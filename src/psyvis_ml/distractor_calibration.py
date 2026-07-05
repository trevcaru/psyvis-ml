"""Calibrate the swept distractor-**size** band for the distractor-robustness suite.

A usable distractor-robustness curve — distracted accuracy falling from a near-ceiling at small
distractor size to an interference floor at large size — is only visible if the swept size range
spans that ceiling→floor transition. :func:`calibrate_distractor_size` probes a few caller-
supplied **extreme** sizes (smallest … largest), measures the undistracted baseline (the clean
ceiling) and the distracted accuracy across the probes, checks that a *ceiling→floor difference*
exists across the size range, and only then places the final swept sizes across the transition.
It returns a :class:`DistractorCalibration`; if the curve is flat (no ceiling→floor difference)
it says so (``transition_detected=False``) so a caller can stop rather than fit noise.

Observer-model note (PRD §14) is inherited from the suite: "correct" is argmax top-1 on the
composited stimulus, a swappable observer-model choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DistractorCalibration", "calibrate_distractor_size"]


@dataclass(frozen=True)
class DistractorCalibration:
    """What the size calibration found, and the swept sizes it placed.

    ``transition_detected`` is True only when the smallest probe is near the clean ceiling and
    the largest probe drops a real margin below it (``small_size_accuracy − large_size_accuracy
    ≥ transition_margin``). ``levels`` is the calibrated size grid to sweep next; when no
    transition is found it spans the full probed range so the (flat) curve is still reported.
    """

    chance_level: float
    baseline_ceiling: float
    small_size_accuracy: float
    large_size_accuracy: float
    transition_detected: bool
    probe_sizes: np.ndarray
    probe_distracted_frac: np.ndarray
    probe_baseline_frac: np.ndarray
    low: float
    high: float
    levels: np.ndarray
    notes: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"baseline(ceiling)={self.baseline_ceiling:.3f}, "
            f"small-size={self.small_size_accuracy:.3f}, "
            f"large-size={self.large_size_accuracy:.3f}, chance={self.chance_level:.3f} -> "
            f"transition_detected={self.transition_detected}; "
            f"swept sizes [{self.low:.1f}, {self.high:.1f}] x{len(self.levels)}"
        )


def _first_downcrossing(sizes, frac, level) -> float:
    """Interpolated size at which a (falling) ``frac`` first drops to ``level``.

    ``sizes`` must be ascending. Falls back to the endpoints when ``level`` is never reached
    (returns the largest) or is already below it at the smallest (returns the smallest).
    """
    sizes = np.asarray(sizes, float)
    frac = np.asarray(frac, float)
    for i in range(sizes.size):
        if frac[i] <= level:
            if i == 0:
                return float(sizes[0])
            x0, x1, y0, y1 = sizes[i - 1], sizes[i], frac[i - 1], frac[i]
            if y1 == y0:
                return float(x1)
            t = (y0 - level) / (y0 - y1)
            return float(x0 + t * (x1 - x0))
    return float(sizes[-1])


def calibrate_distractor_size(model, dataset, *, canvas_shape, probe_sizes, n_levels=11,
                              size_scale="linear", top_k=1, seed=0, distractor_patch=None,
                              n_distractors=4, background=0.0, transition_margin=0.15,
                              band_frac=0.1, pad_frac=0.15):
    """Probe extreme distractor sizes, verify a ceiling→floor transition, and place the sweep.

    Parameters
    ----------
    model, dataset
        As for :func:`psyvis_ml.measure` — ``model`` a callable ``image -> logits`` scored
        against ``dataset.labels`` (use **ground-truth** labels; the target should already be
        big enough to clear the recognition ceiling).
    canvas_shape, distractor_patch, n_distractors, background
        Passed straight to :class:`~psyvis_ml.suites.DistractorRobustness`.
    probe_sizes
        A short list of **extreme** candidate distractor sizes (smallest … largest) to
        characterize the ceiling and floor. Caller-supplied and reported.
    n_levels, size_scale
        Number and spacing ("linear"/"log") of the final swept sizes placed across the span.
    transition_margin
        Minimum small-size − large-size accuracy gap to call it a ceiling→floor transition.
    band_frac, pad_frac
        The final range is placed between where distracted first drops to ``ceiling −
        band_frac*span`` and where it reaches ``floor + band_frac*span``, then padded.

    Returns
    -------
    DistractorCalibration
    """
    from .api import measure
    from .suites import DistractorRobustness

    probe = np.sort(np.asarray(probe_sizes, dtype=float))
    if probe.size < 2:
        raise ValueError("need at least two probe sizes (a small one and a large one).")
    if np.any(probe <= 0):
        raise ValueError("probe sizes must be positive (a distractor size in pixels).")

    suite = DistractorRobustness(canvas_shape=canvas_shape, distractor_patch=distractor_patch,
                                 n_distractors=n_distractors, background=background,
                                 include_baseline=True)
    res = measure(model, suite, dataset, probe, top_k=top_k, seed=seed)

    distracted_label = next(lbl for lbl in res.labels if lbl.startswith("distractors"))
    base_label = next(lbl for lbl in res.labels if lbl.startswith("undistracted"))
    db, bb = res.bundles[distracted_label], res.bundles[base_label]
    distracted = np.asarray(db.n_correct, float) / np.asarray(db.n_trials, float)
    baseline = np.asarray(bb.n_correct, float) / np.asarray(bb.n_trials, float)

    ceiling = float(np.mean(baseline))       # undistracted accuracy = the clean ceiling
    small_acc = float(distracted[0])          # distracted at the smallest probe size
    large_acc = float(distracted[-1])         # distracted at the largest probe size
    span = small_acc - large_acc
    transition_detected = bool(span >= transition_margin)

    if transition_detected:
        low = _first_downcrossing(probe, distracted, small_acc - band_frac * span)
        high = _first_downcrossing(probe, distracted, large_acc + band_frac * span)
        if not (high > low):  # degenerate crossing -> fall back to the full probed range
            low, high = float(probe[0]), float(probe[-1])
        pad = pad_frac * (high - low)
        low = max(1.0, low - pad)
        high = high + pad
        notes = (f"transition detected: small-size {small_acc:.3f} -> large-size {large_acc:.3f} "
                 f"(clean ceiling {ceiling:.3f}); swept sizes placed across the transition.")
    else:
        low, high = float(probe[0]), float(probe[-1])
        notes = (f"NO ceiling->floor transition across sizes (small {small_acc:.3f}, large "
                 f"{large_acc:.3f}, clean ceiling {ceiling:.3f}): the curve is flat. Either the "
                 f"distractors do not interfere or the size range misses the transition — widen "
                 f"probe_sizes before concluding 'flat'.")

    if size_scale == "log":
        levels = np.geomspace(low, high, n_levels)
    elif size_scale == "linear":
        levels = np.linspace(low, high, n_levels)
    else:
        raise ValueError(f"size_scale must be 'linear' or 'log'; got {size_scale!r}.")

    return DistractorCalibration(
        chance_level=float(res.chance_level),
        baseline_ceiling=ceiling,
        small_size_accuracy=small_acc,
        large_size_accuracy=large_acc,
        transition_detected=transition_detected,
        probe_sizes=probe,
        probe_distracted_frac=distracted,
        probe_baseline_frac=baseline,
        low=float(low),
        high=float(high),
        levels=levels,
        notes=notes,
        metadata={"n_distractors": n_distractors, "canvas_shape": tuple(canvas_shape),
                  "transition_margin": transition_margin},
    )
