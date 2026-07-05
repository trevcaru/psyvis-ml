"""Calibrate the swept distractor-spacing range for the distractor-robustness suite.

A useful distractor-robustness curve — distracted performance climbing from an interference
floor back to the undistracted baseline as distractors move away — is only visible if the swept
spacing range spans from "close enough to interfere" to "far enough to clear". Blindly
hardcoding a pixel range can miss the transition entirely and return a flat line you cannot
interpret.

:func:`calibrate_distractor_spacing` removes that ambiguity. It measures, at a few caller-
supplied **extreme** probe spacings, both the **undistracted baseline** (the clean ceiling) and
the **distracted** accuracy across the probes, checks that there is both an *interference floor*
and a *cleared ceiling*, and only then places the final swept levels across the transition. It
returns a :class:`DistractorCalibration` describing exactly what it found. If no
floor→ceiling transition exists in the probed range it says so (``transition_detected=False``).

Observer-model note (PRD §14) is inherited from the suite: "correct" is argmax top-1 on the
composited stimulus, a swappable observer-model choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DistractorCalibration", "calibrate_distractor_spacing"]


@dataclass(frozen=True)
class DistractorCalibration:
    """What the spacing calibration found, and the swept levels it placed.

    ``transition_detected`` is True only when the tightest probe genuinely interferes (baseline
    − floor ≥ ``transition_margin``) *and* the widest probe clears near baseline. ``levels`` is
    the calibrated spacing grid to sweep next; when no transition is found it spans the full
    probed range so the (flat) curve is still reported honestly.
    """

    chance_level: float
    baseline_ceiling: float
    interference_floor: float
    cleared_high: float
    transition_detected: bool
    probe_spacings: np.ndarray
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
            f"floor={self.interference_floor:.3f}, widest-distracted={self.cleared_high:.3f}, "
            f"chance={self.chance_level:.3f} -> transition_detected={self.transition_detected}; "
            f"swept range [{self.low:.2f}, {self.high:.2f}] x{len(self.levels)}"
        )


def _first_upcrossing(spacings, frac, level) -> float:
    """Interpolated spacing at which a (rising) ``frac`` first reaches ``level``.

    ``spacings`` must be ascending. Falls back to the endpoints when ``level`` is never
    reached (returns the widest) or is already exceeded at the tightest (returns the tightest).
    """
    spacings = np.asarray(spacings, float)
    frac = np.asarray(frac, float)
    for i in range(spacings.size):
        if frac[i] >= level:
            if i == 0:
                return float(spacings[0])
            x0, x1, y0, y1 = spacings[i - 1], spacings[i], frac[i - 1], frac[i]
            if y1 == y0:
                return float(x1)
            t = (level - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return float(spacings[-1])


def calibrate_distractor_spacing(model, dataset, *, canvas_shape, probe_spacings,
                                 n_levels=11, spacing_scale="linear", top_k=1, seed=0,
                                 distractor_patch=None, n_distractors=2, angle=0.0,
                                 background=0.0, transition_margin=0.15, band_frac=0.1,
                                 pad_frac=0.15):
    """Probe extreme spacings, verify a floor→ceiling transition, and place the swept levels.

    Parameters
    ----------
    model, dataset
        As for :func:`psyvis_ml.measure` — ``model`` a callable ``image -> logits`` scored
        against ``dataset.labels`` (use **ground-truth** labels; a self-consistency label pins
        the baseline to a trivial 1.0 and there is nothing real to clear toward).
    canvas_shape, distractor_patch, n_distractors, angle, background
        Passed straight to :class:`~psyvis_ml.suites.DistractorRobustness`.
    probe_spacings
        A short list of **extreme** candidate spacings (tight … wide) to characterize the floor
        and ceiling. Caller-supplied and reported — never a blind hardcoded range.
    n_levels, spacing_scale
        Number and spacing ("linear"/"log") of the final swept levels placed across the span.
    transition_margin
        Minimum baseline−floor gap (and maximum baseline−widest gap) to call it a transition.
    band_frac, pad_frac
        The final range is placed between where distracted crosses ``floor + band_frac*span``
        and ``ceiling − band_frac*span``, then padded by ``pad_frac`` of that width each side.

    Returns
    -------
    DistractorCalibration
    """
    from .api import measure
    from .suites import DistractorRobustness

    probe = np.sort(np.asarray(probe_spacings, dtype=float))
    if probe.size < 2:
        raise ValueError("need at least two probe spacings (a tight one and a wide one).")
    if np.any(probe <= 0):
        raise ValueError("probe spacings must be positive (Weibull spacing axis).")

    suite = DistractorRobustness(canvas_shape=canvas_shape, distractor_patch=distractor_patch,
                                 n_distractors=n_distractors, angle=angle, background=background,
                                 include_baseline=True)
    res = measure(model, suite, dataset, probe, top_k=top_k, seed=seed)

    distracted_label = next(lbl for lbl in res.labels if lbl.startswith("distractors"))
    base_label = next(lbl for lbl in res.labels if lbl.startswith("undistracted"))
    db, bb = res.bundles[distracted_label], res.bundles[base_label]
    distracted = np.asarray(db.n_correct, float) / np.asarray(db.n_trials, float)
    baseline = np.asarray(bb.n_correct, float) / np.asarray(bb.n_trials, float)

    ceiling = float(np.mean(baseline))          # undistracted accuracy = the clean ceiling
    floor = float(distracted[0])                 # distracted at the tightest probe
    cleared_high = float(distracted[-1])         # distracted at the widest probe
    span = ceiling - floor
    interferes = span >= transition_margin
    cleared = cleared_high >= ceiling - transition_margin
    transition_detected = bool(interferes and cleared)

    if transition_detected:
        low = _first_upcrossing(probe, distracted, floor + band_frac * span)
        high = _first_upcrossing(probe, distracted, ceiling - band_frac * span)
        if not (high > low):  # degenerate crossing -> fall back to the full probed range
            low, high = float(probe[0]), float(probe[-1])
        pad = pad_frac * (high - low)
        low = max(float(probe[0]) * 0.5, low - pad)  # stay positive; don't overshoot tight
        high = high + pad
        notes = (f"transition detected: distracted floor {floor:.3f} -> baseline ceiling "
                 f"{ceiling:.3f}; swept levels placed across the transition.")
    else:
        low, high = float(probe[0]), float(probe[-1])
        notes = (f"NO transition in probed range (interferes={interferes}, cleared={cleared}): "
                 f"floor {floor:.3f}, widest-distracted {cleared_high:.3f}, ceiling {ceiling:.3f}. "
                 f"Either distractors do not interfere here, or the probe range misses the "
                 f"transition — widen probe_spacings before concluding 'flat'.")

    if spacing_scale == "log":
        levels = np.geomspace(low, high, n_levels)
    elif spacing_scale == "linear":
        levels = np.linspace(low, high, n_levels)
    else:
        raise ValueError(f"spacing_scale must be 'linear' or 'log'; got {spacing_scale!r}.")

    return DistractorCalibration(
        chance_level=float(res.chance_level),
        baseline_ceiling=ceiling,
        interference_floor=floor,
        cleared_high=cleared_high,
        transition_detected=transition_detected,
        probe_spacings=probe,
        probe_distracted_frac=distracted,
        probe_baseline_frac=baseline,
        low=float(low),
        high=float(high),
        levels=levels,
        notes=notes,
        metadata={"n_distractors": n_distractors, "canvas_shape": tuple(canvas_shape),
                  "transition_margin": transition_margin},
    )
