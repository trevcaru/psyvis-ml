"""Descriptive confidence readout: the logit-margin curve and its threshold.

The primary confidence signal is the target-class **logit margin**
``margin(x) = logit[target] − max(logit[others])`` (see :mod:`psyvis_ml.sweep.scoring`). This
module turns the per-image margins the sweep recorded into a **descriptive** readout — a mean
margin curve, a baseline-relative Δ-margin curve, a confidence threshold (the level where the
mean margin crosses a criterion), the local margin slope there, and bootstrap-over-images CIs.

**This is deliberately not fit through the binomial fitting core.** The margin is a continuous
response; forcing it through a binomial psychometric fit would be wrong *and* would couple the
import-isolated core to this layer. Everything here is plain NumPy, computed outside the core.

Analysis discipline (enforced structurally elsewhere):

* **Within-model only** — logit scales are not comparable across models, so absolute margins
  must never be overlaid across models (see :func:`require_relative_for_multimodel`).
* **Baseline-relative** — the cross-model-comparable quantity is Δ margin from the clean
  (best-stimulus) condition; each model is normalized to its own clean baseline.
* **Direction-aware** — margin declines as the stimulus worsens; the clean baseline is the
  ``level 0`` end for a *decreasing* suite and the top-level end for an *increasing* one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConfidenceReadout", "confidence_readout", "require_relative_for_multimodel"]


def require_relative_for_multimodel(kind: str, n_models: int) -> None:
    """Structural guard: block absolute-margin overlays across models (raise, do not warn).

    Logit scales differ across models, so an absolute-margin overlay of two models is
    meaningless. Cross-model confidence overlays must be baseline-relative Δ-margin.
    """
    if n_models > 1 and kind not in ("delta", "relative"):
        raise ValueError(
            "absolute logit margins are not comparable across models (logit scales differ); "
            "cross-model confidence overlays must be baseline-relative Δ-margin. "
            f"Got kind={kind!r} for {n_models} models — pass kind='delta'."
        )


@dataclass(frozen=True)
class ConfidenceReadout:
    """Descriptive margin readout for one condition of one model.

    ``mean_margin`` / ``margin_ci_*`` are **absolute** (within-model only). ``delta_margin`` /
    ``delta_ci_*`` are baseline-relative (Δ from the clean baseline; the cross-model-comparable
    quantity). ``mean_max_softmax`` is reference-only and calibration-sensitive.
    """

    levels: np.ndarray
    mean_margin: np.ndarray
    margin_ci_low: np.ndarray
    margin_ci_high: np.ndarray
    baseline_index: int
    delta_margin: np.ndarray
    delta_ci_low: np.ndarray
    delta_ci_high: np.ndarray
    confidence_threshold: float
    margin_slope: float
    criterion: float
    decreasing: bool
    mean_target_rank: np.ndarray
    mean_max_softmax: np.ndarray
    chance_level: float = float("nan")
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"confidence: margin@clean={self.mean_margin[self.baseline_index]:.3g}, "
            f"threshold(margin={self.criterion:g})={self.confidence_threshold:.4g}, "
            f"slope={self.margin_slope:.3g}, {'decreasing' if self.decreasing else 'increasing'}"
        )


def _crossing(levels, values, criterion):
    """First interpolated ``level`` where ``values`` crosses ``criterion`` (+ local slope).

    Scans in ascending-level order; returns ``(nan, nan)`` if the curve never crosses.
    """
    order = np.argsort(levels)
    x = np.asarray(levels, float)[order]
    y = np.asarray(values, float)[order] - criterion
    for i in range(x.size - 1):
        if y[i] == 0.0:
            slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) if x[i + 1] != x[i] else np.nan
            return float(x[i]), float(slope)
        if y[i] * y[i + 1] < 0.0:  # sign change brackets a crossing
            t = y[i] / (y[i] - y[i + 1])
            xc = x[i] + t * (x[i + 1] - x[i])
            slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            return float(xc), float(slope)
    return float("nan"), float("nan")


def _boot_means(margins, n_boot, rng):
    """Bootstrap the per-level mean margin by resampling IMAGES (paired across levels)."""
    n_images = margins.shape[1]
    boot = np.empty((n_boot, margins.shape[0]), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_images, n_images)
        boot[b] = margins[:, idx].mean(axis=1)
    return boot


def confidence_readout(bundle, *, decreasing, criterion=0.0, n_boot=1000, ci=0.95, seed=0,
                       chance_level=float("nan")):
    """Compute the descriptive margin readout from a sweep bundle's per-image margins.

    Parameters
    ----------
    bundle
        A :class:`~psyvis_ml.sweep.RunBundle` carrying per-image ``margin`` (and, for
        reference, ``target_rank`` / ``max_softmax``) in ``per_image``.
    decreasing
        The suite's declared axis direction (``MeasureResult.decreasing``). Sets which end is
        the clean baseline: ``level 0`` when decreasing, the top level when increasing.
    criterion
        The margin criterion for the confidence threshold; default ``0`` = evidence parity
        between the target and its best competitor.
    n_boot, ci, seed
        Bootstrap-over-images CI settings for the mean and Δ margin curves.
    """
    per = bundle.per_image
    if "margin" not in per:
        raise ValueError(
            "bundle has no per-image margins; this sweep did not record confidence signals."
        )
    margins = np.asarray(per["margin"], dtype=float)   # (n_levels, n_images)
    levels = np.asarray(bundle.levels, dtype=float)
    if margins.ndim != 2 or margins.shape[0] != levels.size:
        raise ValueError(
            f"margins shape {margins.shape} inconsistent with {levels.size} levels."
        )
    n_levels = levels.size

    mean_margin = margins.mean(axis=1)
    # Clean baseline = the best-stimulus end (direction-aware).
    baseline_index = 0 if decreasing else n_levels - 1
    delta_margin = mean_margin - mean_margin[baseline_index]

    rng = np.random.default_rng(seed)
    boot = _boot_means(margins, n_boot, rng)
    lo_q, hi_q = (1.0 - ci) / 2.0 * 100.0, (1.0 + ci) / 2.0 * 100.0
    margin_ci_low = np.percentile(boot, lo_q, axis=0)
    margin_ci_high = np.percentile(boot, hi_q, axis=0)
    boot_delta = boot - boot[:, [baseline_index]]
    delta_ci_low = np.percentile(boot_delta, lo_q, axis=0)
    delta_ci_high = np.percentile(boot_delta, hi_q, axis=0)

    threshold, slope = _crossing(levels, mean_margin, criterion)

    def _mean(name):
        return (np.asarray(per[name], float).mean(axis=1) if name in per
                else np.full(n_levels, np.nan))

    return ConfidenceReadout(
        levels=levels,
        mean_margin=mean_margin,
        margin_ci_low=margin_ci_low,
        margin_ci_high=margin_ci_high,
        baseline_index=int(baseline_index),
        delta_margin=delta_margin,
        delta_ci_low=delta_ci_low,
        delta_ci_high=delta_ci_high,
        confidence_threshold=float(threshold),
        margin_slope=float(slope),
        criterion=float(criterion),
        decreasing=bool(decreasing),
        mean_target_rank=_mean("target_rank"),
        mean_max_softmax=_mean("max_softmax"),
        chance_level=float(chance_level),
        metadata={"n_images": int(margins.shape[1]), "n_boot": int(n_boot), "ci": float(ci)},
    )
