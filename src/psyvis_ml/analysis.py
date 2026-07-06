"""Paired over-images bootstrap analysis for the gallery's ``analysis.md``.

**Why bootstrap over images, not a t-test.** These models are *deterministic*: run the same
image at the same stimulus level twice and you get the same logits. There is no trial-level
sampling noise, so a frequentist t-test / p-value on "trials" would be fabricated. The only
real source of variability is *which images* you happened to measure — so all uncertainty here
comes from **bootstrapping over the image set**, and a difference between two models is called
reliable when the bootstrap CI of the *paired* difference (both models recomputed on the same
resampled images) **excludes zero**.

:func:`analyze_suite` refits each model's psychometric function on each image resample (via the
fitter's warm-started refit — the import-isolated core is untouched), and reports per-model
threshold/slope with over-images CIs, the paired threshold-difference test, a goodness-of-fit
per curve, and the confidence layer's margin slopes / Δ-margin divergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .comparison import resolve_comparison
from .confidence import confidence_readout

__all__ = ["SuiteAnalysis", "analyze_suite", "goodness_of_fit"]


def _ci(values, ci):
    """Percentile CI over the finite entries of ``values`` (nan-safe)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    lo = float(np.percentile(v, (1.0 - ci) / 2.0 * 100.0))
    hi = float(np.percentile(v, (1.0 + ci) / 2.0 * 100.0))
    return lo, hi, int(v.size)


def _excludes_zero(lo, hi):
    return bool(np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0.0 <= hi))


def goodness_of_fit(bundle, fit) -> dict:
    """Per-curve goodness of fit: convergence, at-bound params, and a weighted R².

    ``r2`` is the trial-weighted fraction of variance in the per-level observed proportions
    explained by the fitted curve (1 = perfect; low/negative = a bad fit, made visible).
    """
    levels, nc, nt = bundle.to_fit_inputs()
    p_obs = nc / nt
    p_pred = np.asarray(fit.predict(levels), dtype=float)
    w = nt
    pbar = float(np.sum(w * p_obs) / np.sum(w))
    ss_res = float(np.sum(w * (p_obs - p_pred) ** 2))
    ss_tot = float(np.sum(w * (p_obs - pbar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"converged": bool(fit.converged), "at_bound": list(fit.at_bound), "r2": float(r2)}


@dataclass(frozen=True)
class SuiteAnalysis:
    """Paired over-images analysis of two models on one suite/condition."""

    suite: str
    condition: str
    target: float
    decreasing: bool
    models: list
    n_images: int
    n_boot: int
    ci: float
    threshold: dict
    threshold_ci: dict
    slope: dict
    slope_ci: dict
    margin_slope: dict
    delta_margin_endpoint: dict
    gof: dict
    threshold_diff: float
    threshold_diff_ci: tuple
    threshold_diff_excludes_zero: bool
    delta_margin_diff: float
    delta_margin_diff_ci: tuple
    delta_margin_diff_excludes_zero: bool
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["threshold_diff_ci"] = list(self.threshold_diff_ci)
        d["delta_margin_diff_ci"] = list(self.delta_margin_diff_ci)
        d["threshold_ci"] = {k: list(v) for k, v in self.threshold_ci.items()}
        d["slope_ci"] = {k: list(v) for k, v in self.slope_ci.items()}
        return d


def _resample_curve(bundle, fit, idx, target):
    """Refit on an image resample; return (threshold, slope, per-level mean margin)."""
    correct = np.asarray(bundle.per_image["correct"], dtype=float)
    nc_boot = correct[:, idx].sum(axis=1)
    refit = fit._refit(nc_boot)
    margin = np.asarray(bundle.per_image["margin"], dtype=float)[:, idx].mean(axis=1)
    return refit.threshold(target), refit.slope(target), margin


def analyze_suite(results, *, condition=None, target=0.75, n_boot=2000, ci=0.95, seed=0):
    """Paired over-images bootstrap analysis of exactly two models on a shared suite/condition.

    Returns a :class:`SuiteAnalysis`. The threshold/slope CIs and both difference tests are
    computed by resampling the image set (the only source of variability for deterministic
    models); the paired difference recomputes *both* models on the same resample each iteration.
    """
    condition_label, selected = resolve_comparison(results, condition)
    if len(selected) != 2:
        raise ValueError(
            f"paired difference analysis needs exactly two models; got {len(selected)}."
        )
    decreasing = bool(results[0].decreasing)
    suite_name = results[0].suite.__class__.__name__
    (label_a, cr_a), (label_b, cr_b) = selected

    levels = np.asarray(cr_a.bundle.levels, dtype=float)
    n_levels = levels.size
    baseline_i = 0 if decreasing else n_levels - 1     # clean end
    degraded_i = n_levels - 1 if decreasing else 0     # worst-stimulus end
    n_images = int(np.asarray(cr_a.bundle.per_image["correct"]).shape[1])

    # Point estimates.
    thr = {label_a: float(cr_a.fit.threshold(target)), label_b: float(cr_b.fit.threshold(target))}
    slp = {label_a: float(cr_a.fit.slope(target)), label_b: float(cr_b.fit.slope(target))}
    conf_a = confidence_readout(cr_a.bundle, decreasing=decreasing)
    conf_b = confidence_readout(cr_b.bundle, decreasing=decreasing)
    margin_slope = {label_a: float(conf_a.margin_slope), label_b: float(conf_b.margin_slope)}
    dmarg_end = {label_a: float(conf_a.delta_margin[degraded_i]),
                 label_b: float(conf_b.delta_margin[degraded_i])}
    gof = {label_a: goodness_of_fit(cr_a.bundle, cr_a.fit),
           label_b: goodness_of_fit(cr_b.bundle, cr_b.fit)}

    # Paired over-images bootstrap.
    rng = np.random.default_rng(seed)
    tA, tB, sA, sB, tdiff, dmdiff = [], [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_images, n_images)
        thr_a, slp_a, marg_a = _resample_curve(cr_a.bundle, cr_a.fit, idx, target)
        thr_b, slp_b, marg_b = _resample_curve(cr_b.bundle, cr_b.fit, idx, target)
        tA.append(thr_a)
        tB.append(thr_b)
        sA.append(slp_a)
        sB.append(slp_b)
        if np.isfinite(thr_a) and np.isfinite(thr_b):
            tdiff.append(thr_a - thr_b)
        dmdiff.append((marg_a[degraded_i] - marg_a[baseline_i])
                      - (marg_b[degraded_i] - marg_b[baseline_i]))

    thr_ci = {label_a: _ci(tA, ci)[:2], label_b: _ci(tB, ci)[:2]}
    slp_ci = {label_a: _ci(sA, ci)[:2], label_b: _ci(sB, ci)[:2]}
    td_lo, td_hi, _ = _ci(tdiff, ci)
    dm_lo, dm_hi, _ = _ci(dmdiff, ci)

    return SuiteAnalysis(
        suite=suite_name, condition=condition_label, target=float(target),
        decreasing=decreasing, models=[label_a, label_b], n_images=n_images, n_boot=int(n_boot),
        ci=float(ci), threshold=thr, threshold_ci=thr_ci, slope=slp, slope_ci=slp_ci,
        margin_slope=margin_slope, delta_margin_endpoint=dmarg_end, gof=gof,
        threshold_diff=float(thr[label_a] - thr[label_b]),
        threshold_diff_ci=(td_lo, td_hi), threshold_diff_excludes_zero=_excludes_zero(td_lo, td_hi),
        delta_margin_diff=float(dmarg_end[label_a] - dmarg_end[label_b]),
        delta_margin_diff_ci=(dm_lo, dm_hi),
        delta_margin_diff_excludes_zero=_excludes_zero(dm_lo, dm_hi),
        metadata={"baseline_index": int(baseline_i), "degraded_index": int(degraded_i)},
    )
