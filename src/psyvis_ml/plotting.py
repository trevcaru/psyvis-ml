"""Matplotlib plotting for measurement results: P(correct) plus a logit-margin panel.

Matplotlib is imported lazily inside the plotting function so that importing the package
(and using ``measure()`` headless) does not require a configured backend.

``.plot()`` draws the accuracy psychometric curve and, beneath it, a **within-model**
confidence panel: the mean logit-margin vs. level with its bootstrap CI band, the
criterion (evidence-parity) line, and the confidence-threshold marker. The margin panel is
absolute (within-model only) — cross-model overlays use baseline-relative Δ-margin in
``compare()``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_result"]


def _grid(levels, sigmoid, n=200):
    lo, hi = float(np.min(levels)), float(np.max(levels))
    if sigmoid == "weibull" and lo > 0.0:
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def plot_result(result, *, ax=None, show_ci=True, show_confidence=True, n_boot=200, ci=0.95,
                seed=0, target=0.75, criterion=0.0):
    """Plot P(correct) vs. level (and, by default, the logit-margin panel) for a result.

    Returns the matplotlib ``Figure``. When ``ax`` is supplied, only the accuracy panel is
    drawn on it (no margin panel — a caller-provided single axis cannot host a second panel).
    """
    import matplotlib.pyplot as plt

    if ax is not None:
        fig, acc_ax, conf_ax = ax.figure, ax, None
    elif show_confidence:
        fig, (acc_ax, conf_ax) = plt.subplots(
            2, 1, figsize=(6.0, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    else:
        fig, acc_ax = plt.subplots(figsize=(6.0, 4.5))
        conf_ax = None

    sigmoid = getattr(result.suite, "sigmoid", "weibull")

    conf_by_label = {}
    if conf_ax is not None:
        conf = result.confidence(criterion=criterion, n_boot=n_boot, ci=ci, seed=seed)
        conf_by_label = {result.labels[0]: conf} if result.is_single else conf

    for cr in result.condition_results:
        levels = np.asarray(cr.bundle.levels, dtype=float)
        n_correct = np.asarray(cr.bundle.n_correct, dtype=float)
        n_trials = np.asarray(cr.bundle.n_trials, dtype=float)
        prop = n_correct / n_trials

        points = acc_ax.scatter(levels, prop, s=28, zorder=3, label=cr.label)
        color = points.get_facecolor()[0]

        xgrid = _grid(levels, sigmoid)
        acc_ax.plot(xgrid, cr.fit.predict(xgrid), color=color, lw=2, zorder=2)

        if show_ci:
            low, high, _frac = cr.fit.bootstrap_curve(xgrid, n_boot=n_boot, ci=ci, seed=seed)
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                acc_ax.fill_between(xgrid, low, high, color=color, alpha=0.18, zorder=1)

        thr = cr.fit.threshold(target)
        if np.isfinite(thr):
            acc_ax.plot([thr], [target], marker="o", color=color, ms=7, mfc="white",
                        mec=color, zorder=4)

        if conf_ax is not None:
            r = conf_by_label[cr.label]
            conf_ax.plot(levels, r.mean_margin, color=color, lw=2, marker="o", ms=4, zorder=3)
            conf_ax.fill_between(levels, r.margin_ci_low, r.margin_ci_high, color=color,
                                 alpha=0.18, zorder=1)
            if np.isfinite(r.confidence_threshold):
                conf_ax.axvline(r.confidence_threshold, color=color, ls=":", lw=1.2, zorder=2)

    acc_ax.axhline(target, color="0.6", ls="--", lw=1, zorder=0)
    if sigmoid == "weibull" and np.all(np.asarray(result.levels) > 0):
        acc_ax.set_xscale("log")
    acc_ax.set_xlabel("stimulus level")
    acc_ax.set_ylabel("P(correct)")
    acc_ax.set_ylim(0.0, 1.02)
    acc_ax.legend(loc="lower right", fontsize=9)
    acc_ax.set_title(f"{result.suite.__class__.__name__}: P(correct) vs. level")

    if conf_ax is not None:
        conf_ax.axhline(criterion, color="0.6", ls="--", lw=1, zorder=0)
        conf_ax.set_xlabel("stimulus level")
        conf_ax.set_ylabel("logit margin\n(target − best competitor)")
        conf_ax.set_title(f"confidence (within-model): criterion margin = {criterion:g}",
                          fontsize=9)

    fig.tight_layout()
    return fig
