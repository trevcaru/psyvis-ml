"""Matplotlib plotting for measurement results: P(correct) vs. level with fitted curve + CI.

Matplotlib is imported lazily inside the plotting function so that importing the package
(and using ``measure()`` headless) does not require a configured backend.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_result"]


def _grid(levels, sigmoid, n=200):
    lo, hi = float(np.min(levels)), float(np.max(levels))
    if sigmoid == "weibull" and lo > 0.0:
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def plot_result(result, *, ax=None, show_ci=True, n_boot=200, ci=0.95, seed=0,
                target=0.75):
    """Plot P(correct) vs. stimulus level for each condition of a ``MeasureResult``.

    Draws, per condition: the observed proportion correct at each level, the fitted
    psychometric curve, an optional bootstrap CI band around that curve, and a marker at the
    ``target`` threshold. Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
    else:
        fig = ax.figure

    sigmoid = getattr(result.suite, "sigmoid", "weibull")

    for cr in result.condition_results:
        levels = np.asarray(cr.bundle.levels, dtype=float)
        n_correct = np.asarray(cr.bundle.n_correct, dtype=float)
        n_trials = np.asarray(cr.bundle.n_trials, dtype=float)
        prop = n_correct / n_trials

        points = ax.scatter(levels, prop, s=28, zorder=3, label=cr.label)
        color = points.get_facecolor()[0]

        xgrid = _grid(levels, sigmoid)
        ax.plot(xgrid, cr.fit.predict(xgrid), color=color, lw=2, zorder=2)

        if show_ci:
            low, high, _frac = cr.fit.bootstrap_curve(xgrid, n_boot=n_boot, ci=ci, seed=seed)
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                ax.fill_between(xgrid, low, high, color=color, alpha=0.18, zorder=1)

        thr = cr.fit.threshold(target)
        if np.isfinite(thr):
            ax.plot([thr], [target], marker="o", color=color, ms=7, mfc="white",
                    mec=color, zorder=4)

    ax.axhline(target, color="0.6", ls="--", lw=1, zorder=0)
    if sigmoid == "weibull" and np.all(np.asarray(result.levels) > 0):
        ax.set_xscale("log")
    ax.set_xlabel("stimulus level")
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"{result.suite.__class__.__name__}: P(correct) vs. level")
    fig.tight_layout()
    return fig
