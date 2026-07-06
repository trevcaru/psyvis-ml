"""Matplotlib plotting for measurement results: P(correct) plus a logit-margin panel.

Every figure is drawn inside the shared :mod:`psyvis_ml._style` context so the whole gallery
reads as one restrained, publication-quality set. Matplotlib is imported lazily so importing
the package (and using ``measure()`` headless) needs no configured backend.

The x-axis label is the suite's declared ``x_label`` (e.g. ``"RMS contrast"``, ``"noise σ"``,
``"distractor size (px)"``), falling back to a generic ``"stimulus level"`` only when a suite
declares none. Threshold values (and their CIs) go in the legend, never annotated on the curve.
"""

from __future__ import annotations

import numpy as np

from . import _style

__all__ = ["plot_result"]


def _x_label(suite):
    return getattr(suite, "x_label", None) or "stimulus level"


def _grid(levels, sigmoid, n=200):
    lo, hi = float(np.min(levels)), float(np.max(levels))
    if sigmoid == "weibull" and lo > 0.0:
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def threshold_ci_label(thr, fit, target, n_boot, ci, seed):
    """A compact ``θ=value [lo, hi]`` string for the legend (best-effort CI)."""
    try:
        band = fit.bootstrap_ci("threshold", target=target, n_boot=min(n_boot, 200), ci=ci,
                                seed=seed)
        if np.isfinite(band.low) and np.isfinite(band.high):
            return f"θ={thr:.3g} [{band.low:.3g}, {band.high:.3g}]"
    except Exception:  # noqa: BLE001 - the label is best-effort, never fail the plot
        pass
    return f"θ={thr:.3g}"


def plot_result(result, *, ax=None, show_ci=True, show_confidence=True, n_boot=200, ci=0.95,
                seed=0, target=0.75, criterion=0.0, color_index=0):
    """Plot P(correct) vs. the swept variable (and, by default, the logit-margin panel).

    Returns the matplotlib ``Figure``. ``color_index`` selects the model's colour from the
    shared palette so a model keeps its colour across figures. When ``ax`` is supplied, only
    the accuracy panel is drawn on it.
    """
    import matplotlib.pyplot as plt

    xlabel = _x_label(result.suite)
    with _style.rc_context():
        if ax is not None:
            fig, acc_ax, conf_ax = ax.figure, ax, None
        elif show_confidence:
            fig, (acc_ax, conf_ax) = plt.subplots(
                2, 1, figsize=_style.FIG_STACK, sharex=True,
                gridspec_kw={"height_ratios": [3, 2]})
        else:
            fig, acc_ax = plt.subplots(figsize=_style.FIG_ACCURACY)
            conf_ax = None

        sigmoid = getattr(result.suite, "sigmoid", "weibull")

        conf_by_label = {}
        if conf_ax is not None:
            conf = result.confidence(criterion=criterion, n_boot=n_boot, ci=ci, seed=seed)
            conf_by_label = {result.labels[0]: conf} if result.is_single else conf

        for j, cr in enumerate(result.condition_results):
            color = _style.model_color(color_index + j)
            levels = np.asarray(cr.bundle.levels, dtype=float)
            prop = (np.asarray(cr.bundle.n_correct, float)
                    / np.asarray(cr.bundle.n_trials, float))

            xgrid = _grid(levels, sigmoid)
            if show_ci:
                low, high, _frac = cr.fit.bootstrap_curve(xgrid, n_boot=n_boot, ci=ci, seed=seed)
                if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                    acc_ax.fill_between(xgrid, low, high, color=color, alpha=_style.BAND_ALPHA,
                                        lw=0, zorder=1)

            thr = cr.fit.threshold(target)
            label = cr.label
            if np.isfinite(thr):
                label = f"{cr.label} — {threshold_ci_label(thr, cr.fit, target, n_boot, ci, seed)}"
            acc_ax.plot(xgrid, cr.fit.predict(xgrid), color=color, lw=_style.LINE_WIDTH,
                        zorder=2, label=label)
            acc_ax.scatter(levels, prop, s=_style.POINT_SIZE, color=color, edgecolor="white",
                           linewidth=0.5, zorder=3)
            if np.isfinite(thr):
                _style.threshold_marker(acc_ax, thr, target, color)

            if conf_ax is not None:
                r = conf_by_label[cr.label]
                conf_ax.fill_between(levels, r.margin_ci_low, r.margin_ci_high, color=color,
                                     alpha=_style.BAND_ALPHA, lw=0, zorder=1)
                conf_ax.plot(levels, r.mean_margin, color=color, lw=_style.LINE_WIDTH,
                             marker="o", ms=3.5, mec="white", mew=0.4, zorder=3)
                if np.isfinite(r.confidence_threshold):
                    conf_ax.axvline(r.confidence_threshold, color=color, ls=":", lw=1.1,
                                    zorder=2)

        acc_ax.axhline(target, color=_style.REFERENCE_COLOR, ls="--", lw=0.9, zorder=0)
        if sigmoid == "weibull" and np.all(np.asarray(result.levels) > 0):
            acc_ax.set_xscale("log")
            _style.decimal_log_xticks(acc_ax, float(np.min(result.levels)),
                                      float(np.max(result.levels)))
        acc_ax.set_xlabel(xlabel)
        acc_ax.set_ylabel("P(correct)")
        acc_ax.set_ylim(0.0, 1.02)
        # Legend in the empty corner: below rising curves, above falling ones.
        acc_ax.legend(loc="upper right" if getattr(result, "decreasing", False) else "lower right")
        acc_ax.set_title(f"{result.suite.__class__.__name__}: P(correct) vs. {xlabel}")
        _style.style_axes(acc_ax)

        if conf_ax is not None:
            conf_ax.axhline(criterion, color=_style.REFERENCE_COLOR, ls="--", lw=0.9, zorder=0)
            conf_ax.set_xlabel(xlabel)
            conf_ax.set_ylabel("logit margin\n(target − best competitor)")
            conf_ax.set_title(f"confidence (within-model): criterion margin = {criterion:g}",
                              fontsize=10)
            _style.style_axes(conf_ax)

        fig.tight_layout()
    return fig
