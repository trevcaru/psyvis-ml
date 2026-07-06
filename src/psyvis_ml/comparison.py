"""Multi-model comparison on one psychometric axis — the PRD §5 / §12 "killer plot".

``compare_results`` overlays several models' psychometric curves (observed points, fitted
curve, threshold, CI band) for one suite/condition on a single axis, and — where a credible
published human curve exists — overlays a cited human reference on the same axis (the
human-vs-models figure). It honours each result's *declared* axis direction (``decreasing``),
and refuses to compare results from mismatched suites/conditions with a clear error rather
than silently plotting apples against oranges.
"""

from __future__ import annotations

import numpy as np

from . import _style
from .confidence import confidence_readout, require_relative_for_multimodel
from .reference import REFERENCE_PEAK_SF_CPD, human_reference_for

__all__ = ["compare_results", "comparison_summary", "resolve_comparison",
           "confidence_comparison", "plot_confidence_delta"]


def _curve_grid(levels, sigmoid, decreasing, n=200):
    lo, hi = float(np.min(levels)), float(np.max(levels))
    if sigmoid == "weibull" and lo > 0.0 and not decreasing:
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def _model_label(result, i):
    return result.model_name or f"model {i + 1}"


def resolve_comparison(results, condition=None):
    """Validate that ``results`` are comparable and select one condition from each.

    Returns ``(condition_label, [(model_label, ConditionResult), ...])``. Raises ``ValueError``
    on any mismatch that would make a shared-axis overlay meaningless: differing suites,
    differing declared direction, or a condition absent from some result.
    """
    results = list(results)
    if len(results) < 1:
        raise ValueError("compare needs at least one result.")

    suites = {r.suite.__class__.__name__ for r in results}
    if len(suites) != 1:
        raise ValueError(
            f"cannot compare results from different suites: {sorted(suites)}. "
            "Comparison requires a single shared suite/axis."
        )
    directions = {bool(r.decreasing) for r in results}
    if len(directions) != 1:
        raise ValueError(
            "cannot compare results with different declared axis directions "
            "(decreasing vs. increasing); their thresholds are not on a common scale."
        )
    sigmoids = {getattr(r.suite, "sigmoid", "weibull") for r in results}
    if len(sigmoids) != 1:
        raise ValueError(f"cannot compare results with different sigmoids: {sorted(sigmoids)}.")

    # Resolve the condition label to overlay, shared across every result.
    if condition is None:
        multi = [r for r in results if not r.is_single]
        if multi:
            raise ValueError(
                "some results have multiple conditions; pass condition=<label> to choose "
                f"which to compare. Available labels: {sorted(set(multi[0].labels))}."
            )
        # Every result is single-condition; require they share the same condition label.
        labels = {r.labels[0] for r in results}
        if len(labels) != 1:
            raise ValueError(
                f"single-condition results have different condition labels {sorted(labels)}; "
                "pass condition=<label> or measure them on the same condition."
            )
        condition = results[0].labels[0]

    selected = []
    for i, r in enumerate(results):
        if condition not in r.fits:
            raise ValueError(
                f"condition {condition!r} not found in result for "
                f"{_model_label(r, i)!r}; it has {sorted(r.labels)}."
            )
        cr = next(c for c in r.condition_results if c.label == condition)
        selected.append((_model_label(r, i), cr))
    return condition, selected


def _human_overlay_spec(results, condition_label, selected):
    """Return ``(reference, spatial_freq, human_threshold)`` or ``None`` if unavailable."""
    suite_name = results[0].suite.__class__.__name__
    ref = human_reference_for(suite_name)
    if ref is None:
        return None
    # Use the condition's declared spatial frequency if it has one, else the CSF peak.
    meta = selected[0][1].metadata or {}
    sf = meta.get("spatial_freq")
    sf = float(sf) if sf else REFERENCE_PEAK_SF_CPD
    return ref, sf, ref.threshold_at(sf)


def confidence_comparison(results, *, condition=None, kind="delta", criterion=0.0,
                          n_boot=1000, ci=0.95, seed=0):
    """Per-model **baseline-relative Δ-margin** curves for a shared condition.

    This is the cross-model-safe confidence view: only Δ margin (each model normalized to its
    own clean baseline) is exposed. Absolute margins are blocked across models by
    :func:`~psyvis_ml.confidence.require_relative_for_multimodel` — a structural guard, not a
    warning. Returns ``(condition_label, [per-model curve dicts])``.
    """
    condition_label, selected = resolve_comparison(results, condition)
    require_relative_for_multimodel(kind, len(results))
    decreasing = bool(results[0].decreasing)
    curves = []
    for model_label, cr in selected:
        r = confidence_readout(cr.bundle, decreasing=decreasing, criterion=criterion,
                               n_boot=n_boot, ci=ci, seed=seed)
        curves.append({
            "model": model_label,
            "levels": r.levels,
            "delta_margin": r.delta_margin,
            "delta_ci_low": r.delta_ci_low,
            "delta_ci_high": r.delta_ci_high,
            "confidence_threshold": r.confidence_threshold,
            "margin_slope": r.margin_slope,
            "baseline_index": r.baseline_index,
        })
    return condition_label, curves


def comparison_summary(results, *, condition=None, target=0.75, n_boot=400, ci=0.95, seed=0):
    """Per-model threshold/slope/CI rows (+ confidence threshold/slope) for the report table."""
    condition_label, selected = resolve_comparison(results, condition)
    decreasing = bool(results[0].decreasing)
    rows = []
    for model_label, cr in selected:
        thr = cr.fit.threshold(target)
        slope = cr.fit.slope(target)
        ci_lo = ci_hi = float("nan")
        if np.isfinite(thr):
            band = cr.fit.bootstrap_ci("threshold", target=target, n_boot=n_boot, ci=ci,
                                       seed=seed)
            ci_lo, ci_hi = band.low, band.high
        conf = confidence_readout(cr.bundle, decreasing=decreasing, n_boot=n_boot, ci=ci,
                                  seed=seed)
        rows.append({
            "model": model_label,
            "condition": condition_label,
            "threshold": float(thr),
            "threshold_ci_low": float(ci_lo),
            "threshold_ci_high": float(ci_hi),
            "slope": float(slope),
            "direction": "decreasing" if decreasing else "increasing",
            "confidence_threshold": float(conf.confidence_threshold),
            "margin_slope": float(conf.margin_slope),
        })
    return condition_label, rows


def compare_results(results, *, condition=None, target=0.75, human="auto", ax=None,
                    show_ci=True, show_confidence=True, confidence_kind="delta",
                    n_boot=200, ci=0.95, seed=0, title=None):
    """Overlay several models (and an optional human reference) on one psychometric axis.

    By default a second panel overlays each model's **baseline-relative Δ-margin** (the
    cross-model-comparable confidence view). Absolute-margin cross-model overlays are blocked
    structurally: passing ``confidence_kind="absolute"`` with more than one model raises.

    Parameters
    ----------
    results
        A list of :class:`~psyvis_ml.MeasureResult`, one per model, on the same suite/condition.
    condition
        The condition label to compare; required when results have multiple conditions.
    target
        Threshold criterion (proportion correct) to mark on the accuracy panel.
    human
        ``"auto"``/``True`` overlays a cited human reference when one exists for the suite and
        degrades gracefully (a note, models only) when none does; ``False`` disables it.
    show_ci
        Draw a bootstrap CI band around each fitted curve.
    show_confidence
        Draw the Δ-margin confidence panel beneath the accuracy panel.
    confidence_kind
        Must be ``"delta"``/``"relative"`` for multi-model overlays (structural guard).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from .plotting import threshold_ci_label

    condition_label, selected = resolve_comparison(results, condition)
    # Structural guard: absolute margins are never comparable across models.
    require_relative_for_multimodel(confidence_kind, len(results))
    decreasing = bool(results[0].decreasing)
    sigmoid = getattr(results[0].suite, "sigmoid", "weibull")
    suite_name = results[0].suite.__class__.__name__
    xlabel = getattr(results[0].suite, "x_label", None) or "stimulus level"

    with _style.rc_context():
        if ax is not None:
            fig, acc_ax, conf_ax = ax.figure, ax, None
        elif show_confidence:
            fig, (acc_ax, conf_ax) = plt.subplots(
                2, 1, figsize=_style.FIG_COMPARE, sharex=True,
                gridspec_kw={"height_ratios": [3, 2]})
        else:
            fig, acc_ax = plt.subplots(figsize=_style.FIG_ACCURACY)
            conf_ax = None

        all_levels, model_thresholds = [], []
        for i, (model_label, cr) in enumerate(selected):
            color = _style.model_color(i)
            levels = np.asarray(cr.bundle.levels, dtype=float)
            all_levels.append(levels)
            prop = np.asarray(cr.bundle.n_correct, float) / np.asarray(cr.bundle.n_trials, float)
            xgrid = _curve_grid(levels, sigmoid, decreasing)

            if show_ci:
                low, high, _frac = cr.fit.bootstrap_curve(xgrid, n_boot=n_boot, ci=ci, seed=seed)
                if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                    acc_ax.fill_between(xgrid, low, high, color=color, alpha=_style.BAND_ALPHA,
                                        lw=0, zorder=1)

            thr = cr.fit.threshold(target)
            label = model_label
            if np.isfinite(thr):
                model_thresholds.append(thr)
                tlab = threshold_ci_label(thr, cr.fit, target, n_boot, ci, seed)
                label = f"{model_label} — {tlab}"
            acc_ax.plot(xgrid, cr.fit.predict(xgrid), color=color, lw=_style.LINE_WIDTH,
                        zorder=2, label=label)
            acc_ax.scatter(levels, prop, s=_style.POINT_SIZE, color=color, edgecolor="white",
                           linewidth=0.5, zorder=3)
            if np.isfinite(thr):
                _style.threshold_marker(acc_ax, thr, target, color)

            if conf_ax is not None:
                r = confidence_readout(cr.bundle, decreasing=decreasing, n_boot=n_boot, ci=ci,
                                       seed=seed)
                conf_ax.fill_between(levels, r.delta_ci_low, r.delta_ci_high, color=color,
                                     alpha=_style.BAND_ALPHA, lw=0, zorder=1)
                conf_ax.plot(levels, r.delta_margin, color=color, lw=_style.LINE_WIDTH,
                             marker="o", ms=3.5, mec="white", mew=0.4, zorder=3)

        # Human reference overlay (accuracy panel) or a graceful note when none exists.
        human_note, has_human = None, False
        if human in (True, "auto"):
            spec = _human_overlay_spec(results, condition_label, selected)
            if spec is not None:
                has_human = True
                ref, sf, human_thr = spec
                acc_ax.axvline(human_thr, color=_style.HUMAN_COLOR, ls="-.", lw=1.4, zorder=5,
                               label=(f"human {ref.human_paradigm.split('(')[0].strip()} "
                                      f"threshold @ {sf:g} cpd"))
                approx = "approx.; " if ref.approximate else ""
                # The sensitivity gap goes in the caption (keeps the legend compact).
                gap = ""
                if model_thresholds and human_thr > 0:
                    ratios = sorted(t / human_thr for t in model_thresholds)
                    gap = (f" On this axis the models need ~{ratios[0]:.0f}–{ratios[-1]:.0f}× "
                           f"more contrast than the human threshold.")
                # Name the *paradigm* difference, not just the metric/approximation (PRD §14):
                # human = grating DETECTION sensitivity; model = argmax CLASSIFICATION correctness.
                human_note = (
                    f"Human line = grating-DETECTION contrast sensitivity "
                    f"({ref.metric}, {sf:g} cpd; {approx}see citation). "
                    f"Model curves = argmax-CLASSIFICATION correctness. Different observer "
                    f"paradigms on a shared contrast axis — not identical tasks.{gap}"
                )
            else:
                human_note = f"No published human reference available for {suite_name}."

        acc_ax.axhline(target, color=_style.REFERENCE_COLOR, ls="--", lw=0.9, zorder=0)
        levels0 = np.concatenate(all_levels)
        if sigmoid == "weibull" and np.all(levels0 > 0) and not decreasing:
            acc_ax.set_xscale("log")
            _style.decimal_log_xticks(acc_ax, float(np.min(levels0)), float(np.max(levels0)))
        acc_ax.set_xlabel(xlabel)
        acc_ax.set_ylabel("P(correct)")
        acc_ax.set_ylim(0.0, 1.02)
        # Legend in the empty corner: below rising curves, above falling ones.
        acc_ax.legend(loc="upper right" if decreasing else "lower right")
        default_title = (f"{suite_name} — {condition_label}: models"
                         + (" vs. human threshold" if has_human else ""))
        acc_ax.set_title(title or default_title)
        _style.style_axes(acc_ax)
        if human_note:
            acc_ax.text(0.5, -0.16 if conf_ax is None else -0.30, human_note,
                        transform=acc_ax.transAxes, ha="center", va="top", fontsize=7.5,
                        color="#666666", wrap=True)

        if conf_ax is not None:
            conf_ax.axhline(0.0, color=_style.REFERENCE_COLOR, ls="--", lw=0.9, zorder=0)
            conf_ax.set_xlabel(xlabel)
            conf_ax.set_ylabel("Δ logit margin\n(from clean baseline)")
            conf_ax.set_title("confidence (baseline-relative Δ-margin — cross-model comparable)",
                              fontsize=10)
            _style.style_axes(conf_ax)

        fig.tight_layout()
    return fig


def plot_confidence_delta(results, *, condition=None, target=0.75, n_boot=300, ci=0.95, seed=0,
                          title=None):
    """Standalone styled figure of each model's baseline-relative Δ logit-margin curve.

    The cross-model-comparable confidence view (delta only, structurally guarded). Returns a
    matplotlib ``Figure`` in the shared publication style.
    """
    import matplotlib.pyplot as plt

    condition_label, curves = confidence_comparison(results, condition=condition, kind="delta",
                                                    n_boot=n_boot, ci=ci, seed=seed)
    decreasing = bool(results[0].decreasing)
    sigmoid = getattr(results[0].suite, "sigmoid", "weibull")
    xlabel = getattr(results[0].suite, "x_label", None) or "stimulus level"
    with _style.rc_context():
        fig, ax = plt.subplots(figsize=_style.FIG_DELTA)
        all_levels = []
        for i, c in enumerate(curves):
            color = _style.model_color(i)
            all_levels.append(np.asarray(c["levels"], float))
            ax.fill_between(c["levels"], c["delta_ci_low"], c["delta_ci_high"], color=color,
                            alpha=_style.BAND_ALPHA, lw=0, zorder=1)
            ax.plot(c["levels"], c["delta_margin"], color=color, lw=_style.LINE_WIDTH,
                    marker="o", ms=4, mec="white", mew=0.4, zorder=3, label=c["model"])
        ax.axhline(0.0, color=_style.REFERENCE_COLOR, ls="--", lw=0.9, zorder=0)
        levels0 = np.concatenate(all_levels)
        if sigmoid == "weibull" and np.all(levels0 > 0) and not decreasing:
            ax.set_xscale("log")
            _style.decimal_log_xticks(ax, float(np.min(levels0)), float(np.max(levels0)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Δ logit margin\n(from clean baseline)")
        ax.set_title(title or "Δ logit margin (baseline-relative — cross-model comparable)",
                     fontsize=11)
        ax.legend(loc="best")
        _style.style_axes(ax)
        fig.tight_layout()
    return fig
