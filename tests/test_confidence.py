"""Confidence readout: margin extraction, baseline-relative Δ, threshold recovery, guards.

Covers the logit-margin scoring, the descriptive readout (Δ-margin normalization, margin-
crossing threshold, bootstrap CIs, direction handling), the structural block on absolute
cross-model overlays, and that all three suites emit the readout with no per-suite special-casing.
"""

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.confidence import (
    ConfidenceReadout,
    confidence_readout,
    require_relative_for_multimodel,
)
from psyvis_ml.datasets import Dataset, synthetic_dataset
from psyvis_ml.suites import DegradationSuite, DistractorRobustness
from psyvis_ml.sweep import RunBundle, score_logits

import synthetic as syn

NUM_CLASSES = 8


# --------------------------------------------------------------------------- #
# 1. Margin extraction from raw logits
# --------------------------------------------------------------------------- #
def test_score_logits_margin_rank_softmax():
    logits = np.array([[3.0, 1.0, 0.0],    # target 0: margin 3-1=2, rank 1
                       [0.0, 2.0, 5.0],    # target 2: margin 5-2=3, rank 1
                       [1.0, 4.0, 2.0]])   # target 0: margin 1-4=-3, rank 3 (4 and 2 higher)
    labels = np.array([0, 2, 0])
    s = score_logits(logits, labels)
    assert np.allclose(s.margin, [2.0, 3.0, -3.0])
    assert np.array_equal(s.target_rank, [1, 1, 3])
    assert np.allclose(s.target_logit, [3.0, 5.0, 1.0])
    assert np.array_equal(s.correct, [True, True, False])  # top-1 argmax
    # max_softmax matches a manual stable softmax (reference only).
    row0 = np.exp([3, 1, 0]) / np.exp([3, 1, 0]).sum()
    assert s.max_softmax[0] == pytest.approx(row0.max())
    assert np.all((s.max_softmax > 0) & (s.max_softmax <= 1))


def test_score_logits_margin_sign_matches_correctness():
    # Margin > 0 exactly when the target is top-1.
    logits = np.array([[5.0, 4.9], [4.9, 5.0]])
    s = score_logits(logits, np.array([0, 0]))
    assert s.margin[0] > 0 and s.correct[0]
    assert s.margin[1] < 0 and not s.correct[1]


def test_score_logits_needs_two_classes():
    with pytest.raises(ValueError, match="at least 2 classes"):
        score_logits(np.array([[1.0]]), np.array([0]))


# --------------------------------------------------------------------------- #
# 2/4/5. Readout: baseline-relative Δ, direction, threshold crossing, bootstrap CIs
# --------------------------------------------------------------------------- #
def _bundle(levels, mean_per_level, *, jitter=None, seed=0):
    """A RunBundle whose per-level margins have the given means (optionally with image jitter)."""
    levels = np.asarray(levels, float)
    n_images = 200
    rng = np.random.default_rng(seed)
    rows = []
    for m in mean_per_level:
        col = np.full(n_images, float(m))
        if jitter:
            col = col + rng.normal(0.0, jitter, n_images)
            col = col - col.mean() + m  # keep the exact mean, add spread
        rows.append(col)
    margins = np.vstack(rows)
    per = {"margin": margins,
           "target_rank": np.ones_like(margins, dtype=int),
           "max_softmax": np.full_like(margins, 0.5)}
    return RunBundle(seed=0, config_hash="h", library_version="0", levels=tuple(levels),
                     n_correct=tuple([0] * len(levels)),
                     n_trials=tuple([n_images] * len(levels)), top_k=1, per_image=per)


def test_delta_margin_anchored_at_clean_baseline_by_direction():
    levels = [1.0, 2.0, 3.0, 4.0, 5.0]
    means = [2.0, 1.0, 0.0, -1.0, -2.0]   # declines as level rises
    # Decreasing suite -> clean baseline is level 0 (index 0).
    dec = confidence_readout(_bundle(levels, means), decreasing=True, n_boot=50)
    assert dec.baseline_index == 0
    assert dec.delta_margin[0] == pytest.approx(0.0)
    assert np.allclose(dec.delta_margin, [0, -1, -2, -3, -4])
    # Increasing suite -> clean baseline is the top level (last index).
    inc = confidence_readout(_bundle(levels, means), decreasing=False, n_boot=50)
    assert inc.baseline_index == len(levels) - 1
    assert inc.delta_margin[-1] == pytest.approx(0.0)


def test_confidence_threshold_and_slope_crossing():
    levels = [1.0, 2.0, 3.0, 4.0, 5.0]
    means = [2.0, 1.0, 0.0, -1.0, -2.0]   # crosses margin=0 at level 3, slope -1
    r = confidence_readout(_bundle(levels, means), decreasing=True, n_boot=50)
    assert r.confidence_threshold == pytest.approx(3.0)
    assert r.margin_slope == pytest.approx(-1.0)


def test_confidence_threshold_interpolates_between_levels():
    levels = [0.0, 1.0]
    means = [1.0, -1.0]   # crosses 0 at the midpoint
    r = confidence_readout(_bundle(levels, means), decreasing=True, n_boot=20)
    assert r.confidence_threshold == pytest.approx(0.5)


def test_no_crossing_returns_nan():
    r = confidence_readout(_bundle([1.0, 2.0, 3.0], [3.0, 2.5, 2.0]), decreasing=True, n_boot=20)
    assert np.isnan(r.confidence_threshold)


def test_bootstrap_cis_bracket_the_mean():
    levels = [1.0, 2.0, 3.0]
    means = [2.0, 0.0, -2.0]
    r = confidence_readout(_bundle(levels, means, jitter=1.0, seed=3), decreasing=True,
                           n_boot=500, seed=1)
    assert np.all(r.margin_ci_low < r.mean_margin + 1e-9)
    assert np.all(r.margin_ci_high > r.mean_margin - 1e-9)
    assert np.all(r.margin_ci_low < r.margin_ci_high)          # non-degenerate with spread
    # Δ CI is anchored: exactly 0 width at the baseline level.
    assert r.delta_ci_low[r.baseline_index] == pytest.approx(0.0)
    assert r.delta_ci_high[r.baseline_index] == pytest.approx(0.0)


def test_readout_requires_margins():
    b = RunBundle(seed=0, config_hash="h", library_version="0", levels=(1.0,),
                  n_correct=(0,), n_trials=(1,), top_k=1)  # no per_image
    with pytest.raises(ValueError, match="no per-image margins"):
        confidence_readout(b, decreasing=False)


# --------------------------------------------------------------------------- #
# 3. End-to-end margin-crossing threshold recovery through measure()
# --------------------------------------------------------------------------- #
def test_measure_recovers_known_margin_crossing():
    ds = synthetic_dataset(n=300, num_classes=NUM_CLASSES)
    crossing, slope = 0.20, 10.0
    model = syn.make_margin_observer(NUM_CLASSES, crossing=crossing, slope=slope)
    levels = pe.linspace_levels(0.05, 0.45, 9)   # 0.20 is a grid point
    suite = pe.suites.ContrastThreshold(contrast_metric="rms")
    res = pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=0)

    conf = res.confidence(n_boot=100, seed=0)
    assert isinstance(conf, ConfidenceReadout)
    # Realized contrast == level exactly, so margin = slope*(level - crossing): threshold exact.
    assert conf.confidence_threshold == pytest.approx(crossing, abs=1e-6)
    assert conf.margin_slope == pytest.approx(slope, rel=1e-6)
    # Increasing suite: clean baseline is the top level, Δ anchored there.
    assert conf.baseline_index == len(levels) - 1
    assert conf.delta_margin[-1] == pytest.approx(0.0)
    # Convenience accessor agrees.
    assert res.confidence_threshold(n_boot=100, seed=0) == pytest.approx(crossing, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. Structural guard: no absolute-margin cross-model overlays
# --------------------------------------------------------------------------- #
def test_require_relative_guard():
    require_relative_for_multimodel("delta", 3)       # ok
    require_relative_for_multimodel("absolute", 1)    # single model: absolute is within-model, ok
    with pytest.raises(ValueError, match="not comparable across models"):
        require_relative_for_multimodel("absolute", 2)


def _contrast_result(name, seed=1):
    ds = synthetic_dataset(n=150, num_classes=NUM_CLASSES)
    model = syn.make_contrast_observer(NUM_CLASSES)
    levels = pe.linspace_levels(0.03, 0.6, 7, spacing="log")
    suite = pe.suites.ContrastThreshold(contrast_metric="rms")
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed,
                      model_name=name)


def test_confidence_comparison_blocks_absolute_and_returns_delta():
    a, b = _contrast_result("A"), _contrast_result("B", seed=2)
    with pytest.raises(ValueError, match="not comparable across models"):
        pe.confidence_comparison([a, b], kind="absolute")
    _, curves = pe.confidence_comparison([a, b], kind="delta", n_boot=50)
    assert len(curves) == 2
    for c in curves:  # each model normalized to its own clean baseline -> Δ == 0 there
        assert c["delta_margin"][c["baseline_index"]] == pytest.approx(0.0)


def test_compare_results_blocks_absolute_confidence_overlay():
    a, b = _contrast_result("A"), _contrast_result("B", seed=2)
    with pytest.raises(ValueError, match="not comparable across models"):
        a.compare([b], confidence_kind="absolute", n_boot=20)


def test_compare_adds_delta_margin_panel():
    a, b = _contrast_result("A"), _contrast_result("B", seed=2)
    fig = a.compare([b], n_boot=20, seed=0)
    assert len(fig.axes) == 2                                  # accuracy + Δ-margin panel
    assert "Δ" in fig.axes[1].get_ylabel() or "margin" in fig.axes[1].get_ylabel()
    import matplotlib.pyplot as plt
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. All three suites emit the readout with no per-suite special-casing
# --------------------------------------------------------------------------- #
def _patch_dataset(n, size):
    images, labels = syn.make_patch_data(n, size=size, num_classes=NUM_CLASSES)
    return Dataset(images=images, labels=labels, num_classes=NUM_CLASSES, name="patches")


def _contrast_res():
    ds = synthetic_dataset(n=160, num_classes=NUM_CLASSES)
    return pe.measure(model=syn.make_contrast_observer(NUM_CLASSES),
                      suite=pe.suites.ContrastThreshold(contrast_metric="rms"), dataset=ds,
                      levels=pe.linspace_levels(0.03, 0.6, 7, spacing="log"), seed=0)


def _degradation_res():
    ds = _patch_dataset(160, 24)
    return pe.measure(model=syn.make_degradation_observer(NUM_CLASSES, fill=0.0),
                      suite=DegradationSuite("occlusion", fill=0.0, block=False), dataset=ds,
                      levels=pe.linspace_levels(0.05, 0.6, 7), seed=0)


def _distractor_res():
    ds = _patch_dataset(160, 16)
    suite = DistractorRobustness(distractor_patch=np.full((8, 8), -1.0), canvas_shape=(48, 48),
                                 n_distractors=1, include_baseline=False)
    return pe.measure(model=syn.make_distractor_observer(NUM_CLASSES), suite=suite, dataset=ds,
                      levels=pe.linspace_levels(2, 14, 7), seed=0)


@pytest.mark.parametrize("make_res", [_contrast_res, _degradation_res, _distractor_res])
def test_all_suites_emit_margin_readout(make_res):
    res = make_res()
    conf = res.confidence(n_boot=50, seed=0)
    r = conf if isinstance(conf, ConfidenceReadout) else next(iter(conf.values()))
    assert r.mean_margin.shape == res.levels.shape
    assert r.delta_margin[r.baseline_index] == pytest.approx(0.0)   # Δ anchored, every suite
    assert np.all(np.isfinite(r.mean_margin))
    assert r.decreasing == res.decreasing                          # direction threaded through
