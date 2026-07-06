"""Paired over-images bootstrap analysis: difference test, goodness-of-fit, direction."""

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset, synthetic_dataset

import synthetic as syn

NC = 8


def _contrast_result(alpha, name, *, n=200, seed=1):
    ds = synthetic_dataset(n=n, num_classes=NC)
    model = syn.make_contrast_observer(NC, alpha=alpha)
    return pe.measure(model=model, suite=pe.suites.ContrastThreshold(contrast_metric="rms"),
                      dataset=ds, levels=pe.linspace_levels(0.03, 0.6, 9, spacing="log"),
                      seed=seed, model_name=name)


# --------------------------------------------------------------------------- #
# Paired difference test (the correct test for deterministic models)
# --------------------------------------------------------------------------- #
def test_difference_ci_excludes_zero_when_models_differ():
    a, b = _contrast_result(0.12, "A"), _contrast_result(0.30, "B")
    an = pe.analyze_suite([a, b], n_boot=400, seed=0)
    assert an.threshold["A"] < an.threshold["B"]        # A is the more sensitive model
    assert an.threshold_diff_excludes_zero              # a reliable difference
    lo, hi = an.threshold_diff_ci
    assert hi < 0                                       # A − B is reliably negative


def test_difference_ci_includes_zero_when_models_identical():
    # The contrast observer is deterministic, so two runs of the same observer are identical:
    # every resample gives an exactly zero paired difference -> CI cannot exclude zero.
    a, b = _contrast_result(0.15, "A"), _contrast_result(0.15, "B")
    an = pe.analyze_suite([a, b], n_boot=200, seed=0)
    assert an.threshold_diff == pytest.approx(0.0)
    assert not an.threshold_diff_excludes_zero


# --------------------------------------------------------------------------- #
# Goodness of fit
# --------------------------------------------------------------------------- #
def test_goodness_of_fit_high_for_clean_synthetic():
    a = _contrast_result(0.15, "A")
    gof = pe.goodness_of_fit(a.bundle, a.fit())
    assert gof["converged"]
    assert gof["r2"] > 0.9
    assert gof["at_bound"] == []


# --------------------------------------------------------------------------- #
# Per-model CIs, structure, to_dict
# --------------------------------------------------------------------------- #
def test_per_model_cis_and_to_dict():
    a, b = _contrast_result(0.12, "A"), _contrast_result(0.30, "B")
    an = pe.analyze_suite([a, b], n_boot=200, seed=0)
    assert set(an.models) == {"A", "B"}
    for m in an.models:
        lo, hi = an.threshold_ci[m]
        assert np.isfinite(lo) and np.isfinite(hi) and lo <= an.threshold[m] <= hi
        assert np.isfinite(an.slope_ci[m][0])
        assert an.gof[m]["converged"]
    d = an.to_dict()
    assert isinstance(d["threshold_diff_ci"], list) and isinstance(d["threshold_ci"]["A"], list)
    assert "delta_margin_diff_excludes_zero" in d and "margin_slope" in d


def test_requires_exactly_two_models():
    a = _contrast_result(0.15, "A")
    with pytest.raises(ValueError, match="exactly two"):
        pe.analyze_suite([a], n_boot=10)


# --------------------------------------------------------------------------- #
# Decreasing suite (degradation): direction + Δ-margin endpoint handled
# --------------------------------------------------------------------------- #
def _degradation_result(s50, name, *, n=200, seed=1):
    images, labels = syn.make_patch_data(n, size=24, num_classes=NC)
    ds = Dataset(images=images, labels=labels, num_classes=NC)
    model = syn.make_degradation_observer(NC, fill=0.0, s50=s50)
    return pe.measure(model=model, suite=pe.suites.DegradationSuite("occlusion", fill=0.0,
                                                                    block=False),
                      dataset=ds, levels=pe.linspace_levels(0.05, 0.6, 9), seed=seed,
                      model_name=name)


def test_decreasing_suite_analysis_direction_and_divergence():
    a, b = _degradation_result(0.30, "A"), _degradation_result(0.45, "B")
    an = pe.analyze_suite([a, b], n_boot=300, seed=0)
    assert an.decreasing is True
    assert an.metadata["baseline_index"] == 0            # clean = level 0 for a decreasing suite
    assert np.isfinite(an.delta_margin_diff)
    assert set(an.margin_slope) == {"A", "B"}
    # B tolerates more occlusion (higher s50) -> higher severity threshold.
    assert an.threshold["B"] > an.threshold["A"]
