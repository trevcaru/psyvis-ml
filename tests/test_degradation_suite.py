"""Degradation suite: chance wiring, decreasing-logistic fit, severity-threshold recovery."""

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset
from psyvis_ml.suites import DegradationSuite, Suite

import synthetic as syn

NUM_CLASSES = 8
PATCH = 24  # class encoded across columns; large enough that occlusion rarely erases a column


def _dataset(n=320):
    images, labels = syn.make_patch_data(n, size=PATCH, num_classes=NUM_CLASSES)
    return Dataset(images=images, labels=labels, num_classes=NUM_CLASSES, name="patches")


def _measure_occlusion(n=320, num=11, seed=1):
    ds = _dataset(n)
    model = syn.make_degradation_observer(NUM_CLASSES, fill=0.0)
    # scatter occlusion -> realized occluded fraction matches the swept severity closely.
    suite = DegradationSuite("occlusion", fill=0.0, block=False)
    levels = pe.linspace_levels(0.05, 0.6, num)
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed)


# --------------------------------------------------------------------------- #
# Protocol / config / chance
# --------------------------------------------------------------------------- #
def test_satisfies_protocol_and_decreasing_logistic():
    suite = DegradationSuite("blur")
    assert isinstance(suite, Suite)
    assert suite.sigmoid == "logistic"
    assert suite.decreasing is True  # performance falls with severity


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        DegradationSuite("swirl")


def test_chance_level_is_k_over_num_classes():
    suite = DegradationSuite("gaussian_noise")
    assert suite.chance_level(num_classes=8, top_k=1) == pytest.approx(0.125)
    with pytest.raises(ValueError):
        suite.chance_level(num_classes=0, top_k=1)


@pytest.mark.parametrize("kind", ["gaussian_noise", "blur", "occlusion", "lowpass", "highpass"])
def test_single_condition_per_kind_with_distinct_name(kind):
    conds = DegradationSuite(kind).conditions()
    assert len(conds) == 1
    assert kind in conds[0].stimulus_name
    assert conds[0].metadata["kind"] == kind


def test_params_folded_into_stimulus_name():
    a = DegradationSuite("occlusion", block=True).conditions()[0].stimulus_name
    b = DegradationSuite("occlusion", block=False).conditions()[0].stimulus_name
    assert a != b  # distinct settings hash distinctly


# --------------------------------------------------------------------------- #
# End-to-end: recover the planted severity threshold from a falling curve
# --------------------------------------------------------------------------- #
def test_recovers_known_severity_threshold():
    res = _measure_occlusion()
    assert res.fit().converged
    recovered = res.threshold(0.75)
    known = syn.deg_true_threshold(NUM_CLASSES, 0.75)
    assert abs(recovered - known) / known < 0.15, (recovered, known)


def test_performance_falls_with_severity():
    res = _measure_occlusion()
    frac = np.asarray(res.bundle.n_correct) / np.asarray(res.bundle.n_trials)
    assert frac[0] > 0.9      # low severity -> near ceiling
    assert frac[-1] < 0.3     # high severity -> near chance
    assert np.all(np.diff(frac) <= 0.1)  # loosely monotone decreasing


def test_slope_is_negative_for_falling_curve():
    res = _measure_occlusion()
    assert res.slope(0.75) < 0.0  # dP/dseverity < 0


def test_chance_level_wired_into_guess_rate():
    res = _measure_occlusion()
    # chance is the *high-severity* lower asymptote of the decreasing curve.
    assert res.chance_level == pytest.approx(1.0 / NUM_CLASSES)
    assert res.fit().params["guess"] == pytest.approx(1.0 / NUM_CLASSES)


def test_decreasing_suite_requires_logistic():
    # measure() rejects a decreasing suite that declares a non-logistic sigmoid.
    class BadSuite(DegradationSuite):
        sigmoid = "weibull"

    ds = _dataset(40)
    model = syn.make_degradation_observer(NUM_CLASSES, fill=0.0)
    with pytest.raises(ValueError, match="logistic"):
        pe.measure(model=model, suite=BadSuite("occlusion", fill=0.0, block=False),
                   dataset=ds, levels=pe.linspace_levels(0.05, 0.6, 5), seed=0)
