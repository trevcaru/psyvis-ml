"""Contrast suite: chance-level wiring, conditions, metric validation, SF stub warning."""

import warnings

import numpy as np
import pytest

from psyvis_ml.suites import Condition, ContrastThreshold, Suite


def test_chance_level_is_k_over_num_classes():
    suite = ContrastThreshold()
    assert suite.chance_level(num_classes=1000, top_k=1) == pytest.approx(0.001)
    assert suite.chance_level(num_classes=10, top_k=5) == pytest.approx(0.5)
    # top_k >= num_classes saturates at 1.0.
    assert suite.chance_level(num_classes=4, top_k=8) == 1.0


def test_chance_level_validates_num_classes():
    with pytest.raises(ValueError):
        ContrastThreshold().chance_level(num_classes=0, top_k=1)


def test_single_condition_when_no_spatial_freqs():
    suite = ContrastThreshold(contrast_metric="rms")
    conds = suite.conditions()
    assert len(conds) == 1
    assert isinstance(conds[0], Condition)
    assert "rms" in conds[0].stimulus_name


def test_conditions_apply_target_contrast():
    suite = ContrastThreshold(contrast_metric="rms")
    fn = suite.conditions()[0].apply_stimulus
    img = 0.5 + 0.1 * np.random.default_rng(0).standard_normal(40)
    from psyvis_ml.stimuli.contrast import rms_contrast
    out = fn(img, 0.2, None)  # (image, level, rng) contract
    assert rms_contrast(out) == pytest.approx(0.2, abs=1e-9)


def test_michelson_metric_selected():
    suite = ContrastThreshold(contrast_metric="michelson")
    fn = suite.conditions()[0].apply_stimulus
    img = np.array([0.2, 0.8, 0.5, 0.6])
    from psyvis_ml.stimuli.contrast import michelson_contrast
    out = fn(img, 0.3, None)
    assert michelson_contrast(out) == pytest.approx(0.3, abs=1e-9)


def test_bad_metric_raises():
    with pytest.raises(ValueError):
        ContrastThreshold(contrast_metric="rms-plus")


def test_spatial_freqs_warn_and_make_distinct_labeled_conditions():
    with pytest.warns(RuntimeWarning, match="stub"):
        suite = ContrastThreshold(spatial_freqs=[1, 2, 4, 8])
    conds = suite.conditions()
    assert len(conds) == 4
    labels = [c.label for c in conds]
    assert len(set(labels)) == 4  # distinct labels
    names = [c.stimulus_name for c in conds]
    assert len(set(names)) == 4   # distinct stimulus names -> distinct config hashes


def test_no_spatial_freqs_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ContrastThreshold()  # must not warn


def test_contrast_suite_satisfies_protocol():
    assert isinstance(ContrastThreshold(), Suite)
    assert ContrastThreshold().sigmoid == "weibull"
