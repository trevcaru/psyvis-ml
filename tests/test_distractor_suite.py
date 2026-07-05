"""Distractor-robustness suite: chance wiring, conditions, baseline, threshold recovery."""

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset
from psyvis_ml.suites import DistractorRobustness, Suite

import synthetic as syn

NUM_CLASSES = 8
PATCH = 8          # class encoded as one of PATCH columns -> PATCH >= NUM_CLASSES
CANVAS = (96, 96)
DISTRACTOR = np.full((4, 4), -1.0)  # reserved value so the observer can locate distractors


def _dataset(n=240):
    images, labels = syn.make_patch_data(n, size=PATCH, num_classes=NUM_CLASSES)
    return Dataset(images=images, labels=labels, num_classes=NUM_CLASSES, name="patches")


def _measure(*, include_baseline=False, n=240, num=11, seed=1):
    ds = _dataset(n)
    model = syn.make_distractor_observer(NUM_CLASSES)
    suite = DistractorRobustness(canvas_shape=CANVAS, distractor_patch=DISTRACTOR,
                                 include_baseline=include_baseline)
    levels = pe.linspace_levels(6, 26, num)
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed)


# --------------------------------------------------------------------------- #
# Protocol / chance level / conditions
# --------------------------------------------------------------------------- #
def test_satisfies_suite_protocol_and_sigmoid():
    suite = DistractorRobustness()
    assert isinstance(suite, Suite)
    assert suite.sigmoid == "weibull"  # spacing is a positive, rising axis


def test_chance_level_is_k_over_num_classes():
    suite = DistractorRobustness()
    assert suite.chance_level(num_classes=8, top_k=1) == pytest.approx(0.125)
    assert suite.chance_level(num_classes=10, top_k=2) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        suite.chance_level(num_classes=0, top_k=1)


def test_conditions_distractors_plus_baseline():
    suite = DistractorRobustness(canvas_shape=CANVAS, include_baseline=True)
    conds = suite.conditions()
    assert len(conds) == 2  # distractors + undistracted baseline
    labels = [c.label for c in conds]
    names = [c.stimulus_name for c in conds]
    assert len(set(labels)) == 2 and len(set(names)) == 2
    baselines = [c for c in conds if c.metadata["baseline"]]
    assert len(baselines) == 1 and not baselines[0].metadata["distracted"]


def test_no_baseline_when_disabled():
    conds = DistractorRobustness(include_baseline=False).conditions()
    assert len(conds) == 1 and conds[0].metadata["distracted"]


def test_bad_canvas_shape_raises():
    with pytest.raises(ValueError):
        DistractorRobustness(canvas_shape=(96,))


# --------------------------------------------------------------------------- #
# End-to-end: recover the planted distractor-distance threshold
# --------------------------------------------------------------------------- #
def test_recovers_known_distractor_threshold():
    res = _measure()
    assert res.fit().converged
    recovered = res.threshold(0.75)
    known = syn.distractor_true_threshold(NUM_CLASSES, 0.75)
    assert abs(recovered - known) / known < 0.12, (recovered, known)


def test_performance_rises_with_spacing():
    res = _measure()
    frac = np.asarray(res.bundle.n_correct) / np.asarray(res.bundle.n_trials)
    assert frac[0] < 0.4     # tight spacing -> distractor interferes, near chance
    assert frac[-1] > 0.9    # wide spacing -> target clears
    assert np.all(np.diff(frac) >= -0.1)  # loosely monotone increasing


def test_chance_level_wired_into_guess_rate():
    res = _measure()
    assert res.chance_level == pytest.approx(1.0 / NUM_CLASSES)
    assert res.fit().params["guess"] == pytest.approx(1.0 / NUM_CLASSES)


def test_undistracted_baseline_is_flat_and_high():
    res = _measure(include_baseline=True, n=160, num=7)
    base_label = next(lbl for lbl in res.labels if "undistracted" in lbl)
    b = res.bundles[base_label]
    frac = np.asarray(b.n_correct) / np.asarray(b.n_trials)
    assert frac.min() > 0.9                      # clean ceiling everywhere
    assert frac.max() - frac.min() < 0.1         # spacing has no effect when undistracted
