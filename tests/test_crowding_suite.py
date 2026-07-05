"""Crowding suite: chance wiring, conditions, baseline, and critical-spacing recovery."""

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset
from psyvis_ml.suites import CrowdingSuite, Suite

import synthetic as syn

NUM_CLASSES = 8
PATCH = 8          # class encoded as one of PATCH columns -> PATCH >= NUM_CLASSES
CANVAS = (96, 96)
FLANKER = np.full((4, 4), -1.0)  # reserved value so the observer can locate flankers


def _dataset(n=240):
    images, labels = syn.make_patch_data(n, size=PATCH, num_classes=NUM_CLASSES)
    return Dataset(images=images, labels=labels, num_classes=NUM_CLASSES, name="patches")


def _measure(ecc, *, include_baseline=False, n=240, num=11, seed=1):
    ds = _dataset(n)
    model = syn.make_crowding_observer(NUM_CLASSES)
    suite = CrowdingSuite(eccentricities=[ecc], canvas_shape=CANVAS, flanker_patch=FLANKER,
                          include_baseline=include_baseline)
    levels = pe.linspace_levels(6, 26, num)
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed)


# --------------------------------------------------------------------------- #
# Protocol / chance level / conditions
# --------------------------------------------------------------------------- #
def test_satisfies_suite_protocol_and_sigmoid():
    suite = CrowdingSuite(eccentricities=[8])
    assert isinstance(suite, Suite)
    assert suite.sigmoid == "weibull"  # spacing is a positive, rising axis


def test_chance_level_is_k_over_num_classes():
    suite = CrowdingSuite(eccentricities=[8])
    assert suite.chance_level(num_classes=8, top_k=1) == pytest.approx(0.125)
    assert suite.chance_level(num_classes=10, top_k=2) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        suite.chance_level(num_classes=0, top_k=1)


def test_conditions_flanked_plus_baseline_per_eccentricity():
    suite = CrowdingSuite(eccentricities=[8, 16], canvas_shape=CANVAS, include_baseline=True)
    conds = suite.conditions()
    assert len(conds) == 4  # (flanked + unflanked) x 2 eccentricities
    labels = [c.label for c in conds]
    names = [c.stimulus_name for c in conds]
    assert len(set(labels)) == 4 and len(set(names)) == 4
    baselines = [c for c in conds if c.metadata["baseline"]]
    assert len(baselines) == 2 and all(not c.metadata["flanked"] for c in baselines)


def test_no_baseline_when_disabled():
    suite = CrowdingSuite(eccentricities=[8, 16], include_baseline=False)
    conds = suite.conditions()
    assert len(conds) == 2 and all(c.metadata["flanked"] for c in conds)


def test_empty_eccentricities_raises():
    with pytest.raises(ValueError):
        CrowdingSuite(eccentricities=[])


# --------------------------------------------------------------------------- #
# End-to-end: recover the planted critical spacing
# --------------------------------------------------------------------------- #
def test_recovers_known_critical_spacing():
    res = _measure(ecc=10)
    assert res.fit().converged
    recovered = res.threshold(0.75)
    known = syn.crowding_true_threshold(NUM_CLASSES, 0.75)
    assert abs(recovered - known) / known < 0.12, (recovered, known)


def test_crowding_function_rises_with_spacing():
    res = _measure(ecc=10)
    frac = np.asarray(res.bundle.n_correct) / np.asarray(res.bundle.n_trials)
    assert frac[0] < 0.4     # tight spacing -> crowded, near chance
    assert frac[-1] > 0.9    # wide spacing -> target recovers
    # loosely monotone increasing
    assert np.all(np.diff(frac) >= -0.1)


def test_chance_level_wired_into_guess_rate():
    res = _measure(ecc=10)
    assert res.chance_level == pytest.approx(1.0 / NUM_CLASSES)
    assert res.fit().params["guess"] == pytest.approx(1.0 / NUM_CLASSES)


def test_unflanked_baseline_is_flat_and_high():
    res = _measure(ecc=10, include_baseline=True, n=160, num=7)
    base_label = next(lbl for lbl in res.labels if "unflanked" in lbl)
    b = res.bundles[base_label]
    frac = np.asarray(b.n_correct) / np.asarray(b.n_trials)
    assert frac.min() > 0.9                      # uncrowded ceiling everywhere
    assert frac.max() - frac.min() < 0.1         # spacing has no effect when unflanked
