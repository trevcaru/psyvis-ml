"""Distractor-size calibration: detect a ceiling→floor transition, place sizes, recover it.

Validated on a synthetic observer whose P(correct) falls with distractor size (a known size
threshold), and on a flat observer with no size effect — so the calibration is shown to *detect
a ceiling→floor transition when one exists* and *report its absence honestly* when it does not.
"""

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset
from psyvis_ml.distractor_calibration import calibrate_distractor_size

import synthetic as syn

NUM_CLASSES = 8
PATCH = 16
CANVAS = (48, 48)
DISTRACTOR = np.full((8, 8), -1.0)  # reserved value; resized to each probe size
# Extreme probes: 2 px (small -> near ceiling) up to 14 px (large -> floor).
PROBES = [2.0, 5.0, 8.0, 11.0, 14.0]


def _dataset(n=240):
    images, labels = syn.make_patch_data(n, size=PATCH, num_classes=NUM_CLASSES)
    return Dataset(images=images, labels=labels, num_classes=NUM_CLASSES, name="patches")


def _calibrate(model, *, n=240, n_levels=11, seed=1):
    return calibrate_distractor_size(
        model, _dataset(n), canvas_shape=CANVAS, probe_sizes=PROBES,
        distractor_patch=DISTRACTOR, n_distractors=1, n_levels=n_levels, seed=seed)


# --------------------------------------------------------------------------- #
# Transition present
# --------------------------------------------------------------------------- #
def test_detects_transition():
    cal = _calibrate(syn.make_distractor_observer(NUM_CLASSES))
    assert cal.transition_detected
    assert cal.small_size_accuracy > 0.85   # small distractor -> near ceiling
    assert cal.large_size_accuracy < 0.35   # large distractor -> floor
    assert cal.small_size_accuracy - cal.large_size_accuracy > 0.4


def test_places_levels_bracketing_true_threshold():
    cal = _calibrate(syn.make_distractor_observer(NUM_CLASSES))
    known = syn.distractor_true_threshold(NUM_CLASSES, 0.75)  # ~6.75 px
    assert cal.low < known < cal.high                        # the transition is inside the span
    assert len(cal.levels) == 11
    assert cal.levels[0] == pytest.approx(cal.low)
    assert cal.levels[-1] == pytest.approx(cal.high)


def test_calibrated_sweep_recovers_threshold_with_ci():
    model = syn.make_distractor_observer(NUM_CLASSES)
    cal = _calibrate(model)
    suite = pe.suites.DistractorRobustness(distractor_patch=DISTRACTOR, canvas_shape=CANVAS,
                                           n_distractors=1, include_baseline=False)
    res = pe.measure(model=model, suite=suite, dataset=_dataset(), levels=cal.levels, seed=2)
    known = syn.distractor_true_threshold(NUM_CLASSES, 0.75)
    recovered = res.threshold(0.75)
    assert res.fit().converged
    assert abs(recovered - known) / known < 0.15, (recovered, known)
    ci = res.fit().bootstrap_ci("threshold", target=0.75, n_boot=200, seed=0)
    assert np.isfinite(ci.low) and np.isfinite(ci.high) and ci.low < ci.high


# --------------------------------------------------------------------------- #
# No transition -> honest negative
# --------------------------------------------------------------------------- #
def test_flat_observer_reports_no_transition():
    # s50 far beyond the probe range: the observer is at ceiling for every size -> flat.
    flat = syn.make_distractor_observer(NUM_CLASSES, s50=1000.0)
    cal = _calibrate(flat)
    assert not cal.transition_detected
    assert "NO ceiling->floor transition" in cal.notes
    assert cal.low == pytest.approx(min(PROBES)) and cal.high == pytest.approx(max(PROBES))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_requires_at_least_two_probes():
    with pytest.raises(ValueError, match="two probe sizes"):
        calibrate_distractor_size(syn.make_distractor_observer(NUM_CLASSES), _dataset(40),
                                  canvas_shape=CANVAS, probe_sizes=[8.0])


def test_rejects_nonpositive_probes():
    with pytest.raises(ValueError, match="positive"):
        calibrate_distractor_size(syn.make_distractor_observer(NUM_CLASSES), _dataset(40),
                                  canvas_shape=CANVAS, probe_sizes=[0.0, 8.0])
