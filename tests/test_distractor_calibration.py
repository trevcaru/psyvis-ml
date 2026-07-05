"""Distractor-spacing calibration: detect a floor→ceiling transition, place levels, recover it.

Validated on a synthetic observer whose P(correct) rises with distractor spacing (a known
distance threshold), and on a flat observer with no transition in the probed range — so the
calibration is shown to *detect a transition when one exists* and *report its absence honestly*
when it does not.
"""

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset
from psyvis_ml.distractor_calibration import calibrate_distractor_spacing

import synthetic as syn

NUM_CLASSES = 8
PATCH = 8
CANVAS = (96, 96)
DISTRACTOR = np.full((4, 4), -1.0)  # reserved value so the observer can locate distractors
# Extreme probes: 4 px (tight -> interferes) up to 40 px (wide -> cleared).
PROBES = [4.0, 8.0, 12.0, 18.0, 26.0, 40.0]


def _dataset(n=240):
    images, labels = syn.make_patch_data(n, size=PATCH, num_classes=NUM_CLASSES)
    return Dataset(images=images, labels=labels, num_classes=NUM_CLASSES, name="patches")


def _calibrate(model, *, n=240, n_levels=11, seed=1):
    return calibrate_distractor_spacing(
        model, _dataset(n), canvas_shape=CANVAS, probe_spacings=PROBES,
        distractor_patch=DISTRACTOR, n_levels=n_levels, seed=seed)


# --------------------------------------------------------------------------- #
# Transition present
# --------------------------------------------------------------------------- #
def test_detects_transition():
    cal = _calibrate(syn.make_distractor_observer(NUM_CLASSES))
    assert cal.transition_detected
    # Floor near chance, ceiling well above it (a real clean target, not a trivial 1.0).
    assert cal.interference_floor < 0.35
    assert cal.baseline_ceiling > 0.85
    assert cal.baseline_ceiling - cal.interference_floor > 0.4


def test_places_levels_bracketing_true_threshold():
    cal = _calibrate(syn.make_distractor_observer(NUM_CLASSES))
    known = syn.distractor_true_threshold(NUM_CLASSES, 0.75)  # ~13.1 px
    assert cal.low < known < cal.high                        # the transition is inside the span
    assert len(cal.levels) == 11
    assert cal.levels[0] == pytest.approx(cal.low)
    assert cal.levels[-1] == pytest.approx(cal.high)


def test_calibrated_sweep_recovers_threshold_with_ci():
    model = syn.make_distractor_observer(NUM_CLASSES)
    cal = _calibrate(model)
    suite = pe.suites.DistractorRobustness(canvas_shape=CANVAS, distractor_patch=DISTRACTOR,
                                           include_baseline=False)
    res = pe.measure(model=model, suite=suite, dataset=_dataset(), levels=cal.levels, seed=2)
    known = syn.distractor_true_threshold(NUM_CLASSES, 0.75)
    recovered = res.threshold(0.75)
    assert res.fit().converged
    assert abs(recovered - known) / known < 0.15, (recovered, known)
    ci = res.fit().bootstrap_ci("threshold", target=0.75, n_boot=200, seed=0)
    assert np.isfinite(ci.low) and np.isfinite(ci.high) and ci.low < ci.high


# --------------------------------------------------------------------------- #
# No transition in the probed range -> honest negative
# --------------------------------------------------------------------------- #
def test_flat_observer_reports_no_transition():
    # alpha=1 px: the observer is already cleared at every probe (>=4 px) -> no floor.
    flat = syn.make_distractor_observer(NUM_CLASSES, alpha=1.0)
    cal = _calibrate(flat)
    assert not cal.transition_detected
    assert "NO transition" in cal.notes
    # Levels still span the full probed range so the (flat) curve is reported, not hidden.
    assert cal.low == pytest.approx(min(PROBES)) and cal.high == pytest.approx(max(PROBES))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_requires_at_least_two_probes():
    with pytest.raises(ValueError, match="two probe"):
        calibrate_distractor_spacing(syn.make_distractor_observer(NUM_CLASSES), _dataset(40),
                                     canvas_shape=CANVAS, probe_spacings=[10.0])


def test_rejects_nonpositive_probes():
    with pytest.raises(ValueError, match="positive"):
        calibrate_distractor_spacing(syn.make_distractor_observer(NUM_CLASSES), _dataset(40),
                                     canvas_shape=CANVAS, probe_spacings=[0.0, 10.0])
