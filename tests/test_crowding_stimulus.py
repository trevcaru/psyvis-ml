"""Crowding compositor: placement geometry, flanker spacing, baseline, determinism."""

import numpy as np
import pytest

from psyvis_ml.stimuli.crowding import (
    composite_crowding,
    flanker_centers,
    flanker_offsets,
    paste_patch,
    target_center,
)


def _centroid(mask):
    ys, xs = np.where(mask)
    return np.array([ys.mean(), xs.mean()])


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def test_target_center_at_eccentricity_along_angle():
    shape = (40, 40)
    # angle 0 -> straight right (+col) from the canvas centre (19.5, 19.5).
    tc = target_center(shape, eccentricity=10, angle=0.0)
    assert tc == pytest.approx([19.5, 29.5])
    # angle 90 deg -> straight down (+row).
    tc90 = target_center(shape, eccentricity=10, angle=np.pi / 2)
    assert tc90 == pytest.approx([29.5, 19.5], abs=1e-9)


def test_flanker_offsets_nearest_at_plus_minus_spacing():
    assert flanker_offsets(8, 2) == [8.0, -8.0]
    assert flanker_offsets(8, 1) == [8.0]
    assert flanker_offsets(5, 4) == [5.0, -5.0, 10.0, -10.0]


def test_flanker_centers_symmetric_about_target():
    tc = np.array([20.0, 20.0])
    fcs = flanker_centers(tc, spacing=6, angle=0.0, n_flankers=2)
    assert fcs[0] == pytest.approx([20.0, 26.0])
    assert fcs[1] == pytest.approx([20.0, 14.0])
    # center-to-center target->flanker distance equals the spacing.
    assert np.hypot(*(fcs[0] - tc)) == pytest.approx(6.0)


def test_paste_patch_clips_off_canvas():
    canvas = np.zeros((10, 10))
    patch = np.ones((4, 4))
    paste_patch(canvas, patch, center=(0, 0))  # top-left corner: only a 2x2 quadrant lands
    assert canvas[:2, :2].sum() == 4.0
    assert canvas.sum() == 4.0  # the rest is clipped, not wrapped


# --------------------------------------------------------------------------- #
# Compositing: target placement and flanker spacing on the actual canvas
# --------------------------------------------------------------------------- #
def test_target_lands_at_specified_eccentricity():
    shape = (48, 48)
    target = np.ones((6, 6))
    ecc = 12
    canvas = composite_crowding(target, level=99, rng=None, eccentricity=ecc,
                                canvas_shape=shape, flanked=False)
    centroid = _centroid(canvas > 0)
    expected = target_center(shape, ecc, angle=0.0)
    assert centroid == pytest.approx(expected, abs=0.5)
    # eccentricity == distance from fixation (canvas centre).
    fix = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])
    assert np.hypot(*(centroid - fix)) == pytest.approx(ecc, abs=0.5)


@pytest.mark.parametrize("spacing", [6, 10, 16, 22])
def test_flanker_spacing_is_the_level(spacing):
    shape = (80, 80)
    target = np.ones((6, 6))
    flanker = np.full((4, 4), -1.0)  # reserved value so flankers are separable
    canvas = composite_crowding(target, level=spacing, rng=None, eccentricity=14,
                                canvas_shape=shape, flanked=True, n_flankers=2,
                                flanker_patch=flanker)
    from scipy.ndimage import center_of_mass, label

    lbl, n = label(canvas < 0)
    assert n == 2  # two distinct flankers
    coms = np.asarray(center_of_mass(np.ones_like(canvas), lbl, [1, 2]))
    measured = np.hypot(*(coms[0] - coms[1])) / 2.0  # half the inter-flanker distance
    assert measured == pytest.approx(spacing, abs=0.6)


def test_unflanked_baseline_has_no_flankers():
    shape = (48, 48)
    target = np.ones((6, 6))
    flanker = np.full((4, 4), -1.0)
    canvas = composite_crowding(target, level=10, rng=None, eccentricity=12,
                                canvas_shape=shape, flanked=False, flanker_patch=flanker)
    assert np.count_nonzero(canvas < 0) == 0  # baseline: target alone, no flankers
    assert np.count_nonzero(canvas > 0) == target.size


def test_self_flanking_default_uses_target_patch():
    shape = (60, 60)
    target = np.full((5, 5), 0.7)
    canvas = composite_crowding(target, level=12, rng=None, eccentricity=10,
                                canvas_shape=shape, flanked=True, n_flankers=2)
    # Default flankers are copies of the target: 3 patches of identical positive content.
    assert np.count_nonzero(canvas > 0) == 3 * target.size


# --------------------------------------------------------------------------- #
# Determinism vs. jitter
# --------------------------------------------------------------------------- #
def test_no_jitter_is_deterministic_ignoring_rng():
    shape = (48, 48)
    target = np.ones((6, 6))
    a = composite_crowding(target, 10, np.random.default_rng(0), eccentricity=12,
                           canvas_shape=shape)
    b = composite_crowding(target, 10, np.random.default_rng(999), eccentricity=12,
                           canvas_shape=shape)
    assert np.array_equal(a, b)


def test_position_jitter_draws_from_rng():
    shape = (48, 48)
    target = np.ones((6, 6))
    a = composite_crowding(target, 10, np.random.default_rng(1), eccentricity=12,
                           canvas_shape=shape, position_jitter=2.0)
    b = composite_crowding(target, 10, np.random.default_rng(2), eccentricity=12,
                           canvas_shape=shape, position_jitter=2.0)
    same = composite_crowding(target, 10, np.random.default_rng(1), eccentricity=12,
                              canvas_shape=shape, position_jitter=2.0)
    assert not np.array_equal(a, b)     # different seeds -> different placement
    assert np.array_equal(a, same)      # same seed -> reproducible


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        composite_crowding(np.ones(6), 10, None, eccentricity=5, canvas_shape=(40, 40))
    with pytest.raises(ValueError):
        composite_crowding(np.ones((4, 4)), 10, None, eccentricity=5, canvas_shape=(40,))
