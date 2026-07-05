"""Distractor compositor: centred placement, distractor spacing, baseline, determinism."""

import numpy as np
import pytest

from psyvis_ml.stimuli.distractor import (
    composite_distractor,
    distractor_centers,
    distractor_offsets,
    paste_patch,
    target_center,
)


def _centroid(mask):
    ys, xs = np.where(mask)
    return np.array([ys.mean(), xs.mean()])


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def test_target_center_defaults_to_canvas_centre():
    shape = (40, 40)
    assert target_center(shape) == pytest.approx([19.5, 19.5])              # offset 0 -> centred
    # A non-zero offset shifts the target along the angle axis.
    assert target_center(shape, target_offset=10, angle=0.0) == pytest.approx([19.5, 29.5])


def test_distractor_offsets_nearest_at_plus_minus_spacing():
    assert distractor_offsets(8, 2) == [8.0, -8.0]
    assert distractor_offsets(8, 1) == [8.0]
    assert distractor_offsets(5, 4) == [5.0, -5.0, 10.0, -10.0]


def test_distractor_centers_symmetric_about_target():
    tc = np.array([20.0, 20.0])
    dcs = distractor_centers(tc, spacing=6, angle=0.0, n_distractors=2)
    assert dcs[0] == pytest.approx([20.0, 26.0])
    assert dcs[1] == pytest.approx([20.0, 14.0])
    assert np.hypot(*(dcs[0] - tc)) == pytest.approx(6.0)   # center-to-center == spacing


def test_paste_patch_clips_off_canvas():
    canvas = np.zeros((10, 10))
    patch = np.ones((4, 4))
    paste_patch(canvas, patch, center=(0, 0))  # top-left corner: only a 2x2 quadrant lands
    assert canvas[:2, :2].sum() == 4.0
    assert canvas.sum() == 4.0  # the rest is clipped, not wrapped


# --------------------------------------------------------------------------- #
# Compositing: centred target and distractor spacing on the actual canvas
# --------------------------------------------------------------------------- #
def test_centred_target_lands_at_canvas_centre():
    shape = (48, 48)
    target = np.ones((6, 6))
    canvas = composite_distractor(target, level=99, rng=None, canvas_shape=shape,
                                  distracted=False)
    centroid = _centroid(canvas > 0)
    assert centroid == pytest.approx([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0], abs=0.5)


@pytest.mark.parametrize("spacing", [6, 10, 16, 22])
def test_distractor_spacing_is_the_level(spacing):
    shape = (80, 80)
    target = np.ones((6, 6))
    distractor = np.full((4, 4), -1.0)  # reserved value so distractors are separable
    canvas = composite_distractor(target, level=spacing, rng=None, canvas_shape=shape,
                                  distracted=True, n_distractors=2,
                                  distractor_patch=distractor)
    from scipy.ndimage import center_of_mass, label

    lbl, n = label(canvas < 0)
    assert n == 2  # two distinct distractors
    coms = np.asarray(center_of_mass(np.ones_like(canvas), lbl, [1, 2]))
    measured = np.hypot(*(coms[0] - coms[1])) / 2.0  # half the inter-distractor distance
    assert measured == pytest.approx(spacing, abs=0.6)


def test_undistracted_baseline_has_no_distractors():
    shape = (48, 48)
    target = np.ones((6, 6))
    distractor = np.full((4, 4), -1.0)
    canvas = composite_distractor(target, level=10, rng=None, canvas_shape=shape,
                                  distracted=False, distractor_patch=distractor)
    assert np.count_nonzero(canvas < 0) == 0  # baseline: target alone, no distractors
    assert np.count_nonzero(canvas > 0) == target.size


def test_self_distractor_default_uses_target_patch():
    shape = (60, 60)
    target = np.full((5, 5), 0.7)
    canvas = composite_distractor(target, level=12, rng=None, canvas_shape=shape,
                                  distracted=True, n_distractors=2)
    # Default distractors are copies of the target: 3 patches of identical positive content.
    assert np.count_nonzero(canvas > 0) == 3 * target.size


# --------------------------------------------------------------------------- #
# Determinism vs. jitter
# --------------------------------------------------------------------------- #
def test_no_jitter_is_deterministic_ignoring_rng():
    shape = (48, 48)
    target = np.ones((6, 6))
    a = composite_distractor(target, 10, np.random.default_rng(0), canvas_shape=shape)
    b = composite_distractor(target, 10, np.random.default_rng(999), canvas_shape=shape)
    assert np.array_equal(a, b)


def test_position_jitter_draws_from_rng():
    shape = (48, 48)
    target = np.ones((6, 6))
    a = composite_distractor(target, 10, np.random.default_rng(1), canvas_shape=shape,
                             position_jitter=2.0)
    b = composite_distractor(target, 10, np.random.default_rng(2), canvas_shape=shape,
                             position_jitter=2.0)
    same = composite_distractor(target, 10, np.random.default_rng(1), canvas_shape=shape,
                                position_jitter=2.0)
    assert not np.array_equal(a, b)     # different seeds -> different placement
    assert np.array_equal(a, same)      # same seed -> reproducible


def test_rgb_target_composites_onto_rgb_canvas():
    # Real natural images are (H, W, 3): the canvas must gain the channel axis and preserve
    # per-channel content so the suite runs on real RGB models.
    shape = (64, 64)
    target = np.zeros((8, 8, 3))
    target[..., 0] = 0.9  # a red patch
    distractor = np.zeros((6, 6, 3))
    distractor[..., 2] = 0.7  # blue distractors
    canvas = composite_distractor(target, level=14, rng=None, canvas_shape=shape,
                                  distracted=True, n_distractors=2,
                                  distractor_patch=distractor)
    assert canvas.shape == (64, 64, 3)
    assert canvas[..., 0].max() == pytest.approx(0.9)   # red target present
    assert canvas[..., 2].max() == pytest.approx(0.7)   # blue distractors present


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        composite_distractor(np.ones(6), 10, None, canvas_shape=(40, 40))
    with pytest.raises(ValueError):
        composite_distractor(np.ones((4, 4)), 10, None, canvas_shape=(40,))
