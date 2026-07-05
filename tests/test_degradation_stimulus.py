"""Degradation manipulations: noise/blur/occlusion/filter math on synthetic images."""

import numpy as np
import pytest

from psyvis_ml.stimuli.degradation import (
    add_gaussian_noise,
    gaussian_blur,
    occlude,
    spatial_frequency_filter,
)


# --------------------------------------------------------------------------- #
# Gaussian noise (stochastic)
# --------------------------------------------------------------------------- #
def test_noise_realized_std_matches_severity():
    img = np.full((200, 200), 0.5)
    out = add_gaussian_noise(img, 0.1, np.random.default_rng(0))
    resid = out - img
    assert np.std(resid) == pytest.approx(0.1, rel=0.05)
    assert np.mean(resid) == pytest.approx(0.0, abs=0.01)


def test_noise_zero_severity_is_identity():
    img = np.array([[0.2, 0.8], [0.4, 0.6]])
    assert np.array_equal(add_gaussian_noise(img, 0.0, np.random.default_rng(0)), img)


def test_noise_is_reproducible_and_seed_dependent():
    img = np.zeros((16, 16))
    a = add_gaussian_noise(img, 0.2, np.random.default_rng(7))
    b = add_gaussian_noise(img, 0.2, np.random.default_rng(7))
    c = add_gaussian_noise(img, 0.2, np.random.default_rng(8))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_noise_clip_range_bounds_output():
    img = np.full((50, 50), 0.5)
    out = add_gaussian_noise(img, 0.5, np.random.default_rng(0), clip_range=(0.0, 1.0))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_negative_noise_severity_raises():
    with pytest.raises(ValueError):
        add_gaussian_noise(np.zeros((4, 4)), -0.1, None)


# --------------------------------------------------------------------------- #
# Gaussian blur (deterministic)
# --------------------------------------------------------------------------- #
def test_blur_zero_sigma_is_identity():
    img = np.random.default_rng(0).random((16, 16))
    assert np.array_equal(gaussian_blur(img, 0.0), img)


def test_blur_preserves_constant_image():
    img = np.full((20, 20), 0.37)
    out = gaussian_blur(img, 2.0)
    assert np.allclose(out, 0.37)


def test_blur_reduces_high_frequency_variance():
    # A checkerboard is pure high frequency; blurring must shrink its variance.
    idx = np.add.outer(np.arange(32), np.arange(32))
    board = (idx % 2).astype(float)
    v0 = board.var()
    assert gaussian_blur(board, 1.0).var() < v0
    assert gaussian_blur(board, 3.0).var() < gaussian_blur(board, 1.0).var()


def test_blur_is_deterministic_ignoring_rng():
    img = np.random.default_rng(1).random((16, 16))
    a = gaussian_blur(img, 1.5, np.random.default_rng(0))
    b = gaussian_blur(img, 1.5, np.random.default_rng(999))
    assert np.array_equal(a, b)


def test_blur_needs_spatial_image():
    with pytest.raises(ValueError):
        gaussian_blur(np.arange(10.0), 1.0)


# --------------------------------------------------------------------------- #
# Occlusion (stochastic)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block", [True, False])
def test_occlusion_fraction_matches_severity(block):
    img = np.full((20, 20), 0.5)
    out = occlude(img, 0.25, np.random.default_rng(0), fill=-5.0, block=block)
    frac = np.mean(out == -5.0)
    assert frac == pytest.approx(0.25, abs=0.02)


def test_occlusion_block_is_contiguous_square():
    img = np.full((20, 20), 0.5)
    out = occlude(img, 0.25, np.random.default_rng(3), fill=-5.0, block=True)
    rows = np.where(np.any(out == -5.0, axis=1))[0]
    cols = np.where(np.any(out == -5.0, axis=0))[0]
    # A contiguous block: occupied rows/cols are consecutive runs.
    assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
    assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1))
    assert len(rows) == len(cols)  # square


def test_occlusion_scatter_exact_pixel_count():
    img = np.zeros((10, 10))
    out = occlude(img, 0.3, np.random.default_rng(0), fill=1.0, block=False)
    assert np.count_nonzero(out == 1.0) == 30  # round(0.3 * 100)


def test_occlusion_zero_severity_is_identity():
    img = np.full((8, 8), 0.5)
    assert np.array_equal(occlude(img, 0.0, np.random.default_rng(0)), img)


def test_occlusion_reproducible_and_seed_dependent():
    img = np.full((16, 16), 0.5)
    a = occlude(img, 0.2, np.random.default_rng(1), fill=0.0)
    b = occlude(img, 0.2, np.random.default_rng(1), fill=0.0)
    c = occlude(img, 0.2, np.random.default_rng(2), fill=0.0)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_occlusion_severity_out_of_range_raises():
    with pytest.raises(ValueError):
        occlude(np.zeros((4, 4)), 1.5, None)


# --------------------------------------------------------------------------- #
# Spatial-frequency filtering (deterministic)
# --------------------------------------------------------------------------- #
def test_filter_zero_severity_is_identity():
    img = np.random.default_rng(0).random((16, 16))
    assert np.array_equal(spatial_frequency_filter(img, 0.0, mode="lowpass"), img)
    assert np.array_equal(spatial_frequency_filter(img, 0.0, mode="highpass"), img)


def test_highpass_removes_dc_mean_goes_to_zero():
    img = np.random.default_rng(1).random((32, 32)) + 3.0  # large DC offset
    out = spatial_frequency_filter(img, 0.1, mode="highpass")
    assert np.mean(out) == pytest.approx(0.0, abs=1e-9)


def test_lowpass_full_severity_leaves_only_mean():
    img = np.random.default_rng(2).random((16, 16))
    out = spatial_frequency_filter(img, 1.0, mode="lowpass")
    assert np.allclose(out, img.mean())  # only the DC component survives


def test_lowpass_reduces_high_frequency_energy():
    idx = np.add.outer(np.arange(32), np.arange(32))
    board = (idx % 2).astype(float)
    out = spatial_frequency_filter(board, 0.5, mode="lowpass")
    assert out.var() < board.var()


def test_filter_preserves_dc_under_lowpass():
    img = np.random.default_rng(3).random((16, 16))
    out = spatial_frequency_filter(img, 0.5, mode="lowpass")
    assert np.mean(out) == pytest.approx(np.mean(img), abs=1e-9)  # DC kept by lowpass


def test_filter_bad_mode_raises():
    with pytest.raises(ValueError):
        spatial_frequency_filter(np.zeros((4, 4)), 0.5, mode="bandpass")
