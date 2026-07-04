"""Contrast stimulus: RMS/Michelson math, target-contrast setting, edge cases, SF stub."""

import numpy as np
import pytest

from psyvis_ml.stimuli.contrast import apply_contrast, michelson_contrast, rms_contrast


def test_rms_contrast_math():
    img = np.array([0.4, 0.6, 0.4, 0.6])
    # mean = 0.5, std = 0.1 -> RMS = 0.2
    assert rms_contrast(img) == pytest.approx(0.2)


def test_michelson_contrast_math():
    img = np.array([0.2, 0.8, 0.5])
    # (0.8 - 0.2) / (0.8 + 0.2) = 0.6
    assert michelson_contrast(img) == pytest.approx(0.6)


def test_contrast_metrics_zero_on_uniform():
    img = np.full(10, 0.5)
    assert rms_contrast(img) == 0.0
    assert michelson_contrast(img) == 0.0


@pytest.mark.parametrize("metric,measure", [("rms", rms_contrast),
                                            ("michelson", michelson_contrast)])
@pytest.mark.parametrize("level", [0.05, 0.1, 0.25, 0.5])
def test_apply_contrast_sets_target_exactly(metric, measure, level):
    rng = np.random.default_rng(0)
    img = 0.5 + 0.1 * rng.standard_normal(64)
    img = np.clip(img, 0.05, 0.95)
    out = apply_contrast(img, level, None, metric=metric)
    assert measure(out) == pytest.approx(level, abs=1e-9)


def test_rms_preserves_mean_michelson_preserves_midpoint():
    rng = np.random.default_rng(1)
    img = 0.5 + 0.08 * rng.standard_normal(50)
    mean0 = float(np.mean(img))
    mid0 = 0.5 * (float(np.max(img)) + float(np.min(img)))

    out_rms = apply_contrast(img, 0.3, None, metric="rms")
    assert float(np.mean(out_rms)) == pytest.approx(mean0, abs=1e-9)

    out_mich = apply_contrast(img, 0.3, None, metric="michelson")
    mid1 = 0.5 * (float(np.max(out_mich)) + float(np.min(out_mich)))
    assert mid1 == pytest.approx(mid0, abs=1e-9)


def test_level_zero_gives_uniform_image():
    img = np.array([0.3, 0.5, 0.7, 0.5])
    out = apply_contrast(img, 0.0, None, metric="rms")
    assert np.allclose(out, out[0])  # uniform
    assert rms_contrast(out) == pytest.approx(0.0)


def test_uniform_image_returned_unchanged():
    img = np.full(8, 0.5)
    out = apply_contrast(img, 0.3, None, metric="rms")  # nothing to scale
    assert np.allclose(out, img)


def test_negative_level_raises():
    img = np.array([0.4, 0.6])
    with pytest.raises(ValueError):
        apply_contrast(img, -0.1, None)


def test_bad_metric_raises():
    img = np.array([0.4, 0.6])
    with pytest.raises(ValueError):
        apply_contrast(img, 0.2, None, metric="rms-plus")


def test_rng_is_ignored_deterministic():
    img = np.array([0.4, 0.6, 0.45, 0.55])
    a = apply_contrast(img, 0.15, np.random.default_rng(0), metric="rms")
    b = apply_contrast(img, 0.15, np.random.default_rng(999), metric="rms")
    assert np.array_equal(a, b)


def test_spatial_freq_stub_is_accepted_and_ignored():
    img = np.array([0.4, 0.6, 0.45, 0.55])
    base = apply_contrast(img, 0.2, None, metric="rms")
    with_sf = apply_contrast(img, 0.2, None, metric="rms", spatial_freq=4.0)
    assert np.array_equal(base, with_sf)  # stub: SF does not change the manipulation


def test_clip_range_bounds_output():
    img = np.array([0.1, 0.9, 0.5])
    out = apply_contrast(img, 0.9, None, metric="michelson", clip_range=(0.2, 0.8))
    assert out.min() >= 0.2 - 1e-12
    assert out.max() <= 0.8 + 1e-12
