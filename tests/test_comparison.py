"""Multi-model comparison: shared axis, per-result direction, CI bands, mismatch guards."""

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.comparison import compare_results, resolve_comparison
from psyvis_ml.datasets import Dataset, synthetic_dataset
from psyvis_ml.suites import DegradationSuite

import synthetic as syn

NUM_CLASSES = 8


def _contrast_result(alpha, *, name, n=200, seed=1):
    ds = synthetic_dataset(n=n, num_classes=NUM_CLASSES)
    model = syn.make_contrast_observer(NUM_CLASSES, alpha=alpha)
    levels = pe.linspace_levels(0.03, 0.6, 9, spacing="log")
    suite = pe.suites.ContrastThreshold(contrast_metric="rms")
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed,
                      model_name=name)


def _degradation_result(*, name, n=200, seed=1):
    images, labels = syn.make_patch_data(n, size=24, num_classes=NUM_CLASSES)
    ds = Dataset(images=images, labels=labels, num_classes=NUM_CLASSES)
    model = syn.make_degradation_observer(NUM_CLASSES, fill=0.0)
    suite = DegradationSuite("occlusion", fill=0.0, block=False)
    levels = pe.linspace_levels(0.05, 0.6, 9)
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed,
                      model_name=name)


def _fill_collections(ax):
    return [c for c in ax.collections if "Poly" in type(c).__name__
            or "FillBetween" in type(c).__name__]


# --------------------------------------------------------------------------- #
# Two models on one shared axis
# --------------------------------------------------------------------------- #
def test_compare_two_models_shared_axis_with_ci_bands():
    a = _contrast_result(0.15, name="A")
    b = _contrast_result(0.28, name="B")
    fig = a.compare([b], n_boot=40, seed=0, show_ci=True)
    ax = fig.axes[0]
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert {"A", "B"} <= labels                       # both models on the same axis
    assert len(_fill_collections(ax)) >= 2            # a CI band per model
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_compare_reflects_increasing_direction_log_axis():
    a = _contrast_result(0.15, name="A")
    b = _contrast_result(0.28, name="B")
    fig = a.compare([b], n_boot=20, seed=0, show_ci=False)
    # Contrast is a positive, rising axis -> log x-scale.
    assert fig.axes[0].get_xscale() == "log"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_compare_reflects_decreasing_direction_linear_axis():
    a = _degradation_result(name="A")
    b = _degradation_result(name="B", seed=2)
    assert a.decreasing and b.decreasing
    fig = compare_results([a, b], n_boot=20, seed=0, show_ci=False)
    ax = fig.axes[0]
    assert ax.get_xscale() == "linear"                # decreasing severity axis stays linear
    # No human curve for degradation -> graceful note, models only.
    note = " ".join(t.get_text() for t in ax.texts)
    assert "No published human reference" in note
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_summary_carries_direction_and_ci():
    a = _degradation_result(name="A")
    label, rows = pe.comparison_summary([a], n_boot=40, seed=0)
    assert len(rows) == 1
    assert rows[0]["direction"] == "decreasing"
    assert rows[0]["slope"] < 0                        # falling curve -> negative slope
    assert np.isfinite(rows[0]["threshold_ci_low"])


# --------------------------------------------------------------------------- #
# Mismatch guards
# --------------------------------------------------------------------------- #
def test_mismatched_suites_raise():
    contrast = _contrast_result(0.15, name="A")
    degradation = _degradation_result(name="B")
    with pytest.raises(ValueError, match="different suites"):
        contrast.compare([degradation])


def test_mismatched_direction_raises():
    a = _contrast_result(0.15, name="A")
    b = _contrast_result(0.2, name="B")
    b.decreasing = True  # simulate an inconsistent declared direction
    with pytest.raises(ValueError, match="direction"):
        resolve_comparison([a, b])


def test_missing_condition_raises():
    a = _contrast_result(0.15, name="A")
    b = _contrast_result(0.2, name="B")
    with pytest.raises(ValueError, match="not found"):
        resolve_comparison([a, b], condition="contrast[michelson]")  # wrong metric label


def test_multi_condition_requires_explicit_condition():
    ds = synthetic_dataset(n=120, num_classes=NUM_CLASSES)
    model = syn.make_contrast_observer(NUM_CLASSES)
    levels = pe.linspace_levels(0.03, 0.6, 7, spacing="log")
    with pytest.warns(RuntimeWarning):  # SF stub warning
        suite = pe.suites.ContrastThreshold(spatial_freqs=[1, 2])
    multi = pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=1,
                       model_name="A")
    with pytest.raises(ValueError, match="multiple conditions"):
        resolve_comparison([multi])
