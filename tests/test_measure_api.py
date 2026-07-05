"""measure() API: levels helper, known-threshold recovery, multi-condition, plot smoke."""

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.datasets import synthetic_dataset

import synthetic as syn


# --------------------------------------------------------------------------- #
# linspace_levels
# --------------------------------------------------------------------------- #
def test_linspace_levels_linear():
    lv = pe.linspace_levels(0.0, 1.0, 5)
    assert np.allclose(lv, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_linspace_levels_log():
    lv = pe.linspace_levels(0.01, 1.0, 3, spacing="log")
    assert np.allclose(lv, [0.01, 0.1, 1.0])


def test_linspace_levels_validation():
    with pytest.raises(ValueError):
        pe.linspace_levels(0.0, 1.0, 4, spacing="log")   # non-positive start
    with pytest.raises(ValueError):
        pe.linspace_levels(0.0, 1.0, 3, spacing="quadratic")
    with pytest.raises(ValueError):
        pe.linspace_levels(0.0, 1.0, 0)


# --------------------------------------------------------------------------- #
# End-to-end recovery
# --------------------------------------------------------------------------- #
def _run_single(num_classes=8, n=800, seed=1):
    ds = synthetic_dataset(n=n, num_classes=num_classes)
    model = syn.make_contrast_observer(num_classes)
    levels = pe.linspace_levels(0.03, 0.6, 11, spacing="log")
    suite = pe.suites.ContrastThreshold(contrast_metric="rms")
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed)


def test_measure_recovers_known_contrast_threshold():
    num_classes = 8
    res = _run_single(num_classes=num_classes)
    recovered = res.threshold(0.75)
    known = syn.contrast_true_threshold(num_classes, 0.75)
    assert res.fit().converged
    assert abs(recovered - known) / known < 0.1, (recovered, known)
    # Fitted alpha matches the planted alpha too.
    assert abs(res.fit().params["alpha"] - syn.CONTRAST_ALPHA) / syn.CONTRAST_ALPHA < 0.1


def test_chance_level_wired_into_guess_rate():
    num_classes = 8
    res = _run_single(num_classes=num_classes)
    assert res.chance_level == pytest.approx(1.0 / num_classes)
    # The suite's chance level becomes the fitter's guess (lower asymptote).
    assert res.fit().params["guess"] == pytest.approx(1.0 / num_classes)


def test_single_condition_returns_scalars_not_dicts():
    res = _run_single()
    assert np.isscalar(res.threshold()) or isinstance(res.threshold(), float)
    assert not isinstance(res.fit(), dict)
    assert not isinstance(res.bundle, dict)


# --------------------------------------------------------------------------- #
# Multi-condition
# --------------------------------------------------------------------------- #
def test_multi_condition_returns_dicts_keyed_by_label():
    num_classes = 8
    ds = synthetic_dataset(n=300, num_classes=num_classes)
    model = syn.make_contrast_observer(num_classes)
    levels = pe.linspace_levels(0.03, 0.6, 9, spacing="log")
    with pytest.warns(RuntimeWarning):  # SF stub warning
        suite = pe.suites.ContrastThreshold(spatial_freqs=[1, 2, 4])
    res = pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=3)

    thr = res.threshold()
    assert isinstance(thr, dict) and set(thr) == set(res.labels)
    assert isinstance(res.fit(), dict)
    assert isinstance(res.bundle, dict)
    assert set(res.fits) == set(res.bundles) == set(res.labels)
    # Stub: SF conditions apply identical manipulation -> identical thresholds.
    vals = list(thr.values())
    assert all(v == pytest.approx(vals[0]) for v in vals)


# --------------------------------------------------------------------------- #
# Reproducibility through the API
# --------------------------------------------------------------------------- #
def test_measure_is_reproducible_with_seed():
    a = _run_single(seed=42)
    b = _run_single(seed=42)
    assert a.bundle.config_hash == b.bundle.config_hash
    assert a.bundle.n_correct == b.bundle.n_correct


# --------------------------------------------------------------------------- #
# Plot smoke test (Agg backend via conftest)
# --------------------------------------------------------------------------- #
def test_plot_returns_figure_no_display():
    res = _run_single(n=200)
    fig = res.plot(n_boot=30, seed=0)
    assert fig.__class__.__name__ == "Figure"
    ax = fig.axes[0]
    assert ax.get_xlabel() == "stimulus level"
    assert ax.get_ylabel() == "P(correct)"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_multi_condition_smoke():
    num_classes = 8
    ds = synthetic_dataset(n=150, num_classes=num_classes)
    model = syn.make_contrast_observer(num_classes)
    levels = pe.linspace_levels(0.03, 0.6, 7, spacing="log")
    with pytest.warns(RuntimeWarning):
        suite = pe.suites.ContrastThreshold(spatial_freqs=[1, 2])
    res = pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=1)
    fig = res.plot(n_boot=20, show_ci=False)
    # Two panels by default: the accuracy curve and the within-model confidence (margin) panel.
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "P(correct)"
    assert "margin" in fig.axes[1].get_ylabel()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_confidence_can_be_disabled():
    res = _run_single(n=150)
    fig = res.plot(n_boot=20, show_ci=False, show_confidence=False)
    assert len(fig.axes) == 1  # accuracy only
    import matplotlib.pyplot as plt
    plt.close(fig)
