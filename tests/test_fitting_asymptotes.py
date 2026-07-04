"""Asymptote handling, input validation, threshold inversion, and slope units."""

import numpy as np
import pytest

from psyvis_ml.fitting import fit_psychometric


def _weibull_p(x, alpha, beta, guess, lapse):
    return guess + (1 - guess - lapse) * (1 - np.exp(-((x / alpha) ** beta)))


# --------------------------------------------------------------------------- #
# Fixed vs. free asymptotes
# --------------------------------------------------------------------------- #
def test_fixed_asymptotes_are_held_and_not_counted():
    levels = np.geomspace(0.02, 0.6, 10)
    n = np.full_like(levels, 200)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.5, 0.02)
    fit = fit_psychometric(levels, k, n, guess_rate=0.5, lapse_rate=0.02)
    assert fit.params["guess"] == 0.5
    assert fit.params["lapse"] == 0.02
    assert fit.n_params == 2


def test_free_lapse_is_recovered():
    alpha, beta, guess, lapse = 0.10, 3.0, 0.5, 0.06
    levels = np.geomspace(0.02, 1.5, 14)  # extend high so the ceiling identifies lapse
    n = np.full_like(levels, 500)
    k = n * _weibull_p(levels, alpha, beta, guess, lapse)
    fit = fit_psychometric(levels, k, n, guess_rate=0.5, lapse_rate=None)
    assert fit.n_params == 3
    assert fit.params["lapse"] == pytest.approx(lapse, abs=0.02)


def test_free_guess_is_recovered():
    alpha, beta, guess, lapse = 0.10, 3.0, 0.25, 0.02
    levels = np.geomspace(0.005, 0.6, 14)  # extend low so the floor identifies guess
    n = np.full_like(levels, 500)
    k = n * _weibull_p(levels, alpha, beta, guess, lapse)
    fit = fit_psychometric(levels, k, n, guess_rate=None, lapse_rate=0.02)
    assert fit.n_params == 3
    assert fit.params["guess"] == pytest.approx(guess, abs=0.03)


def test_both_asymptotes_free_counts_four_params():
    levels = np.geomspace(0.005, 1.5, 16)
    n = np.full_like(levels, 500)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.3, 0.05)
    fit = fit_psychometric(levels, k, n, guess_rate=None, lapse_rate=None)
    assert fit.n_params == 4


def test_invalid_asymptote_value_raises():
    levels = np.geomspace(0.02, 0.6, 6)
    n = np.full_like(levels, 100)
    k = n * 0.7
    with pytest.raises(ValueError):
        fit_psychometric(levels, k, n, lapse_rate=0.9)  # lapse must be <= 0.5


# --------------------------------------------------------------------------- #
# Weibull positivity vs. logistic signed axes
# --------------------------------------------------------------------------- #
def test_weibull_rejects_nonpositive_levels():
    levels = np.array([-0.1, 0.0, 0.1, 0.2])
    n = np.full_like(levels, 100)
    k = n * 0.6
    with pytest.raises(ValueError, match="positive"):
        fit_psychometric(levels, k, n, sigmoid="weibull")


def test_logistic_accepts_signed_levels():
    levels = np.linspace(-3.0, 3.0, 9)
    n = np.full_like(levels, 200)
    p = 0.5 + 0.48 / (1 + np.exp(-1.5 * (levels - 0.0)))
    fit = fit_psychometric(levels, n * p, n, sigmoid="logistic")
    assert fit.converged


def test_predict_warns_for_weibull_nonpositive_x():
    levels = np.geomspace(0.02, 0.6, 8)
    n = np.full_like(levels, 100)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.5, 0.02)
    fit = fit_psychometric(levels, k, n, sigmoid="weibull")
    with pytest.warns(RuntimeWarning):
        out = fit.predict(np.array([-1.0, 0.1]))
    assert np.isnan(out[0])


# --------------------------------------------------------------------------- #
# threshold() inversion, including out-of-range targets
# --------------------------------------------------------------------------- #
def test_threshold_inverts_predict():
    levels = np.geomspace(0.02, 0.6, 10)
    n = np.full_like(levels, 300)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.5, 0.02)
    fit = fit_psychometric(levels, k, n, sigmoid="weibull")
    for target in (0.55, 0.6, 0.75, 0.9):
        x = fit.threshold(target)
        assert np.isfinite(x)
        assert float(fit.predict(x)) == pytest.approx(target, abs=1e-8)


def test_threshold_out_of_range_returns_nan_with_warning():
    levels = np.geomspace(0.02, 0.6, 10)
    n = np.full_like(levels, 300)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.5, 0.02)
    fit = fit_psychometric(levels, k, n, guess_rate=0.5, lapse_rate=0.02)
    # Below the guess floor (0.5) and above the 1 - lapse ceiling (0.98).
    for bad in (0.4, 0.99):
        with pytest.warns(RuntimeWarning):
            assert np.isnan(fit.threshold(bad))


# --------------------------------------------------------------------------- #
# slope() units
# --------------------------------------------------------------------------- #
def test_slope_log_equals_x_times_linear():
    levels = np.geomspace(0.02, 0.6, 10)
    n = np.full_like(levels, 300)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.5, 0.02)
    fit = fit_psychometric(levels, k, n, sigmoid="weibull")
    x = fit.threshold(0.75)
    lin = fit.slope(0.75, units="linear")
    log = fit.slope(0.75, units="log")
    assert log == pytest.approx(x * lin, rel=1e-9)


def test_slope_log_nan_for_nonpositive_threshold():
    # Logistic with a negative threshold location: the 0.75 crossing sits at x < 0,
    # so a log-units slope is undefined.
    levels = np.linspace(-5.0, 1.0, 11)
    n = np.full_like(levels, 300)
    p = 0.5 + 0.48 / (1 + np.exp(-1.0 * (levels - (-2.0))))
    fit = fit_psychometric(levels, n * p, n, sigmoid="logistic",
                           guess_rate=0.5, lapse_rate=0.02)
    assert fit.threshold(0.75) < 0
    assert np.isfinite(fit.slope(0.75, units="linear"))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(fit.slope(0.75, units="log"))


def test_slope_log_base10_scales_by_ln10():
    # log10 units: dP/d(log10 x) = ln(10) * x * dP/dx. Vision science reports log10 contrast.
    levels = np.geomspace(0.02, 0.6, 10)
    n = np.full_like(levels, 300)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.5, 0.02)
    fit = fit_psychometric(levels, k, n, sigmoid="weibull")
    x = fit.threshold(0.75)
    lin = fit.slope(0.75, units="linear")
    log_e = fit.slope(0.75, units="log")  # default base = e
    log10 = fit.slope(0.75, units="log", base=10)
    assert log_e == pytest.approx(x * lin, rel=1e-9)
    assert log10 == pytest.approx(np.log(10) * x * lin, rel=1e-9)
    assert log10 == pytest.approx(np.log(10) * log_e, rel=1e-9)


def test_slope_invalid_base_raises():
    levels = np.geomspace(0.02, 0.6, 8)
    n = np.full_like(levels, 100)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.5, 0.02)
    fit = fit_psychometric(levels, k, n)
    for bad in (0.0, 1.0, -2.0):
        with pytest.raises(ValueError):
            fit.slope(0.75, units="log", base=bad)


def test_slope_invalid_units_raises():
    levels = np.geomspace(0.02, 0.6, 8)
    n = np.full_like(levels, 100)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.5, 0.02)
    fit = fit_psychometric(levels, k, n)
    with pytest.raises(ValueError):
        fit.slope(0.75, units="decibel")


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        fit_psychometric([0.1, 0.2], [1, 2, 3], [10, 10, 10])


def test_n_correct_exceeds_trials_raises():
    with pytest.raises(ValueError):
        fit_psychometric([0.1, 0.2], [11, 2], [10, 10])


def test_summary_is_flat_and_has_at_bound():
    levels = np.geomspace(0.02, 0.6, 10)
    n = np.full_like(levels, 300)
    k = n * _weibull_p(levels, 0.1, 3.0, 0.5, 0.02)
    s = fit_psychometric(levels, k, n).summary()
    assert isinstance(s, dict)
    assert "at_bound" in s and isinstance(s["at_bound"], list)
    for key in ("alpha", "slope", "guess", "lapse", "threshold_at_0.75",
                "slope_at_0.75_linear", "slope_at_0.75_log", "log_likelihood",
                "n_params", "n_obs", "aic", "bic", "converged"):
        assert key in s
    # Flat: no nested dicts.
    assert not any(isinstance(v, dict) for v in s.values())
