"""Parametric-bootstrap CI tests: reproducibility, bracketing, failure reporting, coverage."""

import numpy as np
import pytest

from psyvis_ml.fitting import fit_psychometric


def _weibull_p(x, alpha, beta, guess, lapse):
    return guess + (1 - guess - lapse) * (1 - np.exp(-((x / alpha) ** beta)))


def _make_fit(seed=0, n_per=250):
    rng = np.random.default_rng(seed)
    alpha, beta, guess, lapse = 0.10, 3.0, 0.5, 0.02
    levels = np.geomspace(0.02, 0.6, 11)
    n = np.full_like(levels, n_per, dtype=int)
    k = rng.binomial(n, _weibull_p(levels, alpha, beta, guess, lapse)).astype(float)
    return fit_psychometric(levels, k, n.astype(float), sigmoid="weibull",
                            guess_rate=guess, lapse_rate=lapse)


def test_bootstrap_is_reproducible_with_seed():
    fit = _make_fit()
    a = fit.bootstrap_ci("threshold", n_boot=300, seed=42)
    b = fit.bootstrap_ci("threshold", n_boot=300, seed=42)
    assert a.low == b.low and a.high == b.high
    assert a.n_failed == b.n_failed


def test_bootstrap_brackets_point_estimate():
    fit = _make_fit()
    point = fit.threshold(0.75)
    ci = fit.bootstrap_ci("threshold", n_boot=500, seed=1)
    assert ci.low <= point <= ci.high
    assert ci.high > ci.low
    assert ci.fail_fraction == pytest.approx(ci.n_failed / ci.n_boot)


def test_bootstrap_reports_fields():
    fit = _make_fit()
    ci = fit.bootstrap_ci("slope", n_boot=200, seed=3)
    assert ci.n_boot == 200
    assert 0.0 <= ci.fail_fraction <= 1.0
    assert ci.quantity == "slope"
    assert ci.ci == 0.95


def test_bootstrap_all_fail_returns_nan_bounds():
    fit = _make_fit()
    # target below the guess floor -> threshold is NaN for every resample.
    with pytest.warns(RuntimeWarning):
        ci = fit.bootstrap_ci("threshold", target=0.40, n_boot=50, seed=0)
    assert np.isnan(ci.low) and np.isnan(ci.high)
    assert ci.n_failed == ci.n_boot
    assert ci.fail_fraction == 1.0


def test_bootstrap_ci_bad_args():
    fit = _make_fit()
    with pytest.raises(ValueError):
        fit.bootstrap_ci("threshold", ci=1.5)
    with pytest.raises(ValueError):
        fit.bootstrap_ci("nonsense", n_boot=10)


@pytest.mark.slow
def test_bootstrap_coverage_of_true_threshold():
    # Over many simulated datasets, a 95% CI should contain the true threshold ~95% of
    # the time. Loose lower bound (0.75) tolerates Monte-Carlo noise at 25 reps.
    alpha, beta, guess, lapse = 0.10, 3.0, 0.5, 0.02
    levels = np.geomspace(0.02, 0.6, 11)
    n = np.full_like(levels, 200, dtype=int)
    true_thr = alpha * (-np.log(1 - (0.75 - guess) / (1 - guess - lapse))) ** (1 / beta)

    n_reps, hits = 25, 0
    rng = np.random.default_rng(2024)
    for _ in range(n_reps):
        k = rng.binomial(n, _weibull_p(levels, alpha, beta, guess, lapse)).astype(float)
        fit = fit_psychometric(levels, k, n.astype(float), sigmoid="weibull",
                               guess_rate=guess, lapse_rate=lapse)
        ci = fit.bootstrap_ci("threshold", n_boot=120, seed=int(rng.integers(1 << 30)))
        if ci.low <= true_thr <= ci.high:
            hits += 1
    coverage = hits / n_reps
    assert coverage >= 0.75, f"bootstrap coverage too low: {coverage:.2f}"
