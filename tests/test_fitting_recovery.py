"""Synthetic-recovery tests: fit data with known parameters, assert recovery in tolerance.

Two regimes per sigmoid:
  * noiseless expected counts (k = n * P_true) -> MLE recovers true params near-exactly;
  * sampled counts with a fixed seed -> recovery within a looser tolerance.
"""

import numpy as np
import pytest

from psyvis_ml.fitting import fit_psychometric


def _weibull_p(x, alpha, beta, guess, lapse):
    return guess + (1 - guess - lapse) * (1 - np.exp(-((x / alpha) ** beta)))


def _logistic_p(x, alpha, beta, guess, lapse):
    return guess + (1 - guess - lapse) / (1 + np.exp(-beta * (x - alpha)))


def test_weibull_recovery_noiseless():
    alpha, beta, guess, lapse = 0.10, 3.0, 0.5, 0.02
    levels = np.geomspace(0.02, 0.6, 12)
    n = np.full_like(levels, 400)
    p = _weibull_p(levels, alpha, beta, guess, lapse)
    k = n * p  # noiseless expected counts

    fit = fit_psychometric(levels, k, n, sigmoid="weibull",
                           guess_rate=guess, lapse_rate=lapse)
    assert fit.converged
    assert fit.params["alpha"] == pytest.approx(alpha, rel=0.02)
    assert fit.params["slope"] == pytest.approx(beta, rel=0.03)
    # Derived 75%-correct threshold recovered too.
    true_thr = alpha * (-np.log(1 - (0.75 - guess) / (1 - guess - lapse))) ** (1 / beta)
    assert fit.threshold(0.75) == pytest.approx(true_thr, rel=0.02)


def test_weibull_recovery_sampled():
    rng = np.random.default_rng(7)
    alpha, beta, guess, lapse = 0.10, 3.0, 0.5, 0.02
    levels = np.geomspace(0.02, 0.6, 12)
    n = np.full_like(levels, 300, dtype=int)
    p = _weibull_p(levels, alpha, beta, guess, lapse)
    k = rng.binomial(n, p).astype(float)

    fit = fit_psychometric(levels, k, n.astype(float), sigmoid="weibull",
                           guess_rate=guess, lapse_rate=lapse)
    assert fit.converged
    assert fit.params["alpha"] == pytest.approx(alpha, rel=0.15)
    assert fit.params["slope"] == pytest.approx(beta, rel=0.30)


def test_logistic_recovery_noiseless():
    alpha, beta, guess, lapse = 0.0, 2.0, 0.5, 0.02
    levels = np.linspace(-4.0, 4.0, 13)
    n = np.full_like(levels, 400)
    p = _logistic_p(levels, alpha, beta, guess, lapse)
    k = n * p

    fit = fit_psychometric(levels, k, n, sigmoid="logistic",
                           guess_rate=guess, lapse_rate=lapse)
    assert fit.converged
    assert fit.params["alpha"] == pytest.approx(alpha, abs=0.03)
    assert fit.params["slope"] == pytest.approx(beta, rel=0.03)


def test_logistic_recovery_sampled():
    rng = np.random.default_rng(11)
    alpha, beta, guess, lapse = 0.5, 1.5, 0.5, 0.02
    levels = np.linspace(-4.0, 5.0, 13)
    n = np.full_like(levels, 300, dtype=int)
    p = _logistic_p(levels, alpha, beta, guess, lapse)
    k = rng.binomial(n, p).astype(float)

    fit = fit_psychometric(levels, k, n.astype(float), sigmoid="logistic",
                           guess_rate=guess, lapse_rate=lapse)
    assert fit.converged
    assert fit.params["alpha"] == pytest.approx(alpha, abs=0.4)
    assert fit.params["slope"] == pytest.approx(beta, rel=0.30)


def test_loglik_and_information_criteria_are_finite():
    alpha, beta, guess, lapse = 0.10, 3.0, 0.5, 0.02
    levels = np.geomspace(0.02, 0.6, 10)
    n = np.full_like(levels, 200)
    k = n * _weibull_p(levels, alpha, beta, guess, lapse)
    fit = fit_psychometric(levels, k, n, sigmoid="weibull")
    assert np.isfinite(fit.log_likelihood)
    assert np.isfinite(fit.aic())
    assert np.isfinite(fit.bic())
    # log-likelihood is negative (product of probabilities); AIC/BIC penalize params.
    assert fit.n_params == 2  # guess & lapse both fixed by default
