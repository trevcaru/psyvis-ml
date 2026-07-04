"""Binomial likelihood, parameter packing, and default bounds for the fitter.

Import-isolated: NumPy/SciPy only.

The full parameter vector is ``(alpha, beta, guess, lapse)``. Which of ``guess`` and
``lapse`` are *free* (estimated) vs. *fixed* is decided by the caller: a number fixes a
parameter, ``None`` frees it. ``pack``/``unpack`` translate between the compact free
vector the optimizer sees and the full 4-tuple the model needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

__all__ = [
    "ParamSpec",
    "predict_p",
    "neg_log_likelihood",
    "binomial_loglik",
    "default_bounds",
]

# Clamp predicted probabilities away from {0, 1} so log-likelihood stays finite.
_P_EPS = 1e-12


@dataclass(frozen=True)
class ParamSpec:
    """Describes which parameters are free and holds the fixed values.

    Free-parameter order in the packed vector is always: alpha, beta, then guess (if
    free), then lapse (if free).
    """

    free_guess: bool
    free_lapse: bool
    fixed_guess: float  # used when free_guess is False; ignored otherwise
    fixed_lapse: float  # used when free_lapse is False; ignored otherwise

    @property
    def names(self) -> list[str]:
        names = ["alpha", "slope"]
        if self.free_guess:
            names.append("guess")
        if self.free_lapse:
            names.append("lapse")
        return names

    @property
    def n_free(self) -> int:
        return 2 + int(self.free_guess) + int(self.free_lapse)

    def pack(self, alpha: float, beta: float, guess: float, lapse: float) -> np.ndarray:
        theta = [alpha, beta]
        if self.free_guess:
            theta.append(guess)
        if self.free_lapse:
            theta.append(lapse)
        return np.asarray(theta, dtype=float)

    def unpack(self, theta) -> tuple[float, float, float, float]:
        theta = np.asarray(theta, dtype=float)
        alpha, beta = theta[0], theta[1]
        idx = 2
        if self.free_guess:
            guess = theta[idx]
            idx += 1
        else:
            guess = self.fixed_guess
        if self.free_lapse:
            lapse = theta[idx]
        else:
            lapse = self.fixed_lapse
        return float(alpha), float(beta), float(guess), float(lapse)


def predict_p(x, alpha, beta, guess, lapse, f):
    """Full psychometric prediction P(correct | x)."""
    return guess + (1.0 - guess - lapse) * f(x, alpha, beta)


def neg_log_likelihood(theta, x, k, n, f, spec: ParamSpec) -> float:
    """Negative binomial log-likelihood used as the optimizer objective.

    Drops the (parameter-independent) binomial coefficient; adds a large finite penalty
    for degenerate parameterizations (non-positive amplitude, NaN predictions).
    """
    alpha, beta, guess, lapse = spec.unpack(theta)
    amplitude = 1.0 - guess - lapse
    if amplitude <= 0.0:
        return 1e12
    p = predict_p(x, alpha, beta, guess, lapse, f)
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        return 1e12
    p = np.clip(p, _P_EPS, 1.0 - _P_EPS)
    ll = np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p))
    if not np.isfinite(ll):
        return 1e12
    return -float(ll)


def binomial_loglik(x, k, n, alpha, beta, guess, lapse, f) -> float:
    """Full binomial log-likelihood (including the coefficient) for reporting."""
    p = predict_p(x, alpha, beta, guess, lapse, f)
    p = np.clip(np.asarray(p, dtype=float), _P_EPS, 1.0 - _P_EPS)
    coeff = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    return float(np.sum(coeff + k * np.log(p) + (n - k) * np.log(1.0 - p)))


def default_bounds(levels, sigmoid: str, spec: ParamSpec) -> list[tuple[float, float]]:
    """Default box bounds for the free parameters, in packed order.

    alpha/beta bounds are scaled from the observed stimulus range; guess in [0, 1),
    lapse in [0, 0.5). Bounds can be overridden by the caller in ``fit_psychometric``.
    """
    levels = np.asarray(levels, dtype=float)
    lo, hi = float(np.min(levels)), float(np.max(levels))
    span = hi - lo if hi > lo else max(abs(hi), 1.0)

    if sigmoid == "weibull":
        # alpha is a positive scale; allow well inside and beyond the sampled range.
        pos = levels[levels > 0.0]
        floor = float(np.min(pos)) if pos.size else 1e-6
        alpha_bounds = (floor * 1e-3, hi * 1e3)
        beta_bounds = (1e-3, 1e3)  # Weibull shape
    else:  # logistic
        alpha_bounds = (lo - 10.0 * span, hi + 10.0 * span)
        # slope has units 1/x; keep it positive and generously bounded relative to span.
        beta_bounds = (1e-6 / span, 1e6 / span)

    bounds = [alpha_bounds, beta_bounds]
    if spec.free_guess:
        bounds.append((0.0, 1.0 - 1e-6))
    if spec.free_lapse:
        bounds.append((0.0, 0.5 - 1e-6))
    return bounds
