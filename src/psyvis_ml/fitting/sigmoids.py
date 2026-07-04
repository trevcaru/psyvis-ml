"""Sigmoid families and asymptote composition for the psychometric function.

Import-isolated: this module imports only NumPy/SciPy and nothing else from the
package. Each sigmoid ``F(x; alpha, beta)`` maps a stimulus level ``x`` to the range
``(0, 1)`` with location parameter ``alpha`` and slope parameter ``beta``. The full
psychometric function composes a sigmoid with guess/lapse asymptotes:

    P(correct | x) = guess + (1 - guess - lapse) * F(x; alpha, beta)
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

__all__ = [
    "weibull",
    "weibull_deriv",
    "weibull_inverse",
    "logistic",
    "logistic_deriv",
    "logistic_inverse",
    "with_asymptotes",
    "SIGMOIDS",
]


# --------------------------------------------------------------------------- #
# Weibull.  F(x) = 1 - exp(-(x / alpha) ** beta),  requires x > 0, alpha > 0.
# At x = alpha, F = 1 - 1/e ~= 0.632.  alpha is a scale/location, beta the shape.
# --------------------------------------------------------------------------- #
def weibull(x, alpha, beta):
    """Weibull CDF sigmoid. Defined for x > 0; returns NaN for x <= 0."""
    x = np.asarray(x, dtype=float)
    # overflow in z**beta (large beta, z > 1) saturates to +inf -> exp(-inf) = 0 -> F = 1,
    # which is the correct limit, so the overflow is benign and silenced.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = np.where(x > 0.0, x / alpha, np.nan)
        return 1.0 - np.exp(-np.power(z, beta))


def weibull_deriv(x, alpha, beta):
    """dF/dx for the Weibull sigmoid."""
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = np.where(x > 0.0, x / alpha, np.nan)
        return (beta / alpha) * np.power(z, beta - 1.0) * np.exp(-np.power(z, beta))


def weibull_inverse(p, alpha, beta):
    """Inverse: return x such that weibull(x, alpha, beta) == p, for p in (0, 1)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (p > 0.0) & (p < 1.0)
        out = np.where(valid, alpha * np.power(-np.log1p(-p), 1.0 / beta), np.nan)
    return out


# --------------------------------------------------------------------------- #
# Logistic.  F(x) = 1 / (1 + exp(-beta * (x - alpha))).  Handles signed x.
# alpha is the midpoint (F = 0.5), beta the slope.
# --------------------------------------------------------------------------- #
def logistic(x, alpha, beta):
    """Logistic sigmoid. Defined for all real x."""
    x = np.asarray(x, dtype=float)
    return expit(beta * (x - alpha))


def logistic_deriv(x, alpha, beta):
    """dF/dx for the logistic sigmoid."""
    f = logistic(x, alpha, beta)
    return beta * f * (1.0 - f)


def logistic_inverse(p, alpha, beta):
    """Inverse: return x such that logistic(x, alpha, beta) == p, for p in (0, 1)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (p > 0.0) & (p < 1.0)
        out = np.where(valid, alpha + np.log(p / (1.0 - p)) / beta, np.nan)
    return out


def with_asymptotes(f, guess, lapse):
    """Compose a bare sigmoid ``f(x, alpha, beta)`` with guess/lapse asymptotes.

    Returns a callable ``g(x, alpha, beta) = guess + (1 - guess - lapse) * f(...)``.
    """
    amplitude = 1.0 - guess - lapse

    def g(x, alpha, beta):
        return guess + amplitude * f(x, alpha, beta)

    return g


# Registry: name -> everything the fitter needs to know about a family.
SIGMOIDS = {
    "weibull": {
        "f": weibull,
        "deriv": weibull_deriv,
        "inverse": weibull_inverse,
        "requires_positive": True,
    },
    "logistic": {
        "f": logistic,
        "deriv": logistic_deriv,
        "inverse": logistic_inverse,
        "requires_positive": False,
    },
}
