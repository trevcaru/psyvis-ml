"""Parametric bootstrap confidence intervals for a psychometric fit.

Import-isolated: NumPy only (it drives refitting through the fit object's own methods,
so it does not import the fitter directly).

Procedure (Wichmann & Hill, 2001): treat the fitted curve as truth, resample
``n_correct ~ Binomial(n_trials, P_fitted(level))`` at each level, refit, and record the
quantity of interest. The CI is the percentile interval over successful resamples.
Failed resamples (non-convergence, or a quantity that evaluates to NaN — e.g. a threshold
whose target is unreachable for that resample's asymptotes) are **counted, not silently
dropped**: the returned tuple reports how many failed, and if *every* resample fails the
bounds are NaN.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

__all__ = ["BootstrapCI", "parametric_bootstrap_ci", "parametric_bootstrap_curve"]


class BootstrapCI(NamedTuple):
    low: float
    high: float
    ci: float
    n_boot: int
    n_failed: int
    fail_fraction: float
    quantity: str


def parametric_bootstrap_ci(fit, *, quantity="threshold", target=0.75, n_boot=2000,
                            ci=0.95, seed=None, units="linear") -> BootstrapCI:
    """Compute a parametric-bootstrap percentile CI for a derived quantity of ``fit``.

    Parameters
    ----------
    fit
        A fitted ``PsychometricFit``. Used read-only via ``predict``, ``_refit`` and
        ``_quantity``.
    quantity
        ``"threshold"``, ``"slope"``, or a raw parameter name (``"threshold"`` here means
        the derived level at ``target``; the raw sigmoid location is the parameter of the
        same name, so prefer the explicit derived quantities).
    target, units
        Passed through to threshold/slope evaluation.
    n_boot, ci, seed
        Number of resamples, central mass of the interval (e.g. 0.95), and RNG seed.
    """
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1); got {ci}.")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}.")
    # Validate the quantity name up front; otherwise a bad name would be swallowed by the
    # per-resample except-guard below and mis-reported as "all resamples failed".
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit._quantity(quantity, target, units)  # raises ValueError for unknown names

    rng = np.random.default_rng(seed)
    n_trials = np.rint(fit.n_trials).astype(np.int64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p_hat = np.asarray(fit.predict(fit.levels), dtype=float)
    p_hat = np.clip(p_hat, 0.0, 1.0)

    values = []
    n_failed = 0
    for _ in range(n_boot):
        k_boot = rng.binomial(n_trials, p_hat).astype(float)
        try:
            boot_fit = fit._refit(k_boot)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                val = float(boot_fit._quantity(quantity, target, units))
        except Exception:
            val = float("nan")
        if np.isfinite(val):
            values.append(val)
        else:
            n_failed += 1

    fail_fraction = n_failed / n_boot
    if not values:
        warnings.warn(
            f"all {n_boot} bootstrap resamples failed for quantity={quantity!r}; "
            "returning NaN bounds.",
            RuntimeWarning,
            stacklevel=2,
        )
        return BootstrapCI(float("nan"), float("nan"), ci, n_boot, n_failed, fail_fraction,
                           quantity)

    tail = 0.5 * (1.0 - ci)
    lo, hi = np.quantile(values, [tail, 1.0 - tail])
    return BootstrapCI(float(lo), float(hi), ci, n_boot, n_failed, fail_fraction, quantity)


def parametric_bootstrap_curve(fit, x, *, n_boot=400, ci=0.95, seed=None):
    """Parametric-bootstrap confidence band for the *predicted curve* over grid ``x``.

    Resamples counts from the fitted curve, refits, and predicts on ``x``; returns
    ``(low, high)`` percentile bands (each shaped like ``x``) plus the fraction of failed
    resamples. Bands are NaN where every resample failed. Used by the plotting layer to
    shade a CI region around the fitted psychometric function.
    """
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1); got {ci}.")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}.")

    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n_trials = np.rint(fit.n_trials).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p_hat = np.clip(np.asarray(fit.predict(fit.levels), dtype=float), 0.0, 1.0)

    curves = []
    n_failed = 0
    for _ in range(n_boot):
        k_boot = rng.binomial(n_trials, p_hat).astype(float)
        try:
            boot_fit = fit._refit(k_boot)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                curve = np.asarray(boot_fit.predict(x), dtype=float)
            if not np.all(np.isfinite(curve)):
                raise ValueError("non-finite predicted curve")
        except Exception:
            n_failed += 1
            continue
        curves.append(curve)

    fail_fraction = n_failed / n_boot
    if not curves:
        nan = np.full_like(x, np.nan)
        return nan, nan, fail_fraction
    stack = np.vstack(curves)
    tail = 0.5 * (1.0 - ci)
    low = np.quantile(stack, tail, axis=0)
    high = np.quantile(stack, 1.0 - tail, axis=0)
    return low, high, fail_fraction
