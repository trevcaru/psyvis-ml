"""Maximum-likelihood fitting of the psychometric function.

Import-isolated: NumPy/SciPy only.

    P(correct | x) = guess + (1 - guess - lapse) * F(x; alpha, beta)

``fit_psychometric`` returns a :class:`PsychometricFit`. Convention for asymptotes:
pass a **number** to hold a parameter fixed, ``None`` to estimate it by maximum
likelihood (lapse bounded to ``[0, 0.5)``, guess to ``[0, 1)``). The default lapse is a
fixed ``0.02``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import minimize

from .likelihood import (
    ParamSpec,
    binomial_loglik,
    default_bounds,
    neg_log_likelihood,
    predict_p,
)
from .sigmoids import SIGMOIDS

__all__ = ["fit_psychometric", "PsychometricFit"]

# A free parameter is flagged "at_bound" if it lands this close to a box edge.
_BOUND_RTOL = 1e-3
_BOUND_ATOL = 1e-6


def _validate_inputs(levels, n_correct, n_trials, sigmoid):
    levels = np.asarray(levels, dtype=float).ravel()
    n_correct = np.asarray(n_correct, dtype=float).ravel()
    n_trials = np.asarray(n_trials, dtype=float).ravel()

    if not (levels.shape == n_correct.shape == n_trials.shape):
        raise ValueError(
            "levels, n_correct, n_trials must have the same length; got "
            f"{levels.shape}, {n_correct.shape}, {n_trials.shape}."
        )
    if levels.size == 0:
        raise ValueError("no data: levels is empty.")
    if not np.all(np.isfinite(levels)):
        raise ValueError("levels contains non-finite values.")
    if np.any(n_trials <= 0):
        raise ValueError("n_trials must be positive at every level.")
    if np.any(n_correct < 0) or np.any(n_correct > n_trials):
        raise ValueError("n_correct must satisfy 0 <= n_correct <= n_trials at every level.")
    if sigmoid not in SIGMOIDS:
        raise ValueError(f"unknown sigmoid {sigmoid!r}; choose from {sorted(SIGMOIDS)}.")
    if SIGMOIDS[sigmoid]["requires_positive"] and np.any(levels <= 0.0):
        raise ValueError(
            f"sigmoid={sigmoid!r} requires strictly positive stimulus levels (x > 0); "
            "the Weibull is undefined at x <= 0. Use sigmoid='logistic' for signed axes, "
            "or shift/rescale your levels."
        )
    return levels, n_correct, n_trials


def _resolve_asymptote(value, name, hi):
    """Interpret a fix-vs-free asymptote argument. Returns (free, fixed_value)."""
    if value is None:
        return True, np.nan
    v = float(value)
    if not (0.0 <= v <= hi):
        raise ValueError(f"{name} must be in [0, {hi}]; got {v}.")
    return False, v


def _empirical_threshold(levels, emp, target):
    """Linear-interpolate the level at which empirical P(correct) crosses ``target``."""
    order = np.argsort(levels)
    xs, ys = levels[order], emp[order]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - target) * (y1 - target) <= 0.0 and y1 != y0:
            t = (target - y0) / (y1 - y0)
            return xs[i] + t * (xs[i + 1] - xs[i])
    return float(np.median(levels))


def _start_points(levels, emp, sigmoid, spec, guess0, lapse0):
    """Generate a small grid of optimizer start points for multi-start robustness."""
    target_mid = 0.5 * (guess0 + (1.0 - lapse0))
    x_cross = _empirical_threshold(levels, emp, target_mid)

    if sigmoid == "weibull":
        pos = levels[levels > 0.0]
        alpha_cands = [x_cross, *np.quantile(pos, [0.3, 0.5, 0.7])]
        beta_cands = [0.5, 1.0, 2.0, 4.0]
    else:
        span = float(np.max(levels) - np.min(levels)) or 1.0
        alpha_cands = [x_cross, *np.quantile(levels, [0.3, 0.5, 0.7])]
        beta_cands = [1.0 / span, 2.0 / span, 4.0 / span, 8.0 / span]

    starts = []
    for a in alpha_cands:
        for b in beta_cands:
            starts.append(spec.pack(float(a), float(b), guess0, lapse0))
    return starts


def fit_psychometric(
    levels,
    n_correct,
    n_trials,
    *,
    sigmoid="weibull",
    guess_rate=0.5,
    lapse_rate=0.02,
    x0=None,
    bounds=None,
    maxfev=10_000,
    _warm_start=None,
):
    """Fit a psychometric function by maximum likelihood.

    Parameters
    ----------
    levels, n_correct, n_trials
        Equal-length arrays of stimulus level, successes, and trials per condition point.
    sigmoid
        ``"weibull"`` (requires levels > 0) or ``"logistic"`` (signed axes ok).
    guess_rate, lapse_rate
        Number to hold fixed, or ``None`` to estimate by ML (guess in ``[0, 1)``, lapse in
        ``[0, 0.5)``). Default lapse is a fixed ``0.02``.
    x0
        Optional start point for the free parameters, in packed order
        ``(alpha, beta, [guess], [lapse])``. If given, it is tried in addition to the
        built-in multi-start grid.
    bounds
        Optional list of ``(lo, hi)`` box bounds for the free parameters, in packed order,
        overriding the defaults.
    maxfev
        Max objective evaluations / iterations per optimizer run.
    _warm_start
        Private. A single packed start point; when given, the multi-start grid is skipped
        and the optimizer runs once from this point. Used to make bootstrap refits cheap
        (each resample is close to the parent MLE).

    Returns
    -------
    PsychometricFit
    """
    levels, n_correct, n_trials = _validate_inputs(levels, n_correct, n_trials, sigmoid)

    free_guess, fixed_guess = _resolve_asymptote(guess_rate, "guess_rate", 1.0)
    free_lapse, fixed_lapse = _resolve_asymptote(lapse_rate, "lapse_rate", 0.5)
    spec = ParamSpec(
        free_guess=free_guess,
        free_lapse=free_lapse,
        fixed_guess=0.0 if free_guess else fixed_guess,
        fixed_lapse=0.0 if free_lapse else fixed_lapse,
    )

    f = SIGMOIDS[sigmoid]["f"]
    box = bounds if bounds is not None else default_bounds(levels, sigmoid, spec)
    if len(box) != spec.n_free:
        raise ValueError(
            f"bounds has {len(box)} entries but there are {spec.n_free} free parameters "
            f"({spec.names})."
        )

    emp = n_correct / n_trials
    guess0 = fixed_guess if not free_guess else float(np.clip(np.min(emp), 0.0, 0.9))
    lapse0 = fixed_lapse if not free_lapse else 0.02

    if _warm_start is not None:
        starts = [np.asarray(_warm_start, dtype=float)]
    else:
        starts = _start_points(levels, emp, sigmoid, spec, guess0, lapse0)
        if x0 is not None:
            starts.insert(0, np.asarray(x0, dtype=float))

    def objective(theta):
        return neg_log_likelihood(theta, levels, n_correct, n_trials, f, spec)

    best = None
    for start in starts:
        start = np.clip(start, [b[0] for b in box], [b[1] for b in box])
        res = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=box,
            options={"maxfun": maxfev, "maxiter": maxfev},
        )
        if best is None or res.fun < best.fun:
            best = res

    alpha, beta, guess, lapse = spec.unpack(best.x)
    ll = binomial_loglik(levels, n_correct, n_trials, alpha, beta, guess, lapse, f)

    at_bound = []
    for name, val, (lo, hi) in zip(spec.names, best.x, box, strict=True):
        # Compare each bound with a tolerance scaled to that bound's own magnitude, so a
        # small fitted value is not spuriously flagged as sitting on a huge upper bound.
        if abs(val - lo) <= _BOUND_ATOL + _BOUND_RTOL * abs(lo) or abs(
            val - hi
        ) <= _BOUND_ATOL + _BOUND_RTOL * abs(hi):
            at_bound.append(name)

    return PsychometricFit(
        levels=levels,
        n_correct=n_correct,
        n_trials=n_trials,
        sigmoid=sigmoid,
        params={"alpha": alpha, "slope": beta, "guess": guess, "lapse": lapse},
        log_likelihood=ll,
        n_params=spec.n_free,
        converged=bool(best.success),
        at_bound=at_bound,
        _spec=spec,
        _config={
            "sigmoid": sigmoid,
            "guess_rate": guess_rate,
            "lapse_rate": lapse_rate,
            "bounds": bounds,
            "maxfev": maxfev,
        },
    )


class PsychometricFit:
    """Result of :func:`fit_psychometric`.

    ``params`` holds the raw model parameters: ``alpha`` (the sigmoid location), ``slope``
    (the sigmoid slope beta), ``guess`` (gamma) and ``lapse`` (lambda). The
    :meth:`threshold` and :meth:`slope` methods report *derived* quantities at a target
    performance level and are usually what you want to read off a fit; note ``alpha`` (the
    raw location) and ``threshold(target)`` (the level at a given performance) are distinct.
    """

    def __init__(
        self,
        *,
        levels,
        n_correct,
        n_trials,
        sigmoid,
        params,
        log_likelihood,
        n_params,
        converged,
        at_bound,
        _spec,
        _config,
    ):
        self.levels = np.asarray(levels, dtype=float)
        self.n_correct = np.asarray(n_correct, dtype=float)
        self.n_trials = np.asarray(n_trials, dtype=float)
        self.sigmoid = sigmoid
        self.params = dict(params)
        self.log_likelihood = float(log_likelihood)
        self.n_params = int(n_params)
        self.converged = bool(converged)
        self.at_bound = list(at_bound)
        self._spec = _spec
        self._config = dict(_config)
        self._f = SIGMOIDS[sigmoid]["f"]
        self._deriv = SIGMOIDS[sigmoid]["deriv"]
        self._inverse = SIGMOIDS[sigmoid]["inverse"]

    # -- core evaluation ---------------------------------------------------- #
    def predict(self, x):
        """Predicted P(correct) at stimulus level(s) ``x``."""
        p = self.params
        if SIGMOIDS[self.sigmoid]["requires_positive"] and np.any(np.asarray(x) <= 0.0):
            warnings.warn(
                f"sigmoid={self.sigmoid!r} is undefined at x <= 0; those points return NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
        return predict_p(x, p["alpha"], p["slope"], p["guess"], p["lapse"], self._f)

    def threshold(self, target=0.75):
        """Stimulus level at which P(correct) == ``target``.

        Inverts the *full* curve (including guess/lapse). Returns NaN with a warning if
        ``target`` is outside the reachable range ``(guess, 1 - lapse)``.
        """
        p = self.params
        gamma, lam = p["guess"], p["lapse"]
        if not (gamma < target < 1.0 - lam):
            warnings.warn(
                f"target={target} is outside the reachable range "
                f"(guess={gamma:.4g}, 1-lapse={1.0 - lam:.4g}); returning NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        f_target = (target - gamma) / (1.0 - gamma - lam)
        return float(self._inverse(f_target, p["alpha"], p["slope"]))

    def slope(self, target=0.75, units="linear", base=math.e):
        """Slope of the psychometric function at the ``target`` threshold.

        ``units="linear"`` returns dP/dx (probability per stimulus unit). ``units="log"``
        returns dP/d(log_base x) = ln(base) * x * dP/dx, the slope per unit of ``log_base``
        of the stimulus. The default ``base=math.e`` gives natural-log units; pass
        ``base=10`` for the log10 convention commonly reported in vision science. Log-units
        slope is NaN for x <= 0 (e.g. a logistic threshold on a signed axis). ``base`` is
        ignored when ``units="linear"``.
        """
        if units not in ("linear", "log"):
            raise ValueError(f"units must be 'linear' or 'log'; got {units!r}.")
        x_thr = self.threshold(target)
        if not np.isfinite(x_thr):
            return float("nan")
        p = self.params
        amplitude = 1.0 - p["guess"] - p["lapse"]
        dpdx = amplitude * float(self._deriv(x_thr, p["alpha"], p["slope"]))
        if units == "linear":
            return dpdx
        if base <= 0.0 or base == 1.0:
            raise ValueError(f"log base must be positive and != 1; got {base}.")
        if x_thr <= 0.0:
            warnings.warn(
                f"log-units slope needs x > 0 but the threshold is at x={x_thr:.4g}; "
                "returning NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        # d(log_base x) = d(ln x) / ln(base), so dP/d(log_base x) = ln(base) * x * dP/dx.
        return math.log(base) * x_thr * dpdx

    # -- information criteria ---------------------------------------------- #
    def aic(self):
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    def bic(self):
        n_obs = int(np.sum(self.n_trials))
        return self.n_params * np.log(n_obs) - 2.0 * self.log_likelihood

    # -- confidence intervals ---------------------------------------------- #
    def bootstrap_ci(self, quantity="threshold", *, target=0.75, n_boot=2000, ci=0.95,
                     seed=None, units="linear"):
        """Parametric bootstrap CI. See :mod:`psyvis_ml.fitting.bootstrap`.

        Returns a ``BootstrapCI`` named tuple ``(low, high, ci, n_boot, n_failed,
        fail_fraction, quantity)``; NaN resamples are counted, not silently dropped.
        """
        from .bootstrap import parametric_bootstrap_ci

        return parametric_bootstrap_ci(
            self, quantity=quantity, target=target, n_boot=n_boot, ci=ci, seed=seed,
            units=units,
        )

    def bootstrap_curve(self, x, *, n_boot=400, ci=0.95, seed=None):
        """Parametric-bootstrap CI band for the predicted curve over grid ``x``.

        Returns ``(low, high, fail_fraction)``. See :mod:`psyvis_ml.fitting.bootstrap`.
        """
        from .bootstrap import parametric_bootstrap_curve

        return parametric_bootstrap_curve(self, x, n_boot=n_boot, ci=ci, seed=seed)

    def _refit(self, n_correct_boot):
        """Refit with identical configuration on resampled counts (used by bootstrap).

        Warm-starts a single optimization from this fit's own MLE: each bootstrap resample
        lies near the parent, so the full multi-start grid is unnecessary and ~16x slower.
        """
        c = self._config
        p = self.params
        warm = self._spec.pack(p["alpha"], p["slope"], p["guess"], p["lapse"])
        return fit_psychometric(
            self.levels,
            n_correct_boot,
            self.n_trials,
            sigmoid=c["sigmoid"],
            guess_rate=c["guess_rate"],
            lapse_rate=c["lapse_rate"],
            bounds=c["bounds"],
            maxfev=c["maxfev"],
            _warm_start=warm,
        )

    def _quantity(self, name, target, units):
        """Extract a scalar quantity by name for bootstrap aggregation."""
        if name == "threshold":
            return self.threshold(target)
        if name == "slope":
            return self.slope(target, units=units)
        if name in self.params:
            return self.params[name]
        raise ValueError(
            f"unknown bootstrap quantity {name!r}; use 'threshold', 'slope', or a "
            f"parameter name in {sorted(self.params)}."
        )

    # -- reporting ---------------------------------------------------------- #
    def summary(self):
        """Flat dict of the fit: params, derived threshold/slope, fit stats, at_bound."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            thr = self.threshold(0.75)
            slp_lin = self.slope(0.75, units="linear")
            slp_log = self.slope(0.75, units="log")
        return {
            "sigmoid": self.sigmoid,
            "alpha": self.params["alpha"],
            "slope": self.params["slope"],
            "guess": self.params["guess"],
            "lapse": self.params["lapse"],
            "threshold_at_0.75": thr,
            "slope_at_0.75_linear": slp_lin,
            "slope_at_0.75_log": slp_log,
            "log_likelihood": self.log_likelihood,
            "n_params": self.n_params,
            "n_obs": int(np.sum(self.n_trials)),
            "aic": self.aic(),
            "bic": self.bic(),
            "converged": self.converged,
            "at_bound": list(self.at_bound),
        }

    def __repr__(self):
        p = self.params
        return (
            f"PsychometricFit(sigmoid={self.sigmoid!r}, alpha={p['alpha']:.4g}, "
            f"slope={p['slope']:.4g}, guess={p['guess']:.4g}, lapse={p['lapse']:.4g}, "
            f"converged={self.converged})"
        )
