"""Top-level measurement API: ``measure()``, ``linspace_levels()``, ``MeasureResult``.

Composes the pieces built in earlier slices — the sweep engine (method of constant stimuli),
the run bundle, and the import-isolated fitting core — into the PRD §8 target shape::

    result = pe.measure(model=..., suite=pe.suites.ContrastThreshold(...),
                        dataset=..., levels=pe.linspace_levels(...))
    result.threshold(); result.slope(); result.fit(); result.plot()

The suite supplies the stimulus manipulation (per condition) and the observer chance level;
the chance level becomes the fitter's ``guess_rate``, so the fitting core stays classifier-
agnostic (it never needs the class count).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fitting import PsychometricFit, fit_psychometric
from .sweep import RunBundle, run_sweep

__all__ = ["linspace_levels", "measure", "MeasureResult", "ConditionResult"]


def linspace_levels(start, stop, num, *, spacing="linear"):
    """Generate method-of-constant-stimuli levels.

    ``spacing="linear"`` (default) spaces levels evenly; ``spacing="log"`` spaces them
    geometrically (both endpoints must be positive), which suits contrast and other
    positive-axis dimensions.
    """
    if num < 1:
        raise ValueError(f"num must be >= 1; got {num}.")
    if spacing == "linear":
        return np.linspace(float(start), float(stop), int(num))
    if spacing == "log":
        if start <= 0.0 or stop <= 0.0:
            raise ValueError("log spacing requires start > 0 and stop > 0.")
        return np.geomspace(float(start), float(stop), int(num))
    raise ValueError(f"spacing must be 'linear' or 'log'; got {spacing!r}.")


def _model_name(model, explicit):
    if explicit is not None:
        return str(explicit)
    return getattr(model, "__name__", None) or type(model).__name__


def _decreasing_logistic_fit_kwargs(levels, guess_rate, lapse_rate):
    """Bounds + start point for a *decreasing* logistic (falling P vs. an increasing axis).

    Degradation performance falls with severity, so the logistic slope must be negative — but
    the fitter's default logistic bounds force a positive slope. We build negative-slope
    bounds here (in ``measure``, not the import-isolated core) and seed a decreasing start so
    the optimizer does not have to escape a flat region. ``guess`` is always a fixed number in
    ``measure`` (the chance level), so only ``alpha``/``slope`` — and ``lapse`` when
    ``lapse_rate is None`` — are free, matching the fitter's packed free-parameter order.
    """
    levels = np.asarray(levels, dtype=float)
    lo, hi = float(np.min(levels)), float(np.max(levels))
    span = (hi - lo) if hi > lo else max(abs(hi), 1.0)
    alpha_bounds = (lo - 10.0 * span, hi + 10.0 * span)
    slope_bounds = (-1e6 / span, -1e-6 / span)  # strictly negative: force a falling curve
    bounds = [alpha_bounds, slope_bounds]
    x0 = [0.5 * (lo + hi), -4.0 / span]
    if lapse_rate is None:  # lapse becomes a free parameter, appended last
        bounds.append((0.0, 0.5 - 1e-6))
        x0.append(0.02)
    return {"bounds": bounds, "x0": x0}


@dataclass(frozen=True)
class ConditionResult:
    """Per-condition outcome: the label, its run bundle, and its psychometric fit."""

    label: str
    bundle: RunBundle
    fit: PsychometricFit
    metadata: dict = field(default_factory=dict)


class MeasureResult:
    """Result of :func:`measure`, holding one :class:`ConditionResult` per condition.

    Accessors return a bare value when there is a single condition, or a ``{label: value}``
    dict when there are several. The always-dict forms are available as ``.fits``,
    ``.thresholds()`` etc. via the plural properties below.
    """

    def __init__(self, condition_results, *, suite, dataset, chance_level, top_k, seed,
                 model_name=None, decreasing=False):
        self.condition_results = list(condition_results)
        self.suite = suite
        self.dataset = dataset
        self.chance_level = chance_level
        self.top_k = top_k
        self.seed = seed
        self.model_name = model_name
        # Axis direction is *declared* by the suite and threaded through here so downstream
        # code (compare, plot, report) labels direction without re-deriving it from the data.
        # A falling/noisy curve near chance can never flip this — it is not inferred.
        self.decreasing = bool(decreasing)

    # -- shape helpers ------------------------------------------------------ #
    @property
    def is_single(self) -> bool:
        return len(self.condition_results) == 1

    @property
    def labels(self) -> list[str]:
        return [cr.label for cr in self.condition_results]

    def _single_or_dict(self, values):
        if self.is_single:
            return values[0]
        return dict(zip(self.labels, values, strict=True))

    # -- accessors ---------------------------------------------------------- #
    def fit(self):
        """The ``PsychometricFit`` (single condition) or ``{label: fit}`` (multi)."""
        return self._single_or_dict([cr.fit for cr in self.condition_results])

    @property
    def fits(self) -> dict:
        return dict(zip(self.labels, [cr.fit for cr in self.condition_results], strict=True))

    @property
    def bundle(self):
        """The ``RunBundle`` (single condition) or ``{label: bundle}`` (multi)."""
        return self._single_or_dict([cr.bundle for cr in self.condition_results])

    @property
    def bundles(self) -> dict:
        return dict(zip(self.labels, [cr.bundle for cr in self.condition_results],
                        strict=True))

    def threshold(self, target=0.75):
        """Stimulus level at ``target`` proportion correct, per condition."""
        return self._single_or_dict(
            [cr.fit.threshold(target) for cr in self.condition_results]
        )

    def slope(self, target=0.75, units="linear", base=np.e):
        """Slope at the ``target`` threshold, per condition."""
        return self._single_or_dict(
            [cr.fit.slope(target, units=units, base=base) for cr in self.condition_results]
        )

    @property
    def levels(self):
        """The stimulus levels swept (shared across conditions)."""
        return np.asarray(self.condition_results[0].bundle.levels, dtype=float)

    def plot(self, **kwargs):
        """Plot P(correct) vs. level with fitted curve(s) and CI band. Returns a Figure."""
        from .plotting import plot_result

        return plot_result(self, **kwargs)

    def compare(self, others, **kwargs):
        """Overlay this and other models' curves on one axis (the §8 comparison plot).

        ``others`` is a list of :class:`MeasureResult` from other models measured on the same
        suite/condition. Returns a matplotlib ``Figure``. See
        :func:`psyvis_ml.comparison.compare_results`.
        """
        from .comparison import compare_results

        return compare_results([self, *others], **kwargs)

    def report(self, others=None, **kwargs):
        """Write a self-contained methods bundle (text + figures + repro metadata).

        See :func:`psyvis_ml.report.build_report`. Returns a ``ReportBundle``.
        """
        from .report import build_report

        return build_report(self, others=others, **kwargs)

    def __repr__(self) -> str:
        return (
            f"MeasureResult(suite={self.suite.__class__.__name__}, "
            f"model={self.model_name!r}, conditions={self.labels}, "
            f"chance={self.chance_level:.3g}, top_k={self.top_k}, "
            f"decreasing={self.decreasing})"
        )


def measure(model, suite, dataset, levels, *, top_k=1, seed=None, lapse_rate=0.02,
            model_name=None):
    """Run a suite's sweep over ``levels`` and fit a psychometric function per condition.

    Parameters
    ----------
    model
        Any callable ``image -> logits`` (no framework lock-in).
    suite
        A suite instance (e.g. ``ContrastThreshold(...)``) providing conditions, a chance
        level, and the sigmoid family to fit.
    dataset
        Object exposing ``images``, ``labels``, and ``num_classes`` (e.g. from
        ``psyvis_ml.datasets``).
    levels
        The stimulus levels to sweep (method of constant stimuli); see ``linspace_levels``.
    top_k
        Score top-``k`` argmax correctness.
    seed
        Reproducibility seed threaded into every condition's sweep.
    lapse_rate
        Fixed lapse (upper-asymptote) rate for the fit; ``None`` to estimate it.
    model_name
        Optional stable identifier for the config hash (recommended for lambdas).

    Returns
    -------
    MeasureResult
    """
    images = dataset.images
    labels = dataset.labels
    num_classes = dataset.num_classes

    chance = suite.chance_level(num_classes, top_k)
    sigmoid = getattr(suite, "sigmoid", "weibull")
    decreasing = bool(getattr(suite, "decreasing", False))
    if decreasing and sigmoid != "logistic":
        raise ValueError(
            f"a decreasing suite must use sigmoid='logistic'; {suite.__class__.__name__} "
            f"declares sigmoid={sigmoid!r}."
        )
    mname = _model_name(model, model_name)

    condition_results = []
    for cond in suite.conditions():
        bundle = run_sweep(
            model,
            images,
            labels,
            levels,
            cond.apply_stimulus,
            top_k=top_k,
            seed=seed,
            model_name=mname,
            stimulus_name=cond.stimulus_name,
            extra_config={"condition": cond.label, "chance_level": chance,
                          "suite": suite.__class__.__name__},
        )
        lv, nc, nt = bundle.to_fit_inputs()
        extra_fit = _decreasing_logistic_fit_kwargs(lv, chance, lapse_rate) if decreasing else {}
        fit = fit_psychometric(lv, nc, nt, sigmoid=sigmoid, guess_rate=chance,
                               lapse_rate=lapse_rate, **extra_fit)
        condition_results.append(
            ConditionResult(label=cond.label, bundle=bundle, fit=fit,
                            metadata=dict(cond.metadata))
        )

    return MeasureResult(condition_results, suite=suite, dataset=dataset,
                         chance_level=chance, top_k=top_k, seed=seed,
                         model_name=mname, decreasing=decreasing)
