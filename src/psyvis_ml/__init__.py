"""psyvis-ml: treat a vision model as a psychophysical observer.

Sweep a stimulus dimension, fit a psychometric function, and report thresholds and
slopes with confidence intervals instead of accuracy at one severity.

    import psyvis_ml as pe
    result = pe.measure(model=..., suite=pe.suites.ContrastThreshold(...),
                        dataset=pe.datasets.synthetic_dataset(...),
                        levels=pe.linspace_levels(0.01, 0.5, 8, spacing="log"))
    result.threshold(); result.slope(); result.fit(); result.plot()

This ships the fitting core, the sweep engine, the contrast / crowding / degradation stimuli
and suites, and the ``measure()`` API. The comparison/report layer, human-reference overlays,
and real ImageNet/timm loaders are not built yet.
"""

from importlib.metadata import PackageNotFoundError, version

from . import datasets, suites
from .api import ConditionResult, MeasureResult, linspace_levels, measure

try:
    __version__ = version("psyvis-ml")
except PackageNotFoundError:  # running from a source tree without install metadata
    __version__ = "0.0.1"

__all__ = [
    "measure",
    "MeasureResult",
    "ConditionResult",
    "linspace_levels",
    "suites",
    "datasets",
    "__version__",
]
