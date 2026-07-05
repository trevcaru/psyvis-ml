"""psyvis-ml: treat a vision model as a psychophysical observer.

Sweep a stimulus dimension, fit a psychometric function, and report thresholds and
slopes with confidence intervals instead of accuracy at one severity.

    import psyvis_ml as pe
    result = pe.measure(model=..., suite=pe.suites.ContrastThreshold(...),
                        dataset=pe.datasets.synthetic_dataset(...),
                        levels=pe.linspace_levels(0.01, 0.5, 8, spacing="log"))
    result.threshold(); result.slope(); result.fit(); result.plot()

This ships the fitting core, the sweep engine, the contrast / degradation / distractor-
robustness stimuli and suites, the ``measure()`` API, and the comparison / human-reference /
report layer. Real ImageNet/timm convenience loaders are optional (the ``[demo]`` extra).
"""

from importlib.metadata import PackageNotFoundError, version

from . import datasets, models, reference, suites
from .api import ConditionResult, MeasureResult, linspace_levels, measure
from .comparison import compare_results, comparison_summary
from .distractor_calibration import DistractorCalibration, calibrate_distractor_spacing
from .report import build_report

try:
    __version__ = version("psyvis-ml")
except PackageNotFoundError:  # running from a source tree without install metadata
    __version__ = "0.0.1"

__all__ = [
    "measure",
    "MeasureResult",
    "ConditionResult",
    "linspace_levels",
    "compare_results",
    "comparison_summary",
    "build_report",
    "calibrate_distractor_spacing",
    "DistractorCalibration",
    "suites",
    "datasets",
    "models",
    "reference",
    "__version__",
]
