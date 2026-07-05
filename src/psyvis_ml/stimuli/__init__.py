"""Stimulus manipulations following the ``apply_stimulus(image, level, rng)`` contract."""

from .contrast import apply_contrast, michelson_contrast, rms_contrast
from .degradation import (
    add_gaussian_noise,
    gaussian_blur,
    occlude,
    spatial_frequency_filter,
)
from .distractor import composite_distractor

__all__ = [
    "apply_contrast",
    "rms_contrast",
    "michelson_contrast",
    "composite_distractor",
    "add_gaussian_noise",
    "gaussian_blur",
    "occlude",
    "spatial_frequency_filter",
]
