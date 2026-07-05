"""Stimulus manipulations following the ``apply_stimulus(image, level, rng)`` contract."""

from .contrast import apply_contrast, michelson_contrast, rms_contrast
from .crowding import composite_crowding
from .degradation import (
    add_gaussian_noise,
    gaussian_blur,
    occlude,
    spatial_frequency_filter,
)

__all__ = [
    "apply_contrast",
    "rms_contrast",
    "michelson_contrast",
    "composite_crowding",
    "add_gaussian_noise",
    "gaussian_blur",
    "occlude",
    "spatial_frequency_filter",
]
