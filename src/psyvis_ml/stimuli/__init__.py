"""Stimulus manipulations following the ``apply_stimulus(image, level, rng)`` contract."""

from .contrast import apply_contrast, michelson_contrast, rms_contrast

__all__ = ["apply_contrast", "rms_contrast", "michelson_contrast"]
