"""Contrast manipulation stimulus.

Follows the sweep engine's ``apply_stimulus(image, level, rng)`` contract. Contrast
manipulation is *deterministic*, so ``rng`` is accepted and ignored.

``level`` is the *target contrast*: the image is rescaled so its contrast — measured with
the selected metric — equals ``level`` exactly, by scaling pixel deviations around a fixed
center.

* ``metric="rms"``      : RMS contrast = std(image) / mean(image). Deviations are scaled
                          around the **mean**, which the scaling preserves.
* ``metric="michelson"``: Michelson contrast = (max - min) / (max + min). Deviations are
                          scaled around the **midpoint** (max + min) / 2, which the scaling
                          preserves.

Both reduce to ``out = center + (level / current_contrast) * (image - center)``.

Spatial-frequency conditioning (``spatial_freq``) is a **stub** per PRD §7: the parameter is
accepted and threaded through for API stability but is **not yet implemented** — it does not
alter the manipulation. See ``psyvis_ml.suites.ContrastThreshold`` for the condition wiring.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rms_contrast", "michelson_contrast", "apply_contrast"]


def rms_contrast(image) -> float:
    """RMS contrast: std(image) / mean(image). Returns 0.0 for a non-positive mean."""
    image = np.asarray(image, dtype=float)
    mean = float(np.mean(image))
    if mean <= 0.0:
        return 0.0
    return float(np.std(image)) / mean


def michelson_contrast(image) -> float:
    """Michelson contrast: (max - min) / (max + min). Returns 0.0 if max + min == 0."""
    image = np.asarray(image, dtype=float)
    hi = float(np.max(image))
    lo = float(np.min(image))
    denom = hi + lo
    if denom == 0.0:
        return 0.0
    return (hi - lo) / denom


def _center_and_contrast(image, metric):
    if metric == "rms":
        center = float(np.mean(image))
        return center, rms_contrast(image)
    if metric == "michelson":
        hi, lo = float(np.max(image)), float(np.min(image))
        return 0.5 * (hi + lo), michelson_contrast(image)
    raise ValueError(f"unknown contrast metric {metric!r}; use 'rms' or 'michelson'.")


def apply_contrast(image, level, rng=None, *, metric="rms", spatial_freq=None,
                   clip_range=None):
    """Rescale ``image`` to a target contrast ``level``.

    Parameters
    ----------
    image
        Array of pixel values (any shape).
    level
        Target contrast (>= 0) in the selected metric.
    rng
        Unused; present only to satisfy the ``apply_stimulus(image, level, rng)`` contract.
    metric
        ``"rms"`` or ``"michelson"``.
    spatial_freq
        **Stub / not yet implemented** (PRD §7). Accepted but ignored.
    clip_range
        Optional ``(lo, hi)`` to clip the result into a valid pixel range. Clipping can
        perturb the realized contrast away from ``level``; omit it if you need the contrast
        set exactly.

    Returns
    -------
    numpy.ndarray
        The contrast-adjusted image. A uniform (zero-contrast) image is returned unchanged,
        since there are no deviations to scale.
    """
    image = np.asarray(image, dtype=float)
    level = float(level)
    if level < 0.0:
        raise ValueError(f"contrast level must be >= 0; got {level}.")

    center, current = _center_and_contrast(image, metric)
    if current <= 0.0:
        # Uniform / degenerate image: nothing to scale. Return a copy unchanged.
        return image.copy()

    gain = level / current
    out = center + gain * (image - center)
    if clip_range is not None:
        lo, hi = clip_range
        out = np.clip(out, lo, hi)
    return out
