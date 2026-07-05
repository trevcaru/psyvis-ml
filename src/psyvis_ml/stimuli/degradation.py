"""Degradation manipulations swept by severity — the ImageNet-C bridge.

Same corruptions the ImageNet-C robustness benchmark applies at a few fixed severities
(gaussian noise, blur, occlusion, spatial-frequency filtering), but here each is *swept*
parametrically so :func:`psyvis_ml.measure` can fit **P(correct) vs. severity** and report a
**threshold** (the severity at a criterion performance) and a **slope** — a fitted curve
instead of an accuracy scalar at one severity.

All follow the ``apply_stimulus(image, severity, rng)`` contract:

* ``add_gaussian_noise`` and ``occlude`` are **stochastic** — their randomness (noise draws,
  occluder position) comes from the supplied ``rng`` so runs reproduce.
* ``gaussian_blur`` and ``spatial_frequency_filter`` are **deterministic** — they accept
  ``rng`` and ignore it.

Severity convention: ``severity = 0`` is the identity (a clean image) for every manipulation,
and larger severity means more degradation, so P(correct) is a *decreasing* function of
severity. See :class:`psyvis_ml.suites.DegradationSuite` for the decreasing-logistic fit.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = [
    "add_gaussian_noise",
    "gaussian_blur",
    "occlude",
    "spatial_frequency_filter",
]


def add_gaussian_noise(image, severity, rng=None, *, clip_range=None):
    """Add zero-mean Gaussian noise of standard deviation ``severity`` (stochastic).

    The realized noise standard deviation equals ``severity`` in expectation. ``clip_range``
    optionally clips the result to a valid pixel range.
    """
    image = np.asarray(image, dtype=float)
    severity = float(severity)
    if severity < 0.0:
        raise ValueError(f"noise severity (std) must be >= 0; got {severity}.")
    if severity == 0.0:
        return image.copy()
    if rng is None:
        rng = np.random.default_rng()
    out = image + rng.normal(0.0, severity, size=image.shape)
    if clip_range is not None:
        out = np.clip(out, clip_range[0], clip_range[1])
    return out


def gaussian_blur(image, severity, rng=None):
    """Gaussian-blur the spatial dimensions with standard deviation ``severity`` (deterministic).

    ``severity`` is the Gaussian ``sigma`` in pixels; ``0`` returns the image unchanged. For
    images with trailing (channel) axes, blurring is applied per spatial axis only. ``rng`` is
    accepted and ignored.
    """
    image = np.asarray(image, dtype=float)
    severity = float(severity)
    if severity < 0.0:
        raise ValueError(f"blur severity (sigma) must be >= 0; got {severity}.")
    if severity == 0.0:
        return image.copy()
    if image.ndim < 2:
        raise ValueError(f"blur needs a spatial (>=2-D) image; got shape {image.shape}.")
    sigma = (severity, severity) + (0.0,) * (image.ndim - 2)
    return gaussian_filter(image, sigma=sigma, mode="reflect")


def _spatial_mask(shape, severity, rng, block):
    """Boolean (H, W) mask of the pixels to occlude for the given ``severity`` fraction."""
    h, w = shape
    n = h * w
    k = int(round(severity * n))
    mask = np.zeros((h, w), dtype=bool)
    if k <= 0:
        return mask
    if not block:
        # Scatter occlusion: exactly k random pixels, so the realized fraction ~= severity.
        idx = rng.choice(n, size=min(k, n), replace=False)
        flat = mask.ravel()
        flat[idx] = True
        return mask
    # Block occlusion: a single contiguous square of ~severity area at a random position.
    side_h = min(h, int(round(np.sqrt(severity) * h)))
    side_w = min(w, int(round(np.sqrt(severity) * w)))
    if side_h == 0 or side_w == 0:
        return mask
    top = int(rng.integers(0, h - side_h + 1))
    left = int(rng.integers(0, w - side_w + 1))
    mask[top:top + side_h, left:left + side_w] = True
    return mask


def occlude(image, severity, rng=None, *, fill=0.0, block=True):
    """Occlude a ``severity`` fraction of the image, filling it with ``fill`` (stochastic).

    Parameters
    ----------
    severity
        Fraction of pixels to occlude, in ``[0, 1]``. ``0`` returns the image unchanged.
    fill
        Value written into the occluded region.
    block
        If True (default), occlude a single contiguous square of ~``severity`` area at a
        random position (classic occlusion). If False, occlude ``round(severity * N)`` random
        scattered pixels (dropout), whose realized fraction matches ``severity`` closely.

    The occluder's position (or scattered set) is drawn from ``rng``.
    """
    image = np.asarray(image, dtype=float)
    severity = float(severity)
    if not (0.0 <= severity <= 1.0):
        raise ValueError(f"occlusion severity (fraction) must be in [0, 1]; got {severity}.")
    if image.ndim < 2:
        raise ValueError(f"occlusion needs a spatial (>=2-D) image; got shape {image.shape}.")
    out = image.copy()
    if severity == 0.0:
        return out
    if rng is None:
        rng = np.random.default_rng()
    mask = _spatial_mask((image.shape[0], image.shape[1]), severity, rng, block)
    out[mask] = fill  # broadcasts across any trailing channel axes
    return out


def _radial_frequency(h, w):
    """Normalized radial spatial-frequency magnitude in ``[0, 1]`` over an FFT grid."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2)
    return np.clip(r, 0.0, 1.0)


def spatial_frequency_filter(image, severity, rng=None, *, mode="lowpass"):
    """Low- or high-pass filter the image by zeroing a fraction of its spectrum (deterministic).

    Parameters
    ----------
    severity
        Fraction of the (radial) frequency range removed, in ``[0, 1]``. ``0`` is the identity.
    mode
        ``"lowpass"`` keeps radial frequencies below ``(1 - severity)`` (removing the highest
        ``severity`` fraction — progressive blur). ``"highpass"`` removes radial frequencies
        below ``severity`` (including DC at any ``severity > 0``, so the mean is dropped).

    ``rng`` is accepted and ignored. Images with trailing (channel) axes are filtered per
    channel.
    """
    image = np.asarray(image, dtype=float)
    severity = float(severity)
    if not (0.0 <= severity <= 1.0):
        raise ValueError(f"filter severity must be in [0, 1]; got {severity}.")
    if mode not in ("lowpass", "highpass"):
        raise ValueError(f"mode must be 'lowpass' or 'highpass'; got {mode!r}.")
    if image.ndim < 2:
        raise ValueError(f"filter needs a spatial (>=2-D) image; got shape {image.shape}.")
    if severity == 0.0:
        return image.copy()  # identity for both modes at zero severity

    h, w = image.shape[0], image.shape[1]
    r = _radial_frequency(h, w)
    if mode == "lowpass":
        keep = r <= (1.0 - severity)
    else:  # highpass: drop the lowest `severity` fraction (DC included for severity > 0)
        keep = r > severity

    if image.ndim == 2:
        spec = np.fft.fft2(image)
        return np.real(np.fft.ifft2(spec * keep))
    # Filter each trailing channel independently.
    out = np.empty_like(image)
    for c in np.ndindex(image.shape[2:]):
        plane = image[(slice(None), slice(None), *c)]
        spec = np.fft.fft2(plane)
        out[(slice(None), slice(None), *c)] = np.real(np.fft.ifft2(spec * keep))
    return out
