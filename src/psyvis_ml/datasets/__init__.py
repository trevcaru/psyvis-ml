"""Minimal synthetic dataset for running the API end-to-end.

Real ImageNet / timm loaders are intentionally out of scope for this slice. This synthetic
set is designed to be *drivable by a contrast sweep with an analytically known threshold*:

* Each image is a length-``num_classes`` vector ``mean + amp * (e_class - 1/num_classes)``.
  The deviation term is **zero-mean**, so ``mean(image)`` is preserved, and its argmax is the
  true class.
* ``mean(image)`` encodes a per-image *difficulty quantile* ``q`` in ``[0, 1]`` (spread
  deterministically as ``(i + 0.5) / n``). RMS contrast scaling preserves the mean, so ``q``
  survives the stimulus.

A synthetic observer can therefore decode ``q`` (from the mean), the realized contrast (from
std/mean), and the class (from the argmax of the deviations) — see :func:`decode_quantile`
and :func:`decode_class`. If it responds correctly whenever ``q < P_true(realized_contrast)``,
the fraction correct at each contrast level equals ``P_true`` — a known psychometric function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Dataset",
    "synthetic_dataset",
    "decode_quantile",
    "decode_class",
    "MEAN_BASE",
    "MEAN_SPAN",
]

# Mean-luminance encoding of the difficulty quantile q in [0, 1]:  mean = BASE + SPAN*(q-0.5).
MEAN_BASE = 0.5
MEAN_SPAN = 0.3


@dataclass(frozen=True)
class Dataset:
    """A labeled image set. ``images`` is indexable per sample; ``labels`` are class ints."""

    images: np.ndarray
    labels: np.ndarray
    num_classes: int
    name: str = "synthetic"
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.images)


def decode_quantile(image) -> float:
    """Recover the difficulty quantile ``q`` encoded in the image's mean luminance."""
    mean = float(np.mean(image))
    return (mean - MEAN_BASE) / MEAN_SPAN + 0.5


def decode_class(image) -> int:
    """Recover the true class as the argmax of the (zero-mean) deviations."""
    flat = np.asarray(image, dtype=float).ravel()
    return int(np.argmax(flat - float(np.mean(flat))))


def synthetic_dataset(n=400, num_classes=8, amp=0.4) -> Dataset:
    """Build the synthetic contrast-drivable dataset described in the module docstring.

    Parameters
    ----------
    n
        Number of images. Difficulty quantiles are spread deterministically as
        ``(i + 0.5) / n`` so aggregate accuracy is low-variance.
    num_classes
        Class count (also the image length). Chance level is ``1 / num_classes``.
    amp
        Amplitude of the class deviation bump (sets the baseline contrast before the sweep
        rescales it). Must be small enough that pixel values stay positive.
    """
    if n < 1 or num_classes < 2:
        raise ValueError("need n >= 1 and num_classes >= 2.")
    idx = np.arange(n)
    quantiles = (idx + 0.5) / n
    classes = idx % num_classes
    means = MEAN_BASE + MEAN_SPAN * (quantiles - 0.5)

    eye = np.eye(num_classes)
    # Zero-mean bump per class: e_class - 1/num_classes.
    bumps = eye - 1.0 / num_classes
    images = means[:, None] + amp * bumps[classes]
    if np.any(images <= 0.0):
        raise ValueError("amp too large: images went non-positive (RMS contrast needs mean > 0).")
    return Dataset(
        images=images,
        labels=classes.astype(int),
        num_classes=num_classes,
        name="synthetic-contrast",
        metadata={"quantiles": quantiles, "amp": amp},
    )
