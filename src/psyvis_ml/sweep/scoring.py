"""Correctness scoring from classifier logits.

Framework-agnostic: consumes plain NumPy arrays of logits and integer labels, so it works
with any ``model`` callable regardless of the framework that produced the logits.
"""

from __future__ import annotations

import numpy as np

__all__ = ["top_k_correct", "count_correct"]


def _as_logits_2d(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        logits = logits[None, :]
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 1-D or 2-D (n_samples, n_classes); got shape {logits.shape}."
        )
    return logits


def top_k_correct(logits, labels, k=1) -> np.ndarray:
    """Boolean array: is the true label among the model's top-``k`` predictions?

    Parameters
    ----------
    logits
        ``(n_samples, n_classes)`` (or ``(n_classes,)`` for a single sample).
    labels
        ``(n_samples,)`` integer class indices.
    k
        Top-k tolerance; ``k=1`` is standard top-1 argmax correctness.
    """
    logits = _as_logits_2d(logits)
    labels = np.asarray(labels).ravel()
    n, n_classes = logits.shape
    if labels.shape[0] != n:
        raise ValueError(f"labels length {labels.shape[0]} != n_samples {n}.")
    if not (1 <= k <= n_classes):
        raise ValueError(f"k must be in [1, n_classes={n_classes}]; got {k}.")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= n_classes:
        raise ValueError("labels must be valid class indices in [0, n_classes).")

    if k == 1:
        preds = np.argmax(logits, axis=1)
        return preds == labels
    # Top-k: the k largest-logit class indices per row (unordered is fine for membership).
    topk = np.argpartition(logits, n_classes - k, axis=1)[:, n_classes - k:]
    return np.any(topk == labels[:, None], axis=1)


def count_correct(logits, labels, k=1) -> int:
    """Number of top-``k``-correct samples."""
    return int(np.count_nonzero(top_k_correct(logits, labels, k=k)))
