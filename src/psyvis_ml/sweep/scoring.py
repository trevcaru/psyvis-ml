"""Correctness scoring and confidence extraction from classifier logits.

Framework-agnostic: consumes plain NumPy arrays of logits and integer labels, so it works
with any ``model`` callable regardless of the framework that produced the logits.

The **primary confidence signal** is the target-class **logit margin** —
``margin(x) = logit[target] − max(logit[others])`` — read straight off the logit vector the
sweep already computes. It is a raw evidence difference (positive = target beats its best
competitor), *not* an exponentiated softmax probability. We deliberately do **not** use raw
softmax as the primary signal: softmax exponentiates logits, swings wildly with temperature,
and is poorly calibrated. ``max_softmax`` is recorded for reference only and flagged
calibration-sensitive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["top_k_correct", "count_correct", "score_logits", "LogitScores"]


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


@dataclass(frozen=True)
class LogitScores:
    """Per-image scores from a batch of classifier logits at one stimulus level.

    All arrays are length ``n_samples``. ``margin`` is the primary confidence signal;
    ``max_softmax`` is reference-only and **calibration-sensitive** (do not treat as calibrated
    probability). ``target_rank`` is 1 when the target is the top-1 prediction.
    """

    correct: np.ndarray        # bool: is the target among the top-k predictions?
    margin: np.ndarray         # float: logit[target] - max(logit[others]); >0 = target leads
    target_rank: np.ndarray    # int: 1 = target is top-1 (count of strictly-higher logits + 1)
    target_logit: np.ndarray   # float: the raw logit assigned to the target class
    max_softmax: np.ndarray    # float: max softmax probability (REFERENCE ONLY; not calibrated)


def score_logits(logits, labels, k=1) -> LogitScores:
    """Extract correctness + confidence signals from ``(n_samples, n_classes)`` logits.

    The **logit margin** ``logit[target] − max(logit[others])`` is the primary confidence
    measure. Also returns the target's rank, its raw logit, and (for reference only) the
    max-softmax probability. See the module docstring for why softmax is not primary.
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
    if n_classes < 2:
        raise ValueError("margin needs at least 2 classes (target vs. a competitor).")

    rows = np.arange(n)
    target_logit = logits[rows, labels]

    # Best competitor: mask the target column to -inf, then take the row max.
    others = logits.copy()
    others[rows, labels] = -np.inf
    max_other = others.max(axis=1)
    margin = target_logit - max_other

    # Rank: 1 + number of classes with a strictly higher logit (1 == target is top-1).
    target_rank = 1 + np.sum(logits > target_logit[:, None], axis=1)

    # Max softmax (numerically stable) — reference only, calibration-sensitive.
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    max_softmax = e.max(axis=1) / e.sum(axis=1)

    return LogitScores(
        correct=top_k_correct(logits, labels, k=k),
        margin=margin,
        target_rank=target_rank.astype(int),
        target_logit=target_logit,
        max_softmax=max_softmax,
    )
