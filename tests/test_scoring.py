"""Tests for top-1 / top-k argmax correctness scoring."""

import numpy as np
import pytest

from psyvis_ml.sweep.scoring import count_correct, top_k_correct


def test_top1_correctness():
    logits = np.array([
        [0.1, 0.9, 0.0],   # argmax 1
        [2.0, 1.0, 0.5],   # argmax 0
        [0.0, 0.0, 3.0],   # argmax 2
    ])
    labels = np.array([1, 2, 2])
    got = top_k_correct(logits, labels, k=1)
    assert got.tolist() == [True, False, True]
    assert count_correct(logits, labels, k=1) == 2


def test_topk_counts_label_in_top_k_but_not_top1():
    # True label is the 2nd-highest logit: wrong at k=1, right at k=2.
    logits = np.array([[0.1, 0.5, 0.9, 0.2]])
    labels = np.array([1])
    assert top_k_correct(logits, labels, k=1).tolist() == [False]
    assert top_k_correct(logits, labels, k=2).tolist() == [True]
    assert top_k_correct(logits, labels, k=3).tolist() == [True]


def test_single_sample_1d_logits():
    logits = np.array([0.2, 0.1, 0.7])
    assert top_k_correct(logits, np.array([2])).tolist() == [True]
    assert top_k_correct(logits, np.array([0])).tolist() == [False]


def test_topk_ties_and_full_k_equals_all_correct():
    logits = np.array([[1.0, 1.0, 1.0]])
    # k == n_classes: label is always within the top-k.
    for label in (0, 1, 2):
        assert top_k_correct(logits, np.array([label]), k=3).tolist() == [True]


def test_scoring_validation():
    logits = np.array([[0.1, 0.9], [0.5, 0.5]])
    with pytest.raises(ValueError):
        top_k_correct(logits, np.array([0]), k=1)          # label length mismatch
    with pytest.raises(ValueError):
        top_k_correct(logits, np.array([0, 1]), k=3)       # k > n_classes
    with pytest.raises(ValueError):
        top_k_correct(logits, np.array([0, 5]), k=1)       # label out of range
    with pytest.raises(ValueError):
        top_k_correct(np.zeros((2, 2, 2)), np.array([0, 1]))  # bad ndim
