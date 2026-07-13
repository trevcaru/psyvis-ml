"""Tests for top-1 / top-k argmax correctness scoring and the two confidence margins."""

import numpy as np
import pytest

from psyvis_ml.sweep.scoring import count_correct, score_logits, top_k_correct


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


# --------------------------------------------------------------------------- #
# The two margins: target-referenced (margin) vs. decision-referenced (decision_margin)
# --------------------------------------------------------------------------- #
def test_the_two_margins_on_a_hand_computed_case():
    logits = np.array([
        [3.0, 1.0, 0.5],   # target 0 = winner  -> correct
        [3.0, 1.0, 0.5],   # target 1, winner 0 -> error
    ])
    scores = score_logits(logits, np.array([0, 1]), k=1)

    assert scores.correct.tolist() == [True, False]
    # margin is TARGET-referenced: logit[true] - best other.
    assert scores.margin == pytest.approx([3.0 - 1.0, 1.0 - 3.0])      # +2.0, -2.0
    # decision_margin is DECISION-referenced: winner - runner-up. Same logits, so it is the
    # same 2.0 on BOTH rows — it does not know, and must not care, which class was the target.
    assert scores.decision_margin == pytest.approx([2.0, 2.0])


def test_decision_margin_equals_margin_exactly_when_correct():
    """The invariant: on a correct trial the target IS the winner, so the two coincide."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(400, 12))
    labels = rng.integers(0, 12, 400)
    s = score_logits(logits, labels, k=1)

    assert s.correct.any() and (~s.correct).any(), "need both outcomes for this to bite"
    np.testing.assert_array_equal(s.decision_margin[s.correct], s.margin[s.correct])
    # On errors they must diverge: margin < 0 while decision_margin stays > 0.
    assert np.all(s.margin[~s.correct] < 0)
    assert np.all(s.decision_margin[~s.correct] > 0)


def test_decision_margin_matches_an_independent_full_sort_reference():
    """`partition` is an optimization; check it against the obvious (slow) definition."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 7))
    labels = rng.integers(0, 7, 200)

    ordered = np.sort(logits, axis=1)          # ascending; top-1 last, top-2 second-to-last
    expected = ordered[:, -1] - ordered[:, -2]
    np.testing.assert_allclose(score_logits(logits, labels).decision_margin, expected)


def test_decision_margin_is_non_negative_and_zero_only_on_a_tied_winner():
    logits = np.array([
        [2.0, 2.0, 0.0],   # tied winner -> zero confidence in the decision
        [2.0, 1.0, 0.0],
    ])
    s = score_logits(logits, np.array([0, 0]), k=1)
    assert np.all(s.decision_margin >= 0)
    assert s.decision_margin[0] == pytest.approx(0.0)
    assert s.decision_margin[1] == pytest.approx(1.0)


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
