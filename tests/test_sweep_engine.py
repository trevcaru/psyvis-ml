"""Sweep-engine tests: exact per-level counts, top-k, batched equivalence, order-independence."""

import warnings

import numpy as np
import pytest

from psyvis_ml.sweep import run_sweep

import synthetic as syn


def test_per_level_counts_deterministic():
    # 10 images with cutoffs 0.1..1.0; correct iff level >= cutoff.
    cutoffs = np.round(np.linspace(0.1, 1.0, 10), 3)
    images, labels = syn.make_cutoff_data(cutoffs)
    levels = [0.05, 0.35, 0.65, 1.0]

    bundle = run_sweep(syn.onehot_model, images, labels, levels, syn.cutoff_stimulus,
                       seed=0, stimulus_name="cutoff", model_name="onehot")

    # n_correct(level) == number of cutoffs <= level.
    expected = [int(np.count_nonzero(cutoffs <= lv)) for lv in levels]
    assert list(bundle.n_correct) == expected == [0, 3, 6, 10]
    assert list(bundle.n_trials) == [10, 10, 10, 10]  # every image at every level
    assert bundle.metadata["n_classes"] == syn.C


def test_n_trials_equals_n_images():
    cutoffs = np.linspace(0.1, 1.0, 7)
    images, labels = syn.make_cutoff_data(cutoffs)
    bundle = run_sweep(syn.onehot_model, images, labels, [0.5, 0.9], syn.cutoff_stimulus,
                       seed=1)
    assert all(nt == len(images) for nt in bundle.n_trials)


def test_topk_changes_counts():
    # Wrong prediction is always (class + 1) % C, so the true label sits 2nd once we
    # widen k... construct a model that ranks the true label second on failures.
    C = syn.C

    def near_miss_model(stim):
        pred = int(np.asarray(stim).ravel()[0])
        logits = np.full(C, -10.0)
        logits[pred] = 10.0
        logits[(pred - 1) % C] = 5.0  # 2nd place is the "true+1 -> true" neighbor
        return logits

    cutoffs = np.full(6, 0.5)
    images, labels = syn.make_cutoff_data(cutoffs)
    at_low = run_sweep(near_miss_model, images, labels, [0.1], syn.cutoff_stimulus, seed=0)
    at_low_k2 = run_sweep(near_miss_model, images, labels, [0.1], syn.cutoff_stimulus,
                          seed=0, top_k=2)
    # At level 0.1 (< cutoff), top-1 is wrong for all; top-2 recovers the true label.
    assert at_low.n_correct[0] == 0
    assert at_low_k2.n_correct[0] == len(images)


def test_batched_matches_per_image():
    cutoffs = np.round(np.linspace(0.1, 1.0, 12), 3)
    images, labels = syn.make_cutoff_data(cutoffs)
    levels = [0.2, 0.5, 0.8]

    per_image = run_sweep(syn.onehot_model, images, labels, levels, syn.cutoff_stimulus,
                          seed=7)
    batched = run_sweep(syn.onehot_model_batched, images, labels, levels,
                        syn.cutoff_stimulus_batched, seed=7, batched=True)
    assert list(per_image.n_correct) == list(batched.n_correct)


def test_order_independent_map_fn():
    # A map_fn that evaluates levels in reverse must still yield index-aligned counts.
    def reversed_map(fn, iterable):
        return [fn(i) for i in list(iterable)[::-1]]

    cutoffs = np.round(np.linspace(0.1, 1.0, 10), 3)
    images, labels = syn.make_cutoff_data(cutoffs)
    levels = [0.05, 0.35, 0.65, 1.0]

    seq = run_sweep(syn.onehot_model, images, labels, levels, syn.cutoff_stimulus, seed=3)
    rev = run_sweep(syn.onehot_model, images, labels, levels, syn.cutoff_stimulus, seed=3,
                    map_fn=reversed_map)
    assert list(seq.n_correct) == list(rev.n_correct)
    assert seq.config_hash == rev.config_hash


def test_validation_errors():
    images, labels = syn.make_cutoff_data(np.linspace(0.1, 1.0, 5))
    with pytest.raises(ValueError):
        run_sweep(syn.onehot_model, images, labels[:-1], [0.5], syn.cutoff_stimulus)
    with pytest.raises(ValueError):
        run_sweep(syn.onehot_model, images, labels, [], syn.cutoff_stimulus)


# --------------------------------------------------------------------------- #
# Name guard: lambdas / collisions would corrupt the reproducibility hash.
# --------------------------------------------------------------------------- #
def _small_data():
    return syn.make_cutoff_data(np.linspace(0.1, 1.0, 5))


def test_lambda_model_warns_about_unstable_hash():
    images, labels = _small_data()
    with pytest.warns(RuntimeWarning, match="lambda"):
        run_sweep(lambda img: syn.onehot_model(img), images, labels, [0.5],
                  syn.cutoff_stimulus, stimulus_name="cutoff")


def test_lambda_stimulus_warns():
    images, labels = _small_data()
    with pytest.warns(RuntimeWarning, match="lambda"):
        run_sweep(syn.onehot_model, images, labels, [0.5],
                  lambda img, lv, rng: syn.cutoff_stimulus(img, lv, rng),
                  model_name="onehot")


def test_name_collision_warns():
    images, labels = _small_data()
    with pytest.warns(RuntimeWarning, match="same name"):
        run_sweep(syn.onehot_model, images, labels, [0.5], syn.cutoff_stimulus,
                  model_name="samename", stimulus_name="samename")


def test_distinct_named_callables_do_not_warn():
    images, labels = _small_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_sweep(syn.onehot_model, images, labels, [0.5], syn.cutoff_stimulus,
                  model_name="onehot", stimulus_name="cutoff")
