"""RunBundle reproducibility and fingerprinting."""

import numpy as np

from psyvis_ml.sweep import run_sweep
from psyvis_ml.sweep.bundle import data_fingerprint, library_version

import synthetic as syn


def _run(seed):
    images, labels = syn.make_bernoulli_data(n=400, seed=999)  # data fixed; only sweep seed varies
    levels = list(np.geomspace(0.03, 0.5, 8))
    return run_sweep(syn.onehot_model, images, labels, levels, syn.bernoulli_stimulus,
                     seed=seed, model_name="onehot", stimulus_name="bernoulli")


def test_same_seed_and_config_reproduce_hash_and_counts():
    a = _run(seed=1234)
    b = _run(seed=1234)
    assert a.config_hash == b.config_hash
    assert a.n_correct == b.n_correct  # stochastic stimulus, but seeded -> identical
    assert a.seed == b.seed == 1234


def test_different_seed_changes_hash_and_counts():
    a = _run(seed=1)
    b = _run(seed=2)
    assert a.config_hash != b.config_hash          # seed is part of the config
    assert a.n_correct != b.n_correct              # different Bernoulli draws


def test_seed_none_is_resolved_and_recorded():
    images, labels = syn.make_bernoulli_data(n=100, seed=0)
    bundle = run_sweep(syn.onehot_model, images, labels, [0.1, 0.3], syn.bernoulli_stimulus,
                       seed=None)
    assert isinstance(bundle.seed, int)
    # Re-running with the recorded seed reproduces the counts.
    again = run_sweep(syn.onehot_model, images, labels, [0.1, 0.3], syn.bernoulli_stimulus,
                      seed=bundle.seed)
    assert bundle.n_correct == again.n_correct


def test_data_fingerprint_is_stable_and_data_sensitive():
    images, labels = syn.make_bernoulli_data(n=50, seed=5)
    fp1 = data_fingerprint(images, labels)
    fp2 = data_fingerprint(images.copy(), labels.copy())
    assert fp1 == fp2
    changed = labels.copy()
    changed[0] = (changed[0] + 1) % syn.C
    assert data_fingerprint(images, changed) != fp1


def test_to_fit_inputs_shapes_and_types():
    bundle = _run(seed=42)
    levels, n_correct, n_trials = bundle.to_fit_inputs()
    assert levels.shape == n_correct.shape == n_trials.shape
    assert np.all(n_correct <= n_trials)
    assert bundle.library_version == library_version()


def test_to_dict_roundtrip_keys():
    bundle = _run(seed=7)
    d = bundle.to_dict()
    for key in ("seed", "config_hash", "library_version", "levels", "n_correct",
                "n_trials", "top_k", "config", "metadata"):
        assert key in d
