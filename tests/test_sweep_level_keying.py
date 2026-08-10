"""The stimulus RNG must be keyed to a level's VALUE, not its position in the levels list.

A sweep's noise realizations are part of its stimuli. If the RNG key is the level *index*,
then adding, removing, or reordering levels silently re-randomizes the stimuli at levels that
were already measured — so two sweeps that share a level are not showing the same thing there,
and their overlapping rows cannot be pooled or compared. These tests pin the invariance
directly: for a fixed seed, the per-item row at a given (condition, item, level VALUE) must be
bit-identical regardless of where that level sat in the list, how long the list was, or which
other levels were swept alongside it.

The deterministic suites (contrast, distractor size) never consumed the rng in the first place;
they are included as controls that the keying change left them untouched.
"""

import csv
import warnings

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset

N_IMAGES = 12
SIZE = 8
NCLS = 6
SEED = 4242

#: The three level lists the invariance is asserted over: a reference order, the same levels
#: reversed, and a strict subset. Every level shared with the reference must reproduce exactly.
LEVELS_REF = [0.05, 0.1, 0.2, 0.4]
LEVELS_REVERSED = [0.4, 0.2, 0.1, 0.05]
LEVELS_SUBSET = [0.1, 0.2]

#: Distractor size is swept in pixels, so it needs its own (equivalent) triple.
PX_REF = [2.0, 4.0, 6.0, 8.0]
PX_REVERSED = [8.0, 6.0, 4.0, 2.0]
PX_SUBSET = [4.0, 6.0]


def _dataset():
    """A tiny labeled patch set with stable item ids."""
    rng = np.random.default_rng(0)
    images = rng.uniform(0.2, 0.8, size=(N_IMAGES, SIZE, SIZE))
    labels = (np.arange(N_IMAGES) % NCLS).astype(int)
    return Dataset(images=images, labels=labels, num_classes=NCLS, name="keying-fixture",
                   item_ids=tuple(f"item{i:02d}" for i in range(N_IMAGES)))


def _model(dataset):
    """Fixed linear readout, tilted toward each class's own items.

    The tilt puts accuracy in the middle of the range so that a changed noise field flips the
    `correct` bit on some trials, not just the `margin` float — otherwise the test could pass
    on a correctness column that was saturated all along.
    """
    rng = np.random.default_rng(1)
    w = rng.normal(0.0, 0.05, size=(NCLS, SIZE * SIZE))
    for i, img in enumerate(dataset.images):
        flat = img.ravel()
        w[dataset.labels[i]] += 0.30 * flat / np.linalg.norm(flat)

    def linear_probe(image):
        a = np.clip(np.asarray(image, dtype=float), 0.0, 1.0).ravel()
        return w @ a

    linear_probe.__name__ = "linear_probe"
    return linear_probe


def _per_item(tmp_path, model, dataset, suite, levels, tag):
    """Run one sweep and return its per-item rows keyed by (condition, item_id, level).

    Values are the raw CSV strings, so equality is a *bit*-level check: `margin` is written
    with `repr(float)` (shortest round-trip), and any change in the low bits shows up.
    """
    with warnings.catch_warnings():
        # A 2-level list underdetermines the sigmoid; the fit is irrelevant here, only the
        # per-item rows are, so a non-converged fit must not fail the test.
        warnings.simplefilter("ignore")
        result = pe.measure(model, suite, dataset, levels, seed=SEED,
                            model_name="linear_probe")
        export = result.write_per_item(tmp_path / tag, stem="per_item")

    with export.data_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    keyed = {(r["condition"], r["item_id"], r["stimulus_level"]): r for r in rows}
    assert len(keyed) == len(rows), "join key is not unique"
    return keyed


def _assert_shared_levels_identical(ref, other, *, expected_shared):
    shared = sorted(set(ref) & set(other))
    assert len(shared) == expected_shared, (
        f"expected {expected_shared} shared (condition, item, level) cells, got {len(shared)}"
    )
    mismatched = [
        (key, ref[key]["margin"], other[key]["margin"],
         ref[key]["correct"], other[key]["correct"])
        for key in shared
        if ref[key]["margin"] != other[key]["margin"]
        or ref[key]["correct"] != other[key]["correct"]
    ]
    assert not mismatched, (
        f"{len(mismatched)}/{len(shared)} shared cells changed when the levels list changed; "
        f"first three: {mismatched[:3]}"
    )


# --------------------------------------------------------------------------- #
# The invariance itself.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suite_factory,stochastic", [
    (lambda: pe.suites.DegradationSuite("gaussian_noise"), True),
    (lambda: pe.suites.DegradationSuite("occlusion"), True),
    (lambda: pe.suites.ContrastThreshold(), False),
])
def test_level_value_keying_survives_reorder_and_subset(tmp_path, suite_factory, stochastic):
    dataset = _dataset()
    model = _model(dataset)

    ref = _per_item(tmp_path, model, dataset, suite_factory(), LEVELS_REF, "ref")
    rev = _per_item(tmp_path, model, dataset, suite_factory(), LEVELS_REVERSED, "rev")
    sub = _per_item(tmp_path, model, dataset, suite_factory(), LEVELS_SUBSET, "sub")

    # Reversed: every level is shared, so every cell must match.
    _assert_shared_levels_identical(ref, rev,
                                    expected_shared=len(LEVELS_REF) * N_IMAGES)
    # Subset: only the two overlapping levels are shared, and they must match too — a shorter
    # sweep must not re-randomize the levels it has in common with a longer one.
    _assert_shared_levels_identical(ref, sub,
                                    expected_shared=len(LEVELS_SUBSET) * N_IMAGES)


def test_distractor_size_keying_survives_reorder_and_subset(tmp_path):
    dataset = _dataset()
    model = _model(dataset)

    def suite():
        return pe.suites.DistractorRobustness(canvas_shape=(SIZE, SIZE), n_distractors=4)

    n_conditions = 2  # distractors + undistracted baseline
    ref = _per_item(tmp_path, model, dataset, suite(), PX_REF, "d_ref")
    rev = _per_item(tmp_path, model, dataset, suite(), PX_REVERSED, "d_rev")
    sub = _per_item(tmp_path, model, dataset, suite(), PX_SUBSET, "d_sub")

    _assert_shared_levels_identical(ref, rev,
                                    expected_shared=len(PX_REF) * N_IMAGES * n_conditions)
    _assert_shared_levels_identical(ref, sub,
                                    expected_shared=len(PX_SUBSET) * N_IMAGES * n_conditions)


# --------------------------------------------------------------------------- #
# Guards on the key derivation itself.
# --------------------------------------------------------------------------- #
def test_level_key_is_process_stable_and_canonicalizes_equal_floats():
    """The key must not depend on PYTHONHASHSEED, and 0.1 / 0.10 / 1e-1 must agree."""
    from psyvis_ml.sweep.engine import _level_key

    assert _level_key("s", 0.1) == _level_key("s", 0.10) == _level_key("s", 1e-1)
    assert _level_key("s", 0.0) == _level_key("s", -0.0)
    assert _level_key("s", 0.1) != _level_key("s", 0.2)
    # A known constant: if this changes, prior runs stop reproducing — that is a deliberate,
    # documented break, not something to re-baseline silently.
    assert isinstance(_level_key("s", 0.1), int)
    assert 0 <= _level_key("s", 0.1) < 2 ** 64


def test_different_streams_do_not_collide_at_the_same_numeric_level():
    """Two suites/conditions sweeping the same number must draw different noise."""
    from psyvis_ml.sweep.engine import _level_key

    keys = {
        _level_key("DegradationSuite|degradation[gaussian_noise]|deg:gn", 0.2),
        _level_key("DegradationSuite|degradation[occlusion]|deg:occ", 0.2),
        _level_key("ContrastThreshold|contrast[rms]|contrast:rms", 0.2),
    }
    assert len(keys) == 3


def test_stochastic_suites_actually_exercise_the_rng(tmp_path):
    """Sanity: the invariance above would be vacuous if the noise never moved anything.

    Changing only the SEED (same levels, same order) must change the rows for a stochastic
    suite — otherwise the test above proves nothing about noise keying.
    """
    dataset = _dataset()
    model = _model(dataset)
    suite = pe.suites.DegradationSuite("gaussian_noise")

    a = _per_item(tmp_path, model, dataset, suite, LEVELS_REF, "seed_a")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = pe.measure(model, pe.suites.DegradationSuite("gaussian_noise"), dataset,
                           LEVELS_REF, seed=SEED + 1, model_name="linear_probe")
        export = other.write_per_item(tmp_path / "seed_b", stem="per_item")
    with export.data_path.open(newline="", encoding="utf-8") as fh:
        b = {(r["condition"], r["item_id"], r["stimulus_level"]): r
             for r in csv.DictReader(fh)}

    changed = sum(1 for k in a if a[k]["margin"] != b[k]["margin"])
    assert changed > 0.9 * len(a), (
        f"only {changed}/{len(a)} rows moved with the seed; the noise draw is not reaching "
        "the model, so the keying invariance test would be vacuous"
    )
