"""Per-item export: write -> read round-trip, the signed-margin contract, and the schema doc.

This artifact is the contract with downstream type-2 (meta-d′) analysis, so the tests assert
the *contract*, not just that a file appears: every (item x level x condition x model) row is
present, the margin survives the round-trip bit-for-bit and stays SIGNED, correctness pairs
with it at the documented boundary of 0, and the severity axis follows the suite's declared
direction rather than the data.
"""

import csv
import json

import numpy as np
import pytest

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset, synthetic_dataset
from psyvis_ml.per_item import (
    MISSING_LABEL,
    PER_ITEM_COLUMNS,
    SCHEMA_VERSION,
    load_per_item,
    per_item_rows,
    write_per_item,
)
from psyvis_ml.suites import DegradationSuite

import synthetic as syn

NUM_CLASSES = 8
N_IMAGES = 40
N_LEVELS = 5


def _contrast_result(*, name="A", n=N_IMAGES, seed=1):
    """Increasing axis (higher contrast = easier)."""
    ds = synthetic_dataset(n=n, num_classes=NUM_CLASSES)
    model = syn.make_margin_observer(NUM_CLASSES, crossing=0.2, slope=8.0)
    levels = pe.linspace_levels(0.05, 0.5, N_LEVELS)
    suite = pe.suites.ContrastThreshold(contrast_metric="rms")
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed,
                      model_name=name)


def _degradation_result(*, name="B", n=N_IMAGES, seed=1):
    """Decreasing axis (higher severity = harder)."""
    images, labels = syn.make_patch_data(n, size=24, num_classes=NUM_CLASSES)
    ds = Dataset(images=images, labels=labels, num_classes=NUM_CLASSES)
    model = syn.make_degradation_observer(NUM_CLASSES, fill=0.0)
    suite = DegradationSuite("occlusion", fill=0.0, block=False)
    levels = pe.linspace_levels(0.05, 0.6, N_LEVELS)
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed,
                      model_name=name)


# --------------------------------------------------------------------------- #
# Rows: shape and content
# --------------------------------------------------------------------------- #
def test_rows_cover_every_item_level_condition():
    res = _contrast_result()
    rows = per_item_rows(res)
    n_conditions = len(res.condition_results)
    assert len(rows) == N_LEVELS * N_IMAGES * n_conditions
    # Every row carries every declared column, and nothing else.
    names = {c.name for c in PER_ITEM_COLUMNS}
    assert all(set(r) == names for r in rows)
    # Each (condition, level, item) appears exactly once.
    keys = {(r["condition"], r["level_index"], r["item_index"]) for r in rows}
    assert len(keys) == len(rows)


def test_rows_match_the_bundle_they_came_from():
    res = _contrast_result()
    rows = per_item_rows(res)
    cr = res.condition_results[0]
    margins = np.asarray(cr.bundle.per_image["margin"], dtype=float)
    correct = np.asarray(cr.bundle.per_image["correct"], dtype=bool)
    by_cell = {(r["level_index"], r["item_index"]): r for r in rows
               if r["condition"] == cr.label}
    for li in range(N_LEVELS):
        for ii in range(N_IMAGES):
            row = by_cell[(li, ii)]
            assert row["margin"] == pytest.approx(margins[li, ii])
            assert row["correct"] == bool(correct[li, ii])
            assert row["stimulus_level"] == pytest.approx(cr.bundle.levels[li])
            assert row["config_hash"] == cr.bundle.config_hash


def test_labels_are_carried_and_predicted_label_is_the_argmax():
    res = _contrast_result()
    ds_labels = np.asarray(res.dataset.labels)
    rows = per_item_rows(res)
    for r in rows:
        assert r["true_label"] == int(ds_labels[r["item_index"]])
        assert r["true_label"] != MISSING_LABEL
        assert r["predicted_label"] != MISSING_LABEL
        # Correct (top-1) iff the model's prediction is the true class.
        assert r["correct"] == (r["predicted_label"] == r["true_label"])


def test_multiple_results_land_in_one_table():
    a, b = _contrast_result(name="A"), _degradation_result(name="B")
    rows = per_item_rows([a, b])
    models = {r["model"] for r in rows}
    suites = {r["suite"] for r in rows}
    assert models == {"A", "B"}
    assert suites == {"ContrastThreshold", "DegradationSuite"}
    assert len(rows) == sum(N_LEVELS * N_IMAGES * len(r.condition_results) for r in (a, b))


# --------------------------------------------------------------------------- #
# The signed-margin contract (the discipline this artifact exists to preserve)
# --------------------------------------------------------------------------- #
def test_margin_is_persisted_signed_not_absolute(tmp_path):
    """Negative margins must survive to disk: abs() is the consumer's call, never ours."""
    res = _contrast_result()
    export = write_per_item(res, tmp_path)
    table = load_per_item(export.data_path)
    margin = table["margin"]

    assert np.any(margin < 0), "fixture produced no incorrect trials; the test cannot bite"
    assert np.any(margin > 0)
    # The signal is raw, not folded: it is NOT its own absolute value.
    assert not np.allclose(margin, np.abs(margin))
    # And the sidecar says so, in the file itself.
    assert table.metadata["margin_is_signed"] is True
    assert table.metadata["margin_boundary"] == 0.0
    assert "abs" not in table.metadata["margin_definition"].lower()


def test_margin_boundary_at_zero_matches_correctness(tmp_path):
    """The documented boundary: for top_k=1, correct == (margin > 0)."""
    export = write_per_item(_contrast_result(), tmp_path)
    table = load_per_item(export.data_path)
    assert np.array_equal(table["correct"], table["margin"] > 0)


# --------------------------------------------------------------------------- #
# decision_margin: the DECISION-referenced (type-2) signal, schema 1.1
# --------------------------------------------------------------------------- #
def test_decision_margin_equals_margin_on_correct_trials_only(tmp_path):
    """The contract that defines the column: identical when right, divergent when wrong.

    On a correct trial the target IS the winner, so "target vs. its best competitor" and
    "winner vs. runner-up" are the same subtraction. On an error they must come apart — that
    divergence is the entire reason the column exists, so a fixture without both kinds of trial
    cannot test it.
    """
    export = write_per_item(_contrast_result(), tmp_path)
    table = load_per_item(export.data_path)
    margin, decision, correct = table["margin"], table["decision_margin"], table["correct"]

    assert correct.any() and (~correct).any(), "fixture lacks both outcomes; test cannot bite"

    # Correct trials: exactly equal (bit-for-bit through the CSV round-trip, not approx).
    assert np.array_equal(decision[correct], margin[correct])
    # Error trials: margin goes negative while decision_margin cannot, so they always differ.
    # (Not `decision > 0`: this observer sets every competitor to the same logit, so on an error
    # the winner and runner-up tie and decision_margin is legitimately 0. Non-negativity and the
    # strict inequality below are the real contract; a positive value is not.)
    assert np.all(margin[~correct] < 0)
    assert np.all(decision[~correct] >= 0)
    assert np.all(decision[~correct] > margin[~correct])


def test_decision_margin_is_non_negative_everywhere(tmp_path):
    """Winner minus runner-up cannot be negative, on any trial, at any level."""
    export = write_per_item([_contrast_result(), _degradation_result()], tmp_path)
    table = load_per_item(export.data_path)
    assert np.all(table["decision_margin"] >= 0)


def test_decision_margin_is_documented_as_the_type2_signal(tmp_path):
    """The file must tell a consumer which column to score a type-2 ROC on."""
    export = write_per_item(_contrast_result(), tmp_path)
    table = load_per_item(export.data_path)
    assert table.metadata["type2_confidence_column"] == "decision_margin"
    assert "meta-d" in table.metadata["decision_margin_definition"]
    # And it warns off the trap it exists to prevent.
    assert "abs(margin)" in table.metadata["decision_margin_definition"]


def test_decision_margin_is_nan_for_a_bundle_that_never_recorded_it(tmp_path):
    """A pre-1.1 / hand-built bundle must still export, with the new column as NaN — not fail."""
    res = _contrast_result()
    bundle = res.condition_results[0].bundle
    bundle.per_image.pop("decision_margin")     # simulate a bundle written before schema 1.1

    table = load_per_item(write_per_item(res, tmp_path).data_path)
    assert np.all(np.isnan(table["decision_margin"]))
    assert not np.any(np.isnan(table["margin"]))   # the 1.0 columns are untouched


# --------------------------------------------------------------------------- #
# Severity axis: declared by the suite, not inferred from the data
# --------------------------------------------------------------------------- #
def test_severity_rank_follows_the_declared_axis_direction():
    # Decreasing suite (degradation): higher level = worse, so rank rises with level.
    rows = per_item_rows(_degradation_result())
    by_level = {r["level_index"]: (r["stimulus_level"], r["severity_rank"]) for r in rows}
    levels = [by_level[i][0] for i in range(N_LEVELS)]
    ranks = [by_level[i][1] for i in range(N_LEVELS)]
    assert levels == sorted(levels)
    assert ranks == list(range(N_LEVELS))          # cleanest (rank 0) = lowest severity

    # Increasing suite (contrast): higher level = easier, so rank FALLS with level.
    rows = per_item_rows(_contrast_result())
    by_level = {r["level_index"]: (r["stimulus_level"], r["severity_rank"]) for r in rows}
    levels = [by_level[i][0] for i in range(N_LEVELS)]
    ranks = [by_level[i][1] for i in range(N_LEVELS)]
    assert levels == sorted(levels)
    assert ranks == list(range(N_LEVELS - 1, -1, -1))   # cleanest (rank 0) = highest contrast


def test_severity_rank_zero_is_the_confidence_baseline_level():
    """severity_rank 0 must name the same 'clean' level the confidence readout baselines on."""
    for res in (_contrast_result(), _degradation_result()):
        readout = res.confidence(n_boot=20)
        rows = per_item_rows(res)
        clean = {r["level_index"] for r in rows if r["severity_rank"] == 0}
        assert clean == {readout.baseline_index}


# --------------------------------------------------------------------------- #
# Round-trip: write -> read -> same data
# --------------------------------------------------------------------------- #
def test_csv_round_trip_is_exact(tmp_path):
    res = _contrast_result()
    rows = per_item_rows(res)
    export = write_per_item(res, tmp_path)

    assert export.data_path.name == "per_item.csv"
    assert export.metadata_path.name == "per_item.meta.json"
    assert export.n_rows == len(rows)

    table = load_per_item(export.data_path)
    assert len(table) == len(rows)
    assert table.names == [c.name for c in PER_ITEM_COLUMNS]

    back = table.rows()
    for original, restored in zip(rows, back, strict=True):
        for col in PER_ITEM_COLUMNS:
            a, b = original[col.name], restored[col.name]
            if col.dtype == "float":
                # repr() round-trip: floats come back bit-identical, not merely close.
                assert a == b or (np.isnan(a) and np.isnan(b))
            else:
                assert a == b


def test_loaded_dtypes_follow_the_schema(tmp_path):
    export = write_per_item(_contrast_result(), tmp_path)
    table = load_per_item(export.data_path)
    kinds = {"str": "O", "int": "i", "float": "f", "bool": "b"}
    for col in PER_ITEM_COLUMNS:
        assert table[col.name].dtype.kind == kinds[col.dtype], col.name


def test_file_is_readable_without_psyvis(tmp_path):
    """The artifact is a plain CSV + JSON sidecar: stdlib alone must be enough to consume it."""
    export = write_per_item(_contrast_result(), tmp_path)

    with export.data_path.open(encoding="utf-8", newline="") as fh:
        records = list(csv.DictReader(fh))
    assert len(records) == export.n_rows
    assert records[0]["correct"] in ("0", "1")          # documented CSV encoding of bool
    float(records[0]["margin"])                          # parses as a float
    int(records[0]["true_label"])

    meta = json.loads(export.metadata_path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == SCHEMA_VERSION
    assert [c["name"] for c in meta["columns"]] == [c.name for c in PER_ITEM_COLUMNS]


def test_sidecar_carries_reproducibility_metadata(tmp_path):
    res = _contrast_result()
    export = write_per_item(res, tmp_path, extra_metadata={"note": "hello"})
    meta = export.metadata

    bundle = res.condition_results[0].bundle
    run = next(r for r in meta["runs"] if r["config_hash"] == bundle.config_hash)
    assert run["seed"] == bundle.seed
    assert run["library_version"] == bundle.library_version
    assert run["top_k"] == bundle.top_k
    assert run["levels"] == [float(x) for x in bundle.levels]
    assert run["n_correct"] == [int(x) for x in bundle.n_correct]
    assert run["data_fingerprint"] == bundle.config["data_fingerprint"]
    assert run["axis_direction"] == ("decreasing" if res.decreasing else "increasing")
    assert meta["library_versions"]["numpy"] == np.__version__
    assert meta["extra"]["note"] == "hello"
    assert meta["missing_label"] == MISSING_LABEL


def test_load_tolerates_a_missing_sidecar_and_unknown_columns(tmp_path):
    export = write_per_item(_contrast_result(), tmp_path)

    # Sidecar gone: the data file still stands on its own.
    export.metadata_path.unlink()
    table = load_per_item(export.data_path)
    assert table.metadata == {}
    assert len(table) == export.n_rows

    # An extra column from a future (additive) schema bump must not break the reader.
    lines = export.data_path.read_text(encoding="utf-8").splitlines()
    lines[0] += ",future_column"
    lines[1:] = [ln + ",x" for ln in lines[1:]]
    export.data_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = load_per_item(export.data_path)
    assert table["future_column"][0] == "x"
    assert len(table["margin"]) == export.n_rows


def test_load_rejects_a_file_missing_required_columns(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("model,margin\nA,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required per-item columns"):
        load_per_item(path)


def test_item_ids_from_the_dataset_are_used_as_row_ids(tmp_path):
    ds = synthetic_dataset(n=N_IMAGES, num_classes=NUM_CLASSES)
    ids = tuple(f"img_{i:03d}.png" for i in range(N_IMAGES))
    ds = Dataset(images=ds.images, labels=ds.labels, num_classes=NUM_CLASSES, item_ids=ids)
    res = pe.measure(model=syn.make_contrast_observer(NUM_CLASSES),
                     suite=pe.suites.ContrastThreshold(contrast_metric="rms"),
                     dataset=ds, levels=pe.linspace_levels(0.05, 0.5, 3), seed=0,
                     model_name="A")
    table = load_per_item(write_per_item(res, tmp_path).data_path)
    assert set(table["item_id"]) == set(ids)
    # Without ids, rows fall back to the positional index.
    plain = _contrast_result()
    assert {r["item_id"] for r in per_item_rows(plain)} == {str(i) for i in range(N_IMAGES)}


@pytest.mark.parametrize("fmt", ["parquet"])
def test_parquet_round_trip(tmp_path, fmt):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    res = _contrast_result()
    export = write_per_item(res, tmp_path, fmt=fmt)
    assert export.data_path.suffix == ".parquet"
    table = load_per_item(export.data_path)
    rows = per_item_rows(res)
    assert len(table) == len(rows)
    assert np.allclose(table["margin"], [r["margin"] for r in rows])
    assert np.array_equal(table["correct"], [r["correct"] for r in rows])
    assert table.metadata["schema_version"] == SCHEMA_VERSION


def test_write_rejects_an_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="fmt must be"):
        write_per_item(_contrast_result(), tmp_path, fmt="feather")


# --------------------------------------------------------------------------- #
# Wiring: the export ships with a normal run, and the aggregate output is untouched
# --------------------------------------------------------------------------- #
def test_measure_result_writes_the_export(tmp_path):
    res = _contrast_result()
    export = res.write_per_item(tmp_path / "run")
    assert export.data_path.exists() and export.metadata_path.exists()
    assert len(load_per_item(export.data_path)) == export.n_rows


def test_report_bundle_ships_the_per_item_file(tmp_path):
    res = _contrast_result()
    rb = res.report(outdir=tmp_path / "bundle", n_boot=20, seed=0)

    assert rb.per_item is not None
    assert rb.per_item.data_path.exists()
    assert all(p.exists() for p in rb.artifacts())
    # Additive: the aggregate metadata.json still holds its per-level counts and fits, and now
    # also points at the trial-level file.
    meta = json.loads(rb.metadata_path.read_text(encoding="utf-8"))
    assert meta["models"][0]["conditions"][0]["n_correct"]
    assert meta["per_item"]["data"] == rb.per_item.data_path.name
    assert meta["per_item"]["n_rows"] == rb.per_item.n_rows
    # The report tells a reader the margin is signed with a boundary at 0.
    assert "signed" in rb.markdown.lower() and "SCHEMA.md" in rb.markdown


def test_export_refuses_a_bundle_without_per_image_signals(tmp_path):
    from psyvis_ml.sweep import RunBundle

    res = _contrast_result()
    cr = res.condition_results[0]
    stripped = RunBundle(seed=cr.bundle.seed, config_hash=cr.bundle.config_hash,
                         library_version=cr.bundle.library_version, levels=cr.bundle.levels,
                         n_correct=cr.bundle.n_correct, n_trials=cr.bundle.n_trials, top_k=1)
    res.condition_results[0] = type(cr)(label=cr.label, bundle=stripped, fit=cr.fit)
    with pytest.raises(ValueError, match="nothing per-item to export"):
        write_per_item(res, tmp_path)


# --------------------------------------------------------------------------- #
# The schema doc IS the contract: keep it in sync with the code.
# --------------------------------------------------------------------------- #
def test_schema_md_documents_every_column():
    import pathlib

    schema = (pathlib.Path(__file__).resolve().parents[1] / "SCHEMA.md").read_text(
        encoding="utf-8")
    assert f"schema v{SCHEMA_VERSION}" in schema
    for col in PER_ITEM_COLUMNS:
        assert f"`{col.name}`" in schema, f"SCHEMA.md does not document column {col.name!r}"
    # And it states the two things a consumer can get wrong.
    assert "boundary is 0" in schema
    assert "never applies `abs()`" in schema
