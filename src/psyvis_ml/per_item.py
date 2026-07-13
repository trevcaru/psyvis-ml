"""Per-item sweep export — the persisted trial-level artifact.

``results.json`` (and the report bundle's ``metadata.json``) are **aggregate**: per-level
``(n_correct, n_trials)`` counts plus fits. Those cannot reconstruct a type-2 analysis
(meta-d′, type-2 ROC, folded confidence distributions), because every such analysis needs the
*trial-level* pairing of a **confidence signal** with an **outcome**. This module persists
exactly that: one row per (item × stimulus level × condition × model), carrying the signed
logit margin and the correctness bit the sweep already computed but previously threw away.

The artifact is a plain data file (CSV, or Parquet when pandas/pyarrow are installed) plus a
JSON sidecar of run metadata — readable **without psyvis-ml installed**. :func:`load_per_item`
is a convenience, not a requirement; ``SCHEMA.md`` at the repo root is the contract, and
:data:`PER_ITEM_COLUMNS` is its machine-readable form.

Discipline
----------
The persisted ``margin`` is the **raw signal**: signed, with its decision boundary at ``0``
(``margin > 0`` iff the target is the top-1 prediction). We deliberately do **not** fold it to
``abs(margin)`` on export. Absolute value is one *scoring* choice — the natural one for
type-2/meta-d′, where confidence is unsigned magnitude and the outcome carries the sign — and
it is the consumer's to make. Exporting the signed value keeps both the confidence magnitude
and the direction of evidence recoverable; exporting ``abs()`` would destroy the latter
irreversibly.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .sweep.bundle import library_version

__all__ = [
    "PER_ITEM_COLUMNS",
    "SCHEMA_VERSION",
    "MISSING_LABEL",
    "MARGIN_DEFINITION",
    "Column",
    "PerItemExport",
    "PerItemTable",
    "per_item_rows",
    "write_per_item",
    "load_per_item",
]

#: Bumped only on a breaking change to the columns or their semantics (additive columns are a
#: minor bump; consumers must tolerate unknown extra columns).
SCHEMA_VERSION = "1.0"

#: Sentinel for an integer label column whose value is unavailable for this run.
MISSING_LABEL = -1

MARGIN_DEFINITION = (
    "margin = logit[true_label] - max(logit[j] for j != true_label). Signed; decision boundary "
    "at 0 (margin > 0 iff the true label is the top-1 prediction, so for top_k=1 "
    "correct == (margin > 0)). Raw evidence difference, NOT a softmax probability. Logit "
    "scales are not comparable across models: never compare raw margins between models, only "
    "within a model (or as a change from that model's own clean baseline)."
)


@dataclass(frozen=True)
class Column:
    """One column of the per-item table: its name, logical dtype, and meaning."""

    name: str
    dtype: str          # "str" | "int" | "float" | "bool"
    description: str


PER_ITEM_COLUMNS: tuple[Column, ...] = (
    Column("model", "str",
           "Model identifier (MeasureResult.model_name). Rows from different models are "
           "distinguished by this column alone."),
    Column("suite", "str",
           "Measurement suite class name (e.g. ContrastThreshold, DegradationSuite)."),
    Column("condition", "str",
           "Condition/slice label within the suite (e.g. 'distractors', 'baseline'). One "
           "sweep — and one psychometric fit — per (model, suite, condition)."),
    Column("config_hash", "str",
           "SHA-256 config hash of the sweep this row came from; joins the row to its entry "
           "in the sidecar's 'runs' list."),
    Column("item_id", "str",
           "Stable identifier for the source image. Falls back to the decimal image index "
           "when the dataset supplied no ids. The same item_id appears once per stimulus "
           "level (method of constant stimuli: every item is shown at every level)."),
    Column("item_index", "int",
           "Zero-based positional index of the image within the dataset."),
    Column("level_index", "int",
           "Zero-based index of the stimulus level within the swept level list (in the order "
           "the levels were swept, which is the order in the sidecar's 'levels')."),
    Column("stimulus_level", "float",
           "The stimulus level in the suite's native units (RMS contrast, noise sigma, "
           "distractor size in px, ...). Its direction is NOT self-evident — see "
           "severity_rank and the sidecar's axis_direction."),
    Column("severity_rank", "int",
           "Direction-aware ordinal position of this level on the clean -> degraded axis: 0 "
           "is the cleanest/easiest level, n_levels-1 the most degraded/hardest. Derived from "
           "the suite's DECLARED axis direction, never inferred from the data. Use this when "
           "you need a monotone severity axis without knowing the suite's units."),
    Column("margin", "float",
           "PRIMARY CONFIDENCE SIGNAL. Signed logit margin; boundary at 0. See "
           "MARGIN_DEFINITION / the sidecar's margin_definition."),
    Column("correct", "bool",
           "Was the true label among the model's top-k predictions? (top_k is in the "
           "sidecar; it is 1 unless stated otherwise.) Encoded as 0/1 in CSV."),
    Column("true_label", "int",
           f"True class index of the item, or {MISSING_LABEL} if unavailable."),
    Column("predicted_label", "int",
           f"The model's top-1 class index (argmax of the logit vector), or {MISSING_LABEL} "
           "if unavailable. Present here so a future binary/2AFC meta-d' can recover the "
           "response, not just its correctness."),
    Column("target_rank", "int",
           "Rank of the true class in the logit vector; 1 == the true class is top-1."),
    Column("target_logit", "float",
           "Raw logit assigned to the true class (arbitrary per-model scale)."),
    Column("max_softmax", "float",
           "Max softmax probability. REFERENCE ONLY and calibration-sensitive — do not treat "
           "as a calibrated probability, and do not use it as the confidence signal in place "
           "of margin."),
)

_COLUMN_BY_NAME = {c.name: c for c in PER_ITEM_COLUMNS}

# per_image keys on the RunBundle -> per-item column name, for the (n_levels, n_images) arrays.
_PER_IMAGE_COLUMNS = {
    "margin": "margin",
    "correct": "correct",
    "predicted_label": "predicted_label",
    "target_rank": "target_rank",
    "target_logit": "target_logit",
    "max_softmax": "max_softmax",
}


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #
def _severity_ranks(levels, decreasing: bool) -> np.ndarray:
    """Ordinal clean -> degraded position of each level (0 == cleanest).

    Direction is *declared* by the suite, not inferred: a *decreasing* suite (degradation,
    distractor size) gets worse as the level rises, so the lowest level is cleanest; an
    *increasing* suite (contrast) gets better as the level rises, so the highest level is
    cleanest. This mirrors the baseline choice in :mod:`psyvis_ml.confidence`.
    """
    levels = np.asarray(levels, dtype=float)
    order = np.argsort(levels if decreasing else -levels, kind="stable")
    ranks = np.empty(levels.size, dtype=int)
    ranks[order] = np.arange(levels.size)
    return ranks


def _bundle_array(bundle, key, n_levels, n_images):
    arr = bundle.per_image.get(key)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.shape != (n_levels, n_images):
        raise ValueError(
            f"bundle.per_image[{key!r}] has shape {arr.shape}, expected "
            f"{(n_levels, n_images)} (n_levels, n_images)."
        )
    return arr


def per_item_rows(results) -> list[dict]:
    """Build the per-item rows for one :class:`~psyvis_ml.MeasureResult` or a list of them.

    One row per (model × suite × condition × item × stimulus level). Returns plain dicts keyed
    by :data:`PER_ITEM_COLUMNS`; ints/floats/bools are native Python types (JSON/CSV-ready).

    Raises ``ValueError`` if a sweep recorded no per-image signals (nothing to export).
    """
    rows: list[dict] = []
    for result in _as_result_list(results):
        suite_name = result.suite.__class__.__name__
        model = result.model_name or "model"
        for cr in result.condition_results:
            b = cr.bundle
            if "margin" not in b.per_image:
                raise ValueError(
                    f"sweep {model}/{suite_name}/{cr.label} recorded no per-image signals, so "
                    "there is nothing per-item to export. (Bundles produced by run_sweep() "
                    "always record them; this bundle was constructed by hand.)"
                )
            levels = np.asarray(b.levels, dtype=float)
            n_levels = levels.size
            margin = np.asarray(b.per_image["margin"], dtype=float)
            if margin.ndim != 2 or margin.shape[0] != n_levels:
                raise ValueError(
                    f"margins shape {margin.shape} inconsistent with {n_levels} levels."
                )
            n_images = margin.shape[1]
            # Optional signals (a hand-built bundle may omit them) come back as None and are
            # written as the missing sentinel / NaN rather than failing the export.
            arrays = {col: _bundle_array(b, key, n_levels, n_images)
                      for key, col in _PER_IMAGE_COLUMNS.items()}
            if arrays["correct"] is None:
                raise ValueError(
                    f"sweep {model}/{suite_name}/{cr.label} recorded margins but no per-image "
                    "'correct' array; a per-item row needs both (confidence + outcome)."
                )
            ranks = _severity_ranks(levels, result.decreasing)
            true_labels = (tuple(b.true_labels) if b.true_labels
                           else (MISSING_LABEL,) * n_images)
            item_ids = tuple(b.item_ids) if b.item_ids else tuple(str(i) for i in
                                                                  range(n_images))

            for li in range(n_levels):
                for ii in range(n_images):
                    row = {
                        "model": model,
                        "suite": suite_name,
                        "condition": cr.label,
                        "config_hash": b.config_hash,
                        "item_id": item_ids[ii],
                        "item_index": int(ii),
                        "level_index": int(li),
                        "stimulus_level": float(levels[li]),
                        "severity_rank": int(ranks[li]),
                        "margin": float(arrays["margin"][li, ii]),
                        "correct": bool(arrays["correct"][li, ii]),
                        "true_label": int(true_labels[ii]),
                        "predicted_label": _int_or_missing(arrays["predicted_label"], li, ii),
                        "target_rank": _int_or_missing(arrays["target_rank"], li, ii),
                        "target_logit": _float_or_nan(arrays["target_logit"], li, ii),
                        "max_softmax": _float_or_nan(arrays["max_softmax"], li, ii),
                    }
                    rows.append(row)
    return rows


def _int_or_missing(arr, li, ii) -> int:
    return MISSING_LABEL if arr is None else int(arr[li, ii])


def _float_or_nan(arr, li, ii) -> float:
    return float("nan") if arr is None else float(arr[li, ii])


def _as_result_list(results) -> list:
    if hasattr(results, "condition_results"):   # a single MeasureResult
        return [results]
    results = list(results)
    if not results:
        raise ValueError("no MeasureResult given; nothing to export.")
    return results


# --------------------------------------------------------------------------- #
# Metadata sidecar
# --------------------------------------------------------------------------- #
def _run_metadata(result, cr) -> dict:
    """Reproducibility metadata for one sweep — reuses what the RunBundle already captured."""
    b = cr.bundle
    return {
        "model": result.model_name or "model",
        "suite": result.suite.__class__.__name__,
        "condition": cr.label,
        "config_hash": b.config_hash,
        "seed": b.seed,
        "library_version": b.library_version,
        "top_k": b.top_k,
        "chance_level": _jsonable(result.chance_level),
        "axis_direction": "decreasing" if result.decreasing else "increasing",
        "levels": [float(x) for x in b.levels],
        "n_correct": [int(x) for x in b.n_correct],
        "n_trials": [int(x) for x in b.n_trials],
        "n_images": int(b.metadata.get("n_images", 0)),
        "n_classes": int(b.metadata.get("n_classes", 0)),
        "data_fingerprint": b.config.get("data_fingerprint"),
        "stimulus": b.config.get("stimulus"),
        "sigmoid": cr.fit.sigmoid,
    }


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _library_versions() -> dict:
    versions = {"psyvis_ml": library_version(), "python": platform.python_version(),
                "numpy": np.__version__}
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        pass
    return versions


def build_metadata(results, *, data_file: str, n_rows: int, extra=None) -> dict:
    """The JSON sidecar: schema, the margin contract, and per-sweep reproducibility metadata."""
    results = _as_result_list(results)
    runs = [_run_metadata(r, cr) for r in results for cr in r.condition_results]
    axes = sorted({run["axis_direction"] for run in runs})
    meta = {
        "generated_by": "psyvis-ml",
        "artifact": "per_item",
        "schema_version": SCHEMA_VERSION,
        "data_file": data_file,
        "n_rows": int(n_rows),
        "library_versions": _library_versions(),
        "margin_definition": MARGIN_DEFINITION,
        "margin_boundary": 0.0,
        "margin_is_signed": True,
        "missing_label": MISSING_LABEL,
        "severity_rank_definition": (
            "0 = cleanest/easiest level, n_levels-1 = most degraded/hardest, derived from the "
            "suite's declared axis_direction (decreasing: higher stimulus_level is worse; "
            "increasing: higher stimulus_level is better)."
        ),
        "axis_direction": axes[0] if len(axes) == 1 else axes,
        "columns": [
            {"name": c.name, "dtype": c.dtype, "description": c.description}
            for c in PER_ITEM_COLUMNS
        ],
        "runs": runs,
    }
    if extra:
        meta["extra"] = dict(extra)
    return meta


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerItemExport:
    """Paths and content of a written per-item export."""

    data_path: Path
    metadata_path: Path
    n_rows: int
    metadata: dict = field(default_factory=dict)

    def artifacts(self) -> list[Path]:
        return [self.data_path, self.metadata_path]


def _csv_value(row, col: Column):
    v = row[col.name]
    if col.dtype == "bool":
        return int(bool(v))       # 0/1: unambiguous across every CSV reader
    if col.dtype == "float":
        return repr(float(v))     # shortest round-trip repr: read back bit-identical
    return v


def _write_csv(rows, path: Path) -> None:
    names = [c.name for c in PER_ITEM_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for row in rows:
            writer.writerow([_csv_value(row, c) for c in PER_ITEM_COLUMNS])


def _write_parquet(rows, path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "parquet export needs pandas + pyarrow (neither is a psyvis-ml dependency): "
            "pip install pandas pyarrow — or use the default format='csv', which needs "
            "nothing and carries the same schema."
        ) from e
    frame = _to_frame(pd, rows)
    frame.to_parquet(path, index=False)


def _to_frame(pd, rows):
    dtypes = {"str": object, "int": "int64", "float": "float64", "bool": "bool"}
    frame = pd.DataFrame(rows, columns=[c.name for c in PER_ITEM_COLUMNS])
    return frame.astype({c.name: dtypes[c.dtype] for c in PER_ITEM_COLUMNS})


def write_per_item(results, outdir, *, stem="per_item", fmt="csv", extra_metadata=None):
    """Persist the per-item table + its JSON sidecar for one or more :class:`MeasureResult`.

    This is the handoff artifact for trial-level (type-2 / meta-d′) analysis: it carries the
    signed logit margin and the correctness bit per (item × level × condition × model), which
    the aggregate ``results.json`` cannot reconstruct. It is written **alongside**, never
    instead of, the aggregate outputs.

    Parameters
    ----------
    results
        A :class:`~psyvis_ml.MeasureResult`, or a list of them (e.g. several models, or several
        suites) — all rows land in one table, distinguished by the ``model`` / ``suite`` /
        ``condition`` columns.
    outdir
        Destination directory (created if needed).
    stem
        Base filename; writes ``<stem>.csv`` (or ``.parquet``) and ``<stem>.meta.json``.
    fmt
        ``"csv"`` (default; stdlib-only, no extra dependency) or ``"parquet"`` (needs pandas +
        pyarrow). Both carry the identical schema.
    extra_metadata
        Optional dict merged into the sidecar under ``"extra"`` (e.g. the dataset directory).

    Returns
    -------
    PerItemExport
        With ``data_path``, ``metadata_path``, ``n_rows``, and the sidecar ``metadata``.
    """
    if fmt not in ("csv", "parquet"):
        raise ValueError(f"fmt must be 'csv' or 'parquet'; got {fmt!r}.")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = per_item_rows(results)
    data_path = outdir / f"{stem}.{'csv' if fmt == 'csv' else 'parquet'}"
    metadata_path = outdir / f"{stem}.meta.json"

    if fmt == "csv":
        _write_csv(rows, data_path)
    else:
        _write_parquet(rows, data_path)

    metadata = build_metadata(results, data_file=data_path.name, n_rows=len(rows),
                              extra=extra_metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return PerItemExport(data_path=data_path, metadata_path=metadata_path, n_rows=len(rows),
                         metadata=metadata)


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerItemTable:
    """A loaded per-item table: columns as NumPy arrays, plus the sidecar metadata.

    ``columns`` maps each column name to a 1-D array (``str`` -> object, ``int`` -> int64,
    ``float`` -> float64, ``bool`` -> bool). ``metadata`` is the sidecar dict (empty if no
    sidecar was found next to the data file).
    """

    columns: dict
    metadata: dict = field(default_factory=dict)
    path: Path | None = None

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    @property
    def names(self) -> list[str]:
        return list(self.columns)

    def rows(self) -> list[dict]:
        """The table as a list of plain row dicts (native Python scalars)."""
        names = self.names
        return [{n: self.columns[n][i].item() if hasattr(self.columns[n][i], "item")
                 else self.columns[n][i] for n in names} for i in range(len(self))]

    def to_pandas(self):
        """The table as a ``pandas.DataFrame`` (needs pandas; not a psyvis-ml dependency)."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("to_pandas() needs pandas: pip install pandas") from e
        return pd.DataFrame({n: self.columns[n] for n in self.names})


def _cast_column(values, col: Column) -> np.ndarray:
    if col.dtype == "int":
        return np.asarray([int(v) for v in values], dtype=np.int64)
    if col.dtype == "float":
        return np.asarray([float(v) for v in values], dtype=np.float64)
    if col.dtype == "bool":
        # Accept 0/1 (what we write) as well as true/false/True/False from other producers.
        return np.asarray([str(v).strip().lower() in ("1", "true") for v in values], dtype=bool)
    return np.asarray([str(v) for v in values], dtype=object)


def load_per_item(path) -> PerItemTable:
    """Read a per-item file (``.csv`` or ``.parquet``) back into a :class:`PerItemTable`.

    Picks up ``<stem>.meta.json`` next to the data file if present, so ``table.metadata``
    carries the run metadata (seed, config hash, library versions, the margin definition and
    its boundary at 0). Unknown extra columns are loaded as strings rather than rejected, and a
    missing sidecar is tolerated (``metadata == {}``) — the data file stands on its own.

    Nothing here is psyvis-specific: the file is plain CSV/Parquet against the schema in
    ``SCHEMA.md``, so a consumer (e.g. metaeval) can read it with ``pandas.read_csv`` and skip
    this function entirely. It exists so the round-trip is exercised and documented in one
    place.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"per-item file not found: {str(path)!r}")

    if path.suffix == ".parquet":
        columns = _load_parquet_columns(path)
    else:
        columns = _load_csv_columns(path)

    meta_path = path.with_name(path.stem + ".meta.json")
    metadata = (json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists()
                else {})
    return PerItemTable(columns=columns, metadata=metadata, path=path)


def _load_csv_columns(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        names = list(reader.fieldnames or [])
        raw = {n: [] for n in names}
        for record in reader:
            for n in names:
                raw[n].append(record[n])
    missing = [c.name for c in PER_ITEM_COLUMNS if c.name not in raw]
    if missing:
        raise ValueError(
            f"{str(path)!r} is missing required per-item columns {missing}; expected the "
            f"schema-{SCHEMA_VERSION} columns {[c.name for c in PER_ITEM_COLUMNS]}."
        )
    # Unknown columns survive as strings — an additive schema bump must not break old readers.
    return {n: _cast_column(vals, _COLUMN_BY_NAME.get(n, Column(n, "str", "")))
            for n, vals in raw.items()}


def _load_parquet_columns(path: Path) -> dict:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "reading parquet needs pandas + pyarrow: pip install pandas pyarrow"
        ) from e
    frame = pd.read_parquet(path)
    missing = [c.name for c in PER_ITEM_COLUMNS if c.name not in frame.columns]
    if missing:
        raise ValueError(f"{str(path)!r} is missing required per-item columns {missing}.")
    return {n: _cast_column(frame[n].tolist(), _COLUMN_BY_NAME.get(n, Column(n, "str", "")))
            for n in frame.columns}
