# Per-item sweep export — schema v1.0

This is the **contract** for the trial-level artifact psyvis-ml writes next to its aggregate
outputs. It is a plain data file: reading it needs **no psyvis-ml install**, only a CSV (or
Parquet) reader. The machine-readable form of this document is
[`PER_ITEM_COLUMNS`](src/psyvis_ml/per_item.py) and the JSON sidecar each export ships with.

## Why it exists

`results.json` and the report bundle's `metadata.json` are **aggregate**: per-level
`(n_correct, n_trials)` counts plus fitted thresholds/slopes/CIs. Aggregate counts cannot
reconstruct a **type-2** analysis (meta-d′, type-2 ROC, folded confidence distributions),
because every such analysis needs the *trial-level pairing* of a **confidence signal** with an
**outcome**. The sweep computes both (it sees the full logit vector per item per level) but
previously discarded them. This file persists them, so "run psyvis now, evaluate later" works.

## Files

A run writes two files (default stem `per_item`):

| file | contents |
|---|---|
| `per_item.csv` (or `.parquet`) | the table below, one row per (item × stimulus level × condition × model) |
| `per_item.meta.json` | run metadata: seed, config hash, library versions, levels, per-level counts, the column schema, and the margin definition |

Written by:

```python
result = pe.measure(model=..., suite=..., dataset=..., levels=...)
export = result.write_per_item("outdir")          # -> outdir/per_item.csv + per_item.meta.json
pe.write_per_item([res_a, res_b], "outdir")       # several models/suites -> one table
```

`build_report()` writes it into the report directory automatically, and
`examples/generate_gallery.py` writes it next to `results.json`. It is always **additive** —
the aggregate outputs are unchanged.

## Columns

| column | dtype | meaning |
|---|---|---|
| `model` | str | Model identifier (`MeasureResult.model_name`). |
| `suite` | str | Suite class name (`ContrastThreshold`, `DegradationSuite`, `DistractorRobustness`). |
| `condition` | str | Condition/slice label within the suite (e.g. `distractors`, `baseline`). One sweep and one psychometric fit per (`model`, `suite`, `condition`). |
| `config_hash` | str | SHA-256 config hash of the sweep the row came from; joins the row to its entry in the sidecar's `runs` list. |
| `item_id` | str | Stable identifier for the source image (a file path for `imagenet_subset`; otherwise the decimal image index). Every item appears once per level. |
| `item_index` | int | Zero-based position of the image in the dataset. |
| `level_index` | int | Zero-based index of the level within the swept list (same order as the sidecar's `levels`). |
| `stimulus_level` | float | The level in the suite's **native units** (RMS contrast, noise σ, distractor size in px …). Its direction is not self-evident — see `severity_rank`. |
| `severity_rank` | int | **Direction-aware severity axis.** `0` = cleanest/easiest level, `n_levels - 1` = most degraded/hardest. Derived from the suite's *declared* axis direction, never inferred from the data. Use this when you want a monotone severity axis without knowing the suite's units. |
| `margin` | float | **Primary confidence signal.** Signed logit margin; boundary at 0. See below. |
| `correct` | bool | Was the true label among the model's top-`k` predictions? (`top_k` is in the sidecar; it is `1` unless stated otherwise.) Encoded `0`/`1` in CSV. |
| `true_label` | int | True class index, or `-1` if unavailable. |
| `predicted_label` | int | The model's top-1 class (argmax of the logit vector), or `-1` if unavailable. Present so a future binary/2AFC meta-d′ can recover the **response**, not just its correctness. |
| `target_rank` | int | Rank of the true class in the logit vector; `1` = the true class is top-1. |
| `target_logit` | float | Raw logit of the true class (arbitrary per-model scale). |
| `max_softmax` | float | Max softmax probability. **Reference only, calibration-sensitive** — not a calibrated probability, and not a substitute for `margin`. |

Consumers must **tolerate unknown extra columns**: adding a column is a minor schema bump, not
a break. `load_per_item` loads unrecognized columns as strings rather than rejecting the file.

## The margin (read this before scoring it)

```
margin = logit[true_label] − max(logit[j] for j ≠ true_label)
```

* **Signed**, and its **decision boundary is 0**: `margin > 0` iff the true label is the top-1
  prediction. So for `top_k = 1`, `correct == (margin > 0)` exactly.
* It is a **raw evidence difference**, not a softmax probability. Softmax exponentiates the
  logits, swings with temperature, and is poorly calibrated; `max_softmax` is recorded for
  reference only.
* It is persisted **raw and signed. psyvis-ml never applies `abs()` on export.** Folding the
  margin into an unsigned confidence — `abs(margin)`, the natural choice for type-2/meta-d′,
  where confidence is magnitude and the outcome carries the sign — is a **scoring decision that
  belongs to the consumer** (metaeval's bridge), not to the emitter. Exporting the signed value
  keeps both the confidence magnitude and the direction of evidence recoverable; exporting
  `abs()` would destroy the latter irreversibly.
* **Logit scales are not comparable across models.** Never compare raw margins between models.
  Within a model, or as a change from that model's own clean baseline (Δ-margin), is fine.

## Missing values

* Integer label columns (`true_label`, `predicted_label`, `target_rank`) use **`-1`** as the
  "unavailable" sentinel (`psyvis_ml.per_item.MISSING_LABEL`). Filter with `!= -1`; never treat
  `-1` as a class index.
* Float columns use `NaN`.
* In practice, sweeps produced by `measure()` / `run_sweep()` always populate every column;
  the sentinels exist for bundles assembled by hand.

## Sidecar (`per_item.meta.json`)

| key | meaning |
|---|---|
| `schema_version` | This schema's version (`"1.0"`). |
| `data_file`, `n_rows` | The data file this sidecar describes, and its row count. |
| `library_versions` | `psyvis_ml`, `python`, `numpy`, `scipy` versions at export time. |
| `margin_definition`, `margin_boundary`, `margin_is_signed` | The margin contract above, restated in the file itself (boundary `0.0`, signed `true`). |
| `missing_label` | The integer sentinel (`-1`). |
| `severity_rank_definition`, `axis_direction` | How `severity_rank` was derived; `decreasing` (higher level = worse: degradation, distractor size) or `increasing` (higher level = better: contrast). |
| `columns` | The column list above (name, dtype, description). |
| `runs` | One entry per sweep: `model`, `suite`, `condition`, `config_hash`, `seed`, `library_version`, `top_k`, `chance_level`, `axis_direction`, `levels`, `n_correct`, `n_trials`, `n_images`, `n_classes`, `data_fingerprint`, `stimulus`, `sigmoid`. Join to a row via `config_hash`. |
| `extra` | Optional caller-supplied context (the gallery records its dataset dir, for example). |

The `runs` entries reuse the reproducibility metadata already captured on each `RunBundle`, so
the per-item file audits identically to the aggregate one: same seed, same config hash.

## Reading it

Without psyvis-ml — it is just a data file:

```python
import pandas as pd, json

df   = pd.read_csv("per_item.csv")
meta = json.load(open("per_item.meta.json"))

df["correct"] = df["correct"].astype(bool)
# Unsigned confidence for a type-2 analysis is the CONSUMER's choice, made here — not upstream:
df["confidence"] = df["margin"].abs()
```

With psyvis-ml, for the same result plus the sidecar in one object:

```python
import psyvis_ml as pe

table = pe.load_per_item("per_item.csv")
table["margin"]                 # np.float64 array
table["correct"]                # np.bool_ array
table.metadata["runs"][0]["seed"]
table.to_pandas()               # if pandas is installed
```

## Stability

The columns and their semantics above are the stable interface. Additive changes (new columns,
new sidecar keys) bump the minor version; renaming or re-meaning a column bumps the major
version and is announced in `schema_version`.
