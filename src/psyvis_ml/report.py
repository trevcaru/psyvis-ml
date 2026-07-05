"""Methods-ready report bundle — the PRD §8 "audit-bundle" differentiator.

``build_report`` writes a **self-contained** directory an external user can drop into a
methods section: a threshold/slope/CI table, the fitted-curve figure(s), the models-vs-human
comparison figure, and — crucially — the **reproducibility metadata** already captured in each
:class:`~psyvis_ml.sweep.RunBundle` (resolved seed, config hash, library version). Everything
is derived from data already computed; nothing about the human reference is synthesized.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .comparison import compare_results, comparison_summary
from .reference import human_reference_for

__all__ = ["ReportBundle", "build_report"]


@dataclass
class ReportBundle:
    """Paths and content of a written report bundle."""

    directory: Path
    markdown_path: Path
    metadata_path: Path
    figure_paths: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    summary_rows: list = field(default_factory=list)
    markdown: str = ""

    def artifacts(self) -> list[Path]:
        """All files written, for a quick existence check."""
        return [self.markdown_path, self.metadata_path, *self.figure_paths.values()]


def _slug(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
    return s or "model"


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}g}"


def _model_metadata(result, i):
    """Reproducibility metadata for one model, pulled from its run bundles."""
    label = result.model_name or f"model {i + 1}"
    conditions = []
    for cr in result.condition_results:
        b = cr.bundle
        conditions.append({
            "condition": cr.label,
            "seed": b.seed,
            "config_hash": b.config_hash,
            "library_version": b.library_version,
            "top_k": b.top_k,
            "levels": list(b.levels),
            "n_correct": list(b.n_correct),
            "n_trials": list(b.n_trials),
            "sigmoid": cr.fit.sigmoid,
        })
    return {
        "model": label,
        "chance_level": result.chance_level,
        "direction": "decreasing" if result.decreasing else "increasing",
        "conditions": conditions,
    }


def _table_md(rows):
    head = ("| model | threshold | 95% CI | slope | direction |\n"
            "|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        ci = f"[{_fmt(r['threshold_ci_low'])}, {_fmt(r['threshold_ci_high'])}]"
        lines.append(
            f"| {r['model']} | {_fmt(r['threshold'])} | {ci} | "
            f"{_fmt(r['slope'])} | {r['direction']} |"
        )
    return head + "\n".join(lines) + "\n"


def _repro_md(model_meta):
    lines = []
    for m in model_meta:
        lines.append(f"- **{m['model']}** (chance {_fmt(m['chance_level'])}, "
                     f"{m['direction']} axis):")
        for c in m["conditions"]:
            lines.append(
                f"    - `{c['condition']}` — seed `{c['seed']}`, "
                f"config hash `{c['config_hash'][:16]}…`, "
                f"psyvis-ml `{c['library_version']}`, top-k {c['top_k']}, "
                f"{len(c['levels'])} levels, {int(np.sum(c['n_trials']))} trials"
            )
    return "\n".join(lines) + "\n"


def build_report(result, *, others=None, outdir=None, condition=None, target=0.75,
                 human="auto", n_boot=400, ci=0.95, seed=0, title=None,
                 dpi=140):
    """Write a self-contained methods bundle for one or more models.

    Parameters
    ----------
    result
        The primary :class:`~psyvis_ml.MeasureResult`.
    others
        Optional list of other models' results to include in the comparison figure/table.
    outdir
        Destination directory (created if needed). Defaults to a fresh temp directory.
    condition
        Condition label to feature in the comparison/table (required if multi-condition).
    target
        Threshold criterion (proportion correct).
    human
        Passed to the comparison figure; overlays a cited human reference where one exists.

    Returns
    -------
    ReportBundle
    """
    import matplotlib.pyplot as plt

    results = [result, *(others or [])]
    directory = Path(outdir) if outdir is not None else Path(tempfile.mkdtemp(
        prefix="psyvis_report_"))
    directory.mkdir(parents=True, exist_ok=True)
    figs_dir = directory / "figures"
    figs_dir.mkdir(exist_ok=True)

    suite_name = result.suite.__class__.__name__
    condition_label, rows = comparison_summary(
        results, condition=condition, target=target, n_boot=n_boot, ci=ci, seed=seed)

    figure_paths: dict = {}

    # Per-model fitted-curve figures.
    for i, r in enumerate(results):
        fig = r.plot(n_boot=min(n_boot, 200), seed=seed)
        p = figs_dir / f"fit_{_slug(r.model_name or f'model_{i + 1}')}_{i}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figure_paths[f"fit:{r.model_name or f'model {i + 1}'}"] = p

    # The comparison / human-overlay figure (the killer plot).
    cfig = compare_results(results, condition=condition_label, target=target, human=human,
                           n_boot=min(n_boot, 200), ci=ci, seed=seed, title=title)
    comparison_path = figs_dir / "comparison_models_vs_human.png"
    cfig.savefig(comparison_path, dpi=dpi, bbox_inches="tight")
    plt.close(cfig)
    figure_paths["comparison"] = comparison_path

    # Reproducibility metadata + human citation.
    model_meta = [_model_metadata(r, i) for i, r in enumerate(results)]
    ref = human_reference_for(suite_name)
    human_meta = None
    if ref is not None:
        human_meta = {
            "name": ref.name, "citation": ref.citation, "version": ref.version,
            "approximate": ref.approximate, "metric": ref.metric,
            "quantity": ref.quantity, "source_note": ref.source_note,
        }

    metadata = {
        "generated_by": "psyvis-ml",
        "suite": suite_name,
        "featured_condition": condition_label,
        "target": target,
        "n_models": len(results),
        "models": model_meta,
        "summary_rows": rows,
        "human_reference": human_meta,
        "figures": {k: str(v.relative_to(directory)) for k, v in figure_paths.items()},
    }
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # Markdown.
    obs = ("Correctness is scored as **argmax top-1** on the (manipulated) stimulus — a "
           "documented, swappable observer-model choice (PRD §14), not a property of the "
           "fitter, which consumes only aggregated `(level, n_correct, n_trials)` counts.")
    direction = "decreasing" if result.decreasing else "increasing"
    md = [
        f"# psyvis-ml report — {suite_name}",
        "",
        f"Featured condition: **{condition_label}**  ·  criterion: **{target:g} proportion "
        f"correct**  ·  axis direction: **{direction}**.",
        "",
        "## Methods",
        "",
        f"Each model is evaluated as a psychophysical observer by the **method of constant "
        f"stimuli**: every image is presented at each stimulus level, correctness is scored, "
        f"and a psychometric function is fit to the per-level counts by maximum likelihood "
        f"(sigmoid family: `{getattr(result.suite, 'sigmoid', 'weibull')}`; the observer "
        f"chance level is used as the fitted lower asymptote). {obs}",
        "",
        "## Thresholds and slopes",
        "",
        _table_md(rows),
        "## Reproducibility",
        "",
        "Every sweep is captured as an auditable run bundle. To reproduce a row, re-run the "
        "same suite/model on the same dataset with the recorded seed; the config hash below "
        "fingerprints the levels, seed, data, and model/stimulus identity.",
        "",
        _repro_md(model_meta),
        "## Figures",
        "",
        f"![models vs. human]({figure_paths['comparison'].relative_to(directory).as_posix()})",
        "",
    ]
    for key, p in figure_paths.items():
        if key == "comparison":
            continue
        md.append(f"- {key}: `{p.relative_to(directory).as_posix()}`")
    md.append("")
    md.append("## Human reference")
    md.append("")
    if human_meta is not None:
        approx = " (approximate/digitized)" if human_meta["approximate"] else ""
        md.append(f"Human overlay{approx}: {human_meta['citation']}")
        md.append("")
        md.append(f"> {human_meta['source_note']}")
    else:
        md.append(f"No published human reference is shipped for **{suite_name}**; the "
                  f"comparison shows models only (PRD §14 — we do not fabricate human data).")
    md.append("")
    markdown = "\n".join(md)
    markdown_path = directory / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    return ReportBundle(
        directory=directory,
        markdown_path=markdown_path,
        metadata_path=metadata_path,
        figure_paths=figure_paths,
        metadata=metadata,
        summary_rows=rows,
        markdown=markdown,
    )
