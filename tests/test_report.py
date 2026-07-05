"""Report bundle: artifacts written, reproducibility metadata embedded, citations honest."""

import json

import numpy as np

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset, synthetic_dataset
from psyvis_ml.suites import DegradationSuite

import synthetic as syn

NUM_CLASSES = 8


def _contrast_result(alpha, *, name, n=180, seed=1):
    ds = synthetic_dataset(n=n, num_classes=NUM_CLASSES)
    model = syn.make_contrast_observer(NUM_CLASSES, alpha=alpha)
    levels = pe.linspace_levels(0.03, 0.6, 9, spacing="log")
    suite = pe.suites.ContrastThreshold(contrast_metric="rms")
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed,
                      model_name=name)


def _degradation_result(*, name, n=180, seed=1):
    images, labels = syn.make_patch_data(n, size=24, num_classes=NUM_CLASSES)
    ds = Dataset(images=images, labels=labels, num_classes=NUM_CLASSES)
    model = syn.make_degradation_observer(NUM_CLASSES, fill=0.0)
    suite = DegradationSuite("occlusion", fill=0.0, block=False)
    levels = pe.linspace_levels(0.05, 0.6, 9)
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed,
                      model_name=name)


def test_report_writes_all_expected_artifacts(tmp_path):
    a = _contrast_result(0.15, name="A")
    b = _contrast_result(0.28, name="B")
    rb = a.report(others=[b], outdir=tmp_path / "bundle", n_boot=40, seed=0)

    assert rb.markdown_path.exists() and rb.metadata_path.exists()
    # A fitted-curve figure per model + the comparison figure.
    fit_figs = [p for k, p in rb.figure_paths.items() if k.startswith("fit:")]
    assert len(fit_figs) == 2 and all(p.exists() for p in fit_figs)
    assert rb.figure_paths["comparison"].exists()
    assert all(p.exists() for p in rb.artifacts())


def test_report_embeds_reproducibility_metadata(tmp_path):
    a = _contrast_result(0.15, name="A")
    rb = a.report(outdir=tmp_path / "bundle", n_boot=40, seed=0)

    meta = json.loads(rb.metadata_path.read_text(encoding="utf-8"))
    model0 = meta["models"][0]
    cond0 = model0["conditions"][0]
    # Seed, config hash, and library version are all carried through from the run bundle.
    bundle = a.condition_results[0].bundle
    assert cond0["seed"] == bundle.seed
    assert cond0["config_hash"] == bundle.config_hash
    assert cond0["library_version"] == bundle.library_version
    # And they surface in the human-readable methods text.
    assert bundle.config_hash[:16] in rb.markdown
    assert "seed" in rb.markdown.lower()


def test_report_table_has_threshold_slope_ci(tmp_path):
    a = _contrast_result(0.15, name="A")
    b = _contrast_result(0.28, name="B")
    rb = a.report(others=[b], outdir=tmp_path / "bundle", n_boot=40, seed=0)
    assert len(rb.summary_rows) == 2
    for row in rb.summary_rows:
        assert np.isfinite(row["threshold"])
        assert {"threshold_ci_low", "threshold_ci_high", "slope", "direction"} <= set(row)
    # Markdown renders a table header.
    assert "| model | threshold | 95% CI | slope | direction |" in rb.markdown


def test_report_includes_human_citation_for_contrast(tmp_path):
    a = _contrast_result(0.15, name="A")
    rb = a.report(outdir=tmp_path / "bundle", n_boot=40, seed=0)
    assert rb.metadata["human_reference"] is not None
    assert "Campbell" in rb.markdown           # cited
    assert rb.metadata["human_reference"]["approximate"] is True
    # The report must name the paradigm difference (detection vs. classification), not just
    # that the human values are approximate (PRD §14 Frame-B honesty).
    assert "Paradigm difference" in rb.markdown
    low = rb.markdown.lower()
    assert "detection" in low and "classification" in low
    assert rb.metadata["human_reference"]["paradigm_caveat"]


def test_report_degrades_gracefully_without_human(tmp_path):
    a = _degradation_result(name="A")
    rb = a.report(outdir=tmp_path / "bundle", n_boot=40, seed=0)
    # No fabricated human curve; the report says so and still produces all artifacts.
    assert rb.metadata["human_reference"] is None
    assert "No published human reference" in rb.markdown
    assert all(p.exists() for p in rb.artifacts())
