"""Standing figure gallery — run all three suites + the confidence layer on real Imagenette.

Produces a committed ``outputs/gallery/`` folder: per suite (contrast, degradation, distractor-
robustness) and per model (resnet50 + a ViT), the accuracy psychometric curve, the baseline-
relative Δ-margin confidence panel, and the multi-model comparison figure; for contrast, the
human-vs-models overlay with the paradigm-difference caption. Plus ``results.md`` / ``results.json``
with thresholds, CIs, slopes, confidence thresholds, and margin slopes — and ``per_item.csv`` /
``per_item.meta.json``, the trial-level export (one row per item × level × condition × model,
signed logit margin + correctness) that the aggregate numbers cannot reconstruct (see SCHEMA.md).

Run locally against YOUR Imagenette val (never bundled/downloaded here):

    pip install -e ".[demo]"
    export PSYVIS_IMAGENET_DIR=/path/to/imagenette2/val   # ImageFolder (WNID or integer folders)
    python examples/generate_gallery.py

Optional env: PSYVIS_GALLERY_MODELS (comma list), PSYVIS_MAX_PER_CLASS (default 20),
PSYVIS_GALLERY_NBOOT (default 300). Skips cleanly (no fabrication) if deps/data are absent.
"""

import os
import sys


def _slug(text):
    import re
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def main():
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import psyvis_ml as pe

    data_dir = os.environ["PSYVIS_IMAGENET_DIR"]
    model_names = [m.strip() for m in
                   os.environ.get("PSYVIS_GALLERY_MODELS",
                                  "resnet50,vit_small_patch16_224").split(",")]
    max_per_class = int(os.environ.get("PSYVIS_MAX_PER_CLASS", "20"))
    n_boot = int(os.environ.get("PSYVIS_GALLERY_NBOOT", "300"))

    outdir = os.path.join("outputs", "gallery")
    os.makedirs(outdir, exist_ok=True)

    # Full-frame images for contrast/degradation; a big (176-px, ~79% of the 224 frame) centred
    # target for the distractor suite so it clears the recognition ceiling; a 96-px image as the
    # heterogeneous distractor content (resized to each swept size by the compositor).
    ds_full = pe.datasets.imagenet_subset(data_dir, max_per_class=max_per_class, image_size=224)
    ds_target = pe.datasets.imagenet_subset(data_dir, max_per_class=max_per_class, image_size=176)
    ds_small = pe.datasets.imagenet_subset(data_dir, max_per_class=max_per_class, image_size=96)
    print(f"loaded {len(ds_full.images)} full-frame + {len(ds_target.images)} target images "
          f"from {data_dir}", flush=True)

    clfs = {m: pe.models.timm_classifier(m, pretrained=True) for m in model_names}

    specs = [
        {"key": "contrast", "title": "Contrast sensitivity",
         "plot_title": "Contrast sensitivity — models vs. human threshold",
         "suite": lambda: pe.suites.ContrastThreshold(contrast_metric="rms",
                                                      clip_range=(0.0, 1.0)),
         "levels": pe.linspace_levels(0.01, 0.5, 7, spacing="log"),
         "dataset": ds_full, "condition": None, "human": "auto",
         "comparison_name": "contrast_human_vs_models.png"},
        {"key": "degradation", "title": "Degradation (gaussian noise)",
         "plot_title": "Degradation (noise σ) — models",
         "suite": lambda: pe.suites.DegradationSuite("gaussian_noise"),
         "levels": pe.linspace_levels(0.0, 0.8, 7),
         "dataset": ds_full, "condition": None, "human": "auto",
         "comparison_name": "degradation_comparison.png"},
        {"key": "distractor", "title": "Distractor robustness (distractor size)",
         "plot_title": "Distractor robustness — models",
         # Big centred target + 4 heterogeneous distractors grown from the margins. Sweeping
         # distractor SIZE (small -> large) gives a decreasing curve with a real threshold; the
         # size band brackets the ceiling->floor transition found by calibrate_distractor_size.
         "suite": lambda: pe.suites.DistractorRobustness(
             distractor_patch=ds_small.images[0], canvas_shape=(224, 224), n_distractors=4,
             include_baseline=True),
         "levels": pe.linspace_levels(20, 105, 9),
         "dataset": ds_target, "condition": "distractors", "human": "auto",
         "comparison_name": "distractor_comparison.png"},
    ]

    all_rows = []
    all_results = []   # every MeasureResult, for the trial-level per-item export
    analyses = {}
    for spec in specs:
        key, cond = spec["key"], spec["condition"]
        print(f"\n=== {spec['title']} ===", flush=True)
        results = []
        for ci_idx, name in enumerate(model_names):
            print(f"  measuring {name} ...", flush=True)
            res = pe.measure(model=clfs[name], suite=spec["suite"](), dataset=spec["dataset"],
                             levels=spec["levels"], seed=0, model_name=name)
            results.append(res)
            all_results.append(res)
            # Per-model accuracy curve; color_index keeps a model's colour across all figures.
            fig = res.plot(show_confidence=False, n_boot=n_boot, seed=0, color_index=ci_idx)
            fig.savefig(os.path.join(outdir, f"{key}_accuracy_{_slug(name)}.png"),
                        dpi=200, bbox_inches="tight")
            plt.close(fig)

        # Confidence panel (baseline-relative Δ-margin), both models — production-styled.
        dfig = pe.plot_confidence_delta(
            results, condition=cond, n_boot=n_boot,
            title=f"{spec['title']}: Δ logit margin (baseline-relative)")
        dfig.savefig(os.path.join(outdir, f"{key}_confidence_delta.png"),
                     dpi=200, bbox_inches="tight")
        plt.close(dfig)

        # Multi-model comparison (accuracy + Δ-margin panels; human overlay where it exists).
        cfig = pe.compare_results(results, condition=cond, human=spec["human"], n_boot=n_boot,
                                  seed=0, title=spec["plot_title"])
        cfig.savefig(os.path.join(outdir, spec["comparison_name"]),
                     dpi=200, bbox_inches="tight")
        plt.close(cfig)

        # Numbers table (threshold/CI/slope + confidence threshold/margin slope).
        _, rows = pe.comparison_summary(results, condition=cond, n_boot=n_boot, seed=0)
        for r in rows:
            r["suite"] = key
            all_rows.append(r)
            print(f"  {key}/{r['model']}: threshold={r['threshold']:.4g} "
                  f"[{r['threshold_ci_low']:.3g},{r['threshold_ci_high']:.3g}] "
                  f"conf.threshold(margin=0)={r['confidence_threshold']:.4g}", flush=True)

        # Paired over-images bootstrap analysis (the analysis.md content).
        if len(results) == 2:
            an = pe.analyze_suite(results, condition=cond, n_boot=2000, seed=0)
            analyses[key] = an
            d = an.threshold_diff_ci
            print(f"  {key}: threshold diff {an.threshold_diff:+.3g} [{d[0]:.3g}, {d[1]:.3g}] "
                  f"-> {'RELIABLE' if an.threshold_diff_excludes_zero else 'not reliable'}",
                  flush=True)

    # results.json (raw numbers, incl. the analysis) + results.md + analysis.md
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"data_dir": data_dir, "models": model_names,
                   "max_per_class": max_per_class, "rows": all_rows,
                   "analysis": {k: a.to_dict() for k, a in analyses.items()}},
                  fh, indent=2, default=str)

    # per_item.csv + per_item.meta.json — the TRIAL-LEVEL artifact, alongside (not instead of)
    # the aggregate results.json above. One row per item x level x condition x model, carrying
    # the signed logit margin + correctness: the pairing a type-2 / meta-d' analysis needs and
    # the aggregate counts cannot reconstruct. Schema: SCHEMA.md.
    export = pe.write_per_item(
        all_results, outdir,
        extra_metadata={"data_dir": data_dir, "max_per_class": max_per_class,
                        "gallery_suites": [s["key"] for s in specs]},
    )
    print(f"\nper-item export: {export.data_path} ({export.n_rows} rows) "
          f"+ {export.metadata_path.name}", flush=True)

    def _f(x):
        return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.4g}"

    def _thr(x, lo, hi):
        # Flag thresholds the fit extrapolated outside the swept range (ceiling-limited).
        if x is None or (isinstance(x, float) and x != x):
            return "n/a"
        if x < lo or x > hi:
            return f"{x:.3g} (extrapolated)"
        return f"{x:.4g}"

    lvl_range = {"contrast": (0.01, 0.5), "degradation": (0.0, 0.8), "distractor": (20.0, 105.0)}
    md = ["# psyvis-ml results gallery", "",
          f"Real Imagenette val · {len(ds_full.images)} images · models: "
          f"{', '.join(model_names)}.", "",
          "| suite | model | threshold | 95% CI | slope | conf. threshold (margin=0) | "
          "margin slope |", "|---|---|---|---|---|---|---|"]
    for r in all_rows:
        lo, hi = lvl_range.get(r["suite"], (float("-inf"), float("inf")))
        ci = f"[{_f(r['threshold_ci_low'])}, {_f(r['threshold_ci_high'])}]"
        md.append(f"| {r['suite']} | {r['model']} | {_thr(r['threshold'], lo, hi)} | {ci} | "
                  f"{_f(r['slope'])} | {_f(r['confidence_threshold'])} | "
                  f"{_f(r['margin_slope'])} |")
    md += ["", "Figures in this folder: `<suite>_accuracy_<model>.png` (accuracy psychometric "
           "curve), `<suite>_confidence_delta.png` (baseline-relative Δ-margin), "
           "`contrast_human_vs_models.png` / `<suite>_comparison.png` (multi-model + Δ-margin "
           "panels). The human overlay is grating-detection sensitivity vs. argmax "
           "classification — different paradigms on a shared axis (see the figure caption).", "",
           "**Distractor suite (size sweep).** A big centred target (176 px, ~79% of the frame) "
           "clears the recognition ceiling; four heterogeneous distractors grow inward from the "
           "margins and the sweep is over distractor **size**. Performance falls as size grows "
           "(a decreasing suite), giving a **real, non-extrapolated threshold** — the distractor "
           "size at which accuracy hits criterion — over a band gated by "
           "`calibrate_distractor_size`. A larger threshold means a more distractor-robust "
           "model.", ""]
    with open(os.path.join(outdir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    _write_analysis_md(os.path.join(outdir, "analysis.md"), analyses, model_names,
                       len(ds_full.images))

    print(f"\nsaved gallery to {outdir}/ "
          f"({len([f for f in os.listdir(outdir) if f.endswith('.png')])} PNGs + "
          f"results.md + results.json + analysis.md + per_item.csv)", flush=True)


def _write_analysis_md(path, analyses, model_names, n_images):
    """Human-readable per-suite analysis: over-images bootstrap CIs + paired difference tests."""
    def f(x, nd=4):
        return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}g}"

    ma = model_names[0]
    mb = model_names[1] if len(model_names) > 1 else ""
    md = [
        "# psyvis-ml gallery — analysis", "",
        f"Real Imagenette val · {n_images} images · models: {', '.join(model_names)}.", "",
        "## How to read this (deterministic models)", "",
        "These models are **deterministic**: the same image at the same stimulus level always "
        "gives the same logits, so there is **no trial-level sampling noise** and a frequentist "
        "t-test / p-value on \"trials\" would be fabricated. All uncertainty here comes from "
        "**bootstrapping over the image set** — the only thing that varies is *which images* you "
        "measured. A difference between two models is called **reliable** when the bootstrap CI "
        "of the *paired* difference (both models recomputed on the **same** resampled images "
        "each iteration) **excludes zero**. We report no p-values.", "",
    ]
    for key, an in analyses.items():
        m = an.models
        td_lo, td_hi = an.threshold_diff_ci
        dm_lo, dm_hi = an.delta_margin_diff_ci
        md += [f"## {key}", "",
               "| metric | " + " | ".join(m) + " |", "|---|" + "---|" * len(m)]
        md.append("| threshold | " + " | ".join(f(an.threshold[k]) for k in m) + " |")
        md.append("| threshold 95% CI | "
                  + " | ".join(f"[{f(an.threshold_ci[k][0])}, {f(an.threshold_ci[k][1])}]"
                               for k in m) + " |")
        md.append("| slope | " + " | ".join(f(an.slope[k]) for k in m) + " |")
        md.append("| slope 95% CI | "
                  + " | ".join(f"[{f(an.slope_ci[k][0])}, {f(an.slope_ci[k][1])}]"
                               for k in m) + " |")
        md.append("| fit converged | " + " | ".join(str(an.gof[k]["converged"]) for k in m)
                  + " |")
        md.append("| fit R² | " + " | ".join(f(an.gof[k]["r2"], 3) for k in m) + " |")
        md.append("| at-bound params | "
                  + " | ".join(", ".join(an.gof[k]["at_bound"]) or "—" for k in m) + " |")
        md.append("| margin slope | " + " | ".join(f(an.margin_slope[k], 3) for k in m) + " |")
        md.append("| Δ-margin (clean→worst) | "
                  + " | ".join(f(an.delta_margin_endpoint[k], 3) for k in m) + " |")
        thr_verdict = ("**reliable** (CI excludes 0)" if an.threshold_diff_excludes_zero
                       else "not reliably different (CI includes 0)")
        dm_verdict = ("the Δ-margin curves **diverge** (CI excludes 0)"
                      if an.delta_margin_diff_excludes_zero
                      else "no reliable Δ-margin divergence (CI includes 0)")
        md += ["",
               f"**Paired threshold difference** ({m[0]} − {m[1]}): "
               f"**{f(an.threshold_diff)}**, 95% CI [{f(td_lo)}, {f(td_hi)}] → {thr_verdict}.",
               "",
               f"**Confidence divergence** — Δ-margin (clean→worst) difference "
               f"({m[0]} − {m[1]}): {f(an.delta_margin_diff, 3)}, 95% CI "
               f"[{f(dm_lo, 3)}, {f(dm_hi, 3)}] → {dm_verdict}.",
               ""]

    # Summary table.
    md += ["## Summary", "",
           f"| suite | {ma} thr | {mb} thr | diff [95% CI] | reliable? | R² ({ma}/{mb}) |",
           "|---|---|---|---|---|---|"]
    for key, an in analyses.items():
        m = an.models
        td_lo, td_hi = an.threshold_diff_ci
        md.append(f"| {key} | {f(an.threshold[m[0]])} | {f(an.threshold[m[1]])} | "
                  f"{f(an.threshold_diff)} [{f(td_lo)}, {f(td_hi)}] | "
                  f"{'yes' if an.threshold_diff_excludes_zero else 'no'} | "
                  f"{f(an.gof[m[0]]['r2'], 3)} / {f(an.gof[m[1]]['r2'], 3)} |")
    md.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


if __name__ == "__main__":
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: needs the demo extra (torch + timm): pip install -e '.[demo]'")
        sys.exit(0)
    if not (os.environ.get("PSYVIS_IMAGENET_DIR")
            and os.path.isdir(os.environ["PSYVIS_IMAGENET_DIR"])):
        print("SKIP: set PSYVIS_IMAGENET_DIR to your own Imagenette/ImageNet val ImageFolder "
              "(this package does not bundle or download it).")
        sys.exit(0)
    main()
