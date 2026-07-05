"""Standing figure gallery — run all three suites + the confidence layer on real Imagenette.

Produces a committed ``outputs/gallery/`` folder: per suite (contrast, degradation, distractor-
robustness) and per model (resnet50 + a ViT), the accuracy psychometric curve, the baseline-
relative Δ-margin confidence panel, and the multi-model comparison figure; for contrast, the
human-vs-models overlay with the paradigm-difference caption. Plus ``results.md`` / ``results.json``
with thresholds, CIs, slopes, confidence thresholds, and margin slopes.

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


def _save_delta_margin(pe, plt, results, condition, path, title, n_boot):
    """Standalone baseline-relative Δ-margin overlay (cross-model comparable, delta-only)."""
    _, curves = pe.confidence_comparison(results, condition=condition, kind="delta", n_boot=n_boot)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c in curves:
        line, = ax.plot(c["levels"], c["delta_margin"], marker="o", lw=2, ms=5, label=c["model"])
        ax.fill_between(c["levels"], c["delta_ci_low"], c["delta_ci_high"],
                        color=line.get_color(), alpha=0.18)
    ax.axhline(0.0, color="0.6", ls="--", lw=1)  # clean baseline anchor
    ax.set_xlabel("stimulus level")
    ax.set_ylabel("Δ logit margin\n(from clean baseline)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


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
         "suite": lambda: pe.suites.ContrastThreshold(contrast_metric="rms",
                                                      clip_range=(0.0, 1.0)),
         "levels": pe.linspace_levels(0.01, 0.5, 7, spacing="log"),
         "dataset": ds_full, "condition": None, "human": "auto",
         "comparison_name": "contrast_human_vs_models.png"},
        {"key": "degradation", "title": "Degradation (gaussian noise)",
         "suite": lambda: pe.suites.DegradationSuite("gaussian_noise"),
         "levels": pe.linspace_levels(0.0, 0.8, 7),
         "dataset": ds_full, "condition": None, "human": "auto",
         "comparison_name": "degradation_comparison.png"},
        {"key": "distractor", "title": "Distractor robustness (distractor size)",
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
    for spec in specs:
        key, cond = spec["key"], spec["condition"]
        print(f"\n=== {spec['title']} ===", flush=True)
        results = []
        for name in model_names:
            print(f"  measuring {name} ...", flush=True)
            res = pe.measure(model=clfs[name], suite=spec["suite"](), dataset=spec["dataset"],
                             levels=spec["levels"], seed=0, model_name=name)
            results.append(res)
            # Per-model accuracy psychometric curve (single-model, accuracy only).
            fig = res.plot(show_confidence=False, n_boot=n_boot, seed=0)
            fig.savefig(os.path.join(outdir, f"{key}_accuracy_{_slug(name)}.png"),
                        dpi=140, bbox_inches="tight")
            plt.close(fig)

        # Confidence panel (baseline-relative Δ-margin), both models.
        _save_delta_margin(pe, plt, results, cond,
                           os.path.join(outdir, f"{key}_confidence_delta.png"),
                           f"{spec['title']}: Δ logit margin (baseline-relative)", n_boot)

        # Multi-model comparison (accuracy + Δ-margin panels; human overlay where it exists).
        cfig = pe.compare_results(results, condition=cond, human=spec["human"], n_boot=n_boot,
                                  seed=0)
        cfig.savefig(os.path.join(outdir, spec["comparison_name"]), dpi=140, bbox_inches="tight")
        plt.close(cfig)

        # Numbers table (threshold/CI/slope + confidence threshold/margin slope).
        _, rows = pe.comparison_summary(results, condition=cond, n_boot=n_boot, seed=0)
        for r in rows:
            r["suite"] = key
            all_rows.append(r)
            print(f"  {key}/{r['model']}: threshold={r['threshold']:.4g} "
                  f"[{r['threshold_ci_low']:.3g},{r['threshold_ci_high']:.3g}] "
                  f"conf.threshold(margin=0)={r['confidence_threshold']:.4g}", flush=True)

    # results.json + results.md
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"data_dir": data_dir, "models": model_names,
                   "max_per_class": max_per_class, "rows": all_rows}, fh, indent=2, default=str)

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

    print(f"\nsaved gallery to {outdir}/ "
          f"({len([f for f in os.listdir(outdir) if f.endswith('.png')])} PNGs + "
          f"results.md + results.json)", flush=True)


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
