"""Real CONTRAST sweep -> per-item export -> type-2 AUROC vs. severity.

The contrast sibling of ``real_degradation_type2.py``: frozen pretrained ``timm`` ImageNet
classifiers, real Imagenette val images, the real :class:`~psyvis_ml.suites.ContrastThreshold`
stimulus (RMS contrast scaling applied to real pixels), swept log-spaced from near-invisible to
near-full contrast. It writes the trial-level ``per_item.csv`` / ``per_item.meta.json``
(schema 1.1: both the target-referenced ``margin`` and the decision-referenced
``decision_margin``) and reads that file back to score type-2 AUROC per severity level.

The contrast suite is the one **increasing** axis in the library: higher contrast is *easier*,
so ``decreasing`` is False and the direction-aware ``severity_rank`` runs backwards relative to
the raw level — ``severity_rank 0`` is the **highest** contrast (the cleanest, easiest end) and
the last rank is the lowest contrast (the hardest). That inversion is the whole reason
``severity_rank`` exists: a consumer gets a monotone clean -> degraded axis without having to
know which direction this suite's native units run. Do not sort by ``stimulus_level`` and assume
severity.

Why a log-spaced sweep: contrast sensitivity is a positive, multiplicative axis (the suite fits a
Weibull), so equal *ratios*, not equal differences, sample the psychometric function evenly.

Run against YOUR Imagenette val (never bundled/downloaded here):

    pip install -e ".[demo]"
    export PSYVIS_IMAGENET_DIR=/path/to/imagenette2-320/val   # ImageFolder (WNID folders)
    python examples/real_contrast_type2.py

Optional env: PSYVIS_MODELS, PSYVIS_MAX_PER_CLASS (default 30), PSYVIS_OUTDIR, PSYVIS_NBOOT.
"""

import psyvis_ml as pe

import type2_common as t2

# Log-spaced RMS contrast, spanning the floor (0.01, barely visible) to a high-contrast ceiling
# (0.5). The sweep must span ceiling -> chance for the fit to mean anything and for the type-2
# readout to have both correct and incorrect trials at each end.
LEVELS = pe.linspace_levels(0.01, 0.5, 7, spacing="log")


def main():
    cfg = t2.config(outdir="real_contrast_type2")

    dataset = pe.datasets.imagenet_subset(cfg["data_dir"], max_per_class=cfg["max_per_class"],
                                          image_size=224, seed=0)
    print(f"dataset: {len(dataset.images)} real images from {cfg['data_dir']} "
          f"({len(dataset.metadata['classes'])} classes, {cfg['max_per_class']}/class)",
          flush=True)

    clfs = t2.load_models(cfg["model_names"])
    results = []
    for name, clf in clfs.items():
        # clip_range keeps the contrast-scaled pixels in [0, 1] — a real model needs valid
        # pixels, and it makes the persisted stimulus exactly what the model saw.
        suite = pe.suites.ContrastThreshold(contrast_metric="rms", clip_range=(0.0, 1.0))
        print(f"measuring {name} over {len(LEVELS)} contrast levels "
              f"({len(dataset.images) * len(LEVELS)} forward passes) ...", flush=True)
        res = pe.measure(model=clf, suite=suite, dataset=dataset, levels=LEVELS,
                         top_k=1, seed=0, model_name=name)
        acc = [c / t for c, t in zip(res.bundle.n_correct, res.bundle.n_trials, strict=True)]
        print("  accuracy by RMS contrast: " +
              ", ".join(f"{lv:.3f}:{a:.3f}" for lv, a in zip(LEVELS, acc, strict=True)),
              flush=True)
        results.append(res)

    export = pe.write_per_item(
        results, cfg["outdir"],
        extra_metadata={
            "run": "REAL",
            "synthetic": False,
            "models": [f"timm:{m} (pretrained ImageNet-1k, frozen, eval)"
                       for m in cfg["model_names"]],
            "dataset": (f"Imagenette val (real photographs, ImageFolder) at {cfg['data_dir']}"),
            "n_images": int(len(dataset.images)),
            "max_per_class": cfg["max_per_class"],
            "image_size": 224,
            "stimulus": ("ContrastThreshold(contrast_metric='rms', clip_range=(0.0, 1.0)) — real "
                         "RMS-contrast scaling of real pixels"),
            "levels_rms_contrast": [float(x) for x in LEVELS],
            "axis_note": ("INCREASING axis: higher contrast is EASIER, so severity_rank 0 is the "
                          "HIGHEST contrast (cleanest) and the last rank is the lowest contrast "
                          "(hardest). Use severity_rank, not stimulus_level, for a monotone "
                          "clean -> degraded axis."),
            "type2_scoring": ("Score type-2 / meta-d' on decision_margin (decision-referenced, "
                              ">= 0). Do NOT use abs(margin): it is target-referenced and goes "
                              "artifactually anti-predictive at floor accuracy. See SCHEMA.md, "
                              "'The two margins'."),
            "note": ("REAL measured run: pretrained timm classifiers on real Imagenette images "
                     "under the real psyvis contrast stimulus. No synthetic model, no fabricated "
                     "logits."),
        },
    )
    print(f"\nper-item export: {export.data_path} ({export.n_rows} rows) "
          f"+ {export.metadata_path.name}", flush=True)

    return t2.report(export, x_label="contrast", n_boot=cfg["n_boot"])


if __name__ == "__main__":
    main()
