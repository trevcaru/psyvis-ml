"""Real DISTRACTOR-ROBUSTNESS sweep -> per-item export -> type-2 AUROC vs. severity.

The distractor sibling of ``real_degradation_type2.py``: frozen pretrained ``timm`` ImageNet
classifiers, real Imagenette val images, the real
:class:`~psyvis_ml.suites.DistractorRobustness` stimulus — a **big centred target** with real
image-content distractors composited into the margins — swept by **distractor size**. It writes
the trial-level ``per_item.csv`` / ``per_item.meta.json`` (schema 1.1: both the target-referenced
``margin`` and the decision-referenced ``decision_margin``) and reads that file back to score
type-2 AUROC per severity level.

This is DISTRACTOR ROBUSTNESS, not crowding
-------------------------------------------
It measures how a real classifier's accuracy degrades as competing image content grows around a
centred target. It is **not** a model of human visual crowding: no eccentricity, no fixation, no
flanker-spacing law, and the observer is argmax top-1 on a composited canvas. Do not read a
threshold here as a crowding distance.

Size, not spacing
-----------------
The suite sweeps distractor **size**, and the notebook's older "swept spacing" framing is stale.
The reason is in the suite's own docstring: whole-photo targets are only reliably recognised when
large and central, which leaves no room to move a fixed-size distractor from "close" to "clear" —
so a *spacing* sweep sits on the recognition ceiling and yields extrapolated, meaningless
thresholds. Growing the distractors instead gives a real ceiling -> floor transition: small
distractors barely interfere, large ones occlude the target. Accuracy therefore **falls** with
size (a decreasing suite), so ``severity_rank 0`` is the smallest distractor (cleanest).

Geometry (matching ``generate_gallery.py``, which fixed the ceiling problem): a 176-px target
(~79% of the 224-px canvas, comfortably clear of the recognition ceiling), 4 distractors grown
from the margins, each a **real 96-px Imagenette photo** — heterogeneous image content, not a
gray square, so the distractors are genuinely competing objects.

Two conditions
--------------
The suite emits the swept ``distractors`` condition *and* an ``undistracted`` baseline (target
alone). The baseline's stimulus does not depend on the level, so it is a flat ceiling check, not
a severity axis — both are exported, but only ``distractors`` is scored against severity.

Run against YOUR Imagenette val (never bundled/downloaded here):

    pip install -e ".[demo]"
    export PSYVIS_IMAGENET_DIR=/path/to/imagenette2-320/val   # ImageFolder (WNID folders)
    python examples/real_distractor_type2.py

Optional env: PSYVIS_MODELS, PSYVIS_MAX_PER_CLASS (default 30), PSYVIS_OUTDIR, PSYVIS_NBOOT.
"""

import numpy as np

import psyvis_ml as pe

import type2_common as t2

#: Distractor size in px, from barely-there to large enough to occlude the target. Brackets the
#: ceiling -> floor transition found by calibrate_distractor_size (same band as the gallery).
LEVELS = pe.linspace_levels(20, 105, 7)

CANVAS = (224, 224)
TARGET_PX = 176      # ~79% of the canvas: big and centred, so the recognition ceiling holds
DISTRACTOR_PX = 96   # source size of the real-photo distractor content (resized to each level)


def main():
    cfg = t2.config(outdir="real_distractor_type2")

    # The TARGET set (big, centred) and a separate small image used as distractor CONTENT. Both
    # are real Imagenette photos; the compositor resizes the patch to each swept size.
    ds_target = pe.datasets.imagenet_subset(cfg["data_dir"], max_per_class=cfg["max_per_class"],
                                            image_size=TARGET_PX, seed=0)
    ds_small = pe.datasets.imagenet_subset(cfg["data_dir"], max_per_class=1,
                                           image_size=DISTRACTOR_PX, seed=0)
    patch = np.asarray(ds_small.images[0])
    print(f"dataset: {len(ds_target.images)} real {TARGET_PX}px targets from {cfg['data_dir']} "
          f"({len(ds_target.metadata['classes'])} classes, {cfg['max_per_class']}/class); "
          f"distractor content = real {DISTRACTOR_PX}px photo {ds_small.item_ids[0]}", flush=True)

    clfs = t2.load_models(cfg["model_names"])
    results = []
    for name, clf in clfs.items():
        suite = pe.suites.DistractorRobustness(
            distractor_patch=patch, canvas_shape=CANVAS, n_distractors=4, include_baseline=True)
        n_cond = len(suite.conditions())
        print(f"measuring {name} over {len(LEVELS)} distractor sizes x {n_cond} conditions "
              f"({len(ds_target.images) * len(LEVELS) * n_cond} forward passes) ...", flush=True)
        res = pe.measure(model=clf, suite=suite, dataset=ds_target, levels=LEVELS,
                         top_k=1, seed=0, model_name=name)
        for cr in res.condition_results:
            acc = [c / t for c, t in zip(cr.bundle.n_correct, cr.bundle.n_trials, strict=True)]
            print(f"  [{cr.label}] accuracy by size: " +
                  ", ".join(f"{lv:.0f}px:{a:.3f}" for lv, a in zip(LEVELS, acc, strict=True)),
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
            "n_images": int(len(ds_target.images)),
            "max_per_class": cfg["max_per_class"],
            "stimulus": (f"DistractorRobustness(canvas={CANVAS}, n_distractors=4, "
                         f"include_baseline=True) — a real {TARGET_PX}px centred Imagenette "
                         f"target on a {CANVAS[0]}px canvas with 4 real-photo distractors "
                         f"({DISTRACTOR_PX}px source content) grown from the margins"),
            "swept_variable": "distractor size in px (NOT spacing — see the module docstring)",
            "levels_distractor_size_px": [float(x) for x in LEVELS],
            "conditions": ("'distractors' = the swept condition; 'undistracted' = target-alone "
                           "baseline, level-invariant (a flat ceiling check, not a severity "
                           "axis)."),
            "interpretation": ("DISTRACTOR ROBUSTNESS, not crowding: no eccentricity, no "
                               "fixation, no flanker-spacing law. Do not read a threshold here "
                               "as a human crowding distance."),
            "type2_scoring": ("Score type-2 / meta-d' on decision_margin (decision-referenced, "
                              ">= 0). Do NOT use abs(margin): it is target-referenced and goes "
                              "artifactually anti-predictive at floor accuracy. See SCHEMA.md, "
                              "'The two margins'."),
            "note": ("REAL measured run: pretrained timm classifiers on real Imagenette images "
                     "under the real psyvis distractor stimulus. No synthetic model, no "
                     "fabricated logits."),
        },
    )
    print(f"\nper-item export: {export.data_path} ({export.n_rows} rows) "
          f"+ {export.metadata_path.name}", flush=True)

    # Ceiling check: the undistracted baseline should sit high and flat — if it does not, the
    # target is not clearing the recognition ceiling and the swept curve measures the wrong thing.
    table = pe.load_per_item(export.data_path)
    for name in cfg["model_names"]:
        base = (table["model"] == name) & (table["condition"] == "undistracted")
        print(f"ceiling check [{name}] undistracted baseline accuracy: "
              f"{table['correct'][base].mean():.3f} (target alone, no distractors)")

    # Only the swept condition has a severity axis worth scoring.
    return t2.report(export, x_label="size (px)", condition="distractors", n_boot=cfg["n_boot"])


if __name__ == "__main__":
    main()
