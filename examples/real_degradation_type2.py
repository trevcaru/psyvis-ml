"""Real DEGRADATION sweep -> per-item export -> type-2 AUROC vs. severity.

Runs the **real** instrument end to end: frozen pretrained ``timm`` ImageNet classifiers, real
Imagenette val images, the real :class:`~psyvis_ml.suites.DegradationSuite` gaussian-noise
stimulus applied to real pixels, swept from a clean baseline (σ = 0) to heavy noise. It writes
the trial-level ``per_item.csv`` / ``per_item.meta.json`` (see ``SCHEMA.md``) and then reads
that file back to compute the finding this artifact exists to support:

    **type-2 AUROC per severity level** — how well the model's own confidence separates its
    correct from its incorrect trials, at each level of degradation.

Nothing here is synthetic and no logits are fabricated: if the data or the ``[demo]`` extra is
missing, the script fails loudly rather than inventing an observer. The shared scoring rig lives
in ``type2_common.py``; its siblings are ``real_contrast_type2.py`` and
``real_distractor_type2.py``.

The type-2 confidence signal — and why we score TWO of them
-----------------------------------------------------------
The persisted ``margin`` is **signed**, with its decision boundary at 0, and for ``top_k=1``
``correct == (margin > 0)`` exactly. So the *signed* margin would separate correct from
incorrect trials perfectly (AUROC = 1.0) — a tautology, not a metric. Folding it to the unsigned
``|margin|`` looks like the fix, and is a trap.

``|margin|`` is **target-referenced**, and that matters enormously at high severity. The margin
is ``logit[true] - max(logit[others])``, so on an *error* trial ``|margin|`` is
``logit[top-1] - logit[true]`` — the size of the *miss*, a quantity the model cannot observe and
which is not its confidence in the answer it actually gave. Two mechanisms then drag the type-2
AUROC toward 0 as accuracy approaches the floor, with no metacognitive failure required:

1. the surviving correct trials are the *barely*-correct ones, whose ``|margin|`` is small; and
2. the error trials' ``|margin|`` grows precisely because the true class's evidence collapsed.

So the signal a type-2 analysis actually wants is ``decision_margin`` (schema 1.1):
``logit[top-1] - logit[top-2]``, winner minus runner-up — **decision-referenced**, always
``>= 0``, the model's confidence in the answer it gave. It equals ``margin`` on every correct
trial and diverges only on errors, which is exactly where the target-referenced signal misleads.
It is computed inside the sweep (the runner-up logit is nowhere else) and persisted as its own
column.

This script scores **both**, side by side, because the contrast is the finding: a sub-0.5 AUROC
on ``|margin|`` next to a healthy AUROC on ``decision_margin`` is the signature of the artifact,
not of a confidently-wrong model.

Reading the number
------------------
* AUROC > 0.5 — confidence is predictive: when the model is right it is more confident.
* AUROC ~ 0.5 — confidence carries no information about correctness (chance).
* AUROC < 0.5 — **anti-predictive**: the model is *more* confident on the trials it gets wrong.

``decision_margin`` AUROC is expected to *decay toward chance* with severity and to stay at or
above 0.5. If it ever drops reliably below 0.5, that is a genuine and surprising result about the
models' metacognition (not an artifact), and the rig says so loudly rather than burying it.

Run against YOUR Imagenette val (never bundled/downloaded here):

    pip install -e ".[demo]"
    export PSYVIS_IMAGENET_DIR=/path/to/imagenette2-320/val   # ImageFolder (WNID folders)
    python examples/real_degradation_type2.py

Optional env: PSYVIS_MODELS, PSYVIS_MAX_PER_CLASS (default 30), PSYVIS_OUTDIR, PSYVIS_NBOOT.
"""

import psyvis_ml as pe

import type2_common as t2

# Clean (σ = 0) baseline plus six increasing noise levels, up to a σ that drives both models into
# the floor — the sweep must span ceiling -> chance for the psychometric fit to mean anything, and
# for the type-2 readout to have both correct and incorrect trials at each end.
LEVELS = pe.linspace_levels(0.0, 0.8, 7)

KIND = "gaussian_noise"


def main():
    cfg = t2.config(outdir="real_degradation_type2")

    dataset = pe.datasets.imagenet_subset(cfg["data_dir"], max_per_class=cfg["max_per_class"],
                                          image_size=224, seed=0)
    print(f"dataset: {len(dataset.images)} real images from {cfg['data_dir']} "
          f"({len(dataset.metadata['classes'])} classes, {cfg['max_per_class']}/class)",
          flush=True)

    clfs = t2.load_models(cfg["model_names"])
    results = []
    for name, clf in clfs.items():
        suite = pe.suites.DegradationSuite(KIND, clip_range=(0.0, 1.0))
        print(f"measuring {name} over {len(LEVELS)} noise levels "
              f"({len(dataset.images) * len(LEVELS)} forward passes) ...", flush=True)
        res = pe.measure(model=clf, suite=suite, dataset=dataset, levels=LEVELS,
                         top_k=1, seed=0, model_name=name)
        acc = [c / t for c, t in zip(res.bundle.n_correct, res.bundle.n_trials, strict=True)]
        print("  accuracy by sigma: " +
              ", ".join(f"{lv:.2f}:{a:.3f}" for lv, a in zip(LEVELS, acc, strict=True)),
              flush=True)
        results.append(res)

    export = pe.write_per_item(
        results, cfg["outdir"],
        extra_metadata={
            "run": "REAL",
            "synthetic": False,
            "models": [f"timm:{m} (pretrained ImageNet-1k, frozen, eval)"
                       for m in cfg["model_names"]],
            "dataset": f"Imagenette val (real photographs, ImageFolder) at {cfg['data_dir']}",
            "n_images": int(len(dataset.images)),
            "max_per_class": cfg["max_per_class"],
            "image_size": 224,
            "stimulus": (f"DegradationSuite({KIND!r}, clip_range=(0.0, 1.0)) — real additive "
                         "gaussian pixel noise applied to real pixels; σ = 0 is the clean "
                         "baseline"),
            "levels_sigma": [float(x) for x in LEVELS],
            "type2_scoring": ("Score type-2 / meta-d' on decision_margin (decision-referenced, "
                              ">= 0). Do NOT use abs(margin): it is target-referenced and goes "
                              "artifactually anti-predictive at floor accuracy. See SCHEMA.md, "
                              "'The two margins'."),
            "note": ("REAL measured run: pretrained timm classifiers on real Imagenette images "
                     "under the real psyvis gaussian-noise stimulus. No synthetic model, no "
                     "fabricated logits."),
        },
    )
    print(f"\nper-item export: {export.data_path} ({export.n_rows} rows) "
          f"+ {export.metadata_path.name}", flush=True)

    return t2.report(export, x_label="sigma", n_boot=cfg["n_boot"])


if __name__ == "__main__":
    main()
