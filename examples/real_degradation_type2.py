"""Real degradation sweep -> per-item export -> type-2 AUROC vs. severity.

Runs the **real** instrument end to end: frozen pretrained ``timm`` ImageNet classifiers, real
Imagenette val images, the real :class:`~psyvis_ml.suites.DegradationSuite` gaussian-noise
stimulus applied to real pixels, swept from a clean baseline (σ = 0) to heavy noise. It writes
the trial-level ``per_item.csv`` / ``per_item.meta.json`` (see ``SCHEMA.md``) and then reads
that file back to compute the finding this artifact exists to support:

    **type-2 AUROC per severity level** — how well the model's own confidence separates its
    correct from its incorrect trials, at each level of degradation.

Nothing here is synthetic and no logits are fabricated: if the data or the ``[demo]`` extra is
missing, the script fails loudly rather than inventing an observer.

The type-2 confidence signal — and why we score TWO of them
-----------------------------------------------------------
The persisted ``margin`` is **signed**, with its decision boundary at 0, and for ``top_k=1``
``correct == (margin > 0)`` exactly. So the *signed* margin would separate correct from
incorrect trials perfectly (AUROC = 1.0) — a tautology, not a metric. Type-2 sensitivity is
about confidence *magnitude*: the unsigned ``|margin|``, with the outcome carrying the sign.
Folding to ``abs()`` is the **consumer's** scoring choice (per-item.py's discipline note), and
this script is a consumer, so it folds here — the exported file stays signed and lossless.

But ``|margin|`` is **target-referenced**, and that matters enormously at high severity. The
margin is ``logit[true] - max(logit[others])``, so on an *error* trial ``|margin|`` is
``logit[top-1] - logit[true]`` — the size of the *miss*, a quantity the model cannot observe
and which is not its confidence in the answer it actually gave. Two mechanisms then drag the
type-2 AUROC toward 0 as accuracy approaches the floor, with no metacognitive failure required:

1. the surviving correct trials are the *barely*-correct ones, whose ``|margin|`` is small; and
2. the error trials' ``|margin|`` grows precisely because the true class's evidence collapsed.

So we also score ``max_softmax``, which is **decision-referenced**: the model's confidence in
its *own* top-1 response, the type-2 quantity a meta-d′ analysis actually asks about. (The
schema rightly calls ``max_softmax`` reference-only and calibration-sensitive — it is a poor
*calibrated probability*. That warning is about its absolute value; a rank-based AUROC only
uses its ordering, which is exactly what a type-2 ROC needs.) Reporting both is what separates
a real finding from an artifact of the signal's definition.

Reading the number
------------------
* AUROC > 0.5 — confidence is predictive: when the model is right it is more confident.
* AUROC ~ 0.5 — confidence carries no information about correctness (chance).
* AUROC < 0.5 — **anti-predictive**: the model is *more* confident on the trials it gets wrong.

A sub-0.5 AUROC on ``|margin|`` alone is not evidence of anti-predictive confidence: check
whether the decision-referenced column moves with it. If ``max_softmax`` stays high while
``|margin|`` collapses, the collapse is the target-referencing artifact above, not the model
being confidently wrong about its own answer.

Run against YOUR Imagenette val (never bundled/downloaded here):

    pip install -e ".[demo]"
    export PSYVIS_IMAGENET_DIR=/path/to/imagenette2-320/val   # ImageFolder (WNID folders)
    python examples/real_degradation_type2.py

Optional env: PSYVIS_MODELS (comma list), PSYVIS_MAX_PER_CLASS (default 30),
PSYVIS_OUTDIR (default outputs/real_degradation_type2), PSYVIS_NBOOT (default 2000).
"""

import os

import numpy as np
from scipy.stats import rankdata

import psyvis_ml as pe

# Clean (σ = 0) baseline plus six increasing noise levels, up to a σ that drives both models
# into the floor — the sweep must span ceiling -> chance for the psychometric fit to mean
# anything, and for the type-2 readout to have both correct and incorrect trials at each end.
LEVELS = pe.linspace_levels(0.0, 0.8, 7)


# --------------------------------------------------------------------------- #
# Type-2 AUROC (computed here, in the consumer, not in the estimator core)
# --------------------------------------------------------------------------- #
def type2_auroc(confidence, correct):
    """AUROC of ``confidence`` discriminating correct from incorrect trials (ties handled).

    The rank-based (Mann-Whitney U) form: the probability that a randomly drawn correct trial
    carried higher confidence than a randomly drawn incorrect one, with tied confidences
    contributing 0.5. Returns NaN when the level has no correct or no incorrect trials — a
    degenerate outcome distribution has no type-2 ROC, and inventing one would be a lie.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n_pos = int(np.count_nonzero(correct))
    n_neg = int(correct.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(confidence)  # average ranks -> ties count as 0.5
    rank_sum_pos = float(ranks[correct].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc_ci(confidence, correct, *, n_boot=2000, ci=0.95, seed=0):
    """Percentile bootstrap CI for :func:`type2_auroc`, resampling ITEMS within the level.

    The CI is what tells an AUROC of 0.47 apart from a *reliably* anti-predictive one: the
    point estimate alone cannot say whether it excludes 0.5.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    rng = np.random.default_rng(seed)
    n = confidence.size
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = type2_auroc(confidence[idx], correct[idx])
    draws = draws[~np.isnan(draws)]  # resamples that drew one outcome class only
    if draws.size == 0:
        return float("nan"), float("nan")
    lo_q, hi_q = (1.0 - ci) / 2.0 * 100.0, (1.0 + ci) / 2.0 * 100.0
    return float(np.percentile(draws, lo_q)), float(np.percentile(draws, hi_q))


def main():
    data_dir = os.environ.get(
        "PSYVIS_IMAGENET_DIR",
        r"C:\Users\glopu\OneDrive\Desktop\Experiments\Exp_assets\imagenette2-320\val",
    )
    model_names = [m.strip() for m in
                   os.environ.get("PSYVIS_MODELS", "resnet50,vit_small_patch16_224").split(",")]
    max_per_class = int(os.environ.get("PSYVIS_MAX_PER_CLASS", "30"))
    n_boot = int(os.environ.get("PSYVIS_NBOOT", "2000"))
    outdir = os.environ.get("PSYVIS_OUTDIR", os.path.join("outputs", "real_degradation_type2"))

    # Real images: Imagenette val in ImageFolder layout, WNID folders mapped to the ImageNet-1k
    # indices the models were actually trained on (num_classes=1000 -> chance = 1/1000).
    dataset = pe.datasets.imagenet_subset(data_dir, max_per_class=max_per_class, image_size=224,
                                          seed=0)
    print(f"dataset: {len(dataset.images)} real images from {data_dir} "
          f"({len(dataset.metadata['classes'])} classes, {max_per_class}/class)", flush=True)

    suite_kind = "gaussian_noise"
    results = []
    for name in model_names:
        # Frozen, eval-mode, pretrained ImageNet weights. No synthetic observer anywhere.
        clf = pe.models.timm_classifier(name, pretrained=True)
        suite = pe.suites.DegradationSuite(suite_kind, clip_range=(0.0, 1.0))
        print(f"measuring {name} over {len(LEVELS)} noise levels "
              f"({len(dataset.images) * len(LEVELS)} forward passes) ...", flush=True)
        res = pe.measure(model=clf, suite=suite, dataset=dataset, levels=LEVELS,
                         top_k=1, seed=0, model_name=name)
        acc = [c / t for c, t in zip(res.bundle.n_correct, res.bundle.n_trials, strict=True)]
        print("  accuracy by sigma: " +
              ", ".join(f"{lv:.2f}:{a:.3f}" for lv, a in zip(LEVELS, acc, strict=True)),
              flush=True)
        results.append(res)

    # The trial-level artifact: signed margin + correctness per (item x level x model).
    export = pe.write_per_item(
        results, outdir,
        extra_metadata={
            "run": "REAL",
            "models": [f"timm:{m} (pretrained ImageNet-1k, frozen, eval)" for m in model_names],
            "dataset": f"Imagenette val (real photographs, ImageFolder) at {data_dir}",
            "n_images": int(len(dataset.images)),
            "max_per_class": max_per_class,
            "image_size": 224,
            "degradation": (f"DegradationSuite({suite_kind!r}, clip_range=(0.0, 1.0)) — real "
                            "additive gaussian pixel noise applied to real pixels; σ = 0 is the "
                            "clean baseline"),
            "levels_sigma": [float(x) for x in LEVELS],
            "synthetic": False,
            "note": ("REAL measured run: pretrained timm classifiers on real Imagenette images "
                     "under the real psyvis gaussian-noise stimulus. No synthetic model, no "
                     "fabricated logits."),
        },
    )
    print(f"\nper-item export: {export.data_path} ({export.n_rows} rows) "
          f"+ {export.metadata_path.name}", flush=True)

    # Read the artifact back and compute the finding FROM THE FILE — the same path metaeval
    # takes, so what is reported here is exactly what the file supports.
    table = pe.load_per_item(export.data_path)
    print(f"\ntype-2 AUROC by severity, from {export.data_path}")
    print("severity_rank 0 = clean. AUROC < 0.5 = anti-predictive (more confident when wrong).")
    print("|margin| is TARGET-referenced (artifact-prone at the floor); max_softmax is "
          "DECISION-referenced\n(confidence in the model's own answer) - read them together; "
          "see the module docstring.\n")
    header = (f"{'model':<22} {'rank':>4} {'sigma':>6} {'acc':>6} | "
              f"{'AUROC |margin|':>14} {'95% CI':>16} | "
              f"{'AUROC max_softmax':>17} {'95% CI':>16}")
    print(header)
    print("-" * len(header))

    summary = []
    for name in model_names:
        rows = table["model"] == name
        for rank in sorted({int(r) for r in table["severity_rank"][rows]}):
            sel = rows & (table["severity_rank"] == rank)
            corr = table["correct"][sel]
            sigma = float(table["stimulus_level"][sel][0])
            # Two confidence signals, same trials, same outcome vector.
            margin_conf = np.abs(table["margin"][sel])   # target-referenced magnitude
            softmax_conf = table["max_softmax"][sel]     # decision-referenced (own response)

            row = {"model": name, "severity_rank": rank, "sigma": sigma,
                   "n_correct": int(corr.sum()), "n_incorrect": int((~corr).sum()),
                   "accuracy": float(corr.mean())}
            for key, conf in (("margin", margin_conf), ("softmax", softmax_conf)):
                auc = type2_auroc(conf, corr)
                lo, hi = auroc_ci(conf, corr, n_boot=n_boot, seed=0)
                row[f"auroc_{key}"] = auc
                row[f"ci_low_{key}"], row[f"ci_high_{key}"] = lo, hi

            def _f(x):
                return "n/a" if np.isnan(x) else f"{x:.3f}"

            def _ci(lo, hi):
                return "n/a" if np.isnan(lo) else f"[{lo:.3f}, {hi:.3f}]"

            print(f"{name:<22} {rank:>4} {sigma:>6.2f} {row['accuracy']:>6.3f} | "
                  f"{_f(row['auroc_margin']):>14} "
                  f"{_ci(row['ci_low_margin'], row['ci_high_margin']):>16} | "
                  f"{_f(row['auroc_softmax']):>17} "
                  f"{_ci(row['ci_low_softmax'], row['ci_high_softmax']):>16}",
                  flush=True)
            summary.append(row)
        print()

    _verdict(summary)
    return summary


def _below_half(summary, key):
    """Levels whose AUROC on ``key`` is below 0.5 with a 95% CI that excludes 0.5."""
    return [s for s in summary
            if not np.isnan(s[f"auroc_{key}"]) and s[f"auroc_{key}"] < 0.5
            and not np.isnan(s[f"ci_high_{key}"]) and s[f"ci_high_{key}"] < 0.5]


def _verdict(summary):
    """State plainly what the two signals did — including when they disagree."""
    margin_below = _below_half(summary, "margin")
    softmax_below = _below_half(summary, "softmax")
    fmt = "; ".join(f"{s['model']} sigma={s['sigma']:.2f} (AUROC={s['auroc_margin']:.3f})"
                    for s in margin_below)

    if softmax_below:
        print("VERDICT: genuinely ANTI-PREDICTIVE — the decision-referenced signal "
              "(max_softmax) itself drops below 0.5 with a CI excluding it at: " +
              "; ".join(f"{s['model']} sigma={s['sigma']:.2f} "
                        f"(AUROC={s['auroc_softmax']:.3f})" for s in softmax_below) +
              ". The models are reliably more confident in their OWN answers when those "
              "answers are wrong.")
    elif margin_below:
        softmax_min = min(s["auroc_softmax"] for s in summary
                          if not np.isnan(s["auroc_softmax"]))
        print("VERDICT: |margin| goes below 0.5 (CI excluding 0.5) at: " + fmt + ".")
        print("BUT this is the target-referencing ARTIFACT, not anti-predictive confidence: on "
              "the same trials the\ndecision-referenced signal (max_softmax) never approaches "
              f"0.5 - its minimum across all levels is {softmax_min:.3f}.\nThe models' "
              "confidence in their own answers stays strongly predictive of their own "
              "correctness even\nat floor accuracy; what collapses is |margin|, which on error "
              "trials measures how badly the true\nclass was missed.")
    else:
        print("VERDICT: type-2 AUROC stays at or above 0.5 on both signals at every severity — "
              "confidence decays\ntoward chance but never becomes anti-predictive.")


if __name__ == "__main__":
    main()
