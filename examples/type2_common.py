"""Shared rig for the real per-item type-2 runs (degradation / contrast / distractor).

The three ``examples/real_*_type2.py`` scripts differ only in their **suite setup** — the model
list, the dataset, the seeding, the per-item export, the type-2 scoring and the verdict logic are
identical across them, so they live here once rather than three times. Each script supplies its
own suite, levels and metadata; everything below is the common apparatus.

What the scoring here assumes (see ``SCHEMA.md``, "The two margins"):

* the **type-2 confidence signal is** ``decision_margin`` = ``logit[top-1] - logit[top-2]``,
  which is decision-referenced (the model's confidence in the answer it actually gave) and
  always ``>= 0``;
* ``abs(margin)`` is **not** a substitute — it is target-referenced, so on an error trial it
  measures how badly the *true* class was missed and is dragged below 0.5 at floor accuracy for
  definitional reasons. It is scored alongside only to keep that gap visible.

Everything is real: pretrained ``timm`` classifiers, frozen and in eval mode, on real Imagenette
images. If the data or the ``[demo]`` extra is missing these scripts fail loudly rather than
inventing an observer.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.stats import rankdata

import psyvis_ml as pe

#: Your Imagenette val set (ImageFolder layout, WNID folders). Override with PSYVIS_IMAGENET_DIR.
DEFAULT_DATA_DIR = (
    r"C:\Users\glopu\OneDrive\Desktop\Experiments\Exp_assets\imagenette2-320\val"
)

#: A CNN and a ViT, so every finding is checked against two real architectures, not one.
DEFAULT_MODELS = "resnet50,vit_small_patch16_224"

#: A type-2 AUROC is only as good as its *minority* class: with a handful of correct (or
#: incorrect) trials the estimate is noise no matter how many trials the level has in total.
#: Cells below this are flagged rather than silently reported as findings.
THIN_MINORITY = 30


def config(*, outdir, max_per_class=30, n_boot=2000):
    """Resolve the run configuration from the environment (all knobs overridable)."""
    return {
        "data_dir": os.environ.get("PSYVIS_IMAGENET_DIR", DEFAULT_DATA_DIR),
        "model_names": [m.strip() for m in
                        os.environ.get("PSYVIS_MODELS", DEFAULT_MODELS).split(",")],
        "max_per_class": int(os.environ.get("PSYVIS_MAX_PER_CLASS", str(max_per_class))),
        "n_boot": int(os.environ.get("PSYVIS_NBOOT", str(n_boot))),
        "outdir": os.environ.get("PSYVIS_OUTDIR", os.path.join("outputs", outdir)),
    }


def load_models(model_names):
    """Frozen, eval-mode, pretrained ImageNet classifiers as plain ``image -> logits`` callables."""
    return {name: pe.models.timm_classifier(name, pretrained=True) for name in model_names}


# --------------------------------------------------------------------------- #
# Type-2 AUROC (computed here, in the consumer, never in the estimator core)
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
    return (float(ranks[correct].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc_ci(confidence, correct, *, n_boot=2000, ci=0.95, seed=0):
    """Percentile bootstrap CI for :func:`type2_auroc`, resampling ITEMS within the level.

    The CI is what tells an AUROC of 0.47 apart from a *reliably* anti-predictive one: the point
    estimate alone cannot say whether it excludes 0.5.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    rng = np.random.default_rng(seed)
    n = confidence.size
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = type2_auroc(confidence[idx], correct[idx])
    draws = draws[~np.isnan(draws)]   # resamples that happened to draw one outcome class only
    if draws.size == 0:
        return float("nan"), float("nan")
    lo_q, hi_q = (1.0 - ci) / 2.0 * 100.0, (1.0 + ci) / 2.0 * 100.0
    return float(np.percentile(draws, lo_q)), float(np.percentile(draws, hi_q))


# --------------------------------------------------------------------------- #
# The schema-1.1 invariant, checked on the file we just wrote
# --------------------------------------------------------------------------- #
def check_contract(table):
    """Assert the two-margin contract on the exported file, before anything is scored from it.

    Cheap, and it fails the run rather than letting a broken export quietly produce a plausible
    number: ``decision_margin`` must be non-negative everywhere and *identical* to ``margin`` on
    exactly the trials the model got right.
    """
    margin, decision, correct = table["margin"], table["decision_margin"], table["correct"]
    assert np.all(decision >= 0), "decision_margin went negative — winner < runner-up?"
    assert np.array_equal(decision[correct], margin[correct]), \
        "decision_margin != margin on a correct trial — the target was not the winner?"
    assert np.all(margin[~correct] < 0), "an error trial had a non-negative target margin"
    print(f"contract OK: decision_margin >= 0 on all {len(table)} rows; == margin on all "
          f"{int(correct.sum())} correct trials; diverges on all {int((~correct).sum())} errors.")


# --------------------------------------------------------------------------- #
# Score + report
# --------------------------------------------------------------------------- #
def score_by_severity(table, model_names, *, condition=None, n_boot=2000):
    """Type-2 AUROC per (model × severity level), on both confidence signals.

    ``condition`` restricts to one condition label (the distractor suite exports a swept
    ``distractors`` condition *and* a level-invariant ``undistracted`` baseline; only the former
    has a severity axis worth scoring).
    """
    summary = []
    for name in model_names:
        rows = table["model"] == name
        if condition is not None:
            rows = rows & (table["condition"] == condition)
        for rank in sorted({int(r) for r in table["severity_rank"][rows]}):
            sel = rows & (table["severity_rank"] == rank)
            corr = table["correct"][sel]
            signals = {
                "decision": table["decision_margin"][sel],   # the type-2 signal
                "margin": np.abs(table["margin"][sel]),      # target-referenced (artifact-prone)
            }
            row = {
                "model": name,
                "severity_rank": rank,
                "level": float(table["stimulus_level"][sel][0]),
                "n_correct": int(corr.sum()),
                "n_incorrect": int((~corr).sum()),
                "accuracy": float(corr.mean()),
            }
            row["minority"] = min(row["n_correct"], row["n_incorrect"])
            row["thin"] = row["minority"] < THIN_MINORITY
            for key, conf in signals.items():
                row[f"auroc_{key}"] = type2_auroc(conf, corr)
                row[f"ci_low_{key}"], row[f"ci_high_{key}"] = auroc_ci(
                    conf, corr, n_boot=n_boot, seed=0)
            summary.append(row)
    return summary


def _f(x):
    return "n/a" if np.isnan(x) else f"{x:.3f}"


def _ci(lo, hi):
    return "n/a" if np.isnan(lo) else f"[{lo:.3f}, {hi:.3f}]"


def print_table(summary, *, x_label, source):
    """The AUROC-vs-severity table. ``*`` marks a cell too thin for its AUROC to mean much."""
    print(f"\ntype-2 AUROC by severity, from {source}")
    print("severity_rank 0 = cleanest/easiest. AUROC < 0.5 = anti-predictive (more confident "
          "when wrong).")
    print("decision_margin is the TYPE-2 signal (confidence in the model's own answer). "
          "|margin| is\nTARGET-referenced and artifact-prone at the floor - shown alongside "
          "only to expose that gap.")
    print(f"* = thin cell: minority class < {THIN_MINORITY} trials, so its AUROC is noise-"
          "dominated.\n")
    header = (f"{'model':<22} {'rank':>4} {x_label:>11} {'acc':>6} {'n_cor':>6} {'n_err':>6} | "
              f"{'AUROC decision_margin':>21} {'95% CI':>16} | "
              f"{'AUROC |margin|':>14} {'95% CI':>16}")
    print(header)
    print("-" * len(header))
    last = None
    for s in summary:
        if last is not None and s["model"] != last:
            print()
        last = s["model"]
        flag = "*" if s["thin"] else " "
        print(f"{s['model']:<22} {s['severity_rank']:>4} {s['level']:>11.4g} "
              f"{s['accuracy']:>6.3f} {s['n_correct']:>5}{flag} {s['n_incorrect']:>6} | "
              f"{_f(s['auroc_decision']):>21} "
              f"{_ci(s['ci_low_decision'], s['ci_high_decision']):>16} | "
              f"{_f(s['auroc_margin']):>14} "
              f"{_ci(s['ci_low_margin'], s['ci_high_margin']):>16}", flush=True)
    print()


def _below_half(summary, key):
    """Levels whose AUROC on ``key`` is below 0.5 *as a point estimate*."""
    return [s for s in summary
            if not np.isnan(s[f"auroc_{key}"]) and s[f"auroc_{key}"] < 0.5]


def _reliably_below_half(summary, key):
    """Levels below 0.5 with a 95% CI excluding 0.5 — AND enough minority trials to believe it.

    The thinness guard is not fussiness, it is a property of the estimator. The CI here
    bootstraps over ITEMS, so in a cell with (say) 2 correct trials, almost every resample
    redraws those same two items — whose confidences are fixed. The interval then expresses
    uncertainty in the *error* distribution alone and never in *which* items came out correct,
    so it comes back narrow and confidently below 0.5 on a sample that can support no such claim.
    A sub-0.5 AUROC is only evidence of anti-predictive confidence when the minority class is
    actually populated; below that, it is reported as a thin cell, not as a finding.
    """
    return [s for s in _below_half(summary, key)
            if not s["thin"]
            and not np.isnan(s[f"ci_high_{key}"]) and s[f"ci_high_{key}"] < 0.5]


def verdict(summary):
    """State plainly what the two signals did — and flag it hard if decision_margin dips.

    Three outcomes, which must not be blurred together: a sub-0.5 point estimate whose CI
    *excludes* 0.5 is a real claim about the models' metacognition and gets shouted; one whose CI
    *straddles* 0.5 is a noise-dominated cell (at the accuracy floor only a handful of correct
    trials remain) and is reported as exactly that — never silently rounded up to "never dips".
    """
    scored = [s for s in summary if not np.isnan(s["auroc_decision"])]
    if not scored:
        print("VERDICT: no level had both correct and incorrect trials — no type-2 ROC exists.")
        return

    reliably_below = _reliably_below_half(summary, "decision")     # adequately-sampled cells only
    below = _below_half(summary, "decision")
    thin_below = [s for s in below if s["thin"]]
    noisy_below = [s for s in below if s not in reliably_below and s not in thin_below]
    margin_below = _below_half(summary, "margin")

    if reliably_below:
        print("*** UNEXPECTED - STOP AND LOOK ***")
        print("decision_margin AUROC crossed BELOW 0.5 with a 95% CI EXCLUDING 0.5, in a cell "
              f"with >= {THIN_MINORITY} minority\ntrials (so this is NOT the thin-cell "
              "bootstrap failure), at: " +
              "; ".join(f"{s['model']} level={s['level']:.4g} "
                        f"(AUROC={s['auroc_decision']:.3f}, {s['n_correct']} correct trials)"
                        for s in reliably_below))
        print("This is the DECISION-referenced signal, so it is NOT the target-margin artifact "
              "either: it would mean\nthe models are reliably more confident in their own "
              "answers when those answers are wrong. Do not ship\nthis number without "
              "investigating it.")
        return

    solid = [s for s in scored if not s["thin"]] or scored
    d_best = max(s["auroc_decision"] for s in solid)
    d_worst = min(s["auroc_decision"] for s in solid)
    print(f"VERDICT: decision_margin AUROC decays toward chance across the severity axis "
          f"({d_best:.3f} -> {d_worst:.3f} over the\ncells with >= {THIN_MINORITY} minority "
          "trials) and stays ABOVE 0.5 in every adequately-sampled cell. Confidence\nin the "
          "model's own answer stays predictive of that answer's own correctness. That is the "
          "expected,\ncorrect behaviour for a decision-referenced signal.")

    if thin_below:
        print("\nTHIN-CELL DIP (reported, not hidden - and NOT a finding): decision_margin's "
              "point estimate falls below\n0.5 at " +
              "; ".join(f"{s['model']} level={s['level']:.4g} "
                        f"(AUROC={s['auroc_decision']:.3f}, 95% CI "
                        f"[{s['ci_low_decision']:.3f}, {s['ci_high_decision']:.3f}], only "
                        f"{s['n_correct']} correct of "
                        f"{s['n_correct'] + s['n_incorrect']})" for s in thin_below) +
              f".\nThe CI looks decisive but is NOT trustworthy here: it bootstraps over items, "
              f"so with < {THIN_MINORITY} correct\ntrials nearly every resample redraws the SAME "
              "few correct items, and the interval expresses uncertainty in\nthe error "
              "distribution alone - never in which items happened to come out correct. At this "
              "level the\nmodel is essentially never right, so it has almost no correct trials "
              "left to be confident about. To\nturn this cell into evidence either way, it needs "
              "more images, not a re-scoring.")

    if noisy_below:
        print("\nCAVEAT (stated, not buried): the point estimate also falls below 0.5 at " +
              "; ".join(f"{s['model']} level={s['level']:.4g} "
                        f"(AUROC={s['auroc_decision']:.3f}, 95% CI "
                        f"[{s['ci_low_decision']:.3f}, {s['ci_high_decision']:.3f}], minority "
                        f"class {s['minority']} trials)" for s in noisy_below) +
              ".\nEach such CI straddles 0.5, so these are noise-dominated cells, not evidence "
              "of anti-predictive\nconfidence - but they are NOT a clean 'always >= 0.5' either, "
              "and are reported as such.")

    if margin_below:
        print("\nMeanwhile |margin| (target-referenced) goes below 0.5 on the very same trials, "
              "at: " +
              "; ".join(f"{s['model']} level={s['level']:.4g} "
                        f"(AUROC={s['auroc_margin']:.3f})" for s in margin_below) +
              ".\nThat gap is the artifact decision_margin was added to eliminate: on an error "
              "trial |margin| measures\nhow badly the TRUE class was missed, not how confident "
              "the model was. Score type-2 on decision_margin.")


def report(export, *, x_label, condition=None, n_boot=2000):
    """Read the written file back, check the contract, score it, print the table + verdict.

    Scoring reads the **artifact**, not the in-memory results — the same path metaeval takes — so
    what is reported is exactly what the file supports.
    """
    table = pe.load_per_item(export.data_path)
    check_contract(table)
    model_names = sorted(set(table["model"].tolist()))
    summary = score_by_severity(table, model_names, condition=condition, n_boot=n_boot)
    print_table(summary, x_label=x_label, source=export.data_path)
    verdict(summary)
    return summary
