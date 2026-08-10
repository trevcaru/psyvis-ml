"""Test A — threshold split-half reliability (psyvis-ml).

A KILL-CRITERION check, not a feature. One question: is a measured threshold a property
of the model, or an artifact of the particular images we happened to sweep? Method: randomly
split the real Imagenette val set into two disjoint halves, run the IDENTICAL sweep on each,
and compare the resulting thresholds (+ bootstrap CIs) over several split seeds.

REAL DATA ONLY. There is no synthetic/simulate/demo fallback anywhere in this file. Every
number traces to a real forward pass of a real pretrained timm model on a real image file on
disk. If real images or real models are missing, this crashes loudly. The only np.random use
is the seeded split of item indices and the parametric bootstrap inside the fitter.

Consumer of the public psyvis_ml API only — no changes to psyvis_ml/ source.

Usage (PowerShell):
    python experiments/test_a_split_half/run_split_half.py --seeds 0
    python experiments/test_a_split_half/run_split_half.py --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import psyvis_ml as pe

# ---------------------------------------------------------------------------- #
# Fixed configuration — reuse the STANDING GALLERY's exact suites/levels.
# (examples/generate_gallery.py: contrast geomspace(0.01,0.5,7) weibull;
#  degradation linspace(0.0,0.8,7) gaussian_noise logistic/decreasing.)
# ---------------------------------------------------------------------------- #
DEFAULT_DATA_DIR = r"C:/Users/glopu/OneDrive/Desktop/Experiments/Exp_assets/imagenette2-320/val"
MODEL_NAMES = ["resnet50", "vit_small_patch16_224"]
IMAGE_SIZE = 224
SWEEP_SEED = 0          # stimulus-generation seed — IDENTICAL for both halves (spec).
TARGET = 0.75           # threshold criterion (proportion correct) — gallery default.

# Between-model threshold gaps measured by the standing gallery (200 imgs) — the effect the
# library claims to detect. Split-half |Δ| is reported as a fraction of these.
GALLERY_GAP = {"contrast": 0.1458, "degradation": 0.03588}
GALLERY_THR = {  # gallery point thresholds, for orientation only.
    "contrast": {"resnet50": 0.04971, "vit_small_patch16_224": 0.1955},
    "degradation": {"resnet50": 0.08669, "vit_small_patch16_224": 0.1226},
}


def suite_specs():
    """The two suites, each with the gallery's exact level array. Distractor is excluded on
    purpose (halves change the distractor-pool composition = a second confound)."""
    return {
        "contrast": {
            "make": lambda: pe.suites.ContrastThreshold(contrast_metric="rms",
                                                        clip_range=(0.0, 1.0)),
            "levels": pe.linspace_levels(0.01, 0.5, 7, spacing="log"),
            "x_label": "RMS contrast",
        },
        "degradation": {
            "make": lambda: pe.suites.DegradationSuite("gaussian_noise"),
            "levels": pe.linspace_levels(0.0, 0.8, 7),
            "x_label": "noise sigma",
        },
    }


# ---------------------------------------------------------------------------- #
# Verification gate helpers.
# ---------------------------------------------------------------------------- #
def _die(msg):
    raise SystemExit(f"FATAL: {msg}")


def load_full_dataset(data_dir):
    """Load the real Imagenette val set ONCE (every image), or crash. No fallback."""
    root = Path(data_dir)
    if not root.is_dir():
        _die(f"dataset path is not a real directory: {root}")
    # max_per_class=None -> take every file in every class folder.
    ds = pe.datasets.imagenet_subset(str(root), max_per_class=None, image_size=IMAGE_SIZE)
    n = len(ds.images)
    if n == 0:
        _die("loaded 0 images")
    if ds.item_ids is None or len(ds.item_ids) != n:
        _die("dataset did not carry one stable item_id (file path) per image")
    if len(set(ds.item_ids)) != n:
        _die("duplicate item_ids — image files are not distinct")
    return ds, root


def assert_pretrained_not_random(clf, ds, n_probe=40):
    """Prove the weights are real pretrained ImageNet weights, not random init, by a REAL
    forward pass: a pretrained net classifies clean Imagenette far above the 1/1000 chance a
    randomly-initialised net would sit at. This is a behavioural check on actual logits."""
    idx = np.arange(min(n_probe, len(ds.images)))
    correct = 0
    for j in idx:
        logits = np.asarray(clf(ds.images[j]), dtype=float).ravel()
        if int(np.argmax(logits)) == int(ds.labels[j]):
            correct += 1
    acc = correct / len(idx)
    chance = 1.0 / ds.num_classes
    print(f"    [{clf.__name__}] clean top-1 on {len(idx)} real imgs = {acc:.3f} "
          f"(chance={chance:.4f})", flush=True)
    if acc < 0.5:
        _die(f"{clf.__name__} clean accuracy {acc:.3f} is not far above chance "
             f"{chance:.4f} — weights look like random init, NOT pretrained. Aborting.")
    return acc


def split_indices(n_items, split_seed):
    """Seeded random split of item indices into disjoint, exhaustive halves A and B.

    The ONLY permitted np.random beyond the fitter's bootstrap. No class stratification
    beyond what a uniform random partition gives."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_items)
    half = n_items // 2
    idx_A = np.sort(perm[:half])
    idx_B = np.sort(perm[half:])
    # Hard gate: disjoint AND exhaustive.
    sA, sB = set(idx_A.tolist()), set(idx_B.tolist())
    if sA & sB:
        _die(f"split seed {split_seed}: halves overlap ({len(sA & sB)} shared indices)")
    if sA | sB != set(range(n_items)):
        _die(f"split seed {split_seed}: halves are not exhaustive")
    return idx_A, idx_B


def make_half(full_ds, idx):
    """Build a half Dataset from item indices — same public Dataset type, no re-load."""
    return pe.datasets.Dataset(
        images=full_ds.images[idx],
        labels=full_ds.labels[idx],
        num_classes=full_ds.num_classes,
        name=f"{full_ds.name}[half:{len(idx)}]",
        metadata=dict(full_ds.metadata),
        item_ids=tuple(full_ds.item_ids[int(i)] for i in idx),
    )


def class_counts(labels):
    vals, cnts = np.unique(np.asarray(labels), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


# ---------------------------------------------------------------------------- #
# Measurement + bootstrap.
# ---------------------------------------------------------------------------- #
def measure_half(clf, model_name, suite_spec, half_ds, n_boot, boot_seed):
    """Run one identical sweep on one half, fit, and return threshold + bootstrap CI.

    n_trials per level MUST equal the half's item count (method of constant stimuli)."""
    t0 = time.time()
    res = pe.measure(model=clf, suite=suite_spec["make"](), dataset=half_ds,
                     levels=suite_spec["levels"], seed=SWEEP_SEED, model_name=model_name)
    dt = time.time() - t0
    bundle = res.bundle  # single condition
    n_expected = len(half_ds.images)
    if any(int(nt) != n_expected for nt in bundle.n_trials):
        _die(f"n_trials {bundle.n_trials} != half item count {n_expected}")

    fit = res.fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        thr = float(fit.threshold(TARGET))
        ci = fit.bootstrap_ci("threshold", target=TARGET, n_boot=n_boot, ci=0.95,
                              seed=boot_seed)
    return {
        "threshold": thr,
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "ci_fail_fraction": float(ci.fail_fraction),
        "converged": bool(fit.converged),
        "at_bound": list(fit.at_bound),
        "n_trials": int(n_expected),
        "n_correct": [int(c) for c in bundle.n_correct],
        "config_hash": bundle.config_hash,
        "seconds": dt,
        "_fit": fit,
    }


def paired_split_half_diff_ci(fit_A, fit_B, n_boot, seed):
    """Bootstrap the SPLIT-HALF difference (thr_A - thr_B) directly.

    A and B are disjoint, independent image sets, so we resample each half's counts from its
    own fitted curve (parametric bootstrap, Wichmann & Hill), refit, and take the difference
    distribution. CI excluding 0 => the two halves RELIABLY disagree (the failure direction)."""
    rng_A = np.random.default_rng(seed)
    rng_B = np.random.default_rng(seed + 991)  # independent stream

    def phat(fit):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.clip(np.asarray(fit.predict(fit.levels), dtype=float), 0.0, 1.0)

    nt_A = np.rint(fit_A.n_trials).astype(np.int64)
    nt_B = np.rint(fit_B.n_trials).astype(np.int64)
    p_A, p_B = phat(fit_A), phat(fit_B)

    diffs, n_failed = [], 0
    for _ in range(n_boot):
        k_A = rng_A.binomial(nt_A, p_A).astype(float)
        k_B = rng_B.binomial(nt_B, p_B).astype(float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                t_A = float(fit_A._refit(k_A).threshold(TARGET))
                t_B = float(fit_B._refit(k_B).threshold(TARGET))
            d = t_A - t_B
        except Exception:
            d = float("nan")
        if np.isfinite(d):
            diffs.append(d)
        else:
            n_failed += 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        point = float(fit_A.threshold(TARGET) - fit_B.threshold(TARGET))
    if not diffs:
        return {"diff": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "excludes_zero": False, "fail_fraction": 1.0}
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "fail_fraction": n_failed / n_boot}


def cis_overlap(a_lo, a_hi, b_lo, b_hi):
    if any(not np.isfinite(v) for v in (a_lo, a_hi, b_lo, b_hi)):
        return None
    return not (a_hi < b_lo or b_hi < a_lo)


# ---------------------------------------------------------------------------- #
# Reporting.
# ---------------------------------------------------------------------------- #
def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}g}"


def write_csv(path, rows):
    cols = ["model", "suite", "split_seed", "thr_A", "ci_A_low", "ci_A_high",
            "thr_B", "ci_B_low", "ci_B_high", "delta", "abs_delta", "cis_overlap",
            "paired_diff", "paired_ci_low", "paired_ci_high", "paired_excludes_zero",
            "gallery_gap", "abs_delta_over_gap", "n_trials_A", "n_trials_B",
            "converged_A", "converged_B", "seconds_A", "seconds_B"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_plot(path, rows, suites):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {MODEL_NAMES[0]: "#1f77b4", MODEL_NAMES[1]: "#d62728"}
    keys = list(suites.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 5.6))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        sub = [r for r in rows if r["suite"] == key]
        finite = []
        for r in sub:
            if np.isfinite(r["thr_A"]) and np.isfinite(r["thr_B"]):
                finite.append(r)
            c = colors.get(r["model"], "#555555")
            xerr = [[max(0.0, r["thr_A"] - r["ci_A_low"])], [max(0.0, r["ci_A_high"] - r["thr_A"])]]
            yerr = [[max(0.0, r["thr_B"] - r["ci_B_low"])], [max(0.0, r["ci_B_high"] - r["thr_B"])]]
            ax.errorbar(r["thr_A"], r["thr_B"], xerr=xerr, yerr=yerr, fmt="o", ms=7,
                        color=c, ecolor=c, elinewidth=1.1, capsize=2.5, alpha=0.85, zorder=3)
        # identity line spanning the data
        allv = [v for r in finite for v in (r["thr_A"], r["thr_B"],
                r["ci_A_low"], r["ci_A_high"], r["ci_B_low"], r["ci_B_high"])
                if np.isfinite(v)]
        if allv:
            lo, hi = min(allv), max(allv)
            pad = 0.08 * (hi - lo or 1.0)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=1.0, zorder=1,
                    label="identity (thr_A = thr_B)")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xlabel(f"threshold — half A ({suites[key]['x_label']})")
        ax.set_ylabel(f"threshold — half B ({suites[key]['x_label']})")
        ax.set_title(f"{key}: split-half thresholds (95% CI)")
        ax.set_aspect("equal", adjustable="box")
        # legend: models + identity
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=colors[m], label=m)
                   for m in MODEL_NAMES]
        handles.append(plt.Line2D([0], [0], ls="--", color="#888888",
                                  label="identity (thr_A = thr_B)"))
        ax.legend(handles=handles, loc="best", fontsize=8)
    fig.suptitle("Test A — threshold split-half reliability (real Imagenette val)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _suite_verdict(overlap, med_ratio, max_ratio, rank_ok, rank_total, n_seeds):
    """Per-suite PASS/FAIL/AMBIGUOUS.

    The threshold's *direction* (ranking) and its *magnitude* (|Δ| vs the between-model gap)
    are judged separately, because a threshold can be a robust rank signal while its precise
    value is item-sampling-fragile — the degradation case. Rules:

      FAIL       ranking reverses (not a model property), OR median split-half |Δ| ≥ 50% of the
                 between-model gap (typical within-model noise rivals the effect).
      PASS       ranking never reverses AND max |Δ| < 1/3 of the gap AND CIs overlap in ≥ 70%
                 of comparisons (magnitude comfortably above split-half noise).
      AMBIGUOUS  ranking robust but magnitude fragile (worst-case noise 1/3–1/2 of the effect,
                 or overlap dips) — directionally reliable, magnitude not certifiable at this n.
    """
    rank_reversed = rank_ok < rank_total
    if rank_reversed:
        return "FAIL", (f"ranking reverses on {rank_total - rank_ok}/{rank_total} half×seed "
                        f"cells — the threshold is not a model property here")
    if np.isfinite(med_ratio) and med_ratio >= 0.5:
        return "FAIL", (f"median split-half |Δ| is {med_ratio:.0%} of the between-model gap: "
                        f"typical within-model noise rivals the effect")
    if (np.isfinite(max_ratio) and max_ratio < 1 / 3 and np.isfinite(overlap)
            and overlap >= 0.7):
        return "PASS", (f"ranking held {rank_ok}/{rank_total}; worst split-half |Δ| is only "
                        f"{max_ratio:.0%} of the between-model gap; CIs overlap in "
                        f"{overlap:.0%} of comparisons")
    return "AMBIGUOUS", (f"ranking held {rank_ok}/{rank_total} (direction robust), but split-half "
                         f"|Δ| reaches {max_ratio:.0%} of the between-model gap (median "
                         f"{med_ratio:.0%}): the magnitude is item-sampling-fragile at this n")


def compute_verdicts(rows, ranking):
    """Four verdicts + a per-suite and overall PASS/FAIL/AMBIGUOUS call."""
    n = len(rows)
    overlaps = [r["cis_overlap"] for r in rows if r["cis_overlap"] is not None]
    overlap_rate = (sum(1 for o in overlaps if o) / len(overlaps)) if overlaps else float("nan")

    ratio_by_suite = {}
    for key, gap in GALLERY_GAP.items():
        rs = [r["abs_delta"] / gap for r in rows
              if r["suite"] == key and np.isfinite(r["abs_delta"])]
        if rs:
            ratio_by_suite[key] = {"median": float(np.median(rs)), "max": float(np.max(rs)),
                                   "gap": gap, "n": len(rs)}

    rank_total = len(ranking)
    rank_ok = sum(1 for e in ranking if e["holds"] is True)
    rank_bad = [e for e in ranking if e["holds"] is False]
    rank_na = [e for e in ranking if e["holds"] is None]

    excl = [r["paired_excludes_zero"] for r in rows
            if np.isfinite(r["paired_ci_low"]) and np.isfinite(r["paired_ci_high"])]
    excl_rate = (sum(1 for e in excl if e) / len(excl)) if excl else float("nan")

    # Per-suite verdicts.
    n_seeds = len({r["split_seed"] for r in rows})
    per_suite = {}
    for key in ratio_by_suite:
        srows = [r for r in rows if r["suite"] == key]
        sov = [r["cis_overlap"] for r in srows if r["cis_overlap"] is not None]
        s_overlap = (sum(1 for o in sov if o) / len(sov)) if sov else float("nan")
        srank = [e for e in ranking if e["suite"] == key]
        s_rank_ok = sum(1 for e in srank if e["holds"] is True)
        s_excl = [r for r in srows
                  if np.isfinite(r["paired_ci_low"]) and r["paired_excludes_zero"]]
        verdict, reason = _suite_verdict(
            s_overlap, ratio_by_suite[key]["median"], ratio_by_suite[key]["max"],
            s_rank_ok, len(srank), n_seeds)
        per_suite[key] = {
            "verdict": verdict, "reason": reason, "overlap_rate": s_overlap,
            "median_ratio": ratio_by_suite[key]["median"], "max_ratio": ratio_by_suite[key]["max"],
            "rank_ok": s_rank_ok, "rank_total": len(srank),
            "paired_excludes_zero": len(s_excl), "n": len(srows),
        }

    return {
        "n_comparisons": n, "n_seeds": n_seeds,
        "overlap_rate": overlap_rate, "n_overlap_eval": len(overlaps),
        "ratio_by_suite": ratio_by_suite,
        "ranking_total": rank_total, "ranking_ok": rank_ok,
        "ranking_reversals": rank_bad, "ranking_na": rank_na,
        "paired_excludes_zero_rate": excl_rate, "n_paired_eval": len(excl),
        "per_suite": per_suite,
    }


def call_pass_fail(v):
    """Overall call composed from the per-suite verdicts. Explicit, un-spun, per the brief:
    a failure/fragility on any suite is stated plainly, never averaged away."""
    ps = v["per_suite"]
    verdicts = {k: s["verdict"] for k, s in ps.items()}

    # Under-powered guard: with < 12 comparisons we cannot certify a suite PASS at all.
    if v["n_comparisons"] < 12:
        reasons = [f"only {v['n_comparisons']} comparison(s) — under-powered "
                   f"(a smoke run, not the full 20-comparison design)"]
        reasons += [f"{k}: overlap {s['overlap_rate']:.0%}, max |Δ|/gap {s['max_ratio']:.0%}, "
                    f"ranking {s['rank_ok']}/{s['rank_total']}" for k, s in ps.items()]
        reasons.append("run the full 5-seed design (20 comparisons) to certify")
        return "AMBIGUOUS", reasons

    per_lines = [f"**{k}: {s['verdict']}** — {s['reason']}" for k, s in ps.items()]

    if any(x == "FAIL" for x in verdicts.values()):
        failed = [k for k, x in verdicts.items() if x == "FAIL"]
        return ("FAIL (" + ", ".join(failed) + ")",
                [f"split-half reliability fails for: {', '.join(failed)}. The gallery's "
                 f"non-overlapping CIs there are partly an item-sampling artifact and the "
                 f"threshold claim needs requalifying."] + per_lines)
    if all(x == "PASS" for x in verdicts.values()):
        return "PASS", ["threshold is a reproducible model property on every suite tested"] + per_lines
    # Mix of PASS and AMBIGUOUS.
    passed = [k for k, x in verdicts.items() if x == "PASS"]
    amb = [k for k, x in verdicts.items() if x == "AMBIGUOUS"]
    return ("MIXED — PASS (" + ", ".join(passed) + ") / AMBIGUOUS (" + ", ".join(amb) + ")",
            [f"the model ranking is robust on all suites (ranking held "
             f"{v['ranking_ok']}/{v['ranking_total']}); the *magnitude* is a reproducible "
             f"property for {', '.join(passed)} but item-sampling-fragile for {', '.join(amb)}."]
            + per_lines)


def write_md(path, rows, verdicts, call, provenance, suites, n_items, half_sizes, class_bal):
    v = verdicts
    L = []
    L.append("# Test A — threshold split-half reliability")
    L.append("")
    L.append("**Kill-criterion check, not a feature.** Is a measured threshold a property of "
             "the model, or an artifact of which images we swept? Method: randomly split the "
             "real Imagenette val set into two disjoint halves, run the *identical* sweep on "
             "each, compare thresholds + bootstrap CIs, over several split seeds.")
    L.append("")
    L.append(f"Real Imagenette val · **{n_items} images** · halves ≈ "
             f"{half_sizes[0]}/{half_sizes[1]} · models: {', '.join(MODEL_NAMES)} · "
             f"weights: timm pretrained ImageNet-1k · split seeds: "
             f"{', '.join(str(s) for s in provenance['split_seeds'])} · "
             f"bootstrap n={provenance['n_boot']}.")
    L.append("")
    L.append("Every number below traces to a real forward pass of a real pretrained model on "
             "a real image file. No synthetic data, no fallbacks.")
    L.append("")

    # PASS/FAIL up top, not buried.
    L.append(f"## VERDICT: {call[0]}")
    L.append("")
    for r in call[1]:
        L.append(f"- {r}")
    L.append("")
    L.append("_Not spun. This test exists to potentially kill a claim; the call above is "
             "reported as the numbers land, per-suite, without averaging a fragility away._")
    L.append("")
    # "What n would resolve it" for any AMBIGUOUS suite (required by the brief).
    amb = [k for k, s in v["per_suite"].items() if s["verdict"] == "AMBIGUOUS"]
    if amb and v["n_comparisons"] >= 12:
        L.append(f"**What would resolve the AMBIGUOUS suite(s) ({', '.join(amb)}).** Split-half "
                 f"|Δ| shrinks ~1/√n_half, so tightening worst-case noise from ~{max(v['per_suite'][k]['max_ratio'] for k in amb):.0%} "
                 f"of the gap down to a comfortable <⅓ needs roughly **4× the images per half** "
                 f"(≈8k/half, ≈16k total — a larger draw than Imagenette's {n_items}, e.g. the "
                 f"full ImageNet-1k val). Part of the fragility is intrinsic: the degradation "
                 f"between-model gap ({GALLERY_GAP['degradation']:.3g}) is genuinely small, so at "
                 f"any feasible n it should be reported *with* this caveat, not as a crisp "
                 f"threshold separation.")
        L.append("")

    # Per-comparison table.
    L.append("## Per-comparison table")
    L.append("")
    L.append("| model | suite | seed | thr_A | CI_A | thr_B | CI_B | Δ | CIs overlap? | "
             "Δ CI excludes 0? |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ovl = {True: "yes", False: "**no**", None: "n/a"}[r["cis_overlap"]]
        excl = "**yes**" if r["paired_excludes_zero"] else "no"
        L.append(
            f"| {r['model']} | {r['suite']} | {r['split_seed']} | {_fmt(r['thr_A'])} | "
            f"[{_fmt(r['ci_A_low'])}, {_fmt(r['ci_A_high'])}] | {_fmt(r['thr_B'])} | "
            f"[{_fmt(r['ci_B_low'])}, {_fmt(r['ci_B_high'])}] | {_fmt(r['delta'])} | "
            f"{ovl} | {excl} |")
    L.append("")

    # Verdicts.
    L.append("## Verdicts")
    L.append("")
    L.append(f"**1. Overlap rate (headline).** The two halves' 95% CIs overlap in "
             f"**{v['overlap_rate']:.0%}** of the {v['n_overlap_eval']} evaluable comparisons "
             f"(of {v['n_comparisons']} total). "
             + ("High overlap ⇒ a threshold measured on one random half is statistically "
                "consistent with the other half." if np.isfinite(v['overlap_rate']) else ""))
    L.append("")
    L.append("**2. Split-half |Δ| vs. the real between-model effect.** The gallery's "
             "between-model gap is the effect the library claims to detect "
             f"(contrast {GALLERY_GAP['contrast']:.3g}, degradation {GALLERY_GAP['degradation']:.3g}). "
             "Split-half |Δ| as a fraction of that gap:")
    L.append("")
    L.append("| suite | between-model gap | median \\|Δ\\| / gap | max \\|Δ\\| / gap |")
    L.append("|---|---|---|---|")
    for key, s in v["ratio_by_suite"].items():
        L.append(f"| {key} | {s['gap']:.3g} | {s['median']:.0%} | {s['max']:.0%} |")
    L.append("")
    L.append("_If within-model split-half noise is comparable to the between-model effect "
             "(ratio → 1), the effect is not measurable at this n — that is the kill._")
    L.append("")
    L.append(f"**3. Ranking stability.** Does resnet50 < vit_small (lower threshold) hold on "
             f"every half, every seed? Held on **{v['ranking_ok']} / {v['ranking_total']}** "
             f"half×seed×suite cells.")
    if v["ranking_reversals"]:
        L.append("")
        L.append("Reversals (a threshold that flips the ranking on a random half is not a "
                 "model property):")
        for e in v["ranking_reversals"]:
            L.append(f"- {e['suite']} seed {e['seed']} half {e['half']}: "
                     f"resnet50={_fmt(e['resnet'])} vs vit_small={_fmt(e['vit'])}")
    if v["ranking_na"]:
        L.append("")
        L.append(f"({len(v['ranking_na'])} cell(s) not evaluable — a NaN/extrapolated "
                 "threshold on one model.)")
    L.append("")
    L.append(f"**4. Paired Δ CI.** Bootstrapping the split-half difference (thr_A − thr_B) "
             f"directly, the CI **excludes 0** in **{v['paired_excludes_zero_rate']:.0%}** of "
             f"{v['n_paired_eval']} evaluable comparisons. Excluding 0 means the two halves "
             f"reliably differ — but read it against verdict 2, not on its own. At n≈1962 the "
             f"parametric bootstrap is tight enough to resolve a *hair*: a split-half Δ can be "
             f"statistically non-zero while still being a few percent of the between-model gap, "
             f"with the two halves' CIs overlapping and the ranking intact. That is not a kill. "
             f"The kill is when \\|Δ\\| rivals the effect (verdict 2), the CIs stop overlapping "
             f"(verdict 1), or the ranking flips (verdict 3).")
    L.append("")

    # Limitations.
    L.append("## Limitations")
    L.append("")
    L.append("- **Distractor suite excluded.** Splitting the val set changes the distractor-"
             "pool composition, a second confound on top of item sampling. Only contrast and "
             "degradation are tested here.")
    L.append("- **Single dataset, 10 classes.** Imagenette val only; a random split preserves "
             "rough class balance but is not stratified. Per-half class counts are in "
             "`provenance.json`.")
    L.append("- **Deterministic models.** No trial-level noise; all CI width comes from the "
             "parametric bootstrap over the fitted binomial, and the split-half Δ from which "
             "images fell in each half — exactly the artifact this test probes.")
    L.append("- The between-model gaps used as the yardstick come from the standing gallery "
             "(200 images), a different n than these halves (~1962).")
    L.append("")
    L.append(f"_Generated {provenance['timestamp']} · psyvis_ml {provenance['versions']['psyvis_ml']} "
             f"· torch {provenance['versions']['torch']} · timm {provenance['versions']['timm']} "
             f"· device {provenance.get('device', 'unknown')} · "
             f"wall-clock {provenance.get('wall_clock_seconds', 0) / 60:.1f} min._")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def load_rows_from_csv(path):
    """Reconstruct the per-comparison rows + ranking from a written results.csv (for report
    regeneration without re-running the sweeps). Types mirror what the run builds in memory."""
    import csv

    def fnum(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return float("nan")

    def fbool(s):
        return str(s).strip().lower() == "true"

    rows = []
    with open(path, encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            ov = d.get("cis_overlap", "")
            rows.append({
                "model": d["model"], "suite": d["suite"], "split_seed": int(d["split_seed"]),
                "thr_A": fnum(d["thr_A"]), "ci_A_low": fnum(d["ci_A_low"]),
                "ci_A_high": fnum(d["ci_A_high"]), "thr_B": fnum(d["thr_B"]),
                "ci_B_low": fnum(d["ci_B_low"]), "ci_B_high": fnum(d["ci_B_high"]),
                "delta": fnum(d["delta"]), "abs_delta": fnum(d["abs_delta"]),
                "cis_overlap": (None if ov == "" or ov.lower() == "none" else fbool(ov)),
                "paired_diff": fnum(d["paired_diff"]), "paired_ci_low": fnum(d["paired_ci_low"]),
                "paired_ci_high": fnum(d["paired_ci_high"]),
                "paired_excludes_zero": fbool(d["paired_excludes_zero"]),
                "gallery_gap": fnum(d["gallery_gap"]),
                "abs_delta_over_gap": fnum(d["abs_delta_over_gap"]),
                "n_trials_A": int(fnum(d["n_trials_A"])), "n_trials_B": int(fnum(d["n_trials_B"])),
                "converged_A": fbool(d["converged_A"]), "converged_B": fbool(d["converged_B"]),
            })
    # Reconstruct ranking: resnet50 < vit_small on each (suite, seed, half).
    ranking = []
    rn, vt = MODEL_NAMES[0], MODEL_NAMES[1]
    by = {(r["suite"], r["split_seed"], r["model"]): r for r in rows}
    seeds = sorted({r["split_seed"] for r in rows})
    suites_seen = sorted({r["suite"] for r in rows})
    for key in suites_seen:
        for seed in seeds:
            r_rn, r_vt = by.get((key, seed, rn)), by.get((key, seed, vt))
            if not (r_rn and r_vt):
                continue
            for half in ("A", "B"):
                t_rn, t_vt = r_rn[f"thr_{half}"], r_vt[f"thr_{half}"]
                holds = (t_rn < t_vt) if (np.isfinite(t_rn) and np.isfinite(t_vt)) else None
                ranking.append({"suite": key, "seed": seed, "half": half,
                                "resnet": t_rn, "vit": t_vt, "holds": holds})
    return rows, ranking


def regen_report(outdir):
    """Rebuild results.md + split_half.png from an existing results.csv + provenance.json,
    recomputing the verdicts with the current logic. No sweeps re-run."""
    outdir = Path(outdir)
    rows, ranking = load_rows_from_csv(outdir / "results.csv")
    provenance = json.loads((outdir / "provenance.json").read_text(encoding="utf-8"))
    suites = suite_specs()
    verdicts = compute_verdicts(rows, ranking)
    call = call_pass_fail(verdicts)

    # Refresh the verdict fields in provenance (keep everything else as recorded).
    provenance["verdict"] = call[0]
    provenance["verdict_reasons"] = call[1]
    provenance["per_suite_verdict"] = {k: s["verdict"] for k, s in verdicts["per_suite"].items()}
    provenance.setdefault("device", "cuda" if provenance.get("versions", {})
                          .get("torch", "").find("cu") >= 0 else "unknown")

    n_items = provenance["n_items"]
    first_seed = provenance["split_seeds"][0]
    half_sizes0 = tuple(provenance["per_half_sizes"][str(first_seed)])
    class_bal = provenance["per_half_class_counts"][str(first_seed)]

    build_plot(outdir / "split_half.png", rows, suites)
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str),
                                            encoding="utf-8")
    write_md(outdir / "results.md", rows, verdicts, call, provenance, suites,
             n_items, half_sizes0, class_bal)
    print(f"VERDICT: {call[0]}")
    for r in call[1]:
        print(f"  - {r}")
    print(f"regenerated results.md + split_half.png + provenance.json in {outdir}")


# ---------------------------------------------------------------------------- #
# Main.
# ---------------------------------------------------------------------------- #
def main():
    # Windows consoles default to cp1252, which cannot encode Δ/σ/— used below; force UTF-8
    # so progress printing never crashes the run. (Output files are already UTF-8.)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="comma list of split seeds, e.g. '0' or '0,1,2,3,4'")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    ap.add_argument("--max-per-class", default="none",
                    help="'none' = full val set (real design); an int only for plumbing tests")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--regen", action="store_true",
                    help="rebuild results.md + split_half.png from an existing results.csv "
                         "(recompute verdicts), without re-running any sweeps")
    args = ap.parse_args()

    if args.regen:
        regen_report(Path(args.out))
        return

    split_seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    mpc = None if str(args.max_per_class).lower() == "none" else int(args.max_per_class)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("Test A — threshold split-half reliability (REAL DATA ONLY)", flush=True)
    print("=" * 78, flush=True)
    print(f"split seeds       : {split_seeds}", flush=True)
    print(f"bootstrap n / seed: {args.n_boot} / {args.boot_seed}", flush=True)
    print(f"max_per_class     : {'FULL val set' if mpc is None else mpc}", flush=True)
    print(f"output dir        : {outdir}", flush=True)

    # ---- verification gate: dataset ------------------------------------------ #
    print("\n[gate] loading real Imagenette val set once ...", flush=True)
    t_load = time.time()
    if mpc is None:
        full_ds, root = load_full_dataset(args.data_dir)
    else:  # plumbing-test path — still real images, just fewer
        root = Path(args.data_dir)
        if not root.is_dir():
            _die(f"dataset path is not a real directory: {root}")
        full_ds = pe.datasets.imagenet_subset(str(root), max_per_class=mpc,
                                              image_size=IMAGE_SIZE)
    n_items = len(full_ds.images)
    cbal_full = class_counts(full_ds.labels)
    print(f"[gate] path exists & is a directory: {root}", flush=True)
    print(f"[gate] real image files loaded (opened from disk): {n_items}", flush=True)
    print(f"[gate] classes: {len(cbal_full)}  per-class counts: {cbal_full}", flush=True)
    print(f"[gate] load took {time.time() - t_load:.1f}s", flush=True)

    # ---- verification gate: models pretrained -------------------------------- #
    print("\n[gate] loading models (timm, pretrained=True, frozen, eval) ...", flush=True)
    clfs = {m: pe.models.timm_classifier(m, pretrained=True) for m in MODEL_NAMES}
    clean_acc = {}
    for m in MODEL_NAMES:
        clean_acc[m] = assert_pretrained_not_random(clfs[m], full_ds)

    suites = suite_specs()
    for key, s in suites.items():
        print(f"[gate] suite {key}: levels = {np.asarray(s['levels'])}", flush=True)

    # ---- run ----------------------------------------------------------------- #
    rows = []            # one per (model, suite, seed): half-A vs half-B comparison
    ranking = []         # one per (suite, seed, half): resnet50 < vit_small ?
    config_hashes = {}
    per_half_sizes = {}
    per_half_class = {}
    total_sweeps = len(split_seeds) * 2 * len(MODEL_NAMES) * len(suites)  # seeds*halves*models*suites
    done = 0
    t_start = time.time()

    for seed in split_seeds:
        print(f"\n{'#' * 60}\n# split seed {seed}\n{'#' * 60}", flush=True)
        idx_A, idx_B = split_indices(n_items, seed)
        half_A, half_B = make_half(full_ds, idx_A), make_half(full_ds, idx_B)
        per_half_sizes[seed] = (len(idx_A), len(idx_B))
        cA, cB = class_counts(half_A.labels), class_counts(half_B.labels)
        per_half_class[seed] = {"A": cA, "B": cB}
        # Gate: disjoint (by construction, re-checked in split_indices) + distinct files.
        idsA, idsB = set(half_A.item_ids), set(half_B.item_ids)
        if idsA & idsB:
            _die(f"seed {seed}: halves share {len(idsA & idsB)} image files")
        if len(idsA) != len(idx_A) or len(idsB) != len(idx_B):
            _die(f"seed {seed}: item_id count != index count (non-distinct files)")
        print(f"[gate] half A: {len(idx_A)} imgs  half B: {len(idx_B)} imgs  "
              f"(disjoint, union = {len(idsA | idsB)} = n_items {n_items})", flush=True)
        print(f"[gate] class balance A: {cA}", flush=True)
        print(f"[gate] class balance B: {cB}", flush=True)

        # measure every model on both halves for both suites
        # store: measured[(suite, model, half)] = result dict (with _fit)
        measured = {}
        for key in suites:
            for model_name in MODEL_NAMES:
                for half_name, half_ds in (("A", half_A), ("B", half_B)):
                    done_frac = done + 1
                    print(f"\n  -> sweep {done_frac}/{total_sweeps}: {key} / {model_name} / "
                          f"half {half_name} ({len(half_ds.images)} imgs) ...", flush=True)
                    r = measure_half(clfs[model_name], model_name, suites[key], half_ds,
                                     args.n_boot, args.boot_seed)
                    measured[(key, model_name, half_name)] = r
                    config_hashes[f"{key}|{model_name}|s{seed}|{half_name}"] = r["config_hash"]
                    done += 1
                    elapsed = time.time() - t_start
                    eta = (elapsed / done) * (total_sweeps - done)
                    print(f"     thr={_fmt(r['threshold'])} "
                          f"CI=[{_fmt(r['ci_low'])},{_fmt(r['ci_high'])}] "
                          f"converged={r['converged']} at_bound={r['at_bound'] or '—'} "
                          f"({r['seconds']:.1f}s) | elapsed {elapsed/60:.1f}m ETA {eta/60:.1f}m",
                          flush=True)

        # build comparison rows + paired diff + ranking
        for key in suites:
            for model_name in MODEL_NAMES:
                rA = measured[(key, model_name, "A")]
                rB = measured[(key, model_name, "B")]
                paired = paired_split_half_diff_ci(rA["_fit"], rB["_fit"],
                                                   args.n_boot, args.boot_seed)
                delta = (rA["threshold"] - rB["threshold"]
                         if np.isfinite(rA["threshold"]) and np.isfinite(rB["threshold"])
                         else float("nan"))
                ovl = cis_overlap(rA["ci_low"], rA["ci_high"], rB["ci_low"], rB["ci_high"])
                gap = GALLERY_GAP[key]
                rows.append({
                    "model": model_name, "suite": key, "split_seed": seed,
                    "thr_A": rA["threshold"], "ci_A_low": rA["ci_low"], "ci_A_high": rA["ci_high"],
                    "thr_B": rB["threshold"], "ci_B_low": rB["ci_low"], "ci_B_high": rB["ci_high"],
                    "delta": delta, "abs_delta": abs(delta) if np.isfinite(delta) else float("nan"),
                    "cis_overlap": ovl,
                    "paired_diff": paired["diff"], "paired_ci_low": paired["ci_low"],
                    "paired_ci_high": paired["ci_high"],
                    "paired_excludes_zero": paired["excludes_zero"],
                    "gallery_gap": gap,
                    "abs_delta_over_gap": (abs(delta) / gap) if np.isfinite(delta) else float("nan"),
                    "n_trials_A": rA["n_trials"], "n_trials_B": rB["n_trials"],
                    "converged_A": rA["converged"], "converged_B": rB["converged"],
                    "seconds_A": round(rA["seconds"], 1), "seconds_B": round(rB["seconds"], 1),
                })

            # ranking: resnet50 < vit_small on each half
            rn, vt = MODEL_NAMES[0], MODEL_NAMES[1]
            for half_name in ("A", "B"):
                t_rn = measured[(key, rn, half_name)]["threshold"]
                t_vt = measured[(key, vt, half_name)]["threshold"]
                holds = (t_rn < t_vt) if (np.isfinite(t_rn) and np.isfinite(t_vt)) else None
                ranking.append({"suite": key, "seed": seed, "half": half_name,
                                "resnet": t_rn, "vit": t_vt, "holds": holds})

    # ---- verdicts + outputs -------------------------------------------------- #
    verdicts = compute_verdicts(rows, ranking)
    call = call_pass_fail(verdicts)

    # class balance summary for the first seed (representative)
    first_seed = split_seeds[0]
    half_sizes0 = per_half_sizes[first_seed]

    provenance = {
        "test": "A — threshold split-half reliability",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(root),
        "n_items": n_items,
        "max_per_class": ("full" if mpc is None else mpc),
        "per_half_sizes": {str(k): list(v) for k, v in per_half_sizes.items()},
        "per_half_class_counts": {str(k): v for k, v in per_half_class.items()},
        "full_class_counts": cbal_full,
        "split_seeds": split_seeds,
        "sweep_seed": SWEEP_SEED,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "target": TARGET,
        "models": MODEL_NAMES,
        "weight_source": "timm create_model(pretrained=True) — ImageNet-1k",
        "device": ("cuda:" + __import__("torch").cuda.get_device_name(0)
                   if __import__("torch").cuda.is_available() else "cpu"),
        "model_clean_top1": clean_acc,
        "levels": {k: [float(x) for x in np.asarray(s["levels"])]
                   for k, s in suites.items()},
        "image_size": IMAGE_SIZE,
        "gallery_reference_gap": GALLERY_GAP,
        "gallery_reference_thresholds": GALLERY_THR,
        "versions": {
            "psyvis_ml": pe.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
            "timm": __import__("timm").__version__,
            "scipy": __import__("scipy").__version__,
        },
        "config_hashes": config_hashes,
        "verdict": call[0],
        "verdict_reasons": call[1],
        "per_suite_verdict": {k: s["verdict"] for k, s in verdicts["per_suite"].items()},
        "wall_clock_seconds": round(time.time() - t_start, 1),
    }

    write_csv(outdir / "results.csv", rows)
    build_plot(outdir / "split_half.png", rows, suites)
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str),
                                            encoding="utf-8")
    write_md(outdir / "results.md", rows, verdicts, call, provenance, suites,
             n_items, half_sizes0, per_half_class[first_seed])

    print("\n" + "=" * 78, flush=True)
    print(f"VERDICT: {call[0]}", flush=True)
    for r in call[1]:
        print(f"  - {r}", flush=True)
    print(f"\noverlap rate: {verdicts['overlap_rate']:.0%} "
          f"({verdicts['n_overlap_eval']} evaluable) | "
          f"ranking held {verdicts['ranking_ok']}/{verdicts['ranking_total']} | "
          f"paired-Δ excludes-0 rate {verdicts['paired_excludes_zero_rate']:.0%}", flush=True)
    print(f"total wall-clock: {(time.time() - t_start)/60:.1f} min "
          f"({len(rows)} comparisons, {total_sweeps} sweeps)", flush=True)
    print(f"wrote: results.csv, results.md, split_half.png, provenance.json -> {outdir}",
          flush=True)


if __name__ == "__main__":
    main()
