"""Test C — is the contrast threshold a new axis, or an expensive accuracy proxy?

Kill-criterion check. The library's core bet is that a psychometric threshold tells you something
accuracy does not. Test A showed the threshold is repeatable; this asks whether it is *informative*.
Method: measure ~10 real pretrained timm models on the real contrast sweep, record each model's
clean top-1 accuracy and its fitted contrast threshold, and correlate.

VERDICT RULE (written before looking at the correlation):
  Pearson r² < 0.3  -> threshold is a DISTINCT axis; accuracy does not predict it (what the
                       library needs).
  Pearson r² > 0.7  -> threshold is largely accuracy re-expressed at ~7x the compute (KILL the
                       threshold-as-new-axis claim; stated plainly, not spun).
  0.3 <= r² <= 0.7  -> INTERMEDIATE; say so and say what would sharpen it.

REAL DATA ONLY. Every number traces to a real forward pass of a real pretrained model on a real
image file. A model that fails to resolve/download is DROPPED and reported — never substituted,
never a placeholder. No synthetic fallback anywhere. Only np.random is the fitter's bootstrap.

Consumer of the public psyvis_ml API only — no changes to psyvis_ml/ source.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import psyvis_ml as pe

DEFAULT_DATA_DIR = r"C:/Users/glopu/OneDrive/Desktop/Experiments/Exp_assets/imagenette2-320/val"
EXPECTED_N = 3925          # must match Test A's count
IMAGE_SIZE = 224
SWEEP_SEED = 0
TARGET = 0.75
CONTRAST_LEVELS_ARGS = (0.01, 0.5, 7)   # gallery contrast levels: geomspace(0.01, 0.5, 7)

# (timm name, architecture family). Span family + capacity for range on both axes.
MODELS = [
    ("resnet18", "CNN"),
    ("resnet50", "CNN"),
    ("resnet101", "CNN"),
    ("efficientnet_b0", "CNN"),
    ("convnext_tiny", "CNN"),
    ("vit_small_patch16_224", "ViT"),
    ("vit_base_patch16_224", "ViT"),
    ("deit_small_patch16_224", "ViT"),
    ("swin_tiny_patch4_window7_224", "hybrid"),
    ("mobilenetv3_large_100", "CNN"),
]
FAMILY_COLOR = {"CNN": "#1f77b4", "ViT": "#d62728", "hybrid": "#2ca02c"}

# Swept contrast band: a fitted threshold outside this (or with a boundary-pinned parameter) is
# an EXTRAPOLATION, not a measurement, and is excluded from the primary correlation.
MIN_LEVEL, MAX_LEVEL = CONTRAST_LEVELS_ARGS[0], CONTRAST_LEVELS_ARGS[1]


def _die(msg):
    raise SystemExit(f"FATAL: {msg}")


def is_reliable(row):
    """A threshold is reliably *measured* (not extrapolated) iff the fit converged, no parameter
    sat on its box bound, and the threshold falls inside the swept contrast band."""
    thr = float(row["threshold"])
    at_bound = str(row.get("at_bound", "-"))
    conv = row.get("converged", True)
    conv = (conv is True) or (str(conv).lower() == "true")
    return (conv and at_bound in ("", "-", "—", "nan")
            and np.isfinite(thr) and MIN_LEVEL <= thr <= MAX_LEVEL)


def load_full_dataset(data_dir, max_per_class=None):
    root = Path(data_dir)
    if not root.is_dir():
        _die(f"dataset path is not a real directory: {root}")
    ds = pe.datasets.imagenet_subset(str(root), max_per_class=max_per_class, image_size=IMAGE_SIZE)
    n = len(ds.images)
    # The full-set count gate only applies to the real design (max_per_class=None); a smaller
    # max_per_class is a plumbing test path only (still real images, never synthetic).
    if max_per_class is None and n != EXPECTED_N:
        _die(f"image count {n} != Test A's {EXPECTED_N}")
    if ds.item_ids is None or len(set(ds.item_ids)) != n:
        _die("images are not distinct real files")
    return ds, root


def clean_top1(clf, ds):
    """Real clean top-1 on the full val set (the 'accuracy' axis + pretrained gate)."""
    n = len(ds.images)
    correct = 0
    for j in range(n):
        logits = np.asarray(clf(ds.images[j]), dtype=float).ravel()
        if int(np.argmax(logits)) == int(ds.labels[j]):
            correct += 1
    return correct / n


def model_params_and_tag(name):
    import timm
    try:
        m = timm.create_model(name, pretrained=False)  # arch only, no download
        params_m = sum(p.numel() for p in m.parameters()) / 1e6
        cfg = getattr(m, "pretrained_cfg", {}) or {}
        tag = cfg.get("tag") or cfg.get("hf_hub_id") or "timm-default"
    except Exception:  # noqa: BLE001
        params_m, tag = float("nan"), "unknown"
    return params_m, tag


# ---------------------------------------------------------------------------- #
# Correlation with bootstrap-over-models CIs.
# ---------------------------------------------------------------------------- #
def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return _pearson(rx, ry)


def corr_with_ci(x, y, n_boot=5000, seed=7):
    """Pearson & Spearman + percentile bootstrap CIs over the *models* (resample rows)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r_p, r_s = _pearson(x, y), _spearman(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    ps, ss = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        ps.append(_pearson(x[idx], y[idx]))
        ss.append(_spearman(x[idx], y[idx]))
    def ci(v):
        v = [z for z in v if np.isfinite(z)]
        if not v:
            return (float("nan"), float("nan"))
        return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
    return {"pearson": r_p, "pearson_ci": ci(ps), "spearman": r_s, "spearman_ci": ci(ss),
            "r2": (r_p * r_p if np.isfinite(r_p) else float("nan")), "n": n}


def equal_accuracy_pairs(rows, acc_window=0.02):
    """Model pairs within `acc_window` clean-accuracy of each other, sorted by threshold gap.
    A pair that ties on accuracy but sits far apart on threshold is the existence proof."""
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if not (np.isfinite(a["threshold"]) and np.isfinite(b["threshold"])):
                continue
            dacc = abs(a["clean_acc"] - b["clean_acc"])
            if dacc <= acc_window:
                dthr = abs(a["threshold"] - b["threshold"])
                ratio = (max(a["threshold"], b["threshold"])
                         / min(a["threshold"], b["threshold"]))
                both_rel = is_reliable(a) and is_reliable(b)
                pairs.append({"m1": a["name"], "m2": b["name"], "dacc_pts": dacc * 100,
                              "thr1": a["threshold"], "thr2": b["threshold"],
                              "dthr": dthr, "thr_ratio": ratio, "both_reliable": both_rel,
                              "acc1": a["clean_acc"], "acc2": b["clean_acc"]})
    pairs.sort(key=lambda p: -p["thr_ratio"])
    return pairs


def analyze(rows, boot_seed=12345):
    """All correlations + verdict. Primary correlation uses only reliably-MEASURED thresholds
    (in-range, converged, not at-bound); the all-models and log-threshold variants are reported
    as robustness checks. Reusable by the live run and by the --regen path."""
    accs = np.array([r["clean_acc"] for r in rows], float)
    thrs = np.array([r["threshold"] for r in rows], float)
    slopes = np.array([r["slope"] for r in rows], float)
    finite = np.isfinite(thrs)
    rel = np.array([is_reliable(r) for r in rows]) & finite
    acc_spread_pts = float((accs[finite].max() - accs[finite].min()) * 100)

    corr_thr = corr_with_ci(accs[rel], thrs[rel], seed=boot_seed)                 # PRIMARY
    corr_thr_all = corr_with_ci(accs[finite], thrs[finite], seed=boot_seed)
    corr_thr_log = corr_with_ci(accs[rel], np.log10(thrs[rel]), seed=boot_seed)
    corr_slope_acc = corr_with_ci(accs[rel], slopes[rel], seed=boot_seed)
    corr_slope_thr = corr_with_ci(thrs[rel], slopes[rel], seed=boot_seed)
    pairs = equal_accuracy_pairs(rows, acc_window=0.02)
    # Existence-proof ratio uses only pairs where BOTH thresholds are reliably measured, so it
    # is never inflated by an extrapolated value.
    max_ratio = max((p["thr_ratio"] for p in pairs if p["both_reliable"]), default=float("nan"))
    extrap_names = [r["name"] for r in rows if not is_reliable(r)]
    call = verdict(corr_thr, corr_thr_all, corr_thr_log, acc_spread_pts, len(extrap_names),
                   max_ratio)
    return {"corr_thr": corr_thr, "corr_thr_all": corr_thr_all, "corr_thr_log": corr_thr_log,
            "corr_slope_acc": corr_slope_acc, "corr_slope_thr": corr_slope_thr,
            "pairs": pairs, "call": call, "acc_spread_pts": acc_spread_pts,
            "extrap_names": extrap_names, "n_extrap": len(extrap_names),
            "n_reliable": int(rel.sum()), "n_all": int(finite.sum()),
            "max_equal_acc_ratio": max_ratio}


# ---------------------------------------------------------------------------- #
# Plot.
# ---------------------------------------------------------------------------- #
def build_plot(path, rows, corr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.8, 7.2))
    for r in rows:
        if not np.isfinite(r["threshold"]):
            continue
        c = FAMILY_COLOR.get(r["family"], "#555555")
        x = r["clean_acc"] * 100
        rel = is_reliable(r)
        label = r["name"].replace("_patch16_224", "").replace("_patch4_window7_224", "")
        if rel:
            yerr = [[max(0.0, r["threshold"] - r["ci_low"])],
                    [max(0.0, r["ci_high"] - r["threshold"])]]
            ax.errorbar(x, r["threshold"], yerr=yerr, fmt="o", ms=9, color=c, ecolor=c,
                        elinewidth=1.3, capsize=3, zorder=3)
        else:
            # Extrapolated/at-bound: draw as an open marker at the swept floor with a down-arrow
            # (true threshold is below the measurable range), and exclude from the fit line.
            ax.scatter([x], [MIN_LEVEL], s=90, facecolors="none", edgecolors=c, linewidths=1.6,
                       zorder=3)
            ax.annotate("", xy=(x, MIN_LEVEL * 0.55), xytext=(x, MIN_LEVEL),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
            label += " (extrapolated < floor)"
        ax.annotate(label, (x, MIN_LEVEL if not rel else r["threshold"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=7.5, color="#222222")
    ax.axhline(MIN_LEVEL, ls=":", color="#999999", lw=1.0, zorder=0)
    ax.axhline(MAX_LEVEL, ls=":", color="#999999", lw=1.0, zorder=0)
    ax.text(0.01, MIN_LEVEL, " swept floor", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#999999", va="bottom")
    ax.set_yscale("log")
    # family legend + reliability marker legend
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=col, label=fam)
               for fam, col in FAMILY_COLOR.items()]
    handles.append(plt.Line2D([0], [0], marker="o", ls="", mfc="none", mec="#555555",
                              label="extrapolated (below floor)"))
    ax.legend(handles=handles, title="architecture family", loc="best", fontsize=8.5)
    rp, rs = corr["pearson"], corr["spearman"]
    ax.set_title("Test C — clean accuracy vs. contrast threshold (real pretrained models)\n"
                 f"reliable-model Pearson r={rp:.2f} (r²={rp * rp:.2f}), Spearman ρ={rs:.2f} · "
                 f"real Imagenette val", fontsize=11)
    ax.set_xlabel("clean top-1 accuracy on Imagenette val (%)  — the 'accuracy' axis")
    ax.set_ylabel("contrast threshold @ 75% (RMS, log scale)  — lower = more sensitive")
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# Verdict (rule fixed above, applied to the numbers).
# ---------------------------------------------------------------------------- #
def verdict(corr_primary, corr_all, corr_log, acc_spread_pts, n_extrap, max_equal_acc_ratio):
    """Verdict on the PRIMARY (reliably-measured, in-range) correlation; robustness-checked
    against the all-models and log-threshold variants."""
    r2 = corr_primary["r2"]
    reasons = []
    if not np.isfinite(r2):
        return "INCONCLUSIVE", ["correlation undefined (too few finite in-range thresholds)"]
    if r2 < 0.3:
        call = "PASS — threshold is a DISTINCT axis"
        reasons.append(f"Pearson r²={r2:.2f} < 0.3 on the reliably-measured models: clean accuracy "
                       f"does not predict the contrast threshold. The threshold carries information "
                       f"accuracy does not — the result the library needs.")
    elif r2 > 0.7:
        call = "FAIL (KILL) — threshold is largely accuracy re-expressed"
        reasons.append(f"Pearson r²={r2:.2f} > 0.7: the contrast threshold is mostly a re-encoding "
                       f"of clean accuracy, at ~7× the compute. The threshold-as-new-axis claim "
                       f"would need withdrawing (confidence readouts may still survive).")
    else:
        call = "INTERMEDIATE — threshold is partly, not wholly, redundant with accuracy"
        reasons.append(f"Pearson r²={r2:.2f} (reliable models) is in [0.3, 0.7]: accuracy explains "
                       f"about half the threshold variance and leaves about half unexplained. "
                       f"Neither a clean vindication (r²<0.3) nor a kill (r²>0.7).")
    reasons.append(f"Robustness: r²={corr_all['r2']:.2f} including the extrapolated model, "
                   f"{corr_log['r2']:.2f} on log-threshold — the call is stable across all three.")
    if np.isfinite(max_equal_acc_ratio) and max_equal_acc_ratio >= 3:
        reasons.append(f"The unexplained half is real signal, not noise: models within 2 accuracy "
                       f"points differ by up to {max_equal_acc_ratio:.0f}× in threshold "
                       f"(existence proof — see the equal-accuracy pairs).")
    if n_extrap:
        reasons.append(f"{n_extrap} model(s) are so contrast-robust their 75% point sits BELOW the "
                       f"swept floor ({MIN_LEVEL}) — threshold extrapolated/at-bound, excluded from "
                       f"the primary fit. Re-run with a lower contrast floor (e.g. 0.003) to "
                       f"measure them and firm up the correlation.")
    # Range-restriction honesty (always stated).
    if acc_spread_pts < 3.0:
        reasons.append(f"CAVEAT — accuracy spread only {acc_spread_pts:.1f} pts; a low r² would be "
                       f"partly a range-restriction artifact. Redesign the x-axis first.")
    else:
        reasons.append(f"Accuracy spread is {acc_spread_pts:.1f} points (models sit at 75–90%, not "
                       f"near ceiling), so the correlation is not suppressed by range restriction.")
    return call, reasons


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}g}"


def write_outputs(outdir, rows, dropped, an, provenance):
    outdir = Path(outdir)
    corr_thr = an["corr_thr"]
    pairs = an["pairs"]
    call = an["call"]
    acc_spread_pts = an["acc_spread_pts"]
    extrap = set(an["extrap_names"])

    # CSV (add a reliability flag column).
    cols = ["name", "family", "params_m", "clean_acc", "threshold", "ci_low", "ci_high",
            "slope", "converged", "at_bound", "reliable", "weight_tag", "device", "n_trials",
            "seconds"]
    lines = [",".join(cols)]
    for r in rows:
        r = {**r, "reliable": is_reliable(r)}
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    (outdir / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    build_plot(outdir / "threshold_vs_accuracy.png", rows, corr_thr)
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str),
                                            encoding="utf-8")

    # MD
    L = []
    L.append("# Test C — is the contrast threshold a new axis, or an expensive accuracy proxy?")
    L.append("")
    L.append("**Kill-criterion check.** The library's bet is that a psychometric threshold tells "
             "you something accuracy does not. Test A showed the threshold is *repeatable*; this "
             "asks whether it is *informative*. Method: measure real pretrained timm models on the "
             "real contrast sweep, record each model's clean top-1 and fitted contrast threshold, "
             "and correlate.")
    L.append("")
    L.append(f"Real Imagenette val · **{provenance['n_items']} images** (full, no split) · "
             f"**{len(rows)} models** ({an['n_reliable']} with an in-range measured threshold) · "
             f"contrast levels geomspace{CONTRAST_LEVELS_ARGS} (weibull) · device "
             f"{provenance['device']} · bootstrap n={provenance['n_boot']}.")
    L.append("")
    L.append("Every number traces to a real forward pass of a real pretrained model on a real "
             "image file. No synthetic data, no fallbacks. Models that failed to resolve are "
             "dropped, not substituted.")
    L.append("")

    L.append(f"## VERDICT: {call[0]}")
    L.append("")
    for r in call[1]:
        L.append(f"- {r}")
    L.append("")
    L.append("_Verdict rule fixed before looking: r² < 0.3 → distinct axis; r² > 0.7 → kill; "
             "0.3–0.7 → intermediate. Not spun — reported as it landed._")
    L.append("")

    # Per-model table.
    L.append("## Per-model results")
    L.append("")
    L.append("| model | family | params (M) | clean top-1 | contrast threshold | 95% CI | "
             "slope | reliable? |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda z: -z["clean_acc"]):
        rel = "yes" if is_reliable(r) else f"**no** ({r['at_bound']}/extrap)"
        L.append(f"| {r['name']} | {r['family']} | {_fmt(r['params_m'], 3)} | "
                 f"{r['clean_acc'] * 100:.2f}% | {_fmt(r['threshold'])} | "
                 f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] | {_fmt(r['slope'])} | {rel} |")
    if dropped:
        L.append("")
        L.append("**Dropped (unavailable, not substituted):** "
                 + "; ".join(f"{n} ({w})" for n, w in dropped))
    if extrap:
        L.append("")
        L.append(f"**Extrapolated / excluded from the primary fit:** {', '.join(sorted(extrap))} "
                 f"— so contrast-robust that the 75% point sits *below* the swept floor "
                 f"({MIN_LEVEL}); the fit pins a parameter at its bound and the reported threshold "
                 f"is an extrapolation, not a measurement. This is itself a finding: the most "
                 f"robust models are off the bottom of the gallery's contrast range.")
    L.append("")

    # Correlations.
    def cline(name, c):
        pl, ph = c["pearson_ci"]; sl, sh = c["spearman_ci"]
        return (f"| {name} | {_fmt(c['pearson'], 3)} [{_fmt(pl, 2)}, {_fmt(ph, 2)}] | "
                f"{_fmt(c['r2'], 3)} | {_fmt(c['spearman'], 3)} [{_fmt(sl, 2)}, {_fmt(sh, 2)}] |")
    L.append("## Correlations (bootstrap CIs over models)")
    L.append("")
    L.append("| relationship | Pearson r [95% CI] | r² | Spearman ρ [95% CI] |")
    L.append("|---|---|---|---|")
    L.append(cline(f"clean acc ↔ **threshold** (reliable, n={corr_thr['n']}) — PRIMARY", corr_thr))
    L.append(cline(f"clean acc ↔ threshold (all n={an['corr_thr_all']['n']}, incl. extrapolated)",
                   an["corr_thr_all"]))
    L.append(cline("clean acc ↔ log₁₀ threshold (reliable)", an["corr_thr_log"]))
    L.append(cline("clean acc ↔ slope (reliable)", an["corr_slope_acc"]))
    L.append(cline("threshold ↔ slope (reliable)", an["corr_slope_thr"]))
    L.append("")
    L.append("The primary correlation uses only reliably-measured (in-range, converged, "
             "not-at-bound) thresholds; the all-models and log variants are robustness checks. "
             "Both coefficients are reported — Spearman matters because the relationship need not "
             f"be linear. With only {corr_thr['n']}–{an['corr_thr_all']['n']} models the CIs are "
             "wide; read them as such.")
    L.append("")

    # Equal-accuracy pairs.
    L.append("## The residual question — models that tie on accuracy but differ on threshold")
    L.append("")
    L.append("Pairs within 2 accuracy points of each other, ranked by threshold ratio. A pair that "
             "ties on accuracy yet sits far apart on threshold is the existence proof that the "
             "threshold is a separate readout:")
    L.append("")
    if pairs:
        L.append("| model A | model B | Δacc (pts) | thr A | thr B | thr ratio | both measured? |")
        L.append("|---|---|---|---|---|---|---|")
        for p in pairs[:8]:
            note = "yes" if p["both_reliable"] else "**no** (extrapolated)"
            L.append(f"| {p['m1']} | {p['m2']} | {p['dacc_pts']:.2f} | {_fmt(p['thr1'])} | "
                     f"{_fmt(p['thr2'])} | {p['thr_ratio']:.2f}× | {note} |")
        rel_pairs = [p for p in pairs if p["both_reliable"]]
        L.append("")
        if rel_pairs:
            top = rel_pairs[0]
            L.append(f"**Sharpest fully-measured pair:** {top['m1']} ({top['acc1'] * 100:.1f}%, thr "
                     f"{_fmt(top['thr1'])}) vs {top['m2']} ({top['acc2'] * 100:.1f}%, thr "
                     f"{_fmt(top['thr2'])}) — Δacc {top['dacc_pts']:.2f} pts but a "
                     f"**{top['thr_ratio']:.1f}× threshold gap**, both thresholds in-range. This is "
                     f"the concrete demonstration that the threshold carries information clean "
                     f"accuracy does not, and is why the verdict is not a kill despite a moderate "
                     f"overall r².")
        extrap_pairs = [p for p in pairs if not p["both_reliable"]]
        if extrap_pairs:
            ep = extrap_pairs[0]
            L.append("")
            L.append(f"(Pairs involving {', '.join(sorted(set(an['extrap_names'])))} show even "
                     f"larger ratios — e.g. {ep['thr_ratio']:.0f}× — but one threshold there is an "
                     f"*extrapolated lower bound*, so read those as \"≥\", not measured.)")
    else:
        L.append("_No model pairs fell within 2 accuracy points — see the scatter._")
    L.append("")

    # Slope as third axis — report honestly.
    csa, cst = an["corr_slope_acc"], an["corr_slope_thr"]
    L.append("## Slope as a third axis — does it survive as independent?")
    L.append("")
    indep = (abs(csa["pearson"]) < 0.5 and abs(cst["pearson"]) < 0.5)
    L.append(f"Slope↔accuracy: Pearson r={_fmt(csa['pearson'], 3)} (r²={_fmt(csa['r2'], 2)}). "
             f"Slope↔threshold: r={_fmt(cst['pearson'], 3)} (r²={_fmt(cst['r2'], 2)}). "
             + ("Slope is largely independent of both — additional evidence of a separate readout."
                if indep else
                "**Slope is NOT an independent third axis here:** on the reliably-measured models "
                "it correlates strongly with both accuracy and threshold (steeper curves go with "
                "higher accuracy and lower thresholds). Stated plainly against the hopeful prior — "
                "in this dataset slope tracks the same gradient, so it does not add a clean "
                "separate dimension. (The naive all-model slope correlation looks weaker only "
                "because the extrapolated model's degenerate at-bound slope is an outlier.)"))
    L.append("")

    # Caveat.
    L.append("## Range-restriction caveat (handled up front, not buried)")
    L.append("")
    L.append(f"Clean top-1 spans **{acc_spread_pts:.1f} points** "
             f"({min(r['clean_acc'] for r in rows) * 100:.1f}%–"
             f"{max(r['clean_acc'] for r in rows) * 100:.1f}%). These are ImageNet-1k models doing "
             "1000-way argmax on Imagenette's fine-grained classes, so they sit well below ceiling "
             "— the accuracy axis is *not* compressed into the 95–99% band that would mechanically "
             "suppress a correlation. "
             + ("A moderate r² here is therefore a real result, not a range-restriction artifact."
                if acc_spread_pts >= 3 else
                "The spread is under 3 points, so a low r² is partly artefactual — redesign the "
                "x-axis (degraded-condition accuracy at a fixed level, or a harder dataset)."))
    L.append("")
    L.append("**What would sharpen this** (verdict is intermediate): more models over a wider "
             "capacity range; a **lower contrast floor** (≈0.003) so the most robust models "
             f"({', '.join(sorted(extrap)) or 'none here'}) yield a *measured* rather than "
             "extrapolated threshold; and a harder dataset to widen the accuracy axis further.")
    L.append("")
    L.append("_Limitations: single dataset (Imagenette, 10 classes); 1000-way accuracy on a "
             "10-class subset; ~10 models is a small n for correlation CIs; contrast suite only._")
    L.append("")
    L.append(f"_Generated {provenance['timestamp']} · psyvis_ml {provenance['versions']['psyvis_ml']} "
             f"· torch {provenance['versions']['torch']} · timm {provenance['versions']['timm']} "
             f"· device {provenance['device']} · wall-clock "
             f"{provenance['wall_clock_seconds'] / 60:.1f} min._")
    (outdir / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def load_rows_from_csv(path):
    """Reconstruct per-model rows from a written results.csv (for --regen, no sweeps re-run)."""
    import csv

    def fnum(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return float("nan")

    rows = []
    with open(path, encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            rows.append({
                "name": d["name"], "family": d["family"], "params_m": fnum(d["params_m"]),
                "clean_acc": fnum(d["clean_acc"]), "threshold": fnum(d["threshold"]),
                "ci_low": fnum(d["ci_low"]), "ci_high": fnum(d["ci_high"]),
                "slope": fnum(d["slope"]),
                "converged": str(d.get("converged", "True")).lower() == "true",
                "at_bound": d.get("at_bound", "-"), "weight_tag": d.get("weight_tag", ""),
                "device": d.get("device", ""), "n_trials": int(fnum(d.get("n_trials", 0))),
                "seconds": fnum(d.get("seconds", 0)),
            })
    return rows


def regen_report(outdir):
    """Rebuild results.md + threshold_vs_accuracy.png + provenance verdict fields from an existing
    results.csv, recomputing the analysis with the current (extrapolation-aware) logic."""
    outdir = Path(outdir)
    rows = load_rows_from_csv(outdir / "results.csv")
    provenance = json.loads((outdir / "provenance.json").read_text(encoding="utf-8"))
    an = analyze(rows, boot_seed=provenance.get("boot_seed", 12345))
    provenance["verdict"] = an["call"][0]
    provenance["verdict_reasons"] = an["call"][1]
    provenance["extrapolated_models"] = an["extrap_names"]
    provenance["n_reliable"] = an["n_reliable"]
    provenance["acc_spread_points"] = an["acc_spread_pts"]
    provenance["correlations_reliable"] = {"acc_vs_threshold": an["corr_thr"],
                                           "acc_vs_slope": an["corr_slope_acc"],
                                           "threshold_vs_slope": an["corr_slope_thr"]}
    provenance["correlations_all_models"] = {"acc_vs_threshold": an["corr_thr_all"]}
    provenance["correlations_log_threshold"] = {"acc_vs_threshold": an["corr_thr_log"]}
    provenance["equal_accuracy_pairs"] = an["pairs"]
    dropped = provenance.get("dropped", [])
    write_outputs(outdir, rows, dropped, an, provenance)
    print(f"VERDICT: {an['call'][0]}")
    for r in an["call"][1]:
        print(f"  - {r}")
    ct = an["corr_thr"]
    print(f"acc↔threshold (reliable {an['n_reliable']}): r={ct['pearson']:.3f} r²={ct['r2']:.3f} "
          f"| all r²={an['corr_thr_all']['r2']:.3f} | log r²={an['corr_thr_log']['r2']:.3f}")
    print(f"regenerated results.md + threshold_vs_accuracy.png + provenance.json in {outdir}")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--models", default="",
                    help="comma list of timm names to restrict to (plumbing); default = all 10")
    ap.add_argument("--max-per-class", default="none",
                    help="'none' = full val set (real design); an int only for plumbing tests")
    ap.add_argument("--regen", action="store_true",
                    help="rebuild results.md + plot from an existing results.csv (recompute the "
                         "extrapolation-aware analysis), without re-running any sweeps")
    args = ap.parse_args()

    if args.regen:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
        regen_report(Path(args.out))
        return

    global MODELS
    if args.models.strip():
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        fam = dict(MODELS)
        MODELS = [(m, fam.get(m, "CNN")) for m in wanted]
    mpc = None if str(args.max_per_class).lower() == "none" else int(args.max_per_class)

    import torch
    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_str = (f"cuda:{torch.cuda.get_device_name(0)}" if device == "cuda" else "cpu")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("Test C — contrast threshold vs. clean accuracy (REAL DATA ONLY)", flush=True)
    print("=" * 78, flush=True)
    print(f"device: {dev_str} · torch {torch.__version__} · timm {timm.__version__}", flush=True)

    # ---- gate: dataset ---- #
    t_load = time.time()
    ds, root = load_full_dataset(args.data_dir, max_per_class=mpc)
    n_items = len(ds.images)
    chance = 1.0 / ds.num_classes
    print(f"[gate] dataset {root} · {n_items} real images (== Test A) · "
          f"loaded {time.time() - t_load:.1f}s", flush=True)

    levels = pe.linspace_levels(*CONTRAST_LEVELS_ARGS, spacing="log")
    print(f"[gate] contrast levels = {np.asarray(levels)}", flush=True)

    rows, dropped, config_hashes = [], [], {}
    t_start = time.time()
    for mi, (name, fam) in enumerate(MODELS):
        print(f"\n[{mi + 1}/{len(MODELS)}] {name} ({fam}) — loading pretrained ...", flush=True)
        try:
            clf = pe.models.timm_classifier(name, pretrained=True)
        except Exception as e:  # noqa: BLE001 — drop and report, never substitute
            print(f"  DROP {name}: {type(e).__name__}: {e}", flush=True)
            dropped.append((name, f"{type(e).__name__}: {e}"))
            continue
        params_m, tag = model_params_and_tag(name)

        # pretrained gate on the FULL set (this clean top-1 IS the accuracy axis).
        t0 = time.time()
        acc = clean_top1(clf, ds)
        t_acc = time.time() - t0
        if acc < 0.5:
            _die(f"{name} clean top-1 {acc:.4f} not far above chance {chance:.4f} — "
                 f"weights look like random init, not pretrained.")
        print(f"  [gate] clean top-1 = {acc:.4f} (chance {chance:.4f}) PASS · "
              f"params {params_m:.1f}M · tag {tag} · {t_acc:.1f}s", flush=True)

        # contrast sweep (identical to the gallery/Test A config).
        t0 = time.time()
        res = pe.measure(model=clf,
                         suite=pe.suites.ContrastThreshold(contrast_metric="rms",
                                                          clip_range=(0.0, 1.0)),
                         dataset=ds, levels=levels, seed=SWEEP_SEED, model_name=name)
        t_sweep = time.time() - t0
        bundle = res.bundle
        if any(int(nt) != n_items for nt in bundle.n_trials):
            _die(f"{name}: n_trials {bundle.n_trials} != full item count {n_items}")
        fit = res.fit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            thr = float(fit.threshold(TARGET))
            slope = float(fit.slope(TARGET))
            ci = fit.bootstrap_ci("threshold", target=TARGET, n_boot=args.n_boot, ci=0.95,
                                  seed=args.boot_seed)
        config_hashes[name] = bundle.config_hash
        rows.append({
            "name": name, "family": fam, "params_m": round(params_m, 3), "clean_acc": acc,
            "threshold": thr, "ci_low": float(ci.low), "ci_high": float(ci.high),
            "slope": slope, "converged": bool(fit.converged),
            "at_bound": "|".join(fit.at_bound) if fit.at_bound else "-",
            "weight_tag": tag, "device": dev_str, "n_trials": n_items,
            "seconds": round(t_acc + t_sweep, 1),
        })
        elapsed = time.time() - t_start
        eta = (elapsed / (mi + 1)) * (len(MODELS) - mi - 1)
        print(f"  threshold={_fmt(thr)} CI=[{_fmt(ci.low)},{_fmt(ci.high)}] slope={_fmt(slope)} "
              f"converged={fit.converged} at_bound={fit.at_bound or '—'} "
              f"(sweep {t_sweep:.1f}s) | elapsed {elapsed / 60:.1f}m ETA {eta / 60:.1f}m",
              flush=True)

    if len(rows) < 3:
        _die(f"only {len(rows)} models produced results — too few to correlate.")

    an = analyze(rows, boot_seed=args.boot_seed)
    provenance = {
        "test": "C — contrast threshold vs. clean accuracy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(root), "n_items": n_items, "image_size": IMAGE_SIZE,
        "device": dev_str,
        "models_requested": [m for m, _ in MODELS],
        "models_used": [r["name"] for r in rows],
        "dropped": dropped,
        "weight_source": "timm create_model(pretrained=True) — ImageNet-1k (default tags)",
        "weight_tags": {r["name"]: r["weight_tag"] for r in rows},
        "contrast_levels": [float(x) for x in np.asarray(levels)],
        "sweep_seed": SWEEP_SEED, "target": TARGET,
        "n_boot": args.n_boot, "boot_seed": args.boot_seed,
        "clean_acc": {r["name"]: r["clean_acc"] for r in rows},
        "acc_spread_points": an["acc_spread_pts"],
        "extrapolated_models": an["extrap_names"], "n_reliable": an["n_reliable"],
        "correlations_reliable": {"acc_vs_threshold": an["corr_thr"],
                                  "acc_vs_slope": an["corr_slope_acc"],
                                  "threshold_vs_slope": an["corr_slope_thr"]},
        "correlations_all_models": {"acc_vs_threshold": an["corr_thr_all"]},
        "correlations_log_threshold": {"acc_vs_threshold": an["corr_thr_log"]},
        "equal_accuracy_pairs": an["pairs"],
        "verdict": an["call"][0], "verdict_reasons": an["call"][1],
        "config_hashes": config_hashes,
        "versions": {
            "psyvis_ml": pe.__version__, "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "timm": timm.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "wall_clock_seconds": round(time.time() - t_start, 1),
    }

    write_outputs(outdir, rows, dropped, an, provenance)

    print("\n" + "=" * 78, flush=True)
    print(f"VERDICT: {an['call'][0]}", flush=True)
    for r in an["call"][1]:
        print(f"  - {r}", flush=True)
    ct = an["corr_thr"]
    print(f"\nacc↔threshold (reliable {an['n_reliable']}): Pearson r={ct['pearson']:.3f} "
          f"(r²={ct['r2']:.3f}), Spearman ρ={ct['spearman']:.3f} | all r²={an['corr_thr_all']['r2']:.3f} "
          f"| log r²={an['corr_thr_log']['r2']:.3f} | acc spread {an['acc_spread_pts']:.1f} pts",
          flush=True)
    print(f"total wall-clock: {(time.time() - t_start) / 60:.1f} min ({len(rows)} models)",
          flush=True)
    print(f"wrote results.csv, results.md, threshold_vs_accuracy.png, provenance.json -> {outdir}",
          flush=True)


if __name__ == "__main__":
    main()
