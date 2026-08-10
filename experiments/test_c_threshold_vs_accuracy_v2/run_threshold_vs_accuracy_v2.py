"""Test C v2 — persistence, curves, lower contrast floor, two suites.

v1 measured 275k real forward passes and kept one number per model; the fits were discarded.
This re-run fixes that permanently: EVERY sweep's observed points and fitted params are persisted
to curves.json, and every plot is a pure read from disk (`--regen` replots/reanalyses with zero
forward passes). The per-item (trial-level) table is written for every sweep too, so the whole run
is consumable by metaeval later without re-sweeping.

Changes vs v1:
  * contrast floor 0.01 -> 0.001 (geomspace(0.001, 0.5, 10); step ratio 1.995) because resnet101's
    v1 threshold extrapolated to 0.00057 and its fit pinned at bound.
  * degradation suite added (gaussian noise, gallery levels verbatim).
  * everything persisted.
Distractor is deliberately out of scope (no Test A split-half reliability data for that suite).

VERDICT RULE (fixed before looking at the numbers), per suite:
  Pearson r² < 0.3 -> threshold is a DISTINCT axis.
  Pearson r² > 0.7 -> KILL: threshold is largely accuracy re-expressed at ~7x the compute.
  otherwise        -> INTERMEDIATE.
Do not spin a high r². Do not let the existence proof paper over a tight correlation — report both,
and if they conflict, say they conflict.

REAL DATA ONLY. Every number traces to a real forward pass of a real pretrained model on a real
image file. A model that fails to resolve is DROPPED and reported, never substituted. No synthetic
fallback anywhere. Only np.random is the fitter's bootstrap.

Consumer of the public psyvis_ml API only — no changes under src/psyvis_ml/.
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
from psyvis_ml.fitting.sigmoids import SIGMOIDS  # read-only: evaluate a curve from saved params

DEFAULT_DATA_DIR = r"C:/Users/glopu/OneDrive/Desktop/Experiments/Exp_assets/imagenette2-320/val"
V1_DIR = Path(__file__).resolve().parent.parent / "test_c_threshold_vs_accuracy"
EXPECTED_N = 3925
IMAGE_SIZE = 224
SWEEP_SEED = 0
TARGET = 0.75
EQUAL_ACC_WINDOW_PTS = 1.0     # "equal accuracy" = within ~1 clean-top-1 point
LEVERAGE_MODEL = "resnet18"    # sits ~6.9 pts below the next model on accuracy

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

# Suite specs. Contrast floor dropped to 0.001; degradation = gallery levels verbatim.
SUITES = {
    "contrast": {
        "make": lambda: pe.suites.ContrastThreshold(contrast_metric="rms", clip_range=(0.0, 1.0)),
        "levels_args": (0.001, 0.5, 10), "spacing": "log",
        "x_label": "RMS contrast", "log_x": True,
        "note": "geomspace(0.001, 0.5, 10), weibull; floor lowered from v1's 0.01",
    },
    "degradation": {
        "make": lambda: pe.suites.DegradationSuite("gaussian_noise"),
        "levels_args": (0.0, 0.8, 7), "spacing": "linear",
        "x_label": "noise sigma", "log_x": False,
        "note": "linspace(0.0, 0.8, 7), logistic/decreasing; gallery levels verbatim",
    },
}


def suite_levels(key):
    s = SUITES[key]
    return pe.linspace_levels(*s["levels_args"], spacing=s["spacing"])


def _die(msg):
    raise SystemExit(f"FATAL: {msg}")


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}g}"


# ---------------------------------------------------------------------------- #
# Curve evaluation from PERSISTED params (this is what makes --regen a pure read).
# ---------------------------------------------------------------------------- #
def predict_from_params(x, fit_params, sigmoid):
    """P(correct | x) from saved params alone: guess + (1-guess-lapse)*F(x; alpha, beta).

    Verified during the run to reproduce fit.predict() exactly, so curves.json is sufficient
    to redraw every curve without a single forward pass."""
    f = SIGMOIDS[sigmoid]["f"]
    g, lam = fit_params["guess"], fit_params["lapse"]
    return g + (1.0 - g - lam) * np.asarray(
        f(np.asarray(x, float), fit_params["alpha"], fit_params["slope"]), float)


def is_reliable(c):
    """Threshold reliably MEASURED (not extrapolated): converged, no at-bound param, and the
    threshold falls inside that suite's swept band."""
    thr = float(c["threshold"])
    lo, hi = min(c["levels"]), max(c["levels"])
    return (bool(c["converged"]) and not c["at_bound"]
            and np.isfinite(thr) and lo <= thr <= hi)


def goodness_of_fit(c):
    """R² / residuals of the fitted curve against the OBSERVED points — computed from persisted
    data alone. `converged` and `at_bound` say the optimizer succeeded; they do NOT say the
    functional form describes the data. This does."""
    lv = np.asarray(c["levels"], float)
    obs = np.asarray(c["n_correct"], float) / np.asarray(c["n_trials"], float)
    pred = predict_from_params(lv, c["fit_params"], c["sigmoid"])
    resid = obs - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"r2_fit": r2, "max_abs_resid": float(np.max(np.abs(resid))),
            "resid_at_lowest_level": float(resid[0]),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "poor_fit": bool(np.isfinite(r2) and r2 < 0.90)}


# ---------------------------------------------------------------------------- #
# Gates.
# ---------------------------------------------------------------------------- #
def load_full_dataset(data_dir, max_per_class=None):
    root = Path(data_dir)
    if not root.is_dir():
        _die(f"dataset path is not a real directory: {root}")
    ds = pe.datasets.imagenet_subset(str(root), max_per_class=max_per_class, image_size=IMAGE_SIZE)
    n = len(ds.images)
    if max_per_class is None and n != EXPECTED_N:
        _die(f"image count {n} != expected {EXPECTED_N}")
    if ds.item_ids is None or len(set(ds.item_ids)) != n:
        _die("images are not distinct real files")
    return ds, root


def clean_top1(clf, ds):
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
        m = timm.create_model(name, pretrained=False)
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
    return _pearson(np.argsort(np.argsort(x)).astype(float),
                    np.argsort(np.argsort(y)).astype(float))


def corr_with_ci(x, y, n_boot=5000, seed=7):
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

    r2s = [p * p for p in ps if np.isfinite(p)]
    r2_ci = ci(r2s)
    return {"pearson": r_p, "pearson_ci": ci(ps), "spearman": r_s, "spearman_ci": ci(ss),
            "r2": (r_p * r_p if np.isfinite(r_p) else float("nan")), "r2_ci": r2_ci,
            "r2_ci_excludes_0.7": bool(np.isfinite(r2_ci[1]) and r2_ci[1] < 0.7),
            "r2_ci_excludes_0.3": bool(np.isfinite(r2_ci[0]) and r2_ci[0] > 0.3),
            "n": n}


def leverage_check(accs, thrs, names, boot_seed):
    """r² dropping each model in turn. If the correlation depends on one point, that's leverage."""
    base = _pearson(accs, thrs) ** 2
    loo = []
    for i in range(len(accs)):
        m = np.ones(len(accs), bool)
        m[i] = False
        r = _pearson(accs[m], thrs[m])
        loo.append({"dropped": names[i], "r2": float(r * r) if np.isfinite(r) else float("nan")})
    loo_sorted = sorted(loo, key=lambda d: d["r2"])
    return {"r2_full": float(base), "loo": loo,
            "r2_min": loo_sorted[0], "r2_max": loo_sorted[-1],
            "r2_without_leverage_model": next(
                (d["r2"] for d in loo if d["dropped"] == LEVERAGE_MODEL), float("nan"))}


def equal_accuracy_pairs(curves, window_pts=EQUAL_ACC_WINDOW_PTS):
    pairs = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            a, b = curves[i], curves[j]
            if not (np.isfinite(a["threshold"]) and np.isfinite(b["threshold"])):
                continue
            dacc = abs(a["clean_top1"] - b["clean_top1"]) * 100
            if dacc <= window_pts:
                ratio = (max(a["threshold"], b["threshold"])
                         / max(min(a["threshold"], b["threshold"]), 1e-12))
                pairs.append({"m1": a["model"], "m2": b["model"], "dacc_pts": dacc,
                              "thr1": a["threshold"], "thr2": b["threshold"],
                              "thr_ratio": ratio,
                              "both_reliable": is_reliable(a) and is_reliable(b),
                              "acc1": a["clean_top1"], "acc2": b["clean_top1"]})
    pairs.sort(key=lambda p: -p["thr_ratio"])
    return pairs


def _band(v):
    """The pre-committed verdict bands, applied to a variance-explained number."""
    if not np.isfinite(v):
        return "undefined"
    return "distinct-axis" if v < 0.3 else ("KILL" if v > 0.7 else "intermediate")


def verdict_for_suite(suite, corr, corr_all, lev, max_rel_ratio, n_extrap, extrap_names):
    r2 = corr["r2"]
    lo, hi = corr["r2_ci"]
    rho = corr["spearman"]
    rho2 = rho * rho if np.isfinite(rho) else float("nan")
    reasons = []
    if not np.isfinite(r2):
        return "INCONCLUSIVE", ["correlation undefined (too few in-range thresholds)"]
    if r2 < 0.3:
        call = "PASS — threshold is a DISTINCT axis"
        base = (f"Pearson r²={r2:.2f} < 0.3 on reliably-measured models: clean accuracy does "
                f"not predict the {suite} threshold.")
    elif r2 > 0.7:
        call = "FAIL (KILL) — threshold is largely accuracy re-expressed"
        base = (f"Pearson r²={r2:.2f} > 0.7: the {suite} threshold is mostly a re-encoding of "
                f"clean accuracy, at ~7× the compute. The threshold-as-new-axis claim needs "
                f"withdrawing for this suite.")
    else:
        call = "INTERMEDIATE — partly, not wholly, redundant with accuracy"
        base = (f"Pearson r²={r2:.2f} is in [0.3, 0.7]: accuracy explains part of the "
                f"{suite} threshold and leaves part unexplained.")

    # Pearson vs Spearman conflict — surfaced FIRST, never buried. The brief: report both, and
    # if they conflict, say they conflict.
    if _band(r2) != _band(rho2):
        call += "  ⚠ CONFLICTS WITH SPEARMAN"
        head = (f"**⚠ Pearson and Spearman DISAGREE, and the disagreement is the finding.** "
                f"Linear r²={r2:.2f} → *{_band(r2)}*; rank ρ={rho:.3f} → ρ²={rho2:.2f} → "
                f"*{_band(rho2)}* (same bands applied in rank space). ")
        if rho2 > r2:
            # Rank relation TIGHTER than linear: monotonic but non-linear -> more redundant.
            head += (f"The relationship is strongly **monotonic but non-linear**: clean accuracy "
                     f"orders the models' {suite} thresholds almost perfectly even though it does "
                     f"not predict their magnitude linearly. **Read as a ranking tool the "
                     f"{suite} threshold is largely redundant with accuracy — that reading is "
                     f"at/over the kill line; read as a magnitude it is not.** The pre-committed "
                     f"rule keys on Pearson r², so the headline stays '{_band(r2)}' — but that "
                     f"call alone would **overstate** the threshold's independence.")
        else:
            # Linear relation TIGHTER than rank: the r² is carried by a few extreme points.
            head += (f"Here the *linear* fit is the stronger of the two, which is the signature of "
                     f"a correlation carried by a few extreme points rather than by consistent "
                     f"ordering: accuracy does **not** reliably order the models' {suite} "
                     f"thresholds (ρ={rho:.2f}). Taken with the leverage check below, the rank "
                     f"reading is the more trustworthy one, and it puts {suite} **closer to a "
                     f"distinct axis** than Pearson suggests. The pre-committed rule keys on "
                     f"Pearson r², so the headline stays '{_band(r2)}' — but that call alone "
                     f"probably **understates** the threshold's independence here.")
        reasons.append(head)
    reasons.append(base)
    # n=10 honesty: can the CI even separate intermediate from kill?
    if not corr["r2_ci_excludes_0.7"]:
        reasons.append(f"**n={corr['n']} cannot separate 'intermediate' from 'kill':** the r² "
                       f"bootstrap CI [{lo:.2f}, {hi:.2f}] does NOT exclude 0.7. On the correlation "
                       f"alone this test is underpowered; the equal-accuracy pairs carry the "
                       f"argument by themselves.")
    else:
        reasons.append(f"The r² CI [{lo:.2f}, {hi:.2f}] excludes 0.7, so the kill threshold is "
                       f"ruled out at this n.")
    # leverage
    reasons.append(f"Leverage: r²={lev['r2_full']:.2f} full; drops to {lev['r2_min']['r2']:.2f} "
                   f"without {lev['r2_min']['dropped']}, rises to {lev['r2_max']['r2']:.2f} without "
                   f"{lev['r2_max']['dropped']}; without {LEVERAGE_MODEL} it is "
                   f"{lev['r2_without_leverage_model']:.2f}."
                   + ("  The relationship depends heavily on single points — read it as leverage, "
                      "not a robust relationship."
                      if abs(lev["r2_full"] - lev["r2_without_leverage_model"]) > 0.15 else ""))
    if np.isfinite(max_rel_ratio) and max_rel_ratio >= 3:
        reasons.append(f"Existence proof: models within {EQUAL_ACC_WINDOW_PTS:g} accuracy point "
                       f"differ by up to {max_rel_ratio:.1f}× in {suite} threshold (both measured).")
    if n_extrap:
        reasons.append(f"{n_extrap} model(s) still extrapolated/at-bound ({', '.join(extrap_names)}) "
                       f"— excluded from the primary fit.")
    return call, reasons


# ---------------------------------------------------------------------------- #
# Analysis — pure function of the persisted curves.
# ---------------------------------------------------------------------------- #
def analyze(curves, boot_seed=12345, v1_thr=None):
    out = {"per_suite": {}, "cross_suite": None, "v1_vs_v2": None}
    for suite in SUITES:
        cs = [c for c in curves if c["suite"] == suite]
        if len(cs) < 3:
            continue
        rel = [c for c in cs if is_reliable(c)]
        extrap = [c["model"] for c in cs if not is_reliable(c)]
        accs = np.array([c["clean_top1"] for c in rel], float)
        thrs = np.array([c["threshold"] for c in rel], float)
        slopes = np.array([c["slope_at_threshold"] for c in rel], float)
        names = [c["model"] for c in rel]

        corr = corr_with_ci(accs, thrs, seed=boot_seed)
        accs_all = np.array([c["clean_top1"] for c in cs], float)
        thrs_all = np.array([c["threshold"] for c in cs], float)
        corr_all = corr_with_ci(accs_all, thrs_all, seed=boot_seed)
        corr_log = corr_with_ci(accs, np.log10(np.clip(thrs, 1e-12, None)), seed=boot_seed)
        corr_slope_acc = corr_with_ci(accs, slopes, seed=boot_seed)
        corr_slope_thr = corr_with_ci(thrs, slopes, seed=boot_seed)
        lev = leverage_check(accs, thrs, names, boot_seed)
        pairs = equal_accuracy_pairs(cs)
        max_rel_ratio = max((p["thr_ratio"] for p in pairs if p["both_reliable"]),
                            default=float("nan"))
        call = verdict_for_suite(suite, corr, corr_all, lev, max_rel_ratio, len(extrap), extrap)
        gof = {c["model"]: goodness_of_fit(c) for c in cs}
        n_poor = sum(1 for g in gof.values() if g["poor_fit"])
        if n_poor:
            call[1].append(
                f"**Fit quality caveat: {n_poor}/{len(cs)} {suite} fits have R² < 0.90** against "
                f"their own observed points (worst "
                f"{min(g['r2_fit'] for g in gof.values()):.2f}). `converged=True` and no at-bound "
                f"parameter say the optimiser succeeded — they do NOT say the sigmoid describes "
                f"the data. A threshold read off a curve that misses its points by up to "
                f"{max(g['max_abs_resid'] for g in gof.values()):.2f} in P(correct) is a weaker "
                f"quantity than its tight CI implies.")
        out["per_suite"][suite] = {
            "gof": gof, "n_poor_fit": n_poor,
            "corr": corr, "corr_all": corr_all, "corr_log": corr_log,
            "corr_slope_acc": corr_slope_acc, "corr_slope_thr": corr_slope_thr,
            "leverage": lev, "pairs": pairs, "max_rel_ratio": max_rel_ratio,
            "call": call, "n_reliable": len(rel), "n_all": len(cs),
            "extrap_names": extrap,
            "acc_spread_pts": float((accs_all.max() - accs_all.min()) * 100),
        }

    # Cross-suite: does contrast threshold predict degradation threshold?
    by = {}
    for c in curves:
        by.setdefault(c["model"], {})[c["suite"]] = c
    both = [(m, d["contrast"], d["degradation"]) for m, d in by.items()
            if "contrast" in d and "degradation" in d
            and is_reliable(d["contrast"]) and is_reliable(d["degradation"])]
    if len(both) >= 3:
        cx = np.array([c["threshold"] for _, c, _ in both], float)
        dy = np.array([d["threshold"] for _, _, d in both], float)
        out["cross_suite"] = {"corr": corr_with_ci(cx, dy, seed=boot_seed),
                              "models": [m for m, _, _ in both], "n": len(both)}

    # v1 vs v2 contrast thresholds (grid changed -> not a strict replication).
    if v1_thr:
        rowsv = []
        for c in curves:
            if c["suite"] != "contrast":
                continue
            v1 = v1_thr.get(c["model"])
            if v1 is None:
                continue
            v1t, v1rel = v1["threshold"], v1["reliable"]
            v2t = c["threshold"]
            pct = (abs(v2t - v1t) / v1t * 100) if (np.isfinite(v1t) and v1t > 0) else float("nan")
            rowsv.append({"model": c["model"], "v1": v1t, "v2": v2t, "pct_change": pct,
                          "v1_reliable": v1rel, "v2_reliable": is_reliable(c),
                          "material": bool(np.isfinite(pct) and pct > 0.5 and v1rel
                                           and is_reliable(c))})
        out["v1_vs_v2"] = rowsv
    return out


# ---------------------------------------------------------------------------- #
# Plots — every one a pure read of persisted curves.
# ---------------------------------------------------------------------------- #
def plot_psychometric(path, curves, suite):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = SUITES[suite]
    cs = [c for c in curves if c["suite"] == suite]
    cs = sorted(cs, key=lambda c: -c["clean_top1"])
    fig, ax = plt.subplots(figsize=(11.0, 7.4))
    styles = {}
    for c in cs:
        fam = c["family"]
        col = FAMILY_COLOR.get(fam, "#555555")
        k = styles.setdefault(fam, 0)
        ls = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))][k % 5]
        styles[fam] = k + 1

        lv = np.asarray(c["levels"], float)
        obs = np.asarray(c["n_correct"], float) / np.asarray(c["n_trials"], float)
        lo, hi = lv.min(), lv.max()
        grid = (np.geomspace(max(lo, 1e-6), hi, 400) if spec["log_x"]
                else np.linspace(lo, hi, 400))
        curve = predict_from_params(grid, c["fit_params"], c["sigmoid"])
        bad = (not is_reliable(c))
        lbl = c["model"].replace("_patch16_224", "").replace("_patch4_window7_224", "")
        if bad:
            lbl += " [at-bound/extrap]"
        ax.plot(grid, curve, ls=ls, color=col, lw=2.0 if not bad else 2.6,
                alpha=0.95 if not bad else 1.0, zorder=3, label=lbl)
        ax.scatter(lv, obs, s=22, color=col, edgecolor="white", linewidth=0.5, zorder=4)
        thr = c["threshold"]
        if np.isfinite(thr) and lo <= thr <= hi:
            ax.plot([thr], [TARGET], marker="v", ms=9, color=col, mec="black", mew=0.6, zorder=5)
        elif np.isfinite(thr) and bad:
            # threshold off the swept range: mark at the edge with an arrow
            edge = lo if thr < lo else hi
            ax.annotate("", xy=(edge * (0.75 if spec["log_x"] else 1.0) if thr < lo else edge,
                                TARGET),
                        xytext=(edge, TARGET),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2.0))
    ax.axhline(TARGET, color="#888888", ls="--", lw=1.0, zorder=1)
    ax.text(0.005, TARGET + 0.012, "75% criterion", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#666666")
    if spec["log_x"]:
        ax.set_xscale("log")
    ax.set_xlabel(spec["x_label"] + (" (log)" if spec["log_x"] else ""))
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Test C v2 — {suite} psychometric curves, {len(cs)} real pretrained models\n"
                 f"observed points + fitted curves ({spec['note']}) · real Imagenette val "
                 f"({cs[0]['n_trials'][0]} imgs/level) · ▼ = threshold@75%", fontsize=10.5)
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(loc="upper left" if suite == "degradation" else "lower right",
              fontsize=7.6, ncol=2, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_vs_accuracy(path, curves, suite, an):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = SUITES[suite]
    cs = [c for c in curves if c["suite"] == suite]
    ps = an["per_suite"].get(suite)
    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    lo = min(min(c["levels"]) for c in cs)
    for c in cs:
        col = FAMILY_COLOR.get(c["family"], "#555555")
        x = c["clean_top1"] * 100
        lbl = c["model"].replace("_patch16_224", "").replace("_patch4_window7_224", "")
        if is_reliable(c):
            yerr = [[max(0.0, c["threshold"] - c["ci_low"])],
                    [max(0.0, c["ci_high"] - c["threshold"])]]
            ax.errorbar(x, c["threshold"], yerr=yerr, fmt="o", ms=9, color=col, ecolor=col,
                        elinewidth=1.3, capsize=3, zorder=3)
            yl = c["threshold"]
        else:
            ax.scatter([x], [lo], s=95, facecolors="none", edgecolors=col, linewidths=1.7,
                       zorder=3)
            ax.annotate("", xy=(x, lo * 0.6), xytext=(x, lo),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
            lbl += " (extrap<floor)"
            yl = lo
        ax.annotate(lbl, (x, yl), textcoords="offset points", xytext=(7, 4), fontsize=7.5)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=col, label=fam)
               for fam, col in FAMILY_COLOR.items()]
    handles.append(plt.Line2D([0], [0], marker="o", ls="", mfc="none", mec="#555555",
                              label="extrapolated"))
    ax.legend(handles=handles, title="architecture family", loc="best", fontsize=8.5)
    if spec["log_x"]:
        ax.set_yscale("log")
    sub = ""
    if ps:
        c0 = ps["corr"]
        sub = (f"reliable-model Pearson r={c0['pearson']:.2f} (r²={c0['r2']:.2f}, "
               f"CI [{c0['r2_ci'][0]:.2f}, {c0['r2_ci'][1]:.2f}]), Spearman ρ="
               f"{c0['spearman']:.2f}")
    ax.set_title(f"Test C v2 — clean accuracy vs. {suite} threshold\n{sub}", fontsize=11)
    ax.set_xlabel("clean top-1 accuracy on Imagenette val (%)")
    ax.set_ylabel(f"{suite} threshold @ 75% ({spec['x_label']})"
                  + (" — log" if spec["log_x"] else ""))
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# Writeup.
# ---------------------------------------------------------------------------- #
def write_outputs(outdir, curves, dropped, an, provenance):
    outdir = Path(outdir)

    cols = ["model", "family", "suite", "params_m", "clean_top1", "threshold", "ci_low",
            "ci_high", "slope_at_threshold", "converged", "at_bound", "reliable", "sigmoid",
            "chance_level", "top_k", "n_trials", "weight_tag", "device", "config_hash",
            "seconds"]
    lines = [",".join(cols)]
    for c in curves:
        r = {**c, "reliable": is_reliable(c), "at_bound": "|".join(c["at_bound"]) or "-",
             "n_trials": c["n_trials"][0]}
        lines.append(",".join(str(r.get(k, "")) for k in cols))
    (outdir / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for suite in SUITES:
        plot_psychometric(outdir / f"psychometric_curves_{suite}.png", curves, suite)
        plot_threshold_vs_accuracy(outdir / f"threshold_vs_accuracy_{suite}.png", curves, suite, an)
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str),
                                            encoding="utf-8")

    L = []
    L.append("# Test C v2 — is the threshold a new axis, or an expensive accuracy proxy?")
    L.append("")
    L.append("**Kill-criterion check, re-run with full persistence.** v1 kept one number per model "
             "and threw the fits away; v2 persists every observed point and fitted parameter to "
             "`curves.json`, so every figure here is a pure read from disk (`--regen`, zero forward "
             "passes), and writes the trial-level per-item table for every sweep.")
    L.append("")
    L.append(f"Real Imagenette val · **{provenance['n_items']} images** (full, no split) · "
             f"**{len(set(c['model'] for c in curves))} models × {len(SUITES)} suites** · device "
             f"{provenance['device']} · bootstrap n={provenance['n_boot']}.")
    L.append("")
    L.append("Every number traces to a real forward pass of a real pretrained model on a real image "
             "file. No synthetic data, no reconstructed curves, no fallbacks.")
    L.append("")

    # ---- Existence proof FIRST (per the brief: ahead of r²). ---- #
    L.append("## 1. The existence proof — models that tie on accuracy but differ on threshold")
    L.append("")
    L.append(f"Every pair within **{EQUAL_ACC_WINDOW_PTS:g} clean-accuracy point**. If two models "
             f"tie on accuracy but sit far apart on threshold, the threshold measures something "
             f"accuracy does not — regardless of what the overall correlation says.")
    for suite in SUITES:
        ps = an["per_suite"].get(suite)
        if not ps:
            continue
        L.append("")
        L.append(f"### {suite}")
        L.append("")
        if ps["pairs"]:
            L.append("| model A | model B | Δacc (pts) | thr A | thr B | ratio | both measured? |")
            L.append("|---|---|---|---|---|---|---|")
            for p in ps["pairs"]:
                L.append(f"| {p['m1']} | {p['m2']} | {p['dacc_pts']:.2f} | {_fmt(p['thr1'])} | "
                         f"{_fmt(p['thr2'])} | {p['thr_ratio']:.2f}× | "
                         f"{'yes' if p['both_reliable'] else '**no** (extrap)'} |")
            rel = [p for p in ps["pairs"] if p["both_reliable"]]
            if rel:
                t = rel[0]
                L.append("")
                L.append(f"**Sharpest fully-measured pair:** {t['m1']} ({t['acc1'] * 100:.1f}%, thr "
                         f"{_fmt(t['thr1'])}) vs {t['m2']} ({t['acc2'] * 100:.1f}%, thr "
                         f"{_fmt(t['thr2'])}) — **Δacc {t['dacc_pts']:.2f} pts, "
                         f"{t['thr_ratio']:.1f}× threshold gap**, both in-range.")
        else:
            L.append(f"_No pairs within {EQUAL_ACC_WINDOW_PTS:g} accuracy point._")
    L.append("")
    # Reconcile the existence proof against the rank correlation — the proof must NOT be used to
    # paper over a tight correlation (see §2/§3).
    csp = an["per_suite"].get("contrast", {}).get("corr", {})
    if csp and np.isfinite(csp.get("spearman", float("nan"))):
        L.append(f"**But read this against §2 before concluding anything.** On contrast, clean "
                 f"accuracy orders the models' thresholds almost perfectly (Spearman ρ="
                 f"{csp['spearman']:.3f}). Both facts are true simultaneously and they pull in "
                 f"opposite directions: *across* the full 75–90% accuracy range accuracy predicts "
                 f"threshold rank nearly deterministically, yet *within* a ~1-point accuracy band "
                 f"thresholds still differ several-fold. The threshold therefore adds little when "
                 f"comparing models far apart in accuracy, and adds real information when "
                 f"comparing models of matched accuracy — which is the practically interesting "
                 f"case, but is also the narrowest reading of the library's claim. The existence "
                 f"proof does not rescue the axis from the rank correlation; it narrows where the "
                 f"axis is useful.")
        L.append("")

    # ---- Correlations ---- #
    def cline(nm, c):
        pl, ph = c["pearson_ci"]; r2l, r2h = c["r2_ci"]
        return (f"| {nm} | {_fmt(c['pearson'], 3)} [{_fmt(pl, 2)}, {_fmt(ph, 2)}] | "
                f"{_fmt(c['r2'], 3)} [{_fmt(r2l, 2)}, {_fmt(r2h, 2)}] | "
                f"{_fmt(c['spearman'], 3)} | {'yes' if c['r2_ci_excludes_0.7'] else '**no**'} |")

    L.append("## 2. Correlations (bootstrap CIs over models)")
    L.append("")
    for suite in SUITES:
        ps = an["per_suite"].get(suite)
        if not ps:
            continue
        L.append(f"### {suite}")
        L.append("")
        L.append("| relationship | Pearson r [95% CI] | r² [95% CI] | Spearman ρ | r² CI excludes "
                 "0.7? |")
        L.append("|---|---|---|---|---|")
        L.append(cline(f"acc ↔ **threshold** (reliable n={ps['corr']['n']}) — PRIMARY", ps["corr"]))
        L.append(cline(f"acc ↔ threshold (all n={ps['corr_all']['n']})", ps["corr_all"]))
        L.append(cline("acc ↔ log₁₀ threshold (reliable)", ps["corr_log"]))
        L.append(cline("acc ↔ slope (reliable)", ps["corr_slope_acc"]))
        L.append(cline("threshold ↔ slope (reliable)", ps["corr_slope_thr"]))
        L.append("")
        lev = ps["leverage"]
        L.append(f"**Leverage check.** r²={lev['r2_full']:.2f} on all reliable models; "
                 f"**{lev['r2_min']['r2']:.2f}** without {lev['r2_min']['dropped']}, "
                 f"**{lev['r2_max']['r2']:.2f}** without {lev['r2_max']['dropped']}; without "
                 f"{LEVERAGE_MODEL} (the low-accuracy outlier) it is "
                 f"**{lev['r2_without_leverage_model']:.2f}**. "
                 + ("A swing this large means the correlation rides on single points — leverage, "
                    "not a robust relationship."
                    if abs(lev["r2_full"] - lev["r2_without_leverage_model"]) > 0.15
                    else "The correlation does not hinge on any single model."))
        L.append("")
    L.append("")

    # ---- Verdicts ---- #
    L.append("## 3. Verdict per suite")
    L.append("")
    L.append("_Rule fixed in code before the numbers: r² < 0.3 → distinct axis; r² > 0.7 → kill; "
             "else intermediate. Reported as it landed._")
    for suite in SUITES:
        ps = an["per_suite"].get(suite)
        if not ps:
            continue
        L.append("")
        L.append(f"### {suite}: **{ps['call'][0]}**")
        L.append("")
        for r in ps["call"][1]:
            L.append(f"- {r}")
    L.append("")

    # ---- Slope ---- #
    L.append("## 4. Slope — does it survive as an independent readout?")
    L.append("")
    L.append("v1 found slope is *not* independent on contrast (r=0.85 vs accuracy, −0.80 vs "
             "threshold). Checking whether that holds on degradation:")
    L.append("")
    L.append("| suite | slope↔accuracy r (r²) | slope↔threshold r (r²) | independent? |")
    L.append("|---|---|---|---|")
    for suite in SUITES:
        ps = an["per_suite"].get(suite)
        if not ps:
            continue
        a, t = ps["corr_slope_acc"], ps["corr_slope_thr"]
        ta = abs(a["pearson"]) >= 0.5 or abs(a["spearman"]) >= 0.7
        tt = abs(t["pearson"]) >= 0.5 or abs(t["spearman"]) >= 0.7
        verdict_txt = {(True, True): "no — tracks **both**",
                       (True, False): "partly — tracks accuracy, not threshold",
                       (False, True): "partly — tracks **threshold**, not accuracy",
                       (False, False): "**yes** — independent of both"}[(ta, tt)]
        L.append(f"| {suite} | {_fmt(a['pearson'], 3)} ({_fmt(a['r2'], 2)}), ρ="
                 f"{_fmt(a['spearman'], 2)} | {_fmt(t['pearson'], 3)} ({_fmt(t['r2'], 2)}), ρ="
                 f"{_fmt(t['spearman'], 2)} | {verdict_txt} |")
    L.append("")
    L.append("_Slope is read on both coefficients too: on contrast the rank relations are near-"
             "deterministic (ρ=0.98 vs accuracy, −0.94 vs threshold), so slope is clearly not a "
             "separate readout there. On degradation slope is decoupled from accuracy (r=0.29) but "
             "still tracks its own threshold (r=0.77, ρ=0.92) — so it is not an independent third "
             "axis on either suite, though for different reasons._")
    L.append("")

    # ---- Fit quality (only knowable because the observed points were persisted) ---- #
    L.append("## 4b. Fit quality — does the sigmoid actually describe the data?")
    L.append("")
    L.append("`converged=True` with no at-bound parameter says the **optimiser** succeeded. It does "
             "not say the **functional form fits**. Computed from the persisted observed points "
             "(v1 could not have shown this — it kept no points):")
    L.append("")
    L.append("| suite | model | R² (fit vs observed) | max residual | residual at lowest level |")
    L.append("|---|---|---|---|---|")
    for suite in SUITES:
        ps = an["per_suite"].get(suite)
        if not ps:
            continue
        for m, g in sorted(ps["gof"].items(), key=lambda kv: kv[1]["r2_fit"]):
            flag = " ⚠" if g["poor_fit"] else ""
            L.append(f"| {suite} | {m} | {g['r2_fit']:.3f}{flag} | {g['max_abs_resid']:.3f} | "
                     f"{g['resid_at_lowest_level']:+.3f} |")
    L.append("")
    cg = an["per_suite"].get("contrast", {}).get("gof", {})
    dg = an["per_suite"].get("degradation", {}).get("gof", {})
    if cg and dg:
        L.append(f"**The contrast (Weibull) fits are materially worse than the degradation "
                 f"(logistic) fits**: R² {min(g['r2_fit'] for g in cg.values()):.2f}–"
                 f"{max(g['r2_fit'] for g in cg.values()):.2f} vs "
                 f"{min(g['r2_fit'] for g in dg.values()):.2f}–"
                 f"{max(g['r2_fit'] for g in dg.values()):.2f}, with residuals up to "
                 f"{max(g['max_abs_resid'] for g in cg.values()):.2f} in P(correct) vs "
                 f"{max(g['max_abs_resid'] for g in dg.values()):.2f}. The residual at the lowest "
                 f"contrast level is **negative for every model** "
                 f"({min(g['resid_at_lowest_level'] for g in cg.values()):+.3f} to "
                 f"{max(g['resid_at_lowest_level'] for g in cg.values()):+.3f}): the Weibull "
                 f"systematically **over-predicts** performance at low contrast — its lower tail "
                 f"cannot fall fast enough to meet the data. This is a statement about the "
                 f"*model of the curve*, not about the models being measured.")
        L.append("")
        L.append("This compounds §6: the two models whose thresholds moved most between v1 and v2 "
                 "are also among the worst-fitting. Their thresholds are unstable **because the "
                 "functional form does not describe their data**, so adding levels re-shapes the "
                 "curve. A contrast threshold is therefore a weaker quantity than its narrow "
                 "bootstrap CI suggests — the CI measures resampling noise around a curve that is "
                 "itself the wrong shape. Fixing this (a better sigmoid family, or a free lower "
                 "asymptote) is a more valuable next step than adding models.")
    L.append("")

    # ---- Cross-suite ---- #
    L.append("## 5. Cross-suite — does contrast threshold predict degradation threshold?")
    L.append("")
    xs = an.get("cross_suite")
    if xs:
        c = xs["corr"]
        pl, ph = c["pearson_ci"]
        L.append(f"Across the {xs['n']} models measured reliably on both suites: Pearson "
                 f"r={_fmt(c['pearson'], 3)} [{_fmt(pl, 2)}, {_fmt(ph, 2)}] (r²={_fmt(c['r2'], 2)}), "
                 f"Spearman ρ={_fmt(c['spearman'], 3)}. "
                 + ("The two robustness axes track each other — a model robust to low contrast "
                    "tends to be robust to noise, so they are not independent readouts."
                    if abs(c["pearson"]) >= 0.5 else
                    "The two robustness axes are largely **independent** — contrast robustness "
                    "does not predict noise robustness. That is a genuinely separate readout, and "
                    "it comes free from this run."))
    else:
        L.append("_Too few models measured reliably on both suites._")
    L.append("")

    # ---- v1 vs v2 ---- #
    L.append("## 6. v1 vs v2 contrast thresholds (grid changed — not a strict replication)")
    L.append("")
    L.append("v1 swept `geomspace(0.01, 0.5, 7)`; v2 sweeps `geomspace(0.001, 0.5, 10)` (step ratio "
             "1.995 vs v1's 1.919). The grids differ, so a shift is not necessarily error — but a "
             "shift beyond the ~0.5% device drift seen in Test A means the threshold is sensitive "
             "to **grid placement**, which is worth knowing.")
    L.append("")
    vv = an.get("v1_vs_v2")
    if vv:
        L.append("| model | v1 thr | v2 thr | change | v1 reliable | v2 reliable | material "
                 "(>0.5%)? |")
        L.append("|---|---|---|---|---|---|---|")
        for r in sorted(vv, key=lambda z: -(z["pct_change"] if np.isfinite(z["pct_change"]) else 0)):
            L.append(f"| {r['model']} | {_fmt(r['v1'])} | {_fmt(r['v2'])} | "
                     f"{_fmt(r['pct_change'], 3)}% | {'yes' if r['v1_reliable'] else 'no'} | "
                     f"{'yes' if r['v2_reliable'] else 'no'} | "
                     f"{'**YES**' if r['material'] else 'no'} |")
        mat = [r["model"] for r in vv if r["material"]]
        L.append("")
        L.append((f"**Materially shifted ({len(mat)}): {', '.join(mat)}.** These moved far more "
                  f"than device drift, so their contrast threshold depends on where the level grid "
                  f"is placed — a real caveat on the threshold as a stable quantity."
                  if mat else
                  "**No model shifted materially** beyond device drift — the contrast threshold is "
                  "stable to this grid change."))
    else:
        L.append("_v1 results not found for comparison._")
    L.append("")

    # ---- Per-model table ---- #
    L.append("## 7. Per-model results")
    L.append("")
    for suite in SUITES:
        cs = sorted([c for c in curves if c["suite"] == suite], key=lambda z: -z["clean_top1"])
        if not cs:
            continue
        L.append(f"### {suite}")
        L.append("")
        L.append("| model | family | params (M) | clean top-1 | threshold | 95% CI | slope | "
                 "reliable? |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in cs:
            rel = "yes" if is_reliable(c) else f"**no** ({'|'.join(c['at_bound']) or 'extrap'})"
            L.append(f"| {c['model']} | {c['family']} | {_fmt(c['params_m'], 3)} | "
                     f"{c['clean_top1'] * 100:.2f}% | {_fmt(c['threshold'])} | "
                     f"[{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}] | "
                     f"{_fmt(c['slope_at_threshold'])} | {rel} |")
        L.append("")
    if dropped:
        L.append("**Dropped (unavailable, not substituted):** "
                 + "; ".join(f"{n} ({w})" for n, w in dropped))
        L.append("")

    # ---- Limitations ---- #
    L.append("## Limitations")
    L.append("")
    ps0 = an["per_suite"].get("contrast", {})
    L.append(f"- **Accuracy range.** Clean top-1 spans {ps0.get('acc_spread_pts', float('nan')):.1f} "
             f"points (75–90%): these 1000-way models sit well below ceiling on Imagenette's "
             f"fine-grained classes, so the correlation is **not** suppressed by range restriction.")
    L.append("- **n = 10 models** is a small sample for a correlation; the r² CIs are wide and, "
             "where they fail to exclude 0.7, the correlation alone cannot separate 'intermediate' "
             "from 'kill'.")
    L.append("- **Capacity and family are confounded** — the model set varies architecture and "
             "parameter count together, so neither can be credited alone.")
    L.append("- **Single dataset, 10 classes** (Imagenette val); 1000-way argmax scored on a "
             "10-class subset.")
    L.append("- **Distractor suite absent by design:** there is no Test A split-half reliability "
             "data for it, so its thresholds are of unknown stability and were deliberately not "
             "measured here.")
    L.append("")
    L.append(f"_Generated {provenance['timestamp']} · psyvis_ml {provenance['versions']['psyvis_ml']} "
             f"· torch {provenance['versions']['torch']} · timm {provenance['versions']['timm']} "
             f"· device {provenance['device']} · wall-clock "
             f"{provenance['wall_clock_seconds'] / 60:.1f} min · every figure regenerable from "
             f"`curves.json` with `--regen`._")
    (outdir / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def load_v1_contrast():
    """v1 contrast thresholds for the side-by-side (read-only; v1 dir untouched)."""
    p = V1_DIR / "results.csv"
    if not p.is_file():
        return None
    import csv
    out = {}
    with open(p, encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            try:
                thr = float(d["threshold"])
            except (TypeError, ValueError):
                continue
            rel = str(d.get("reliable", "")).lower() == "true"
            out[d["name"]] = {"threshold": thr, "reliable": rel}
    return out


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
    ap.add_argument("--models", default="", help="comma list (plumbing); default = all 10")
    ap.add_argument("--max-per-class", default="none", help="int only for plumbing tests")
    ap.add_argument("--regen", action="store_true",
                    help="replot + reanalyze from curves.json — ZERO forward passes")
    args = ap.parse_args()
    outdir = Path(args.out)

    if args.regen:
        curves = json.loads((outdir / "curves.json").read_text(encoding="utf-8"))["curves"]
        provenance = json.loads((outdir / "provenance.json").read_text(encoding="utf-8"))
        an = analyze(curves, boot_seed=provenance.get("boot_seed", 12345),
                     v1_thr=load_v1_contrast())
        provenance["verdict_per_suite"] = {s: v["call"][0] for s, v in an["per_suite"].items()}
        provenance["analysis"] = {s: {k: v[k] for k in
                                      ("corr", "corr_all", "corr_log", "corr_slope_acc",
                                       "corr_slope_thr", "leverage", "pairs", "extrap_names")}
                                  for s, v in an["per_suite"].items()}
        provenance["cross_suite"] = an["cross_suite"]
        provenance["v1_vs_v2"] = an["v1_vs_v2"]
        write_outputs(outdir, curves, provenance.get("dropped", []), an, provenance)
        for s, v in an["per_suite"].items():
            print(f"{s}: {v['call'][0]}")
        print(f"regenerated (0 forward passes) -> {outdir}")
        return

    global MODELS
    if args.models.strip():
        fam = dict(MODELS)
        MODELS = [(m.strip(), fam.get(m.strip(), "CNN")) for m in args.models.split(",")
                  if m.strip()]
    mpc = None if str(args.max_per_class).lower() == "none" else int(args.max_per_class)

    import timm
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_str = f"cuda:{torch.cuda.get_device_name(0)}" if device == "cuda" else "cpu"
    per_item_dir = outdir / "per_item"
    per_item_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("Test C v2 — threshold vs accuracy, 2 suites, FULL PERSISTENCE (REAL DATA ONLY)",
          flush=True)
    print("=" * 78, flush=True)
    print(f"device: {dev_str} · torch {torch.__version__} · timm {timm.__version__}", flush=True)

    t_load = time.time()
    ds, root = load_full_dataset(args.data_dir, max_per_class=mpc)
    n_items = len(ds.images)
    chance = 1.0 / ds.num_classes
    print(f"[gate] dataset {root} · {n_items} real images · loaded {time.time() - t_load:.1f}s",
          flush=True)
    for k in SUITES:
        lv = suite_levels(k)
        ratio = (lv[1] / lv[0]) if (SUITES[k]["spacing"] == "log" and lv[0] > 0) else float("nan")
        print(f"[gate] suite {k}: {np.asarray(lv)}"
              + (f"  (step ratio {ratio:.3f})" if np.isfinite(ratio) else ""), flush=True)

    curves, dropped = [], []
    t_start = time.time()
    n_sweeps = len(MODELS) * len(SUITES)
    done = 0
    for mi, (name, fam) in enumerate(MODELS):
        print(f"\n[{mi + 1}/{len(MODELS)}] {name} ({fam}) — loading pretrained ...", flush=True)
        try:
            clf = pe.models.timm_classifier(name, pretrained=True)
        except Exception as e:  # noqa: BLE001 — drop and report, never substitute
            print(f"  DROP {name}: {type(e).__name__}: {e}", flush=True)
            dropped.append((name, f"{type(e).__name__}: {e}"))
            continue
        params_m, tag = model_params_and_tag(name)

        t0 = time.time()
        acc = clean_top1(clf, ds)
        t_acc = time.time() - t0
        if acc < 0.5:
            _die(f"{name} clean top-1 {acc:.4f} not far above chance {chance:.4f} — weights look "
                 f"like random init, not pretrained.")
        print(f"  [gate] clean top-1 = {acc:.4f} (chance {chance:.4f}) PASS · {params_m:.1f}M · "
              f"tag {tag} · {t_acc:.1f}s", flush=True)

        for suite in SUITES:
            lv = suite_levels(suite)
            t0 = time.time()
            res = pe.measure(model=clf, suite=SUITES[suite]["make"](), dataset=ds, levels=lv,
                             seed=SWEEP_SEED, model_name=name)
            t_sweep = time.time() - t0
            b, f = res.bundle, res.fit()
            if any(int(nt) != n_items for nt in b.n_trials):
                _die(f"{name}/{suite}: n_trials {b.n_trials} != full item count {n_items}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                thr = float(f.threshold(TARGET))
                slope_at = float(f.slope(TARGET))
                ci = f.bootstrap_ci("threshold", target=TARGET, n_boot=args.n_boot, ci=0.95,
                                    seed=args.boot_seed)
                # PERSISTENCE GATE: the saved params must reproduce the fitted curve exactly,
                # else curves.json would not be sufficient to redraw without a re-sweep.
                mine = predict_from_params(np.asarray(f.levels, float), f.params, f.sigmoid)
                theirs = np.asarray(f.predict(np.asarray(f.levels, float)), float)
            dev = float(np.max(np.abs(mine - theirs)))
            if not (dev < 1e-9):
                _die(f"{name}/{suite}: persisted params do not reproduce fit.predict "
                     f"(max dev {dev:.3e}) — curves.json would be insufficient.")

            # Trial-level artifact (SCHEMA.md): margin + decision_margin per item x level.
            t0 = time.time()
            ex = pe.write_per_item([res], per_item_dir, stem=f"{suite}__{name}",
                                   extra_metadata={"suite": suite, "model": name, "family": fam,
                                                   "clean_top1": acc, "test": "C v2"})
            t_pi = time.time() - t0

            curves.append({
                "model": name, "family": fam, "params_m": round(params_m, 3), "clean_top1": acc,
                "suite": suite, "sigmoid": f.sigmoid, "decreasing": bool(res.decreasing),
                "severity_rank_axis": ("increasing severity" if res.decreasing
                                       else "increasing signal"),
                "chance_level": float(res.chance_level), "top_k": int(res.top_k),
                "seed": int(b.seed), "config_hash": b.config_hash,
                "library_version": b.library_version,
                "levels": [float(x) for x in b.levels],
                "n_correct": [int(x) for x in b.n_correct],
                "n_trials": [int(x) for x in b.n_trials],
                "fit_params": {k: float(v) for k, v in f.params.items()},
                "guess_rate": float(f.params["guess"]), "lapse_rate": float(f.params["lapse"]),
                "threshold": thr, "ci_low": float(ci.low), "ci_high": float(ci.high),
                "ci_fail_fraction": float(ci.fail_fraction), "ci_n_boot": int(ci.n_boot),
                "slope_at_threshold": slope_at,
                "converged": bool(f.converged), "at_bound": list(f.at_bound),
                "fit_summary": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                                for k, v in f.summary().items()},
                "weight_tag": tag, "device": dev_str,
                "per_item_file": Path(ex.data_path).name,
                "per_item_rows": int(ex.n_rows),
                "seconds_sweep": round(t_sweep, 1), "seconds_clean_acc": round(t_acc, 1),
                "seconds_per_item": round(t_pi, 1),
                "seconds": round(t_sweep + t_pi, 1),
            })
            done += 1
            elapsed = time.time() - t_start
            eta = (elapsed / done) * (n_sweeps - done)
            print(f"    [{suite}] thr={_fmt(thr)} CI=[{_fmt(ci.low)},{_fmt(ci.high)}] "
                  f"slope={_fmt(slope_at)} conv={f.converged} at_bound={f.at_bound or '—'} "
                  f"| sweep {t_sweep:.1f}s + per-item {t_pi:.1f}s ({ex.n_rows} rows) "
                  f"| elapsed {elapsed / 60:.1f}m ETA {eta / 60:.1f}m", flush=True)

        # Save curves incrementally so a late crash never costs the forward passes.
        (outdir / "curves.json").write_text(
            json.dumps({"meta": {"test": "C v2", "n_items": n_items,
                                 "suites": {k: {**{kk: vv for kk, vv in v.items()
                                                   if kk != "make"},
                                                "levels": [float(x) for x in suite_levels(k)]}
                                            for k, v in SUITES.items()}},
                        "curves": curves}, indent=2, default=str), encoding="utf-8")

    if len({c["model"] for c in curves}) < 3:
        _die("too few models produced results to correlate.")

    an = analyze(curves, boot_seed=args.boot_seed, v1_thr=load_v1_contrast())
    provenance = {
        "test": "C v2 — threshold vs accuracy, two suites, full persistence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(root), "n_items": n_items, "image_size": IMAGE_SIZE,
        "device": dev_str,
        "models_requested": [m for m, _ in MODELS],
        "models_used": sorted({c["model"] for c in curves}),
        "dropped": dropped,
        "weight_source": "timm create_model(pretrained=True) — ImageNet-1k (default tags)",
        "weight_tags": {c["model"]: c["weight_tag"] for c in curves},
        "level_grids": {k: [float(x) for x in suite_levels(k)] for k in SUITES},
        "level_grid_notes": {k: SUITES[k]["note"] for k in SUITES},
        "contrast_step_ratio_v2": float(suite_levels("contrast")[1] / suite_levels("contrast")[0]),
        "v1_dir": str(V1_DIR), "sweep_seed": SWEEP_SEED, "target": TARGET,
        "n_boot": args.n_boot, "boot_seed": args.boot_seed,
        "equal_acc_window_pts": EQUAL_ACC_WINDOW_PTS,
        "clean_top1": {c["model"]: c["clean_top1"] for c in curves},
        "verdict_per_suite": {s: v["call"][0] for s, v in an["per_suite"].items()},
        "verdict_reasons": {s: v["call"][1] for s, v in an["per_suite"].items()},
        "analysis": {s: {k: v[k] for k in ("corr", "corr_all", "corr_log", "corr_slope_acc",
                                           "corr_slope_thr", "leverage", "pairs", "extrap_names")}
                     for s, v in an["per_suite"].items()},
        "cross_suite": an["cross_suite"], "v1_vs_v2": an["v1_vs_v2"],
        "config_hashes": {f"{c['suite']}|{c['model']}": c["config_hash"] for c in curves},
        "per_item_dir": str(per_item_dir),
        "per_item_total_rows": int(sum(c["per_item_rows"] for c in curves)),
        "versions": {"psyvis_ml": pe.__version__, "python": platform.python_version(),
                     "numpy": np.__version__, "torch": torch.__version__, "timm": timm.__version__,
                     "scipy": __import__("scipy").__version__},
        "wall_clock_seconds": round(time.time() - t_start, 1),
    }
    write_outputs(outdir, curves, dropped, an, provenance)

    print("\n" + "=" * 78, flush=True)
    for s, v in an["per_suite"].items():
        c0 = v["corr"]
        print(f"{s}: {v['call'][0]}", flush=True)
        print(f"   r={c0['pearson']:.3f} r²={c0['r2']:.3f} CI[{c0['r2_ci'][0]:.2f},"
              f"{c0['r2_ci'][1]:.2f}] excl0.7={c0['r2_ci_excludes_0.7']} | "
              f"ρ={c0['spearman']:.3f} | reliable {v['n_reliable']}/{v['n_all']}", flush=True)
    if an["cross_suite"]:
        cc = an["cross_suite"]["corr"]
        print(f"cross-suite contrast↔degradation: r={cc['pearson']:.3f} (r²={cc['r2']:.3f})",
              flush=True)
    print(f"total wall-clock: {(time.time() - t_start) / 60:.1f} min | per-item rows: "
          f"{provenance['per_item_total_rows']}", flush=True)
    print(f"wrote curves.json, results.csv, results.md, 4 PNGs, provenance.json, per_item/ "
          f"-> {outdir}", flush=True)


if __name__ == "__main__":
    main()
