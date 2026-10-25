"""Poster figure: all 10 Test C v2 contrast curves under the corrected instrument.

Reads `experiments/test_c_threshold_vs_accuracy_v2/curves.json` (contrast suite, 10 models,
n=3925 per level) and refits every model with the corrected settings:

  * sigmoid    weibull
  * guess      chance (0.001), fixed
  * lapse      FREE (estimated)
  * criterion  midpoint `guess + 0.5 * (ceiling - guess)`, ceiling = 1 - lapse

Thresholds carry a joint parametric bootstrap CI (ceiling and criterion recomputed per resample,
as in experiments/gallery_contrast_corrected/run_gallery_contrast_corrected.py).

DESCRIPTIVE ONLY: no correlation, no claim. READ-ONLY on data: no sweep, no forward pass. The
persisted counts are cross-checked against the per-item CSVs before fitting.

Writes to outputs/poster_10model_curves/ and refuses to run if that folder already exists.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import psyvis_ml as pe
from psyvis_ml.fitting import fit_psychometric
from psyvis_ml.fitting.sigmoids import SIGMOIDS

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "experiments" / "test_c_threshold_vs_accuracy_v2"
V2_CURVES = V2 / "curves.json"
V2_PER_ITEM = V2 / "per_item"
OUT = ROOT / "outputs" / "poster_10model_curves"

SUITE = "contrast"
EXPECTED_N = 3925
EXPECTED_MODELS = 10
CHANCE = 0.001
N_BOOT = 2000
BOOT_SEED = 12345

# Poster spec: 14 in wide @ 300 dpi; 24 pt ticks/legend, 28 pt axis labels.
FIG_W, FIG_H, DPI = 14.0, 10.0, 300
FS_TICK, FS_LEGEND, FS_LABEL = 24, 24, 28

LABEL = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_small_patch16_224": "ViT-Small",
    "vit_base_patch16_224": "ViT-Base",
    "deit_small_patch16_224": "DeiT-Small",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
    "mobilenetv3_large_100": "MobileNetV3-Large",
}
# Highlighted models, in the reference categorical palette's slot order (slots 1-3 validate
# all-pairs in light mode: dataviz references/palette.md). Everything else is background grey.
HIGHLIGHT = {
    "resnet50": "#2a78d6",               # slot 1 blue
    "vit_small_patch16_224": "#eb6834",  # slot 2 orange
    "resnet101": "#1baf7a",              # slot 3 aqua
}
GREY = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#ffffff"

# Variant (e) midpoint thresholds already published in criterion_scoping.md §1 — the refit
# here must reproduce them (same data, same settings).
SCOPING_THR = {
    "resnet101": 0.003587, "vit_base_patch16_224": 0.009085, "convnext_tiny": 0.01399,
    "resnet50": 0.01065, "swin_tiny_patch4_window7_224": 0.005238,
    "deit_small_patch16_224": 0.00656, "mobilenetv3_large_100": 0.06694,
    "efficientnet_b0": 0.04785, "vit_small_patch16_224": 0.0289, "resnet18": 0.02057,
}


def _die(msg):
    raise SystemExit(f"FATAL: {msg}")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def midpoint_target(ceiling, guess=CHANCE):
    return guess + 0.5 * (ceiling - guess)


def predict_from_params(x, p, sigmoid="weibull"):
    f = SIGMOIDS[sigmoid]["f"]
    g, lam = p["guess"], p["lapse"]
    return g + (1.0 - g - lam) * np.asarray(f(np.asarray(x, float), p["alpha"], p["slope"]), float)


def load_curves():
    d = json.loads(V2_CURVES.read_text())
    if d["meta"]["n_items"] != EXPECTED_N:
        _die(f"curves.json n_items {d['meta']['n_items']} != {EXPECTED_N}")
    curves = [c for c in d["curves"] if c["suite"] == SUITE]
    if len(curves) != EXPECTED_MODELS:
        _die(f"{len(curves)} {SUITE} curves, expected {EXPECTED_MODELS}")
    for c in curves:
        if set(c["n_trials"]) != {EXPECTED_N}:
            _die(f"{c['model']}: n_trials {c['n_trials']} not all {EXPECTED_N}")
        if c["model"] not in LABEL:
            _die(f"no display label for {c['model']}")
    return curves


def check_per_item(c):
    """Recount n_correct per level from the per-item CSV; must equal curves.json exactly."""
    path = V2_PER_ITEM / c["per_item_file"]
    lv = [float(x) for x in c["levels"]]
    k = np.zeros(len(lv), np.int64)
    n = np.zeros(len(lv), np.int64)
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] != c["model"]:
                _die(f"{path.name}: row for model {row['model']}")
            i = int(row["level_index"])
            if not np.isclose(float(row["stimulus_level"]), lv[i], rtol=1e-9):
                _die(f"{path.name}: level_index {i} has level {row['stimulus_level']} != {lv[i]}")
            k[i] += int(row["correct"])
            n[i] += 1
    if k.tolist() != list(c["n_correct"]) or n.tolist() != list(c["n_trials"]):
        _die(f"{c['model']}: per-item recount {k.tolist()}/{n.tolist()} != curves.json")
    return {"file": path.name, "sha256": _sha256(path), "rows": int(n.sum())}


def boot_joint(fit, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap ceiling and the midpoint threshold JOINTLY — the criterion is a function of the
    fitted ceiling, so each resample must recompute its own criterion."""
    rng = np.random.default_rng(seed)
    nt = np.rint(fit.n_trials).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.clip(np.asarray(fit.predict(fit.levels), float), 0.0, 1.0)
    ceils, thrs, n_fail = [], [], 0
    for _ in range(n_boot):
        k = rng.binomial(nt, p).astype(float)
        try:
            bf = fit._refit(k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = 1.0 - bf.params["lapse"]
                t = float(bf.threshold(midpoint_target(c)))
        except Exception:  # noqa: BLE001
            n_fail += 1
            continue
        if np.isfinite(c) and np.isfinite(t):
            ceils.append(c)
            thrs.append(t)
        else:
            n_fail += 1

    def ci(v):
        return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v else (np.nan,) * 2
    return {"ceil_ci": ci(ceils), "thr_ci": ci(thrs), "fail_frac": n_fail / n_boot}


def refit(c):
    lv = np.asarray(c["levels"], float)
    nc = np.asarray(c["n_correct"], float)
    nt = np.asarray(c["n_trials"], float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fit_psychometric(lv, nc, nt, sigmoid="weibull", guess_rate=CHANCE, lapse_rate=None)
        ceiling = 1.0 - f.params["lapse"]
        crit = midpoint_target(ceiling)
        thr = float(f.threshold(crit))
        pred = np.asarray(f.predict(lv), float)
    obs = nc / nt
    resid = obs - pred
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    b = boot_joint(f)
    return {
        "model": c["model"], "label": LABEL[c["model"]], "family": c["family"],
        "clean_top1": c["clean_top1"],
        "levels": lv.tolist(), "n_correct": c["n_correct"], "n_trials": c["n_trials"],
        "fit_params": {k: float(v) for k, v in f.params.items()},
        "converged": bool(f.converged), "at_bound": list(f.at_bound),
        "ceiling": float(ceiling), "ceiling_ci": b["ceil_ci"],
        "criterion": float(crit),
        "threshold": thr, "ci_low": b["thr_ci"][0], "ci_high": b["thr_ci"][1],
        "ci_fail_fraction": b["fail_frac"], "ci_n_boot": N_BOOT,
        "r2": float(1 - np.sum(resid ** 2) / ss_tot), "max_resid": float(np.max(np.abs(resid))),
    }


# ---------------------------------------------------------------------------- #
def build_figure(fits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x_lo, x_hi = fits[0]["levels"][0], fits[0]["levels"][-1]
    grid = np.geomspace(x_lo, x_hi, 500)

    # Background: every model in grey, points + curve. Highlighted ones are redrawn on top.
    for r in fits:
        if r["model"] in HIGHLIGHT:
            continue
        obs = np.asarray(r["n_correct"], float) / np.asarray(r["n_trials"], float)
        ax.plot(grid, 100 * predict_from_params(grid, r["fit_params"]), color=GREY, lw=2.5,
                zorder=2, solid_capstyle="round")
        ax.scatter(r["levels"], 100 * obs, s=70, color=GREY, edgecolor=SURFACE, linewidth=1.2,
                   zorder=3)

    for model, col in HIGHLIGHT.items():
        r = next(f for f in fits if f["model"] == model)
        obs = np.asarray(r["n_correct"], float) / np.asarray(r["n_trials"], float)
        ax.plot(grid, 100 * predict_from_params(grid, r["fit_params"]), color=col, lw=4.5,
                zorder=5, solid_capstyle="round")
        ax.scatter(r["levels"], 100 * obs, s=190, color=col, edgecolor=SURFACE, linewidth=2.0,
                   zorder=6)
        # Threshold: drop line to the x-axis + diamond at (threshold, midpoint criterion).
        ax.plot([r["threshold"], r["threshold"]], [0, 100 * r["criterion"]], color=col, lw=2.5,
                ls=(0, (4, 3)), zorder=4)
        ax.scatter([r["threshold"]], [100 * r["criterion"]], marker="D", s=330, color=col,
                   edgecolor=INK, linewidth=2.0, zorder=7)

    ax.set_xscale("log")
    ax.set_xlim(x_lo / 1.25, x_hi * 1.25)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.001", "0.01", "0.1"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("RMS contrast", fontsize=FS_LABEL, color=INK, labelpad=12)
    ax.set_ylabel("Accuracy (%)", fontsize=FS_LABEL, color=INK, labelpad=12)
    ax.tick_params(labelsize=FS_TICK, colors=INK_2, length=8, width=1.5, pad=8)
    ax.grid(axis="y", color=GRID, lw=1.2, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(1.5)

    # Legend below the axes: the three highlighted models + the threshold key, then one grey
    # entry naming the seven background models.
    hi = [Line2D([], [], color=col, lw=4.5, marker="o", ms=13, mec=SURFACE, mew=2.0,
                 label=LABEL[m]) for m, col in HIGHLIGHT.items()]
    hi.append(Line2D([], [], color="none", marker="D", ms=16, mfc=INK_2, mec=INK, mew=2.0,
                     label="Threshold"))
    others = [LABEL[f["model"]] for f in sorted(fits, key=lambda f: LABEL[f["model"]])
              if f["model"] not in HIGHLIGHT]
    half = (len(others) + 1) // 2
    grey = [Line2D([], [], color=GREY, lw=2.5, marker="o", ms=8, mec=SURFACE, mew=1.2,
                   label=", ".join(others[:half]) + ",\n" + ", ".join(others[half:]))]
    common = dict(fontsize=FS_LEGEND, frameon=False, labelcolor=INK, handlelength=2.2,
                  handletextpad=0.6, borderaxespad=0.0)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.31)
    fig.legend(handles=hi, ncol=4, loc="lower left", bbox_to_anchor=(0.095, 0.10),
               columnspacing=1.6, **common)
    fig.legend(handles=grey, ncol=1, loc="lower left", bbox_to_anchor=(0.095, 0.015), **common)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def fmt(x, nd=4):
    return f"{x:.{nd}g}"


def write_results_md(fits, cross, path):
    lines = [
        "# Poster — 10-model contrast curves, corrected instrument",
        "",
        "Descriptive only: no correlation, no claim. Read-only on data — no sweep, no forward pass.",
        "",
        f"Source: `experiments/test_c_threshold_vs_accuracy_v2/curves.json`, contrast suite, "
        f"10 models, n = {EXPECTED_N} images per level, 10 levels `geomspace(0.001, 0.5, 10)`. "
        "Counts re-verified against the per-item CSVs (exact match, every model, every level).",
        "",
        f"Fit: Weibull · guess = chance ({CHANCE}) fixed · lapse estimated · threshold at the "
        "midpoint criterion `guess + 0.5 × (ceiling − guess)`, ceiling = 1 − lapse. 95% CI: joint "
        f"parametric bootstrap, {N_BOOT} resamples (seed {BOOT_SEED}), ceiling and criterion "
        "recomputed in every resample.",
        "",
        "Figure: `poster_10model_curves.png` — all 10 models drawn (points + fitted curve); "
        "ResNet-50, ViT-Small and ResNet-101 highlighted in colour with their threshold marked "
        "(◆ at threshold × criterion, dashed drop line to the x-axis); the other seven in grey.",
        "",
        "## Thresholds (sorted by threshold)",
        "",
        "| model | ceiling [95% CI] | criterion | threshold | 95% CI | fit R² | max resid |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(fits, key=lambda r: r["threshold"]):
        star = " ★" if r["model"] in HIGHLIGHT else ""
        lines.append(
            f"| {r['label']}{star} | {r['ceiling']:.3f} [{r['ceiling_ci'][0]:.3f}, "
            f"{r['ceiling_ci'][1]:.3f}] | {r['criterion']:.3f} | {fmt(r['threshold'])} | "
            f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {r['r2']:.3f} | "
            f"{r['max_resid']:.3f} |")
    n_fail = sum(r["ci_fail_fraction"] > 0 for r in fits)
    lines += [
        "",
        "★ = highlighted in the figure. Criterion and ceiling are proportions correct; "
        "threshold is RMS contrast.",
        "",
        f"All fits converged: {all(r['converged'] for r in fits)}; parameters at bound: "
        f"{[r['label'] for r in fits if r['at_bound']] or 'none'}; models with any failed "
        f"bootstrap resample: {n_fail}.",
        "",
        "## Reproduction check",
        "",
        "These settings are variant (e) of `criterion_scoping.md` §1, which reported midpoint "
        "thresholds (4 s.f.) from the same counts. Max relative difference from those values: "
        f"**{100 * cross['max_rel_diff']:.3f}%** ({cross['worst']}).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if OUT.exists():
        _die(f"{OUT} already exists — refusing to overwrite")

    curves = load_curves()
    per_item = {c["model"]: check_per_item(c) for c in curves}
    print(f"per-item recount OK for {len(per_item)} models")

    fits = []
    for c in curves:
        r = refit(c)
        fits.append(r)
        print(f"{r['label']:<18} ceil {r['ceiling']:.3f}  crit {r['criterion']:.3f}  "
              f"thr {fmt(r['threshold'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]  "
              f"R2 {r['r2']:.3f}")

    rel = {r["model"]: abs(r["threshold"] / SCOPING_THR[r["model"]] - 1) for r in fits}
    worst = max(rel, key=rel.get)
    cross = {"max_rel_diff": rel[worst], "worst": LABEL[worst],
             "per_model": {LABEL[m]: v for m, v in rel.items()}}
    if rel[worst] > 1e-3:  # scoping values are 4 s.f.
        _die(f"refit does not reproduce criterion_scoping.md: {LABEL[worst]} off by "
             f"{100 * rel[worst]:.3f}%")

    OUT.mkdir(parents=True, exist_ok=False)
    build_figure(fits, OUT / "poster_10model_curves.png")

    with (OUT / "thresholds.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "label", "highlighted", "clean_top1", "ceiling", "ceiling_ci_low",
                    "ceiling_ci_high", "criterion", "threshold", "ci_low", "ci_high",
                    "ci_fail_fraction", "fit_r2", "max_resid", "alpha", "beta", "lapse"])
        for r in sorted(fits, key=lambda r: r["threshold"]):
            p = r["fit_params"]
            w.writerow([r["model"], r["label"], r["model"] in HIGHLIGHT, r["clean_top1"],
                        r["ceiling"], *r["ceiling_ci"], r["criterion"], r["threshold"],
                        r["ci_low"], r["ci_high"], r["ci_fail_fraction"], r["r2"],
                        r["max_resid"], p["alpha"], p["slope"], p["lapse"]])

    (OUT / "fits.json").write_text(json.dumps({"fits": fits}, indent=2), encoding="utf-8")
    write_results_md(fits, cross, OUT / "results.md")

    import matplotlib
    prov = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "git_head_at_run": _git_head(),
        "inputs": {"curves_json": {"path": str(V2_CURVES.relative_to(ROOT)).replace("\\", "/"),
                                   "sha256": _sha256(V2_CURVES)},
                   "per_item": per_item},
        "settings": {"suite": SUITE, "sigmoid": "weibull", "guess_rate": CHANCE,
                     "lapse_rate": "free", "criterion": "guess + 0.5*(ceiling - guess)",
                     "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                     "bootstrap": "parametric, joint ceiling + criterion per resample"},
        "figure": {"width_in": FIG_W, "height_in": FIG_H, "dpi": DPI, "tick_pt": FS_TICK,
                   "legend_pt": FS_LEGEND, "axis_label_pt": FS_LABEL,
                   "highlight": {LABEL[m]: c for m, c in HIGHLIGHT.items()}, "grey": GREY},
        "reproduction_vs_criterion_scoping": cross,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__,
                     "psyvis_ml": getattr(pe, "__version__", "unknown")},
        "argv": sys.argv,
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
