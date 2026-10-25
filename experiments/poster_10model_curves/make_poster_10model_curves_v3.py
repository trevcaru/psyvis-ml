"""Poster figure v3: the 10 Test C v2 contrast curves WITHOUT a parametric fit.

  * points     observed accuracy per level, recounted from the per-item CSVs
  * lines      monotone PCHIP interpolation through the points, in log10 contrast
  * threshold  "model-free threshold": the first upward crossing of
               chance + 0.5 * (ceiling - chance), ceiling = max observed accuracy, located by
               linear interpolation in log10 contrast
  * CI         bootstrap over images (2000 resamples). Every model saw the same 3925 images at
               every level, so one shared set of image resamples is applied to all models.

Also documents the misfit of the Weibull drawn on poster v1 (weibull, guess = chance, lapse
free): deviance goodness-of-fit test per model, computed from the parameters already stored in
outputs/poster_10model_curves/fits.json. No refit.

DESCRIPTIVE ONLY: no correlation, no claim. READ-ONLY on data: no sweep, no forward pass.
Writes to outputs/poster_10model_curves_v3/ and refuses to run if that folder already exists.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import xlogy
from scipy.stats import chi2

from make_poster_10model_curves import (
    CHANCE, DPI, EXPECTED_N, FIG_H, FIG_W, FS_LABEL, FS_LEGEND, FS_TICK, GREY, GRID, INK, INK_2,
    LABEL, ROOT, SURFACE, V2_CURVES, V2_PER_ITEM, _die, _git_head, _sha256, fmt,
    predict_from_params,
)

V1_FITS = ROOT / "outputs" / "poster_10model_curves" / "fits.json"
OUT = ROOT / "outputs" / "poster_10model_curves_v3"

SUITE = "contrast"
N_LEVELS = 10
N_BOOT = 2000
BOOT_SEED = 12345
F_CRIT = 0.5

# Reference categorical slots 1-3 (all-pairs safe in light mode). ResNet-50 keeps slot 1 blue
# from poster v1 so the colour follows the model across both figures.
HIGHLIGHT = {
    "swin_tiny_patch4_window7_224": "#eb6834",  # slot 2 orange
    "resnet50": "#2a78d6",                      # slot 1 blue
    "mobilenetv3_large_100": "#1baf7a",         # slot 3 aqua
}
# Marker areas (pt^2) — also used by the overlap check.
S_POINT_HI, S_POINT_GREY = 190, 70
S_DIAMOND_HI, S_DIAMOND_GREY = 330, 200
EDGE_HI, EDGE_GREY = 2.0, 1.5


def load_meta():
    d = json.loads(V2_CURVES.read_text())
    curves = [c for c in d["curves"] if c["suite"] == SUITE]
    if len(curves) != len(LABEL):
        _die(f"{len(curves)} {SUITE} curves, expected {len(LABEL)}")
    return curves


def load_correct_matrix(c):
    """(n_items x n_levels) 0/1 matrix from the per-item CSV, plus the item-id order."""
    path = V2_PER_ITEM / c["per_item_file"]
    lv = np.asarray(c["levels"], float)
    C = np.full((EXPECTED_N, N_LEVELS), -1, np.int8)
    ids = [None] * EXPECTED_N
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] != c["model"]:
                _die(f"{path.name}: row for model {row['model']}")
            i, j = int(row["item_index"]), int(row["level_index"])
            if not np.isclose(float(row["stimulus_level"]), lv[j], rtol=1e-9):
                _die(f"{path.name}: level_index {j} has level {row['stimulus_level']}")
            if C[i, j] != -1:
                _die(f"{path.name}: duplicate (item {i}, level {j})")
            if ids[i] is None:
                ids[i] = row["item_id"]
            elif ids[i] != row["item_id"]:
                _die(f"{path.name}: item_index {i} maps to two item_ids")
            C[i, j] = int(row["correct"])
    if (C < 0).any():
        _die(f"{path.name}: missing (item, level) cells")
    if C.sum(axis=0).tolist() != list(c["n_correct"]):
        _die(f"{c['model']}: per-item recount != curves.json n_correct")
    return C, ids, {"file": path.name, "sha256": _sha256(path), "rows": int(C.size)}


def crossing(logx, acc, crit):
    """First upward crossing of `crit`, linear in log10 contrast. NaN if already above at the
    lowest level (threshold below the grid)."""
    above = acc >= crit
    j = int(np.argmax(above))
    if j == 0:
        return np.nan
    i = j - 1
    t = (crit - acc[i]) / (acc[j] - acc[i])
    return float(10 ** (logx[i] + t * (logx[j] - logx[i])))


def model_free(acc, logx):
    ceiling = float(acc.max())
    crit = CHANCE + F_CRIT * (ceiling - CHANCE)
    return ceiling, crit, crossing(logx, acc, crit)


def n_crossings(acc, crit):
    s = np.sign(acc - crit)
    return int(np.sum(s[1:] != s[:-1]))


def weibull_gof(fit):
    """Deviance G² of the stored poster-v1 Weibull against the observed counts, chi² on
    (levels − free params) df. Free params: alpha, beta, lapse (guess fixed at chance)."""
    x = np.asarray(fit["levels"], float)
    k = np.asarray(fit["n_correct"], float)
    n = np.asarray(fit["n_trials"], float)
    p = predict_from_params(x, fit["fit_params"])
    dev = float(2 * np.sum(xlogy(k, k / (n * p)) + xlogy(n - k, (n - k) / (n * (1 - p)))))
    df = len(x) - 3
    return {"deviance": dev, "df": df, "p": float(chi2.sf(dev, df)),
            "log10_p": float(chi2.logsf(dev, df) / np.log(10)),
            "max_resid": float(np.max(np.abs(k / n - p)))}


def fmt_p(r):
    if r["p"] > 0:
        return f"{r['p']:.2g}"
    return f"10^{r['log10_p']:.0f}"


# ---------------------------------------------------------------------------- #
def build_figure(rows, path):
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

    lv = np.asarray(rows[0]["levels"], float)
    grid = np.geomspace(lv[0], lv[-1], 500)
    diamonds = []  # (label, x, y, area, edge) for the overlap check

    for r in rows:
        if r["model"] in HIGHLIGHT:
            continue
        acc = np.asarray(r["acc"])
        ax.plot(grid, 100 * PchipInterpolator(np.log10(lv), acc)(np.log10(grid)), color=GREY,
                lw=2.5, zorder=2, solid_capstyle="round")
        ax.scatter(lv, 100 * acc, s=S_POINT_GREY, color=GREY, edgecolor=SURFACE, linewidth=1.2,
                   zorder=3)
        ax.scatter([r["threshold"]], [100 * r["criterion"]], marker="D", s=S_DIAMOND_GREY,
                   color=GREY, edgecolor=INK_2, linewidth=EDGE_GREY, zorder=4)
        diamonds.append((r["label"], r["threshold"], 100 * r["criterion"], S_DIAMOND_GREY,
                         EDGE_GREY))

    for model, col in HIGHLIGHT.items():
        r = next(x for x in rows if x["model"] == model)
        acc = np.asarray(r["acc"])
        ax.plot(grid, 100 * PchipInterpolator(np.log10(lv), acc)(np.log10(grid)), color=col,
                lw=4.5, zorder=5, solid_capstyle="round")
        ax.scatter(lv, 100 * acc, s=S_POINT_HI, color=col, edgecolor=SURFACE, linewidth=2.0,
                   zorder=6)
        ax.plot([r["threshold"]] * 2, [0, 100 * r["criterion"]], color=col, lw=2.5,
                ls=(0, (4, 3)), zorder=4)
        ax.scatter([r["threshold"]], [100 * r["criterion"]], marker="D", s=S_DIAMOND_HI,
                   color=col, edgecolor=INK, linewidth=EDGE_HI, zorder=7)
        diamonds.append((r["label"], r["threshold"], 100 * r["criterion"], S_DIAMOND_HI,
                         EDGE_HI))

    ax.set_xscale("log")
    ax.set_xlim(lv[0] / 1.25, lv[-1] * 1.25)
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

    hi = []
    for m, col in HIGHLIGHT.items():
        r = next(x for x in rows if x["model"] == m)
        hi.append(Line2D([], [], color=col, lw=4.5, marker="o", ms=13, mec=SURFACE, mew=2.0,
                         label=f"{r['label']} ({100 * r['clean_top1']:.1f}%)"))
    hi.append(Line2D([], [], color="none", marker="D", ms=16, mfc=INK_2, mec=INK, mew=2.0,
                     label="Threshold"))
    others = [LABEL[r["model"]] for r in sorted(rows, key=lambda r: LABEL[r["model"]])
              if r["model"] not in HIGHLIGHT]
    half = (len(others) + 1) // 2
    grey = [Line2D([], [], color=GREY, lw=2.5, marker="o", ms=8, mec=SURFACE, mew=1.2,
                   label=", ".join(others[:half]) + ",\n" + ", ".join(others[half:]))]
    common = dict(fontsize=FS_LEGEND, frameon=False, labelcolor=INK, handlelength=2.2,
                  handletextpad=0.6, borderaxespad=0.0)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.37)
    fig.legend(handles=hi, ncol=2, loc="lower left", bbox_to_anchor=(0.095, 0.115),
               columnspacing=1.6, **common)
    fig.legend(handles=grey, ncol=1, loc="lower left", bbox_to_anchor=(0.095, 0.015), **common)

    fig.canvas.draw()
    overlaps = diamond_overlaps(ax, diamonds)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return overlaps


def diamond_overlaps(ax, diamonds):
    """Pairs of threshold diamonds that may touch in the rendered figure. Marker 'D' of area
    s pt² is a sqrt(s)-pt square rotated 45°, so its vertices sit sqrt(s)/sqrt(2) pt from the
    centre; that plus half the edge width is used as a radius (conservative: it can flag a pair
    whose corners do not quite meet, never misses one that does)."""
    px_per_pt = DPI / 72.0
    pts = [(lab, ax.transData.transform((x, y)), (np.sqrt(s / 2) + e / 2) * px_per_pt)
           for lab, x, y, s, e in diamonds]
    out = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (a, pa, ra), (b, pb, rb) = pts[i], pts[j]
            d = float(np.hypot(*(pa - pb)))
            if d < 1.5 * (ra + rb):
                out.append({"a": a, "b": b, "center_dist_px": round(d, 1),
                            "touch_dist_px": round(ra + rb, 1), "overlap": bool(d < ra + rb)})
    return out


def write_results_md(rows, overlaps, max_offset, path):
    lines = [
        "# Poster v3 — 10-model contrast curves, model-free threshold",
        "",
        "Descriptive only: no correlation, no claim. Read-only on data — no sweep, no forward "
        "pass, no refit.",
        "",
        f"Source: Test C v2 per-item CSVs (`experiments/test_c_threshold_vs_accuracy_v2/"
        f"per_item/contrast__*.csv`), 10 models, n = {EXPECTED_N} images × 10 levels "
        "`geomspace(0.001, 0.5, 10)`. Recounts match `curves.json` exactly.",
        "",
        "## Method",
        "",
        "- **Points**: observed accuracy per contrast level.",
        "- **Lines**: monotone cubic (PCHIP) interpolation through the points in log₁₀ contrast. "
        "Not a fitted model; drawn to guide the eye.",
        f"- **Model-free threshold**: ceiling = max observed accuracy; criterion = chance + "
        f"{F_CRIT} × (ceiling − chance), chance = {CHANCE}; threshold = contrast at the first "
        "upward crossing of the criterion, by linear interpolation in log₁₀ contrast between "
        "the two bracketing levels.",
        f"- **95% CI**: percentile bootstrap over images, {N_BOOT} resamples (seed {BOOT_SEED}). "
        "An image keeps all 10 of its levels when drawn; ceiling, criterion and threshold are "
        "recomputed in every resample. The same image resamples are used for every model.",
        "- The diamond sits at (threshold, criterion). Because the threshold is linear in log "
        "contrast while the drawn line is PCHIP, a diamond can sit slightly off its line: "
        f"max vertical offset **{max_offset:.2f} accuracy points**.",
        "",
        "## Model-free thresholds (sorted)",
        "",
        "| model | clean top-1 | ceiling (max obs.) | criterion | model-free threshold | 95% CI "
        "| crossings |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["threshold"]):
        star = " ★" if r["model"] in HIGHLIGHT else ""
        lines.append(
            f"| {r['label']}{star} | {100 * r['clean_top1']:.1f}% | {r['ceiling']:.3f} | "
            f"{r['criterion']:.3f} | {fmt(r['threshold'])} | [{fmt(r['ci_low'])}, "
            f"{fmt(r['ci_high'])}] | {r['n_crossings']} |")
    lines += [
        "",
        "★ = highlighted in the figure. Crossings = number of times the observed points cross "
        "the criterion (1 = unambiguous). Failed bootstrap resamples: "
        f"{sum(r['ci_fail'] for r in rows)} of {N_BOOT * len(rows)}.",
        "",
        "## Weibull goodness of fit (documenting the misfit)",
        "",
        "The Weibull drawn on poster v1 (guess = chance fixed, lapse free; parameters read from "
        "`outputs/poster_10model_curves/fits.json`, not refit), tested against the observed "
        "counts: deviance G² = 2 Σ [k ln(k / nπ) + (n − k) ln((n − k) / n(1 − π))], "
        "χ² on 10 levels − 3 free parameters = 7 df.",
        "",
        "| model | deviance G² | df | p | max \\|residual\\| |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["threshold"]):
        g = r["weibull_gof"]
        lines.append(f"| {r['label']} | {g['deviance']:.1f} | {g['df']} | {fmt_p(g)} | "
                     f"{g['max_resid']:.3f} |")
    lines += [
        "",
        "Caveats on reading p: the test assumes independent binomial counts at each level. "
        f"With {EXPECTED_N} trials per level it has power to reject small absolute misfits, "
        "and the same images recur across levels, so the levels are not independent samples. "
        "The p-values document that the Weibull is rejected as an exact description of the "
        "data; the max residual column gives the size of the misfit in accuracy units.",
        "",
        "## Threshold-marker overlap check",
        "",
    ]
    if overlaps:
        lines += ["Diamond pairs closer than 1.5× touching distance in the rendered figure:",
                  "", "| pair | centre distance (px) | touching distance (px) | overlap? |",
                  "|---|---|---|---|"]
        for o in overlaps:
            lines.append(f"| {o['a']} / {o['b']} | {o['center_dist_px']} | "
                         f"{o['touch_dist_px']} | {'**yes**' if o['overlap'] else 'no'} |")
    else:
        lines.append("No pair of threshold diamonds is within 1.5× touching distance.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if OUT.exists():
        _die(f"{OUT} already exists — refusing to overwrite")

    curves = load_meta()
    v1 = {f["model"]: f for f in json.loads(V1_FITS.read_text())["fits"]}
    mats, per_item, id_order = {}, {}, None
    for c in curves:
        C, ids, info = load_correct_matrix(c)
        if id_order is None:
            id_order = ids
        elif ids != id_order:
            _die(f"{c['model']}: item order differs from other models — cannot pair resamples")
        mats[c["model"]] = C
        per_item[c["model"]] = info
    print(f"per-item matrices OK: {len(mats)} models x {EXPECTED_N} items x {N_LEVELS} levels, "
          "identical item order")

    # Shared image resamples: W[b, i] = how many times image i is drawn in resample b.
    rng = np.random.default_rng(BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, EXPECTED_N, EXPECTED_N), minlength=EXPECTED_N)
                  for _ in range(N_BOOT)]).astype(np.float64)

    rows = []
    for c in curves:
        C = mats[c["model"]]
        lv = np.asarray(c["levels"], float)
        logx = np.log10(lv)
        acc = C.mean(axis=0)
        ceiling, crit, thr = model_free(acc, logx)
        if not np.isfinite(thr):
            _die(f"{c['model']}: criterion already met at the lowest level")
        boot_acc = (W @ C) / EXPECTED_N
        bt = np.array([model_free(a, logx)[2] for a in boot_acc])
        ok = np.isfinite(bt)
        r = {
            "model": c["model"], "label": LABEL[c["model"]], "clean_top1": c["clean_top1"],
            "levels": lv.tolist(), "acc": acc.tolist(),
            "ceiling": ceiling, "ceiling_level": float(lv[int(np.argmax(acc))]),
            "criterion": crit, "threshold": thr,
            "ci_low": float(np.quantile(bt[ok], 0.025)),
            "ci_high": float(np.quantile(bt[ok], 0.975)),
            "ci_fail": int((~ok).sum()), "n_crossings": n_crossings(acc, crit),
            "pchip_at_threshold": float(PchipInterpolator(logx, acc)(np.log10(thr))),
            "weibull_gof": weibull_gof(v1[c["model"]]),
        }
        rows.append(r)
        g = r["weibull_gof"]
        print(f"{r['label']:<18} ceil {ceiling:.3f}  crit {crit:.3f}  thr {fmt(thr)} "
              f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]  x{r['n_crossings']}  "
              f"Weibull G2 {g['deviance']:.1f} p {fmt_p(g)}")

    max_offset = max(100 * abs(r["pchip_at_threshold"] - r["criterion"]) for r in rows)

    OUT.mkdir(parents=True, exist_ok=False)
    overlaps = build_figure(rows, OUT / "poster_10model_curves_v3.png")
    for o in overlaps:
        print(f"diamonds {o['a']} / {o['b']}: {o['center_dist_px']} px apart "
              f"(touch {o['touch_dist_px']}) overlap={o['overlap']}")

    with (OUT / "thresholds.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "label", "highlighted", "clean_top1", "ceiling_max_observed",
                    "ceiling_level", "criterion", "model_free_threshold", "ci_low", "ci_high",
                    "ci_failed_resamples", "n_crossings", "weibull_deviance", "weibull_df",
                    "weibull_p", "weibull_log10_p", "weibull_max_resid"])
        for r in sorted(rows, key=lambda r: r["threshold"]):
            g = r["weibull_gof"]
            w.writerow([r["model"], r["label"], r["model"] in HIGHLIGHT, r["clean_top1"],
                        r["ceiling"], r["ceiling_level"], r["criterion"], r["threshold"],
                        r["ci_low"], r["ci_high"], r["ci_fail"], r["n_crossings"],
                        g["deviance"], g["df"], g["p"], g["log10_p"], g["max_resid"]])
    with (OUT / "observed_accuracy.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "level_index", "rms_contrast", "accuracy", "n_correct", "n_trials"])
        for r in rows:
            for j, (x, a) in enumerate(zip(r["levels"], r["acc"])):
                w.writerow([r["model"], j, x, a, int(mats[r["model"]][:, j].sum()), EXPECTED_N])
    write_results_md(rows, overlaps, max_offset, OUT / "results.md")

    import matplotlib
    import scipy
    prov = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "git_head_at_run": _git_head(),
        "inputs": {"curves_json": {"path": str(V2_CURVES.relative_to(ROOT)).replace("\\", "/"),
                                   "sha256": _sha256(V2_CURVES)},
                   "v1_fits_json": {"path": str(V1_FITS.relative_to(ROOT)).replace("\\", "/"),
                                    "sha256": _sha256(V1_FITS)},
                   "per_item": per_item},
        "settings": {"suite": SUITE, "threshold": "model-free; first upward crossing of "
                     "chance + 0.5*(max observed - chance), linear in log10 contrast",
                     "line": "PCHIP in log10 contrast", "chance": CHANCE, "n_boot": N_BOOT,
                     "boot_seed": BOOT_SEED, "bootstrap": "over images, shared across models",
                     "weibull_gof": "deviance vs chi2(levels - 3), poster v1 params, no refit"},
        "figure": {"width_in": FIG_W, "height_in": FIG_H, "dpi": DPI, "tick_pt": FS_TICK,
                   "legend_pt": FS_LEGEND, "axis_label_pt": FS_LABEL,
                   "highlight": {LABEL[m]: c for m, c in HIGHLIGHT.items()}, "grey": GREY},
        "diamond_overlaps": overlaps,
        "max_diamond_offset_from_line_pts": max_offset,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "argv": sys.argv,
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
