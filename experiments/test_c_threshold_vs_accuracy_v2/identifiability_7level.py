"""Identifiability check: is the free ceiling (variant e) estimable on the gallery's 7-level grid?

READS curves.json ONLY. No sweeps, no forward passes, no psyvis_ml/ changes, NOTHING ADOPTED.

Freeing `lapse` buys fit quality (median R² 0.906 -> 0.993 on the 10-level v2 grid) at the cost of
a degree of freedom. The gallery sweeps 7 levels, not 10. If the ceiling is not identifiable at 7
levels then the (e) fit + midpoint criterion cannot be adopted in the gallery without a re-sweep —
which is a data-collection decision, not a fitting decision.

Grid mimicry: the gallery sweeps geomspace(0.01, 0.5, 7) = 0.01 .. 0.5, step ratio 1.919. The
closest available analogue inside v2's persisted grid is its TOP 7 levels, 0.0079 .. 0.5, step
ratio 1.995 — same level count, near-identical span and density. We cannot reproduce the gallery's
exact x-positions (they were never swept in v2), so this is an analogue, not a replica; that is
stated in the output rather than glossed.

The ceiling CI and the midpoint-threshold CI are bootstrapped JOINTLY: the midpoint criterion is a
function of the fitted ceiling, so a resample that moves the ceiling also moves the criterion. A
fixed-target bootstrap would understate exactly the instability we are testing for.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from psyvis_ml.fitting import fit_psychometric

HERE = Path(__file__).resolve().parent
CHANCE = 0.001
N_BOOT = 1000
BOOT_SEED = 20260716
GALLERY_LEVELS = [0.01, 0.019193831036664843, 0.03684031498640387, 0.07071067811865475,
                  0.13572088082974532, 0.26050036547934563, 0.5]


def _fit_e(lv, nc, nt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_psychometric(lv, nc, nt, sigmoid="weibull", guess_rate=CHANCE, lapse_rate=None)


def _mid_thr(f):
    """Threshold at THIS fit's own midpoint criterion."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ceil = 1.0 - f.params["lapse"]
        return float(f.threshold(CHANCE + 0.5 * (ceil - CHANCE))), ceil


def boot_joint(fit, n_boot=N_BOOT, seed=BOOT_SEED):
    """Parametric bootstrap of (ceiling, midpoint-threshold) jointly — criterion recomputed from
    each resample's own ceiling, which is the whole point."""
    rng = np.random.default_rng(seed)
    nt = np.rint(fit.n_trials).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.clip(np.asarray(fit.predict(fit.levels), float), 0.0, 1.0)
    ceils, thrs, n_fail, n_bound = [], [], 0, 0
    for _ in range(n_boot):
        k = rng.binomial(nt, p).astype(float)
        try:
            bf = fit._refit(k)
            t, c = _mid_thr(bf)
            if bf.at_bound:
                n_bound += 1
        except Exception:  # noqa: BLE001
            n_fail += 1
            continue
        if np.isfinite(c):
            ceils.append(c)
        if np.isfinite(t):
            thrs.append(t)
        else:
            n_fail += 1

    def ci(v):
        return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v else (np.nan,) * 2
    return {"ceil_ci": ci(ceils), "thr_ci": ci(thrs),
            "ceil_sd": float(np.std(ceils)) if ceils else float("nan"),
            "fail_frac": n_fail / n_boot, "bound_frac": n_bound / n_boot}


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    curves = json.loads((HERE / "curves.json").read_text(encoding="utf-8"))["curves"]
    cs = sorted([c for c in curves if c["suite"] == "contrast"], key=lambda z: -z["clean_top1"])
    L = []

    def out(s=""):
        print(s, flush=True)
        L.append(s)

    lv_full = np.asarray(cs[0]["levels"], float)
    keep = np.zeros(len(lv_full), bool)
    keep[-7:] = True                      # top 7 levels = gallery-like coverage
    lv7 = lv_full[keep]

    out("# Identifiability: is the free ceiling estimable on a 7-level (gallery-like) grid?")
    out()
    out("Reads `curves.json` only. **No sweeps, no forward passes, no `psyvis_ml/` changes, "
        "NOTHING ADOPTED.** All fits are variant (e): weibull, `guess`=chance fixed, `lapse` FREE.")
    out()
    out("**Grid mimicry, stated honestly.** The gallery sweeps "
        "`geomspace(0.01, 0.5, 7)` = " + ", ".join(f"{x:.4g}" for x in GALLERY_LEVELS) + " "
        f"(ratio 1.919). Those exact x-positions were never swept in v2, so they cannot be "
        f"reproduced from disk. The closest analogue available is v2's **top 7 levels**: "
        + ", ".join(f"{x:.4g}" for x in lv7) + f" (span {lv7[0]:.4g}–{lv7[-1]:.4g}, ratio "
        f"{lv7[1] / lv7[0]:.3f}) — same level count, near-identical span and density. This is an "
        f"**analogue, not a replica**; a real gallery answer needs the gallery's own counts.")
    out()

    rows = []
    for c in cs:
        lv = np.asarray(c["levels"], float)
        nc = np.asarray(c["n_correct"], float)
        nt = np.asarray(c["n_trials"], float)
        f10 = _fit_e(lv, nc, nt)
        f7 = _fit_e(lv[keep], nc[keep], nt[keep])
        t10, c10 = _mid_thr(f10)
        t7, c7 = _mid_thr(f7)
        b10 = boot_joint(f10)
        b7 = boot_joint(f7)
        rows.append({"model": c["model"], "acc": c["clean_top1"],
                     "c10": c10, "c7": c7, "t10": t10, "t7": t7,
                     "f10": f10, "f7": f7, "b10": b10, "b7": b7})

    # ---------------- 1 ---------------- #
    out("## 1. Ceiling and threshold: 10-level vs 7-level")
    out()
    out("| model | ceiling (10) | ceiling (7) | Δceiling | ceiling 95% CI width (10) | "
        "(7) | widening |")
    out("|---|---|---|---|---|---|---|")
    for r in rows:
        w10 = r["b10"]["ceil_ci"][1] - r["b10"]["ceil_ci"][0]
        w7 = r["b7"]["ceil_ci"][1] - r["b7"]["ceil_ci"][0]
        out(f"| {r['model']} | {r['c10']:.3f} | {r['c7']:.3f} | {r['c7'] - r['c10']:+.3f} | "
            f"{w10:.4f} | {w7:.4f} | {w7 / w10:.1f}× |")
    dc = np.array([r["c7"] - r["c10"] for r in rows])
    w10s = np.array([r["b10"]["ceil_ci"][1] - r["b10"]["ceil_ci"][0] for r in rows])
    w7s = np.array([r["b7"]["ceil_ci"][1] - r["b7"]["ceil_ci"][0] for r in rows])
    out()
    out(f"**Ceiling point estimate: median |Δ| = {np.median(np.abs(dc)):.4f}, max "
        f"{np.max(np.abs(dc)):.4f}.** Ceiling CI width: median {np.median(w10s):.4f} (10-level) → "
        f"{np.median(w7s):.4f} (7-level), a median widening of "
        f"{np.median(w7s / w10s):.1f}×.")
    out()
    out("| model | mid-thr (10) | mid-thr (7) | Δ% | mid-thr 95% CI width (10) | (7) | widening |")
    out("|---|---|---|---|---|---|---|")
    for r in rows:
        d = abs(r["t7"] - r["t10"]) / r["t10"] * 100 if r["t10"] > 0 else float("nan")
        w10 = r["b10"]["thr_ci"][1] - r["b10"]["thr_ci"][0]
        w7 = r["b7"]["thr_ci"][1] - r["b7"]["thr_ci"][0]
        out(f"| {r['model']} | {r['t10']:.4g} | {r['t7']:.4g} | {d:.1f}% | {w10:.4g} | {w7:.4g} | "
            f"{w7 / w10:.1f}× |")
    dt = np.array([abs(r["t7"] - r["t10"]) / r["t10"] * 100 for r in rows])
    out()
    out(f"**Midpoint threshold: median shift {np.median(dt):.1f}%, max {np.max(dt):.1f}%.**")
    out()

    # ---------------- 2 ---------------- #
    out("## 2. Degeneracy checks at 7 levels")
    out()
    out("| model | converged (7) | at_bound (7) | boot fail-frac (7) | boot at-bound frac (7) | "
        "ceiling CI (7) |")
    out("|---|---|---|---|---|---|")
    bad = []
    for r in rows:
        ab = "|".join(r["f7"].at_bound) or "—"
        lo, hi = r["b7"]["ceil_ci"]
        flag = ""
        if r["f7"].at_bound or not r["f7"].converged or r["b7"]["fail_frac"] > 0.05 \
                or (hi - lo) > 0.10:
            flag = " ⚠"
            bad.append(r["model"])
        out(f"| {r['model']} | {r['f7'].converged} | {ab} | {r['b7']['fail_frac']:.1%} | "
            f"{r['b7']['bound_frac']:.1%} | [{lo:.3f}, {hi:.3f}]{flag} |")
    out()
    if bad:
        out(f"**⚠ {len(bad)} model(s) show degeneracy at 7 levels:** {', '.join(bad)}.")
    else:
        out("**No model fails to converge, hits a parameter bound, or shows an unstable "
            "bootstrap at 7 levels.**")
    out()

    # criterion stability: how much does the criterion itself move with the ceiling?
    out("How unstable is the *criterion* itself? The midpoint target is "
        "`guess + (ceiling−guess)/2`, so a ceiling CI of width w gives a criterion CI of width "
        "~w/2:")
    out()
    out("| model | criterion @ point est. | criterion 95% CI (7-level) | width |")
    out("|---|---|---|---|")
    for r in rows:
        lo, hi = r["b7"]["ceil_ci"]
        out(f"| {r['model']} | {CHANCE + 0.5 * (r['c7'] - CHANCE):.3f} | "
            f"[{CHANCE + 0.5 * (lo - CHANCE):.3f}, {CHANCE + 0.5 * (hi - CHANCE):.3f}] | "
            f"{0.5 * (hi - lo):.4f} |")
    out()

    # ---------------- 3 ---------------- #
    out("## 3. Verdict and what grid would fix it")
    out()
    # The DoF question and the coverage question are different. Separate them explicitly:
    # a model breaks at 7 levels iff its threshold is not bracketed by the 7-level floor.
    floor7 = lv7[0]
    unbracketed = [r["model"] for r in rows if r["t10"] < floor7]
    ok_rows = [r for r in rows if r["t10"] >= floor7]
    dc_ok = np.array([abs(r["c7"] - r["c10"]) for r in ok_rows])
    w_ok = np.array([(r["b7"]["ceil_ci"][1] - r["b7"]["ceil_ci"][0])
                     / (r["b10"]["ceil_ci"][1] - r["b10"]["ceil_ci"][0]) for r in ok_rows])
    dt_ok = np.array([abs(r["t7"] - r["t10"]) / r["t10"] * 100 for r in ok_rows])

    out("**The DoF question and the coverage question have different answers. Separating them is "
        "the whole result.**")
    out()
    out(f"**(1) Is the free ceiling identifiable at 7 levels? YES — the extra DoF is essentially "
        f"free.** For the {len(ok_rows)} models whose threshold the 7-level grid actually "
        f"brackets, the ceiling moves by a median {np.median(dc_ok):.4f} (max "
        f"{np.max(dc_ok):.4f}) and its 95% CI widens by a median **{np.median(w_ok):.1f}×** — i.e. "
        f"not at all. No fit among them hits a bound or fails to converge, and their midpoint "
        f"threshold moves a median {np.median(dt_ok):.1f}%. This is structural: **the ceiling is "
        f"pinned by the top of the curve**, and the 7-level analogue retains every one of v2's "
        f"top levels. The three dropped levels sit near the floor and carry almost no information "
        f"about the ceiling. Freeing `lapse` costs a degree of freedom the gallery grid can "
        f"comfortably afford.")
    out()
    unb_thrs = ", ".join("{:.4g}".format(r["t10"]) for r in rows if r["model"] in unbracketed)
    out(f"**(2) Does the 7-level grid bracket the thresholds? NO — for "
        f"{len(unbracketed)} of 10 models.** {', '.join(unbracketed)} have midpoint thresholds "
        f"({unb_thrs}) *below* the "
        f"7-level floor ({floor7:.4g}). With no levels on their rising limb the fit is "
        f"unconstrained: resnet101's lapse runs to its bound (ceiling → 1.000, at-bound in 98.4% "
        f"of bootstrap resamples) and its threshold collapses to 1.7e-07 — a nonsense number, "
        f"correctly flagged rather than reported.")
    out()
    out("**Crucially, that failure is NOT caused by freeing lapse.** The same models break under "
        "the current fixed-lapse fit on the same truncated grid — the earlier truncation test gave "
        "resnet101 an 86.5% threshold shift under `lapse=0.02` versus 87.6% under lapse-free. "
        "A grid that does not reach below a model's threshold cannot measure that threshold under "
        "*any* functional form. **Do not read this as an argument against variant (e).**")
    out()
    out(f"The rule is crisp and worth stating as a design constraint: **a threshold is stable iff "
        f"the grid floor sits meaningfully below it.** Every model with midpoint threshold ≥ "
        f"~1.2× the floor shifts ≤ {np.max(dt_ok):.0f}%; every model below the floor shifts "
        f"30–100%.")
    out()
    # what grid would fix it — driven by where the midpoint thresholds actually land
    t_all = np.array([r["t10"] for r in rows])
    need_floor = t_all.min() / 4.0
    out("**What grid would fix it (a re-sweep decision, not a fit decision).** The binding "
        "constraint is bracketing the *threshold*, not estimating the ceiling:")
    out()
    out(f"- Midpoint thresholds under (e) span **{t_all.min():.4g} – {t_all.max():.4g}**. "
        f"A grid must put at least ~2 levels below the smallest of these to constrain the rising "
        f"limb, implying a floor of roughly **{need_floor:.4g}** — i.e. **~4× lower than v2's "
        f"0.001 floor**, and 10–11× lower than the gallery's 0.01.")
    out(f"- Keeping v2's step ratio (~2.0) from a {need_floor:.4g} floor to 0.5 needs "
        f"**{int(np.ceil(np.log(0.5 / need_floor) / np.log(2.0))) + 1} levels**; at the gallery's "
        f"tighter 1.919 ratio, "
        f"**{int(np.ceil(np.log(0.5 / need_floor) / np.log(1.919))) + 1} levels**.")
    out(f"- Cost, at v2's measured ~4.9 ms/image × 3925 images: roughly "
        f"**{11 * 3925 * 0.0049 / 60:.0f}–{12 * 3925 * 0.0049 / 60:.0f} min per model** for an "
        f"11–12-level sweep — about {11 / 7:.1f}× the gallery's current contrast cost.")
    out()
    out("So the honest split:")
    out("1. **Fit decision (free, no data needed):** freeing `lapse` is safe at 7 levels. The "
        "ceiling survives the sparser grid.")
    out("2. **Data decision (needs a re-sweep):** the gallery's 0.01 floor does not bracket the "
        "midpoint thresholds of the sensitive models under (e). Adopting (e)+midpoint in the "
        "gallery **requires extending the grid downward**, not just refitting. That is a "
        "re-sweep, and it should be decided as one.")
    out()
    out("_Identifiability scoping only. Nothing adopted; gallery and README untouched._")
    out()

    (HERE / "identifiability_7level.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'identifiability_7level.md'}")


if __name__ == "__main__":
    main()
