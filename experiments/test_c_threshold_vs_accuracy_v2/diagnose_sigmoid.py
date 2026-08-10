"""Diagnostic: why is the contrast Weibull misspecified?

READS curves.json ONLY. No sweeps, no forward passes, no changes to psyvis_ml/. Nothing here is
adopted into the gallery/README — this is evidence for a decision, not the decision.

Hypothesis under test (from the brief): `measure()` fixes guess_rate = chance = 1/1000, but a model
at 0.001 RMS contrast has class priors and will not guess uniformly; if the empirical floor is well
above chance, forcing the curve through ~0 would over-predict at low contrast.

Result: REFUTED — and the real culprit is at the other end of the curve. See the writeup.

Variants (all refit from persisted per-level counts; refitting needs no model):
  current : weibull,  guess 0.001 fixed, lapse 0.02 fixed   (what the run persisted)
  (a)     : weibull,  guess FREE
  (b)     : logistic, guess 0.001 fixed
  (c)     : logistic, guess FREE
  (d)     : weibull,  guess fixed at each model's OBSERVED floor  (strongest form of hypothesis)
  (e)     : weibull,  lapse FREE                                   <-- the fix
  (f)     : weibull,  lapse fixed at 1 - clean_top1                (CIRCULAR — see caveat)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from psyvis_ml.fitting import fit_psychometric

HERE = Path(__file__).resolve().parent
TARGET = 0.75
V1_FLOOR = 0.01
CHANCE = 0.001


def _fit(lv, nc, nt, sigmoid="weibull", guess=CHANCE, lapse=0.02):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_psychometric(lv, nc, nt, sigmoid=sigmoid, guess_rate=guess, lapse_rate=lapse)


def _thr(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f.threshold(TARGET))


def _gof(f, lv, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = np.asarray(f.predict(lv), float)
    r = obs - pred
    st = np.sum((obs - obs.mean()) ** 2)
    return {"r2": float(1 - np.sum(r ** 2) / st) if st > 0 else float("nan"),
            "max_resid": float(np.max(np.abs(r))), "resid_low": float(r[0]),
            "resid_top": float(r[-1]), "thr": _thr(f),
            "guess": float(f.params["guess"]), "lapse": float(f.params["lapse"])}


def pear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return pear(np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok])))


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

    def data(c):
        lv = np.asarray(c["levels"], float)
        nc = np.asarray(c["n_correct"], float)
        nt = np.asarray(c["n_trials"], float)
        return lv, nc, nt, nc / nt

    out("# Contrast sigmoid diagnostic — the Weibull misspecification")
    out()
    out("Reads `curves.json` only. **No sweeps, no forward passes, no `psyvis_ml/` changes, "
        "nothing adopted.** Refitting uses the persisted per-level counts, so every variant is "
        "free. This is evidence for a decision, not the decision.")
    out()
    out("## TL;DR")
    out()
    out("- **The guess_rate hypothesis is REFUTED.** The empirical floor is at or *below* chance, "
        "not above it: 7 of 10 models score **exactly 0 correct out of 3925** at 0.001 contrast "
        "(uniform guessing would give ~4). Freeing `guess` changes nothing (it drives it to 0, its "
        "bound); pinning it at the observed floor makes the two elevated-floor models *worse*.")
    out("- **The real misspecification is at the other end: `lapse_rate = 0.02` forces a 98% "
        "ceiling that no model reaches.** These models cap at their clean accuracy (75–90%). "
        "Freeing `lapse` lifts median R² from **0.906 → 0.993** (worst 0.827 → 0.989), matching "
        "the degradation suite's fit quality.")
    out("- **The ρ = −0.903 result does NOT survive.** It was substantially an artifact of the "
        "fixed ceiling. Correctly specified: ρ = −0.733, Pearson r² 0.606 → **0.325**.")
    out("- **Recommend variant (e): weibull, `lapse_rate=None` (free).** Not adopted here.")
    out()

    # ---------------- 1. observed floor ---------------- #
    out("## 1. Observed P(correct) at the two lowest contrast levels")
    out()
    out(f"Chance for 1000-way is **{CHANCE}** (~4 of 3925 images). `measure()` pins `guess_rate` "
        f"there. What do the models actually do at near-zero contrast?")
    out()
    lv0 = cs[0]["levels"]
    out(f"| model | clean top-1 | n_correct @ {lv0[0]:.4g} | P | n_correct @ {lv0[1]:.4g} | P | "
        f"vs chance |")
    out("|---|---|---|---|---|---|---|")
    floors = []
    for c in cs:
        lv, nc, nt, obs = data(c)
        floors.append(obs[0])
        rel = "**0× (below chance)**" if nc[0] == 0 else f"{obs[0] / CHANCE:.0f}×"
        out(f"| {c['model']} | {c['clean_top1'] * 100:.2f}% | {int(nc[0])} | {obs[0]:.4f} | "
            f"{int(nc[1])} | {obs[1]:.4f} | {rel} |")
    fl = np.array(floors)
    n_zero = int(np.sum(fl == 0))
    out()
    out(f"**Hypothesis REFUTED.** {n_zero}/10 models get **exactly zero** correct at the lowest "
        f"contrast — *below* the ~4 that uniform guessing would deliver. Median floor "
        f"{np.median(fl):.4f}. Only swin_tiny (0.069), deit_small (0.086) and resnet101 (0.009) "
        f"sit above chance.")
    out()
    out("The class-prior intuition is right about the *mechanism* but wrong about the *direction*: "
        "at zero signal a model does collapse onto a few classes — but those classes are generally "
        "**not** among Imagenette's 10, so P(correct) collapses to 0, not up to some elevated "
        "floor. The two exceptions (swin, deit) are the models that happen to collapse *onto* one "
        "of the 10 classes, which is why their floor ≈ that class's base rate (~8–10%).")
    out()

    # ---------------- 2. variants ---------------- #
    out("## 2. Refit under seven variants (from persisted counts)")
    out()
    variants = {}
    for c in cs:
        lv, nc, nt, obs = data(c)
        floor = float(max(obs[0], 1e-6))
        variants.setdefault("current (weibull, guess=chance, lapse=0.02)", {})[c["model"]] = \
            _gof(_fit(lv, nc, nt), lv, obs)
        variants.setdefault("(a) weibull, guess FREE", {})[c["model"]] = \
            _gof(_fit(lv, nc, nt, guess=None), lv, obs)
        variants.setdefault("(b) logistic, guess=chance", {})[c["model"]] = \
            _gof(_fit(lv, nc, nt, sigmoid="logistic"), lv, obs)
        variants.setdefault("(c) logistic, guess FREE", {})[c["model"]] = \
            _gof(_fit(lv, nc, nt, sigmoid="logistic", guess=None), lv, obs)
        variants.setdefault("(d) weibull, guess=observed floor", {})[c["model"]] = \
            _gof(_fit(lv, nc, nt, guess=floor), lv, obs)
        variants.setdefault("(e) weibull, lapse FREE", {})[c["model"]] = \
            _gof(_fit(lv, nc, nt, lapse=None), lv, obs)
        variants.setdefault("(f) weibull, lapse=1-clean_top1 [CIRCULAR]", {})[c["model"]] = \
            _gof(_fit(lv, nc, nt, lapse=min(0.499, 1 - c["clean_top1"])), lv, obs)

    out("| variant | median R² | worst R² | median max-resid | low-level resid | top-level resid |")
    out("|---|---|---|---|---|---|")
    for vn, g in variants.items():
        r2 = np.array([g[m]["r2"] for m in g], float)
        mr = np.array([g[m]["max_resid"] for m in g], float)
        rl = np.array([g[m]["resid_low"] for m in g], float)
        rt = np.array([g[m]["resid_top"] for m in g], float)
        out(f"| {vn} | **{np.nanmedian(r2):.3f}** | {np.nanmin(r2):.3f} | {np.nanmedian(mr):.3f} "
            f"| {int(np.sum(rl < 0))}/10 neg | {int(np.sum(rt < 0))}/10 neg |")
    out()
    out("**Neither the guess rate nor the sigmoid family is the problem.** Freeing `guess` is a "
        "no-op (it goes to 0, its bound — the data wants the floor *lower*, not higher). Pinning "
        "guess at the observed floor makes swin (0.827→0.807) and deit (0.838→0.811) **worse**. "
        "The logistic is worse across the board (worst R² 0.646). Only **freeing `lapse` fixes it "
        "(0.906 → 0.993)**.")
    out()

    # ---------------- 2b. the smoking gun ---------------- #
    out("## 2b. The smoking gun — residual at maximum contrast")
    out()
    out("Under the current fit the residual at the *top* level is negative for all 10 models. "
        "Compare it to `0.98 − clean_top1`, i.e. how far the forced 98% ceiling sits above where "
        "the model actually tops out:")
    out()
    out("| model | clean top-1 | resid @ 0.5 | 0.98 − clean top-1 | match? |")
    out("|---|---|---|---|---|")
    for c in cs:
        g = variants["current (weibull, guess=chance, lapse=0.02)"][c["model"]]
        gap = 0.98 - c["clean_top1"]
        out(f"| {c['model']} | {c['clean_top1']:.3f} | {g['resid_top']:+.3f} | {gap:.3f} | "
            f"{'✓' if abs(abs(g['resid_top']) - gap) < 0.05 else '~'} |")
    out()
    out("`lapse_rate=0.02` asserts every observer reaches 98% given enough signal. These models "
        "cap at their clean accuracy (75–90%) — they *never* reach 98%, even on undegraded images. "
        "The fit is forced upward, and it pays for that by distorting the shape: hence the "
        "sign-alternating (− + −) residuals and the over-prediction at low contrast that prompted "
        "this diagnostic. **The floor was the symptom; the ceiling is the cause.**")
    out()

    # ---------------- 2c. fitted ceiling ---------------- #
    out("## 2c. What does the free ceiling recover?")
    out()
    out("| model | fitted ceiling (1−λ̂) | clean top-1 | diff |")
    out("|---|---|---|---|")
    ceil, clean = [], []
    for c in cs:
        g = variants["(e) weibull, lapse FREE"][c["model"]]
        cc = 1 - g["lapse"]
        ceil.append(cc)
        clean.append(c["clean_top1"])
        out(f"| {c['model']} | {cc:.3f} | {c['clean_top1']:.3f} | {cc - c['clean_top1']:+.3f} |")
    out()
    out(f"corr(fitted ceiling, clean top-1) = **r = {pear(ceil, clean):+.3f}**. The free ceiling "
        f"recovers clean accuracy from the sweep alone (consistently ~0.03 below it, as expected — "
        f"even the top level, 0.5 RMS, is still mildly degraded relative to a raw image). "
        f"**This is the key structural point: the curve has two informative parameters — a ceiling "
        f"that is essentially accuracy, and a threshold that is contrast sensitivity. Fixing the "
        f"ceiling at 0.98 forced the threshold to absorb the accuracy information, which is "
        f"precisely why the old thresholds correlated with accuracy.**")
    out()

    # ---------------- 3. grid sensitivity ---------------- #
    out("## 3. Does the grid sensitivity collapse under (e)?")
    out()
    out(f"**Why not v1's numbers:** v1 never persisted per-level counts (the sin v2 fixed), so its "
        f"curves cannot be refit under a new variant — only its published thresholds survive. "
        f"Instead, a *controlled* test on v2's own data: full 10-level grid vs a **v1-like "
        f"truncation keeping only levels ≥ {V1_FLOOR}** (same images, same counts; only the floor "
        f"moves).")
    out()
    out("| model | current shift | (e) lapse-FREE shift | verdict |")
    out("|---|---|---|---|")
    sc, se = [], []
    for c in cs:
        lv, nc, nt, obs = data(c)
        k = lv >= V1_FLOOR
        a, b = _thr(_fit(lv, nc, nt)), _thr(_fit(lv[k], nc[k], nt[k]))
        p = abs(b - a) / a * 100 if np.isfinite(a) and a > 0 and np.isfinite(b) else float("nan")
        a2, b2 = _thr(_fit(lv, nc, nt, lapse=None)), _thr(_fit(lv[k], nc[k], nt[k], lapse=None))
        q = abs(b2 - a2) / a2 * 100 if np.isfinite(a2) and a2 > 0 and np.isfinite(b2) \
            else float("nan")
        sc.append(p)
        se.append(q)
        if not np.isfinite(q):
            v = "n/a (ceiling < 0.75 criterion)"
        elif np.isfinite(a2) and a2 < V1_FLOOR:
            v = "**grid coverage** — true thr < truncation floor"
        elif q < p * 0.5:
            v = "**collapses** — was misspecification"
        elif q > p * 1.2:
            v = "worse"
        else:
            v = "persists"
        out(f"| {c['model']} | {p:.1f}% | {q:.1f}% | {v} |")
    sc, se = np.array(sc, float), np.array(se, float)
    out(f"| **median** | **{np.nanmedian(sc):.1f}%** | **{np.nanmedian(se):.1f}%** | |")
    out()
    out("**Partially — and the answer splits cleanly in two.** deit_small's shift collapses "
        "**84.7% → 11.1%**, so *its* grid sensitivity was misspecification, exactly as the "
        "hypothesis-chain predicted. But the median barely moves (28.7% → 23.6%) and the max does "
        "not improve. The residue has a different cause: under (e) the corrected thresholds fall "
        "**below the 0.01 truncation floor** for several models (e.g. resnet101 0.0051), so "
        "truncating removes every level beneath the threshold and it becomes an extrapolation "
        "again. That is **grid coverage, not functional form** — no sigmoid can fix a threshold "
        "you did not sweep low enough to bracket.")
    out()
    out("So: fixing the sigmoid does **not** make the contrast threshold grid-stable. Both fixes "
        "are needed — a correctly-specified curve *and* a grid that brackets the threshold.")
    out()

    # ---------------- 4. correlations ---------------- #
    out("## 4. Does ρ = −0.903 survive under the best-fitting variant?")
    out()
    accs = np.array([c["clean_top1"] for c in cs], float)
    cur_thr = np.array([variants["current (weibull, guess=chance, lapse=0.02)"][c["model"]]["thr"]
                        for c in cs], float)
    e_thr = np.array([variants["(e) weibull, lapse FREE"][c["model"]]["thr"] for c in cs], float)
    ok = np.isfinite(e_thr) & np.isfinite(cur_thr)
    nan_models = [c["model"] for c, o in zip(cs, np.isfinite(e_thr)) if not o]
    out("| fit | n | Pearson r | r² | Spearman ρ | ρ² | band |")
    out("|---|---|---|---|---|---|---|")

    def band(v):
        return "distinct-axis" if v < 0.3 else ("KILL" if v > 0.7 else "intermediate")
    for lbl, t in [("current (lapse=0.02)", cur_thr), ("**(e) lapse FREE**", e_thr)]:
        r, rho = pear(accs[ok], t[ok]), spear(accs[ok], t[ok])
        out(f"| {lbl} | {int(ok.sum())} | {r:+.3f} | {r * r:.3f} | {rho:+.3f} | {rho * rho:.3f} | "
            f"r²→{band(r * r)}, ρ²→{band(rho * rho)} |")
    out()
    out(f"_Both rows on the same {int(ok.sum())} models for an apples-to-apples comparison. "
        f"Excluded: {', '.join(nan_models)} — under (e) its fitted ceiling (0.726) is **below the "
        f"0.75 criterion**, so a 75%-correct threshold does not exist for it. That is not a fit "
        f"failure; it is the criterion being wrong for a model that never reaches 75%._")
    out()
    out("**No — ρ = −0.903 does not survive.** Correctly specified it falls to **ρ = −0.733** "
        "(ρ² 0.840 → 0.538, out of kill territory into intermediate) and Pearson r² falls "
        "**0.606 → 0.325**, at the edge of the distinct-axis band. ")
    out()
    out("The mechanism is now clear and it is circular contamination: the forced 98% ceiling sat "
        "`0.98 − clean_top1` above each model's real ceiling, so the *amount of distortion the fit "
        "had to absorb was itself a function of clean accuracy*. That pushed accuracy into the "
        "threshold estimate. Fix the ceiling and roughly half the apparent redundancy disappears. "
        "**The strongest evidence against the library's core claim was substantially an artifact "
        "of its own default `lapse_rate`.**")
    out()
    out("⚠ **Variant (f) is excluded from this test on purpose.** Setting `lapse = 1 − "
        "clean_top1` feeds clean accuracy *into* the fit, so its very low r² (0.152) is partly "
        "decorrelation by construction. It is not admissible evidence for a threshold-vs-accuracy "
        "claim. (e) estimates the ceiling from the sweep alone and is the honest variant.")
    out()

    # ---------------- recommendation ---------------- #
    out("## Recommendation (not adopted)")
    out()
    out("**Adopt variant (e): `weibull` with `lapse_rate=None` (free) for the contrast suite.**")
    out()
    out("Why:")
    out("1. **Fit quality**: median R² 0.906 → 0.993, worst 0.827 → 0.989 — parity with the "
        "degradation suite, whose logistic already fits at 0.98+. The sign-alternating residual "
        "pattern disappears.")
    out("2. **It is the physically correct model.** A 98% ceiling is a claim about the observer "
        "that is false for every model here. The lapse parameter exists precisely to absorb a "
        "sub-100% ceiling; pinning it at 0.02 is the human-psychophysics default imported "
        "unexamined into a setting where observers top out at 75–90%.")
    out("3. **It de-contaminates the headline result.** The threshold stops absorbing accuracy, "
        "and the ceiling — which *is* accuracy (r = +0.954) — becomes a separately readable "
        "parameter. That is a better instrument *and* a better argument for the library.")
    out()
    out("Costs / open issues, honestly:")
    out("- **The 75% criterion breaks for low-ceiling models.** resnet18's fitted ceiling is 0.726, "
        "so `threshold(0.75)` is undefined (NaN). A criterion defined *relative to each model's own "
        "range* — e.g. the midpoint `guess + (ceiling − guess)/2` — would be well-defined for every "
        "model and is the natural companion change. This needs its own decision.")
    out("- **Freeing lapse costs a degree of freedom** on a 10-point grid. It is well-identified "
        "here (the top 3–4 levels pin the ceiling), but on the gallery's 7-point grid it may be "
        "less so — worth checking before changing the gallery.")
    out("- **It does not fix grid sensitivity** (§3). That needs a floor low enough to bracket the "
        "threshold — under (e) the corrected thresholds are *lower* (e.g. swin 0.063 → 0.014), so "
        "an adequate grid needs to reach lower still than v2's 0.001 for the most sensitive models.")
    out("- **Everything above is contrast-only.** Degradation already fits at R² ≥ 0.98 with the "
        "decreasing logistic and needs no change; its lower asymptote is chance at high severity, "
        "which *is* correct there.")
    out()
    out("Not adopted in the gallery or README. Suggested next step: re-run the v2 analysis under "
        "(e) via `--regen` (free — no sweeps) and, separately, decide the criterion question.")
    out()

    (HERE / "sigmoid_diagnostic.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'sigmoid_diagnostic.md'}")


if __name__ == "__main__":
    main()
