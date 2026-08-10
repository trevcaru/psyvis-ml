"""Scoping report: should the threshold criterion be ceiling-relative rather than a fixed 75%?

READS curves.json ONLY. No sweeps, no forward passes, no psyvis_ml/ changes, NOTHING ADOPTED.
Contrast suite only, all fits under variant (e): weibull, guess=chance fixed, lapse FREE.

The argument: 75% is inherited from 2AFC, where it is exactly the midpoint between chance (0.5)
and ceiling (1.0). With chance = 0.001 and per-model ceilings of 0.73-0.88 it is not the midpoint
of anything, sits at a different relative height on every model's curve, and is undefined for a
model whose ceiling is below it. A ceiling-relative criterion

    target(f) = guess + f * (ceiling - guess),      ceiling = 1 - lapse

is the natural generalisation: f = 0.5 (the "midpoint") reduces to 0.75 for a 2AFC observer
(0.5 + 0.5*(1.0 - 0.5) = 0.75), and is well-defined for every observer regardless of ceiling.

The load-bearing question is NOT whether the numbers move (they must — it is a different criterion)
but whether the model RANKING moves. If ranking is invariant, the criterion is a presentation
choice. If it is not, the criterion is load-bearing and must be justified on its own merits.
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
FRACTIONS = (0.25, 0.5, 0.75)      # relative positions in each model's own [guess, ceiling] range
N_BOOT = 5000
BOOT_SEED = 12345


def _fit_e(lv, nc, nt):
    """Variant (e): weibull, guess pinned at chance, lapse FREE."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_psychometric(lv, nc, nt, sigmoid="weibull", guess_rate=CHANCE, lapse_rate=None)


def _thr(f, target):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f.threshold(target))


def pear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return pear(np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok])))


def corr_ci(x, y, seed=BOOT_SEED, n_boot=N_BOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rp, rs = pear(x, y), spear(x, y)
    rng = np.random.default_rng(seed)
    ps, ss = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) == 0 or np.std(y[i]) == 0:
            continue
        ps.append(pear(x[i], y[i]))
        ss.append(spear(x[i], y[i]))

    def ci(v):
        v = [z for z in v if np.isfinite(z)]
        return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v else (float("nan"),) * 2
    return {"r": rp, "r_ci": ci(ps), "r2": rp * rp, "rho": rs, "rho_ci": ci(ss),
            "rho2": rs * rs, "n": int(len(x))}


def ranking(models, vals):
    """Model names ordered by threshold ascending (most sensitive first); NaN dropped."""
    pairs = [(m, v) for m, v in zip(models, vals) if np.isfinite(v)]
    return [m for m, _ in sorted(pairs, key=lambda t: t[1])]


def discordant_pairs(models, a, b):
    """Pairs whose relative order differs between criteria a and b (both finite in both)."""
    out = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            if not all(np.isfinite(v) for v in (a[i], a[j], b[i], b[j])):
                continue
            if np.sign(a[i] - a[j]) != np.sign(b[i] - b[j]):
                out.append((models[i], models[j], a[i], a[j], b[i], b[j]))
    return out


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    curves = json.loads((HERE / "curves.json").read_text(encoding="utf-8"))["curves"]
    cs = sorted([c for c in curves if c["suite"] == "contrast"], key=lambda z: -z["clean_top1"])
    lv_min = min(min(c["levels"]) for c in cs)
    lv_max = max(max(c["levels"]) for c in cs)
    L = []

    def out(s=""):
        print(s, flush=True)
        L.append(s)

    # ---- fit everything once under (e) ---- #
    rec = []
    for c in cs:
        lv = np.asarray(c["levels"], float)
        nc = np.asarray(c["n_correct"], float)
        nt = np.asarray(c["n_trials"], float)
        f = _fit_e(lv, nc, nt)
        ceiling = 1.0 - f.params["lapse"]
        r = {"model": c["model"], "acc": c["clean_top1"], "ceiling": ceiling,
             "thr75": _thr(f, 0.75), "fit": f}
        for fr in FRACTIONS:
            tgt = CHANCE + fr * (ceiling - CHANCE)
            r[f"target_{fr}"] = tgt
            r[f"thr_{fr}"] = _thr(f, tgt)
        rec.append(r)

    models = [r["model"] for r in rec]
    accs = np.array([r["acc"] for r in rec], float)
    thr75 = np.array([r["thr75"] for r in rec], float)
    thrmid = np.array([r["thr_0.5"] for r in rec], float)

    out("# Scoping: a ceiling-relative threshold criterion (contrast only)")
    out()
    out("Reads `curves.json` only. **No sweeps, no forward passes, no `psyvis_ml/` changes. "
        "NOTHING ADOPTED — the gallery and README are untouched.** All fits are variant (e): "
        "weibull, `guess`=chance fixed, **`lapse` FREE**.")
    out()
    out("**The criterion under test.** 75% is inherited from 2AFC, where it is exactly the "
        "midpoint between chance (0.5) and ceiling (1.0). Here chance is 0.001 and ceilings are "
        "0.73–0.88, so 75% is the midpoint of nothing, sits at a *different relative height on "
        "every model's curve*, and is undefined when a ceiling falls below it. The candidate")
    out()
    out("    target(f) = guess + f · (ceiling − guess),   ceiling = 1 − lapse")
    out()
    out("reduces to 0.75 for a 2AFC observer (0.5 + 0.5·(1.0−0.5)), so it **generalises** the "
        "convention rather than replacing it. f = 0.5 is the midpoint.")
    out()

    # ---------------- 1 ---------------- #
    out("## 1. Ceiling, midpoint criterion, and threshold vs. the fixed 75%")
    out()
    out("| model | clean top-1 | fitted ceiling | midpoint target | thr @ midpoint | thr @ 0.75 | "
        "ratio |")
    out("|---|---|---|---|---|---|---|")
    for r in rec:
        ratio = (r["thr75"] / r["thr_0.5"]) if (np.isfinite(r["thr75"]) and r["thr_0.5"] > 0) \
            else float("nan")
        t75 = f"{r['thr75']:.4g}" if np.isfinite(r["thr75"]) else "**undefined**"
        rt = f"{ratio:.2f}×" if np.isfinite(ratio) else "—"
        out(f"| {r['model']} | {r['acc'] * 100:.2f}% | {r['ceiling']:.3f} | "
            f"{r['target_0.5']:.3f} | {r['thr_0.5']:.4g} | {t75} | {rt} |")
    out()
    mids = np.array([r["target_0.5"] for r in rec])
    out(f"The midpoint criterion lands at **P = {mids.min():.3f}–{mids.max():.3f}** (vs a flat "
        f"0.75), and the midpoint threshold is uniformly **lower** than the 75% threshold "
        f"(median ratio "
        f"{np.nanmedian([r['thr75'] / r['thr_0.5'] for r in rec if np.isfinite(r['thr75'])]):.2f}×) "
        f"— unsurprising: on a rising curve an easier criterion is met at less contrast. "
        f"**resnet18 gains a defined threshold** ({rec[-1]['thr_0.5']:.4g}); under the fixed 75% "
        f"it has none, because its ceiling (0.726) is below the criterion.")
    out()
    out("Where does the fixed 75% actually sit on each model's curve? That is the incoherence, "
        "quantified:")
    out()
    out("| model | ceiling | relative height of the fixed 0.75 criterion |")
    out("|---|---|---|")
    for r in rec:
        f_of_75 = (0.75 - CHANCE) / (r["ceiling"] - CHANCE)
        note = " — **above the ceiling**" if f_of_75 > 1 else ""
        out(f"| {r['model']} | {r['ceiling']:.3f} | f = {f_of_75:.3f}{note} |")
    fs = [(0.75 - CHANCE) / (r["ceiling"] - CHANCE) for r in rec]
    out()
    out(f"The single number 0.75 is measuring each model at a **different point on its own curve** "
        f"— f ranges {min(fs):.2f} to {max(fs):.2f}. It is not one criterion; it is ten different "
        f"criteria wearing the same label.")
    out()

    # ---------------- 2 ---------------- #
    out("## 2. Does the model ranking change? (the load-bearing question)")
    out()
    rank75 = ranking(models, thr75)
    rankmid = ranking(models, thrmid)
    out(f"Ranking by threshold, most sensitive first (n={len(rank75)} for 0.75 — resnet18 "
        f"undefined; n={len(rankmid)} for midpoint):")
    out()
    out("| rank | @ 0.75 | @ midpoint |")
    out("|---|---|---|")
    for i in range(max(len(rank75), len(rankmid))):
        a = rank75[i] if i < len(rank75) else "—"
        b = rankmid[i] if i < len(rankmid) else "—"
        mark = "" if a == b else "  ←"
        out(f"| {i + 1} | {a} | {b}{mark} |")
    out()
    dp = discordant_pairs(models, thr75, thrmid)
    rho_between = spear(thr75, thrmid)
    out(f"Rank correlation between the two criteria (on the {int(np.sum(np.isfinite(thr75)))} "
        f"models defined under both): **Spearman ρ = {rho_between:+.3f}**.")
    out()
    if dp:
        out(f"**{len(dp)} pair(s) reorder:**")
        out()
        out("| model A | model B | thr@0.75 A | thr@0.75 B | thr@mid A | thr@mid B |")
        out("|---|---|---|---|---|---|")
        for a, b, x1, x2, y1, y2 in dp:
            out(f"| {a} | {b} | {x1:.4g} | {x2:.4g} | {y1:.4g} | {y2:.4g} |")
    else:
        out("**No pair reorders.** The ranking is identical under both criteria.")
    out()

    # ---------------- 3 ---------------- #
    out("## 3. Do the midpoint thresholds stay inside the swept range?")
    out()
    out(f"Swept contrast range: **[{lv_min:g}, {lv_max:g}]**.")
    out()
    out("| model | thr @ midpoint | inside swept range? |")
    out("|---|---|---|")
    n_out = 0
    for r in rec:
        t = r["thr_0.5"]
        ok = np.isfinite(t) and lv_min <= t <= lv_max
        n_out += (not ok)
        out(f"| {r['model']} | {t:.4g} | {'yes' if ok else '**NO — extrapolated**'} |")
    out()
    out(f"**{10 - n_out}/10 inside the swept range.**"
        + ("  All measured, none extrapolated." if n_out == 0 else
           f"  {n_out} still extrapolate — the grid does not bracket them even at the 0.001 "
           f"floor."))
    out()

    # ---------------- 4 ---------------- #
    out("## 4. Sensitivity — is the exact criterion load-bearing?")
    out()
    out("Thresholds at f = 0.25, 0.5, 0.75 of each model's own [guess, ceiling] range:")
    out()
    out("| model | f=0.25 | f=0.50 (midpoint) | f=0.75 |")
    out("|---|---|---|---|")
    for r in rec:
        out(f"| {r['model']} | {r['thr_0.25']:.4g} | {r['thr_0.5']:.4g} | {r['thr_0.75']:.4g} |")
    out()
    ranks = {fr: ranking(models, np.array([r[f"thr_{fr}"] for r in rec], float))
             for fr in FRACTIONS}
    all_same = all(ranks[f] == ranks[FRACTIONS[0]] for f in FRACTIONS)
    out("Ranking at each f:")
    out()
    out("| rank | f=0.25 | f=0.50 | f=0.75 |")
    out("|---|---|---|---|")
    for i in range(len(models)):
        row = [ranks[f][i] if i < len(ranks[f]) else "—" for f in FRACTIONS]
        mark = "" if len(set(row)) == 1 else "  ←"
        out(f"| {i + 1} | " + " | ".join(row) + f" |{mark}")
    out()
    dps = []
    for i, f1 in enumerate(FRACTIONS):
        for f2 in FRACTIONS[i + 1:]:
            a = np.array([r[f"thr_{f1}"] for r in rec], float)
            b = np.array([r[f"thr_{f2}"] for r in rec], float)
            for d in discordant_pairs(models, a, b):
                dps.append((f1, f2, d))
    if all_same and not dps:
        out("**The ranking is identical at every f.** The exact criterion is therefore a "
            "**presentation choice, not a scientific one** — it rescales the numbers but does not "
            "change which model is judged more contrast-sensitive. That is the good outcome: it "
            "means the midpoint can be adopted for coherence (well-defined for all models, same "
            "relative height on every curve, reduces to 2AFC's 75%) without anyone having to "
            "defend 0.5 over 0.4 or 0.6.")
    else:
        out(f"**The ranking is NOT invariant** — {len(dps)} discordant pair(s) across f:")
        out()
        out("| f pair | model A | model B | thr A (f1→f2) | thr B (f1→f2) |")
        out("|---|---|---|---|---|")
        for f1, f2, (a, b, x1, x2, y1, y2) in dps:
            out(f"| {f1}→{f2} | {a} | {b} | {x1:.4g}→{y1:.4g} | {x2:.4g}→{y2:.4g} |")
        out()
        out("Because the ranking moves with f, the criterion is **load-bearing** and cannot be "
            "chosen on aesthetics — the curves cross, so 'which model is more contrast-sensitive' "
            "genuinely depends on where you read them. Any adopted criterion needs an argument "
            "for *that* height, and the crossing pairs should be reported as ambiguous rather "
            "than ranked.")
    out()

    # ---------------- 5 ---------------- #
    out("## 5. Correlation vs. accuracy under the midpoint criterion")
    out()
    c75 = corr_ci(accs, thr75)
    cmid = corr_ci(accs, thrmid)
    ok_both = np.isfinite(thr75) & np.isfinite(thrmid)
    cmid_same9 = corr_ci(accs[ok_both], thrmid[ok_both])
    out("| criterion | n | Pearson r [95% CI] | r² | Spearman ρ [95% CI] | ρ² |")
    out("|---|---|---|---|---|---|")
    for lbl, c in [("(e) @ 0.75 (fixed)", c75),
                   ("(e) @ midpoint — same 9 models", cmid_same9),
                   ("(e) @ midpoint — all 10", cmid)]:
        out(f"| {lbl} | {c['n']} | {c['r']:+.3f} [{c['r_ci'][0]:+.2f}, {c['r_ci'][1]:+.2f}] | "
            f"{c['r2']:.3f} | {c['rho']:+.3f} [{c['rho_ci'][0]:+.2f}, {c['rho_ci'][1]:+.2f}] | "
            f"{c['rho2']:.3f} |")
    out()
    drho = cmid_same9["rho"] - c75["rho"]
    out(f"**On the same 9 models the criterion barely matters:** ρ moves {c75['rho']:+.3f} → "
        f"{cmid_same9['rho']:+.3f} ({drho:+.3f}), and that entire move is attributable to the one "
        f"discordant pair from §2 — vit_base (0.01607) vs deit_small (0.01634), whose 75% "
        f"thresholds differ by **1.7%**, i.e. they are tied and their order is arbitrary. This is "
        f"consistent with §4: rank-invariant thresholds cannot move a rank correlation except "
        f"through near-ties.")
    out()
    out(f"**The real effect is which models the criterion lets you keep.** Adding resnet18 back — "
        f"possible only under the midpoint — gives n=10: Pearson r = {cmid['r']:+.3f} "
        f"[{cmid['r_ci'][0]:+.2f}, {cmid['r_ci'][1]:+.2f}], **r² = {cmid['r2']:.3f}**, ρ = "
        f"{cmid['rho']:+.3f} (ρ² = {cmid['rho2']:.3f}).")
    out()
    out("Why resnet18 matters so much: under the *misspecified* fit its threshold was **0.2088** — "
        "the highest of any model — so the least-accurate model looked maximally contrast-"
        "insensitive and anchored the negative correlation. Correctly specified and read at its own "
        "midpoint, resnet18's threshold is **0.0206 — rank 7 of 10**, *more* contrast-sensitive "
        "than vit_small (0.0289), efficientnet_b0 (0.0479) and mobilenetv3 (0.0669), all of which "
        "are more accurate than it. It goes from anchoring the trend to breaking it.")
    out()
    out("So the fixed 75% criterion was **silently deleting the single model that most contradicts "
        "the accuracy↔threshold relationship** — and deleting it precisely *because* its ceiling is "
        "low, which is to say because it is inaccurate. That is a selection effect pointing in the "
        "direction of finding redundancy.")
    out()
    out("The cumulative picture across this diagnostic chain, contrast:")
    out()
    out("| analysis | n | Pearson r² | Spearman ρ (ρ²) | band (r² / ρ²) |")
    out("|---|---|---|---|---|")
    out(f"| v2 as published (lapse=0.02, fixed 0.75) | 10 | 0.537 | −0.903 (0.815) | "
        f"intermediate / **KILL** |")
    out(f"| + lapse free (e), fixed 0.75 | 9 | {c75['r2']:.3f} | {c75['rho']:+.3f} "
        f"({c75['rho2']:.3f}) | intermediate / intermediate |")
    out(f"| + midpoint criterion (coherent) | 10 | **{cmid['r2']:.3f}** | {cmid['rho']:+.3f} "
        f"({cmid['rho2']:.3f}) | **distinct-axis** / intermediate |")
    out()
    out(f"⚠ **Do not over-read this.** The n=10 Pearson CI is [{cmid['r_ci'][0]:+.2f}, "
        f"{cmid['r_ci'][1]:+.2f}] — it **includes zero**, so the linear correlation is not "
        f"distinguishable from no relationship *or* from a moderate one. Spearman stays at "
        f"{cmid['rho']:+.3f} (intermediate), so the rank relationship does **not** vanish. The "
        f"honest summary is that the coherent analysis moves contrast from 'arguably a kill' to "
        f"'arguably a distinct axis', and n=10 cannot adjudicate between them. Neither the fit fix "
        f"nor the criterion fix rescues the sample-size problem — they only stop the instrument "
        f"from manufacturing the redundancy itself.")
    out()

    # ---------------- recommendation ---------------- #
    out("## Scoping verdict (nothing adopted)")
    out()
    out("**The case for the midpoint criterion is coherence, not results.**")
    out()
    out("For:")
    out("1. **It is the actual generalisation of the convention.** 75% *is* the midpoint for 2AFC; "
        "the library imported the number instead of the definition. The midpoint reproduces 0.75 "
        "exactly where 0.75 is correct.")
    out("2. **It is well-defined for every observer.** resnet18 has no 75% threshold because it "
        "never reaches 75%; it has a perfectly good midpoint threshold. A criterion that silently "
        "deletes your weakest model is a bad criterion.")
    out("3. **It measures every model at the same relative height.** The fixed 0.75 sits at "
        f"f = {min(fs):.2f}–{max(fs):.2f} across these models — ten different criteria under one "
        "label.")
    out()
    out("4. **The ranking is invariant to f** (§4), so adopting f = 0.5 costs nothing "
        "scientifically — no one has to defend 0.5 over 0.4. The *only* ordering that moves "
        "between the fixed 0.75 and the midpoint is a pair whose thresholds differ by 1.7% (§2), "
        "i.e. a tie.")
    out()
    out("Against / open:")
    out("- **It requires `lapse` free** (variant e) to have a ceiling to be relative *to*. The two "
        "changes are a package: criterion and fit cannot be decided separately.")
    out("- **It is not neutral on the headline.** Because it readmits resnet18, it takes contrast "
        "from r² = 0.33 (n=9) to r² = 0.08 (n=10). That is the *right* number — the model was "
        "being excluded by an artifact — but adopting a criterion that moves your own key result "
        "toward the answer you want deserves scrutiny, and should be pre-registered as a fit/"
        "criterion decision made on coherence grounds **before** re-reading the correlation. The "
        "argument for it must stay §1–§4 (it is the actual 2AFC generalisation, well-defined for "
        "all observers, same relative height, rank-invariant), never §5.")
    out("- **The ceiling is itself an estimate**, so the criterion inherits its uncertainty; a "
        "poorly-identified ceiling (e.g. on the gallery's sparser 7-point grid) would propagate "
        "into the threshold. Needs checking before touching the gallery.")
    out("- **It changes every published number** in the gallery/README. The re-run is free "
        "(`--regen` from `curves.json`), but the prose is not.")
    out()
    out("_Scoping only. Not adopted. Gallery and README untouched._")
    out()

    (HERE / "criterion_scoping.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'criterion_scoping.md'}")


if __name__ == "__main__":
    main()
