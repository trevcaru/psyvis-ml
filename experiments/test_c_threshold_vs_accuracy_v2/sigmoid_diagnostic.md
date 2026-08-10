# Contrast sigmoid diagnostic — the Weibull misspecification

Reads `curves.json` only. **No sweeps, no forward passes, no `psyvis_ml/` changes, nothing adopted.** Refitting uses the persisted per-level counts, so every variant is free. This is evidence for a decision, not the decision.

## TL;DR

- **The guess_rate hypothesis is REFUTED.** The empirical floor is at or *below* chance, not above it: 7 of 10 models score **exactly 0 correct out of 3925** at 0.001 contrast (uniform guessing would give ~4). Freeing `guess` changes nothing (it drives it to 0, its bound); pinning it at the observed floor makes the two elevated-floor models *worse*.
- **The real misspecification is at the other end: `lapse_rate = 0.02` forces a 98% ceiling that no model reaches.** These models cap at their clean accuracy (75–90%). Freeing `lapse` lifts median R² from **0.906 → 0.993** (worst 0.827 → 0.989), matching the degradation suite's fit quality.
- **The ρ = −0.903 result does NOT survive.** It was substantially an artifact of the fixed ceiling. Correctly specified: ρ = −0.733, Pearson r² 0.606 → **0.325**.
- **Recommend variant (e): weibull, `lapse_rate=None` (free).** Not adopted here.

## 1. Observed P(correct) at the two lowest contrast levels

Chance for 1000-way is **0.001** (~4 of 3925 images). `measure()` pins `guess_rate` there. What do the models actually do at near-zero contrast?

| model | clean top-1 | n_correct @ 0.001 | P | n_correct @ 0.001995 | P | vs chance |
|---|---|---|---|---|---|---|
| resnet101 | 89.83% | 36 | 0.0092 | 417 | 0.1062 | 9× |
| vit_base_patch16_224 | 89.50% | 0 | 0.0000 | 64 | 0.0163 | **0× (below chance)** |
| convnext_tiny | 88.18% | 0 | 0.0000 | 6 | 0.0015 | **0× (below chance)** |
| resnet50 | 86.27% | 0 | 0.0000 | 14 | 0.0036 | **0× (below chance)** |
| swin_tiny_patch4_window7_224 | 85.99% | 271 | 0.0690 | 412 | 0.1050 | 69× |
| deit_small_patch16_224 | 85.73% | 338 | 0.0861 | 361 | 0.0920 | 86× |
| mobilenetv3_large_100 | 85.20% | 0 | 0.0000 | 0 | 0.0000 | **0× (below chance)** |
| efficientnet_b0 | 84.69% | 0 | 0.0000 | 0 | 0.0000 | **0× (below chance)** |
| vit_small_patch16_224 | 82.27% | 0 | 0.0000 | 0 | 0.0000 | **0× (below chance)** |
| resnet18 | 75.41% | 0 | 0.0000 | 1 | 0.0003 | **0× (below chance)** |

**Hypothesis REFUTED.** 7/10 models get **exactly zero** correct at the lowest contrast — *below* the ~4 that uniform guessing would deliver. Median floor 0.0000. Only swin_tiny (0.069), deit_small (0.086) and resnet101 (0.009) sit above chance.

The class-prior intuition is right about the *mechanism* but wrong about the *direction*: at zero signal a model does collapse onto a few classes — but those classes are generally **not** among Imagenette's 10, so P(correct) collapses to 0, not up to some elevated floor. The two exceptions (swin, deit) are the models that happen to collapse *onto* one of the 10 classes, which is why their floor ≈ that class's base rate (~8–10%).

## 2. Refit under seven variants (from persisted counts)

| variant | median R² | worst R² | median max-resid | low-level resid | top-level resid |
|---|---|---|---|---|---|
| current (weibull, guess=chance, lapse=0.02) | **0.906** | 0.827 | 0.185 | 10/10 neg | 10/10 neg |
| (a) weibull, guess FREE | **0.906** | 0.827 | 0.185 | 10/10 neg | 10/10 neg |
| (b) logistic, guess=chance | **0.891** | 0.646 | 0.200 | 10/10 neg | 10/10 neg |
| (c) logistic, guess FREE | **0.891** | 0.646 | 0.200 | 10/10 neg | 10/10 neg |
| (d) weibull, guess=observed floor | **0.905** | 0.807 | 0.192 | 10/10 neg | 10/10 neg |
| (e) weibull, lapse FREE | **0.993** | 0.989 | 0.060 | 8/10 neg | 0/10 neg |
| (f) weibull, lapse=1-clean_top1 [CIRCULAR] | **0.990** | 0.978 | 0.070 | 9/10 neg | 9/10 neg |

**Neither the guess rate nor the sigmoid family is the problem.** Freeing `guess` is a no-op (it goes to 0, its bound — the data wants the floor *lower*, not higher). Pinning guess at the observed floor makes swin (0.827→0.807) and deit (0.838→0.811) **worse**. The logistic is worse across the board (worst R² 0.646). Only **freeing `lapse` fixes it (0.906 → 0.993)**.

## 2b. The smoking gun — residual at maximum contrast

Under the current fit the residual at the *top* level is negative for all 10 models. Compare it to `0.98 − clean_top1`, i.e. how far the forced 98% ceiling sits above where the model actually tops out:

| model | clean top-1 | resid @ 0.5 | 0.98 − clean top-1 | match? |
|---|---|---|---|---|
| resnet101 | 0.898 | -0.083 | 0.082 | ✓ |
| vit_base_patch16_224 | 0.895 | -0.084 | 0.085 | ✓ |
| convnext_tiny | 0.882 | -0.099 | 0.098 | ✓ |
| resnet50 | 0.863 | -0.118 | 0.117 | ✓ |
| swin_tiny_patch4_window7_224 | 0.860 | -0.090 | 0.120 | ✓ |
| deit_small_patch16_224 | 0.857 | -0.106 | 0.123 | ✓ |
| mobilenetv3_large_100 | 0.852 | -0.105 | 0.128 | ✓ |
| efficientnet_b0 | 0.847 | -0.112 | 0.133 | ✓ |
| vit_small_patch16_224 | 0.823 | -0.130 | 0.157 | ✓ |
| resnet18 | 0.754 | -0.152 | 0.226 | ~ |

`lapse_rate=0.02` asserts every observer reaches 98% given enough signal. These models cap at their clean accuracy (75–90%) — they *never* reach 98%, even on undegraded images. The fit is forced upward, and it pays for that by distorting the shape: hence the sign-alternating (− + −) residuals and the over-prediction at low contrast that prompted this diagnostic. **The floor was the symptom; the ceiling is the cause.**

## 2c. What does the free ceiling recover?

| model | fitted ceiling (1−λ̂) | clean top-1 | diff |
|---|---|---|---|
| resnet101 | 0.870 | 0.898 | -0.029 |
| vit_base_patch16_224 | 0.877 | 0.895 | -0.018 |
| convnext_tiny | 0.857 | 0.882 | -0.024 |
| resnet50 | 0.837 | 0.863 | -0.026 |
| swin_tiny_patch4_window7_224 | 0.818 | 0.860 | -0.042 |
| deit_small_patch16_224 | 0.843 | 0.857 | -0.014 |
| mobilenetv3_large_100 | 0.810 | 0.852 | -0.042 |
| efficientnet_b0 | 0.791 | 0.847 | -0.056 |
| vit_small_patch16_224 | 0.765 | 0.823 | -0.057 |
| resnet18 | 0.726 | 0.754 | -0.028 |

corr(fitted ceiling, clean top-1) = **r = +0.954**. The free ceiling recovers clean accuracy from the sweep alone (consistently ~0.03 below it, as expected — even the top level, 0.5 RMS, is still mildly degraded relative to a raw image). **This is the key structural point: the curve has two informative parameters — a ceiling that is essentially accuracy, and a threshold that is contrast sensitivity. Fixing the ceiling at 0.98 forced the threshold to absorb the accuracy information, which is precisely why the old thresholds correlated with accuracy.**

## 3. Does the grid sensitivity collapse under (e)?

**Why not v1's numbers:** v1 never persisted per-level counts (the sin v2 fixed), so its curves cannot be refit under a new variant — only its published thresholds survive. Instead, a *controlled* test on v2's own data: full 10-level grid vs a **v1-like truncation keeping only levels ≥ 0.01** (same images, same counts; only the floor moves).

| model | current shift | (e) lapse-FREE shift | verdict |
|---|---|---|---|
| resnet101 | 86.5% | 87.6% | **grid coverage** — true thr < truncation floor |
| vit_base_patch16_224 | 50.6% | 21.0% | **collapses** — was misspecification |
| convnext_tiny | 16.5% | 23.6% | worse |
| resnet50 | 35.0% | 43.5% | worse |
| swin_tiny_patch4_window7_224 | 71.0% | 53.2% | persists |
| deit_small_patch16_224 | 84.7% | 11.1% | **collapses** — was misspecification |
| mobilenetv3_large_100 | 3.5% | 2.1% | persists |
| efficientnet_b0 | 5.3% | 4.6% | persists |
| vit_small_patch16_224 | 6.7% | 24.2% | worse |
| resnet18 | 22.4% | nan% | n/a (ceiling < 0.75 criterion) |
| **median** | **28.7%** | **23.6%** | |

**Partially — and the answer splits cleanly in two.** deit_small's shift collapses **84.7% → 11.1%**, so *its* grid sensitivity was misspecification, exactly as the hypothesis-chain predicted. But the median barely moves (28.7% → 23.6%) and the max does not improve. The residue has a different cause: under (e) the corrected thresholds fall **below the 0.01 truncation floor** for several models (e.g. resnet101 0.0051), so truncating removes every level beneath the threshold and it becomes an extrapolation again. That is **grid coverage, not functional form** — no sigmoid can fix a threshold you did not sweep low enough to bracket.

So: fixing the sigmoid does **not** make the contrast threshold grid-stable. Both fixes are needed — a correctly-specified curve *and* a grid that brackets the threshold.

## 4. Does ρ = −0.903 survive under the best-fitting variant?

| fit | n | Pearson r | r² | Spearman ρ | ρ² | band |
|---|---|---|---|---|---|---|
| current (lapse=0.02) | 9 | -0.778 | 0.606 | -0.917 | 0.840 | r²→intermediate, ρ²→KILL |
| **(e) lapse FREE** | 9 | -0.570 | 0.325 | -0.733 | 0.538 | r²→intermediate, ρ²→intermediate |

_Both rows on the same 9 models for an apples-to-apples comparison. Excluded: resnet18 — under (e) its fitted ceiling (0.726) is **below the 0.75 criterion**, so a 75%-correct threshold does not exist for it. That is not a fit failure; it is the criterion being wrong for a model that never reaches 75%._

**No — ρ = −0.903 does not survive.** Correctly specified it falls to **ρ = −0.733** (ρ² 0.840 → 0.538, out of kill territory into intermediate) and Pearson r² falls **0.606 → 0.325**, at the edge of the distinct-axis band. 

The mechanism is now clear and it is circular contamination: the forced 98% ceiling sat `0.98 − clean_top1` above each model's real ceiling, so the *amount of distortion the fit had to absorb was itself a function of clean accuracy*. That pushed accuracy into the threshold estimate. Fix the ceiling and roughly half the apparent redundancy disappears. **The strongest evidence against the library's core claim was substantially an artifact of its own default `lapse_rate`.**

⚠ **Variant (f) is excluded from this test on purpose.** Setting `lapse = 1 − clean_top1` feeds clean accuracy *into* the fit, so its very low r² (0.152) is partly decorrelation by construction. It is not admissible evidence for a threshold-vs-accuracy claim. (e) estimates the ceiling from the sweep alone and is the honest variant.

## Recommendation (not adopted)

**Adopt variant (e): `weibull` with `lapse_rate=None` (free) for the contrast suite.**

Why:
1. **Fit quality**: median R² 0.906 → 0.993, worst 0.827 → 0.989 — parity with the degradation suite, whose logistic already fits at 0.98+. The sign-alternating residual pattern disappears.
2. **It is the physically correct model.** A 98% ceiling is a claim about the observer that is false for every model here. The lapse parameter exists precisely to absorb a sub-100% ceiling; pinning it at 0.02 is the human-psychophysics default imported unexamined into a setting where observers top out at 75–90%.
3. **It de-contaminates the headline result.** The threshold stops absorbing accuracy, and the ceiling — which *is* accuracy (r = +0.954) — becomes a separately readable parameter. That is a better instrument *and* a better argument for the library.

Costs / open issues, honestly:
- **The 75% criterion breaks for low-ceiling models.** resnet18's fitted ceiling is 0.726, so `threshold(0.75)` is undefined (NaN). A criterion defined *relative to each model's own range* — e.g. the midpoint `guess + (ceiling − guess)/2` — would be well-defined for every model and is the natural companion change. This needs its own decision.
- **Freeing lapse costs a degree of freedom** on a 10-point grid. It is well-identified here (the top 3–4 levels pin the ceiling), but on the gallery's 7-point grid it may be less so — worth checking before changing the gallery.
- **It does not fix grid sensitivity** (§3). That needs a floor low enough to bracket the threshold — under (e) the corrected thresholds are *lower* (e.g. swin 0.063 → 0.014), so an adequate grid needs to reach lower still than v2's 0.001 for the most sensitive models.
- **Everything above is contrast-only.** Degradation already fits at R² ≥ 0.98 with the decreasing logistic and needs no change; its lower asymptote is chance at high severity, which *is* correct there.

Not adopted in the gallery or README. Suggested next step: re-run the v2 analysis under (e) via `--regen` (free — no sweeps) and, separately, decide the criterion question.

