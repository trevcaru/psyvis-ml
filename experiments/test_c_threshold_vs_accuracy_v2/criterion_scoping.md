# Scoping: a ceiling-relative threshold criterion (contrast only)

Reads `curves.json` only. **No sweeps, no forward passes, no `psyvis_ml/` changes. NOTHING ADOPTED — the gallery and README are untouched.** All fits are variant (e): weibull, `guess`=chance fixed, **`lapse` FREE**.

**The criterion under test.** 75% is inherited from 2AFC, where it is exactly the midpoint between chance (0.5) and ceiling (1.0). Here chance is 0.001 and ceilings are 0.73–0.88, so 75% is the midpoint of nothing, sits at a *different relative height on every model's curve*, and is undefined when a ceiling falls below it. The candidate

    target(f) = guess + f · (ceiling − guess),   ceiling = 1 − lapse

reduces to 0.75 for a 2AFC observer (0.5 + 0.5·(1.0−0.5)), so it **generalises** the convention rather than replacing it. f = 0.5 is the midpoint.

## 1. Ceiling, midpoint criterion, and threshold vs. the fixed 75%

| model | clean top-1 | fitted ceiling | midpoint target | thr @ midpoint | thr @ 0.75 | ratio |
|---|---|---|---|---|---|---|
| resnet101 | 89.83% | 0.870 | 0.435 | 0.003587 | 0.005105 | 1.42× |
| vit_base_patch16_224 | 89.50% | 0.877 | 0.439 | 0.009085 | 0.01607 | 1.77× |
| convnext_tiny | 88.18% | 0.857 | 0.429 | 0.01399 | 0.02403 | 1.72× |
| resnet50 | 86.27% | 0.837 | 0.419 | 0.01065 | 0.01817 | 1.71× |
| swin_tiny_patch4_window7_224 | 85.99% | 0.818 | 0.410 | 0.005238 | 0.01368 | 2.61× |
| deit_small_patch16_224 | 85.73% | 0.843 | 0.422 | 0.00656 | 0.01634 | 2.49× |
| mobilenetv3_large_100 | 85.20% | 0.810 | 0.405 | 0.06694 | 0.1342 | 2.00× |
| efficientnet_b0 | 84.69% | 0.791 | 0.396 | 0.04785 | 0.08981 | 1.88× |
| vit_small_patch16_224 | 82.27% | 0.765 | 0.383 | 0.0289 | 0.06296 | 2.18× |
| resnet18 | 75.41% | 0.726 | 0.364 | 0.02057 | **undefined** | — |

The midpoint criterion lands at **P = 0.364–0.439** (vs a flat 0.75), and the midpoint threshold is uniformly **lower** than the 75% threshold (median ratio 1.88×) — unsurprising: on a rising curve an easier criterion is met at less contrast. **resnet18 gains a defined threshold** (0.02057); under the fixed 75% it has none, because its ceiling (0.726) is below the criterion.

Where does the fixed 75% actually sit on each model's curve? That is the incoherence, quantified:

| model | ceiling | relative height of the fixed 0.75 criterion |
|---|---|---|
| resnet101 | 0.870 | f = 0.862 |
| vit_base_patch16_224 | 0.877 | f = 0.855 |
| convnext_tiny | 0.857 | f = 0.875 |
| resnet50 | 0.837 | f = 0.896 |
| swin_tiny_patch4_window7_224 | 0.818 | f = 0.916 |
| deit_small_patch16_224 | 0.843 | f = 0.889 |
| mobilenetv3_large_100 | 0.810 | f = 0.926 |
| efficientnet_b0 | 0.791 | f = 0.949 |
| vit_small_patch16_224 | 0.765 | f = 0.980 |
| resnet18 | 0.726 | f = 1.033 — **above the ceiling** |

The single number 0.75 is measuring each model at a **different point on its own curve** — f ranges 0.86 to 1.03. It is not one criterion; it is ten different criteria wearing the same label.

## 2. Does the model ranking change? (the load-bearing question)

Ranking by threshold, most sensitive first (n=9 for 0.75 — resnet18 undefined; n=10 for midpoint):

| rank | @ 0.75 | @ midpoint |
|---|---|---|
| 1 | resnet101 | resnet101 |
| 2 | swin_tiny_patch4_window7_224 | swin_tiny_patch4_window7_224 |
| 3 | vit_base_patch16_224 | deit_small_patch16_224  ← |
| 4 | deit_small_patch16_224 | vit_base_patch16_224  ← |
| 5 | resnet50 | resnet50 |
| 6 | convnext_tiny | convnext_tiny |
| 7 | vit_small_patch16_224 | resnet18  ← |
| 8 | efficientnet_b0 | vit_small_patch16_224  ← |
| 9 | mobilenetv3_large_100 | efficientnet_b0  ← |
| 10 | — | mobilenetv3_large_100  ← |

Rank correlation between the two criteria (on the 9 models defined under both): **Spearman ρ = +0.983**.

**1 pair(s) reorder:**

| model A | model B | thr@0.75 A | thr@0.75 B | thr@mid A | thr@mid B |
|---|---|---|---|---|---|
| vit_base_patch16_224 | deit_small_patch16_224 | 0.01607 | 0.01634 | 0.009085 | 0.00656 |

## 3. Do the midpoint thresholds stay inside the swept range?

Swept contrast range: **[0.001, 0.5]**.

| model | thr @ midpoint | inside swept range? |
|---|---|---|
| resnet101 | 0.003587 | yes |
| vit_base_patch16_224 | 0.009085 | yes |
| convnext_tiny | 0.01399 | yes |
| resnet50 | 0.01065 | yes |
| swin_tiny_patch4_window7_224 | 0.005238 | yes |
| deit_small_patch16_224 | 0.00656 | yes |
| mobilenetv3_large_100 | 0.06694 | yes |
| efficientnet_b0 | 0.04785 | yes |
| vit_small_patch16_224 | 0.0289 | yes |
| resnet18 | 0.02057 | yes |

**10/10 inside the swept range.**  All measured, none extrapolated.

## 4. Sensitivity — is the exact criterion load-bearing?

Thresholds at f = 0.25, 0.5, 0.75 of each model's own [guess, ceiling] range:

| model | f=0.25 | f=0.50 (midpoint) | f=0.75 |
|---|---|---|---|
| resnet101 | 0.00267 | 0.003587 | 0.004528 |
| vit_base_patch16_224 | 0.005571 | 0.009085 | 0.01336 |
| convnext_tiny | 0.009063 | 0.01399 | 0.01969 |
| resnet50 | 0.007156 | 0.01065 | 0.01456 |
| swin_tiny_patch4_window7_224 | 0.002702 | 0.005238 | 0.008826 |
| deit_small_patch16_224 | 0.003273 | 0.00656 | 0.01135 |
| mobilenetv3_large_100 | 0.04217 | 0.06694 | 0.09635 |
| efficientnet_b0 | 0.0327 | 0.04785 | 0.0646 |
| vit_small_patch16_224 | 0.01945 | 0.0289 | 0.03949 |
| resnet18 | 0.01376 | 0.02057 | 0.02823 |

Ranking at each f:

| rank | f=0.25 | f=0.50 | f=0.75 |
|---|---|---|---|
| 1 | resnet101 | resnet101 | resnet101 |
| 2 | swin_tiny_patch4_window7_224 | swin_tiny_patch4_window7_224 | swin_tiny_patch4_window7_224 |
| 3 | deit_small_patch16_224 | deit_small_patch16_224 | deit_small_patch16_224 |
| 4 | vit_base_patch16_224 | vit_base_patch16_224 | vit_base_patch16_224 |
| 5 | resnet50 | resnet50 | resnet50 |
| 6 | convnext_tiny | convnext_tiny | convnext_tiny |
| 7 | resnet18 | resnet18 | resnet18 |
| 8 | vit_small_patch16_224 | vit_small_patch16_224 | vit_small_patch16_224 |
| 9 | efficientnet_b0 | efficientnet_b0 | efficientnet_b0 |
| 10 | mobilenetv3_large_100 | mobilenetv3_large_100 | mobilenetv3_large_100 |

**The ranking is identical at every f.** The exact criterion is therefore a **presentation choice, not a scientific one** — it rescales the numbers but does not change which model is judged more contrast-sensitive. That is the good outcome: it means the midpoint can be adopted for coherence (well-defined for all models, same relative height on every curve, reduces to 2AFC's 75%) without anyone having to defend 0.5 over 0.4 or 0.6.

## 5. Correlation vs. accuracy under the midpoint criterion

| criterion | n | Pearson r [95% CI] | r² | Spearman ρ [95% CI] | ρ² |
|---|---|---|---|---|---|
| (e) @ 0.75 (fixed) | 9 | -0.570 [-0.90, -0.32] | 0.325 | -0.733 [-0.88, -0.07] | 0.538 |
| (e) @ midpoint — same 9 models | 9 | -0.525 [-0.87, -0.22] | 0.276 | -0.667 [-0.85, +0.08] | 0.444 |
| (e) @ midpoint — all 10 | 10 | -0.277 [-0.81, +0.05] | 0.077 | -0.685 [-0.83, -0.07] | 0.469 |

**On the same 9 models the criterion barely matters:** ρ moves -0.733 → -0.667 (+0.067), and that entire move is attributable to the one discordant pair from §2 — vit_base (0.01607) vs deit_small (0.01634), whose 75% thresholds differ by **1.7%**, i.e. they are tied and their order is arbitrary. This is consistent with §4: rank-invariant thresholds cannot move a rank correlation except through near-ties.

**The real effect is which models the criterion lets you keep.** Adding resnet18 back — possible only under the midpoint — gives n=10: Pearson r = -0.277 [-0.81, +0.05], **r² = 0.077**, ρ = -0.685 (ρ² = 0.469).

Why resnet18 matters so much: under the *misspecified* fit its threshold was **0.2088** — the highest of any model — so the least-accurate model looked maximally contrast-insensitive and anchored the negative correlation. Correctly specified and read at its own midpoint, resnet18's threshold is **0.0206 — rank 7 of 10**, *more* contrast-sensitive than vit_small (0.0289), efficientnet_b0 (0.0479) and mobilenetv3 (0.0669), all of which are more accurate than it. It goes from anchoring the trend to breaking it.

So the fixed 75% criterion was **silently deleting the single model that most contradicts the accuracy↔threshold relationship** — and deleting it precisely *because* its ceiling is low, which is to say because it is inaccurate. That is a selection effect pointing in the direction of finding redundancy.

The cumulative picture across this diagnostic chain, contrast:

| analysis | n | Pearson r² | Spearman ρ (ρ²) | band (r² / ρ²) |
|---|---|---|---|---|
| v2 as published (lapse=0.02, fixed 0.75) | 10 | 0.537 | −0.903 (0.815) | intermediate / **KILL** |
| + lapse free (e), fixed 0.75 | 9 | 0.325 | -0.733 (0.538) | intermediate / intermediate |
| + midpoint criterion (coherent) | 10 | **0.077** | -0.685 (0.469) | **distinct-axis** / intermediate |

⚠ **Do not over-read this.** The n=10 Pearson CI is [-0.81, +0.05] — it **includes zero**, so the linear correlation is not distinguishable from no relationship *or* from a moderate one. Spearman stays at -0.685 (intermediate), so the rank relationship does **not** vanish. The honest summary is that the coherent analysis moves contrast from 'arguably a kill' to 'arguably a distinct axis', and n=10 cannot adjudicate between them. Neither the fit fix nor the criterion fix rescues the sample-size problem — they only stop the instrument from manufacturing the redundancy itself.

## Scoping verdict (nothing adopted)

**The case for the midpoint criterion is coherence, not results.**

For:
1. **It is the actual generalisation of the convention.** 75% *is* the midpoint for 2AFC; the library imported the number instead of the definition. The midpoint reproduces 0.75 exactly where 0.75 is correct.
2. **It is well-defined for every observer.** resnet18 has no 75% threshold because it never reaches 75%; it has a perfectly good midpoint threshold. A criterion that silently deletes your weakest model is a bad criterion.
3. **It measures every model at the same relative height.** The fixed 0.75 sits at f = 0.86–1.03 across these models — ten different criteria under one label.

4. **The ranking is invariant to f** (§4), so adopting f = 0.5 costs nothing scientifically — no one has to defend 0.5 over 0.4. The *only* ordering that moves between the fixed 0.75 and the midpoint is a pair whose thresholds differ by 1.7% (§2), i.e. a tie.

Against / open:
- **It requires `lapse` free** (variant e) to have a ceiling to be relative *to*. The two changes are a package: criterion and fit cannot be decided separately.
- **It is not neutral on the headline.** Because it readmits resnet18, it takes contrast from r² = 0.33 (n=9) to r² = 0.08 (n=10). That is the *right* number — the model was being excluded by an artifact — but adopting a criterion that moves your own key result toward the answer you want deserves scrutiny, and should be pre-registered as a fit/criterion decision made on coherence grounds **before** re-reading the correlation. The argument for it must stay §1–§4 (it is the actual 2AFC generalisation, well-defined for all observers, same relative height, rank-invariant), never §5.
- **The ceiling is itself an estimate**, so the criterion inherits its uncertainty; a poorly-identified ceiling (e.g. on the gallery's sparser 7-point grid) would propagate into the threshold. Needs checking before touching the gallery.
- **It changes every published number** in the gallery/README. The re-run is free (`--regen` from `curves.json`), but the prose is not.

_Scoping only. Not adopted. Gallery and README untouched._

