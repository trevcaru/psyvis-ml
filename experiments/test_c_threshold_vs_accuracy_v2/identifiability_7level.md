# Identifiability: is the free ceiling estimable on a 7-level (gallery-like) grid?

Reads `curves.json` only. **No sweeps, no forward passes, no `psyvis_ml/` changes, NOTHING ADOPTED.** All fits are variant (e): weibull, `guess`=chance fixed, `lapse` FREE.

**Grid mimicry, stated honestly.** The gallery sweeps `geomspace(0.01, 0.5, 7)` = 0.01, 0.01919, 0.03684, 0.07071, 0.1357, 0.2605, 0.5 (ratio 1.919). Those exact x-positions were never swept in v2, so they cannot be reproduced from disk. The closest analogue available is v2's **top 7 levels**: 0.007937, 0.01583, 0.03158, 0.063, 0.1257, 0.2507, 0.5 (span 0.007937–0.5, ratio 1.995) — same level count, near-identical span and density. This is an **analogue, not a replica**; a real gallery answer needs the gallery's own counts.

## 1. Ceiling and threshold: 10-level vs 7-level

| model | ceiling (10) | ceiling (7) | Δceiling | ceiling 95% CI width (10) | (7) | widening |
|---|---|---|---|---|---|---|
| resnet101 | 0.870 | 1.000 | +0.130 | 0.0080 | 0.0074 | 0.9× |
| vit_base_patch16_224 | 0.877 | 0.888 | +0.011 | 0.0093 | 0.0103 | 1.1× |
| convnext_tiny | 0.857 | 0.867 | +0.009 | 0.0106 | 0.0109 | 1.0× |
| resnet50 | 0.837 | 0.848 | +0.011 | 0.0102 | 0.0109 | 1.1× |
| swin_tiny_patch4_window7_224 | 0.818 | 0.846 | +0.028 | 0.0101 | 0.0154 | 1.5× |
| deit_small_patch16_224 | 0.843 | 0.850 | +0.006 | 0.0103 | 0.0114 | 1.1× |
| mobilenetv3_large_100 | 0.810 | 0.811 | +0.001 | 0.0164 | 0.0168 | 1.0× |
| efficientnet_b0 | 0.791 | 0.791 | +0.001 | 0.0147 | 0.0149 | 1.0× |
| vit_small_patch16_224 | 0.765 | 0.777 | +0.012 | 0.0140 | 0.0150 | 1.1× |
| resnet18 | 0.726 | 0.729 | +0.003 | 0.0141 | 0.0144 | 1.0× |

**Ceiling point estimate: median |Δ| = 0.0099, max 0.1302.** Ceiling CI width: median 0.0104 (10-level) → 0.0129 (7-level), a median widening of 1.0×.

| model | mid-thr (10) | mid-thr (7) | Δ% | mid-thr 95% CI width (10) | (7) | widening |
|---|---|---|---|---|---|---|
| resnet101 | 0.003587 | 1.718e-07 | 100.0% | 0.0001007 | 5.086e-07 | 0.0× |
| vit_base_patch16_224 | 0.009085 | 0.008065 | 11.2% | 0.0003502 | 0.0006479 | 1.9× |
| convnext_tiny | 0.01399 | 0.01394 | 0.3% | 0.0005395 | 0.0006791 | 1.3× |
| resnet50 | 0.01065 | 0.01003 | 5.8% | 0.000399 | 0.0006184 | 1.5× |
| swin_tiny_patch4_window7_224 | 0.005238 | 0.001749 | 66.6% | 0.0002931 | 0.0009059 | 3.1× |
| deit_small_patch16_224 | 0.00656 | 0.004554 | 30.6% | 0.0003511 | 0.0008993 | 2.6× |
| mobilenetv3_large_100 | 0.06694 | 0.06709 | 0.2% | 0.003073 | 0.003274 | 1.1× |
| efficientnet_b0 | 0.04785 | 0.04792 | 0.1% | 0.001818 | 0.002053 | 1.1× |
| vit_small_patch16_224 | 0.0289 | 0.03049 | 5.5% | 0.001402 | 0.001551 | 1.1× |
| resnet18 | 0.02057 | 0.02056 | 0.0% | 0.0009018 | 0.001035 | 1.1× |

**Midpoint threshold: median shift 5.6%, max 100.0%.**

## 2. Degeneracy checks at 7 levels

| model | converged (7) | at_bound (7) | boot fail-frac (7) | boot at-bound frac (7) | ceiling CI (7) |
|---|---|---|---|---|---|
| resnet101 | True | lapse | 0.0% | 98.4% | [0.993, 1.000] ⚠ |
| vit_base_patch16_224 | True | — | 0.0% | 0.0% | [0.883, 0.893] |
| convnext_tiny | True | — | 0.0% | 0.0% | [0.861, 0.872] |
| resnet50 | True | — | 0.0% | 0.0% | [0.842, 0.853] |
| swin_tiny_patch4_window7_224 | True | — | 0.0% | 0.0% | [0.839, 0.855] |
| deit_small_patch16_224 | True | — | 0.0% | 0.0% | [0.844, 0.856] |
| mobilenetv3_large_100 | True | — | 0.0% | 0.0% | [0.803, 0.820] |
| efficientnet_b0 | True | — | 0.0% | 0.0% | [0.784, 0.799] |
| vit_small_patch16_224 | True | — | 0.0% | 0.0% | [0.770, 0.785] |
| resnet18 | True | — | 0.0% | 0.0% | [0.721, 0.736] |

**⚠ 1 model(s) show degeneracy at 7 levels:** resnet101.

How unstable is the *criterion* itself? The midpoint target is `guess + (ceiling−guess)/2`, so a ceiling CI of width w gives a criterion CI of width ~w/2:

| model | criterion @ point est. | criterion 95% CI (7-level) | width |
|---|---|---|---|
| resnet101 | 0.500 | [0.497, 0.500] | 0.0037 |
| vit_base_patch16_224 | 0.444 | [0.442, 0.447] | 0.0052 |
| convnext_tiny | 0.434 | [0.431, 0.437] | 0.0054 |
| resnet50 | 0.424 | [0.422, 0.427] | 0.0055 |
| swin_tiny_patch4_window7_224 | 0.424 | [0.420, 0.428] | 0.0077 |
| deit_small_patch16_224 | 0.425 | [0.423, 0.428] | 0.0057 |
| mobilenetv3_large_100 | 0.406 | [0.402, 0.410] | 0.0084 |
| efficientnet_b0 | 0.396 | [0.393, 0.400] | 0.0074 |
| vit_small_patch16_224 | 0.389 | [0.385, 0.393] | 0.0075 |
| resnet18 | 0.365 | [0.361, 0.368] | 0.0072 |

## 3. Verdict and what grid would fix it

**The DoF question and the coverage question have different answers. Separating them is the whole result.**

**(1) Is the free ceiling identifiable at 7 levels? YES — the extra DoF is essentially free.** For the 7 models whose threshold the 7-level grid actually brackets, the ceiling moves by a median 0.0091 (max 0.0117) and its 95% CI widens by a median **1.0×** — i.e. not at all. No fit among them hits a bound or fails to converge, and their midpoint threshold moves a median 0.3%. This is structural: **the ceiling is pinned by the top of the curve**, and the 7-level analogue retains every one of v2's top levels. The three dropped levels sit near the floor and carry almost no information about the ceiling. Freeing `lapse` costs a degree of freedom the gallery grid can comfortably afford.

**(2) Does the 7-level grid bracket the thresholds? NO — for 3 of 10 models.** resnet101, swin_tiny_patch4_window7_224, deit_small_patch16_224 have midpoint thresholds (0.003587, 0.005238, 0.00656) *below* the 7-level floor (0.007937). With no levels on their rising limb the fit is unconstrained: resnet101's lapse runs to its bound (ceiling → 1.000, at-bound in 98.4% of bootstrap resamples) and its threshold collapses to 1.7e-07 — a nonsense number, correctly flagged rather than reported.

**Crucially, that failure is NOT caused by freeing lapse.** The same models break under the current fixed-lapse fit on the same truncated grid — the earlier truncation test gave resnet101 an 86.5% threshold shift under `lapse=0.02` versus 87.6% under lapse-free. A grid that does not reach below a model's threshold cannot measure that threshold under *any* functional form. **Do not read this as an argument against variant (e).**

The rule is crisp and worth stating as a design constraint: **a threshold is stable iff the grid floor sits meaningfully below it.** Every model with midpoint threshold ≥ ~1.2× the floor shifts ≤ 11%; every model below the floor shifts 30–100%.

**What grid would fix it (a re-sweep decision, not a fit decision).** The binding constraint is bracketing the *threshold*, not estimating the ceiling:

- Midpoint thresholds under (e) span **0.003587 – 0.06694**. A grid must put at least ~2 levels below the smallest of these to constrain the rising limb, implying a floor of roughly **0.0008968** — i.e. **~4× lower than v2's 0.001 floor**, and 10–11× lower than the gallery's 0.01.
- Keeping v2's step ratio (~2.0) from a 0.0008968 floor to 0.5 needs **11 levels**; at the gallery's tighter 1.919 ratio, **11 levels**.
- Cost, at v2's measured ~4.9 ms/image × 3925 images: roughly **4–4 min per model** for an 11–12-level sweep — about 1.6× the gallery's current contrast cost.

So the honest split:
1. **Fit decision (free, no data needed):** freeing `lapse` is safe at 7 levels. The ceiling survives the sparser grid.
2. **Data decision (needs a re-sweep):** the gallery's 0.01 floor does not bracket the midpoint thresholds of the sensitive models under (e). Adopting (e)+midpoint in the gallery **requires extending the grid downward**, not just refitting. That is a re-sweep, and it should be decided as one.

_Identifiability scoping only. Nothing adopted; gallery and README untouched._

