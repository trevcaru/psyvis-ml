# Test C v2 — is the threshold a new axis, or an expensive accuracy proxy?

**Kill-criterion check, re-run with full persistence.** v1 kept one number per model and threw the fits away; v2 persists every observed point and fitted parameter to `curves.json`, so every figure here is a pure read from disk (`--regen`, zero forward passes), and writes the trial-level per-item table for every sweep.

Real Imagenette val · **3925 images** (full, no split) · **10 models × 2 suites** · device cuda:NVIDIA GeForce RTX 4070 SUPER · bootstrap n=2000.

Every number traces to a real forward pass of a real pretrained model on a real image file. No synthetic data, no reconstructed curves, no fallbacks.

## 1. The existence proof — models that tie on accuracy but differ on threshold

Every pair within **1 clean-accuracy point**. If two models tie on accuracy but sit far apart on threshold, the threshold measures something accuracy does not — regardless of what the overall correlation says.

### contrast

| model A | model B | Δacc (pts) | thr A | thr B | ratio | both measured? |
|---|---|---|---|---|---|---|
| deit_small_patch16_224 | mobilenetv3_large_100 | 0.54 | 0.05903 | 0.2304 | 3.90× | yes |
| swin_tiny_patch4_window7_224 | mobilenetv3_large_100 | 0.79 | 0.06314 | 0.2304 | 3.65× | yes |
| resnet101 | vit_base_patch16_224 | 0.33 | 0.01021 | 0.02478 | 2.43× | yes |
| resnet50 | swin_tiny_patch4_window7_224 | 0.28 | 0.04654 | 0.06314 | 1.36× | yes |
| resnet50 | deit_small_patch16_224 | 0.54 | 0.04654 | 0.05903 | 1.27× | yes |
| efficientnet_b0 | mobilenetv3_large_100 | 0.51 | 0.2011 | 0.2304 | 1.15× | yes |
| deit_small_patch16_224 | swin_tiny_patch4_window7_224 | 0.25 | 0.05903 | 0.06314 | 1.07× | yes |

**Sharpest fully-measured pair:** deit_small_patch16_224 (85.7%, thr 0.05903) vs mobilenetv3_large_100 (85.2%, thr 0.2304) — **Δacc 0.54 pts, 3.9× threshold gap**, both in-range.

### degradation

| model A | model B | Δacc (pts) | thr A | thr B | ratio | both measured? |
|---|---|---|---|---|---|---|
| deit_small_patch16_224 | mobilenetv3_large_100 | 0.54 | 0.167 | 0.06214 | 2.69× | yes |
| resnet101 | vit_base_patch16_224 | 0.33 | 0.1205 | 0.2654 | 2.20× | yes |
| swin_tiny_patch4_window7_224 | mobilenetv3_large_100 | 0.79 | 0.1255 | 0.06214 | 2.02× | yes |
| resnet50 | deit_small_patch16_224 | 0.54 | 0.09753 | 0.167 | 1.71× | yes |
| deit_small_patch16_224 | swin_tiny_patch4_window7_224 | 0.25 | 0.167 | 0.1255 | 1.33× | yes |
| resnet50 | swin_tiny_patch4_window7_224 | 0.28 | 0.09753 | 0.1255 | 1.29× | yes |
| efficientnet_b0 | mobilenetv3_large_100 | 0.51 | 0.06706 | 0.06214 | 1.08× | yes |

**Sharpest fully-measured pair:** deit_small_patch16_224 (85.7%, thr 0.167) vs mobilenetv3_large_100 (85.2%, thr 0.06214) — **Δacc 0.54 pts, 2.7× threshold gap**, both in-range.

**But read this against §2 before concluding anything.** On contrast, clean accuracy orders the models' thresholds almost perfectly (Spearman ρ=-0.903). Both facts are true simultaneously and they pull in opposite directions: *across* the full 75–90% accuracy range accuracy predicts threshold rank nearly deterministically, yet *within* a ~1-point accuracy band thresholds still differ several-fold. The threshold therefore adds little when comparing models far apart in accuracy, and adds real information when comparing models of matched accuracy — which is the practically interesting case, but is also the narrowest reading of the library's claim. The existence proof does not rescue the axis from the rank correlation; it narrows where the axis is useful.

## 2. Correlations (bootstrap CIs over models)

### contrast

| relationship | Pearson r [95% CI] | r² [95% CI] | Spearman ρ | r² CI excludes 0.7? |
|---|---|---|---|---|
| acc ↔ **threshold** (reliable n=10) — PRIMARY | -0.733 [-0.99, -0.53] | 0.537 [0.28, 0.98] | -0.903 | **no** |
| acc ↔ threshold (all n=10) | -0.733 [-0.99, -0.53] | 0.537 [0.28, 0.98] | -0.903 | **no** |
| acc ↔ log₁₀ threshold (reliable) | -0.756 [-0.96, -0.58] | 0.572 [0.33, 0.93] | -0.903 | **no** |
| acc ↔ slope (reliable) | 0.597 [0.53, 0.97] | 0.357 [0.28, 0.94] | 0.976 | **no** |
| threshold ↔ slope (reliable) | -0.619 [-0.92, -0.51] | 0.384 [0.26, 0.84] | -0.939 | **no** |

**Leverage check.** r²=0.54 on all reliable models; **0.47** without resnet101, **0.70** without mobilenetv3_large_100; without resnet18 (the low-accuracy outlier) it is **0.61**. The correlation does not hinge on any single model.

### degradation

| relationship | Pearson r [95% CI] | r² [95% CI] | Spearman ρ | r² CI excludes 0.7? |
|---|---|---|---|---|
| acc ↔ **threshold** (reliable n=10) — PRIMARY | 0.667 [0.03, 0.91] | 0.445 [0.0053, 0.83] | 0.467 | **no** |
| acc ↔ threshold (all n=10) | 0.667 [0.03, 0.91] | 0.445 [0.0053, 0.83] | 0.467 | **no** |
| acc ↔ log₁₀ threshold (reliable) | 0.839 [0.052, 0.97] | 0.704 [0.0079, 0.95] | 0.467 | **no** |
| acc ↔ slope (reliable) | 0.288 [-0.19, 0.77] | 0.0831 [0.00068, 0.6] | 0.333 | yes |
| threshold ↔ slope (reliable) | 0.769 [0.57, 0.97] | 0.592 [0.33, 0.93] | 0.915 | **no** |

**Leverage check.** r²=0.45 on all reliable models; **0.24** without resnet18, **0.51** without resnet101; without resnet18 (the low-accuracy outlier) it is **0.24**. A swing this large means the correlation rides on single points — leverage, not a robust relationship.


## 3. Verdict per suite

_Rule fixed in code before the numbers: r² < 0.3 → distinct axis; r² > 0.7 → kill; else intermediate. Reported as it landed._

### contrast: **INTERMEDIATE — partly, not wholly, redundant with accuracy  ⚠ CONFLICTS WITH SPEARMAN**

- **⚠ Pearson and Spearman DISAGREE, and the disagreement is the finding.** Linear r²=0.54 → *intermediate*; rank ρ=-0.903 → ρ²=0.82 → *KILL* (same bands applied in rank space). The relationship is strongly **monotonic but non-linear**: clean accuracy orders the models' contrast thresholds almost perfectly even though it does not predict their magnitude linearly. **Read as a ranking tool the contrast threshold is largely redundant with accuracy — that reading is at/over the kill line; read as a magnitude it is not.** The pre-committed rule keys on Pearson r², so the headline stays 'intermediate' — but that call alone would **overstate** the threshold's independence.
- Pearson r²=0.54 is in [0.3, 0.7]: accuracy explains part of the contrast threshold and leaves part unexplained.
- **n=10 cannot separate 'intermediate' from 'kill':** the r² bootstrap CI [0.28, 0.98] does NOT exclude 0.7. On the correlation alone this test is underpowered; the equal-accuracy pairs carry the argument by themselves.
- Leverage: r²=0.54 full; drops to 0.47 without resnet101, rises to 0.70 without mobilenetv3_large_100; without resnet18 it is 0.61.
- Existence proof: models within 1 accuracy point differ by up to 3.9× in contrast threshold (both measured).
- **Fit quality caveat: 3/10 contrast fits have R² < 0.90** against their own observed points (worst 0.83). `converged=True` and no at-bound parameter say the optimiser succeeded — they do NOT say the sigmoid describes the data. A threshold read off a curve that misses its points by up to 0.22 in P(correct) is a weaker quantity than its tight CI implies.

### degradation: **INTERMEDIATE — partly, not wholly, redundant with accuracy  ⚠ CONFLICTS WITH SPEARMAN**

- **⚠ Pearson and Spearman DISAGREE, and the disagreement is the finding.** Linear r²=0.45 → *intermediate*; rank ρ=0.467 → ρ²=0.22 → *distinct-axis* (same bands applied in rank space). Here the *linear* fit is the stronger of the two, which is the signature of a correlation carried by a few extreme points rather than by consistent ordering: accuracy does **not** reliably order the models' degradation thresholds (ρ=0.47). Taken with the leverage check below, the rank reading is the more trustworthy one, and it puts degradation **closer to a distinct axis** than Pearson suggests. The pre-committed rule keys on Pearson r², so the headline stays 'intermediate' — but that call alone probably **understates** the threshold's independence here.
- Pearson r²=0.45 is in [0.3, 0.7]: accuracy explains part of the degradation threshold and leaves part unexplained.
- **n=10 cannot separate 'intermediate' from 'kill':** the r² bootstrap CI [0.01, 0.83] does NOT exclude 0.7. On the correlation alone this test is underpowered; the equal-accuracy pairs carry the argument by themselves.
- Leverage: r²=0.45 full; drops to 0.24 without resnet18, rises to 0.51 without resnet101; without resnet18 it is 0.24.  The relationship depends heavily on single points — read it as leverage, not a robust relationship.

## 4. Slope — does it survive as an independent readout?

v1 found slope is *not* independent on contrast (r=0.85 vs accuracy, −0.80 vs threshold). Checking whether that holds on degradation:

| suite | slope↔accuracy r (r²) | slope↔threshold r (r²) | independent? |
|---|---|---|---|
| contrast | 0.597 (0.36), ρ=0.98 | -0.619 (0.38), ρ=-0.94 | no — tracks **both** |
| degradation | 0.288 (0.083), ρ=0.33 | 0.769 (0.59), ρ=0.92 | partly — tracks **threshold**, not accuracy |

_Slope is read on both coefficients too: on contrast the rank relations are near-deterministic (ρ=0.98 vs accuracy, −0.94 vs threshold), so slope is clearly not a separate readout there. On degradation slope is decoupled from accuracy (r=0.29) but still tracks its own threshold (r=0.77, ρ=0.92) — so it is not an independent third axis on either suite, though for different reasons._

## 4b. Fit quality — does the sigmoid actually describe the data?

`converged=True` with no at-bound parameter says the **optimiser** succeeded. It does not say the **functional form fits**. Computed from the persisted observed points (v1 could not have shown this — it kept no points):

| suite | model | R² (fit vs observed) | max residual | residual at lowest level |
|---|---|---|---|---|
| contrast | swin_tiny_patch4_window7_224 | 0.827 ⚠ | 0.182 | -0.156 |
| contrast | deit_small_patch16_224 | 0.838 ⚠ | 0.195 | -0.108 |
| contrast | resnet18 | 0.853 ⚠ | 0.220 | -0.046 |
| contrast | resnet50 | 0.901 | 0.209 | -0.056 |
| contrast | vit_small_patch16_224 | 0.903 | 0.210 | -0.025 |
| contrast | resnet101 | 0.909 | 0.147 | -0.089 |
| contrast | efficientnet_b0 | 0.935 | 0.188 | -0.009 |
| contrast | convnext_tiny | 0.948 | 0.140 | -0.022 |
| contrast | mobilenetv3_large_100 | 0.949 | 0.162 | -0.006 |
| contrast | vit_base_patch16_224 | 0.953 | 0.126 | -0.042 |
| degradation | resnet101 | 0.980 | 0.106 | +0.009 |
| degradation | vit_small_patch16_224 | 0.994 | 0.047 | -0.047 |
| degradation | deit_small_patch16_224 | 0.994 | 0.048 | -0.048 |
| degradation | efficientnet_b0 | 0.995 | 0.047 | -0.024 |
| degradation | swin_tiny_patch4_window7_224 | 0.996 | 0.027 | -0.012 |
| degradation | convnext_tiny | 0.997 | 0.034 | -0.002 |
| degradation | resnet18 | 0.997 | 0.033 | -0.024 |
| degradation | mobilenetv3_large_100 | 0.998 | 0.033 | -0.024 |
| degradation | vit_base_patch16_224 | 0.999 | 0.008 | -0.008 |
| degradation | resnet50 | 1.000 | 0.014 | -0.010 |

**The contrast (Weibull) fits are materially worse than the degradation (logistic) fits**: R² 0.83–0.95 vs 0.98–1.00, with residuals up to 0.22 in P(correct) vs 0.11. The residual at the lowest contrast level is **negative for every model** (-0.156 to -0.006): the Weibull systematically **over-predicts** performance at low contrast — its lower tail cannot fall fast enough to meet the data. This is a statement about the *model of the curve*, not about the models being measured.

This compounds §6: the two models whose thresholds moved most between v1 and v2 are also among the worst-fitting. Their thresholds are unstable **because the functional form does not describe their data**, so adding levels re-shapes the curve. A contrast threshold is therefore a weaker quantity than its narrow bootstrap CI suggests — the CI measures resampling noise around a curve that is itself the wrong shape. Fixing this (a better sigmoid family, or a free lower asymptote) is a more valuable next step than adding models.

## 5. Cross-suite — does contrast threshold predict degradation threshold?

Across the 10 models measured reliably on both suites: Pearson r=-0.675 [-0.89, -0.31] (r²=0.46), Spearman ρ=-0.588. The two robustness axes track each other — a model robust to low contrast tends to be robust to noise, so they are not independent readouts.

## 6. v1 vs v2 contrast thresholds (grid changed — not a strict replication)

v1 swept `geomspace(0.01, 0.5, 7)`; v2 sweeps `geomspace(0.001, 0.5, 10)` (step ratio 1.995 vs v1's 1.919). The grids differ, so a shift is not necessarily error — but a shift beyond the ~0.5% device drift seen in Test A means the threshold is sensitive to **grid placement**, which is worth knowing.

| model | v1 thr | v2 thr | change | v1 reliable | v2 reliable | material (>0.5%)? |
|---|---|---|---|---|---|---|
| resnet101 | 0.0005736 | 0.01021 | 1.68e+03% | no | yes | no |
| deit_small_patch16_224 | 0.02269 | 0.05903 | 160% | yes | yes | **YES** |
| swin_tiny_patch4_window7_224 | 0.02596 | 0.06314 | 143% | yes | yes | **YES** |
| convnext_tiny | 0.06253 | 0.03934 | 37.1% | yes | yes | **YES** |
| resnet18 | 0.2363 | 0.2088 | 11.6% | yes | yes | **YES** |
| vit_base_patch16_224 | 0.02785 | 0.02478 | 11% | yes | yes | **YES** |
| resnet50 | 0.05208 | 0.04654 | 10.6% | yes | yes | **YES** |
| vit_small_patch16_224 | 0.1917 | 0.1826 | 4.72% | yes | yes | **YES** |
| efficientnet_b0 | 0.2076 | 0.2011 | 3.1% | yes | yes | **YES** |
| mobilenetv3_large_100 | 0.2346 | 0.2304 | 1.81% | yes | yes | **YES** |

**Materially shifted (9): resnet18, resnet50, efficientnet_b0, convnext_tiny, vit_small_patch16_224, vit_base_patch16_224, deit_small_patch16_224, swin_tiny_patch4_window7_224, mobilenetv3_large_100.** These moved far more than device drift, so their contrast threshold depends on where the level grid is placed — a real caveat on the threshold as a stable quantity.

## 7. Per-model results

### contrast

| model | family | params (M) | clean top-1 | threshold | 95% CI | slope | reliable? |
|---|---|---|---|---|---|---|---|
| resnet101 | CNN | 44.5 | 89.83% | 0.01021 | [0.009971, 0.01047] | 36.84 | yes |
| vit_base_patch16_224 | ViT | 86.6 | 89.50% | 0.02478 | [0.02416, 0.02541] | 14.76 | yes |
| convnext_tiny | CNN | 28.6 | 88.18% | 0.03934 | [0.03838, 0.0403] | 9.721 | yes |
| resnet50 | CNN | 25.6 | 86.27% | 0.04654 | [0.04528, 0.04781] | 6.02 | yes |
| swin_tiny_patch4_window7_224 | hybrid | 28.3 | 85.99% | 0.06314 | [0.06072, 0.06577] | 2.19 | yes |
| deit_small_patch16_224 | ViT | 22.1 | 85.73% | 0.05903 | [0.05684, 0.06115] | 2.609 | yes |
| mobilenetv3_large_100 | CNN | 5.48 | 85.20% | 0.2304 | [0.2247, 0.2362] | 1.519 | yes |
| efficientnet_b0 | CNN | 5.29 | 84.69% | 0.2011 | [0.1961, 0.2064] | 1.616 | yes |
| vit_small_patch16_224 | ViT | 22.1 | 82.27% | 0.1826 | [0.1774, 0.188] | 1.431 | yes |
| resnet18 | CNN | 11.7 | 75.41% | 0.2088 | [0.2021, 0.2162] | 1.024 | yes |

### degradation

| model | family | params (M) | clean top-1 | threshold | 95% CI | slope | reliable? |
|---|---|---|---|---|---|---|---|
| resnet101 | CNN | 44.5 | 89.83% | 0.1205 | [0.115, 0.1261] | -1.616 | yes |
| vit_base_patch16_224 | ViT | 86.6 | 89.50% | 0.2654 | [0.2578, 0.2735] | -0.8519 | yes |
| convnext_tiny | CNN | 28.6 | 88.18% | 0.1232 | [0.1174, 0.1288] | -1.484 | yes |
| resnet50 | CNN | 25.6 | 86.27% | 0.09753 | [0.09189, 0.103] | -1.65 | yes |
| swin_tiny_patch4_window7_224 | hybrid | 28.3 | 85.99% | 0.1255 | [0.1191, 0.132] | -1.27 | yes |
| deit_small_patch16_224 | ViT | 22.1 | 85.73% | 0.167 | [0.1612, 0.173] | -1.381 | yes |
| mobilenetv3_large_100 | CNN | 5.48 | 85.20% | 0.06214 | [0.05787, 0.06628] | -2.686 | yes |
| efficientnet_b0 | CNN | 5.29 | 84.69% | 0.06706 | [0.06248, 0.07151] | -2.347 | yes |
| vit_small_patch16_224 | ViT | 22.1 | 82.27% | 0.1242 | [0.1177, 0.1307] | -1.249 | yes |
| resnet18 | CNN | 11.7 | 75.41% | 0.01579 | [0.01017, 0.02118] | -1.87 | yes |

## Limitations

- **Accuracy range.** Clean top-1 spans 14.4 points (75–90%): these 1000-way models sit well below ceiling on Imagenette's fine-grained classes, so the correlation is **not** suppressed by range restriction.
- **n = 10 models** is a small sample for a correlation; the r² CIs are wide and, where they fail to exclude 0.7, the correlation alone cannot separate 'intermediate' from 'kill'.
- **Capacity and family are confounded** — the model set varies architecture and parameter count together, so neither can be credited alone.
- **Single dataset, 10 classes** (Imagenette val); 1000-way argmax scored on a 10-class subset.
- **Distractor suite absent by design:** there is no Test A split-half reliability data for it, so its thresholds are of unknown stability and were deliberately not measured here.

_Generated 2026-07-16T12:34:26.644065+00:00 · psyvis_ml 0.0.1 · torch 2.6.0+cu124 · timm 1.0.27 · device cuda:NVIDIA GeForce RTX 4070 SUPER · wall-clock 61.6 min · every figure regenerable from `curves.json` with `--regen`._
