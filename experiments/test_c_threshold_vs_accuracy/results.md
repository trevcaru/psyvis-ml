# Test C — is the contrast threshold a new axis, or an expensive accuracy proxy?

**Kill-criterion check.** The library's bet is that a psychometric threshold tells you something accuracy does not. Test A showed the threshold is *repeatable*; this asks whether it is *informative*. Method: measure real pretrained timm models on the real contrast sweep, record each model's clean top-1 and fitted contrast threshold, and correlate.

Real Imagenette val · **3925 images** (full, no split) · **10 models** (9 with an in-range measured threshold) · contrast levels geomspace(0.01, 0.5, 7) (weibull) · device cuda:NVIDIA GeForce RTX 4070 SUPER · bootstrap n=2000.

Every number traces to a real forward pass of a real pretrained model on a real image file. No synthetic data, no fallbacks. Models that failed to resolve are dropped, not substituted.

## VERDICT: INTERMEDIATE — threshold is partly, not wholly, redundant with accuracy

- Pearson r²=0.48 (reliable models) is in [0.3, 0.7]: accuracy explains about half the threshold variance and leaves about half unexplained. Neither a clean vindication (r²<0.3) nor a kill (r²>0.7).
- Robustness: r²=0.54 including the extrapolated model, 0.40 on log-threshold — the call is stable across all three.
- The unexplained half is real signal, not noise: models within 2 accuracy points differ by up to 10× in threshold (existence proof — see the equal-accuracy pairs).
- 1 model(s) are so contrast-robust their 75% point sits BELOW the swept floor (0.01) — threshold extrapolated/at-bound, excluded from the primary fit. Re-run with a lower contrast floor (e.g. 0.003) to measure them and firm up the correlation.
- Accuracy spread is 14.4 points (models sit at 75–90%, not near ceiling), so the correlation is not suppressed by range restriction.

_Verdict rule fixed before looking: r² < 0.3 → distinct axis; r² > 0.7 → kill; 0.3–0.7 → intermediate. Not spun — reported as it landed._

## Per-model results

| model | family | params (M) | clean top-1 | contrast threshold | 95% CI | slope | reliable? |
|---|---|---|---|---|---|---|---|
| resnet101 | CNN | 44.5 | 89.83% | 0.0005736 | [0.0005351, 0.0009047] | 53.14 | **no** (alpha/extrap) |
| vit_base_patch16_224 | ViT | 86.6 | 89.50% | 0.02785 | [0.02581, 0.02983] | 3.209 | yes |
| convnext_tiny | CNN | 28.6 | 88.18% | 0.06253 | [0.05984, 0.06505] | 2.094 | yes |
| resnet50 | CNN | 25.6 | 86.27% | 0.05208 | [0.04899, 0.05528] | 1.726 | yes |
| swin_tiny_patch4_window7_224 | hybrid | 28.3 | 85.99% | 0.02596 | [0.02233, 0.02957] | 1.824 | yes |
| deit_small_patch16_224 | ViT | 22.1 | 85.73% | 0.02269 | [0.01981, 0.0257] | 2.284 | yes |
| mobilenetv3_large_100 | CNN | 5.48 | 85.20% | 0.2346 | [0.2288, 0.2404] | 1.386 | yes |
| efficientnet_b0 | CNN | 5.29 | 84.69% | 0.2076 | [0.2019, 0.2133] | 1.378 | yes |
| vit_small_patch16_224 | ViT | 22.1 | 82.27% | 0.1917 | [0.185, 0.1983] | 1.069 | yes |
| resnet18 | CNN | 11.7 | 75.41% | 0.2363 | [0.2246, 0.2477] | 0.6034 | yes |

**Extrapolated / excluded from the primary fit:** resnet101 — so contrast-robust that the 75% point sits *below* the swept floor (0.01); the fit pins a parameter at its bound and the reported threshold is an extrapolation, not a measurement. This is itself a finding: the most robust models are off the bottom of the gallery's contrast range.

## Correlations (bootstrap CIs over models)

| relationship | Pearson r [95% CI] | r² | Spearman ρ [95% CI] |
|---|---|---|---|
| clean acc ↔ **threshold** (reliable, n=9) — PRIMARY | -0.691 [-0.94, -0.44] | 0.477 | -0.65 [-0.85, 0.15] |
| clean acc ↔ threshold (all n=10, incl. extrapolated) | -0.736 [-0.95, -0.51] | 0.541 | -0.745 [-0.9, -0.067] |
| clean acc ↔ log₁₀ threshold (reliable) | -0.636 [-0.89, -0.36] | 0.404 | -0.65 [-0.85, 0.15] |
| clean acc ↔ slope (reliable) | 0.85 [0.7, 0.97] | 0.722 | 0.883 [0.37, 1] |
| threshold ↔ slope (reliable) | -0.803 [-0.95, -0.63] | 0.645 | -0.817 [-0.92, -0.32] |

The primary correlation uses only reliably-measured (in-range, converged, not-at-bound) thresholds; the all-models and log variants are robustness checks. Both coefficients are reported — Spearman matters because the relationship need not be linear. With only 9–10 models the CIs are wide; read them as such.

## The residual question — models that tie on accuracy but differ on threshold

Pairs within 2 accuracy points of each other, ranked by threshold ratio. A pair that ties on accuracy yet sits far apart on threshold is the existence proof that the threshold is a separate readout:

| model A | model B | Δacc (pts) | thr A | thr B | thr ratio | both measured? |
|---|---|---|---|---|---|---|
| resnet101 | convnext_tiny | 1.66 | 0.0005736 | 0.06253 | 109.02× | **no** (extrapolated) |
| resnet101 | vit_base_patch16_224 | 0.33 | 0.0005736 | 0.02785 | 48.55× | **no** (extrapolated) |
| deit_small_patch16_224 | mobilenetv3_large_100 | 0.54 | 0.02269 | 0.2346 | 10.34× | yes |
| efficientnet_b0 | deit_small_patch16_224 | 1.04 | 0.2076 | 0.02269 | 9.15× | yes |
| swin_tiny_patch4_window7_224 | mobilenetv3_large_100 | 0.79 | 0.02596 | 0.2346 | 9.04× | yes |
| efficientnet_b0 | swin_tiny_patch4_window7_224 | 1.30 | 0.2076 | 0.02596 | 7.99× | yes |
| resnet50 | mobilenetv3_large_100 | 1.07 | 0.05208 | 0.2346 | 4.51× | yes |
| resnet50 | efficientnet_b0 | 1.58 | 0.05208 | 0.2076 | 3.99× | yes |

**Sharpest fully-measured pair:** deit_small_patch16_224 (85.7%, thr 0.02269) vs mobilenetv3_large_100 (85.2%, thr 0.2346) — Δacc 0.54 pts but a **10.3× threshold gap**, both thresholds in-range. This is the concrete demonstration that the threshold carries information clean accuracy does not, and is why the verdict is not a kill despite a moderate overall r².

(Pairs involving resnet101 show even larger ratios — e.g. 109× — but one threshold there is an *extrapolated lower bound*, so read those as "≥", not measured.)

## Slope as a third axis — does it survive as independent?

Slope↔accuracy: Pearson r=0.85 (r²=0.72). Slope↔threshold: r=-0.803 (r²=0.64). **Slope is NOT an independent third axis here:** on the reliably-measured models it correlates strongly with both accuracy and threshold (steeper curves go with higher accuracy and lower thresholds). Stated plainly against the hopeful prior — in this dataset slope tracks the same gradient, so it does not add a clean separate dimension. (The naive all-model slope correlation looks weaker only because the extrapolated model's degenerate at-bound slope is an outlier.)

## Range-restriction caveat (handled up front, not buried)

Clean top-1 spans **14.4 points** (75.4%–89.8%). These are ImageNet-1k models doing 1000-way argmax on Imagenette's fine-grained classes, so they sit well below ceiling — the accuracy axis is *not* compressed into the 95–99% band that would mechanically suppress a correlation. A moderate r² here is therefore a real result, not a range-restriction artifact.

**What would sharpen this** (verdict is intermediate): more models over a wider capacity range; a **lower contrast floor** (≈0.003) so the most robust models (resnet101) yield a *measured* rather than extrapolated threshold; and a harder dataset to widen the accuracy axis further.

_Limitations: single dataset (Imagenette, 10 classes); 1000-way accuracy on a 10-class subset; ~10 models is a small n for correlation CIs; contrast suite only._

_Generated 2026-07-15T23:57:27.863401+00:00 · psyvis_ml 0.0.1 · torch 2.6.0+cu124 · timm 1.0.27 · device cuda:NVIDIA GeForce RTX 4070 SUPER · wall-clock 27.9 min._
