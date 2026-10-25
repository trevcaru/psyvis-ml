# Poster — 10-model contrast curves, corrected instrument

Descriptive only: no correlation, no claim. Read-only on data — no sweep, no forward pass.

Source: `experiments/test_c_threshold_vs_accuracy_v2/curves.json`, contrast suite, 10 models, n = 3925 images per level, 10 levels `geomspace(0.001, 0.5, 10)`. Counts re-verified against the per-item CSVs (exact match, every model, every level).

Fit: Weibull · guess = chance (0.001) fixed · lapse estimated · threshold at the midpoint criterion `guess + 0.5 × (ceiling − guess)`, ceiling = 1 − lapse. 95% CI: joint parametric bootstrap, 2000 resamples (seed 12345), ceiling and criterion recomputed in every resample.

Figure: `poster_10model_curves.png` — all 10 models drawn (points + fitted curve); ResNet-50, ViT-Small and ResNet-101 highlighted in colour with their threshold marked (◆ at threshold × criterion, dashed drop line to the x-axis); the other seven in grey.

## Thresholds (sorted by threshold)

| model | ceiling [95% CI] | criterion | threshold | 95% CI | fit R² | max resid |
|---|---|---|---|---|---|---|
| ResNet-101 ★ | 0.870 [0.866, 0.874] | 0.435 | 0.003587 | [0.003535, 0.00364] | 0.990 | 0.087 |
| Swin-Tiny | 0.818 [0.813, 0.823] | 0.410 | 0.005238 | [0.005096, 0.005381] | 0.989 | 0.052 |
| DeiT-Small | 0.843 [0.838, 0.849] | 0.422 | 0.00656 | [0.006391, 0.006732] | 0.990 | 0.063 |
| ViT-Base | 0.877 [0.872, 0.881] | 0.439 | 0.009085 | [0.008899, 0.009271] | 0.993 | 0.060 |
| ResNet-50 ★ | 0.837 [0.832, 0.842] | 0.419 | 0.01065 | [0.01045, 0.01085] | 0.993 | 0.065 |
| ConvNeXt-Tiny | 0.857 [0.852, 0.862] | 0.429 | 0.01399 | [0.0137, 0.01426] | 0.996 | 0.059 |
| ResNet-18 | 0.726 [0.719, 0.733] | 0.364 | 0.02057 | [0.0201, 0.02106] | 0.993 | 0.054 |
| ViT-Small ★ | 0.765 [0.759, 0.773] | 0.383 | 0.0289 | [0.02824, 0.02958] | 0.991 | 0.083 |
| EfficientNet-B0 | 0.791 [0.783, 0.798] | 0.396 | 0.04785 | [0.04692, 0.04882] | 0.992 | 0.061 |
| MobileNetV3-Large | 0.810 [0.801, 0.819] | 0.405 | 0.06694 | [0.06527, 0.06849] | 0.995 | 0.052 |

★ = highlighted in the figure. Criterion and ceiling are proportions correct; threshold is RMS contrast.

All fits converged: True; parameters at bound: none; models with any failed bootstrap resample: 0.

## Reproduction check

These settings are variant (e) of `criterion_scoping.md` §1, which reported midpoint thresholds (4 s.f.) from the same counts. Max relative difference from those values: **0.033%** (ResNet-50).
