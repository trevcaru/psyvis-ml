# Poster v3 — 10-model contrast curves, model-free threshold

Descriptive only: no correlation, no claim. Read-only on data — no sweep, no forward pass, no refit.

Source: Test C v2 per-item CSVs (`experiments/test_c_threshold_vs_accuracy_v2/per_item/contrast__*.csv`), 10 models, n = 3925 images × 10 levels `geomspace(0.001, 0.5, 10)`. Recounts match `curves.json` exactly.

## Method

- **Points**: observed accuracy per contrast level.
- **Lines**: monotone cubic (PCHIP) interpolation through the points in log₁₀ contrast. Not a fitted model; drawn to guide the eye.
- **Model-free threshold**: ceiling = max observed accuracy; criterion = chance + 0.5 × (ceiling − chance), chance = 0.001; threshold = contrast at the first upward crossing of the criterion, by linear interpolation in log₁₀ contrast between the two bracketing levels.
- **95% CI**: percentile bootstrap over images, 2000 resamples (seed 12345). An image keeps all 10 of its levels when drawn; ceiling, criterion and threshold are recomputed in every resample. The same image resamples are used for every model.
- The diamond sits at (threshold, criterion). Because the threshold is linear in log contrast while the drawn line is PCHIP, a diamond can sit slightly off its line: max vertical offset **0.64 accuracy points**.

## Model-free thresholds (sorted)

| model | clean top-1 | ceiling (max obs.) | criterion | model-free threshold | 95% CI | crossings |
|---|---|---|---|---|---|---|
| ResNet-101 | 89.8% | 0.897 | 0.449 | 0.003474 | [0.003407, 0.003548] | 1 |
| Swin-Tiny ★ | 86.0% | 0.858 | 0.430 | 0.005056 | [0.004904, 0.005203] | 1 |
| DeiT-Small | 85.7% | 0.854 | 0.427 | 0.006148 | [0.006012, 0.00629] | 1 |
| ViT-Base | 89.5% | 0.896 | 0.448 | 0.008359 | [0.008068, 0.008652] | 1 |
| ResNet-50 ★ | 86.3% | 0.862 | 0.431 | 0.01007 | [0.009814, 0.01033] | 1 |
| ConvNeXt-Tiny | 88.2% | 0.881 | 0.441 | 0.01313 | [0.01281, 0.01348] | 1 |
| ResNet-18 | 75.4% | 0.750 | 0.376 | 0.01969 | [0.01912, 0.02028] | 1 |
| ViT-Small | 82.3% | 0.810 | 0.406 | 0.028 | [0.02721, 0.02888] | 1 |
| EfficientNet-B0 | 84.7% | 0.839 | 0.420 | 0.04714 | [0.04593, 0.04841] | 1 |
| MobileNetV3-Large ★ | 85.2% | 0.838 | 0.419 | 0.06216 | [0.06023, 0.0644] | 1 |

★ = highlighted in the figure. Crossings = number of times the observed points cross the criterion (1 = unambiguous). Failed bootstrap resamples: 0 of 20000.

## Weibull goodness of fit (documenting the misfit)

The Weibull drawn on poster v1 (guess = chance fixed, lapse free; parameters read from `outputs/poster_10model_curves/fits.json`, not refit), tested against the observed counts: deviance G² = 2 Σ [k ln(k / nπ) + (n − k) ln((n − k) / n(1 − π))], χ² on 10 levels − 3 free parameters = 7 df.

| model | deviance G² | df | p | max \|residual\| |
|---|---|---|---|---|
| ResNet-101 | 335.2 | 7 | 1.8e-68 | 0.087 |
| Swin-Tiny | 246.3 | 7 | 1.7e-49 | 0.052 |
| DeiT-Small | 262.2 | 7 | 7.1e-53 | 0.063 |
| ViT-Base | 355.1 | 7 | 9.9e-73 | 0.060 |
| ResNet-50 | 304.2 | 7 | 7.8e-62 | 0.065 |
| ConvNeXt-Tiny | 260.2 | 7 | 1.9e-52 | 0.059 |
| ResNet-18 | 206.0 | 7 | 6.3e-41 | 0.054 |
| ViT-Small | 294.3 | 7 | 1e-59 | 0.083 |
| EfficientNet-B0 | 294.4 | 7 | 9.5e-60 | 0.061 |
| MobileNetV3-Large | 138.8 | 7 | 9.1e-27 | 0.052 |

Caveats on reading p: the test assumes independent binomial counts at each level. With 3925 trials per level it has power to reject small absolute misfits, and the same images recur across levels, so the levels are not independent samples. The p-values document that the Weibull is rejected as an exact description of the data; the max residual column gives the size of the misfit in accuracy units.

## Threshold-marker overlap check

Diamond pairs closer than 1.5× touching distance in the rendered figure:

| pair | centre distance (px) | touching distance (px) | overlap? |
|---|---|---|---|
| EfficientNet-B0 / MobileNetV3-Large | 151.7 | 102.5 | no |
| ConvNeXt-Tiny / ResNet-50 | 146.8 | 102.5 | no |
| ViT-Base / ResNet-50 | 106.5 | 102.5 | no |
| DeiT-Small / Swin-Tiny | 107.3 | 102.5 | no |
