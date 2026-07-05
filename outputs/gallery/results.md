# psyvis-ml results gallery

Real Imagenette val · 200 images · models: resnet50, vit_small_patch16_224.

| suite | model | threshold | 95% CI | slope | conf. threshold (margin=0) | margin slope |
|---|---|---|---|---|---|---|
| contrast | resnet50 | 0.04971 | [0.03866, 0.0631] | 1.957 | n/a | n/a |
| contrast | vit_small_patch16_224 | 0.1955 | [0.1688, 0.2305] | 1.008 | 0.03401 | 99.59 |
| degradation | resnet50 | 0.08669 | [0.06454, 0.1111] | -1.557 | 0.2613 | -31.81 |
| degradation | vit_small_patch16_224 | 0.1226 | [0.09189, 0.1537] | -1.172 | 0.3162 | -15.93 |
| distractor | resnet50 | 53.1 | [50.88, 55.41] | -0.01409 | 75.47 | -0.508 |
| distractor | vit_small_patch16_224 | 38.64 | [36.12, 41.22] | -0.0112 | 65.18 | -0.2878 |

Figures in this folder: `<suite>_accuracy_<model>.png` (accuracy psychometric curve), `<suite>_confidence_delta.png` (baseline-relative Δ-margin), `contrast_human_vs_models.png` / `<suite>_comparison.png` (multi-model + Δ-margin panels). The human overlay is grating-detection sensitivity vs. argmax classification — different paradigms on a shared axis (see the figure caption).

**Distractor suite (size sweep).** A big centred target (176 px, ~79% of the frame) clears the recognition ceiling; four heterogeneous distractors grow inward from the margins and the sweep is over distractor **size**. Performance falls as size grows (a decreasing suite), giving a **real, non-extrapolated threshold** — the distractor size at which accuracy hits criterion — over a band gated by `calibrate_distractor_size`. A larger threshold means a more distractor-robust model.
