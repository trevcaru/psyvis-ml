# psyvis-ml results gallery

Real Imagenette val · 200 images · models: resnet50, vit_small_patch16_224.

| suite | model | threshold | 95% CI | slope | conf. threshold (margin=0) | margin slope |
|---|---|---|---|---|---|---|
| contrast | resnet50 | 0.04971 | [0.03866, 0.0631] | 1.957 | n/a | n/a |
| contrast | vit_small_patch16_224 | 0.1955 | [0.1688, 0.2305] | 1.008 | 0.03401 | 99.59 |
| degradation | resnet50 | 0.08669 | [0.06454, 0.1111] | -1.557 | 0.2613 | -31.81 |
| degradation | vit_small_patch16_224 | 0.1226 | [0.09189, 0.1537] | -1.172 | 0.3162 | -15.93 |
| distractor | resnet50 | 90 (extrapolated) | [86.36, 94.58] | 0.02862 | n/a | n/a |
| distractor | vit_small_patch16_224 | 90.3 (extrapolated) | [87.03, 94.86] | 0.03623 | n/a | n/a |

Figures in this folder: `<suite>_accuracy_<model>.png` (accuracy psychometric curve), `<suite>_confidence_delta.png` (baseline-relative Δ-margin), `contrast_human_vs_models.png` / `<suite>_comparison.png` (multi-model + Δ-margin panels). The human overlay is grating-detection sensitivity vs. argmax classification — different paradigms on a shared axis (see the figure caption).

**Distractor caveat.** Distractor-robustness on whole-photo ImageNet targets is ceiling-limited: a centred target large enough to recognise leaves little room to move a distractor clear on a 224-px input, so the accuracy threshold is often extrapolated. Read the Δ-margin gradient as the robustness signal (consistent with the project's Q1/window diagnostics).
