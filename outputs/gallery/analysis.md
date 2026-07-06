# psyvis-ml gallery — analysis

Real Imagenette val · 200 images · models: resnet50, vit_small_patch16_224.

## How to read this (deterministic models)

These models are **deterministic**: the same image at the same stimulus level always gives the same logits, so there is **no trial-level sampling noise** and a frequentist t-test / p-value on "trials" would be fabricated. All uncertainty here comes from **bootstrapping over the image set** — the only thing that varies is *which images* you measured. A difference between two models is called **reliable** when the bootstrap CI of the *paired* difference (both models recomputed on the **same** resampled images each iteration) **excludes zero**. We report no p-values.

## contrast

| metric | resnet50 | vit_small_patch16_224 |
|---|---|---|
| threshold | 0.04971 | 0.1955 |
| threshold 95% CI | [0.03103, 0.08536] | [0.1438, 0.2644] |
| slope | 1.957 | 1.008 |
| slope 95% CI | [1.008, 3.666] | [0.6985, 1.511] |
| fit converged | True | True |
| fit R² | 0.719 | 0.845 |
| at-bound params | — | — |
| margin slope | n/a | 99.6 |
| Δ-margin (clean→worst) | -8.06 | -6.91 |

**Paired threshold difference** (resnet50 − vit_small_patch16_224): **-0.1458**, 95% CI [-0.1954, -0.1069] → **reliable** (CI excludes 0).

**Confidence divergence** — Δ-margin (clean→worst) difference (resnet50 − vit_small_patch16_224): -1.16, 95% CI [-1.87, -0.435] → the Δ-margin curves **diverge** (CI excludes 0).

## degradation

| metric | resnet50 | vit_small_patch16_224 |
|---|---|---|
| threshold | 0.08669 | 0.1226 |
| threshold 95% CI | [0.05479, 0.1181] | [0.0793, 0.1648] |
| slope | -1.557 | -1.172 |
| slope 95% CI | [-1.756, -1.404] | [-1.333, -1.043] |
| fit converged | True | True |
| fit R² | 0.999 | 0.997 |
| at-bound params | — | — |
| margin slope | -31.8 | -15.9 |
| Δ-margin (clean→worst) | -13.8 | -8.7 |

**Paired threshold difference** (resnet50 − vit_small_patch16_224): **-0.03588**, 95% CI [-0.06708, -0.003973] → **reliable** (CI excludes 0).

**Confidence divergence** — Δ-margin (clean→worst) difference (resnet50 − vit_small_patch16_224): -5.15, 95% CI [-5.89, -4.39] → the Δ-margin curves **diverge** (CI excludes 0).

## distractor

| metric | resnet50 | vit_small_patch16_224 |
|---|---|---|
| threshold | 53.1 | 38.64 |
| threshold 95% CI | [46.31, 60.49] | [32.72, 44.1] |
| slope | -0.01409 | -0.0112 |
| slope 95% CI | [-0.01945, -0.01141] | [-0.01284, -0.009901] |
| fit converged | True | True |
| fit R² | 0.917 | 0.947 |
| at-bound params | — | — |
| margin slope | -0.508 | -0.288 |
| Δ-margin (clean→worst) | -16.9 | -9.62 |

**Paired threshold difference** (resnet50 − vit_small_patch16_224): **14.46**, 95% CI [8.541, 20.87] → **reliable** (CI excludes 0).

**Confidence divergence** — Δ-margin (clean→worst) difference (resnet50 − vit_small_patch16_224): -7.33, 95% CI [-8.08, -6.57] → the Δ-margin curves **diverge** (CI excludes 0).

## Summary

| suite | resnet50 thr | vit_small_patch16_224 thr | diff [95% CI] | reliable? | R² (resnet50/vit_small_patch16_224) |
|---|---|---|---|---|---|
| contrast | 0.04971 | 0.1955 | -0.1458 [-0.1954, -0.1069] | yes | 0.719 / 0.845 |
| degradation | 0.08669 | 0.1226 | -0.03588 [-0.06708, -0.003973] | yes | 0.999 / 0.997 |
| distractor | 53.1 | 38.64 | 14.46 [8.541, 20.87] | yes | 0.917 / 0.947 |
