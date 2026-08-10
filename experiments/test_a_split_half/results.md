# Test A — threshold split-half reliability

**Kill-criterion check, not a feature.** Is a measured threshold a property of the model, or an artifact of which images we swept? Method: randomly split the real Imagenette val set into two disjoint halves, run the *identical* sweep on each, compare thresholds + bootstrap CIs, over several split seeds.

Real Imagenette val · **3925 images** · halves ≈ 1962/1963 · models: resnet50, vit_small_patch16_224 · weights: timm pretrained ImageNet-1k · split seeds: 0, 1, 2, 3, 4 · bootstrap n=2000.

Every number below traces to a real forward pass of a real pretrained model on a real image file. No synthetic data, no fallbacks.

## VERDICT: MIXED — PASS (contrast) / AMBIGUOUS (degradation)

- the model ranking is robust on all suites (ranking held 20/20); the *magnitude* is a reproducible property for contrast but item-sampling-fragile for degradation.
- **contrast: PASS** — ranking held 10/10; worst split-half |Δ| is only 12% of the between-model gap; CIs overlap in 80% of comparisons
- **degradation: AMBIGUOUS** — ranking held 10/10 (direction robust), but split-half |Δ| reaches 67% of the between-model gap (median 26%): the magnitude is item-sampling-fragile at this n

_Not spun. This test exists to potentially kill a claim; the call above is reported as the numbers land, per-suite, without averaging a fragility away._

**What would resolve the AMBIGUOUS suite(s) (degradation).** Split-half |Δ| shrinks ~1/√n_half, so tightening worst-case noise from ~67% of the gap down to a comfortable <⅓ needs roughly **4× the images per half** (≈8k/half, ≈16k total — a larger draw than Imagenette's 3925, e.g. the full ImageNet-1k val). Part of the fragility is intrinsic: the degradation between-model gap (0.0359) is genuinely small, so at any feasible n it should be reported *with* this caveat, not as a crisp threshold separation.

## Per-comparison table

| model | suite | seed | thr_A | CI_A | thr_B | CI_B | Δ | CIs overlap? | Δ CI excludes 0? |
|---|---|---|---|---|---|---|---|---|---|
| resnet50 | contrast | 0 | 0.04895 | [0.04451, 0.05291] | 0.05539 | [0.05055, 0.06022] | -0.006438 | yes | **yes** |
| vit_small_patch16_224 | contrast | 0 | 0.1919 | [0.1827, 0.2014] | 0.1914 | [0.1824, 0.2009] | 0.0005793 | yes | no |
| resnet50 | degradation | 0 | 0.09487 | [0.08722, 0.1027] | 0.09855 | [0.09091, 0.1064] | -0.003681 | yes | no |
| vit_small_patch16_224 | degradation | 0 | 0.1172 | [0.1077, 0.1269] | 0.1257 | [0.1167, 0.1348] | -0.00847 | yes | no |
| resnet50 | contrast | 1 | 0.05783 | [0.0531, 0.06279] | 0.04693 | [0.0429, 0.05083] | 0.0109 | **no** | **yes** |
| vit_small_patch16_224 | contrast | 1 | 0.198 | [0.1883, 0.2079] | 0.1855 | [0.1768, 0.1947] | 0.01256 | yes | no |
| resnet50 | degradation | 1 | 0.0898 | [0.08211, 0.09796] | 0.1045 | [0.09705, 0.1119] | -0.01467 | yes | **yes** |
| vit_small_patch16_224 | degradation | 1 | 0.1163 | [0.1066, 0.1261] | 0.1283 | [0.1192, 0.1376] | -0.01199 | yes | no |
| resnet50 | contrast | 2 | 0.04744 | [0.04315, 0.05142] | 0.05721 | [0.05254, 0.06223] | -0.009768 | **no** | **yes** |
| vit_small_patch16_224 | contrast | 2 | 0.1827 | [0.1739, 0.1918] | 0.2008 | [0.1912, 0.2106] | -0.0181 | yes | **yes** |
| resnet50 | degradation | 2 | 0.09925 | [0.09159, 0.107] | 0.09492 | [0.08752, 0.1029] | 0.00433 | yes | no |
| vit_small_patch16_224 | degradation | 2 | 0.1345 | [0.1252, 0.1439] | 0.1105 | [0.101, 0.12] | 0.02397 | **no** | **yes** |
| resnet50 | contrast | 3 | 0.05293 | [0.04862, 0.0573] | 0.05115 | [0.04663, 0.05559] | 0.001783 | yes | no |
| vit_small_patch16_224 | contrast | 3 | 0.1833 | [0.1748, 0.192] | 0.2005 | [0.1905, 0.2111] | -0.01723 | yes | **yes** |
| resnet50 | degradation | 3 | 0.09398 | [0.08616, 0.1015] | 0.09599 | [0.08811, 0.1037] | -0.002014 | yes | no |
| vit_small_patch16_224 | degradation | 3 | 0.1283 | [0.1196, 0.1372] | 0.1167 | [0.1076, 0.1263] | 0.01159 | yes | no |
| resnet50 | contrast | 4 | 0.05556 | [0.0508, 0.06032] | 0.04892 | [0.04487, 0.05325] | 0.006646 | yes | no |
| vit_small_patch16_224 | contrast | 4 | 0.1986 | [0.1892, 0.2085] | 0.185 | [0.1765, 0.1939] | 0.01357 | yes | **yes** |
| resnet50 | degradation | 4 | 0.09048 | [0.08229, 0.09837] | 0.1006 | [0.09313, 0.1083] | -0.0101 | yes | no |
| vit_small_patch16_224 | degradation | 4 | 0.1186 | [0.1099, 0.1282] | 0.1256 | [0.1166, 0.135] | -0.007022 | yes | no |

## Verdicts

**1. Overlap rate (headline).** The two halves' 95% CIs overlap in **85%** of the 20 evaluable comparisons (of 20 total). High overlap ⇒ a threshold measured on one random half is statistically consistent with the other half.

**2. Split-half |Δ| vs. the real between-model effect.** The gallery's between-model gap is the effect the library claims to detect (contrast 0.146, degradation 0.0359). Split-half |Δ| as a fraction of that gap:

| suite | between-model gap | median \|Δ\| / gap | max \|Δ\| / gap |
|---|---|---|---|
| contrast | 0.146 | 7% | 12% |
| degradation | 0.0359 | 26% | 67% |

_If within-model split-half noise is comparable to the between-model effect (ratio → 1), the effect is not measurable at this n — that is the kill._

**3. Ranking stability.** Does resnet50 < vit_small (lower threshold) hold on every half, every seed? Held on **20 / 20** half×seed×suite cells.

**4. Paired Δ CI.** Bootstrapping the split-half difference (thr_A − thr_B) directly, the CI **excludes 0** in **40%** of 20 evaluable comparisons. Excluding 0 means the two halves reliably differ — but read it against verdict 2, not on its own. At n≈1962 the parametric bootstrap is tight enough to resolve a *hair*: a split-half Δ can be statistically non-zero while still being a few percent of the between-model gap, with the two halves' CIs overlapping and the ranking intact. That is not a kill. The kill is when \|Δ\| rivals the effect (verdict 2), the CIs stop overlapping (verdict 1), or the ranking flips (verdict 3).

## Limitations

- **Distractor suite excluded.** Splitting the val set changes the distractor-pool composition, a second confound on top of item sampling. Only contrast and degradation are tested here.
- **Single dataset, 10 classes.** Imagenette val only; a random split preserves rough class balance but is not stratified. Per-half class counts are in `provenance.json`.
- **Deterministic models.** No trial-level noise; all CI width comes from the parametric bootstrap over the fitted binomial, and the split-half Δ from which images fell in each half — exactly the artifact this test probes.
- The between-model gaps used as the yardstick come from the standing gallery (200 images), a different n than these halves (~1962).

_Generated 2026-07-15T22:48:46.396556+00:00 · psyvis_ml 0.0.1 · torch 2.6.0+cu124 · timm 1.0.27 · device cuda · wall-clock 53.5 min._
