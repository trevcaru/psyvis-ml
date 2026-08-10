# Does the fit-bootstrap CI capture item-sampling variance?

Derived from `results.csv` (the 20-comparison split-half run) — **no sweeps re-run**. The question:
the reported threshold CIs come from a *parametric bootstrap over the fitted binomial* (resample
`n_correct ~ Binomial(n_trials, P_fitted)` at each level, refit). That models trial noise **given
the fitted curve**. Does it also capture the variance from *which images you drew* — the thing the
split-half test actually perturbs?

## Method — direct σ estimate (no normality assumption)

thr_A and thr_B are two independent threshold estimates on disjoint random halves (n ≈ 1962 each).
By the arbitrary A/B labelling, Δ = thr_A − thr_B is mean-zero, and

    Var(Δ) = Var(thr_A) + Var(thr_B) = 2·σ_half²   ⟹   σ_half = RMS(Δ) / √2

with **RMS(Δ) = √(mean Δ²)** taken over the 5 seeds (using the known-zero mean, not a fitted one).
This estimates the true item-sampling SD of a single half's threshold **without any assumption about
the shape of the Δ distribution**. We compare it to the CI-implied SD, `σ_CI = (mean CI half-width)
/ 1.96`. Computed **per model**, because the two models' thresholds differ ~4× in scale (resnet ≈
0.05, vit ≈ 0.19 on contrast) and pooling them mixes variances.

## Normality / robustness check first

- **`at_bound`: none on any of the 40 fits.** No threshold sat on a parameter box edge, so no
  fit-boundary artifact is distorting a CI or a Δ.
- **Skew of signed Δ (n = 5 per model):** contrast −0.07 / −0.12; degradation −0.03 / **+0.60**
  (vit). Small for three of four; the degradation-vit tail is the one mild asymmetry (driven by the
  seed-2 |Δ| = 0.024 outlier). The pooled-per-suite skews (−0.29, **+1.06**) are inflated by
  scale-mixing two models and should be ignored.
- **Verdict:** nothing ugly (no at-bound, roughly symmetric per model), **but n = 5 cannot certify
  normality** — which is exactly why the direct σ estimate below, not the median route, is the
  headline. The median route assumed |Δ| ~ folded-normal (the 0.674 SD↔median factor); the direct
  route does not.

## Result — σ_half directly vs the CI

| suite / model | σ_half (direct, RMS/√2) | σ_CI (half-width/1.96) | **understatement σ_half/σ_CI** |
|---|---|---|---|
| contrast / resnet50        | 0.00550 | 0.00227 | **2.4×** |
| contrast / vit_small       | 0.00983 | 0.00476 | **2.1×** |
| contrast (pooled 10)       | 0.00797 | 0.00352 | 2.3× |
| degradation / resnet50     | 0.00595 | 0.00395 | **1.5×** |
| degradation / vit_small    | 0.00987 | 0.00474 | **2.1×** |
| degradation (pooled 10)    | 0.00815 | 0.00434 | 1.9× |

**Headline: the fit-bootstrap CI understates item-sampling SD by ≈ 2× — contrast ~2.1–2.4×,
degradation ~1.5–2.1×.**

### This revises the earlier median-route number down

The first pass reported 3.1× (contrast) / 2.3× (degradation) from
`R = median|Δ|/median(half-width)` × 2.05. The direct estimate lands at **~2.2× / ~1.9×**. The median
route **overstated contrast** because (a) it pooled the two models, so `median|Δ|` sat among the
larger-scale vit values while `median half-width` sat between the models, and (b) it relied on the
0.674 folded-normal factor. The direct per-model σ is the trustworthy figure; where the two methods
disagree, the median route was the problem. **The qualitative conclusion is unchanged — the CIs are
roughly a factor of two too narrow — only the size of the factor moved.**

## What it means (plainly)

The parametric bootstrap treats every image at a level as an identical Bernoulli trial with the
fitted `P`, so it only sees idealised binomial noise. Real images have **heterogeneous difficulty**,
and swapping in a different random half moves the fitted threshold about **twice** as much as that
idealised CI admits. The bootstrap CI is a **lower bound** on threshold uncertainty — it captures
curve/trial noise but misses image-set (item-sampling) noise.

This is consistent with the split-half verdicts: contrast passes because its between-model gap
(0.146) dwarfs even a 2–2.4×-widened CI, whereas degradation's small gap (0.036) does not survive a
~2×-widened interval — which is why degradation reads AMBIGUOUS.

## Recommendation

- **Widen the reported threshold CIs by ≈ 2× (per-suite: ~2.2× contrast, ~1.9× degradation)** before
  reading any single threshold's uncertainty — or, cleaner, **report the split-half spread itself**
  as the item-sampling CI, since it measures the real quantity directly.
- **Re-check between-model "reliable" calls under the wider interval.** A paired difference whose CI
  barely excludes 0 at the un-widened scale (the gallery's degradation gap, CI ≈ [0.004, 0.067])
  generally will **not** exclude 0 once ~2× item-sampling variance is folded in. Contrast is
  unaffected — its gap is ~30% of the swept axis and survives comfortably.
- Longer term, an **item-level (cluster/jackknife-over-images) bootstrap** inside the fitter would
  capture this variance directly, removing the need for an after-the-fact inflation factor.

## Caveats — indicative, not exact

1. **n mismatch.** The half CIs are at **n ≈ 1962**; the gallery reports at **n ≈ 3925** (~√2
   tighter). So this does not translate one-to-one to the gallery's printed intervals.
2. **n-invariance of the ratio is an ASSUMPTION, not a property.** The understatement factor would
   carry over to n ≈ 3925 *only if the fit-bootstrap variance and the item-sampling variance both
   scale as 1/√n* — which is precisely the equivalence under test here. It is probably true to first
   order, but it is an assumption; if item-difficulty heterogeneity does not average out as 1/√n
   (e.g. a few systematically hard images), the factor could drift with n.
3. **Small n for σ itself.** σ_half is estimated from 5 pairs per model, so each factor carries its
   own ±30–40% sampling error. Read them as "roughly 2×", not as calibrated multipliers.
