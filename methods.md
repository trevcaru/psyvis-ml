# Methods

## Models and dataset

Two pretrained ImageNet classifiers served as observers: a convolutional network (`resnet50`) and a Vision Transformer (`vit_small_patch16_224`), loaded via `timm`, frozen, in evaluation mode. Each model is treated as a callable mapping an image to a logit vector; no fine-tuning or probing was performed. Stimuli were drawn from the Imagenette validation set (320 px), organized in ImageFolder layout. The task in all experiments is multiclass identification: a trial is scored correct if the true label is the model's top-1 prediction (top-k is recorded in the run metadata and is 1 throughout).

## Psychophysical procedure

Each experiment used the method of constant stimuli, implemented in `psyvis-ml`. A stimulus dimension was swept over a fixed set of levels, every image was presented at every level, and per-level correct/total counts were aggregated. Three measurement suites were run: (1) **contrast**, sweeping RMS contrast (increasing axis: higher level is easier); (2) **degradation**, sweeping gaussian-noise σ (decreasing axis: higher level is harder); and (3) **distractor robustness**, sweeping the size of distractors placed around a large centred target, with a level-invariant `undistracted` condition retained as a ceiling check and held out of the severity axis. We describe suite 3 as distractor robustness, not crowding: there is no eccentricity manipulation, no fixation, and no flanker-spacing law. Every run captures a resolved seed, a SHA-256 configuration hash, and library versions in a reproducibility bundle.

## Psychometric fitting

Per-level counts were fit by maximum likelihood with the psychometric model

P(correct | x) = γ + (1 − γ − λ) · F(x; α, β),

where F is a Weibull or logistic sigmoid with location α and slope β, γ is the guess rate (set to the task's chance level, top_k / n_classes), and λ is the lapse rate (fixed at 0.02 by default). The fitter (`psyvis_ml.fitting`) is import-isolated (numpy/scipy only) and observer-model-agnostic: it consumes only `(level, n_correct, n_trials)` counts. Reported thresholds are the derived stimulus level at a criterion performance (75% correct), which is distinct from the raw location parameter α. Slopes are dP/dx at the criterion level. Confidence intervals on thresholds and slopes are percentile bootstrap intervals with a fixed seed. Cross-model comparison respects each suite's declared axis direction and refuses overlays of mismatched suites or conditions. For contrast, a digitized human photopic CSF (Campbell & Robson, 1968; Michelson contrast, grating detection) is overlaid as cited, static reference data, flagged approximate; the paradigms differ (grating detection vs. argmax classification on natural images), so the overlay is a shared axis, not a claim of equivalent observers.

## Confidence signals

The sweep records the full logit vector per image per level, from which two distinct confidence signals are persisted (per-item schema v1.1):

- **margin** (target-referenced): logit[true label] − max(logit[others]), signed, decision boundary at 0 (margin > 0 iff the true class is top-1). This is the signal for the psychometric-adjacent confidence readout: the baseline-relative decline in margin (Δ from each model's own clean condition), with a confidence threshold at the margin's criterion crossing and bootstrap-over-images CIs.
- **decision_margin** (decision-referenced): logit[top-1] − logit[top-2], non-negative by construction — the model's confidence in the answer it actually gave. The two are equal on correct trials and diverge on errors, where `margin` measures the size of the miss (a quantity the observer cannot access) and `decision_margin` measures confidence in the wrong answer.

Raw softmax probability is recorded (`max_softmax`) but flagged reference-only: softmax exponentiates the logits, is temperature-sensitive, and is poorly calibrated. Logit scales are not comparable across architectures, so no raw-margin quantity of either kind is ever compared across models; all cross-model confidence comparisons are within-model, baseline-relative changes, and the plotting layer structurally refuses raw cross-model confidence overlays.

## Trial-level export

Because aggregate counts cannot support type-2 analysis (which requires the trial-level pairing of a confidence with an outcome), each run exports a per-item table (one row per item × level × condition × model) with both margins, correctness, true and predicted labels, target rank and logit, and a severity rank (a direction-aware axis derived from the suite's declared direction, never inferred from data). A JSON sidecar restates the margin contracts, the axis direction, per-run reproducibility metadata, and `type2_confidence_column = "decision_margin"`, so the downstream consumer scores the intended signal rather than inferring one.

## Metacognitive scoring

Type-2 and SDT analyses were computed in `metaeval` from the persisted per-item data via its `psyvis` bridge, which maps sweep levels to the severity axis, conditions to slices, and the clean condition to the baseline. All type-2 measures were scored on `decision_margin`, as designated by the sidecar; the choice is recorded in the result's confidence provenance. Scoring `abs(margin)` instead is a known artifact source: at floor accuracy it drags type-2 AUROC below 0.5 for definitional reasons (large misses masquerade as high confidence), which we verified empirically and retain as a labelled illustration, not a finding.

The primary metacognition measure is the **type-2 ROC/AUROC**: rank-based (Mann–Whitney, tie-corrected), model-free, computed on the raw confidence signal, asking whether confidence discriminates correct from incorrect responses. **Calibration** (ECE with reliability diagrams) is computed explicitly as a contrast term — it asks whether confidence magnitudes match accuracy, a different question that can rank models in the opposite order — and requires a probability scale, so it is computed on `max_softmax`, never on a raw margin. **Risk–coverage** curves summarize selective prediction, with RC-AUC defined as mean risk across the full coverage grid (a Riemann sum over [0, 1], well-defined at n = 1). **meta-d′ / M-ratio** implements the Maniscalco & Lau (2012) SDT-based maximum-likelihood estimator (verified against the published worked example: d′ = 1.535, meta-d′ = 1.634, M-ratio = 1.064, and cross-checked against an independent implementation, metadpy, to ~1e-6 on a 9-case degenerate battery), with the log-linear correction of Hautus (1995) applied to extreme hit/false-alarm proportions in type-1 d′. On the present multiclass identification sweeps, meta-d′ and d′ are undefined and are refused with a stated reason: identification supplies no target-absent trials, hence no false-alarm rate and no type-1 SDT structure; constructing a binary task from correctness would fabricate the structure the estimator measures. meta-d′ therefore appears only where a genuine binary/2AFC design supplies it.

## Statistical analysis

The models are deterministic, so the only sampling variability is the finite item set; no p-values are reported anywhere. All uncertainty is expressed as seeded, deterministic bootstrap-over-items percentile CIs (n_boot = 2000, seed = 0 in the standing analyses). Within `metaeval`, every measure shares a single bootstrap resample-index matrix drawn once per evaluation, so CIs are comparable within a resample rather than independently drawn. Model differences are assessed by paired bootstrap difference tests, with a CI-excludes-zero criterion. Per-slice and along-severity analyses use within-slice bootstrap and are baseline-relative to the clean condition.

Sparse data are gated rather than papered over. meta-d′ estimates below ~100 items per response×confidence cell (where its coefficient of variation is empirically 15–20%) are flagged `unstable` with the cell counts, not refused. Slices with a minority class below 30 are drawn hollow and flagged as thin support; all-correct (ceiling) slices return NaN with the reason (nothing to rank against); meta-d′ at d′ = 0 returns NaN flagged `degenerate`. Every measure honours one invariant — a finite value, a NaN with an attached reason, or a loud error — enforced by a fuzz layer of ~6,400 randomized inputs spanning the operating range.

## Reproducibility

All runs capture seed, configuration hash, and library versions; the per-item sidecar carries the same metadata as the aggregate outputs, so trial-level and aggregate artifacts audit identically. The full gallery (three experiments, both models) regenerates from the committed per-item files with a single script and no upstream install. `psyvis-ml` emits the confidence signal; `metaeval` scores it; neither reimplements the other's role.

## References

Campbell, F. W., & Robson, J. G. (1968). Application of Fourier analysis to the visibility of gratings. *The Journal of Physiology*, 197(3), 551–566.

Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values of d′. *Behavior Research Methods, Instruments, & Computers*, 27, 46–51.

Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings. *Consciousness and Cognition*, 21(1), 422–430.
