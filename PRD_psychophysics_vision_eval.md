# PRD — Psychophysical Evaluation for Vision Models (working title: `psyvis-eval`)

**Status:** Draft v0.1
**Owner:** Trevor Caruso
**Build order:** Project #1 of 3 (first to implement)
**Positioning:** Both audiences, leading with ML robustness eval; vision-science / human-model comparison as the second frame.

---

## 1. One-line summary

A Python library that treats a vision model as a psychophysical observer — running controlled stimulus manipulations (contrast, noise, blur, eccentricity/crowding) to measure *thresholds and sensitivity functions* rather than single benchmark scores, and expressing model behavior in the same measurement language used for human observers.

## 2. Problem

ML robustness is mostly measured with accuracy on perturbed datasets (ImageNet-C and friends): a scalar per corruption at fixed severity. This tells you *that* a model degrades but not the *shape* of its degradation, its *threshold*, or how its sensitivity compares to a human observer's. There is no standard tool that:

- Sweeps a stimulus dimension parametrically and fits a psychometric function to a model's responses.
- Reports thresholds (e.g., contrast at 75% correct) and slopes instead of accuracy-at-one-severity.
- Puts model and human sensitivity on a common axis for direct comparison.

Vision scientists have this machinery (staircases, psychometric fitting, threshold estimation) but it lives in human-experiment tooling and has not been packaged to point at models. ML researchers can perturb images but rarely build a rigorous measurement apparatus. The gap sits exactly at that intersection.

## 3. Why this is defensible / why the gap persists

- Requires *both* real psychophysics knowledge (threshold estimation, psychometric fitting, what a valid sensitivity measurement is) and ML engineering. Few people hold both.
- Academics get citations for findings, not for measurement infrastructure.
- The incumbent approach (accuracy-on-corruptions) is "good enough" to publish, so nobody has been forced to build the better instrument.

## 4. Positioning: two frames, one tool

The same measurements serve two audiences. The PRD leads with the first for market legibility, keeps the second for credibility and papers.

**Frame A — Robustness eval (lead).** "Measure the *threshold* and *degradation shape* of your vision model under controlled distortions, not just accuracy at one severity." Buyers/users: robustness and safety-eval teams, anyone shipping vision models into safety-relevant settings.

**Frame B — Human-model comparison (credibility).** "Measure a model's contrast sensitivity function, crowding zones, and noise tolerance in the same units as a human observer." Audience: vision science, cog-sci, model-human alignment researchers; paper and collaborator pipeline.

The single differentiating claim, stated plainly and defended up front in the README:
> Accuracy-on-corruptions gives you a point. A psychometric sweep gives you the curve — threshold, slope, and the severity at which the model fails — which is what you need to compare models fairly (same accuracy, different thresholds) and to compare a model against a human.

## 5. Goals / non-goals

**Goals (v1):**
- Parametrically sweep a stimulus dimension over a labeled image set, collect model responses, fit a psychometric function, and report threshold + slope + CI.
- Ship three measurement suites (see §7).
- Produce the "killer plot": human vs. several models on one psychometric axis.
- Clean, boring API that an ML engineer can run without knowing psychophysics vocabulary.

**Non-goals (v1):**
- No adaptive staircase engine in v1 (method of constant stimuli only — simpler, fully parallelizable, no sequential dependency). Adaptive is v2.
- No LLM / language models. This tool is vision-only; the SDT-for-ML-eval work (Project #1 in the earlier ranking) is a *separate* library.
- No CLIP / multimodal in v1 (scoped out per decision below).
- No training, no model zoo hosting, no proprietary datasets.

## 6. MVP scope decisions (locked)

- **Models:** ImageNet classifiers only in v1. Cleanest response model (argmax / top-k correct, softmax available as a confidence signal for later reuse). CLIP-style and human-reference curves are fast-follows, not v1.
- **Response definition:** correct/incorrect from the classifier on a controlled image set; psychometric function is P(correct) vs. stimulus level.
- **Human reference:** ship *pre-collected/published* human sensitivity curves as static reference data where they exist (e.g., contrast sensitivity), rather than running human experiments. Clearly cite sources. Full human-data collection is out of scope.

## 7. The three v1 measurement suites

1. **Contrast sensitivity / contrast threshold.** Sweep contrast (optionally × spatial frequency), fit P(correct) vs. contrast, report threshold and slope. The classic, most-shareable result.
2. **Peripheral / eccentricity + crowding.** Present targets at controlled eccentricity with and without flankers; measure the performance drop and crowding zone. This is directly your dissertation expertise and is *the* differentiated suite — nobody in ML has a crowding measurement tool.
3. **Degradation robustness (noise / blur / occlusion / spatial-frequency).** Sweep severity for each, fit curves, report thresholds. This is the bridge to ImageNet-C users — same corruptions, but as fitted curves instead of scalars.

## 8. Core API sketch (target shape)

```python
import psyvis_eval as pe

result = pe.measure(
    model=my_model,                 # any callable image -> logits
    suite=pe.suites.ContrastThreshold(spatial_freqs=[1, 2, 4, 8]),
    dataset=pe.datasets.ImageNetSubset(...),
    levels=pe.linspace_levels(...),  # method of constant stimuli
)

result.threshold()        # e.g. contrast at 75% correct, per condition
result.slope()
result.fit()              # psychometric fit object + CIs
result.plot()             # curve(s); overlay human reference if available
result.compare([other_result_a, other_result_b])  # multi-model on one axis
result.report()           # methods-ready text + figures bundle
```

Design constraints:
- `model` is just a callable; no framework lock-in (works with torch, tf, jax, or a wrapper).
- Fitting engine is a separable module (reused later by Project #1's SDT-for-ML library).
- Everything reproducible: seed capture, config hash, versioned outputs (the audit-bundle idea, folded in here as a differentiator).

## 9. Technical stack / dependencies

- Python, NumPy, SciPy.
- Psychometric fitting: start with a clean maximum-likelihood Weibull/logistic fit (your own, small, well-tested) rather than a heavy dependency; optionally interoperate with psignifit later.
- Plotting: matplotlib.
- Model I/O: framework-agnostic callable; provide torch/timm convenience loaders for the ImageNet models used in demos.
- Optional: your NumPyro experience is a v2 lever (Bayesian/hierarchical fitting), not a v1 dependency.

## 10. Success criteria (sequenced — read §"note" below)

**v1 primary (the bar to hit):**
- A public repo + preprint/short paper with the human-vs-models psychometric comparison figure as the centerpiece.
- Early external traction: installs, a handful of GitHub stars/issues from people you didn't recruit, and at least one unsolicited collaborator or citation inquiry.

**v1.5 (explicit follow-on, not v1):**
- "Something an industry safety/robustness team would actually run" — requires polish, docs, stability, and a robustness-framed quickstart. Gated on v1 landing first, because it needs trust v1 won't have on day one.

*Note on "all three":* the paper/demo and GitHub traction are compatible v1 goals and reinforce each other. Industry-usability is a distinct, later bar with its own polish requirements; forcing it into v1 is the main way a solo build stalls. It is therefore a milestone, not a v1 gate. Revisit if you decide industry pull matters more than the paper.

## 11. Timeline (solo, part-time)

- **Weeks 1–3:** psychometric fitting core + method-of-constant-stimuli sweep engine + one suite (contrast) working end-to-end on 2–3 ImageNet models.
- **Weeks 4–6:** eccentricity/crowding suite (the differentiator) + degradation suite.
- **Weeks 7–9:** the comparison/plot layer, human-reference overlays, report/bundle output, docs.
- **Weeks 10–12:** polish, worked-example notebook, preprint draft, public launch.

## 12. Go-to-market (no company pretense)

- Lead artifact: the human-vs-models psychometric figure. It is instantly legible and shareable.
- Post in vision-science channels (VSS crowd, Bluesky/Mastodon vision circles) *and* ML robustness channels — the dual framing is the asset.
- Short methods paper (Behavior Research Methods / JORS, or an ML-robustness workshop). Choose venue by which frame is pulling harder at that point.
- Offer the crowding/eccentricity suite as the "you can't get this anywhere else" hook.

## 13. Kill criteria (weeks 4–8)

- The contrast suite works but produces *uninteresting* results — models and humans look trivially the same or trivially different, with no structure worth a figure. (Test this early; it's the core bet.)
- No external installs or engagement after posting in 2–3 real communities.
- The crowding suite doesn't reveal anything a plain accuracy-drop wouldn't have shown — i.e., the psychophysical framing adds no diagnostic value over ImageNet-C scalars.
- An existing tool (someone else's) already does parametric threshold fitting for vision models and you'd be duplicating. (Scan before building.)

Any two firing = stop adding suites and reassess the framing.

## 14. Open risks

- **"Why not just accuracy-on-corruptions?"** Your README's first paragraph must answer this concretely, or ML readers bounce. The answer is the threshold/slope/curve argument in §4.
- **Response model validity:** argmax-correct is a coarse observer model; make sure the psychometric fits are meaningful and not artifacts of the classifier's decision boundary. Validate on a case where the human answer is known.
- **Human-reference availability:** good published human curves exist for contrast sensitivity but are thinner for crowding-in-natural-images; be honest where the human overlay is approximate.
- **Scope creep toward adaptive methods / LLMs / CLIP.** All explicitly deferred. Guard the v1 boundary.

## 15. Relationship to the other two projects

- The **fitting core and reproducibility bundle** built here are reused by Project #1 (SDT-for-ML evaluation). Build them clean and separable.
- The **confidence/softmax signals** you collect here are the on-ramp to Project #1's meta-d′/type-2 ROC work: same models, same harness, extended to uncertainty quality.
- Project #3 (SBI for psychophysical models) can later sit *underneath* this library's fitting engine. Not a v1 concern.
