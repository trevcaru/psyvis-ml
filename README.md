# psyvis-ml

**Why not just accuracy-on-corruptions?** Accuracy on a perturbed dataset (ImageNet-C
and friends) gives you a *point*: one scalar per corruption at one fixed severity. That
tells you *that* a model degrades, but not the *shape* of the degradation, the *threshold*
at which it fails, or how *steeply* it falls off. Two models can post identical accuracy at
severity 3 and yet have thresholds an octave apart — one gracefully declining, one about to
fall off a cliff. A psychometric sweep gives you the *curve*: sweep a stimulus dimension
parametrically, fit P(correct) vs. stimulus level, and read off a **threshold** (e.g. the
contrast at 75% correct), a **slope** (how fast performance changes near threshold), and a
**confidence interval** on each. That is the measurement you need to compare models fairly
(same accuracy, different thresholds) and to put a model's sensitivity on the same axis as a
human observer's. `psyvis-ml` treats a vision model as a psychophysical observer and
reports thresholds and sensitivity functions, not accuracy at one severity.

> Status: early development. This repository ships the **import-isolated fitting core**
> (`psyvis_ml.fitting`), the **method-of-constant-stimuli sweep engine** with a reproducible
> run bundle (`psyvis_ml.sweep`), **three measurement suites** — contrast, degradation, and
> distractor-robustness (`psyvis_ml.stimuli`, `psyvis_ml.suites`), the top-level
> **`pe.measure(...)` API** (sweep → fit → threshold/slope/CI/plot), a **graded
> confidence-readout layer** — per-image target-class logit-margin curves across every suite,
> baseline-relative Δ-margin, and a confidence threshold with bootstrap CIs
> (`result.confidence(...)`, `psyvis_ml.confidence`) — and the **comparison + human-reference
> overlay + report-bundle** layer (`result.compare(...)`, `result.report(...)`,
> `psyvis_ml.reference`). Still to come: more human-reference curves; adaptive staircases,
> Bayesian/hierarchical fitting, and meta-d′/type-2 scoring remain v2.

## Install

```bash
pip install -e .            # core: numpy, scipy, matplotlib
pip install -e ".[demo]"    # + torch, timm, Pillow convenience loaders for ImageNet demos
pip install -e ".[dev]"     # + pytest, ruff
```

`import psyvis_ml` and the whole fitting/measurement path work on the **core deps alone**;
torch/timm/Pillow are imported lazily and only needed when you call the real-model helpers.

## Quickstart — the human-vs-models figure

The worked example ([`examples/worked_example.ipynb`](examples/worked_example.ipynb)) runs the
whole instrument on real models end to end: load 2–3 `timm` ImageNet classifiers, sweep
contrast, fit, and produce the human-vs-models contrast figure with the cited human CSF
overlay.

**You supply the images.** This package never bundles or downloads ImageNet. Point it at your
own subset in ImageFolder layout (`<root>/<class>/img.JPEG`; class folders named by integer
ImageNet index or WordNet ID):

```bash
pip install -e ".[demo]"
export IMAGENET_DIR=/path/to/your/imagenet_subset   # your own data
jupyter notebook examples/worked_example.ipynb
```

In code the same path is just a few lines:

```python
import psyvis_ml as pe

ds = pe.datasets.imagenet_subset("/path/to/your/imagenet_subset", max_classes=5)
models = {n: pe.models.timm_classifier(n) for n in ["resnet18", "resnet50"]}  # frozen, eval
suite = pe.suites.ContrastThreshold(contrast_metric="rms", clip_range=(0.0, 1.0))
levels = pe.linspace_levels(0.01, 0.5, 9, spacing="log")

results = [pe.measure(model=m, suite=suite, dataset=ds, levels=levels, model_name=n)
           for n, m in models.items()]
fig = results[0].compare(results[1:], human="auto")   # models + cited human CSF, one axis
results[0].report(others=results[1:], outdir="report_bundle")   # methods-ready bundle
```

`model` is just a callable `image -> logits`; `timm_classifier` is a convenience wrapper, not
a requirement — any framework (or a plain function) works.

## Results gallery

Standing figures on real **Imagenette** val (resnet50 + `vit_small_patch16_224`), regenerated
by [`examples/generate_gallery.py`](examples/generate_gallery.py) into
[`outputs/gallery/`](outputs/gallery/) (numbers in
[`outputs/gallery/results.md`](outputs/gallery/results.md) /
[`results.json`](outputs/gallery/results.json)):

**Contrast — human vs. models (lead figure).**

![contrast: human vs. models](outputs/gallery/contrast_human_vs_models.png)

> **Paradigm caveat (read with the figure).** The human line is *grating-detection* contrast
> sensitivity (Campbell & Robson 1968, Michelson contrast); the model curves are *argmax-
> classification* correctness on natural images (RMS contrast). Different observer paradigms on
> a shared contrast axis — a model threshold far from the human line is a difference in task,
> not proof a model does or doesn't "see like" a human.

**Degradation (gaussian noise) — fitted falling curves + Δ-margin.**

![degradation comparison](outputs/gallery/degradation_comparison.png)

**Distractor robustness — ceiling-limited (see caveat in `results.md`).**

![distractor comparison](outputs/gallery/distractor_comparison.png)

**Confidence layer — baseline-relative Δ logit margin (contrast).** The primary confidence
signal is the target-class **logit margin** (not softmax); across models it is compared only
as Δ from each model's own clean baseline.

![contrast Δ-margin](outputs/gallery/contrast_confidence_delta.png)

Regenerate locally (you supply the data — nothing is bundled or downloaded):

```bash
pip install -e ".[demo]"
export PSYVIS_IMAGENET_DIR=/path/to/imagenette2/val
python examples/generate_gallery.py
```

## The fitting core

`psyvis_ml.fitting` is a small, self-contained maximum-likelihood fitter for the
psychometric function. It is **import-isolated**: it imports only `numpy` and `scipy` and
nothing else from the package, because two sibling projects reuse it. You can point it at any
`(stimulus_level, n_correct, n_trials)` data:

```python
import numpy as np
from psyvis_ml.fitting import fit_psychometric

levels    = np.array([0.02, 0.04, 0.08, 0.16, 0.32])   # e.g. RMS contrast
n_correct = np.array([  12,   19,   34,   46,   49])
n_trials  = np.array([  50,   50,   50,   50,   50])

fit = fit_psychometric(levels, n_correct, n_trials, sigmoid="weibull",
                       guess_rate=0.5, lapse_rate=0.02)

fit.threshold(target=0.75)          # stimulus level at 75% correct
fit.slope(target=0.75)              # dP/dx at that point (units="linear" | "log", base=10 for log10)
fit.bootstrap_ci("threshold", target=0.75, seed=0)   # percentile CI
fit.summary()                       # flat dict of everything, incl. at_bound
```

The model is

```
P(correct | x) = γ + (1 − γ − λ) · F(x; α, β)
```

where `F` is a Weibull or logistic sigmoid with location `α` (stored as `params["alpha"]`)
and slope `β` (`params["slope"]`), `γ` is the guess rate (lower asymptote) and `λ` is the
lapse rate (upper asymptote). Note `params["alpha"]` (the raw sigmoid location) and
`fit.threshold(target)` (the derived stimulus level at a given performance) are distinct;
the latter is usually what you report. Pass a **number** to hold `guess_rate`/`lapse_rate` fixed, or
`None` to estimate it by maximum likelihood (`λ` bounded to `[0, 0.5)`, `γ` to `[0, 1)`). The
default lapse is a fixed `λ = 0.02`.

### Methods note: the fitter is observer-model-agnostic

This fitting core makes **no assumption about how "correct" is defined.** It consumes only
aggregated `(level, n_correct, n_trials)` counts. The eventual decision to score a model by
**argmax top-1 correctness** (in the measurement suites, not here) is a documented,
*swappable* observer-model choice — not a limitation of the fitter. Other observer models
(top-k, a trained contrast-discrimination probe, a 2AFC pairing, a softmax-confidence
criterion) produce the same `(level, n_correct, n_trials)` shape and fit identically. We name
this explicitly because the closest prior art measures a different observer:
[Akbarinia et al. (2023), *Contrast Sensitivity Function in Deep Networks*](https://pubmed.ncbi.nlm.nih.gov/37156217/)
measure DNN CSFs via a trained linear contrast-discrimination probe on frozen features rather
than argmax-top-1 on a labeled classification set. That is a legitimate, different observer
model, and a comparison worth making — not a duplication.

## Comparison, human overlays, and the report bundle

Once you have measured several models, `result.compare([other_a, other_b, ...])` puts their
psychometric curves — observed points, fitted curves, thresholds, and CI bands — on **one
axis** for a shared suite/condition (the PRD §5/§12 "killer plot"). It honours each result's
*declared* axis direction (rising for contrast/distractor-robustness, falling for degradation)
and refuses to overlay results from mismatched suites/conditions rather than compare apples to
oranges.

Where a credible **published human curve** exists, the same axis carries a cited human
overlay. We ship pre-collected/published human sensitivity data as static, versioned reference
data and **never run human experiments or synthesize** a curve; where no good human data
exists (e.g. distractor robustness, which has no established human sensitivity curve), the
overlay **degrades gracefully** — models only, with a note — instead of fabricating one.

Shipped human reference data (`psyvis_ml.reference`, files under `reference/data/`):

- **Contrast sensitivity function** — an *approximate, digitized* human photopic CSF for
  sinusoidal gratings (Michelson contrast), representative of the classic band-pass curve of
  **Campbell, F. W., & Robson, J. G. (1968). Application of Fourier analysis to the visibility
  of gratings. _The Journal of Physiology_, 197(3), 551–566.**
  ([doi:10.1113/jphysiol.1968.sp008574](https://doi.org/10.1113/jphysiol.1968.sp008574)) and
  consistent with the ModelFest / standard-observer literature. It is flagged `approximate` in
  both the data file and the API, and is for illustration and cross-checking — not a
  substitute for measuring your own observers, and its contrast metric (Michelson grating) may
  differ from a given model dataset's.

`result.report(others=[...])` writes a self-contained methods bundle — a threshold/slope/CI
table, the fitted-curve figures, the models-vs-human comparison figure, and the
**reproducibility metadata** captured in every run (resolved seed, config hash, library
version) — that an external user can drop into a methods section (the §8 audit-bundle idea).

### Relationship to existing tools

For fitting human observer data, [**psignifit 4**](https://psignifit.readthedocs.io/en/latest/)
is the gold standard (Bayesian beta-binomial, goodness-of-fit, CIs). `psyvis-ml` is not a
replacement for it: this fitter is deliberately small and dependency-light so it can be
embedded in a *model*-evaluation harness, and interoperating with psignifit for cross-checking
fits is an explicit later goal. The differentiator of `psyvis-ml` is the packaging — a
reproducible sweep engine, measurement suites, and human-reference overlays that point this
machinery at models — not the fitting math itself.

## License

MIT. See [LICENSE](LICENSE).
