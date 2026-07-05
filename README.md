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
> run bundle (`psyvis_ml.sweep`), the **contrast stimulus and `ContrastThreshold` suite**
> (`psyvis_ml.stimuli`, `psyvis_ml.suites`), and the top-level **`pe.measure(...)` API** that
> composes them (sweep → fit → threshold/slope/CI/plot). Still to come: the remaining
> measurement suites (eccentricity/crowding, degradation), real ImageNet/timm loaders, and
> the multi-model comparison and human-reference overlay layer.

## Install

```bash
pip install -e .            # core: numpy, scipy, matplotlib
pip install -e ".[demo]"    # + torch, timm convenience loaders for ImageNet demos
pip install -e ".[dev]"     # + pytest, ruff
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
