# psyvis-ml — Quickstart

Measure a vision model's **threshold**, **slope**, and **CI** under a swept stimulus, instead of one accuracy number at one severity.

---

## 1. Install

```bash
pip install -e .            # core: numpy, scipy, matplotlib
pip install -e ".[demo]"    # + torch, timm, Pillow (real-model helpers)
pip install -e ".[dev]"     # + pytest, ruff
```

`import psyvis_ml` and the whole fitting/measurement path run on the core deps alone. torch, timm, and Pillow load lazily.

## 2. Bring your own images

Nothing is bundled or downloaded. Point it at your own subset in ImageFolder layout:

```
<root>/<class>/img.JPEG
```

Class folders named by integer ImageNet index or WordNet ID.

```bash
export PSYVIS_IMAGENET_DIR=/path/to/imagenette2/val
```

## 3. One measurement

```python
import psyvis_ml as pe

ds     = pe.datasets.imagenet_subset("/path/to/subset", max_classes=5)
model  = pe.models.timm_classifier("resnet50")          # frozen, eval
suite  = pe.suites.ContrastThreshold(contrast_metric="rms", clip_range=(0.0, 1.0))
levels = pe.linspace_levels(0.01, 0.5, 9, spacing="log")

result = pe.measure(model=model, suite=suite, dataset=ds,
                    levels=levels, model_name="resnet50")
```

`model` is just a callable `image -> logits`. `timm_classifier` is convenience, not a requirement — any framework or a plain function works.

## 4. Read it off

```python
result.threshold(target=0.75)   # stimulus level at 75% correct
result.slope(target=0.75)       # dP/dx there
result.fit()                    # the psychometric fit object + CIs
result.plot()                   # observed points, fitted curve, threshold, CI band
result.confidence(...)          # logit-margin decline, baseline-relative, + bootstrap CI
```

Lower contrast threshold = sharper vision. On real Imagenette, resnet50 (~0.05) beats vit_small (~0.20).

## 5. The three suites

| suite | sweeps | axis |
|---|---|---|
| `ContrastThreshold` | RMS contrast | increasing (higher = easier) |
| `DegradationSuite` | noise σ / blur / occlusion | decreasing (higher = worse) |
| `DistractorRobustness` | distractor size (px) | decreasing |

`DistractorRobustness` uses a big centred target (clears the recognition ceiling) with distractors swept inward from the margins. Size band set by `calibrate_distractor_size`. This is **not** crowding — no eccentricity, no fixation, no flanker-spacing law.

## 6. Compare models

```python
fig = result_a.compare([result_b], human="auto")   # one axis + cited human CSF
```

`compare` honors each result's declared axis direction and **refuses** to overlay mismatched suites or conditions. Where no credible human curve exists (distractor robustness), the overlay degrades to models-only with a note rather than fabricating one.

> **Paradigm caveat for the human overlay.** The human line is grating-detection contrast sensitivity (Campbell & Robson 1968, Michelson contrast); the model curves are argmax classification on natural images (RMS contrast). Different observer paradigms sharing an axis. A model far from the human line is a task difference, not proof about "seeing like a human."

## 7. Ship a bundle

```python
result_a.report(others=[result_b], outdir="report_bundle")
```

Writes a threshold/slope/CI table, fitted-curve figures, the models-vs-human figure, reproducibility metadata (resolved seed, config hash, library versions), **and** the per-item table automatically.

## 8. Per-item export (the handoff to metaeval)

`results.json` is aggregate — enough for a threshold, not enough for type-2 work, which needs a confidence paired with an outcome trial by trial.

```python
export = result.write_per_item("outdir")     # outdir/per_item.csv + per_item.meta.json
table  = pe.load_per_item(export.data_path)  # or just pandas.read_csv — it's a plain file
```

Always additive; aggregate outputs are unchanged. Full contract in [`SCHEMA.md`](SCHEMA.md).

---

## The two margins (read before scoring either)

```
margin          = logit[true_label] - max(logit[others])   # target-referenced, SIGNED, boundary at 0
decision_margin = logit[top-1]      - logit[top-2]         # decision-referenced, >= 0
```

- Use **`margin`** for the psychometric/confidence curve and Δ-margin.
- Use **`decision_margin`** for type-2 ROC / meta-d′.
- They're **equal on correct trials**, diverge only on errors.
- `abs(margin)` is a trap: on an error it's the size of the *miss*, so a big miss scores as high confidence.

psyvis-ml never applies `abs()` on export — folding the margin is a *scoring* decision that belongs to the consumer.

## The disciplines

- **Margin, not softmax.** Softmax exponentiates logits, swings with temperature, is poorly calibrated. `max_softmax` is recorded for reference only.
- **Never compare raw margins across models.** Logit scales aren't comparable. Report each model's change from its own clean baseline.
- **Method of constant stimuli only.** No adaptive staircase in v1.
- **The fitter is observer-model-agnostic.** It sees only `(level, n_correct, n_trials)`. Argmax top-1 is a documented, swappable choice made in the suites — not a limitation of the fitting core.

## The fitting core, standalone

`psyvis_ml.fitting` is import-isolated (numpy + scipy only) so sibling projects can reuse it. Point it at any counts:

```python
from psyvis_ml.fitting import fit_psychometric
import numpy as np

fit = fit_psychometric(
    np.array([0.02, 0.04, 0.08, 0.16, 0.32]),   # levels
    np.array([  12,   19,   34,   46,   49]),   # n_correct
    np.array([  50,   50,   50,   50,   50]),   # n_trials
    sigmoid="weibull", guess_rate=0.5, lapse_rate=0.02,
)

fit.threshold(target=0.75)
fit.slope(target=0.75)                            # units="linear"|"log", base=10
fit.bootstrap_ci("threshold", target=0.75, seed=0)
fit.summary()                                     # flat dict, incl. at_bound
```

The model:

```
P(correct | x) = γ + (1 − γ − λ) · F(x; α, β)
```

F is Weibull or logistic with location α (`params["alpha"]`) and slope β (`params["slope"]`); γ = guess rate, λ = lapse rate (default fixed 0.02; pass `None` to estimate by ML).

**Gotcha:** `params["alpha"]` (raw sigmoid location) ≠ `fit.threshold(target)` (derived stimulus level at a performance). Report the latter.

## Regenerate the gallery

```bash
pip install -e ".[demo]"
export PSYVIS_IMAGENET_DIR=/path/to/imagenette2/val
python examples/generate_gallery.py
```

Outputs land in `outputs/gallery/`: figures plus `results.md`, `results.json`, `analysis.md` (paired bootstrap difference tests), and `per_item.csv`.

## Where to go next

`psyvis-ml` **emits** the confidence signal. `metaeval` **scores** it — type-2 ROC, risk–coverage, calibration-as-contrast. See the metaeval quickstart.
