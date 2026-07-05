"""Smoke test: the PRD §5/§12 human-vs-models contrast figure renders end-to-end."""

import numpy as np

import psyvis_ml as pe
from psyvis_ml.datasets import Dataset, synthetic_dataset
from psyvis_ml.suites import DegradationSuite

import synthetic as syn

NUM_CLASSES = 8


def _contrast_result(alpha, *, name, n=160, seed=1):
    ds = synthetic_dataset(n=n, num_classes=NUM_CLASSES)
    model = syn.make_contrast_observer(NUM_CLASSES, alpha=alpha)
    levels = pe.linspace_levels(0.03, 0.6, 9, spacing="log")
    suite = pe.suites.ContrastThreshold(contrast_metric="rms")
    return pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=seed,
                      model_name=name)


def test_human_vs_models_contrast_figure_renders():
    """The killer plot: several models + a cited human reference on one axis."""
    a = _contrast_result(0.15, name="ResNet-ish")
    b = _contrast_result(0.30, name="ViT-ish")
    fig = a.compare([b], human="auto", n_boot=40, seed=0)
    ax = fig.axes[0]

    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    # Both models present...
    assert any("ResNet" in lbl for lbl in legend_labels)
    assert any("ViT" in lbl for lbl in legend_labels)
    # ...and a cited human overlay on the same axis.
    assert any("human" in lbl.lower() for lbl in legend_labels)
    # The human threshold is drawn as a vertical reference line within the data range.
    xs = [line.get_xdata()[0] for line in ax.lines
          if len(set(line.get_xdata())) == 1 and len(line.get_xdata()) == 2]
    assert xs, "expected a vertical human-threshold line"
    assert np.isfinite(fig.axes[0].get_xlim()).all()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_figure_degrades_without_human_reference():
    """Same rendering path with no human curve available -> models only, with a note."""
    images, labels = syn.make_patch_data(160, size=24, num_classes=NUM_CLASSES)
    ds = Dataset(images=images, labels=labels, num_classes=NUM_CLASSES)
    model = syn.make_degradation_observer(NUM_CLASSES, fill=0.0)
    suite = DegradationSuite("occlusion", fill=0.0, block=False)
    levels = pe.linspace_levels(0.05, 0.6, 9)
    res = pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=1,
                     model_name="M")

    fig = res.compare([], human="auto", n_boot=20, seed=0)
    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert not any("human" in lbl.lower() for lbl in legend_labels)  # nothing fabricated
    note = " ".join(t.get_text() for t in ax.texts)
    assert "No published human reference" in note

    import matplotlib.pyplot as plt
    plt.close(fig)
