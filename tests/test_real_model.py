"""Real-model demo path: loader error handling (core-safe) + a deps/data-gated smoke test.

The end-to-end test runs only when torch+timm are installed AND a user ImageNet subset path
is provided via ``PSYVIS_IMAGENET_DIR``; otherwise it *skips* (never fails), so the core suite
stays green in a bare environment.
"""

import os

import numpy as np
import pytest

from psyvis_ml.datasets import imagenet_subset


def _has(mod):
    # Broad except on purpose: an absent OR broken/partial optional dep (e.g. a torch mid-
    # install raising OSError on its DLLs) must lead to a clean *skip*, never a collection
    # error that would take down the core suite.
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Core-safe: loader validation (no torch/timm/real images needed)
# --------------------------------------------------------------------------- #
def test_missing_root_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="supply your own ImageNet"):
        imagenet_subset(tmp_path / "does_not_exist")


def test_empty_root_raises(tmp_path):
    with pytest.raises(ValueError, match="no class subdirectories"):
        imagenet_subset(tmp_path)


def test_integer_folder_names_are_used_as_labels_without_mapping(tmp_path):
    # Folders named by integer index need no WNID mapping (so no timm import is triggered).
    (tmp_path / "207").mkdir()
    (tmp_path / "817").mkdir()
    # No image files -> the loader resolves labels, finds no images, and says so clearly.
    with pytest.raises(ValueError, match="no images"):
        imagenet_subset(tmp_path, wnid_to_index={})


def test_wnid_folder_without_mapping_entry_raises(tmp_path):
    (tmp_path / "n02099601").mkdir()
    # Providing an (empty) mapping avoids the timm lookup; the missing WNID is a clear KeyError.
    with pytest.raises(KeyError, match="integer ImageNet index or supply the mapping"):
        imagenet_subset(tmp_path, wnid_to_index={})


# --------------------------------------------------------------------------- #
# timm-gated: the WNID -> ImageNet-index map derived from the installed timm
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _has("timm"), reason="timm not installed (the [demo] extra)")
def test_imagenet_wnid_to_index_has_1000_known_mappings():
    from psyvis_ml.datasets import imagenet_wnid_to_index

    mapping = imagenet_wnid_to_index()
    assert len(mapping) == 1000
    assert all(k[0] == "n" and k[1:].isdigit() for k in mapping)  # every key is a WNID
    # Known ImageNet-1k anchors (stable across the standard class ordering).
    assert mapping["n01440764"] == 0     # tench
    assert mapping["n01443537"] == 1     # goldfish
    assert mapping["n02099601"] == 207   # golden retriever


@pytest.mark.skipif(not (_has("timm") and _has("PIL")),
                    reason="needs timm + Pillow (the [demo] extra)")
def test_wnid_folder_subset_loads_end_to_end(tmp_path):
    # Regression: a WNID-folder subset must load without the timm mapping crashing.
    from PIL import Image

    for wnid in ("n01440764", "n02099601"):
        d = tmp_path / wnid
        d.mkdir()
        for k in range(2):
            Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8)).save(d / f"{k}.png")

    ds = imagenet_subset(tmp_path, max_per_class=2, image_size=24)  # derives the WNID map
    assert ds.images.shape == (4, 24, 24, 3)
    assert set(ds.labels.tolist()) == {0, 207}     # tench, golden retriever
    assert ds.num_classes == 1000


# --------------------------------------------------------------------------- #
# Pillow-gated: a real image round-trips into a Dataset with the right labels
# --------------------------------------------------------------------------- #
def test_loads_integer_labeled_images(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    for cls in ("207", "817"):
        d = tmp_path / cls
        d.mkdir()
        for k in range(3):
            arr = np.full((16, 16, 3), (k * 20) % 256, dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img{k}.png")

    ds = imagenet_subset(tmp_path, max_per_class=2, image_size=24, num_classes=1000)
    assert ds.images.shape == (4, 24, 24, 3)          # 2 classes x 2 per class
    assert ds.images.dtype == np.float32 and ds.images.max() <= 1.0
    assert set(ds.labels.tolist()) == {207, 817}
    assert ds.num_classes == 1000


# --------------------------------------------------------------------------- #
# Deps + data gated: one tiny real model straight through measure()
# --------------------------------------------------------------------------- #
_DATA_DIR = os.environ.get("PSYVIS_IMAGENET_DIR")
_DEPS = _has("torch") and _has("timm")


@pytest.mark.skipif(not _DEPS, reason="torch+timm not installed (the [demo] extra)")
@pytest.mark.skipif(not (_DATA_DIR and os.path.isdir(_DATA_DIR)),
                    reason="set PSYVIS_IMAGENET_DIR to a prepared ImageFolder to run this")
def test_real_timm_model_end_to_end():
    import psyvis_ml as pe

    ds = pe.datasets.imagenet_subset(_DATA_DIR, max_classes=3, max_per_class=5, image_size=224)
    pretrained = os.environ.get("PSYVIS_TIMM_PRETRAINED", "1") == "1"
    try:
        model = pe.models.timm_classifier(
            os.environ.get("PSYVIS_TIMM_MODEL", "resnet18"), pretrained=pretrained)
    except Exception as e:  # noqa: BLE001 - e.g. offline weight download; skip, don't fail
        pytest.skip(f"could not build the timm model (offline?): {e}")

    suite = pe.suites.ContrastThreshold(contrast_metric="rms", clip_range=(0.0, 1.0))
    levels = pe.linspace_levels(0.05, 1.0, 5)
    res = pe.measure(model=model, suite=suite, dataset=ds, levels=levels, seed=0)

    frac = np.asarray(res.bundle.n_correct, float) / np.asarray(res.bundle.n_trials, float)
    assert frac.shape == (5,)
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    assert res.fit() is not None
