"""Optional ImageNet-subset loader for the real-model demos (behind the ``[demo]`` extra).

**You supply the images.** We never bundle ImageNet or download copyrighted data. Point this
loader at a directory *you* have prepared in the standard ImageFolder layout::

    <root>/<class>/img0.JPEG
    <root>/<class>/img1.JPEG
    ...

``<class>`` folder names may be either **integer ImageNet indices** (``"207"``) — the simplest
unambiguous contract — or **WordNet IDs** (``"n02099601"``), in which case you pass a
``wnid_to_index`` mapping (or let the loader derive one from ``timm`` when it is installed).

Pillow is imported lazily inside the functions, so ``import psyvis_ml`` still works on the core
dependencies alone; only *calling* these loaders needs the ``[demo]`` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import Dataset

__all__ = ["imagenet_subset", "load_image", "imagenet_wnid_to_index"]

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG", ".bmp", ".webp")


def load_image(path, image_size=224):
    """Load one image as a float32 ``(H, W, 3)`` array in ``[0, 1]`` (Pillow, lazy import)."""
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover - exercised only without the demo extra
        raise ImportError(
            "loading real images needs Pillow; install the demo extra: "
            'pip install -e ".[demo]"'
        ) from e
    with Image.open(path) as im:
        im = im.convert("RGB")
        if image_size is not None:
            im = im.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def _is_wnid(s) -> bool:
    """A WordNet ID looks like ``n01440764`` — an ``n`` followed by digits."""
    return isinstance(s, str) and len(s) > 1 and s[0] == "n" and s[1:].isdigit()


def _imagenet_label_names(info) -> list:
    """Ordered list of the 1000 ImageNet-1k label names (WNIDs), across ``timm`` versions.

    Current ``timm`` exposes ``label_names()`` (the ordered list) plus ``num_classes()`` and
    ``index_to_label_name(i)``; older builds exposed ``len(info)`` and
    ``index_to_synset``/``index_to_wnid``. We try the accessors an installed ``timm`` actually
    has rather than assuming any one of them.
    """
    names_fn = getattr(info, "label_names", None)
    if callable(names_fn):
        try:
            names = list(names_fn())
            if names:
                return names
        except Exception:  # noqa: BLE001 - fall back to a per-index accessor
            pass

    n = None
    num_fn = getattr(info, "num_classes", None)
    if callable(num_fn):
        n = int(num_fn())
    else:
        try:
            n = len(info)  # older timm made ImageNetInfo sized
        except TypeError:
            n = None

    if n:
        for meth in ("index_to_label_name", "index_to_synset", "index_to_wnid"):
            fn = getattr(info, meth, None)
            if callable(fn):
                try:
                    return [fn(i) for i in range(n)]
                except Exception:  # noqa: BLE001 - try the next accessor
                    continue
    return []


def imagenet_wnid_to_index() -> dict:
    """WordNet-ID → ImageNet-1k index map, derived from ``timm`` (not hardcoded).

    Raises a clear error (rather than guessing) if ``timm`` is missing or exposes an
    unexpected API — pass ``wnid_to_index`` explicitly in that case.
    """
    try:
        from timm.data import ImageNetInfo
    except ImportError as e:  # pragma: no cover - demo-only path
        raise ImportError(
            'deriving the WNID->index map needs timm; install the demo extra '
            '(pip install -e ".[demo]") or pass wnid_to_index explicitly.'
        ) from e

    info = ImageNetInfo()
    labels = _imagenet_label_names(info)
    mapping = {wnid: i for i, wnid in enumerate(labels)}
    if len(mapping) == 1000 and all(_is_wnid(k) for k in mapping):
        return mapping
    available = [m for m in dir(info) if not m.startswith("_")]
    raise RuntimeError(
        "could not derive a 1000-class WNID->index map from timm.data.ImageNetInfo "
        f"(got {len(mapping)} WNID-shaped entries; installed timm exposes {available}); "
        "pass wnid_to_index={'nXXXXXXXX': index, ...} explicitly."
    )


def _resolve_label(folder_name, wnid_to_index):
    if folder_name.isdigit():
        return int(folder_name)
    if folder_name not in wnid_to_index:
        raise KeyError(
            f"class folder {folder_name!r} is not an integer index and is absent from "
            "wnid_to_index; name folders by integer ImageNet index or supply the mapping."
        )
    return int(wnid_to_index[folder_name])


def imagenet_subset(root, *, classes=None, max_classes=None, max_per_class=20,
                    image_size=224, wnid_to_index=None, num_classes=1000, seed=0):
    """Load a small ImageNet subset from a user-provided ImageFolder directory.

    Parameters
    ----------
    root
        Path to the ImageFolder-style directory you prepared (see module docstring).
    classes
        Optional explicit list of class folder names to include; otherwise all are used
        (subject to ``max_classes``).
    max_classes, max_per_class
        Caps for a quick demo curve — a handful of classes and a few hundred images is plenty.
    image_size
        Images are loaded/resized to ``(image_size, image_size)``; the model wrapper resizes
        again to its own input if needed.
    wnid_to_index
        Mapping from WordNet-ID folder names to ImageNet indices (only needed for WNID
        folders; integer-named folders are used directly).
    num_classes
        Class count for the returned :class:`~psyvis_ml.datasets.Dataset` (1000 for standard
        ImageNet models, so chance level is 1/1000).
    seed
        Deterministic shuffling of the per-class file order before applying ``max_per_class``.

    Returns
    -------
    Dataset
        ``images`` is ``(N, image_size, image_size, 3)`` float32 in ``[0, 1]``; ``labels`` are
        ImageNet class indices; ``item_ids`` are the source file paths relative to ``root``, so
        a per-item export can be traced back to the exact image.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"ImageNet subset root {str(root)!r} not found. You must supply your own ImageNet "
            "images (this package does not bundle or download them); see the docstring."
        )
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if classes is not None:
        wanted = set(classes)
        class_dirs = [d for d in class_dirs if d.name in wanted]
    if not class_dirs:
        raise ValueError(f"no class subdirectories found under {str(root)!r}.")
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]

    # Build the WNID map once if any class folder is named by WordNet ID rather than index.
    if wnid_to_index is None and any(not d.name.isdigit() for d in class_dirs):
        wnid_to_index = imagenet_wnid_to_index()

    rng = np.random.default_rng(seed)
    images, labels, item_ids, used = [], [], [], []
    for d in class_dirs:
        label = _resolve_label(d.name, wnid_to_index or {})
        files = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix in _IMAGE_EXTS)
        if not files:
            continue
        order = rng.permutation(len(files))
        for j in order[:max_per_class]:
            images.append(load_image(files[j], image_size=image_size))
            labels.append(label)
            # Path relative to the root: a stable item id that survives moving the dataset.
            item_ids.append(files[j].relative_to(root).as_posix())
        used.append(d.name)

    if not images:
        raise ValueError(
            f"no images with extensions {_IMAGE_EXTS} found under the class folders in "
            f"{str(root)!r}."
        )
    return Dataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=int),
        num_classes=int(num_classes),
        name=f"imagenet-subset({root.name})",
        metadata={"classes": used, "image_size": image_size, "n_images": len(images)},
        item_ids=tuple(item_ids),
    )
