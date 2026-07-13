"""Reproducible run bundle for a method-of-constant-stimuli sweep.

A :class:`RunBundle` captures everything needed to reproduce and audit a sweep: the
resolved seed, a config hash, the library version, and the per-level ``(n_correct,
n_trials)`` counts. The same seed + config must produce an identical hash and identical
counts.

Framework-agnostic: NumPy + stdlib only, no model framework imports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

import numpy as np

__all__ = ["RunBundle", "compute_config_hash", "data_fingerprint", "library_version"]


def library_version() -> str:
    try:
        return version("psyvis-ml")
    except PackageNotFoundError:  # running from a source tree without install metadata
        return "0.0.0+unknown"


def data_fingerprint(images, labels) -> str:
    """Stable SHA-256 over the image set and labels, so the config hash is data-aware."""
    h = hashlib.sha256()
    arr = np.ascontiguousarray(np.asarray(images))
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    lab = np.ascontiguousarray(np.asarray(labels))
    h.update(str(lab.dtype).encode())
    h.update(lab.tobytes())
    return h.hexdigest()


def compute_config_hash(config: dict) -> str:
    """SHA-256 of a canonical JSON encoding of the run config.

    Callables (model, stimulus) are identified by name in the config, not by source, so a
    stable name yields a stable hash. Pass explicit ``model_name`` / ``stimulus_name`` to
    the sweep for reproducible hashing across processes.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class RunBundle:
    """Immutable record of one sweep.

    ``per_image`` optionally holds the raw per-image confidence signals as
    ``(n_levels, n_images)`` arrays keyed by ``"margin"``, ``"target_rank"``,
    ``"target_logit"``, ``"max_softmax"``, ``"predicted_label"``, and ``"correct"``. It feeds
    the descriptive confidence readout (:mod:`psyvis_ml.confidence`) and the per-item export
    (:mod:`psyvis_ml.per_item`) — it is *not* part of the config hash and does not feed the
    binomial fitting core.

    ``true_labels`` and ``item_ids`` are per-*image* (length ``n_images``, constant across
    levels): the class index each image was scored against, and a stable identifier for it.
    ``item_ids`` is empty when the caller supplied none, in which case consumers fall back to
    the image's positional index. Neither is part of the config hash — they identify rows, they
    are not experimental configuration (the image *content* is already fingerprinted).
    """

    seed: int
    config_hash: str
    library_version: str
    levels: tuple[float, ...]
    n_correct: tuple[int, ...]
    n_trials: tuple[int, ...]
    top_k: int
    config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    per_image: dict = field(default_factory=dict)
    true_labels: tuple[int, ...] = ()
    item_ids: tuple[str, ...] = ()

    def to_fit_inputs(self):
        """Return ``(levels, n_correct, n_trials)`` as arrays, ready for ``fit_psychometric``."""
        return (
            np.asarray(self.levels, dtype=float),
            np.asarray(self.n_correct, dtype=float),
            np.asarray(self.n_trials, dtype=float),
        )

    def margins(self) -> np.ndarray:
        """Per-image logit margins as a ``(n_levels, n_images)`` array (raises if absent)."""
        if "margin" not in self.per_image:
            raise KeyError("this bundle has no per-image margins (no confidence signal recorded).")
        return np.asarray(self.per_image["margin"], dtype=float)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "config_hash": self.config_hash,
            "library_version": self.library_version,
            "levels": list(self.levels),
            "n_correct": list(self.n_correct),
            "n_trials": list(self.n_trials),
            "top_k": self.top_k,
            "config": dict(self.config),
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            f"RunBundle(seed={self.seed}, config_hash={self.config_hash[:12]}…, "
            f"n_levels={len(self.levels)}, top_k={self.top_k}, "
            f"library_version={self.library_version!r})"
        )
