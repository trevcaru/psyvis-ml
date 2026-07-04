"""Method-of-constant-stimuli sweep engine.

Present a labeled image set at each of a fixed list of stimulus levels, score the model's
correctness, and return per-level ``(n_correct, n_trials)`` as a reproducible
:class:`~psyvis_ml.sweep.bundle.RunBundle`.

Design constraints:
  * ``model`` is just a callable ``image -> logits`` (or ``images -> logits`` when
    ``batched=True``). No torch/tf/jax import here — framework-agnostic by construction.
  * No sequential dependency: every level (and every image within a level) is scored
    independently, so the work is embarrassingly parallel. Pass ``map_fn`` (e.g. a thread
    pool's ``map``) to parallelize across levels; the default is sequential ``map``.
  * Reproducible: a resolved integer seed drives a per-element ``numpy`` RNG derived from
    ``SeedSequence([seed, level_index, image_index])`` — independent of evaluation order,
    so parallel and sequential runs with the same seed produce identical counts.

The stimulus contract is ``apply_stimulus(image, level, rng)`` (or, when ``batched=True``,
``apply_stimulus(images, level, rng)``): any *stochastic* manipulation must draw from the
supplied ``rng`` so runs stay reproducible; deterministic manipulations simply ignore it.
"""

from __future__ import annotations

import warnings

import numpy as np

from .bundle import RunBundle, compute_config_hash, data_fingerprint, library_version
from .scoring import count_correct

__all__ = ["run_sweep"]


def _resolve_seed(seed) -> int:
    if seed is None:
        return int(np.random.SeedSequence().entropy)
    return int(seed)


def _child_rng(seed: int, *keys: int) -> np.random.Generator:
    """Deterministic, order-independent RNG for one sweep element."""
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def _callable_name(fn, explicit) -> str:
    if explicit is not None:
        return str(explicit)
    return getattr(fn, "__name__", None) or type(fn).__name__


def _resolve_and_check_names(model, model_name, apply_stimulus, stimulus_name):
    """Resolve model/stimulus identifiers for the config hash and warn on unstable ones.

    The config hash identifies callables by name, so a ``<lambda>`` (whose name is not a
    stable identifier) or a model/stimulus name collision would let two *different* runs
    hash identically — silently corrupting the reproducibility bundle. We warn rather than
    error so exploratory use still works; pass explicit ``model_name``/``stimulus_name`` to
    silence it.
    """
    m = _callable_name(model, model_name)
    s = _callable_name(apply_stimulus, stimulus_name)
    for role, name in (("model", m), ("stimulus", s)):
        if name == "<lambda>":
            warnings.warn(
                f"{role} resolved to the non-stable name '<lambda>'; the config hash will "
                f"not distinguish it from other lambdas. Pass an explicit "
                f"{role}_name=... for a reproducible hash.",
                RuntimeWarning,
                stacklevel=3,
            )
    if m == s:
        warnings.warn(
            f"model and stimulus resolved to the same name {m!r}; the config hash cannot "
            f"tell them apart. Pass distinct model_name=/stimulus_name= to avoid a silent "
            f"hash collision.",
            RuntimeWarning,
            stacklevel=3,
        )
    return m, s


def run_sweep(
    model,
    images,
    labels,
    levels,
    apply_stimulus,
    *,
    top_k=1,
    batched=False,
    seed=None,
    map_fn=map,
    model_name=None,
    stimulus_name=None,
    extra_config=None,
):
    """Run a method-of-constant-stimuli sweep.

    Parameters
    ----------
    model
        Callable. Per-image mode: ``image -> logits`` (1-D, length ``n_classes``). Batched
        mode (``batched=True``): ``images_stack -> logits`` (``(n, n_classes)``).
    images, labels
        The labeled image set. ``images`` is any sequence/array indexable per sample;
        ``labels`` are integer class indices, one per image. Every image is presented at
        every level (method of constant stimuli), so ``n_trials`` per level == ``len(images)``.
    levels
        The stimulus levels to sweep (e.g. contrast values).
    apply_stimulus
        ``apply_stimulus(image, level, rng)`` returning the manipulated image (batched:
        ``apply_stimulus(images, level, rng)`` returning the manipulated stack).
    top_k
        Score top-``k`` argmax correctness (``top_k=1`` is standard top-1).
    batched
        If True, the stimulus and model operate on the whole image stack per level.
    seed
        Integer seed for reproducibility. If None, a fresh seed is drawn and recorded in the
        returned bundle so the run can still be reproduced.
    map_fn
        Mapping function over level indices (default ``map``); pass a parallel map to run
        levels concurrently. Results are collected and re-sorted by level, so ordering does
        not matter.
    model_name, stimulus_name
        Optional stable identifiers folded into the config hash (recommended for lambdas /
        cross-process reproducibility).
    extra_config
        Optional extra dict merged into the hashed config (e.g. suite parameters).

    Returns
    -------
    RunBundle
    """
    labels = np.asarray(labels).ravel()
    n_images = len(images)
    if n_images != labels.shape[0]:
        raise ValueError(f"len(images)={n_images} != len(labels)={labels.shape[0]}.")
    if n_images == 0:
        raise ValueError("empty image set.")
    levels = [float(x) for x in levels]
    if len(levels) == 0:
        raise ValueError("no stimulus levels given.")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1; got {top_k}.")

    seed = _resolve_seed(seed)
    model_id, stimulus_id = _resolve_and_check_names(
        model, model_name, apply_stimulus, stimulus_name
    )

    def score_level(i: int):
        level = levels[i]
        if batched:
            rng = _child_rng(seed, i)
            stim = apply_stimulus(images, level, rng)
            logits = np.asarray(model(stim), dtype=float)
        else:
            rows = []
            for j in range(n_images):
                rng = _child_rng(seed, i, j)
                stim = apply_stimulus(images[j], level, rng)
                rows.append(np.asarray(model(stim), dtype=float).ravel())
            logits = np.vstack(rows)
        n_correct = count_correct(logits, labels, k=top_k)
        return i, n_correct, logits.shape[1]

    results = list(map_fn(score_level, range(len(levels))))
    results.sort(key=lambda r: r[0])  # re-sort: map_fn may be parallel / out of order
    n_correct = tuple(int(r[1]) for r in results)
    n_classes = int(results[0][2]) if results else 0
    n_trials = tuple(n_images for _ in levels)

    config = {
        "levels": levels,
        "top_k": int(top_k),
        "batched": bool(batched),
        "seed": seed,
        "n_images": n_images,
        "data_fingerprint": data_fingerprint(images, labels),
        "model": model_id,
        "stimulus": stimulus_id,
        "extra": dict(extra_config) if extra_config else {},
    }
    config_hash = compute_config_hash(config)

    return RunBundle(
        seed=seed,
        config_hash=config_hash,
        library_version=library_version(),
        levels=tuple(levels),
        n_correct=n_correct,
        n_trials=n_trials,
        top_k=int(top_k),
        config=config,
        metadata={"n_images": n_images, "n_classes": n_classes, "batched": bool(batched)},
    )
