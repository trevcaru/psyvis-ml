"""Distractor compositing stimulus.

Follows the sweep engine's ``apply_stimulus(image, level, rng)`` contract. The ``image`` is
the *target patch* (whatever labeled content the model classifies); this module pastes it onto
a larger canvas — **centred** — and surrounds it with **distractor** patches.

Two compositors, two swept axes:

* :func:`composite_distractor_size` (the one :class:`psyvis_ml.suites.DistractorRobustness`
  uses): a **big centred target** that clears the recognition ceiling, with distractors of
  swept **size** pasted flush against the frame margins. Small distractors sit in the margins
  and barely interfere; large ones grow inward and occlude the target — so P(correct) *falls*
  as the size grows (a *decreasing* suite). This is the geometry that yields a real,
  non-extrapolated threshold on whole-photo ImageNet targets.
* :func:`composite_distractor`: the lower-level compositor that places distractors at a
  controlled center-to-center **spacing** about the target — kept as generic machinery.

This is deliberately *not* a model of human crowding: the target is centred (no visual-field
eccentricity) and there is no perceptual critical-spacing claim.

Geometry convention: image coordinates are ``(row, col)`` with ``row`` increasing downward.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "radial_unit",
    "target_center",
    "distractor_offsets",
    "distractor_centers",
    "margin_anchor_centers",
    "resize_square",
    "paste_patch",
    "composite_distractor",
    "composite_distractor_size",
]

DEFAULT_BACKGROUND = 0.0


def radial_unit(angle: float) -> np.ndarray:
    """Unit ``(drow, dcol)`` step along the layout axis at ``angle`` (radians)."""
    return np.array([np.sin(angle), np.cos(angle)], dtype=float)


def _fixation(canvas_shape, fixation) -> np.ndarray:
    if fixation is not None:
        return np.asarray(fixation, dtype=float)
    h, w = canvas_shape[0], canvas_shape[1]
    return np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=float)


def target_center(canvas_shape, target_offset=0.0, angle=0.0, fixation=None) -> np.ndarray:
    """``(row, col)`` of the target centre: fixation + ``target_offset`` along ``angle``."""
    fix = _fixation(canvas_shape, fixation)
    return fix + float(target_offset) * radial_unit(angle)


def distractor_offsets(spacing: float, n_distractors: int) -> list[float]:
    """Signed along-axis offsets of ``n_distractors`` distractors about the target.

    The nearest distractor on each side sits at ``±spacing`` (center-to-center), the next at
    ``±2*spacing``, and so on — so the target-to-nearest-distractor distance is ``spacing`` and
    adjacent centres on one side are ``spacing`` apart.
    """
    if n_distractors < 0:
        raise ValueError(f"n_distractors must be >= 0; got {n_distractors}.")
    offs: list[float] = []
    k = 1
    while len(offs) < n_distractors:
        offs.append(+k * float(spacing))
        if len(offs) < n_distractors:
            offs.append(-k * float(spacing))
        k += 1
    return offs


def distractor_centers(t_center, spacing, angle=0.0, n_distractors=2) -> list[np.ndarray]:
    """``(row, col)`` centres of each distractor along the layout axis about ``t_center``."""
    t_center = np.asarray(t_center, dtype=float)
    unit = radial_unit(angle)
    return [t_center + off * unit for off in distractor_offsets(spacing, n_distractors)]


def paste_patch(canvas, patch, center) -> None:
    """Overwrite ``canvas`` in place with ``patch`` centred at ``center`` (row, col).

    The patch is rounded to the nearest integer top-left and clipped to the canvas bounds, so
    a patch partially off-canvas pastes only its visible part.
    """
    patch = np.asarray(patch, dtype=float)
    ph, pw = patch.shape[0], patch.shape[1]
    cy, cx = float(center[0]), float(center[1])
    top = int(round(cy - (ph - 1) / 2.0))
    left = int(round(cx - (pw - 1) / 2.0))

    ch, cw = canvas.shape[0], canvas.shape[1]
    # Intersection of the patch rectangle with the canvas.
    r0, r1 = max(0, top), min(ch, top + ph)
    c0, c1 = max(0, left), min(cw, left + pw)
    if r0 >= r1 or c0 >= c1:
        return  # entirely off-canvas
    pr0, pc0 = r0 - top, c0 - left
    canvas[r0:r1, c0:c1] = patch[pr0:pr0 + (r1 - r0), pc0:pc0 + (c1 - c0)]


def composite_distractor(
    image,
    level,
    rng=None,
    *,
    canvas_shape,
    target_offset=0.0,
    angle=0.0,
    distracted=True,
    n_distractors=2,
    distractor_patch=None,
    fixation=None,
    background=DEFAULT_BACKGROUND,
    position_jitter=0.0,
):
    """Paste a target patch (centred by default) and optional distractors at spacing ``level``.

    Parameters
    ----------
    image
        The target patch the model must classify: a 2-D ``(H, W)`` grayscale array or a 3-D
        ``(H, W, C)`` array (e.g. an RGB natural image).
    level
        Distractor **spacing** (center-to-center). Ignored when ``distracted=False`` (the
        undistracted baseline).
    rng
        Used **only** for positional jitter; ignored when ``position_jitter == 0`` so
        jitter-free composites are deterministic.
    canvas_shape
        ``(H, W)`` of the output canvas.
    target_offset
        Distance of the target centre from fixation, in pixels (default ``0`` = centred).
    angle
        Layout-axis angle in radians from the ``+col`` axis (``0`` -> distractors horizontal).
    distracted
        If True, place ``n_distractors`` distractors at spacing ``level``. If False, the target
        is presented alone (the undistracted baseline); ``level`` has no effect.
    n_distractors
        Number of distractors (default 2, one on each side along the layout axis).
    distractor_patch
        The patch pasted for each distractor. Defaults to a copy of the target patch. Pass a
        distinct patch to surround the target with other content.
    fixation
        ``(row, col)`` fixation centre; defaults to the canvas centre.
    background
        Fill value for the canvas.
    position_jitter
        If ``> 0``, the target centre is jittered by ``rng.uniform(-j, +j)`` per axis; the
        distractors move with it, preserving their spacing.

    Returns
    -------
    numpy.ndarray
        The composited canvas: shape ``canvas_shape`` for a 2-D target, or
        ``canvas_shape + (C,)`` for a 3-D ``(H, W, C)`` target (e.g. an RGB natural image).
    """
    image = np.asarray(image, dtype=float)
    if image.ndim not in (2, 3):
        raise ValueError(
            f"target patch must be 2-D (H, W) or 3-D (H, W, C); got shape {image.shape}."
        )
    if len(canvas_shape) != 2:
        raise ValueError(f"canvas_shape must be (H, W); got {canvas_shape}.")

    # Match the target's channel layout so RGB targets composite onto an RGB canvas.
    canvas = np.full(tuple(canvas_shape) + image.shape[2:], float(background), dtype=float)
    t_center = target_center(canvas_shape, target_offset, angle, fixation)

    if position_jitter and position_jitter > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        t_center = t_center + rng.uniform(-position_jitter, position_jitter, size=2)

    paste_patch(canvas, image, t_center)

    if distracted:
        dp = image if distractor_patch is None else np.asarray(distractor_patch, dtype=float)
        for dc in distractor_centers(t_center, float(level), angle, n_distractors):
            paste_patch(canvas, dp, dc)

    return canvas


def resize_square(patch, size):
    """Resize a patch to ``size × size`` (bilinear), preserving any trailing channel axis."""
    patch = np.asarray(patch, dtype=float)
    size = int(size)
    if patch.shape[0] == size and patch.shape[1] == size:
        return patch
    from scipy.ndimage import zoom
    factors = [size / patch.shape[0], size / patch.shape[1]] + [1.0] * (patch.ndim - 2)
    return zoom(patch, factors, order=1)


def margin_anchor_centers(canvas_shape, size, n_distractors) -> list[tuple[float, float]]:
    """``(row, col)`` distractor centres flush to the frame edges (N, S, W, E, cycled).

    Each centre is half a distractor-size inside its edge, so an ``size × size`` distractor sits
    against that edge and grows *inward* (toward the centred target) as ``size`` increases.
    """
    h, w = canvas_shape[0], canvas_shape[1]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    half = size / 2.0
    edges = [(half, cx), (h - 1 - half, cx), (cy, half), (cy, w - 1 - half)]  # N, S, W, E
    return [edges[i % len(edges)] for i in range(n_distractors)]


def composite_distractor_size(image, level, rng=None, *, canvas_shape, distractor_patch=None,
                              n_distractors=4, distracted=True,
                              background=DEFAULT_BACKGROUND):
    """Centre a target and paste ``n_distractors`` of SIZE ``level`` in the frame margins.

    Parameters
    ----------
    image
        The (already big) target patch, centred on the canvas as-is: a 2-D ``(H, W)`` or 3-D
        ``(H, W, C)`` array.
    level
        Distractor **size** in pixels (the swept axis). Small = little interference; large =
        distractors grow inward and occlude the target, so P(correct) falls with ``level``.
    rng
        Accepted for the ``apply_stimulus`` contract and ignored (placement is deterministic).
    canvas_shape
        ``(H, W)`` of the output canvas (typically the model input size, e.g. ``(224, 224)``).
    distractor_patch
        Content pasted for each distractor, resized to ``level × level``. Defaults to a neutral
        mid-gray square (an occluding distractor) when ``None``.
    n_distractors
        Number of distractors placed in the margins (default 4: N, S, W, E).
    distracted
        If False, the target is presented alone (the undistracted baseline ceiling); ``level``
        has no effect.
    background
        Canvas fill value.

    Returns
    -------
    numpy.ndarray
        The composited canvas of shape ``canvas_shape`` (+ channels for an RGB target).
    """
    image = np.asarray(image, dtype=float)
    if image.ndim not in (2, 3):
        raise ValueError(
            f"target patch must be 2-D (H, W) or 3-D (H, W, C); got shape {image.shape}."
        )
    if len(canvas_shape) != 2:
        raise ValueError(f"canvas_shape must be (H, W); got {canvas_shape}.")

    canvas = np.full(tuple(canvas_shape) + image.shape[2:], float(background), dtype=float)
    paste_patch(canvas, image, target_center(canvas_shape))  # centred target

    size = int(round(float(level)))
    if distracted and size >= 1 and n_distractors > 0:
        if distractor_patch is None:
            dp = np.full((size, size) + image.shape[2:], 0.5, dtype=float)  # neutral gray
        else:
            dp = resize_square(np.asarray(distractor_patch, dtype=float), size)
        for center in margin_anchor_centers(canvas_shape, size, n_distractors):
            paste_patch(canvas, dp, center)

    return canvas
