"""Eccentricity + crowding compositing stimulus.

Follows the sweep engine's ``apply_stimulus(image, level, rng)`` contract. The ``image`` is
the *target patch* (whatever labeled content the model classifies); this module pastes it
onto a larger canvas at a controlled **eccentricity** (distance from a fixation center) and,
optionally, surrounds it with **flankers** at a controlled center-to-center **spacing**.

* ``level`` is the flanker **spacing** — the crowding axis. Tight spacing crowds the target
  (flankers abut / overwrite its neighbourhood); wide spacing lets the target recover.
* **Eccentricity** is a *condition* parameter (like spatial frequency for contrast), baked
  into the per-condition ``functools.partial`` by :class:`psyvis_ml.suites.CrowdingSuite` —
  it is not the swept axis.
* Placement is **deterministic** given the position parameters. Any positional **jitter**
  draws from the supplied ``rng`` (and only then), so a jitter-free run ignores ``rng`` and
  reproduces exactly.

Geometry convention: image coordinates are ``(row, col)`` with ``row`` increasing downward.
``angle`` is measured from the ``+col`` (horizontal) axis, so ``angle=0`` places the target
directly to the right of fixation. The displacement of the target from fixation is
``(ecc * sin(angle), ecc * cos(angle))`` and flankers are placed along that same radial axis.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "radial_unit",
    "target_center",
    "flanker_offsets",
    "flanker_centers",
    "paste_patch",
    "composite_crowding",
]

DEFAULT_BACKGROUND = 0.0


def radial_unit(angle: float) -> np.ndarray:
    """Unit ``(drow, dcol)`` step along the eccentricity axis at ``angle`` (radians)."""
    return np.array([np.sin(angle), np.cos(angle)], dtype=float)


def _fixation(canvas_shape, fixation) -> np.ndarray:
    if fixation is not None:
        return np.asarray(fixation, dtype=float)
    h, w = canvas_shape[0], canvas_shape[1]
    return np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=float)


def target_center(canvas_shape, eccentricity, angle=0.0, fixation=None) -> np.ndarray:
    """``(row, col)`` of the target centre: fixation + ``eccentricity`` along ``angle``."""
    fix = _fixation(canvas_shape, fixation)
    return fix + float(eccentricity) * radial_unit(angle)


def flanker_offsets(spacing: float, n_flankers: int) -> list[float]:
    """Signed along-axis offsets of ``n_flankers`` flankers about the target.

    The nearest flanker on each side sits at ``±spacing`` (center-to-center), the next at
    ``±2*spacing``, and so on — so the target-to-nearest-flanker distance is ``spacing`` and
    adjacent centres on one side are ``spacing`` apart.
    """
    if n_flankers < 0:
        raise ValueError(f"n_flankers must be >= 0; got {n_flankers}.")
    offs: list[float] = []
    k = 1
    while len(offs) < n_flankers:
        offs.append(+k * float(spacing))
        if len(offs) < n_flankers:
            offs.append(-k * float(spacing))
        k += 1
    return offs


def flanker_centers(t_center, spacing, angle=0.0, n_flankers=2) -> list[np.ndarray]:
    """``(row, col)`` centres of each flanker along the radial axis about ``t_center``."""
    t_center = np.asarray(t_center, dtype=float)
    unit = radial_unit(angle)
    return [t_center + off * unit for off in flanker_offsets(spacing, n_flankers)]


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


def composite_crowding(
    image,
    level,
    rng=None,
    *,
    eccentricity,
    canvas_shape,
    angle=0.0,
    flanked=True,
    n_flankers=2,
    flanker_patch=None,
    fixation=None,
    background=DEFAULT_BACKGROUND,
    position_jitter=0.0,
):
    """Paste a target patch at ``eccentricity`` and optional flankers at spacing ``level``.

    Parameters
    ----------
    image
        The target patch (2-D array of pixel values) the model must classify.
    level
        Flanker **spacing** (center-to-center), the crowding axis. Ignored when
        ``flanked=False`` (the unflanked baseline).
    rng
        Used **only** for positional jitter; ignored when ``position_jitter == 0`` so
        jitter-free composites are deterministic.
    eccentricity
        Distance of the target centre from fixation, in pixels (a *condition* parameter).
    canvas_shape
        ``(H, W)`` of the output canvas.
    angle
        Radial-axis angle in radians from the ``+col`` axis (``0`` -> target to the right).
    flanked
        If True, place ``n_flankers`` flankers at spacing ``level``. If False, the target is
        presented alone (the unflanked baseline condition); ``level`` has no effect.
    n_flankers
        Number of flankers (default 2, one on each side along the radial axis).
    flanker_patch
        The patch pasted for each flanker. Defaults to a copy of the target patch
        (self-flanking, the classic crowding stimulus). Pass a distinct patch to flank with
        other content.
    fixation
        ``(row, col)`` fixation centre; defaults to the canvas centre.
    background
        Fill value for the canvas.
    position_jitter
        If ``> 0``, the target centre is jittered by ``rng.uniform(-j, +j)`` per axis; the
        flankers move with it, preserving their spacing.

    Returns
    -------
    numpy.ndarray
        The composited canvas of shape ``canvas_shape``.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"target patch must be 2-D (H, W); got shape {image.shape}.")
    if len(canvas_shape) != 2:
        raise ValueError(f"canvas_shape must be (H, W); got {canvas_shape}.")

    canvas = np.full(canvas_shape, float(background), dtype=float)
    t_center = target_center(canvas_shape, eccentricity, angle, fixation)

    if position_jitter and position_jitter > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        t_center = t_center + rng.uniform(-position_jitter, position_jitter, size=2)

    paste_patch(canvas, image, t_center)

    if flanked:
        fp = image if flanker_patch is None else np.asarray(flanker_patch, dtype=float)
        for fc in flanker_centers(t_center, float(level), angle, n_flankers):
            paste_patch(canvas, fp, fc)

    return canvas
