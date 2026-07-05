"""Eccentricity + crowding suite — the PRD §7 differentiator.

Sweeps flanker **spacing** (the crowding axis) and fits **P(correct) vs. spacing** per
**eccentricity**, so :meth:`~psyvis_ml.MeasureResult.threshold` reports the **critical
spacing** — the spacing at which a crowded target recovers criterion performance, i.e. the
size of the crowding zone. Eccentricity is a *condition* parameter (one condition per value),
like spatial frequency for the contrast suite. An optional **unflanked baseline** condition
(target alone, spacing has no effect) gives the uncrowded ceiling to compare against.

Observer-model caveat (PRD §14). As with every suite here, "correct" defaults to
**argmax top-1** on the composited stimulus. That is a deliberately coarse observer model for
crowding: a classifier's top-1 decision on a target-plus-flankers composite is not the same
measurement as a human's forced-choice identification of the target, and the fitted critical
spacing should be read as *this* observer's crowding zone, not a direct human analogue. The
observer model is swappable (top-k, a 2AFC pairing, a trained probe) without changing the
suite; the fitter consumes only ``(spacing, n_correct, n_trials)`` either way.
"""

from __future__ import annotations

import functools

from ..stimuli.crowding import composite_crowding
from .base import Condition

__all__ = ["CrowdingSuite"]


class CrowdingSuite:
    """Sweep flanker spacing and fit a crowding function per eccentricity.

    Parameters
    ----------
    eccentricities
        One *condition* per value: the target's distance from fixation, in pixels.
    canvas_shape
        ``(H, W)`` of the composited canvas. Must be large enough to hold the target and its
        flankers at the widest swept spacing for the largest eccentricity.
    angle
        Radial-axis angle (radians) along which eccentricity and flankers are laid out.
    n_flankers
        Number of flankers per composite (default 2, one on each side).
    flanker_patch
        Patch pasted for each flanker; defaults to a copy of the target (self-flanking).
    include_baseline
        If True (default), add an unflanked baseline condition per eccentricity.
    position_jitter
        Positional jitter (pixels) applied to the target centre, drawn from the sweep ``rng``.
    background
        Canvas fill value.
    """

    # Spacing is a positive axis and P(correct) rises with it (wider spacing -> less
    # crowding), so the Weibull is the natural family; threshold() is the critical spacing.
    sigmoid = "weibull"

    def __init__(self, eccentricities, *, canvas_shape=(96, 96), angle=0.0, n_flankers=2,
                 flanker_patch=None, include_baseline=True, position_jitter=0.0,
                 background=0.0):
        self.eccentricities = list(eccentricities)
        if not self.eccentricities:
            raise ValueError("eccentricities must be non-empty.")
        if len(canvas_shape) != 2:
            raise ValueError(f"canvas_shape must be (H, W); got {canvas_shape}.")
        self.canvas_shape = tuple(int(s) for s in canvas_shape)
        self.angle = float(angle)
        self.n_flankers = int(n_flankers)
        self.flanker_patch = flanker_patch
        self.include_baseline = bool(include_baseline)
        self.position_jitter = float(position_jitter)
        self.background = float(background)

    def chance_level(self, num_classes: int, top_k: int) -> float:
        """Chance ~= k / num_classes for top-k argmax correctness."""
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive; got {num_classes}.")
        return min(1.0, float(top_k) / float(num_classes))

    def _condition(self, ecc, flanked: bool) -> Condition:
        fn = functools.partial(
            composite_crowding,
            eccentricity=ecc,
            canvas_shape=self.canvas_shape,
            angle=self.angle,
            flanked=flanked,
            n_flankers=self.n_flankers,
            flanker_patch=self.flanker_patch,
            background=self.background,
            position_jitter=self.position_jitter,
        )
        kind = "crowding" if flanked else "unflanked"
        return Condition(
            label=f"{kind} ecc={ecc}",
            apply_stimulus=fn,
            stimulus_name=f"crowding:ecc={ecc}:flanked={flanked}:n={self.n_flankers}",
            metadata={"eccentricity": ecc, "flanked": flanked,
                      "n_flankers": self.n_flankers, "baseline": not flanked},
        )

    def conditions(self) -> list[Condition]:
        conds = []
        for ecc in self.eccentricities:
            conds.append(self._condition(ecc, flanked=True))
            if self.include_baseline:
                conds.append(self._condition(ecc, flanked=False))
        return conds
