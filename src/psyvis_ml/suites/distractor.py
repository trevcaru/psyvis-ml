"""Distractor-robustness suite.

Sweeps distractor **spacing** around a **centred** target and fits **P(correct) vs. spacing**,
so :meth:`~psyvis_ml.MeasureResult.threshold` reports the distractor-distance at which a
centred target recovers criterion accuracy — a *robustness* measure, not a perceptual crowding
zone. An optional **undistracted baseline** condition (target alone) gives the clean ceiling to
compare against.

Why centred (and not the earlier eccentricity/crowding framing): a diagnostic on real
ImageNet classifiers found that whole-photo targets are only reliably recognised when large and
central — the exact regime that leaves no room to place distractors at a peripheral
eccentricity. Ceiling and flanker-room were mutually exclusive, so the "crowding critical
spacing / human signature" claim was invalid for this stimulus. This suite keeps the compositor
machinery but makes only the honest claim it supports: how a nearby distractor's distance
degrades a centred classifier.

Observer-model note (PRD §14): "correct" defaults to **argmax top-1** on the composited
stimulus — a swappable observer-model choice, not a property of the fitter, which consumes only
``(spacing, n_correct, n_trials)``.
"""

from __future__ import annotations

import functools

from ..stimuli.distractor import composite_distractor
from .base import Condition

__all__ = ["DistractorRobustness"]


class DistractorRobustness:
    """Sweep distractor spacing around a centred target and fit P(correct) vs. spacing.

    Parameters
    ----------
    canvas_shape
        ``(H, W)`` of the composited canvas. Must be large enough to hold the centred target
        and its distractors at the widest swept spacing.
    angle
        Layout-axis angle (radians) along which distractors are placed about the target.
    n_distractors
        Number of distractors per composite (default 2, one on each side).
    distractor_patch
        Patch pasted for each distractor; defaults to a copy of the target.
    include_baseline
        If True (default), add an undistracted baseline condition (target alone).
    position_jitter
        Positional jitter (pixels) applied to the target centre, drawn from the sweep ``rng``.
    background
        Canvas fill value.
    """

    # Spacing is a positive axis and P(correct) rises with it (distant distractors interfere
    # less), so the Weibull is the natural family; threshold() is the distractor-distance
    # tolerance at criterion performance.
    sigmoid = "weibull"

    def __init__(self, *, canvas_shape=(96, 96), angle=0.0, n_distractors=2,
                 distractor_patch=None, include_baseline=True, position_jitter=0.0,
                 background=0.0):
        if len(canvas_shape) != 2:
            raise ValueError(f"canvas_shape must be (H, W); got {canvas_shape}.")
        self.canvas_shape = tuple(int(s) for s in canvas_shape)
        self.angle = float(angle)
        self.n_distractors = int(n_distractors)
        self.distractor_patch = distractor_patch
        self.include_baseline = bool(include_baseline)
        self.position_jitter = float(position_jitter)
        self.background = float(background)

    def chance_level(self, num_classes: int, top_k: int) -> float:
        """Chance ~= k / num_classes for top-k argmax correctness."""
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive; got {num_classes}.")
        return min(1.0, float(top_k) / float(num_classes))

    def _condition(self, distracted: bool) -> Condition:
        fn = functools.partial(
            composite_distractor,
            canvas_shape=self.canvas_shape,
            angle=self.angle,
            distracted=distracted,
            n_distractors=self.n_distractors,
            distractor_patch=self.distractor_patch,
            background=self.background,
            position_jitter=self.position_jitter,
        )  # target_offset defaults to 0 -> centred target
        kind = "distractors" if distracted else "undistracted"
        return Condition(
            label=kind,
            apply_stimulus=fn,
            stimulus_name=f"distractor:distracted={distracted}:n={self.n_distractors}",
            metadata={"distracted": distracted, "n_distractors": self.n_distractors,
                      "baseline": not distracted},
        )

    def conditions(self) -> list[Condition]:
        conds = [self._condition(distracted=True)]
        if self.include_baseline:
            conds.append(self._condition(distracted=False))
        return conds
