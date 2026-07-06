"""Distractor-robustness suite.

Sweeps distractor **size** around a **big centred target** and fits **P(correct) vs. distractor
size**, so :meth:`~psyvis_ml.MeasureResult.threshold` reports the distractor size at which a
centred target's accuracy falls to criterion — a *robustness* measure. An optional
**undistracted baseline** condition (target alone) gives the clean ceiling.

Why size, not spacing. A diagnostic on real ImageNet classifiers found that whole-photo targets
are only reliably recognised when large and central — the exact regime that leaves no room to
move a fixed-size distractor from "close" to "clear", so a *spacing* sweep is ceiling-limited
and returns extrapolated thresholds. This suite instead keeps the target big enough to clear the
recognition ceiling and grows the distractors from the margins: small distractors barely
interfere, large ones occlude the target. Performance therefore **falls** with distractor size
(``decreasing = True``, fit with a decreasing logistic), and the threshold is a real,
non-extrapolated distractor size at criterion.

Observer-model note (PRD §14): "correct" defaults to **argmax top-1** on the composited
stimulus — a swappable observer-model choice, not a property of the fitter, which consumes only
``(size, n_correct, n_trials)``.
"""

from __future__ import annotations

import functools

from ..stimuli.distractor import composite_distractor_size
from .base import Condition

__all__ = ["DistractorRobustness"]


class DistractorRobustness:
    """Sweep distractor size around a big centred target and fit P(correct) vs. size.

    Parameters
    ----------
    distractor_patch
        Content pasted for each distractor (resized to the swept size). Defaults to a neutral
        mid-gray square (an occluding distractor); pass a real image for a content distractor.
    canvas_shape
        ``(H, W)`` of the composited canvas — typically the model input size, e.g.
        ``(224, 224)``. The target should be pre-sized to clear the recognition ceiling (e.g.
        ~62% of the frame) so distractor size, not target loss, drives the curve.
    n_distractors
        Number of distractors placed in the margins (default 4: N, S, W, E).
    include_baseline
        If True (default), add an undistracted baseline condition (target alone).
    background
        Canvas fill value.
    """

    # Performance falls as distractor size grows, so this is a *decreasing* suite fit with a
    # decreasing logistic; threshold() is the distractor size at criterion accuracy.
    sigmoid = "logistic"
    decreasing = True
    #: Axis label for the swept variable (used by the plotting layer).
    x_label = "distractor size (px)"

    def __init__(self, *, distractor_patch=None, canvas_shape=(224, 224), n_distractors=4,
                 include_baseline=True, background=0.0):
        if len(canvas_shape) != 2:
            raise ValueError(f"canvas_shape must be (H, W); got {canvas_shape}.")
        self.distractor_patch = distractor_patch
        self.canvas_shape = tuple(int(s) for s in canvas_shape)
        self.n_distractors = int(n_distractors)
        self.include_baseline = bool(include_baseline)
        self.background = float(background)

    def chance_level(self, num_classes: int, top_k: int) -> float:
        """Chance ~= k / num_classes for top-k argmax correctness."""
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive; got {num_classes}.")
        return min(1.0, float(top_k) / float(num_classes))

    def _condition(self, distracted: bool) -> Condition:
        fn = functools.partial(
            composite_distractor_size,
            canvas_shape=self.canvas_shape,
            distractor_patch=self.distractor_patch,
            n_distractors=self.n_distractors,
            distracted=distracted,
            background=self.background,
        )
        kind = "distractors" if distracted else "undistracted"
        return Condition(
            label=kind,
            apply_stimulus=fn,
            stimulus_name=f"distractor_size:distracted={distracted}:n={self.n_distractors}",
            metadata={"distracted": distracted, "n_distractors": self.n_distractors,
                      "baseline": not distracted},
        )

    def conditions(self) -> list[Condition]:
        conds = [self._condition(distracted=True)]
        if self.include_baseline:
            conds.append(self._condition(distracted=False))
        return conds
