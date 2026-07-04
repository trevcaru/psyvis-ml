"""Contrast-threshold suite: sweep contrast, fit P(correct) vs. contrast per condition."""

from __future__ import annotations

import functools
import warnings

from ..stimuli.contrast import apply_contrast
from .base import Condition

__all__ = ["ContrastThreshold"]


class ContrastThreshold:
    """Sweep RMS or Michelson contrast and fit a psychometric function per condition.

    Parameters
    ----------
    spatial_freqs
        Optional list of spatial frequencies, one *condition* each. **Spatial-frequency
        conditioning is a stub** (PRD §7): it is not yet implemented, so the conditions apply
        identical contrast manipulation and differ only by label. A warning is emitted at
        construction when this is set, so results are not mistaken for a real SF sweep. When
        ``None`` (default), a single condition is produced.
    contrast_metric
        ``"rms"`` or ``"michelson"``.
    clip_range
        Optional ``(lo, hi)`` passed through to the contrast manipulation for pixel clipping.
    """

    sigmoid = "weibull"  # contrast is a positive axis, so Weibull is the natural family.

    def __init__(self, spatial_freqs=None, contrast_metric="rms", clip_range=None):
        if contrast_metric not in ("rms", "michelson"):
            raise ValueError(
                f"contrast_metric must be 'rms' or 'michelson'; got {contrast_metric!r}."
            )
        self.spatial_freqs = list(spatial_freqs) if spatial_freqs is not None else None
        self.contrast_metric = contrast_metric
        self.clip_range = clip_range
        if self.spatial_freqs is not None:
            warnings.warn(
                "spatial-frequency conditioning is a stub (not yet implemented); the "
                "requested spatial_freqs produce separate conditions by label only, with "
                "identical contrast manipulation. Do not read them as a real SF sweep.",
                RuntimeWarning,
                stacklevel=2,
            )

    def chance_level(self, num_classes: int, top_k: int) -> float:
        """Chance ~= k / num_classes for top-k argmax correctness."""
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive; got {num_classes}.")
        return min(1.0, float(top_k) / float(num_classes))

    def conditions(self) -> list[Condition]:
        freqs = self.spatial_freqs if self.spatial_freqs is not None else [None]
        conds = []
        for sf in freqs:
            fn = functools.partial(
                apply_contrast,
                metric=self.contrast_metric,
                spatial_freq=sf,
                clip_range=self.clip_range,
            )
            if sf is None:
                label = f"contrast[{self.contrast_metric}]"
            else:
                label = f"contrast[{self.contrast_metric}] sf={sf}"
            conds.append(
                Condition(
                    label=label,
                    apply_stimulus=fn,
                    stimulus_name=f"contrast:{self.contrast_metric}:sf={sf}",
                    metadata={"spatial_freq": sf, "contrast_metric": self.contrast_metric},
                )
            )
        return conds
