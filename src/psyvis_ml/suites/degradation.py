"""Degradation-robustness suite — the ImageNet-C bridge (PRD §7).

Sweeps a single degradation (gaussian noise, blur, occlusion, or spatial-frequency
filtering) by **severity** and fits **P(correct) vs. severity**, reporting a **threshold**
(the severity at criterion performance) and a **slope** — the same corruptions ImageNet-C
scores at a few fixed severities, but as *fitted curves* instead of accuracy scalars, so two
models with equal accuracy at one severity but thresholds an octave apart are told apart.

Because more severity means more degradation, P(correct) *decreases* with severity: the fit
uses a **decreasing logistic** (``sigmoid="logistic"`` with ``decreasing=True``, honoured by
:func:`psyvis_ml.measure`). The lower asymptote is still the observer's chance level — chance
is reached at *high* severity — so the same ``chance_level`` -> ``guess_rate`` wiring as every
other suite applies. The default argmax-top-1 observer model is the documented, swappable
choice noted throughout (PRD §14).
"""

from __future__ import annotations

import functools

from ..stimuli.degradation import (
    add_gaussian_noise,
    gaussian_blur,
    occlude,
    spatial_frequency_filter,
)
from .base import Condition

__all__ = ["DegradationSuite"]

# Registry: kind -> (manipulation callable, extra kwargs bound into the partial).
_KINDS = {
    "gaussian_noise": (add_gaussian_noise, {}),
    "blur": (gaussian_blur, {}),
    "occlusion": (occlude, {}),
    "lowpass": (spatial_frequency_filter, {"mode": "lowpass"}),
    "highpass": (spatial_frequency_filter, {"mode": "highpass"}),
}


class DegradationSuite:
    """Sweep one degradation by severity and fit a decreasing psychometric function.

    Parameters
    ----------
    kind
        One of ``"gaussian_noise"``, ``"blur"``, ``"occlusion"``, ``"lowpass"``,
        ``"highpass"``.
    **params
        Extra keyword arguments forwarded to the manipulation (e.g. ``clip_range`` for
        gaussian noise, ``fill`` / ``block`` for occlusion). They are folded into the
        condition's ``stimulus_name`` so distinct settings hash distinctly.
    """

    # Performance falls with severity, so we fit a *decreasing* logistic on the severity
    # axis; `decreasing` is read by measure() to bound the slope negative.
    sigmoid = "logistic"
    decreasing = True

    def __init__(self, kind, **params):
        if kind not in _KINDS:
            raise ValueError(
                f"unknown degradation kind {kind!r}; choose from {sorted(_KINDS)}."
            )
        self.kind = kind
        self.params = dict(params)

    def chance_level(self, num_classes: int, top_k: int) -> float:
        """Chance ~= k / num_classes for top-k argmax correctness (the high-severity asymptote)."""
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive; got {num_classes}.")
        return min(1.0, float(top_k) / float(num_classes))

    def conditions(self) -> list[Condition]:
        fn_base, fixed = _KINDS[self.kind]
        kwargs = {**fixed, **self.params}
        fn = functools.partial(fn_base, **kwargs) if kwargs else fn_base
        param_tag = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return [
            Condition(
                label=f"degradation[{self.kind}]",
                apply_stimulus=fn,
                stimulus_name=f"degradation:{self.kind}:{param_tag}",
                metadata={"kind": self.kind, **kwargs},
            )
        ]
