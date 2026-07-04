"""Suite protocol: what ``measure()`` needs from a measurement suite.

A suite declares (1) how to manipulate the stimulus at a given level — as one or more
*conditions*, each a callable following the ``apply_stimulus(image, level, rng)`` contract —
and (2) the observer's **chance level**, so the API can set the fitter's ``guess_rate`` from
chance and keep the fitting core classifier-agnostic (it never needs to know how many
classes the model has).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["Condition", "Suite"]


@dataclass(frozen=True)
class Condition:
    """One condition within a suite: a labeled stimulus-application function.

    ``apply_stimulus`` is a 3-arg callable ``(image, level, rng) -> image``. ``stimulus_name``
    is a stable identifier folded into the run bundle's config hash (avoids the ``<lambda>``
    / collision pitfalls the sweep engine warns about).
    """

    label: str
    apply_stimulus: Callable
    stimulus_name: str
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Suite(Protocol):
    """Structural protocol implemented by measurement suites."""

    #: Psychometric sigmoid to fit for this suite ("weibull" for positive-axis contrast).
    sigmoid: str

    def conditions(self) -> list[Condition]:
        """The conditions to sweep (one per, e.g., spatial frequency)."""
        ...

    def chance_level(self, num_classes: int, top_k: int) -> float:
        """Observer chance level, used as the fitter's ``guess_rate`` (lower asymptote)."""
        ...
