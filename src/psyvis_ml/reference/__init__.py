"""Static, cited human psychophysical reference data (PRD §6, §14).

We **ship pre-collected / published** human sensitivity curves as static reference data where
good ones exist, and we **never run human experiments or synthesize** human data. Each curve
carries its citation and an ``approximate`` flag, and lookups **degrade gracefully**: when no
credible human reference exists for a suite (e.g. distractor robustness, which has no
established human sensitivity curve), the lookup returns ``None`` so callers plot models only
rather than fabricating a curve.

Currently shipped:
  * ``contrast_sensitivity`` — the human photopic contrast sensitivity function (CSF), an
    approximate digitization of Campbell & Robson (1968). See
    ``data/contrast_sensitivity_human.json`` for the full citation and caveats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

import numpy as np

__all__ = [
    "HumanReference",
    "contrast_sensitivity_reference",
    "human_reference_for",
    "REFERENCE_PEAK_SF_CPD",
]

# Spatial frequency (cycles/deg) used for the human overlay when a contrast condition does not
# itself declare one: the CSF band-pass peak. Documented so the overlay is never silently
# picking an arbitrary operating point.
REFERENCE_PEAK_SF_CPD = 4.0


@dataclass(frozen=True)
class HumanReference:
    """A static, cited human reference curve.

    ``sensitivity`` is contrast sensitivity (1 / contrast-threshold) at each ``spatial_freq``;
    ``threshold()`` returns the corresponding contrast thresholds. ``approximate`` flags a
    digitized / representative curve (per §14) rather than exact tabulated trial data.
    """

    name: str
    suite: str
    quantity: str
    metric: str
    version: str
    approximate: bool
    citation: str
    source_note: str
    spatial_freq: np.ndarray
    sensitivity: np.ndarray
    human_paradigm: str = "grating detection"
    paradigm_caveat: str = ""

    def threshold(self) -> np.ndarray:
        """Contrast thresholds = 1 / sensitivity at each tabulated spatial frequency."""
        return 1.0 / self.sensitivity

    def threshold_at(self, spatial_freq) -> float:
        """Human contrast threshold at ``spatial_freq`` (log-log interpolated, clamped).

        Interpolation is linear in log-frequency / log-sensitivity, matching how the CSF is
        conventionally plotted; queries outside the tabulated range clamp to the endpoints.
        """
        sf = float(spatial_freq)
        logf = np.log(self.spatial_freq)
        logs = np.log(self.sensitivity)
        log_sens = float(np.interp(np.log(sf), logf, logs))  # np.interp clamps at the ends
        return float(np.exp(-log_sens))

    def label(self) -> str:
        approx = " (approx.)" if self.approximate else ""
        short = self.citation.split(".")[0]  # first author + year fragment
        return f"human {self.quantity}{approx} [{short}]"


def _load_json(filename: str) -> dict:
    data_pkg = resources.files(__package__).joinpath("data", filename)
    with data_pkg.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def contrast_sensitivity_reference() -> HumanReference:
    """The shipped human photopic CSF (approximate; Campbell & Robson, 1968)."""
    d = _load_json("contrast_sensitivity_human.json")
    return HumanReference(
        name=d["name"],
        suite=d["suite"],
        quantity=d["quantity"],
        metric=d["metric"],
        version=d["version"],
        approximate=bool(d["approximate"]),
        citation=d["citation"],
        source_note=d["source_note"],
        spatial_freq=np.asarray(d["spatial_frequency_cpd"], dtype=float),
        sensitivity=np.asarray(d["contrast_sensitivity"], dtype=float),
        human_paradigm=d.get("human_paradigm", "grating detection"),
        paradigm_caveat=d.get("paradigm_caveat", ""),
    )


# Registry of suite-class-name -> reference loader. Suites absent here have no human curve.
_REFERENCES = {
    "ContrastThreshold": contrast_sensitivity_reference,
}


def human_reference_for(suite_name: str):
    """Return the :class:`HumanReference` for a suite class name, or ``None`` if none exists.

    ``None`` is the honest, graceful answer for suites without a credible published human
    curve (e.g. ``DegradationSuite``, ``DistractorRobustness``) — callers overlay nothing and
    note the absence rather than inventing data (§14).
    """
    loader = _REFERENCES.get(suite_name)
    return loader() if loader is not None else None
