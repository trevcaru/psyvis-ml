"""Human reference data: cited, versioned, interpolable, and graceful when absent."""

import numpy as np
import pytest

from psyvis_ml.reference import (
    HumanReference,
    contrast_sensitivity_reference,
    human_reference_for,
)


def test_contrast_csf_loads_with_citation_and_flags():
    ref = contrast_sensitivity_reference()
    assert isinstance(ref, HumanReference)
    assert ref.suite == "ContrastThreshold"
    assert ref.metric == "michelson"
    assert ref.approximate is True                     # digitized/representative (PRD §14)
    assert "Campbell" in ref.citation and "1968" in ref.citation
    assert ref.version and ref.source_note             # non-empty provenance
    assert ref.spatial_freq.shape == ref.sensitivity.shape
    assert np.all(ref.sensitivity > 0)


def test_threshold_is_inverse_sensitivity():
    ref = contrast_sensitivity_reference()
    assert np.allclose(ref.threshold(), 1.0 / ref.sensitivity)


def test_threshold_at_interpolates_and_clamps():
    ref = contrast_sensitivity_reference()
    # At a tabulated peak (4 cpd, sensitivity 250) the threshold is 1/250.
    assert ref.threshold_at(4.0) == pytest.approx(1.0 / 250.0, rel=1e-6)
    # Between tabulated points, the threshold lies between the neighbours' thresholds.
    thr = ref.threshold_at(5.0)
    assert min(1 / 250.0, 1 / 220.0) <= thr <= max(1 / 250.0, 1 / 220.0)
    # Outside the tabulated range, clamp to the endpoints (no extrapolation blow-up).
    lo_sf = ref.spatial_freq.min()
    assert ref.threshold_at(lo_sf / 10.0) == pytest.approx(1.0 / ref.sensitivity[0], rel=1e-6)


def test_human_reference_for_contrast_present():
    ref = human_reference_for("ContrastThreshold")
    assert ref is not None and ref.suite == "ContrastThreshold"


@pytest.mark.parametrize("suite_name",
                         ["DistractorRobustness", "DegradationSuite", "SomethingElse"])
def test_human_reference_absent_returns_none(suite_name):
    # No fabricated human data for suites without a credible published curve (PRD §14).
    assert human_reference_for(suite_name) is None


def test_label_mentions_approximation():
    ref = contrast_sensitivity_reference()
    assert "approx" in ref.label().lower()
