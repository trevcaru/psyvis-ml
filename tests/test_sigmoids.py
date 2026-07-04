"""Unit tests for the sigmoid family: values, derivatives, inverses, composition."""

import numpy as np
import pytest

from psyvis_ml.fitting import sigmoids as sg


def test_weibull_reference_points():
    # At x == alpha, F = 1 - 1/e.
    assert sg.weibull(1.0, 1.0, 3.0) == pytest.approx(1.0 - np.exp(-1.0))
    # Monotone increasing in x.
    xs = np.linspace(0.01, 5.0, 50)
    f = sg.weibull(xs, 1.0, 2.0)
    assert np.all(np.diff(f) > 0)
    assert np.all((f > 0) & (f < 1))


def test_weibull_undefined_nonpositive():
    out = sg.weibull(np.array([-1.0, 0.0, 1.0]), 1.0, 2.0)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert np.isfinite(out[2])


def test_logistic_reference_points():
    # Midpoint at x == alpha.
    assert sg.logistic(0.5, 0.5, 3.0) == pytest.approx(0.5)
    # Handles signed axes.
    assert 0.0 < sg.logistic(-10.0, 0.0, 1.0) < 0.5
    assert 0.5 < sg.logistic(10.0, 0.0, 1.0) < 1.0


@pytest.mark.parametrize("name", ["weibull", "logistic"])
def test_inverse_roundtrip(name):
    fam = sg.SIGMOIDS[name]
    f, inv = fam["f"], fam["inverse"]
    alpha, beta = (1.3, 2.5) if name == "weibull" else (0.2, 1.7)
    xs = np.linspace(0.05, 4.0, 20) if name == "weibull" else np.linspace(-4.0, 4.0, 20)
    p = f(xs, alpha, beta)
    x_back = inv(p, alpha, beta)
    assert np.allclose(x_back, xs, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("name", ["weibull", "logistic"])
def test_deriv_matches_finite_difference(name):
    fam = sg.SIGMOIDS[name]
    f, d = fam["f"], fam["deriv"]
    alpha, beta = (1.1, 3.0) if name == "weibull" else (0.0, 2.0)
    x0 = 1.4 if name == "weibull" else 0.3
    h = 1e-6
    fd = (f(x0 + h, alpha, beta) - f(x0 - h, alpha, beta)) / (2 * h)
    assert d(x0, alpha, beta) == pytest.approx(float(fd), rel=1e-4)


def test_with_asymptotes_bounds_range():
    g = sg.with_asymptotes(sg.logistic, guess=0.25, lapse=0.05)
    # As x -> -inf, P -> guess; as x -> +inf, P -> 1 - lapse.
    assert g(-50.0, 0.0, 1.0) == pytest.approx(0.25, abs=1e-6)
    assert g(50.0, 0.0, 1.0) == pytest.approx(0.95, abs=1e-6)
