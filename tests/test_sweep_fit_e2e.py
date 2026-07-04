"""End-to-end: sweep a synthetic observer with a known threshold, then fit and recover it.

This is the composition test — it proves the sweep engine's output feeds straight into the
import-isolated fitting core and recovers the planted psychometric threshold.
"""

import numpy as np

from psyvis_ml.fitting import fit_psychometric
from psyvis_ml.sweep import run_sweep

import synthetic as syn


def test_sweep_then_fit_recovers_known_threshold():
    images, labels = syn.make_bernoulli_data(n=800, seed=2024)
    levels = list(np.geomspace(0.03, 0.5, 11))

    bundle = run_sweep(syn.onehot_model, images, labels, levels, syn.bernoulli_stimulus,
                       seed=2024, model_name="onehot", stimulus_name="bernoulli")

    # Counts should climb with contrast (loosely monotone) and span the psychometric range.
    frac = np.asarray(bundle.n_correct) / np.asarray(bundle.n_trials)
    assert frac[0] < 0.4       # near chance at low contrast
    assert frac[-1] > 0.9      # near ceiling at high contrast

    # Feed the bundle straight into the fitting core.
    lv, nc, nt = bundle.to_fit_inputs()
    fit = fit_psychometric(lv, nc, nt, sigmoid="weibull",
                           guess_rate=syn.GUESS, lapse_rate=syn.LAPSE)

    assert fit.converged
    recovered = fit.threshold(0.75)
    known = syn.true_threshold(0.75)
    assert abs(recovered - known) / known < 0.15, (recovered, known)

    # The recovered curve passes ~0.75 at the recovered threshold, by construction.
    assert abs(float(fit.predict(recovered)) - 0.75) < 1e-6


def test_sweep_fit_alpha_recovered():
    images, labels = syn.make_bernoulli_data(n=1000, seed=77)
    levels = list(np.geomspace(0.03, 0.5, 12))
    bundle = run_sweep(syn.onehot_model, images, labels, levels, syn.bernoulli_stimulus,
                       seed=77)
    lv, nc, nt = bundle.to_fit_inputs()
    fit = fit_psychometric(lv, nc, nt, sigmoid="weibull",
                           guess_rate=syn.GUESS, lapse_rate=syn.LAPSE)
    assert abs(fit.params["alpha"] - syn.ALPHA_TRUE) / syn.ALPHA_TRUE < 0.2
