"""Shared synthetic model/stimulus builders for sweep-engine tests.

Two synthetic observers:

* A **stochastic Bernoulli** observer whose P(correct) at a level is *exactly* a known
  Weibull psychometric function — used to check that sweep -> fit recovers a known
  threshold. Randomness flows only through the engine-supplied ``rng`` so runs reproduce.
* A **deterministic cutoff** observer (correct iff level >= a per-image cutoff) — used to
  assert exact per-level counts and per-image/batched equivalence.

Both encode the "image" as small integer arrays and use a one-hot classifier as the model,
so no image-processing or ML framework is involved.
"""

import numpy as np

C = 10  # number of classes; chance = 1/C = 0.1
ALPHA_TRUE = 0.12
BETA_TRUE = 3.0
GUESS = 0.1
LAPSE = 0.02


def p_true(level):
    """The exact P(correct) the Bernoulli observer realizes at a given level."""
    f = 1.0 - np.exp(-((level / ALPHA_TRUE) ** BETA_TRUE))
    return GUESS + (1.0 - GUESS - LAPSE) * f


def true_threshold(target=0.75):
    f_target = (target - GUESS) / (1.0 - GUESS - LAPSE)
    return ALPHA_TRUE * (-np.log(1.0 - f_target)) ** (1.0 / BETA_TRUE)


# --------------------------------------------------------------------------- #
# One-hot classifier: turns a predicted-class code into logits.
# --------------------------------------------------------------------------- #
def onehot_model(stim):
    pred = int(np.asarray(stim).ravel()[0])
    logits = np.full(C, -10.0)
    logits[pred] = 10.0
    return logits


def onehot_model_batched(preds):
    preds = np.asarray(preds).ravel().astype(int)
    n = preds.shape[0]
    logits = np.full((n, C), -10.0)
    logits[np.arange(n), preds] = 10.0
    return logits


# --------------------------------------------------------------------------- #
# Stochastic Bernoulli observer (known Weibull threshold).
# --------------------------------------------------------------------------- #
def make_bernoulli_data(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, C, size=n)
    images = labels.copy()  # the image encodes its own true class
    return images, labels


def bernoulli_stimulus(image, level, rng):
    true = int(image)
    correct = rng.random() < p_true(level)
    pred = true if correct else (true + 1) % C
    return np.array([pred], dtype=int)


# --------------------------------------------------------------------------- #
# Deterministic cutoff observer (exact counts).
# --------------------------------------------------------------------------- #
def make_cutoff_data(cutoffs):
    """images: column 0 = true class, column 1 = per-image cutoff. labels = true class."""
    cutoffs = np.asarray(cutoffs, dtype=float)
    classes = np.arange(cutoffs.shape[0]) % C
    images = np.column_stack([classes.astype(float), cutoffs])
    return images, classes


def cutoff_stimulus(image, level, rng):  # rng unused: deterministic
    cls = int(image[0])
    cutoff = float(image[1])
    pred = cls if level >= cutoff else (cls + 1) % C
    return np.array([pred], dtype=int)


def cutoff_stimulus_batched(images, level, rng):  # rng unused: deterministic
    classes = images[:, 0].astype(int)
    cutoffs = images[:, 1]
    correct = level >= cutoffs
    return np.where(correct, classes, (classes + 1) % C)


# --------------------------------------------------------------------------- #
# Contrast observer with a known Weibull threshold, driven through the REAL contrast
# pipeline. Reads the difficulty quantile from the image mean, the realized contrast from
# std/mean, and the class from the deviation argmax (see psyvis_ml.datasets encoding). It
# responds correctly iff q < P_true(realized_contrast), so fraction-correct at each level
# equals P_true -- an analytically known psychometric function.
# --------------------------------------------------------------------------- #
CONTRAST_ALPHA = 0.15
CONTRAST_BETA = 3.0
CONTRAST_LAPSE = 0.02


def contrast_p_true(contrast, num_classes, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA,
                    lapse=CONTRAST_LAPSE):
    guess = 1.0 / num_classes
    f = 1.0 - np.exp(-((contrast / alpha) ** beta))
    return guess + (1.0 - guess - lapse) * f


def contrast_true_threshold(num_classes, target=0.75, alpha=CONTRAST_ALPHA,
                            beta=CONTRAST_BETA, lapse=CONTRAST_LAPSE):
    guess = 1.0 / num_classes
    f_target = (target - guess) / (1.0 - guess - lapse)
    return alpha * (-np.log(1.0 - f_target)) ** (1.0 / beta)


def make_contrast_observer(num_classes, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA,
                           lapse=CONTRAST_LAPSE):
    from psyvis_ml.datasets import decode_class, decode_quantile
    from psyvis_ml.stimuli.contrast import rms_contrast

    def contrast_observer(image):
        q = decode_quantile(image)
        contrast = rms_contrast(image)
        cls = decode_class(image)
        correct = q < contrast_p_true(contrast, num_classes, alpha, beta, lapse)
        pred = cls if correct else (cls + 1) % num_classes
        logits = np.full(num_classes, -10.0)
        logits[pred] = 10.0
        return logits

    return contrast_observer
