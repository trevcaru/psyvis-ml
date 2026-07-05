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


# --------------------------------------------------------------------------- #
# 2-D target patches for the crowding + degradation suites.
#
# A patch is a `size x size` array at a base luminance encoding the difficulty quantile q
# (base = MEAN_BASE + MEAN_SPAN*(q-0.5)), with the whole class-`cls` column raised by `amp`.
# Reading class as the argmax of column means, and q as the *median* pixel (the base, since a
# single raised column is a minority), is robust to partial occlusion or flanker overwrite --
# so a synthetic observer can recover class and q even from a degraded composite.
# --------------------------------------------------------------------------- #
MEAN_BASE = 0.5   # mirror psyvis_ml.datasets so patches decode with the same convention
MEAN_SPAN = 0.3
PATCH_AMP = 0.3


def make_patch(cls, q, size, amp=PATCH_AMP):
    base = MEAN_BASE + MEAN_SPAN * (q - 0.5)
    patch = np.full((size, size), base, dtype=float)
    patch[:, cls] += amp  # raise the whole class column (redundant, occlusion-robust)
    return patch


def _q_from_base(base):
    return (base - MEAN_BASE) / MEAN_SPAN + 0.5


def _col_argmax_and_base(arr, valid):
    """Class = argmax of per-column means over `valid` pixels; base = their median."""
    col_cnt = valid.sum(axis=0)
    col_sum = np.where(valid, arr, 0.0).sum(axis=0)
    col_mean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), -np.inf)
    cls = int(np.argmax(col_mean))
    base = float(np.median(arr[valid])) if valid.any() else 0.0
    return cls, base


def decode_patch(image, fill=0.0):
    """Recover (class, q) from a bare (possibly occluded) patch; occluded pixels == ``fill``."""
    image = np.asarray(image, dtype=float)
    valid = image != fill
    cls, base = _col_argmax_and_base(image, valid)
    return cls, _q_from_base(base)


def decode_crowding_target(canvas):
    """Recover (class, q) from the target patch on a crowding canvas.

    The target is the only strictly-positive content (background is 0, flankers are negative),
    so its bounding box isolates it; class/q read exactly when flankers do not overlap it.
    """
    canvas = np.asarray(canvas, dtype=float)
    valid = canvas > 0.0
    ys, xs = np.where(valid)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = canvas[r0:r1 + 1, c0:c1 + 1]
    cmask = valid[r0:r1 + 1, c0:c1 + 1]
    cls, base = _col_argmax_and_base(crop, cmask)
    return cls, _q_from_base(base)


def decode_flanker_spacing(canvas):
    """Recover the flanker spacing from a crowding canvas (inf if unflanked).

    Flankers are the negative-valued blobs; for two symmetric flankers at +-spacing about the
    target, the inter-flanker centroid distance is 2*spacing, so spacing = distance / 2.
    """
    from scipy.ndimage import center_of_mass, label

    canvas = np.asarray(canvas, dtype=float)
    lbl, n = label(canvas < 0.0)
    if n < 2:
        return np.inf
    coms = np.asarray(center_of_mass(np.ones_like(canvas), lbl, range(1, n + 1)))
    # Two blobs (n_flankers=2): half their separation is the target-to-flanker spacing.
    d = float(np.hypot(*(coms[0] - coms[1])))
    return d / 2.0


# --------------------------------------------------------------------------- #
# Crowding observer: P(correct) RISES with flanker spacing (Weibull in spacing).
# --------------------------------------------------------------------------- #
CROWD_ALPHA = 12.0   # critical-spacing scale (pixels)
CROWD_BETA = 3.0
CROWD_LAPSE = 0.02


def crowding_p_true(spacing, num_classes, alpha=CROWD_ALPHA, beta=CROWD_BETA,
                    lapse=CROWD_LAPSE):
    guess = 1.0 / num_classes
    if not np.isfinite(spacing):
        f = 1.0  # unflanked baseline: no crowding, target at ceiling
    else:
        f = 1.0 - np.exp(-((max(spacing, 0.0) / alpha) ** beta))
    return guess + (1.0 - guess - lapse) * f


def crowding_true_threshold(num_classes, target=0.75, alpha=CROWD_ALPHA, beta=CROWD_BETA,
                            lapse=CROWD_LAPSE):
    guess = 1.0 / num_classes
    f_target = (target - guess) / (1.0 - guess - lapse)
    return alpha * (-np.log(1.0 - f_target)) ** (1.0 / beta)


def make_crowding_observer(num_classes, alpha=CROWD_ALPHA, beta=CROWD_BETA,
                           lapse=CROWD_LAPSE):
    def crowding_observer(canvas):
        spacing = decode_flanker_spacing(canvas)
        cls, q = decode_crowding_target(canvas)
        correct = q < crowding_p_true(spacing, num_classes, alpha, beta, lapse)
        pred = cls if correct else (cls + 1) % num_classes
        logits = np.full(num_classes, -10.0)
        logits[pred] = 10.0
        return logits

    return crowding_observer


# --------------------------------------------------------------------------- #
# Degradation observer: P(correct) FALLS with severity (decreasing logistic in severity).
# Uses occlusion so realized severity is readable exactly as the occluded-pixel fraction.
# --------------------------------------------------------------------------- #
DEG_S50 = 0.35   # severity at the curve midpoint
DEG_K = 15.0     # logistic steepness
DEG_LAPSE = 0.02


def deg_p_true(severity, num_classes, s50=DEG_S50, k=DEG_K, lapse=DEG_LAPSE):
    guess = 1.0 / num_classes
    f = 1.0 / (1.0 + np.exp(k * (severity - s50)))  # 1 at low severity, 0 at high
    return guess + (1.0 - guess - lapse) * f


def deg_true_threshold(num_classes, target=0.75, s50=DEG_S50, k=DEG_K, lapse=DEG_LAPSE):
    guess = 1.0 / num_classes
    f_target = (target - guess) / (1.0 - guess - lapse)
    return s50 + np.log((1.0 - f_target) / f_target) / k


def make_degradation_observer(num_classes, fill=0.0, s50=DEG_S50, k=DEG_K, lapse=DEG_LAPSE):
    def degradation_observer(image):
        image = np.asarray(image, dtype=float)
        severity = float(np.mean(image == fill))  # occluded fraction
        cls, q = decode_patch(image, fill=fill)
        correct = q < deg_p_true(severity, num_classes, s50, k, lapse)
        pred = cls if correct else (cls + 1) % num_classes
        logits = np.full(num_classes, -10.0)
        logits[pred] = 10.0
        return logits

    return degradation_observer


def make_patch_data(n, size, num_classes, amp=PATCH_AMP):
    """A labeled set of 2-D patches: quantiles spread uniformly, classes cycled."""
    idx = np.arange(n)
    quantiles = (idx + 0.5) / n
    classes = idx % num_classes
    images = np.stack([make_patch(int(c), float(q), size, amp)
                       for c, q in zip(classes, quantiles, strict=True)])
    return images, classes.astype(int)
