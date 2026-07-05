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


def make_margin_observer(num_classes, crossing, slope):
    """Graded-logit observer whose target-class logit margin is ``slope * (contrast - crossing)``.

    The target logit is set to that margin and all competitors to 0, so the extracted margin
    equals ``slope * (realized_contrast - crossing)`` exactly. Since ``apply_contrast`` sets the
    realized RMS contrast to the swept level, the mean margin crosses 0 at contrast==``crossing``
    — a deterministic ground truth for the confidence-threshold recovery test.
    """
    from psyvis_ml.datasets import decode_class
    from psyvis_ml.stimuli.contrast import rms_contrast

    def margin_observer(image):
        contrast = rms_contrast(image)
        cls = decode_class(image)
        logits = np.zeros(num_classes, dtype=float)   # competitors at 0
        logits[cls] = slope * (contrast - crossing)    # target logit == the margin
        return logits

    return margin_observer


# --------------------------------------------------------------------------- #
# 2-D target patches for the distractor + degradation suites.
#
# A patch is a `size x size` array at a base luminance encoding the difficulty quantile q
# (base = MEAN_BASE + MEAN_SPAN*(q-0.5)), with the whole class-`cls` column raised by `amp`.
# Reading class as the argmax of column means, and q as the *median* pixel (the base, since a
# single raised column is a minority), is robust to partial occlusion or distractor overwrite
# -- so a synthetic observer can recover class and q even from a degraded composite.
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


def decode_distractor_target(canvas):
    """Recover (class, q) from the target patch on a distractor canvas.

    The target is the only strictly-positive content (background is 0, distractors negative),
    so its bounding box isolates it; class/q read exactly when distractors do not overlap it.
    """
    canvas = np.asarray(canvas, dtype=float)
    valid = canvas > 0.0
    ys, xs = np.where(valid)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = canvas[r0:r1 + 1, c0:c1 + 1]
    cmask = valid[r0:r1 + 1, c0:c1 + 1]
    cls, base = _col_argmax_and_base(crop, cmask)
    return cls, _q_from_base(base)


def decode_distractor_size(canvas):
    """Recover the (single) distractor's size from its reserved-value blob (0 if undistracted).

    Distractors are the negative-valued pixels; the bounding-box side of that region is the
    distractor size. Assumes a single distractor (``n_distractors=1``) so the box is one square.
    """
    canvas = np.asarray(canvas, dtype=float)
    ys, xs = np.where(canvas < 0.0)
    if ys.size == 0:
        return 0.0
    return float(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))


# --------------------------------------------------------------------------- #
# Distractor observer: P(correct) FALLS with distractor SIZE (decreasing logistic in size).
# --------------------------------------------------------------------------- #
DIST_S50 = 8.0   # distractor size (pixels) at the curve midpoint
DIST_K = 0.8
DIST_LAPSE = 0.02


def distractor_size_p_true(size, num_classes, s50=DIST_S50, k=DIST_K, lapse=DIST_LAPSE):
    guess = 1.0 / num_classes
    if size <= 0:
        f = 1.0  # undistracted baseline: target at ceiling
    else:
        f = 1.0 / (1.0 + np.exp(k * (size - s50)))  # 1 at small size, 0 at large -> decreasing
    return guess + (1.0 - guess - lapse) * f


def distractor_true_threshold(num_classes, target=0.75, s50=DIST_S50, k=DIST_K, lapse=DIST_LAPSE):
    guess = 1.0 / num_classes
    f_target = (target - guess) / (1.0 - guess - lapse)
    return s50 + np.log((1.0 - f_target) / f_target) / k


def make_distractor_observer(num_classes, s50=DIST_S50, k=DIST_K, lapse=DIST_LAPSE):
    def distractor_observer(canvas):
        size = decode_distractor_size(canvas)
        cls, q = decode_distractor_target(canvas)
        correct = q < distractor_size_p_true(size, num_classes, s50, k, lapse)
        pred = cls if correct else (cls + 1) % num_classes
        logits = np.full(num_classes, -10.0)
        logits[pred] = 10.0
        return logits

    return distractor_observer


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
