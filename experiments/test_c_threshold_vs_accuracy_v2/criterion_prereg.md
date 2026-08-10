# Pre-registration: ceiling-relative threshold criterion (contrast suite)

**Date: 2026-07-16**
**Status: PRE-REGISTERED — not adopted. Gallery and README untouched.**
**Scope: contrast suite only.**

This document fixes a measurement decision *before* its consequences for any result are consulted.
It rests **solely** on whether the criterion is internally coherent. It deliberately cites **no
correlation, no relationship to accuracy, and no test outcome of any kind**. If the argument below
is not sufficient on its own, the decision should not be made.

---

## The decision

For the contrast suite, replace the fixed 75%-correct threshold criterion with the
**ceiling-relative midpoint**:

    ceiling   = 1 − lapse                       (lapse estimated, not fixed — see Prerequisite)
    target(f) = guess + f · (ceiling − guess)
    criterion = target(0.5)

and fit with `sigmoid="weibull", guess_rate=chance, lapse_rate=None`.

## Rationale (coherence only)

### 1. 75% is the 2AFC midpoint. The library imported the number, not the definition.

In two-alternative forced choice, chance is 0.5 and the ceiling is 1.0, so

    target(0.5) = 0.5 + 0.5 · (1.0 − 0.5) = 0.75

**75% *is* the midpoint criterion** — it is not an independent convention. What transfers between
paradigms is the *definition* (halfway up the observer's own range), not the *number*. This library
inherited the number and applied it to a 1000-way task where chance is 0.001 and observers cap well
below 1.0. The midpoint criterion reproduces 0.75 exactly wherever 0.75 is correct, and departs from
it only where 0.75 was never justified. It is a generalisation, not a replacement.

### 2. Under a 1000-way task the fixed 0.75 is ten different criteria wearing one label.

Measured on the ten contrast curves (Test C v2, variant (e)), the relative height at which the fixed
0.75 sits on each model's own range — `f = (0.75 − guess) / (ceiling − guess)`:

| model | fitted ceiling | relative height of the fixed 0.75 |
|---|---|---|
| resnet101 | 0.870 | f = 0.862 |
| vit_base_patch16_224 | 0.877 | f = 0.855 |
| convnext_tiny | 0.857 | f = 0.875 |
| resnet50 | 0.837 | f = 0.896 |
| swin_tiny_patch4_window7_224 | 0.818 | f = 0.916 |
| deit_small_patch16_224 | 0.843 | f = 0.889 |
| mobilenetv3_large_100 | 0.810 | f = 0.926 |
| efficientnet_b0 | 0.791 | f = 0.949 |
| vit_small_patch16_224 | 0.765 | f = 0.980 |
| resnet18 | 0.726 | **f = 1.033 — above the ceiling** |

A single number, 0.75, reads each observer at a **different point on its own psychometric function**
(f = 0.86 to 1.03). Two models' "thresholds at 75%" are therefore not the same measurement, and
comparing them is comparing quantities defined differently for each. The midpoint reads every
observer at f = 0.5 by construction, which is what makes the numbers commensurable.

### 3. The fixed criterion is undefined for observers it should describe.

resnet18's ceiling is 0.726. It never reaches 75% correct at any contrast, so `threshold(0.75)` does
not exist for it (f = 1.033 lies above its range). Under the fixed criterion this model silently
drops out of the analysis. This is not a fitting failure — it is the criterion asking a question the
observer cannot answer.

A criterion that is undefined for the weakest observer in the set, *because* it is the weakest, is a
criterion that decides which observers are eligible to be measured. Eligibility should not be a side
effect of an inherited constant. The midpoint is well-defined for every observer with a ceiling above
chance, regardless of how low that ceiling is.

### 4. The choice of f is not load-bearing, so f = 0.5 costs nothing to assume.

Model ranking by contrast threshold is **identical at f = 0.25, f = 0.50 and f = 0.75** — zero
discordant pairs across all three (Test C v2 §4). The curves do not cross within the observers'
ranges, so which model is judged more contrast-sensitive does not depend on the height at which the
curves are read.

This matters for pre-registration specifically: it means adopting f = 0.5 requires **no defence of
0.5 over 0.4 or 0.6**, and cannot be a researcher-degree-of-freedom used to select an outcome. The
criterion rescales the reported numbers without reordering the models. f = 0.5 is then chosen purely
because it is the value that reduces to the established 2AFC convention (§1).

## Prerequisite (decided on its own merits, not on results)

The midpoint requires a ceiling to be relative *to*, so it requires `lapse_rate=None` (variant (e))
rather than the current fixed `lapse_rate=0.02`. That change is justified independently of this
document, on two grounds, neither of which is a result about accuracy:

- **Correctness of the observer model.** `lapse_rate=0.02` asserts that every observer reaches 98%
  given sufficient signal. That claim is false for all ten observers measured (ceilings 0.73–0.88).
  It is a human-psychophysics default carried unexamined into a setting where it does not hold.
- **Goodness of fit to the observed points.** Freeing lapse raises the fit's R² against its own
  observed data from a median of 0.906 (worst 0.827) to 0.993 (worst 0.989), and removes a
  systematic sign-alternating residual pattern present in every model.

Identifiability of the free ceiling has been checked separately and holds on a 7-level
gallery-like grid (median CI widening 1.0×) for every model whose threshold the grid brackets.

## What this document does NOT rest on

This decision is made **without reference to**, and must not be defended by:

- any correlation between threshold and clean accuracy, under any fit or criterion;
- any change to such a correlation caused by adopting the midpoint;
- any change in which models are included in such an analysis;
- any verdict (distinct-axis / intermediate / kill) in Test C or its successors.

Those quantities are to be read **only after** this decision is fixed. If the midpoint criterion is
right, it is right for the reasons in §1–§4 and would be right had it moved the headline the other
way, or not at all.

## Commitments / falsifiers

The decision should be **withdrawn** if any of the following is shown:

1. **Ranking is not invariant to f** on a broader model set — §4's invariance is established on ten
   models on one dataset. If curves cross within observers' ranges elsewhere, f becomes load-bearing
   and the midpoint needs a positive argument for 0.5 specifically, which this document does not
   supply.
2. **The ceiling is not identifiable** on the grid actually in use, making the criterion a function
   of a poorly-estimated parameter.
3. **A principled alternative appears** that is defined for all observers, reduces to 0.75 for 2AFC,
   and is not ceiling-dependent.

## Scope and blockers

- **Contrast suite only.** The degradation suite is out of scope: its lower asymptote at high
  severity genuinely *is* chance, and its decreasing logistic already fits its observed points at
  R² ≥ 0.98. Nothing here argues for changing it.
- **Not adopted anywhere.** Gallery, README and `src/psyvis_ml/` are untouched.
- **Gallery adoption is blocked on a data decision, not this one.** The gallery's 0.01 contrast
  floor does not bracket the midpoint thresholds of the most sensitive models, so adopting this
  criterion there requires extending the grid downward (a re-sweep), independent of whether the
  criterion itself is accepted.

---

_Pre-registered 2026-07-16, before re-reading the correlation results in
`criterion_scoping.md` §5._
