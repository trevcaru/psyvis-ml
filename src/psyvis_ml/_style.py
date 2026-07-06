"""Shared publication-quality matplotlib styling for every psyvis-ml figure.

One place defines the look — palette, typography, spines, grid, sizes — so ``plot()``,
``compare()``, and the confidence panels all render as one coherent, restrained, scientific
set (no per-figure hacks, no chartjunk). Styling is applied via :func:`rc_context` (a scoped
``matplotlib.rc_context``) so it never mutates a caller's global rcParams.

The categorical colours are the **Okabe–Ito colourblind-safe palette**; the two demo models
map to the first two entries *by order*, so a model keeps the same colour across all figures.

Matplotlib is imported lazily inside :func:`rc_context` so that importing the package (and the
whole fitting/measurement path) never eagerly loads matplotlib or configures a backend.
"""

from __future__ import annotations

__all__ = [
    "MODEL_PALETTE", "HUMAN_COLOR", "REFERENCE_COLOR", "BAND_ALPHA", "LINE_WIDTH",
    "POINT_SIZE", "THRESHOLD_MS", "FIG_ACCURACY", "FIG_STACK", "FIG_COMPARE", "FIG_DELTA",
    "model_color", "rc_context", "style_axes", "threshold_marker", "decimal_log_xticks",
]

# Okabe–Ito (colourblind-safe): blue, vermillion, bluish-green, reddish-purple, orange, sky.
MODEL_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
HUMAN_COLOR = "#2B2B2B"        # human reference line (neutral, distinct from model colours)
REFERENCE_COLOR = "#9AA0A6"    # criterion / baseline guide lines (muted gray)

# Consistent figure sizes (inches) per figure *type* — same across suites for one coherent set.
FIG_ACCURACY = (5.6, 4.0)      # single model, accuracy only
FIG_STACK = (5.6, 6.2)         # single model, accuracy + margin panels
FIG_COMPARE = (6.6, 7.0)       # multi-model, accuracy + Δ-margin panels
FIG_DELTA = (5.8, 3.9)         # standalone Δ-margin

BAND_ALPHA = 0.15              # CI band fill, soft and low
LINE_WIDTH = 1.9               # fitted curve
POINT_SIZE = 22                # data points, small
THRESHOLD_MS = 7               # open-circle threshold marker

_RC = {
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.titlepad": 9,
    "axes.labelsize": 11,
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#5A5A5A",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "text.color": "#222222",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#5A5A5A",
    "ytick.color": "#5A5A5A",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.4,
    "axes.grid": True,
    "grid.color": "#E4E4E4",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "lines.linewidth": LINE_WIDTH,
    "lines.solid_capstyle": "round",
}


def model_color(i: int) -> str:
    """Consistent colour for the ``i``-th model (by order), colourblind-safe."""
    return MODEL_PALETTE[i % len(MODEL_PALETTE)]


def rc_context():
    """Scoped matplotlib style context — wrap figure building in ``with rc_context():``."""
    import matplotlib as mpl
    return mpl.rc_context(_RC)


def style_axes(ax) -> None:
    """Final touches an axis: keep the grid behind data, drop the top/right spines."""
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def threshold_marker(ax, x, y, color):
    """An open-circle threshold marker on the curve (the value/CI go in the legend)."""
    ax.plot([x], [y], marker="o", ms=THRESHOLD_MS, mfc="white", mec=color, mew=1.5, zorder=6,
            linestyle="none")


def decimal_log_xticks(ax, lo, hi):
    """Readable **decimal** major ticks (1-2-5 per decade) on a log x-axis (0.01, 0.02, 0.05...)."""
    import numpy as np
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ticks, decade = [], 10.0 ** np.floor(np.log10(lo))
    while decade <= hi * 1.0001:
        for mult in (1, 2, 5):
            t = mult * decade
            if lo * 0.9999 <= t <= hi * 1.0001:
                ticks.append(t)
        decade *= 10
    if len(ticks) >= 2:
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
