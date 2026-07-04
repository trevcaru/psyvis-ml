"""Method-of-constant-stimuli sweep engine.

Applies a stimulus manipulation across a labeled image set at each of a fixed list of
levels, scores model correctness, and returns a reproducible run bundle whose per-level
``(levels, n_correct, n_trials)`` feed straight into ``psyvis_ml.fitting.fit_psychometric``.
"""

from .bundle import RunBundle, compute_config_hash, data_fingerprint, library_version
from .engine import run_sweep
from .scoring import count_correct, top_k_correct

__all__ = [
    "run_sweep",
    "RunBundle",
    "compute_config_hash",
    "data_fingerprint",
    "library_version",
    "top_k_correct",
    "count_correct",
]
