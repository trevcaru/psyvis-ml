# Changelog

## Unreleased

### Changed — stimulus RNG is keyed to a level's VALUE, not its position (deliberate break)

**What changed.** `run_sweep` derived each element's RNG from
`SeedSequence([seed, level_index, image_index])`. It now derives it from
`SeedSequence([seed, level_key, image_index])`, where `level_key` is a SHA-256 digest of
`"<suite>|<condition>|<stimulus>|<level>"` with the level canonicalized as `%.17g`
(`src/psyvis_ml/sweep/engine.py`, `_level_key`). The hashed run config gains an
`rng_keying` field recording the scheme (`"level_value_sha256_v2"`).

**Why.** Because `level_index` is positional, adding, removing, or reordering entries in
`levels` re-randomized the noise at levels that had already been measured. Two sweeps sharing
a level were therefore not showing the same stimuli there, so their overlapping rows could not
be pooled, joined, or compared — and extending a level grid silently invalidated the part of
it you already had. Under the new keying, the stimulus at a given
(suite, condition, level value, image) is fixed by the seed alone, whatever else was swept
alongside it. Audit check C2 (`outputs/audit/sweep_randomization.md`) demonstrated the old
behavior: reversing a 4-level list changed 100% of `margin` values and 20–25% of `correct`
bits on `gaussian_noise` and `occlusion`.

**This is bookkeeping and reproducibility, NOT a correctness fix.** No prior result is biased,
wrong, or invalidated by having used the old keying. Any RNG key yields an equally valid draw
from the same noise distribution: the old scheme's realizations were legitimate samples, the
per-level counts they produced were unbiased estimates, and every published curve, threshold,
and CI computed from them stands. What the old scheme lacked was *addressability* — a way to
name a stimulus independently of the list it happened to be swept in. Nothing about the
statistics changes; what changes is which sweeps can be joined to which.

**Compatibility — pre-change runs are NOT bit-reproducible under the new keying.** Re-running
a sweep recorded before this change, with the same seed and the same config, reproduces the
old per-item table only for the deterministic suites. Specifically:

| Suite | Reproduces pre-change output? |
|---|---|
| `ContrastThreshold` | **Yes** — byte-identical; `apply_contrast` never consumed the rng |
| `DistractorRobustness` | **Yes** — byte-identical; `composite_distractor_size` ignores the rng |
| `DegradationSuite("blur")`, `("lowpass")`, `("highpass")` | **Yes** — deterministic manipulations |
| `DegradationSuite("gaussian_noise")` | **No** — new noise fields |
| `DegradationSuite("occlusion")` | **No** — new occluder positions |

Verified by re-running audit check C1 across the change: the `contrast` and `distractor`
per-item files hash to the same sha256 before and after (`a27e1eeb45bf9cb6…`,
`e02763edd32b220b…`), while `gaussian_noise` moved `69550b1df5f6d0d7… → 62356ff6502f8bf7…`
and `occlusion` moved `c625edbe030b5ce8… → b40edd03fecc1f2e…`.

Existing artifacts under `outputs/` were **not** regenerated and are still valid results; they
are simply not byte-reproducible on the stochastic suites by a current checkout. Because
`rng_keying` is part of the hashed config, an old bundle and a new one can never present the
same `config_hash` while carrying different stimuli — the two are distinguishable rather than
silently conflated. The cost of that guarantee is that config hashes changed for *every*
suite, including the deterministic ones whose data did not change; a `config_hash` from before
this change will not match one produced after it even when the underlying rows are identical.
Join old and new artifacts on `(model, suite, condition, item_id, stimulus_level)` rather than
on `config_hash` if you need to compare across the boundary.

**Tests.** `tests/test_sweep_level_keying.py` asserts the invariance directly: for
`gaussian_noise`, `occlusion`, `ContrastThreshold`, and `DistractorRobustness`, sweeps over
`[0.05, 0.1, 0.2, 0.4]`, its reverse, and the subset `[0.1, 0.2]` at one seed must produce
bit-identical `margin` and `correct` on every shared (condition, item, level) cell. It also
pins that `_level_key` is process-stable (SHA-256, not `hash()`), that `0.1`/`0.10`/`1e-1`
canonicalize together, and that different suites at the same numeric level do not collide.
