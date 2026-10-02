# sim-speed

## Goal
Make the MOD-25/SIM-09 season simulator much faster with bit-identical outputs, opt-in only
(`SIM_FAST=1` env or cfg key `fast=1`), in `scripts/sim_fast.py`. Default code paths unchanged.

## State
- `scripts/sim_fast.py`: init memo, exact FastGBM, scalar feature_matrix, FastReg (EP regressor, 400 trees, was 108,800 predictor calls per season = 27 of 49 s). `SIM_FAST_PARTS` default `memo,gbm,fm,ep`.
- Seeding fixed: `task_rng` in mod25d_variance.py keys pace and lattice RNGs by (cfg seed, salt, world, season, task seed); no pid. Slow-path RNG streams change vs old runs (reported). `sim09_u4e.py:191` still uses getpid (not touched).
- Measured, one process, crz, 1 world x 2 seasons (544 games): slow init 221 s, 0.4025 s/game; fast cold init 116 s then 1.9 s warm, 0.0926 s/game (4.3x). Game rows, season ids, plays arrays byte-equal slow vs fast.
- The only per-play HGB in sim is sim04_engine fourth_down (already routed, verified earlier); other policies are precomputed grids.

## Tried
- Profile of fast before ep: pick_index_nn_conditioned 8 s tottime of 49 s, 41k calls; numba not added.

## Next
1. Profile again; if pick_index_nn_conditioned dominates, vectorize miss path keeping RNG draws.
2. Check variants with fourth_down_clf (non-crz) equality.

## Open
- numba not installed. Cache invalidates on edits to the five hashed scripts.
