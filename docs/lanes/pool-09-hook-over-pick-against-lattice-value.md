# POOL-09

## Goal
Hook over-pick against lattice value: is the field's half-point lean proportional to the discrete-margin value of the half point (ROADMAP POOL-09). Context for the pool-standing panel only; one fitted cover probability selects every side.

## State
Unit 3, 2026-11-21 10:01 local (Week 10 rerun, both covariates), measured and recorded (6 cells replaced). Family `pool09_hook_value_slope`. Anchor: served multi-book Tuesday line.
- `scripts/pool09_hook_value_slope.py --last-week 10 --covariate push_mass --out artifacts/pool_rank_replay/pool09_hook_push_mass_slope_w10.json`: 57 hook games W1-10 (Week 10 adds 5), push mass 1.0-10.4 pp, mean 6.3.
  - Within-week (week-demeaned) slope: **+1.02 [-0.38, 2.44] P+ 0.931**, permutation p .148 (permuted range [-1.36, 1.36]).
  - Pooled slope, over-pick pp per pp of push mass: +0.62 [-0.58, 1.84] P+ 0.842; centred permutation p .152 (permuted mean -0.17).
  - Spearman +0.11 [-0.16, 0.35] P+ 0.781. Half split: W1-5 -0.10 (26 games), W6-10 +0.84 (31): signs still disagree.
- `--covariate served_effect --out ..._value_slope_w10.json`, 46 hook games W3-10: within-week +2.02 [-2.38, 6.26] P+ 0.824; pooled +1.84 [-1.71, 5.47] P+ 0.846, centred p .324; Spearman +0.13 [-0.16, 0.41] P+ 0.804; early +0.01, late +1.91.
- Implies: card unchanged. Pool-standing panel: the field's half-point lean (POOL-08, about 5.5 share points at Week 10) is still not shown to scale with what the half point is worth; one-for-one pricing would be a slope near 0.5 on push mass (inferred) and both 0.5 and 0 sit inside every interval. Any ownership proxy keeps one uniform hook-side coefficient. All 6 cells `unresolved_below_power`; nothing closes.

## Tried
Unit 1 (2026-11-03): served effect, 33 games W3-8, slope -1.27 [-5.07, 3.06] P+ 0.283. Unit 2 (2026-11-14, W1-9): push mass +0.40 [-0.88, 1.76] P+ 0.729, within-week +1.07 P+ 0.931. Unit 3 looks: 2 covariates x (pooled slope, within-week slope, Spearman, 2 half-split slopes) = 10; half splits unrecorded (no interval), counted here. The within-week slope was added in Unit 2 after the pooled null was seen (disclosed).

## Next
1. Week 11+: rerun both covariates with `--last-week N --out ..._wNN.json`; report the within-week slope first; replace the 6 cells.
2. Push mass is mostly a 3-versus-rest contrast; a key-number-matched contrast (hook games on 3 v hook games off 3, same pool line) would separate the number from the hook.

## Open
- W1-2 lack the discrete base for the served-effect covariate; push mass needs no backfill.
- Zero crossing closes nothing; no side rule; no entrant names stored.
