# POOL-09

## Goal
Hook over-pick against lattice value: is the field's half-point lean proportional to the discrete-margin value of the half point (ROADMAP POOL-09). Context for the pool-standing panel only; one fitted cover probability selects every side.

## State
Unit 2, 2026-11-14 (declared push-mass covariate, Week 9 added): measured and written to artifacts; NOT yet recorded in the registry (agent hit the tool cap before `record`). Family `pool09_hook_value_slope`. Anchor is the served multi-book Tuesday line (src change 2026-11-10, see POOL-08 lane).
- Covariate as declared: push mass at the market whole number from the served discrete push table (`load_pick_probability_model(...).hook_push_mass` through `crossed_push_mass`), i.e. the margin distribution conditional on the line, not a smooth residual read. 52 hook games W1-9, none without a table entry; push mass 1.0-10.4 pp, mean 6.6 (half-point cover value mean 3.3 pp); 24 of 52 games sit on 3. Output `artifacts/pool_rank_replay/pool09_hook_push_mass_slope.json`.
  - Pooled slope, field over-pick (pp) per pp of push mass: +0.40 [-0.88, 1.76] P+ 0.729.
  - Permuted-covariate control (within-week, 20000): permuted slopes mean -0.50, 95% range [-1.67, 0.66]; centred two-sided p .133, one-sided .066; the script's uncentred two-sided p is .63 and misleads because the null is not centred on zero (the between-week part is fixed under within-week permutation).
  - Within-week (week-demeaned) slope: +1.07 [-0.41, 2.53] P+ 0.931, permutation p .133, permuted range [-1.39, 1.39].
  - Spearman +0.06 [-0.22, 0.32] P+ 0.660. Half split: W1-5 -0.11 (26 games), W6-9 +0.69 (26): signs disagree.
- Served-effect covariate rerun, 41 hook games W3-9 (`pool09_hook_value_slope.json`): slope +1.34 [-2.45, 5.66] P+ 0.743; control centred p .298, one-sided .149; within-week +2.22 [-2.27, 6.74] P+ 0.854, p .292; Spearman +0.07 [-0.23, 0.37] P+ 0.669; early +0.01 (15), late +1.50 (26). It was -1.27 on 2026-11-03 (W3-8, earlier anchor and served model): the sign moved.
- Implies: card unchanged. Pool-standing panel: the lean (POOL-08, about 6.5 share points) is not shown to scale with what the half point is worth; a field that moved its share one for one with cover value would show a slope near 0.5 on push mass (inferred), and the interval holds 0, 0.5 and larger. Any ownership proxy keeps one uniform hook-side coefficient for now. All cells `unresolved_below_power`; nothing closes.

## Tried
Unit 1 (2026-11-03): served effect, 33 games W3-8, slope -1.27 [-5.07, 3.06] P+ 0.283. Unit 2 looks: 2 covariates x (pooled slope, within-week slope, Spearman, 2 half-split slopes) = 10. The within-week slope was added after the pooled slope and its off-centre null were seen; that is disclosed, not a pre-declared look. Script now takes `--last-week`, `--covariate push_mass|served_effect`, `--out`; `ruff check` and `ruff format --check` pass on it.

## Next
1. Record with `nfl-ats weak-signals record --batch <file> --replace` (units slope_difference, correlation; category market; season 2026): `pool09_hook_push_mass_slope` 0.402 [-0.881, 1.757] P+ 0.72925 n52; `pool09_hook_push_mass_slope_within_week` 1.073 [-0.411, 2.526] P+ 0.93125 n52; `pool09_hook_push_mass_spearman` 0.057 [-0.220, 0.323] P+ 0.66025 n52; replace `pool09_hook_value_slope` 1.344 [-2.452, 5.665] P+ 0.743 n41; `pool09_hook_value_slope_within_week` 2.219 [-2.274, 6.744] P+ 0.854 n41; `pool09_hook_value_spearman` 0.070 [-0.229, 0.374] P+ 0.669 n41. Sample blocks 9 and 7.
2. Run `nfl-ats weak-signals pool --league nfl --effect-units accuracy_points` once and confirm it loads (shared with POOL-08 step 3).
3. Rerun both covariates at Week 10+ (`--last-week N`); report the within-week slope first.

## Open
- Push mass is nearly a 3-versus-rest contrast here (24 of 52 on 3), so the slope is mostly that contrast.
- W1-2 lack the discrete base for the served-effect covariate; the push-mass covariate needs no backfill.
- Zero crossing closes nothing; no side rule; weights and cut points chosen out of season; no entrant names stored.

- 2026-11-14 15:20: recorded 6 cells in family pool09_hook_value_slope (replace; units slope_difference, correlation). The two half-split slopes (4 looks) stay unrecorded: they are not separate effects with their own interval in the artifacts and are counted in the look total in the notes. pool loads. Next steps 1-2 done.
