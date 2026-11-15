# LEAD-108

## Goal
Wording-read availability inside the injury scenario mixture: the LEAD-106 per-player play probability replaces the ordinal ladder as the mixture weight (ROADMAP LEAD-108).

## State
Unit 1 done 2026-11-15 (all measured, `scripts/lead108_wording_mixture.py` -> `artifacts/lead108_wording_mixture/report.json`, `games.parquet`; ruff clean).
Declared before signs: family `lead108_wording_mixture`, 24 looks = 3 contrasts (read vs ladder, read vs served, ladder vs served) x 2 metrics x 3 subsets (weeks 2-8, week 9, pooled 2-9) + 3 controls x 2 metrics pooled.
Design: term per game = sum over verified listed players of LEAD-110 signed spread points per Out (OLS 2011-2025, `artifacts/lead110_absorption/report.json`) x (1 - P_play), home minus away. Ladder = designation x practice, expanding prior weeks (week 9 matches the forward scorer within .0009). Read arm swaps in the LEAD-106 read where one exists. One slope on a logit(served) offset, leave-one-week-out (2026 only; LOSO impossible). Served base = last published card probability before the deadline (`clv_ledger/published_picks.parquet`), because 2026 is absent from `pick_probability` per_game. Graded at the Tuesday opener, pushes dropped: 116 games.
Coverage (verified listed / with read / value mass read): weeks 2-8 218-329 listed per week, 15-61 read, 8-40% of value mass; week 9 309/309, 100%.
Pooled 2-9, log loss gain (positive = first arm better), week-stratified game bootstrap 2000:
- read vs ladder: decisive 2-2; +.0003 [-.0018, .0024] P+ .62 (perm p .15); Brier +.0002 P+ .60. Weeks 2-8 +.0005 P+ .66; week 9 -.0009 P+ .30 (0 decisive).
- read shift vs served: decisive 5-3; -.0021 [-.0062, .0019] P+ .14. Ladder shift vs served: decisive 3-1; -.0024 [-.0060, .0009] P+ .08.
- Mean log loss: served .6871, read .6892, ladder .6895, intercept .7039, line-move .7113. Served beats LOWO raw-model by .037 [-.0004, .073] P+ .97 (raw recalibrated).
- Slopes per fold: ladder .048 to -.023 (3 of 8 negative), read .057 to -.007 (2 of 8); in-sample gain read .0002 vs OOS -.0021.
Reading (inferred): the served probability already prices most listed-player availability; swapping the read in for the ladder moves the term (sd of difference .21 pts vs 1.5) too little to show at 116 games. Nothing changes a served pick.
Registry: 4 cells unresolved_below_power in `registry/weak_signals/` family lead108_wording_mixture (`lead108_read_vs_ladder_{ll,brier}_wk2_9`, `lead108_{read,ladder}_shift_vs_served_ll_wk2_9`). `weak-signals pool --league nfl --effect-units accuracy_points` exit 0, no coherence problems.

## Tried
Novelty check: nearest PER-10 (ordinal mixture, 2020-2024, unresolved), LEAD-106 (read), LEAD-110 (read absorption into line move), LEAD-109 (position weights). Unit 1 uses expected-lineup shift (linear mean of the mixture), not the full lattice re-read of PER-10.

## Next
1. Score weeks 10+ as forward weeks with full read coverage (forward reads on disk for week 10; needs snap counts for verification) by re-running the script with LAST_WEEK raised.
2. Full mixture: re-read the served discrete lattice at line minus each scenario shift (PER-10 Unit 5 method) instead of the linear term, same arms.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights chosen out of sample (here leave-one-week-out); every cell counted as a look; unresolved results go through weak-signals record; graded at the opener with intercept-only and line-move-only controls.
