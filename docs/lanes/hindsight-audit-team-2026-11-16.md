# Hindsight audit, team and schedule columns (2026-11-16)

## Goal
For each team-level and schedule-level column of the served weak_stack market_residual profile, state the as-of time of its training value against the pick deadline min(kickoff, Sun 4 PM ET); rebuild post-deadline columns as of the deadline for 2009-2025 and grade them through the served stage LOSO 2020-25 at the opener like scripts/v4_candidate.py, with QW (836-667) as reference. Player, availability and QB columns belong to a sibling unit; temp/wind is already handled by v4 W.

## State
Timing table (read):
- spread_line: table value is the nflverse close (features from schedules; target ats_margin also vs close, margin.py:431-434). Training rows feed the close; opener scoring rows replace it with the Tuesday opener (clv.py:2185-2208). Close is after the deadline for 4 PM, night and Monday games. POST-DEADLINE in training.
- total_line: nflverse close in training AND in the opener scoring rows (clv.py:2207 only swaps spread_line). POST-DEADLINE everywhere; the served live row instead holds the forecast-time total (serve-parity lane, mean |diff| 1.34).
- rest_diff, div_game, week_sin/cos: schedule fields fixed before Tuesday; rest agrees with gamedays on 4597/4613 (serve-parity lane). OK.
- neutral_site / elo location: final corrected schedule (one 2026 correction, PHI_JAX); no as-of schedule archive on disk. Minor, not rebuilt.
- elo_diff, elo_home_win_prob: updated only from completed earlier games in gameday order (features.py:468-497). OK.
- home/away team states, *_team_games, point_diff, ats_residual: EWMA through the last game strictly before gameday (features.py:430-433); ats_residual uses past games' closes, known before the deadline. OK.
- bias_prior_week_ats / week2_anchor / playoff_holdover: last completed game before gameday; prior-season postseason (features.py:563-585). OK.

Declared 2026-11-16 16:46 -05:00, before grading. Arms on the QW table (QB repair + forecast weather), stage 1 clv.opener_pick_evaluation, stage 2 the served four-term logistic LOSO 2020-25, per_game population of pick_probability/20261116T171555Z, paired vs QW season and week blocks:
- T: total_line = Tuesday-opener consensus total (tue_open captures, same store as the opener spread) in every 2009-2025 row; rows without a capture keep the close (count reported).
- P: training spread_line = Tuesday opener where captured (target unchanged, vs close).
- TP: both.
Amended 2026-11-16 16:54 -05:00 after run 1 (graded only the 1,537 Tuesday-captured 2020-25 rows; 2009-19 training rows kept the close): as-of source = Tuesday capture, else data/processed/sbr_odds.parquet open_total / open_home_spread (2009-2021, sign aligned to spread_line by its close), else close. Run 1 kept as T1/P1/TP1 in Tried; arms T/P/TP are re-graded with the SBR fill.
Cells: family hindsight_audit_team, log_loss_improvement and accuracy_points per arm, unresolved_below_power unless an admissible closing ground applies.

## Tried
DONE 2026-11-16 (measured, artifacts/hindsight_audit_team/results.json, run2.log exit 0). Identity: S 829-674, max gap 0.0. As-of fill: 4415 of 4431 REG 2009-25 rows (Tuesday 1602, SBR 3491 rows available, sign +1); total changed on 3948 rows, mean |diff| 1.26; spread on 3642, mean |diff| 1.20. Of 2009-25 REG games, 1709 of 4431 (38.6%) kick off at or after the Sunday 4 PM deadline.
Vs QW 836-667 (season block; week-block P+ in brackets):
- T 834-669: acc -0.13 [-0.34, +0.14] P+ .097 (.150); LL -0.00005 [-0.00014, +0.00002] P+ .068 (.072); changed 6 (arm 2-4).
- P 826-677: acc -0.67 [-1.14, -0.27] P+ 0.0 (.011); LL -0.00010 P+ .159; changed 26 (arm 8-18).
- TP 827-676: acc -0.60 [-0.95, -0.27] P+ 0.0 (.022); LL -0.00012 P+ .188; changed 27 (9-18).
Run 1 (Tuesday captures only, 2009-19 training kept close; results_run1_tuesday_only.json): T 834-669, P 832-671, TP 834-669; all P+ below .3.
Recorded 6 cells, family hindsight_audit_team: T both and P/TP log loss unresolved_below_power; P and TP accuracy refuted_mechanism, wrong_sign_resolved (season-block interval below zero). `weak-signals pool` exit 0.
Reading (inferred): the opener-spread training arm fails because the target stays the residual against the close; feeding the opener while learning the close residual misaligns feature and target. A consistent fix would also re-base the target to the opener (not tested). The as-of total moves nothing.

## Next
1. Optional: arm with target re-based to the opener (ats_margin from opener) plus opener spread feature, if the orchestrator wants the full opener-parity test.
2. Nothing to serve from this unit; QW stays the reference.

## Open
- neutral_site as-of schedule not reconstructable from disk.

## Root note 2026-11-16
- The two refuted_mechanism cells (P and TP accuracy) refute only those arms as built: opener spread as a feature while the target stays the residual against the close. They do not refute opener parity (feature and target both against the opener); that arm is untested here and is the next unit (check docs/lanes/mod57-p2c-uncorrected-error-model.md first).
