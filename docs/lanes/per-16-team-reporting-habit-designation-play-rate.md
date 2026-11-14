# PER-16

## Goal
Team reporting habit: play rate given the same designation, by team and coach, shrunk, as the availability prior (ROADMAP PER-16).

## State
Units 1-3 (2026-11-03; scripts/per16_team_designation_habit.py, per16_unit2_confounds.py, played_by_gsis.py; artifacts/per16/, artifacts/per03/), as measured then:
- Questionable habit is a reliable trait: team split-half .421 [.335,.508], coach .420; on complete-source rows (2016-25 participation) team .505 [.40,.60], coach with position and practice controlled .501, coach year to year on the same team .382. Doubtful has none (r -.06).
- Held-out who-plays gain over the league ladder (2019-25, 9,784 rows): coach log loss .0089 [.0053,.0127] P+ 1.0; team .0066 P+ .9995. Over position x practice cells: coach .0044 [-.0001,.0087] P+ .97 (Brier .0032 P+ 1.0); team .0007 P+ .61. Most of the raw gain is position mix.
- Snap-verified rows overstate play rates (unverified skill rows -19.3 pp); the offensive line is only visible in participation seasons.
- 13 cells in family per16_team_designation_habit, all unresolved_below_power.

Unit 4 done 2026-11-14 14:57 ET: scripts/per16_unit4_availability_addone.py; artifacts/per16/unit4_addone.json, unit4_addone.log, unit4_games.parquet, unit4_forward_games.parquet.
Declaration (written 14:49 ET, before any cover outcome was joined). Family `per16_habit_availability_addone`. Games: newest artifacts/pick_probability/*/per_game.parquet, 2020-25, opener grade. Rows: nflverse injuries, last row per player-week, Questionable, matched to the team's game; rows stamped at or after 00:00 UTC of game day are dropped; 2025 has no stamps, so it is kept, flagged, and a 2020-24 stamped-only grade sits beside the full one. Habit for season S is fit on complete-source Questionable rows of 2016..S-1: position x practice cells give p_ctrl, the head coach's shrunk offset gives p_coach; no within-season update, no role weight. Terms, home minus away: league_avail = -sum(1 - p_ctrl); habit_avail = -sum(1 - p_coach); habit_adj = sum(p_coach - p_ctrl) (primary). Arms: unpenalised logistic, leave-one-season-out, logit of out_of_season_home_probability as a fixed offset: served, intercept_only (control), the three terms (intercept plus one term per training-fold SD), habit_adj shuffled within season 500 times (control). No line is an input. Six comparisons x three metrics, paired game bootstrap 5000. Looks: 18 + 3 shuffle summaries + 6 stamped-only + 6 forward = 33. No side rule.

Results (measured; base 20261114T184614Z, 1,503 games, served 55.16%; habit_adj SD 0.30 players, nonzero in 1,343 games, 2.64 listed per team-game):
- habit_adj vs served: side changed on 43 games, new right 17, served right 26. Accuracy -0.60 pts [-1.46,+0.27] P+ .086; log loss -.00033 [-.00101,+.00036] P+ .188; Brier -.00016 [-.00049,+.00018] P+ .193. In-sample -0.27 pts, +.00004, +.00002; gap +0.33 pts, +.00037, +.00018.
- habit_adj vs intercept_only (primary): 13-24; -0.73 pts [-1.53,+0.07] P+ .034; log loss -.00028 [-.00096,+.00039] P+ .217; Brier -.00013 P+ .224.
- Fold coefficients per SD: -.027, -.020, -.008, -.047, -.012, +.001 (1 of 6 positive; all seasons -.019).
- Controls: intercept_only vs served 7-5, +0.13 pts P+ .711, log loss -.00005 P+ .292. Shuffled term vs served: accuracy mean -0.23, 5th-95th [-1.20,+0.67], real -0.60 (76% of shuffles at or above); log loss mean -.00045 [-.00151,+.00064], real -.00033 (39% at or above); coefficient [-.078,+.088], real -.019. The real term is indistinguishable from a shuffled one; the loss is the cost of fitting a coefficient.
- habit_avail vs league_avail (the swap inside the availability input): 13-12; +0.07 pts [-0.60,+0.73] P+ .586; log loss -.00001 P+ .479; Brier P+ .459.
- habit_avail vs served 49-57, -0.53 pts [-1.86,+0.80] P+ .224, log loss P+ .498; league_avail vs served 52-61, -0.60 pts [-2.00,+0.80] P+ .200, log loss P+ .507. Both coefficients positive in 6 of 6 folds (.025 to .063 per SD), in-sample log loss +.0003, held-out zero.
- Stamped-only 2020-24 (1,236 games): habit_adj vs served 25-20, +0.40 pts [-0.65,+1.46] P+ .769, log loss -.00035 P+ .138; vs intercept_only 10-15, -0.40 pts P+ .164; folds 3 of 5 positive. The 5-of-6 negative sign depends on 2025.
- 2026 forward (artifacts/forward_ledger, 88 graded games weeks 4-9, served 46.6%, coefficients frozen on 2020-25, no report stamps): habit_adj vs served 1-1, 0.00 pts [-3.41,+3.41] P+ .497, log loss -.00220 [-.00404,-.00049] P+ .006, Brier -.00109 P+ .006; frozen intercept control log loss -.00034 [-.00061,-.00004] P+ .014 (home covered 34 of 88); habit_avail vs served 4-1, +3.41 pts [-1.14,+7.95] P+ .913, log loss +.00058 P+ .616. 2026 correlation of habit_adj with home cover is +.22, opposite the frozen negative coefficient.
- 30 cells recorded (family per16_habit_availability_addone, all unresolved_below_power); `weak-signals pool --league nfl --effect-units accuracy_points` loads (exit 0).
- Card: no change. The habit predicts who plays but does not move the cover probability at this resolution; nothing here is proposed for serving and nothing is closed.

## Tried
Team and coach keys are near-collinear; not separable. No decay weighting. Unit 4 ran no line-move-only control: the served base already holds the market-move term and no line was declared as an input. Each listed player counts one, so a backup counts like a starter.

## Next
1. Role weight: each listed player weighted by his earlier-in-season snap share (needs a gsis-to-snap crosswalk that covers the offensive line; data/players/raw has no 2026 snaps).
2. Within-season habit update: a first-year head coach has offset zero (12 to 29% of listed rows carried none).
3. Report timestamps are missing for 2025 and 2026 in data/raw/nflverse_injuries; find a capture-time source so the cut is enforced at runtime.
4. Orchestrator adjudicates the 2026 forward log-loss cell (whole interval below zero on 88 games; it contradicts the fitted 2020-25 sign, not the availability mechanism; the frozen intercept control is below zero too).

## Open
Entry route if a weighted version holds: coach-shrunk Questionable play rate replaces the league ladder inside the player availability input, then the one calibrated probability; no side rule. Availability features are pinned source, so any served change needs a new enrolled cohort.
