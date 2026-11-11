# MOD-57 P2-O how wrong is the line (hindsight and pregame ratings, own game removed)

## Goal
Measure distance between the line and the best hindsight estimate of each game with the game itself removed, and what that distance is worth in pick accuracy. Corrects P2-M, whose unit-rate arm included each game's own result. Families: mod57_how_wrong_is_the_line (H, P), mod57_availability_ceiling (U). Script scripts/mod57_p2o_how_wrong_is_the_line.py, outputs artifacts/mod57_p2o/ (leak files named LEAK_NOT_SERVABLE).

## State
2026-11-11 10:11:51 DECLARED before any outcome read.
- Population: REG games 2009-2025 with spread_line and score (served features table, home margin = home_score - away_score, home spread positive = home favoured); opener = tue_open_home_spread in per_game.parquet, 2020-25 (pushes already dropped there; the decisive-game filter is margin != line).
- H (LEAK, uses future games, positive control, not servable): per game g, unpenalised least squares of home margin on team indicators (+1 home, -1 away) plus one home-field term over every other REG game of the season, min-norm solution (ratings sum to zero). Removal verified: refit with g changes its prediction; perturbing g's margin by 10 leaves H unchanged (ten games).
- P (servable analogue): same regression on that season's games with gameday strictly before g; previous season's full-season fitted ratings and home field used when either team has fewer than four prior games that season (count reported). 2009 has no prior season, those games are dropped from P. Truncation check: deleting every row from g's date on leaves P unchanged, ten games.
- Statistics per rating x line (close 2009-25, open 2020-25), 4 cells x 3 looks = 12: (1) paired MAE difference line minus rating, (2) paired MSE difference (RMSE shown), (3) calibration slope of (margin - line) on (rating margin - line); all season-block bootstrap, 4000 draws. Descriptive (not looks): mean/median |rating - line| overall and by thirds (weeks 1-6, 7-12, 13-18), pick accuracy of the side the rating puts away from the line on decisive games overall, by third, by |gap| (<1, 1-3, >3) with season-block interval.
- U (LEAK, positive control, not servable): each team's season mean of the 13 unit-rate inputs over its other games of the season (own row removed), added to served walk-forward ridge as P2-M. Primary arm U_A = O1+O3 oracles + units (comparator 926-577); 3 populations (all/quiet/moved) = 3 looks. Descriptive extra arms: U_alone (units only, own row removed; comparator 916-587), U_prior_only (units from the team's rows strictly before g, not a leak), shuffled control of U_alone. Recorded too (6 extra cells, accuracy_points only, family mod57_availability_ceiling). Caveat (inferred): served unit values are as-of smoothed, so later rows may still carry g's result; own-row removal may not fully clean U, which U_prior_only checks.
- Looks: 12 + 3 primary = 15, plus 6 descriptive-arm cells.

## Tried
none yet

## Next
Write and run script, record batch, report.

## Open
2026-11-11 10:13:41 script written, ruff clean, run launched in background (artifacts/mod57_p2o/run.out)
2026-11-11 10:13:53 RUN IN FLIGHT (background bash id b2ghcswop, started 10:13; no output yet). When it finishes read artifacts/mod57_p2o/run.out and run.err; then record: uv run --project . --no-sync nfl-ats weak-signals record --batch artifacts/mod57_p2o/batch_lines_LEAK_NOT_SERVABLE.json and batch_units_LEAK_NOT_SERVABLE.json; confirm registry/weak_signals/mod57_how_wrong_is_the_line.json and mod57_availability_ceiling.json changed; fill Tried; report. If effect_units is rejected, adjust unit name in the script cells.
2026-11-11 10:17:42 RUN DONE and recorded (12 cells family mod57_how_wrong_is_the_line, 9 cells mod57_availability_ceiling; shards updated 10:17). Measured: H close MAE 10.58 vs line 10.14, slope .365 [.316,.411]; pick 2407-1910 55.8%. P close MAE 11.09, slope .078, pick 2143-2109 50.4%; P open 743-760. Opener H 843-660 56.1%. U_A 929-574 (+8.45), U_alone 916-587 (same as contaminated), U_prior_only 779-724 (-1.53), shuffled 795-708. Inferred: own-row removal does not clean U (later as-of rows carry the game). Fallback to prior season 1102 games; 66 2009 games no P. Truncation diff 0.0 on 12 games. Next: leave-game-out unit rates need raw per-game pbp stats, not as-of rows.
