# rating-dispersion-2026-11-16

## Goal
Prediction Tracker computer-rating predictions (NFL 1999-2026, college 2001-2026) as a pregame source; test cross-system disagreement D as one fitted tempering term (college first), with the systems-median minus opener gap as the priced control.

## State
- 2026-11-16T21:00:02Z part 1 done: scripts/rating_dispersion.py fetch|join; raw CSVs in data/external/prediction_tracker/{nfl,cfb}/<season>.csv (gitignored); joined artifacts/rating_dispersion/{nfl,cfb}_dispersion.parquet + join_report.json. NFL 4720 games joined (2009-2025 ~99.6% of repo, 2026 121 so far); college 11799 (2007-2025 ~97%; 2006 31; 2008 is a partial PT year). ncaa20000.csv (2000) is a dead link (404).
- Outcome columns (hscore, rscore, vscore, actual, total, difference, phcover, phwin) are dropped at read.
- Pregame evidence: site posts each week's predictions before games (reported); the live nflpredictions.csv / ncaapredictions.csv carry predictions for unplayed games (measured 2026-11-16). Archive values are the last pre-kickoff posting, so some systems may move after the Tuesday opener (inferred caveat).
- 2026-11-16T21:00:02Z part 2 declared in artifacts/rating_dispersion/declaration.json before any outcome join.

- 2026-11-16T21:01:52Z part 2 college screen done (`rating_dispersion.py screen`, exit 0; artifacts/rating_dispersion/summary.{json,txt}, screen.log). 8575 games, 13 seasons, 200 draws. Looks: 2 gating log-loss reads, 4 season-half coefficients, 6 cells.
  - dispersion_temper: coef -0.044/SD (declared negative), 13/13 folds, halves -0.053/-0.034; held-out ll +0.000078 [-0.000400,+0.000535] pp 0.650, null pct 92.0; acc -0.152 pts, 41 changed picks 14-27. Not a survivor; unresolved_below_power.
  - median_open_gap (control): coef +0.076/SD, 13/13 folds, halves +0.098/+0.061; ll +0.000309 [-0.000260,+0.000905] pp 0.845, pct 99.0; changed 936, record 468-468. Survivor by the declared rule. Inferred: archive values are each system's last pre-kickoff posting, so the gap likely carries post-opener line movement (line-following), not new pregame information.
  - Recorded 6 cells `rating_dispersion_cfb_<candidate>_<units>` (family rating_dispersion_cfb_u1, unresolved_below_power) in registry/weak_signals/rating_dispersion_cfb_u1.json: record exit 0 (6); pool --league nfl --effect-units accuracy_points exit 0.

## Tried
- Swapped-orientation join without week double-matched NFL division games; fixed with week keys (NFL) and unmatched-only swap.

## Next
- NFL confirmation of median_open_gap (declared, not run): same column on xlg14_family_screen.nfl_frame, 2020-2025, 200 within-season shuffles, same survivor rule. Before running, add the declared timing control: the same gap against the closing line (and an intercept-only read), since a gap that vanishes against the close is line-following.

## Open
- Whether any PT system columns are themselves market-derived or posted after the Tuesday opener; per-system posting times are not in the archive.
- dispersion_temper sat at pct 92 with the declared sign in every fold: a larger-sample read (more college seasons, 2001-2006 PT files exist but repo frame starts later) is the obvious next power step.

## NFL confirmation
- 2026-11-16T21:03:29Z declared before any NFL outcome join (scripts/rating_dispersion_nfl.py, artifacts/rating_dispersion_nfl/). Population: build_fit_population(keep_pushes=True) 2020-2025, offset = eng62 stage_one leave-one-season-out served logit (checked against per_game out_of_season_home_probability); opener pushes kept with y=0.5 (log loss and Brier score them, accuracy counts them wrong for both arms); inner join to PT nfl_dispersion by game_id. Term median_open_gap = PT system median minus tue_open_home_spread (home-margin units), declared sign positive. Control column close_move = close_home_spread minus tue_open_home_spread. Family rating_dispersion_nfl_u1, seed stream 42.
  - Read 1 (gating, the declared confirmation): one offset-logistic coefficient per SD on the served logit, LOSO, 200 within-season shuffles of the gap; survivor = held-out log-loss gain above null 95th percentile and same sign in both season halves (2020-22, 2023-25). Controls reported: intercept-only, close_move alone.
  - Read 2: gap plus close_move as two fitted terms vs close_move alone; null shuffles the gap within season with close_move fixed. Answers whether the systems' median adds beyond the move to the close.
  - Read 3: read 1 restricted to 2023-2025 (the served stage already has a market-move term there), own 200-draw null.
  - Looks: 3 reads x 3 metrics = 9 recorded cells; 3 gating log-loss reads; 2 season-half coefficients; 2 control reads. Correlations of gap with close_move and the served logit reported.
- 2026-11-16T21:06:01Z NFL confirmation run (`rating_dispersion_nfl.py`, exit 0, 49 s; artifacts/rating_dispersion_nfl/{summary.json,run.log,weak_signals_batch.json}). 1537 games 2020-25, 34 pushes, PT join 100%, stage gap 0.0 vs per_game; median systems 46; corr(PT median, opener) .920.
  - Read 1: coef -0.023/SD [-0.119,+0.080] P+ .32, folds 0/6 positive, halves -0.016/-0.026; ll -0.000224 [-0.001012,+0.000482] pp .262 null pct 60.5; changed 57 (2 pushes) 30-27; acc +0.20 pts. Not confirmed (opposite sign, unresolved; not resolved wrong sign).
  - Read 2 (beyond close move): coef -0.035, folds 0/6; ll -0.000183 pp .355 pct 70.5; changed 64, 38-26 vs move-only. Median adds nothing measurable beyond the close.
  - Read 3 (2023-25): coef -0.029 [-0.169,+0.106], folds 1/3; ll -0.000523 pp .282 pct 55.0; changed 28, 17-11.
  - Correlations: gap vs close-minus-open +0.095 (2023-25 +0.115); gap vs served logit +0.008. Controls (log loss): served .686913, +intercept .686956, +close move .685603 (close move coef +0.09..+0.12/SD every fold: line-following at the opener, not admissible).
  - Recorded 9 cells family rating_dispersion_nfl_u1 (unresolved_below_power): record exit 0 (9); pool --league nfl --effect-units accuracy_points exit 0.
  - Inferred: the college gap survivor does not carry to the NFL; NFL PT median sits tight on the opener (r .92) and its residual gap is nearly orthogonal to the served probability.
