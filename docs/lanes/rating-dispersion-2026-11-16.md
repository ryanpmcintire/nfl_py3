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
