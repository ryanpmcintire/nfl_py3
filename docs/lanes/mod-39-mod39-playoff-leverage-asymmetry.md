# MOD-39 Playoff leverage asymmetry for Weeks 10-16

## Goal
Pregame playoff-probability swing (P(playoffs|win) - P(playoffs|loss)), home minus away (levdiff), as one LOSO term at the opener, and a frozen forward log of it for 2026 Weeks 10-16. Family mod39_playoff_leverage (declared before running; 8 looks counted, 6 cells recorded). Research only; nothing here changes a pick.

## State
Unit 1 (2026-11-03): scripts/mod39_leverage.py -> artifacts/mod39/mod39_results.json. Swing from artifacts/sim09/stakes_teamweek.csv (standings Monte Carlo, 6000 sims, as-of margin ratings; 4 rating parameters fit on 2009-17). Raw-model base, 2020-25, n=592: slope +3.18 ats pts per 1.0 levdiff (-2.85..9.33, p+ 0.849); LOSO vs intercept-only accuracy +0.68 pp (p+ 0.74), log loss -0.00094 (p+ 0.18), decisive 32: term 18, control 14. Close-graded 2009-19 slope -1.59 (p+ 0.257); odd/even halves -3.82 / +3.49.

Unit 2 (2026-11-14), all measured this session:
- Unit repair: the two slope cells were per-1.0-levdiff slopes stored as ats_points (1.0 is about six SDs). Re-recorded through `weak-signals record --batch --replace` as ATS points per one SD of the term: opener 2020-25 +0.498 (-0.447..1.463, p+ 0.849, SD 0.1568); close 2009-19 -0.261 (-1.041..0.506, p+ 0.257, SD 0.1644). The accuracy and log-loss cells already carried the unit they measure. `weak-signals pool --league nfl --effect-units accuracy_points` exits 0.
- Served-base refit (scripts/mod39_freeze_levdiff_served.py, same fit code, offset = out_of_season_home_probability from artifacts/pick_probability/20261103T195411Z, 576 games, 16 opener pushes excluded): decisive 77: term 36, control 41; LOSO accuracy -0.87 pp (-3.82..+2.08, p+ 0.270); log loss -0.00060 (-0.0044..0.0031, p+ 0.374); in-sample +0.00083, gap 0.0014. Held-out-fold coefficient 0.349..0.930, positive in all six. Both recorded unresolved_below_power (mod39_levdiff_served_loso_*).
- Frozen 2026-11-14T20:07:58Z: artifacts/forward_terms/mod39_levdiff_served/frozen_coefficients.json. All-six-season fit: b = +0.5252 logit per 1.0 levdiff, intercept +0.0128 (intercept-only control +0.0098), base logit as offset. Rating parameters copied from artifacts/sim09/stakes_summary.json. Hash b5be8c08...
- Forward: term mod39_levdiff_served registered in scripts/forward_terms_week.py (TERMS, served base, skips weeks outside 10-16) and scripts/forward_ledger.py TERM_DIRS. 2026 swing = same simulation on the newest schedule snapshot, standings and ratings from weeks before the target week only; target-week and later results are blanked in code and it fails closed if an earlier result is missing. Seed is [season, week].
- Week 10 record written 2026-11-14T20:09:48Z, 13 games, Thursday WAS at NYG excluded (kicked off): 2026_week10_20261114T200948Z.json. levdiff: JAX@TEN -0.012, BUF@NYJ -0.007, HOU@CLE +0.051, NE@DET -0.092, KC@ATL +0.047, PIT@CIN -0.045, CAR@NO -0.042, MIN@GB +0.085, SF@DAL +0.134, LAC@BAL +0.081, LA@ARI -0.217, SEA@LV -0.156, MIA@IND +0.005. One game crosses 0.5 versus the served base (MIN@GB 0.489 -> 0.504).

## Tried
Raw-base and served-base reads disagree in sign on accuracy (+0.68 vs -0.87 pp); neither interval is wholly on one side, no positive control, so nothing closes. forward_weekly.py not run: Week 10 is unsettled and the ledger is unchanged.

## Next
Tue 2026-11-17: the existing forward job (`forward_terms_week.py --week auto`, default all terms) writes Week 11; `forward_weekly.py` (13:15 ET) scores Week 10 as candidate term_mod39_levdiff_served once Monday's game settles. After Week 16 record the forward read (decisive games first, then log loss, Brier, accuracy, probability_positive) with `weak-signals record`. Still owed: accuracy_points cells for the elimination and clinch differences, then the three-measure pool.

## Open
- The two Unit 1 LOSO cells score 16 opener pushes as home non-covers and use the raw model column; kept as recorded, superseded for serving questions by the served-base cells.
- The ledger compares the term against the served base, not the intercept-only control; the control intercept is in the frozen file for that comparison.
- If the newest schedule snapshot at job time lacks Monday's result the term fails closed for that week (inferred from the code path; last week's snapshot landed Tuesday evening).
- Swing is Monte Carlo with ties unmodeled and tiebreakers by random jitter. Opener coverage is 2020-25 only. New cells may need a base tag for the pool's by-base split.
