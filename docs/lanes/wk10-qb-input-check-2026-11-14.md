# Week 10 quarterback input check (2026-11-14, read-only audit)

## Goal
Check every unplayed Week 10 game for a quarterback the served model assumes but the newest data on disk contradicts; quantify the stake; change nothing.

## State
Forecast `margin_predictions/2026-week-10-20261114T184227Z`; report `artifacts/wk10_qb_input_check/report.json`; scripts `tests/scratch/wk10_qb/s1..s5`.
**Inputs (measured, 5 of 90 model columns, all home minus away):** `diff_qb_expected_epa_per_dropback`, `diff_qb_starter_epa_per_dropback`, `diff_qb_starter_cpoe`, `diff_qb_start_probability`, `diff_qb_starter_experience_log`.
**Read at (read):** starter = rank 1 of the newest depth-chart observation (players.py:1965-1973; depth snapshot 20261114T183101Z, last observation 11-14 14:04Z); EPA/CPOE/experience = that player's state before gameday from pbp 20261111T010519Z (players.py:1977-1981); start probability = 1 minus his injury-row unavailability (1982-1988; injuries first seen 11-14 11:00Z, 21:30Z capture identical for QBs); expected EPA blends to constant -0.15 (players.py:126, 1989-1991). Rebuilt by every `lineups_*` pass (scripts/refresh_lineup_forecast.py:32-54); `refresh_*` refits on the existing table (capture_scheduler.py:2019-2021 job text, reported).

Assumed starter (start probability), away / home; every slot equals depth rank 1:
- BUF@NYJ J.Allen 1.0 / G.Smith 1.0; CAR@NO Young 1.0 / Shough 1.0; JAX@TEN Lawrence 1.0 / Ward 1.0
- KC@ATL Mahomes .94 / Penix 1.0; NE@DET Maye .94 / Goff .94; PIT@CIN Rodgers 1.0 / Burrow 1.0; SF@DAL Purdy 1.0 / Prescott 1.0
- HOU@CLE Stroud 1.0 / Watson 1.0 (last game Sanders)
- LAC@BAL Herbert 1.0 / Brissett 1.0 (state from 8 ARI games; last BAL game Fagnano; no BAL QB on report)
- MIN@GB Cousins 1.0 (state from 9 LV games; Murray Out; last MIN game Wentz) / Love 1.0
- **SEA@LV Darnold 0.0003, listed Out, expected Lock** / Mendoza 1.0, EPA and CPOE missing (8 career dropbacks)
- **LA@ARI** Stafford .94 / Beck, all five inputs missing (no prior dropbacks)
- **MIA@IND** McCord, all five missing / D.Jones 1.0

**Flagged (measured; base rerun reproduces served to 1e-12; card numbers from CURRENT_PREDICTIONS.md; card shift inferred through the served model-term coefficient 0.308):**
- SEA@LV: Lock's values (EPA +0.114, experience 7.00) move model-only SEA 50.2% -> 52.9% (52.8% at start probability .94); card SEA -3.5 62.4% -> 63.2%. No side change; the stale read understates the pick.
- MIN@GB if Wentz starts: model-only MIN 51.5% -> 51.6/51.8%; card 51.1% -> 51.1/51.2%. No side change.
- LA@ARI, MIA@IND sensitivity (start probability 1, experience log1p(0)): card LA -9.5 57.1% -> 56.0%; IND -2.5 61.0% -> 60.4%. No side change; experience diffs -9.17/+8.07 lie outside the fitted range (-5.18..+5.36), and these two diffs are never missing in 1,829 graded 2020+ games.

## Tried
s2 scores through `score_outcome_week` with the served offsets and reads on a copied feature frame; other games unchanged (0.0). Transactions capture holds URL slugs only; injury-news current file had no hits for these names.

## Next
1. players.py:1965-1999: when the rank-1 QB is ruled out, project the next depth-chart QB not ruled out and use his state instead of the -0.15 constant. Feature-definition change: refit and leave-one-season-out evaluation before serving. Sunday passes will not fix SEA unless the depth feed demotes Darnold (still rank 1 three hours after the Out listing).
2. players.py:1977-1999: a debut QB with no state leaves start probability and experience missing; assign them.
3. players.py:126: derivation of -0.15 not located; measure or remove.
players.py is outside the pinned-name rule as recorded in memory (reported, guard not checked).

## Open
- `decision_hours` used by weekly-run (qb-features default 24, cli_commands/features.py:608): if 24, Sunday observations are invisible to Sunday games (players.py:1968-1970).
- MIN and BAL starters not confirmable beyond the depth chart; Fagnano alternative not scored; NaN imputation path not inspected.
