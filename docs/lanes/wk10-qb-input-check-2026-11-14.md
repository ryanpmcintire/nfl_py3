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

## Repairs (2026-11-14, unit mod22_qb_input_repair; nothing under src edited)
Script `scripts/mod22_qb_input_repair.py` (`patches`, then `run`); outputs `artifacts/mod22_qb_input_repair/{summary.txt,results.json,cells.json,patches.json}`.
- **Constant origin (measured, git log -S):** `_REPLACEMENT_QB_EPA = -0.15` entered with the first commit of players.py (2c26e544, 2026-08-12, empty body); only other mention is docs/qb_dependence.md:96-101, which flags it as underived. No derivation on record. Mechanism (read, players.py:1989-1991): EPA per dropback the team gets when the projected starter does not play, blended in by 1 minus his start probability.
- **Patches (measured: `git apply --check` exit 0, ruff check and format clean on the patched copies):** `tests/scratch/qb_input_repair/repair_a.patch` (starter = highest quarterback of the newest visible depth chart not listed Out as of the decision time; without a depth chart the same rule runs down the previous game's dropback order), `repair_b.patch` (a starter with no usable state gets the dropback-weighted first-start EPA and CPOE of earlier seasons' starters who also had no state, his own injury-row start probability, and zero career dropbacks), `repair_ab.patch` (both; a and b do not stack, apply this one). Patched copies under `tests/scratch/qb_input_repair/<name>/src/nfl_ats/players.py`.
- **Fence (read, independent_validation.py:40-72):** players.py is not in `PINNED_NAMES` and matches none of the name rules, so the source hash is untouched; the patch still changes five model inputs, so it needs a refit and belongs with the next pinned change.
- **Constant measured (measured, `constant` step, EPA per dropback of the quarterbacks who played when the projected starter had none; week-block bootstrap, held-out-season range):** starter listed Out, the case the blend weights: -0.123 [-0.175, -0.075], 132 team-games 2009-25, held-out -0.137..-0.113; 2020-25 -0.104 [-0.183, -0.030], 62. -0.15 is inside both intervals, 0.03 to 0.05 below the estimates. Any absence: -0.046 [-0.069, -0.023] (901), because unlisted absences are mostly a regular starter returning (opening week +0.024, later weeks -0.047). Units agree: starters' state +0.044 against realised +0.034. Debut starters' first start: -0.142 [-0.193, -0.095] (248); the patch's fill for 2026 is EPA -0.142, CPOE -3.38.
- **How often (measured; team-games; as of 24 h before kickoff):** season: listed Out / Doubtful; projected quarterback did not take the first dropback (opening week + later) of played; all five missing; EPA and CPOE missing.
  2020: 9/3; 61 (13+48) of 512; 0; 1. 2021: 8/3; 74 (18+56) of 543; 0; 3. 2022: 15/2; 71 (16+55) of 542; 0; 2. 2023: 9/3; 65 (15+50) of 544; 0; 1. 2024: 4/5; 66 (20+46) of 544; 0; 0. 2025: 17/1; 73 (21+52) of 544; 0; 1. 2026 weeks 1-10: 16/0; 20 (1+19) of 271; 2; 1.
  No Out-listed projected quarterback took a first dropback in any season. Timestamps: injury rows carry real modification times 2020-24, a week proxy only in 2025, first-seen capture times in 2026; depth charts exist only from 2026-03-22, so before 2026 the projected starter is the previous game's top passer.
- **Evidence (measured; identity first: scratch copy of the served builder reproduces the served table exactly and the base scores 802-701 on 1,503, max gap 3.8e-15; arm minus base, week-block bootstrap):**
  a: 26 affected, 14-12 against 12-14, changed picks 2 (repair 2-0), log loss -0.024 [-0.062, +0.005] P+ .06; all games 805-698, +0.20 pts [-0.46, +0.85] P+ .70, changed picks 21 (12-9), log loss -0.0002 [-0.0010, +0.0004] P+ .23, margin error to the close 1.891 -> 1.884, +0.007 [-0.003, +0.017] P+ .92.
  b: 8 affected, 1-7 against 2-6, changed picks 1 (base), margin error to the close 3.10 -> 2.07, +1.03 [+0.10, +1.73] P+ .99; all games 804-699, +0.13 [-0.27, +0.54] P+ .68, log loss +0.0001 P+ .65.
  ab: 33 affected, 15-18 against 14-19, changed picks 3 (2-1), log loss -0.0006 [-0.029, +0.030] P+ .47, margin error to the close 2.75 -> 2.25, +0.49 [+0.14, +0.89] P+ .998; all games 804-699, +0.13 [-0.55, +0.80] P+ .62, changed picks 28 (15-13), log loss +0.0002 [-0.0005, +0.0009] P+ .68, margin error to the result +0.001 [-0.010, +0.012] P+ .59.
  Reading (inferred, against mod-36 "Use in future packets"): 33 of 1,503 games cannot move accuracy by the 1.4 to 1.6 points that harness resolves, so these numbers neither earn nor refuse the repair; the case is that the input describes the wrong player, and the evidence shows no harm. 30 looks (3 arms x 2 populations x 5 metrics); 18 recorded `unresolved_below_power`, family `mod22_qb_input_repair`; `weak-signals pool` exit 0.
- **Week 10 under ab (measured):** SEA projected Darnold -> Lock, start probability 0.000 -> 0.938, expected EPA -0.150 -> +0.097; Beck, McCord and Mendoza get the debut fill, start probability 1.0, experience from their own dropbacks.
- **Apply at the next pinned change:** 1 `git apply tests/scratch/qb_input_repair/repair_ab.patch` (rerun `patches` first if players.py moved); 2 rebuild the feature tables and refit; 3 rerun the identity and this script; 4 separately replace -0.15 with the measured listed-Out value.
- **Open:** 2026 is not graded by this harness, so the depth-chart branch (16 changed 2026 games for a, 39 for ab) has no graded record; zero experience for a true debut lies outside the fitted range; `_REPLACEMENT_QB_EPA` is unchanged by the patches.

## Open
- `decision_hours` used by weekly-run (qb-features default 24, cli_commands/features.py:608): if 24, Sunday observations are invisible to Sunday games (players.py:1968-1970).
- MIN and BAL starters not confirmable beyond the depth chart; Fagnano alternative not scored; NaN imputation path not inspected.

## Root decision (2026-11-14 17:21 ET)
- Not applied tonight: a change to the feature builder hours before the Sunday lineup and refresh passes, verified only in a research harness, risks the passes failing at the lock for a gain of under a point on one game.
- Apply on Tuesday 2026-11-17 before the weekly lock, when the pipeline retrains: tests/scratch/qb_input_repair/repair_ab.patch (starter = highest depth-chart quarterback not listed Out; measured fill for a debut starter), and replace the underived replacement constant -0.15 with its measured value -0.123 [-0.175, -0.075] (132 team-games 2009-25), citing artifacts/mod22_qb_input_repair. players.py is not in the validation cohort's fence (read by the unit; confirm with one read of independent_validation.py before applying). Then run the durable tests, the weekly forecast, and scripts/check_served_terms.py.
- Evidence the repair does no harm (measured): affected games 15-18 against 14-19; all 1,503 games 804-699 against 802-701, changed picks 15-13. Accuracy cannot judge a 33-game repair; the case for it is that the input describes a player who is not playing.
