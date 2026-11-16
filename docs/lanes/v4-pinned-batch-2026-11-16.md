# v4 pinned batch candidate (2026-11-16)

## Goal
Build and grade the post-season base-model candidate (served weak_stack, market_residual, ridge 10) with two correctness fixes: QB input repair (tests/scratch/qb_input_repair/repair_ab.patch) and pregame-forecast temp/wind in training and serving (or drop temp/wind). Grade through the served pick-probability stage, LOSO 2020-25 at the opener; frozen replay 2026 W1-10; record cells. Serving decision is the orchestrator's after Week 18.

## State
- Script scripts/v4_candidate.py; outputs artifacts/v4_candidate/{results.json, replay_2026.csv, opener_<arm>.parquet}.
- Arms: S served, Q QB repair, W forecast weather, D weather drop, QW, QD.
- Stage 1 = clv.opener_pick_evaluation (served walk-forward) on each arm's table; stage 2 = the served four-term logistic (mod27 Fit, FIT_RIDGE) refit LOSO on pick_probability/20261116T171555Z per_game with model_logit replaced.
- Historical forecast source: data/raw/forecast_archive/pool_decision_2009_2025_aware (GFS MOS walked back from min(kickoff, Sun 16:00 ET), every game 2009-25); open-air roofs only, else NaN. 2026: mos_bulletins with cached_at before the same cutoff (wk6n.py rule).
- 2026 replay: score_outcome_week per arm on served input rows; card shift = card logit + arm model_logit coefficient x model-logit delta (delta method; gaussian read, no discrete offset).

## Tried
DONE 2026-11-16 (measured, results.json; `--cached` reuses opener_<arm>.parquet). Identity: S reproduces stored model prob and served out-of-season prob exactly (max gap 0.0), 829-674 on 1,503 (34 pushes excluded from the binary metrics; 1-3 push model-side changes per arm, no effect on the record).
Arm vs S (season-block; LL and accuracy gain, point [95%] P+):
- Q 832-671: acc +0.20 [0.00, +0.45] P+ .91; LL -0.00002 [-0.00043, +0.00026] P+ .49; changed 17 (10-7).
- W 836-667: acc +0.47 [+0.13, +0.85] P+ .997 (week-block P+ .95); LL +0.00003 P+ .74; changed 15 (11-4).
- D 830-673: acc +0.07 P+ .61; LL +0.00036 [+0.00001, +0.00070] P+ .98; Brier P+ .98; changed 41 (21-20).
- QW 836-667: acc +0.47 [+0.13, +0.82] P+ .997; LL -0.00002 P+ .47; changed 23 (15-8).
- QD 828-675: acc -0.07 P+ .32; LL +0.00034 P+ .99; changed 41 (20-21).
Stage-2 model_logit coef per fold: S .22-.38 (full .308); W .22-.38; D .27-.42 (full higher, dropped weather sharpens the model term). In-sample vs LOSO log loss gap about 0.001 for every arm.
2026 W1-10 replay (card delta method): Q changes BAL@ATL, LA@ARI, SEA@LV (graded 1-0); W changes DEN@KC (0-1); D DEN@KC, TEN@NYG (0-2); QW 4 changes (1-1); QD 5 (1-2).
Recorded 10 cells, family v4_pinned_batch, unresolved_below_power; `weak-signals pool` exit 0.
Bug fixed in-session: mod27 `paired(frame, pa, pb)` returns gain of pb over pa's sign convention mixed; v4 calls paired(frame, served, arm).

## Next
1. After Week 18 the orchestrator picks the candidate (QW: correctness fixes, best accuracy; or QD/D: best log loss). Apply repair_ab.patch + the forecast feed or weather_drop.patch, rebuild, refit, identity check, re-run this script.

## Open
- QB arm reuses the 11-14 scratch copies of players.py (orig, repair_ab); rerun mod22 `patches` if players.py moved.
- Archive forecast_valid_utc is not always nearest kickoff (2009_01_TEN_PIT valid 06:00 for a 00:30 kickoff); history and 2026 matching rules differ slightly.
