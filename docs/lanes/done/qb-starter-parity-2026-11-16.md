# qb-starter-parity-2026-11-16

## Goal
Train/serve parity of the QB input. Before 2026 the training QB is the previous game's top passer (plus repair A/B); live, the starter is usually known by the deadline. Rebuild training QB columns for the game's actual starter (first QB dropback; form strictly before the game) and grade through the served stage. Family qb_starter_parity_2026_11_16. Owns scripts/qb_starter_parity.py, artifacts/qb_starter_parity/.

## State
2026-11-16T22:55Z declared before grading: base QW (836-667). Arm A = actual starter on every 2009-25 team-game; arm A1 (A') = actual starter only where it changed from the previous top passer AND the previous starter was listed Out/Doubtful at kickoff-24h or was off the team's ACT/INA weekly roster; 2026 rows unchanged in both (live_rows_changed 0; unchanged games max_abs 0).
2026-11-16T23:08Z done, recorded 2 cells (batch exit 0; pool ran once). Measured, LOSO 2020-25, 1,503 opener games, artifacts/qb_starter_parity/results.json:
- Starter changes 2009-25: 909 of 8,862 team-games (10.3%); knowable 393 (Out/Doubtful 173, off roster 239); week-1 changes 225. Graded 2020-25: 386 team-games differ from QW's starter, 157 knowable, 229 hindsight.
- A 838-665 vs QW 836-667: +0.13 pts, P+ 0.596 season / 0.603 week; changed 30 (16-14 v 14-16); log loss P+ 0.35, Brier P+ 0.36. A's test rows carry 229 hindsight starters, so it is not servable as graded.
- A1 833-670: -0.20 pts, season block [-0.33,-0.07] P+ 0.00 (each season 0 or -1 game), week block P+ 0.22; changed 21 (9-12 v 12-9); log loss P+ 0.84, Brier P+ 0.88. Recorded unresolved_below_power (scores disagree in sign).
- Model-only: QW 801-702, A 788-715, A1 800-703. model_logit folds stable (A1 0.23-0.39).

## Tried
Patched copy of the repair_ab players.py (tests/scratch/qb_starter_parity/parity/); v4_candidate grade() reused.

## Next
- Not a serving candidate. Optional: 2026 W1-10 forward replay of A1 (train-side change only), via v4_candidate replay().

## Open
- Off-roster knowability uses the nflverse weekly roster for the game week (inferred to be pre-game; INA treated as not knowable).
