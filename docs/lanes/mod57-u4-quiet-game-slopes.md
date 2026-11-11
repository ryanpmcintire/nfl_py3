# MOD-57 U4 quiet-game slopes

## Goal
Test two declared slope splits inside the one served four-term probability (model_logit, flag_sum, market_move_toward_home, market_move_available): Arm A splits the move slope by move size, Arm M splits the model slope by state (small/zero move, large move, no move input). Zero crossing closes nothing; slopes inside one probability, not gates.

## State
Declared 2026-11-11 08:49 before any outcome read (stamp from date). Novelty (read): neither arm scored as stated. four-term-nonlinear-check N2 scored pooled model_logit*move_toward_home and *move_available products (unresolved), a partial overlap with Arm M; mod56 scored |move| buckets (2 looks) as precision-weight buckets, partial overlap with Arm A. Looks are counted in family mod57_quiet_game_slopes (12) and also share the N2/mod56 families.
Declaration: harness = lead98_price_skew loso/fit (served ridge), 1,503 games, week-block bootstrap (u3). Split point: print abs(market_move_toward_home) lattice first; small = available and abs<=0.5, large = available and abs>0.5. Arm columns are NaN outside their regime so the served z-score-then-zero treatment applies. Baselines: four-term, model only, 0.5, intercept-only. Populations: all 1,503; quiet (abs(open_move)<=0.5). Metrics accuracy points, log loss, Brier vs four-term; decisive games, fold coefficients, in-sample gap, quiet reliability. 12 looks.
Script: scripts/mod57_u4_quiet_game_slopes.py; output tests/scratch/mod57/u4/.

## Tried
Ran once 2026-11-11 08:50 (tests/scratch/mod57/u4/results.json, tests/scratch/mod57/u4.out). Lattice of abs move among 799 available: 0 30.4%, 0.5 37.3%, 1 14.0%, >1 18.3% (no 0.25/0.75); small 541, large 258. Convention: positive move_toward_home = home gets more favoured; corr with open_move .84. Four-term reproduced 829-674.
All games: Arm A 842-661 (+0.87 pts [-0.87,+2.60], P+ .83; ll P+ .24, Brier P+ .26; 147 changed picks 80-67); Arm M 834-669 (+0.33 [-0.80,+1.47], P+ .69; ll P+ .17; 89 changed 47-42).
Quiet 837: four-term 430-407, model only 441-396; Arm A 453-384 (+2.73 [+0.13,+5.29], P+ .98; ll improvement +0.0012 P+ .78; 107 changed 65-42); Arm M 444-393 (+1.68 [+0.12,+3.23], P+ .98; ll P+ .007 i.e. worse; 54 changed 34-20).
Folds: move_small mean -0.06 (-0.24..+0.04, 2 of 6 positive), move_large 0.23 (0.20..0.28, 6/6), served single 0.22. model_quiet 0.35 (0.08..0.54, 6/6), model_large -0.20 (-0.41..+0.09, 1/6), model_nomove 0.42, served 0.31. In-sample minus OOS accuracy gap all games: four-term +0.6, A +0.7, M -0.3.
12 cells recorded unresolved_below_power, family mod57_quiet_game_slopes.

## Next
Not served: accuracy gains on quiet games come with flat or worse log loss. Next: whether move_small ~0 and model_large <0 repeat on 2026 forward games (A: drop move slope on 0-0.5 moves).

## Open
None.
