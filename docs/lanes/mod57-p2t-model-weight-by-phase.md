# MOD-57 P2-T model weight by season phase

## Goal
Test whether the one served pick probability should weight the model (arm M4), the line move (arm V4) or both (arm B) differently by season phase, as fitted slopes chosen leave-one-season-out. Slopes inside one probability, not rules; nothing served changes. Zero crossing closes nothing. Script scripts/mod57_p2t_model_weight_by_phase.py, outputs artifacts/mod57_p2t/. Family mod57_quiet_game_slopes (prior count 18 read from shard; this unit adds 18).

## State
2026-11-11 10:58:40 DECLARED before any outcome of this unit was read (stamp from Get-Date).
Novelty (read): conditional-signal-atlas split composition_flag_sum, market_move_toward_home and market_move_available by week_in_season (registry shards market_move_*__week_in_season), accuracy-point reads, retrospective; it did NOT split model_logit. four-term-nonlinear-check has pairwise products only, no week term. MOD-47 is a short-week slope, not phase. No shard holds model slope by season phase. So M4 and B are new; V4 partially overlaps the atlas move-by-week split (that was a paired term-on/off read, not a fitted four-way slope with held-out log loss).
CAUTION: the phase boundaries came from descriptive reads on these same 1,503 games (P2-S, 40-48 W1, 259-177 wk2-6, 268-215 wk7-12, 262-234 wk13-18); leave-one-season-out protects the slopes, not the boundaries. The 2026 forward read is the only untouched evidence.
Declaration: phases W1, wk2-6, wk7-12, wk13-18. M4 model_logit -> four phase columns (NaN outside phase, z-then-zero as unit 4). V4 same for market_move_toward_home (availability and flag terms single, model single). B both. Fit LOSO 2020-25 with the served ridge (lead98 loso/fit), 1,503 games; baselines four-term (must read 829-674), model only, intercept-only. Control: 200 permutations of phase labels across week blocks within season per arm. Reads: decisive games first, accuracy/log loss/Brier vs four-term with week-block bootstrap and probability_positive overall and per phase; fold slopes; in-sample vs OOS gap; weeks 13-18 reliability. Forward: slopes frozen on 2020-25, 2026 Wk1-9 harness (root_m2_games.csv, pool-line stage as score.py), compared with current 85-51. Week 10: harness csv and u3 parquets stop at Week 9, so no Week 10 build; skipped, no hand-rolled inputs. Looks: 18 recorded (3 arms x 3 metrics x 2020-25 and forward); per-phase bootstraps are 36 extra descriptive looks, reported not recorded.

## Tried
2026-11-11 11:02:25 RUN DONE once, 18 cells recorded (shard registry/weak_signals/mod57_quiet_game_slopes.json, 18 to 36 signals). Output artifacts/mod57_p2t/ (results.json, batch.json, run.out). Four-term reproduced 829-674.
2020-25 vs four-term: M4 818-685 (acc -0.72 [-2.76,+1.33] P+ .23; ll +.0004 better P+ .59; changed 231, arm 110-121), V4 822-681 (acc -0.46 P+ .21; ll .0044 worse, P+ .004; 81 changed 37-44), B 819-684 (acc -0.67 P+ .28; ll worse .0044 P+ .05; 278 changed 134-144). Weeks 13-18: M4 258-238 vs 262-234 (acc -0.82 P+ .34, ll better .0018 P+ .70, 128 changed 62-66); V4 248-248 (acc -2.84 [-5.67,-0.20] P+ .01, ll worse P+ .003); B 256-240.
Model slope by phase (mean, 6 folds): W1 .40 (-.10..1.07), wk2-6 .78 (.66..85, 6/6), wk7-12 .43 (6/6), wk13-18 -.22 (-.35..-.11, 0/6); single .31. Move slope: W1 -.24, 2-6 .20, 7-12 .30, 13-18 .20.
Permuted-phase control (200): actual M4 acc -0.73 vs control mean -0.30 (share control >= actual .76); ll P(control>=actual) .11; arbitrary splits score as badly as the declared ones.
Forward 2026 wk1-9 (current 85-51 ll .6787): M4 76-60 (changed 17, arm 4-13, acc -6.65, P+ .02), V4 77-59 (12 changed, arm 2-10), B 77-59 (22 changed, arm 7-15); ll all worse or equal. Current by phase W1 12-4, wk2-6 50-27, wk7-9 23-20.
Week 10: harness csv and u3 parquets stop at Week 9; not built.

## Next
Not served. The model slope falls across the season as predicted but the split does not improve held-out accuracy; do not pursue unless boundaries are re-declared on untouched data (2026 wks 10-18).

## Open
Week 10 harness build absent.
