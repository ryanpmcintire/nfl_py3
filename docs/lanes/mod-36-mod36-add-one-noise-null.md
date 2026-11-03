# MOD-36 Add-one noise ensemble

## Goal
Noise floor for add-one candidates on the 88-input served base (ROADMAP MOD-36).

## State
Unit 1 done 2026-11-03 (measured). Family mod36_noise_floor, 4 looks recorded (K x accuracy/log loss), unresolved_below_power. 50 draws per K in {1,12}, N(0,1) columns seeded [36,K,i], ridge alpha 10, opener 2020-25 vs base 802-701 (scripts/mod36_noise_floor.py, mod36_analyze.py; artifacts/mod36_noise/draws.jsonl, summary.json). ~30-45 s per draw.
Null K=1: acc pts q05/50/95 -0.73/-0.07/+0.34; LL gain -6e-4/-2e-5/+6e-4; flips 10/36/73; P+ q50 .33, share P+>=.9 .02, >=.95 .02.
Null K=12: acc -1.03/-0.33/+0.71; LL gain -2.9e-3/-8.5e-4/+5.7e-4; flips 102/149/201; P+ share >=.9 0, >=.95 0; P+<=.1 .26.
Placement (two-sided acc p vs K=12): u13 A -0.47 .54, B -0.47 .54, u14 D -0.33 .60, P -0.86 .28 (K=1: .04).
Not a closing ground for anyone.

## Tried
Noise is not mean zero: median acc is negative for K=12, so excess over noise, not raw delta, is the read.

## Next
Place MOD-29, MKT-21, XLG-10 only via their own pipelines (different estimator/period/harness: not exchangeable with this null). More draws or K=2,4,16 matched arms if wanted.

## Open
Draws per arm are 50, so p-values resolve to 0.02.
