# POOL-07

## Goal
Standings-conditional variance policy: how far ahead or behind we sit changes which picks to take (ROADMAP POOL-07). Challenger in the pool simulator only; one fitted cover probability selects every served side.

## State
Unit 3, 2026-11-21 10:34 local: field-model arms, the expected-wins cost and the decisive record, measured and recorded. `scripts/pool07_standings_policy.py --mode lowo --arm <arm> --paths 10000 --chunk 2000 --weeks 8 --points 77 --rank 90`, one file per arm `artifacts/pool07/pool07_lowo_<arm>_w10.json`; `--mode lowo_summary` -> `pool07_lowo_field_arms_w10.json`. Rule and start state as Unit 2 (rank 90, 77 pts, 8 weeks left). Proxies are count-weighted logits on the home line (intercept + line; curve adds line*|line|), refit on each fold's training weeks. The realised arm uses the observed Splash pick counts for Weeks 1-10 (Weeks 3-10 match the raw pick-distribution file, checked on Week 10), and it reproduces Unit 2 exactly.
Decisive record first (measured): tilted held-out games, where the tilt side is always the less-picked side and never the served side. Realised 11-10 (10.22 expected under served p; exact P(>=11) 0.45, half-null 0.50); home-line-decides 6-3 (4.38; 0.23); curve-decides 6-4 (4.91; 0.36); home-line-field 9-3 (5.86; 0.06); curve-field 11-7 (8.76; 0.21); no-skill 11-9 (9.73; 0.37). The tilt sides covered at about the rate the served probabilities predict. Nothing suggests a cover mechanism.
Held-out first-place change, 10-fold bootstrap (measured; expected correct picks given up per week, simulated, trailing-conditional):
- realised (oracle ownership, not knowable pre-lock, inferred): +0.54 pp [0.32, 0.77], P+ 1.00, 9/10, gap 0.22; cost 0.056 [0.025, 0.094]
- home_line_decides (deployable): +0.06 [-0.12, 0.24], P+ 0.71, 3/10 (5 folds tilt nothing), gap 0.21; cost 0.025
- curve_decides (deployable): +0.17 [0.03, 0.32], P+ 0.99, 5/10 (4 tilt nothing), gap 0.17; cost 0.018
- home_line_field: +0.13 [0.05, 0.22], P+ 1.00, 6/10; cost 0.029
- curve_field: +0.17 [0.08, 0.27], P+ 1.00, 7/10; cost 0.047; top-10 -0.56 pp [-1.18, -0.01]
- no_skill: +0.54 [0.27, 0.84], P+ 1.00, 8/10; cost 0.053
Point range across the field models is +0.06 to +0.54 pp, and the envelope of the intervals is [-0.12, +0.84]. Most of the Unit 2 gain needs ownership we only see after lock. With pre-lock proxies, the gain is +0.06 to +0.17 pp at a cost of about 0.02 to 0.03 correct picks a week. Looks: 6 held-out cells in family pool07_standings_policy, plus inner selection over 59-75 candidate cuts per fold x 10 folds per arm.
Recorded (unresolved_below_power, payout_first_pp): `pool07_lowo_{home_line_decides,curve_decides,home_line_field,curve_field,no_skill}_first`. The realised arm is the existing `pool07_lowo_leverage_tilt_first`.
Serving path (not built): ownership does not change whether a side covers, so a tilt cannot enter the card as a flip rule. The only admissible route is a fitted term in the one calibrated probability: a pre-lock ownership proxy share with a named cover mechanism (crowd-side overpricing), with its coefficient fit leave-one-season-out on the opener grade and checked against the market and model-only baselines. The decisive records above show no such mechanism yet. A pool-objective decision (maximising first-place chance instead of cover probability) would need an owner change of objective, and the pick would still come out of one fitted score.

## Tried
Unit 1 (2026-11-03): quartile cuts, 24 looks, 20 cells. Unit 2: one held-out cell (realised). Unit 3: six field-model arms (above).

## Next
1. Rerun each week with the new standings (`--points`, `--rank`, `--weeks`), deployable arms only (curve_decides, home_line_decides).
2. Week 1-4 versus 5-8 ownership-share arms (Unit 1 sign flip) in lowo mode, if still wanted.

## Open
- The behind check uses standings before each simulated week; trailing is true in nearly all paths, so the conditioning is weak.
- Zero crossing closes nothing; no side rule enters the card; no entrant names stored.

ROADMAP POOL-07 update: Unit 3 field-model arms: held-out first-place change +0.06 to +0.54 pp across 6 field models (deployable pre-lock proxies +0.06 [-0.12, 0.24] and +0.17 [0.03, 0.32]; oracle realised +0.54); cost 0.02-0.06 correct picks/week; tilt side 11-10 on decisive games (exact P 0.45); 5 cells recorded, all unresolved_below_power; challenger only.
