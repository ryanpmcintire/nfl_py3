# MOD-47

## Goal
Short-week confidence slope: calibration slope of the served logit inside short-week games, as a fitted interaction on the one calibrated probability.

## State
Unit 1 done 2020-25 / 2011-19 (2026-11-03 ~17:40). Script scripts/mod47_short_week_slope.py (log in scratchpad m47.log). Short = min(home,away) team rest days < median team rest (7.0, derived); week-1 rest unknown excluded; pushes excluded. Looks: 4 slope fits, 2 LOSO interaction arms, ~8 dLL contrasts (family mod47_short_week_slope).
Slopes (served logit): 2020-25 short 0.54 [-0.22,1.33] n=309, normal 0.31 [-0.10,0.73] n=1106; 2011-19 SBR short 0.10 [-0.26,0.46] n=444, normal 0.11 [-0.06,0.30] n=1646. Served prob is overconfident in BOTH groups (slope far below 1); short vs normal not separable.
LOSO z x short interaction vs served: 2020-25 dLL -0.00149 [-0.0036,+0.0006] P+ .08; decisive 256 games, interaction 117 / served 139; zs coef per fold -0.19..+0.72 unstable. 2011-19: dLL -0.00103 [-0.0019,-0.0002] P+ .008; decisive 217, 105/112; zs coef ~0 (-0.19..+0.07). Intercept-only control also not beaten.
Recorded 2 cells (log_loss_improvement, unresolved_below_power).

## Tried
Interaction (z, short, z*short) LOSO; intercept-only and served-only controls.

## Next
Overconfidence is global (slope 0.1-0.5), not short-week-specific: a global slope shrink (z scale) LOSO is the real calibration question; check whether it is already in the served path before proposing. Then short-week as side-specific term (home short minus away short) interacting with z.

## Open
Does the pooled +0.47 accuracy of short_week_diff survive as an additive term near 0.5 in this harness (not tested here)?

## Unit 2: global calibration of the served probability (2026-11-03 17:26 local)
Script scripts/mod47_unit2_global_calibration.py (log in scratchpad m47u2.log). Unit 1's z = raw model_probability (= home_cover_probability_at_open, max diff 0), NOT the served probability. Served = four_term_pick_probability_v1 (pick_probability.py:795 calibrated_pick_probability; active artifacts/pick_probability/20261103T195411Z, model_logit coef 0.308 already the global shrink). Served OOS = per_game.parquet out_of_season_home_probability, n=1503 2020-25.
Slope [95% boot]: raw 0.347 [0.009,0.671]; served OOS 0.922 [0.507,1.363] P(slope<1) .64, intercept 0.001 [-0.099,0.105]; in-sample 0.999. LL: coin 0.69315, raw 0.69692, served 0.68670. Served vs coin dLL +0.00644 P+ .986; vs raw +0.01021 P+ .995. Market-neutral baseline is 0.5 at the opener (same as coin).
LOSO slope+intercept shrink on served: LL 0.68692, dLL vs served -0.00021 [-0.00077,+0.00035] P+ .206 (worse, 2 of 6 seasons >0; fold slopes .86-1.01). 10 side changes only because of the fitted intercept; slope-only shrink is monotone and never flips.
Live 2026 Wk1-8 n=121: slope 1.42 [-0.89,4.51], LL 0.68586 vs coin 0.69315, mean p .551 obs .537 (small sample, descriptive).
Looks: 3 slope fits + 1 live fit, 1 LOSO arm, 3 dLL contrasts, 2 reliability tables (family mod47_global_calibration). Recorded 1 cell.
Conclusion: served is NOT overconfident; unit 1's overconfidence was the raw model, already corrected inside the served fit. No global shrink warranted; no v8 enrolment. Next: short-week as side-specific term (home short minus away short) inside the four-term fit.
