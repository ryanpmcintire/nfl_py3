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
