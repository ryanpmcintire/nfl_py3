# POOL-05

## Goal
Ownership proxy v2: field share of the served side from Tuesday covariates (line, home, primetime, team brand effect) validated out of week (ROADMAP POOL-05).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: POOL-03 found the line-only proxy is the limit (within-week Spearman 0.50). The field loves certain teams and slots beyond the line; a shrunk team effect plus kickoff slot may raise the held-out fit, and the pool-objective term is only as good as the proxy.

## Tried
Novelty check only. Nearest: POOL-03 unit 1 (line-only proxy), POOL-01 (field model), POL-04 (closed: shares only appear at the deadline). Grep for team-effect ownership or brand covariates found none. Differs: covariates and team effects, validated before any leverage replay.

## Next
First unit: Fit leave-one-week-out logistic of served-side share on logit(line proxy), home, slot and a shrunk team effect; report held-out log loss against the POOL-03 line-only 0.6796 and the team-effect split-half correlation across odd and even weeks.

## Open
Changes: Better ownership proxy for the pool objective. Data: data/splash/field per-game counts (121 games W1-8, growing 15 per week). Out-of-week fit of ownership on Tuesday-known covariates. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.

## Unit 1 (2026-11-03 16:03, family pool05_ownership_v2)
scripts/pool05_ownership_v2.py -> artifacts/pool_rank_replay/pool05_ownership_v2.json. Looks: 5 outer cells x 8 weeks = 40 fits, plus 1,920 inner ridge-lambda fits (6 lambdas x 7 x 8 x 5; lambda chosen inside training weeks). Home-share logit on spread, slot (thu/sun_early ref/sun_late/sun_night/mon), shrunk team effect (+1 home, -1 away), optional x|x|.
- Baseline verified (measured): 0.6796 = entry-weighted log loss per pick of model-side share on its line, leave-one-week-out; recomputed on same 121 rows.
- Improvement over 0.6796 (positive = better), games bootstrap: M0h home+line +0.0010 [-0.0020,0.0040] P+ 0.745; M1 slot +0.0010 P+ 0.728; M2 team -0.0007 [-0.0038,0.0025] P+ 0.323; M3 slot+team (primary) -0.0009 [-0.0042,0.0024] P+ 0.273 (weeks bootstrap [-0.0039,0.0020] P+ 0.283); M4 +curve +0.0002 P+ 0.542.
- In-sample 0.6767 v out-of-week 0.6805 for M3 (gap 0.0038). Chosen lambda mostly 1e5 (slot and team effects shrunk to ~0); team effect odd/even split-half r 0.34. Fold coefficients in JSON.
- Recorded M3 and M0h as unresolved_below_power. Team and slot effects add nothing measurable at 8 weeks; the gain, if any, is the home-side parametrization.
- Pool term entry (not implemented): proxy share enters the pool-relative objective as logit(share) alongside logit(p) in one fitted score; no side flips by itself.
## Next
Re-run when W9+ lands (more games per team); try team effect on model-side residual of the line proxy.
