# POOL-05

## Goal
Ownership proxy v2: field share of a side from Tuesday covariates (line, home, kickoff slot, team effect), validated out of week (ROADMAP POOL-05). Context for the pool-standing panel and the standing-aware Best Pick question (revisit Week 15). Never a pick input: one fitted cover probability selects every side.

## State
Unit 3, 2026-11-21 10:03 local (Week 10 rerun), measured and recorded (8 cells replaced). Family `pool05_ownership_v2`. `scripts/pool05_ownership_v2.py --last-week 10` -> `artifacts/pool_rank_replay/pool05_ownership_v2_w10.json` (W1-8 and W1-9 files untouched).
- 150 games W1-10. Line-only proxy held out: log loss 0.6792 per pick, mean absolute error 11.37 share points, rmse 14.03.
- Log-loss gain over line-only, game bootstrap [95%] P+ (weeks-bootstrap P+; in-sample gap): home+line +0.0017 [-0.0012, 0.0046] 0.880 (0.855; 0.0011); +slot +0.0014 [-0.0015, 0.0043] 0.831; +team +0.0005 [-0.0026, 0.0035] 0.644; slot+team (primary) +0.0002 [-0.0029, 0.0032] 0.541 (0.536; gap 0.0038); +curve +0.0038 [-0.0005, 0.0081] 0.962 (0.969; gap 0.0079, 7 of 10 weeks better).
- Team effect on the line-proxy residual: log loss -0.0013 [-0.0029, 0.0002] P+ 0.052 (weeks P+ 0.074); mean absolute error -0.26 points [-0.53, -0.01] P+ 0.019, whole interval below zero, recorded `refuted_mechanism` / `wrong_sign_resolved` for this ridge-residual parametrization only (game bootstrap; no week-cluster interval for this cell).
- Do teams differ: odd v even weeks r **+0.25 [0.01, 0.47] P+ 0.978** (one-sided permutation p .074; was 0.18 at W9); between-team spread 5.07 v null [3.42, 5.52], permutation p .122; largest team 10.4 points, family-wise p .48; 5 of 32 teams outside 0.025-0.975 (about 1.6 expected, inferred).
- Implies: card unchanged. Pool panel keeps the spread-only proxy (miss about 11.4 share points). The team pattern repeats a little more with Week 10, but the ridge team term still loses held out, so team popularity is real-but-unusable in this form. The curved-line term is now the strongest held-out gain (P+ 0.96) but carries the largest in-sample gap.

## Tried
Unit 1 (2026-11-03, W1-8): primary -0.0009 P+ 0.273. Unit 2 (2026-11-14, W1-9): primary +0.0002 P+ 0.564; team residual MAE -0.24 [-0.52, 0.03]. Unit 3 looks: 5 models x 10 weeks; 5 team-residual cells; 32 team intervals; inner lambda fits per fold.

## Next
1. Week 11+: `--last-week N`; replace the 8 cells (keep the MAE cell's closing ground only if its interval stays wholly negative).
2. Candidate proxy for unplayed weeks: home + curved line + POOL-08 half-point term in one fit, leave-one-week-out against line-only.
3. Root verified (2026-11-21 10:05 EST, measured): week-cluster bootstrap of the team-residual MAE cell [-0.621, -0.005], P+ .021 (game bootstrap [-0.527, -0.012]); the closure stands for this ridge form only. The loss sits in Weeks 1-4 (-0.33, -0.48, -1.68, -0.20) and Weeks 6-10 run slightly positive (+0.01 to +0.11), so a form that waits for several weeks of team history is a different construction, not closed.

## Open
- Not recorded: between-team permutation statistic and largest-team family-wise p (no signed effect); 32 team rows (components of one test), kept in the artifact.
- Intervals are game-level bootstraps; ten weeks is too few for a week-block interval. No entrant names stored.
