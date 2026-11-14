# POOL-05

## Goal
Ownership proxy v2: field share of a side from Tuesday covariates (line, home, kickoff slot, team effect), validated out of week (ROADMAP POOL-05). Context for the pool-standing panel and the standing-aware Best Pick question (revisit Week 15). Never a pick input: one fitted cover probability selects every side.

## State
Unit 2, 2026-11-14 16:30 (Week 9 rerun + owed team-effect unit), measured and recorded. Family `pool05_ownership_v2`. `scripts/pool05_ownership_v2.py --last-week 9` -> `artifacts/pool_rank_replay/pool05_ownership_v2_w09.json` (the Week 1-8 file is untouched; `--last-week 8` reproduces it for all five models, measured).
- 136 games; per week 16,16,16,16,15,14,14,14,15.
- Held-out error of the line-only proxy (leave-one-week-out): log loss 0.6798 per pick, mean absolute error 11.22 share points, rmse 13.96.
- Log-loss gain over line-only (positive = better), game bootstrap [95%] P+: home+line +0.0019 [-0.0013, 0.0050] 0.873; +slot +0.0019 [-0.0011, 0.0051] 0.884 (slot terms shrunk to zero in every fold); +team +0.0006 [-0.0026, 0.0038] 0.645; slot+team (primary) +0.0002 [-0.0029, 0.0035] 0.564 (weeks bootstrap P+ 0.562; in-sample 0.6755 v held-out 0.6795, gap 0.0040); +curve +0.0024 [-0.0018, 0.0066] 0.868 (gap 0.0090). Weeks 1-8 primary was -0.0009, P+ 0.273.
- Owed unit, team effect on the line-proxy residual (model-side share logit = proxy offset + ridge team effect, lambda chosen in training weeks: 1e3 x4, 1e4 x4, 1e5 x1): held-out 0.6811, gain -0.0014 [-0.0032, 0.0002] P+ 0.047 (weeks bootstrap P+ 0.095; 6 of 9 weeks better, the losing weeks lose more). Mean absolute error 11.45 v 11.22: -0.24 points [-0.52, 0.03] P+ 0.043.
- Do teams differ at all (per-team mean residual, home-oriented residuals shuffled within week, 20,000 draws): between-team spread 5.17 share points v 4.67 expected by chance [3.59, 5.75], permutation p 0.171. Largest team +11.5 [5.3, 16.7], family-wise permutation p 0.358. Odd v even weeks r 0.184 [-0.086, 0.427] P+ 0.92, permutation one-sided p 0.142. Team effects range -7.9 to +11.5 on 8-9 games each; 4 of 32 have P+ outside 0.025-0.975 (HOU -7.1, BUF +4.0, LV +8.8, NE +11.5) against about 1.6 expected by chance (inferred). All 32 with intervals are in the artifact.
- Implies: card unchanged. Pool panel: the spread alone is still the supported way to predict the field's split before lock, and it misses a game by about 11 share points, which is the uncertainty to state wherever a predicted (not observed) share is shown. Do not add team popularity: held out it makes the fit worse and the team pattern repeats only weakly. The small supported gain is the home-team term. Nothing closes: every cell is `unresolved_below_power` (no interval wholly on the wrong side; split-half is positive, not zero).
- Recorded 8 cells with replace (5 models, `pool05_team_residual_logloss`, `pool05_team_residual_mae`, `pool05_team_effect_split_half`); `weak-signals pool` loads (exit 0).

## Tried
Unit 1 (2026-11-03): 121 games Weeks 1-8, baseline 0.6796; primary -0.0009 [-0.0042, 0.0024] P+ 0.273; home+line +0.0010 P+ 0.745; team split-half r 0.34. Unit 2 looks: 5 models x 9 weeks; 5 team-residual cells; 32 team intervals; inner lambda fits 6 x 8 x 9 per model. The team term costs about 0.0013 held out in both parametrizations (home-share model and residual model), so the result does not depend on orientation.

## Next
1. Week 10+: `--last-week N` (writes `pool05_ownership_v2_wNN.json`).
2. Candidate proxy for unplayed weeks: home + line + POOL-08 half-point term in one fit, leave-one-week-out against line-only; this is the predicted-share step the POOL-01 replay lacks.

## Open
- Not recorded, with reason: the between-team permutation statistic and the largest-team family-wise p have no signed effect or probability_positive (carried in the split-half cell's description); the 32 team rows are components of one test (each game enters two teams with opposite sign), kept in the artifact.
- Intervals are game-level bootstraps; nine weeks is too few for a week-block interval. No entrant names stored.
