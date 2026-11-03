# MOD-24 finer grade (2026-11-03)

## Goal
Finer grading lever: grade model variants on the opener margin error (squared and absolute, margin_vs_open vs the model's own residual_at_open), a continuous quantity per game, and measure the effective sample multiplier vs binary cover/accuracy on 2020-25 (artifacts/opener_evaluation/20261101T151614Z/per_game.parquet). Then regrade MOD-23 unit 13/14 arms and the MOD-24 postseason arm.

## State
Novelty (read 2026-11-03): acc-u2/u2b + scripts/mod24_u2*.py already did log loss, Brier and single-line discrete RPS (RPS = the sum of cover Briers over all lines, so a line sweep adds nothing new) on 17 arms; all unresolved. Not done anywhere: continuous margin-error grade, and any measured power multiplier (grep effective sample/multiplier: none in acc-* lanes). Declared before outcomes: family mod24_finer_grade, looks = 2 fine metrics (sq err, abs err) x arms; multiplier = mean t^2 fine / mean t^2 binary across the 17 arms, game-level paired t, season-block bootstrap interval. Unit 13/14/postseason files hold only p, so their margin is inferred by inverting p on the base (p, residual, line) relation (inferred, flagged).
Script: scripts/mod24_finer_grade.py -> artifacts/mod24_finer_grade/.

## Tried
Measured 2026-11-03 (scripts/mod24_finer_grade.py, artifacts/mod24_finer_grade/results.json), 16 arms x 2 fine metrics = 32 looks, 16 sq-error cells recorded (family mod24_finer_grade, unresolved_below_power). Multiplier vs binary accuracy, mean t^2 ratio: log loss 1.19 [0.54,2.72], sq error 1.39 [0.72,2.59], abs error 1.29 [0.65,2.36] (arm-bootstrap). Sign agreement with accuracy 11-12 of 16; corr of t 0.21 (sq), -0.10 (ll). Fine grade ranks arms differently: u1_greedy/u3_compact_net/u4_shrink gain sq error (P+ >=.999) while losing accuracy (decisive 152-168). Base margin RMSE 13.03 (sq 169.8).
Unit 13/14 and postseason files store only p; inverting p to margin has R2 0.69 on the base, too lossy, so not fine-graded.

## Next
Finer grade is exhausted as a power lever (about 1.2-1.4x, interval spans 1). Margin error rewards shrinkage/calibration the way log loss does. Remaining: more graded games (postseason rows, other-sport transfer) or new information.

## Open
p-only arms (unit 13/14, postseason) need residual saved by their scripts to be fine-graded.
