# acc-u2-proper-scores

## Goal
Regrade every MOD-23 arm on log loss, Brier, discrete margin RPS, 2020-2025 at the opener. Parent: docs/lanes/accuracy-ceiling-theory.md.

## State
Done. scripts/mod24_u2.py writes artifacts/mod24_u2/results.{csv,json}. 17 arms x 3 metrics = 51 looks. 17 log_loss_improvement cells recorded (family mod24_proper_scores, unresolved_below_power).
Method: cover p = arm's stored home_cover_probability_at_open (discrete lattice, same policy as production); RPS rebuilt by solving the tilt theta that reproduces that p on the week's prior band pool, then scoring the whole-margin pmf (lead66 RPS). Season-blocked bootstrap, 6 seasons, 10000 draws. Positive = arm better.
Results: lattice-pass (LL and RPS P+ >= 0.90, positive): u1_greedy, u3_compact_net, u4_blend_compact_net, u4_blend_unit1_trimmed, u4_shrink_served (5/5 seasons). All five lose 1.1 accuracy points (flip W-L 152-168 etc). u5_man_zone: LL P+ 0.955, RPS 0.83, accuracy +0.47. u1 "better log loss, worse picks" still holds under the lattice.

## Tried
Reused per-game parquet p/ll/brier; no new pipeline. 34 push games have no arm p so RPS pairs exclude them (39 for u5 b/c).

## Next
Decide if proper-score gain from shrinkage-type arms is calibration only (flip net -17 identical across u3/u4 arms, suspicious; check). Fit one calibrated probability combining pick signal.

## Open
Base log loss 0.6969 > 0.693 (worse than a coin): the base's p is overconfident; improvement may be recalibration, not information.

## U2b (recalibration test, done)
scripts/mod24_u2b.py -> artifacts/mod24_u2b/{results.csv,json,pick_identity.json}. Each season's cover logit rescaled by T (and Platt a+bz) fitted on the other 5 seasons; RPS via same tilt. 102 looks; 17 temp cells recorded, family mod24_recalibrated_scores.
Result (measured): base LL 0.6969 -> 0.6922 recalibrated (T~0.34; base was heavily overconfident). Gains vanish: best arm u3_compact_net / u4_blend_compact_net +0.00049 LL, P+ 0.79, 4/6 seasons; u5_man_zone +0.00012 P+ 0.895; u1_greedy +0.0003 P+ 0.64; u4_trimmed 0.63; u4_shrink -0.0005 P+ 0.16. Raw gains (u1 +0.004) were calibration (u1 T 0.53 vs base 0.34). No arm resolved; none closed.
u3_compact_net and u4_blend_compact_net: identical p and picks (0 diffs; inferred blend weight 1 = same series). u4_trimmed/shrink picks differ 276/316 games from them; MOD-23's equal 785-718 is a tally coincidence, not identical picks.
Next: unit 3/4 compact_net is the only live information candidate; needs more seasons or a pooled fitted-probability test.
