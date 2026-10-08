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

## Net -17 check (done, genuine)
Measured from artifacts/mod23_unit{3,4}/*_per_game.parquet: compact_net, blend_compact_net, blend_unit1_trimmed, shrink_served each score 785-718 on 1503 games (base 802-701), so net -17 vs base for all. Recomputed correct == (pick_home == margin>0) on every row, 1503/1503, so the acc column matches the stored pick; no shared file, no wrong column, baseline is the opener frame. Pairwise: compact_net vs blend identical (0 differ); vs trimmed 276 differ, 138-138; vs shrink 316 differ, 158-158. Equal nets are an exact-tie coincidence of different pick sets (MOD-23 tally), not a bug. No recorded cell changes; no --replace needed.

Orchestrator recheck 2026-10-08 (measured): four rule columns hit 785 (cn, tr, sh rule; sh raw) while differing pairwise on 276-498 games; correctness code is standard (src/nfl_ats/clv.py:2315-2335) and per-season wins differ across arms (2025: 140/131/135), so no mechanical constraint found; treated as coincidence (it was noticed because it was odd, so its rarity is not evidence of a defect).
