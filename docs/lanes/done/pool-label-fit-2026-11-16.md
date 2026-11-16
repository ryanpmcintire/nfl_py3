# pool-label-fit-2026-11-16

## Goal
Goal-65 slot: train/serve parity. The served four-term stage is fitted on cover labels at the opener and the hook term (0.805) patches the half point; the card is decided and graded at the pool line (FanDuel Tuesday noon, half point off whole numbers; mod57 u7, u1c, u8). Test the stage fitted directly on pool-line labels. Script scripts/pool_label_fit.py, outputs artifacts/pool_label_fit/. Family pool_label_fit_2026_11_16.

## Prior work (read)
- Served fit: src/nfl_ats/pick_probability_fit.py:503-516 stage 1 = ridge logistic on FIT_FEATURES (:55) over non-push rows labelled home_covered at the opener; :520-565 stage 2 = one offset coefficient on hook_rows (pool_hook.py:61-82: Tuesday opener +-0.5, prior-season push mass), fold stage-1 logit as offset. Serve: pool_hook.py:136-175 crossed push mass Tuesday market line -> card line.
- No unit fitted the stage itself on pool-line labels: u1c/u8 refit only the hook coefficient on FD-rebuilt rows; mod27-u4 stacked five-term refit used simulated opener +-0.5 pool lines (regressed 472 v 487 non-whole); fix-picks u5 graded at simulated pool lines.

## Declaration (stamped Mon 2026-11-16 17:45:11 EST by `date`, before any outcome read)
- Population: build_fit_population(keep_pushes=True), active move version, 2020-25 (1,537). FD quote: first FanDuel quote <=1 h after Tuesday noon (2023-25, u1_arch.pkl), last within 4 h before (2020-22, u1_pre.pkl), as u1c; no book-count filter.
- Pool rule (u7): FD whole number -> pool at both FD+-0.5, weight 0.5 each (side unidentified, 12/7 dog/fav); FD half-point -> pool = FD. Label y = margin > pool.
- Hook h = hook_value(sign(pool - Tuesday opener), crossed_push_mass(opener, pool, push table without the held season)).
- Arm S (served two-stage): src LOSO stage 1 on opener labels plus src fold hook coefficient, recomputed with src functions and checked against the active artifact.
- Arm L (primary): one weighted ridge logistic (FIT_RIDGE, standardised) on the four FIT_FEATURES + h, fitted on FD-built pool rows of every season except the held one. Arm L4 (secondary): same without h.
- Grade: held 2023-25 rows at the rebuilt pool line, weighted. Decisive record first, record, accuracy, log loss, Brier; L minus S week-block (4000) and season-block bootstrap P+; fold coefficients; in-sample minus OOS gap; per season.
- 2026 W1-10: frozen L/L4 fit on all FD-built 2020-25 rows vs active artifact coefficients; real pool lines and served inputs as tests/scratch/fixpicks/u9_replay.py (root_m2_games W1-9 + forward-log W10, served flags, served Tuesday anchors). Settled non-push games. Changed picks and their record.
- Looks: 2 arms x 2 periods x 3 units = 12 cells, one family. No variants.

## State (measured, run 2026-11-16 17:47 EST, exit 0; artifacts/pool_label_fit/run.out, result.json, history_rows.parquet, forward_2026.parquet)
- Checks: stage 1 recomputed with src functions equals the active artifact (20261116T171555Z) on 1,503 games (max diff 0.0); hook full .8047 and folds identical. FD coverage 2020-22 196/227, 195/239, 255/255; 2023-25 270-271/272. Graded 2023-25: 812 games (973 rows), 161 whole-FD games split +-0.5; h nonzero on 52%.
- 2023-25 held out at the rebuilt pool line: S 459.5-352.5 (56.59%, ll .68406); L 451.0-361.0 (55.54%, ll .68437); L4 451.0-361.0 (ll .68402). Decisive: L 10.5 v S 19.0 of 29.5; L4 33.0 v 41.5 of 74.5.
- L minus S, week block (54): acc -1.05 pts [-2.00,-0.06] P+ .015; ll -.00031 [-.00133,+.00067] P+ .28; Brier P+ .29. Season block (3): acc P+ .00, ll P+ .00. L4 minus S: acc -1.05 [-2.54,+0.43] P+ .07; ll +.00004 P+ .51 (season block .59).
- In-sample minus OOS: S .12 pts / ll .00122; L .92 / .00174; L4 0 / .00118.
- Fold coefs: L hook term h .30-.74 (full .446) v served hook .77-.875 (.805); L model_logit .25-.44 (full .357) v S .22-.38 (.308); flags and move nearly equal; L move_available -.07..+.02 v S -.01..+.04.
- 2026 W1-10 frozen fit, real pool lines, 149 settled: S 97-52 (ll .66826), L 93-56 (.67019), L4 91-58 (.67342). L changes 8 sides, L right 2; L4 changes 14, right 4. Week block L acc -2.68 [-6.90,+1.35] P+ .075, ll P+ .11; L4 acc P+ .06, ll P+ .046. Caveat: S formula was chosen after W1-9 (picks-getting-worse lane), so its 97-52 is optimistic.
- Recorded 12 cells unresolved_below_power, family pool_label_fit_2026_11_16 (artifacts/pool_label_fit/batch.json); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.

## Tried
One declared run, arms L and L4; no variants.

## Next
Nothing served. Fitting the stage on FD-rebuilt pool labels loses to the served opener-label stage plus hook; the gap is the hook shrinking (.45 v .805) when its coefficient must fit noisier rebuilt pool rows. Orchestrator: the L 2023-25 accuracy cell interval lies wholly below zero (one of 12 looks; log loss crosses), recorded unresolved.

## Open
- History model_logit is at the opener; 2026 served base probability is at the card line. Same for both arms.
- Rebuilt FD pool side for whole numbers unidentified (weight 0.5 each side); a real side rule could change L's hook fit.
- Script lint: 18 style findings remain (dict() calls, zip strict); ruff safe fixes applied after the run, no semantic change.
