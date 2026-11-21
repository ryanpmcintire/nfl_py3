# Pool-field reruns at Week 10 (2026-11-21)

## Goal
Rerun POOL-05, 07, 08, 09, 10 on the Week 10 pool read and record each in the weak-signals registry. Rows' own lanes hold detail.

## State
Done 2026-11-21 10:03 local. All shards changed today (measured mtimes): pool08 09:59, pool09 10:01, pool07 10:01, pool05 10:01, pool10 10:02. `nfl-ats weak-signals pool --league nfl --effect-units accuracy_points` exit 0 (472 lines). Batch files built by a scratchpad script from the artifacts; 32 cells recorded with replace. No processes left running (measured).
- POOL-08: hook-side share minus proxy +5.49 pp [1.36, 9.35] P+ 0.995, perm p .078, 57 games; Week 10 alone -5.66 (5 games). 10 cells.
- POOL-09: within-week push-mass slope +1.02 [-0.38, 2.44] P+ 0.931; pooled +0.62 P+ 0.842. 6 cells.
- POOL-05: slot+team +0.0002 P+ 0.541; split-half r +0.25 [0.01, 0.47] P+ 0.978; team-residual MAE -0.26 [-0.53, -0.01] recorded `refuted_mechanism`/`wrong_sign_resolved` (that parametrization only). 8 cells.
- POOL-10: coefficient 0.212 [0.016, 0.382] P+ 0.998; Week 10 alone -0.09. 7 cells.
- POOL-07: new `--mode lowo` fitted leverage cut, held-out +0.54 pp first-place chance [0.32, 0.77] P+ 1.00, in-sample 0.76, gap 0.22; 1 cell (payout_first_pp).

ROADMAP POOL-05 update: **2026-11-21 Week 10 (measured, lane):** 150 games; slot+team +0.0002 P+ .54, home+line +0.0017 P+ .88, curve +0.0038 P+ .96 (gap .008); teams repeat odd/even r .25 [.01, .47] but the team term still loses held out (MAE -0.26 [-0.53, -0.01], closed wrong_sign_resolved for that form only). Next unit: home + curve + POOL-08 hook term as one predicted-share proxy, leave-one-week-out.
ROADMAP POOL-07 update: **2026-11-21 (measured, lane):** fitted per-game leverage cut chosen leave-one-week-out: held-out +0.54 pp first-place chance [0.32, 0.77] P+ 1.00, 9 of 10 folds, gap 0.22; recorded payout_first_pp unresolved. Challenger only. Next unit: add field-model arms and expected-wins cost to --mode lowo.
ROADMAP POOL-08 update: **2026-11-21 Week 10 (measured, lane):** +5.49 pp [1.36, 9.35] P+ .995, perm p .078, 57 games; Week 10 alone -5.7; coefficient 0.229 [0.057, 0.407]; 10 cells. Next unit: rerun at Week 11.
ROADMAP POOL-09 update: **2026-11-21 Week 10 (measured, lane):** within-week push-mass slope +1.02 [-0.38, 2.44] P+ .93; pooled +0.62 P+ .84; 6 cells. Next unit: key-number-matched contrast, rerun at Week 11.
ROADMAP POOL-10 update: **2026-11-21 Week 10 (measured, lane):** favourite-adjusted half-point coefficient 0.212 [0.016, 0.382] P+ .998; Week 10 alone -0.09; 7 cells. Next unit: matched-line contrast, rerun at Week 11.

## Tried
POOL-08 first-snapshot anchor has no Week 10 hook games (open in POOL-08 lane).

## Next
Root: apply the five ROADMAP lines; verify the POOL-05 MAE closure (game bootstrap only) and the POOL-07 interval (fold bootstrap, excludes field-model uncertainty).

## Open
New files to commit: scripts/pool07_standings_policy.py (lowo mode), artifacts listed in each lane.
