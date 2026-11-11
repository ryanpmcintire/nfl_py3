# MOD-57 P2-M knowable-information ceiling (deliberate leaks, positive controls, NOT SERVABLE)

## Goal
One number: base model given true answers for all pregame-knowable information, vs Tuesday opener, all and quiet games; plus a power check showing the harness sees a large effect. Extends mod57-p2l-availability-ceiling.md. Family mod57_availability_ceiling (the earlier 36 looks are shared).

## State
2026-11-11 10:05 DECLARED before any outcome read. Script scripts/mod57_p2m_knowable_ceiling.py, outputs artifacts/mod57_p2m/ (LEAK_NOT_SERVABLE names).
- Arm A: O1 actives + O3 QB (artifacts/mod57_p2l oracle table) + same-season full-season team means of the 13 served offense/defense rate inputs (home minus away). Reason for fallback (read): the PER-14 oracle source (scripts/unit_prior_features_on_production.py, src/nfl_ats/unit_prior_features.py) no longer exists, only pycache. Weather (read): temp and wind are already among the 88 served inputs (FULL_COLUMNS, nflverse realised values); no precipitation field exists in the repo; so no weather input is added. No snap share.
- Arm B: O1, O3, UNIT alone (3 components x 9 looks). Arm P: realised turnover margin (home takeaways minus giveaways, pbp, 2020-25) added. Shuffled control: Arm A permuted within season.
- Model/grading as P2-L: ridge alpha 10, walk-forward opener evaluation. Looks: A 9, B 27, P 9 = 45 new, plus 36 earlier.

## Tried
2026-11-11 10:09 RUN DONE, 45 cells recorded (registry mod57_availability_ceiling.json updated 10:09, 45 p2m names). Base 802-701 / 439-398 / 359-294 reproduced, max abs diff 0.
- All / quiet / moved: A 926-577 / 508-329 / 412-241 (+8.25 pts [5.10,11.44] P+ 1.000 all); O1 813-690; O3 812-691; unit ratings alone 916-587 / 509-328 / 401-252 (+7.58); turnover margin 1041-462 / 580-257 / 452-201 (+15.90); shuffled A 789-714 (-0.86, P+ .11).
- Inferred: unit-rate arm is contaminated, each season mean includes the game's own result (as-of rates are smoothed over that game); the per14 oracle (OL/skill, +0.13) did not. A leave-own-game-out unit arm would be a new look, not run.

## Next
Optional undeclared: unit ratings with the game's own contribution removed, to get a clean knowable ceiling.

## Open
Per14 oracle source absent (pycache only); weather already served (temp, wind), no precipitation in repo.

