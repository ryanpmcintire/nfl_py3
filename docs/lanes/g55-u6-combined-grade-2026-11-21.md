# g55-u6-combined-grade-2026-11-21

## Goal
The one declared G55-U6 grade (goal-55-backwards-2026-11-21.md "G55-U6 declaration", committed e3e84cbc before any outcome read): arm C = S_ALL_v2 (frozen U5b table) plus MOD-23 unit 5a man/zone coverage inputs (home/away/diff_cov_mz, repaired algebraic deviation, Unit 5b) in one walk-forward ridge, model alone at the Tuesday opener, 1,503 decisive 2020-25 games. POST-SELECTION in every write-up. Looks all/quiet/moved = 3, family g55_combined; 50-draw within-season null; serving test >= 827 wins, null pct >= 95, log loss no worse than base. Zero crossing closes nothing; one fitted probability selects every side.

## State (2026-11-21 15:42, script written, nothing graded yet)
- scripts/g55_u6_combined_grade.py (modes terms/grade/null/summarize), ruff format + check clean. Reuses U3 run_eval/season_paired/grade_frame and U5b guard_v2/changed unchanged; reads artifacts/g55_u5b/servable_features.parquet as the frozen S_ALL_v2 block (no rebuild); coverage terms rebuilt with mod23_unit5 build_terms + the Unit 5b exact deviation, pinned to the frozen inputs (participation 20260813T131635Z, pbp 20260929T191306Z, schedules data/raw/20260929T191306Z) and compared to artifacts/mod23_unit5/terms.parquet. Guards: U3 nonzero-on-unknown-game abort, U5b v2 guard, coverage zero before 2018, no missing terms, base must be 802-701.
- Null implementation (fixed before any outcome read): per draw, seed [20261121, k] as U3; within each season from 2018, one permutation of the season's deadline-known games moves all six inputs (3 S_ALL_v2 + 3 coverage) together as rows; the coverage block on that season's games not known at the deadline (all of 2018-19, the rest of 2020+) is permuted among themselves in the same draw, so S_ALL_v2 stays zero off known games.
- Outputs: artifacts/g55_u6/ only.
- terms 15:43 exit 0: 4,902 rows, max abs diff v frozen mod23 terms 1.1e-16, zero before 2018, none missing.
- grade 15:45 exit 0 (measured): base 802-701 (p identical to U5b, max diff 0.0); S_ALL_v2 812-691 (max diff v U5b p_arm 0.0); arm a alone 806-697 (NOT 809-694; same 1,503 games; frozen unrepaired terms in this harness also 806-697, so the repair is not the cause; inferred drift in shared code or the features file since 2026-09-30).
- Arm C (POST-SELECTION) all 807-696 v base +0.33 pts season-block [-1.44, 1.90] P+ .637 (week-block .644); changed 131, arm 68-63. v S_ALL_v2 -0.33 [-1.21, 0.51] P+ .221; changed 71, 33-38. Log loss 0.69841 v base 0.69678 (worse) v S_ALL_v2 0.69881; Brier 0.25197 v 0.25164 v 0.25213.
- quiet 837: 450-387 v 437-400 +1.55 [0.29, 2.53] P+ .989; v S_ALL_v2 -0.12 P+ .32. moved 653: 353-300 v 361-292 -1.23 [-3.88, 1.36] P+ .171; v S_ALL_v2 -0.61 P+ .14.
- Per season arm C/base: 112-108/119-101, 128-108/130-106, 139-109/133-115, 139-127/132-134, 144-122/148-118, 145-122/140-127 (3 of 6 up).
- Coverage coefficients (season means, scaled): home/away_cov_mz near 0 in 2020-21, -0.12..-0.32 by 2022-25; O1/O2 as in U5b.

## Tried
- One grade run, no variants; arm a frozen-term attribution run (scratchpad u6_arm_a_attr.py, artifacts/g55_u6/arm_a_attribution.json), a reproduction check, not a look.

- Null 50 draws (15:46-15:49, 3 workers, all exit 0, no process left): all wins pct 88.0 (mean 794.1, range 763-815), log loss pct 80; quiet 90.0 / 80; moved 68.0 / 68.
- Serving test FAILED on all three legs: wins 807 < 827; null pct 88 < 95; log loss 0.69841 worse than base 0.69678.
- 15:50 recorded 3 cells unresolved_below_power (record exit 0, family g55_combined, notes POST-SELECTION), new shard registry/weak_signals/g55_combined.json (untracked); pool exit 0 (artifacts/g55_u6/pool.log).
- Artifacts: artifacts/g55_u6/ results.json, per_game.parquet (game_id, season, week, home_covered, p_base, p_s_all_v2, p_arm), coverage_terms.parquet, terms_summary.json, batch.json, arm_a_attribution.json, null/draw_000..049.json, logs.
- DONE 2026-11-21 15:50.

## Next
1. Root: verify, commit scripts/g55_u6_combined_grade.py + registry/weak_signals/g55_combined.json, move this lane to done/.
2. ROADMAP G55 update: arm C (POST-SELECTION) 807-696, +0.33 [-1.44, 1.90] P+ .64 v base, -0.33 P+ .22 v S_ALL_v2, null pct 88; serving test failed on all three legs. Adding coverage to S_ALL_v2 does not add wins on history; S_ALL_v2 alone stays the better arm for the U7 forward logger.

## Open
- Arm a alone no longer reproduces 809-694 in today's harness (806-697 with repaired or frozen terms, same games); the recorded mod23_unit5 arm a cell predates the drift. The cause (shared margin code or the features file since 2026-09-30) is not located yet.
