# g55-u5b-s-all-v2-grade-2026-11-21

## Goal
The one declared G55-U5b grade (goal-55-backwards-2026-11-21.md "G55-U5b declaration", committed a80fc28b before any outcome read): S_ALL_v2 = U3 harness with O2 from artifacts/g55_u5a/expected_snaps_v2.parquet (`expected_share`, LOSO), O1 and O3 as U3. 1,503 decisive 2020-25 games at the opener; looks all/quiet/moved = 3, family g55_deadline_availability; 50-draw within-season null; serving test as U3 (wins >= 827, null pct >= 95, log loss no worse than base); also v S_ALL 809-694. Zero crossing closes nothing; one fitted probability selects every side.

## State (2026-11-21 15:37 DONE, all measured)
- scripts/g55_u5b_grade.py: thin wrapper over scripts/g55_u3_servable_all.py (patches only OUT/SYNTH_TABLE/NULL_DIR/EXPECTED/EXPECTED_MANIFEST = the U5a swap; U3 logic and leak guards unchanged; extra v2 guard: realized_share_stored false, inactive forecasts 0, rows only on known team-games; own summarize with cells g55_u5b_s_all_v2_*). ruff format + check clean.
- Build 15:25-15:29 exit 0: synthetic team-games by season/slot = known counts; 0 on unknown games; O1 rows and QB counts identical to U3; sev2 rows 98,065 v 92,833.
- Base 802-701 reproduced; p_base identical to U3 (max diff 0.0), so v S_ALL pairs on U3 per_game p_arm (809-694 re-read).
- all: 812-691 v base 802-701, +0.67 pts season-block [-0.57, 1.56] P+ .867 (week-block .82); changed picks 94, arm 52-42. Log loss 0.69881 v 0.69678 (worse), Brier 0.25213 v 0.25164 (worse). v S_ALL +0.20 [-0.13, 0.64] P+ .771, changed 39, 21-18; log loss 0.69881 v 0.69904.
- quiet 837: 451-386 v 437-400 +1.67 [0.92, 2.33] P+ 1.0, v S_ALL +0.12 P+ .56. moved 653: 357-296 v 361-292 -0.61 [-2.53, 1.08] P+ .25, v S_ALL +0.31 P+ .74.
- Per season arm/base: 114-106/119-101, 131-105/130-106, 138-110/133-115, 136-130/132-134, 149-117/148-118, 144-123/140-127 (5 of 6 seasons up; 2020 down 5).
- Season-mean coefficients: O2 2020 -0.48, 2021-25 +0.08..+0.85; O1 2020 +0.43, 2021-25 -0.33..-1.20 (O1/O2 partly offsetting, collinear).
- Null 50 draws (18.6 s each; 3 workers 15:30-15:37): all wins pct 98.0 (mean 796.7, range 780-817), log loss pct 66; quiet 96.0 / 56; moved 81.0 / 50.
- Serving test FAILED: wins 812 < 827 FAIL; null pct 98 >= 95 PASS; log loss worse than base FAIL.
- Recorded 3 cells unresolved_below_power (record exit 0), shard registry/weak_signals/g55_deadline_availability.json modified; pool exit 0 (artifacts/g55_u5b/pool.log).
- Artifacts: artifacts/g55_u5b/ results.json, per_game.parquet (game_id, season, week, home_covered, p_base, p_arm, p_s_all, terms), build_summary.json, batch.json, null/draw_000..049.json, logs.

## Tried
- Wrapper instead of a copied script; cell names changed so U3's registry cells are not overwritten. One run, no variants.

## Next
1. Root: verify per_game.parquet records, commit scripts/g55_u5b_grade.py + shard, move this lane to done/.
2. ROADMAP G55 update: S_ALL_v2 812-691 (+0.67 P+ .867, null pct 98) v S_ALL 809-694 (+0.20 P+ .77); serving test failed on wins and log loss, passed null pct.

## Open
- Log loss worse than base in all and moved despite more wins: the arm sharpens probabilities without calibration gain; O1/O2 coefficient sign flip in 2020 v later seasons.
- Degenerate OL Doubtful LAD cell from U5a left as declared.
