# MOD-57 u8 - FanDuel-anchored pool term
Created 2026-11-11 09:08 -0500 (declaration written before any outcome was read).

## Goal
Find which reference line gives the pool-line term real value when the pool is rebuilt from FanDuel (unit 7: pool = FD 77 of 101). Run once; changes nothing served.

## State
Declared (pool = FD-rebuilt in every arm; 2023-25 primary, 2020-22 secondary separately; population = u1c D set with FD quote and 5+ book anchor, pushes kept, 1,537-game build):
- Arm L: anchor = median of bovada, williamhill_us, mybookieag from the same quote window as FD (post-noon first quote 2023+, last pre-quote 2020-22). Coverage 3/2/1/0 per season reported; zero-leader games get zero term.
- Arm S: anchor = FD own line; term nonzero only where pool moved off FD whole number.
- Reference: all-books median (5+).
- Machinery of u1c_run.py (push table from other seasons, LOSO coefficient, stage 1 = served OOS logit, intercept control, week-block bootstrap 4000).
- Reports: decisive record first, all-games record/accuracy, log loss, Brier, diff + P+, fold coefs vs 0.805, nonzero share, better-number cover, in vs out gap, per season.
- 2026 descriptive (weeks 3-9, u7_table joined to root_m2_games, graded at pool line): split pool != leaders into (i) pool = FD Tuesday line, (ii) half point off FD whole number, (iii) other. No fit.
- Looks: 12 paired + 3 descriptive, family mod57_pool_source_book.

## Results (measured, tests/scratch/mod57/u8_run.py, u8_run.out, u8_batch.json; run 2026-11-11 09:12 -0500)
Leader coverage (5+ set, 3/2/1/0 books): 2023 255/17/0/0, 2024 156/112/4/0, 2025 244/27/1/0; 2020-22 mostly 3, 2020 has 10 with none. Push check: whole-line margin==line 5.1% (3: 10.0%, 7: 4.1%). Sign: 2020_01_ARI_SF line 7.0, home SF, margin -4.
2023-25 two-stage / stage1 acc: L 55.23/55.30 (diff -0.06 pts, P+ .45; ll P+ .42), S 55.54/55.30 (+0.25, P+ .64; ll P+ .71, brier .74), ref 55.17/55.30 (-0.12, P+ .40). Decisive: L 29.0 vs 29.5 of 58.5; S 20.0 vs 18.0 of 38; ref 21.5 vs 22.5 of 44.
2020-22: L -1.35 pts P+ .10; S -0.95 pts P+ .09 but ll +0.00092 P+ .995; ref -1.35 P+ .03.
Fold coefs: L .42-1.08 (sd .19), S .79-.98 (sd .065, all .90), ref .40-.84; served 0.805. Nonzero share 2023-25: L .58, S .20, ref .54. Better-number cover on nonzero: L .534, S .525, ref .517.
2026 wk3-9 (76 games pool != leaders, graded at pool): (i) pool=FD 52, better covers 32-20, decisive 10-6; (ii) half off FD whole 15, 7-8, decisive 1-3; (iii) other 9, 6-3, decisive 3-1. 32 of 105 joined games had nearest FD scan more than 6h from Tuesday noon.
Registry: 15 cells recorded unresolved_below_power, family mod57_pool_source_book.

## Tried
One declared run, no variants.

## Next
Replicate S on 2026 once Tuesday-noon FD captures exist (wk11+); check S in 2025-only vs earlier.

## Open
Half-point shift rule for FD whole numbers unknown.
