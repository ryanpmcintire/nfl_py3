# ENG-55

## Goal
Pool-line inference audit: is the pool line always the Tuesday market plus or minus half a point (ROADMAP ENG-55).

## State
Unit 1 done 2026-11-03 17:19 EST. scripts/eng55_pool_line_audit.py -> artifacts/eng55_pool_line_audit.csv (136 games Wk1-9). Pool minus market, absolute: 0 = 66, 0.25 = 2 (Wk9), 0.5 = 53, 1.0 = 13, 2.5 = 2. Market whole in 55 games: 53 are exactly 0.5 (hook), 2 are 2.5 (ARI_NYG Wk4, LV_NYJ Wk8; first post-noon snapshot is a thin transient, later Tuesday consensus = pool 1.5, not hooks). Hook set Wk1-8 = 44 of 121, identical to POOL-08; no rerun, no registry change. Field parquet line vs capture: 0 disagreements. Wk9 hooks 9 of 15. Served path (pool_hook.served_hook_frame, board_content pool_edge) reads captured pool; hook_rows (training) infers market+-0.5 on history only (no captures). 13 games with |diff| 1.0 are half-point markets that moved between first snapshot and capture.

## Tried
Join captures to tuesday_market_lines; compared field parquet.

## Next
Explain the 2 transient-snapshot games (ARI_NYG sign/size) and 13 one-point moves: is "first snapshot after Tuesday noon" the right anchor vs consensus at capture.

## Open
Wk9 badge check by code reading only (no rendered page inspected).
