# MOD-57 u1c - regrade with pushes kept

## Goal
Rerun the u1b regrade on the served pool-hook population (build_fit_population keep_pushes=True) and replace the u1b cells. Stamped 2026-11-11 08:45 -0500.

## State (measured; tests/scratch/mod57/u1c_run.py, u1c_run.out, u1c_cells.json, u1c_batch.json)
Population 1537 games, 34 pushes, all with stage-1 logits from hook_rows.parquet (artifact vs hook_rows diff 0.0 on 627 shared games). 1520 games with a 5+ book anchor.
OOS accuracy two-stage / stage1: FD 2023-25 55.17/55.30 (448 of 812); FD 2020-22 54.05/55.41; DK 2023-25 57.12/55.09 (465.5 of 815); DK 2020-22 52.43/54.78. Two-stage minus stage1 pts: FD -0.12 [-1.35,1.11] P+ .40; FD 20-22 -1.35 P+ .035; DK +2.02 [0.85,3.23] P+ 1.00 (log loss P+ .96); DK 20-22 -2.35 [-4.24,-0.56] P+ .004.
Fold coef FD .40-.84 (sd .15), all-seasons .47; DK .68-.97 (sd .09), all-seasons .805 = served.
Decomposition 2023-25 better-number cover: FD a .531 (128 games), b .500 (242), c .592 (51.5); DK a .524 (268), b .524 (105), c .547 (53).
Registry: 12 u1b cells invalidated (superseded_by), 12 u1c cells recorded unresolved_below_power, family mod57_pool_source_book.

## Tried
One declared run; no variants. First batch record ran from a wrong cwd and wrote a stray registry dir under tests/scratch/mod57; removed, rerun from repo root.

## Next
DK two-stage gain sits where the pool line differs by one half from the anchor; replicate on 2020-22 fails (negative). Check whether the DK 2025 gain is a single-season effect.

## Open
Pool posting time and side rule unknown; FD and DK disagree on sign of two-stage gain.
