# MOD-57 u1b - real offset regrade

VOID 2026-11-11: this unit dropped the 34 games that pushed at the Tuesday opener, which removes the half-point value it measured. Its numbers and its 12 registry cells are superseded by mod57-u1c-regrade-with-pushes.md. Kept as the record of what was run.

## Goal
Held-out all-games accuracy of the served pick probability at pool lines rebuilt from FD and DK source books. Stamped 2026-11-11 08:37 -05:00.

## State (measured; tests/scratch/mod57/u1b_run.py, output u1b_run.out, cells u1b_cells.json)
OOS all-games accuracy, two-stage / stage1 / intercept: FD 2023-25 55.41/55.53/49.37 (440.5 of 795); FD 2020-22 55.46/55.55; DK 2023-25 55.39/55.08 (442 of 798); DK 2020-22 53.73/54.90. Two-stage minus stage1 accuracy pts: FD -0.13 [-1.00,0.75] P+ .36 (23-25), -0.08 (20-22); DK +0.31 [-0.44,1.06] P+ .76, -1.17 [-2.50,0.07] P+ .03. Log loss ~0.684 (23-25). Fold coef FD -0.2 (sd .18, 2025 +0.20), DK +0.37 (sd .10). Offset 0/.5/1+ share 47/45/6.5 pct; better-number cover 0.501 FD, 0.507 DK (23-25). open_move <=0.5: 50.9-51.0 pct; >=1: 62.1-62.7 pct (23-25). 12 cells recorded unresolved_below_power, family mod57_pool_source_book. Sign check ok (corr anchor vs margin +0.44, SF home known game).

## Tried
Declared design run once; no variants.

## Next
Gap to 60 is about 4.5 pts; served structure is at stage-1 level at real-source lines. Next: where the 62 pct large-move games vs 51 pct small-move split comes from (stage 1 only reaches 60 where the line moved).

## Open
FD coefficient sign negative at FD-vs-median offsets (not the whole-number hook construct); pool posting time/side rule unknown.
