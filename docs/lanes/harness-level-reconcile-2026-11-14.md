# harness-level-reconcile-2026-11-14

## Goal
Explain the 28-row gap between U5 (757-565) and the hook lane (729-593) on the 1,322 whole-number Tuesday rows, 2020-25, and name the harness a stage-2 refit may use. Parent: fix-picks-2026-11-10.md Next 4. Audit only: nothing served changed, nothing fitted, no registry record.

## State (2026-11-14 16:02 ET; measured unless labelled)
- Identity: `u5_loso.py` rerun unchanged gives 757-565 (stage 1 742-580), largest difference from its saved rows 0.0. 729-593 (stage 1 726-596, log loss .68343 / .68223) reproduced from `artifacts/pick_probability/20261102T155750Z/hook_rows.parquet`. That number is the served fit's own hook rows (lane mod27-u4-hook-serve.md), not `tests/scratch/pool_hook/hook.py`.
- Join on game and side: 1,322 rows in both, 0 in one only, 0 pool-line differences, 0 grade differences.

| bucket | rows | U5 | hook lane | gap |
|---|---|---|---|---|
| same pick, same grade | 1,210 | 687-523 | 687-523 | 0 |
| pick differs, same grade | 112 | 70-42 | 42-70 | +28 |
| grade differs, either pick | 0 | - | - | 0 |
| in one harness only | 0 | - | - | 0 |

- Cause 1, 26 of the 28 (110 rows, 68-42 v 42-68): the fit's counted flag set changed between the two runs. Through fit 20261102T171412Z the flag sum counted protection only; from 20261103T173458Z it counts division and protection (metadata `counted_flag_columns`), after the division flag's 2026 record went 0-0 to 3-0 (pick_probability_fit.py:351-359; the gate's condition is inferred from the two metadata states). The flag sum changed on 264 of 1,503 games; model logit, move, move availability, Tuesday line, margin and grade have 0 differences. No source commit in between. Every fit from 11-03 through the active 20261114T184614Z reads stage 1 742-580, two-stage 755-567, log loss .67976 / .67856; the game-level record went 833-670 to 829-674.
- Cause 2, 2 of the 28 (4 rows, 3-1 v 1-3): U5's S arm leaves the held season out of every training row's push mass and weights rows 0.5 (u5_loso.py:35, :113, :122); the served fit uses all earlier seasons, unweighted (pool_hook.py:71-72, pick_probability_fit.py:554-558). U5's served-style arm equals the served rows at equal inputs: 755-567, 0 picks differ, largest logit difference .0001.
- Pitfalls tested, none is a cause: pushes kept in both (34 games, same 1,322 rows); same base (fold four-term fit on the at-open model logit, stage-1 logits equal to 0.0 at equal inputs); both stack both sides; both grade margin > pool line (u5_loso.py:22, pool_hook.py:80); move input identical (799 games, zero in 2020-22); neither harness contains the at-market conversion.
- Served path today (read: pick_probability.py:872-890, :232-233) feeds the model's stated probability at the pool line plus the hook. Both harnesses feed the probability at the whole-number Tuesday line, so neither matches it. Measured with served functions and the U8 stated probabilities (`tests/scratch/fixpicks/u8_plus.csv`, `u8_minus.csv`; U8 base equals the served base on all 1,537 games): stated base plus hook at fold coefficients 754-568, log loss .67833; without the hook 741-581, .67905. Against the fit harness 49 sides differ, 24-25 v 25-24.

## Correct level
Served two-stage probability as the code stands: 754-568, log loss .67833. Beside it: fit harness (at-market base) 755-567, .67856; U5 757-565, .67853; hook lane 729-593, .68223 (right on 11-02, stale since the 11-03 fit). These are fitted-history levels with flags discovered on the same seasons, not evidence of a forward edge.

## Tried
`tests/scratch/harness_level/reconcile.py`, `flagdiff.py`, `served_path_level.py` -> `artifacts/harness_level/reconcile.json`, `reconcile_rows.csv`, `flagdiff.json`, `reconcile_log.txt`; U5 originals backed up in `artifacts/harness_level/u5_saved/`.

## Next
1. A stage-2 refit uses the construction in `tests/scratch/fixpicks/u8_analysis.py:34-42` (served `build_fit_population` and `hook_rows`, stated probability at M+0.5 / M-0.5 as the offset base) with U5's held-season push rule (u5_loso.py:35), every arm inside one `build_fit_population` call, and prints `counted_flag_columns` with its result.
2. `u8_plus.csv` / `u8_minus.csv` date from 2026-11-10 (`u8_run.py`, not read here); the base-equality check in `served_path_level.py` says whether they still match the active model.

## Open
- The counted flag set moves with the live record, so a 2020-25 level is comparable only within one fit; the division flag entered on 3-0. Whether research harnesses should freeze that set is the orchestrator's call.
- No recorded cell shown wrong: the hook cells are gains (hook minus stage 1), .00120 log loss at both levels. Registry not opened.
