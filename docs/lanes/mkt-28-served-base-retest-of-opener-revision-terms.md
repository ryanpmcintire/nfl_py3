# MKT-28

## Goal
Served-base re-test (base = out_of_season_home_probability, artifacts/pick_probability/20261103T195411Z) of raw-base add-on terms. Family mkt28_served_base_retest.

## State
Unit 1 run 2026-11-03 17:35-17:50. Patched copies and logs in tests/scratch/mkt28/ (served.py swaps home_cover_probability_at_open for the served column; run with PYTHONPATH=F:/Repos/nfl_py3/scripts).
Bases (read): MKT-24 line 63, MKT-25 lines 103 and 135, PER-15 line 224, SIM-17 line 260, SIM-18 line 505 all raw home_cover_probability_at_open. MKT-23 unit 2 already served (lines 55-79), no rerun. XLG-14 is CFB, no served base, skipped. MKT-24 SBR 2011-19 proxy has no served column, not rerun.
Recorded (weak-signals, unresolved_below_power): MKT-24 1 cell, MKT-25 unit1 3 cells, MKT-25 unit2 8 cells, SIM-17 3 cells.
Recorded 2026-11-03 17:42: PER-15 2 cells (batch_per15.json) and SIM-18 3 cells (batch_sim18.json), all unresolved_below_power (sec_adj and sec_raw CI below zero but coef sign 5/6 consistent, not resolved). Registry family file now 20 signals.

## Tried
Old vs new dLL vs refit-served arm (interval; P+): MKT-24 slot -.00088 -> -.00110 (season CI -.0033,.0003; P+ .15); flips slot 77-114 -> 35-58. MKT-25 R_raw -.00029 -> -.00036 (P+ .25; flips 71-58 old -> 36-46 new); e_cm -.0001 -> -.00011; e_full -.00007 -> -.00007. MKT-25 u2 all within .0001 except w8 -.00042 -> -.0005 (P+ .031), w12 -.00065 -> -.00075 (P+ .005). PER-15 x_value -.00007 -> -.00000 (P+ .48; flips 78-76 -> 41-49); x_unweighted -.00115 -> -.00124. SIM-17 f_tot +.00034 -> +.00030 (P+ .60; flips 113-131 -> 87-92); f_off +.00004 -> +.00006; f_def -.00058 -> -.00061. SIM-18 sec_adj -.00109 -> -.00111 (P+ .013); run_adj +.00009 -> +.00030 (P+ .59; coef negative 6/6 folds); sec_raw -.00115 -> -.00109.
Intercept-only control: served beats intercept by .0064 (was .0011 on raw). Served-logit fold coef now .82-1.00 (was .27-.45).

## Next
ROADMAP note for MKT-28 unit 1. Unit 2 candidate: none needed; no conclusion changed.

## Open
No conclusion changed: all dLL differences under .0003 and same signs; only decisive flip records shrank. The refit served arm in these scripts already rescaled raw z, which is why dLL barely moved.
