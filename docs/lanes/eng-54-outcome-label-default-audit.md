# ENG-54

## Goal
Outcome-label default audit: every scorer that joins players or games to outcomes and labels a miss as a negative (ROADMAP ENG-54).

## State
Unit 1 done 2026-11-03 17:25 (date): table artifacts/eng54/audit.csv, 17 joins: A name-key 3, B id-crosswalk 5, C feature default 5, D bool NaN-False 4. Measured 8 of 17; the rest listed "not measured". Scratch: tests/scratch/eng54/.
Measured: id crosswalk is safe (snap rows lacking gsis 0.0002; unmatched injury rows that played under another key 6 of 33398, 0.0002). Name-key join is not: mod33 labels 5.86% of starter-weeks absent that the id join shows played (OL 6.6%, WR1 7.1%); non-exit OL absent 11.1% -> 4.5%, absent-undesignated 7.9% -> 1.6% (exit 17.7%). Qualitative timing finding survives, baseline was inflated; recorded mod33 cells use the injury gsis join, so none replaced.
sim16 (no registry cells): id join plus exclusion of ppg-missing slots, n 1368 -> 1283: LL gain -0.0015 (P+ 0.004) -> -0.0009 (P+ 0.15), coef +0.033 -> -0.054; still no OOS gain.
coach_speak positive (accuracy +0.213 pts, P+ 0.9876) survives: mislabel 0.0002.

## Tried
Greps of fillna(0/False) on label columns, name merges, snap_counts users; ran only the joins.

## Next
Unit 2: measure the 9 "not measured" rows (mod22_unit5 name key, veteran_rest_day_tell, per15, xlg06, sky01, overlay flags); decide whether sim16 id-join version is recorded; re-key mod33 timing on ids.

## Open
Whether to record the sim16 id-join cells (population changed, so not a --replace).
