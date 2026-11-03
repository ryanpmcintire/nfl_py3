# MOD-40 Next Gen Stats process gap

## Goal
Next Gen Stats process gap (completion over expected, rush yards over expected vs actual) as a fitted term at the opener (ROADMAP MOD-40). Family: mod40_ngs_process_gap.

## State
Unit 1 DONE 2026-11-03 (measured). Data: data/raw/ngs/20261001T173403Z; features reuse data/processed/game_features_ngs.parquet (pre-game, shrunk, leak-checked in acc-u3-ngs). Scratch scripts: scripts/mod40_ngs_reliability.py, mod40_fit.py (ignored). Batch: artifacts/mod40/weak_signals_batch.json.
Reliability (odd/even REG weeks, team-seasons, season-demeaned, season-shuffle permutation null 5000): CPOE r=0.418, Spearman-Brown 0.59, perm p<0.0002, 10/10 seasons positive (n=320). RYOE/att r=0.175, SB 0.30, perm p=0.003, 7/8 seasons positive (n=256, 2018+). Gates cleared; the gap is a stable team trait, not noise.
Opener LOSO 2020-25, 1503 decisive games, opener probability as input. Decisive record: intercept-only 745-758, served 800-703, +CPOE gap 802-701, +RYOE gap 796-707, +both 787-716, +sum 797-706. Log loss gain vs served: CPOE +0.00004 (P+ 0.499), RYOE -0.00002 (0.496), both -0.00006 (0.473), sum +0.00024 (P+ 0.590, CI -0.0017..0.0022, 4/6 seasons). Served vs intercept +0.0011 (P+ 0.80). Coefficient on expected-minus-actual completion negative 6/6 folds (-0.05..-0.10 per sd): the gap persists as skill, opposite to regression.
Looks: 2 reliability + 6 arms (4 recorded as cells) = 8. Recorded 4 cells unresolved_below_power.

## Tried
Regression-to-process hypothesis: wrong sign in point estimate but interval spans zero; not closed (no resolved wrong sign, split-half reliable).

## Next
Optional: orthogonalise against MOD-34 and MOD-36 noise floor; add receiving separation and time to throw as a declared family.

## Open
Time to throw and separation were not tested (no result counterpart declared). 80 games lack Tuesday opener.
