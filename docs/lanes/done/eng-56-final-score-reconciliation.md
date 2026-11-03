# ENG-56

## Goal
Final-score reconciliation: nflverse finals vs a second free archive for every graded game (ROADMAP ENG-56).

## State
Unit 1 done 2026-11-03 (date 5:2x PM). scripts/eng56_score_reconcile.py -> artifacts/eng56/reconciliation.csv (4217 games 2011-2026 scored, REG+post). Sources (both allowed): SBR archive (sbr_odds.parquet, 2011-2021, 2957 games) and nflverse pbp last-play scores (data/pbp/raw/20261103T172043Z, same upstream org so only partly independent). Measured: SBR 2956/2957 exact; pbp 4216/4217 exact. Diffs: 2016_04_KC_PIT (SBR away 7 vs nflverse 14; PIT 43 KC 14 is correct, SBR typo) and 2011_13_DET_NO (pbp 37-23 vs 31-17, pbp has bad rows; nflverse correct). Grade changes at nflverse close, SBR open/close, open_close opener (260 games), 2026 pool line first/last capture (121 games): 0. No cited number moves.

## Tried
SBR sign convention is same as nflverse spread_line (no negation). Pool lines from data/splash/2026_week*.json.

## Next
Openers for 2022-2026 not in the compared set beyond open_close (260) and pool (121); a Tuesday-opener rerun adds nothing since scores equal. PFR box-score source not captured on disk. Close lane.

## Open
Two-source agreement means the archive tells nothing about nflverse errors shared by both; independence limited.
