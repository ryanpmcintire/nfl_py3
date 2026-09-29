# LEAD-71 unit 1: local Tuesday-open market inventory

**Measured:** 1602/1693 games (94.62%) have valid paired h2h, spread and total quotes in the same snapshot.
**Measured:** SOURCE GATE MET: a separately predeclared unit 2 may proceed. No model was fitted.
**Read:** ROADMAP.md:854 requires strictly greater than 80% coverage in 2020–2025 before unit 2.

## Reproduction and scope
Command: `.tools/uv.exe run --no-sync python scripts/lead71_unit1.py --archive data/market/raw --features data/processed/game_features.parquet --report docs/lead71_unit1_inventory.md`.
**Measured:** inspected 8834 local archive manifests; population: CON: 12, DIV: 24, REG: 1615, SB: 6, WC: 36.
Only schedule identifiers/timing and raw market quotes were read; no final scores, margins or picks.
**Read:** `src/nfl_ats/odds_backfill.py:40` defines tue_open as Tuesday 09:00 America/New_York.
The canonical season/week cutoff comes from that module's schedule planner.
Historical response hashes and source timestamps must match their manifests; snapshots and quote
updates must precede the cutoff and kickoff. Games must match teams and kickoff within 12 hours.
Paired quotes require both home/away or over/under prices, opposing spread lines and equal totals.
All-three coverage permits different books within one snapshot; same-book coverage is also shown.
The denominator includes every locally scheduled 2020–2025 game, including games without quotes.
Coverage is an exact local source census, not a sampled performance estimate; no confidence interval applies.

## Coverage by season

| Season | Games | Snapshots | Weeks | h2h | Spread | Total | All three | Same book | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 269 | 21 | 21 | 239 | 239 | 228 | 228 | 227 | 84.76% |
| 2021 | 285 | 22 | 22 | 252 | 252 | 252 | 252 | 252 | 88.42% |
| 2022 | 284 | 22 | 22 | 268 | 268 | 268 | 268 | 268 | 94.37% |
| 2023 | 285 | 22 | 22 | 284 | 284 | 284 | 284 | 284 | 99.65% |
| 2024 | 285 | 22 | 22 | 285 | 285 | 285 | 285 | 285 | 100.00% |
| 2025 | 285 | 22 | 22 | 285 | 285 | 285 | 285 | 285 | 100.00% |

## Inventory diagnostics

**Measured:** 131 target Tuesday-open manifests; 21762 matching pregame raw h2h market blocks; 0 archive read/integrity errors.
- **Measured:** event belongs to another scheduled week: 882.
- **Measured:** event lacks a unique schedule match: 79.

## Research status

Predictive looks: **0** executed; the roadmap reserves **2** for unit 2.
IS/OOS performance, their gap, per-fold coefficients, probability_positive and decisive-game
record: **not estimated**. There were no fits, LOSO scores, outcome reads or prediction rows.
This inventory establishes source availability only, with no claim of predictive gain or its absence.
Zero crossing never closes a signal. A later model must select sides through one fitted calibrated probability.
No registry write was performed; a source census cannot supply an effect estimate for a signal record.
