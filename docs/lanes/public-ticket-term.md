# Public ticket term
## Goal
Test ticket-share and money-minus-ticket as a fifth term in the served four-term fit, LOSO, opener-graded.
## State
Done. Scripts tests/scratch/public_ticket/{build,loso}.py; output result.json, batch.json, splits.parquet. Recorded 4 cells, family public_ticket_term_2026_10_27, all unresolved_below_power.
Coverage (pre-lock, capture within 168h of cutoff): ticket 255 of 1503 games (2020 26, 2021 58, 2022 32, 2023 35, 2024 73, 2025 31); money only 2023-25 (139).
Results (gain = base minus new, negative = worse): ticket allfit covered Brier -0.0022 P+ 0.003; coveredfit -0.0019 P+ 0.005; money-minus-ticket -0.0006 P+ 0.40. Majority-ticket cover 125-124, perm p 1.0.
## Tried
Missing term filled with 0, no availability flag; also covered-only refit.
## Next
Forward log of ticket splits at lock; live capture 2026 coverage (public_betting_live) is the only way to get power.
## Open
Ticket coefficient is positive (follow public) in 4 of 6 folds, opposite to the shading mechanism, unstable. Pre-2023 splits have no money.
