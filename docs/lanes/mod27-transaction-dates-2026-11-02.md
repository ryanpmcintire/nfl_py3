# MOD-27 day-level transaction dates for the deadline flag

## Goal
Repair the month-end chronology leak in the deadline-integration research flag (LEAD-23 / acc-u7) with day-level transaction dates, assert pregame dating at runtime, rerun the declared fifth-term LOSO measurement at the opener.

## State
- DONE 2026-11-02. scripts/mod27_transaction_dates.py; outputs in artifacts/mod27_transaction_dates/ (results.json, repaired_events.csv, leaky_events.csv, run.log).
- Dates: article datePublished from data/raw/pfr_transactions/*/sample_articles (all 81 flag rows covered); lastmod fallback never used. lastmod never earlier than published (0 of 13,496), median +0.02 h.
- Runtime assertion: every transaction used is strictly before the game's Tuesday noon ET; passes for repaired flag.
- Measured leak: month-end rule counted 81 team-game flags (27 transactions); only 3 flags/games/transactions (2020, 2023, 2024) postdate the prediction timestamp at day level (earlier 15 was month-end overcounting). Repair also shifts which games are flagged: 45 decisive flagged games vs 52.
- Not served: flag is only in challenger feature sets; served path untouched.
- Results (1,503 decisive 2020-25, base 833-670): repaired five-term 827-676, -0.40 pts [-0.94, +0.07] P+ .029; 2022-25 -0.57 [-1.41, 0.00] P+ .024; changed sides 8, 1-7 vs 7-1; flagged-only 20-25 vs 26-19. In-sample acc .5516 vs OOS .5502 (gap .0013); log loss .68877 vs base .68719. Coef per fold -0.09 -0.11 -0.01 -0.09 0.00 +0.49 (sign unstable). Leaky: 831-672, -0.13 P+ .185. Repaired minus leaky -0.27 [-0.93, +0.34] P+ .18, 20 changed picks 8-12 vs 12-8.
- Recorded 3 cells (total_signals 8025) under family deadline_integration_drag_on_production, unresolved_below_power. Looks 5 (leaky repro/2022-25 recorded earlier by L23).

## Tried
- Month-end vs day-level; sitemap lastmod as fallback not needed.

## Next
- Nothing for this flag: not close-able (interval touches +0.07, no control), not servable (negative, unstable sign). Optionally restore 2020-21 events (index reproduces 2 vs registry 25).

## Open
- None.
