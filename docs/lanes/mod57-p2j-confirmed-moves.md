# mod57-p2j confirmed moves

## Goal
Give the served pick probability the rest of the market's deadline line: a consensus move (all-books median of each book's last quote strictly before the game's pick deadline, minus tue_open_home_spread) so the fit can tell a market-wide move from a leader-only move, and so 2020-22 get a move input.

## State
Declared 2026-11-11 09:41 (date stamp) before any outcome read.
- Consensus: median over >=5 books of each book's last quote strictly before deadline = min(kickoff, Sunday 16:00 ET of the week, pick_refresh.sunday_pick_lock). 2023-25 from artifacts/experiments/sharp_book_movement/quotes.parquet (hourly archive cache, spreads); 2020-22 from the historical decision snapshots (labels feeding close_home_spread). Games with no admissible quote set (>=5 books, all before deadline) get consensus unavailable, never a post-deadline line. Per-game snapshot time vs deadline printed to tests/scratch/mod57/p2j/times.csv; per-season coverage and lead time reported.
- consensus_move = consensus line - tue_open_home_spread, positive = home more favoured (sign checked against market_move_toward_home).
- Arm J1 (both): model_logit, composition_flag_sum, market_move_toward_home, market_move_available, consensus_move, consensus_available. Arm J2 (replace): model_logit, composition_flag_sum, consensus_move, consensus_available.
- Baselines: served four-term (must reproduce 829-674), model only, intercept-only, consensus-move-only control (consensus_move, consensus_available).
- Harness: lead98_price_skew loso/fit (served ridge), 1,503 games, week-block bootstrap (mod57_u3). Reports: decisive games first vs four-term, 2020-22 and 2023-25 separately, fold coefficients and sign of leader coefficient in J1, in-sample beside OOS, reliability.
- 2026 read: frozen 2020-25 coefficients; Weeks 3-9 consensus from tests/scratch/mod57/p2e payloads at last scan before deadline; swap only stage-1 logit in the u5 harness.
- Looks: 12 (2 arms x 3 metrics x {2020-25, 2026}); family mod57_confirmed_moves.
- Script scripts/mod57_p2j_confirmed_moves.py; output tests/scratch/mod57/p2j/.

## Tried
Ran once 2026-11-11 09:45 (tests/scratch/mod57/p2j/run.out, results_history.json, results_2026.json; scripts/mod57_p2j_confirmed_moves.py + mod57_p2j_consensus.py). 12 cells recorded unresolved_below_power, family mod57_confirmed_moves.
Coverage (measured): all six seasons 100 percent admissible (>=5 books, all before deadline); 2020-22 newest quote 35 min before deadline (Sunday snapshot), 2023-25 median ~125 min (min 4). Sign: corr consensus_move vs leader move .86 (799 games), same sign 96 percent; vs open_move .97.
2020-25 (1,503): four-term 829-674 (reproduced), J1 820-683 (acc -0.61 [-2.28,+1.01] P+ .22; ll P+ .82; Brier P+ .81; decisive 185 changed, 88-97), J2 843-660 (+0.92 [-1.06,+2.87] P+ .81; ll P+ .64; decisive 268 changed, 141-127). 2020-22: J1 -1.15 P+ .20, J2 +0.40 P+ .58. 2023-25: J1 -0.12 P+ .43, J2 +1.38 [-0.74,+3.42] P+ .89. Controls: consensus-only 808-695, model only 800-703, intercept 745-758.
J1 leader coefficient positive in 6 of 6 folds (0.077-0.140), consensus 0.095-0.145: leader move keeps weight beyond the consensus.
2026 Wk3-9 (104): current 59-45, J1 56-48 (changed 5, 1-4; acc -2.92 P+ .00), J2 55-49 (changed 8, 2-6; acc -3.86 P+ .00); log loss improvements +-.001 P+ .68/.75.

## Next
Not served. Consensus move gives 2020-22 a move input but no resolved gain; 2026 changed picks lose. Untested: horizon mismatch (2020-22 consensus 35 min before deadline, 2023-25 about 2 h).

## Open
None.
