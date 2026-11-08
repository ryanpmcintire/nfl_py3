# MOD-55 move anchor at pool line

Goal: test whether the line-move input should be anchored at the Tuesday-noon pool-line capture instead of the Wednesday-onward per-book diff.

State: Measured. Fit move comes from artifacts/sunday_market_probability/20260920_fixed/market_move.parquet (pick_probability_fit.py:151, 361), built by scripts/sunday_market_probability_eval.py sunday_move (Monday window, diffs counted only from Wednesday on, first-in-window quote as prior; cutoff Sun 12:45 ET or kickoff). Serve uses the same sharp_book_movement_features logic (pick_probability.py:734, sharp_book_movement_features.py:136-140), cutoff kickoff/now. Same anchor, no skew in anchor; only the cutoff differs (12:45 vs kickoff). Both exclude Tue-noon..Wed moves.
Q2 (2026 wk1-9, 136 games, deadline cutoff): differ >=0.5 in 11, >=1.0 in 3. Wk5 has leader_books=0 so served move 0 while Tuesday anchor gives up to 1.5 (CHI_GB, HOU_TEN). Top: CLE_NO served 2.5 vs Tue-anchored 1.0.
Q3: only 2023-25 supports alt (hourly quotes absent before 2023; move unavailable 2020-22 in both arms). Held-out 2023-25 (799 games) LL current 0.68348 vs Tuesday 0.68349 (diff -1.3e-5, 95% [-0.0026,+0.0025], p_pos 0.497); Brier diff -5.5e-5 (p_pos 0.469); accuracy -0.25 pts (p_pos 0.394); decisive games 50, 26 vs 24; weekly top pick 32-54 vs 33-54. Move coef per fold current 0.19-0.24, Tuesday 0.17-0.23.
Recorded: mod55_move_anchor_tuesday_noon_vs_wednesday, unresolved_below_power.
Tried: 2 arms (current, Tuesday-noon first quote, leader median of last-first). Scripts tests/scratch/mod55_q2.py, mod55_q3.py, mod55_fit.py.
Next: the anchor question is not the lever; the CLE_NO case stems from the 2 leader book coverage and a median over few books.
Open: wk5 leader_books=0 in serve data (why).
