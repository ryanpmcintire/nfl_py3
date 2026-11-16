# move-book-count-2026-11-16

## Goal
Does the served line-move term's reliability depend on how many leader books had a Tuesday baseline? Mechanism: one book's move is noisier than a consensus, so its fitted slope should be smaller. Research only; no src edit, no serve change.

## Declaration (stamped 2026-11-16 20:39:37 UTC, before any outcome read)
- Fit population (measured before outcomes): artifacts/sunday_market_probability/20260920_fixed/market_move.parquet, the served stage's move input, has leader_books = 3 on 799 of 799 games (sunday_books 3 on 799). Commit 52555216 message (read): under current code 7 of 2,397 book-game pairs lack a baseline, 2 of 799 values change. So test (a), slope split 1 vs 2+, has 0 one-book games in 2023-25 and cannot be read. Count recomputed by the script from the source quotes under current code (secondary, reported).
- Fallback primary: 2026 Weeks 1-10. Per game, last served capture before kickoff (artifacts/prospective/independent_validation*/captures). Book count = leader_books from sharp_book_movement_features at that capture time on today's live store with the late-written *-odds-gap-private-reparse files excluded; agreement of the replayed move with the captured served move is reported, mismatches flagged and kept on the served move. Primary cell: games with nonzero served move on exactly 1 leader book; record of the side the move points to, graded at the capture's served spread line, pushes counted apart; exact binomial v even, Beta(1,1) posterior P(rate>.5) as probability_positive.
- Contrast cell: same metric on nonzero moves over 2-3 books; difference 1-book minus 2+ in accuracy points with a matched null (book-count label permuted within week, 2000 draws), percentile reported.
- Secondary: games where the move terms (move + availability, coefficient set in force at the capture) decide the served side (side without them differs): record of the served side, 1 book v 2+.
- Looks: 5 (primary, contrast, difference, two changed-side cells), one family move_book_count_2026_11_16, effect_units accuracy_points. An interval crossing zero closes nothing; only a resolved wrong sign or a positive control closes.

## State (2026-11-16 20:42 UTC, done; nothing served touched)
- scripts/move_book_count.py -> artifacts/move_book_count/ (games.csv, summary.json, record_batch.json). Ran once, exit 0, single process; ruff format + check clean after. Imports scripts/flip_attribution_audit.py and the untracked tests/scratch/serve_time_move/serve_time_move.py (store loader, old-code replay).
- (a) unreadable (measured): fit population 799 of 799 games on 3 leader books, 0 one-book, 0 two-book. Current-code recount from source quotes not run; commit 52555216 message (read) says 7 of 2,397 pairs lack a baseline.
- 2026 coverage (measured): validation captures exist Weeks 4-10 only (100 games); Week 5 had no move input. Replay equals the served move on 98 of 100 (misses 2026_08_TEN_CIN, 2026_08_LV_NYJ, labelled by the current-code replay). Books by week: 1 book W4 11, W6 8, W7 9, W8 12, W9 1, W10 1; 2 books W9 14, W10 2; 3 books W4 5, W6 5, W7 5, W8 1, W10 11.
- Results (measured, pool Tuesday line, 0 pushes): one-book nonzero moves, move side 14-10 (+8.33 pts, [-13.4, +27.9], P+ .79, p .54); 2-3 books 13-14 (-1.85, [-21.3, +18.1], P+ .43). Difference +10.2 pts, week-block [-29.2, +47.7], P+ .66, matched-null percentile .65 (2000 draws). Move decides the served side: one book 4-4, 2-3 books 4-3. Mechanism (one book worse) not supported; sign opposite, unresolved_below_power.
- Registry: 5 cells, family move_book_count_2026_11_16, accuracy_points, unresolved_below_power, record exit 0; pool ran once exit 0 (4,648 nfl accuracy_points signals).
## Tried

## Next
1. Orchestrator verifies, commits script, artifact pointers, registry shards, lane; move lane to done/.
2. If wanted: recompute the 2023-25 book count under current code from artifacts/sharp_book_weighted_movement/spread_quotes.parquet (expected at most 7 two-book games), still unreadable for a slope split.

## Open
- Weeks 1-3 have no captures; their served moves would need the ledger or feeds.
- The 2026 test grades the move's side, not a LOSO slope; the fitted slope cannot be split by depth until history has thin-depth games.
