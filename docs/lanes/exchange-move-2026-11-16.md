# Exchange-confirmed line move 2026-11-16

## Goal
Second, independent measurement of the served line-move term (leader-book median, half-point quantised): the Kalshi/Polymarket win-price change from Tuesday noon ET to each game's pick deadline, mapped to points, as one fitted term on the served probability. Research only; serves nothing. Goal lane: goal-65-2026-11-16.md.

## Declaration (2026-11-16T22:29:02Z, from `date -u`, before any fit or outcome join)
- Mapping as prediction_market_gap.py: L*(p) = -quantile(r, 1 - p), r = result - spread_line, REG seasons before 2024.
- Venue move = L*(deadline price) - L*(Tuesday-noon price), same venue, home-positive points; price = latest point in (T - 1 h, T]; deadline via `game_deadlines`. Term x = venue mean; games lacking a venue pair are dropped.
- Arm beside: offset logit(served out-of-season home probability). Arm replace: offset minus held-out fold move coefficient times the book move (2026: active coefficients.json 0.2204). Label home cover vs Tuesday opener, pushes dropped. Fit 2024-2025, frozen; LOSO over the 2 seasons; scored once on 2026 W1-10 (published card; book move recomputed at each deadline); forward W11-18.
- Looks: 2 arms + 1 zero-book-move diagnostic = 3. Family exchange_move.

## State (2026-11-16T22:34:47Z, measured)
- scripts/exchange_move.py: coverage, fit (refuses refit), score2026 (refuses rerun), pergame, pregame [--week] [--dry], score, argv. Artifacts in artifacts/exchange_move/ (frozen.json, fit_2024_2025.json, score_2026_w1_10.json, per_game.parquet, zero_book_direction.json, record_batch.json).
- Coverage: 2024 208/266 (Polymarket only), 2025 267/267, 2026 W1-10 145/145. Corr with book move 0.745 (2024-25), 0.846 (2026); signs agree 83-85% where book moved; book move zero on 31-38% of games.
- Frozen: beside 0.0322/pt (SE 0.070; folds 0.054/0.003); replace 0.1503/pt (SE 0.069; folds 0.173/0.120).
- LOSO 2024-25 (475): beside gain -0.00021 [-0.0022, 0.0019] P+ 0.42, changed 4-5, record 255-220 v served 256-219; replace -0.0020 [-0.0087, 0.0051] P+ 0.29, changed 35-34, 257-218; replace in-sample gain also negative (-0.0015).
- 2026 W1-10 once (140): beside -0.00002 [-0.0031, 0.0029] P+ 0.51, changed 0-1, 74-66 v 75-65; replace -0.00058 [-0.0108, 0.0094] P+ 0.46, changed 9-7, 77-63.
- Zero book move, nonzero exchange move, side the exchange moved toward: 2024-25 59-56 (+1.3 pp [-6.5, 9.1] P+ 0.63); 2026 25-17 (+9.5 pp [-4.8, 21.5] P+ 0.89); beside arm changed no 2026 pick there (coefficient too small).
- Recorded 6 cells unresolved_below_power (registry/weak_signals/exchange_move.json); pool ran clean for log_loss_improvement and rate_pp.
- Dry pregame week 10 (Monday night LAC_BAL): both venues read, move 0.0, book -0.5; nothing written.

## Tried
- Pregame fetches both windows live (Tuesday noon and now) through prediction_market_gap.live_prices; Kalshi lists open markets only, so it works for unplayed games.

## Next
- Orchestrator: scheduler entries from `exchange_move.py argv`: pregame Thu 19:45, Sat 15:45 (when games), Sun 12:30, Sun 15:45 ET; score Tue 09:00 ET. Forward W11-18 graded by `score`.
- Candidate follow-up (not run): the zero-book-move subset as its own declared term (exchange direction where the book is flat), fitted 2024-25 only.

## Open
- 2024 has no Kalshi; fit rests on one venue for half the seasons.
- Mean x +0.30 pt (2024-25), +0.20 (2026): mapping drift from pre-2024 residuals, absorbed by no intercept.
