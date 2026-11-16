# Prediction-market gap (Kalshi, Polymarket) 2026-11-16

## Goal
Test whether the gap between prediction-market win prices and the sportsbook line at the same timestamp carries information the pool's frozen Tuesday line lacks, as one fitted term on the served probability. Research only; serves nothing.

## Declaration (2026-11-16T21:03:10Z, before any outcome join)
- Mapping (repo's own): `MarginModel` win probability = share of (center + market residuals) above 0. Here center = sportsbook home line L, residuals r = result - spread_line over REG games of seasons before 2024 (game_features). Spread-equivalent of win price p is L* = -quantile(r, 1 - p).
- Term: gap = L*(PM home win price) - sportsbook home line, points, home minus away. Two timestamps: Tuesday noon ET of game week (book = Tuesday opener, the pool line) and the pick deadline min(kickoff, Sun 4 PM ET) via `game_deadlines` (book = close line). Price = latest point in (T - 1 h, T]; Kalshi traded `price_close` only (no quote mids); Polymarket CLOB point. Venue-combined term = mean of available venues; venue-only terms are diagnostics.
- Fit: one logistic coefficient, offset logit(served out-of-season home probability, active pick_probability per_game), label home cover vs Tuesday opener, pushes dropped, fitted on 2024-2025, then frozen. Also leave-one-season-out within 2024-2025 for in-sample vs held-out gap.
- Grade: changed-pick record and served-side record, mean log loss vs served, week-block bootstrap probability_positive; scored once on 2026 W1-10 (published card probability) and forward W11-18.
- Looks: 2 primary (tue, deadline) + 4 venue diagnostics = 6.

## State (2026-11-16, measured)
- scripts/prediction_market_gap.py: coverage, fit (writes frozen.json, refuses refit), score2026 (refuses rerun), pregame --slot tue|deadline [--dry], score, argv. Artifacts in artifacts/prediction_market_gap/.
- Coverage (price within 1 h): 2024 tue 208/266 (PM only), deadline 257; 2025 tue 267 (K 258, PM 262), deadline 267; 2026 W1-10 tue 107/145, deadline 100 (PM only). Mean gap +0.37 tue, +0.42 deadline, SD about 1.0 pt.
- Frozen: tue 0.0411 logit/pt (SE 0.086; folds -0.017/+0.147), deadline -0.0283 (SE 0.082; folds -0.081/+0.044). LOSO log-loss gain tue -0.0027 [-0.0081, 0.0028] P+ 0.19; deadline -0.0015 [-0.0040, 0.0011] P+ 0.14.
- 2026 W1-10 once: tue 102 games, served 61-41, term 60-42, changed 4-5, gain -0.0028 [-0.0065, 0.0009] P+ 0.065; deadline 95 games, served 57-38, term 59-36, changed 3-1, gain -0.0001 [-0.0036, 0.0038] P+ 0.49.
- Recorded 4 cells unresolved_below_power, family prediction_market_gap (registry/weak_signals/prediction_market_gap.json); pool ran clean.
- Dry pregame week 11 tue (Monday): 26 live prices, 0 games (no Tuesday opener or card yet), expected.

## Tried
- game_deadlines must be called per week (whole-season call gives one Sunday lock); fixed in stamps().
- Live Kalshi candles use price.close_dollars and volume_fp; Polymarket open events need tag_slug=nfl (series_id 10187 returns none).

## Next
- Orchestrator: add scheduler entries (argv from `prediction_market_gap.py argv`): pregame --slot tue Tue 12:30 ET; pregame --slot deadline Thu 19:45, Sat 15:45 (when games), Sun 12:30, Sun 15:45 ET; score Tue 09:00 ET.
- Forward W11-18 graded by `score`.

## Open
- Kalshi 2026 hourly file has no prices (source pull parsed close, API now close_dollars): re-pull would add Kalshi to 2026 scoring (source-pull lane).
- Mapping residuals are pre-2024; mean gap +0.4 pt may be mapping drift, not venue view.
