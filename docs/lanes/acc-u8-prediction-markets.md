# acc-u8-prediction-markets

## Goal
Acquire Kalshi and Polymarket NFL game and spread market prices (backfill plus forward capture) as information independent of the sportsbook opener. Parent: docs/lanes/accuracy-ceiling-theory.md. Acquisition only, no grading.

## State
scripts/mod24_u8_markets.py written (modes backfill, derive, capture, report). Capture ran once: raw data/raw/prediction_markets/20261001T182528Z-capture, 2,412 rows into data/processed/prediction_markets_snapshots.parquet.
Backfill run 20261001-backfill (log tests/scratch/u8_backfill.log) was at poly 800 of 824 events when the tool cap hit, so the final parquet write and manifest may or may not have completed. Kalshi: 702 game markets and 6,043 spread markets with volume fetched (cached under data/raw/prediction_markets/20261001-backfill/kalshi). Outputs when done: data/processed/prediction_markets_series.parquet (long), prediction_markets.parquet (per game x point freeze/deadline/kickoff), manifest.json with sha256 and unmatched counts. Coverage table and the freeze-vs-opener comparison NOT yet produced.

## Endpoints (measured)
- Kalshi, unauthenticated: /series, /markets?series_ticker=, /historical/markets (cutoff 2026-08-02), candlesticks /historical/markets/{t}/candlesticks (period 60 or 1, 5000 cap) and /series/{s}/markets/{t}/candlesticks. KXNFLGAME (one market per team), KXNFLSPREAD (ladder "team wins by over k.5", floor_strike), KXNFLTOTAL. Games exist from Aug 2025 only. Event ticker AWAYHOME has variable-length codes.
- Polymarket, unauthenticated: gamma /events?tag_slug=nfl (slug nfl-away-home-YYYY-MM-DD, sportsMarketType moneyline/spreads/totals), clob /prices-history needs startTs/endTs plus fidelity (interval=max is empty for closed markets). Games from Aug 2024. Spread ladder only on the favourite side.

## Tried
Kalshi spread-ladder implied spread = logit interpolation of P(home margin > x)=0.5 across both teams' strikes. Fixed an event-code parse bug (2-letter codes) before the run.

## Next
1. Check log tail for "done"; if absent or errored, rerun `.tools/uv.exe run --no-sync python scripts/mod24_u8_markets.py backfill --run-id 20261001-backfill` (cache makes it resumable).
2. Run `... mod24_u8_markets.py report` for the coverage table and freeze-vs-opener numbers (opener = open_close archive opening_home_spread, 2025 only), and read manifest unmatched counts.
3. Orchestrator adds scheduler job: `.tools/uv.exe run --no-sync python scripts/mod24_u8_markets.py capture`.

## Open
Pool opener comparison uses the book opener archive, not the Splash pool line (Splash only 2026). Polymarket price semantics (trade vs mid) are source-defined.

## Result 2026-10-08 (measured, tests/scratch/u8_grade.py)
Backfill done (log "done", manifest unmatched kalshi 100, poly 235, mostly Aug preseason ids). Report run: sign bug confirmed (freeze vs open corr -0.95; flip implied_home_spread to home margin). Scheduler job prediction_markets_{tue,wed,thu}_1200 exists and OK in data/scheduler_log.txt through 2026-10-08 (1,826-1,987 rows each).
Grade: y = home margin - opener (2025 open_close archive only, 2026 has no archive opener), d = freeze market spread - opener, 2025 only, one season so no LOSO.
- Kalshi n 208: beta(y on d) 0.94 pts per pt, bootstrap 95% [0.03, 1.91], probability_positive 0.978. Side-with-d record 107-91 (0.540, binomial p .29); |d|>=1: 58-40 (0.592, n 98).
- Poly n 78: beta 0.33 [-1.15, 1.94], probability_positive 0.68; side 41-34 (0.547).
- d correlates 0.78 (Kalshi) / 0.75 (Poly) with open-to-close move: the freeze price is largely the early line move, not independent information.
Looks: 2 sources x (beta, side, 3 thresholds) = 10, family prediction_market_freeze_vs_opener.
Draft (not run): nfl-ats weak-signals record --name u8_kalshi_freeze_vs_opener --source tests/scratch/u8_grade.py --effect 0.94 --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2025 --season-end 2025 --standard-error 0.49 --interval-low 0.03 --interval-high 1.91 --probability-positive 0.978 --sample-games 208 --family prediction_market_freeze_vs_opener --category market (poly: effect 0.33, [-1.15,1.94], pp 0.68, n 78). Units note: beta is margin points per spread point, not strictly ats_points; orchestrator confirm.
Recorded 2026-10-08 by orchestrator (rerun reproduced every number) in accuracy_points, not the slope: Kalshi +4.04 [-2.92,+11.00] P+ .872 n198; Polymarket +4.67 [-6.65,+15.98] P+ .791 n75; both unresolved_below_power, slopes in the descriptions. Report sign fixed (scripts/mod24_u8_markets.py:536): freeze vs open corr .949 Kalshi, .962 Poly.
Next: (2) add 2026 weeks once a Splash pool opener is joined (Splash-line keyed, game_features_weak_stack_splash); (3) fit d as a term inside the calibrated probability LOSO once >1 season.
