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
