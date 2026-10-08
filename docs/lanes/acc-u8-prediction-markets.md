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

## 2026 W1-4 + LOSO (measured 2026-10-08, tests/scratch/u8_grade26.py)
Opener 2026 = Splash locked line (data/splash/2026_weekNN_*.json home_spread, nflverse sign); freeze = Tuesday noon ET, before every deadline, ml_last_ts<=freeze on 100% rows. Qualifying games with opener+result: Kalshi 64 (2025: 208), Poly 58 (2025: 78). Sign flip by corr: -0.995/-0.994 in 2026.
Caveat: sd d 2026 0.51 vs 2025 1.94 (2025 opener is the early book open, 2026 is the Tuesday-noon line = freeze time), so the seasons measure different things; only 33 (Kalshi) / 32 (Poly) 2026 games have |d|>0.25.
Accuracy points (side of d vs 50%, decisive games): Kalshi 2026 -2.46 [-13.9,+10.7] P+ .356 (29-32); 2025 +4.04 P+ .856; pooled +2.51 [-3.7,+8.7] P+ .770 (136-123). Poly 2026 -5.17 [-19.0,+6.9] P+ .169 (26-32); pooled +0.38 [-7.9,+8.7] P+ .520 (67-66).
Slope beta (y on d): Kalshi 2026 0.59 [-4.7,10.7]; pooled 0.93 [0.04,1.84]. Poly 2026 -0.02; pooled 0.27.
LOSO logistic (z = home beats opener): Kalshi fit25->score26 slope .126, logloss .6958 vs const .6931, Brier .2513 vs .2500, acc_pts +3.12; fit26->score25 slope -.274, logloss .7571 vs .6931, acc -4.41. Poly fit25->26 slope .061, ll .6933 vs .6926, acc +1.72; fit26->25 slope -.144, ll .7098 vs .6929, acc -1.28. Slope sign flips across folds; held-out log loss never beats constant. No closing line for 2026, so side-of-market-move baseline not computed.
Looks: +2 sources x (2026, pooled, 2 folds) = 8 new; family prediction_market_freeze_vs_opener now 18.
Draft (not run), accuracy_points, unresolved_below_power: nfl-ats weak-signals record --name u8_kalshi_freeze_vs_opener_2026 --effect-units accuracy_points --effect -2.46 --interval-low -13.93 --interval-high 10.66 --probability-positive 0.356 --sample-games 61 --season-start 2026 --season-end 2026 --family prediction_market_freeze_vs_opener; Poly: --effect -5.17 --interval-low -18.97 --interval-high 6.90 --probability-positive 0.169 --sample-games 58 (decisive 58); pooled Kalshi +2.51 [-3.67,8.69] P+ .770 n259; pooled Poly +0.38 [-7.89,8.65] P+ .520 n133. Match other flags to the existing 2025 records.
Next: wait for more 2026 weeks; need the 2025 early-open vs Tuesday-line distinction resolved before pooling slopes; fitting d inside the calibrated probability needs an opener-consistent definition.

Recorded 2026-10-08 (orchestrator reran u8_grade26.py, numbers reproduced): 2026-alone Kalshi -2.46 and Polymarket -5.17 accuracy points, unresolved_below_power. Pooled 2025+2026 cells NOT recorded: the two openers differ (early book open v Tuesday-noon Splash line), so d is not commensurable. Next: rebuild 2025 d against a Tuesday-noon book line (same definition as 2026) before any pooled or LOSO fit; add 2026 weeks as they settle.

## 2025 d rebuilt vs Tuesday-noon book line (measured 2026-10-08, tests/scratch/u8_grade_tue.py)
Opener 2025 = median over books of last quote with observed, snapshot <= freeze (point_utc), book update <= observed, within 36h of freeze (week start), 12 books in artifacts/sharp_book_weighted_movement/spread_quotes.parquet. Coverage 224 games (median 11 books, cross-book SD 0.22); early open v Tuesday line corr .949, SD diff 1.93. Kalshi 216/216 market games, Poly 84/84 matched. SD d now 2025 0.32 / 2026 0.51 (Kalshi), 0.36 / 0.52 (Poly): commensurable.
Accuracy points (side of d, decisive): Kalshi 2025 -1.06 [-7.98,+5.85] P+ .350 (92-96); pooled -1.41 [-7.83,+4.62] P+ .340 (121-128). Poly 2025 +14.18 [+3.73,+26.12] P+ .991 (43-24); pooled +5.20 [-3.60,+13.22] P+ .887 (69-56). Slopes beta pooled Kalshi -0.52, Poly -1.45 (2025 Poly beta -4.41, P+ .20: side and slope disagree).
LOSO logistic Kalshi: fit25->26 slope -.119 ll .6917 v const .6932 Brier .2493 v .2500; fit26->25 slope -.274 ll .6936 v .6931. Poly: fit25->26 slope -.002 ll .6937 v .6937; fit26->25 slope -.144 ll .6949 v .6942. Slope negative in all four folds; held-out gain only Kalshi fit25->26 (n64).
Looks: +8 (2 sources x 2025, pooled, 2 folds); family prediction_market_freeze_vs_opener now 26.
Draft (not run), accuracy_points, unresolved_below_power, match other flags to prior records, name suffix _tue: kalshi 2025 --effect -1.06 --interval-low -7.98 --interval-high 5.85 --probability-positive 0.350 --sample-games 188; kalshi pooled -1.41 [-7.83,4.62] 0.340 n249; poly 2025 +14.18 [3.73,26.12] 0.991 n67; poly pooled +5.20 [-3.60,13.22] 0.887 n125 (all --effect-units accuracy_points --family prediction_market_freeze_vs_opener --league nfl). Poly 2025 is 1 of 4 cells; do not headline.
Next: refresh with 2026 weeks 5+ as they settle; fit d inside the calibrated probability only with the Tuesday definition.
Recorded 2026-10-08 (orchestrator reran u8_grade_tue.py, reproduced): pooled Tuesday-line cells Kalshi -1.41 [-7.83,+4.62] P+ .340 and Polymarket +5.20 [-3.60,+13.22] P+ .887, unresolved_below_power. Single-season _tue cells not recorded separately (in the descriptions). Next: refresh the pooled cells each few 2026 weeks; fit d as a term in the calibrated probability only after more 2026 weeks.
