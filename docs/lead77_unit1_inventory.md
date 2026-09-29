# LEAD-77 unit 1 - local quote timestamp inventory

**Measured:** `timestamp_audit_required`. Only schemas, capture metadata and quote clocks were read; no game outcome or price-response target was read or calculated.

**Read:** ROADMAP.md:860 and docs/lanes/ideation-2026-09-29b.md:11-25 define the 2023-2025 NFL opener games and NFL/CFB quote panels. The declaration was copied to
docs/lanes/lead77.md before execution. The partially pooled response remains the declared challenger.

## Local source coverage

**Measured:** data root `data`; recursive Parquet metadata inventory covers `market/`, `cfb/` and `processed/sbr_odds.parquet`. Counts include out-of-population seasons. Quote
panels require event, book, market, side, price, line, observation and event-time fields. File/row counts are not independent games.

| Source | Present | Parquet files | Rows | Quote files | Quote rows | Quote files with ingestion field |
| --- | --- | --- | --- | --- | --- | --- |
| market archive | True | 8834 | 6207318 | 8832 | 6189933 | 9 |
| college archive | True | 395 | 18479746 | 0 | 0 | 0 |
| legacy SBR | True | 1 | 4025 | 0 | 0 | 0 |

## Target-season quote clocks

**Measured:** event dates assign football seasons (January-June use the preceding year). These are inventory labels, not folds. Provider event IDs are not certified opener-game
counts.

| League | Season | Files | Rows | Event IDs | Events with 2+ observations | Provider-clock rows | Nonnegative-age rows | Pregame rows | Files with ingestion field |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NFL | 2023 | 2548 | 1816750 | 492 | 488 | 1816750 | 1816750 | 1809856 | 0 |
| NFL | 2024 | 2548 | 1193130 | 308 | 307 | 1193130 | 1193130 | 1188564 | 0 |
| NFL | 2025 | 2548 | 1411344 | 285 | 285 | 1411344 | 1411344 | 1405704 | 0 |
| CFB | 2023 | 80 | 182594 | 912 | 791 | 182594 | 182594 | 182324 | 0 |
| CFB | 2024 | 239 | 223418 | 948 | 934 | 223418 | 223418 | 217064 | 0 |
| CFB | 2025 | 138 | 203940 | 959 | 941 | 203940 | 203940 | 193934 | 0 |

**Measured:** cadence uses unique observations per event/book, pooling market/side duplicates. It does not estimate synchronized-consensus response.

| League | Season | Cadence intervals | Median seconds | Min seconds | Max seconds |
| --- | --- | --- | --- | --- | --- |
| NFL | 2023 | 843693 | 3600.0 | 3000.0 | 9754197.0 |
| NFL | 2024 | 551849 | 3600.0 | 3594.0 | 10987200.0 |
| NFL | 2025 | 656008 | 3600.0 | 3597.0 | 10285198.0 |
| CFB | 2023 | 35996 | 86400.0 | 61197.0 | 6908401.0 |
| CFB | 2024 | 58155 | 28799.0 | 3598.0 | 8409598.0 |
| CFB | 2025 | 40065 | 61200.0 | 14398.0 | 18363601.0 |

## Source limitations

- **Measured:** market archive: 7 schema variants; relevant field names: bookmaker_key, bookmaker_last_update_utc, bookmaker_title, capture_kind, closing_books, closing_spread_max,
closing_spread_min, closing_spread_std, commence_time_utc, consensus_closing_home_spread, game_date, home_spread_line, inconsistent_closing_books, is_timestamped_quote, line,
market, market_last_update_utc, nflverse_spread_line, observed_at_utc, opening_home_spread, outcome_side, provider, provider_event_id, quote_timestamp_basis,
reported_closing_books, reported_opening_home_spread, season, source_scan_at_utc, sport_key, spread_moneyline_direction_consistent. Manifest ingestion names: retrieved_at_utc.
- **Measured:** market archive: 0 quote rows without event dates; 1779 quote rows with unrecognized sport keys.
- **Measured:** college archive: 28 schema variants; relevant field names: awayTimeoutCalled, book, date_of_birth, date_time, draft_year, drive.end.yardLine, drive.start.yardLine,
drive.timeElapsed.displayValue, end.ExpScoreDiff_Time_Ratio, end.TimeSecsRem, end.adj_TimeSecsRem, end.awayTeamTimeouts, end.defPosTeamTimeouts, end.homeTeamTimeouts,
end.posTeamTimeouts, end.pos_team_spread, end.spread_time, end.yardLine, experience_years, gameSpread, gameSpreadAvailable, game_spread, game_spread_available, homeTeamSpread,
homeTimeoutCalled, home_team_spread, line_yards, lines, odds, odds_source, opening_lines, opening_odds, penalty_declined, season, seasonType, season_type,
start.ExpScoreDiff_Time_Ratio, start.ExpScoreDiff_Time_Ratio_touchback, start.TimeSecsRem, start.adj_TimeSecsRem, start.awayTeamTimeouts, start.defPosTeamTimeouts,
start.homeTeamTimeouts, start.posTeamTimeouts, start.pos_team_spread, start.spread_time, start.yardLine, start_date, start_time_tbd, transferDate, year. Manifest ingestion names:
none.
- **Measured:** college archive: 0 quote rows without event dates; 0 quote rows with unrecognized sport keys.
- **Measured:** legacy SBR: 1 schema variants; relevant field names: away_moneyline, close_home_spread, game_date, home_moneyline, open_home_spread, sbr_date_raw, sbr_season_slug,
season, week_match_date_diff_days. Manifest ingestion names: none.
- **Measured:** legacy SBR: 0 quote rows without event dates; 0 quote rows with unrecognized sport keys.
- **Measured:** quote capture kinds: {'event_halves': 14, 'historical_backfill': 8746, 'live': 17, 'unspecified': 55}.
- **Measured:** inventory errors: 0.
- **Read:** src/nfl_ats/odds_backfill.py:242-264 uses provider snapshot time as historical observed time. This does not establish local ingestion time.
src/nfl_ats/market_data.py:309-331 stores observed time and permits extra manifest fields.
- **Inferred:** historical observed time, file modification time or pre-kickoff availability cannot substitute for issuance and ingestion before the authentic pool deadline. No
deadline join or synchronization target has been certified.

## Protocol gate

**Measured:** a separate metadata-only review found 8,104 target-season manifests;
all 8,104 declare `historical_backfill`. A recursive inspection of manifest key
names found no ingestion/retrieval/receipt/fetch/collection/download/storage/creation
time field; `capture_kind` was the only matching key. The real inventory command
found zero target-season quote files carrying one of its explicit ingestion fields.

**Inferred:** the quote panels exist, but the local metadata does not establish
historical ingestion before the authentic pool deadline. The required source
provenance is absent from the inspected artifacts. Protocol B at
`docs/lanes/ideation-2026-09-29b.md:12` requires issuance and ingestion before that
deadline, so this unit stops at inventory without estimating an effect. This is a
source-evidence gap, not a refuted mechanism or a negative research finding.

The supplemental review used `.tools/uv.exe run --no-sync python -` to inspect
manifest field names only. An earlier PowerShell variant failed because its JSON
parser lacked `-AsHashtable`; its output was discarded. No outcome data was read.
## Research reporting status

**Measured:** target quote rows NFL=4421224, CFB=609952. Missing quote-panel populations: none identified by schema. Fitted/scored games=0; consumed outcome looks=0. The declared
family remains B=4, F=3, E=1, L=217. Inventory tables are metadata diagnostics, not outcome looks.

**Measured:** decisive-game record is unavailable (no games scored), not evidence of a 0-0 record. IS/OOS accuracy, Brier, log loss, margin MAE, response MAE, gaps, five-band
reliability, season-block 95% intervals and probability_positive are unestimated. No effect estimate or closure verdict is supported.

| Held-out season | IS | OOS | Gap | Coefficients | 95% interval | probability_positive |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | NA | NA | NA | not fitted | not estimated | not estimated |
| 2024 | NA | NA | NA | not fitted | not estimated | not estimated |
| 2025 | NA | NA | NA | not fitted | not estimated | not estimated |

**Inferred:** remain at the inventory unit. Obtain documented pre-deadline NFL/CFB panels with provider clocks and ingestion provenance, then certify authentic opener/deadline
joins and enough separate earlier training, selection and calibration seasons. Keep the pooled arm fixed, exclude target/later seasons from both leagues, normalize on training only
and weight games equally. Missing earlier folds remain unavailable. Missing sources neither refute nor close the mechanism (AGENTS.md:65-85).

## Reproduction

```bash
.tools/uv.exe run --no-sync python scripts/lead77_unit1.py
```

No registry command is supplied: no effect, interval, probability_positive or research verdict was computed. A placeholder estimate would not be a valid measurement.

