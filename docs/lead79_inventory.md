# LEAD-79 unit 1 — local source inventory

**Measured:** `source_gap`. No outcomes, fits, scores, or registry writes.
The source gate does not close or reject the proposed mechanism.

## Declared population and scope

**Read:** `ROADMAP.md:862`; `docs/lanes/lead79.md` preserves Protocol B before execution.
2020-2025 REG; same-fixture lookahead issued before either intervening game,
issuance/ingestion before the pool deadline, and an authentic frozen Tuesday pool line.
B=2, F=6, E=0: 291 planned looks; **measured:** 0 outcome looks used.

**Measured:** data root `data`; schedule `data/raw/20260929T191306Z/schedules.parquet`.
Read only schedule identity fields and market identity/timestamp metadata;
Parquet enumeration includes ignored local files. No scores or prediction rows loaded.
The scan covers every Parquet under the data root's `market/`,
`market/raw/*/manifest.json`, `splash/*.json` filenames, and names containing
lookahead/look_ahead/look-ahead under the configured data and artifact roots.
Artifact root: `artifacts`.
Other unnamed/custom stores and external sources are outside this inventory.

## Coverage

**Measured:** counts are inventory totals, without a sampling interval.
Quote rows repeat bookmakers, sides and markets; they are not independent games.
A calendar-season row is a date candidate, not a certified lookahead quote.
Pool file/week counts are filename coverage, not issuance or ingestion certification.

| Season | REG schedule games | Quote rows by NFL calendar | Matched REG games | Pool files | Pool weeks |
|---|---:|---:|---:|---:|---:|
| 2020 | 256 | 188691 | 240 | 0 | 0 |
| 2021 | 272 | 317239 | 255 | 0 | 0 |
| 2022 | 271 | 403913 | 271 | 0 | 0 |
| 2023 | 272 | 1999664 | 272 | 0 | 0 |
| 2024 | 272 | 1416804 | 272 | 0 | 0 |
| 2025 | 272 | 1615508 | 272 | 0 | 0 |

**Measured:** 8834 market Parquet files / 6207318 stored rows; 8832 files / 6189933 rows expose fixture kickoff metadata.
Quote kickoff range: 2020-09-04 to 2027-01-10.
Pool filename season counts: {2026: 4}.
Files without fixture kickoff metadata: 2.
Manifest files: 8834; kinds: {'historical_backfill': 8746, 'the-odds-api': 57, 'event_halves': 14, 'live': 17}.
Explicit manifest ingestion fields: {'retrieved_at_utc': 9}; quote rows with a source-scan date: 636.

**Read:** `src/nfl_ats/odds_backfill.py:242` writes historical snapshot time as
`observed_at`; that value alone does not prove contemporaneous ingestion.
`src/nfl_ats/pool_decision_lines.py:27` identifies authentic pool capture storage.
No closing-line, retrospective-news, or motivational-lookahead substitute was used.

**Measured:** 3 lookahead-named candidate files.
- `artifacts/lookahead_fade_screen/20260911T031611Z/results.json`
- `artifacts/lookahead_fade_screen/20260911T031622Z/results.json`
- `artifacts/lookahead_fade_screen/20260911T032046Z/results.json`

**Measured:** 0 inventory read errors.

## Statistical outputs and next unit

**Measured:** decisive-game record unavailable (0 games scored); IS/OOS opener
accuracy, Brier, log loss, margin MAE and their gaps unavailable; season-block
95% intervals and `probability_positive` unavailable; calibration bands unavailable.
2020, 2021, 2022, 2023, 2024 and 2025 fold coefficients are all unavailable
because no fold was fitted. No effect estimate or terminal verdict exists to record.

**Inferred:** continue only after an authentic same-fixture lookahead archive and
Tuesday pool captures pass the declared timestamp gates. Then execute the fixed
chronological LOSO protocol with cached predictions and four paired baselines.
No evidence here refutes the mechanism or establishes serving eligibility.

Reproduce: `.tools/uv.exe run --no-sync python scripts/lead79_unit1.py`.
