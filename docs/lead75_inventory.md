# LEAD-75 unit 1 — local forecast inventory

**Measured:** source gate `data_gap`; 0 eligible matched forecast pairs. No fit or score ran.

Command: `.tools/uv.exe run --no-sync python scripts/lead75_unit1.py`.
Protocol: `docs/lanes/lead75.md`, saved before outcomes; ROADMAP.md:858 and Protocol B at
docs/lanes/ideation-2026-09-29b.md:11-23.

**Measured:** scanned 9 parquet archives under `data/raw/forecast_archive`; 0 additional partial
archive directories. Population: 1062 REG open-air games in 2020-2025, from
`data/raw/20260824T115346Z/schedules.parquet` using only game identity, season, game type and roof.
Forecast reads project only forecast values and provenance; no realized weather, scores, margins,
picks or prediction outcomes were loaded.

All counts below are **measured** inventory counts, not effect estimates.

| Archive | Product | Cutoff | Rows | Usable wind/issuance | Timely ingestion |
|---|---|---|---:|---:|---:|
| full_2020_2025 | MEX | tuesday_noon | 1062 | 1045 | 0 |
| kickoff_nearest_2009_2025 | GFS | kickoff_nearest | 1062 | 1045 | 0 |
| kickoff_nearest_2009_2025_checkpoint | unknown | unknown | 0 | 0 | 0 |
| kickoff_nearest_2024 | GFS | kickoff_nearest | 178 | 173 | 0 |
| pool_decision_2009_2025 | GFS | pool_decision | 1062 | 1045 | 0 |
| spotcheck_kickoff_nearest_2020 | GFS | kickoff_nearest | 10 | 10 | 0 |
| spotcheck_kickoff_nearest_2021 | GFS | kickoff_nearest | 10 | 10 | 0 |
| spotcheck_kickoff_nearest_2022 | GFS | kickoff_nearest | 7 | 7 | 0 |
| spotcheck_kickoff_nearest_2023 | GFS | kickoff_nearest | 10 | 10 | 0 |

Usable wind/issuance requires fetch success, a nonmissing wind forecast and valid time, and issuance
no later than the archive cutoff. It is not certification for replay. Rows from spot checks and
checkpoints overlap; archive row counts are not additive.

**Measured:** unique Tuesday games 1045; unique pool-deadline games 1045; overlap 1045; same-product
overlap 0; same-product/station/kickoff/forecast-valid-time pairs 0; pairs also satisfying issuance
order and ingestion cutoffs 0.

**Measured:** 0/9 archives carry a recognized row-level ingestion timestamp. Archive build
timestamps are collection metadata, not evidence of ingestion before historical pool deadlines. The
legacy Tuesday product is identified from its manifest's explicit `model=` source and Tuesday cutoff
column; an unlabelled checkpoint remains unknown.

**Read:** scripts/ingest_forecast_archive.py:23-28 selects Tuesday MEX and deadline GFS; lines
176-187 select each bulletin's nearest valid time, so a shared game identifier alone cannot certify
an identical forecast target. The inventory requires matching product/source, station, kickoff and
forecast valid time before certifying a revision.

| Season fold | Population games | IS / OOS / gap | Coefficients |
|---|---:|---|---|
| 2020 | 165 | Not estimated: source gate | Not fitted |
| 2021 | 182 | Not estimated: source gate | Not fitted |
| 2022 | 177 | Not estimated: source gate | Not fitted |
| 2023 | 180 | Not estimated: source gate | Not fitted |
| 2024 | 178 | Not estimated: source gate | Not fitted |
| 2025 | 180 | Not estimated: source gate | Not fitted |

**Measured:** decisive games scored: 0; win-loss record not estimated. IS/OOS opener accuracy,
Brier, log loss, margin MAE, gaps, fold coefficients, calibration bands, season-block 95% intervals
and `probability_positive` are not estimated because the declared source gate blocks replay. An
inventory count has no sampling interval. Planned looks: (2 + 36 + 8*0)*(6 + 1) + 25 = 291; executed
outcome looks: 0.

**Inferred:** the existing cross-product pair cannot identify forecast innovation separately from
product differences. This is a data gap, not a refuted mechanism or a negative effect. No research
closure or registry record is supported. The roadmap source gate requires stopping here. Next:
obtain comparable paired forecasts with identical valid times and historical issuance/ingestion
provenance, then re-audit before the predeclared replay. No external source access or serving change
ran.
