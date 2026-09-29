# LEAD-74 unit 1: local quote provenance inventory

**Measured:** source_gated_missing_ingestion_provenance. Metadata only; no outcomes were read.
The frozen declaration is in docs/lanes/lead74.md:9.

**Measured:** data/market/raw contains 8,835 snapshot directories; 8,835 manifests were readable.
There are 8,104 target-season manifests, 8,104 readable quote-file footers, and 5,031,980 quote rows across all markets and game types.
Source counts are not paired quotes, REG games, independent observations, or picks.

| Season | Manifests | Quote files | Quote rows | Price-schema files | Ingestion-field candidates |
|---|---:|---:|---:|---:|---:|
| 2023 | 2,629 | 2,629 | 1,999,664 | 2,629 | 0 |
| 2024 | 2,794 | 2,794 | 1,417,196 | 2,794 | 0 |
| 2025 | 2,681 | 2,681 | 1,615,120 | 2,681 | 0 |

**Measured:** capture kinds: historical_backfill = 8,104.
Target-season metadata errors: 0; season labels inferred from timestamps: 0.
All-archive access/parse errors: none.

**Measured:** decision-label counts follow; labels alone do not prove pool deadlines.

| Decision label | 2023 | 2024 | 2025 |
|---|---:|---:|---:|
| fri_pre | 20 | 22 | 22 |
| intraday_hourly | 2322 | 2322 | 2322 |
| mon_pre_mnf | 40 | 40 | 40 |
| preseason_futures | 1 | 1 | 1 |
| sat_1500 | 0 | 22 | 22 |
| sat_1900 | 0 | 22 | 22 |
| sat_intraday | 0 | 113 | 0 |
| sat_last_pre_kick | 20 | 22 | 22 |
| sat_midday | 40 | 40 | 40 |
| sun_early_close | 22 | 22 | 22 |
| sun_late_close | 40 | 40 | 40 |
| thu_pre | 20 | 22 | 22 |
| thu_pre_tnf | 40 | 40 | 40 |
| true_open | 22 | 22 | 22 |
| tue_open | 42 | 44 | 44 |

**Measured:** 1 normalized quote schema(s). Explicit ingestion/retrieval field candidates: none.

**Read:** Protocol B requires issuance/ingestion before the pool deadline (docs/lanes/ideation-2026-09-29b.md:12). The backfill writer stores requested historical time and provider snapshot time without a retrieval timestamp (src/nfl_ats/odds_backfill.py:258). Those times cannot stand in for ingestion; file modification times are also not proof.

**Inferred:** the audited archive lacks the required predeadline-ingestion evidence. The declared replay is source-gated. This is a provenance gap, not absence of historical prices or evidence against the proposed mechanism.
Same-book, two-sided, unchanged-spread matching; REG-only filtering; authentic frozen-opener and pool-deadline matching; and certified prediction coverage remain unaudited because the prerequisite failed.

| Requested result | Availability |
|---|---|
| Decisive-game record | Unscored; not a 0-0 result |
| IS/OOS accuracy, Brier, log loss, margin MAE, and gap | Not estimated |
| Season-block 95% intervals and probability_positive | Not estimated |
| 2023, 2024, 2025 fold coefficients and stability | No folds fitted |
| Five training-quantile reliability bands | Not estimated |
| Looks | 177 predeclared; 0 fitting/outcome-reporting looks used |

**Inferred:** descriptive inventory counts have no sampling interval. No effect, closing ground, promotion verdict, or registry record is justified. An interval crossing zero would not close this signal; one fitted calibrated discrete-margin probability would select the side in a future eligible replay.

Next: supply verifiable predeadline ingestion provenance, then audit Tuesday/deadline quote pairs and frozen openers without changing the declaration.

Command: .tools/uv.exe run --no-sync python scripts/lead74_unit1.py
