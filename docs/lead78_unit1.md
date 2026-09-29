# LEAD-78 unit one: completion and state-update source audit

**Measured:** `source_gap`; zero fits, scored games and outcome looks; 298 looks preregistered.

**Inferred:** this source gap leaves the effect unmeasured. It neither refutes the mechanism nor closes research.

## Declaration and scope

**Read:** the unchanged roadmap row and Protocol B were copied to `docs/lanes/lead78.md` before this command. Population: 2020-2025 REG games with a later pick deadline; retain
every eligible game. Target: authentic Tuesday frozen-opener cover. Terms: fixed score-only opponent-graph state, updated-minus-Tuesday implied home-margin delta, market move, and
their joint fitted discrete-margin probability. Chronology-purged LOSO excludes target/later seasons and separates earlier training, selection and calibration. B=3, F=6, E=0;
L=(3+36)(6+1)+25=298.

**Measured:** inspected schedule parquet schemas and calendar columns, adjacent snapshot manifest metadata, Splash capture metadata and the referenced baseline CSV header. No
score, margin, cover label, probability or market-line value was used. No source fetch, registry write, rebuild or served-card change ran.

Schedule scope: `data/raw/*/schedules.parquet`; capture scope: `data/splash/YYYY_weekWW_*.json`; baseline header: `artifacts/four_term_probability/20260914T222345Z/per_game.csv`.

## Source counts

**Measured:** exact inventory counts; statistical intervals do not apply.

| Inventory | Count / value |
| --- | --- |
| Schedule schemas inspected | 12 |
| Schemas with recognized completion fields | 0 |
| Manifests with timezone-aware snapshot fetch times | 12 |
| Splash capture files, all seasons | 4 |
| Splash capture files, 2020-2025 | 0 |
| 2020-2025 captures with timezone-aware capture times | 0 |
| 2020-2025 captures with timezone-aware pool lock times | 0 |
| Referenced baseline exists / column count | True / 26 |

| Snapshot count | Temporal column names |
| --- | --- |
| 12 | gameday, gametime, overtime, weekday |

Snapshot fetch-time range: `2026-08-12T10:12:44.833533+00:00` to `2026-09-29T19:13:06.219719+00:00`.

Baseline timestamp-header matches: none.

## Calendar and capture coverage

**Measured:** calendar counts are not certified eligible games; Thu/Sat kickoff dates do not establish final publication. Capture counts can include duplicates and do not establish
Tuesday issuance or deadline validity. For each game, calendar metadata comes from the latest lexically named inspected snapshot containing it.

| Season | REG games | Thu/Sat | Sunday | Capture files | Weeks | Lock times |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | 256 | 19 | 213 | 0 | 0 | 0 |
| 2021 | 272 | 23 | 229 | 0 | 0 | 0 |
| 2022 | 271 | 35 | 219 | 0 | 0 | 0 |
| 2023 | 272 | 27 | 223 | 0 | 0 | 0 |
| 2024 | 272 | 26 | 221 | 0 | 0 | 0 |
| 2025 | 272 | 27 | 222 | 0 | 0 | 0 |

| Calendar source | Games |
| --- | --- |
| data/raw/20260929T191306Z/schedules.parquet | 1615 |

## Gate and requested evaluation

**Measured:** No inspected schedule archive exposes a recognized final/completion timestamp field. No local Splash capture files for scheduled seasons 2020, 2021, 2022, 2023, 2024,
2025. Timezone-aware pool lock timestamps do not cover all scheduled seasons.

**Read:** LEAD-78 requires published-final timestamps and excludes unfinished games and later corrections. Protocol B requires issuance/ingestion before the pool deadline and
authentic frozen openers. Snapshot fetch times and kickoff dates cannot substitute for historical final-publication times.

**Inferred:** without that provenance no eligible population or state replay is certified. The baseline CSV header alone also cannot certify chronology-purged folds or a calibrated
discrete-margin distribution.

Decisive-game W-L-P record; IS/OOS opener accuracy, Brier, log loss, margin MAE and gaps; five-band reliability; season-block 95% intervals; and `probability_positive`:
unavailable, not zero. Zero games scored; zero folds fitted. Calendar/capture counts above are not performance looks.

| Scheduled fold | IS / OOS / gap | Coefficients | 95% CI / probability_positive |
| --- | --- | --- | --- |
| 2020 | unavailable | not fitted | unavailable |
| 2021 | unavailable | not fitted | unavailable |
| 2022 | unavailable | not fitted | unavailable |
| 2023 | unavailable | not fitted | unavailable |
| 2024 | unavailable | not fitted | unavailable |
| 2025 | unavailable | not fitted | unavailable |

## Next unit

Recover deadline-valid historical frozen openers and lock metadata, and score-only histories with published-final and ingestion timestamps preserving the version available then.
Verify the fixed earlier-trained rating architecture and certified baseline folds before replay. Retain all eligible games, including zero-delta weeks. No registry command is
supplied because no effect estimate or research verdict exists.

Verification command: `.tools/uv.exe run --no-sync python scripts/lead78_unit1.py`.
