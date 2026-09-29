# LEAD-76 unit 1: rules and history inventory

**Measured:** `source_gated_missing_authentic_openers`. Decisive-game record: unavailable; zero games scored.
**Measured:** 2,816 unique REG score-present games in 1999-2019;
0 authentic-pool capture files in the six scheduled outer seasons.
Performance intervals and probability_positive are unavailable because no candidate was fitted or scored.
**Inferred:** LEAD-76 remains open at its source gate; this inventory supplies no closure evidence.

## Reproduction and scope

`.tools/uv.exe run --no-sync python scripts/lead76_unit1.py`

**Read:** the exact roadmap row and Protocol B were saved in `docs/lanes/lead76.md` before inventory.
**Read:** `src/nfl_ats/splash_lines.py:343` identifies the canonical pool-capture directory.
**Read:** Protocol B requires authentic frozen openers and pre-deadline issuance/ingestion;
bookmaker Tuesday quotes, reconstructed labels and closing lines cannot fill this source gap.
**Measured:** scanned `data/raw/**/schedules.parquet` and `data/splash/*.json`.
Only schedule identifiers, REG labels, seasons, score-null masks and capture source/season were inventoried.
No score values were converted into margins, wins, cover labels, fitted terms or performance metrics.
Schedule snapshots are digest-checked and deduplicated by game ID, retaining the last snapshot path.
A score-present count is availability only; historical score integrity and game completion need later validation.
Recent retrieval of old final scores does not certify historical pregame prediction or quote availability.

## Historical coverage

| Season | Unique REG schedule rows | Both scores present |
| --- | ---: | ---: |
| 1999 | 0 | 0 |
| 2000 | 0 | 0 |
| 2001 | 0 | 0 |
| 2002 | 0 | 0 |
| 2003 | 0 | 0 |
| 2004 | 0 | 0 |
| 2005 | 0 | 0 |
| 2006 | 0 | 0 |
| 2007 | 0 | 0 |
| 2008 | 0 | 0 |
| 2009 | 256 | 256 |
| 2010 | 256 | 256 |
| 2011 | 256 | 256 |
| 2012 | 256 | 256 |
| 2013 | 256 | 256 |
| 2014 | 256 | 256 |
| 2015 | 256 | 256 |
| 2016 | 256 | 256 |
| 2017 | 256 | 256 |
| 2018 | 256 | 256 |
| 2019 | 256 | 256 |

**Measured:** missing declared score-history seasons: 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008.

## Rules inventory

**Inferred:** the following REG rule chronology is an inventory checklist, pending documentary verification.
PAT refers to the snap spot for the one-point kick. No rule dates or categories were fitted to outcomes.
**Measured:** candidate rules fields in inspected schedule schemas: none.
No documentary rules source was verified or downloaded in this unit.

| Proposed era | Rule checklist | Score-present REG games |
| --- | --- | ---: |
| 1999-2011 | 2-yard PAT; 15-minute sudden death | 768 |
| 2012-2014 | 2-yard PAT; 15-minute modified sudden death | 768 |
| 2015-2016 | 15-yard PAT; 15-minute modified sudden death | 512 |
| 2017-2024 | 15-yard PAT; 10-minute modified sudden death | 2111 |
| 2025-2025 | 15-yard PAT; both teams receive possession, 10-minute limit | 272 |

**Read:** the declaration requires an unseen rules regime to inherit the pooled prior.
**Inferred:** after documentary verification, the 2025 rules category would require that fallback.
The 1999-2019 score-only prior must remain distinct from later earlier-season fitting data.

## Scheduled folds and unavailable results

| Outer season | REG rows | Score-present rows | Pool capture files | Coefficients |
| --- | ---: | ---: | ---: | --- |
| 2020 | 256 | 256 | 0 | Not fitted |
| 2021 | 272 | 272 | 0 | Not fitted |
| 2022 | 271 | 271 | 0 | Not fitted |
| 2023 | 272 | 272 | 0 | Not fitted |
| 2024 | 272 | 272 | 0 | Not fitted |
| 2025 | 272 | 272 | 0 | Not fitted |

**Measured:** zero fits, zero scores, zero predictive looks; the unchanged full protocol reserves 361 looks
from B=4, F=6, E=1: (4 + 36 + 8)(6 + 1) + 25.
For pooled-prior, rule-conditioned-prior, probability-term and joint-fit arms, all six folds remain unavailable.
Combined-model, model-only, market-only and Elo paired comparisons were not run.
IS and OOS accuracy, Brier, log loss, margin MAE, whole-margin log score and their gaps are unavailable.
Season-block 95% intervals, probability_positive, fold coefficients, season stability and five-band
training-quantile reliability tables are unavailable. An unmeasured probability is not reported as 0.5.
Coverage is a census of these local files; a statistical effect interval is not applicable.

## Source accounting

| Verified schedule file | Rows | Seasons |
| --- | ---: | --- |
| `data/raw/20260812T101244Z/schedules.parquet` | 4630 | 2009-2025 |
| `data/raw/20260812T130036Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260817T235649Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260824T110229Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260824T115346Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260905T211016Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260908T162105Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260915T102254Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260915T170343Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260923T002027Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260923T005026Z/schedules.parquet` | 4902 | 2009-2026 |
| `data/raw/20260929T191306Z/schedules.parquet` | 4902 | 2009-2026 |

**Measured:** pool capture files by season: 2026: 4.
**Measured:** 0 input errors. Canonical pool directory exists: True.

## Next unit

Obtain and verify the missing score histories, documentary REG rules chronology, and
2020-2025 authentic frozen-pool opener captures with deadline provenance before cached replay.
Audit certified pregame predictions and earlier training/selection/calibration partitions once source coverage exists.
Keep the declared folds, terms, endpoints and 361-look budget; do not replace missing sources with proxy grades.
No effect or uncertainty was estimated; no statistical record command is prepared.
