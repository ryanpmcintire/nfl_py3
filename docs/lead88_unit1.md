# LEAD-88 unit 1: paired-total join

**Measured:** outcome-free source/clock inventory and paired-total join; no fit, score, registry write or serving change.
The roadmap separates this paired-total unit from its one-parameter replay. Declaration was saved first in
`docs/lanes/lead88.md`; verbatim row and shared Protocol C are in `docs/lead88_protocol.md`.

## Source coverage

**Measured:** 262 verified source files / 316,530 quote rows;
16,720 same-book total pairs give 1,343 games and
101 actual weekly last games of 107 scheduled last games.
The historical sources are present. Missing games are not replaced with zero movement or closing lines.

| Season | Scheduled REG | Paired games | Last games | Four-term rows | Chronological p present | Upstream day before freeze |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | 256 | 197 | 13 | 190 | 0 | 162 |
| 2021 | 272 | 216 | 16 | 213 | 0 | 208 |
| 2022 | 271 | 216 | 18 | 210 | 210 | 216 |
| 2023 | 272 | 239 | 18 | 233 | 233 | 239 |
| 2024 | 272 | 238 | 18 | 232 | 232 | 238 |
| 2025 | 272 | 237 | 18 | 232 | 232 | 237 |

**Measured:** rejected-row clock/identity counts overlap; they are not additive:

| Check | Rows failing |
| --- | --- |
| target_season_week | 21762 |
| team_identity | 5956 |
| source_identity | 0 |
| snapshot_window | 21762 |
| requested_window | 21762 |
| sunday_request | 15312 |
| observed_clock | 0 |
| book_clock | 0 |
| market_clock | 0 |
| pregame | 6164 |
| positive_total | 0 |
| book_and_outcome | 0 |

## Join rules

**Read/inferred implementation:** select historical target-season/week Tuesday and Sunday-12:30 manifests,
verify quote hashes, require exactly one matching OVER/UNDER pair at the same total per book/capture,
then retain the latest admissible capture per game/book/endpoint. Require source and schedule team identity,
observation = snapshot, bookmaker/market updates no later than observation, and requests before both
source and schedule kickoff. Tuesday anchors precede noon; they are not evidence of a noon quote.
Sunday requests must equal the declared 12:30 boundary. Only books present at both endpoints enter the
game-level medians. Preserve each original bookmaker pair and clock in the scratch artifact.
The last-game flag is computed from the complete REG schedule before quote filtering. No score, margin,
cover outcome, or closing total column is loaded. Probability availability is checked without scoring.

## Replay fold inventory

| Outer year | Fit through Y-3 | Tune Y-2 | Calibrate Y-1 | Outer games | Outer last games |
| --- | --- | --- | --- | --- | --- |
| 2023 | 197 | 216 | 216 | 239 | 18 |
| 2024 | 413 | 216 | 239 | 238 | 18 |
| 2025 | 629 | 239 | 238 | 237 | 18 |

**Measured:** no in-sample/OOS result, gap, coefficient, decisive record, interval or probability_positive
has been computed. These are not zero or 0.5. Planned family: 713 looks; executed statistical looks: 0.
The source counts are census counts, not sampled outcome estimates; an outcome interval is inapplicable.

## Open replay requirements

**Read:** `scripts/tiebreaker_total_study.py:33` rounds half-up, whereas the current served path in
`src/nfl_ats/tiebreaker.py:427` uses `pick_consistent_top_score` with a fixed side and margin centre.
Its no-model fallback at line 466 rounds a neighborhood median. The declaration says existing rounding;
the replay must resolve the intended entry point before fitting, rather than silently substitute a rule.
**Read:** the frozen four-term source excludes opener pushes and lacks chronological probabilities in
its earliest seasons. The coverage table identifies missing rows; keep pushes in the total population.
**Inferred:** a training calendar day ending before freeze is a date-granularity check, not proof of
completion-time availability or Protocol C fit/tune/calibration separation. Reconstruct matched four-term features,
verify fold lineage, retain a fixed margin probability/side/Best Pick, calibrate the discrete total law on
the separate year, and report all declared comparisons. This unit makes no signal or pool-rank verdict.

## Saved artifacts

`tests/scratch/codex/lead88_unit1/{book_pairs,paired_games}.parquet` contains quote features and clocks only.
`summary.json` contains source hashes, coverage, rejected checks and fold counts. No outcome rows enter docs.
Record commands: none, because no effect has been estimated or adjudicated.
