# Injury timing leakage audit

## Verdict

**Measured:** the historical injury inputs behind the served model's graded record
postdate the Tuesday prediction clock in 1,534 of 1,537 games (2020-2025 openers),
by a median 75.8 hours (5th to 95th percentile 28.3 to 100.4). The clock the
feature builder enforces is kickoff minus 24 hours, so its guard passed every row
and could not fire. **Read:** the pool lock is Tuesday noon Eastern. **Measured:**
removing every injury-report-dependent input from training and prediction moves
the held-out record by 3 games (below), so the leakage is real but its measured
effect on the record is small and unresolved.

## Decisive record first

**Measured** (1,503 decisive games; 34 pushes excluded; season-held-out replay,
opener as the frozen line, one calibrated probability at 0.5 picks the side):

| Arm | W-L | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| Served four-term, original features | 865-638 | 57.55% | 0.68177 | 0.24427 |
| Same, injury inputs removed (constant zero) | 862-641 | 57.35% | 0.68195 | 0.24437 |
| Frozen served evaluation | 863-640 | 57.42% | 0.68277 | 0.24484 |
| Model only, original | 785-718 | 52.23% | 0.69495 | 0.25082 |
| Model only, injury removed | 783-720 | 52.10% | 0.69566 | 0.25119 |
| Market even | n/a | n/a | 0.69315 | 0.25000 |

The baseline replay reproduces the saved upstream artifact exactly (max absolute
probability difference 0.0).

**Measured** paired difference, original minus repaired (2,000 season-week block
draws, seed 8902, 107 blocks, fixed forecasts):

| Metric | Difference | 95% interval | probability_positive (original better) |
| --- | ---: | :---: | ---: |
| Accuracy (points) | +0.200 | -0.811 to +1.274 | 0.659 |
| Log loss | -0.00018 | -0.00113 to +0.00074 | 0.657 |
| Brier | -0.00010 | -0.00056 to +0.00033 | 0.678 |
| Model-only accuracy (points) | +0.133 | -1.284 to +1.563 | 0.562 |
| Model-only log loss | -0.00071 | -0.00293 to +0.00138 | 0.740 |

Negative log loss and Brier differences mean the repaired arm had marginally
lower loss; the original has 3 more wins. The result is
`unresolved_below_power`; nothing here closes an injury signal and no positive
effect is established.

By season, accuracy difference in points (probability original is better):
2020 -0.45 (0.42), 2021 0.00 (0.49), 2022 -2.02 (0.12), 2023 +1.13 (0.85),
2024 +2.63 (0.99; interval +0.39 to +4.58), 2025 -0.37 (0.33). Signs alternate;
no stable season direction.

In-sample versus held-out (four-term accuracy / log loss), original: in-sample
0.5675-0.5809 / 0.6800-0.6821; held-out by season 0.545-0.594 / 0.6755-0.6855.
Repaired: in-sample 0.5643-0.5825 / 0.6807-0.6824. Fold coefficients, original:
model logit 0.246-0.558, composition flags 0.245-0.304, market move
0.189-0.248, move available 0.087-0.148; repaired model logit 0.222-0.481, the
rest unchanged.

Reliability (five equal-count bands, mean pick probability to observed win
rate), original: 0.511 to 0.542, 0.528 to 0.577, 0.548 to 0.558, 0.576 to
0.560, 0.640 to 0.641. Repaired: 0.510 to 0.532, 0.528 to 0.580, 0.547 to
0.548, 0.575 to 0.573, 0.638 to 0.635.

Looks: 151 declared (21 comparisons, 105 descriptive cells, 25 calibration
cells), no tuning. The protocol is preserved unchanged in
`docs/lanes/injury-timing-leakage-audit.md`.

**Inferred:** most of the four-term skill over model-only (57.6% versus 52.2%)
comes from the market-movement and composition terms, not injuries. The
LEAD-89 note that the market-movement input is a Sunday quantity applies to the
same evaluation and is outside this injury-only repair; it is the larger open
timing question.

## Source audit

1. **Injury-derived served columns (read, `src/nfl_ats/constants.py:565`,
   90 margin inputs):** nine `diff_injury_*` columns (offense, defense, special
   teams, offensive line, skill, front, secondary unavailability; skill EPA
   value lost; defense disruption value lost), plus two indirect terms built
   from the same report rows, `diff_qb_expected_epa_per_dropback` and
   `diff_qb_start_probability` (`players.py` around 1985-2003 uses
   `visible_injuries`). Starter EPA, CPOE and experience come from history and
   the depth chart, not injury reports.
2. **Observation timestamp per row (measured):** each side stores the latest
   report revision at or before kickoff minus 24 hours (`players.py:1824-1829`;
   `--decision-hours` default 24 at `cli_commands/features.py:608`). Across 3,074
   team-sides: 2,518 real report edit times, 544 week proxy (kickoff minus 24
   hours, floored at Tuesday midnight, `players.py:207-250`), 12 missing.
   Observed a median 45.8 hours before kickoff (5th to 95th percentile 24.0 to
   55.1). No row is after kickoff or after the builder's own cutoff.
3. **Prediction timestamp (read):** the opener grade and pool lock are Tuesday
   noon Eastern (the saved quote clock is hour 12 Eastern on all 1,537 games).
   Sunday-game injury reports are published Wednesday to Friday, so nearly every
   observation is later. **Measured:** late in 1,534 of 1,537 games (2020 227/227,
   2021 238/239, 2022 255/255, 2023 272/272, 2024 270/272, 2025 272/272); 1 game
   has no observation on a side; 272 games have a proxy on at least one side.
   Lag after Tuesday noon: 5th percentile 28.3 hours, quartiles 74.5 and 92.8,
   median 75.8, 95th percentile 100.4.
4. **Guard (read):** the only runtime clock check on these rows is inside the
   builder against kickoff minus 24 hours; `lineage.py:846-865` compares lineage
   effective times to a card's prediction time for live cards only. Nothing
   compares the historical rows' observation times to the evaluation's Tuesday
   clock. It did not fire because every row honestly satisfied the clock it was
   given; that clock is later than the served decision.
   `prediction_safety.py` `injury_feature_presence` only checks values are
   nonzero.
5. **Live Tuesday card (measured):** the week 4 forecast
   (`margin_predictions/2026-week-04-20260929T192403Z`) uses the player snapshot
   captured 2026-09-27 13:15:40 UTC (Sunday morning, before any week 4 report)
   for all 16 games, basis `snapshot_captured_at`. No leakage, but the whole
   injury block is exactly zero (share nonzero: week 1 0.82, week 2 0.95, week 3
   0.99, week 4 0.00) while 95 to 100 percent of training rows are nonzero
   (offensive line 0.953, secondary 0.971, quarterback start probability 0.342).
   The live card serves a state the model never saw in training. The repaired
   arm is the matching configuration; its held-out record is within 3 games of
   the original.

## Proposed guard fix

Exact diff: `tests/scratch/codex/injury_timing_audit/injury_clock_guard.diff`
(compiled, not applied), reproduced in the lane. It adds
`decision_clock="tuesday_noon"` to `enrich_with_player_features`, capping the
decision time at Tuesday noon Eastern of the game week. Rebuilding the feature
table under that clock, with a matching CLI flag, is the follow-up.

## Files and commands

`scripts/injury_timing_audit.py` (ruff check and ruff format pass). Outputs in
`tests/scratch/codex/injury_timing_audit/` (`results.json`, `clock_census.json`,
`clock_census.parquet`, `baseline/`, `candidate/`). Run:
`.tools/uv.exe run --no-sync python scripts/injury_timing_audit.py` (one
process, 2 threads). No src change, registry write or publication.
