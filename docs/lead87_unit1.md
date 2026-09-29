# LEAD-87 unit 1: playoff clock and label join

**Measured:** `.tools/uv.exe run --no-sync python scripts/lead87_unit1.py` completed.
**Read:** the immutable declaration is in `docs/lead87_protocol.md`; it was copied
into `docs/lanes/lead87.md` before source inventory or outcome access.

## Scope and decisive record

This executes the row's first unit: playoff clock/label join. No probability was fitted
or scored, and no side was selected. Decisive record, IS/OOS results and gap, coefficients,
95% effect intervals, and probability_positive are **unmeasured**, pending joint-likelihood replay.
**Read:** the complete replay declares 497 looks; this join consumes zero fit/score looks.

## Joined source inventory

**Measured:** counts below are complete local inventories, not estimates; sampling intervals do not apply.

| Season | POST scheduled | POST paired | POST leader paired | POST pushes | POST leader nonpush | REG frozen | REG paired | REG leader paired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 13 | 8 | 8 | 0 | 8 | 220 | 201 | 176 |
| 2021 | 13 | 9 | 9 | 0 | 9 | 236 | 213 | 208 |
| 2022 | 13 | 9 | 9 | 0 | 9 | 248 | 210 | 202 |
| 2023 | 13 | 8 | 8 | 0 | 8 | 266 | 233 | 231 |
| 2024 | 13 | 9 | 9 | 1 | 8 | 266 | 232 | 224 |
| 2025 | 13 | 9 | 9 | 0 | 9 | 267 | 232 | 231 |

**Measured:** 262 verified quote files; 18246 unique same-book pairs.
**Measured:** 52 paired postseason games; 26 lack an admissible Tuesday/Sunday pair.
The latter remain in the saved join with missing moves; missing values are never converted to zero.
Pushes remain in the join and are excluded only by the separately saved conditional-fit eligibility flag.

**Measured:** quote hashes, target season/week, team identity, signed handicap, bookmaker update
at or before observation, observation before both recorded kickoffs, Tuesday observation by noon,
and Sunday observation before min(kickoff, 12:45 ET) were checked. Late quotes are paired within
book before taking a median. Tuesday anchors precede noon; they are not noon-capture evidence.
The archive label sun_early_close names a pre-kick capture; no final closing-line column is read.
Postseason frozen line is the median of eligible Tuesday books. REG keeps its original frozen opener.
Leader moves use the served recipe's declared bookmaker set. All-book moves are saved as source
evidence only, without creating another fitted arm. No postseason model or flag columns are fabricated.

## Chronological partitions

**Measured:** counts below are available postseason conditional-fit rows; later seasons never enter an earlier partition.

| Outer season | Fit through | POST fit | Tune season | POST tune | Calibration season | POST calibration | REG outer source-complete |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| 2023 | 2020 | 8 | 2021 | 9 | 2022 | 9 | 231 |
| 2024 | 2021 | 17 | 2022 | 9 | 2023 | 8 | 224 |
| 2025 | 2022 | 26 | 2023 | 8 | 2024 | 8 | 231 |

**Measured:** 1503/1503 frozen REG games match companion margin-cache cutoff rows;
0 cutoffs reach their own game date. Last training dates by prediction season:
2020: 2020-12-28; 2021: 2022-01-03; 2022: 2023-01-01; 2023: 2023-12-31; 2024: 2024-12-30; 2025: 2025-12-29.
**Measured:** 1415 companion predictions train on earlier games in their own season.
**Inferred:** pregame chronology is verified for the matched companion cache; strict season exclusion
and exact opener-probability lineage are not established. The replay must resolve this before
claiming the prescribed outer-season result and preserve distinct fit, tune, calibration, and outer periods.

## Next unit and handoff

Implement the declared single joint likelihood: retain every REG term, add a POST-only intercept,
and share only the move coefficient with the auxiliary market-only logit. Keep all headline
grades on the REG population and compare the four predeclared baselines. No research verdict
or registry command is warranted by source availability alone; no signal is closed.
The full replay still owes decisive records, IS/OOS/gaps, per-fold coefficients, four endpoints,
season/week-block intervals, probability_positive, and five fixed-width reliability bands.
**Read:** AGENTS.md requires one fitted calibrated probability to select a side; a zero-crossing
interval cannot close a signal. No serving or promotion decision is made by this inventory.

**Measured:** detailed joins, book evidence, folds, source hashes and cutoff audit are saved only under
`tests/scratch/codex/lead87_unit1/`; no prediction-row dumps are written to docs.
