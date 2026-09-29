# LEAD-84 unit 1: mix book lattices before the frozen line

**Measured:** .tools/uv.exe run --no-sync python scripts/lead84_unit1.py

**Read:** ROADMAP.md:867 assigns the two constructions to unit 1 and
the paired combined-probability replay to unit 2. Protocol C and
numerical conventions were saved in the lane before fitting.

## Decisive games and requested performance statistics

No side was selected or scored. Decisive records, IS/OOS performance
and their gap, combined per-fold coefficients, effect intervals and
probability_positive are **unmeasured**, pending the unit-2 replay.
The Jensen-gap feature must enter the fitted four-term calibrated
probability; it cannot select a side alone. No closure or promotion.

## Source and chronology

**Measured:** 131 local Sunday files;
171,190 quote rows; 1,354 opener games;
33 pushes retained and
1,321 games in the four-term nonpush population.
Each game has at least two complete books at the same snapshot.
Quote rows are not independent games.

**Measured:** source hashes, target week, observation, bookmaker,
market and kickoff clocks passed. Snapshots are at or before the fixed
Sunday cutoff and before scheduled and quoted kickoff. Upstream model
training completed at least two days before that cutoff.
No closing columns or pre-2026 pool captures were read. The historical
opener is the frozen pool-line proxy. Archives remain retrospective.

| Season | Games | Pushes | Nonpush games |
|---|---:|---:|---:|
| 2020 | 208 | 7 | 201 |
| 2021 | 216 | 3 | 213 |
| 2022 | 216 | 6 | 210 |
| 2023 | 239 | 6 | 233 |
| 2024 | 238 | 6 | 232 |
| 2025 | 237 | 5 | 232 |

## Fold construction

**Measured:** the common prior fits only seasons through Y-3.
Training rows are explicitly optimistic IS; tune, calibration and
outer rows are separate. All stages are saved for the replay.

| Outer | Prior through | Prior games | Tune | Calibrate | Outer games |
|---|---:|---:|---:|---:|---:|
| 2023 | 2020 | 208 | 2021 | 2022 | 239 |
| 2024 | 2021 | 424 | 2022 | 2023 | 238 |
| 2025 | 2022 | 640 | 2023 | 2024 | 237 |

**Read:** the fixed integer support is -100..100. Both constructions
share the earlier-season empirical line-band prior and variance
coefficient. Total points adjust variance; entropic projection targets
no-vig moneyline and spread constraints. Two-sided total prices
validate the source; the total point sets variance, as in LEAD-71.
The mixture averages book PMFs equally before conditioning on nonpush
mass at the opener. The comparator averages the same books' spread
points, total points and de-vigged probabilities before one projection.

**Measured:** 3,350 paired distributions;
51,184 numerical projections; 714 outer rows.
Maximum PMF normalization error: 8.88e-16.
Maximum projection stationarity: 3.15e-07.
Maximum constraint price error: 0.134682.
Price multipliers and variance parameters are saved per fold/game/book.
They are not combined-probability coefficients selecting a side.

## Constraint feasibility review

**Measured:** read-only inspection of saved book panels and projection
diagnostics found 17 incompatible book panels across nine games. Seventeen
fold projections across two games miss a constraint by more than one
percentage point; 59 projections reach a multiplier bound. These are
numerical/source checks, not outcome-based effect estimates or new arms.

**Measured:** in data/market/raw/20211017T162500Z/quotes.parquet, Unibet lists
Denver +5.5 at -105, Las Vegas -5.5 at -115, Denver moneyline -213 and
Las Vegas moneyline +185. The resulting home win probability is 0.659801,
but the home probability of covering +5.5 is 0.489166. Maximum fitted
price error is 0.134682. The second game above the one-point diagnostic
threshold is 2022_12_DEN_CAR: William Hill's pick'em win probability is
0.500000 while its cover probability at the same zero threshold is
0.489166. The exact source rows remain in scratch provenance.

**Inferred:** a home side cannot be more likely to win than to cover +5.5;
at pick'em the nonpush win and cover probabilities must agree. A converged
bounded projection cannot reconcile those constraints. The saved PMFs are
construction diagnostics requiring source review before the paired replay.
No books or games were removed after inspecting residuals, no parameter was
retuned, and the declared population is retained. This source problem does
not refute the proposed mixture mechanism or close the research line.

**Read:** F=3, B=8, K=4; **505 planned looks** for the full family.
**Measured:** zero performance looks in this construction unit.

## Saved handoff

Under tests/scratch/codex/lead84_unit1/: features.parquet indexes
pmfs.npz; projection_diagnostics.parquet, book_panels.parquet,
folds.json and manifest.json retain numerical checks and provenance.
Prediction rows remain outside docs/.

Next: review incompatible source constraints, reconstruct timestamp-matched
moves in every year, fit the
four-term probability plus the sole Jensen-gap term, and perform the
declared five-arm replay using the reserved tune/calibration years.
Report season/week-block intervals, reliability, IS/OOS/gap, per-fold
coefficients, decisive records and all 505 looks. The orchestrator
alone runs record commands; none is warranted before scoring.
