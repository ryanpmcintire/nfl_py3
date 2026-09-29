# LEAD-69 unit 2

## Goal
Extend the horizon-weighted market move to 2020-2025 and reproduce unit 1.

## State
**Measured:** 1,285 games; decisive 4-12; gain -0.623 pp [-1.238,-0.077],
probability_positive 0.0174. Horizon c1 negative 6/6, range [-0.393,-0.079].
IS/OOS accuracy 56.809%/55.875%, gap 0.934 pp [0.077,1.800].
Original 685-game replay exactly reproduced +0.292 pp [-0.249,+0.779], P+=0.7963
with its original season intervals; week interval [-0.717,+1.327], P+=0.7186.

## Protocol amendment
Population: pinned `pick_probability/20260929T192747Z/per_game.parquet`, 2020-2025
regular-season nonpush historical OPENER proxy; move available; retain unit-1
SNF/MNF exclusions. Older quote/game alignment is LEAD-73 unit 2; six missing
older moves are unavailable, never imputed. Recent predictors use unit-1 builder.
Target: home covers opener. Base: served four terms plus intercept on identical
rows. Primary: one move-times-log-hours term; descriptive second look: unchanged
fixed slot interactions. Availability amendment: for each book's Wednesday-onward
net move, start at its start quote's availability, max(capture, snapshot, update).
Added feature: median book net move times log(hours to min(kickoff, Sunday 16:00
ET)); old quote alignment ends at 12:45 Sunday. Strict predeadline gates apply.
Folds: six LOSO plus full-data IS per arm; ridge 0.001, train-only scaling,
no tuning. Replay original Tuesday-horizon unit 1 with three LOSO plus IS; also
report 2023-2025 inside six-season predictions. Metrics: opener accuracy, log
loss, Brier, evaluation-only move toward pick; model-only and neutral-market
baselines; decisive record first; coefficients, season stability, IS/OOS gaps,
five equal-width reliability bins. One fitted probability selects each side.
Uncertainty: 10,000 paired season-stratified week-block draws, seed 20260929,
fixed predictions, percentile 95% intervals; probability_positive counts ties
half. Exact three-season bootstrap additionally checks original replay parity.
Budget: two research looks; 33 numerical fits; 660 reporting looks = 33 fits +
120 aggregate metrics + 144 contrasts + 36 gaps + 240 season metrics + 75
calibration bands + 12 decisive records. Coefficients describe those fits.
Zero crossing does not close a signal. No registry action or serving change.

## Tried
Ran `.tools/uv.exe run --no-sync --no-cache python scripts/lead69_unit2.py` once.
33 fits; source/parity gates, Ruff and 8 read-only registry payload validations passed.
Report: `docs/lead69_unit2.md`; rows/logs/batch: `tests/scratch/codex/lead69_unit2/`.

## Next
Owner reviews the timing amendment and executes the prepared batch serially.

## Open
**Inferred:** unresolved_below_power, no serving change; adverse accuracy does not
close the broader mechanism. Proper-score gains unresolved; LOSO is not an outer test.
Quote density and the predeclared availability amendment limit direct unit-1 comparison.

## Record commands
Prepared, not run; eight correlated cells, slots descriptive; no continuity pooling.
```bash
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record \
  --batch tests/scratch/codex/lead69_unit2/registry_batch.json \
  --source docs/lead69_unit2.md \
  --plain-summary 'Weighting sportsbook moves by the time available produced fewer winning picks in this check. Keep the current picks while the wider idea remains unsettled.'
```
