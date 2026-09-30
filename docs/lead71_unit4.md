# LEAD-71 unit 4: market joint-score tiebreaker

**Measured:** .tools/uv.exe run --no-sync python scripts/lead71_unit4.py; two primary candidate-versus-served MAE contrasts. No serving changes or registry writes.

## Predeclared protocol

**Declared before Unit 4 outcome access:** two candidate-versus-served MAE looks: last REG
game of each season/week (selected from the full schedule before coverage filtering), and all
eligible REG games, 2020-2025, with frozen Tuesday opener moneyline/spread/total, Unit 2
lattice coverage, four-term inputs and reconstructable served scores. Target: actual combined score.
Terms: retain Unit 2 implied margin PMF; couple to opener-centred empirical score-lattice
conditional totals on integer team scores 0..100, with one Gaussian pseudo-observation using
training residual variances; fit one over/under price multiplier while preserving the margin PMF.
Candidate: joint total median passed to production pick-consistent score selection, preserving
the served four-term side/centre. Served: joint ridge total residual (existing features/alpha/blend)
plus existing shade, production score lattice and score selection. Market half-up is context only.
Folds: six chronological season holdouts, prior trailing-season history as Unit 2; IS adds the
held season to the same history. Four-term LOSO probabilities stay fixed for both arms/schemes.
All fitted quantities use their fold only; no target-game outcomes in OOS construction. This
reconstructs the served rule with season-frozen training, not archived weekly guesses.
Metrics: MAE, paired gain=served error minus candidate error, closer/worse/tied with two-sided
exact binomial null; IS/OOS gap and per-season descriptive stability. No added arms/subgroups.
Uncertainty: 10,000 paired week-block resamples within season, seed 7104; percentile 95% CI,
probability_positive=P(gain>0)+0.5P(gain=0). Two decision looks; supporting cells descriptive.
No threshold tuning or side flips. Zero crossing never closes a signal; no registry writes.

## Paired results

**Measured:** points of absolute error; improvement is served minus candidate. 95% percentile intervals use 10,000 paired week-block draws within each season. Closer/worse counts precede the average effects; ties leave the exact binomial null.

| Population | Scheme | Games | Weeks | Closer | Worse | Tied | Exact two-sided p | Closer share exact 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| last | IS | 98 | 98 | 29 | 44 | 25 | 0.100644 | [0.284527, 0.518599] |
| last | OOS | 98 | 98 | 30 | 40 | 28 | 0.281979 | [0.310868, 0.552513] |
| all | IS | 1472 | 103 | 500 | 519 | 453 | 0.572858 | [0.459557, 0.521851] |
| all | OOS | 1472 | 103 | 512 | 500 | 460 | 0.729525 | [0.474652, 0.537171] |

| Population | Scheme | Metric | Estimate | 95% interval | probability_positive |
| --- | --- | --- | --- | --- | --- |
| last | IS | error_candidate | 10.183673 | [8.765306, 11.642857] | n/a |
| last | IS | error_served | 10.204082 | [8.765306, 11.693878] | n/a |
| last | IS | error_market | 10.265306 | [8.816071, 11.755102] | n/a |
| last | IS | gain | 0.020408 | [-0.367347, 0.377806] | 0.556200 |
| last | OOS | error_candidate | 10.163265 | [8.724490, 11.632908] | n/a |
| last | OOS | error_served | 10.234694 | [8.785714, 11.724490] | n/a |
| last | OOS | error_market | 10.265306 | [8.816071, 11.755102] | n/a |
| last | OOS | gain | 0.071429 | [-0.326531, 0.438776] | 0.653800 |
| all | IS | error_candidate | 10.482337 | [10.117365, 10.842652] | n/a |
| all | IS | error_served | 10.624321 | [10.200537, 11.055980] | n/a |
| all | IS | error_market | 10.641304 | [10.234563, 11.053492] | n/a |
| all | IS | gain | 0.141984 | [-0.008781, 0.316726] | 0.964300 |
| all | OOS | error_candidate | 10.489810 | [10.118029, 10.856277] | n/a |
| all | OOS | error_served | 10.688859 | [10.260211, 11.129834] | n/a |
| all | OOS | error_market | 10.641304 | [10.234563, 11.053492] | n/a |
| all | OOS | gain | 0.199049 | [0.048877, 0.366854] | 0.997200 |

## IS/OOS gap

| Population | Metric | IS | OOS | OOS minus IS |
| --- | --- | --- | --- | --- |
| last | error_candidate | 10.183673 | 10.163265 | -0.020408 |
| last | error_served | 10.204082 | 10.234694 | 0.030612 |
| last | gain | 0.020408 | 0.071429 | 0.051020 |
| all | error_candidate | 10.482337 | 10.489810 | 0.007473 |
| all | error_served | 10.624321 | 10.688859 | 0.064538 |
| all | gain | 0.141984 | 0.199049 | 0.057065 |

## Season stability

**Measured:** all six seasons, descriptive only; no season selected for its sign.

| Population | Season | N | Candidate MAE | Served MAE | Market MAE | Gain | Closer | Worse | Tied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| last | 2020 | 11 | 7.454545 | 9.909091 | 9.000000 | 2.454545 | 8 | 2 | 1 |
| last | 2021 | 15 | 11.000000 | 10.866667 | 10.800000 | -0.133333 | 2 | 4 | 9 |
| last | 2022 | 18 | 8.666667 | 8.000000 | 8.500000 | -0.666667 | 2 | 11 | 5 |
| last | 2023 | 18 | 10.888889 | 10.222222 | 10.888889 | -0.666667 | 2 | 8 | 8 |
| last | 2024 | 18 | 11.666667 | 12.000000 | 11.722222 | 0.333333 | 9 | 7 | 2 |
| last | 2025 | 18 | 10.388889 | 10.388889 | 10.277778 | 0.000000 | 7 | 8 | 3 |
| all | 2020 | 180 | 11.094444 | 12.977778 | 12.333333 | 1.883333 | 96 | 49 | 35 |
| all | 2021 | 227 | 11.048458 | 10.841410 | 11.000000 | -0.207048 | 53 | 78 | 96 |
| all | 2022 | 254 | 10.488189 | 10.338583 | 10.433071 | -0.149606 | 74 | 102 | 78 |
| all | 2023 | 270 | 10.477778 | 10.485185 | 10.603704 | 0.007407 | 87 | 88 | 95 |
| all | 2024 | 269 | 9.717472 | 9.828996 | 9.698885 | 0.111524 | 103 | 87 | 79 |
| all | 2025 | 272 | 10.400735 | 10.426471 | 10.386029 | 0.025735 | 99 | 96 | 77 |

## Reconstruction and chronology

**Measured:** 1480 source games; 1472 paired games; 8 excluded when either scheme lacked a production-admissible score. Full schedule: 107 last-of-week games; 98 have Unit 2 source coverage. 34 four-term rows reconstructed with production inputs; maximum cached-probability difference 1.11e-16.

**Measured:** 317 opener quote spreads differ from the historical pool proxy spread. Price projection uses its quoted spread; both guesses preserve the four-term side at the pool proxy spread.

**Read:** tiebreaker.json method_note points to docs/tiebreaker.md; scripts/lead88_unit2.py supplies the served total and score-selection recipe. src/nfl_ats/clv.py:943 sets the model spread to tue_open_home_spread. This implementation uses home-minus-away spread thresholds, verified against the opener result and margin_vs_open identity, rather than negating that threshold.

**Measured:** fold training (five prior seasons for OOS; IS adds the scored season):

| season | scheme | training_games | lattice_games | margin_games | training_end |
| --- | --- | --- | --- | --- | --- |
| 2020 | OOS | 1280 | 1280 | 1280 | 2019-12-29 |
| 2020 | IS | 1536 | 1536 | 1536 | 2021-01-03 |
| 2021 | OOS | 1280 | 1280 | 1280 | 2021-01-03 |
| 2021 | IS | 1552 | 1552 | 1552 | 2022-01-09 |
| 2022 | OOS | 1296 | 1296 | 1296 | 2022-01-09 |
| 2022 | IS | 1567 | 1567 | 1567 | 2023-01-08 |
| 2023 | OOS | 1311 | 1311 | 1311 | 2023-01-08 |
| 2023 | IS | 1583 | 1583 | 1583 | 2024-01-07 |
| 2024 | OOS | 1327 | 1327 | 1327 | 2024-01-07 |
| 2024 | IS | 1599 | 1599 | 1599 | 2025-01-05 |
| 2025 | OOS | 1343 | 1343 | 1343 | 2025-01-05 |
| 2025 | IS | 1615 | 1615 | 1615 | 2026-01-04 |

**Measured:** maximum marginal-preservation error 1.53e-16; total-price error 6.07e-13; original Unit 2 margin-price error 2.5e-06. All quote timestamps and OOS training-completion guards passed.

**Read:** the total residual fit retains the production feature union, ridge 10, blend 0.1 and shade -1. The joint score extension retains Unit 2 moneyline/spread projection and fits only the over/under multiplier. Its total median feeds the production pick-consistent score selector; mass breaks nearby-cell ties.

**Measured:** four-term coefficients, unchanged between candidate and served:

| Season | Intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available |
| --- | --- | --- | --- | --- | --- |
| 2020 | -0.045251 | 0.341246 | 0.244789 | 0.220730 | 0.028113 |
| 2021 | -0.062332 | 0.177916 | 0.271867 | 0.223716 | 0.044183 |
| 2022 | -0.055698 | 0.249647 | 0.238476 | 0.222903 | 0.039674 |
| 2023 | -0.038064 | 0.341480 | 0.246114 | 0.230272 | 0.002286 |
| 2024 | -0.056178 | 0.256116 | 0.262262 | 0.246066 | 0.039146 |
| 2025 | -0.054099 | 0.292328 | 0.295997 | 0.188739 | 0.047462 |

## Look ledger and interpretation

**Measured:** two primary contrasts; 12 arm-MAE cells, four paired-gain cells, six IS/OOS gap cells, 48 season-MAE/gain cells, 48 closer/worse/tied count cells, four exact-null tests and four exact-interval cells; 30 reused coefficient cells. Twelve total ridge fits plus 2960 margin and 2960 total-price projections were executed. These supporting views overlap and supply no additional selection choices.

**Measured:** OOS MAE gain is positive in four of six all-game seasons; last-of-week gain is positive in two, negative in three and tied in one. The 2020 gain is 1.883333 points over all games and 2.454545 on last games. **Inferred:** the pooled all-game improvement merits follow-up, but does not establish stable last-game usefulness; the current serving decision is unchanged.

**Inferred:** the market joint-score extension remains unresolved_below_power; no positive control or mechanism-refuting analysis was performed. AGENTS.md's interval and promotion rules keep research closure separate from serving. The orchestrator must run the proposed records before treating any verdict as recorded.

**Inferred:** this is a season-frozen reconstruction, not archived weekly guesses. Both score lattices use regular-season training finals. Historical training residuals inherit schedule line provenance; target lines are dated openers. Cached four-term LOSO coefficients include later seasons in earlier folds, making this retrospective. IS/OOS gaps vary the empirical lattices and total fit while holding that four-term base fixed. The week bootstrap conditions on fitted predictions, does not refit, and does not establish new-season uncertainty. The exact binomial null treats decisive games as independent; the blocked MAE interval is the primary uncertainty calculation. Six reused seasons are not an independent outer test. No side changes, pick accuracy claim, or profit claim follows.

**Measured:** predictions, exclusions, fold metadata, source hashes, frozen protocol and record payloads are under tests/scratch/codex/lead71_unit4/. No prediction rows are written under docs.

## Prior pending serial records (unchanged)

Proposed only; Bash-compatible. Set UV_CACHE_DIR as above if the default cache is inaccessible.
Unit 3 (both unresolved; pool-player summaries; validated without registry writes):

```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit3-logloss --description 'Frozen-opener push logloss; 2 primary and 125 descriptive looks' --source docs/lead71_unit3.md --league nfl --category market --effect=1.12700029517e-05 --effect-units log_loss_improvement --standard-error=0.000513855014418 --interval-low=-0.000903542138385 --interval-high=0.00107979328137 --probability-positive=0.4851 --sample-games 1480 --sample-blocks 103 --season-start 2020 --season-end 2025 --family LEAD-71-unit3 --classification unresolved_below_power --classification-evidence 'No closure or serving decision; paired week bootstrap with inherited LOSO folds' --plain-summary 'We checked how often a game lands exactly on the pool spread. The market prices and the current margin table give different chances. This check leaves the idea open and does not change any pool pick.' --notes '2 primary endpoints; 127 total estimands; unadjusted intervals; no new fits'
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit3-brier --description 'Frozen-opener push brier; 2 primary and 125 descriptive looks' --source docs/lead71_unit3.md --league nfl --category market --effect=-2.1862293285e-05 --effect-units brier_improvement --standard-error=7.3328728204e-05 --interval-low=-0.000160194332112 --interval-high=0.000127629197674 --probability-positive=0.3718 --sample-games 1480 --sample-blocks 103 --season-start 2020 --season-end 2025 --family LEAD-71-unit3 --classification unresolved_below_power --classification-evidence 'No closure or serving decision; paired week bootstrap with inherited LOSO folds' --plain-summary 'We checked how often a game lands exactly on the pool spread. The market prices and the current margin table give different chances. This check leaves the idea open and does not change any pool pick.' --notes '2 primary endpoints; 127 total estimands; unadjusted intervals; no new fits'
```

Unit 2 retained pending the orchestrator's serial handling:

```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit2-look1 --description 'Reserved market-implied lattice look 1; two-look family' --source docs/lead71_unit2.md --league nfl --category market --effect=0.0348581399733 --effect-units rps_improvement --standard-error=0.0226235062059 --interval-low=-0.000317990775327 --interval-high=0.0858635838172 --probability-positive=0.9741 --sample-games 1480 --sample-blocks 6 --season-start 2020 --season-end 2025 --family LEAD-71-unit2 --classification unresolved_below_power --classification-evidence 'The margin-lattice mechanism remains open; no admissible closure established' --plain-summary 'Moneyline, spread and total prices gave a slightly better picture of final margins. That result leaves this idea open; the pool picks are unchanged.' --notes 'Proposed serial record; 2 primary contrasts; diagnostic looks listed in source'
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit2-look2 --description 'Reserved market-implied lattice look 2; two-look family' --source docs/lead71_unit2.md --league nfl --category market --effect=-0.00117333787234 --effect-units log_loss_improvement --standard-error=0.000400826685553 --interval-low=-0.00198424742371 --interval-high=-0.000436330749859 --probability-positive=0 --sample-games 1273 --sample-blocks 5 --season-start 2021 --season-end 2025 --family LEAD-71-unit2 --classification refuted_mechanism --closing-ground wrong_sign_resolved --classification-evidence 'Fixed fifth-term extension only: log-loss improvement interval entirely negative; the margin-lattice family remains open' --plain-summary 'Adding the new market chance to the pick formula made held-out chances worse. This finding concerns that extra term; the separate margin table idea remains open.' --notes 'Proposed serial record; 2 primary contrasts; diagnostic looks listed in source'
```
