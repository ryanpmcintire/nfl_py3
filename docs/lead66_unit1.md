# LEAD-66 whole-margin PMF unit 1

**Measured:** .tools/uv.exe run --no-sync python scripts/lead66_unit1.py.
Protocol was frozen in docs/lanes/lead66.md before outcome scoring.

## Population and decisive records

**Measured:** 1537 opener games, 1503 non-push and 34 pushes, 2020-2025; 1537 source rows before filtering.
All PMF scores include pushes. No zero-mass outcome was excluded. The roadmap's 1,503 non-push count is a declaration, not an imposed sample-size filter.

**Measured:** RPS records are candidate lower-score wins/losses/ties. Cover records use PMF-implied side disagreements on non-push games: diagnostics only, not served picks. Wilson intervals describe game counts; inferential RPS intervals resample seasons.

| Candidate vs baseline | RPS W-L-T | Win fraction 95% | Cover W-L | Cover 95% |
| --- | --- | --- | --- | --- |
| served_lattice vs market_residual | 746-791-0 | [46.04%, 51.04%] | 356-294 | [50.93%, 58.56%] |
| model_centred_lattice vs market_residual | 750-787-0 | [46.30%, 51.30%] | 367-306 | [50.75%, 58.26%] |
| model_centred_lattice vs served_lattice | 722-762-53 | [46.12%, 51.20%] | 26-27 | [36.12%, 62.12%] |

## Paired RPS improvements

**Measured:** positive means lower candidate RPS. RPS sums squared CDF error across every integer threshold (margin-point units). Bootstrap: 10,000 paired season-block draws, seed 20260929, game-weighted aggregation, percentile 95% intervals; six blocks.

| Contrast | IS gain | OOS gain | OOS 95% | OOS-IS gap | probability_positive |
| --- | --- | --- | --- | --- | --- |
| served_lattice vs market_residual | -0.024680 | -0.026309 | [-0.070629, 0.004613] | -0.001629 | 0.076500 |
| model_centred_lattice vs market_residual | -0.037600 | -0.037718 | [-0.081865, -0.010954] | -0.000119 | 0.000000 |
| model_centred_lattice vs served_lattice | -0.012920 | -0.011409 | [-0.019049, -0.003794] | 0.001510 | 0.002900 |

## Arm metrics and IS-OOS gap

**Measured:** exact log scores retain infinity for zero realized mass. Infinite IS/OOS log scores give an undefined gap; no clipping or smoothing is applied. Cover Brier/log loss exclude pushes; full-margin scores include them.

| Arm | IS RPS | OOS RPS | OOS RPS 95% | RPS gap | IS log | OOS log | OOS log 95% | OOS zero mass | Cover Brier | Cover log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_residual | 7.144089 | 7.165375 | [6.797266, 7.556190] | 0.021286 | infinity | infinity | [infinity, infinity] | 26 | 0.250095 | 0.693336 |
| served_lattice | 7.168769 | 7.191684 | [6.869740, 7.561935] | 0.022915 | 3.706584 | infinity | [infinity, infinity] | 69 | 0.251726 | 0.696936 |
| model_centred_lattice | 7.181689 | 7.203093 | [6.882341, 7.572147] | 0.021405 | infinity | infinity | [infinity, infinity] | 81 | 0.251950 | 0.697465 |

## Fold populations and metrics

**Measured:** chronological LOSO mass fitting uses preceding seasons in the production five-season window. No held-out-season outcomes enter OOS mass fits. IS adds the whole held-out season to that same pool and is deliberately optimistic. Base model forecasts remain frozen walk-forward forecasts; base models were not refitted LOSO.

| Held-out season | Games | OOS pool | IS pool | Latest OOS outcome |
| --- | --- | --- | --- | --- |
| 2020 | 227 | 1280 | 1536 | 2019-12-29 |
| 2021 | 239 | 1280 | 1552 | 2021-01-03 |
| 2022 | 255 | 1296 | 1567 | 2022-01-09 |
| 2023 | 272 | 1311 | 1583 | 2023-01-08 |
| 2024 | 272 | 1327 | 1599 | 2024-01-07 |
| 2025 | 272 | 1343 | 1615 | 2025-01-05 |

| Season | Arm | Games | IS RPS | OOS RPS | Gap | IS log | OOS log | Zero mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | market_residual | 227 | 7.038180 | 7.050222 | 0.012041 | infinity | infinity | 4 |
| 2020 | model_centred_lattice | 227 | 7.053355 | 7.076400 | 0.023045 | 3.715190 | infinity | 11 |
| 2020 | served_lattice | 227 | 7.035999 | 7.052229 | 0.016229 | 3.670992 | infinity | 10 |
| 2021 | market_residual | 239 | 8.043413 | 8.072356 | 0.028943 | infinity | infinity | 4 |
| 2021 | model_centred_lattice | 239 | 8.049145 | 8.072770 | 0.023625 | infinity | infinity | 13 |
| 2021 | served_lattice | 239 | 8.029558 | 8.062747 | 0.033189 | 3.807211 | infinity | 11 |
| 2022 | market_residual | 255 | 6.399334 | 6.420486 | 0.021152 | infinity | infinity | 1 |
| 2022 | model_centred_lattice | 255 | 6.547657 | 6.565404 | 0.017748 | infinity | infinity | 11 |
| 2022 | served_lattice | 255 | 6.532741 | 6.553466 | 0.020725 | 3.663161 | infinity | 11 |
| 2023 | market_residual | 272 | 7.352993 | 7.371335 | 0.018342 | 3.925388 | infinity | 5 |
| 2023 | model_centred_lattice | 272 | 7.366884 | 7.384229 | 0.017344 | infinity | infinity | 20 |
| 2023 | served_lattice | 272 | 7.346223 | 7.362155 | 0.015932 | 3.717461 | infinity | 16 |
| 2024 | market_residual | 272 | 7.048490 | 7.082203 | 0.033713 | infinity | infinity | 7 |
| 2024 | model_centred_lattice | 272 | 7.075613 | 7.109886 | 0.034273 | infinity | infinity | 18 |
| 2024 | served_lattice | 272 | 7.079948 | 7.114547 | 0.034599 | 3.709432 | infinity | 15 |
| 2025 | market_residual | 272 | 7.027163 | 7.040079 | 0.012916 | 3.866663 | infinity | 5 |
| 2025 | model_centred_lattice | 272 | 7.041862 | 7.054568 | 0.012705 | infinity | infinity | 8 |
| 2025 | served_lattice | 272 | 7.030861 | 7.047681 | 0.016820 | 3.674852 | infinity | 6 |

## Coefficients and their stability

**Measured:** market_residual coefficient is minus the selected training residual mean, recentering its PMF at the market line. Fractional shifts split mass between adjacent integers. Lattice coefficient is exponential tilt theta using production tilted_atoms and the frozen model centre. These coefficients vary by game; the table gives each fold's full range and median. Individual values are in the prediction appendix. No logistic or combined-pick coefficients apply because this unit fits no side-selection model.

| Season | Arm | Read | Min | Median | Max | Max band | Min band games |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | market_residual | OOS | -0.777429 | 0.246285 | 1.856796 | 12.0 | 201 |
| 2020 | market_residual | IS | -0.691906 | 0.393333 | 2.620000 | 11.5 | 200 |
| 2020 | model_centred_lattice | OOS | -0.051890 | -0.000163 | 0.057317 | 14.5 | 200 |
| 2020 | model_centred_lattice | IS | -0.044408 | 0.001615 | 0.058815 | 13.0 | 201 |
| 2020 | served_lattice | OOS | -0.056622 | -0.008777 | 0.057317 | 12.0 | 201 |
| 2020 | served_lattice | IS | -0.059115 | -0.007253 | 0.058815 | 11.5 | 200 |
| 2021 | market_residual | OOS | -1.045455 | 0.327532 | 2.338443 | 11.5 | 200 |
| 2021 | market_residual | IS | -1.054545 | 0.349206 | 2.338592 | 10.0 | 206 |
| 2021 | model_centred_lattice | OOS | -0.052244 | 0.001100 | 0.070125 | 15.0 | 200 |
| 2021 | model_centred_lattice | IS | -0.050428 | 0.001731 | 0.056700 | 13.5 | 202 |
| 2021 | served_lattice | OOS | -0.052244 | -0.011784 | 0.070125 | 11.5 | 200 |
| 2021 | served_lattice | IS | -0.050428 | -0.010954 | 0.056700 | 10.0 | 206 |
| 2022 | market_residual | OOS | -0.784958 | 0.475658 | 2.206468 | 9.5 | 200 |
| 2022 | market_residual | IS | -0.814644 | 0.561441 | 1.875000 | 8.5 | 200 |
| 2022 | model_centred_lattice | OOS | -0.032912 | 0.002699 | 0.065419 | 14.5 | 200 |
| 2022 | model_centred_lattice | IS | -0.035763 | 0.002602 | 0.063836 | 13.5 | 202 |
| 2022 | served_lattice | OOS | -0.037611 | -0.001810 | 0.065419 | 9.5 | 200 |
| 2022 | served_lattice | IS | -0.038456 | -0.002445 | 0.063836 | 8.5 | 200 |
| 2023 | market_residual | OOS | -0.842879 | 0.824257 | 1.487685 | 9.0 | 202 |
| 2023 | market_residual | IS | -0.890152 | 0.389175 | 0.969376 | 8.0 | 200 |
| 2023 | model_centred_lattice | OOS | -0.024036 | 0.003758 | 0.041737 | 11.0 | 201 |
| 2023 | model_centred_lattice | IS | -0.023029 | 0.002108 | 0.038708 | 10.0 | 200 |
| 2023 | served_lattice | OOS | -0.034931 | 0.002860 | 0.043471 | 9.0 | 202 |
| 2023 | served_lattice | IS | -0.035979 | 0.001485 | 0.040385 | 8.0 | 200 |
| 2024 | market_residual | OOS | -0.899183 | 0.482484 | 0.955529 | 10.0 | 204 |
| 2024 | market_residual | IS | -1.403941 | 0.160163 | 1.234375 | 9.5 | 201 |
| 2024 | model_centred_lattice | OOS | -0.036479 | 0.002476 | 0.057299 | 13.0 | 202 |
| 2024 | model_centred_lattice | IS | -0.033716 | 0.000960 | 0.044220 | 12.5 | 200 |
| 2024 | served_lattice | OOS | -0.036479 | 0.003538 | 0.057299 | 10.0 | 204 |
| 2024 | served_lattice | IS | -0.033716 | 0.002657 | 0.044220 | 9.5 | 201 |
| 2025 | market_residual | OOS | -2.077114 | -0.115566 | 0.473094 | 9.0 | 200 |
| 2025 | market_residual | IS | -2.419725 | -0.211316 | 0.625000 | 9.0 | 201 |
| 2025 | model_centred_lattice | OOS | -0.032045 | 0.000281 | 0.049135 | 13.5 | 200 |
| 2025 | model_centred_lattice | IS | -0.028817 | -0.000261 | 0.049694 | 13.5 | 200 |
| 2025 | served_lattice | OOS | -0.031630 | 0.002921 | 0.049135 | 9.0 | 200 |
| 2025 | served_lattice | IS | -0.028785 | 0.002747 | 0.049694 | 9.0 | 201 |

## Predicted-tail calibration

**Measured:** five fixed bands per arm, integer thresholds -70..70, equal total weight per game. Counts are game-equivalent weights, not independent observations. Gap intervals resample the same season blocks. No band is selected as a finding.

| Arm | Predicted tail band | Game-equivalent count | Predicted | Observed | Gap 95% |
| --- | --- | --- | --- | --- | --- |
| market_residual | 0.0-0.2 | 651.01 | 0.027850 | 0.028619 | [-0.004234, 0.005585] |
| market_residual | 0.2-0.4 | 78.61 | 0.290840 | 0.303140 | [-0.010129, 0.034135] |
| market_residual | 0.4-0.6 | 61.41 | 0.502526 | 0.506756 | [-0.007858, 0.015631] |
| market_residual | 0.6-0.8 | 78.01 | 0.702928 | 0.709364 | [-0.018210, 0.028669] |
| market_residual | 0.8-1.0 | 667.95 | 0.975736 | 0.976120 | [-0.003709, 0.004263] |
| served_lattice | 0.0-0.2 | 653.48 | 0.027460 | 0.029368 | [-0.002346, 0.007235] |
| served_lattice | 0.2-0.4 | 80.19 | 0.292378 | 0.310604 | [-0.000489, 0.040609] |
| served_lattice | 0.4-0.6 | 63.62 | 0.502334 | 0.524749 | [0.008932, 0.036628] |
| served_lattice | 0.6-0.8 | 77.62 | 0.704512 | 0.728162 | [0.004007, 0.042421] |
| served_lattice | 0.8-1.0 | 662.09 | 0.974949 | 0.977087 | [0.000014, 0.005466] |
| model_centred_lattice | 0.0-0.2 | 653.31 | 0.027481 | 0.029582 | [-0.002199, 0.007489] |
| model_centred_lattice | 0.2-0.4 | 80.01 | 0.291111 | 0.309636 | [-0.000447, 0.039287] |
| model_centred_lattice | 0.4-0.6 | 63.46 | 0.502387 | 0.520675 | [0.003685, 0.033744] |
| model_centred_lattice | 0.6-0.8 | 78.20 | 0.702243 | 0.727281 | [0.004430, 0.044667] |
| model_centred_lattice | 0.8-1.0 | 662.02 | 0.975037 | 0.977171 | [0.000077, 0.005291] |

## Scope, provenance and next action

**Read:** ROADMAP.md LEAD-66 declares three primary looks. **Measured:** three arms and three paired contrasts plus 15 fixed descriptive calibration cells: 18 reported looks including diagnostics. No parameter search or additional arm was run.

**Measured:** secondary 2009-2019 inventory lacks seasons 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017; matching predictions begin in 2018. The secondary metric is unavailable, not zero. No partial 2018-2019 substitute or full-history rebuild was run.

**Measured:** maximum archived cover/push/loss reconstruction discrepancy 1.36e-13; recovered residual location agrees within each week. Source feature hashes match; each frozen model training date precedes its game. Inputs were hashed before and after evaluation.

**Read:** served_lattice retains production market-line conditioning; model_centred_lattice uses the existing conditioning_line mechanism at the same recovered model centre. OOS pools omit the whole target season, so this is a chronological LOSO mapping reconstruction, not a replay of the production in-season reference pool. Historical reference lines use archived openers when available and feature-table spreads otherwise, as production prior_pool does; all target grades use the opener.

**Inferred:** results concern PMF mapping conditional on frozen forecasts. They establish no new fitted pick model, profitable edge or serving change. AGENTS.md requires one fitted calibrated probability to choose the served side; no serving decision changes here. The served-versus-market contrast remains unresolved_below_power. Both model-centred contrasts have wholly negative 95% intervals; their proposed registry classification is refuted_mechanism with closing ground wrong_sign_resolved. These are pending commands for the orchestrator, not recorded closures. The LEAD-66 research programme itself remains open.

**Read:** weak_signals.py EFFECT_UNITS does not include rps_improvement, required by the roadmap. Exact commands in the lane remain unexecuted and parser-blocked until the orchestrator resolves that schema. Do not substitute accuracy, Brier, MAE or log-loss units.

Prediction-level output: docs/lead66_prediction_scores.md. Source model: b578fbea1c5c706f. Fixed band 2.5; minimum 200; step 0.5; cap 20.0.

| Input | SHA256 |
| --- | --- |
| artifacts\margins\20260929T192312Z\metadata.json | c13201ffa44d2fe4f2ddb32eac60e4b2c24616c0ae776e2227303c123ff18256 |
| artifacts\margins\20260929T192312Z\predictions.parquet | 93a0ad9e446e152eeebe4de34043d7b7bcca0cc1a3899ef8d77a358e12ab1fa3 |
| artifacts\opener_evaluation\20260929T192743Z\metadata.json | 2b2d31e2edf683dd0fb3654d6681f7bc564b4b2600a929a20d861fc5be6237d2 |
| artifacts\opener_evaluation\20260929T192743Z\per_game.parquet | de3b0ef63819e3416b4c6698e564b22a63cf43dcbfa6bc94a27f9ee8a477a201 |
| data\processed\game_features_weak_stack.parquet | a3eedb0323818f95b029047c5b80de3ce8d458a48b82ae4489c1fb70d16d5685 |

## Pending record payloads

| Signal | Estimate | CI lower | CI upper | probability_positive | Blocks |
| --- | --- | --- | --- | --- | --- |
| lead66_served_lattice_vs_market_residual | -0.0263092311 | -0.0706287621 | 0.0046126176 | 0.0765000000 | 6 |
| lead66_model_centred_lattice_vs_market_residual | -0.0377183695 | -0.0818652969 | -0.0109540449 | 0.0000000000 | 6 |
| lead66_model_centred_lattice_vs_served_lattice | -0.0114091385 | -0.0190492490 | -0.0037943140 | 0.0029000000 | 6 |
