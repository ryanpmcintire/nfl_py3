# LEAD-85 unit 1: integrated coefficient probability adapter

**Measured:** `.tools/uv.exe run --no-sync python scripts/lead85_unit1.py`; protocol saved before outcomes in `docs/lanes/lead85.md`.

## Decisive games first

| Outer season | Decisive n | Candidate W-L | Four-term / candidate wins | Nonpush n |
| --- | --- | --- | --- | --- |
| 2023 | 0 | 0-0 | 158 / 158 | 266 |
| 2024 | 0 | 0-0 | 148 / 148 | 266 |
| 2025 | 0 | 0-0 | 150 / 150 | 267 |
| pooled | 0 | 0-0 | 456 / 456 | 799 |

**Inferred:** symmetric Gaussian logit integration and positive temperature retain each point-fit side; heterogeneous uncertainty can still change weekly ranking. Zero decisive games provide no accuracy mechanism test.

## Population and chronology

**Measured:** 1537 frozen opener rows; 1531 source-complete games, including 34 pushes; 6 excluded for missing dated moves. No historical pool captures required.
**Measured:** counts {2020: 225, 2021: 237, 2022: 253, 2023: 272, 2024: 272, 2025: 272}; 455 verified early quote files, 11362 admissible early rows. Max late-move error 0; archived PMF error 2.13e-13.
**Read:** frozen upstream per-game training dates must precede their game; archived PMFs are reconstructed from completed prior games. These previously examined archives and upstream feature choices are retrospective, not untouched prospective evidence.
**Measured:** outer 2023/2024/2025; combined fit through Y-3, reserve Y-2, calibrate on Y-1. Existing ridge 0.001 is fixed with no search. Both arms use identical source-complete rows. Pushes remain in PMFs/RPS and are excluded from conditional fitting/scoring.

## Adapter and uncertainty

**Measured:** inverse penalized Hessian, fixed 20-node Gauss-Hermite integration, same PMF reweighted within each home/away region at every node with unchanged push mass. Explicit node mixtures match integrated probabilities. Separate positive temperatures use the same calibration season and coherently reweight each final PMF.
**Measured:** 10,000 paired hierarchical season/week bootstrap draws, seed 85; 95% percentile intervals, half-weight zero effects. Fixed predictions are not refitted. Gap is OOS minus optimistic IS; training games repeated across folds stay in their original week blocks.
**Read:** 713 declared study looks: (27*6+6+4)*(3+1)+25. Unit 1 reports two arms and four ordinary endpoints; nominee-Brier primary, weekly reward and all five arms belong to the row's unit 2. No variant is selected from these results.

## In-sample, held-out and gap

| Arm | Metric | IS [95% interval] | OOS [95% interval] | OOS-IS [95% interval] |
| --- | --- | --- | --- | --- |
| four_term | accuracy_points | 55.774854 [51.857444, 60.195495] | 57.071339 [53.409091, 60.714396] | 1.296485 [-4.362711, 6.567294] |
| four_term | brier | 0.244686 [0.238852, 0.249492] | 0.245609 [0.240746, 0.250529] | 0.000923 [-0.005835, 0.008512] |
| four_term | log_loss | 0.682312 [0.670484, 0.692108] | 0.684473 [0.674556, 0.694496] | 0.002161 [-0.011671, 0.017601] |
| four_term | rps | 7.193047 [6.388013, 8.052093] | 7.098231 [6.698407, 7.531081] | -0.094816 [-1.058524, 0.823136] |
| integrated | accuracy_points | 55.774854 [51.857444, 60.195495] | 57.071339 [53.409091, 60.714396] | 1.296485 [-4.362711, 6.567294] |
| integrated | brier | 0.244725 [0.238837, 0.249570] | 0.245552 [0.240651, 0.250483] | 0.000828 [-0.005976, 0.008444] |
| integrated | log_loss | 0.682400 [0.670432, 0.692262] | 0.684344 [0.674390, 0.694433] | 0.001944 [-0.011941, 0.017481] |
| integrated | rps | 7.193978 [6.388386, 8.052185] | 7.097876 [6.698298, 7.530837] | -0.096102 [-1.058645, 0.822754] |

## Paired held-out improvement over the fitted four-term recipe

| Panel | Metric | Improvement [95% interval] | probability_positive |
| --- | --- | --- | --- |
| 2023 | accuracy_points | 0.000000 [0.000000, 0.000000] | 0.500000 |
| 2023 | brier | 0.000154 [-0.000000, 0.000318] | 0.974900 |
| 2023 | log_loss | 0.000355 [-0.000001, 0.000746] | 0.974700 |
| 2023 | rps | 0.001435 [-0.000382, 0.003412] | 0.933200 |
| 2024 | accuracy_points | 0.000000 [0.000000, 0.000000] | 0.500000 |
| 2024 | brier | 0.000041 [-0.000047, 0.000136] | 0.809400 |
| 2024 | log_loss | 0.000089 [-0.000099, 0.000295] | 0.814000 |
| 2024 | rps | 0.000053 [-0.000802, 0.000888] | 0.555000 |
| 2025 | accuracy_points | 0.000000 [0.000000, 0.000000] | 0.500000 |
| 2025 | brier | -0.000027 [-0.000070, 0.000014] | 0.113300 |
| 2025 | log_loss | -0.000058 [-0.000151, 0.000028] | 0.105900 |
| 2025 | rps | -0.000422 [-0.000878, 0.000048] | 0.039900 |
| pooled | accuracy_points | 0.000000 [0.000000, 0.000000] | 0.500000 |
| pooled | brier | 0.000056 [-0.000031, 0.000176] | 0.846500 |
| pooled | log_loss | 0.000129 [-0.000067, 0.000403] | 0.848700 |
| pooled | rps | 0.000355 [-0.000525, 0.001685] | 0.676700 |

## Fold coefficients and calibration

**Measured:** natural-scale point coefficients and Laplace 95% intervals. Both arms share the combined fit. Availability is constant in this source-complete population, so its coefficient is unidentified by the likelihood; ridge supplies precision. Full covariance and standardizers are saved in the scratch summary.

| Fold | Train n | Coefficient | Estimate [95% interval] |
| --- | --- | --- | --- |
| 2023 | 218 | model_logit | -0.006681 [-0.886177, 0.872815] |
| 2023 | 218 | composition_flag_sum | 0.363213 [0.009087, 0.717339] |
| 2023 | 218 | market_move_toward_home | 0.048984 [-0.132864, 0.230832] |
| 2023 | 218 | market_move_available | 0.000000 [-61.980642, 61.980642] |
| 2023 | 218 | intercept | -0.113420 [-62.094962, 61.868123] |
| 2024 | 452 | model_logit | 0.370792 [-0.271486, 1.013069] |
| 2024 | 452 | composition_flag_sum | 0.260210 [0.030957, 0.489463] |
| 2024 | 452 | market_move_toward_home | 0.061742 [-0.058919, 0.182402] |
| 2024 | 452 | market_move_available | 0.000000 [-61.980642, 61.980642] |
| 2024 | 452 | intercept | -0.030925 [-62.012030, 61.950181] |
| 2025 | 698 | model_logit | 0.362850 [-0.148936, 0.874637] |
| 2025 | 698 | composition_flag_sum | 0.287314 [0.108833, 0.465796] |
| 2025 | 698 | market_move_toward_home | 0.090751 [-0.012021, 0.193523] |
| 2025 | 698 | market_move_available | 0.000000 [-61.980642, 61.980642] |
| 2025 | 698 | intercept | -0.041454 [-62.022356, 61.939448] |

| Fold | Cal n | Four-term inverse temperature | Integrated inverse temperature | Hessian condition |
| --- | --- | --- | --- | --- |
| 2023 | 246 | 1.052180 | 1.081192 | 57753.9 |
| 2024 | 266 | 1.226058 | 1.252461 | 115297 |
| 2025 | 266 | 0.988894 | 1.000000 | 176696 |

## Five equal-width reliability bands

| Arm | Band | n | Mean probability | Observed home cover |
| --- | --- | --- | --- | --- |
| four_term | 0.0-0.2 | 0 | n/a | n/a |
| four_term | 0.2-0.4 | 96 | 0.363264 | 0.447917 |
| four_term | 0.4-0.6 | 636 | 0.499811 | 0.498428 |
| four_term | 0.6-0.8 | 67 | 0.638683 | 0.686567 |
| four_term | 0.8-1.0 | 0 | n/a | n/a |
| integrated | 0.0-0.2 | 0 | n/a | n/a |
| integrated | 0.2-0.4 | 96 | 0.362997 | 0.447917 |
| integrated | 0.4-0.6 | 634 | 0.499446 | 0.498423 |
| integrated | 0.6-0.8 | 69 | 0.637394 | 0.681159 |
| integrated | 0.8-1.0 | 0 | n/a | n/a |

## State and next unit

**Inferred:** unresolved_below_power; the predeclared nominee-Brier primary endpoint is unmeasured in unit 1. Zero crossing closes nothing. No promotion, rejection, served change or wagering. The orchestrator records the diagnostic serially; no registry command was executed here.
Prediction rows, OOS discrete PMFs, source hashes, coefficients and all fold/pooled IS/OOS/gap intervals are in `tests/scratch/codex/lead85_unit1/`. Unit 2 replays fixed weekly contenders against four-term, model-only, timestamp-matched market and Elo, with the existing push/tie/reward rules and the declared 713-look family.
