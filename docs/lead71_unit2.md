# LEAD-71 unit 2: moneyline, spread and total margin lattice

**Measured:** two reserved contrasts executed; research only; served card unchanged.
**Inferred:** records remain proposed until the orchestrator runs the lane commands.

## Decisive opener records and scores

**Measured:** cover scores exclude pushes; look-1 whole-margin scores include pushes.
Accuracy intervals are Wilson 95%; improvement intervals use paired season blocks.

| Look | Scheme | Arm | Games | W-L | Accuracy [95% CI] | Cover log loss | Brier | RPS | Margin log score | Zero-mass outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | IS | implied_lattice | 1480 | 727-719 | 50.28% [47.70%, 52.85%] | 0.693283 | 0.250087 | 7.069152 | 3.718252 | 0 |
| 1 | IS | market_residual | 1480 | 709-737 | 49.03% [46.46%, 51.61%] | 0.693160 | 0.250008 | 7.075087 | infinity | 6 |
| 1 | IS | served_lattice | 1480 | 765-681 | 52.90% [50.33%, 55.47%] | 0.695876 | 0.251208 | 7.102634 | 3.700140 | 0 |
| 1 | OOS | implied_lattice | 1480 | 727-719 | 50.28% [47.70%, 52.85%] | 0.693161 | 0.250025 | 7.090196 | 4.108286 | 0 |
| 1 | OOS | market_residual | 1480 | 703-743 | 48.62% [46.05%, 51.19%] | 0.693520 | 0.250186 | 7.096387 | infinity | 24 |
| 1 | OOS | served_lattice | 1480 | 762-684 | 52.70% [50.12%, 55.26%] | 0.697027 | 0.251767 | 7.125054 | infinity | 65 |
| 2 | IS | four_plus_implied | 1273 | 734-539 | 57.66% [54.93%, 60.35%] | 0.681360 | 0.244157 | 7.172914 | 3.719867 | 0 |
| 2 | IS | four_term | 1273 | 732-541 | 57.50% [54.77%, 60.19%] | 0.680856 | 0.243914 | 7.166756 | 3.719364 | 0 |
| 2 | IS | market_residual | 1273 | 627-646 | 49.25% [46.51%, 52.00%] | 0.692680 | 0.249768 | 7.243139 | infinity | 4 |
| 2 | IS | served_lattice | 1273 | 676-597 | 53.10% [50.36%, 55.83%] | 0.693456 | 0.250065 | 7.267608 | 3.731964 | 0 |
| 2 | IS | simple_logit | 1273 | 679-594 | 53.34% [50.59%, 56.06%] | 0.691330 | 0.249092 | 7.225566 | 3.729838 | 0 |
| 2 | OOS | four_plus_implied | 1273 | 720-553 | 56.56% [53.82%, 59.26%] | 0.686511 | 0.246634 | 7.226297 | infinity | 59 |
| 2 | OOS | four_term | 1273 | 732-541 | 57.50% [54.77%, 60.19%] | 0.685338 | 0.246098 | 7.213208 | infinity | 59 |
| 2 | OOS | market_residual | 1273 | 620-653 | 48.70% [45.97%, 51.45%] | 0.693407 | 0.250130 | 7.265449 | infinity | 22 |
| 2 | OOS | served_lattice | 1273 | 672-601 | 52.79% [50.04%, 55.52%] | 0.694814 | 0.250731 | 7.290955 | infinity | 59 |
| 2 | OOS | simple_logit | 1273 | 655-618 | 51.45% [48.71%, 54.19%] | 0.693200 | 0.250024 | 7.248778 | infinity | 59 |

## Paired improvements

**Measured:** positive means the challenger improves on its reserved comparator.

| Look | Scheme | Metric | Primary | Improvement | 95% season-block CI | probability_positive |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | OOS | rps | True | 0.034858 | [-0.000318, 0.085864] | 0.974100 |
| 1 | OOS | log_loss | False | 0.003866 | [-0.001174, 0.011102] | 0.914700 |
| 1 | OOS | brier | False | 0.001742 | [-0.000678, 0.005123] | 0.902500 |
| 1 | OOS | accuracy | False | -0.024205 | [-0.051266, 0.006354] | 0.062450 |
| 1 | IS | rps | True | 0.033482 | [0.006233, 0.078552] | 0.999900 |
| 1 | IS | log_loss | False | 0.002593 | [-0.003120, 0.010510] | 0.769900 |
| 1 | IS | brier | False | 0.001121 | [-0.001631, 0.004864] | 0.750100 |
| 1 | IS | accuracy | False | -0.026279 | [-0.052632, 0.000000] | 0.028150 |
| 2 | OOS | rps | False | -0.013089 | [-0.024229, -0.005955] | 0.000000 |
| 2 | OOS | log_loss | True | -0.001173 | [-0.001984, -0.000436] | 0.000000 |
| 2 | OOS | brier | False | -0.000537 | [-0.000876, -0.000208] | 0.000000 |
| 2 | OOS | accuracy | False | -0.009427 | [-0.015188, -0.003180] | 0.000150 |
| 2 | IS | rps | False | -0.006158 | [-0.009387, -0.004198] | 0.000000 |
| 2 | IS | log_loss | True | -0.000503 | [-0.001002, -0.000021] | 0.008800 |
| 2 | IS | brier | False | -0.000242 | [-0.000497, 0.000001] | 0.028500 |
| 2 | IS | accuracy | False | 0.001571 | [-0.014471, 0.014937] | 0.598850 |

## OOS-minus-IS gaps

**Measured:** same-game score differences.

| Look | Arm | Log loss gap | Brier gap | RPS gap | Accuracy gap |
| --- | --- | --- | --- | --- | --- |
| 1 | implied_lattice | -0.000122 | -0.000062 | 0.021044 | 0.000000 |
| 1 | market_residual | 0.000360 | 0.000179 | 0.021300 | -0.004149 |
| 1 | served_lattice | 0.001151 | 0.000559 | 0.022420 | -0.002075 |
| 2 | four_plus_implied | 0.005152 | 0.002478 | 0.053384 | -0.010998 |
| 2 | four_term | 0.004481 | 0.002183 | 0.046452 | 0.000000 |
| 2 | market_residual | 0.000727 | 0.000362 | 0.022310 | -0.005499 |
| 2 | served_lattice | 0.001358 | 0.000666 | 0.023347 | -0.003142 |
| 2 | simple_logit | 0.001871 | 0.000932 | 0.023212 | -0.018853 |

## Season stability

**Measured:** descriptive season summaries; no season selected for its result.

| Look | Season | Arm | Games | W-L | Accuracy [95% CI] | Cover log loss | Brier | RPS | Margin log score | Zero-mass outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2020 | implied_lattice | 180 | 95-78 | 54.91% [47.47%, 62.14%] | 0.689166 | 0.248163 | 6.566797 | 4.080361 | 0 |
| 1 | 2020 | market_residual | 180 | 83-90 | 47.98% [40.66%, 55.38%] | 0.694347 | 0.250600 | 6.554213 | infinity | 2 |
| 1 | 2020 | served_lattice | 180 | 90-83 | 52.02% [44.62%, 59.34%] | 0.713309 | 0.259390 | 6.581412 | infinity | 6 |
| 1 | 2021 | implied_lattice | 229 | 126-100 | 55.75% [49.23%, 62.08%] | 0.693244 | 0.250048 | 8.003368 | 4.238876 | 0 |
| 1 | 2021 | market_residual | 229 | 100-126 | 44.25% [37.92%, 50.77%] | 0.697465 | 0.252157 | 8.032344 | infinity | 4 |
| 1 | 2021 | served_lattice | 229 | 127-99 | 56.19% [49.68%, 62.51%] | 0.688440 | 0.247538 | 8.011613 | infinity | 11 |
| 1 | 2022 | implied_lattice | 255 | 137-111 | 55.24% [49.02%, 61.30%] | 0.693071 | 0.249961 | 6.404085 | 3.980944 | 0 |
| 1 | 2022 | market_residual | 255 | 134-114 | 54.03% [47.82%, 60.13%] | 0.689054 | 0.247957 | 6.420486 | infinity | 1 |
| 1 | 2022 | served_lattice | 255 | 132-116 | 53.23% [47.01%, 59.34%] | 0.692566 | 0.249587 | 6.553466 | infinity | 11 |
| 1 | 2023 | implied_lattice | 272 | 123-143 | 46.24% [40.35%, 52.24%] | 0.693285 | 0.250069 | 7.379565 | 4.216719 | 0 |
| 1 | 2023 | market_residual | 272 | 126-140 | 47.37% [41.45%, 53.36%] | 0.693891 | 0.250369 | 7.371335 | infinity | 5 |
| 1 | 2023 | served_lattice | 272 | 134-132 | 50.38% [44.41%, 56.34%] | 0.700584 | 0.253531 | 7.362155 | infinity | 16 |
| 1 | 2024 | implied_lattice | 272 | 123-143 | 46.24% [40.35%, 52.24%] | 0.693465 | 0.250159 | 7.083845 | 4.178029 | 0 |
| 1 | 2024 | market_residual | 272 | 134-132 | 50.38% [44.41%, 56.34%] | 0.692037 | 0.249447 | 7.082203 | infinity | 7 |
| 1 | 2024 | served_lattice | 272 | 143-123 | 53.76% [47.76%, 59.65%] | 0.695637 | 0.251165 | 7.114547 | infinity | 15 |
| 1 | 2025 | implied_lattice | 272 | 123-144 | 46.07% [40.19%, 52.06%] | 0.695335 | 0.251093 | 7.027963 | 3.958028 | 0 |
| 1 | 2025 | market_residual | 272 | 126-141 | 47.19% [41.29%, 53.18%] | 0.694898 | 0.250876 | 7.040079 | infinity | 5 |
| 1 | 2025 | served_lattice | 272 | 136-131 | 50.94% [44.97%, 56.88%] | 0.695729 | 0.251274 | 7.047681 | infinity | 6 |
| 2 | 2021 | four_plus_implied | 226 | 124-102 | 54.87% [48.35%, 61.22%] | 0.699413 | 0.252762 | 8.130365 | infinity | 11 |
| 2 | 2021 | four_term | 226 | 127-99 | 56.19% [49.68%, 62.51%] | 0.696897 | 0.251709 | 8.094435 | infinity | 11 |
| 2 | 2021 | market_residual | 226 | 100-126 | 44.25% [37.92%, 50.77%] | 0.697465 | 0.252157 | 8.104049 | infinity | 4 |
| 2 | 2021 | served_lattice | 226 | 127-99 | 56.19% [49.68%, 62.51%] | 0.688440 | 0.247538 | 8.078027 | infinity | 11 |
| 2 | 2021 | simple_logit | 226 | 112-114 | 49.56% [43.10%, 56.03%] | 0.697532 | 0.252182 | 8.066560 | infinity | 11 |
| 2 | 2022 | four_plus_implied | 248 | 149-99 | 60.08% [53.88%, 65.98%] | 0.678772 | 0.242866 | 6.515267 | infinity | 11 |
| 2 | 2022 | four_term | 248 | 150-98 | 60.48% [54.28%, 66.36%] | 0.678771 | 0.242846 | 6.509516 | infinity | 11 |
| 2 | 2022 | market_residual | 248 | 134-114 | 54.03% [47.82%, 60.13%] | 0.689054 | 0.247957 | 6.524289 | infinity | 1 |
| 2 | 2022 | served_lattice | 248 | 132-116 | 53.23% [47.01%, 59.34%] | 0.692566 | 0.249587 | 6.659248 | infinity | 11 |
| 2 | 2022 | simple_logit | 248 | 133-115 | 53.63% [47.41%, 59.73%] | 0.690830 | 0.248842 | 6.577634 | infinity | 11 |
| 2 | 2023 | four_plus_implied | 266 | 151-115 | 56.77% [50.76%, 62.58%] | 0.685565 | 0.246105 | 7.407143 | infinity | 16 |
| 2 | 2023 | four_term | 266 | 151-115 | 56.77% [50.76%, 62.58%] | 0.684710 | 0.245719 | 7.393582 | infinity | 16 |
| 2 | 2023 | market_residual | 266 | 126-140 | 47.37% [41.45%, 53.36%] | 0.693891 | 0.250369 | 7.474973 | infinity | 5 |
| 2 | 2023 | served_lattice | 266 | 134-132 | 50.38% [44.41%, 56.34%] | 0.700584 | 0.253531 | 7.461961 | infinity | 16 |
| 2 | 2023 | simple_logit | 266 | 128-138 | 48.12% [42.19%, 54.11%] | 0.694387 | 0.250613 | 7.434749 | infinity | 16 |
| 2 | 2024 | four_plus_implied | 266 | 148-118 | 55.64% [49.63%, 61.49%] | 0.684335 | 0.245472 | 7.145411 | infinity | 15 |
| 2 | 2024 | four_term | 266 | 153-113 | 57.52% [51.51%, 63.31%] | 0.683703 | 0.245178 | 7.140338 | infinity | 15 |
| 2 | 2024 | market_residual | 266 | 134-132 | 50.38% [44.41%, 56.34%] | 0.692037 | 0.249447 | 7.180347 | infinity | 7 |
| 2 | 2024 | served_lattice | 266 | 143-123 | 53.76% [47.76%, 59.65%] | 0.695637 | 0.251165 | 7.210134 | infinity | 15 |
| 2 | 2024 | simple_logit | 266 | 145-121 | 54.51% [48.51%, 60.39%] | 0.691828 | 0.249340 | 7.159333 | infinity | 15 |
| 2 | 2025 | four_plus_implied | 267 | 148-119 | 55.43% [49.43%, 61.27%] | 0.685888 | 0.246633 | 7.021905 | infinity | 6 |
| 2 | 2025 | four_term | 267 | 151-116 | 56.55% [50.56%, 62.37%] | 0.683907 | 0.245662 | 7.013815 | infinity | 6 |
| 2 | 2025 | market_residual | 267 | 126-141 | 47.19% [41.29%, 53.18%] | 0.694898 | 0.250876 | 7.120084 | infinity | 5 |
| 2 | 2025 | served_lattice | 267 | 136-131 | 50.94% [44.97%, 56.88%] | 0.695729 | 0.251274 | 7.121649 | infinity | 6 |
| 2 | 2025 | simple_logit | 267 | 137-130 | 51.31% [45.34%, 57.24%] | 0.691921 | 0.249389 | 7.083792 | infinity | 6 |

## Cover reliability

**Measured:** five fixed bands per arm, including empty bands; descriptive only.

| Look | Arm | Band | Games | Predicted | Observed |
| --- | --- | --- | --- | --- | --- |
| 1 | implied_lattice | 0.0-0.2 | 0 | undefined | undefined |
| 1 | implied_lattice | 0.2-0.4 | 0 | undefined | undefined |
| 1 | implied_lattice | 0.4-0.6 | 1440 | 0.500720 | 0.494444 |
| 1 | implied_lattice | 0.6-0.8 | 5 | 0.667236 | 0.800000 |
| 1 | implied_lattice | 0.8-1.0 | 1 | 0.851330 | 1.000000 |
| 1 | market_residual | 0.0-0.2 | 0 | undefined | undefined |
| 1 | market_residual | 0.2-0.4 | 0 | undefined | undefined |
| 1 | market_residual | 0.4-0.6 | 1446 | 0.498296 | 0.495851 |
| 1 | market_residual | 0.6-0.8 | 0 | undefined | undefined |
| 1 | market_residual | 0.8-1.0 | 0 | undefined | undefined |
| 1 | served_lattice | 0.0-0.2 | 0 | undefined | undefined |
| 1 | served_lattice | 0.2-0.4 | 200 | 0.366424 | 0.500000 |
| 1 | served_lattice | 0.4-0.6 | 1176 | 0.492366 | 0.492347 |
| 1 | served_lattice | 0.6-0.8 | 70 | 0.624260 | 0.542857 |
| 1 | served_lattice | 0.8-1.0 | 0 | undefined | undefined |
| 2 | four_plus_implied | 0.0-0.2 | 0 | undefined | undefined |
| 2 | four_plus_implied | 0.2-0.4 | 180 | 0.355903 | 0.461111 |
| 2 | four_plus_implied | 0.4-0.6 | 966 | 0.496397 | 0.485507 |
| 2 | four_plus_implied | 0.6-0.8 | 127 | 0.649831 | 0.637795 |
| 2 | four_plus_implied | 0.8-1.0 | 0 | undefined | undefined |
| 2 | four_term | 0.0-0.2 | 0 | undefined | undefined |
| 2 | four_term | 0.2-0.4 | 139 | 0.360087 | 0.431655 |
| 2 | four_term | 0.4-0.6 | 1002 | 0.497756 | 0.490020 |
| 2 | four_term | 0.6-0.8 | 132 | 0.646139 | 0.621212 |
| 2 | four_term | 0.8-1.0 | 0 | undefined | undefined |
| 2 | market_residual | 0.0-0.2 | 0 | undefined | undefined |
| 2 | market_residual | 0.2-0.4 | 0 | undefined | undefined |
| 2 | market_residual | 0.4-0.6 | 1273 | 0.498608 | 0.497251 |
| 2 | market_residual | 0.6-0.8 | 0 | undefined | undefined |
| 2 | market_residual | 0.8-1.0 | 0 | undefined | undefined |
| 2 | served_lattice | 0.0-0.2 | 0 | undefined | undefined |
| 2 | served_lattice | 0.2-0.4 | 148 | 0.368516 | 0.486486 |
| 2 | served_lattice | 0.4-0.6 | 1057 | 0.493833 | 0.495743 |
| 2 | served_lattice | 0.6-0.8 | 68 | 0.624673 | 0.544118 |
| 2 | served_lattice | 0.8-1.0 | 0 | undefined | undefined |
| 2 | simple_logit | 0.0-0.2 | 0 | undefined | undefined |
| 2 | simple_logit | 0.2-0.4 | 0 | undefined | undefined |
| 2 | simple_logit | 0.4-0.6 | 1273 | 0.494707 | 0.497251 |
| 2 | simple_logit | 0.6-0.8 | 0 | undefined | undefined |
| 2 | simple_logit | 0.8-1.0 | 0 | undefined | undefined |

## Fitting and price diagnostics

**Measured:** chronological training pools:

| season | games | training_games | training_last_day | in_sample_games |
| --- | --- | --- | --- | --- |
| 2020 | 180 | 1280 | 2019-12-29 | 1536 |
| 2021 | 229 | 1280 | 2021-01-03 | 1552 |
| 2022 | 255 | 1296 | 2022-01-09 | 1567 |
| 2023 | 272 | 1311 | 2023-01-08 | 1583 |
| 2024 | 272 | 1327 | 2024-01-07 | 1599 |
| 2025 | 272 | 1343 | 2025-01-05 | 1615 |

**Measured:** natural-scale coefficients (constant inputs have zero weight):

| season | scheme | arm | training_games | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available | implied_logit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | OOS | simple_logit | 173 | -0.126637 | -0.314547 | undefined | undefined | undefined | undefined |
| 2021 | OOS | four_term | 173 | -0.195176 | -0.275264 | 0.415847 | 0.000000 | 0.000000 | undefined |
| 2021 | OOS | four_plus_implied | 173 | -0.294319 | -0.299158 | 0.466451 | 0.000000 | 0.000000 | 1.444700 |
| 2021 | IS | simple_logit | 399 | -0.034334 | 0.256397 | undefined | undefined | undefined | undefined |
| 2021 | IS | four_term | 399 | -0.070551 | 0.287892 | 0.287889 | 0.000000 | 0.000000 | undefined |
| 2021 | IS | four_plus_implied | 399 | -0.105788 | 0.257026 | 0.302490 | 0.000000 | 0.000000 | 1.168721 |
| 2022 | OOS | simple_logit | 399 | -0.034334 | 0.256397 | undefined | undefined | undefined | undefined |
| 2022 | OOS | four_term | 399 | -0.070551 | 0.287892 | 0.287889 | 0.000000 | 0.000000 | undefined |
| 2022 | OOS | four_plus_implied | 399 | -0.105788 | 0.257026 | 0.302490 | 0.000000 | 0.000000 | 1.168721 |
| 2022 | IS | simple_logit | 647 | -0.011334 | 0.357970 | undefined | undefined | undefined | undefined |
| 2022 | IS | four_term | 647 | -0.060681 | 0.325715 | 0.314552 | 0.000000 | 0.000000 | undefined |
| 2022 | IS | four_plus_implied | 647 | -0.079898 | 0.315946 | 0.323090 | 0.000000 | 0.000000 | 1.156326 |
| 2023 | OOS | simple_logit | 647 | -0.011334 | 0.357970 | undefined | undefined | undefined | undefined |
| 2023 | OOS | four_term | 647 | -0.060681 | 0.325715 | 0.314552 | 0.000000 | 0.000000 | undefined |
| 2023 | OOS | four_plus_implied | 647 | -0.079898 | 0.315946 | 0.323090 | 0.000000 | 0.000000 | 1.156326 |
| 2023 | IS | simple_logit | 913 | -0.004368 | 0.296932 | undefined | undefined | undefined | undefined |
| 2023 | IS | four_term | 913 | -0.077571 | 0.236062 | 0.317169 | 0.210855 | 0.084301 | undefined |
| 2023 | IS | four_plus_implied | 913 | -0.096658 | 0.221717 | 0.322259 | 0.216466 | 0.105942 | 1.116356 |
| 2024 | OOS | simple_logit | 913 | -0.004368 | 0.296932 | undefined | undefined | undefined | undefined |
| 2024 | OOS | four_term | 913 | -0.077571 | 0.236062 | 0.317169 | 0.210855 | 0.084301 | undefined |
| 2024 | OOS | four_plus_implied | 913 | -0.096658 | 0.221717 | 0.322259 | 0.216466 | 0.105942 | 1.116356 |
| 2024 | IS | simple_logit | 1179 | 0.002048 | 0.323190 | undefined | undefined | undefined | undefined |
| 2024 | IS | four_term | 1179 | -0.070518 | 0.262593 | 0.301819 | 0.189122 | 0.063125 | undefined |
| 2024 | IS | four_plus_implied | 1179 | -0.087929 | 0.248476 | 0.306012 | 0.191457 | 0.078827 | 1.002108 |
| 2025 | OOS | simple_logit | 1179 | 0.002048 | 0.323190 | undefined | undefined | undefined | undefined |
| 2025 | OOS | four_term | 1179 | -0.070518 | 0.262593 | 0.301819 | 0.189122 | 0.063125 | undefined |
| 2025 | OOS | four_plus_implied | 1179 | -0.087929 | 0.248476 | 0.306012 | 0.191457 | 0.078827 | 1.002108 |
| 2025 | IS | simple_logit | 1446 | 0.009352 | 0.335086 | undefined | undefined | undefined | undefined |
| 2025 | IS | four_term | 1446 | -0.067161 | 0.251599 | 0.263505 | 0.222248 | 0.049163 | undefined |
| 2025 | IS | four_plus_implied | 1446 | -0.079468 | 0.242372 | 0.266378 | 0.223789 | 0.061105 | 0.707296 |

**Measured:** coefficient stability across the five OOS combination folds:

| Arm | Term | Minimum | Maximum | Positive folds | Negative folds |
| --- | --- | --- | --- | --- | --- |
| four_plus_implied | intercept | -0.294319 | -0.079898 | 0 | 5 |
| four_plus_implied | model_logit | -0.299158 | 0.315946 | 4 | 1 |
| four_plus_implied | composition_flag_sum | 0.302490 | 0.466451 | 5 | 0 |
| four_plus_implied | market_move_toward_home | 0.000000 | 0.216466 | 2 | 0 |
| four_plus_implied | market_move_available | 0.000000 | 0.105942 | 2 | 0 |
| four_plus_implied | implied_logit | 1.002108 | 1.444700 | 5 | 0 |
| four_term | intercept | -0.195176 | -0.060681 | 0 | 5 |
| four_term | model_logit | -0.275264 | 0.325715 | 4 | 1 |
| four_term | composition_flag_sum | 0.287889 | 0.415847 | 5 | 0 |
| four_term | market_move_toward_home | 0.000000 | 0.210855 | 2 | 0 |
| four_term | market_move_available | 0.000000 | 0.084301 | 2 | 0 |
| simple_logit | intercept | -0.126637 | 0.002048 | 1 | 4 |
| simple_logit | model_logit | -0.314547 | 0.357970 | 4 | 1 |

**Measured:** per-game variance scale and price-tilt ranges:

| Season | Scheme | Variance scale min/max | Moneyline tilt min/max | Spread tilt min/max | Max price error | Price error > 0.01 |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | IS | 3.002880/4.426902 | -2.341946/0.777918 | -1.286084/1.912818 | 0.000002 | 0 |
| 2020 | OOS | 3.031862/4.350345 | -2.419624/0.843590 | -1.378886/2.246944 | 0.000002 | 0 |
| 2021 | IS | 3.082296/4.430936 | -0.759976/0.629700 | -0.746312/0.870327 | 0.000001 | 0 |
| 2021 | OOS | 2.940208/4.538032 | -0.874300/0.557001 | -0.610958/0.934749 | 0.000001 | 0 |
| 2022 | IS | 3.154484/4.386878 | -1.359970/1.107837 | -1.199785/1.546495 | 0.000002 | 0 |
| 2022 | OOS | 3.304432/4.510603 | -1.432745/1.567387 | -1.641870/1.619206 | 0.000002 | 0 |
| 2023 | IS | 3.135964/4.005568 | -0.953758/0.936253 | -1.015764/1.093974 | 0.000001 | 0 |
| 2023 | OOS | 3.152826/4.128621 | -0.845922/1.440832 | -1.477761/0.986767 | 0.000002 | 0 |
| 2024 | IS | 2.961171/4.038066 | -1.278917/1.077173 | -1.165576/1.485881 | 0.000002 | 0 |
| 2024 | OOS | 3.194993/4.081893 | -1.056510/1.144384 | -0.880596/1.205902 | 0.000001 | 0 |
| 2025 | IS | 3.074449/4.111423 | -1.147567/1.341224 | -1.133891/1.386853 | 0.000001 | 0 |
| 2025 | OOS | 3.073664/4.063618 | -1.131105/1.354590 | -1.103424/1.411641 | 0.000001 | 0 |

## Interpretation and scope

**Read:** AGENTS.md research rules prohibit closure from a zero-crossing interval and require one fitted probability to select a side.
**Inferred:** these are two fixed-family measurements, not a serving decision. No positive control or split-half reliability adjudication was performed.
**Inferred:** leave the margin-lattice mechanism open. The fixed extra-term extension has a proposed wrong_sign_resolved record when its whole improvement interval is adverse, limited to that extension. AGENTS.md and weak_signals.validate_closure impose that condition; the table above supplies the interval. Registry execution remains with the orchestrator.
**Measured:** look 1 has primary RPS; look 2 has primary cover log loss. The other metrics are descriptive views of the same two contrasts.
Look ledger for LEAD-71-unit2: 2 reserved primary contrasts; 16 arm/scheme views; 40 fixed reliability-band cells; 43 OOS season/arm cells; 16 paired metric readouts (including IS); 30 logistic fits; and 2,960 per-game price projections. These overlap and are not independent votes. All were predetermined views; no best cell changed the family.
IS is an explicitly optimistic same-game diagnostic: the held season enters the lattice pool or logistic fit. Logistic IS retains the frozen OOS implied feature.
Look 2 uses archived prekick situational flags and movement. Only the new market feature freezes Tuesday 09:00 Eastern; the combined model is not Tuesday-only.
Model centres remain frozen. Rows whose training day plus one day is not before the Tuesday cutoff are excluded by the source-timing contract before scoring. Their game-level training chronology and reconstructed probabilities are checked; archived base models are not refitted here.
Historical totals and pre-2020 line proxies enter only completed prior-season OOS pools; held games are graded at the frozen pool opener.
The total market supplies its quoted scoring line. Moneyline and spread prices are proportionally de-vigged; no external retrieval occurs.
Exact margin log scores are infinite for empirical baselines assigning zero mass to an observed margin. Those baselines are not silently smoothed.
Look 2 excludes pushes because its archived input rows do. Look 1 retains pushes for distribution scoring. 2020 initializes chronological calibration.

## Source census and reproduction

Command: .tools/uv.exe run --no-sync python scripts/lead71_unit2.py
Thread limits: 2; one research process; no full-history rebuild, registry write, publication or Git mutation.

- **Measured:** source_rows: 1537.
- **Measured:** archived_first_season: 2018.
- **Measured:** missing_secondary_seasons: [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017].
- **Measured:** reconstruction_max_error: 1.3627987627273797e-13.
- **Measured:** source_model_id: b578fbea1c5c706f.
- **Measured:** model_timestamp_rejections: 46.
- **Measured:** opener_quote_matches_before_timing: 1526.
- **Measured:** quote_games: 1602.
- **Measured:** same_book_games: 1601.
- **Measured:** source_opener_games: 1537.
- **Measured:** matched_opener_games: 1480.
- **Measured:** unavailable_opener_games: 11.
- **Measured:** look1_pushes: 34.
- **Measured:** quote_pool_line_disagreements: 317.

**Measured:** source hashes, prediction-level scores, selected quotes, coefficients and price residuals are saved under tests/scratch/codex/lead71_unit2/; no prediction-row dump is stored in docs.

## Predeclared protocol

— declared before unit 2 outcome reads
Population: completed 2020–2025 REG games in LEAD-66's pinned opener/margin
artifacts, intersect valid unit-1 Tuesday 09:00 Eastern three-market quotes.
Prefer the alphabetically first same-book triplet in the latest eligible snapshot;
otherwise take the first valid book per market. No outcome-based exclusions.
Enforce manifest hashes, source/update <= snapshot <= cutoff < kickoff and unique IDs.
Target: integer home margin; cover/push at the archived pool opener, never the close.
Family: support -100..100; production line-band empirical integer counts plus one
Gaussian pseudocount; reweight by a Gaussian variance ratio using quoted total.
Variance scale = historical squared margin-minus-line residual / historical total;
reference total = historical mean total. Preserve empirical key-number atoms.
Use two exponential payoff tilts to fit de-vigged home moneyline and quoted-spread
conditional probabilities; dual ridge 1e-6, bounds [-40,40], report pricing residuals.
No distribution, hyperparameter, book or band search; production band constants.
LOSO is chronological: lattice fits use only prior five completed seasons, excluding
held season; frozen model centres reconstructed and checked as in LEAD-66.
IS diagnostic adds the held season's outcomes to that same pool; explicitly optimistic.
Look 1: implied lattice versus served lattice; primary RPS improvement, secondary
exact margin log score, conditional-cover log loss/Brier and opener accuracy.
Same-game market-residual PMF and uncalibrated model-only lattice are baselines.
Look 2: production four-term logit versus those terms plus implied-cover logit;
intercept, standardized inputs and ridge 1e-3 as production, no tuning. Prior scored
seasons fit coefficients; 2020 initializes calibration, so OOS scoring is 2021–2025.
Use pinned pick_probability/20260929T192747Z input rows; existing flags and movement
are prekick inputs, not Tuesday features. Only the new market lattice freezes Tuesday.
Primary look-2 metric: cover log-loss improvement; also Brier, accuracy, RPS after
rescaling served non-push tails to the one calibrated probability, keeping push mass.
IS fits include the scored season; report IS/OOS and OOS-minus-IS, fold coefficients,
season stability, fixed cover-reliability quintiles and decisive-game win/loss first.
Two planned contrasts total; secondary metrics/bands are descriptive, not new searches.
Paired season-block bootstrap: 10,000, seed 20260929, game weighted, percentile 95% CI,
probability_positive with half credit for ties; accuracy also Wilson CI.
Zero crossing never closes a signal; no flip rule; no serving or promotion decision.
