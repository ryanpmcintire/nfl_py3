# LEAD-67 unit 1

**Measured:** 799 opener-graded games; seasons 2023/2024/2025 = 266/266/267. Two predeclared experimental looks: replacement and additive. All results are offline.

## Decisive-game record first

| Arm vs base | Wins | Losses | Disagreements |
| --- | --- | --- | --- |
| replacement | 40 | 51 | 91 |
| additive | 9 | 14 | 23 |

## Metrics

**Measured:** IS fits all 799 rows; OOS holds each season out. Accuracy is a fraction; line move is points toward the selected side. A 0.5 market tie selects home solely for the mandatory accuracy/movement benchmark; it has no directional edge.

| Arm | Protocol | Correct | Accuracy | Brier | Log loss | Line move |
| --- | --- | --- | --- | --- | --- | --- |
| base | is | 458/799 | 0.573217 | 0.243798 | 0.680511 | 0.594493 |
| base | oos | 458/799 | 0.573217 | 0.244680 | 0.682352 | 0.598248 |
| replacement | is | 448/799 | 0.560701 | 0.244965 | 0.683030 | 0.624531 |
| replacement | oos | 447/799 | 0.559449 | 0.245625 | 0.684390 | 0.636421 |
| additive | is | 457/799 | 0.571965 | 0.243773 | 0.680473 | 0.612015 |
| additive | oos | 453/799 | 0.566959 | 0.244851 | 0.682764 | 0.605131 |
| model | is | 416/799 | 0.520651 | 0.251974 | 0.697288 | 0.114518 |
| model | oos | 416/799 | 0.520651 | 0.251974 | 0.697288 | 0.114518 |
| market | is | 406/799 | 0.508135 | 0.250000 | 0.693147 | 0.156446 |
| market | oos | 406/799 | 0.508135 | 0.250000 | 0.693147 | 0.156446 |

## Paired gains, intervals, and in-sample optimism

**Measured:** positive effects favor the challenger. Gap = IS gain minus OOS gain. Season blocks are primary; season-week blocks are sensitivity. 10,000 draws, seed 6701; 95% percentile intervals; zero draws receive half credit in probability_positive.

| Arm | Metric | Block | OOS gain | 95% interval | probability_positive | IS gain | IS-OOS gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| replacement | accuracy | season | -0.013767 | [-0.030075, 0.003745] | 0.0360 | -0.012516 | 0.001252 |
| replacement | accuracy | season-week | -0.013767 | [-0.037313, 0.008952] | 0.1218 | -0.012516 | 0.001252 |
| replacement | brier | season | -0.000945 | [-0.002215, 0.000508] | 0.0355 | -0.001167 | -0.000222 |
| replacement | brier | season-week | -0.000945 | [-0.003487, 0.001499] | 0.2359 | -0.001167 | -0.000222 |
| replacement | log_loss | season | -0.002039 | [-0.004992, 0.001378] | 0.1441 | -0.002519 | -0.000481 |
| replacement | log_loss | season-week | -0.002039 | [-0.007352, 0.003064] | 0.2299 | -0.002519 | -0.000481 |
| replacement | line_move | season | 0.038173 | [0.018797, 0.059925] | 1.0000 | 0.030038 | -0.008135 |
| replacement | line_move | season-week | 0.038173 | [-0.019159, 0.097118] | 0.9058 | 0.030038 | -0.008135 |
| additive | accuracy | season | -0.006258 | [-0.018797, 0.000000] | 0.1489 | -0.001252 | 0.005006 |
| additive | accuracy | season-week | -0.006258 | [-0.018774, 0.006321] | 0.1681 | -0.001252 | 0.005006 |
| additive | brier | season | -0.000171 | [-0.000291, -0.000004] | 0.0000 | 0.000025 | 0.000196 |
| additive | brier | season-week | -0.000171 | [-0.000711, 0.000356] | 0.2676 | 0.000025 | 0.000196 |
| additive | log_loss | season | -0.000413 | [-0.000651, -0.000056] | 0.0000 | 0.000038 | 0.000451 |
| additive | log_loss | season-week | -0.000413 | [-0.001524, 0.000666] | 0.2272 | 0.000038 | 0.000451 |
| additive | line_move | season | 0.006884 | [-0.084586, 0.078652] | 0.6370 | 0.017522 | 0.010638 |
| additive | line_move | season-week | 0.006884 | [-0.037268, 0.045963] | 0.6397 | 0.017522 | 0.010638 |

## Season stability

| Season | Arm | N | Correct | Decisive W | Decisive L | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | base | 266 | 151 | 0 | 0 | 0.243069 | 0.679210 |
| 2023 | replacement | 266 | 143 | 13 | 21 | 0.244192 | 0.681700 |
| 2023 | additive | 266 | 151 | 2 | 2 | 0.243073 | 0.679266 |
| 2024 | base | 266 | 159 | 0 | 0 | 0.245594 | 0.684547 |
| 2024 | replacement | 266 | 155 | 16 | 20 | 0.245087 | 0.683169 |
| 2024 | additive | 266 | 154 | 3 | 8 | 0.245885 | 0.685198 |
| 2025 | base | 267 | 148 | 0 | 0 | 0.245374 | 0.683295 |
| 2025 | replacement | 267 | 149 | 11 | 10 | 0.247589 | 0.688287 |
| 2025 | additive | 267 | 148 | 4 | 4 | 0.245591 | 0.683825 |

## Natural coefficients

**Measured:** availability is constant one; its standardized coefficient is zero.

| Arm | Fold | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available | dP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | in_sample | -0.014058 | 0.171468 | 0.220145 | 0.225153 | 0.000000 | absent |
| replacement | in_sample | -0.012053 | 0.145652 | 0.227759 | absent | 0.000000 | 4.526732 |
| additive | in_sample | -0.015122 | 0.163193 | 0.220429 | 0.206164 | 0.000000 | 0.633203 |
| base | 2023 | -0.028997 | 0.291782 | 0.168564 | 0.234173 | 0.000000 | absent |
| replacement | 2023 | -0.021455 | 0.272257 | 0.178917 | absent | 0.000000 | 4.678544 |
| additive | 2023 | -0.030448 | 0.281751 | 0.169182 | 0.208981 | 0.000000 | 0.869643 |
| base | 2024 | -0.010537 | 0.069390 | 0.204199 | 0.252537 | 0.000000 | absent |
| replacement | 2024 | -0.001676 | 0.061785 | 0.209804 | absent | 0.000000 | 4.614166 |
| additive | 2024 | -0.009670 | 0.083094 | 0.204847 | 0.283651 | 0.000000 | -1.048057 |
| base | 2025 | -0.007380 | 0.162071 | 0.284877 | 0.191006 | 0.000000 | absent |
| replacement | 2025 | -0.012765 | 0.127197 | 0.291480 | absent | 0.000000 | 4.322083 |
| additive | 2025 | -0.010470 | 0.143809 | 0.286266 | 0.146154 | 0.000000 | 1.506031 |

| Arm | Term | LOSO min | LOSO max | Positive folds | Negative folds |
| --- | --- | --- | --- | --- | --- |
| base | intercept | -0.028997 | -0.007380 | 0 | 3 |
| base | model_logit | 0.069390 | 0.291782 | 3 | 0 |
| base | composition_flag_sum | 0.168564 | 0.284877 | 3 | 0 |
| base | market_move_toward_home | 0.191006 | 0.252537 | 3 | 0 |
| base | market_move_available | 0.000000 | 0.000000 | 0 | 0 |
| replacement | intercept | -0.021455 | -0.001676 | 0 | 3 |
| replacement | model_logit | 0.061785 | 0.272257 | 3 | 0 |
| replacement | composition_flag_sum | 0.178917 | 0.291480 | 3 | 0 |
| replacement | market_move_available | 0.000000 | 0.000000 | 0 | 0 |
| replacement | dP | 4.322083 | 4.678544 | 3 | 0 |
| additive | intercept | -0.030448 | -0.009670 | 0 | 3 |
| additive | model_logit | 0.083094 | 0.281751 | 3 | 0 |
| additive | composition_flag_sum | 0.169182 | 0.286266 | 3 | 0 |
| additive | market_move_toward_home | 0.146154 | 0.283651 | 3 | 0 |
| additive | market_move_available | 0.000000 | 0.000000 | 0 | 0 |
| additive | dP | -1.048057 | 1.506031 | 2 | 1 |

## Reliability

**Measured:** fixed 0.1-wide home-probability bins; all nonempty bins shown. No bin outcome selected a model.

| Arm | Bin | N | Mean predicted | Observed home cover |
| --- | --- | --- | --- | --- |
| base | 0.2-0.3 | 9 | 0.268275 | 0.444444 |
| base | 0.3-0.4 | 54 | 0.366684 | 0.407407 |
| base | 0.4-0.5 | 313 | 0.462395 | 0.434505 |
| base | 0.5-0.6 | 335 | 0.540122 | 0.567164 |
| base | 0.6-0.7 | 74 | 0.633368 | 0.554054 |
| base | 0.7-0.8 | 14 | 0.736568 | 0.928571 |
| replacement | 0.2-0.3 | 4 | 0.272888 | 0.250000 |
| replacement | 0.3-0.4 | 39 | 0.374112 | 0.384615 |
| replacement | 0.4-0.5 | 336 | 0.460736 | 0.455357 |
| replacement | 0.5-0.6 | 340 | 0.541312 | 0.570588 |
| replacement | 0.6-0.7 | 73 | 0.630921 | 0.534247 |
| replacement | 0.7-0.8 | 7 | 0.716841 | 0.571429 |
| additive | 0.2-0.3 | 10 | 0.269719 | 0.500000 |
| additive | 0.3-0.4 | 54 | 0.368321 | 0.351852 |
| additive | 0.4-0.5 | 315 | 0.461648 | 0.450794 |
| additive | 0.5-0.6 | 327 | 0.539746 | 0.559633 |
| additive | 0.6-0.7 | 80 | 0.633902 | 0.587500 |
| additive | 0.7-0.8 | 13 | 0.737497 | 0.769231 |
| model | 0.2-0.3 | 1 | 0.284015 | 1.000000 |
| model | 0.3-0.4 | 48 | 0.376546 | 0.520833 |
| model | 0.4-0.5 | 331 | 0.458284 | 0.480363 |
| model | 0.5-0.6 | 361 | 0.540546 | 0.529086 |
| model | 0.6-0.7 | 58 | 0.623338 | 0.517241 |
| market | 0.5-0.6 | 799 | 0.500000 | 0.508135 |

## Interpretation and limits

**Inferred:** neither serving nor research closure is authorized by this unit. Candidate registry status is unresolved_below_power pending orchestrator serial registration and adjudication under AGENTS.md:65-105. No positive-control or split-half reliability study was run. Only three season blocks support the primary interval.

**Read:** the frozen model probabilities retain their walk-forward production lineage. LOSO combining coefficients are retrospective; this is not a prospective deployment test. Lattice priors exclude the held season and every row's current/future seasons. The endpoint is the actual leader-median quote; the base input sums within-book moves. Consequently this declared endpoint study includes the frozen-pool/leader basis difference as well as key-number pricing and cannot attribute any gain solely to units.

**Measured:** all 799 latest quote times precede min(kickoff, Sunday 12:45 Eastern). LOSO lattice prior rows range 1311-1343. Close lines enter the yardstick only, never predictors. The yardstick overlaps the movement input and is not independent evidence.

## Reproduction and local prediction output

Command: .tools/uv.exe run --no-sync python scripts/lead67_unit1.py (UV_NO_CACHE=1 avoids an inaccessible shared uv cache). No network or pipeline rebuild. Prediction rows: docs/lead67_predictions.md (local research artifact; do not commit).

Protocol: docs/lead67_protocol.md; lane: docs/lanes/lead67.md.

| Read-only source | SHA256 |
| --- | --- |
| artifacts\pick_probability\20260929T192747Z\per_game.parquet | 0490c806caf9e9707abe28f3e3e42f85df312084ae52d9d1588d173bba5b01e5 |
| artifacts\opener_evaluation\20260929T192743Z\per_game.parquet | de3b0ef63819e3416b4c6698e564b22a63cf43dcbfa6bc94a27f9ee8a477a201 |
| data\processed\game_features_pbp.parquet | 219e0a016f7556a6cd5acfef89e8d19d689b17d294002b1aef11959c89486eef |
| artifacts\sharp_book_weighted_movement\spread_quotes.parquet | 6ef7ff80d493a24af9d18c1e14d438c1c869677e1e6b2e8f9c901a2990953590 |

## Registry estimates for the orchestrator

The values below use accuracy fractions; lane commands convert them to
percentage points, as required by docs/weak_signal_registry.md:57.

[
  {
    "arm": "replacement",
    "effect": -0.01376720901126408,
    "low": -0.03007518796992481,
    "high": 0.003745318352059925,
    "probability_positive": 0.036,
    "standard_error": 0.007945144488468921,
    "blocks": 3
  },
  {
    "arm": "additive",
    "effect": -0.006257822277847309,
    "low": -0.018796992481203006,
    "high": 0.0,
    "probability_positive": 0.1489,
    "standard_error": 0.005079152538724801,
    "blocks": 3
  }
]

## Review accounting

**Measured:** family `lead67_key_number_move` has two predeclared experimental
arm comparisons, with three comparison models (four-term base, frozen model,
neutral market). The full display inventory is 12 fitted coefficient instances
(three all-row fits and nine LOSO fits), nine season-by-fitted-arm cells,
24 nonempty reliability bins, ten arm-by-protocol metric rows, 16 paired
metric/block summaries, and 16 coefficient-stability summaries. These repeated,
dependent diagnostics do not constitute independent votes or a selection pool.
Every cell is retained; no best-bin or best-fold claim is made.

**Measured:** the single numerical run exited 0. Scoped Ruff format and lint
checks passed. Repository-wide mypy passed on 253 source files. Broader Ruff
checks found errors outside this lane in concurrent/unrelated files. Pytest's
default shared temporary root was inaccessible. The isolated-root retry passed:
1,645 tests, 87 warnings, 82.72 seconds. Command:
`.tools/uv.exe run --no-sync pytest -q --basetemp $lead67TestRoot`, where
`$lead67TestRoot` was a fresh directory under the Windows temporary directory.
Logs are `%TEMP%/nfl-lead67-{format,lint,types,tests,tests-isolated}.log`.

**Inferred:** with only three season blocks, the primary "95%" intervals are
nominal bootstrap percentile ranges; their repeated-sampling coverage is not
validated. They must not be interpreted as calibrated 95% confidence or used
alone to establish a resolved wrong sign. The already reported season-week
sensitivity, season records, and probability_positive accompany every effect.
