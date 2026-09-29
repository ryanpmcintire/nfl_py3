# LEAD-71 unit 3: push chances at the frozen pool line

**Measured:** saved Unit 2 probabilities were rescored without fitting or changing served code.

## Decisive opener record

**Measured:** pushes are excluded only from this inherited side-selection record. All push scores below retain them. Accuracy intervals are Wilson 95%; rates are fractions.

| Arm | W-L | Accuracy [95% CI] |
| --- | --- | --- |
| implied_lattice | 727-719 | 0.502766 [0.477022, 0.528496] |
| served_lattice | 762-684 | 0.526971 [0.501200, 0.552599] |
| market_residual | 703-743 | 0.486169 [0.460478, 0.511933] |

## Primary push comparisons

**Measured:** 1480 games and 103 season/week blocks. Positive improvement is served minus implied loss; smaller loss is better. Intervals use 10,000 paired week-block draws within season.

| Metric | Improvement [95% CI] | probability_positive |
| --- | --- | --- |
| push_log_loss | 0.000011 [-0.000904, 0.001080] | 0.485100 |
| push_brier | -0.000022 [-0.000160, 0.000128] | 0.371800 |

**Inferred:** neither metric establishes better push calibration against served. These push results describe distribution quality, not a new pick rule or proof of a profitable edge. The lattice mechanism remains open pending serial recording. No promotion or closure is made here. AGENTS.md:54-59 requires the discrete distribution at the line; AGENTS.md:65-79 governs closure; AGENTS.md:89-100 requires one fitted probability and out-of-season parameters before side selection changes.

## IS/OOS push scores and gaps

**Measured:** each score has a week-block 95% CI. The integer panel excludes noninteger lines with structurally zero push chance. Gap = OOS minus optimistic IS; positive is worse. Log loss clips only for numeric evaluation at 1e-12. Raw impossible counts reveal any infinite exact log losses; zero support is not smoothed in the probabilities.

| Panel | Scheme | Arm | Games / pushes | Metric | Score [95% CI] | Raw impossible |
| --- | --- | --- | --- | --- | --- | --- |
| all | IS | implied_lattice | 1480 / 34 | push_log_loss | 0.084103 [0.064127, 0.105679] | 0 |
| all | IS | served_lattice | 1480 / 34 | push_log_loss | 0.083917 [0.064057, 0.105316] | 0 |
| all | IS | market_residual | 1480 / 34 | push_log_loss | 0.089468 [0.067916, 0.112522] | 0 |
| all | IS | implied_lattice | 1480 / 34 | push_brier | 0.021225 [0.015027, 0.027907] | 0 |
| all | IS | served_lattice | 1480 / 34 | push_brier | 0.021189 [0.015016, 0.027829] | 0 |
| all | IS | market_residual | 1480 / 34 | push_brier | 0.021733 [0.015208, 0.028670] | 0 |
| all | OOS | implied_lattice | 1480 / 34 | push_log_loss | 0.085182 [0.064726, 0.107314] | 0 |
| all | OOS | served_lattice | 1480 / 34 | push_log_loss | 0.085193 [0.064658, 0.107421] | 0 |
| all | OOS | market_residual | 1480 / 34 | push_log_loss | 0.089864 [0.068146, 0.113014] | 0 |
| all | OOS | implied_lattice | 1480 / 34 | push_brier | 0.021313 [0.015073, 0.028044] | 0 |
| all | OOS | served_lattice | 1480 / 34 | push_brier | 0.021291 [0.015060, 0.028009] | 0 |
| all | OOS | market_residual | 1480 / 34 | push_brier | 0.021750 [0.015261, 0.028684] | 0 |
| integer | IS | implied_lattice | 645 / 34 | push_log_loss | 0.192980 [0.147817, 0.242121] | 0 |
| integer | IS | served_lattice | 645 / 34 | push_log_loss | 0.192555 [0.147502, 0.241202] | 0 |
| integer | IS | market_residual | 645 / 34 | push_log_loss | 0.205292 [0.155812, 0.258525] | 0 |
| integer | IS | implied_lattice | 645 / 34 | push_brier | 0.048703 [0.034495, 0.063979] | 0 |
| integer | IS | served_lattice | 645 / 34 | push_brier | 0.048619 [0.034514, 0.063802] | 0 |
| integer | IS | market_residual | 645 / 34 | push_brier | 0.049867 [0.034981, 0.065673] | 0 |
| integer | OOS | implied_lattice | 645 / 34 | push_log_loss | 0.195456 [0.148838, 0.245846] | 0 |
| integer | OOS | served_lattice | 645 / 34 | push_log_loss | 0.195482 [0.148743, 0.246067] | 0 |
| integer | OOS | market_residual | 645 / 34 | push_log_loss | 0.206201 [0.156387, 0.259193] | 0 |
| integer | OOS | implied_lattice | 645 / 34 | push_brier | 0.048905 [0.034596, 0.064289] | 0 |
| integer | OOS | served_lattice | 645 / 34 | push_brier | 0.048854 [0.034609, 0.064214] | 0 |
| integer | OOS | market_residual | 645 / 34 | push_brier | 0.049907 [0.035069, 0.065697] | 0 |

| Panel | Arm | Metric | OOS minus IS |
| --- | --- | --- | --- |
| all | implied_lattice | push_log_loss | 0.001079 |
| all | implied_lattice | push_brier | 0.000088 |
| all | served_lattice | push_log_loss | 0.001276 |
| all | served_lattice | push_brier | 0.000103 |
| all | market_residual | push_log_loss | 0.000396 |
| all | market_residual | push_brier | 0.000017 |
| integer | implied_lattice | push_log_loss | 0.002476 |
| integer | implied_lattice | push_brier | 0.000202 |
| integer | served_lattice | push_log_loss | 0.002928 |
| integer | served_lattice | push_brier | 0.000236 |
| integer | market_residual | push_log_loss | 0.000909 |
| integer | market_residual | push_brier | 0.000039 |

## All predeclared paired comparisons

**Measured:** comparator minus implied loss. Market residual is the simple market baseline; served lattice is the model-only baseline. Season panels use OOS probabilities.

| Panel | Scheme | Comparator | Metric | Improvement [95% CI] | probability_positive |
| --- | --- | --- | --- | --- | --- |
| all | IS | served_lattice | push_log_loss | -0.000185 [-0.001138, 0.000847] | 0.342900 |
| all | IS | market_residual | push_log_loss | 0.005365 [0.000925, 0.009922] | 0.991900 |
| all | IS | served_lattice | push_brier | -0.000037 [-0.000179, 0.000117] | 0.301900 |
| all | IS | market_residual | push_brier | 0.000508 [0.000012, 0.001039] | 0.978500 |
| all | OOS | served_lattice | push_log_loss | 0.000011 [-0.000904, 0.001080] | 0.485100 |
| all | OOS | market_residual | push_log_loss | 0.004682 [-0.000333, 0.009841] | 0.964900 |
| all | OOS | served_lattice | push_brier | -0.000022 [-0.000160, 0.000128] | 0.371800 |
| all | OOS | market_residual | push_brier | 0.000437 [-0.000093, 0.001001] | 0.942700 |
| integer | IS | served_lattice | push_log_loss | -0.000426 [-0.002620, 0.001944] | 0.342900 |
| integer | IS | market_residual | push_log_loss | 0.012312 [0.002081, 0.022864] | 0.991900 |
| integer | IS | served_lattice | push_brier | -0.000084 [-0.000409, 0.000271] | 0.301900 |
| integer | IS | market_residual | push_brier | 0.001165 [0.000028, 0.002393] | 0.978500 |
| integer | OOS | served_lattice | push_log_loss | 0.000026 [-0.002072, 0.002478] | 0.485100 |
| integer | OOS | market_residual | push_log_loss | 0.010744 [-0.000771, 0.022725] | 0.964900 |
| integer | OOS | served_lattice | push_brier | -0.000050 [-0.000364, 0.000292] | 0.371800 |
| integer | OOS | market_residual | push_brier | 0.001002 [-0.000216, 0.002304] | 0.942700 |
| 2020 | OOS | served_lattice | push_log_loss | -0.000124 [-0.005075, 0.006113] | 0.454700 |
| 2020 | OOS | market_residual | push_log_loss | 0.009508 [-0.009868, 0.029016] | 0.830600 |
| 2020 | OOS | served_lattice | push_brier | -0.000142 [-0.001133, 0.000972] | 0.381300 |
| 2020 | OOS | market_residual | push_brier | 0.001254 [-0.000687, 0.003473] | 0.884900 |
| 2021 | OOS | served_lattice | push_log_loss | 0.002038 [-0.000652, 0.006491] | 0.828200 |
| 2021 | OOS | market_residual | push_log_loss | -0.002047 [-0.014156, 0.007974] | 0.384300 |
| 2021 | OOS | served_lattice | push_brier | 0.000153 [-0.000027, 0.000392] | 0.942200 |
| 2021 | OOS | market_residual | push_brier | -0.000282 [-0.001446, 0.000878] | 0.317100 |
| 2022 | OOS | served_lattice | push_log_loss | -0.001456 [-0.002142, -0.000850] | 0.000000 |
| 2022 | OOS | market_residual | push_log_loss | 0.006715 [-0.008494, 0.024926] | 0.776300 |
| 2022 | OOS | served_lattice | push_brier | -0.000189 [-0.000286, -0.000103] | 0.000000 |
| 2022 | OOS | market_residual | push_brier | 0.000556 [-0.001007, 0.002508] | 0.713300 |
| 2023 | OOS | served_lattice | push_log_loss | -0.000091 [-0.001136, 0.001005] | 0.423700 |
| 2023 | OOS | market_residual | push_log_loss | 0.010300 [0.000628, 0.020693] | 0.981300 |
| 2023 | OOS | served_lattice | push_brier | 0.000045 [-0.000109, 0.000217] | 0.689400 |
| 2023 | OOS | market_residual | push_brier | 0.000889 [-0.000274, 0.002163] | 0.934300 |
| 2024 | OOS | served_lattice | push_log_loss | -0.000700 [-0.002064, 0.000422] | 0.134300 |
| 2024 | OOS | market_residual | push_log_loss | 0.000312 [-0.009519, 0.011500] | 0.512700 |
| 2024 | OOS | served_lattice | push_brier | -0.000047 [-0.000180, 0.000072] | 0.232400 |
| 2024 | OOS | market_residual | push_brier | -0.000069 [-0.001046, 0.001144] | 0.432900 |
| 2025 | OOS | served_lattice | push_log_loss | 0.000583 [-0.000935, 0.001977] | 0.788500 |
| 2025 | OOS | market_residual | push_log_loss | 0.004002 [-0.002823, 0.011044] | 0.871000 |
| 2025 | OOS | served_lattice | push_brier | 0.000025 [-0.000211, 0.000195] | 0.633000 |
| 2025 | OOS | market_residual | push_brier | 0.000442 [-0.000246, 0.001256] | 0.875800 |

## Key-number reliability

**Measured:** each cell contains games whose absolute pool spread equals that key. A push still means the signed margin equals the signed pool threshold; opposite-side margins are not combined as events. These are line-conditional push masses, not the unconditional chance of either team winning by that number. Rates are fractions. Observed intervals are Wilson; predicted minus observed intervals use paired week blocks.

| Key | Arm | Games / pushes | Predicted | Observed [Wilson 95% CI] | Predicted minus observed [block 95% CI] | Empty draws |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | implied_lattice | 194 / 20 | 0.089627 | 0.103093 [0.067732, 0.153866] | -0.013465 [-0.055171, 0.025245] | 0 |
| 3 | served_lattice | 194 / 20 | 0.089961 | 0.103093 [0.067732, 0.153866] | -0.013132 [-0.054996, 0.025597] | 0 |
| 3 | market_residual | 194 / 20 | 0.043714 | 0.103093 [0.067732, 0.153866] | -0.059378 [-0.101080, -0.020787] | 0 |
| 7 | implied_lattice | 72 / 3 | 0.057051 | 0.041667 [0.014271, 0.115493] | 0.015384 [-0.034912, 0.056931] | 0 |
| 7 | served_lattice | 72 / 3 | 0.056526 | 0.041667 [0.014271, 0.115493] | 0.014859 [-0.034812, 0.056100] | 0 |
| 7 | market_residual | 72 / 3 | 0.041458 | 0.041667 [0.014271, 0.115493] | -0.000208 [-0.050024, 0.040983] | 0 |
| 10 | implied_lattice | 22 / 0 | 0.040691 | 0.000000 [0.000000, 0.148655] | 0.040691 [0.036300, 0.044558] | 0 |
| 10 | served_lattice | 22 / 0 | 0.038356 | 0.000000 [0.000000, 0.148655] | 0.038356 [0.034346, 0.041637] | 0 |
| 10 | market_residual | 22 / 0 | 0.039345 | 0.000000 [0.000000, 0.148655] | 0.039345 [0.032712, 0.046792] | 0 |
| 14 | implied_lattice | 11 / 1 | 0.072284 | 0.090909 [0.016232, 0.377358] | -0.018625 [-0.238325, 0.082719] | 0 |
| 14 | served_lattice | 11 / 1 | 0.072295 | 0.090909 [0.016232, 0.377358] | -0.018614 [-0.235882, 0.081002] | 0 |
| 14 | market_residual | 11 / 1 | 0.039574 | 0.090909 [0.016232, 0.377358] | -0.051335 [-0.264530, 0.043783] | 0 |

## Season stability and inherited folds

**Measured:** no season or key cell was selected after scoring. Sparse-event bootstrap intervals condition on observed events: a zero-push cell or season cannot generate unseen pushes. Reliability Wilson intervals show event-rate uncertainty in those cells. Week-block intervals do not measure between-season sampling or refit uncertainty.

| Season | Arm | Games / pushes | Metric | Score [95% CI] |
| --- | --- | --- | --- | --- |
| 2020 | implied_lattice | 180 / 7 | push_log_loss | 0.122068 [0.056968, 0.201212] |
| 2020 | served_lattice | 180 / 7 | push_log_loss | 0.121944 [0.056842, 0.200633] |
| 2020 | market_residual | 180 / 7 | push_log_loss | 0.131576 [0.062941, 0.207834] |
| 2020 | implied_lattice | 180 / 7 | push_brier | 0.034518 [0.014805, 0.057877] |
| 2020 | served_lattice | 180 / 7 | push_brier | 0.034375 [0.014730, 0.057466] |
| 2020 | market_residual | 180 / 7 | push_brier | 0.035772 [0.015388, 0.059191] |
| 2021 | implied_lattice | 229 / 3 | push_log_loss | 0.059864 [0.019858, 0.106179] |
| 2021 | served_lattice | 229 / 3 | push_log_loss | 0.061902 [0.020003, 0.110227] |
| 2021 | market_residual | 229 / 3 | push_log_loss | 0.057817 [0.020699, 0.098186] |
| 2021 | implied_lattice | 229 / 3 | push_brier | 0.012944 [0.001242, 0.025426] |
| 2021 | served_lattice | 229 / 3 | push_brier | 0.013097 [0.001295, 0.025725] |
| 2021 | market_residual | 229 / 3 | push_brier | 0.012662 [0.001021, 0.024555] |
| 2022 | implied_lattice | 255 / 7 | push_log_loss | 0.101195 [0.053611, 0.154673] |
| 2022 | served_lattice | 255 / 7 | push_log_loss | 0.099739 [0.052447, 0.152827] |
| 2022 | market_residual | 255 / 7 | push_log_loss | 0.107910 [0.048452, 0.174953] |
| 2022 | implied_lattice | 255 / 7 | push_brier | 0.025501 [0.009419, 0.043072] |
| 2022 | served_lattice | 255 / 7 | push_brier | 0.025312 [0.009278, 0.042836] |
| 2022 | market_residual | 255 / 7 | push_brier | 0.026057 [0.008370, 0.045258] |
| 2023 | implied_lattice | 272 / 6 | push_log_loss | 0.078727 [0.043200, 0.116997] |
| 2023 | served_lattice | 272 / 6 | push_log_loss | 0.078636 [0.043023, 0.116846] |
| 2023 | market_residual | 272 / 6 | push_log_loss | 0.089027 [0.043916, 0.135298] |
| 2023 | implied_lattice | 272 / 6 | push_brier | 0.020183 [0.007939, 0.032771] |
| 2023 | served_lattice | 272 / 6 | push_brier | 0.020228 [0.007964, 0.032853] |
| 2023 | market_residual | 272 / 6 | push_brier | 0.021072 [0.007685, 0.034612] |
| 2024 | implied_lattice | 272 / 6 | push_log_loss | 0.087762 [0.031580, 0.167425] |
| 2024 | served_lattice | 272 / 6 | push_log_loss | 0.087062 [0.031670, 0.166160] |
| 2024 | market_residual | 272 / 6 | push_log_loss | 0.088075 [0.032217, 0.164066] |
| 2024 | implied_lattice | 272 / 6 | push_brier | 0.020990 [0.004473, 0.044143] |
| 2024 | served_lattice | 272 / 6 | push_brier | 0.020943 [0.004501, 0.044009] |
| 2024 | market_residual | 272 / 6 | push_brier | 0.020921 [0.004283, 0.043888] |
| 2025 | implied_lattice | 272 / 5 | push_log_loss | 0.070950 [0.034725, 0.112735] |
| 2025 | served_lattice | 272 / 5 | push_log_loss | 0.071533 [0.034794, 0.114060] |
| 2025 | market_residual | 272 / 5 | push_log_loss | 0.074952 [0.037569, 0.119191] |
| 2025 | implied_lattice | 272 / 5 | push_brier | 0.017149 [0.007091, 0.029641] |
| 2025 | served_lattice | 272 / 5 | push_brier | 0.017174 [0.007030, 0.029636] |
| 2025 | market_residual | 272 / 5 | push_brier | 0.017591 [0.007367, 0.030609] |

| Held season | Games | Prior training games | Last training day | Optimistic IS games |
| --- | --- | --- | --- | --- |
| 2020 | 180 | 1280 | 2019-12-29 | 1536 |
| 2021 | 229 | 1280 | 2021-01-03 | 1552 |
| 2022 | 255 | 1296 | 2022-01-09 | 1567 |
| 2023 | 272 | 1311 | 2023-01-08 | 1583 |
| 2024 | 272 | 1327 | 2024-01-07 | 1599 |
| 2025 | 272 | 1343 | 2025-01-05 | 1615 |

**Read:** scripts/lead71_unit2.py:276-324 constructs each OOS lattice only from completed prior seasons; its IS companion adds the held season. It is chronological LOSO, not training on future seasons. Archived model forecasts are inherited, not refitted. There are no Unit 3 coefficients or fitted cut points; coefficient stability is inapplicable. Price constraints and lattice parameters remain exactly those from Unit 2.

## Reproduction and limits

Run `.tools/uv.exe run --no-sync python scripts/lead71_unit3.py` with `UV_CACHE_DIR=tests/scratch/codex/lead71_unit3/uv-cache` and `PYTHONDONTWRITEBYTECODE=1`. Threads are capped at two.

**Measured:** 127 predeclared estimands: 60 score cells, 40 paired contrasts, 12 IS/OOS gaps, 12 reliability cells and 3 decisive records. Only the two all-game OOS comparisons against served are primary; all other results are descriptive. Intervals are unadjusted and correlated; no minimum cell or best season is selected. All primary and score-panel bootstrap draws retain a nonempty denominator.

**Measured:** source hashes, frozen protocol, scored prediction rows and full summary are saved in tests/scratch/codex/lead71_unit3/. Hashes for the three joined original data sources match Unit 2; saved score/quote/fold bytes are hashed for reproducibility. Raw quote archives are not re-parsed. Runtime checks recheck game IDs, pool-line signs, push outcomes, noninteger-line support, prediction-time and fold chronology.

**Inferred:** LOSO reuse prevents choosing push parameters on these outcomes, but this is a new endpoint on the already researched Unit 2 population, not an independent outer test. Key cells can have few or no pushes; neither an unadjusted cell interval nor an overall push score establishes reliable alternative-line or flip-line behavior. The interval rule leaves the mechanism unresolved_below_power pending the orchestrator's serial records. Proposed commands are in docs/lanes/lead71.md; none was executed.

## Predeclared protocol

Population: exactly Unit 2 look-1 game IDs, 2020-2025; no outcome-dependent exclusions.
Target: result equals signed Tuesday pool-opener margin threshold; half-point lines have zero push probability.
Terms/arms: reuse saved implied_lattice, served_lattice (model-only), market_residual (simple market baseline) masses at that line; no new coefficients, smoothing, recalibration, thresholds, or side flips.
Folds: reuse Unit 2 chronological LOSO (completed prior seasons only) and its optimistic IS companion; assert matched IDs, frozen-quote timing, training chronology and input hashes. No pipeline rebuild.
Primary: served-minus-implied push-event log loss and Brier, all games OOS; clip probabilities to [1e-12,1-1e-12] only for log evaluation, report raw impossible-event counts separately. Positive improvement favors implied.
Uncertainty: 10,000 paired season-stratified (season, week) block bootstraps, seed 7103, percentile 95% CI and probability_positive with half weight for exact ties; reuse draws across arms/endpoints; no refitting in bootstrap.
Diagnostics fixed now: both metrics for each arm and IS/OOS overall and integer-line games; IS/OOS gaps; each season's OOS scores; implied-minus-comparator improvements against served and simple market in every scored panel; OOS reliability at absolute pool lines exactly 3/7/10/14 (signed push event), observed Wilson CI and predicted-minus-observed week-block CI; OOS decisive records with Wilson CI.
New look budget: 127 estimands = 24 overall/integer score cells + 16 paired contrasts + 12 IS/OOS gaps + 36 season score cells + 24 season contrasts + 12 key-line/arm reliability cells + 3 decisive records. Two primary contrasts; all other looks descriptive, unadjusted intervals, no best-cell selection. Denominators/support/timing checks are audits.
Decision: an interval crossing zero never closes a signal; report probability_positive. No promotion or closure here; unresolved_below_power pending serial recording. One fitted calibrated probability selects the side. No served changes. Report bootstrap sparsity and inherited-fold limitations; preserve rows only under tests/scratch/codex/lead71_unit3/.
