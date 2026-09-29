# LEAD-87 unit 2: whole-season-excluded postseason replay

**Measured:** `.tools/uv.exe run --no-sync python scripts/lead87_unit2.py` executed the declared replay once.
The pre-score amendment and hash are saved in scratch. No registry or served-card changes.

## Decisive record

**Measured:** candidate record on held-out REG games where calibrated sides disagree, with exact binomial
95% intervals and two-sided equal-chance null. Historical forced picks are not each game's probability.

| Comparator | Candidate W-L | Win-rate 95% interval | Exact null p |
| --- | --- | --- | --- |
| served | 1-1 | [0.01257911709342506, 0.9874208829065749] | 1.0 |
| model_only | 151-116 | [0.5037647988828063, 0.6258500983717689] | 0.0372567831589444 |
| market | 150-115 | [0.5040213105064498, 0.6265604620242355] | 0.03654396752012622 |
| elo | 177-132 | [0.5155657977616882, 0.6286556349725164] | 0.01218807913318276 |

## Protocol and season exclusion

**Read:** `docs/lead87_protocol.md` plus the lane amendment declared before scoring.
Original 2020–2025 opener population; historical opener is the frozen pool-line proxy.
One joint likelihood retains all four REG terms and shares only the move coefficient with a POST-only
intercept. No postseason model or situational flags are invented.

**Measured:** 1503 original nonpush REG rows; 1272 with admissible leader moves;
686 outer REG games. 52 paired POST, 51 nonpush.
Only earlier fit-partition POST labels enter a fold; later POST games are unused.

**Measured:** fresh margin ridge fits, residual samples, discrete readers, and home-side offsets exclude each
entire prediction season and later seasons. Four-term likelihoods and REG-only Platt calibration are independently
refitted for each arm. Reserved tuning seasons select nothing: ridge and the single specification were fixed.
Observed earlier-game rolling inputs remain allowed; cached companion predictions are never consumed.

| Prediction season | Margin training games | Last training season/date | Discrete prior games | Offset prior games |
| --- | --- | --- | --- | --- |
| 2020 | 2816 | 2019 / 2019-12-29 | 1280 | 0 |
| 2021 | 3072 | 2020 / 2021-01-03 | 1280 | 220 |
| 2022 | 3344 | 2021 / 2022-01-09 | 1296 | 456 |
| 2023 | 3615 | 2022 / 2023-01-08 | 1311 | 704 |
| 2024 | 3887 | 2023 / 2024-01-07 | 1327 | 970 |
| 2025 | 4159 | 2024 / 2025-01-05 | 1343 | 1236 |

| Outer | Fit through | REG/POST fit | Tune REG | Calibrate REG | Outer REG |
| --- | --- | --- | --- | --- | --- |
| 2023 | 2020 | 176/8 | 208 | 202 | 231 |
| 2024 | 2021 | 384/17 | 202 | 231 | 224 |
| 2025 | 2022 | 586/26 | 231 | 224 | 231 |

**Read/inferred limitation:** feature-table hash matches the frozen served artifact. This repairs fitted-model
season exclusion, not historical feature construction. Weather, totals, injury/roster features and cached flags
are inherited without a new source-clock audit. Margin fitting retains the served historical spread labels;
the discrete prior pool substitutes known openers where available and otherwise retains historical spreads.
Scoring uses the opener and verified pre-kick move. This is a season-excluded recipe replay, not live-fit parity.
The discrete reader keeps its served five-year pool. Home offsets cold-start at zero in 2020, then use only
prior fresh season-excluded opener predictions; no contaminated offset cache is borrowed.

## Paired held-out results

**Measured:** 95% week-block bootstrap intervals, 10,000 paired draws stratified by season, seed 20260929.
Positive gains favor the candidate: comparator minus candidate for losses, candidate minus comparator for
accuracy points. Tied draws receive half credit in probability_positive. Intervals condition on fitted models.
Conditional binary RPS equals Brier exactly; it is not full-margin RPS.

| Arm | Log loss [95%] | Brier / conditional RPS [95%] | Accuracy points [95%] |
| --- | --- | --- | --- |
| candidate | 0.683436 [0.673440, 0.694011] | 0.245091 [0.240236, 0.250228] | 56.122449 [52.380952, 59.827834] |
| served | 0.683424 [0.673322, 0.694132] | 0.245088 [0.240200, 0.250266] | 56.122449 [52.366705, 59.827834] |
| model_only | 0.692129 [0.688605, 0.695827] | 0.249489 [0.247732, 0.251334] | 51.020408 [47.577093, 54.464286] |
| market | 0.691195 [0.682101, 0.700412] | 0.249106 [0.244728, 0.253552] | 51.020408 [47.611772, 54.376148] |
| elo | 0.696420 [0.690248, 0.702640] | 0.251626 [0.248569, 0.254713] | 49.562682 [45.481471, 53.471196] |

| Comparator | Endpoint | Candidate gain [95%] | probability_positive |
| --- | --- | --- | --- |
| served | log_loss | -0.000013 [-0.000272, 0.000248] | 0.4705 |
| served | brier | -0.000004 [-0.000126, 0.000119] | 0.4865 |
| served | accuracy_points | 0.000000 [-0.431655, 0.434153] | 0.4936 |
| model_only | log_loss | 0.008692 [-0.001156, 0.018638] | 0.9573 |
| model_only | brier | 0.004398 [-0.000387, 0.009226] | 0.9639 |
| model_only | accuracy_points | 5.102041 [0.888856, 9.251101] | 0.9909 |
| market | log_loss | 0.007758 [-0.005705, 0.021376] | 0.8690 |
| market | brier | 0.004015 [-0.002518, 0.010633] | 0.8872 |
| market | accuracy_points | 5.102041 [0.144713, 10.132159] | 0.9776 |
| elo | log_loss | 0.012984 [0.001656, 0.023702] | 0.9884 |
| elo | brier | 0.006535 [0.001028, 0.011781] | 0.9904 |
| elo | accuracy_points | 6.559767 [1.636905, 11.396104] | 0.9968 |

## In-sample, out-of-sample, and gap

**Measured:** IS reuses likelihood-fit REG games and is optimistic; IS/OOS apply the same independent
calibration map. Pooled IS includes repeated training games across folds; bootstrap keeps those repeats
together in their original season/week block. Gap is OOS minus IS; positive loss gaps mean worse OOS.
Complete interval-valued fold/pooled arm and contrast IS/OOS/gaps are in scratch summary.json.

| Arm | Endpoint | IS [95%] | OOS [95%] | OOS−IS [95%] |
| --- | --- | --- | --- | --- |
| candidate | log_loss | 0.682331 [0.671816, 0.692404] | 0.683436 [0.673440, 0.694011] | 0.001106 [-0.013762, 0.015996] |
| candidate | brier | 0.244726 [0.239562, 0.249690] | 0.245091 [0.240236, 0.250228] | 0.000365 [-0.006867, 0.007620] |
| candidate | accuracy_points | 54.450262 [50.304586, 58.599304] | 56.122449 [52.380952, 59.827834] | 1.672187 [-3.860330, 7.261193] |
| served | log_loss | 0.682322 [0.671742, 0.692454] | 0.683424 [0.673322, 0.694132] | 0.001102 [-0.013822, 0.016081] |
| served | brier | 0.244726 [0.239514, 0.249703] | 0.245088 [0.240200, 0.250266] | 0.000362 [-0.006921, 0.007681] |
| served | accuracy_points | 54.363002 [50.226214, 58.487434] | 56.122449 [52.366705, 59.827834] | 1.759447 [-3.735195, 7.321621] |
| model_only | log_loss | 0.692936 [0.687973, 0.698149] | 0.692129 [0.688605, 0.695827] | -0.000807 [-0.007052, 0.005439] |
| model_only | brier | 0.249887 [0.247417, 0.252485] | 0.249489 [0.247732, 0.251334] | -0.000398 [-0.003508, 0.002714] |
| model_only | accuracy_points | 51.570681 [48.194918, 54.718776] | 51.020408 [47.577093, 54.464286] | -0.550272 [-5.303151, 4.193150] |
| market | log_loss | 0.693210 [0.685204, 0.701029] | 0.691195 [0.682101, 0.700412] | -0.002015 [-0.013952, 0.010256] |
| market | brier | 0.250137 [0.246259, 0.253906] | 0.249106 [0.244728, 0.253552] | -0.001031 [-0.006811, 0.004884] |
| market | accuracy_points | 50.436300 [46.728929, 54.103860] | 51.020408 [47.611772, 54.376148] | 0.584108 [-4.366698, 5.596377] |
| elo | log_loss | 0.698162 [0.695081, 0.701132] | 0.696420 [0.690248, 0.702640] | -0.001742 [-0.008565, 0.005143] |
| elo | brier | 0.252493 [0.250969, 0.253965] | 0.251626 [0.248569, 0.254713] | -0.000867 [-0.004261, 0.002550] |
| elo | accuracy_points | 47.294939 [44.840167, 49.776606] | 49.562682 [45.481471, 53.471196] | 2.267743 [-2.427841, 6.840907] |

| Outer | Endpoint | Candidate gain IS | Candidate gain OOS [95%] | Gain gap | probability_positive |
| --- | --- | --- | --- | --- | --- |
| 2023 | log_loss | -0.000034 | 0.000283 [-0.000206, 0.000797] | 0.000317 | 0.8654 |
| 2023 | brier | -0.000024 | 0.000147 [-0.000074, 0.000387] | 0.000171 | 0.8928 |
| 2023 | accuracy_points | 0.000000 | -0.432900 [-1.298701, 0.000000] | -0.432900 | 0.1746 |
| 2024 | log_loss | 0.000017 | -0.000087 [-0.000629, 0.000421] | -0.000104 | 0.3917 |
| 2024 | brier | 0.000019 | -0.000049 [-0.000310, 0.000192] | -0.000068 | 0.3674 |
| 2024 | accuracy_points | 0.000000 | 0.000000 [0.000000, 0.000000] | 0.000000 | 0.5000 |
| 2025 | log_loss | -0.000018 | -0.000237 [-0.000523, 0.000005] | -0.000218 | 0.0276 |
| 2025 | brier | -0.000006 | -0.000110 [-0.000246, 0.000006] | -0.000104 | 0.0319 |
| 2025 | accuracy_points | 0.170648 | 0.432900 [0.000000, 1.315789] | 0.262252 | 0.8205 |

## Fitted coefficients

**Measured:** natural log-odds move coefficients per point. Wald intervals describe likelihood fits, not
paired held-out effects. Constant move-availability remains present with coefficient zero on this subset.

| Outer | Arm | Move [Wald 95%] | Intercept | Model logit | Flags | POST intercept | Calibration intercept/slope | Calibrated move |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | served | 0.051850 [-0.148873, 0.252573] | -0.044496 | -0.029970 | 0.353664 | n/a | 0.025222/1.205355 | 0.062498 |
| 2023 | candidate | 0.056053 [-0.144397, 0.256504] | -0.043850 | -0.031730 | 0.353885 | -0.521194 | 0.023805/1.207127 | 0.067663 |
| 2024 | served | 0.053179 [-0.074652, 0.181010] | -0.009723 | 0.427340 | 0.284516 | n/a | 0.074015/1.132728 | 0.060237 |
| 2024 | candidate | 0.047359 [-0.079793, 0.174511] | -0.009758 | 0.428084 | 0.285219 | -0.113602 | 0.074603/1.122621 | 0.053167 |
| 2025 | served | 0.086168 [-0.023955, 0.196290] | -0.000951 | 0.386293 | 0.313876 | n/a | -0.029908/0.855342 | 0.073703 |
| 2025 | candidate | 0.082389 [-0.026149, 0.190927] | -0.000866 | 0.386660 | 0.314172 | -0.183664 | -0.029550/0.856566 | 0.070571 |

## Reliability

**Measured:** five predeclared equal-width home-cover bands; empty cells remain explicit.

| Arm | Band | Games | Mean predicted | Observed home cover |
| --- | --- | --- | --- | --- |
| candidate | 0.0-0.2 | 0 | empty | empty |
| candidate | 0.2-0.4 | 65 | 0.360344 | 0.446154 |
| candidate | 0.4-0.6 | 521 | 0.505847 | 0.497121 |
| candidate | 0.6-0.8 | 100 | 0.637319 | 0.620000 |
| candidate | 0.8-1.0 | 0 | empty | empty |
| served | 0.0-0.2 | 0 | empty | empty |
| served | 0.2-0.4 | 68 | 0.362238 | 0.441176 |
| served | 0.4-0.6 | 515 | 0.505938 | 0.497087 |
| served | 0.6-0.8 | 103 | 0.636513 | 0.621359 |
| served | 0.8-1.0 | 0 | empty | empty |
| model_only | 0.0-0.2 | 0 | empty | empty |
| model_only | 0.2-0.4 | 0 | empty | empty |
| model_only | 0.4-0.6 | 686 | 0.510187 | 0.510204 |
| model_only | 0.6-0.8 | 0 | empty | empty |
| model_only | 0.8-1.0 | 0 | empty | empty |
| market | 0.0-0.2 | 0 | empty | empty |
| market | 0.2-0.4 | 36 | 0.350047 | 0.416667 |
| market | 0.4-0.6 | 604 | 0.502492 | 0.508278 |
| market | 0.6-0.8 | 45 | 0.646979 | 0.600000 |
| market | 0.8-1.0 | 1 | 0.811184 | 1.000000 |
| elo | 0.0-0.2 | 0 | empty | empty |
| elo | 0.2-0.4 | 7 | 0.386804 | 0.142857 |
| elo | 0.4-0.6 | 672 | 0.506198 | 0.514881 |
| elo | 0.6-0.8 | 7 | 0.607003 | 0.428571 |
| elo | 0.8-1.0 | 0 | empty | empty |

## Interpretation and handoff

**Inferred:** one retrospective auxiliary arm, not a serving decision. The 497-look family is unchanged
(F=3, B=6, K=4); repeated identical refits add no specification. The conditional RPS panel repeats Brier.
Week intervals do not establish postseason transportability; no positive control was run.
AGENTS.md lines 65–105 require that zero crossing closes nothing and one fitted probability selects the side.
Provisional classification: unresolved_below_power, pending serial orchestrator recording.
No terminal classification or promotion is asserted.

**Measured:** candidate-versus-served Brier gain is positive in 2023 and negative in 2024/2025.
Move slopes remain positive in all fits: served 0.051850 to 0.086168,
candidate 0.047359 to 0.082389.
The extra POST labels change the slope upward in 2023 and downward in 2024/2025; no stable improvement
is established. Only 8/17/26 POST games enter the respective chronological fit partitions.

**Measured:** predictions, training membership, source hashes, coefficients, metric panels, and the pre-score
amendment are in `tests/scratch/codex/lead87_unit2/`. Record commands are in `docs/lanes/lead87.md`.
Only candidate-versus-served commands are prepared; this script never writes to a registry.
