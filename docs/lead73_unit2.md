# LEAD-73 unit 2: extended dated market moves

Protocol was saved in docs/lanes/lead73.md before outcomes were loaded.
**Measured:** one frozen-population comparison; 14 fits, 144 declared reporting looks; no tuning.
Both arms score the same 1503 regular-season non-push opener games, 2020-2025.
**Measured, decisive games first:** the extended arm won 69-94 on 163 side disagreements; accuracy 42.33 [34.88, 49.69]%.

## Declared protocol

Saved before outcomes in the lane; copied here without changing the design.
Population: frozen pick_probability/20260929T192747Z/per_game.parquet; identical
complete 2020-2025 regular-season rows in both arms; opener pushes excluded.
Target: home covers the Tuesday opener, conditional on no push.
Terms: model logit, signed move toward home, availability, composition sum, intercept;
ridge 0.001, six LOSO folds with train-fold standardization; one full-data IS fit.
Current retains all inputs. Extended adds only verified 2020-2022 leader moves and
availability. Monday history; Wednesday-onward differences; capture strictly before
min(kickoff, Sunday 12:45 ET); book update no later than capture; median book net move.
The 2023-2025 inputs stay identical. Fail on timestamp, quote, sign or schema conflict.
Metrics: opener accuracy, log loss and Brier against current, raw model and 0.5 market.
Primary contrast: extended-minus-current accuracy. Fixed home-probability bins:
0, .40, .45, .50, .55, .60, 1. No tuning, new subgroup, challenger or standalone flip.
10,000 paired season-stratified week-block bootstraps; seed 20260929; 95% percentile
intervals; probability_positive=P(gain>0)+0.5P(gain=0); hold predictions fixed.
Look accounting: 14 fits + 12 OOS metric cells + 6 IS metric cells + 72 season metric
cells + 24 reliability cells + 9 paired contrasts + 1 decisive comparison + 6 gaps
= 144 reporting looks, one primary comparison; coefficients shown for every fit.
No multiple-comparison adjustment; diagnostics are not independent discoveries.

## Population and source alignment

| Season | Fit games | Early source pairs | Added | Outside fit | Current available | Extended available |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 220 | 235 | 218 | 17 | 0 | 218 |
| 2021 | 236 | 253 | 234 | 19 | 0 | 234 |
| 2022 | 248 | 269 | 246 | 23 | 0 | 246 |
| 2023 | 266 | 0 | 0 | 0 | 266 | 266 |
| 2024 | 266 | 0 | 0 | 0 | 266 | 266 |
| 2025 | 267 | 0 | 0 | 0 | 267 | 267 |

**Measured, exclusions from the 757 source games:**

- 2020: 10 absent_from_opener.
- 2020: 7 opener_push.
- 2021: 16 absent_from_opener.
- 2021: 3 opener_push.
- 2022: 16 absent_from_opener.
- 2022: 7 opener_push.

**Measured:** quote hashes, schedule team mapping, HOME/AWAY handicap sign, capture/update gates,
and current-artifact move parity passed. Bookmaker home handicap is negated by the parser;
a positive feature is movement toward home. Both arms retain identical late-season inputs.
Source games outside the frozen model/composition population are excluded from both arms;
their outcomes are not used. The frozen fit already excludes opener pushes and ungraded rows.
Verified manifests: 455; rejected quote rows: 0;
admissible unique leader quotes: 11362.
Current refit maximum probability error versus saved fit: 0.

## Paired out-of-season results

All intervals are 95% percentile intervals from 10,000 paired week-block resamples within season.
There are 107 season/week blocks; seed 20260929. Accuracy is percent; gains are percentage points.
| Arm | Accuracy [95% interval] | Log loss [95% interval] | Brier [95% interval] |
|---|---:|---:|---:|
| current | 57.42 [55.04, 59.65] | 0.6828 [0.6760, 0.6896] | 0.2448 [0.2416, 0.2481] |
| extended | 55.76 [53.47, 57.99] | 0.6830 [0.6765, 0.6895] | 0.2450 [0.2419, 0.2482] |
| model | 53.36 [50.79, 55.82] | 0.6969 [0.6893, 0.7046] | 0.2517 [0.2480, 0.2554] |
| market | 49.63 [47.30, 52.00] | 0.6931 [0.6931, 0.6931] | 0.2500 [0.2500, 0.2500] |

Positive gain means extended is better. The market comparator is neutral 0.5 at the opener,
with home selected on exact ties; this is not a tradable price or profitability estimate.

| Baseline | Metric | Extended gain [95% interval] | probability_positive |
|---|---|---:|---:|
| current | accuracy_points | -1.663340 [-3.275401, -0.066534] | 0.0202 |
| current | log_loss | -0.000182 [-0.004530, 0.003981] | 0.4707 |
| current | brier | -0.000145 [-0.002226, 0.001832] | 0.4496 |
| model | accuracy_points | 2.395210 [-0.333572, 5.214005] | 0.9570 |
| model | log_loss | 0.013961 [0.004915, 0.023015] | 0.9981 |
| model | brier | 0.006732 [0.002350, 0.011131] | 0.9980 |
| market | accuracy_points | 6.121091 [2.692049, 9.533411] | 0.9996 |
| market | log_loss | 0.010190 [0.003602, 0.016632] | 0.9982 |
| market | brier | 0.005017 [0.001829, 0.008144] | 0.9985 |

## In-sample versus out-of-season

Gap is optimistic IS gain: IS-OOS accuracy; OOS-IS loss. Both use the same games.

| Arm | Metric | IS [95% interval] | OOS [95% interval] | Gap [95% interval] |
|---|---|---:|---:|---:|
| current | accuracy_points | 57.5516 [55.2047, 59.8552] | 57.4185 [55.0403, 59.6538] | 0.1331 [-0.6645, 0.9309] |
| current | log_loss | 0.6818 [0.6750, 0.6886] | 0.6828 [0.6760, 0.6896] | 0.0010 [0.0002, 0.0017] |
| current | brier | 0.2444 [0.2411, 0.2477] | 0.2448 [0.2416, 0.2481] | 0.0004 [0.0001, 0.0008] |
| extended | accuracy_points | 56.4870 [54.2418, 58.7103] | 55.7552 [53.4727, 57.9909] | 0.7319 [0.1325, 1.3898] |
| extended | log_loss | 0.6816 [0.6752, 0.6879] | 0.6830 [0.6765, 0.6895] | 0.0014 [0.0004, 0.0024] |
| extended | brier | 0.2443 [0.2412, 0.2474] | 0.2450 [0.2419, 0.2482] | 0.0007 [0.0002, 0.0012] |

## Coefficients and stability

Natural units; ridge 0.001; train-fold standardization. IS fits are descriptive only.

| Arm | Held out | Train n | Intercept | Model logit | Move | Available | Composition |
|---|---|---:|---:|---:|---:|---:|---:|
| current | 2020 | 1283 | -0.045251 | 0.341246 | 0.220730 | 0.028113 | 0.244789 |
| current | 2021 | 1267 | -0.062332 | 0.177916 | 0.223716 | 0.044183 | 0.271867 |
| current | 2022 | 1255 | -0.055698 | 0.249647 | 0.222903 | 0.039674 | 0.238476 |
| current | 2023 | 1237 | -0.038064 | 0.341480 | 0.230272 | 0.002286 | 0.246114 |
| current | 2024 | 1237 | -0.056178 | 0.256116 | 0.246066 | 0.039146 | 0.262262 |
| current | 2025 | 1236 | -0.054099 | 0.292328 | 0.188739 | 0.047462 | 0.295997 |
| current | IS | 1503 | -0.052065 | 0.276510 | 0.221778 | 0.034174 | 0.259974 |
| extended | 2020 | 1283 | 0.101703 | 0.375483 | 0.165644 | -0.127131 | 0.236637 |
| extended | 2021 | 1267 | 0.125995 | 0.194640 | 0.165099 | -0.153083 | 0.271052 |
| extended | 2022 | 1255 | 0.075086 | 0.286269 | 0.137255 | -0.095633 | 0.235292 |
| extended | 2023 | 1237 | 0.123026 | 0.355464 | 0.129987 | -0.151975 | 0.241052 |
| extended | 2024 | 1237 | 0.100344 | 0.289964 | 0.139572 | -0.130843 | 0.256399 |
| extended | 2025 | 1236 | 0.109306 | 0.315883 | 0.119636 | -0.138798 | 0.289540 |
| extended | IS | 1503 | 0.105414 | 0.304607 | 0.141871 | -0.132292 | 0.254944 |

**Measured:** current move coefficient is positive in 6/6 folds;
range 0.188739 to 0.246066; mean 0.222071; fold SD 0.018757.

**Measured:** extended move coefficient is positive in 6/6 folds;
range 0.119636 to 0.165644; mean 0.142865; fold SD 0.018767.

## Season stability

| Season | Arm | n | Accuracy [95% interval] | Log loss [95% interval] | Brier [95% interval] |
|---|---|---:|---:|---:|---:|
| 2020 | current | 220 | 55.45 [48.13, 61.97] | 0.6858 [0.6742, 0.6991] | 0.2464 [0.2406, 0.2530] |
| 2020 | extended | 220 | 52.73 [45.93, 59.03] | 0.6897 [0.6746, 0.7052] | 0.2485 [0.2412, 0.2561] |
| 2020 | market | 220 | 48.64 [42.20, 54.26] | 0.6931 [0.6931, 0.6931] | 0.2500 [0.2500, 0.2500] |
| 2020 | model | 220 | 53.18 [45.70, 59.62] | 0.7084 [0.6827, 0.7358] | 0.2571 [0.2447, 0.2702] |
| 2021 | current | 236 | 56.36 [50.00, 62.76] | 0.6864 [0.6705, 0.7017] | 0.2467 [0.2388, 0.2542] |
| 2021 | extended | 236 | 52.97 [47.41, 59.58] | 0.6872 [0.6663, 0.7079] | 0.2471 [0.2371, 0.2569] |
| 2021 | market | 236 | 47.46 [41.20, 53.91] | 0.6931 [0.6931, 0.6931] | 0.2500 [0.2500, 0.2500] |
| 2021 | model | 236 | 55.93 [48.46, 63.75] | 0.6896 [0.6605, 0.7189] | 0.2481 [0.2339, 0.2623] |
| 2022 | current | 248 | 60.48 [55.42, 65.43] | 0.6797 [0.6684, 0.6913] | 0.2433 [0.2377, 0.2490] |
| 2022 | extended | 248 | 55.24 [49.14, 61.48] | 0.6740 [0.6593, 0.6885] | 0.2406 [0.2334, 0.2477] |
| 2022 | market | 248 | 48.79 [44.49, 53.33] | 0.6931 [0.6931, 0.6931] | 0.2500 [0.2500, 0.2500] |
| 2022 | model | 248 | 55.24 [49.20, 60.78] | 0.6924 [0.6784, 0.7087] | 0.2495 [0.2427, 0.2575] |
| 2023 | current | 266 | 56.39 [51.81, 60.74] | 0.6780 [0.6608, 0.6963] | 0.2423 [0.2340, 0.2511] |
| 2023 | extended | 266 | 57.89 [53.79, 61.76] | 0.6790 [0.6656, 0.6935] | 0.2429 [0.2363, 0.2500] |
| 2023 | market | 266 | 50.75 [45.28, 55.93] | 0.6931 [0.6931, 0.6931] | 0.2500 [0.2500, 0.2500] |
| 2023 | model | 266 | 50.38 [46.82, 54.33] | 0.7018 [0.6910, 0.7124] | 0.2541 [0.2489, 0.2592] |
| 2024 | current | 266 | 58.27 [52.63, 63.26] | 0.6837 [0.6675, 0.7013] | 0.2451 [0.2374, 0.2535] |
| 2024 | extended | 266 | 58.65 [53.87, 63.16] | 0.6829 [0.6705, 0.6967] | 0.2449 [0.2388, 0.2516] |
| 2024 | market | 266 | 50.38 [43.85, 57.09] | 0.6931 [0.6931, 0.6931] | 0.2500 [0.2500, 0.2500] |
| 2024 | model | 266 | 54.51 [47.84, 61.01] | 0.6944 [0.6809, 0.7081] | 0.2506 [0.2439, 0.2573] |
| 2025 | current | 267 | 57.30 [51.70, 62.82] | 0.6837 [0.6619, 0.7041] | 0.2456 [0.2353, 0.2553] |
| 2025 | extended | 267 | 56.18 [50.74, 61.71] | 0.6859 [0.6679, 0.7031] | 0.2465 [0.2378, 0.2548] |
| 2025 | market | 267 | 51.31 [45.90, 56.23] | 0.6931 [0.6931, 0.6931] | 0.2500 [0.2500, 0.2500] |
| 2025 | model | 267 | 51.31 [45.15, 56.93] | 0.6956 [0.6814, 0.7112] | 0.2512 [0.2442, 0.2589] |

## Fixed reliability table

Probabilities and observed rates below refer to the home side; empty bins remain reported.

| Arm | Home probability bin | n | Mean probability | Home cover rate |
|---|---|---:|---:|---:|
| current | 0.00-0.40 | 114 | 0.3573 | 0.3684 |
| current | 0.40-0.45 | 243 | 0.4254 | 0.4774 |
| current | 0.45-0.50 | 488 | 0.4752 | 0.4221 |
| current | 0.50-0.55 | 347 | 0.5267 | 0.5447 |
| current | 0.55-0.60 | 182 | 0.5698 | 0.5934 |
| current | 0.60-1.00 | 129 | 0.6437 | 0.6589 |
| extended | 0.00-0.40 | 143 | 0.3613 | 0.3986 |
| extended | 0.40-0.45 | 252 | 0.4288 | 0.4325 |
| extended | 0.45-0.50 | 419 | 0.4769 | 0.4654 |
| extended | 0.50-0.55 | 344 | 0.5242 | 0.5174 |
| extended | 0.55-0.60 | 219 | 0.5711 | 0.5936 |
| extended | 0.60-1.00 | 126 | 0.6449 | 0.6111 |
| model | 0.00-0.40 | 215 | 0.3650 | 0.4884 |
| model | 0.40-0.45 | 292 | 0.4260 | 0.4384 |
| model | 0.45-0.50 | 387 | 0.4751 | 0.4806 |
| model | 0.50-0.55 | 360 | 0.5226 | 0.5361 |
| model | 0.55-0.60 | 176 | 0.5725 | 0.5341 |
| model | 0.60-1.00 | 73 | 0.6245 | 0.5479 |
| market | 0.00-0.40 | 0 | — | — |
| market | 0.40-0.45 | 0 | — | — |
| market | 0.45-0.50 | 0 | — | — |
| market | 0.50-0.55 | 1503 | 0.5000 | 0.4963 |
| market | 0.55-0.60 | 0 | — | — |
| market | 0.60-1.00 | 0 | — | — |

## Interpretation and limits

**Inferred:** this is a source-extension estimate conditional on the frozen upstream model and
composition features. LOSO keeps calibration parameters out of each scored season, but uses
future seasons in training and is not a chronological deployment simulation. Week resampling
holds fitted predictions fixed; it does not include full refit or source-selection uncertainty.
Equal aggregation rules do not establish equal capture density across archives.
No source-frequency adjustment was selected.
One calibrated fitted probability chooses each side. Historical forced-pick accuracy does not
establish a profitable edge or represent an individual game's probability.
**Measured:** primary accuracy gain is -1.6633 [-3.2754, -0.0665] percentage points; probability_positive=0.0202.
**Inferred:** the source extension does not support replacing the current fit.
For this fixed accuracy comparison, the entire interval is adverse: the proposed
registry row names wrong_sign_resolved under AGENTS.md. This is a claim about
the tested extension, not a rejection of book moves or pre-2023 source research.
Proper-score differences versus current remain unresolved_below_power. No
positive-control or split-half reliability experiment was run; no broader closure.
All registry entries await the owner. No served fit was changed.

## Reproduction and artifacts

.tools/uv.exe run --no-sync --no-cache python scripts/lead73_unit2.py

Prediction rows, quote evidence, coefficients, provenance hashes and bootstrap summaries:
tests/scratch/codex/lead73_unit2/. No registry writes or served artifacts changed.
The lane contains the exact owner-run record commands; --no-cache avoids the inaccessible uv cache.
