# Confidence and Best Pick audit

## Question and frozen audit plan

Declared 2026-09-20 before the measurements below: does the probability
actually served distinguish stronger picks within a week, and are its
numbers calibrated? Agreement between the star and the displayed number is
an implementation property, not evidence of either claim.

Use the active model's matched opener evaluation and the production
`build_fit_population` inputs. Save the population, prediction-level results,
coefficients and source identifiers. Never activate this audit's fits.

Compare four fixed probability arms: the existing raw model, a simple
intercept-plus-model-logit recalibration, the existing four-term production
formula, and a neutral 0.5 market reference at the grading line. The last is
not a reconstructed historical no-vig sportsbook probability.

Evaluate both the existing leave-one-season-out protocol and chronological
expanding training with at least two earlier seasons. Also report the
full-population fit beside leave-one-season-out performance to expose the
optimism gap. No features, penalty, thresholds or model variants are selected
from the audit results. The current formula and its ingredients have already
been selected using these years; neither replay is an untouched outer test.

For each fitted protocol, report Brier and log loss against the raw, simple
and neutral baselines, fixed confidence-band reliability, season stability
and coefficients. Separately measure a continuous within-week confidence
slope, and the weekly top pick's result against the rest of the same week
and against the raw model's top pick. Report decisive weeks first. No new
spread or bookmaker filter is optimized. Ties use game ID, fixed in advance.

Uncertainty uses resampling whole season/week blocks with a fixed seed;
season-level results remain visible. Every inferential comparison, slope,
band and arm is counted. Paired effects are oriented so positive favours
the combined probability. Unresolved effects are recorded through
`nfl-ats weak-signals record`; no interval crossing zero closes anything.

The operational objective is one probability and one selection rule for
publication and refresh. A separate ranking fit cannot repair bad
probabilities. Any remaining inability to distinguish the top pick must be
reported as unresolved, not renamed a solved confidence problem.

## Input-alignment amendment, before the corrected replay

The first reconstruction reproduced the existing smooth-input fitter. Code
inspection then confirmed that serving substituted discrete probabilities at
key lines. That replay is labelled `legacy_smooth` and does not validate the
served path. Correct the input contract to discrete cover divided by
cover-plus-loss at every line, for both training and serving. Repeat the
same frozen arms and diagnostics on the corrected population; this is one
additional structural comparison, not a search over mappings or thresholds.
Preserve the discrete push mass when applying the fitted conditional chance.
Percentages describe cover conditional on a non-push, matching the training
target; report that definition explicitly. Keep all original outputs.

## Corrected replay and operational decision

Read: `artifacts/confidence_ranking_audit/20260920_aligned/audit.json` and its retained prediction tables. Measured registry command: `nfl-ats weak-signals record --batch artifacts/confidence_ranking_audit/20260920_registry_batch.json`.

Keep the existing four-term probability formula and correct its discrete inputs; do not introduce a separately fitted Best Pick ranker. The combined formula improves probability loss, while the extra advantage of its single top pick remains unresolved. These are reused research years, not an untouched outer test. The audit fits remain inactive; production must be refitted through the normal command against the corrected input contract.

| Protocol / arm | Games | Correct | Brier | Log loss |
|---|---:|---:|---:|---:|
| in_sample / raw | 1503 | 802 | 0.251718 | 0.696922 |
| in_sample / simple | 1503 | 803 | 0.249327 | 0.691804 |
| in_sample / combined | 1503 | 862 | 0.244861 | 0.682830 |
| in_sample / neutral | 1503 | 746 | 0.250000 | 0.693147 |
| leave_one_season_out / raw | 1503 | 802 | 0.251718 | 0.696922 |
| leave_one_season_out / simple | 1503 | 800 | 0.249502 | 0.692159 |
| leave_one_season_out / combined | 1503 | 859 | 0.245388 | 0.683952 |
| leave_one_season_out / neutral | 1503 | 746 | 0.250000 | 0.693147 |
| chronological / raw | 1047 | 553 | 0.251395 | 0.696144 |
| chronological / simple | 1047 | 553 | 0.249506 | 0.692162 |
| chronological / combined | 1047 | 602 | 0.245259 | 0.683638 |
| chronological / neutral | 1047 | 527 | 0.250000 | 0.693147 |

### leave_one_season_out

Decisive weeks: combined wins 28, raw wins 17; two-sided exact paired-sign null p=0.1352. Nominees differ in 89/107 weeks. Combined top record 65/107; mean top estimate 64.10%. Rest-of-week accuracy, with weeks weighted equally, 56.71%.

| Comparison | Effect | 95% interval | probability_positive |
|---|---:|---:|---:|
| combined_vs_raw_brier | 0.006329 | [0.002321, 0.010301] | 0.9992 |
| combined_vs_raw_log_loss | 0.012970 | [0.004738, 0.021168] | 0.9992 |
| combined_vs_simple_brier | 0.004113 | [0.001246, 0.007020] | 0.9978 |
| combined_vs_simple_log_loss | 0.008207 | [0.002237, 0.014213] | 0.9969 |
| combined_vs_neutral_brier | 0.004612 | [0.001585, 0.007699] | 0.9985 |
| combined_vs_neutral_log_loss | 0.009195 | [0.002917, 0.015556] | 0.9982 |
| combined_top_vs_rest_accuracy | 4.040912 | [-5.679478, 13.847075] | 0.7926 |
| combined_top_vs_raw_top_accuracy | 10.280374 | [-1.869159, 22.429907] | 0.9526 |
| raw within-week slope, accuracy points per 10 confidence points | -0.509 | [-6.529, 5.453] | 0.4308 |
| simple within-week slope, accuracy points per 10 confidence points | -2.249 | [-17.972, 13.982] | 0.3997 |
| combined within-week slope, accuracy points per 10 confidence points | 4.185 | [-1.931, 9.963] | 0.9139 |

| Combined probability band | Games | Mean estimate | Actual | Gap interval, accuracy points | probability_positive |
|---|---:|---:|---:|---:|---:|
| 50%â€“52% | 313 | 51.03% | 57.19% | [0.72, 11.71] | 0.9865 |
| 52%â€“55% | 516 | 53.42% | 55.23% | [-2.73, 6.36] | 0.7867 |
| 55%â€“58% | 297 | 56.38% | 55.56% | [-5.30, 3.69] | 0.3655 |
| 58%â€“62% | 247 | 59.77% | 61.54% | [-4.40, 7.83] | 0.7102 |
| 62%â€“100% | 130 | 65.78% | 60.00% | [-13.94, 2.08] | 0.0768 |

| Season | Combined top record | Combined Brier | Raw Brier | Slope probability_positive |
|---|---:|---:|---:|---:|
| 2020 | 12/17 | 0.246400 | 0.257110 | 0.7479 |
| 2021 | 11/18 | 0.246641 | 0.248123 | 0.7078 |
| 2022 | 11/18 | 0.243257 | 0.249514 | 0.3900 |
| 2023 | 11/18 | 0.243672 | 0.254129 | 0.9470 |
| 2024 | 9/18 | 0.246187 | 0.250570 | 0.5085 |
| 2025 | 11/18 | 0.246341 | 0.251240 | 0.7362 |

### chronological

Decisive weeks: combined wins 19, raw wins 11; two-sided exact paired-sign null p=0.2005. Nominees differ in 63/72 weeks. Combined top record 43/72; mean top estimate 64.97%. Rest-of-week accuracy, with weeks weighted equally, 57.24%.

| Comparison | Effect | 95% interval | probability_positive |
|---|---:|---:|---:|
| combined_vs_raw_brier | 0.006136 | [0.001712, 0.010808] | 0.9977 |
| combined_vs_raw_log_loss | 0.012507 | [0.003371, 0.022160] | 0.9973 |
| combined_vs_simple_brier | 0.004247 | [0.000521, 0.008054] | 0.9872 |
| combined_vs_simple_log_loss | 0.008525 | [0.000778, 0.016389] | 0.9853 |
| combined_vs_neutral_brier | 0.004741 | [0.000869, 0.008561] | 0.9907 |
| combined_vs_neutral_log_loss | 0.009509 | [0.001547, 0.017391] | 0.9885 |
| combined_top_vs_rest_accuracy | 2.484969 | [-8.925900, 13.855476] | 0.6658 |
| combined_top_vs_raw_top_accuracy | 11.111111 | [-4.166667, 25.000000] | 0.9274 |
| raw within-week slope, accuracy points per 10 confidence points | -0.337 | [-7.233, 6.720] | 0.4720 |
| simple within-week slope, accuracy points per 10 confidence points | 0.007 | [-19.232, 19.959] | 0.5073 |
| combined within-week slope, accuracy points per 10 confidence points | 2.425 | [-4.382, 9.195] | 0.7525 |

| Combined probability band | Games | Mean estimate | Actual | Gap interval, accuracy points | probability_positive |
|---|---:|---:|---:|---:|---:|
| 50%â€“52% | 249 | 50.97% | 54.22% | [-2.96, 9.69] | 0.8458 |
| 52%â€“55% | 281 | 53.42% | 58.01% | [-1.51, 10.47] | 0.9271 |
| 55%â€“58% | 222 | 56.51% | 58.11% | [-4.58, 7.66] | 0.6972 |
| 58%â€“62% | 183 | 59.72% | 57.38% | [-9.75, 5.11] | 0.2621 |
| 62%â€“100% | 112 | 65.51% | 62.50% | [-12.17, 5.95] | 0.2592 |

| Season | Combined top record | Combined Brier | Raw Brier | Slope probability_positive |
|---|---:|---:|---:|---:|
| 2022 | 11/18 | 0.242871 | 0.249514 | 0.8228 |
| 2023 | 9/18 | 0.245732 | 0.254129 | 0.5706 |
| 2024 | 12/18 | 0.245927 | 0.250570 | 0.4904 |
| 2025 | 11/18 | 0.246341 | 0.251240 | 0.7362 |

### Fold coefficients and counted looks

| Held-out season | Intercept | Model logit | Flag sum | Market move | Move available |
|---|---:|---:|---:|---:|---:|
| 2020 | -0.044059 | 0.350165 | 0.247496 | 0.208955 | 0.032826 |
| 2021 | -0.061307 | 0.187054 | 0.274666 | 0.212145 | 0.049069 |
| 2022 | -0.053897 | 0.259329 | 0.241574 | 0.211137 | 0.043770 |
| 2023 | -0.037051 | 0.347358 | 0.246704 | 0.225369 | 0.011746 |
| 2024 | -0.055550 | 0.262501 | 0.266336 | 0.246874 | 0.043893 |
| 2025 | -0.053458 | 0.296682 | 0.297235 | 0.158283 | 0.048714 |

In-sample optimism: Brier 0.000527; accuracy 0.200 points. All fold coefficients are retained in `coefficients.csv`, including chronological fits. The model-logit, flag-sum and market-move terms have positive signs in each leave-one-season-out fold; no coefficient is selected from the winning fold.

Family `confidence_best_pick_audit_20260920` contains 183 recorded inferential cells across the original and corrected replays. Inventories: `{"aligned_discrete": {"arm_metric_cells": 36, "coefficient_fit_calls": 32, "paired_comparisons": 24, "pooled_slopes": 9, "probability_arms": 4, "protocols": 3, "reliability_bands": 43, "season_slopes": 16}, "legacy_smooth": {"not_recorded_in_earlier_replay": true}}`. The corrected mapping is one structural comparison; the two paired-sign nulls above are two additional reported looks. These cells overlap heavily and must not be pooled as independent confirmations. Degenerate tiny-band bootstraps were recorded without a spurious certainty estimate.

Research-residue proposal: the existing Gaussian cover-curve fixture test overlaps the calibrated no-Gaussian-payload test and retains a legacy name. Retained pending owner decision; no test files/functions were added or removed.
