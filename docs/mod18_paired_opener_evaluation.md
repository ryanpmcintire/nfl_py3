# MOD-18 paired opener measurement, 2026-09-27

**Measured:** on 950 jointly decisive games, FanDuel finished 494-456 (52.00%) and consensus 502-448 (52.84%). The predeclared FanDuel-minus-consensus accuracy difference is **-0.8421 percentage points**, with a 95% season-bootstrap percentile interval **[-2.1921, +0.2137]** and **probability_positive 0.10115**.

This measurement does not justify a serving change. The signal remains `unresolved_below_power`, recorded under `fanduel_minus_consensus_calibrated_accuracy` in `registry/weak_signals.json`. No refuted mechanism, zero split-half reliability, or positive-control bound has been established. AGENTS.md permits closure only on those admissible grounds; historical forced-pick accuracy does not establish profitability or a current game's probability.

## Declared comparison and population

**Read:** the frozen packet `.tmp/mod18-evaluator-recipe.md` declares family `MOD18_FANDUEL_SINGLE_BOOK_V1`, one primary look, and the sole accuracy contrast above. The primary contrast remains separate from the full analysis accounting required by AGENTS.md: two source arms; three probability streams per source (six reported arms and eighteen arm-level metric cells); nine aggregate contrast cells (the primary calibrated accuracy comparison, two calibrated proper-score comparisons, and six baseline diagnostic comparisons), making twenty-seven aggregate metric cells in total; four season contrasts; eight Platt calibration fits with their in/out diagnostics; five prior-season margin fits and eight fitted home-offset folds; and sixty predeclared reliability cells (six streams times ten bins), of which twenty-two are nonempty. All belong to the declared family. No best cell, cut point or calibration method was selected after observing signs.

**Measured:** current source manifests were regenerated against an immutable feature copy because the earlier feature hash was no longer available. Both arms authenticated the same 241 raw sources. Configuration: weak_stack, market_residual, ridge alpha 10, gaussian_median, minimum 500 prior training games; discrete line-conditional cover/push probabilities and prior-season home offsets. Arm-specific Platt calibration is fitted on completed prior seasons only. Each calibrated probability selects home at or above 0.5, away below 0.5.

The producer emitted 1,153 matched games for 2021-2025: 190, 220, 240, 250 and 253 respectively, with ten arm/fold records. It retained 287 exclusion rows: 195 missing authenticated price, 76 missing feature row, six no valid book at the selected line and ten season/week mismatches. Exclusion rows can overlap by arm or source and are not a unique-game attrition count. There is no authenticated pre-2020 lattice; 2021 supplies calibration history. Evaluation covers 2022-2025: 963 paired games, then 950 jointly decisive games after excluding 13 FanDuel pushes and zero consensus pushes.

An initial preparation using the CLI's default ECDF setting was stopped before evaluation. The sole evaluated run used the explicit frozen configuration; there was no method comparison.

## Same-game baselines

**Measured:** all rows below use the same 950 jointly decisive games. Lower Brier/log loss is better. Model-only means the raw discrete model probability before the fitted home offset and Platt calibration.

| Source | Arm | Accuracy | Brier | Log loss |
| --- | --- | --- | --- | --- |
| calibrated | fanduel | 52.00% | 0.249851 | 0.692885 |
| calibrated | consensus | 52.84% | 0.249824 | 0.692845 |
| market | fanduel | 48.21% | 0.250108 | 0.693362 |
| market | consensus | 49.68% | 0.250063 | 0.693273 |
| model_only | fanduel | 51.05% | 0.252129 | 0.697624 |
| model_only | consensus | 52.42% | 0.251630 | 0.696619 |

**Measured:** FanDuel-minus-consensus Brier difference is +0.00002714 (season-bootstrap interval [-0.00026001, +0.00029899], probability_positive 0.549); log-loss difference is +0.00004017 ([-0.00052606, +0.00057623], probability_positive 0.549). Positive proper-score differences favor consensus. These are secondary comparisons included in the analysis accounting above.

## Season stability

**Measured:** jointly decisive records precede any aggregate interpretation.

| Season | Games | FanDuel W-L | Consensus W-L | Accuracy delta, pp |
| --- | --- | --- | --- | --- |
| 2022 | 215 | 122-93 | 121-94 | +0.47 |
| 2023 | 235 | 116-119 | 123-112 | -2.98 |
| 2024 | 247 | 121-126 | 123-124 | -0.81 |
| 2025 | 253 | 135-118 | 135-118 | +0.00 |

The direction is not stable across seasons: one positive, two negative, one tie. There are only four season blocks. The 20,000 joint draws (seed 20260817) preserve pairing, but the percentile interval is coarse and should not be read as finely calibrated coverage. No split-half reliability or positive-control sensitivity was measured.

## Calibration folds

**Measured:** folds use each arm's own nonpush rows; their denominators differ from the jointly decisive comparison above. Gaps are out-of-sample minus in-sample accuracy in percentage points. Coefficients remain positive but are smaller by 2025; the table preserves their full trajectory.

| Season | Arm | Fit rows | Coefficient | Intercept | In accuracy | Out accuracy | Gap, pp | In Brier | Out Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | consensus | 190 | 0.734616 | 0.028423 | 53.68% | 56.82% | +3.13 | 0.247316 | 0.247273 |
| 2022 | fanduel | 186 | 0.613771 | 0.028474 | 51.61% | 56.74% | +5.13 | 0.248308 | 0.246996 |
| 2023 | consensus | 410 | 0.776415 | 0.035037 | 55.37% | 52.50% | -2.87 | 0.247286 | 0.250994 |
| 2023 | fanduel | 401 | 0.754863 | 0.043736 | 54.86% | 49.36% | -5.50 | 0.247519 | 0.251519 |
| 2024 | consensus | 650 | 0.581832 | 0.047383 | 54.15% | 50.40% | -3.75 | 0.248452 | 0.251277 |
| 2024 | fanduel | 636 | 0.533386 | 0.059552 | 50.79% | 48.99% | -1.80 | 0.248721 | 0.251660 |
| 2025 | consensus | 900 | 0.427602 | 0.030542 | 53.33% | 53.36% | +0.03 | 0.249130 | 0.248621 |
| 2025 | fanduel | 883 | 0.354114 | 0.041265 | 50.28% | 53.36% | +3.08 | 0.249398 | 0.248961 |

## Reliability

**Measured:** calibrated home-cover probability and observed home-cover rate on the same 950 jointly decisive games. Small tail bins are descriptive only; no bin was selected as a rule. Market and model-only reliability rows are retained in the artifact.

| Arm | Probability bin | Games | Mean probability | Observed rate |
| --- | --- | --- | --- | --- |
| fanduel | 0.3-0.4 | 9 | 0.3813 | 0.5556 |
| fanduel | 0.4-0.5 | 384 | 0.4712 | 0.4844 |
| fanduel | 0.5-0.6 | 548 | 0.5313 | 0.5255 |
| fanduel | 0.6-0.7 | 9 | 0.6156 | 0.4444 |
| consensus | 0.3-0.4 | 15 | 0.3777 | 0.5333 |
| consensus | 0.4-0.5 | 409 | 0.4677 | 0.4743 |
| consensus | 0.5-0.6 | 515 | 0.5337 | 0.5379 |
| consensus | 0.6-0.7 | 11 | 0.6115 | 0.2727 |

## Reproduction and evidence

- Producer: `artifacts/paired_opener_inputs/20260927-frozen-config`.
- Evaluation: `artifacts/paired_opener_evaluation/20260927-frozen-config`, containing predictions, fold diagnostics, metrics, reliability, bootstrap summary/draws and hashed metadata.
- Root verification: `.tmp/resume-paired-real-verification.json` and `.tmp/resume-paired-evaluation-review.json`; every output hash, canonical result-minus-line grade, probability bound and prior-season provenance passed. Selected sides matched the single calibrated probability.
- Registry command evidence: `.tmp/resume-mod18-weak-record.log`.
- Producer command: `nfl-ats paired-opener-inputs --features .tmp/resume-mod18-source-inputs/game_features_weak_stack.parquet --fd-manifest .tmp/resume-mod18-source-inputs/fanduel/input_manifest.json --consensus-manifest .tmp/resume-mod18-source-inputs/consensus/input_manifest.json --seasons 2021 2022 2023 2024 2025 --feature-profile weak_stack --regressor ridge --ridge-alpha 10 --probability-method gaussian_median --min-train-games 500 --output artifacts/paired_opener_inputs/20260927-frozen-config`.
- Controlled driver: `.tmp/mod18_run_evaluation.py --producer-dir artifacts/paired_opener_inputs/20260927-frozen-config --output-dir artifacts/paired_opener_evaluation/20260927-frozen-config`. It authenticates output hashes before calling `evaluate_paired_openers`, with completed seasons 2021-2025, evaluation seasons 2022-2025, FanDuel as arm A, consensus as arm B, market/model-only baseline columns, ten reliability bins, confidence 0.95 and the declared bootstrap settings.

Raw data and model/evaluation artifacts remain local and untracked. The measurement is research evidence only; no enrollment, side override or serving promotion is made.

**Measured performance verification:** the bounded season-bootstrap optimization reproduces all 20,000 accuracy draws and ordering exactly, with unchanged probability_positive for all metrics. Maximum Brier/log-loss draw drift is 4.44e-16. Recomputing from cached predictions took 0.50 seconds; this excludes the roughly 13-minute authenticated input preparation. The original measurement artifacts remain unchanged. Evidence: `.tmp/resume-paired-bootstrap-real-parity.json`.
