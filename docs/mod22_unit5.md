# MOD-22 unit 5: expected QB quality loss

**Measured:** repaired-timestamp remeasurement of the one declared quality-gap look; look count
remains 1, graded at the Tuesday opener. Decisive-game record: candidate 4-12 on 16 changed sides
(two-sided exact null p=0.0768).

**Measured:** 1503 nonpush games in 107 season/week blocks; 53 games have a nonzero quality-loss
difference. Late-out team games: 55; missing starter quality: 5; missing replacement quality: 17; no
eligible backup: 0. Starter identity matches: 3006/3006.

**Measured:** starter Out/Doubtful timestamp audit on scored team games: 73 cutoff-only flags;
excluded reports: 18 proxy, 18 total unevidenced (including proxies), 1 evidenced but late. All-
source starter matches: 3230/3230.

**Measured:** positive effects mean improvement over the refitted four-term baseline. Paired 95%
intervals resample seasons and whole weeks within season (2,000 draws).

| Metric | Effect | 95% interval | probability_positive |
|---|---:|---:|---:|
| Accuracy, percentage points | -0.5322688 | [-1.5042758, 0.1944296] | 0.0963 |
| Brier improvement | -0.0010598 | [-0.0031625, -0.0000801] | 0.0045 |
| Log-loss improvement | -0.0026542 | [-0.0083148, -0.0001811] | 0.0045 |

## Probability scores and fit gaps

**Measured:** p >= 0.5 selects home. The 0.5 market reference breaks ties toward home.

| Fit | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| candidate OOS | 0.568862 | 0.245898 | 0.685428 |
| four-term OOS | 0.574185 | 0.244838 | 0.682774 |
| candidate in-sample | 0.576181 | 0.244388 | 0.681812 |
| four-term in-sample | 0.575516 | 0.244389 | 0.681816 |
| model only | 0.533599 | 0.251715 | 0.696917 |
| market | 0.496341 | 0.250000 | 0.693147 |

**Measured:** gaps are in-sample minus out-of-season scores.

| Fit | Accuracy gap | Brier gap | Log-loss gap |
|---|---:|---:|---:|
| candidate | 0.007319 | -0.001511 | -0.003616 |
| four-term | 0.001331 | -0.000450 | -0.000958 |

## Season stability

**Measured:** descriptive held-out-season diagnostics; no season selection.

| Season | Games | Candidate accuracy | Baseline accuracy | Accuracy delta, pp | Brier improvement | Log-loss improvement |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 220 | 0.554545 | 0.554545 | 0.000000 | -0.0001269 | -0.0002581 |
| 2021 | 236 | 0.559322 | 0.563559 | -0.423729 | -0.0004281 | -0.0008608 |
| 2022 | 248 | 0.580645 | 0.604839 | -2.419355 | -0.0045076 | -0.0119708 |
| 2023 | 266 | 0.560150 | 0.563910 | -0.375940 | -0.0008379 | -0.0018683 |
| 2024 | 266 | 0.582707 | 0.582707 | 0.000000 | -0.0004690 | -0.0010050 |
| 2025 | 267 | 0.573034 | 0.573034 | 0.000000 | 0.0000057 | 0.0000139 |

## Combined coefficients

**Measured:** natural feature units; all five terms and the intercept are fitted outside the scored
season.

| Held-out season | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available | qb_quality_loss_diff |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | -0.0450176 | 0.3410619 | 0.2447994 | 0.2198484 | 0.0281575 | 0.1148011 |
| 2021 | -0.0618005 | 0.1786715 | 0.2716061 | 0.2218004 | 0.0442737 | 0.2489231 |
| 2022 | -0.0569206 | 0.2475427 | 0.2362878 | 0.2327210 | 0.0380999 | -1.2681167 |
| 2023 | -0.0371675 | 0.3429517 | 0.2456117 | 0.2281666 | 0.0019275 | 0.4030649 |
| 2024 | -0.0564619 | 0.2573103 | 0.2621540 | 0.2483335 | 0.0384880 | -0.3496348 |
| 2025 | -0.0541744 | 0.2922920 | 0.2959971 | 0.1892597 | 0.0474025 | -0.0455858 |
| In-sample | -0.0521707 | 0.2764637 | 0.2599728 | 0.2222864 | 0.0341202 | -0.0651725 |

**Measured:** quality-loss coefficient positive in 3/6 folds; range [-1.2681167, 0.4030649].

## Reliability

**Measured:** fixed inherited bins; all references use the same games.

| Fit | Probability bin | Games | Mean probability | Home-cover rate |
|---|---|---:|---:|---:|
| full | 0.00-0.40 | 117 | 0.355387 | 0.376068 |
| full | 0.40-0.45 | 242 | 0.425155 | 0.483471 |
| full | 0.45-0.50 | 484 | 0.475459 | 0.425620 |
| full | 0.50-0.55 | 346 | 0.526659 | 0.531792 |
| full | 0.55-0.60 | 185 | 0.569735 | 0.600000 |
| full | 0.60-1.00 | 129 | 0.645809 | 0.651163 |
| reduced | 0.00-0.40 | 114 | 0.357291 | 0.368421 |
| reduced | 0.40-0.45 | 243 | 0.425372 | 0.477366 |
| reduced | 0.45-0.50 | 488 | 0.475198 | 0.422131 |
| reduced | 0.50-0.55 | 347 | 0.526717 | 0.544669 |
| reduced | 0.55-0.60 | 182 | 0.569783 | 0.593407 |
| reduced | 0.60-1.00 | 129 | 0.643683 | 0.658915 |
| model | 0.00-0.40 | 215 | 0.365033 | 0.488372 |
| model | 0.40-0.45 | 292 | 0.425995 | 0.438356 |
| model | 0.45-0.50 | 387 | 0.475136 | 0.480620 |
| model | 0.50-0.55 | 360 | 0.522609 | 0.536111 |
| model | 0.55-0.60 | 176 | 0.572526 | 0.534091 |
| model | 0.60-1.00 | 73 | 0.624468 | 0.547945 |
| market | 0.50-0.55 | 1503 | 0.500000 | 0.496341 |

## Interpretation and reproducibility

**Inferred:** this single quality-weighted construct remains pending orchestrator adjudication. It
does not generalize the binary-flag verdict. No power-matched positive control or split-half
reliability was measured. AGENTS.md:65-85 governs research closure; AGENTS.md:87-105 requires one
fitted probability and out-of-season evaluation.

**Read:** the protocol was saved before outcomes in `docs/lanes/mod22-unit5-qb-quality.md`. Starter
names use the suffix-aware normalizer imported from repaired unit 4. Injury visibility mirrors its
evidence filter: basis is date_modified, proxy is false, date_modified exists and equals
effective_observed_at, and is no later than the decision. The declared latest-status, prior-
roster/prior-quality and post-Tuesday checks remain. Quality uses only the immediately preceding
season; missing quality is fixed at -0.15 EPA/dropback.

**Inferred:** evidenced source report times do not establish a complete announcement archive. The
prior-game starter and prior-week roster can miss depth changes and new signings. Missing prior-
season quality can blur real gaps. LOSO is retrospective: training includes other years on both
sides of each fold; pre-existing model predictions are inherited, not independently retrained. No
serving or registry changes were made.

Command: `.tools/uv.exe run --no-sync python scripts/mod22_unit5.py`.

**Measured:** full metrics, baseline coefficients, audit fields and prediction rows:
`tests/scratch/codex/mod22_unit5/20260929T214747Z`. Active four-term market-move version:
`leader_median_through_sunday_prekick_v1`.

**Read:** the paired bootstrap resamples saved OOS scores; it holds the fitted folds fixed. The
inherited four-term baseline uses Sunday prekick market information. The added QB term is frozen at
kickoff minus 24 hours. This is a comparison with the final served baseline, not a claim that all
baseline inputs were available at the earlier QB cutoff.

**Measured:** full OOS records are candidate 855-648 and baseline 863-640.

**Inferred:** the entirely adverse Brier-improvement interval supports a proposed `refuted_mechanism
/ wrong_sign_resolved` entry scoped only to this declared quality proxy, timing gate and fitted
fifth-term construction (AGENTS.md:70-78). It does not establish a reversed football mechanism or
close other measures of QB quality. Accuracy alone is unresolved; the orchestrator must verify and
run the lane command. No registry entry has been written by this worker.
