# LEAD-68 unit 1: model slope by weeks of information

**Measured:** `.tools/uv.exe run --no-sync python scripts/lead68_unit1.py`.

## Population and protocol

**Read:** frozen source `artifacts/four_term_probability/20260914T222345Z/per_game.csv`, SHA-256 `fd17c523355f4e2bb5aa2b91d121447799c2c5af32514fd92ac4fb66678adeaa`. **Measured:** 1,503 unique non-push opener games, seasons 2020-2025. The baseline has model logit, flag_sum, move, move_available and an intercept. The candidate replaces only model logit with logit multiplied by the fixed weeks 1-4, 5-9 and 10-18 indicators. All coefficients and standardization fit on the other five seasons, fixed L2=0.001, 50 Newton iterations; no tuning. Four declared looks: three phase slopes and the joint fit. Reference baselines and descriptive reliability cells do not select new candidates.

**Read:** protocol frozen before outcome access in `docs/lanes/lead68.md`. **Inferred:** requested retrospective LOSO includes later seasons in earlier holdout training; this is not a prospective rolling evaluation. Cached inputs inherit the original evaluator's timing and model limitations; this unit does not re-audit feature provenance or change served output.

**Measured:** four-term reproduction maximum absolute probability error 1.66533453694e-16.

## Decisive record and uncertainty

**Measured:** 121 games change side: candidate 57-64, baseline 64-57. All games: candidate 840-663, baseline 847-656. One fitted probability selects each side at 0.5; no isolated flip rule.

**Measured:** 2,000 paired whole-season bootstrap draws, seed 68, six seasons sampled with replacement, 95% percentile intervals. probability_positive assigns half weight to exact zero draws. Fixed LOSO predictions are not refitted in the bootstrap; six clusters limit precision. Positive effects favor the candidate. No best phase or reliability bin is selected.

| Metric | Candidate improvement | 95% interval | probability_positive |
| --- | --- | --- | --- |
| log loss | 0.000339 | [-0.001465, 0.002021] | 0.659500 |
| Brier | 0.000136 | [-0.000750, 0.000966] | 0.630500 |
| accuracy points | -0.465735 | [-1.797603, 0.593276] | 0.255750 |

## Held-out metrics

**Measured:** metric estimates with season-bootstrap 95% intervals. Market-even is the explicit uninformative 0.5 opener-cover baseline with no directional pick; raw_model is the unchanged model-only reference.

| Arm | Log loss | Brier | Accuracy % | W-L |
| --- | --- | --- | --- | --- |
| four_term | 0.684032 [0.681448, 0.686571] | 0.245437 [0.244074, 0.246706] | 56.353959 [55.236140, 57.738095] | 847-656 |
| phase_slopes | 0.683693 [0.680781, 0.687128] | 0.245301 [0.243844, 0.246943] | 55.888224 [54.320622, 57.532379] | 840-663 |
| raw_model | 0.697002 [0.692136, 0.702656] | 0.251769 [0.249446, 0.254399] | 53.825682 [52.671233, 54.808959] | 809-694 |
| market_even | 0.693147 [0.693147, 0.693147] | 0.250000 [0.250000, 0.250000] | n/a | n/a (no directional pick) |

## Training and held-out gap

**Measured:** training pools 7,515 fit/game evaluations (five appearances per game); descriptive only. Gap = held-out minus training; accuracy in points.

| Arm | Metric | Training | Held out | Gap |
| --- | --- | --- | --- | --- |
| four_term | log loss | 0.682606 | 0.684032 | 0.001426 |
| four_term | Brier | 0.244764 | 0.245437 | 0.000673 |
| four_term | accuracy % | 57.045908 | 56.353959 | -0.691949 |
| phase_slopes | log loss | 0.681508 | 0.683693 | 0.002185 |
| phase_slopes | Brier | 0.244259 | 0.245301 | 0.001041 |
| phase_slopes | accuracy % | 56.606786 | 55.888224 | -0.718563 |

## Season stability

**Measured:** each metric is training / held-out / gap; accuracy in percent.

| Arm | Holdout | n train/test | Log loss | Brier | Accuracy % |
| --- | --- | --- | --- | --- | --- |
| four_term | 2020 | 1283/220 | 0.682600 / 0.684277 / 0.001677 | 0.244744 / 0.245631 / 0.000887 | 57.443492 / 55.000000 / -2.443492 |
| four_term | 2021 | 1267/236 | 0.682093 / 0.687022 / 0.004929 | 0.244515 / 0.246960 / 0.002444 | 57.221784 / 54.661017 / -2.560767 |
| four_term | 2022 | 1255/248 | 0.683160 / 0.680768 / -0.002391 | 0.245040 / 0.243830 / -0.001210 | 56.414343 / 59.677419 / 3.263077 |
| four_term | 2023 | 1237/266 | 0.683649 / 0.678702 / -0.004947 | 0.245305 / 0.242667 / -0.002638 | 56.669361 / 56.390977 / -0.278384 |
| four_term | 2024 | 1237/266 | 0.681782 / 0.687703 / 0.005921 | 0.244370 / 0.247145 / 0.002776 | 57.235247 / 56.390977 / -0.844269 |
| four_term | 2025 | 1236/267 | 0.682355 / 0.685872 / 0.003516 | 0.244612 / 0.246481 / 0.001869 | 57.281553 / 55.805243 / -1.476310 |
| phase_slopes | 2020 | 1283/220 | 0.681823 / 0.681700 / -0.000123 | 0.244398 / 0.244393 / -0.000004 | 56.975838 / 55.909091 / -1.066747 |
| phase_slopes | 2021 | 1267/236 | 0.681201 / 0.686037 / 0.004836 | 0.244110 / 0.246501 / 0.002391 | 56.511444 / 54.237288 / -2.274156 |
| phase_slopes | 2022 | 1255/248 | 0.682242 / 0.679548 / -0.002695 | 0.244624 / 0.243212 / -0.001411 | 56.015936 / 59.274194 / 3.258257 |
| phase_slopes | 2023 | 1237/266 | 0.682191 / 0.680025 / -0.002167 | 0.244604 / 0.243450 / -0.001154 | 56.588521 / 56.015038 / -0.573483 |
| phase_slopes | 2024 | 1237/266 | 0.679922 / 0.691053 / 0.011132 | 0.243517 / 0.248789 / 0.005272 | 56.588521 / 53.007519 / -3.581002 |
| phase_slopes | 2025 | 1236/267 | 0.681657 / 0.683438 / 0.001781 | 0.244296 / 0.245294 / 0.000999 | 56.957929 / 56.928839 / -0.029090 |

## Reliability

**Measured:** fixed evaluator bins, left closed/right open except 1.0. Five descriptive bins per arm; no cell-based side selection.

| Arm | Probability bin | n | Mean p(home cover) | Home-cover rate |
| --- | --- | --- | --- | --- |
| four_term | 0.0-0.2 | 0 | n/a | n/a |
| four_term | 0.2-0.4 | 104 | 0.362331 | 0.413462 |
| four_term | 0.4-0.6 | 1274 | 0.493416 | 0.486656 |
| four_term | 0.6-0.8 | 125 | 0.637399 | 0.664000 |
| four_term | 0.8-1.0 | 0 | n/a | n/a |
| phase_slopes | 0.0-0.2 | 0 | n/a | n/a |
| phase_slopes | 0.2-0.4 | 129 | 0.359451 | 0.387597 |
| phase_slopes | 0.4-0.6 | 1250 | 0.496042 | 0.491200 |
| phase_slopes | 0.6-0.8 | 124 | 0.640200 | 0.661290 |
| phase_slopes | 0.8-1.0 | 0 | n/a | n/a |
| raw_model | 0.0-0.2 | 0 | n/a | n/a |
| raw_model | 0.2-0.4 | 176 | 0.367203 | 0.500000 |
| raw_model | 0.4-0.6 | 1272 | 0.492725 | 0.490566 |
| raw_model | 0.6-0.8 | 55 | 0.626208 | 0.618182 |
| raw_model | 0.8-1.0 | 0 | n/a | n/a |
| market_even | 0.0-0.2 | 0 | n/a | n/a |
| market_even | 0.2-0.4 | 0 | n/a | n/a |
| market_even | 0.4-0.6 | 1503 | 0.500000 | 0.496341 |
| market_even | 0.6-0.8 | 0 | n/a | n/a |
| market_even | 0.8-1.0 | 0 | n/a | n/a |

## Per-fold coefficients

**Measured:** natural units, conditional penalized-Hessian normal 95% working intervals and normal-approximation probability_positive. These are not season-cluster intervals or multiplicity-adjusted evidence. Folds overlap in training. Intercept variance includes the centering transformation.

### four_term

| Holdout | Term | Coefficient | 95% working interval | probability_positive |
| --- | --- | --- | --- | --- |
| 2020 | intercept | -0.051222 | [-0.247371, 0.144928] | 0.304390 |
| 2020 | x0_model_logit | 0.316892 | [-0.125872, 0.759656] | 0.919658 |
| 2020 | flag_sum | 0.243357 | [0.116135, 0.370578] | 0.999911 |
| 2020 | move | 0.209201 | [0.059676, 0.358727] | 0.996948 |
| 2020 | move_available | 0.030751 | [-0.214259, 0.275760] | 0.597155 |
| 2021 | intercept | -0.079635 | [-0.274781, 0.115512] | 0.211909 |
| 2021 | x0_model_logit | 0.092304 | [-0.336982, 0.521590] | 0.663279 |
| 2021 | flag_sum | 0.278927 | [0.147935, 0.409918] | 0.999985 |
| 2021 | move | 0.213517 | [0.063733, 0.363300] | 0.997396 |
| 2021 | move_available | 0.062053 | [-0.181437, 0.305543] | 0.691283 |
| 2022 | intercept | -0.066205 | [-0.276803, 0.144393] | 0.268899 |
| 2022 | x0_model_logit | 0.243226 | [-0.192923, 0.679374] | 0.862805 |
| 2022 | flag_sum | 0.244771 | [0.113215, 0.376327] | 0.999867 |
| 2022 | move | 0.210817 | [0.061304, 0.360329] | 0.997142 |
| 2022 | move_available | 0.047626 | [-0.209851, 0.305104] | 0.641526 |
| 2023 | intercept | -0.059866 | [-0.229866, 0.110133] | 0.245031 |
| 2023 | x0_model_logit | 0.250225 | [-0.182534, 0.682983] | 0.871449 |
| 2023 | flag_sum | 0.248096 | [0.116751, 0.379441] | 0.999893 |
| 2023 | move | 0.224807 | [0.034387, 0.415227] | 0.989664 |
| 2023 | move_available | 0.029781 | [-0.219588, 0.279151] | 0.592535 |
| 2024 | intercept | -0.077872 | [-0.247432, 0.091687] | 0.184023 |
| 2024 | x0_model_logit | 0.176599 | [-0.249289, 0.602487] | 0.791810 |
| 2024 | flag_sum | 0.278200 | [0.146327, 0.410072] | 0.999982 |
| 2024 | move | 0.249887 | [0.068818, 0.430955] | 0.996584 |
| 2024 | move_available | 0.058788 | [-0.186038, 0.303614] | 0.681048 |
| 2025 | intercept | -0.075086 | [-0.245316, 0.095143] | 0.193651 |
| 2025 | x0_model_logit | 0.210137 | [-0.220565, 0.640838] | 0.830528 |
| 2025 | flag_sum | 0.299966 | [0.168992, 0.430940] | 0.999996 |
| 2025 | move | 0.158914 | [-0.021877, 0.339706] | 0.957537 |
| 2025 | move_available | 0.058869 | [-0.184452, 0.302191] | 0.682320 |

### phase_slopes

| Holdout | Term | Coefficient | 95% working interval | probability_positive |
| --- | --- | --- | --- | --- |
| 2020 | intercept | -0.059399 | [-0.257426, 0.138629] | 0.278302 |
| 2020 | weeks_1_4 | 0.803275 | [-0.010879, 1.617429] | 0.973430 |
| 2020 | weeks_5_9 | 0.131636 | [-0.622852, 0.886124] | 0.633808 |
| 2020 | weeks_10_18 | 0.132997 | [-0.525879, 0.791873] | 0.653810 |
| 2020 | flag_sum | 0.243427 | [0.116103, 0.370750] | 0.999911 |
| 2020 | move | 0.213244 | [0.063441, 0.363048] | 0.997365 |
| 2020 | move_available | 0.047156 | [-0.199113, 0.293424] | 0.646280 |
| 2021 | intercept | -0.092510 | [-0.289482, 0.104462] | 0.178652 |
| 2021 | weeks_1_4 | 0.600166 | [-0.218236, 1.418568] | 0.924686 |
| 2021 | weeks_5_9 | -0.209997 | [-0.984347, 0.564353] | 0.297527 |
| 2021 | weeks_10_18 | -0.013841 | [-0.633569, 0.605887] | 0.482543 |
| 2021 | flag_sum | 0.278039 | [0.146942, 0.409137] | 0.999984 |
| 2021 | move | 0.217507 | [0.067469, 0.367545] | 0.997754 |
| 2021 | move_available | 0.078625 | [-0.166150, 0.323399] | 0.735510 |
| 2022 | intercept | -0.073476 | [-0.285352, 0.138399] | 0.248349 |
| 2022 | weeks_1_4 | 0.741135 | [-0.067017, 1.549287] | 0.963866 |
| 2022 | weeks_5_9 | 0.203956 | [-0.530548, 0.938460] | 0.706862 |
| 2022 | weeks_10_18 | -0.018204 | [-0.633152, 0.596745] | 0.476867 |
| 2022 | flag_sum | 0.243960 | [0.112342, 0.375577] | 0.999860 |
| 2022 | move | 0.214767 | [0.064997, 0.364538] | 0.997527 |
| 2022 | move_available | 0.068382 | [-0.190776, 0.327540] | 0.697478 |
| 2023 | intercept | -0.063948 | [-0.235577, 0.107681] | 0.232612 |
| 2023 | weeks_1_4 | 0.903639 | [0.082536, 1.724743] | 0.984496 |
| 2023 | weeks_5_9 | 0.147440 | [-0.575281, 0.870161] | 0.655365 |
| 2023 | weeks_10_18 | -0.032314 | [-0.633827, 0.569199] | 0.458072 |
| 2023 | flag_sum | 0.243740 | [0.112236, 0.375244] | 0.999860 |
| 2023 | move | 0.226660 | [0.036174, 0.417146] | 0.990154 |
| 2023 | move_available | 0.047351 | [-0.203067, 0.297768] | 0.644533 |
| 2024 | intercept | -0.092012 | [-0.263134, 0.079111] | 0.145973 |
| 2024 | weeks_1_4 | 0.876660 | [0.097779, 1.655541] | 0.986308 |
| 2024 | weeks_5_9 | -0.087048 | [-0.794660, 0.620564] | 0.404735 |
| 2024 | weeks_10_18 | -0.081326 | [-0.697821, 0.535169] | 0.397991 |
| 2024 | flag_sum | 0.275623 | [0.143569, 0.407677] | 0.999979 |
| 2024 | move | 0.255674 | [0.074076, 0.437272] | 0.997105 |
| 2024 | move_available | 0.086763 | [-0.159836, 0.333361] | 0.754774 |
| 2025 | intercept | -0.080796 | [-0.252184, 0.090592] | 0.177751 |
| 2025 | weeks_1_4 | 0.657124 | [-0.143620, 1.457868] | 0.946129 |
| 2025 | weeks_5_9 | 0.095799 | [-0.607243, 0.798840] | 0.605293 |
| 2025 | weeks_10_18 | 0.029061 | [-0.592500, 0.650622] | 0.536507 |
| 2025 | flag_sum | 0.299378 | [0.168344, 0.430412] | 0.999996 |
| 2025 | move | 0.164588 | [-0.016529, 0.345706] | 0.962551 |
| 2025 | move_available | 0.072762 | [-0.171749, 0.317272] | 0.720136 |

## Decision and limitations

**Inferred:** retain unresolved_below_power pending the orchestrator's serial registry recording; no terminal closure or serving change. AGENTS.md:65-84 requires admissible closing evidence; AGENTS.md:115-120 separates closure from serving. Three phase slopes form one joint candidate. Historical accuracy is not a game probability or profitability evidence.

**Read:** this is the historic four-term population, not a newly materialized active-card forecast. **Measured:** prediction rows preserved in `docs/lead68_predictions.md`, a generated local research artifact; do not commit that processed-data file. No registry command was run.
