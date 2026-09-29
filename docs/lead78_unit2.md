# LEAD-78 unit 2: within-week common-opponent news

**Measured:** `.tools/uv.exe run --no-sync python scripts/lead78_unit2.py`.

## Declaration and timing

**Read:** the unit-two amendment in `docs/lanes/lead78.md` was saved before outcome computation. Historical openers replace unavailable pool captures; scheduled kickoff + four hours replaces unavailable published-final timestamps. The original score-state delta remains one fitted fifth term. B=3, F=6, E=0; the look budget is unchanged at 298, including baseline/reporting cells. No searched cutoffs, selected surprise games, or extra challenger specifications.

**Measured:** 1,503 unique non-push REG games, 2020-2025; 107 season-week blocks. 1339 targets have eligible Thursday/Saturday news; 1168 nonzero state deltas. All 1886 source-target pairs passed completion-after-Tuesday and completion-before-deadline/kickoff assertions. No-news games remain with zero delta.

**Read:** completion is a proxy, not an observed publication time. Tuesday freeze is noon Eastern; each deadline is the earlier of kickoff and Sunday 12:45 Eastern. Prior-season scores establish a unit-precision ridge rating prior and home advantage; only scores completed before Tuesday enter the initial current-season graph. Eligible early results update that graph, and the home-minus-away rating change is the term. All four existing coefficients and the fifth coefficient are fitted together; no independent rule flips picks.

**Measured:** refitted four-term held-out probabilities reproduce the frozen served base to maximum error 2.22e-16.

**Inferred:** six-fold retrospective LOSO trains on the other five seasons, including later seasons for earlier holdouts; this is not prospective forward validation. Cached discrete-model and situational inputs inherit their existing provenance. The news feature itself uses only earlier completed scores; no full-history pipeline was rebuilt. Final schedule scores cannot verify absence of later corrections.

## Decisive record and paired gains

**Measured:** on 13 games where candidate and four-term sides differ, the candidate is 4-9-0 and the base is 9-4-0. The population excludes opener pushes. Historical forced-pick records do not establish a profitable edge.

**Measured:** positive gains favor the candidate: accuracy is candidate minus baseline; losses and margin MAE are baseline minus candidate. 95% paired intervals use 10,000 season-stratified week-block draws, seed 78. `probability_positive` counts half of exact ties. These intervals condition on the fitted LOSO predictions.

| Baseline | Metric | Gain | 95% interval | probability_positive |
| --- | --- | --- | --- | --- |
| four_term | accuracy_points | -0.332668 | [-0.805369, 0.133511] | 0.084350 |
| four_term | log_loss | -0.000566 | [-0.001149, -0.000027] | 0.019700 |
| four_term | brier | -0.000283 | [-0.000569, -0.000018] | 0.018100 |
| four_term | margin_mae | -0.007324 | [-0.016144, 0.000529] | 0.033600 |
| model | accuracy_points | 3.725882 | [1.067360, 6.360656] | 0.997150 |
| model | log_loss | 0.013577 | [0.005059, 0.022274] | 0.998800 |
| model | brier | 0.006594 | [0.002531, 0.010787] | 0.998900 |
| model | margin_mae | 0.152789 | [0.045064, 0.265220] | 0.997600 |
| market | accuracy_points | 7.451763 | [3.735749, 11.155556] | 0.999700 |
| market | log_loss | 0.009807 | [0.002869, 0.016652] | 0.997900 |
| market | brier | 0.004879 | [0.001539, 0.008172] | 0.998300 |
| market | margin_mae | 0.155034 | [0.044324, 0.272054] | 0.997300 |
| elo | accuracy_points | 8.316700 | [5.063123, 11.548792] | 1.000000 |
| elo | log_loss | 0.011157 | [0.004121, 0.018147] | 0.999100 |
| elo | brier | 0.005553 | [0.002141, 0.008906] | 0.999400 |
| elo | margin_mae | -0.648527 | [-0.898973, -0.396170] | 0.000000 |

## In-sample and held-out calibration

**Measured:** training means aggregate each fold's five training seasons; each game appears five times. Held-out means include each game once; gap is held-out minus training. Market probability is 0.5 with a deterministic home tie rule. Elo uses K=20, scale=400 and offseason carry=0.75 with training-fold logistic calibration of rating gap and opener. Margin MAE uses a reporting-only linear calibration of each probability's logit to home margin on the training fold; it is not a served margin distribution.

| Arm | Metric | Training | Held out | Held-out 95% interval | Gap |
| --- | --- | --- | --- | --- | --- |
| fifth_term | accuracy_points | 57.418496 | 57.085828 | [54.787190, 59.391585] | -0.332668 |
| fifth_term | log_loss | 0.681656 | 0.683341 | [0.676495, 0.690278] | 0.001685 |
| fifth_term | brier | 0.244311 | 0.245121 | [0.241828, 0.248461] | 0.000810 |
| fifth_term | margin_mae | 11.028933 | 11.057857 | [10.626665, 11.498329] | 0.028925 |
| four_term | accuracy_points | 57.511643 | 57.418496 | [55.149490, 59.667829] | -0.093147 |
| four_term | log_loss | 0.681732 | 0.682774 | [0.675942, 0.689664] | 0.001042 |
| four_term | brier | 0.244350 | 0.244838 | [0.241533, 0.248178] | 0.000488 |
| four_term | margin_mae | 11.031289 | 11.050533 | [10.618964, 11.491732] | 0.019244 |
| model | accuracy_points | 53.359947 | 53.359947 | [50.830537, 55.931164] | 0.000000 |
| model | log_loss | 0.696917 | 0.696917 | [0.689274, 0.704649] | -0.000000 |
| model | brier | 0.251715 | 0.251715 | [0.247992, 0.255487] | -0.000000 |
| model | margin_mae | 11.191343 | 11.210646 | [10.785687, 11.642063] | 0.019304 |
| market | accuracy_points | 49.634065 | 49.634065 | [47.281167, 52.024291] | 0.000000 |
| market | log_loss | 0.693147 | 0.693147 | [0.693147, 0.693147] | -0.000000 |
| market | brier | 0.250000 | 0.250000 | [0.250000, 0.250000] | 0.000000 |
| market | margin_mae | 11.203064 | 11.212891 | [10.787751, 11.640746] | 0.009827 |
| elo | accuracy_points | 50.884897 | 48.769128 | [46.251673, 51.306183] | -2.115768 |
| elo | log_loss | 0.692823 | 0.694497 | [0.693244, 0.695753] | 0.001675 |
| elo | brier | 0.249838 | 0.250675 | [0.250048, 0.251302] | 0.000837 |
| elo | margin_mae | 10.407761 | 10.409330 | [10.013953, 10.808656] | 0.001569 |

## Season stability

**Measured:** season intervals resample weeks within that season.

| Arm | Held out | n train/test | Metric | Training | Held out | 95% interval | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fifth_term | 2020 | 1283/220 | accuracy_points | 57.989088 | 55.909091 | [48.372093, 62.978775] | -2.079997 |
| fifth_term | 2020 | 1283/220 | log_loss | 0.681198 | 0.685896 | [0.673947, 0.698924] | 0.004698 |
| fifth_term | 2020 | 1283/220 | brier | 0.244074 | 0.246413 | [0.240488, 0.252912] | 0.002339 |
| fifth_term | 2020 | 1283/220 | margin_mae | 11.034122 | 11.135439 | [10.106748, 12.282434] | 0.101317 |
| four_term | 2020 | 1283/220 | accuracy_points | 57.989088 | 55.454545 | [48.165138, 62.100809] | -2.534543 |
| four_term | 2020 | 1283/220 | log_loss | 0.681199 | 0.685840 | [0.673875, 0.698929] | 0.004642 |
| four_term | 2020 | 1283/220 | brier | 0.244074 | 0.246386 | [0.240452, 0.252890] | 0.002312 |
| four_term | 2020 | 1283/220 | margin_mae | 11.034038 | 11.134904 | [10.106308, 12.281889] | 0.100866 |
| model | 2020 | 1283/220 | accuracy_points | 53.390491 | 53.181818 | [45.754717, 59.825328] | -0.208673 |
| model | 2020 | 1283/220 | log_loss | 0.694940 | 0.708450 | [0.681580, 0.734928] | 0.013510 |
| model | 2020 | 1283/220 | brier | 0.250790 | 0.257111 | [0.244179, 0.269831] | 0.006321 |
| model | 2020 | 1283/220 | margin_mae | 11.177732 | 11.322091 | [10.325162, 12.427780] | 0.144359 |
| market | 2020 | 1283/220 | accuracy_points | 49.805144 | 48.636364 | [42.241379, 54.377880] | -1.168781 |
| market | 2020 | 1283/220 | log_loss | 0.693147 | 0.693147 | [0.693147, 0.693147] | 0.000000 |
| market | 2020 | 1283/220 | brier | 0.250000 | 0.250000 | [0.250000, 0.250000] | 0.000000 |
| market | 2020 | 1283/220 | margin_mae | 11.177362 | 11.337274 | [10.338432, 12.444503] | 0.159913 |
| elo | 2020 | 1283/220 | accuracy_points | 50.116913 | 51.363636 | [43.438914, 59.259259] | 1.246723 |
| elo | 2020 | 1283/220 | log_loss | 0.692956 | 0.693380 | [0.691473, 0.695335] | 0.000424 |
| elo | 2020 | 1283/220 | brier | 0.249905 | 0.250117 | [0.249163, 0.251094] | 0.000212 |
| elo | 2020 | 1283/220 | margin_mae | 11.026053 | 11.128008 | [10.143745, 12.207503] | 0.101955 |
| fifth_term | 2021 | 1267/236 | accuracy_points | 57.300710 | 56.355932 | [50.000000, 62.761506] | -0.944778 |
| fifth_term | 2021 | 1267/236 | log_loss | 0.681077 | 0.686369 | [0.670311, 0.701736] | 0.005292 |
| fifth_term | 2021 | 1267/236 | brier | 0.244036 | 0.246639 | [0.238742, 0.254233] | 0.002603 |
| fifth_term | 2021 | 1267/236 | margin_mae | 10.757349 | 12.448039 | [11.437322, 13.589573] | 1.690690 |
| four_term | 2021 | 1267/236 | accuracy_points | 57.221784 | 56.355932 | [50.000000, 62.761506] | -0.865852 |
| four_term | 2021 | 1267/236 | log_loss | 0.681083 | 0.686447 | [0.670541, 0.701788] | 0.005364 |
| four_term | 2021 | 1267/236 | brier | 0.244040 | 0.246677 | [0.238831, 0.254257] | 0.002637 |
| four_term | 2021 | 1267/236 | margin_mae | 10.758058 | 12.449229 | [11.439365, 13.590002] | 1.691171 |
| model | 2021 | 1267/236 | accuracy_points | 52.880821 | 55.932203 | [48.132780, 63.636364] | 3.051383 |
| model | 2021 | 1267/236 | log_loss | 0.698275 | 0.689630 | [0.660779, 0.718752] | -0.008645 |
| model | 2021 | 1267/236 | brier | 0.252384 | 0.248123 | [0.234088, 0.262253] | -0.004262 |
| model | 2021 | 1267/236 | margin_mae | 10.933132 | 12.606248 | [11.554009, 13.744372] | 1.673115 |
| market | 2021 | 1267/236 | accuracy_points | 50.039463 | 47.457627 | [41.227981, 53.813559] | -2.581836 |
| market | 2021 | 1267/236 | log_loss | 0.693147 | 0.693147 | [0.693147, 0.693147] | 0.000000 |
| market | 2021 | 1267/236 | brier | 0.250000 | 0.250000 | [0.250000, 0.250000] | 0.000000 |
| market | 2021 | 1267/236 | margin_mae | 10.941023 | 12.617009 | [11.579473, 13.745480] | 1.675986 |
| elo | 2021 | 1267/236 | accuracy_points | 49.092344 | 50.000000 | [43.932554, 55.462185] | 0.907656 |
| elo | 2021 | 1267/236 | log_loss | 0.692866 | 0.693863 | [0.690647, 0.697419] | 0.000998 |
| elo | 2021 | 1267/236 | brier | 0.249859 | 0.250358 | [0.248751, 0.252135] | 0.000499 |
| elo | 2021 | 1267/236 | margin_mae | 10.079639 | 11.657271 | [10.613424, 12.868853] | 1.577632 |
| fifth_term | 2022 | 1255/248 | accuracy_points | 56.494024 | 59.677419 | [54.886989, 64.897959] | 3.183395 |
| fifth_term | 2022 | 1255/248 | log_loss | 0.682020 | 0.681851 | [0.670889, 0.692792] | -0.000169 |
| fifth_term | 2022 | 1255/248 | brier | 0.244507 | 0.244371 | [0.238943, 0.249790] | -0.000136 |
| fifth_term | 2022 | 1255/248 | margin_mae | 11.328967 | 9.512667 | [8.592818, 10.419072] | -1.816300 |
| four_term | 2022 | 1255/248 | accuracy_points | 56.892430 | 60.483871 | [55.465587, 65.447154] | 3.591441 |
| four_term | 2022 | 1255/248 | log_loss | 0.682292 | 0.679696 | [0.668319, 0.691086] | -0.002596 |
| four_term | 2022 | 1255/248 | brier | 0.244641 | 0.243307 | [0.237695, 0.248928] | -0.001334 |
| four_term | 2022 | 1255/248 | margin_mae | 11.336361 | 9.489694 | [8.567930, 10.403863] | -1.846666 |
| model | 2022 | 1255/248 | accuracy_points | 52.988048 | 55.241935 | [49.180159, 60.956175] | 2.253888 |
| model | 2022 | 1255/248 | log_loss | 0.697804 | 0.692429 | [0.678152, 0.708649] | -0.005375 |
| model | 2022 | 1255/248 | brier | 0.252150 | 0.249514 | [0.242463, 0.257471] | -0.002636 |
| model | 2022 | 1255/248 | margin_mae | 11.499538 | 9.639034 | [8.779975, 10.499543] | -1.860504 |
| market | 2022 | 1255/248 | accuracy_points | 49.800797 | 48.790323 | [44.531250, 53.359684] | -1.010474 |
| market | 2022 | 1255/248 | log_loss | 0.693147 | 0.693147 | [0.693147, 0.693147] | 0.000000 |
| market | 2022 | 1255/248 | brier | 0.250000 | 0.250000 | [0.250000, 0.250000] | 0.000000 |
| market | 2022 | 1255/248 | margin_mae | 11.511852 | 9.650749 | [8.801300, 10.497361] | -1.861103 |
| elo | 2022 | 1255/248 | accuracy_points | 49.880478 | 45.967742 | [41.897233, 50.000000] | -3.912736 |
| elo | 2022 | 1255/248 | log_loss | 0.693049 | 0.694974 | [0.693790, 0.696394] | 0.001925 |
| elo | 2022 | 1255/248 | brier | 0.249951 | 0.250913 | [0.250321, 0.251623] | 0.000962 |
| elo | 2022 | 1255/248 | margin_mae | 10.405976 | 9.123215 | [8.254895, 9.967967] | -1.282762 |
| fifth_term | 2023 | 1237/266 | accuracy_points | 58.124495 | 56.390977 | [51.893939, 60.740741] | -1.733517 |
| fifth_term | 2023 | 1237/266 | log_loss | 0.682729 | 0.678268 | [0.660404, 0.696500] | -0.004461 |
| fifth_term | 2023 | 1237/266 | brier | 0.244861 | 0.242470 | [0.233904, 0.251252] | -0.002391 |
| fifth_term | 2023 | 1237/266 | margin_mae | 11.051210 | 10.969575 | [9.891853, 12.083026] | -0.081634 |
| four_term | 2023 | 1237/266 | accuracy_points | 58.205335 | 56.390977 | [51.893939, 60.740741] | -1.814358 |
| four_term | 2023 | 1237/266 | log_loss | 0.682741 | 0.677980 | [0.660084, 0.696256] | -0.004761 |
| four_term | 2023 | 1237/266 | brier | 0.244865 | 0.242324 | [0.233743, 0.251080] | -0.002541 |
| four_term | 2023 | 1237/266 | margin_mae | 11.051441 | 10.962633 | [9.887640, 12.073150] | -0.088808 |
| model | 2023 | 1237/266 | accuracy_points | 54.001617 | 50.375940 | [46.816479, 54.263566] | -3.625677 |
| model | 2023 | 1237/266 | log_loss | 0.695859 | 0.701840 | [0.690977, 0.712586] | 0.005981 |
| model | 2023 | 1237/266 | brier | 0.251196 | 0.254129 | [0.248877, 0.259274] | 0.002933 |
| model | 2023 | 1237/266 | margin_mae | 11.175933 | 11.286788 | [10.181897, 12.420057] | 0.110855 |
| market | 2023 | 1237/266 | accuracy_points | 49.393694 | 50.751880 | [45.283019, 55.925926] | 1.358185 |
| market | 2023 | 1237/266 | log_loss | 0.693147 | 0.693147 | [0.693147, 0.693147] | 0.000000 |
| market | 2023 | 1237/266 | brier | 0.250000 | 0.250000 | [0.250000, 0.250000] | 0.000000 |
| market | 2023 | 1237/266 | margin_mae | 11.189229 | 11.292561 | [10.189922, 12.428766] | 0.103332 |
| elo | 2023 | 1237/266 | accuracy_points | 52.223120 | 48.872180 | [42.804428, 54.850746] | -3.350940 |
| elo | 2023 | 1237/266 | log_loss | 0.692663 | 0.694822 | [0.691517, 0.697878] | 0.002159 |
| elo | 2023 | 1237/266 | brier | 0.249758 | 0.250837 | [0.249185, 0.252363] | 0.001079 |
| elo | 2023 | 1237/266 | margin_mae | 10.251433 | 10.417103 | [9.341277, 11.458881] | 0.165669 |
| fifth_term | 2024 | 1237/266 | accuracy_points | 57.396928 | 58.270677 | [52.631579, 63.369963] | 0.873749 |
| fifth_term | 2024 | 1237/266 | log_loss | 0.681470 | 0.683600 | [0.667007, 0.701149] | 0.002129 |
| fifth_term | 2024 | 1237/266 | brier | 0.244258 | 0.245085 | [0.237208, 0.253462] | 0.000827 |
| fifth_term | 2024 | 1237/266 | margin_mae | 10.971758 | 11.333193 | [10.289438, 12.467853] | 0.361435 |
| four_term | 2024 | 1237/266 | accuracy_points | 57.235247 | 58.270677 | [52.631579, 63.369963] | 1.035430 |
| four_term | 2024 | 1237/266 | log_loss | 0.681473 | 0.683667 | [0.666957, 0.701327] | 0.002195 |
| four_term | 2024 | 1237/266 | brier | 0.244259 | 0.245124 | [0.237214, 0.253510] | 0.000865 |
| four_term | 2024 | 1237/266 | margin_mae | 10.972242 | 11.333695 | [10.291467, 12.466904] | 0.361453 |
| model | 2024 | 1237/266 | accuracy_points | 53.112369 | 54.511278 | [47.876448, 61.132075] | 1.398910 |
| model | 2024 | 1237/266 | log_loss | 0.697456 | 0.694413 | [0.680648, 0.708439] | -0.003043 |
| model | 2024 | 1237/266 | brier | 0.251962 | 0.250570 | [0.243773, 0.257498] | -0.001392 |
| model | 2024 | 1237/266 | margin_mae | 11.167176 | 11.308467 | [10.235365, 12.421803] | 0.141291 |
| market | 2024 | 1237/266 | accuracy_points | 49.474535 | 50.375940 | [43.773585, 57.142857] | 0.901405 |
| market | 2024 | 1237/266 | log_loss | 0.693147 | 0.693147 | [0.693147, 0.693147] | 0.000000 |
| market | 2024 | 1237/266 | brier | 0.250000 | 0.250000 | [0.250000, 0.250000] | 0.000000 |
| market | 2024 | 1237/266 | margin_mae | 11.189175 | 11.280314 | [10.217653, 12.386057] | 0.091139 |
| elo | 2024 | 1237/266 | accuracy_points | 51.738076 | 44.360902 | [36.781609, 51.908397] | -7.377174 |
| elo | 2024 | 1237/266 | log_loss | 0.692426 | 0.696508 | [0.691795, 0.701442] | 0.004082 |
| elo | 2024 | 1237/266 | brier | 0.249639 | 0.251680 | [0.249326, 0.254144] | 0.002041 |
| elo | 2024 | 1237/266 | margin_mae | 10.377440 | 10.080872 | [9.235417, 10.916565] | -0.296568 |
| fifth_term | 2025 | 1236/267 | accuracy_points | 57.200647 | 55.805243 | [50.000000, 61.397059] | -1.395404 |
| fifth_term | 2025 | 1236/267 | log_loss | 0.681464 | 0.684737 | [0.662565, 0.705360] | 0.003274 |
| fifth_term | 2025 | 1236/267 | brier | 0.244143 | 0.246090 | [0.235562, 0.255951] | 0.001947 |
| fifth_term | 2025 | 1236/267 | margin_mae | 11.032221 | 11.014037 | [10.022439, 12.094953] | -0.018184 |
| four_term | 2025 | 1236/267 | accuracy_points | 57.524272 | 57.303371 | [51.514867, 62.773723] | -0.220901 |
| four_term | 2025 | 1236/267 | log_loss | 0.681634 | 0.683748 | [0.661783, 0.704196] | 0.002114 |
| four_term | 2025 | 1236/267 | brier | 0.244234 | 0.245581 | [0.235173, 0.255308] | 0.001347 |
| four_term | 2025 | 1236/267 | margin_mae | 11.037685 | 10.999950 | [9.996853, 12.103670] | -0.037735 |
| model | 2025 | 1236/267 | accuracy_points | 53.802589 | 51.310861 | [45.353160, 57.142857] | -2.491728 |
| model | 2025 | 1236/267 | log_loss | 0.697198 | 0.695617 | [0.681216, 0.710680] | -0.001582 |
| model | 2025 | 1236/267 | brier | 0.251821 | 0.251226 | [0.244152, 0.258669] | -0.000595 |
| model | 2025 | 1236/267 | margin_mae | 11.196832 | 11.171719 | [10.222337, 12.198377] | -0.025113 |
| market | 2025 | 1236/267 | accuracy_points | 49.271845 | 51.310861 | [45.864662, 56.343284] | 2.039017 |
| market | 2025 | 1236/267 | log_loss | 0.693147 | 0.693147 | [0.693147, 0.693147] | 0.000000 |
| market | 2025 | 1236/267 | brier | 0.250000 | 0.250000 | [0.250000, 0.250000] | 0.000000 |
| market | 2025 | 1236/267 | margin_mae | 11.212571 | 11.173748 | [10.229312, 12.194775] | -0.038823 |
| elo | 2025 | 1236/267 | accuracy_points | 52.346278 | 52.434457 | [47.328244, 57.518797] | 0.088179 |
| elo | 2025 | 1236/267 | log_loss | 0.692967 | 0.693209 | [0.690998, 0.695426] | 0.000242 |
| elo | 2025 | 1236/267 | brier | 0.249910 | 0.250031 | [0.248926, 0.251139] | 0.000121 |
| elo | 2025 | 1236/267 | margin_mae | 10.290924 | 10.228192 | [9.350245, 11.099115] | -0.062731 |

## Fold coefficients

**Measured:** the news coefficient is positive in 4/6 folds; range [-0.030466, 0.158661] per margin point. Natural-scale coefficients follow; approximate Hessian intervals describe training fits, not six independent studies.

| Arm | Held out | Term | Coefficient | Approximate 95% interval |
| --- | --- | --- | --- | --- |
| fifth_term | 2020 | intercept | -0.045134 | [-0.238488, 0.148220] |
| fifth_term | 2020 | model_logit | 0.341353 | [-0.065021, 0.747726] |
| fifth_term | 2020 | composition_flag_sum | 0.244732 | [0.114861, 0.374604] |
| fifth_term | 2020 | market_move_toward_home | 0.220698 | [0.085929, 0.355466] |
| fifth_term | 2020 | market_move_available | 0.027970 | [-0.211986, 0.267926] |
| fifth_term | 2020 | common_opponent_news | -0.006867 | [-0.379185, 0.365451] |
| four_term | 2020 | intercept | -0.045251 | [-0.238502, 0.147999] |
| four_term | 2020 | model_logit | 0.341246 | [-0.065081, 0.747572] |
| four_term | 2020 | composition_flag_sum | 0.244789 | [0.114954, 0.374623] |
| four_term | 2020 | market_move_toward_home | 0.220730 | [0.085974, 0.355485] |
| four_term | 2020 | market_move_available | 0.028113 | [-0.211719, 0.267944] |
| elo | 2020 | intercept | -0.015405 | [-0.133503, 0.102692] |
| elo | 2020 | elo_gap | -0.000650 | [-0.002681, 0.001381] |
| elo | 2020 | tue_open_home_spread | 0.004884 | [-0.025184, 0.034952] |
| fifth_term | 2021 | intercept | -0.062914 | [-0.257248, 0.131420] |
| fifth_term | 2021 | model_logit | 0.177688 | [-0.224129, 0.579505] |
| fifth_term | 2021 | composition_flag_sum | 0.272021 | [0.138540, 0.405502] |
| fifth_term | 2021 | market_move_toward_home | 0.223813 | [0.088739, 0.358887] |
| fifth_term | 2021 | market_move_available | 0.044844 | [-0.195719, 0.285407] |
| fifth_term | 2021 | common_opponent_news | 0.021220 | [-0.321570, 0.364009] |
| four_term | 2021 | intercept | -0.062332 | [-0.256434, 0.131771] |
| four_term | 2021 | model_logit | 0.177916 | [-0.223901, 0.579732] |
| four_term | 2021 | composition_flag_sum | 0.271867 | [0.138409, 0.405326] |
| four_term | 2021 | market_move_toward_home | 0.223716 | [0.088645, 0.358786] |
| four_term | 2021 | market_move_available | 0.044183 | [-0.196139, 0.284504] |
| elo | 2021 | intercept | 0.005342 | [-0.113192, 0.123877] |
| elo | 2021 | elo_gap | -0.000379 | [-0.002448, 0.001690] |
| elo | 2021 | tue_open_home_spread | -0.003065 | [-0.033665, 0.027534] |
| fifth_term | 2022 | intercept | -0.057954 | [-0.266786, 0.150879] |
| fifth_term | 2022 | model_logit | 0.242161 | [-0.166709, 0.651031] |
| fifth_term | 2022 | composition_flag_sum | 0.238957 | [0.103882, 0.374032] |
| fifth_term | 2022 | market_move_toward_home | 0.223891 | [0.089105, 0.358677] |
| fifth_term | 2022 | market_move_available | 0.042560 | [-0.210225, 0.295345] |
| fifth_term | 2022 | common_opponent_news | 0.158661 | [-0.220230, 0.537553] |
| four_term | 2022 | intercept | -0.055698 | [-0.264375, 0.152978] |
| four_term | 2022 | model_logit | 0.249647 | [-0.158709, 0.658002] |
| four_term | 2022 | composition_flag_sum | 0.238476 | [0.103455, 0.373497] |
| four_term | 2022 | market_move_toward_home | 0.222903 | [0.088132, 0.357675] |
| four_term | 2022 | market_move_available | 0.039674 | [-0.212921, 0.292269] |
| elo | 2022 | intercept | -0.016334 | [-0.135804, 0.103136] |
| elo | 2022 | elo_gap | -0.000140 | [-0.002306, 0.002027] |
| elo | 2022 | tue_open_home_spread | 0.005959 | [-0.025138, 0.037057] |
| fifth_term | 2023 | intercept | -0.037602 | [-0.205362, 0.130158] |
| fifth_term | 2023 | model_logit | 0.341453 | [-0.060075, 0.742981] |
| fifth_term | 2023 | composition_flag_sum | 0.245698 | [0.111456, 0.379940] |
| fifth_term | 2023 | market_move_toward_home | 0.230223 | [0.056802, 0.403645] |
| fifth_term | 2023 | market_move_available | 0.001964 | [-0.243127, 0.247055] |
| fifth_term | 2023 | common_opponent_news | -0.030466 | [-0.387577, 0.326645] |
| four_term | 2023 | intercept | -0.038064 | [-0.205739, 0.129611] |
| four_term | 2023 | model_logit | 0.341480 | [-0.060025, 0.742984] |
| four_term | 2023 | composition_flag_sum | 0.246114 | [0.111966, 0.380261] |
| four_term | 2023 | market_move_toward_home | 0.230272 | [0.056857, 0.403687] |
| four_term | 2023 | market_move_available | 0.002286 | [-0.242776, 0.247347] |
| elo | 2023 | intercept | -0.018886 | [-0.138034, 0.100262] |
| elo | 2023 | elo_gap | -0.000388 | [-0.002465, 0.001690] |
| elo | 2023 | tue_open_home_spread | -0.004510 | [-0.034644, 0.025625] |
| fifth_term | 2024 | intercept | -0.056392 | [-0.223937, 0.111153] |
| fifth_term | 2024 | model_logit | 0.256237 | [-0.142272, 0.654746] |
| fifth_term | 2024 | composition_flag_sum | 0.262397 | [0.127678, 0.397115] |
| fifth_term | 2024 | market_move_toward_home | 0.246054 | [0.085036, 0.407073] |
| fifth_term | 2024 | market_move_available | 0.039375 | [-0.202005, 0.280755] |
| fifth_term | 2024 | common_opponent_news | 0.015959 | [-0.381307, 0.413224] |
| four_term | 2024 | intercept | -0.056178 | [-0.223635, 0.111279] |
| four_term | 2024 | model_logit | 0.256116 | [-0.142387, 0.654618] |
| four_term | 2024 | composition_flag_sum | 0.262262 | [0.127587, 0.396937] |
| four_term | 2024 | market_move_toward_home | 0.246066 | [0.085029, 0.407103] |
| four_term | 2024 | market_move_available | 0.039146 | [-0.202162, 0.280453] |
| elo | 2024 | intercept | -0.015591 | [-0.135705, 0.104524] |
| elo | 2024 | elo_gap | -0.000580 | [-0.002757, 0.001596] |
| elo | 2024 | tue_open_home_spread | -0.004499 | [-0.035317, 0.026320] |
| fifth_term | 2025 | intercept | -0.055895 | [-0.224049, 0.112259] |
| fifth_term | 2025 | model_logit | 0.292047 | [-0.110826, 0.694920] |
| fifth_term | 2025 | composition_flag_sum | 0.296412 | [0.162356, 0.430468] |
| fifth_term | 2025 | market_move_toward_home | 0.190314 | [0.026860, 0.353769] |
| fifth_term | 2025 | market_move_available | 0.051027 | [-0.188195, 0.290249] |
| fifth_term | 2025 | common_opponent_news | 0.120841 | [-0.244841, 0.486523] |
| four_term | 2025 | intercept | -0.054099 | [-0.222122, 0.113925] |
| four_term | 2025 | model_logit | 0.292328 | [-0.110527, 0.695183] |
| four_term | 2025 | composition_flag_sum | 0.295997 | [0.161950, 0.430043] |
| four_term | 2025 | market_move_toward_home | 0.188739 | [0.025391, 0.352087] |
| four_term | 2025 | market_move_available | 0.047462 | [-0.191454, 0.286379] |
| elo | 2025 | intercept | -0.026538 | [-0.146535, 0.093459] |
| elo | 2025 | elo_gap | -0.000168 | [-0.002270, 0.001934] |
| elo | 2025 | tue_open_home_spread | -0.001986 | [-0.031983, 0.028011] |

## Reliability

**Measured:** five bands use each fold's training probability quintiles and pool held-out predictions. Tied market probabilities leave empty bands; no held-out cutoff selection.

| Arm | Training-quantile band | Held-out n | Mean probability | Observed home-cover rate |
| --- | --- | --- | --- | --- |
| fifth_term | 1 | 303 | 0.396820 | 0.438944 |
| fifth_term | 2 | 311 | 0.461271 | 0.437299 |
| fifth_term | 3 | 278 | 0.490240 | 0.442446 |
| fifth_term | 4 | 326 | 0.532242 | 0.561350 |
| fifth_term | 5 | 285 | 0.605385 | 0.600000 |
| four_term | 1 | 305 | 0.396731 | 0.432787 |
| four_term | 2 | 311 | 0.461424 | 0.440514 |
| four_term | 3 | 280 | 0.490181 | 0.435714 |
| four_term | 4 | 324 | 0.532378 | 0.564815 |
| four_term | 5 | 283 | 0.605133 | 0.607774 |
| model | 1 | 315 | 0.379615 | 0.469841 |
| model | 2 | 282 | 0.441338 | 0.453901 |
| model | 3 | 303 | 0.481104 | 0.488449 |
| model | 4 | 286 | 0.517867 | 0.541958 |
| model | 5 | 317 | 0.578295 | 0.526814 |
| market | 1 | 0 | unavailable | unavailable |
| market | 2 | 0 | unavailable | unavailable |
| market | 3 | 0 | unavailable | unavailable |
| market | 4 | 0 | unavailable | unavailable |
| market | 5 | 1503 | 0.500000 | 0.496341 |
| elo | 1 | 311 | 0.480889 | 0.536977 |
| elo | 2 | 280 | 0.490176 | 0.514286 |
| elo | 3 | 319 | 0.496162 | 0.467085 |
| elo | 4 | 312 | 0.502714 | 0.487179 |
| elo | 5 | 281 | 0.511217 | 0.476868 |

## Interpretation and saved evidence

**Inferred:** provisional `unresolved_below_power`, pending the orchestrator's registry write. AGENTS.md lines 67-78 allow closure only with a refuted mechanism or demonstrated positive-control power; no such closure is established here. AGENTS.md lines 89-100 require one fitted probability and out-of-season parameters. The estimates inform research; they do not authorize serving this term. Correlated reporting cells and the look budget preclude presenting a best cell as independent evidence.

**Measured:** prediction rows, timing audit, coefficient rows and structured summary are under `tests/scratch/codex/lead78_unit2/`. Exact bash record commands are in the lane and were not executed by this worker.

| Input | SHA-256 |
| --- | --- |
| artifacts\pick_probability\20260929T192747Z\per_game.parquet | 0490c806caf9e9707abe28f3e3e42f85df312084ae52d9d1588d173bba5b01e5 |
| data\raw\20260908T162105Z\schedules.parquet | a075d909f48e0a4301dbe7a90085c3fd4ad0c3ef5c487d0e3e29baf9a640650a |

## Original unit-one declaration (archived unchanged)

**Read:** copied from the pre-outcome lane; the unit-two amendment supersedes its capture, publication-time and forward-only restrictions.

### Predeclared protocol (verbatim words; wrapped for the bounded reader)
| LEAD-78 | ⬜ | Within-week common-opponent news (batch B rank 5 of 8) | **Inferred proposal, 2026-09-29; unmeasured.** Mechanism: a completed Thursday or Saturday game changes what
earlier performances against those teams mean; Tuesday's pool line cannot contain that update for Sunday's teams. Predeclare 2020-2025 REG games with a later pick deadline, preserving
every eligible game rather than selecting surprising early results. Build one score-based common-opponent state update from games publicly final between Tuesday and the target deadline,
holding the earlier-season-trained rating architecture fixed. Feature: updated-minus-Tuesday implied home margin propagated through the already-observed opponent graph; enter it
alongside the existing market move so news already absorbed by books is controlled. No target-game plays, unfinished games or later stat corrections; retain published-final timestamps.
Protocol B supplies chronology-purged LOSO, paired baselines, IS/OOS gap, calibration and season intervals. B=3 (state-update specification, delta term, joint fit), F=6, E=0; 298
counted looks. **Read/checked:** PBP-05, RWB-01, MOD-06/10/21 and LEAD-68. Those cover opponent adjustment, season state, dynamic/graph ratings and seasonal calibration; the new
estimand is the incremental information arriving within the target week, not a replacement rating system or a rerun of closing-line ratings. Units: as-of completion/state-delta audit,
15-20 calls; cached score-only replay, 20-25. Rank rationale: no new external feed and many historical weeks; expected effect is smaller because only a few early games inform each
slate. |

Rows fix population, terms, target, units and specification budget B. Inventory metadata first; require issuance/ingestion before the pool deadline. Grade authentic frozen openers.
Missing archives are data gaps, never replaced by closes, realized weather or retrospective news.

Chronology-purged LOSO excludes target/later seasons from fitting; separate earlier training, selection and calibration seasons. Folds lacking training remain unavailable. Reuse
certified pregame predictions; choose coefficients/penalties on earlier seasons without new grids. Freeze before prospective evaluation. One calibrated discrete-margin distribution
combines information; its cover probability selects the side.

Pair combined-model, model-only, market-only and Elo baselines. Report IS/OOS opener accuracy, Brier, log loss and margin MAE, their gap, decisive-game record first, fold coefficients,
season stability, five training-quantile reliability bands, season-block 95% intervals and `probability_positive`. Zero crossing closes nothing; unresolved effects remain
`unresolved_below_power`.

Count reporting cells too: L = (B + 36 + 8E)(F + 1) + 25 for F scheduled folds, E extra endpoints: B candidate/four baseline specifications, four-comparator IS/OOS metric cells per
fold/pooled, five bands for five models. Identical nested refits add no specification. New splits/specifications need separate preregistration. Correlated cells are not independent
evidence.

LEAD-78 budget: B=3, F=6, E=0; L=298.
