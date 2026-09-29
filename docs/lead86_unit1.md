# LEAD-86 unit 1: observed-market training regularization

**Measured:** one local fixed-grid replay; no served changes. Protocol was saved before outcomes.

## Decisive games

| Fold | Against | Wins-losses | Disagreements | 95% Wilson |
| --- | --- | --- | --- | --- |
| 2023 | four_term | 5-4 | 9 | [26.67%, 81.12%] |
| 2023 | model_only | 63-45 | 108 | [48.91%, 67.19%] |
| 2023 | market | 43-27 | 70 | [49.72%, 71.95%] |
| 2023 | elo | 72-48 | 120 | [51.06%, 68.32%] |
| 2024 | four_term | 0-0 | 0 | unavailable |
| 2024 | model_only | 40-34 | 74 | [42.78%, 64.93%] |
| 2024 | market | 47-36 | 83 | [45.91%, 66.76%] |
| 2024 | elo | 54-32 | 86 | [52.23%, 72.25%] |
| 2025 | four_term | 0-0 | 0 | unavailable |
| 2025 | model_only | 42-39 | 81 | [41.14%, 62.40%] |
| 2025 | market | 42-38 | 80 | [41.70%, 63.08%] |
| 2025 | elo | 69-54 | 123 | [47.27%, 64.55%] |
| pooled | four_term | 5-4 | 9 | [26.67%, 81.12%] |
| pooled | model_only | 145-118 | 263 | [49.09%, 61.03%] |
| pooled | market | 132-101 | 233 | [50.23%, 62.86%] |
| pooled | elo | 195-134 | 329 | [53.88%, 64.44%] |

## Population and clocks

**Measured:** 262 capture files, 316530 quote rows; 1309 common-book complete non-push opener games.
Tuesday anchors predate noon; Sunday captures are the archived deadline anchors. No closing predictors.
Push rows remain in the lattice training prior; the frozen four-term artifact excludes pushes, so scored rows are non-push.
The lattice nuisance fit uses only the declared fit seasons, including training outcomes; no selection, calibration or outer outcomes enter it.
Five arms use the same discrete margin lattice at the frozen opener; only conditional cover mass changes.
Model-only and market retain their probabilities; candidate, four-term and Elo use separate-season calibration.
**Inferred limitation:** archived model features and model selection are retrospective; this is not an untouched prospective test.

## Metrics and gaps

**Measured:** 95% season-stratified week-block percentile intervals, 10,000 replicates; gap = OOS minus optimistic fit-period IS.
Predictions are held fixed. Accuracy is a fraction; Brier/log loss/RPS are losses. Pooled IS repeats training games across outer folds.

| Fold | Set | Arm | Metric | Value | 95% interval |
| --- | --- | --- | --- | --- | --- |
| 2023 | IS | regularized | accuracy | 0.550265 | [0.468077, 0.633883] |
| 2023 | IS | regularized | brier | 0.245394 | [0.235451, 0.255287] |
| 2023 | IS | regularized | log_loss | 0.683228 | [0.663137, 0.703818] |
| 2023 | IS | regularized | rps | 7.024241 | [6.404240, 7.747356] |
| 2023 | IS | four_term | accuracy | 0.539683 | [0.468750, 0.614213] |
| 2023 | IS | four_term | brier | 0.244183 | [0.235221, 0.252658] |
| 2023 | IS | four_term | log_loss | 0.680972 | [0.662683, 0.698105] |
| 2023 | IS | four_term | rps | 7.021842 | [6.390759, 7.745275] |
| 2023 | IS | model_only | accuracy | 0.529101 | [0.454545, 0.592040] |
| 2023 | IS | model_only | brier | 0.258187 | [0.244980, 0.271930] |
| 2023 | IS | model_only | log_loss | 0.710793 | [0.683504, 0.738921] |
| 2023 | IS | model_only | rps | 7.065162 | [6.399883, 7.850535] |
| 2023 | IS | market | accuracy | 0.539683 | [0.484833, 0.585863] |
| 2023 | IS | market | brier | 0.246194 | [0.238613, 0.254280] |
| 2023 | IS | market | log_loss | 0.685093 | [0.669612, 0.701584] |
| 2023 | IS | market | rps | 7.018357 | [6.362497, 7.778433] |
| 2023 | IS | elo | accuracy | 0.539683 | [0.483871, 0.589474] |
| 2023 | IS | elo | brier | 0.249499 | [0.243734, 0.255354] |
| 2023 | IS | elo | log_loss | 0.692425 | [0.680863, 0.704438] |
| 2023 | IS | elo | rps | 7.054340 | [6.358599, 7.860277] |
| 2023 | OOS | regularized | accuracy | 0.583691 | [0.528378, 0.637132] |
| 2023 | OOS | regularized | brier | 0.240893 | [0.229696, 0.251916] |
| 2023 | OOS | regularized | log_loss | 0.675313 | [0.652008, 0.697956] |
| 2023 | OOS | regularized | rps | 7.425774 | [6.626655, 8.246383] |
| 2023 | OOS | four_term | accuracy | 0.579399 | [0.513043, 0.645570] |
| 2023 | OOS | four_term | brier | 0.242223 | [0.231794, 0.253010] |
| 2023 | OOS | four_term | log_loss | 0.678002 | [0.656209, 0.700971] |
| 2023 | OOS | four_term | rps | 7.450089 | [6.645174, 8.293584] |
| 2023 | OOS | model_only | accuracy | 0.506438 | [0.461856, 0.553097] |
| 2023 | OOS | model_only | brier | 0.255391 | [0.249821, 0.260711] |
| 2023 | OOS | model_only | log_loss | 0.704461 | [0.692980, 0.715872] |
| 2023 | OOS | model_only | rps | 7.602820 | [6.825639, 8.414423] |
| 2023 | OOS | market | accuracy | 0.515021 | [0.453696, 0.581897] |
| 2023 | OOS | market | brier | 0.246463 | [0.240908, 0.251448] |
| 2023 | OOS | market | log_loss | 0.685934 | [0.674619, 0.696178] |
| 2023 | OOS | market | rps | 7.491045 | [6.705205, 8.321678] |
| 2023 | OOS | elo | accuracy | 0.480687 | [0.406926, 0.550847] |
| 2023 | OOS | elo | brier | 0.253895 | [0.246245, 0.261677] |
| 2023 | OOS | elo | log_loss | 0.700978 | [0.685444, 0.717008] |
| 2023 | OOS | elo | rps | 7.592319 | [6.809642, 8.407426] |
| 2024 | IS | regularized | accuracy | 0.542289 | [0.493671, 0.589109] |
| 2024 | IS | regularized | brier | 0.244599 | [0.237612, 0.251272] |
| 2024 | IS | regularized | log_loss | 0.681868 | [0.667504, 0.695800] |
| 2024 | IS | regularized | rps | 7.622267 | [7.106250, 8.222032] |
| 2024 | IS | four_term | accuracy | 0.542289 | [0.493573, 0.590244] |
| 2024 | IS | four_term | brier | 0.244599 | [0.237754, 0.251267] |
| 2024 | IS | four_term | log_loss | 0.681868 | [0.667828, 0.695772] |
| 2024 | IS | four_term | rps | 7.622267 | [7.102324, 8.210457] |
| 2024 | IS | model_only | accuracy | 0.547264 | [0.492424, 0.600000] |
| 2024 | IS | model_only | brier | 0.252503 | [0.242335, 0.263238] |
| 2024 | IS | model_only | log_loss | 0.698863 | [0.677648, 0.720203] |
| 2024 | IS | model_only | rps | 7.687795 | [7.135422, 8.334672] |
| 2024 | IS | market | accuracy | 0.559701 | [0.525300, 0.593676] |
| 2024 | IS | market | brier | 0.245412 | [0.241344, 0.249695] |
| 2024 | IS | market | log_loss | 0.683597 | [0.675165, 0.692143] |
| 2024 | IS | market | rps | 7.625587 | [7.109113, 8.229286] |
| 2024 | IS | elo | accuracy | 0.485075 | [0.434889, 0.533333] |
| 2024 | IS | elo | brier | 0.250890 | [0.248868, 0.252856] |
| 2024 | IS | elo | log_loss | 0.694929 | [0.690951, 0.698954] |
| 2024 | IS | elo | rps | 7.711066 | [7.178712, 8.329880] |
| 2024 | OOS | regularized | accuracy | 0.586207 | [0.521739, 0.654012] |
| 2024 | OOS | regularized | brier | 0.244254 | [0.236716, 0.252317] |
| 2024 | OOS | regularized | log_loss | 0.681833 | [0.666461, 0.698596] |
| 2024 | OOS | regularized | rps | 7.226975 | [6.663011, 7.803110] |
| 2024 | OOS | four_term | accuracy | 0.586207 | [0.523207, 0.655319] |
| 2024 | OOS | four_term | brier | 0.244254 | [0.236612, 0.252335] |
| 2024 | OOS | four_term | log_loss | 0.681833 | [0.666262, 0.698462] |
| 2024 | OOS | four_term | rps | 7.226975 | [6.656149, 7.811654] |
| 2024 | OOS | model_only | accuracy | 0.560345 | [0.482906, 0.633621] |
| 2024 | OOS | model_only | brier | 0.249274 | [0.241471, 0.257183] |
| 2024 | OOS | model_only | log_loss | 0.691784 | [0.676415, 0.707929] |
| 2024 | OOS | model_only | rps | 7.288314 | [6.730233, 7.861058] |
| 2024 | OOS | market | accuracy | 0.538793 | [0.493333, 0.584071] |
| 2024 | OOS | market | brier | 0.247933 | [0.243901, 0.251415] |
| 2024 | OOS | market | log_loss | 0.688953 | [0.680901, 0.696050] |
| 2024 | OOS | market | rps | 7.251536 | [6.710058, 7.798672] |
| 2024 | OOS | elo | accuracy | 0.491379 | [0.424893, 0.557610] |
| 2024 | OOS | elo | brier | 0.250704 | [0.248059, 0.253347] |
| 2024 | OOS | elo | log_loss | 0.694556 | [0.689349, 0.699801] |
| 2024 | OOS | elo | rps | 7.274553 | [6.721328, 7.848777] |
| 2025 | IS | regularized | accuracy | 0.555556 | [0.517571, 0.593443] |
| 2025 | IS | regularized | brier | 0.242651 | [0.237026, 0.248622] |
| 2025 | IS | regularized | log_loss | 0.678013 | [0.666414, 0.689867] |
| 2025 | IS | regularized | rps | 7.284708 | [6.876567, 7.738536] |
| 2025 | IS | four_term | accuracy | 0.557190 | [0.519606, 0.594829] |
| 2025 | IS | four_term | brier | 0.242623 | [0.236932, 0.248216] |
| 2025 | IS | four_term | log_loss | 0.677979 | [0.666502, 0.689954] |
| 2025 | IS | four_term | rps | 7.285778 | [6.879520, 7.732013] |
| 2025 | IS | model_only | accuracy | 0.540850 | [0.498344, 0.580894] |
| 2025 | IS | model_only | brier | 0.252139 | [0.244993, 0.259499] |
| 2025 | IS | model_only | log_loss | 0.698000 | [0.683072, 0.712831] |
| 2025 | IS | model_only | rps | 7.407125 | [6.977214, 7.882314] |
| 2025 | IS | market | accuracy | 0.562092 | [0.530000, 0.594341] |
| 2025 | IS | market | brier | 0.244971 | [0.241616, 0.248347] |
| 2025 | IS | market | log_loss | 0.682768 | [0.675929, 0.689815] |
| 2025 | IS | market | rps | 7.317436 | [6.907699, 7.788576] |
| 2025 | IS | elo | accuracy | 0.508170 | [0.472667, 0.544408] |
| 2025 | IS | elo | brier | 0.249933 | [0.249316, 0.250560] |
| 2025 | IS | elo | log_loss | 0.693014 | [0.691754, 0.694265] |
| 2025 | IS | elo | rps | 7.388912 | [6.973128, 7.846333] |
| 2025 | OOS | regularized | accuracy | 0.547414 | [0.495536, 0.595833] |
| 2025 | OOS | regularized | brier | 0.245025 | [0.235405, 0.253907] |
| 2025 | OOS | regularized | log_loss | 0.682753 | [0.662477, 0.701536] |
| 2025 | OOS | regularized | rps | 7.058383 | [6.416197, 7.710250] |
| 2025 | OOS | four_term | accuracy | 0.547414 | [0.495495, 0.597510] |
| 2025 | OOS | four_term | brier | 0.245249 | [0.235725, 0.254142] |
| 2025 | OOS | four_term | log_loss | 0.683276 | [0.663567, 0.701614] |
| 2025 | OOS | four_term | rps | 7.064504 | [6.409111, 7.748666] |
| 2025 | OOS | model_only | accuracy | 0.534483 | [0.473451, 0.591304] |
| 2025 | OOS | model_only | brier | 0.251394 | [0.243981, 0.259340] |
| 2025 | OOS | model_only | log_loss | 0.696063 | [0.680855, 0.712065] |
| 2025 | OOS | model_only | rps | 7.177302 | [6.506609, 7.880560] |
| 2025 | OOS | market | accuracy | 0.530172 | [0.480176, 0.581596] |
| 2025 | OOS | market | brier | 0.244917 | [0.239125, 0.250330] |
| 2025 | OOS | market | log_loss | 0.682795 | [0.671187, 0.693890] |
| 2025 | OOS | market | rps | 7.051859 | [6.422615, 7.695529] |
| 2025 | OOS | elo | accuracy | 0.482759 | [0.426724, 0.543481] |
| 2025 | OOS | elo | brier | 0.250372 | [0.249311, 0.251327] |
| 2025 | OOS | elo | log_loss | 0.693890 | [0.691796, 0.695842] |
| 2025 | OOS | elo | rps | 7.161819 | [6.525049, 7.786084] |
| pooled | IS | regularized | accuracy | 0.550291 | [0.512906, 0.588854] |
| pooled | IS | regularized | brier | 0.243733 | [0.238022, 0.249466] |
| pooled | IS | regularized | log_loss | 0.680121 | [0.668022, 0.692492] |
| pooled | IS | regularized | rps | 7.356587 | [6.927069, 7.829525] |
| pooled | IS | four_term | accuracy | 0.549460 | [0.511666, 0.586735] |
| pooled | IS | four_term | brier | 0.243528 | [0.237827, 0.249207] |
| pooled | IS | four_term | log_loss | 0.679749 | [0.668325, 0.691275] |
| pooled | IS | four_term | rps | 7.356755 | [6.919959, 7.830094] |
| pooled | IS | model_only | accuracy | 0.541147 | [0.494068, 0.584460] |
| pooled | IS | model_only | brier | 0.253211 | [0.244900, 0.261710] |
| pooled | IS | model_only | log_loss | 0.700298 | [0.683279, 0.718123] |
| pooled | IS | model_only | rps | 7.447190 | [6.987522, 7.979758] |
| pooled | IS | market | accuracy | 0.557772 | [0.525738, 0.587200] |
| pooled | IS | market | brier | 0.245310 | [0.241472, 0.249275] |
| pooled | IS | market | log_loss | 0.683410 | [0.675378, 0.691621] |
| pooled | IS | market | rps | 7.373422 | [6.938162, 7.870035] |
| pooled | IS | elo | accuracy | 0.505403 | [0.493796, 0.516104] |
| pooled | IS | elo | brier | 0.250185 | [0.249143, 0.251291] |
| pooled | IS | elo | log_loss | 0.693561 | [0.691369, 0.695734] |
| pooled | IS | elo | rps | 7.444000 | [6.979824, 7.956665] |
| pooled | OOS | regularized | accuracy | 0.572453 | [0.539589, 0.604816] |
| pooled | OOS | regularized | brier | 0.243387 | [0.237804, 0.248826] |
| pooled | OOS | regularized | log_loss | 0.679960 | [0.668129, 0.691574] |
| pooled | OOS | regularized | rps | 7.237315 | [6.842388, 7.644932] |
| pooled | OOS | four_term | accuracy | 0.571019 | [0.535613, 0.606790] |
| pooled | OOS | four_term | brier | 0.243906 | [0.238674, 0.249191] |
| pooled | OOS | four_term | log_loss | 0.681033 | [0.669803, 0.692483] |
| pooled | OOS | four_term | rps | 7.247480 | [6.866920, 7.644833] |
| pooled | OOS | model_only | accuracy | 0.533716 | [0.498559, 0.569018] |
| pooled | OOS | model_only | brier | 0.252024 | [0.247930, 0.256219] |
| pooled | OOS | model_only | log_loss | 0.697446 | [0.689298, 0.705963] |
| pooled | OOS | model_only | rps | 7.356499 | [6.961720, 7.763213] |
| pooled | OOS | market | accuracy | 0.527977 | [0.497842, 0.558613] |
| pooled | OOS | market | brier | 0.246438 | [0.243557, 0.249233] |
| pooled | OOS | market | log_loss | 0.685894 | [0.679976, 0.691615] |
| pooled | OOS | market | rps | 7.265138 | [6.881347, 7.665232] |
| pooled | OOS | elo | accuracy | 0.484935 | [0.446064, 0.523273] |
| pooled | OOS | elo | brier | 0.251660 | [0.248898, 0.254423] |
| pooled | OOS | elo | log_loss | 0.696481 | [0.690951, 0.702108] |
| pooled | OOS | elo | rps | 7.343255 | [6.952029, 7.740613] |
| 2023 | gap | regularized | accuracy | 0.033426 | [-0.067524, 0.132001] |
| 2023 | gap | regularized | brier | -0.004501 | [-0.019479, 0.010473] |
| 2023 | gap | regularized | log_loss | -0.007915 | [-0.038928, 0.022808] |
| 2023 | gap | regularized | rps | 0.401533 | [-0.656510, 1.436065] |
| 2023 | gap | four_term | accuracy | 0.039717 | [-0.062412, 0.139210] |
| 2023 | gap | four_term | brier | -0.001960 | [-0.015787, 0.012145] |
| 2023 | gap | four_term | log_loss | -0.002970 | [-0.031254, 0.026301] |
| 2023 | gap | four_term | rps | 0.428246 | [-0.649828, 1.445115] |
| 2023 | gap | model_only | accuracy | -0.022663 | [-0.099494, 0.064246] |
| 2023 | gap | model_only | brier | -0.002797 | [-0.017537, 0.011594] |
| 2023 | gap | model_only | log_loss | -0.006333 | [-0.036590, 0.023265] |
| 2023 | gap | model_only | rps | 0.537657 | [-0.579077, 1.593215] |
| 2023 | gap | market | accuracy | -0.024661 | [-0.102978, 0.060535] |
| 2023 | gap | market | brier | 0.000269 | [-0.009355, 0.009195] |
| 2023 | gap | market | log_loss | 0.000841 | [-0.019167, 0.019549] |
| 2023 | gap | market | rps | 0.472687 | [-0.634417, 1.546209] |
| 2023 | gap | elo | accuracy | -0.058996 | [-0.148309, 0.030090] |
| 2023 | gap | elo | brier | 0.004396 | [-0.005386, 0.014083] |
| 2023 | gap | elo | log_loss | 0.008553 | [-0.011289, 0.028706] |
| 2023 | gap | elo | rps | 0.537979 | [-0.581992, 1.621691] |
| 2024 | gap | regularized | accuracy | 0.043918 | [-0.034578, 0.128122] |
| 2024 | gap | regularized | brier | -0.000345 | [-0.010429, 0.010047] |
| 2024 | gap | regularized | log_loss | -0.000035 | [-0.021146, 0.021886] |
| 2024 | gap | regularized | rps | -0.395292 | [-1.219184, 0.384438] |
| 2024 | gap | four_term | accuracy | 0.043918 | [-0.034732, 0.127611] |
| 2024 | gap | four_term | brier | -0.000345 | [-0.010640, 0.010188] |
| 2024 | gap | four_term | log_loss | -0.000035 | [-0.021650, 0.021878] |
| 2024 | gap | four_term | rps | -0.395292 | [-1.193180, 0.394258] |
| 2024 | gap | model_only | accuracy | 0.013081 | [-0.082490, 0.105133] |
| 2024 | gap | model_only | brier | -0.003230 | [-0.016249, 0.009881] |
| 2024 | gap | model_only | log_loss | -0.007079 | [-0.033524, 0.019090] |
| 2024 | gap | model_only | rps | -0.399481 | [-1.252383, 0.401314] |
| 2024 | gap | market | accuracy | -0.020908 | [-0.077178, 0.036686] |
| 2024 | gap | market | brier | 0.002521 | [-0.003185, 0.007900] |
| 2024 | gap | market | log_loss | 0.005355 | [-0.006282, 0.016679] |
| 2024 | gap | market | rps | -0.374051 | [-1.172198, 0.386008] |
| 2024 | gap | elo | accuracy | 0.006305 | [-0.075899, 0.088160] |
| 2024 | gap | elo | brier | -0.000186 | [-0.003530, 0.003158] |
| 2024 | gap | elo | log_loss | -0.000373 | [-0.006896, 0.006316] |
| 2024 | gap | elo | rps | -0.436513 | [-1.261331, 0.377270] |
| 2025 | gap | regularized | accuracy | -0.008142 | [-0.072554, 0.054730] |
| 2025 | gap | regularized | brier | 0.002374 | [-0.008843, 0.012926] |
| 2025 | gap | regularized | log_loss | 0.004740 | [-0.017944, 0.027254] |
| 2025 | gap | regularized | rps | -0.226325 | [-1.004502, 0.564554] |
| 2025 | gap | four_term | accuracy | -0.009776 | [-0.071818, 0.054716] |
| 2025 | gap | four_term | brier | 0.002626 | [-0.008323, 0.013249] |
| 2025 | gap | four_term | log_loss | 0.005298 | [-0.017694, 0.026729] |
| 2025 | gap | four_term | rps | -0.221274 | [-1.010357, 0.584512] |
| 2025 | gap | model_only | accuracy | -0.006367 | [-0.080442, 0.064432] |
| 2025 | gap | model_only | brier | -0.000745 | [-0.011452, 0.009935] |
| 2025 | gap | model_only | log_loss | -0.001937 | [-0.023407, 0.019984] |
| 2025 | gap | model_only | rps | -0.229822 | [-1.052601, 0.605800] |
| 2025 | gap | market | accuracy | -0.031919 | [-0.090503, 0.028642] |
| 2025 | gap | market | brier | -0.000054 | [-0.006724, 0.006295] |
| 2025 | gap | market | log_loss | 0.000028 | [-0.013474, 0.013253] |
| 2025 | gap | market | rps | -0.265577 | [-1.048662, 0.519543] |
| 2025 | gap | elo | accuracy | -0.025411 | [-0.091972, 0.043944] |
| 2025 | gap | elo | brier | 0.000438 | [-0.000753, 0.001570] |
| 2025 | gap | elo | log_loss | 0.000876 | [-0.001597, 0.003180] |
| 2025 | gap | elo | rps | -0.227092 | [-1.020552, 0.543207] |
| pooled | gap | regularized | accuracy | 0.022162 | [-0.029256, 0.072827] |
| pooled | gap | regularized | brier | -0.000346 | [-0.008289, 0.007592] |
| pooled | gap | regularized | log_loss | -0.000161 | [-0.016763, 0.016347] |
| pooled | gap | regularized | rps | -0.119272 | [-0.738021, 0.478330] |
| pooled | gap | four_term | accuracy | 0.021559 | [-0.030334, 0.073156] |
| pooled | gap | four_term | brier | 0.000378 | [-0.007264, 0.008093] |
| pooled | gap | four_term | log_loss | 0.001284 | [-0.014742, 0.017142] |
| pooled | gap | four_term | rps | -0.109274 | [-0.719177, 0.492651] |
| pooled | gap | model_only | accuracy | -0.007431 | [-0.063050, 0.050216] |
| pooled | gap | model_only | brier | -0.001187 | [-0.010681, 0.008103] |
| pooled | gap | model_only | log_loss | -0.002852 | [-0.022240, 0.016237] |
| pooled | gap | model_only | rps | -0.090691 | [-0.752561, 0.538209] |
| pooled | gap | market | accuracy | -0.029795 | [-0.073487, 0.015740] |
| pooled | gap | market | brier | 0.001127 | [-0.003854, 0.005878] |
| pooled | gap | market | log_loss | 0.002484 | [-0.007447, 0.012398] |
| pooled | gap | market | rps | -0.108284 | [-0.729728, 0.492514] |
| pooled | gap | elo | accuracy | -0.020468 | [-0.060678, 0.019939] |
| pooled | gap | elo | brier | 0.001475 | [-0.001494, 0.004447] |
| pooled | gap | elo | log_loss | 0.002920 | [-0.003002, 0.009101] |
| pooled | gap | elo | rps | -0.100745 | [-0.735934, 0.503890] |

## Paired gains

Positive gains favor regularization. probability_positive assigns half of bootstrap ties to each side.

| Fold | Set | Baseline | Metric | Gain | 95% interval | probability_positive |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | IS | four_term | accuracy | 0.010582 | [-0.027782, 0.048649] | 0.7090 |
| 2023 | IS | four_term | brier | -0.001210 | [-0.005816, 0.003328] | 0.2998 |
| 2023 | IS | four_term | log_loss | -0.002255 | [-0.012327, 0.007781] | 0.3307 |
| 2023 | IS | four_term | rps | -0.002399 | [-0.065044, 0.061849] | 0.4550 |
| 2023 | IS | model_only | accuracy | 0.021164 | [-0.074468, 0.116674] | 0.6719 |
| 2023 | IS | model_only | brier | 0.012794 | [-0.003924, 0.028450] | 0.9344 |
| 2023 | IS | model_only | log_loss | 0.027566 | [-0.006396, 0.060075] | 0.9424 |
| 2023 | IS | model_only | rps | 0.040921 | [-0.167241, 0.255276] | 0.6418 |
| 2023 | IS | market | accuracy | 0.010582 | [-0.082902, 0.111676] | 0.5702 |
| 2023 | IS | market | brier | 0.000800 | [-0.008554, 0.011199] | 0.5463 |
| 2023 | IS | market | log_loss | 0.001865 | [-0.017374, 0.023411] | 0.5600 |
| 2023 | IS | market | rps | -0.005884 | [-0.163562, 0.158904] | 0.4639 |
| 2023 | IS | elo | accuracy | 0.010582 | [-0.089005, 0.114943] | 0.5729 |
| 2023 | IS | elo | brier | 0.004106 | [-0.008219, 0.016048] | 0.7430 |
| 2023 | IS | elo | log_loss | 0.009198 | [-0.016037, 0.033184] | 0.7715 |
| 2023 | IS | elo | rps | 0.030099 | [-0.165633, 0.216512] | 0.6132 |
| 2023 | OOS | four_term | accuracy | 0.004292 | [-0.022026, 0.033473] | 0.6014 |
| 2023 | OOS | four_term | brier | 0.001330 | [-0.003306, 0.006056] | 0.7080 |
| 2023 | OOS | four_term | log_loss | 0.002689 | [-0.007577, 0.012954] | 0.6885 |
| 2023 | OOS | four_term | rps | 0.024315 | [-0.032624, 0.081008] | 0.8066 |
| 2023 | OOS | model_only | accuracy | 0.077253 | [-0.004465, 0.150862] | 0.9659 |
| 2023 | OOS | model_only | brier | 0.014498 | [0.002712, 0.026200] | 0.9920 |
| 2023 | OOS | model_only | log_loss | 0.029148 | [0.004530, 0.053974] | 0.9894 |
| 2023 | OOS | model_only | rps | 0.177046 | [0.048918, 0.310424] | 0.9971 |
| 2023 | OOS | market | accuracy | 0.068670 | [-0.012876, 0.141026] | 0.9544 |
| 2023 | OOS | market | brier | 0.005570 | [-0.004641, 0.016196] | 0.8593 |
| 2023 | OOS | market | log_loss | 0.010621 | [-0.010914, 0.032501] | 0.8299 |
| 2023 | OOS | market | rps | 0.065271 | [-0.051926, 0.184198] | 0.8550 |
| 2023 | OOS | elo | accuracy | 0.103004 | [0.017019, 0.192829] | 0.9901 |
| 2023 | OOS | elo | brier | 0.013002 | [0.003981, 0.022480] | 0.9986 |
| 2023 | OOS | elo | log_loss | 0.025666 | [0.006469, 0.045199] | 0.9967 |
| 2023 | OOS | elo | rps | 0.166546 | [0.045475, 0.289193] | 0.9972 |
| 2024 | IS | four_term | accuracy | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | IS | four_term | brier | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | IS | four_term | log_loss | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | IS | four_term | rps | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | IS | model_only | accuracy | -0.004975 | [-0.063572, 0.053529] | 0.4288 |
| 2024 | IS | model_only | brier | 0.007905 | [-0.002296, 0.018280] | 0.9384 |
| 2024 | IS | model_only | log_loss | 0.016995 | [-0.003874, 0.038427] | 0.9450 |
| 2024 | IS | model_only | rps | 0.065527 | [-0.086766, 0.220075] | 0.8044 |
| 2024 | IS | market | accuracy | -0.017413 | [-0.075000, 0.040407] | 0.2798 |
| 2024 | IS | market | brier | 0.000813 | [-0.005824, 0.007486] | 0.5942 |
| 2024 | IS | market | log_loss | 0.001730 | [-0.012066, 0.015261] | 0.5892 |
| 2024 | IS | market | rps | 0.003319 | [-0.089277, 0.103498] | 0.5228 |
| 2024 | IS | elo | accuracy | 0.057214 | [-0.017414, 0.129032] | 0.9342 |
| 2024 | IS | elo | brier | 0.006291 | [-0.000501, 0.013104] | 0.9657 |
| 2024 | IS | elo | log_loss | 0.013061 | [-0.000830, 0.027056] | 0.9686 |
| 2024 | IS | elo | rps | 0.088798 | [-0.016356, 0.197948] | 0.9492 |
| 2024 | OOS | four_term | accuracy | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | OOS | four_term | brier | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | OOS | four_term | log_loss | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | OOS | four_term | rps | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | OOS | model_only | accuracy | 0.025862 | [-0.051502, 0.103604] | 0.7494 |
| 2024 | OOS | model_only | brier | 0.005020 | [-0.003432, 0.013302] | 0.8770 |
| 2024 | OOS | model_only | log_loss | 0.009951 | [-0.007366, 0.027352] | 0.8636 |
| 2024 | OOS | model_only | rps | 0.061339 | [-0.042344, 0.163056] | 0.8776 |
| 2024 | OOS | market | accuracy | 0.047414 | [-0.029170, 0.128102] | 0.8870 |
| 2024 | OOS | market | brier | 0.003679 | [-0.003859, 0.011058] | 0.8295 |
| 2024 | OOS | market | log_loss | 0.007120 | [-0.007742, 0.022087] | 0.8232 |
| 2024 | OOS | market | rps | 0.024560 | [-0.072913, 0.116554] | 0.6931 |
| 2024 | OOS | elo | accuracy | 0.094828 | [0.017776, 0.175003] | 0.9922 |
| 2024 | OOS | elo | brier | 0.006450 | [-0.001258, 0.013827] | 0.9499 |
| 2024 | OOS | elo | log_loss | 0.012723 | [-0.003279, 0.027978] | 0.9430 |
| 2024 | OOS | elo | rps | 0.047578 | [-0.051193, 0.140181] | 0.8383 |
| 2025 | IS | four_term | accuracy | -0.001634 | [-0.006861, 0.003344] | 0.2843 |
| 2025 | IS | four_term | brier | -0.000028 | [-0.000287, 0.000226] | 0.4176 |
| 2025 | IS | four_term | log_loss | -0.000035 | [-0.000565, 0.000493] | 0.4319 |
| 2025 | IS | four_term | rps | 0.001070 | [-0.002425, 0.004688] | 0.7121 |
| 2025 | IS | model_only | accuracy | 0.014706 | [-0.023103, 0.052723] | 0.7798 |
| 2025 | IS | model_only | brier | 0.009489 | [0.002886, 0.016051] | 0.9972 |
| 2025 | IS | model_only | log_loss | 0.019987 | [0.006420, 0.033907] | 0.9986 |
| 2025 | IS | model_only | rps | 0.122417 | [0.028219, 0.220755] | 0.9950 |
| 2025 | IS | market | accuracy | -0.006536 | [-0.050408, 0.037037] | 0.3816 |
| 2025 | IS | market | brier | 0.002320 | [-0.003583, 0.008248] | 0.7748 |
| 2025 | IS | market | log_loss | 0.004754 | [-0.006967, 0.016838] | 0.7831 |
| 2025 | IS | market | rps | 0.032728 | [-0.044987, 0.109480] | 0.7984 |
| 2025 | IS | elo | accuracy | 0.047386 | [0.006568, 0.087949] | 0.9891 |
| 2025 | IS | elo | brier | 0.007283 | [0.001844, 0.012856] | 0.9951 |
| 2025 | IS | elo | log_loss | 0.015001 | [0.003656, 0.026697] | 0.9954 |
| 2025 | IS | elo | rps | 0.104204 | [0.022302, 0.189737] | 0.9941 |
| 2025 | OOS | four_term | accuracy | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2025 | OOS | four_term | brier | 0.000224 | [-0.000118, 0.000585] | 0.8989 |
| 2025 | OOS | four_term | log_loss | 0.000523 | [-0.000188, 0.001299] | 0.9176 |
| 2025 | OOS | four_term | rps | 0.006121 | [0.002059, 0.010816] | 0.9998 |
| 2025 | OOS | model_only | accuracy | 0.012931 | [-0.053812, 0.082988] | 0.6312 |
| 2025 | OOS | model_only | brier | 0.006369 | [-0.002707, 0.017578] | 0.8947 |
| 2025 | OOS | model_only | log_loss | 0.013310 | [-0.005507, 0.036838] | 0.8950 |
| 2025 | OOS | model_only | rps | 0.118919 | [0.001739, 0.250763] | 0.9767 |
| 2025 | OOS | market | accuracy | 0.017241 | [-0.056034, 0.090129] | 0.6702 |
| 2025 | OOS | market | brier | -0.000108 | [-0.007616, 0.008377] | 0.4827 |
| 2025 | OOS | market | log_loss | 0.000042 | [-0.016244, 0.017220] | 0.4882 |
| 2025 | OOS | market | rps | -0.006524 | [-0.109073, 0.090063] | 0.4423 |
| 2025 | OOS | elo | accuracy | 0.064655 | [-0.017699, 0.141667] | 0.9377 |
| 2025 | OOS | elo | brier | 0.005347 | [-0.003746, 0.015128] | 0.8607 |
| 2025 | OOS | elo | log_loss | 0.011137 | [-0.007797, 0.031083] | 0.8599 |
| 2025 | OOS | elo | rps | 0.103436 | [-0.017721, 0.234497] | 0.9512 |
| pooled | IS | four_term | accuracy | 0.000831 | [-0.005800, 0.007282] | 0.6044 |
| pooled | IS | four_term | brier | -0.000204 | [-0.000967, 0.000599] | 0.2994 |
| pooled | IS | four_term | log_loss | -0.000372 | [-0.002082, 0.001387] | 0.3360 |
| pooled | IS | four_term | rps | 0.000167 | [-0.010492, 0.011678] | 0.5000 |
| pooled | IS | model_only | accuracy | 0.009144 | [-0.033970, 0.052102] | 0.6716 |
| pooled | IS | model_only | brier | 0.009479 | [0.001213, 0.017562] | 0.9879 |
| pooled | IS | model_only | log_loss | 0.020178 | [0.002848, 0.036945] | 0.9893 |
| pooled | IS | model_only | rps | 0.090603 | [-0.025643, 0.204419] | 0.9404 |
| pooled | IS | market | accuracy | -0.007481 | [-0.052804, 0.038983] | 0.3712 |
| pooled | IS | market | brier | 0.001578 | [-0.004326, 0.007523] | 0.6992 |
| pooled | IS | market | log_loss | 0.003290 | [-0.008869, 0.015471] | 0.7074 |
| pooled | IS | market | rps | 0.016835 | [-0.068526, 0.100172] | 0.6484 |
| pooled | IS | elo | accuracy | 0.044888 | [0.004170, 0.085576] | 0.9841 |
| pooled | IS | elo | brier | 0.006452 | [0.000424, 0.012495] | 0.9820 |
| pooled | IS | elo | log_loss | 0.013441 | [0.001177, 0.025474] | 0.9835 |
| pooled | IS | elo | rps | 0.087413 | [-0.001049, 0.178941] | 0.9735 |
| pooled | OOS | four_term | accuracy | 0.001435 | [-0.007225, 0.011348] | 0.6080 |
| pooled | OOS | four_term | brier | 0.000519 | [-0.001065, 0.002170] | 0.7320 |
| pooled | OOS | four_term | log_loss | 0.001073 | [-0.002338, 0.004561] | 0.7288 |
| pooled | OOS | four_term | rps | 0.010165 | [-0.008868, 0.029147] | 0.8563 |
| pooled | OOS | model_only | accuracy | 0.038737 | [-0.005626, 0.082173] | 0.9562 |
| pooled | OOS | model_only | brier | 0.008638 | [0.002725, 0.014713] | 0.9984 |
| pooled | OOS | model_only | log_loss | 0.017486 | [0.005344, 0.030138] | 0.9969 |
| pooled | OOS | model_only | rps | 0.119184 | [0.051580, 0.190114] | 0.9998 |
| pooled | OOS | market | accuracy | 0.044476 | [0.000000, 0.088319] | 0.9750 |
| pooled | OOS | market | brier | 0.003051 | [-0.001833, 0.008116] | 0.8877 |
| pooled | OOS | market | log_loss | 0.005934 | [-0.004236, 0.016318] | 0.8724 |
| pooled | OOS | market | rps | 0.027823 | [-0.032099, 0.088244] | 0.8132 |
| pooled | OOS | elo | accuracy | 0.087518 | [0.039435, 0.133527] | 0.9997 |
| pooled | OOS | elo | brier | 0.008273 | [0.003162, 0.013399] | 0.9993 |
| pooled | OOS | elo | log_loss | 0.016522 | [0.006190, 0.027390] | 0.9990 |
| pooled | OOS | elo | rps | 0.105940 | [0.038872, 0.173690] | 0.9994 |
| 2023 | gap | four_term | accuracy | -0.006290 | [-0.052859, 0.041622] | 0.3898 |
| 2023 | gap | four_term | brier | 0.002541 | [-0.003943, 0.009121] | 0.7761 |
| 2023 | gap | four_term | log_loss | 0.004945 | [-0.009540, 0.019460] | 0.7478 |
| 2023 | gap | four_term | rps | 0.026714 | [-0.056796, 0.112261] | 0.7379 |
| 2023 | gap | model_only | accuracy | 0.056089 | [-0.070149, 0.176055] | 0.8061 |
| 2023 | gap | model_only | brier | 0.001704 | [-0.017991, 0.021860] | 0.5572 |
| 2023 | gap | model_only | log_loss | 0.001582 | [-0.038859, 0.043392] | 0.5267 |
| 2023 | gap | model_only | rps | 0.136125 | [-0.116764, 0.378264] | 0.8583 |
| 2023 | gap | market | accuracy | 0.058088 | [-0.069519, 0.180629] | 0.8212 |
| 2023 | gap | market | brier | 0.004770 | [-0.009616, 0.019258] | 0.7508 |
| 2023 | gap | market | log_loss | 0.008756 | [-0.021860, 0.038505] | 0.7178 |
| 2023 | gap | market | rps | 0.071155 | [-0.132689, 0.270680] | 0.7576 |
| 2023 | gap | elo | accuracy | 0.092422 | [-0.043766, 0.227976] | 0.9073 |
| 2023 | gap | elo | brier | 0.008897 | [-0.006175, 0.024018] | 0.8722 |
| 2023 | gap | elo | log_loss | 0.016468 | [-0.014281, 0.048743] | 0.8474 |
| 2023 | gap | elo | rps | 0.136447 | [-0.087982, 0.366761] | 0.8820 |
| 2024 | gap | four_term | accuracy | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | gap | four_term | brier | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | gap | four_term | log_loss | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | gap | four_term | rps | 0.000000 | [0.000000, 0.000000] | 0.5000 |
| 2024 | gap | model_only | accuracy | 0.030837 | [-0.063160, 0.130524] | 0.7389 |
| 2024 | gap | model_only | brier | -0.002884 | [-0.016421, 0.010375] | 0.3300 |
| 2024 | gap | model_only | log_loss | -0.007044 | [-0.034696, 0.020441] | 0.3121 |
| 2024 | gap | model_only | rps | -0.004189 | [-0.193358, 0.184426] | 0.4777 |
| 2024 | gap | market | accuracy | 0.064827 | [-0.028877, 0.162866] | 0.9049 |
| 2024 | gap | market | brier | 0.002866 | [-0.007274, 0.012926] | 0.7052 |
| 2024 | gap | market | log_loss | 0.005390 | [-0.015095, 0.025783] | 0.6986 |
| 2024 | gap | market | rps | 0.021241 | [-0.116629, 0.153795] | 0.6207 |
| 2024 | gap | elo | accuracy | 0.037614 | [-0.067152, 0.145967] | 0.7570 |
| 2024 | gap | elo | brier | 0.000159 | [-0.010022, 0.010032] | 0.5197 |
| 2024 | gap | elo | log_loss | -0.000338 | [-0.021215, 0.020443] | 0.4936 |
| 2024 | gap | elo | rps | -0.041221 | [-0.188270, 0.100823] | 0.2885 |
| 2025 | gap | four_term | accuracy | 0.001634 | [-0.003344, 0.006861] | 0.7157 |
| 2025 | gap | four_term | brier | 0.000253 | [-0.000174, 0.000698] | 0.8755 |
| 2025 | gap | four_term | log_loss | 0.000558 | [-0.000344, 0.001511] | 0.8836 |
| 2025 | gap | four_term | rps | 0.005051 | [-0.000514, 0.010944] | 0.9619 |
| 2025 | gap | model_only | accuracy | -0.001775 | [-0.077002, 0.077859] | 0.4727 |
| 2025 | gap | model_only | brier | -0.003119 | [-0.014742, 0.010090] | 0.2948 |
| 2025 | gap | model_only | log_loss | -0.006677 | [-0.030286, 0.020154] | 0.2897 |
| 2025 | gap | model_only | rps | -0.003497 | [-0.158565, 0.158905] | 0.4688 |
| 2025 | gap | market | accuracy | 0.023777 | [-0.060607, 0.109202] | 0.7004 |
| 2025 | gap | market | brier | -0.002428 | [-0.012158, 0.007789] | 0.3148 |
| 2025 | gap | market | log_loss | -0.004713 | [-0.025139, 0.015780] | 0.3234 |
| 2025 | gap | market | rps | -0.039252 | [-0.166849, 0.083322] | 0.2635 |
| 2025 | gap | elo | accuracy | 0.017270 | [-0.074928, 0.105401] | 0.6573 |
| 2025 | gap | elo | brier | -0.001936 | [-0.012746, 0.009526] | 0.3578 |
| 2025 | gap | elo | log_loss | -0.003864 | [-0.026589, 0.019088] | 0.3569 |
| 2025 | gap | elo | rps | -0.000768 | [-0.150461, 0.152040] | 0.4999 |
| pooled | gap | four_term | accuracy | 0.000603 | [-0.010042, 0.011987] | 0.5360 |
| pooled | gap | four_term | brier | 0.000724 | [-0.001056, 0.002539] | 0.7880 |
| pooled | gap | four_term | log_loss | 0.001445 | [-0.002418, 0.005390] | 0.7684 |
| pooled | gap | four_term | rps | 0.009998 | [-0.011939, 0.031850] | 0.8206 |
| pooled | gap | model_only | accuracy | 0.029594 | [-0.032672, 0.090711] | 0.8267 |
| pooled | gap | model_only | brier | -0.000841 | [-0.010768, 0.009182] | 0.4299 |
| pooled | gap | model_only | log_loss | -0.002691 | [-0.023323, 0.019035] | 0.3984 |
| pooled | gap | model_only | rps | 0.028582 | [-0.105840, 0.163638] | 0.6605 |
| pooled | gap | market | accuracy | 0.051958 | [-0.011127, 0.116219] | 0.9452 |
| pooled | gap | market | brier | 0.001473 | [-0.006227, 0.009251] | 0.6428 |
| pooled | gap | market | log_loss | 0.002645 | [-0.013402, 0.018510] | 0.6251 |
| pooled | gap | market | rps | 0.010988 | [-0.093297, 0.116405] | 0.5862 |
| pooled | gap | elo | accuracy | 0.042630 | [-0.019637, 0.105605] | 0.9094 |
| pooled | gap | elo | brier | 0.001821 | [-0.006128, 0.009488] | 0.6746 |
| pooled | gap | elo | log_loss | 0.003081 | [-0.012839, 0.019478] | 0.6422 |
| pooled | gap | elo | rps | 0.018527 | [-0.096516, 0.131108] | 0.6329 |

## Fold coefficients

**Measured:** natural input units; calibration is a nonnegative slope on the raw probability logit.

| Fold | Penalty | Selection Brier | Selected | Intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 0.0 | 0.247961 | False | -0.111586 | -0.135139 | 0.379288 | 0.104850 | 0.000000 |
| 2023 | 0.1 | 0.247430 | False | -0.098455 | -0.125791 | 0.341507 | 0.103683 | 0.000000 |
| 2023 | 1.0 | 0.246633 | True | -0.040526 | -0.085965 | 0.175209 | 0.098806 | 0.000000 |
| 2024 | 0.0 | 0.239426 | True | -0.021183 | 0.384875 | 0.288541 | 0.093723 | 0.000000 |
| 2024 | 0.1 | 0.239718 | False | -0.017644 | 0.350477 | 0.259703 | 0.093979 | 0.000000 |
| 2024 | 1.0 | 0.242184 | False | -0.001916 | 0.198386 | 0.132194 | 0.095535 | 0.000000 |
| 2025 | 0.0 | 0.242543 | False | -0.008638 | 0.385503 | 0.328726 | 0.116794 | 0.000000 |
| 2025 | 0.1 | 0.242423 | True | -0.006592 | 0.350159 | 0.296617 | 0.115574 | 0.000000 |
| 2025 | 1.0 | 0.243002 | False | 0.002711 | 0.195497 | 0.155580 | 0.110738 | 0.000000 |

| Fold | Arm | Calibration intercept | Calibration slope |
| --- | --- | --- | --- |
| 2023 | regularized | 0.037866 | 2.147595 |
| 2023 | four_term | 0.085457 | 1.177225 |
| 2023 | elo | 0.085591 | 1.761244 |
| 2024 | regularized | 0.085012 | 1.060121 |
| 2024 | four_term | 0.085012 | 1.060121 |
| 2024 | elo | 0.090220 | 0.203347 |
| 2025 | regularized | -0.066971 | 1.090745 |
| 2025 | four_term | -0.064511 | 0.996884 |
| 2025 | elo | -0.034486 | 0.000000 |

| Fold | Elo intercept | Elo difference | Opener | Prior games | Prior pushes |
| --- | --- | --- | --- | --- | --- |
| 2023 | -0.085907 | 0.001861 | -0.036938 | 196 | 7 |
| 2024 | -0.099848 | 0.001120 | -0.014667 | 412 | 10 |
| 2025 | -0.028457 | 0.000666 | -0.028162 | 628 | 16 |

## Reliability

| Arm | Band | Games | Mean probability | Observed home cover |
| --- | --- | --- | --- | --- |
| regularized | 0.0-0.2 | 2 | 0.163533 | 0.500000 |
| regularized | 0.2-0.4 | 74 | 0.349208 | 0.405405 |
| regularized | 0.4-0.6 | 511 | 0.503782 | 0.493151 |
| regularized | 0.6-0.8 | 109 | 0.650116 | 0.651376 |
| regularized | 0.8-1.0 | 1 | 0.804848 | 1.000000 |
| four_term | 0.0-0.2 | 1 | 0.182780 | 1.000000 |
| four_term | 0.2-0.4 | 80 | 0.352261 | 0.400000 |
| four_term | 0.4-0.6 | 497 | 0.503494 | 0.486922 |
| four_term | 0.6-0.8 | 119 | 0.644005 | 0.672269 |
| four_term | 0.8-1.0 | 0 | nan | nan |
| model_only | 0.0-0.2 | 0 | nan | nan |
| model_only | 0.2-0.4 | 45 | 0.373670 | 0.577778 |
| model_only | 0.4-0.6 | 600 | 0.500693 | 0.506667 |
| model_only | 0.6-0.8 | 52 | 0.621717 | 0.480769 |
| model_only | 0.8-1.0 | 0 | nan | nan |
| market | 0.0-0.2 | 0 | nan | nan |
| market | 0.2-0.4 | 10 | 0.365678 | 0.200000 |
| market | 0.4-0.6 | 674 | 0.505723 | 0.508902 |
| market | 0.6-0.8 | 13 | 0.630679 | 0.769231 |
| market | 0.8-1.0 | 0 | nan | nan |
| elo | 0.0-0.2 | 0 | nan | nan |
| elo | 0.2-0.4 | 7 | 0.384306 | 0.142857 |
| elo | 0.4-0.6 | 683 | 0.504466 | 0.513909 |
| elo | 0.6-0.8 | 7 | 0.607182 | 0.428571 |
| elo | 0.8-1.0 | 0 | nan | nan |

## Decision and saved artifacts

**Measured:** 505 declared reporting looks, F=3, B=8, K=4; no added variants. No multiple-comparison adjustment.
**Inferred:** unresolved_below_power; this experiment provides no admissible terminal closing ground. No serving promotion.
Prediction rows, input inventory/hashes, cached market targets, coefficients and summary JSON are under tests/scratch/codex/lead86_unit1/.
The frozen declaration is retained in docs/lead86_protocol.md. Record commands are in docs/lanes/lead86.md for the orchestrator.

## Combined coefficients and stability

**Measured:** these coefficients include the separately fitted calibration.

| Fold | Arm | Intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | regularized | -0.049166 | -0.184619 | 0.376279 | 0.212196 | 0.000000 |
| 2023 | four_term | -0.045905 | -0.159089 | 0.446507 | 0.123432 | 0.000000 |
| 2024 | regularized | 0.062556 | 0.408014 | 0.305888 | 0.099358 | 0.000000 |
| 2024 | four_term | 0.062556 | 0.408014 | 0.305888 | 0.099358 | 0.000000 |
| 2025 | regularized | -0.074161 | 0.381934 | 0.323533 | 0.126062 | 0.000000 |
| 2025 | four_term | -0.073122 | 0.384302 | 0.327701 | 0.116430 | 0.000000 |

**Measured:** selected penalties are 1, 0 and 0.1 across 2023, 2024 and 2025.
The model-logit coefficient changes sign; composition and movement remain positive.
Availability is constant on this complete-source population, so its coefficient is zero;
that is not evidence that availability has no effect.

**Read:** src/nfl_ats/clv.py:2184-2208 trains on completed games before each target
week and substitutes the opener before prediction. Upstream selection is retrospective.
**Measured:** saved upstream model identifiers agree; OOS game IDs are unique.
Maximum market-price projection error: 0.169936813;
maximum projected-gradient residual: 3.47861163e-08.
**Measured:** 3 of 3231 market fits exceed one point of price error;
they concern 1 unique game. Other games' maximum error is 9.74499431e-06.
The Denver-Las Vegas 2021 source gives Denver a 66.03% win chance but a 49.03%
cover chance as a 5.5-point underdog. Those prices cannot share a valid margin
distribution: winning implies covering the positive home handicap.
The inherited constrained projection reaches its parameter bounds on this source.
The declared population was retained; no exclusion, penalty change or refit followed.
**Inferred limitation:** reconcile this source discrepancy before treating the
regularizer result as evidence for serving. The mechanism remains unresolved.

**Inferred decision:** the small estimated Brier gain is unresolved_below_power.
The 5-4 disagreement record and varying selected penalty do not establish a stable
improvement. Under AGENTS.md research rules, uncertainty does not close this mechanism;
this diagnostic does not authorize changing the served card.
