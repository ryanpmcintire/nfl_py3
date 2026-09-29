# LEAD-83 unit 2: shrink noisy movement

Historical 2020-2025 OPENER is the frozen pool-line proxy; the served four-term
recipe is the base. No pre-2026 Splash captures are used.

## Decisive games first

**Measured:** records count only outer games where the candidate and comparator
selected different sides. Exact p is the two-sided fair-coin null on these games;
no population split is claimed.

| Outer | Comparator | Decisive n | Candidate W-L | Exact p |
| --- | --- | --- | --- | --- |
| 2023 | four_term | 46 | 24-22 | 0.882996 |
| 2023 | model_only | 63 | 30-33 | 0.801306 |
| 2023 | market | 104 | 50-54 | 0.768792 |
| 2023 | elo | 63 | 30-33 | 0.801306 |
| 2024 | four_term | 0 | 0-0 | 1.000000 |
| 2024 | model_only | 0 | 0-0 | 1.000000 |
| 2024 | market | 83 | 40-43 | 0.826405 |
| 2024 | elo | 63 | 31-32 | 1.000000 |
| 2025 | four_term | 1 | 0-1 | 1.000000 |
| 2025 | model_only | 39 | 26-13 | 0.053252 |
| 2025 | market | 81 | 44-37 | 0.505236 |
| 2025 | elo | 110 | 64-46 | 0.104613 |
| pooled | four_term | 47 | 24-23 | 1.000000 |
| pooled | model_only | 102 | 56-46 | 0.372944 |
| pooled | market | 268 | 134-134 | 1.000000 |
| pooled | elo | 236 | 125-111 | 0.397474 |

## Protocol and upstream blocker

**Read:** the lane's pre-outcome amendment retains 501 looks:
(27*4+7+4)*(3+1)+25. Five arms, four candidate contrasts, four endpoints, three
outer seasons plus pooled panels, IS/OOS/gaps, coefficients, decisive records
and 25 reliability cells are the declared family. No outcome-selected variants.

**Measured:** upstream weak-stack margin ridge, home-side offsets, line-local
discrete pools, Elo margin fit and four-term/candidate coefficients were refitted
using only 2020 through Y-3. Pushes remain in upstream fits/distributions.
The 1,311 non-push rows retain unit 1's clocks and original/shrunk movements;
outer 2023-2025 has 697 games. Cached weekly-trained model probabilities are
never loaded. The active parent feature hash was verified. Later spread/total
inputs were replaced with opener spreads and archived Tuesday totals.

**Measured:** a nonnegative logit slope is tuned on Y-2 by Brier, then an
intercept is calibrated on Y-1 by log loss. The final probability reweights
the training discrete margin lattice's cover/loss groups, preserving push
mass and within-side shape; it alone selects sides. Market uses matched-book
Sunday median; Elo uses training-only linear margin on pregame Elo difference.
RPS is the full integer-margin CDF score on -100...100, evaluated on the same
non-push games as the other endpoints.

**Measured:** missing Tuesday totals: 3;
missing values stay missing. Thread pools are limited to one. No full-history
feature rebuild or served change was run.

**Inferred:** this fixed-cutoff refit of the served recipe does not reproduce
its weekly-trained cached scores. Upstream training predictions and downstream
training scores are optimistic; IS is diagnostic. Retrospective source reuse
still requires prospective confirmation.

## Outer scores and uncertainty

**Measured:** 95% intervals use 10,000 hierarchical season/whole-week bootstrap
replicates, seed 20260929. Single-season panels resample weeks; pooled panels
resample seasons then weeks. Positive contrasts mean candidate improvement;
probability_positive gives half weight to exact ties. Primary endpoint: Brier.
Historical forced picks are not per-game probabilities or profitability evidence.

| Outer | Arm | Record | Accuracy % | Log loss | Brier | RPS |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | shrunk_move | 118-115 | 50.643777 [43.621173, 57.142857] | 0.692686 [0.690239, 0.695340] | 0.249769 [0.248546, 0.251095] | 7.903128 [7.055032, 8.775778] |
| 2023 | four_term | 116-117 | 49.785408 [43.277311, 55.982906] | 0.692781 [0.691063, 0.694610] | 0.249817 [0.248958, 0.250731] | 7.890841 [7.046435, 8.759778] |
| 2023 | model_only | 121-112 | 51.931330 [45.777778, 57.894737] | 0.692825 [0.691689, 0.693997] | 0.249839 [0.249271, 0.250425] | 7.876955 [7.037398, 8.744481] |
| 2023 | market | 122-111 | 52.360515 [46.382979, 58.474765] | 0.689702 [0.685247, 0.694019] | 0.248282 [0.246058, 0.250438] | 7.466989 [6.702241, 8.268039] |
| 2023 | elo | 121-112 | 51.931330 [45.777778, 57.894737] | 0.692825 [0.691689, 0.693997] | 0.249839 [0.249271, 0.250425] | 7.559587 [6.806316, 8.327947] |
| 2024 | shrunk_move | 114-118 | 49.137931 [42.727273, 55.605381] | 0.694560 [0.689561, 0.699515] | 0.250706 [0.248208, 0.253182] | 7.391016 [6.853203, 7.905347] |
| 2024 | four_term | 114-118 | 49.137931 [42.727273, 55.605381] | 0.694560 [0.689561, 0.699515] | 0.250706 [0.248208, 0.253182] | 7.391016 [6.853203, 7.905347] |
| 2024 | model_only | 114-118 | 49.137931 [42.727273, 55.605381] | 0.694560 [0.689561, 0.699515] | 0.250706 [0.248208, 0.253182] | 7.391016 [6.853203, 7.905347] |
| 2024 | market | 117-115 | 50.431034 [43.478261, 56.735245] | 0.702934 [0.679633, 0.726400] | 0.254649 [0.243486, 0.266086] | 7.294938 [6.741965, 7.837913] |
| 2024 | elo | 115-117 | 49.568966 [43.805218, 55.144033] | 0.699350 [0.687774, 0.711878] | 0.253009 [0.247284, 0.259156] | 7.300027 [6.731521, 7.879947] |
| 2025 | shrunk_move | 130-102 | 56.034483 [50.216450, 61.842105] | 0.690404 [0.670397, 0.709205] | 0.248628 [0.238854, 0.257829] | 7.227952 [6.559549, 7.944881] |
| 2025 | four_term | 131-101 | 56.465517 [51.054852, 62.068966] | 0.689257 [0.670389, 0.706904] | 0.248070 [0.238778, 0.256707] | 7.221059 [6.552812, 7.936848] |
| 2025 | model_only | 117-115 | 50.431034 [43.277311, 58.078603] | 0.692787 [0.686048, 0.699424] | 0.249818 [0.246459, 0.253125] | 7.207346 [6.550685, 7.899395] |
| 2025 | market | 123-109 | 53.017241 [45.568773, 60.775862] | 0.686082 [0.669753, 0.700495] | 0.246541 [0.238461, 0.253655] | 7.056747 [6.436724, 7.695153] |
| 2025 | elo | 112-120 | 48.275862 [42.672414, 54.273504] | 0.693890 [0.691822, 0.695823] | 0.250372 [0.249338, 0.251338] | 7.217431 [6.594993, 7.852739] |
| pooled | shrunk_move | 362-335 | 51.936872 [47.058617, 56.932260] | 0.692550 [0.684403, 0.699280] | 0.249701 [0.245686, 0.252997] | 7.507933 [7.025743, 8.085466] |
| pooled | four_term | 361-336 | 51.793400 [46.857143, 57.080925] | 0.692200 [0.684006, 0.698310] | 0.249531 [0.245500, 0.252527] | 7.501531 [7.019463, 8.074795] |
| pooled | model_only | 352-345 | 50.502152 [46.407904, 54.493902] | 0.693390 [0.690342, 0.696484] | 0.250121 [0.248599, 0.251665] | 7.492325 [7.013769, 8.060327] |
| pooled | market | 362-335 | 51.936872 [47.832370, 56.150507] | 0.692901 [0.681745, 0.707023] | 0.249822 [0.244385, 0.256674] | 7.273170 [6.854743, 7.730102] |
| pooled | elo | 348-349 | 49.928264 [46.153846, 53.735632] | 0.695351 [0.691769, 0.701568] | 0.251071 [0.249300, 0.254114] | 7.359303 [6.958491, 7.805997] |

## Candidate improvements

**Measured:** all four contrasts are reported; registry commands are restricted
to candidate versus four-term. Positive loss contrasts mean lower candidate loss.

| Outer | Comparator | Endpoint | Improvement [95% interval] | probability_positive |
| --- | --- | --- | --- | --- |
| 2023 | four_term | accuracy_points | 0.858369 [-2.531646, 4.310345] | 0.68430 |
| 2023 | four_term | log_loss | 0.000095 [-0.000889, 0.000968] | 0.60060 |
| 2023 | four_term | brier | 0.000048 [-0.000443, 0.000484] | 0.60130 |
| 2023 | four_term | rps | -0.012288 [-0.022138, -0.003485] | 0.00110 |
| 2023 | model_only | accuracy_points | -1.287554 [-5.063291, 2.608696] | 0.25355 |
| 2023 | model_only | log_loss | 0.000139 [-0.001825, 0.001899] | 0.57740 |
| 2023 | model_only | brier | 0.000070 [-0.000910, 0.000949] | 0.57780 |
| 2023 | model_only | rps | -0.026173 [-0.046060, -0.008278] | 0.00060 |
| 2023 | market | accuracy_points | -1.716738 [-7.792208, 3.879310] | 0.29405 |
| 2023 | market | log_loss | -0.002984 [-0.007390, 0.001411] | 0.09330 |
| 2023 | market | brier | -0.001487 [-0.003685, 0.000707] | 0.09350 |
| 2023 | market | rps | -0.436139 [-0.731172, -0.164295] | 0.00040 |
| 2023 | elo | accuracy_points | -1.287554 [-5.063291, 2.608696] | 0.25355 |
| 2023 | elo | log_loss | 0.000139 [-0.001825, 0.001899] | 0.57740 |
| 2023 | elo | brier | 0.000070 [-0.000910, 0.000949] | 0.57780 |
| 2023 | elo | rps | -0.343541 [-0.647998, -0.057207] | 0.00890 |
| 2024 | four_term | accuracy_points | 0.000000 [0.000000, 0.000000] | 0.50000 |
| 2024 | four_term | log_loss | 0.000000 [0.000000, 0.000000] | 0.50000 |
| 2024 | four_term | brier | 0.000000 [0.000000, 0.000000] | 0.50000 |
| 2024 | four_term | rps | 0.000000 [0.000000, 0.000000] | 0.50000 |
| 2024 | model_only | accuracy_points | 0.000000 [0.000000, 0.000000] | 0.50000 |
| 2024 | model_only | log_loss | 0.000000 [0.000000, 0.000000] | 0.50000 |
| 2024 | model_only | brier | 0.000000 [0.000000, 0.000000] | 0.50000 |
| 2024 | model_only | rps | 0.000000 [0.000000, 0.000000] | 0.50000 |
| 2024 | market | accuracy_points | -1.293103 [-9.349593, 6.696429] | 0.37190 |
| 2024 | market | log_loss | 0.008374 [-0.014046, 0.030915] | 0.76240 |
| 2024 | market | brier | 0.003943 [-0.006854, 0.014826] | 0.75830 |
| 2024 | market | rps | -0.096078 [-0.329302, 0.154889] | 0.20910 |
| 2024 | elo | accuracy_points | -0.431034 [-6.723398, 6.167401] | 0.43075 |
| 2024 | elo | log_loss | 0.004790 [-0.005411, 0.015594] | 0.81170 |
| 2024 | elo | brier | 0.002303 [-0.002720, 0.007601] | 0.80650 |
| 2024 | elo | rps | -0.090989 [-0.353713, 0.214231] | 0.25270 |
| 2025 | four_term | accuracy_points | -0.431034 [-1.333333, 0.000000] | 0.17940 |
| 2025 | four_term | log_loss | -0.001147 [-0.002986, 0.000416] | 0.08430 |
| 2025 | four_term | brier | -0.000559 [-0.001450, 0.000200] | 0.08490 |
| 2025 | four_term | rps | -0.006894 [-0.016984, 0.002203] | 0.07580 |
| 2025 | model_only | accuracy_points | 5.603448 [0.000000, 11.111111] | 0.97615 |
| 2025 | model_only | log_loss | 0.002383 [-0.010272, 0.016615] | 0.62120 |
| 2025 | model_only | brier | 0.001190 [-0.004977, 0.008123] | 0.62500 |
| 2025 | model_only | rps | -0.020606 [-0.098052, 0.071990] | 0.29680 |
| 2025 | market | accuracy_points | 3.017241 [-5.454736, 11.842663] | 0.75030 |
| 2025 | market | log_loss | -0.004322 [-0.019063, 0.011132] | 0.28120 |
| 2025 | market | brier | -0.002087 [-0.009359, 0.005544] | 0.28440 |
| 2025 | market | rps | -0.171206 [-0.358979, 0.020649] | 0.04230 |
| 2025 | elo | accuracy_points | 7.758621 [-0.854793, 16.595745] | 0.96260 |
| 2025 | elo | log_loss | 0.003487 [-0.015414, 0.023924] | 0.62490 |
| 2025 | elo | brier | 0.001743 [-0.007511, 0.011779] | 0.62740 |
| 2025 | elo | rps | -0.010521 [-0.186633, 0.190418] | 0.44360 |
| pooled | four_term | accuracy_points | 0.143472 [-0.995733, 1.702128] | 0.49900 |
| pooled | four_term | log_loss | -0.000350 [-0.001419, 0.000353] | 0.24515 |
| pooled | four_term | brier | -0.000170 [-0.000690, 0.000176] | 0.24715 |
| pooled | four_term | rps | -0.006402 [-0.014020, 0.000000] | 0.02965 |
| pooled | model_only | accuracy_points | 1.434720 [-2.008680, 5.932234] | 0.72850 |
| pooled | model_only | log_loss | 0.000839 [-0.003490, 0.006531] | 0.62905 |
| pooled | model_only | brier | 0.000419 [-0.001683, 0.003204] | 0.63225 |
| pooled | model_only | rps | -0.015608 [-0.048453, 0.017615] | 0.14585 |
| pooled | market | accuracy_points | 0.000000 [-4.748372, 5.339105] | 0.49170 |
| pooled | market | log_loss | 0.000351 [-0.009293, 0.012876] | 0.47070 |
| pooled | market | brier | 0.000120 [-0.004559, 0.006123] | 0.46600 |
| pooled | market | rps | -0.234764 [-0.460451, -0.035945] | 0.00890 |
| pooled | elo | accuracy_points | 2.008608 [-2.970402, 8.757245] | 0.72000 |
| pooled | elo | log_loss | 0.002801 [-0.004546, 0.011381] | 0.77510 |
| pooled | elo | brier | 0.001370 [-0.002215, 0.005574] | 0.77460 |
| pooled | elo | rps | -0.148631 [-0.374905, 0.051156] | 0.08320 |

## Optimistic IS, OOS and gap

**Measured:** IS scores each final fold model on its training rows. Pooled IS
repeats earlier games across folds; bootstrap blocks keep repetitions together.
Gap is OOS minus IS, including positively oriented contrasts. Full IS/OOS/gap
intervals for every arm/contrast are also saved in scratch summary.json.

| Outer | Arm/contrast | Endpoint | IS | OOS | Gap [95% interval] |
| --- | --- | --- | --- | --- | --- |
| 2023 | shrunk_move | accuracy_points | 63.350785 | 50.643777 | -12.707009 [-22.385554, -2.587310] |
| 2023 | shrunk_move | log_loss | 0.687521 | 0.692686 | 0.005165 [0.001777, 0.008517] |
| 2023 | shrunk_move | brier | 0.247188 | 0.249769 | 0.002581 [0.000888, 0.004255] |
| 2023 | shrunk_move | rps | 6.606344 | 7.903128 | 1.296784 [0.332587, 2.283585] |
| 2023 | four_term | accuracy_points | 52.356021 | 49.785408 | -2.570613 [-11.040472, 6.062826] |
| 2023 | four_term | log_loss | 0.690219 | 0.692781 | 0.002562 [0.000074, 0.004998] |
| 2023 | four_term | brier | 0.248536 | 0.249817 | 0.001281 [0.000037, 0.002498] |
| 2023 | four_term | rps | 6.624459 | 7.890841 | 1.266382 [0.301323, 2.246750] |
| 2023 | model_only | accuracy_points | 50.261780 | 51.931330 | 1.669550 [-7.095877, 10.901640] |
| 2023 | model_only | log_loss | 0.693143 | 0.692825 | -0.000318 [-0.002077, 0.001352] |
| 2023 | model_only | brier | 0.249998 | 0.249839 | -0.000159 [-0.001038, 0.000676] |
| 2023 | model_only | rps | 6.644150 | 7.876955 | 1.232805 [0.264114, 2.211088] |
| 2023 | market | accuracy_points | 58.638743 | 52.360515 | -6.278228 [-15.537555, 2.749695] |
| 2023 | market | log_loss | 0.690598 | 0.689702 | -0.000896 [-0.007423, 0.005797] |
| 2023 | market | brier | 0.248730 | 0.248282 | -0.000448 [-0.003707, 0.002893] |
| 2023 | market | rps | 6.990812 | 7.466989 | 0.476177 [-0.588294, 1.481544] |
| 2023 | elo | accuracy_points | 50.261780 | 51.931330 | 1.669550 [-7.095877, 10.901640] |
| 2023 | elo | log_loss | 0.693143 | 0.692825 | -0.000318 [-0.002077, 0.001352] |
| 2023 | elo | brier | 0.249998 | 0.249839 | -0.000159 [-0.001038, 0.000676] |
| 2023 | elo | rps | 6.963884 | 7.559587 | 0.595703 [-0.483420, 1.602248] |
| 2023 | gain_vs_four_term | accuracy_points | 10.994764 | 0.858369 | -10.136395 [-16.479051, -3.697847] |
| 2023 | gain_vs_four_term | log_loss | 0.002698 | 0.000095 | -0.002602 [-0.003779, -0.001505] |
| 2023 | gain_vs_four_term | brier | 0.001348 | 0.000048 | -0.001300 [-0.001888, -0.000752] |
| 2023 | gain_vs_four_term | rps | 0.018115 | -0.012288 | -0.030403 [-0.042411, -0.019667] |
| 2023 | gain_vs_model_only | accuracy_points | 13.089005 | -1.287554 | -14.376559 [-21.106577, -7.638523] |
| 2023 | gain_vs_model_only | log_loss | 0.005621 | 0.000139 | -0.005483 [-0.007905, -0.003176] |
| 2023 | gain_vs_model_only | brier | 0.002810 | 0.000070 | -0.002740 [-0.003948, -0.001588] |
| 2023 | gain_vs_model_only | rps | 0.037806 | -0.026173 | -0.063979 [-0.089125, -0.041787] |
| 2023 | gain_vs_market | accuracy_points | 4.712042 | -1.716738 | -6.428780 [-17.021814, 3.806755] |
| 2023 | gain_vs_market | log_loss | 0.003077 | -0.002984 | -0.006061 [-0.012251, 0.000160] |
| 2023 | gain_vs_market | brier | 0.001542 | -0.001487 | -0.003029 [-0.006120, 0.000074] |
| 2023 | gain_vs_market | rps | 0.384468 | -0.436139 | -0.820607 [-1.217670, -0.463946] |
| 2023 | gain_vs_elo | accuracy_points | 13.089005 | -1.287554 | -14.376559 [-21.106577, -7.638523] |
| 2023 | gain_vs_elo | log_loss | 0.005621 | 0.000139 | -0.005483 [-0.007905, -0.003176] |
| 2023 | gain_vs_elo | brier | 0.002810 | 0.000070 | -0.002740 [-0.003948, -0.001588] |
| 2023 | gain_vs_elo | rps | 0.357540 | -0.343541 | -0.701081 [-1.139308, -0.297832] |
| 2024 | shrunk_move | accuracy_points | 49.009901 | 49.137931 | 0.128030 [-8.123383, 8.427653] |
| 2024 | shrunk_move | log_loss | 0.694659 | 0.694560 | -0.000099 [-0.006514, 0.006279] |
| 2024 | shrunk_move | brier | 0.250755 | 0.250706 | -0.000049 [-0.003255, 0.003138] |
| 2024 | shrunk_move | rps | 7.537898 | 7.391016 | -0.146882 [-1.141719, 0.800784] |
| 2024 | four_term | accuracy_points | 49.009901 | 49.137931 | 0.128030 [-8.123383, 8.427653] |
| 2024 | four_term | log_loss | 0.694659 | 0.694560 | -0.000099 [-0.006514, 0.006279] |
| 2024 | four_term | brier | 0.250755 | 0.250706 | -0.000049 [-0.003255, 0.003138] |
| 2024 | four_term | rps | 7.537898 | 7.391016 | -0.146882 [-1.141719, 0.800784] |
| 2024 | model_only | accuracy_points | 49.009901 | 49.137931 | 0.128030 [-8.123383, 8.427653] |
| 2024 | model_only | log_loss | 0.694659 | 0.694560 | -0.000099 [-0.006514, 0.006279] |
| 2024 | model_only | brier | 0.250755 | 0.250706 | -0.000049 [-0.003255, 0.003138] |
| 2024 | model_only | rps | 7.537898 | 7.391016 | -0.146882 [-1.141719, 0.800784] |
| 2024 | market | accuracy_points | 48.762376 | 50.431034 | 1.668658 [-7.072362, 9.866437] |
| 2024 | market | log_loss | 0.703654 | 0.702934 | -0.000720 [-0.031666, 0.031656] |
| 2024 | market | brier | 0.255251 | 0.254649 | -0.000602 [-0.015258, 0.014792] |
| 2024 | market | rps | 7.710857 | 7.294938 | -0.415919 [-1.509476, 0.669500] |
| 2024 | elo | accuracy_points | 48.762376 | 49.568966 | 0.806589 [-6.663583, 8.309967] |
| 2024 | elo | log_loss | 0.693969 | 0.699350 | 0.005381 [-0.010388, 0.022670] |
| 2024 | elo | brier | 0.250409 | 0.253009 | 0.002600 [-0.005192, 0.011165] |
| 2024 | elo | rps | 7.711652 | 7.300027 | -0.411625 [-1.633651, 0.805262] |
| 2024 | gain_vs_four_term | accuracy_points | 0.000000 | 0.000000 | 0.000000 [0.000000, 0.000000] |
| 2024 | gain_vs_four_term | log_loss | 0.000000 | 0.000000 | 0.000000 [0.000000, 0.000000] |
| 2024 | gain_vs_four_term | brier | 0.000000 | 0.000000 | 0.000000 [0.000000, 0.000000] |
| 2024 | gain_vs_four_term | rps | 0.000000 | 0.000000 | 0.000000 [0.000000, 0.000000] |
| 2024 | gain_vs_model_only | accuracy_points | 0.000000 | 0.000000 | 0.000000 [0.000000, 0.000000] |
| 2024 | gain_vs_model_only | log_loss | 0.000000 | 0.000000 | 0.000000 [0.000000, 0.000000] |
| 2024 | gain_vs_model_only | brier | 0.000000 | 0.000000 | 0.000000 [0.000000, 0.000000] |
| 2024 | gain_vs_model_only | rps | 0.000000 | 0.000000 | 0.000000 [0.000000, 0.000000] |
| 2024 | gain_vs_market | accuracy_points | 0.247525 | -1.293103 | -1.540628 [-11.755837, 8.650639] |
| 2024 | gain_vs_market | log_loss | 0.008995 | 0.008374 | -0.000621 [-0.029728, 0.029785] |
| 2024 | gain_vs_market | brier | 0.004495 | 0.003943 | -0.000553 [-0.014477, 0.014007] |
| 2024 | gain_vs_market | rps | 0.172959 | -0.096078 | -0.269037 [-0.595349, 0.072222] |
| 2024 | gain_vs_elo | accuracy_points | 0.247525 | -0.431034 | -0.678559 [-8.068949, 6.782610] |
| 2024 | gain_vs_elo | log_loss | -0.000690 | 0.004790 | 0.005480 [-0.007898, 0.020072] |
| 2024 | gain_vs_elo | brier | -0.000346 | 0.002303 | 0.002649 [-0.003949, 0.009838] |
| 2024 | gain_vs_elo | rps | 0.173754 | -0.090989 | -0.264743 [-0.684957, 0.194885] |
| 2025 | shrunk_move | accuracy_points | 63.843648 | 56.034483 | -7.809165 [-15.455216, -0.273718] |
| 2025 | shrunk_move | log_loss | 0.660815 | 0.690404 | 0.029589 [0.005060, 0.053653] |
| 2025 | shrunk_move | brier | 0.234079 | 0.248628 | 0.014549 [0.002595, 0.026308] |
| 2025 | shrunk_move | rps | 7.035211 | 7.227952 | 0.192741 [-0.876916, 1.191006] |
| 2025 | four_term | accuracy_points | 63.680782 | 56.465517 | -7.215265 [-14.602132, 0.112129] |
| 2025 | four_term | log_loss | 0.661745 | 0.689257 | 0.027511 [0.004372, 0.050348] |
| 2025 | four_term | brier | 0.234503 | 0.248070 | 0.013567 [0.002251, 0.024770] |
| 2025 | four_term | rps | 7.038998 | 7.221059 | 0.182060 [-0.888528, 1.180566] |
| 2025 | model_only | accuracy_points | 58.631922 | 50.431034 | -8.200887 [-17.244169, 0.881843] |
| 2025 | model_only | log_loss | 0.681055 | 0.692787 | 0.011732 [0.002154, 0.021456] |
| 2025 | model_only | brier | 0.243972 | 0.249818 | 0.005847 [0.001072, 0.010692] |
| 2025 | model_only | rps | 7.165315 | 7.207346 | 0.042031 [-1.044468, 1.034602] |
| 2025 | market | accuracy_points | 53.908795 | 53.017241 | -0.891553 [-9.753145, 8.265019] |
| 2025 | market | log_loss | 0.686159 | 0.686082 | -0.000078 [-0.018713, 0.017139] |
| 2025 | market | brier | 0.246593 | 0.246541 | -0.000052 [-0.009242, 0.008462] |
| 2025 | market | rps | 7.325687 | 7.056747 | -0.268940 [-1.417065, 0.752365] |
| 2025 | elo | accuracy_points | 50.488599 | 48.275862 | -2.212737 [-9.304747, 4.870733] |
| 2025 | elo | log_loss | 0.693127 | 0.693890 | 0.000763 [-0.001680, 0.003209] |
| 2025 | elo | brier | 0.249990 | 0.250372 | 0.000382 [-0.000840, 0.001604] |
| 2025 | elo | rps | 7.406454 | 7.217431 | -0.189023 [-1.357690, 0.832282] |
| 2025 | gain_vs_four_term | accuracy_points | 0.162866 | -0.431034 | -0.593901 [-1.946739, 0.657922] |
| 2025 | gain_vs_four_term | log_loss | 0.000930 | -0.001147 | -0.002077 [-0.004603, 0.000011] |
| 2025 | gain_vs_four_term | brier | 0.000424 | -0.000559 | -0.000982 [-0.002159, 0.000010] |
| 2025 | gain_vs_four_term | rps | 0.003787 | -0.006894 | -0.010681 [-0.022075, -0.000378] |
| 2025 | gain_vs_model_only | accuracy_points | 5.211726 | 5.603448 | 0.391722 [-6.125693, 6.841646] |
| 2025 | gain_vs_model_only | log_loss | 0.020240 | 0.002383 | -0.017857 [-0.033165, -0.001784] |
| 2025 | gain_vs_model_only | brier | 0.009893 | 0.001190 | -0.008703 [-0.016115, -0.000849] |
| 2025 | gain_vs_model_only | rps | 0.130104 | -0.020606 | -0.150710 [-0.256184, -0.044551] |
| 2025 | gain_vs_market | accuracy_points | 9.934853 | 3.017241 | -6.917612 [-17.729385, 4.478448] |
| 2025 | gain_vs_market | log_loss | 0.025344 | -0.004322 | -0.029666 [-0.052311, -0.006477] |
| 2025 | gain_vs_market | brier | 0.012514 | -0.002087 | -0.014601 [-0.025746, -0.003177] |
| 2025 | gain_vs_market | rps | 0.290476 | -0.171206 | -0.461681 [-0.728148, -0.200347] |
| 2025 | gain_vs_elo | accuracy_points | 13.355049 | 7.758621 | -5.596428 [-15.659454, 4.737718] |
| 2025 | gain_vs_elo | log_loss | 0.032312 | 0.003487 | -0.028825 [-0.052797, -0.004108] |
| 2025 | gain_vs_elo | brier | 0.015911 | 0.001743 | -0.014168 [-0.025871, -0.002059] |
| 2025 | gain_vs_elo | rps | 0.371243 | -0.010521 | -0.381764 [-0.661945, -0.117570] |
| pooled | shrunk_move | accuracy_points | 58.808933 | 51.936872 | -6.872061 [-12.758599, -0.727205] |
| pooled | shrunk_move | log_loss | 0.676344 | 0.692550 | 0.016207 [0.006264, 0.027404] |
| pooled | shrunk_move | brier | 0.241723 | 0.249701 | 0.007979 [0.003101, 0.013464] |
| pooled | shrunk_move | rps | 7.135436 | 7.507933 | 0.372497 [-0.597914, 1.276094] |
| pooled | four_term | accuracy_points | 56.989247 | 51.793400 | -5.195847 [-11.669397, 0.978675] |
| pooled | four_term | log_loss | 0.677242 | 0.692200 | 0.014958 [0.004964, 0.026224] |
| pooled | four_term | brier | 0.242151 | 0.249531 | 0.007381 [0.002459, 0.012925] |
| pooled | four_term | rps | 7.140221 | 7.501531 | 0.361310 [-0.608056, 1.264539] |
| pooled | model_only | accuracy_points | 54.094293 | 50.502152 | -3.592141 [-8.682041, 1.308794] |
| pooled | model_only | log_loss | 0.687511 | 0.693390 | 0.005879 [0.001776, 0.010497] |
| pooled | model_only | brier | 0.247191 | 0.250121 | 0.002930 [0.000884, 0.005233] |
| pooled | model_only | rps | 7.207483 | 7.492325 | 0.284842 [-0.694689, 1.196959] |
| pooled | market | accuracy_points | 52.936311 | 51.936872 | -0.999439 [-7.342346, 4.573665] |
| pooled | market | log_loss | 0.692707 | 0.692901 | 0.000195 [-0.015775, 0.019452] |
| pooled | market | brier | 0.249824 | 0.249822 | -0.000002 [-0.007726, 0.009349] |
| pooled | market | rps | 7.401491 | 7.273170 | -0.128322 [-1.134752, 0.793869] |
| pooled | elo | accuracy_points | 49.875931 | 49.928264 | 0.052333 [-4.064924, 4.213409] |
| pooled | elo | log_loss | 0.693411 | 0.695351 | 0.001940 [-0.003561, 0.008776] |
| pooled | elo | brier | 0.250131 | 0.251071 | 0.000940 [-0.001771, 0.004294] |
| pooled | elo | rps | 7.438521 | 7.359303 | -0.079218 [-1.151731, 0.861820] |
| pooled | gain_vs_four_term | accuracy_points | 1.819686 | 0.143472 | -1.676214 [-3.898747, 0.898855] |
| pooled | gain_vs_four_term | log_loss | 0.000899 | -0.000350 | -0.001249 [-0.002532, 0.000052] |
| pooled | gain_vs_four_term | brier | 0.000428 | -0.000170 | -0.000598 [-0.001211, 0.000032] |
| pooled | gain_vs_four_term | rps | 0.004785 | -0.006402 | -0.011187 [-0.019428, -0.002816] |
| pooled | gain_vs_model_only | accuracy_points | 4.714640 | 1.434720 | -3.279920 [-7.694038, 2.093179] |
| pooled | gain_vs_model_only | log_loss | 0.011167 | 0.000839 | -0.010328 [-0.017716, -0.003753] |
| pooled | gain_vs_model_only | brier | 0.005468 | 0.000419 | -0.005049 [-0.008657, -0.001837] |
| pooled | gain_vs_model_only | rps | 0.072047 | -0.015608 | -0.087655 [-0.142434, -0.045308] |
| pooled | gain_vs_market | accuracy_points | 5.872622 | 0.000000 | -5.872622 [-11.940929, 1.177286] |
| pooled | gain_vs_market | log_loss | 0.016363 | 0.000351 | -0.016012 [-0.030632, 0.000781] |
| pooled | gain_vs_market | brier | 0.008101 | 0.000120 | -0.007981 [-0.014981, 0.000154] |
| pooled | gain_vs_market | rps | 0.266055 | -0.234764 | -0.500819 [-0.768961, -0.242720] |
| pooled | gain_vs_elo | accuracy_points | 8.933002 | 2.008608 | -6.924394 [-13.014495, 0.618368] |
| pooled | gain_vs_elo | log_loss | 0.017067 | 0.002801 | -0.014266 [-0.026838, -0.003247] |
| pooled | gain_vs_elo | brier | 0.008409 | 0.001370 | -0.007039 [-0.013215, -0.001628] |
| pooled | gain_vs_elo | rps | 0.303085 | -0.148631 | -0.451716 [-0.761682, -0.172150] |

## Fold coefficients

**Measured:** final coefficients include tune slope and calibration intercept.
Availability is one throughout this complete-source population, so its
standardized coefficient is zero. It is not identifiable from the intercept.
Opposed movement coefficients can reflect collinearity; stability is shown below.

| Outer | Arm | Stage | Intercept | Model logit | Flags | Original move | Availability | Shrunk move |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | shrunk_move | fit | -1.14337471 | 1.55182935 | 0.37437738 | 2.59208263 | 0.00000000 | -2.63566443 |
| 2023 | shrunk_move | final | 0.00090810 | 0.02324307 | 0.00560737 | 0.03882383 | 0.00000000 | -0.03947659 |
| 2023 | four_term | fit | -1.14864175 | 1.53121776 | 0.36486530 | -0.00265778 | 0.00000000 | 0.00000000 |
| 2023 | four_term | final | 0.00959894 | 0.01224257 | 0.00291721 | -0.00002125 | 0.00000000 | 0.00000000 |
| 2024 | shrunk_move | fit | -0.98913124 | 1.41191809 | 0.18653874 | 0.78951593 | 0.00000000 | -0.78009882 |
| 2024 | shrunk_move | final | 0.07729167 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | -0.00000000 |
| 2024 | four_term | fit | -0.99351750 | 1.41851832 | 0.18583142 | 0.02675399 | 0.00000000 | 0.00000000 |
| 2024 | four_term | final | 0.07729167 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| 2025 | shrunk_move | fit | -0.26829025 | 1.07339074 | 0.32583663 | 0.67159286 | 0.00000000 | -0.59959687 |
| 2025 | shrunk_move | final | -0.17987441 | 0.45816324 | 0.13907924 | 0.28666091 | 0.00000000 | -0.25593033 |
| 2025 | four_term | fit | -0.26767422 | 1.07632359 | 0.32394395 | 0.08518841 | 0.00000000 | 0.00000000 |
| 2025 | four_term | final | -0.17525018 | 0.44934159 | 0.13523952 | 0.03556430 | 0.00000000 | 0.00000000 |

| Outer | Fit through | Arm | Tune slope | Calibration intercept |
| --- | --- | --- | --- | --- |
| 2023 | 2020 | shrunk_move | 0.01497785 | 0.01803339 |
| 2023 | 2020 | four_term | 0.00799531 | 0.01878269 |
| 2023 | 2020 | model_only | 0.00000000 | 0.01904819 |
| 2023 | 2020 | market | 0.33741734 | 0.00298493 |
| 2023 | 2020 | elo | 0.00000000 | 0.01904819 |
| 2024 | 2021 | shrunk_move | 0.00000000 | 0.07729167 |
| 2024 | 2021 | four_term | 0.00000000 | 0.07729167 |
| 2024 | 2021 | model_only | 0.00000000 | 0.07729167 |
| 2024 | 2021 | market | 2.24083397 | 0.11207085 |
| 2024 | 2021 | elo | 0.34643645 | 0.10650653 |
| 2025 | 2022 | shrunk_move | 0.42683733 | -0.06535811 |
| 2025 | 2022 | four_term | 0.41747816 | -0.06350204 |
| 2025 | 2022 | model_only | 0.19762292 | -0.09137657 |
| 2025 | 2022 | market | 0.90055469 | -0.02716521 |
| 2025 | 2022 | elo | 0.00000000 | -0.03448618 |

| Outer | Upstream n incl pushes | Last training day | tau^2 | Elo intercept | Elo slope |
| --- | --- | --- | --- | --- | --- |
| 2023 | 198 | 2021-01-03 | 2.31476587 | -3.47256784 | 0.07376231 |
| 2024 | 414 | 2022-01-09 | 2.71310943 | -2.71927260 | 0.07113877 |
| 2025 | 630 | 2023-01-08 | 2.37679326 | -1.76671033 | 0.05925920 |

## Reliability

**Measured:** five equal-width bins per arm; predicted and observed entries
are home-cover probabilities/frequencies conditional on a non-push.

| Arm | Band | Games | Predicted | Observed |
| --- | --- | --- | --- | --- |
| shrunk_move | 0.0-0.2 | 0 | - | - |
| shrunk_move | 0.2-0.4 | 23 | 0.366724 | 0.391304 |
| shrunk_move | 0.4-0.6 | 660 | 0.507285 | 0.509091 |
| shrunk_move | 0.6-0.8 | 14 | 0.629690 | 0.714286 |
| shrunk_move | 0.8-1.0 | 0 | - | - |
| four_term | 0.0-0.2 | 0 | - | - |
| four_term | 0.2-0.4 | 24 | 0.369495 | 0.375000 |
| four_term | 0.4-0.6 | 661 | 0.507777 | 0.509834 |
| four_term | 0.6-0.8 | 12 | 0.629956 | 0.750000 |
| four_term | 0.8-1.0 | 0 | - | - |
| model_only | 0.0-0.2 | 0 | - | - |
| model_only | 0.2-0.4 | 0 | - | - |
| model_only | 0.4-0.6 | 697 | 0.504551 | 0.509326 |
| model_only | 0.6-0.8 | 0 | - | - |
| model_only | 0.8-1.0 | 0 | - | - |
| market | 0.0-0.2 | 0 | - | - |
| market | 0.2-0.4 | 20 | 0.357812 | 0.400000 |
| market | 0.4-0.6 | 615 | 0.500720 | 0.507317 |
| market | 0.6-0.8 | 62 | 0.648690 | 0.564516 |
| market | 0.8-1.0 | 0 | - | - |
| elo | 0.0-0.2 | 0 | - | - |
| elo | 0.2-0.4 | 3 | 0.361679 | 0.666667 |
| elo | 0.4-0.6 | 687 | 0.506097 | 0.508006 |
| elo | 0.6-0.8 | 7 | 0.627017 | 0.571429 |
| elo | 0.8-1.0 | 0 | - | - |

## Interpretation and handoff

**Measured:** candidate accuracy IS=58.808933%,
OOS=51.936872%; gap=-6.872061 [-12.758599, -0.727205] points.
The tune multiplier is zero in 2024.
A zero tune multiplier makes that fold's conditional cover probability
constant after intercept calibration; it is not a split-half reliability test.
The 2023 secondary RPS improvement is -0.012288 [-0.022138, -0.003485];
probability_positive=0.00110.
**Inferred:** this adverse secondary result does not settle the predeclared
pooled Brier mechanism. The opposed original/shrunk coefficients and strong
tuning attenuation do not establish a stable noise-shrinkage gain.

**Measured:** primary Brier improvement: -0.000170 [-0.000690, 0.000176];
probability_positive=0.24715.
Opener accuracy improvement: 0.143472 [-0.995733, 1.702128] points;
probability_positive=0.49900.

**Inferred:** unresolved_below_power; no research closure or serving decision
follows. AGENTS.md:65-85 permits closure only through admissible refutation or
demonstrated positive-control power; AGENTS.md:113-126 separates promotion from
research closure. No positive control was run. The report remains pending the
orchestrator's serial registry commands in the lane.

**Measured:** executed:
.tools/uv.exe run --no-sync --no-cache python scripts/lead83_unit2.py.
Prediction rows, fixed-cutoff upstream probabilities, lineage, intervals and
fold parameters are under tests/scratch/codex/lead83_unit2/. No prediction-row
dumps are in docs. No registry command was run.

## Frozen amendment (verbatim)

Keep unit 1's population, clocks, tau² and jackknife weights. Outer 2023/24/25:
fit through Y−3, tune Y−2, calibrate Y−1, score Y. Candidate adds shrunk movement
to the four terms: model logit, signed flags, original movement, availability.
Rebuild upstream ridge margins, home-side offsets and discrete line-local pools
from training games including pushes; use no cached model probabilities. Keep
the active weak-stack recipe and fixed ridge constants, with cached pregame
features, opener spread and archived Tuesday total (missing remains missing).
Upstream fits use this same 2020–Y−3 population; training reads are optimistic.
Reuse cached signed flags only on non-push rows. Base/candidate logistic ridge
is 0.001. Market uses matched-book Sunday median; Elo uses training-only linear
margin on pregame Elo difference. Tune each arm's nonnegative logit multiplier
in [0,10] by Brier on Y−2; fit its intercept by log loss on Y−1. Reweight the
training discrete lattice's cover/loss masses to that one calibrated probability,
preserving push mass and within-side shape. One probability selects the side.
No outcome-selected grids, bands, or further variants.

Primary: opener Brier. Four endpoints (accuracy points, log loss, Brier, full-margin
RPS) use the declared non-push games; pushes remain in upstream fits/distributions.
RPS grid: integer margins −100…100. Compare candidate with four-term, model-only,
timestamp-matched market and Elo. Report decisive records first, IS/OOS/gaps,
fold coefficients, and five equal-width reliability bins. IS scores each final
fold model on its training rows; pooled IS repeats rows across folds. Bootstrap
10,000 replicates, seed 20260929: seasons then whole season/week blocks; ties
count half in probability_positive. F=3, B=7, K=4: retain 501 looks. Tune/calibration
are parts of their arm; no additional scored panels. Zero crossing closes nothing;
default unresolved_below_power. Reused archives require prospective confirmation.
