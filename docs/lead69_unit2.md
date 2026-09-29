# LEAD-69 unit 2: quote-availability horizon

Protocol amendment saved in docs/lanes/lead69.md before outcomes.
**Measured:** two research looks; 33 numerical fits, 660 reporting looks; no tuning.
Historical frozen lines are openers, not pre-2026 Splash captures.

## Decisive games first

| Population | Mode | Arm vs base | Record | Decisive accuracy % [95%] |
| --- | --- | --- | --- | --- |
| extended | is | horizon | 11-8 | 57.894737 [36.842105, 78.947368] |
| extended | is | slots | 8-7 | 53.333333 [27.272727, 78.947368] |
| extended | oos | horizon | 4-12 | 25.000000 [5.555556, 47.619048] |
| extended | oos | slots | 13-17 | 43.333333 [25.000000, 61.111111] |
| extended_recent | is | horizon | 3-2 | 60.000000 [0.000000, 100.000000] |
| extended_recent | is | slots | 6-1 | 85.714286 [50.000000, 100.000000] |
| extended_recent | oos | horizon | 2-6 | 25.000000 [0.000000, 60.000000] |
| extended_recent | oos | slots | 8-11 | 42.105263 [20.000000, 65.217391] |
| original_replay | is | horizon | 5-7 | 41.666667 [15.384615, 69.230769] |
| original_replay | is | slots | 10-11 | 47.619048 [26.086957, 72.000000] |
| original_replay | oos | horizon | 8-6 | 57.142857 [30.769231, 81.818182] |
| original_replay | oos | slots | 23-19 | 54.761905 [38.235294, 71.435310] |

## Protocol and sources

**Read:** unit-1 terms/metrics: docs/lead69_protocol.md; older alignment: docs/lead73_unit2.md.
The pre-outcome amendment changes horizon timing to the start quote's observed availability.
For each book, sum Wednesday-onward changes, multiply by log(hours from start-quote
availability to min(kickoff, Sunday 16:00 ET)), then take the median across books.
Availability=max(capture, snapshot, book update). The base move remains the median
unweighted book net move. The original Tuesday-noon definition is replayed separately.
Those two interactions need not agree when book start times differ.
Older evidence ends at 12:45 Sunday; this archive limit was not a tuned cutoff.
**Measured:** source manifest/quote hashes, schedule/population hashes, timestamp gates
and move reconstruction parity against both prior units passed before outcome loading.
Regular-season nonpush rows with verified moves only; retain unit-1 SNF/MNF exclusions.
Missing older moves: 6; book horizons
13.073-152.083 hours over 3820 book/game pairs.

| Season | Eligible games |
| --- | --- |
| 2020 | 185 |
| 2021 | 202 |
| 2022 | 213 |
| 2023 | 229 |
| 2024 | 228 |
| 2025 | 228 |

**Read:** base is intercept plus model logit, signed move, availability, composition sum;
horizon adds one term; slots add five fixed non-reference interactions. Ridge=0.001;
training-only standardization; six LOSO folds plus IS, original three-season replay plus IS.
One fitted probability chooses each side at 0.5. Model-only uses the same discrete
conditional nonpush probability; neutral market p=0.5 selects home on ties.
Opener-to-close movement is evaluation only, never an input.
**Measured:** 10,000 paired season-stratified week-block draws, seed 20260929;
95% percentile intervals; probability_positive=P(gain>0)+0.5P(gain=0); predictions fixed.
Reporting budget: 33 fits + 120 aggregate metrics + 144 contrasts + 36 gaps +
240 season metric cells + 75 calibration bands + 12 decisive records = 660 looks.
These correlated diagnostics are not independent discoveries.

## Unit-1 continuity

**Measured:** original 685-game replay accuracy gain 0.291971 [-0.248538, 0.779438] pp
using its exact season bootstrap; probability_positive=0.796296.
Tables below use week-block intervals. extended_recent is the same 685 games under
six-season fits and quote-availability horizons; original_replay keeps three-season
fits and the Tuesday horizon. This separates extended training from exact reproduction.

## extended: 1285 games, 107 season/week blocks

**Measured:** accuracy is percent; movement is spread points; losses are natural units.

| Arm | Metric | IS [95%] | OOS [95%] | Optimistic IS/OOS gap [95%] |
| --- | --- | --- | --- | --- |
| base | accuracy | 56.575875 [54.097056, 59.077011] | 56.498054 [54.032890, 58.934236] | 0.077821 [-0.835882, 0.933852] |
| base | log_loss | 0.681379 [0.674209, 0.688513] | 0.683031 [0.675819, 0.690448] | 0.001653 [0.000373, 0.003044] |
| base | brier | 0.244235 [0.240782, 0.247668] | 0.245005 [0.241524, 0.248584] | 0.000770 [0.000178, 0.001402] |
| base | line_move | 0.602335 [0.521663, 0.690729] | 0.602335 [0.519512, 0.690263] | 0.000000 [-0.019795, 0.017844] |
| horizon | accuracy | 56.809339 [54.346036, 59.253499] | 55.875486 [53.499134, 58.263624] | 0.933852 [0.077220, 1.799722] |
| horizon | log_loss | 0.680937 [0.673772, 0.688151] | 0.683536 [0.676065, 0.691268] | 0.002598 [0.001081, 0.004242] |
| horizon | brier | 0.244031 [0.240552, 0.247522] | 0.245247 [0.241637, 0.248951] | 0.001215 [0.000519, 0.001950] |
| horizon | line_move | 0.593774 [0.511911, 0.681413] | 0.600778 [0.521803, 0.685754] | -0.007004 [-0.029781, 0.016627] |
| slots | accuracy | 56.653696 [54.126984, 59.178931] | 56.186770 [53.840244, 58.574866] | 0.466926 [-0.457326, 1.339664] |
| slots | log_loss | 0.678921 [0.671710, 0.686208] | 0.684285 [0.676405, 0.692496] | 0.005364 [0.003162, 0.007927] |
| slots | brier | 0.243320 [0.239846, 0.246842] | 0.245678 [0.241997, 0.249532] | 0.002358 [0.001451, 0.003400] |
| slots | line_move | 0.606226 [0.524117, 0.693999] | 0.584436 [0.506245, 0.669656] | 0.021790 [-0.001543, 0.047659] |
| model_only | accuracy | 52.529183 [49.731379, 55.226266] | 52.529183 [49.731379, 55.226266] | same baseline |
| model_only | log_loss | 0.698619 [0.690432, 0.706911] | 0.698619 [0.690432, 0.706911] | same baseline |
| model_only | brier | 0.252539 [0.248551, 0.256594] | 0.252539 [0.248551, 0.256594] | same baseline |
| model_only | line_move | 0.092607 [0.003653, 0.178903] | 0.092607 [0.003653, 0.178903] | same baseline |
| market | accuracy | 50.116732 [47.538462, 52.754119] | 50.116732 [47.538462, 52.754119] | same baseline |
| market | log_loss | 0.693147 [0.693147, 0.693147] | 0.693147 [0.693147, 0.693147] | same baseline |
| market | brier | 0.250000 [0.250000, 0.250000] | 0.250000 [0.250000, 0.250000] | same baseline |
| market | line_move | 0.092607 [0.002998, 0.184196] | 0.092607 [0.002998, 0.184196] | same baseline |

Positive gap means higher IS accuracy/movement or lower IS loss.
Positive contrast means challenger improves on the comparator.

| Contrast | Metric | Gain [95%] | probability_positive |
| --- | --- | --- | --- |
| horizon_base_is | accuracy | 0.233463 [-0.386399, 0.858034] | 0.770100 |
| horizon_base_is | log_loss | 0.000441 [-0.001210, 0.002055] | 0.704100 |
| horizon_base_is | brier | 0.000204 [-0.000571, 0.000963] | 0.702000 |
| horizon_base_is | line_move | -0.008560 [-0.027090, 0.010261] | 0.185100 |
| horizon_model_only_is | accuracy | 4.280156 [1.263773, 7.392769] | 0.997800 |
| horizon_model_only_is | log_loss | 0.017681 [0.007229, 0.028094] | 0.999500 |
| horizon_model_only_is | brier | 0.008508 [0.003430, 0.013550] | 0.999500 |
| horizon_model_only_is | line_move | 0.501167 [0.387305, 0.622058] | 1.000000 |
| horizon_market_is | accuracy | 6.692607 [2.911036, 10.493912] | 0.999700 |
| horizon_market_is | log_loss | 0.012210 [0.004997, 0.019375] | 0.999300 |
| horizon_market_is | brier | 0.005969 [0.002478, 0.009448] | 0.999300 |
| horizon_market_is | line_move | 0.501167 [0.398072, 0.610717] | 1.000000 |
| slots_base_is | accuracy | 0.077821 [-0.470219, 0.630915] | 0.607100 |
| slots_base_is | log_loss | 0.002458 [0.000396, 0.004966] | 0.993300 |
| slots_base_is | brier | 0.000916 [0.000113, 0.001883] | 0.989800 |
| slots_base_is | line_move | 0.003891 [-0.006216, 0.014718] | 0.760700 |
| slots_model_only_is | accuracy | 4.124514 [1.021170, 7.313459] | 0.996250 |
| slots_model_only_is | log_loss | 0.019698 [0.009563, 0.029836] | 1.000000 |
| slots_model_only_is | brier | 0.009219 [0.004327, 0.014076] | 0.999900 |
| slots_model_only_is | line_move | 0.513619 [0.398733, 0.636366] | 1.000000 |
| slots_market_is | accuracy | 6.536965 [2.755906, 10.411932] | 0.999400 |
| slots_market_is | log_loss | 0.014226 [0.006940, 0.021437] | 1.000000 |
| slots_market_is | brier | 0.006680 [0.003158, 0.010154] | 1.000000 |
| slots_market_is | line_move | 0.513619 [0.413498, 0.620694] | 1.000000 |
| horizon_base_oos | accuracy | -0.622568 [-1.238390, -0.076567] | 0.017400 |
| horizon_base_oos | log_loss | -0.000504 [-0.002549, 0.001525] | 0.325000 |
| horizon_base_oos | brier | -0.000242 [-0.001174, 0.000684] | 0.314800 |
| horizon_base_oos | line_move | -0.001556 [-0.015760, 0.012481] | 0.415500 |
| horizon_model_only_oos | accuracy | 3.346304 [0.233636, 6.547190] | 0.982550 |
| horizon_model_only_oos | log_loss | 0.015083 [0.003976, 0.026088] | 0.995100 |
| horizon_model_only_oos | brier | 0.007292 [0.001952, 0.012543] | 0.995300 |
| horizon_model_only_oos | line_move | 0.508171 [0.393389, 0.631008] | 1.000000 |
| horizon_market_oos | accuracy | 5.758755 [2.078442, 9.490826] | 0.998700 |
| horizon_market_oos | log_loss | 0.009611 [0.001879, 0.017082] | 0.992900 |
| horizon_market_oos | brier | 0.004753 [0.001049, 0.008363] | 0.993900 |
| horizon_market_oos | line_move | 0.508171 [0.404367, 0.614731] | 1.000000 |
| slots_base_oos | accuracy | -0.311284 [-1.101517, 0.537242] | 0.225150 |
| slots_base_oos | log_loss | -0.001254 [-0.004051, 0.001665] | 0.190600 |
| slots_base_oos | brier | -0.000673 [-0.001777, 0.000487] | 0.120300 |
| slots_base_oos | line_move | -0.017899 [-0.044997, 0.006855] | 0.076350 |
| slots_model_only_oos | accuracy | 3.657588 [0.546854, 6.872647] | 0.988800 |
| slots_model_only_oos | log_loss | 0.014333 [0.003241, 0.025149] | 0.993400 |
| slots_model_only_oos | brier | 0.006861 [0.001573, 0.011972] | 0.993700 |
| slots_model_only_oos | line_move | 0.491829 [0.378955, 0.613355] | 1.000000 |
| slots_market_oos | accuracy | 6.070039 [2.403101, 9.765680] | 0.998800 |
| slots_market_oos | log_loss | 0.008862 [0.000651, 0.016743] | 0.982600 |
| slots_market_oos | brier | 0.004322 [0.000468, 0.008003] | 0.985900 |
| slots_market_oos | line_move | 0.491829 [0.390062, 0.597988] | 1.000000 |

OOS season stability, descriptive:

| Season | Arm | Games | Accuracy % | Log loss | Brier | Line move |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | base | 185 | 54.0541 | 0.687573 | 0.247502 | 0.781081 |
| 2020 | horizon | 185 | 53.5135 | 0.686919 | 0.247114 | 0.786486 |
| 2020 | slots | 185 | 54.5946 | 0.688207 | 0.247800 | 0.786486 |
| 2020 | model_only | 185 | 52.9730 | 0.708603 | 0.257189 | 0.302703 |
| 2020 | market | 185 | 48.6486 | 0.693147 | 0.250000 | -0.162162 |
| 2021 | base | 202 | 53.4653 | 0.692691 | 0.249436 | 0.797030 |
| 2021 | horizon | 202 | 53.4653 | 0.691093 | 0.248742 | 0.797030 |
| 2021 | slots | 202 | 53.4653 | 0.700611 | 0.252500 | 0.831683 |
| 2021 | model_only | 202 | 55.4455 | 0.687555 | 0.247091 | -0.113861 |
| 2021 | market | 202 | 46.5347 | 0.693147 | 0.250000 | 0.217822 |
| 2022 | base | 213 | 55.3991 | 0.678946 | 0.243098 | 0.620892 |
| 2022 | horizon | 213 | 53.9906 | 0.685652 | 0.246271 | 0.545775 |
| 2022 | slots | 213 | 54.4601 | 0.683765 | 0.245381 | 0.597418 |
| 2022 | model_only | 213 | 52.1127 | 0.700219 | 0.253334 | 0.038732 |
| 2022 | market | 213 | 50.7042 | 0.693147 | 0.250000 | 0.043427 |
| 2023 | base | 229 | 58.0786 | 0.678099 | 0.242428 | 0.497817 |
| 2023 | horizon | 229 | 57.2052 | 0.677999 | 0.242381 | 0.537118 |
| 2023 | slots | 229 | 59.3886 | 0.675655 | 0.241621 | 0.458515 |
| 2023 | model_only | 229 | 50.6550 | 0.699722 | 0.253057 | 0.126638 |
| 2023 | market | 229 | 51.9651 | 0.693147 | 0.250000 | -0.010917 |
| 2024 | base | 228 | 58.7719 | 0.679657 | 0.243331 | 0.469298 |
| 2024 | horizon | 228 | 58.7719 | 0.678385 | 0.242700 | 0.469298 |
| 2024 | slots | 228 | 58.7719 | 0.674542 | 0.241588 | 0.390351 |
| 2024 | model_only | 228 | 54.8246 | 0.693516 | 0.250110 | 0.072368 |
| 2024 | market | 228 | 51.7544 | 0.693147 | 0.250000 | 0.135965 |
| 2025 | base | 228 | 58.3333 | 0.682933 | 0.245099 | 0.505482 |
| 2025 | horizon | 228 | 57.4561 | 0.682829 | 0.245104 | 0.523026 |
| 2025 | slots | 228 | 55.7018 | 0.685537 | 0.246352 | 0.509868 |
| 2025 | model_only | 228 | 49.5614 | 0.702820 | 0.254758 | 0.141447 |
| 2025 | market | 228 | 50.4386 | 0.693147 | 0.250000 | 0.294956 |

Fixed OOS reliability bins; no bin selected:

| Arm | Band | Games | Mean p(home) | Home-cover rate |
| --- | --- | --- | --- | --- |
| base | 0.0-0.2 | 1 | 0.120247 | 0.000000 |
| base | 0.2-0.4 | 111 | 0.363394 | 0.396396 |
| base | 0.4-0.6 | 1046 | 0.499302 | 0.500000 |
| base | 0.6-0.8 | 127 | 0.646989 | 0.606299 |
| base | 0.8-1.0 | 0 | empty | empty |
| horizon | 0.0-0.2 | 2 | 0.146773 | 0.000000 |
| horizon | 0.2-0.4 | 112 | 0.363284 | 0.375000 |
| horizon | 0.4-0.6 | 1034 | 0.498634 | 0.503868 |
| horizon | 0.6-0.8 | 137 | 0.648234 | 0.591241 |
| horizon | 0.8-1.0 | 0 | empty | empty |
| slots | 0.0-0.2 | 4 | 0.073004 | 0.000000 |
| slots | 0.2-0.4 | 112 | 0.364007 | 0.401786 |
| slots | 0.4-0.6 | 1046 | 0.500801 | 0.503824 |
| slots | 0.6-0.8 | 120 | 0.648542 | 0.591667 |
| slots | 0.8-1.0 | 3 | 0.894975 | 0.333333 |
| model_only | 0.0-0.2 | 0 | empty | empty |
| model_only | 0.2-0.4 | 186 | 0.365222 | 0.494624 |
| model_only | 0.4-0.6 | 1037 | 0.490010 | 0.498554 |
| model_only | 0.6-0.8 | 62 | 0.623530 | 0.564516 |
| model_only | 0.8-1.0 | 0 | empty | empty |
| market | 0.0-0.2 | 0 | empty | empty |
| market | 0.2-0.4 | 0 | empty | empty |
| market | 0.4-0.6 | 1285 | 0.500000 | 0.501167 |
| market | 0.6-0.8 | 0 | empty | empty |
| market | 0.8-1.0 | 0 | empty | empty |

## extended_recent: 685 games, 54 season/week blocks

**Measured:** accuracy is percent; movement is spread points; losses are natural units.

| Arm | Metric | IS [95%] | OOS [95%] | Optimistic IS/OOS gap [95%] |
| --- | --- | --- | --- | --- |
| base | accuracy | 58.248175 [55.192878, 61.209440] | 58.394161 [55.325444, 61.373391] | -0.145985 [-1.156069, 0.877193] |
| base | log_loss | 0.679231 [0.669558, 0.688393] | 0.680227 [0.670959, 0.689203] | 0.000996 [-0.000036, 0.002069] |
| base | brier | 0.243140 [0.238443, 0.247614] | 0.243617 [0.239093, 0.247988] | 0.000477 [-0.000021, 0.000991] |
| base | line_move | 0.495255 [0.377581, 0.614495] | 0.490876 [0.367580, 0.617108] | 0.004380 [-0.020498, 0.026549] |
| horizon | accuracy | 58.394161 [55.407114, 61.333333] | 57.810219 [54.862119, 60.714414] | 0.583942 [-0.715308, 1.886792] |
| horizon | log_loss | 0.678492 [0.668383, 0.688343] | 0.679735 [0.669813, 0.689457] | 0.001243 [0.000158, 0.002353] |
| horizon | brier | 0.242790 [0.237885, 0.247557] | 0.243393 [0.238606, 0.248117] | 0.000603 [0.000077, 0.001135] |
| horizon | line_move | 0.508394 [0.389546, 0.625924] | 0.509854 [0.387554, 0.635511] | -0.001460 [-0.031163, 0.029284] |
| slots | accuracy | 58.978102 [55.817293, 62.043796] | 57.956204 [54.896002, 60.975700] | 1.021898 [-0.145985, 2.199413] |
| slots | log_loss | 0.674376 [0.664599, 0.684225] | 0.678574 [0.668821, 0.688478] | 0.004198 [0.001549, 0.007666] |
| slots | brier | 0.241287 [0.236675, 0.245963] | 0.243185 [0.238528, 0.247902] | 0.001897 [0.000735, 0.003332] |
| slots | line_move | 0.495255 [0.375744, 0.617948] | 0.452920 [0.333575, 0.575150] | 0.042336 [0.010145, 0.082742] |
| model_only | accuracy | 51.678832 [48.148148, 55.328541] | 51.678832 [48.148148, 55.328541] | same baseline |
| model_only | log_loss | 0.698687 [0.689758, 0.707621] | 0.698687 [0.689758, 0.707621] | same baseline |
| model_only | brier | 0.252642 [0.248257, 0.257012] | 0.252642 [0.248257, 0.257012] | same baseline |
| model_only | line_move | 0.113504 [0.012061, 0.212869] | 0.113504 [0.012061, 0.212869] | same baseline |
| market | accuracy | 51.386861 [47.819609, 54.862119] | 51.386861 [47.819609, 54.862119] | same baseline |
| market | log_loss | 0.693147 [0.693147, 0.693147] | 0.693147 [0.693147, 0.693147] | same baseline |
| market | brier | 0.250000 [0.250000, 0.250000] | 0.250000 [0.250000, 0.250000] | same baseline |
| market | line_move | 0.139781 [0.021527, 0.267478] | 0.139781 [0.021527, 0.267478] | same baseline |

Positive gap means higher IS accuracy/movement or lower IS loss.
Positive contrast means challenger improves on the comparator.

| Contrast | Metric | Gain [95%] | probability_positive |
| --- | --- | --- | --- |
| horizon_base_is | accuracy | 0.145985 [-0.442494, 0.739645] | 0.669550 |
| horizon_base_is | log_loss | 0.000738 [-0.001085, 0.002560] | 0.793100 |
| horizon_base_is | brier | 0.000350 [-0.000525, 0.001222] | 0.790700 |
| horizon_base_is | line_move | 0.013139 [-0.001471, 0.035088] | 0.937950 |
| horizon_model_only_is | accuracy | 6.715328 [2.366864, 10.911875] | 0.998200 |
| horizon_model_only_is | log_loss | 0.020195 [0.007964, 0.033118] | 0.999500 |
| horizon_model_only_is | brier | 0.009852 [0.003868, 0.016156] | 0.999500 |
| horizon_model_only_is | line_move | 0.394891 [0.249242, 0.549360] | 1.000000 |
| horizon_market_is | accuracy | 7.007299 [2.170610, 11.936116] | 0.997300 |
| horizon_market_is | log_loss | 0.014655 [0.004805, 0.024764] | 0.998100 |
| horizon_market_is | brier | 0.007210 [0.002443, 0.012115] | 0.998300 |
| horizon_market_is | line_move | 0.368613 [0.236043, 0.506750] | 1.000000 |
| slots_base_is | accuracy | 0.729927 [0.000000, 1.461988] | 0.982300 |
| slots_base_is | log_loss | 0.004855 [0.001031, 0.009513] | 0.996400 |
| slots_base_is | brier | 0.001853 [0.000389, 0.003630] | 0.996300 |
| slots_base_is | line_move | 0.000000 [-0.011799, 0.012987] | 0.480450 |
| slots_model_only_is | accuracy | 7.299270 [2.962853, 11.600587] | 0.999050 |
| slots_model_only_is | log_loss | 0.024312 [0.012342, 0.037122] | 1.000000 |
| slots_model_only_is | brier | 0.011355 [0.005553, 0.017514] | 0.999900 |
| slots_model_only_is | line_move | 0.381752 [0.237532, 0.537848] | 1.000000 |
| slots_market_is | accuracy | 7.591241 [2.548726, 12.680115] | 0.998200 |
| slots_market_is | log_loss | 0.018772 [0.008922, 0.028548] | 1.000000 |
| slots_market_is | brier | 0.008713 [0.004037, 0.013325] | 1.000000 |
| slots_market_is | line_move | 0.355474 [0.224181, 0.491293] | 1.000000 |
| horizon_base_oos | accuracy | -0.583942 [-1.418440, 0.148148] | 0.076600 |
| horizon_base_oos | log_loss | 0.000491 [-0.001461, 0.002450] | 0.697700 |
| horizon_base_oos | brier | 0.000224 [-0.000706, 0.001156] | 0.691200 |
| horizon_base_oos | line_move | 0.018978 [0.004329, 0.037846] | 0.995650 |
| horizon_model_only_oos | accuracy | 6.131387 [1.793455, 10.443817] | 0.995900 |
| horizon_model_only_oos | log_loss | 0.018952 [0.006890, 0.031499] | 0.999100 |
| horizon_model_only_oos | brier | 0.009249 [0.003348, 0.015378] | 0.999100 |
| horizon_model_only_oos | line_move | 0.396350 [0.247436, 0.557785] | 1.000000 |
| horizon_market_oos | accuracy | 6.423358 [1.605839, 11.240876] | 0.995750 |
| horizon_market_oos | log_loss | 0.013412 [0.003690, 0.023334] | 0.995700 |
| horizon_market_oos | brier | 0.006607 [0.001883, 0.011394] | 0.996700 |
| horizon_market_oos | line_move | 0.370073 [0.230709, 0.510394] | 1.000000 |
| slots_base_oos | accuracy | -0.437956 [-1.615272, 0.865801] | 0.239600 |
| slots_base_oos | log_loss | 0.001653 [-0.002085, 0.006112] | 0.769200 |
| slots_base_oos | brier | 0.000433 [-0.001064, 0.002180] | 0.677700 |
| slots_base_oos | line_move | -0.037956 [-0.082112, -0.001462] | 0.020400 |
| slots_model_only_oos | accuracy | 6.277372 [1.769846, 10.775862] | 0.995550 |
| slots_model_only_oos | log_loss | 0.020113 [0.007604, 0.032626] | 0.998900 |
| slots_model_only_oos | brier | 0.009457 [0.003479, 0.015440] | 0.998700 |
| slots_model_only_oos | line_move | 0.339416 [0.195456, 0.497123] | 1.000000 |
| slots_market_oos | accuracy | 6.569343 [1.472700, 11.600751] | 0.994850 |
| slots_market_oos | log_loss | 0.014573 [0.004669, 0.024326] | 0.997400 |
| slots_market_oos | brier | 0.006815 [0.002098, 0.011472] | 0.997100 |
| slots_market_oos | line_move | 0.313139 [0.179394, 0.452066] | 1.000000 |

OOS season stability, descriptive:

| Season | Arm | Games | Accuracy % | Log loss | Brier | Line move |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | base | 229 | 58.0786 | 0.678099 | 0.242428 | 0.497817 |
| 2023 | horizon | 229 | 57.2052 | 0.677999 | 0.242381 | 0.537118 |
| 2023 | slots | 229 | 59.3886 | 0.675655 | 0.241621 | 0.458515 |
| 2023 | model_only | 229 | 50.6550 | 0.699722 | 0.253057 | 0.126638 |
| 2023 | market | 229 | 51.9651 | 0.693147 | 0.250000 | -0.010917 |
| 2024 | base | 228 | 58.7719 | 0.679657 | 0.243331 | 0.469298 |
| 2024 | horizon | 228 | 58.7719 | 0.678385 | 0.242700 | 0.469298 |
| 2024 | slots | 228 | 58.7719 | 0.674542 | 0.241588 | 0.390351 |
| 2024 | model_only | 228 | 54.8246 | 0.693516 | 0.250110 | 0.072368 |
| 2024 | market | 228 | 51.7544 | 0.693147 | 0.250000 | 0.135965 |
| 2025 | base | 228 | 58.3333 | 0.682933 | 0.245099 | 0.505482 |
| 2025 | horizon | 228 | 57.4561 | 0.682829 | 0.245104 | 0.523026 |
| 2025 | slots | 228 | 55.7018 | 0.685537 | 0.246352 | 0.509868 |
| 2025 | model_only | 228 | 49.5614 | 0.702820 | 0.254758 | 0.141447 |
| 2025 | market | 228 | 50.4386 | 0.693147 | 0.250000 | 0.294956 |

Fixed OOS reliability bins; no bin selected:

| Arm | Band | Games | Mean p(home) | Home-cover rate |
| --- | --- | --- | --- | --- |
| base | 0.0-0.2 | 0 | empty | empty |
| base | 0.2-0.4 | 48 | 0.365045 | 0.437500 |
| base | 0.4-0.6 | 574 | 0.503041 | 0.506969 |
| base | 0.6-0.8 | 63 | 0.639150 | 0.634921 |
| base | 0.8-1.0 | 0 | empty | empty |
| horizon | 0.0-0.2 | 0 | empty | empty |
| horizon | 0.2-0.4 | 51 | 0.364274 | 0.431373 |
| horizon | 0.4-0.6 | 561 | 0.501827 | 0.508021 |
| horizon | 0.6-0.8 | 73 | 0.641440 | 0.616438 |
| horizon | 0.8-1.0 | 0 | empty | empty |
| slots | 0.0-0.2 | 2 | 0.002757 | 0.000000 |
| slots | 0.2-0.4 | 50 | 0.367145 | 0.460000 |
| slots | 0.4-0.6 | 570 | 0.504937 | 0.508772 |
| slots | 0.6-0.8 | 62 | 0.640666 | 0.612903 |
| slots | 0.8-1.0 | 1 | 1.000000 | 1.000000 |
| model_only | 0.0-0.2 | 0 | empty | empty |
| model_only | 0.2-0.4 | 43 | 0.373810 | 0.534884 |
| model_only | 0.4-0.6 | 594 | 0.499818 | 0.511785 |
| model_only | 0.6-0.8 | 48 | 0.621592 | 0.520833 |
| model_only | 0.8-1.0 | 0 | empty | empty |
| market | 0.0-0.2 | 0 | empty | empty |
| market | 0.2-0.4 | 0 | empty | empty |
| market | 0.4-0.6 | 685 | 0.500000 | 0.513869 |
| market | 0.6-0.8 | 0 | empty | empty |
| market | 0.8-1.0 | 0 | empty | empty |

## original_replay: 685 games, 54 season/week blocks

**Measured:** accuracy is percent; movement is spread points; losses are natural units.

| Arm | Metric | IS [95%] | OOS [95%] | Optimistic IS/OOS gap [95%] |
| --- | --- | --- | --- | --- |
| base | accuracy | 57.372263 [54.093567, 60.667634] | 56.058394 [52.615666, 59.443732] | 1.313869 [-1.151079, 3.790226] |
| base | log_loss | 0.678350 [0.666424, 0.689481] | 0.680301 [0.668197, 0.691684] | 0.001951 [-0.000184, 0.004114] |
| base | brier | 0.242779 [0.237132, 0.248100] | 0.243705 [0.237928, 0.249131] | 0.000926 [-0.000103, 0.001967] |
| base | line_move | 0.595985 [0.482566, 0.714441] | 0.586496 [0.473370, 0.707310] | 0.009489 [-0.026471, 0.040936] |
| horizon | accuracy | 57.080292 [53.947124, 60.287822] | 56.350365 [53.052326, 59.568345] | 0.729927 [-1.775148, 3.244838] |
| horizon | log_loss | 0.677854 [0.665865, 0.689419] | 0.681281 [0.668626, 0.693751] | 0.003427 [0.000629, 0.006485] |
| horizon | brier | 0.242535 [0.236804, 0.248020] | 0.244140 [0.238081, 0.250065] | 0.001605 [0.000310, 0.002996] |
| horizon | line_move | 0.571168 [0.452965, 0.689593] | 0.595255 [0.481850, 0.716364] | -0.024088 [-0.080962, 0.022897] |
| slots | accuracy | 57.226277 [54.049978, 60.518826] | 56.642336 [53.362573, 59.854041] | 0.583942 [-1.321634, 2.507467] |
| slots | log_loss | 0.671684 [0.658533, 0.684620] | 0.689719 [0.670891, 0.710231] | 0.018036 [0.007460, 0.031394] |
| slots | brier | 0.240257 [0.234155, 0.246244] | 0.245899 [0.238585, 0.253272] | 0.005642 [0.002706, 0.009094] |
| slots | line_move | 0.552920 [0.435020, 0.675111] | 0.539781 [0.417880, 0.667511] | 0.013139 [-0.011645, 0.037411] |
| model_only | accuracy | 51.678832 [48.148148, 55.328541] | 51.678832 [48.148148, 55.328541] | same baseline |
| model_only | log_loss | 0.698687 [0.689758, 0.707621] | 0.698687 [0.689758, 0.707621] | same baseline |
| model_only | brier | 0.252642 [0.248257, 0.257012] | 0.252642 [0.248257, 0.257012] | same baseline |
| model_only | line_move | 0.113504 [0.012061, 0.212869] | 0.113504 [0.012061, 0.212869] | same baseline |
| market | accuracy | 51.386861 [47.819609, 54.862119] | 51.386861 [47.819609, 54.862119] | same baseline |
| market | log_loss | 0.693147 [0.693147, 0.693147] | 0.693147 [0.693147, 0.693147] | same baseline |
| market | brier | 0.250000 [0.250000, 0.250000] | 0.250000 [0.250000, 0.250000] | same baseline |
| market | line_move | 0.139781 [0.021527, 0.267478] | 0.139781 [0.021527, 0.267478] | same baseline |

Positive gap means higher IS accuracy/movement or lower IS loss.
Positive contrast means challenger improves on the comparator.

| Contrast | Metric | Gain [95%] | probability_positive |
| --- | --- | --- | --- |
| horizon_base_is | accuracy | -0.291971 [-1.167883, 0.586510] | 0.258700 |
| horizon_base_is | log_loss | 0.000495 [-0.001727, 0.002732] | 0.674800 |
| horizon_base_is | brier | 0.000243 [-0.000796, 0.001285] | 0.682700 |
| horizon_base_is | line_move | -0.024818 [-0.058908, 0.004380] | 0.052650 |
| horizon_model_only_is | accuracy | 5.401460 [0.892824, 9.812764] | 0.989700 |
| horizon_model_only_is | log_loss | 0.020833 [0.006535, 0.036118] | 0.998500 |
| horizon_model_only_is | brier | 0.010107 [0.003216, 0.017421] | 0.998500 |
| horizon_model_only_is | line_move | 0.457664 [0.312405, 0.613636] | 1.000000 |
| horizon_market_is | accuracy | 5.693431 [1.002794, 10.495627] | 0.990500 |
| horizon_market_is | log_loss | 0.015293 [0.003729, 0.027283] | 0.995700 |
| horizon_market_is | brier | 0.007465 [0.001980, 0.013196] | 0.996400 |
| horizon_market_is | line_move | 0.431387 [0.309550, 0.556207] | 1.000000 |
| slots_base_is | accuracy | -0.145985 [-1.601164, 1.176471] | 0.443500 |
| slots_base_is | log_loss | 0.006666 [-0.000067, 0.013857] | 0.973100 |
| slots_base_is | brier | 0.002522 [-0.000305, 0.005489] | 0.958000 |
| slots_base_is | line_move | -0.043066 [-0.078488, -0.012408] | 0.001850 |
| slots_model_only_is | accuracy | 5.547445 [1.014456, 10.000000] | 0.990950 |
| slots_model_only_is | log_loss | 0.027004 [0.011311, 0.043747] | 0.999500 |
| slots_model_only_is | brier | 0.012385 [0.005065, 0.020196] | 0.999500 |
| slots_model_only_is | line_move | 0.439416 [0.293462, 0.595482] | 1.000000 |
| slots_market_is | accuracy | 5.839416 [1.017442, 10.698381] | 0.991650 |
| slots_market_is | log_loss | 0.021464 [0.008527, 0.034614] | 0.999500 |
| slots_market_is | brier | 0.009743 [0.003756, 0.015845] | 0.999200 |
| slots_market_is | line_move | 0.413139 [0.294999, 0.536191] | 1.000000 |
| horizon_base_oos | accuracy | 0.291971 [-0.717386, 1.327434] | 0.718600 |
| horizon_base_oos | log_loss | -0.000980 [-0.004922, 0.002874] | 0.320100 |
| horizon_base_oos | brier | -0.000436 [-0.002160, 0.001292] | 0.316400 |
| horizon_base_oos | line_move | 0.008759 [-0.001445, 0.020290] | 0.956350 |
| horizon_model_only_oos | accuracy | 4.671533 [-0.435429, 9.544787] | 0.965750 |
| horizon_model_only_oos | log_loss | 0.017406 [0.002656, 0.032711] | 0.988600 |
| horizon_model_only_oos | brier | 0.008502 [0.001366, 0.015825] | 0.989600 |
| horizon_model_only_oos | line_move | 0.481752 [0.332851, 0.640178] | 1.000000 |
| horizon_market_oos | accuracy | 4.963504 [0.000000, 9.899861] | 0.976200 |
| horizon_market_oos | log_loss | 0.011866 [-0.000603, 0.024521] | 0.967900 |
| horizon_market_oos | brier | 0.005860 [-0.000065, 0.011919] | 0.974100 |
| horizon_market_oos | line_move | 0.455474 [0.334052, 0.582368] | 1.000000 |
| slots_base_oos | accuracy | 0.583942 [-1.451432, 2.590021] | 0.727000 |
| slots_base_oos | log_loss | -0.009418 [-0.026127, 0.004616] | 0.108900 |
| slots_base_oos | brier | -0.002194 [-0.007106, 0.002155] | 0.179800 |
| slots_base_oos | line_move | -0.046715 [-0.092678, -0.007236] | 0.009400 |
| slots_model_only_oos | accuracy | 4.963504 [0.144077, 9.784375] | 0.976700 |
| slots_model_only_oos | log_loss | 0.008968 [-0.014550, 0.030817] | 0.788500 |
| slots_model_only_oos | brier | 0.006743 [-0.002176, 0.015465] | 0.932200 |
| slots_model_only_oos | line_move | 0.426277 [0.276580, 0.587091] | 1.000000 |
| slots_market_oos | accuracy | 5.255474 [0.432230, 10.249182] | 0.981600 |
| slots_market_oos | log_loss | 0.003428 [-0.017084, 0.022256] | 0.645300 |
| slots_market_oos | brier | 0.004101 [-0.003272, 0.011415] | 0.866700 |
| slots_market_oos | line_move | 0.400000 [0.273577, 0.530703] | 1.000000 |

OOS season stability, descriptive:

| Season | Arm | Games | Accuracy % | Log loss | Brier | Line move |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | base | 229 | 58.0786 | 0.677687 | 0.242212 | 0.685590 |
| 2023 | horizon | 229 | 58.9520 | 0.678056 | 0.242391 | 0.698690 |
| 2023 | slots | 229 | 58.0786 | 0.707767 | 0.249353 | 0.558952 |
| 2023 | model_only | 229 | 50.6550 | 0.699722 | 0.253057 | 0.126638 |
| 2023 | market | 229 | 51.9651 | 0.693147 | 0.250000 | -0.010917 |
| 2024 | base | 228 | 54.3860 | 0.680455 | 0.243714 | 0.541667 |
| 2024 | horizon | 228 | 54.8246 | 0.680212 | 0.243594 | 0.550439 |
| 2024 | slots | 228 | 56.5789 | 0.673191 | 0.240871 | 0.506579 |
| 2024 | model_only | 228 | 54.8246 | 0.693516 | 0.250110 | 0.072368 |
| 2024 | market | 228 | 51.7544 | 0.693147 | 0.250000 | 0.135965 |
| 2025 | base | 228 | 55.7018 | 0.682773 | 0.245195 | 0.531798 |
| 2025 | horizon | 228 | 55.2632 | 0.685589 | 0.246444 | 0.536184 |
| 2025 | slots | 228 | 55.2632 | 0.688121 | 0.247458 | 0.553728 |
| 2025 | model_only | 228 | 49.5614 | 0.702820 | 0.254758 | 0.141447 |
| 2025 | market | 228 | 50.4386 | 0.693147 | 0.250000 | 0.294956 |

Fixed OOS reliability bins; no bin selected:

| Arm | Band | Games | Mean p(home) | Home-cover rate |
| --- | --- | --- | --- | --- |
| base | 0.0-0.2 | 0 | empty | empty |
| base | 0.2-0.4 | 54 | 0.344201 | 0.425926 |
| base | 0.4-0.6 | 533 | 0.505078 | 0.491557 |
| base | 0.6-0.8 | 95 | 0.648942 | 0.673684 |
| base | 0.8-1.0 | 3 | 0.805411 | 1.000000 |
| horizon | 0.0-0.2 | 0 | empty | empty |
| horizon | 0.2-0.4 | 59 | 0.347408 | 0.440678 |
| horizon | 0.4-0.6 | 527 | 0.505206 | 0.493359 |
| horizon | 0.6-0.8 | 97 | 0.655819 | 0.670103 |
| horizon | 0.8-1.0 | 2 | 0.866634 | 0.500000 |
| slots | 0.0-0.2 | 7 | 0.056828 | 0.428571 |
| slots | 0.2-0.4 | 57 | 0.352914 | 0.473684 |
| slots | 0.4-0.6 | 515 | 0.506740 | 0.491262 |
| slots | 0.6-0.8 | 99 | 0.653150 | 0.646465 |
| slots | 0.8-1.0 | 7 | 0.871339 | 0.714286 |
| model_only | 0.0-0.2 | 0 | empty | empty |
| model_only | 0.2-0.4 | 43 | 0.373810 | 0.534884 |
| model_only | 0.4-0.6 | 594 | 0.499818 | 0.511785 |
| model_only | 0.6-0.8 | 48 | 0.621592 | 0.520833 |
| model_only | 0.8-1.0 | 0 | empty | empty |
| market | 0.0-0.2 | 0 | empty | empty |
| market | 0.2-0.4 | 0 | empty | empty |
| market | 0.4-0.6 | 685 | 0.500000 | 0.513869 |
| market | 0.6-0.8 | 0 | empty | empty |
| market | 0.8-1.0 | 0 | empty | empty |

## Natural-unit coefficients

**Measured:** fold identifies the held-out season; IS is descriptive. Availability
is constant one in this population, so its standardized coefficient is zero.

| Population | Arm | Fold | Train n | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available | move_log_hours | move_slot_thursday | move_slot_friday | move_slot_saturday | move_slot_sunday_late | move_slot_other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extended | base | IS | 1285 | -0.012897 | 0.233932 | 0.243522 | 0.155736 | 0.000000 | — | — | — | — | — | — |
| extended | base | 2020 | 1100 | -0.007013 | 0.294118 | 0.228091 | 0.174225 | 0.000000 | — | — | — | — | — | — |
| extended | base | 2021 | 1083 | -0.004404 | 0.080797 | 0.264290 | 0.182388 | 0.000000 | — | — | — | — | — | — |
| extended | base | 2022 | 1072 | -0.015672 | 0.262309 | 0.247635 | 0.146745 | 0.000000 | — | — | — | — | — | — |
| extended | base | 2023 | 1056 | -0.028230 | 0.240319 | 0.228873 | 0.149478 | 0.000000 | — | — | — | — | — | — |
| extended | base | 2024 | 1057 | -0.027018 | 0.189876 | 0.236134 | 0.150256 | 0.000000 | — | — | — | — | — | — |
| extended | base | 2025 | 1057 | 0.006616 | 0.325546 | 0.257136 | 0.137648 | 0.000000 | — | — | — | — | — | — |
| extended | horizon | IS | 1285 | -0.013941 | 0.233893 | 0.243864 | 1.142888 | 0.000000 | -0.209512 | — | — | — | — | — |
| extended | horizon | 2020 | 1100 | -0.007510 | 0.288136 | 0.228736 | 1.199815 | 0.000000 | -0.217598 | — | — | — | — | — |
| extended | horizon | 2021 | 1083 | -0.005186 | 0.083670 | 0.264116 | 0.550904 | 0.000000 | -0.078709 | — | — | — | — | — |
| extended | horizon | 2022 | 1072 | -0.017777 | 0.262228 | 0.250795 | 1.997076 | 0.000000 | -0.393348 | — | — | — | — | — |
| extended | horizon | 2023 | 1056 | -0.028888 | 0.241307 | 0.228717 | 1.188402 | 0.000000 | -0.219999 | — | — | — | — | — |
| extended | horizon | 2024 | 1057 | -0.027918 | 0.188203 | 0.236299 | 0.710211 | 0.000000 | -0.118708 | — | — | — | — | — |
| extended | horizon | 2025 | 1057 | 0.006482 | 0.327524 | 0.257069 | 1.228558 | 0.000000 | -0.230788 | — | — | — | — | — |
| extended | slots | IS | 1285 | -0.007062 | 0.258260 | 0.242432 | 0.152720 | 0.000000 | — | 0.067023 | -18.572359 | 0.043156 | -0.002077 | 11.548279 |
| extended | slots | 2020 | 1100 | -0.000990 | 0.320709 | 0.226741 | 0.182803 | 0.000000 | — | 0.041096 | -18.338147 | 0.041303 | -0.028073 | 11.403671 |
| extended | slots | 2021 | 1083 | 0.004555 | 0.124778 | 0.261827 | 0.143468 | 0.000000 | — | 0.021169 | -18.260310 | 0.343469 | 0.086716 | 11.369973 |
| extended | slots | 2022 | 1072 | -0.009305 | 0.288099 | 0.248551 | 0.138634 | 0.000000 | — | 0.242111 | -18.232313 | 0.034401 | -0.010556 | 11.401352 |
| extended | slots | 2023 | 1056 | -0.022806 | 0.264982 | 0.227206 | 0.138092 | 0.000000 | — | 0.062257 | -10.519218 | 0.150510 | 0.010985 | 11.368202 |
| extended | slots | 2024 | 1057 | -0.025989 | 0.192745 | 0.236720 | 0.153441 | 0.000000 | — | 0.011057 | -22.985613 | -0.173748 | 0.003647 | 0.000000 |
| extended | slots | 2025 | 1057 | 0.014153 | 0.352340 | 0.256111 | 0.161023 | 0.000000 | — | 0.066098 | -18.277521 | -0.097965 | -0.058698 | 11.409853 |
| original_replay | base | IS | 685 | 0.007085 | 0.093011 | 0.257237 | 0.222241 | 0.000000 | — | — | — | — | — | — |
| original_replay | base | 2023 | 456 | -0.023948 | 0.101588 | 0.228034 | 0.241442 | 0.000000 | — | — | — | — | — | — |
| original_replay | base | 2024 | 457 | -0.004531 | -0.064602 | 0.243977 | 0.232904 | 0.000000 | — | — | — | — | — | — |
| original_replay | base | 2025 | 457 | 0.048438 | 0.271898 | 0.297421 | 0.200650 | 0.000000 | — | — | — | — | — | — |
| original_replay | horizon | IS | 685 | 0.005769 | 0.096832 | 0.259500 | 1.761497 | 0.000000 | -0.323625 | — | — | — | — | — |
| original_replay | horizon | 2023 | 456 | -0.023270 | 0.112086 | 0.230785 | 2.307984 | 0.000000 | -0.434823 | — | — | — | — | — |
| original_replay | horizon | 2024 | 457 | -0.004783 | -0.064836 | 0.244301 | 0.371261 | 0.000000 | -0.029133 | — | — | — | — | — |
| original_replay | horizon | 2025 | 457 | 0.045408 | 0.284572 | 0.300003 | 3.316706 | 0.000000 | -0.653790 | — | — | — | — | — |
| original_replay | slots | IS | 685 | 0.016719 | 0.173206 | 0.251369 | 0.164059 | 0.000000 | — | 0.287157 | -17.483695 | 0.728509 | 0.107783 | 10.958897 |
| original_replay | slots | 2023 | 456 | -0.008331 | 0.190443 | 0.207289 | 0.127760 | 0.000000 | — | 0.354778 | -9.734077 | 2.979106 | 0.190634 | 10.603194 |
| original_replay | slots | 2024 | 457 | -0.001207 | -0.035575 | 0.249866 | 0.172043 | 0.000000 | — | 0.097283 | -21.490777 | 0.189364 | 0.196675 | 0.000000 |
| original_replay | slots | 2025 | 457 | 0.059221 | 0.376730 | 0.295199 | 0.194328 | 0.000000 | — | 0.641516 | -16.887987 | 0.421129 | -0.059709 | 10.664876 |

**Measured:** extended horizon coefficient range [-0.393348, -0.078709], mean -0.209858, fold SD 0.109341; positive 0/6, negative 6/6.

**Measured:** original_replay horizon coefficient range [-0.653790, -0.029133], mean -0.372582, fold SD 0.316945; positive 0/3, negative 3/3.

## Interpretation and limits

**Measured:** primary accuracy gain -0.622568 [-1.238390, -0.076567] pp, probability_positive=0.017400;
log-loss gain -0.000504 [-0.002549, 0.001525], probability_positive=0.325000.
**Inferred:** unresolved_below_power pending the owner's serial registry record;
no terminal closure or serving proposal. AGENTS.md separates promotion from closure:
a zero-crossing interval closes nothing. No powered control or split-half study ran.
The accuracy interval is wholly adverse for this declared version; its proper-score
contrasts remain unresolved. This does not support changing the served picks or
closing the broader horizon mechanism, especially given the timing amendment.
LOSO coefficients are out of season but use future seasons and inherited frozen upstream
model/composition features. This is not a chronological outer test. Bootstrap omits
refit/selection uncertainty. Archive quote density and early 12:45 cutoff differ.
Historical forced-pick accuracy establishes neither profit nor game-level probability.

## Reproduction

.tools/uv.exe run --no-sync --no-cache python scripts/lead69_unit2.py

Prediction rows, quote horizons, coefficients, provenance and summaries:
tests/scratch/codex/lead69_unit2/. Owner-run record commands are in the lane; none ran.
