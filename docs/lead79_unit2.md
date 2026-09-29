# LEAD-79 unit 2: earlier same-fixture price

Protocol: `docs/lanes/lead79.md`, saved before outcomes. Historical opener is the frozen
pool-line proxy on 2020-2025 REG games; base is the served four-term feature set.
**Conditional assumption:** provider `observed_at_utc` counts as historical availability
under `docs/lanes/quote-provenance.md` Decision. This does not prove archive immutability.

**Measured decisive record first:** 8-14 on 22 changed picks;
accuracy 36.36364 [15.99474, 60.00000]. Eligible games: 1059;
chronological OOS games: 970; week blocks: 76.

**Measured:** all numerical tables below come from the single run.

## Source coverage

| Season | Schedule | Frozen non-push | Lookahead | Eligible |
| --- | --- | --- | --- | --- |
| 2020 | 256 | 220 | 92 | 89 |
| 2021 | 272 | 236 | 45 | 40 |
| 2022 | 271 | 248 | 195 | 181 |
| 2023 | 272 | 266 | 254 | 248 |
| 2024 | 272 | 266 | 254 | 251 |
| 2025 | 272 | 267 | 254 | 250 |

**Measured:** snapshots=7,883; quote_rows=4,767,704; same_fixture_spread_rows=4,282,112; prior_game_available_rows=4,120,720; before_both_prior_games_rows=1,424,736; clock_rejected_rows=0; valid_book_pairs=712,342; incomplete_pairs=13; lookahead_games=1,094.

Each anchor is the latest complete opposite spread pair snapshot before BOTH teams'
previous same-season kickoffs and Tuesday 09:00 Eastern, aggregated equally across books.
Week-one games cannot qualify. Home margin equals minus the HOME handicap, so the
anchor term is lookahead margin minus the canonical opener margin. No largest-move screen.

## Estimation and limits

One joint ridge-logistic calibration fits the four served inputs plus the anchor term.
The comparator refits the same four served inputs on the same rows. Six forward-only
season-held-out folds use earlier seasons only; fixed existing ridge is 0.001. There is
no tuning or selection sample; calibration is the fitted logistic layer, not a separate
held-out calibration stage. These are research fits and do not authorize serving.
The base movement horizon is through the existing pregame cutoff: opener-graded is not
a claim that all base inputs existed Tuesday. Raw model probabilities use the cached
discrete conditional non-push read; upstream model-selection chronology is inherited.
Neutral market probability is 0.5 and has no directional accuracy. Elo is unavailable
in the frozen paired artifact. Candidate/four-term margin MAE is undefined for a
binary probability calibration; raw-model and opener-margin MAE are reported when scored.
IS predictions come from a full-population fit and are evaluated on the same OOS-eligible
rows for the IS/OOS gap. Positive accuracy gap is optimism; negative loss gap is optimism.
Intervals are paired 10,000-draw week-block bootstraps within season (seed 79), conditional
on the fitted predictions; they do not include refit uncertainty. probability_positive
is bootstrap mass above zero plus half the mass at zero, not a Bayesian posterior.

| Held out | Earlier training | Test quotes | Status |
| --- | --- | --- | --- |
| 2020 | 0 | 89 | no_earlier_training_classes |
| 2021 | 89 | 40 | available |
| 2022 | 129 | 181 | available |
| 2023 | 310 | 248 | available |
| 2024 | 558 | 251 | available |
| 2025 | 809 | 250 | available |

## Pooled scores and IS/OOS gap

| Sample | Arm | Metric | Estimate [95% interval] |
| --- | --- | --- | --- |
| oos | anchor | accuracy_points | 57.01031 [54.03141, 59.93789] |
| oos | anchor | log_loss | 0.68701 [0.67344, 0.70064] |
| oos | anchor | brier | 0.24650 [0.24013, 0.25291] |
| oos | four_term | accuracy_points | 57.62887 [54.68729, 60.52638] |
| oos | four_term | log_loss | 0.68489 [0.67109, 0.69872] |
| oos | four_term | brier | 0.24549 [0.23903, 0.25197] |
| oos | model | accuracy_points | 53.71134 [50.71575, 56.70875] |
| oos | model | log_loss | 0.69443 [0.68681, 0.70233] |
| oos | model | brier | 0.25055 [0.24681, 0.25441] |
| oos | market | accuracy_points | unavailable |
| oos | market | log_loss | 0.69315 [0.69315, 0.69315] |
| oos | market | brier | 0.25000 [0.25000, 0.25000] |
| is | anchor | accuracy_points | 59.07216 [56.48621, 61.57895] |
| is | anchor | log_loss | 0.67743 [0.66746, 0.68724] |
| is | anchor | brier | 0.24220 [0.23748, 0.24690] |
| is | four_term | accuracy_points | 58.96907 [56.13487, 61.67851] |
| is | four_term | log_loss | 0.67776 [0.66778, 0.68768] |
| is | four_term | brier | 0.24237 [0.23760, 0.24709] |
| is | model | accuracy_points | 53.71134 [50.71575, 56.70875] |
| is | model | log_loss | 0.69443 [0.68681, 0.70233] |
| is | model | brier | 0.25055 [0.24681, 0.25441] |
| is | market | accuracy_points | unavailable |
| is | market | log_loss | 0.69315 [0.69315, 0.69315] |
| is | market | brier | 0.25000 [0.25000, 0.25000] |

| Arm | Metric | IS minus OOS [95% interval] |
| --- | --- | --- |
| anchor | accuracy_points | 2.06186 [-0.10111, 4.15336] |
| anchor | log_loss | -0.00958 [-0.01792, -0.00112] |
| anchor | brier | -0.00430 [-0.00811, -0.00040] |
| four_term | accuracy_points | 1.34021 [-0.61921, 3.25311] |
| four_term | log_loss | -0.00714 [-0.01552, 0.00134] |
| four_term | brier | -0.00313 [-0.00693, 0.00075] |
| model | accuracy_points | 0.00000 [0.00000, 0.00000] |
| model | log_loss | 0.00000 [0.00000, 0.00000] |
| model | brier | 0.00000 [0.00000, 0.00000] |
| market | accuracy_points | unavailable |
| market | log_loss | 0.00000 [0.00000, 0.00000] |
| market | brier | 0.00000 [0.00000, 0.00000] |

## Paired gains

Positive gains favor the anchor arm; all scored comparisons share the same rows.

| Sample | Comparator | Metric | Gain [95% interval] | probability_positive |
| --- | --- | --- | --- | --- |
| oos | four_term | accuracy_points | -0.61856 [-1.61780, 0.41537] | 0.11820 |
| oos | four_term | log_loss | -0.00211 [-0.00428, -0.00006] | 0.02180 |
| oos | four_term | brier | -0.00101 [-0.00204, -0.00004] | 0.01930 |
| oos | model | accuracy_points | 3.29897 [0.00000, 6.67362] | 0.97575 |
| oos | model | log_loss | 0.00743 [-0.00500, 0.02007] | 0.88220 |
| oos | model | brier | 0.00405 [-0.00176, 0.00996] | 0.91570 |
| oos | market | accuracy_points | unavailable | unavailable |
| oos | market | log_loss | 0.00614 [-0.00749, 0.01971] | 0.81090 |
| oos | market | brier | 0.00350 [-0.00291, 0.00987] | 0.85870 |
| is | four_term | accuracy_points | 0.10309 [-0.71140, 0.93170] | 0.58910 |
| is | four_term | log_loss | 0.00033 [-0.00052, 0.00116] | 0.77610 |
| is | four_term | brier | 0.00016 [-0.00025, 0.00056] | 0.77880 |
| is | model | accuracy_points | 5.36082 [2.07674, 8.65907] | 0.99945 |
| is | model | log_loss | 0.01701 [0.00682, 0.02792] | 0.99970 |
| is | model | brier | 0.00835 [0.00346, 0.01353] | 0.99970 |
| is | market | accuracy_points | unavailable | unavailable |
| is | market | log_loss | 0.01572 [0.00591, 0.02568] | 0.99930 |
| is | market | brier | 0.00780 [0.00310, 0.01252] | 0.99970 |

## Fold coefficients

| Arm | Held out | Training | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available | anchor_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor | 2021 | 89 | -0.163238 | 0.467619 | 0.347300 | 0.000000 | 0.000000 | -0.065722 |
| four_term | 2021 | 89 | -0.194364 | 0.513877 | 0.401758 | 0.000000 | 0.000000 | — |
| anchor | 2022 | 129 | -0.281093 | 0.617748 | 0.548068 | 0.000000 | 0.000000 | -0.033940 |
| four_term | 2022 | 129 | -0.296947 | 0.621809 | 0.581743 | 0.000000 | 0.000000 | — |
| anchor | 2023 | 310 | -0.220549 | 0.437434 | 0.531771 | 0.000000 | 0.000000 | 0.027908 |
| four_term | 2023 | 310 | -0.210503 | 0.447039 | 0.521503 | 0.000000 | 0.000000 | — |
| anchor | 2024 | 558 | -0.217385 | 0.259114 | 0.409783 | 0.201634 | 0.297404 | 0.022016 |
| four_term | 2024 | 558 | -0.208950 | 0.274907 | 0.403856 | 0.204529 | 0.288582 | — |
| anchor | 2025 | 809 | -0.192287 | 0.325671 | 0.348594 | 0.183683 | 0.201836 | 0.001860 |
| four_term | 2025 | 809 | -0.191624 | 0.327115 | 0.348370 | 0.183940 | 0.201108 | — |
| anchor | IS | 1059 | -0.184840 | 0.312834 | 0.289931 | 0.226447 | 0.172986 | 0.011163 |
| four_term | IS | 1059 | -0.181194 | 0.319213 | 0.288924 | 0.226639 | 0.168638 | — |

## Season stability

| Season | Arm | Metric | Estimate [95% interval] |
| --- | --- | --- | --- |
| 2021 | anchor | accuracy_points | 57.50000 [40.00000, 72.50000] |
| 2021 | anchor | log_loss | 0.64541 [0.59392, 0.71024] |
| 2021 | anchor | brier | 0.22703 [0.20180, 0.25896] |
| 2021 | four_term | accuracy_points | 62.50000 [41.66667, 82.50000] |
| 2021 | four_term | log_loss | 0.63237 [0.57598, 0.69198] |
| 2021 | four_term | brier | 0.22046 [0.19277, 0.24968] |
| 2021 | model | accuracy_points | 62.50000 [48.78049, 80.00000] |
| 2021 | model | log_loss | 0.67713 [0.61108, 0.73568] |
| 2021 | model | brier | 0.24185 [0.20979, 0.27031] |
| 2021 | market | log_loss | 0.69315 [0.69315, 0.69315] |
| 2021 | market | brier | 0.25000 [0.25000, 0.25000] |
| 2022 | anchor | accuracy_points | 61.32597 [53.63128, 68.57143] |
| 2022 | anchor | log_loss | 0.67781 [0.63800, 0.72298] |
| 2022 | anchor | brier | 0.24176 [0.22307, 0.26289] |
| 2022 | four_term | accuracy_points | 62.43094 [55.13514, 69.14894] |
| 2022 | four_term | log_loss | 0.67284 [0.63248, 0.71852] |
| 2022 | four_term | brier | 0.23940 [0.22065, 0.26086] |
| 2022 | model | accuracy_points | 55.24862 [47.22222, 63.18690] |
| 2022 | model | log_loss | 0.69180 [0.66828, 0.71612] |
| 2022 | model | brier | 0.24923 [0.23773, 0.26110] |
| 2022 | market | log_loss | 0.69315 [0.69315, 0.69315] |
| 2022 | market | brier | 0.25000 [0.25000, 0.25000] |
| 2023 | anchor | accuracy_points | 57.66129 [51.63934, 63.05221] |
| 2023 | anchor | log_loss | 0.70100 [0.67377, 0.73077] |
| 2023 | anchor | brier | 0.25273 [0.24010, 0.26659] |
| 2023 | four_term | accuracy_points | 57.66129 [52.01573, 62.75304] |
| 2023 | four_term | log_loss | 0.70034 [0.67241, 0.73158] |
| 2023 | four_term | brier | 0.25239 [0.23940, 0.26682] |
| 2023 | model | accuracy_points | 50.80645 [47.36842, 54.77178] |
| 2023 | model | log_loss | 0.70222 [0.69065, 0.71382] |
| 2023 | model | brier | 0.25430 [0.24868, 0.25985] |
| 2023 | market | log_loss | 0.69315 [0.69315, 0.69315] |
| 2023 | market | brier | 0.25000 [0.25000, 0.25000] |
| 2024 | anchor | accuracy_points | 52.58964 [45.84980, 58.84774] |
| 2024 | anchor | log_loss | 0.69057 [0.67009, 0.71218] |
| 2024 | anchor | brier | 0.24824 [0.23872, 0.25826] |
| 2024 | four_term | accuracy_points | 53.38645 [47.32510, 59.23077] |
| 2024 | four_term | log_loss | 0.68859 [0.66852, 0.71002] |
| 2024 | four_term | brier | 0.24736 [0.23794, 0.25737] |
| 2024 | model | accuracy_points | 55.37849 [48.76033, 61.92308] |
| 2024 | model | log_loss | 0.69245 [0.67854, 0.70637] |
| 2024 | model | brier | 0.24960 [0.24272, 0.25648] |
| 2024 | market | log_loss | 0.69315 [0.69315, 0.69315] |
| 2024 | market | brier | 0.25000 [0.25000, 0.25000] |
| 2025 | anchor | accuracy_points | 57.60000 [52.80000, 62.34818] |
| 2025 | anchor | log_loss | 0.68287 [0.65823, 0.70556] |
| 2025 | anchor | brier | 0.24513 [0.23358, 0.25583] |
| 2025 | four_term | accuracy_points | 57.60000 [52.80000, 62.34818] |
| 2025 | four_term | log_loss | 0.68299 [0.65830, 0.70568] |
| 2025 | four_term | brier | 0.24519 [0.23361, 0.25590] |
| 2025 | model | accuracy_points | 52.40000 [46.34146, 58.06452] |
| 2025 | model | log_loss | 0.69338 [0.67968, 0.70897] |
| 2025 | model | brier | 0.25012 [0.24338, 0.25782] |
| 2025 | market | log_loss | 0.69315 [0.69315, 0.69315] |
| 2025 | market | brier | 0.25000 [0.25000, 0.25000] |

## Reliability

Five quantile bands use only each fold's training probabilities; ties may leave empty bands.

| Arm | Band | Games | Mean prediction | Home cover [95% interval] |
| --- | --- | --- | --- | --- |
| anchor | 1 | 157 | 0.32082 | 0.44586 [0.37931, 0.51351] |
| anchor | 2 | 124 | 0.41552 | 0.42742 [0.36066, 0.50000] |
| anchor | 3 | 203 | 0.46566 | 0.40887 [0.33333, 0.48417] |
| anchor | 4 | 205 | 0.51413 | 0.51220 [0.43814, 0.58333] |
| anchor | 5 | 281 | 0.62177 | 0.61922 [0.57664, 0.66084] |
| four_term | 1 | 164 | 0.32429 | 0.43293 [0.36471, 0.50318] |
| four_term | 2 | 104 | 0.41547 | 0.40385 [0.32323, 0.48485] |
| four_term | 3 | 224 | 0.46355 | 0.42857 [0.36283, 0.49510] |
| four_term | 4 | 199 | 0.51789 | 0.51759 [0.44615, 0.58659] |
| four_term | 5 | 279 | 0.62191 | 0.62007 [0.57762, 0.66154] |
| model | 1 | 106 | 0.38377 | 0.50943 [0.42574, 0.59813] |
| model | 2 | 131 | 0.43533 | 0.41985 [0.34821, 0.49123] |
| model | 3 | 212 | 0.47144 | 0.47170 [0.41025, 0.53521] |
| model | 4 | 250 | 0.51389 | 0.51200 [0.44690, 0.57959] |
| model | 5 | 271 | 0.57827 | 0.54613 [0.49821, 0.59244] |
| market | 1 | 0 | unavailable | unavailable |
| market | 2 | 0 | unavailable | unavailable |
| market | 3 | 0 | unavailable | unavailable |
| market | 4 | 0 | unavailable | unavailable |
| market | 5 | 970 | 0.50000 | 0.50000 [0.47168, 0.52824] |

## Margin MAE

| Cached margin | MAE [95% interval] |
| --- | --- |
| model | 9.97999 [9.49577, 10.47361] |
| market | 9.95103 [9.49338, 10.42371] |

**Measured looks:** 209 numeric reporting looks, including fitted coefficients;
291-look ceiling retained (B=2, six scheduled folds, no extra endpoints). Missing cells
are reported unavailable and consume no outcome look. Correlated looks are not independent evidence.

## Decision and handoff

**Measured:** the candidate went 553-417, versus 559-411 for the four-term base.
Its paired log-loss and Brier gain intervals lie wholly below zero (tables above);
accuracy gain is -0.61856 points [-1.61780, 0.41537], probability_positive=0.1182.
The anchor coefficient is negative in 2021-2022 and positive in 2023-2025,
shrinking to 0.001860 per point in the last fold.
**Inferred:** this fit does not support adding the anchor to the served probability.
The adverse loss evidence is retained for the orchestrator to adjudicate; the
primary accuracy question is not closed, and no reliability or power control was estimated.

**Inferred:** unresolved_below_power pending serial registry review; no positive control
or admissible closing ground is claimed. Under AGENTS.md, a zero-crossing interval does
not close a signal, and a favorable probability_positive alone does not authorize serving.

**Measured command:** `UV_NO_CACHE=1 .tools/uv.exe run --no-sync python scripts/lead79_unit2.py`.
Detailed provenance, coefficients, bootstrap summaries, and prediction rows are saved under
`tests/scratch/codex/lead79_unit2/`; no row dumps are stored in docs. Registry commands belong
to the orchestrator. No served card, registry, publication, or Git history was changed.

## Frozen pre-outcome amendment

Copied verbatim from the saved declaration; no protocol revision after outcomes.

### Unit-2 amendment — fixed before outcomes, 2026-09-29
Population: 2020–2025 REG games in the frozen four-term fit, restricted to valid
same-fixture lookahead quotes. Historical OPENER is the frozen pool-line proxy;
no pre-2026 Splash capture is required. Under the root Decision in
`quote-provenance.md`, provider `observed_at_utc` is assumed to be availability
time. All results are conditional on that assumption; archive revision risk remains.
Require the snapshot and book/market clocks before BOTH teams' immediately
preceding same-season games and before Tuesday 09:00 Eastern of the target week.
Require both prior games, matching teams/game identity, and complete opposing spread
pairs. Take the latest qualifying snapshot and its median book home spread, with
all books equally weighted; no outcome-based book selection or revision screen.
Term: lookahead-implied home margin minus opener-implied home margin, in points
(`opener_home_handicap - lookahead_home_handicap`; canonical margin fields use
`lookahead_margin - tue_open_home_spread`). One logistic fit adds that term
to the served four inputs, including existing availability and market movement.
Both arms use identical eligible rows and base inputs. Movement retains the served
pregame horizon; this is an opener-GRADED pregame study, not a Tuesday-only forecast.
Folds: six scheduled forward-only season-held-out folds; fit only earlier seasons,
never the held-out or later season. Fixed existing ridge 0.001; training-only
standardization, no search, selection set, or second calibration fit. Missing
training/class variation leaves a fold unavailable. Full-population IS is diagnostic.
Metrics retained: opener accuracy, Brier, log loss; margin MAE and Elo only if
matching cached predictions exist, otherwise explicitly unavailable, never invented.
Paired comparators: four-term, raw discrete model, neutral market, cached Elo if
available. Pushes excluded by frozen non-push target. One fitted probability picks
the side, with 0.5 choosing home; neutral market has no directional pick.
Report decisive changed-pick record first, fold coefficients, IS/OOS gaps, season
stability, five training-quantile reliability bands, and paired season-stratified
week-block bootstrap (10,000 draws, seed 79), 95% intervals and probability_positive.
Retain B=2, F=6, E=0 and the 291-look ceiling; count actual reporting looks without
new specifications. A zero-crossing interval closes nothing; no positive control
is claimed. Any inconclusive fitted effect remains unresolved_below_power.

