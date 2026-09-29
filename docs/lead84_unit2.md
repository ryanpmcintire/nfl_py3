# LEAD-84 unit 2: combine book distributions before the opener

**Measured:** `.tools/uv.exe run --no-sync python scripts/lead84_unit2.py`.
Historical OPENER is the frozen Splash-line proxy on the 2020-2025 source-complete population; pool captures begin in 2026. The base is the served four-term recipe, refitted within each chronological fold. No target closing-line inputs, served-card changes or registry writes.

## Decisive record first

| Outer | Comparator | Different picks | Candidate W-L |
| --- | --- | --- | --- |
| 2023 | served_base | 37 | 16-21 |
| 2023 | model | 112 | 63-49 |
| 2023 | market | 109 | 54-55 |
| 2023 | elo | 131 | 74-57 |
| 2024 | served_base | 8 | 4-4 |
| 2024 | model | 71 | 45-26 |
| 2024 | market | 84 | 52-32 |
| 2024 | elo | 90 | 55-35 |
| 2025 | served_base | 5 | 2-3 |
| 2025 | model | 82 | 43-39 |
| 2025 | market | 75 | 36-39 |
| 2025 | elo | 125 | 69-56 |
| pooled | served_base | 50 | 22-28 |
| pooled | model | 265 | 151-114 |
| pooled | market | 268 | 142-126 |
| pooled | elo | 346 | 198-148 |

**Measured:** records exclude opener pushes. Each side follows its sole calibrated PMF probability; a tie selects home.

**Measured:** primary RPS improvement -0.009385 [-0.024230, +0.005969]; probability_positive 0.1048. **Inferred:** the primary mixture question remains unresolved_below_power; this does not authorize a served change.

## Locked design and chronology

**Read:** the unit-two amendment in `docs/lanes/lead84.md` was saved before outcomes; its immutable copy and SHA-256 are in scratch. Outer 2023/2024/2025 fits through Y-3, tunes one shared logit multiplier on Y-2 base log loss, and calibrates each intercept on Y-1. Logistic ridge is 0.001 throughout. No candidate was chosen after scoring.
**Measured:** weekly upstream model training cutoffs precede predictions by more than two days. Combined cached probabilities are never read. Base and candidate receive reconstructed timestamp-matched leader-book moves in every year. Fit, tune, calibration and outer seasons are disjoint. Elo uses 2019 warm-up, K=20, 0.75 annual carryover and a four-hour result delay.
**Measured:** the fifth term is mixture logit minus averaged-constraint logit. Candidate probabilities reweight mixture PMF nonpush tails; other arms reweight averaged-constraint PMF tails. Push mass is preserved. RPS sums squared CDF errors on integer margins -100..100, on all games; binary endpoints omit pushes.
**Inferred:** this RPS contrast measures both changed PMF shape and the fitted fifth term, without isolating their contributions. Baseline RPS uses the declared market lattice, not a reconstruction of the production model's full served margin distribution. Weekly upstream model archives are retrospective pregame predictions, not a new full-pipeline outer refit.

| Outer | Fit n | Tune n | Calibrate n | Common multiplier |
| --- | --- | --- | --- | --- |
| 2023 | 200 | 213 | 210 | 0.728776 |
| 2024 | 413 | 210 | 233 | 1.494018 |
| 2025 | 623 | 233 | 232 | 1.027767 |

## Source repair and population

**Measured:** the pre-outcome clock amendment excludes 2020_11_DET_CAR from every arm and prior because only one usable leader-book timestamp exists. The first two attempts stopped before fitting/scoring; the third completed. Production push-row flags use historical completed-game spreads solely for prior-week standings eligibility. No target closing line enters a term or grade.

**Measured:** excluded 17 price-incompatible book panels in 9 games, from both constructions, without outcomes. Retained 18387 panels and 1353 games (1320 nonpush, 33 pushes); 0 games lost two-book coverage. Both priors and all distributions were rebuilt.
**Measured:** moves use 7394 archived files and 558024 admitted quote rows; every retained game has a reconstructed move. Hash, target-week, observation/book/market-clock and kickoff checks ran before fitting.
**Measured:** repaired projections: maximum constraint error 0.00887640; maximum stationarity 2.8e-07; maximum PMF normalization error 9.99e-16.

| Season | Games | Nonpush |
| --- | --- | --- |
| 2020 | 207 | 200 |
| 2021 | 216 | 213 |
| 2022 | 216 | 210 |
| 2023 | 239 | 233 |
| 2024 | 238 | 232 |
| 2025 | 237 | 232 |

## Effects and uncertainty

**Measured:** 10,000 paired bootstrap draws, seed 84 (85 for OOS), resample seasons then whole weeks; single-season panels resample weeks. Intervals are percentile 95%. Positive contrasts favor the candidate: baseline minus candidate for losses, candidate minus baseline for accuracy in percentage points. Historical accuracy is not a game's probability or evidence of profitability.
**Measured:** optimistic IS scores final fold parameters on fitting rows; repeated historical rows across folds remain blocked by their original season/week. Contrast gap is IS benefit minus OOS benefit; absolute loss gaps are OOS minus IS. Tune and calibration rows are not outer results.

### Five arms

| Panel | Stage | Arm/comparator | Metric | Estimate [95% interval] | probability_positive |
| --- | --- | --- | --- | --- | --- |
| 2023 | IS | candidate | rps | +6.915751 [+6.328870, +7.592250] | - |
| 2023 | IS | served_base | rps | +6.927248 [+6.332218, +7.608271] | - |
| 2023 | IS | model | rps | +6.938003 [+6.332021, +7.647881] | - |
| 2023 | IS | market | rps | +6.926021 [+6.327933, +7.629537] | - |
| 2023 | IS | elo | rps | +6.924544 [+6.308183, +7.647246] | - |
| 2023 | IS | candidate | log_loss | +0.684112 [+0.673139, +0.695705] | - |
| 2023 | IS | served_base | log_loss | +0.686080 [+0.677136, +0.694112] | - |
| 2023 | IS | model | log_loss | +0.693126 [+0.690816, +0.695523] | - |
| 2023 | IS | market | log_loss | +0.691660 [+0.685709, +0.697770] | - |
| 2023 | IS | elo | log_loss | +0.689256 [+0.682838, +0.695566] | - |
| 2023 | IS | candidate | brier | +0.245522 [+0.240091, +0.251260] | - |
| 2023 | IS | served_base | brier | +0.246489 [+0.242045, +0.250472] | - |
| 2023 | IS | model | brier | +0.249990 [+0.248835, +0.251188] | - |
| 2023 | IS | market | brier | +0.249259 [+0.246290, +0.252305] | - |
| 2023 | IS | elo | brier | +0.248036 [+0.244848, +0.251189] | - |
| 2023 | IS | candidate | accuracy_points | +54.000000 [+47.596154, +60.396040] | - |
| 2023 | IS | served_base | accuracy_points | +51.000000 [+44.278607, +58.762887] | - |
| 2023 | IS | model | accuracy_points | +48.500000 [+41.951220, +54.871795] | - |
| 2023 | IS | market | accuracy_points | +51.500000 [+45.672554, +57.211538] | - |
| 2023 | IS | elo | accuracy_points | +55.000000 [+50.500000, +59.788360] | - |
| 2023 | OOS | candidate | rps | +7.379608 [+6.594626, +8.226907] | - |
| 2023 | OOS | served_base | rps | +7.368537 [+6.585999, +8.214978] | - |
| 2023 | OOS | model | rps | +7.426274 [+6.635031, +8.273216] | - |
| 2023 | OOS | market | rps | +7.391252 [+6.592277, +8.243975] | - |
| 2023 | OOS | elo | rps | +7.433405 [+6.641306, +8.270761] | - |
| 2023 | OOS | candidate | log_loss | +0.685271 [+0.673059, +0.698130] | - |
| 2023 | OOS | served_base | log_loss | +0.682322 [+0.671355, +0.693594] | - |
| 2023 | OOS | model | log_loss | +0.692857 [+0.691585, +0.694083] | - |
| 2023 | OOS | market | log_loss | +0.688098 [+0.682975, +0.693161] | - |
| 2023 | OOS | elo | log_loss | +0.695107 [+0.685577, +0.704629] | - |
| 2023 | OOS | candidate | brier | +0.246076 [+0.240050, +0.252424] | - |
| 2023 | OOS | served_base | brier | +0.244618 [+0.239197, +0.250198] | - |
| 2023 | OOS | model | brier | +0.249855 [+0.249219, +0.250468] | - |
| 2023 | OOS | market | brier | +0.247482 [+0.244928, +0.250007] | - |
| 2023 | OOS | elo | brier | +0.250983 [+0.246231, +0.255730] | - |
| 2023 | OOS | candidate | accuracy_points | +56.652361 [+48.936170, +63.404255] | - |
| 2023 | OOS | served_base | accuracy_points | +58.798283 [+52.340426, +64.830508] | - |
| 2023 | OOS | model | accuracy_points | +50.643777 [+45.991561, +55.701754] | - |
| 2023 | OOS | market | accuracy_points | +57.081545 [+51.489362, +62.917026] | - |
| 2023 | OOS | elo | accuracy_points | +49.356223 [+42.060086, +56.170213] | - |
| 2023 | IS_minus_OOS | candidate | rps | +0.463857 [-0.580701, +1.491563] | - |
| 2023 | IS_minus_OOS | candidate | log_loss | +0.001158 [-0.016141, +0.018323] | - |
| 2023 | IS_minus_OOS | candidate | brier | +0.000554 [-0.007994, +0.009042] | - |
| 2023 | IS_minus_OOS | candidate | accuracy_points | -2.652361 [-12.081701, +7.346620] | - |
| 2023 | IS_minus_OOS | served_base | rps | +0.441289 [-0.607901, +1.473628] | - |
| 2023 | IS_minus_OOS | served_base | log_loss | -0.003758 [-0.017523, +0.010663] | - |
| 2023 | IS_minus_OOS | served_base | brier | -0.001871 [-0.008695, +0.005276] | - |
| 2023 | IS_minus_OOS | served_base | accuracy_points | -7.798283 [-16.853028, +2.280536] | - |
| 2023 | IS_minus_OOS | model | rps | +0.488271 [-0.566147, +1.531516] | - |
| 2023 | IS_minus_OOS | model | log_loss | -0.000270 [-0.002970, +0.002370] | - |
| 2023 | IS_minus_OOS | model | brier | -0.000135 [-0.001484, +0.001185] | - |
| 2023 | IS_minus_OOS | model | accuracy_points | -2.143777 [-10.349741, +5.709175] | - |
| 2023 | IS_minus_OOS | market | rps | +0.465231 [-0.587339, +1.512320] | - |
| 2023 | IS_minus_OOS | market | log_loss | -0.003563 [-0.011631, +0.004221] | - |
| 2023 | IS_minus_OOS | market | brier | -0.001777 [-0.005802, +0.002107] | - |
| 2023 | IS_minus_OOS | market | accuracy_points | -5.581545 [-13.961240, +2.449446] | - |
| 2023 | IS_minus_OOS | elo | rps | +0.508861 [-0.561246, +1.555442] | - |
| 2023 | IS_minus_OOS | elo | log_loss | +0.005851 [-0.005329, +0.017351] | - |
| 2023 | IS_minus_OOS | elo | brier | +0.002947 [-0.002630, +0.008692] | - |
| 2023 | IS_minus_OOS | elo | accuracy_points | +5.643777 [-2.529592, +14.068112] | - |
| 2024 | IS | candidate | rps | +7.545380 [+6.682486, +8.509342] | - |
| 2024 | IS | served_base | rps | +7.550189 [+6.670683, +8.528899] | - |
| 2024 | IS | model | rps | +7.564941 [+6.653444, +8.581412] | - |
| 2024 | IS | market | rps | +7.544579 [+6.681369, +8.487342] | - |
| 2024 | IS | elo | rps | +7.600548 [+6.645067, +8.641633] | - |
| 2024 | IS | candidate | log_loss | +0.686133 [+0.665871, +0.704412] | - |
| 2024 | IS | served_base | log_loss | +0.686225 [+0.666159, +0.703789] | - |
| 2024 | IS | model | log_loss | +0.691932 [+0.680405, +0.703244] | - |
| 2024 | IS | market | log_loss | +0.689139 [+0.671597, +0.706870] | - |
| 2024 | IS | elo | log_loss | +0.694295 [+0.685431, +0.703468] | - |
| 2024 | IS | candidate | brier | +0.246655 [+0.237013, +0.255362] | - |
| 2024 | IS | served_base | brier | +0.246699 [+0.237026, +0.255178] | - |
| 2024 | IS | model | brier | +0.249386 [+0.243670, +0.254984] | - |
| 2024 | IS | market | brier | +0.248128 [+0.239716, +0.256634] | - |
| 2024 | IS | elo | brier | +0.250543 [+0.246086, +0.255131] | - |
| 2024 | IS | candidate | accuracy_points | +54.237288 [+49.160468, +59.310429] | - |
| 2024 | IS | served_base | accuracy_points | +54.963680 [+49.871449, +60.001142] | - |
| 2024 | IS | model | accuracy_points | +54.479419 [+46.786521, +62.060890] | - |
| 2024 | IS | market | accuracy_points | +53.268765 [+48.826013, +57.692308] | - |
| 2024 | IS | elo | accuracy_points | +50.605327 [+44.444444, +56.068071] | - |
| 2024 | OOS | candidate | rps | +7.158187 [+6.603250, +7.739720] | - |
| 2024 | OOS | served_base | rps | +7.146203 [+6.585662, +7.739813] | - |
| 2024 | OOS | model | rps | +7.161731 [+6.619548, +7.719431] | - |
| 2024 | OOS | market | rps | +7.187118 [+6.670368, +7.733522] | - |
| 2024 | OOS | elo | rps | +7.168642 [+6.633866, +7.724688] | - |
| 2024 | OOS | candidate | log_loss | +0.687899 [+0.669701, +0.707951] | - |
| 2024 | OOS | served_base | log_loss | +0.684245 [+0.665067, +0.705343] | - |
| 2024 | OOS | model | log_loss | +0.692575 [+0.680847, +0.704753] | - |
| 2024 | OOS | market | log_loss | +0.700831 [+0.685150, +0.715477] | - |
| 2024 | OOS | elo | log_loss | +0.698446 [+0.686808, +0.710871] | - |
| 2024 | OOS | candidate | brier | +0.246861 [+0.238139, +0.256487] | - |
| 2024 | OOS | served_base | brier | +0.245211 [+0.236071, +0.255283] | - |
| 2024 | OOS | model | brier | +0.249717 [+0.243887, +0.255771] | - |
| 2024 | OOS | market | brier | +0.253739 [+0.246180, +0.260758] | - |
| 2024 | OOS | elo | brier | +0.252594 [+0.246844, +0.258730] | - |
| 2024 | OOS | candidate | accuracy_points | +58.620690 [+52.789700, +64.317181] | - |
| 2024 | OOS | served_base | accuracy_points | +58.620690 [+52.380441, +65.126530] | - |
| 2024 | OOS | model | accuracy_points | +50.431034 [+42.666514, +58.050847] | - |
| 2024 | OOS | market | accuracy_points | +50.000000 [+44.493392, +54.878049] | - |
| 2024 | OOS | elo | accuracy_points | +50.000000 [+44.540282, +55.364807] | - |
| 2024 | IS_minus_OOS | candidate | rps | -0.387193 [-1.497294, +0.670334] | - |
| 2024 | IS_minus_OOS | candidate | log_loss | +0.001766 [-0.024867, +0.029992] | - |
| 2024 | IS_minus_OOS | candidate | brier | +0.000206 [-0.012460, +0.013716] | - |
| 2024 | IS_minus_OOS | candidate | accuracy_points | -4.383402 [-12.060360, +3.362378] | - |
| 2024 | IS_minus_OOS | served_base | rps | -0.403987 [-1.538620, +0.678263] | - |
| 2024 | IS_minus_OOS | served_base | log_loss | -0.001980 [-0.028587, +0.026954] | - |
| 2024 | IS_minus_OOS | served_base | brier | -0.001488 [-0.014332, +0.012366] | - |
| 2024 | IS_minus_OOS | served_base | accuracy_points | -3.657009 [-11.952534, +4.478121] | - |
| 2024 | IS_minus_OOS | model | rps | -0.403210 [-1.550362, +0.693075] | - |
| 2024 | IS_minus_OOS | model | log_loss | +0.000643 [-0.015726, +0.017378] | - |
| 2024 | IS_minus_OOS | model | brier | +0.000331 [-0.007809, +0.008655] | - |
| 2024 | IS_minus_OOS | model | accuracy_points | +4.048384 [-6.997997, +15.019757] | - |
| 2024 | IS_minus_OOS | market | rps | -0.357461 [-1.432847, +0.682952] | - |
| 2024 | IS_minus_OOS | market | log_loss | +0.011691 [-0.011703, +0.034159] | - |
| 2024 | IS_minus_OOS | market | brier | +0.005612 [-0.005616, +0.016401] | - |
| 2024 | IS_minus_OOS | market | accuracy_points | +3.268765 [-3.372126, +10.219940] | - |
| 2024 | IS_minus_OOS | elo | rps | -0.431906 [-1.598021, +0.700684] | - |
| 2024 | IS_minus_OOS | elo | log_loss | +0.004151 [-0.010765, +0.019376] | - |
| 2024 | IS_minus_OOS | elo | brier | +0.002051 [-0.005373, +0.009608] | - |
| 2024 | IS_minus_OOS | elo | accuracy_points | +0.605327 [-7.444057, +8.530826] | - |
| 2025 | IS | candidate | rps | +7.201553 [+6.445380, +8.111413] | - |
| 2025 | IS | served_base | rps | +7.204204 [+6.436157, +8.131554] | - |
| 2025 | IS | model | rps | +7.276338 [+6.531223, +8.192359] | - |
| 2025 | IS | market | rps | +7.214687 [+6.468424, +8.092569] | - |
| 2025 | IS | elo | rps | +7.282017 [+6.499811, +8.230442] | - |
| 2025 | IS | candidate | log_loss | +0.680451 [+0.666923, +0.694551] | - |
| 2025 | IS | served_base | log_loss | +0.680753 [+0.667103, +0.695010] | - |
| 2025 | IS | model | log_loss | +0.692633 [+0.683907, +0.700934] | - |
| 2025 | IS | market | log_loss | +0.684270 [+0.673362, +0.695196] | - |
| 2025 | IS | elo | log_loss | +0.690343 [+0.682990, +0.698603] | - |
| 2025 | IS | candidate | brier | +0.243785 [+0.237157, +0.250712] | - |
| 2025 | IS | served_base | brier | +0.243929 [+0.237203, +0.250947] | - |
| 2025 | IS | model | brier | +0.249727 [+0.245388, +0.253840] | - |
| 2025 | IS | market | brier | +0.245752 [+0.240479, +0.251103] | - |
| 2025 | IS | elo | brier | +0.248587 [+0.244943, +0.252686] | - |
| 2025 | IS | candidate | accuracy_points | +56.179775 [+51.575326, +60.949647] | - |
| 2025 | IS | served_base | accuracy_points | +56.019262 [+51.592230, +60.301107] | - |
| 2025 | IS | model | accuracy_points | +52.648475 [+48.107210, +57.613946] | - |
| 2025 | IS | market | accuracy_points | +54.253612 [+50.240751, +58.211440] | - |
| 2025 | IS | elo | accuracy_points | +54.574639 [+51.741274, +57.120259] | - |
| 2025 | OOS | candidate | rps | +7.001364 [+6.350987, +7.693988] | - |
| 2025 | OOS | served_base | rps | +6.996289 [+6.337979, +7.694987] | - |
| 2025 | OOS | model | rps | +7.052803 [+6.405821, +7.722405] | - |
| 2025 | OOS | market | rps | +6.959228 [+6.334699, +7.611632] | - |
| 2025 | OOS | elo | rps | +7.114401 [+6.473688, +7.766761] | - |
| 2025 | OOS | candidate | log_loss | +0.687029 [+0.667439, +0.705953] | - |
| 2025 | OOS | served_base | log_loss | +0.686204 [+0.666588, +0.704847] | - |
| 2025 | OOS | model | log_loss | +0.692837 [+0.686426, +0.699410] | - |
| 2025 | OOS | market | log_loss | +0.685973 [+0.669215, +0.702869] | - |
| 2025 | OOS | elo | log_loss | +0.697425 [+0.689397, +0.705651] | - |
| 2025 | OOS | candidate | brier | +0.247021 [+0.237551, +0.256170] | - |
| 2025 | OOS | served_base | brier | +0.246603 [+0.237129, +0.255586] | - |
| 2025 | OOS | model | brier | +0.249843 [+0.246648, +0.253121] | - |
| 2025 | OOS | market | brier | +0.246483 [+0.238397, +0.254655] | - |
| 2025 | OOS | elo | brier | +0.252129 [+0.248133, +0.256225] | - |
| 2025 | OOS | candidate | accuracy_points | +55.603448 [+50.431034, +60.759494] | - |
| 2025 | OOS | served_base | accuracy_points | +56.034483 [+51.489362, +60.593220] | - |
| 2025 | OOS | model | accuracy_points | +53.879310 [+48.458150, +59.545455] | - |
| 2025 | OOS | market | accuracy_points | +56.896552 [+51.754386, +61.276596] | - |
| 2025 | OOS | elo | accuracy_points | +50.000000 [+44.052863, +55.701754] | - |
| 2025 | IS_minus_OOS | candidate | rps | -0.200189 [-1.307629, +0.813728] | - |
| 2025 | IS_minus_OOS | candidate | log_loss | +0.006578 [-0.017279, +0.029627] | - |
| 2025 | IS_minus_OOS | candidate | brier | +0.003236 [-0.008380, +0.014411] | - |
| 2025 | IS_minus_OOS | candidate | accuracy_points | +0.576327 [-6.435064, +7.663427] | - |
| 2025 | IS_minus_OOS | served_base | rps | -0.207915 [-1.326650, +0.815223] | - |
| 2025 | IS_minus_OOS | served_base | log_loss | +0.005451 [-0.018320, +0.028397] | - |
| 2025 | IS_minus_OOS | served_base | brier | +0.002674 [-0.008907, +0.013799] | - |
| 2025 | IS_minus_OOS | served_base | accuracy_points | -0.015221 [-6.417886, +6.335067] | - |
| 2025 | IS_minus_OOS | model | rps | -0.223535 [-1.330433, +0.770828] | - |
| 2025 | IS_minus_OOS | model | log_loss | +0.000204 [-0.010446, +0.010922] | - |
| 2025 | IS_minus_OOS | model | brier | +0.000116 [-0.005184, +0.005445] | - |
| 2025 | IS_minus_OOS | model | accuracy_points | -1.230835 [-8.607329, +6.152630] | - |
| 2025 | IS_minus_OOS | market | rps | -0.255459 [-1.325326, +0.725725] | - |
| 2025 | IS_minus_OOS | market | log_loss | +0.001703 [-0.017993, +0.021387] | - |
| 2025 | IS_minus_OOS | market | brier | +0.000730 [-0.008785, +0.010198] | - |
| 2025 | IS_minus_OOS | market | accuracy_points | -2.642940 [-8.669533, +3.658350] | - |
| 2025 | IS_minus_OOS | elo | rps | -0.167616 [-1.306881, +0.843303] | - |
| 2025 | IS_minus_OOS | elo | log_loss | +0.007082 [-0.004289, +0.018530] | - |
| 2025 | IS_minus_OOS | elo | brier | +0.003542 [-0.002111, +0.009230] | - |
| 2025 | IS_minus_OOS | elo | accuracy_points | +4.574639 [-1.784777, +11.072162] | - |
| pooled | IS | candidate | rps | +7.269542 [+6.484358, +8.140348] | - |
| pooled | IS | served_base | rps | +7.274355 [+6.473815, +8.156890] | - |
| pooled | IS | model | rps | +7.317349 [+6.553909, +8.220355] | - |
| pooled | IS | market | rps | +7.277563 [+6.510851, +8.127732] | - |
| pooled | IS | elo | rps | +7.329882 [+6.520218, +8.261681] | - |
| pooled | IS | candidate | log_loss | +0.682942 [+0.667081, +0.694618] | - |
| pooled | IS | served_base | log_loss | +0.683444 [+0.667196, +0.695291] | - |
| pooled | IS | model | log_loss | +0.692479 [+0.683550, +0.698924] | - |
| pooled | IS | market | log_loss | +0.687093 [+0.674540, +0.697758] | - |
| pooled | IS | elo | log_loss | +0.691488 [+0.683602, +0.698129] | - |
| pooled | IS | candidate | brier | +0.245025 [+0.237334, +0.250709] | - |
| pooled | IS | served_base | brier | +0.245269 [+0.237366, +0.251021] | - |
| pooled | IS | model | brier | +0.249655 [+0.245221, +0.252848] | - |
| pooled | IS | market | brier | +0.247113 [+0.241032, +0.252269] | - |
| pooled | IS | elo | brier | +0.249151 [+0.245232, +0.252461] | - |
| pooled | IS | candidate | accuracy_points | +55.177994 [+51.377218, +60.062096] | - |
| pooled | IS | served_base | accuracy_points | +54.854369 [+50.987393, +59.628571] | - |
| pooled | IS | model | accuracy_points | +52.588997 [+48.722218, +58.490697] | - |
| pooled | IS | market | accuracy_points | +53.478964 [+49.938342, +57.703998] | - |
| pooled | IS | elo | accuracy_points | +53.317152 [+50.304359, +56.521826] | - |
| pooled | OOS | candidate | rps | +7.180249 [+6.756471, +7.632106] | - |
| pooled | OOS | served_base | rps | +7.170864 [+6.747411, +7.623714] | - |
| pooled | OOS | model | rps | +7.214126 [+6.796231, +7.669498] | - |
| pooled | OOS | market | rps | +7.179804 [+6.756201, +7.634858] | - |
| pooled | OOS | elo | rps | +7.239262 [+6.834460, +7.676329] | - |
| pooled | OOS | candidate | log_loss | +0.686731 [+0.676831, +0.697162] | - |
| pooled | OOS | served_base | log_loss | +0.684255 [+0.674507, +0.694639] | - |
| pooled | OOS | model | log_loss | +0.692756 [+0.688032, +0.697489] | - |
| pooled | OOS | market | log_loss | +0.691629 [+0.681484, +0.702826] | - |
| pooled | OOS | elo | log_loss | +0.696990 [+0.691146, +0.703181] | - |
| pooled | OOS | candidate | brier | +0.246652 [+0.241882, +0.251654] | - |
| pooled | OOS | served_base | brier | +0.245476 [+0.240744, +0.250483] | - |
| pooled | OOS | model | brier | +0.249805 [+0.247454, +0.252167] | - |
| pooled | OOS | market | brier | +0.249232 [+0.244340, +0.254709] | - |
| pooled | OOS | elo | brier | +0.251901 [+0.248997, +0.254964] | - |
| pooled | OOS | candidate | accuracy_points | +56.958393 [+53.010237, +60.703823] | - |
| pooled | OOS | served_base | accuracy_points | +57.819225 [+54.301738, +61.626495] | - |
| pooled | OOS | model | accuracy_points | +51.649928 [+47.564428, +55.680264] | - |
| pooled | OOS | market | accuracy_points | +54.662841 [+49.566443, +59.169202] | - |
| pooled | OOS | elo | accuracy_points | +49.784792 [+46.121282, +53.257986] | - |
| pooled | IS_minus_OOS | candidate | rps | -0.089292 [-1.059969, +0.823265] | - |
| pooled | IS_minus_OOS | candidate | log_loss | +0.003789 [-0.011221, +0.022730] | - |
| pooled | IS_minus_OOS | candidate | brier | +0.001627 [-0.005611, +0.010799] | - |
| pooled | IS_minus_OOS | candidate | accuracy_points | -1.780400 [-7.096614, +4.339989] | - |
| pooled | IS_minus_OOS | served_base | rps | -0.103491 [-1.086072, +0.818711] | - |
| pooled | IS_minus_OOS | served_base | log_loss | +0.000811 [-0.014335, +0.020012] | - |
| pooled | IS_minus_OOS | served_base | brier | +0.000207 [-0.007203, +0.009553] | - |
| pooled | IS_minus_OOS | served_base | accuracy_points | -2.964856 [-8.377610, +2.994101] | - |
| pooled | IS_minus_OOS | model | rps | -0.103224 [-1.100415, +0.794961] | - |
| pooled | IS_minus_OOS | model | log_loss | +0.000278 [-0.007773, +0.010209] | - |
| pooled | IS_minus_OOS | model | brier | +0.000150 [-0.003846, +0.005089] | - |
| pooled | IS_minus_OOS | model | accuracy_points | +0.939068 [-4.980599, +8.025507] | - |
| pooled | IS_minus_OOS | market | rps | -0.097759 [-1.040598, +0.794087] | - |
| pooled | IS_minus_OOS | market | log_loss | +0.004536 [-0.010049, +0.021060] | - |
| pooled | IS_minus_OOS | market | brier | +0.002119 [-0.004949, +0.010157] | - |
| pooled | IS_minus_OOS | market | accuracy_points | -1.183876 [-6.720158, +5.445971] | - |
| pooled | IS_minus_OOS | elo | rps | -0.090620 [-1.104296, +0.847377] | - |
| pooled | IS_minus_OOS | elo | log_loss | +0.005502 [-0.003162, +0.015355] | - |
| pooled | IS_minus_OOS | elo | brier | +0.002749 [-0.001573, +0.007636] | - |
| pooled | IS_minus_OOS | elo | accuracy_points | +3.532360 [-0.938582, +8.405713] | - |

### Candidate paired improvements

| Panel | Stage | Arm/comparator | Metric | Estimate [95% interval] | probability_positive |
| --- | --- | --- | --- | --- | --- |
| 2023 | IS | served_base | rps | +0.011497 [-0.013164, +0.033666] | 0.8243 |
| 2023 | IS | model | rps | +0.022252 [-0.041145, +0.088221] | 0.7527 |
| 2023 | IS | market | rps | +0.010270 [-0.061365, +0.089334] | 0.6052 |
| 2023 | IS | elo | rps | +0.008793 [-0.071588, +0.099310] | 0.5796 |
| 2023 | IS | served_base | log_loss | +0.001968 [-0.004877, +0.007433] | 0.7396 |
| 2023 | IS | model | log_loss | +0.009014 [-0.002749, +0.019958] | 0.9354 |
| 2023 | IS | market | log_loss | +0.007548 [-0.006328, +0.020597] | 0.8606 |
| 2023 | IS | elo | log_loss | +0.005144 [-0.009868, +0.018449] | 0.7714 |
| 2023 | IS | served_base | brier | +0.000967 [-0.002431, +0.003675] | 0.7380 |
| 2023 | IS | model | brier | +0.004467 [-0.001352, +0.009899] | 0.9359 |
| 2023 | IS | market | brier | +0.003736 [-0.003151, +0.010207] | 0.8601 |
| 2023 | IS | elo | brier | +0.002513 [-0.004940, +0.009126] | 0.7672 |
| 2023 | IS | served_base | accuracy_points | +3.000000 [-2.051282, +8.095238] | 0.8693 |
| 2023 | IS | model | accuracy_points | +5.500000 [-4.523184, +15.094340] | 0.8656 |
| 2023 | IS | market | accuracy_points | +2.500000 [-4.950495, +10.256410] | 0.7372 |
| 2023 | IS | elo | accuracy_points | -1.000000 [-9.743902, +7.614213] | 0.4262 |
| 2023 | OOS | served_base | rps | -0.011071 [-0.020633, -0.000296] | 0.0228 |
| 2023 | OOS | model | rps | +0.046666 [-0.019712, +0.112466] | 0.9119 |
| 2023 | OOS | market | rps | +0.011644 [-0.060448, +0.082989] | 0.6194 |
| 2023 | OOS | elo | rps | +0.053797 [-0.021643, +0.128788] | 0.9172 |
| 2023 | OOS | served_base | log_loss | -0.002949 [-0.006421, +0.000559] | 0.0531 |
| 2023 | OOS | model | log_loss | +0.007586 [-0.005574, +0.020112] | 0.8736 |
| 2023 | OOS | market | log_loss | +0.002827 [-0.012713, +0.018197] | 0.6467 |
| 2023 | OOS | elo | log_loss | +0.009837 [-0.005126, +0.024793] | 0.9038 |
| 2023 | OOS | served_base | brier | -0.001459 [-0.003178, +0.000267] | 0.0514 |
| 2023 | OOS | model | brier | +0.003778 [-0.002740, +0.009961] | 0.8758 |
| 2023 | OOS | market | brier | +0.001405 [-0.006291, +0.009010] | 0.6481 |
| 2023 | OOS | elo | brier | +0.004906 [-0.002496, +0.012312] | 0.9051 |
| 2023 | OOS | served_base | accuracy_points | -2.145923 [-7.522124, +3.265306] | 0.2157 |
| 2023 | OOS | model | accuracy_points | +6.008584 [-3.508772, +14.782609] | 0.8950 |
| 2023 | OOS | market | accuracy_points | -0.429185 [-12.017167, +10.000000] | 0.4834 |
| 2023 | OOS | elo | accuracy_points | +7.296137 [-2.564378, +17.334322] | 0.9228 |
| 2023 | IS_minus_OOS | served_base | rps | +0.022568 [-0.004340, +0.047181] | 0.9503 |
| 2023 | IS_minus_OOS | served_base | log_loss | +0.004916 [-0.002640, +0.011505] | 0.9052 |
| 2023 | IS_minus_OOS | served_base | brier | +0.002425 [-0.001311, +0.005693] | 0.9044 |
| 2023 | IS_minus_OOS | served_base | accuracy_points | +5.145923 [-2.298789, +12.429211] | 0.9080 |
| 2023 | IS_minus_OOS | model | rps | -0.024414 [-0.115602, +0.069102] | 0.3171 |
| 2023 | IS_minus_OOS | model | log_loss | +0.001428 [-0.016138, +0.018763] | 0.5662 |
| 2023 | IS_minus_OOS | model | brier | +0.000689 [-0.007990, +0.009273] | 0.5637 |
| 2023 | IS_minus_OOS | model | accuracy_points | -0.508584 [-13.820535, +12.848905] | 0.4722 |
| 2023 | IS_minus_OOS | market | rps | -0.001374 [-0.104137, +0.106526] | 0.4932 |
| 2023 | IS_minus_OOS | market | log_loss | +0.004721 [-0.016306, +0.025064] | 0.6723 |
| 2023 | IS_minus_OOS | market | brier | +0.002331 [-0.008094, +0.012414] | 0.6717 |
| 2023 | IS_minus_OOS | market | accuracy_points | +2.929185 [-9.897509, +16.903636] | 0.6559 |
| 2023 | IS_minus_OOS | elo | rps | -0.045004 [-0.153158, +0.071688] | 0.2207 |
| 2023 | IS_minus_OOS | elo | log_loss | -0.004693 [-0.025919, +0.015449] | 0.3289 |
| 2023 | IS_minus_OOS | elo | brier | -0.002393 [-0.012901, +0.007671] | 0.3241 |
| 2023 | IS_minus_OOS | elo | accuracy_points | -8.296137 [-21.593631, +4.959523] | 0.1119 |
| 2024 | IS | served_base | rps | +0.004809 [-0.034467, +0.037781] | 0.6127 |
| 2024 | IS | model | rps | +0.019560 [-0.096949, +0.153310] | 0.6073 |
| 2024 | IS | market | rps | -0.000802 [-0.141073, +0.135333] | 0.5034 |
| 2024 | IS | elo | rps | +0.055167 [-0.096979, +0.219015] | 0.7482 |
| 2024 | IS | served_base | log_loss | +0.000092 [-0.004335, +0.004221] | 0.5200 |
| 2024 | IS | model | log_loss | +0.005799 [-0.007863, +0.020594] | 0.7914 |
| 2024 | IS | market | log_loss | +0.003006 [-0.018558, +0.026187] | 0.5993 |
| 2024 | IS | elo | log_loss | +0.008162 [-0.014443, +0.030868] | 0.7494 |
| 2024 | IS | served_base | brier | +0.000045 [-0.002070, +0.002039] | 0.5190 |
| 2024 | IS | model | brier | +0.002731 [-0.003813, +0.009847] | 0.7892 |
| 2024 | IS | market | brier | +0.001473 [-0.008985, +0.012683] | 0.6015 |
| 2024 | IS | elo | brier | +0.003888 [-0.007128, +0.014853] | 0.7482 |
| 2024 | IS | served_base | accuracy_points | -0.726392 [-2.619048, +1.033592] | 0.2180 |
| 2024 | IS | model | accuracy_points | -0.242131 [-8.177676, +7.635908] | 0.4834 |
| 2024 | IS | market | accuracy_points | +0.968523 [-6.356968, +7.655530] | 0.6323 |
| 2024 | IS | elo | accuracy_points | +3.631961 [-4.314721, +11.576544] | 0.8159 |
| 2024 | OOS | served_base | rps | -0.011985 [-0.042472, +0.018391] | 0.2196 |
| 2024 | OOS | model | rps | +0.003544 [-0.106199, +0.100720] | 0.5441 |
| 2024 | OOS | market | rps | +0.028930 [-0.110092, +0.162117] | 0.6571 |
| 2024 | OOS | elo | rps | +0.010454 [-0.141416, +0.152553] | 0.5608 |
| 2024 | OOS | served_base | log_loss | -0.003654 [-0.007454, +0.000056] | 0.0265 |
| 2024 | OOS | model | log_loss | +0.004675 [-0.012641, +0.020117] | 0.7185 |
| 2024 | OOS | market | log_loss | +0.012932 [-0.009444, +0.037088] | 0.8605 |
| 2024 | OOS | elo | log_loss | +0.010547 [-0.011930, +0.031534] | 0.8242 |
| 2024 | OOS | served_base | brier | -0.001649 [-0.003500, +0.000137] | 0.0363 |
| 2024 | OOS | model | brier | +0.002857 [-0.005496, +0.010286] | 0.7625 |
| 2024 | OOS | market | brier | +0.006879 [-0.004020, +0.018592] | 0.8838 |
| 2024 | OOS | elo | brier | +0.005733 [-0.005220, +0.015866] | 0.8543 |
| 2024 | OOS | served_base | accuracy_points | +0.000000 [-2.597403, +2.690886] | 0.4933 |
| 2024 | OOS | model | accuracy_points | +8.189655 [+1.709402, +14.754444] | 0.9926 |
| 2024 | OOS | market | accuracy_points | +8.620690 [+0.829790, +17.316017] | 0.9832 |
| 2024 | OOS | elo | accuracy_points | +8.620690 [+3.138663, +14.102564] | 0.9993 |
| 2024 | IS_minus_OOS | served_base | rps | +0.016794 [-0.031330, +0.063019] | 0.7568 |
| 2024 | IS_minus_OOS | served_base | log_loss | +0.003746 [-0.001903, +0.009288] | 0.8984 |
| 2024 | IS_minus_OOS | served_base | brier | +0.001694 [-0.001042, +0.004377] | 0.8829 |
| 2024 | IS_minus_OOS | served_base | accuracy_points | -0.726392 [-4.081417, +2.460287] | 0.3328 |
| 2024 | IS_minus_OOS | model | rps | +0.016017 [-0.135149, +0.186555] | 0.5661 |
| 2024 | IS_minus_OOS | model | log_loss | +0.001123 [-0.019201, +0.023605] | 0.5278 |
| 2024 | IS_minus_OOS | model | brier | -0.000125 [-0.009870, +0.010666] | 0.4814 |
| 2024 | IS_minus_OOS | model | accuracy_points | -8.431786 [-18.626913, +1.791180] | 0.0555 |
| 2024 | IS_minus_OOS | market | rps | -0.029732 [-0.220478, +0.166138] | 0.3863 |
| 2024 | IS_minus_OOS | market | log_loss | -0.009925 [-0.042194, +0.022511] | 0.2741 |
| 2024 | IS_minus_OOS | market | brier | -0.005406 [-0.021111, +0.010202] | 0.2510 |
| 2024 | IS_minus_OOS | market | accuracy_points | -7.652167 [-18.853345, +2.763907] | 0.0779 |
| 2024 | IS_minus_OOS | elo | rps | +0.044713 [-0.167746, +0.266245] | 0.6577 |
| 2024 | IS_minus_OOS | elo | log_loss | -0.002385 [-0.033901, +0.029782] | 0.4403 |
| 2024 | IS_minus_OOS | elo | brier | -0.001845 [-0.017086, +0.013646] | 0.4068 |
| 2024 | IS_minus_OOS | elo | accuracy_points | -4.988728 [-14.666802, +4.747331] | 0.1517 |
| 2025 | IS | served_base | rps | +0.002651 [-0.020274, +0.027936] | 0.5656 |
| 2025 | IS | model | rps | +0.074785 [-0.017704, +0.167304] | 0.9371 |
| 2025 | IS | market | rps | +0.013134 [-0.077602, +0.090648] | 0.6356 |
| 2025 | IS | elo | rps | +0.080464 [-0.019537, +0.192596] | 0.9408 |
| 2025 | IS | served_base | log_loss | +0.000302 [-0.001963, +0.002787] | 0.5809 |
| 2025 | IS | model | log_loss | +0.012182 [+0.001021, +0.023774] | 0.9832 |
| 2025 | IS | market | log_loss | +0.003819 [-0.009738, +0.016105] | 0.7183 |
| 2025 | IS | elo | log_loss | +0.009891 [-0.005498, +0.025154] | 0.8962 |
| 2025 | IS | served_base | brier | +0.000144 [-0.000958, +0.001355] | 0.5806 |
| 2025 | IS | model | brier | +0.005942 [+0.000445, +0.011654] | 0.9820 |
| 2025 | IS | market | brier | +0.001967 [-0.004676, +0.007988] | 0.7291 |
| 2025 | IS | elo | brier | +0.004802 [-0.002756, +0.012251] | 0.8943 |
| 2025 | IS | served_base | accuracy_points | +0.160514 [-1.754526, +2.037698] | 0.5619 |
| 2025 | IS | model | accuracy_points | +3.531300 [-1.618189, +8.746122] | 0.9118 |
| 2025 | IS | market | accuracy_points | +1.926164 [-2.623552, +6.549589] | 0.8015 |
| 2025 | IS | elo | accuracy_points | +1.605136 [-2.995258, +6.139115] | 0.7475 |
| 2025 | OOS | served_base | rps | -0.005075 [-0.035246, +0.024653] | 0.3773 |
| 2025 | OOS | model | rps | +0.051439 [-0.042660, +0.156762] | 0.8409 |
| 2025 | OOS | market | rps | -0.042136 [-0.171009, +0.078109] | 0.2596 |
| 2025 | OOS | elo | rps | +0.113037 [-0.035455, +0.258560] | 0.9314 |
| 2025 | OOS | served_base | log_loss | -0.000825 [-0.003689, +0.001856] | 0.2922 |
| 2025 | OOS | model | log_loss | +0.005807 [-0.010820, +0.025014] | 0.7187 |
| 2025 | OOS | market | log_loss | -0.001056 [-0.020882, +0.019454] | 0.4527 |
| 2025 | OOS | elo | log_loss | +0.010396 [-0.012092, +0.033026] | 0.8088 |
| 2025 | OOS | served_base | brier | -0.000417 [-0.001815, +0.000890] | 0.2836 |
| 2025 | OOS | model | brier | +0.002822 [-0.005171, +0.012048] | 0.7197 |
| 2025 | OOS | market | brier | -0.000538 [-0.010098, +0.009370] | 0.4480 |
| 2025 | OOS | elo | brier | +0.005109 [-0.005819, +0.016079] | 0.8125 |
| 2025 | OOS | served_base | accuracy_points | -0.431034 [-2.212389, +1.339286] | 0.3344 |
| 2025 | OOS | model | accuracy_points | +1.724138 [-3.524229, +6.694561] | 0.7426 |
| 2025 | OOS | market | accuracy_points | -1.293103 [-7.826941, +5.106383] | 0.3581 |
| 2025 | OOS | elo | accuracy_points | +5.603448 [-3.508772, +14.285714] | 0.8874 |
| 2025 | IS_minus_OOS | served_base | rps | +0.007726 [-0.030259, +0.046337] | 0.6415 |
| 2025 | IS_minus_OOS | served_base | log_loss | +0.001126 [-0.002391, +0.004900] | 0.7142 |
| 2025 | IS_minus_OOS | served_base | brier | +0.000562 [-0.001146, +0.002405] | 0.7206 |
| 2025 | IS_minus_OOS | served_base | accuracy_points | +0.591548 [-2.093654, +3.267838] | 0.6567 |
| 2025 | IS_minus_OOS | model | rps | +0.023346 [-0.116005, +0.156757] | 0.6317 |
| 2025 | IS_minus_OOS | model | log_loss | +0.006374 [-0.015662, +0.026702] | 0.7277 |
| 2025 | IS_minus_OOS | model | brier | +0.003120 [-0.007576, +0.013012] | 0.7305 |
| 2025 | IS_minus_OOS | model | accuracy_points | +1.807162 [-5.368297, +9.076924] | 0.6870 |
| 2025 | IS_minus_OOS | market | rps | +0.055270 [-0.092899, +0.205350] | 0.7667 |
| 2025 | IS_minus_OOS | market | log_loss | +0.004875 [-0.019218, +0.028387] | 0.6558 |
| 2025 | IS_minus_OOS | market | brier | +0.002505 [-0.009210, +0.013916] | 0.6638 |
| 2025 | IS_minus_OOS | market | accuracy_points | +3.219267 [-4.632431, +11.256893] | 0.7796 |
| 2025 | IS_minus_OOS | elo | rps | -0.032573 [-0.207294, +0.152440] | 0.3555 |
| 2025 | IS_minus_OOS | elo | log_loss | -0.000504 [-0.028054, +0.026767] | 0.4901 |
| 2025 | IS_minus_OOS | elo | brier | -0.000306 [-0.013692, +0.013019] | 0.4855 |
| 2025 | IS_minus_OOS | elo | accuracy_points | -3.998312 [-13.973610, +6.222470] | 0.2177 |
| pooled | IS | served_base | rps | +0.004813 [-0.017395, +0.026996] | 0.6437 |
| pooled | IS | model | rps | +0.047808 [-0.031732, +0.162281] | 0.8712 |
| pooled | IS | market | rps | +0.008022 [-0.089160, +0.096964] | 0.5887 |
| pooled | IS | elo | rps | +0.060341 [-0.037597, +0.187288] | 0.8768 |
| pooled | IS | served_base | log_loss | +0.000501 [-0.001293, +0.002928] | 0.6366 |
| pooled | IS | model | log_loss | +0.009536 [-0.000498, +0.022528] | 0.9684 |
| pooled | IS | market | log_loss | +0.004151 [-0.010037, +0.019059] | 0.7258 |
| pooled | IS | elo | log_loss | +0.008545 [-0.005590, +0.025794] | 0.8744 |
| pooled | IS | served_base | brier | +0.000244 [-0.000631, +0.001409] | 0.6396 |
| pooled | IS | model | brier | +0.004631 [-0.000333, +0.011025] | 0.9665 |
| pooled | IS | market | brier | +0.002088 [-0.004854, +0.009314] | 0.7320 |
| pooled | IS | elo | brier | +0.004126 [-0.002782, +0.012505] | 0.8726 |
| pooled | IS | served_base | accuracy_points | +0.323625 [-0.669248, +1.384768] | 0.7586 |
| pooled | IS | model | accuracy_points | +2.588997 [-3.095169, +8.308637] | 0.8317 |
| pooled | IS | market | accuracy_points | +1.699029 [-3.395354, +6.405091] | 0.7557 |
| pooled | IS | elo | accuracy_points | +1.860841 [-3.359027, +7.279788] | 0.7684 |
| pooled | OOS | served_base | rps | -0.009385 [-0.024230, +0.005969] | 0.1048 |
| pooled | OOS | model | rps | +0.033876 [-0.026532, +0.090172] | 0.8729 |
| pooled | OOS | market | rps | -0.000445 [-0.078567, +0.072106] | 0.5112 |
| pooled | OOS | elo | rps | +0.059013 [-0.028055, +0.148832] | 0.9135 |
| pooled | OOS | served_base | log_loss | -0.002476 [-0.004876, -0.000195] | 0.0157 |
| pooled | OOS | model | log_loss | +0.006025 [-0.003374, +0.015102] | 0.9033 |
| pooled | OOS | market | log_loss | +0.004898 [-0.008037, +0.018931] | 0.7634 |
| pooled | OOS | elo | log_loss | +0.010259 [-0.001501, +0.021998] | 0.9553 |
| pooled | OOS | served_base | brier | -0.001176 [-0.002330, -0.000083] | 0.0160 |
| pooled | OOS | model | brier | +0.003153 [-0.001387, +0.007550] | 0.9188 |
| pooled | OOS | market | brier | +0.002580 [-0.003856, +0.009648] | 0.7747 |
| pooled | OOS | elo | brier | +0.005249 [-0.000522, +0.010968] | 0.9623 |
| pooled | OOS | served_base | accuracy_points | -0.860832 [-3.551136, +1.176514] | 0.2295 |
| pooled | OOS | model | accuracy_points | +5.308465 [+0.279711, +10.466761] | 0.9793 |
| pooled | OOS | market | accuracy_points | +2.295552 [-4.762077, +9.640288] | 0.7283 |
| pooled | OOS | elo | accuracy_points | +7.173601 [+2.002790, +12.057043] | 0.9956 |
| pooled | IS_minus_OOS | served_base | rps | +0.014199 [-0.012197, +0.040441] | 0.8364 |
| pooled | IS_minus_OOS | served_base | log_loss | +0.002978 [-0.000060, +0.006307] | 0.9721 |
| pooled | IS_minus_OOS | served_base | brier | +0.001420 [-0.000028, +0.003021] | 0.9723 |
| pooled | IS_minus_OOS | served_base | accuracy_points | +1.184457 [-1.141664, +4.023672] | 0.8309 |
| pooled | IS_minus_OOS | model | rps | +0.013931 [-0.085339, +0.140903] | 0.6220 |
| pooled | IS_minus_OOS | model | log_loss | +0.003511 [-0.010006, +0.019614] | 0.7021 |
| pooled | IS_minus_OOS | model | brier | +0.001477 [-0.005116, +0.009267] | 0.6783 |
| pooled | IS_minus_OOS | model | accuracy_points | -2.719468 [-10.391393, +5.054712] | 0.2493 |
| pooled | IS_minus_OOS | market | rps | +0.008467 [-0.112006, +0.126502] | 0.5697 |
| pooled | IS_minus_OOS | market | log_loss | -0.000747 [-0.020424, +0.018711] | 0.4853 |
| pooled | IS_minus_OOS | market | brier | -0.000492 [-0.010203, +0.009135] | 0.4761 |
| pooled | IS_minus_OOS | market | accuracy_points | -0.596523 [-9.357342, +7.943401] | 0.4483 |
| pooled | IS_minus_OOS | elo | rps | +0.001328 [-0.128211, +0.151334] | 0.5161 |
| pooled | IS_minus_OOS | elo | log_loss | -0.001714 [-0.019915, +0.018588] | 0.4433 |
| pooled | IS_minus_OOS | elo | brier | -0.001122 [-0.010052, +0.008667] | 0.4195 |
| pooled | IS_minus_OOS | elo | accuracy_points | -5.312760 [-12.479585, +2.066394] | 0.0788 |

## Effective coefficients and stability

**Measured:** effective Jensen coefficients for 2023/2024/2025 are +1.149124, -1.027235, -0.736015. The sign changes after the first fold; there is no stable positive weight.

**Measured:** effective coefficients incorporate the tuned multiplier and calibration intercept. Unscaled coefficients, standardizers and stage sizes remain in scratch. Signs and variation describe three folds, without selecting one.

| Outer | Arm | Intercept | Model logit | Flags | Move | Move available | Jensen gap | Market logit | Elo gap | Opener |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | candidate | -0.020614 | -0.055274 | +0.250498 | +0.024138 | +0.000000 | +1.149124 | - | - | - |
| 2023 | served_base | -0.009912 | -0.058141 | +0.230083 | +0.027454 | +0.000000 | - | - | - | - |
| 2023 | model | +0.010406 | -0.072366 | - | - | - | - | - | - | - |
| 2023 | market | +0.016738 | - | - | - | - | - | +0.394844 | - | - |
| 2023 | elo | +0.075076 | - | - | - | - | - | - | +0.001946 | -0.036657 |
| 2024 | candidate | +0.069699 | +0.588950 | +0.396613 | +0.095790 | +0.000000 | -1.027235 | - | - | - |
| 2024 | served_base | +0.065726 | +0.581989 | +0.401086 | +0.093234 | +0.000000 | - | - | - | - |
| 2024 | model | +0.098282 | +0.544766 | - | - | - | - | - | - | - |
| 2024 | market | +0.075293 | - | - | - | - | - | +1.800838 | - | - |
| 2024 | elo | +0.149552 | - | - | - | - | - | - | +0.002779 | -0.039015 |
| 2025 | candidate | -0.069877 | +0.401217 | +0.321591 | +0.099031 | +0.000000 | -0.736015 | - | - | - |
| 2025 | served_base | -0.073033 | +0.394291 | +0.322438 | +0.097451 | +0.000000 | - | - | - | - |
| 2025 | model | -0.040790 | +0.409617 | - | - | - | - | - | - | - |
| 2025 | market | -0.067241 | - | - | - | - | - | +1.377126 | - | - |
| 2025 | elo | +0.006925 | - | - | - | - | - | - | +0.000616 | -0.030606 |

## Five equal-width reliability bands

**Measured:** pooled OOS nonpush games; empty cells remain in the look count.

| Arm | Home-probability band | Games | Mean p | Observed home cover |
| --- | --- | --- | --- | --- |
| candidate | 0.0-0.2 | 0 | - | - |
| candidate | 0.2-0.4 | 59 | 0.362755 | 0.406780 |
| candidate | 0.4-0.6 | 543 | 0.503099 | 0.497238 |
| candidate | 0.6-0.8 | 94 | 0.644595 | 0.648936 |
| candidate | 0.8-1.0 | 1 | 0.834959 | 0.000000 |
| served_base | 0.0-0.2 | 0 | - | - |
| served_base | 0.2-0.4 | 60 | 0.367221 | 0.416667 |
| served_base | 0.4-0.6 | 547 | 0.504043 | 0.499086 |
| served_base | 0.6-0.8 | 89 | 0.645373 | 0.640449 |
| served_base | 0.8-1.0 | 1 | 0.803989 | 0.000000 |
| model | 0.0-0.2 | 0 | - | - |
| model | 0.2-0.4 | 0 | - | - |
| model | 0.4-0.6 | 693 | 0.507354 | 0.509380 |
| model | 0.6-0.8 | 4 | 0.607113 | 0.500000 |
| model | 0.8-1.0 | 0 | - | - |
| market | 0.0-0.2 | 0 | - | - |
| market | 0.2-0.4 | 18 | 0.336845 | 0.444444 |
| market | 0.4-0.6 | 635 | 0.505999 | 0.508661 |
| market | 0.6-0.8 | 44 | 0.639445 | 0.545455 |
| market | 0.8-1.0 | 0 | - | - |
| elo | 0.0-0.2 | 0 | - | - |
| elo | 0.2-0.4 | 3 | 0.372706 | 0.666667 |
| elo | 0.4-0.6 | 688 | 0.505078 | 0.507267 |
| elo | 0.6-0.8 | 6 | 0.622189 | 0.666667 |
| elo | 0.8-1.0 | 0 | - | - |

## Interpretation and handoff

**Measured:** 505 declared looks: 432 metric cells, 32 fit/nuisance summaries, 16 decisive records and 25 reliability bands. Five arms plus shared-prior, prior-variance and common-multiplier summaries account for B=8. Per-fold prior and variance parameters are in construction diagnostics; pooled summaries are descriptive, not new fits.
**Inferred:** provisional endpoint records await the orchestrator. Primary RPS and accuracy remain unresolved_below_power. Log-loss and Brier improvement intervals are wholly negative; the proposed refuted_mechanism records apply only to this fixed extension on those metrics, with closing ground wrong_sign_resolved under AGENTS.md research rules. The RPS mixture mechanism remains open. Intervals are unadjusted across the declared 505 looks; no family-wide significance or promotion is claimed. No power-matched positive control was run.
Prediction rows, normalized calibrated PMFs, exclusions, provenance, coefficients, all estimates and the protocol snapshot are under `tests/scratch/codex/lead84_unit2/`. Proposed bash record commands are in the lane, not executed.
