# LEAD-88 unit 3: isolate the total-news adjustment

**Measured:** one LOSO replay, two declared candidate-versus-served looks.
Command: .tools/uv.exe run --no-sync python scripts/lead88_unit3.py with local UV_CACHE_DIR.
The frozen declaration below preceded outcome access; full declaration and hashes are in scratch.

## Closer, worse and tied

**Measured:** exact two-sided null is 0.5 among games with unequal absolute errors.
Ties stay in MAE and the unconditional closer share; exact intervals are Clopper-Pearson 95%.
| Population | Closer | Worse | Tied | Decisive share [95% CI] | Unconditional closer share [95% CI] | Exact p |
| --- | --- | --- | --- | --- | --- | --- |
| last | 9 | 4 | 88 | 0.692308 [0.385738, 0.909080] | 0.089109 [0.041559, 0.162426] | 0.266846 |
| all | 176 | 157 | 1000 | 0.528529 [0.473363, 0.583182] | 0.132033 [0.114312, 0.151394] | 0.323944 |

## MAE and generalization

**Measured:** positive improvement means a smaller total-score error than the served guess.
Intervals use 10,000 paired season-stratified week-block draws, seed 88; probability_positive gives ties half weight.
| Population | Games | Weeks | Served MAE [95% CI] | Candidate OOS MAE [95% CI] | MAE improvement [95% CI] | probability_positive |
| --- | --- | --- | --- | --- | --- | --- |
| last | 101 | 101 | 10.405941 [8.930693, 12.009901] | 10.306931 [8.821782, 11.911139] | 0.099010 [-0.019802, 0.217822] | 0.952550 |
| all | 1333 | 107 | 10.368342 [10.029432, 10.704374] | 10.315079 [9.984193, 10.646618] | 0.053263 [0.009901, 0.097524] | 0.992300 |

| Population | Optimistic IS MAE [95% CI] | IS improvement [95% CI] | OOS improvement | OOS minus IS improvement [95% CI] |
| --- | --- | --- | --- | --- |
| last | 10.306931 [8.821535, 11.920792] | 0.099010 [-0.039604, 0.247525] | 0.099010 | 0.000000 [-0.079208, 0.069307] |
| all | 10.293323 [9.961374, 10.624454] | 0.075019 [0.026219, 0.126583] | 0.053263 | -0.021755 [-0.048013, 0.003779] |

**Inferred:** these measurements do not establish a serving change or settle the mechanism.
Proposed classification is unresolved_below_power, pending serial orchestrator recording.
AGENTS.md permits closure only for a refuted mechanism or a positive control with sufficient power; neither is established here.

## Fold coefficients and season stability

**Measured:** full-data optimistic coefficient b=0.800000000; all six LOSO slopes exactly reproduce Unit 2.
| Held season | Train | Held | b | Last improvement [95% CI] | Last probability_positive | All improvement [95% CI] | All probability_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 1136 | 197 | 0.666667 | -0.230769 [-0.615385, 0.000000] | 0.058950 | 0.203046 [0.091891, 0.320197] | 0.999400 |
| 2021 | 1117 | 216 | 0.666667 | 0.187500 [0.000000, 0.500000] | 0.941400 | 0.037037 [-0.033658, 0.109093] | 0.839150 |
| 2022 | 1118 | 215 | 0.800000 | 0.222222 [0.000000, 0.555556] | 0.943150 | 0.060465 [-0.050459, 0.182222] | 0.848150 |
| 2023 | 1099 | 234 | 0.500000 | 0.222222 [0.000000, 0.555556] | 0.938050 | 0.055556 [0.000000, 0.113646] | 0.978550 |
| 2024 | 1095 | 238 | 1.000000 | 0.000000 [-0.333333, 0.333333] | 0.503350 | -0.008403 [-0.117910, 0.100000] | 0.443800 |
| 2025 | 1100 | 233 | 1.000000 | 0.111111 [0.000000, 0.333333] | 0.819350 | -0.004292 [-0.156128, 0.144105] | 0.481700 |

**Measured:** frozen base-probability coefficients below are reused, never refitted or changed by total news.
| Held season | composition_flag_sum | intercept | market_move_available | market_move_toward_home | model_logit |
| --- | --- | --- | --- | --- | --- |
| 2020 | 0.244789 | -0.045251 | 0.028113 | 0.220730 | 0.341246 |
| 2021 | 0.271867 | -0.062332 | 0.044183 | 0.223716 | 0.177916 |
| 2022 | 0.238476 | -0.055698 | 0.039674 | 0.222903 | 0.249647 |
| 2023 | 0.246114 | -0.038064 | 0.002286 | 0.230272 | 0.341480 |
| 2024 | 0.262262 | -0.056178 | 0.039146 | 0.246066 | 0.256116 |
| 2025 | 0.295997 | -0.054099 | 0.047462 | 0.188739 | 0.292328 |

## Mapping checks and coverage

**Measured:** 1343 paired rows retained; 1333 scored, 101 last games; 10 production-declined baselines remain unscored.
All 33 restored four-term rows remain. Zero coefficient reproduced both served team scores on 1333 games.
Of 255 zero-move games, 0 changed; 0 last games changed at zero move.
The centre crossed a cell on 612 games; 0 same-cell guesses changed; 0 adjusted selectors declined and retained the served score.
Overall 340 total guesses changed, including 13 last games. Fixed-side and finite-score guards passed.
**Read:** src/nfl_ats/score_lattice.py:182 uses floor coordinates for team-score cells; scripts/lead88_unit2.py:222 defines the original served continuous total.
**Inferred:** the declared cell gate is a fixed conservative rule for this study, not a claim that the production selector has constant output everywhere inside that cell.

## Provenance and limits

**Measured:** reused Unit 2 baseline; no upstream model rebuild. Source hashes, clocks, population, probability-side identity and response coefficients were checked in this command.
Baseline guess = production score lattice at Tuesday total + 0.1 * joint residual - 1, with opener margin and frozen four-term probability.
The historical pool-line proxy is the opener. This study changes no ATS picks and requires no pre-2026 pool capture.
**Inferred limitations:** upstream artifacts and feature vintages are retrospective. LOSO includes future seasons for earlier holdouts; baseline histories can include other held-season prior games.
Bootstrap intervals condition on fitted coefficients and baseline; they exclude parameter refitting, prior-unit selection and prospective capture uncertainty.
Two new looks belong to lead88_total_response; they do not erase Unit 2's 82 reported looks or its broader 713-look parent protocol.
No market comparator was rescored; this bounded unit isolates the requested candidate-versus-served mapping correction.
No serving change, registry write, publication, new tests, commit or push occurred.

## Frozen protocol

Population: reuse Unit 2's frozen 2020-2025 1,343 paired games; historical OPENER is the pool-line proxy.
Retain all rows and 33 restored four-term rows; paired MAE uses the 1,333 defined served guesses, including 101 last games.
Target: actual combined final score. Fixed base: served four-term LOSO probability selects the side; no independent side flips.
Terms/folds: same single LAD slope b, no intercept, minimizing |actual total - served integer total - b * total move|.
Fit b on the other five seasons and score the held season; one full-data slope supplies optimistic IS only. No refitting upstream models.
Mapping: T is the original served continuous lattice centre, M the fixed served centre margin, delta=b*(deadline total-Tuesday total).
Define the cell as (floor((T+M)/2), floor((T-M)/2)), matching the unit team-score lattice coordinates.
Return the exact served home/away scores when delta=0 or the cell at T+delta equals the cell at T.
Otherwise rederive with Unit 2's production score selector at T+delta, fixed M and side; if it declines, retain the served score and count it.
Metrics: served MAE minus candidate MAE, primary last game and secondary all games; exactly 2 candidate-versus-served looks, no market regrading.
Uncertainty: 10,000 paired season-stratified week-block bootstrap draws, seed 88; percentile 95% intervals and half-weight ties in probability_positive.
Report closer/worse/tied and two-sided exact binomial null (0.5 on unequal errors), exact share intervals, six fold slopes/results, IS/OOS and paired gap.
Fold, count, and IS diagnostics describe these two looks; no subgroup selection, additional candidate, or post-outcome protocol revision.
Zero crossing closes nothing. No power control or refuted mechanism is supplied; any result remains proposed unresolved_below_power, pending orchestrator record.

## Verification

**Measured:** replay exit 0. Both .tools/uv.exe run --no-sync ruff check scripts/lead88_unit3.py and .tools/uv.exe run --no-sync ruff format --check scripts/lead88_unit3.py passed (exit 0).
