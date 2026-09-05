# SIM-03 lineup mixture

Predeclared 2026-09-05 by CX21 before scoring.

The frozen screen draws 200 independent Bernoulli availability scenarios per
team-game with seed 2026090521. All visible depth-chart players are sampled;
only rank-one QB/offense/defense players contribute, exactly as LEAD-62:
missing availability times trailing-four-week snap share, home minus away.
Missing snap history contributes zero. Reuse LEAD-62 visibility, injury and
walk-forward probability helpers at min(kickoff, Sunday 16:00 Eastern).
Legacy week-labelled depth observations retain their documented archive proxy.

Use production's weekly chronological ridge fit and a separate fit with the
three LEAD-62 columns. Extract the latter's three coefficients by unit
perturbations through its fitted estimator, including scaling. Apply those
coefficients to (scenario loss minus expected loss) around the production
margin. This isolates uncertainty rather than changing the mean forecast or
refitting 200 models. Retain production's Gaussian residual mean and sigma.
Primary mixture probability averages the conditional Gaussian cover
probabilities; also report the literal share of scenario centers above the
opener, which omits game residual noise and is not the primary probability.

Evaluate requested 2020–2025 regular-season opener games in weekly folds;
fit margins strictly before each week's first game, with the production
chronological residual holdout; fit player probabilities before the target
season, calibrating on the preceding season. No tuning/selection on these
scored seasons. Report unavailable seasons rather than impute entire lineups.
Read (`play_probability_panel.parquet`, startup inspection): rows end in 2024.
Retain the opener evaluator's inherited close-era non-spread inputs; this is
an opener-graded refresh screen, not reconstructed Tuesday information.

Outputs: paired Brier improvement (baseline minus mixture), 10 fixed equal-width
reliability bins, 20,000 week-blocked bootstrap draws with the fixed seed and
probability_positive; forced-pick flips at >=0.5, paired accuracy improvement
in percentage points across all non-push games and conditional on flips;
season stability; mean scenario-center SD and total predictive SD increase.
No within-week correlation estimate or inflation. Pushes excluded identically.
Controls use the same coefficients and production center: shuffle probabilities
across all players within team-game before starter aggregation, and replace
probabilities with the actual played label (snap-derived participation oracle,
not a claim that zero snaps proves an official inactive designation). Center
both controls against the original expected loss so they can test the mapping.
One fixed permutation is a null diagnostic, not a permutation p-value.

For Week 1 use the current serving probabilities, prior snap history and
the current forecast centers/probabilities, preserving its residual mapping;
report base-rule crossing changes separately from published policy picks.
Store predictions, controls, coefficients, coverage, source hashes and results
under artifacts/experiments/lineup_mixture/. Record accuracy_points and
brier_improvement separately via the weak-signals CLI.

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. Only two grounds ever close a line of work: (1) refuted mechanism - a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to detect an effect that size. Everything else is unresolved_below_power: record it, report probability_positive, never the binary "contains zero". If a record command errors, the verdict is wrong, not the validator. Decisions are expected value: probability_positive above 0.5 favours playing it; state what the result implies for the DECISION (would serving the mixture probability change any Week 1 pick? read the current forecast and say) before what is wrong with it. Never say a lead needs more games; the data is fixed and the project is model-limited. Within-week game correlation is zero by owner mandate; never estimate or pad it.

## Results appended 2026-09-05

Measured (`artifacts/experiments/lineup_mixture/week1.csv`): serving the mixture
changes **0 of 16 Week 1 base picks**; the largest absolute probability movement
is 0.000883845 (0.08838 percentage points). Inferred: retaining the published
policy actions therefore keeps all current card sides, including MIA as the
existing Best Pick; the nomination policy itself was not rerun. Measured
(`results.json` in the same directory): Brier improvement is +0.0000325123,
95% [-0.0000267991, +0.0000930683], probability_positive **0.859**. Inferred:
this favours serving the mixture probability on expected Brier loss; no Week 1
side change is implied, and this lane does not modify serving code.

Measured (`results.json`): **1,236 non-push games / 89 weeks**, 2020–2024;
29 pushes excluded. Baseline Brier **0.2516537129**, mixture **0.2516212006**.
Both forced-pick accuracies are **53.72168%**; paired improvement **0.00000
accuracy points**, 95% [-0.24135, +0.24311], probability_positive **0.34535**
(strictly positive bootstrap draws; ties are not positive).
Measured (`paired_predictions.csv`): two flips, one lost correct pick
(`2020_03_CIN_PHI`) and one gained correct pick (`2021_03_TB_LA`); conditional
accuracy improvement among flips is 0 points. Measured (`results.json`):
mean scenario-center SD **0.449575 margin points**, mean increase in total
Gaussian-plus-lineup predictive SD **0.0102103 points**.

Measured (`results.json`): season stability, with accuracy in percentage points
and Brier improvement defined as baseline loss minus mixture loss:

| Season | Games | Accuracy improvement | Brier improvement |
|---|---:|---:|---:|
| 2020 | 220 | -0.454545 | +0.0000789631 |
| 2021 | 236 | +0.423729 | +0.0000268147 |
| 2022 | 248 | 0 | -0.0000012090 |
| 2023 | 266 | 0 | -0.0000225278 |
| 2024 | 266 | 0 | +0.0000856287 |

Measured (`results.json`): the fixed within-team-game permutation has 110 flips,
accuracy improvement **-1.61812 points**, 95% [-3.24149, 0.00000],
probability_positive **0.02320**; Brier improvement **-0.00134429**, 95%
[-0.00310732, +0.000402767], probability_positive **0.06635**.
Measured (same artifact): snap-participation oracle has 52 flips, accuracy
improvement **-0.485437 points**, 95% [-1.77857, +0.876494],
probability_positive **0.21715**; Brier improvement **-0.000442077**, 95%
[-0.00138304, +0.000488087], probability_positive **0.17690**.
Inferred: this oracle does not demonstrate sensitivity to a benefit of the
candidate's size, so it supplies no positive-control closing ground.

Measured (`reliability.csv`): the nonempty points of the fixed-bin reliability
curves are below; the CSV retains all ten bins for both arms and controls.

| Probability bin | Baseline n | Baseline mean / observed | Mixture n | Mixture mean / observed |
|---|---:|---:|---:|---:|
| 0.2–0.3 | 1 | 0.218985 / 1.000000 | 1 | 0.217735 / 1.000000 |
| 0.3–0.4 | 127 | 0.367806 / 0.496063 | 127 | 0.367935 / 0.496063 |
| 0.4–0.5 | 561 | 0.454074 / 0.450980 | 559 | 0.453950 / 0.450805 |
| 0.5–0.6 | 483 | 0.541798 / 0.525880 | 485 | 0.541622 / 0.525773 |
| 0.6–0.7 | 64 | 0.626102 / 0.593750 | 64 | 0.625885 / 0.593750 |

Measured (weak-signals CLI; `record_commands.txt`): six records
`sim03_lineup_{mixture,permutation,oracle}_{accuracy,brier}_cx21` were recorded
as `unresolved_below_power`, under the shared family `sim03_lineup_mixture_cx21`;
accuracy and Brier use `accuracy_points` and `brier_improvement` respectively.
Inferred: these overlapping arms and units must not be treated as independent
votes or pooled together as one commensurable effect.

Measured (`results.json`): 272 archived opener games in 2025 lack lineup panel
coverage and were not scored; the requested 2020–2025 evaluation is therefore
incomplete for 2025. Measured (serving artifact inspection): seven live rows
have no player ID/probability (five unnamed slots and two named reserves);
they are omitted because they have no linked snap-history weight, not assigned
invented probabilities. Read (`play_probability.py:759`): the oracle label
is snap participation, not official inactive status. Read
(`artifacts/lineups/current/lineups.json`, probability_provenance): the live
player model reports uncalibrated_insufficient_history because 2025 calibration
rows are absent. Inferred: legacy weekly depth timing, the inherited close-era
inputs, independent player sampling, the single permutation and K=200 Monte
Carlo error limit interpretation; no candidate-sized positive-control bound
or resolved mechanism refutation was established.

Measured (live reconstruction check): weekly-refitted Gaussian probabilities
match the active forecast probabilities to **2.22e-16** maximum absolute error;
the current forecast is filtered to `market_residual` before comparison.
Measured (verification commands): Ruff format/check passed, `mypy src` passed
245 source files, and the required three-file targeted pytest run passed
**30 tests**. The fixture tests cover post-cutoff depth, outcome isolation,
zero history, bench weights, deterministic sampling and Gaussian convolution.
