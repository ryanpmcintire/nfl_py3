# MOD-18 spread regime program — frozen before scoring

Declared 2026-09-07, lane H. Read: `artifacts/active_ats_model.json` identifies
incumbent `a4c757efd2525da6`, weak_stack ridge alpha 10, gaussian_median.
Primary archive: `artifacts/opener_evaluation/20260907T152026Z/per_game.parquet`.
All evaluation outputs and registry commands use isolated scratch roots seeded
with that manifest. Data are read-only; no activation, publication or ledger writes.

## Frozen family and decisions

Family `mod18_spread_regime_v1`: C1, C2, C3, C1_C2, C1_C3, with no tuning after
scoring. Primary metric is paired non-push opener accuracy improvement against
the incumbent's own probability-rule picks on the full 1,537-row archive.
Select maximum point improvement, ties in the order above. Report each arm,
season, and assigned two-season block, including negative measurements.
Bootstrap whole season-week blocks, 20,000 draws, seed 20260817; within-week
correlation parameter ZERO, no variance inflation. Frozen-pick null permutes
home-cover outcomes within each week (2,000 draws, same seed); report the
maximum over all five arms as selection context. Thresholds govern claims only;
the coordinator decides the played card on expected value.

Reuse discount: this is a mined archive, inherited active-model selection,
mined spread boundaries, and a four-rule union selected among 127 subsets.
Five correlated candidate arms add selection. These are descriptive reused-era
measurements, not independent confirmation; no numerical discount invented.
Declare/assign before scoring; record each arm using weak-signals record and
each assigned block using rotation record. Record full-archive overlaps explicitly.

## Part 1: diagnosis (no window spent)

Reproduce current opener buckets and extend by favourite/underdog pick and
stated confidence versus accuracy. Disjoint buckets are [0,3], (3,6.5],
(6.5,7.5), [7.5,10], (10,infinity); the source table's overlapping labels
are resolved by giving 7.5 to the zone. Also report the alternate boundary
allocation to make reproduction discrepancies explicit.
Use the pre-2018 script's walk-forward margin fitting recipe on 2011-2017
and 2018-2019 with current gaussian_median. These eras use the archived
nflverse spread (close proxy), NOT the unavailable Tuesday opener; keep grade
labels explicit. Training strictly precedes the full prediction week.
Per-era estimates and week-bootstrap intervals accompany reliability tables.
Weaker eras are magnitudes, never absence. Diagnosis does not choose new arms.

## Part 2: fixed candidates

C1: `weak_stack_spread_regime` adds eight home-oriented features to weak_stack:
sign(spread) times indicators for [3.5,6.5], [7.5,10], [10.5,infinity],
signed absolute spread, and sign(spread) times absolute distance of |spread|
to each of 3,7,10,14. Positive spread means home favourite in this repository.
Reference is 0-3; the 7-point gap has no separate indicator. All columns are
row-local, use the scoring line at prediction, and ridge alpha stays 10.

C2: additive `discrete_residual` probability method. Shift each trailing
out-of-time residual by predicted margin, round to the nearest integer final
margin, then count strict covers at the quoted line with the existing half-count
continuity correction. Integer pushes are not covers. This preserves empirical
mass rather than fitting a Gaussian. Inferred limitation: rounding translated
residuals does not guarantee special excess mass at the football key numbers;
we will quantify that distinction, not claim an empirical error distribution
alone learns a score-lattice mechanism.

C3: simple predeclared calibration cells: the five disjoint spread buckets
above, pick side (favourite/underdog/pickem), and stated pick probability bands
[0.5,0.55), [0.55,0.60), [0.60,1]. Each cell estimates correctness with
20 pseudo-observations at the current stated probability: (prior cell wins +
20 * stated p)/(prior cell n + 20). The count 20 is fixed regularization,
not tuned. No history means unchanged probability. Only completed games
before the target week's first game enter; use earlier-season walk-forward
predictions for initial history. At prediction transform back to home probability;
flip only if calibrated pick correctness falls below 0.5. No fitted in-sample
predictions enter. Historical calibration uses its available line, clearly
labeling close-proxy history versus opener history.

C1_C2 combines C1 features and C2 mapping; C1_C3 calibrates C1's Gaussian-median
probabilities using C1's own earlier walk-forward stream.

## Part 3: played-card comparison

Run overlay-composition in scratch on the selected arm's opener per-game artifact.
Evaluate the existing four-member OR union on candidate and incumbent, and
candidate union without spread-gap against the incumbent four-member played card.
All member triggers are recomputed against each incoming card. Same paired
bootstrap, no subset search. Also report zone marginal on candidate. Produce
read-only Week 1 2026 margin predictions and compare sides with
`artifacts/margin_predictions/2026-week-01-20260907T151637Z/predictions.csv`.

## Closing-grounds taxonomy (verbatim from fleet brief)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. At this
evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains zero". The
registry code hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not
the validator. Never use 95%, 0.90, or any threshold as a DECISION bar; decide on expected value
(`probability_positive` above 0.5 favours playing it), thresholds only govern what docs may CLAIM.
Grade at the OPENER (the pool's grade); a close-graded number may never veto a play. Within-week game
correlation is ZERO by owner mandate: never estimate or pad it. Never say something "needs N more
games": the data is fixed and the project is model-limited.

## Gates and scope

Leakage regression: future outcome/line cannot move earlier features/calibration;
same-week outcomes cannot train that week's calibration. Fixed mapping fixture,
feature orientation and profile membership tests. Run the user-listed targeted
pytest suite with two workers and scratch basetemp, ruff format/check on changed
Python files, and repo-wide mypy src. Report to scratch/reports/laneH_spread_regime.md.
Dashboard, scheduler writes, handoff, publication and commits are outside this
lane's explicit file scope and fleet restrictions; coordinator retains ownership.

## Post-run interpretation addendum (does not alter the frozen arms)

Measured: `laneH/results.json` selects C2 under the frozen five-arm ranking.
Measured: `laneH/diagnosis.parquet` reproduces the supplied bucket counts and
accuracies but corrects reversed favourite/underdog labels: the zone's 111
picks are favourites, 83 are underdogs. Read: `src/nfl_ats/score_lattice.py:261`
documents the positive-home-favoured sign convention explicitly.

Read during the run: new commits `fe03294` and `d300244` add binding instructions
in `AGENTS.md:56-92`: conditional discrete margins and retirement of unexplained
threshold flips. The frozen four-member comparison remains a research
counterfactual; it is not a recommendation to reinstate the zone.
Inferred: the requested rounded pooled-residual C2 recipe is an additive
integer-support control, not a completed conditional key-number model. Its
measurement may not be cited as testing or rejecting the newly explicit
conditional-distribution requirement. A conditional lattice is follow-up work;
the original five arms are not silently redefined after their signs were read.
