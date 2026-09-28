# September 28 research backlog checkpoint

The current-week repair was followed by substantive work on POL-10, POOL-01,
positive-control precision, and SIM-08. No result here promotes a candidate or
selects a serving coefficient. The independent Week 4–18 study remains enrolled
with unchanged source hashes and results withheld until its declared cohort ends.

## Frozen prospective scorecard

**Superseded baseline comparison:** the later [seven-category report](seven_backlog_completion_20260928.md)
requires exact same-book quote proof and corrects the raw baseline from cover mass
to conditional non-push probability. The current artifact is
`artifacts/prospective_scorecard/20260928T175751Z/`: 24 raw-eligible games (23 decisive)
and only three market-eligible games. The historical table below is retained as an
audit trail, not current comparative evidence. The frozen 28-19 record is unchanged.

**Measured:** the former scorecard reported 30–17 from a different decision ledger.
Eight sides and 44 displayed probabilities disagreed with the actual 48 frozen
published rows. The corrected record is **28–19, one pending**; Best Pick is 1–2.
Every row is graded from its immutable published side, probability, and original line.

**Measured:** 11 archived raw forecasts postdated the exact frozen publication;
24 decision bindings postdated publication. The corrected comparison requires
the named forecast and binding to exist by publication and match the original line.
All 48 served rows remain; 24 unavailable baseline bindings are explicit. Of the
24 eligible rows, 23 are decisive. The four arms below use those same 23 games.

| Probability arm | Brier | Log loss |
|---|---:|---:|
| market | 0.254919 | 0.702992 |
| neutral | 0.250000 | 0.693147 |
| raw | 0.236740 | 0.666282 |
| served | 0.251553 | 0.696079 |

These are descriptive archived baselines, not an isolated estimate of the fitted
situational terms. The served-only 47-game Brier is 0.238437 and log loss 0.669702.
The matched subset must not be compared with those whole-card values as an effect.
No three-week performance result selects or rejects a model.

**Measured:** all 20 fixed reliability cells and all 63 challenger status cells
are retained: 45 scored, 10 original-line mismatches, three missing scored rows,
and five unsupported sources. Challenger grades must agree with current results
before pairing. Pending games cannot suppress a valid decisive comparison.
Look family: 48 probability metrics, 20 reliability cells, 63 challenger status
cells, and nine record cells. No best-cell selection. Source hashes cover every
referenced forecast CSV/metadata file, ledgers, results, and implementation.

Evidence: `artifacts/prospective_scorecard/20260928T162048Z/summary.json` and `predictions.csv`.
Independent arithmetic checks reproduced every matched Brier/log-loss value.

## Pool field model

**Measured:** the corrected `python scripts/pool_field_share_fit.py` fails closed
for `2026_02_DET_BUF`: no saved deadline-eligible public split exists. The deadline
is September 18 at 00:15 UTC; the first saved pair-bearing capture is September 19
at 16:00 UTC. The 32-game population was neither reduced nor backfilled.

The code now uses frozen served probabilities, validates the field/decision line
signs and push outcomes, and predeclares Week 1 training → Week 2 scoring only.
The former future-to-past and same-week fits are invalid. Its earlier numerical
verdict is superseded pending a valid remeasurement; this is not a negative result.
Exact contemporaneous spread lines are absent from the public-split source, and
rank simulations remain conditional on no push. The fitted branch was not rerun
because its required input is missing. Any rank search is an unserved counterfactual.

## Positive-control bootstrap precision

**Measured:** `artifacts/positive_control_precision_replay/20260928T160726Z/results.json` preserves 1,202,400
prediction rows and 2,400 replicate/draw-prefix rows. All four historical 400-draw
cells reproduced exactly, including optimizer diagnostics. The population is the
pinned 1,503-game historical evaluator input, not a current-model analogue.

The predeclared four cells used the original 200 synthetic replicates, fixed
coefficients 0/1.80, LOSO fits, and season-then-week bootstrap seeds. One 20,000-draw
stream provides strict 400/2,000 prefixes. All 24 diagnostic looks follow.

| Control | Coefficient | Draws | Detections / 200 | Rate, Wilson 95% interval |
|---|---:|---:|---:|---:|
| DPI | 0.00 | 400 | 1 | 0.5% [0.09, 2.78] |
| DPI | 0.00 | 2,000 | 1 | 0.5% [0.09, 2.78] |
| DPI | 0.00 | 20,000 | 1 | 0.5% [0.09, 2.78] |
| DPI | 1.80 | 400 | 77 | 38.5% [32.03, 45.40] |
| DPI | 1.80 | 2,000 | 71 | 35.5% [29.20, 42.35] |
| DPI | 1.80 | 20,000 | 69 | 34.5% [28.26, 41.32] |
| Holding | 0.00 | 400 | 1 | 0.5% [0.09, 2.78] |
| Holding | 0.00 | 2,000 | 1 | 0.5% [0.09, 2.78] |
| Holding | 0.00 | 20,000 | 1 | 0.5% [0.09, 2.78] |
| Holding | 1.80 | 400 | 156 | 78.0% [71.76, 83.18] |
| Holding | 1.80 | 2,000 | 148 | 74.0% [67.51, 79.59] |
| Holding | 1.80 | 20,000 | 150 | 75.0% [68.57, 80.49] |

| Control | Coefficient | Draw comparison | Left-only / right-only | Disagreement, Wilson 95% interval |
|---|---:|---|---:|---:|
| DPI | 0.00 | 400 / 2,000 | 0 / 0 | 0.0% [0.00, 1.88] |
| DPI | 0.00 | 400 / 20,000 | 0 / 0 | 0.0% [0.00, 1.88] |
| DPI | 0.00 | 2,000 / 20,000 | 0 / 0 | 0.0% [0.00, 1.88] |
| DPI | 1.80 | 400 / 2,000 | 10 / 4 | 7.0% [4.22, 11.41] |
| DPI | 1.80 | 400 / 20,000 | 12 / 4 | 8.0% [4.98, 12.60] |
| DPI | 1.80 | 2,000 / 20,000 | 7 / 5 | 6.0% [3.47, 10.19] |
| Holding | 0.00 | 400 / 2,000 | 0 / 0 | 0.0% [0.00, 1.88] |
| Holding | 0.00 | 400 / 20,000 | 0 / 0 | 0.0% [0.00, 1.88] |
| Holding | 0.00 | 2,000 / 20,000 | 0 / 0 | 0.0% [0.00, 1.88] |
| Holding | 1.80 | 400 / 2,000 | 10 / 2 | 6.0% [3.47, 10.19] |
| Holding | 1.80 | 400 / 20,000 | 9 / 3 | 6.0% [3.47, 10.19] |
| Holding | 1.80 | 2,000 / 20,000 | 1 / 3 | 2.0% [0.78, 5.03] |

**Inferred:** the 400-draw classification is not interchangeable with the production
20,000-draw result for these injected effects. The numerical precision question is
now measured for these four cells; unmeasured coefficients and smaller natural
effects remain unbounded. Monte Carlo intervals quantify 200 synthetic replicates,
not generalization across data-generating mechanisms. No natural signal is closed.
Detection rates have no admissible weak-signal effect unit, so they are retained
as control diagnostics rather than mislabeled accuracy effects in pooled research.

## Simulator terminal rules

**Measured:** three deterministic defects were repaired: zero elapsed transitions
were floored to one second; exact quarter boundaries used the previous quarter;
overtime started with three timeouts rather than two. Focused production-call
probes verified the repaired states, including preservation of the play-cap guard.

Before replay, the lane fixed seed 20260925, training 2009–2014, validation
2015–2017, 3,334 simulations per validation season, and a copied feature artifact.
Both engines and all input hashes are preserved. The before arm exactly reproduces
the previous 10,002 simulated margins; both arms completed with zero caps.

| Absolute margin | Before mass | Repaired mass | Historical reference 90% band |
|---|---:|---:|---:|
| 3 | 10.368% | 10.858% | [13.802, 16.667]% |
| 7 | 8.038% | 7.648% | [7.552, 10.677]% |
| 10 | 5.469% | 5.029% | [3.906, 5.729]% |
| 14 | 4.659% | 4.679% | [4.557, 5.599]% |
| 17 | 3.609% | 4.019% | [2.083, 3.906]% |

**Measured:** repaired log-loss improvement over the training histogram is
+0.000939, conditional 95% season-resampling interval [-0.011431, +0.015928],
probability_positive **0.5926**. Before repair: −0.000885 [-0.012462, +0.012007],
probability_positive **0.3704**. These use the exact 27 resamples of three seasons;
the intervals exclude simulator Monte Carlo and model-selection uncertainty.

| Validation season | Before improvement | Repaired improvement |
|---|---:|---:|
| 2015 | -0.014625 | -0.012782 |
| 2016 | +0.016611 | +0.022144 |
| 2017 | -0.004641 | -0.006546 |

**Inferred:** correct terminal rules do not establish a better served model. Three
of five key-number bands pass after repair, versus four before; margin 3 remains
deficient and margin 17 is above its reference band. The engine's aggregate gate
is NO_GO. Under AGENTS.md, neither gate failure nor this interval closes a signal.
The two log-loss effects were recorded with `nfl-ats weak-signals record --batch`
as `unresolved_below_power`, names `sim08-terminal-repair-{before,after}-20260928`.

All 66 diagnostics remain in the report/uncertainty artifacts: 30 per arm plus
six season cells. The six season cells and exact uncertainty summary were added
after replay for transparent stability reporting, without parameter selection.
All 20 terminal sequence comparisons are retained, including negative results.
Evidence: `artifacts/sim08_terminal_repair/20260928T161258Z/`; `log_loss_predictions.parquet` preserves
the 768 paired actual-game losses and both `simulations_*.parquet` retain all
20,004 simulation rows. No simulator distribution was promoted to production.

## Verification and next work

Run commands and evidence are retained in the task lane. Existing tests cover
publication/chronology contracts; no test files or functions were added. No
redundant test deletion was identified in this bounded review.

Next: capture the already-declared Week 4 cohort without resetting enrollment;
continue the simulator's mechanism-led endgame diagnosis; obtain genuinely
predeadline pool inputs prospectively before fitting the field model. Expand
positive-control power at a matched effect only under a separate predeclaration.
