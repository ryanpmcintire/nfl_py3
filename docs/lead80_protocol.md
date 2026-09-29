# LEAD-80 predeclared protocol

**Read:** copied from ROADMAP.md:863 and docs/lanes/ideation-2026-09-29b.md before inventory, fitting, or scoring.

| LEAD-80 | ⬜ | Tiebreaker scores conditional on a weekly tie (batch B rank 7 of 8) | **Inferred proposal, 2026-09-29; unmeasured.** Mechanism: post-freeze scoring news and margin-total dependence
change which score paths produce both an ATS tie and the tiebreak game's result; an unconditional total forecast can be wrong for precisely the contest states where the guess matters. Predeclare
2020-2025 weeks with authentic pool rules, archived guesses/cards for evaluation, and pre-deadline spread/total captures. First verify closest-total versus exact-score rules, lock timing, push/Best
Pick points and field observations. Fit the joint discrete score law and field distribution only on earlier seasons; held-out opponent cards/guesses are evaluation outcomes, never pre-deadline inputs.
With sides and Best Pick selection fixed, choose the guess minimizing the verified rule loss conditional on a weekly score tie, integrating uncertainty in the field. Protocol B: chronological LOSO,
paired ordinary prediction baselines plus the same-law unconditional guess, IS/OOS gap and season-block intervals. B=3 (joint score fit, field fit, conditional-choice arm), F=6, E=2 (joint-score log
score and actual tiebreak utility); 445 counted looks including the additional unconditional-guess comparator (35 cells beyond Protocol B). **Read/checked:** MOD-17, SIM-04/08, LEAD-53/54/72,
POL-05/09 and POOL-01. This conditions the score distribution on the tie event; it does not repeat marginal lattice shading, nomination, push arithmetic or side/rank optimization. Units: rules/data
inventory, 15-20 calls; conditional replay, 25-30 only if real observations exist. Rank rationale: a separate pool benefit with a smaller eligible population; do not present assumed-field simulations
as measured pool improvement. |

## Protocol B

Rows fix population, terms, target, units and specification budget B. Inventory metadata first; require issuance/ingestion before the pool deadline. Grade authentic frozen openers. Missing archives
are data gaps, never replaced by closes, realized weather or retrospective news.

Chronology-purged LOSO excludes target/later seasons from fitting; separate earlier training, selection and calibration seasons. Folds lacking training remain unavailable. Reuse certified pregame
predictions; choose coefficients/penalties on earlier seasons without new grids. Freeze before prospective evaluation. One calibrated discrete-margin distribution combines information; its cover
probability selects the side.

Pair combined-model, model-only, market-only and Elo baselines. Report IS/OOS opener accuracy, Brier, log loss and margin MAE, their gap, decisive-game record first, fold coefficients, season
stability, five training-quantile reliability bands, season-block 95% intervals and `probability_positive`. Zero crossing closes nothing; unresolved effects remain `unresolved_below_power`.

Count reporting cells too: L = (B + 36 + 8E)(F + 1) + 25 for F scheduled folds, E extra endpoints: B candidate/four baseline specifications, four-comparator IS/OOS metric cells per fold/pooled, five
bands for five models. Identical nested refits add no specification. New splits/specifications need separate preregistration. Correlated cells are not independent evidence.

| 80 | 3 / 6 / 2 | 445 |

LEAD-80 adds 35 looks for the unconditional-guess comparator: seven arm cells plus 28 IS/OOS endpoint cells.
