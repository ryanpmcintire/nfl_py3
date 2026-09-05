# PER-14: season-lagged offensive unit priors

## Predeclaration — 2026-09-05, before scoring

Construct: append two columns, `unit_prior_off_ol_diff` and
`unit_prior_off_skill_diff`, to production at runtime only. Each is the
home-minus-away final team-unit rating from exactly season S-1 for a game in
season S. Do not substitute an older season or the current season. Require
the rating's finalization time to precede the prediction timestamp. Missing
ratings remain missing, never zero. Do not change the active model or any
permanent feature-profile definitions.

Read (`scripts/unit_apm_screen.py:266` and
`artifacts/unit_apm/20260903T202022Z/results.json`): the existing screen pools
2019–2024 plays for odd/even player-coefficient reliability and saves summary
statistics. It does not save final team-season ratings. Source-season labels
in this summary are not six annual rating observations. Inventory these
artifacts before deciding whether an ATS experiment is supported.

Conditional evaluation plan, frozen before any scores: annual source ratings
2019–2024 would support candidate game seasons 2020–2025. Use expanding
chronological outer folds, one held-out season at a time, preserving all
earlier production training rows; fit/calibrate only before the held-out
season. Use the existing production opener confirmation harness, ridge alpha
10, market residual, weak_stack profile, production probability rule. Pair
candidate and production by game, excluding opener pushes and games without
both prior units. Save prediction-level output and per-season results. These
reused historical windows are exploratory, with a stated selection discount,
not independent confirmation. No hyperparameter or season selection by score.

Uncertainty: season/week-blocked bootstrap, 20,000 draws, seed 20260902;
report accuracy-point effect, 95% interval and `probability_positive`.
Permutation null: 1,000 within-season/week permutations of opener outcomes
against fixed baseline and candidate picks, same seed, using the existing harness. Positive
control: separately join **same-season final unit ratings**, deliberately
leaky and diagnostic only, on the same paired games. Never substitute realized
game margin for this oracle. Report its measured detectability without
assuming success. Odd/even reliability: independently fit each source
season's ratings on odd and even weeks, correlate matched team-unit ratings
across team-seasons, and report Pearson and Spearman-Brown per unit; never
call repeated copies of the same annual value across game weeks reliability.

Owner memory rule: features that only measure team quality better are bounded
near zero because team quality is already priced. Inferred: I think the
distinct construct worth separating is unit-level roster continuity across
the offseason, rather than another estimate of total team quality. If dated
pregame personnel membership and prior-season unit membership exist, classify
each unit as stayed (identical member set) or changed (any member difference).
Predeclare separate stayed/stayed and any-changed game strata for each unit,
with no outcome-driven threshold. Unknown membership is a separate missing
category, not stayed. Postseason/current-season participation must not label
pregame continuity. Neither a pooled correlation nor an undated roster proves
this construct. Inventory annual ratings and dated membership independently.

Binding taxonomy for every verdict (verbatim):

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. Only two grounds ever close a line of work: (1) refuted mechanism - a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to detect an effect that size. Everything else is unresolved_below_power: record it, report probability_positive, never the binary "contains zero". If a record command errors, the verdict is wrong, not the validator. Decisions are expected value: probability_positive above 0.5 favours playing it; state what the result implies for the DECISION before what is wrong with it. Never say a lead needs more games; the data is fixed and the project is model-limited.

Record every measured ATS result with `nfl-ats weak-signals record` in
`accuracy_points`. If annual rating artifacts are absent, use the explicitly
authorized missing-artifact fallback: implement and test the join contract,
save a provenance-stamped inventory, and do not manufacture a zero ATS effect
or a registry measurement. No scientific closing verdict follows from missing
serialized ratings. Reconstructing annual ratings from raw plays is a separate
producer task, not treating pooled multi-season coefficients as annual priors.

## Inventory result — appended 2026-09-05

Inferred: I think no play decision can be derived from this inventory; no
candidate effect was measured, and this is not a rejection of the mechanism.

Measured (`artifacts/experiments/unit_prior_features/inventory.json`, produced
by `scripts/unit_prior_features_on_production.py`): four unit-APM files,
zero annual rating rows, zero annual rating seasons, zero membership rows in
those artifacts, and zero paired games scored. The files list source seasons
2019–2024 but serialize only pooled reliability summaries. Missing for both
OFF_OL and OFF_SKILL: season/team final coefficients and availability times;
also missing are annual odd/even team-unit estimates and dated unit membership
in these artifacts. No broader absence of raw roster data is claimed.

Read (`artifacts/unit_apm/20260903T202022Z/results.json` and
`artifacts/unit_apm/20260903T202044Z/results.json`): the existing pooled
player-level Spearman-Brown summaries are 0.39396485091203975 (840 players)
and 0.32466292237035616 (501 players), respectively. These inherited values
were not remeasured and do not establish annual team-unit reliability.

Measured (`tests/test_unit_prior_features.py`): the adapter preserves game
rows, joins exactly S-1, leaves missing teams/seasons unknown, excludes
current/future ratings, and excludes ratings finalized at or after the
decision time. Duplicate keys and malformed inputs fail closed.

Measured (`nfl-ats weak-signals record --help`): the recorder requires a
numeric `--effect`. No accuracy-point record was written because no such
measurement exists; assigning zero would create false evidence in the pool.
Effect, interval, `probability_positive`, annual reliability, per-season ATS
rows, permutation results, leaky-oracle results, and continuity subgroup
scores are unmeasured. The missing-artifact fallback is complete; the
production screen remains unrun. The executable is explicitly an inventory
runner, not an implemented production evaluator.
