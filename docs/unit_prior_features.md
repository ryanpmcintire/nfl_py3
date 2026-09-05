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

## CX20 annual producer and completed screen — 2026-09-05

Inferred: I think this measured comparison favors retaining the baseline card,
without closing the unit-prior mechanism. Measured
(`artifacts/experiments/unit_prior_features/cx20_20260905/results.json`): the
lagged two-column candidate scores **795/1,503 (52.8942%)**, versus the paired
baseline **801/1,503 (53.2934%)**, an effect of **−0.399202 accuracy points**,
week-blocked 95% **[−1.490591, +0.720862]**, `probability_positive=0.2208`.
Measured (`nfl-ats weak-signals record`): this is recorded as
`unit_prior_off_ol_skill_on_production_cx20`, `unresolved_below_power`, in
`accuracy_points`; the same-season oracle is separately recorded as
`unit_prior_same_season_oracle_cx20`, category `control`, in its own family.

Measured (`artifacts/unit_apm/cx20_20260905/annual_summary.json` and
`team_unit_ratings.parquet` in that directory): **1,280 unique
season/team/unit rows**, **32 teams × four units × ten seasons, 2016–2025**;
2013–2015 lack local participation partitions. Measured (producer run): all
120 annual/odd/even unit fits use the imported frozen Ridge alpha 1000,
team scale 11, EPA clip 5 recipe; the original pooled reliability artifacts
were not rewritten. Measured (aggregation fixture): team ratings are means
of fitted player coefficients weighted by that team's actual player-snaps,
with traded-player snaps attributed to the team on each play; nuisance team
coefficients are excluded. Measured (output schema): `players` is distinct
member count, `snaps` is player-plays; `members`, `rating_odd`, `rating_even`,
and `finalized_at` supplement the six required columns.

Measured (`annual_summary.json`): each between-season result below pairs
288 team/year transitions; each odd/even result pairs 320 team-seasons.
Measured (producer command): intervals resample 32 team clusters, retaining
all years for a sampled team, with 20,000 draws and seed 20260902.

| Unit | Measured year t vs t+1 Pearson, 95% interval | Measured probability_positive | Measured odd/even Pearson, 95% interval | Measured odd/even probability_positive | Measured Spearman-Brown |
|---|---|---|---|---|---|
| OFF_OL | 0.057336 [−0.081294, +0.188399] | 0.77685 | 0.062615 [−0.063150, +0.176066] | 0.84915 | 0.117851 |
| OFF_SKILL | 0.063458 [−0.058465, +0.180381] | 0.84250 | 0.056246 [−0.058796, +0.169813] | 0.83160 | 0.106503 |
| DEF_FRONT | 0.090512 [−0.019494, +0.184282] | 0.94495 | −0.042989 [−0.163943, +0.069935] | 0.22805 | −0.089839 |
| DEF_SECONDARY | 0.016519 [−0.093944, +0.117585] | 0.60820 | −0.032111 [−0.109608, +0.047871] | 0.20735 | −0.066352 |

Measured (`results.json`, `fold_audit`): held-out game seasons remain the
predeclared 2020–2025; all earlier production training rows are retained,
with 2,816/3,072/3,344/3,615/3,887/4,159 training rows respectively in each
arm. Measured (same audit): every training season is strictly earlier than
its test season, and residual-distribution calibration is inside that earlier
training population. Measured (configuration): production is `weak_stack`,
Ridge alpha 10, market residual, Gaussian probability rule at 0.5, no added
calibration; no profile or active-model definition is changed. Measured
(paired output): 1,503 non-push games in 107 season/week blocks.

| Held-out season | Measured games | Measured baseline correct | Measured candidate correct | Measured effect, accuracy points | Measured oracle correct | Measured oracle effect, accuracy points |
|---|---|---|---|---|---|---|
| 2020 | 220 | 114 | 112 | −0.909091 | 116 | +0.909091 |
| 2021 | 236 | 131 | 129 | −0.847458 | 129 | −0.847458 |
| 2022 | 248 | 130 | 126 | −1.612903 | 127 | −1.209677 |
| 2023 | 266 | 147 | 149 | +0.751880 | 148 | +0.375940 |
| 2024 | 266 | 141 | 137 | −1.503759 | 144 | +1.127820 |
| 2025 | 267 | 138 | 142 | +1.498127 | 139 | +0.374532 |

Measured (`results.json`, oracle): deliberately leaky same-season final unit
ratings score **803/1,503 (53.4265%)**, **+0.133067 accuracy points**,
week-blocked 95% **[−1.430591, +1.673360]**,
`probability_positive=0.5503`, on exactly the candidate's paired games.
Inferred: I think this diagnostic has not established a candidate-sized
positive-control bound and provides no closing ground for the lagged feature.
Measured (same artifact): 1,000 within-season/week outcome permutations give
candidate null mean **+0.049634 points**, 95% quantiles
**[−0.931470, +1.064538]**, fraction below observed **0.166**; oracle null
mean **+0.530406 points**, quantiles **[−0.931470, +2.132402]**, fraction
below observed **0.298**. Measured (secondary season-block bootstrap):
candidate interval **[−1.333333, +0.632911]**, `probability_positive=0.17645`;
oracle interval **[−0.594845, +0.796813]**, `probability_positive=0.6267`.

Measured (weekly-roster parquet schema inventory and `results.json`): the
roster source has season/week labels but no publication timestamps;
all 1,503 games have unknown dated continuity, with zero scored stayed/stayed
or any-changed groups. Inferred: I think none of the tested two columns
isolates a non-quality mechanism; these are player-quality estimates after
team adjustment, not a measured roster-continuity signal. Read (owner memory
rule in this predeclaration): better team-quality measurement is already
priced; a continuity interpretation must not be attached to this result.

Measured (producer and evaluator configuration): source availability is
reconstructed conservatively as March 1 following the source season, checked
against seven days before gameday; it is not a contemporaneous archive.
Read (`src/nfl_ats/clv.py:2022`): inherited opener evaluation replaces only
the spread and retains close-era covariates. Inferred: I treat this as an
exploratory reused-window result with a selection discount, not independent
confirmation or a new audit of every production feature's historical vintage.
Measured (fold implementation): annual refits follow this predeclaration and
therefore differ from the published weekly-refit historical baseline.
