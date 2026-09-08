# PER-07: coordinator turnover on production

Predeclared 2026-09-07 before any candidate scoring. Family: per07_coord_change_on_production.

**Inferred (design):** turnover can affect scheme familiarity and early-season execution.
Exactly one candidate, weak_stack_coord_change, adds three home-minus-away flags:
coord_new_oc_diff, coord_new_dc_diff, coord_new_hc_diff. Each compares the
September 1 staff observation with the prior season's September 1 observation.
This measures season-start staff turnover, not a claim about the hiring day.
Both observations must be strictly before the decision timestamp, and the
current season cutoff itself must also have been reached. Missing or ambiguous names
stay unknown. No in-season flag enters this look: the bounded 2022-2025
revision census is a separate source assessment, with incomplete earlier coverage.
No alternative N, sign selection, interactions or tuning will be tried.

**Read** (`artifacts/active_ats_model.json`): baseline model a4c757efd2525da6,
weak_stack, ridge alpha 10, gaussian_median, no calibration. The newest
matching opener archive is resolved by find_matching_opener_evaluation.
Baseline probabilities and picks must replay exactly. Same chronological
weekly fitting and training-only imputation as the established v5 evaluator.
For archive games use the saved Tuesday snapshot timestamp; for training-only
rows use September 1 at 00:00 UTC as a conservative staff information boundary
(or the day before kickoff if earlier). Only these new columns use that boundary.

**Inferred (measurement plan):** paired opener probability-rule accuracy in
accuracy points, 20,000 week-blocked bootstrap draws, seed 20260817; within-week
correlation ZERO, no design-effect padding. Full 1,537-game archive with baseline
fallback on incomplete coordinator coverage is the decision read; covered-only,
season tables, assigned window and a 200-draw within-week frozen-pick null are
reported too. No close-grade veto. The assigned rotation window is confirmation
accounting; the full archive is a predeclared descriptive decision comparison.
The archive was mined by many prior families; this one-look result carries that
selection/multiplicity discount and is not untouched confirmation. Positive
expected opener delta favours the candidate; no promotion threshold gates play.
This lane records the evidence and never changes the model or played card.

## Closing-grounds taxonomy (verbatim, binding, applies to every experiment you run or judge)

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



## Completed look (2026-09-07)

**Measured** (`laneM_artifacts/eval_final/opener_summary.json` under the fleet
scratch directory): expected-value decision favours the incumbent: candidate
53.6261% vs active 53.9587%, delta **-0.332668 accuracy points**, week-blocked
95% **[-1.592568, +0.926539]**, `probability_positive` **0.28905**. There are
1,537 paired archive games, 1,503 nonpush games, 107 weeks and 97 changed nonpush
picks; the candidate gets five fewer games right. All 1,537 baseline probabilities
and picks replayed against the newest active-matched opener run
`artifacts/opener_evaluation/20260907T152026Z` before this completed scoring run.
**Inferred:** retain the incumbent on this comparison; the signal remains
`unresolved_below_power`, not closed, because no admissible closing ground exists.

**Measured** (same artifact): complete turnover coverage is 1,019/1,537 games
(66.30%); the remaining 518 use the baseline pick. Covered-only: 994 nonpush,
delta -0.503018, 95% [-2.423681, +1.388958], probability_positive 0.28905.
Assigned 2020-2021 window: 466 games / 456 nonpush, +0.657895 points,
95% [-0.884956, +2.178649], probability_positive 0.7657. This is a distinct
window read, not a substitute for the predeclared full-archive decision read.
Season-blocked full-archive interval [-1.370757, +0.730897], probability_positive
0.2581. Within-week correlation remains zero; no extra design-effect correction.

**Measured** (same artifact): season deltas in accuracy points are 2020 0.0000,
2021 +1.2712, 2022 -0.4032, 2023 -2.2556, 2024 -1.5038, 2025 +1.1236.
The 200-draw frozen-pick null has mean -0.23686, 95% [-1.66667, +0.99800], and
observed percentile 0.48. This is an outcome-shuffle diagnostic, not a positive
control demonstrating power or admissible grounds to close the mechanism.

**Measured** (`registry/weak_signals.json`): CLI recorded
`per07_coord_change_on_production_opener` under the exact family prefix with
`unresolved_below_power` and a pool-player plain summary. **Measured**
(`registry/rotation_registry.json`): `rotation record` spent the assigned
2020-2021 window as `unresolved`, carrying its own +0.657895 / 0.7657 result
and notes explicitly linking the separate full-archive comparison. The 2022-2025
portion is predeclared descriptive reuse, not untouched confirmation.

**Measured** (execution record): an initial run replayed the baseline, then was
interrupted during source-parser repair before any candidate result was read;
no result artifact was emitted. The only completed ATS look uses
`data/raw/coordinators/20260907T213814366437Z/coordinator_history.parquet`, SHA-256
`c09ec5a712a19206801b9538dc3747b1191f5074caf15e33e851dde8bd36d243`.
No result-driven variant, tuning or second completed look ran. The final runner
sets isolated NFL_ATS_ARTIFACTS_DIR and NFL_ATS_REGISTRY_DIR, and holds BLAS/OpenMP
threads to one; recorder CLIs deliberately write the shared project registries.

**Inferred:** the mined-archive discount, incomplete parsed staff coverage,
September-to-September definition (not exact tenure or playcaller identity),
and inherited close-era non-spread baseline features limit interpretation.
The in-season audit exposes revision errors and role conflation; it is not fed
into this season-start experiment. No model, played policy or published card
was changed by this lane.
