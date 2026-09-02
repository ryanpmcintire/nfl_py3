# PER-13 opener confirmation: durability P(plays) on production

Work package WP26, PER-13 Stage 2 follow-up. This is a new opener-graded
family descended from `per13_durability_on_production`, whose first look was
close-graded on [2011, 2013] (**read**,
`docs/per13_durability_stage2_on_production.md` §7.7). The opener grade is
required by the standing rule that play and promotion decisions are graded at
the opener; a close-graded result may not veto or establish one.

## 1. Predeclaration (before outcome scoring)

The candidate is exactly the existing `weak_stack_durability` profile and
implementation. It replaces production's nine availability-derived columns
with the durability-transported twins; it does not add columns, refit the
active model, or change the active model files. The baseline is the unmodified
`weak_stack` profile. Both use `market_residual`, ridge, and alpha 10.0, as
read from `artifacts/active_ats_model.json` before this look.

The opener grade substitutes the Tuesday-opener home spread for the scoring
line and settles each arm against that line. Weekly fits remain forward
chaining on the production feature table: training games kicked off before the
target week's earliest kickoff only. The opener pairing is the existing
archived Tuesday-opener population in
`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`; the child
family is assigned by `nfl-ats rotation assign`, never by hand.

The inherited implementation's point-in-time and identity contracts remain
binding: no future outcome may move an earlier durability offset, same-week
roster history is excluded, and when the durability offset is zero the nine
candidate columns equal production bit-for-bit. The regression coverage is in
`tests/test_per13_durability_production_feature.py` and
`tests/test_durability_prior.py` (**read** those files before scoring).

## 2. Frozen order, null, and positive control

Run exactly once in this order: null → positive control → screen. The null
shuffles the opener settlement margins within week while reusing the same
fitted arm probabilities. It is not required to centre on zero because each
arm can have a different home-pick rate.

The positive control replaces
`diff_injury_offense_unavailability_durability` with the realised ATS margin
in the candidate arm. A large detected effect is an instrument check only; it
does not bound detectability at the small effect scale unless it is explicitly
sized to that scale.

## 3. Decision and recording rule

The primary quantity is paired candidate-minus-baseline forced-pick accuracy
in `accuracy_points`, using the production probability rule
`home_cover_probability >= 0.5`. Week-blocked bootstrap (1,000 samples,
seed 20260826) is primary; season-blocked is secondary. Expected value is the
decision rule: `probability_positive > 0.5` favours playing the candidate, and
no 0.90 threshold may veto a forced pick. The result remains a research
record, not wagering advice.

An interval crossing zero is NEVER grounds to reject, fail or close this
experiment. A result that is not one of the two admissible closing grounds is
`unresolved_below_power` and must be recorded with its `probability_positive`.
The opener grade is the decision grade; a terminal classification is allowed
only when the recorded evidence and one admissible closing ground support it.
If the opener archive or rotation assignment cannot provide a valid paired
window, stop before scoring and report the exact command output.

The screen, if admissible, is recorded in both registries: one
`nfl-ats weak-signals record` row under
`per13_durability_on_production_opener`, and one `nfl-ats rotation record`
for the same assigned window. The opener child explicitly acknowledges the
2018–2025 mined-season multiplicity ledger because its assigned pool begins in
2020 and the opener archive has already supported other families.

## 4. Results (added only after the frozen look)

All figures in this section are **measured** by the commands named inline;
the source artifacts are local and immutable experiment outputs.

### 4.1 Assignment and instrument order

The opener child was declared **measured** by
`nfl-ats rotation declare --name per13_durability_on_production_opener` on
2026-09-02 with grade `opener`, inheritance
`per13_durability_on_production`, and mined-season acknowledgement. It was
assigned **measured** by `nfl-ats rotation assign --name
per13_durability_on_production_opener` to [2020, 2021], the earliest eligible
two-season opener block for that family. This is family-specific reuse: the
opener pool's global season-use count is not a prohibition under
`src/nfl_ats/rotation.py`; the inherited PER-13 close window [2011, 2013] is
the only blocked lineage window.

The frozen order was run exactly once: null artifact
`artifacts/per13_durability_on_production_opener/20260902T144411Z/results.json`,
then positive control artifact
`.../20260902T144431Z/results.json`, then screen artifact
`.../20260902T144444Z/results.json`. The **measured** null used 200
within-week permutations: mean +0.1086 points, standard deviation 0.6808,
95% permutation range [−1.10197, +1.75439], and the observed screen delta was
at the 3.0th percentile. The **measured** positive control detected +43.8596
points, week-blocked P+ 1.000, with 95% [+38.4605, +49.6613].

### 4.2 Opener confirmation

The screen artifact **measures** 456 paired opener games across 35 weeks and
two seasons. Baseline `weak_stack` accuracy was 53.7281%; candidate
`weak_stack_durability` accuracy was 52.8509%; candidate-minus-baseline was
**−0.8772 accuracy points**, week-blocked 95% [**−2.3864, +0.4464**],
`probability_positive` **0.084**. The season-blocked secondary was −1.2712 to
−0.4545 points with P+ 0.000 (two blocks, so it is reported context rather
than averaged into the primary). The arms' opener probabilities were
bit-identical on 9.01% of rows and their forced picks differed on 2.15%.

This is **unresolved_below_power**, recorded under weak-signal entry
`per13_durability_on_production_opener_ats` and rotation window
`per13_durability_on_production_opener` (both **measured** by the two record-command
outputs). The primary interval is not wholly below zero, the inherited
durability trait's split-half reliability is 0.793 (**read**,
`docs/per13_durability_prior.md` §3), and the large positive control is not
sized to bound an effect of roughly one point. No admissible closing ground
therefore applies. In EV terms the measured opener P+ of 0.084 favours the
unchanged production arm for this window; this is a research decision read,
not a wagering recommendation or an activation of the candidate.

The close and opener rows are correlated lineage and are not pooled as
independent votes. The close look's [2011, 2013] result remains in
`per13_durability_on_production_ats`; this opener confirmation is the distinct
opener-grade read required by the project's grade-at-opener rule.
