# XLG-06 Stage 1: pure-CFB rookie-prior screen

Written 2026-09-01, after the fact -- `scripts/xlg06_rookie_prior_cfb_screen.py`
was committed 2026-08-18 (`git log`, commit `a3addf5`, **measured**) and run
twice the same day, but the result was never written up in `docs/` and never
recorded to `registry/weak_signals.json` until this document. Every claim
below is tagged **measured** (re-derived this session from the artifact JSON
or the script), **read** (opened this session, path:line given), **reported**
(a prior session's claim, unverified further here), or **inferred** (my
reasoning, not evidence).

## Binding closing-grounds taxonomy (verbatim, restated per AGENTS.md's rule
for any document, script, or subagent that scores or adjudicates an
experiment)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry code hard-rejects inadmissible
> closures; if a record command errors, the verdict is wrong, not the
> validator.

Also binding: within-week (here, within-cohort) correlation is not padded or
assumed; every claim is labeled by provenance; numbers and intervals are
reported before verdicts.

## What the screen asked

XLG-06 (`ROADMAP.md:180`, **read**) is "Link college usage/value, recruiting,
transfers, and draft identity to NFL players with explicit uncertainty and
decay as NFL evidence accumulates." `scripts/xlg06_rookie_prior_cfb_screen.py`
is explicitly scoped as **Stage 1 only**, a cheap pure-CFB feasibility screen
before any crosswalk or NFL work is attempted (script docstring, **read**,
`scripts/xlg06_rookie_prior_cfb_screen.py:1-115`):

> "Question: does recruiting pedigree (known before a single college snap)
> predict a true freshman's realized CFB usage in their first meaningful
> season? Reliability of the outcome construct is measured before any
> predictive claim, per AGENTS.md's binding 'split-half reliability before
> any predictive claim' rule."

Hard constraints stated in the docstring (**read**): (1) no external API
calls, locally-ingested CFB data only; (2) no portal/transfer data, no
name-based joining; (3) QB-first population phasing (RB/WR/TE reported
separately, never pooled, because `usage.overall`'s scale differs
structurally by position -- measured means QB 0.279, RB 0.113, WR 0.043, TE
0.029); (4) Stages 2-3 -- the CFB-to-NFL player crosswalk, the NFL roster
join, and any registry write -- are **explicitly out of scope for this
script**. This document supplies the missing write-up and the registry write
that Stage 1's own scope always intended to hand off to a follow-up step.

"True freshman" is operationalized as recruiting class year == CFB usage
season (script docstring, **read**): the repository has no separate
enrollment-year source to confirm redshirt/grayshirt status, so this is a
pregame-knowable approximation, not a verified enrollment date.

## The frozen gate, as the script states it

The script does not carry a single predeclared numeric pass/fail threshold
(no "promote to Stage 2 if r > X"); its gate is a **procedural** one, stated
directly in the docstring and the code path: **the outcome construct's
split-half reliability must be measured before any predictive correlation is
treated as a claim** (the AGENTS.md "split-half reliability before any
predictive claim" rule, cited verbatim in the script). Concretely, the script
runs two logically separate checks in order:

1. **Construct-facet reliability** (`construct_facet_reliability`,
   `scripts/xlg06_rookie_prior_cfb_screen.py:350-386`): a Spearman-Brown
   corrected split-half proxy -- correlating CFBD's own two disjoint
   down-type usage facets (`usage.standardDowns` vs `usage.passingDowns`)
   against each other for the same freshmen -- as a substitute for the
   canonical temporal (odd/even-week) split-half, which the docstring
   explains is **not executable** from this data (`usage` is a
   season-aggregate table with no week column, confirmed by schema
   inspection, script docstring point 2, **reported**, taken from the
   script's own comment). The docstring is explicit that this substitute
   result alone does not license invoking `no_split_half_reliability` as a
   registry closing ground, since it is not the canonical repeated-occasion
   recipe.
2. **Predictive correlation** (`predictor_outcome_correlation`,
   `scripts/xlg06_rookie_prior_cfb_screen.py:389-435`): recruiting
   `rating`/`stars` vs. realized `usage.overall`/`usage.pass`/`usage.rush`,
   QB-first, cohort-blocked bootstrap (whole recruiting-class cohorts
   resampled, not individual players -- the project's standard "week" block
   translated to this source's finest available temporal grain, a season).

What Stage 2 was supposed to be (docstring, **read**): build the actual
CFB-to-NFL player crosswalk (`build_cfb_nfl_player_crosswalk`, named but not
built), join to NFL rosters, and only then attempt an NFL-side predictive or
market claim with a registry write. None of that exists in this repository
yet (**measured**, `grep -rln "build_cfb_nfl_player_crosswalk" src/ scripts/`
returns zero hits beyond the one docstring mention).

## The two runs, verified

Two artifact directories exist, both **measured** this session by opening
`results.json` directly (not summarized from a prior write-up, since none
existed):

### Run 1 -- `artifacts/xlg06_rookie_prior_cfb/20260818T213509Z/results.json`

The original run, before a same-day CFB backfill. Cohort coverage was
`recruiting_players` 2024-2026 (local) intersected with `usage` 2023-2025
(local), leaving exactly **2 usable cohort-seasons (2024, 2025)** -- below
this project's own `MIN_BLOCKS_FOR_INTERVAL = 10` floor, so no cohort-blocked
bootstrap was computable; every interval in this run is a **player-row-blocked**
bootstrap instead (each matched player is its own resampling block, not each
cohort).

- QB construct-facet reliability (player-blocked, n=136): Pearson r=0.9244,
  **Spearman-Brown full-length reliability (Pearson) = 0.9607**
  (Spearman-rho variant 0.9176).
- QB `rating -> usage.overall` (player-blocked, n=125): Pearson r=**+0.1121**,
  95% CI **[-0.0773, +0.2926]**, `pearson_probability_positive` **0.8773**.
  A modest positive lean, interval crossing zero, on a small sample.

### Run 2 -- `artifacts/xlg06_rookie_prior_cfb/20260818T215305Z/results.json`

The same-day re-run after a backfill agent ingested `usage` 2013-2022 and
`recruiting_players` 2013-2023 through this repo's own `cfb-ingest` CLI (no
ad hoc API calls, per the script's own addendum, **reported**, taken from the
script docstring's account of the backfill). Usable cohorts expanded to
**2013-2025, 13 cohort-seasons** -- above `MIN_BLOCKS_FOR_INTERVAL`, so a
genuine **cohort-blocked** bootstrap (whole recruiting-class years resampled)
is now computable and is reported as primary; the player-row bootstrap is
kept as a secondary coherence check.

- QB construct-facet reliability (cohort-blocked, n=602, 13 blocks): Pearson
  r=0.9323, **Spearman-Brown full-length reliability (Pearson) = 0.9650**
  (Spearman-rho variant 0.9479), CI [0.9241, 0.9402]. Player-blocked
  secondary (n=602): identical point estimate and reliability (0.9650, same
  underlying r), a slightly wider CI [0.9218, 0.9424] -- the two blocking
  schemes agree on the point estimate.
- QB `rating -> usage.overall` (cohort-blocked, primary, n=557, 13 blocks):
  Pearson r = **-0.0018** (exactly -0.00175181), 95% CI **[-0.0920, +0.0882]**,
  `pearson_probability_positive` **0.4842** (audit's -0.484 wording; re-derived
  here to 0.48415). Spearman rho -0.0101, 95% CI [-0.0969, +0.0844], P+
  0.4043. Player-blocked secondary (n=557) reads nearly identically:
  r=-0.0018, CI [-0.0843, +0.0800], P+ 0.4799 -- the two blocking schemes
  agree with each other within run 2.

### Do the two runs agree? No -- and the disagreement is informative, not a
red flag

This is not a same-test-run-twice reproduction check; it is a **scope
expansion**. Run 1's population (2 cohorts, n=125 for the QB predictive cell)
leaned mildly positive (P+ 0.8773, r=+0.112, CI still crossing zero). Run 2,
after the cohort backfill (13 cohorts, n=557, a 4.5x larger sample with a
methodologically sounder cohort-blocked interval instead of a treat-every-row-
as-independent player bootstrap), reads as a **dead coin flip** (P+ 0.4842,
r=-0.0018). More data moved the estimate *toward* zero, not away from it --
exactly the pattern expected if run 1's positive lean was small-sample noise
rather than a real effect being confirmed. The **reliability** side of the
picture, by contrast, genuinely does agree across both runs: QB construct-facet
Spearman-Brown reliability was 0.9607 on n=136 (run 1) and 0.9650 on n=602
(run 2) -- stable and high regardless of cohort depth. **The trait is
reliably measured; the correlation is not reliably positive.** Run 2, with
the larger and more defensible cohort-blocked interval, is treated as the
current primary reading; run 1 is superseded, not confirmed.

Per-cohort trend (run 2, `primary_qb_per_cohort_trend`, **measured**,
player-blocked within each single cohort year, descriptive only -- not
itself a blocked interval per the script's own labeling): the 13 individual
cohort-year point estimates range from r=-0.215 (2023) to r=+0.259 (2019),
with no monotonic trend and every single cohort's own 95% CI crossing zero.
This is consistent with the pooled null read, not a hidden pattern the pooled
number is washing out.

Secondary positions (run 2, cohort-blocked, `rating -> usage.overall`,
**measured**, reported for coherence only, never pooled with QB per the
script's own position-scale caveat): RB r=+0.0644 CI [+0.0187,+0.1153] P+
0.9971; WR r=+0.0473 CI [-0.0044,+0.1023] P+ 0.9646; TE r=+0.0037 CI
[-0.0316,+0.1065] P+ 0.6082. RB's interval sits entirely positive at this
n (1,204) -- worth noting as a genuinely different-positioned reading from
QB's null, but out of scope for QB-first Stage 1's primary claim and not
itself claimed here as a finding (no reliability-before-claim check was run
for RB's own outcome construct at this depth beyond what's reported above).

## What the gate outcome implies for Stage 2, under this project's rules

Applying the taxonomy above to the QB primary cell (recruiting rating vs.
true-freshman usage, run 2, cohort-blocked, n=557): Pearson r=-0.0018, 95% CI
[-0.0920, +0.0882].

- **`wrong_sign_resolved` does not apply.** The interval straddles zero; it
  is not "resolved wrong-signed" (that requires the whole interval on the
  wrong side of zero, and there is no predeclared "wrong" sign here in the
  first place -- either sign would have supported the prior-value hypothesis).
- **`no_split_half_reliability` does not apply, and this is the substantive
  finding of this write-up.** The construct itself is reliably measured
  (Spearman-Brown 0.965, stable across both runs and both blocking schemes)
  -- this is NOT a case of "we can't tell because the trait is noise." A
  near-zero correlation riding on top of a highly reliable predictor is a
  materially different result than a near-zero correlation on an unreliable
  one: it says the *measurement* is trustworthy and the *relationship* (at
  this n, this construct, this outcome) reads flat, not that nothing could
  be concluded either way for instrumentation reasons.
- **No positive control was run in this script** at a size matched to this
  question, so `bounded_by_control` is not available either.
- **Neither admissible closing ground is met. This is `unresolved_below_power`.**
  A reliably-measured near-zero correlation at n=557 (13 cohort blocks) does
  not rule out a real small effect that this instrument's block count cannot
  yet resolve -- the per-AGENTS.md default. It also does not, on its own,
  justify treating recruiting pedigree as a validated Stage-2 prior input: no
  finding here says "build the crosswalk because this works," and none says
  "never build it because this is refuted." The honest state is: the pure-CFB
  predictor-outcome link for QB, at this sample size, does not clear an EV
  bar to promote on its own, and does not close the door on XLG-06 either.
  Before any Stage 2 (crosswalk/NFL work) is funded on the strength of *this*
  screen alone, it would need either (a) a larger n (more cohorts, once
  available, or a secondary predictor/outcome combination already computed
  above -- `rating -> usage.pass` reads P+ 0.5356 (r=+0.0047), `stars ->
  usage.overall` P+ 0.5853 (r=+0.0068), all similarly flat) or (b) a
  different predictor construct
  entirely; the RB positive read above (P+ 0.9971) is a candidate worth a
  dedicated look, not folded into this QB-scoped write-up.

## Proposed ROADMAP XLG-06 status

Move from ⬜ to 🚧: Stage 1 (pure-CFB screen) is built, ran twice, and is now
written up and recorded. Stages 2-3 (crosswalk, NFL roster join, registry
write for an NFL-side claim) remain unbuilt, exactly as the script's own
docstring scopes them. See the addendum text below.

## Registry record

Recorded via the cross-process-locked wrapper (command and output pasted in
the session report). `--league cfb` (this is a pure-CFB screen, no NFL join).
`--family xlg06_rookie_prior_stage1`.

**Effect-units mapping, stated explicitly.** The CLI's `--effect-units`
choices are `{ats_points, accuracy_points, brier, log_loss, mae}` -- none of
which is a correlation coefficient's native unit. This screen has no ATS
pick, no probability forecast graded against an outcome, and no error metric
in any of those senses; its effect is a raw Pearson r on a [-1, +1] scale.
The closest available option is treated as a plain numeric container:
`accuracy_points` is used **only** as a numeric field, with the raw Pearson r
(-0.0018) recorded as-is (NOT scaled by 100 the way this project's other
`accuracy_points` entries are, since that scaling convention exists to turn a
win-rate fraction into a point figure, which does not apply to a correlation
coefficient). This is flagged in both `--notes` and `--plain-summary` so a
future reader does not mistake -0.0018 for a point-scale accuracy effect. The
unit actually wanted here, and not currently in the CLI's supported set, is a
**`correlation` (or `pearson_r`) effect unit** with a native [-1, +1] range --
worth adding to `weak_signals.py`'s `EFFECT_UNITS` if this project accumulates
more correlation-gate screens like this one (XLG-05/XLG-07 CFB-side work is
plausibly the same shape).

## RB cell: dedicated look (2026-09-01, WP46)

### Disclosure: the sign was already seen

This is a follow-up on the RB read reported above (**read**, this document,
"Secondary positions" paragraph): `rating -> usage.overall`, run 2,
cohort-blocked, n=1204, r=+0.0644, 95% CI [+0.0187, +0.1153], P+ 0.9971 --
computed inside the QB-scoped screen's own Step 6 before this dedicated look
was declared. The cell below is frozen (population, predictor, outcome,
statistic, reliability method, positive-control design) BEFORE any new
outcome number for this look is generated. That RB sign was a **secondary
read in the prior run**, so what follows is a **sequential confirmation of an
already-observed sign, not a first look** -- stated here so a future reader
does not mistake the run below for a blind first test. Run 1's RB read
(player-blocked, n=253, r=+0.0732, CI [-0.0501, +0.2033], P+ 0.878, **read**,
`artifacts/xlg06_rookie_prior_cfb/20260818T213509Z/results.json`
`secondary_position_correlations.RB`) also leaned positive with a wider,
zero-crossing interval -- both prior observations of this cell pointed the
same direction, which this dedicated look does not treat as independent
confirmation (same underlying data, overlapping cohorts) but does disclose.

### The cell, exactly as the original script already computes it (no redesign)

- **Population**: true-freshman running backs -- `position_usage == "RB"`
  rows of the same `matched` population `build_true_freshman_population`
  already builds for every position (recruiting class year == CFB usage
  season, recruit `athleteId` joined to `usage.id` within that season).
  Cohort-years available: 2013-2025 (13 usable cohorts, the same
  post-backfill span the QB primary cell uses; `join_diagnostics.cohort_years_usable`,
  identical for every position since the join is position-agnostic).
- **Predictor**: recruiting `rating` (`PREDICTOR_COLUMNS[0]` in the original
  script; the QB primary cell's own predictor, and the one the RB secondary
  read above used).
- **Outcome**: `usage.overall` -- CFBD's season-level composite usage-share
  column (`OUTCOME_COLUMNS[0]`), the same outcome the QB primary cell and the
  RB/WR/TE secondary reads all use.
- **Statistic**: Pearson r (Spearman rho reported alongside), cohort-blocked
  percentile bootstrap as primary (`predictor_outcome_correlation(...,
  block="cohort")`, whole recruiting-class-year cohorts resampled) with the
  player-blocked bootstrap (`block="player"`) as a secondary coherence check
  -- identical machinery to every other position's cell, unmodified. 20,000
  bootstrap samples (`BOOTSTRAP_SAMPLES`). `--mode screen` reuses the exact
  seeds the original script's Step 6 already assigned to RB (cohort seed
  4000, player seed 4100 -- RB is index 0 of `SECONDARY_POSITIONS`), so it
  reproduces the already-computed RB numbers rather than drawing a fresh
  bootstrap.
- **Reliability**: the same construct-facet split-half substitute the QB cell
  uses (`construct_facet_reliability`: `usage.standardDowns` vs
  `usage.passingDowns`, Spearman-Brown corrected, cohort-blocked primary +
  player-blocked secondary) -- explicitly NOT the canonical temporal
  odd/even-week recipe, for the same reason the QB section already gives
  (`usage` has no week column). `--mode screen` reuses the seeds the original
  script's Step 3 already assigned to RB (cohort seed 1001, player seed
  1101 -- RB is index 1 of `(PRIMARY_POSITION, *SECONDARY_POSITIONS)`).
- **Positive control**: a code-path sanity check on the bootstrap correlation
  machinery, not a `bounded_by_control`-eligible control sized to this cell's
  own small effect. `scripts/xlg06_rb_stage1.py --mode positive-control`
  overwrites each RB row's `usage.overall` with a deliberately LEAKED,
  monotone (rank-preserving, non-linear) function of `rating` -- the
  predictor's within-RB rank, min-max rescaled into the RB population's own
  observed outcome range -- then runs the identical
  `predictor_outcome_correlation` call. A working instrument must report
  Pearson r and P+ driven toward 1 on this leaked input; it says nothing
  about whether the instrument could detect an effect the size of the real
  RB reading (r~0.06), so it can never be cited as the admissible
  `positive_control_bound` closing ground for this cell.
- **Recording plan**: family `xlg06_rookie_prior_stage1` (same family as the
  QB cell -- both are Stage-1 pure-CFB predictor/outcome cells on the same
  population and instrument, differing only in position), name
  `xlg06_rookie_prior_stage1_rb`, `--effect-units correlation` (the unit this
  project added since the QB cell was first recorded under a numeric-container
  workaround), `--league cfb`, `--season-start 2013 --season-end 2025`,
  reliability filled from the RB construct-facet cohort-blocked Spearman-Brown
  reading, classification per the taxonomy below -- per this session's binding
  instruction, a resolved POSITIVE result (this interval sits entirely
  positive) has no terminal state either: it is recorded
  `unresolved_below_power` with its interval and P+ stated, never promoted to
  a closing ground on the strength of a positive sign alone. What a
  resolved-positive RB cell implies for Stage 2 is an EV question (P+ > 0.5),
  addressed in the results section below, not a validation claim: a positive
  resolved r here is Stage-1 evidence for whether building an RB usage prior
  is worth attempting at Stage 2 (the crosswalk + NFL join), never itself an
  NFL-side or wagering claim -- Stage 2 remains unbuilt regardless of this
  cell's outcome.

### Results (measured this session)

`scripts/xlg06_rb_stage1.py` was run twice per mode this session (an initial
pass, then a second pass after adding the per-cohort trend diagnostic below
-- the correlation and reliability numbers are byte-identical across both
passes, since they are deterministic given the same seeds and the same local
snapshots). Numbers below are from the fuller pair: **measured**,
`artifacts/xlg06_rookie_prior_cfb/rb_20260901T195232Z/results.json` (`--mode
screen`) and `artifacts/xlg06_rookie_prior_cfb/rb_20260901T195447Z/results.json`
(`--mode positive-control`).

**Positive control** (`usage.overall` leaked as a rank-monotone function of
`rating`, RB rows only): cohort-blocked Pearson r=**0.9690**, 95% CI
[0.9646, 0.9725]; player-blocked Pearson r=**0.9690**, CI [0.9663, 0.9717];
Spearman rho=**1.0** exactly in both blockings; P+ = 1.0 in both. The
instrument correctly reports a strong effect when one is deliberately
engineered into the data -- the code path is not silently broken. As stated
in the cell definition above, this is a coarse sanity check, not a control
sized to the real ~0.06 effect, and is not cited as this cell's closing
ground.

**Reliability** (construct-facet split-half substitute, RB, same recipe as
QB): cohort-blocked primary Spearman-Brown(Pearson)=**0.8017**, n=1273, 13
blocks, CI [0.6276, 0.7061] (on the underlying facet-facet r); player-blocked
secondary identical point estimate (0.8017), CI [0.6251, 0.7099]. Lower than
QB's 0.965, but still a clearly reliable trait, not noise -- `no_split_half_reliability`
is inadmissible for the same reason it was inadmissible for QB.

**Predictive correlation, `rating -> usage.overall`, RB**: cohort-blocked
primary (n=1204, 13 blocks) Pearson r=**+0.06443**, 95% CI **[+0.01874,
+0.11532]**, P+ **0.99705**; Spearman rho=+0.07482, CI [+0.03614, +0.12408],
P+ 0.99995. Player-blocked secondary (n=1204): r=+0.06443 (identical point
estimate), CI [+0.00438, +0.12327], P+ 0.98175 -- the two blocking schemes
agree. This exactly reproduces the already-computed run-2 RB secondary read
(`artifacts/xlg06_rookie_prior_cfb/20260818T215305Z/results.json`) to the
last printed digit, confirmed by `tests/test_xlg06_rb_stage1.py`'s
1e-6-tolerance reproduction test.

**Per-cohort trend** (13 individual cohort years, player-blocked within each
single cohort, descriptive only, **measured**,
`artifacts/xlg06_rookie_prior_cfb/rb_20260901T195232Z/results.json`
`cohort_trend`):

| cohort | r | 95% CI | P+ | n |
|---|---|---|---|---|
| 2013 | -0.0902 | [-0.3569, +0.2232] | 0.2796 | 58 |
| 2014 | +0.1797 | [-0.0431, +0.3996] | 0.9428 | 68 |
| 2015 | +0.0995 | [-0.1497, +0.3291] | 0.7834 | 64 |
| 2016 | +0.1548 | [-0.0294, +0.3429] | 0.9511 | 80 |
| 2017 | +0.0365 | [-0.2026, +0.2751] | 0.6208 | 85 |
| 2018 | +0.1282 | [-0.0482, +0.3009] | 0.9242 | 97 |
| 2019 | +0.0130 | [-0.1722, +0.2006] | 0.5578 | 115 |
| 2020 | +0.1576 | [-0.0256, +0.3385] | 0.9540 | 104 |
| 2021 | +0.1525 | [-0.1264, +0.4201] | 0.8540 | 71 |
| 2022 | +0.2079 | [+0.0457, +0.3660] | 0.9930 | 107 |
| 2023 | +0.1485 | [-0.0042, +0.2998] | 0.9718 | 102 |
| 2024 | +0.0274 | [-0.1526, +0.2299] | 0.6152 | 119 |
| 2025 | +0.0848 | [-0.0857, +0.2640] | 0.8341 | 134 |

12 of 13 individual cohort years read a positive point estimate (only 2013
is negative); every cohort's own individual 95% CI crosses zero except 2022,
whose CI sits entirely positive. This is a **materially more consistent**
per-cohort pattern than the QB cell's own per-cohort trend reported above
(13 cohorts ranging r=-0.215 to +0.259, no consistent sign, "no monotonic
trend"). The RB read is not one lucky pooled draw riding on a couple of
outlier cohort-years; the positive lean recurs across most of a 13-year span.

### What this implies for the decision, before what is wrong with it

At the pooled, cohort-blocked, primary reading: P+ = **0.99705** that
recruiting rating positively predicts a true-freshman RB's realized CFB
usage. Per this project's binding EV rule, a decision is EV (P+ > 0.5), not a
promotion-bar classification -- and 0.99705 clears that bar by a wide margin,
independent of whether the interval also happens to exclude zero (it does
here, but that is not what makes this decision go one way rather than the
other). Building the RB usage prior at Stage 2 (the CFB-to-NFL crosswalk +
NFL roster join, still unbuilt for every position) is the higher-EV action
on this evidence, not a marginal call. The per-cohort trend reinforces this
rather than undercutting it: unlike QB's flat, sign-inconsistent per-cohort
pattern, RB's per-cohort point estimates are positive in 12 of 13 years, a
pattern a single-cohort artifact of the pooled estimate would not produce.

What this evidence does NOT establish, stated plainly rather than folded into
a hedge: this is still a Stage-1, pure-CFB, small (r~0.06) correlation on a
reliability-limited construct (0.80, lower than QB's 0.965) -- it says
recruiting pedigree is *worth building a prior around* for RBs, not that the
eventual NFL-side prior will itself carry a usable NFL edge; that question is
unanswered until Stage 2 exists and is tested against NFL data. The
positive-control result bounds nothing about the real effect's size (stated
in the cell definition above) -- it only rules out "the harness is broken"
as an explanation for the near-zero QB read, and, by extension, confirms this
RB read is a real measured signal from the same trustworthy instrument, not
an artifact of a broken correlation calculation. Classification stays
`unresolved_below_power` (recorded, `nfl-ats weak-signals record`, family
`xlg06_rookie_prior_stage1`, name `xlg06_rookie_prior_stage1_rb`) because
neither admissible closing ground (`wrong_sign_resolved`,
`no_split_half_reliability`, `positive_control_bound`) applies -- classification
and the EV decision are separate questions, and this document's answer to
the EV question is: **yes, Stage 2 for RBs is warranted on EV**, subject to
the scope, cost, and prioritization of building the crosswalk itself, which
this document does not evaluate.

### Registry record (RB cell)

Recorded via the cross-process-locked wrapper. `--family
xlg06_rookie_prior_stage1`, `--effect-units correlation` (this project's
dedicated correlation unit, added since the QB cell's original
numeric-container workaround), `--league cfb`, `--season-start 2013
--season-end 2025`, `--reliability 0.8017`, `--classification
unresolved_below_power` (no `--closing-ground`, per the taxonomy applied
above). Effect **0.0644**, interval **[0.0187, 0.1153]**,
`--probability-positive 0.99705`, `--sample-games 1204 --sample-blocks 13`.
