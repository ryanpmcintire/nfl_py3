# FluView home-market illness indicator, OPENER-graded confirmation on production: predeclaration

Written **before any opener-grade accuracy, cover-rate, or `probability_positive`
number against NFL outcomes exists for this comparison**, per the same rule
that governs every predeclared document in this tree
(`docs/fluview_on_production.md`, `docs/graph_team_stat_on_production.md`).
**Sections 1-6 are the predeclaration.** A dated results section is appended
after the look; it changes nothing above it.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md/CLAUDE.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
**bounded by a positive control** -- the instrument was PROVEN able to detect
an effect that size and it was absent. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

## 1. What this closes, and why it is a different question from the close-graded look

`docs/fluview_on_production.md` (section 7, results added 2026-08-31) measured
both FluView elevated-illness cells stacked on PRODUCTION `weak_stack`, at the
**close** grade, on an era-stratified window (legs 2011 and 2025):

- Away-market (that document's PRIMARY cell): pooled delta **0.000** accuracy
  points, week-blocked 95% CI **[-1.156, +1.161]**, P+ **0.403** -- close to a
  coin flip, the bare-baseline screen's lean (P+ 0.883) did not survive being
  stacked on production.
- Home-market (that document's SECONDARY cell): pooled delta **+0.969**
  accuracy points, week-blocked 95% CI **[-1.150, +3.119]**, P+ **0.792** --
  comparable to or slightly stronger than its own bare-baseline screen reading
  (P+ 0.818), and did NOT evaporate under stacking.

Per AGENTS.md's binding "grade the decision at the OPENER" rule -- a
close-graded number may never veto (or, symmetrically, canonize) a play --
that result settled nothing about play/no-play. This document is the deciding
opener-graded look the close-graded result earned: **the home-market cell,
not the away-market cell, is now the PRIMARY (and only) cell carried
forward.** This primary-cell switch is informed by the close-graded result
(disclosed here, not hidden) -- the home-market cell is the one whose
close-graded reading looked worth an opener-graded confirmation; the
away-market cell reads as a coin flip at the grade that matters and is not
re-run here (it remains recorded `unresolved_below_power`, open, at
`docs/fluview_on_production.md`; nothing about this document reopens or
revises that entry).

This document predeclares exactly **ONE** look: the home-market cell,
opener-graded.

## 2. The candidate feature and profiles, reused unchanged

Identical construction to the close-graded look, not re-derived:
`fluview_home_market_elevated` (`nfl_ats.fluview_production_feature`), the
frozen per-state top-decile threshold read from the frozen battery's own
recorded results artifact, the point-in-time-safe as-of construction
(`build_checkpoint_tables`/`attach_asof_ili`, imported from
`scripts/fluview_battery_screen.py`), the `"Home"`-location restriction
(NaN elsewhere). Same feature table,
`data/processed/game_features_weak_stack_fluview.parquet` (already built by
`scripts/build_weak_stack_fluview_table.py`; not rebuilt here). Same
candidate profile, `weak_stack_fluview_home` (production `weak_stack` plus
exactly the one new column), already registered in `src/nfl_ats/constants.py`
and `src/nfl_ats/margin.py`. Baseline is production `weak_stack`, unmodified,
on the same feature table (`weak_stack`'s own 274 columns are byte-identical
across the plain and FluView-widened files, per
`scripts/build_weak_stack_fluview_table.py`'s own additivity check).

Measured, this session, on the feature table directly: FluView home-column
coverage in 2020 is 85.9%, in 2021 is 92.3% -- well above the coverage floor,
no degenerate leg expected for this window (unlike the close-graded look's
2011 leg).

## 3. Grade: the opener, via the paired Tuesday-opener archive

**Grade = opener**, using `nfl_ats.clv.opener_pick_evaluation` -- the exact
machinery behind `docs/opener_evaluation.md`'s incumbent numbers and every
opener-graded confirmation since (`scripts/mod07_weak_stack.py`,
`scripts/surface_profile_opener_eval.py`, `scripts/ridge_alpha_promotion_eval.py`).
For every archived game with both a resolvable `tue_open` consensus and a
close (2020-2025 historical snapshot archive), one weekly-refit
`market_residual`/ridge/alpha-10 model (trained on completed games strictly
before that week's first kickoff) is evaluated with `spread_line` swapped to
the opener; the forced pick settles against the opener line
(`result - tue_open`).

**Both pick rules are reported, since the evaluator emits both natively**
(`opener_pick_evaluation`/`opener_evaluation_metrics`, per
`docs/opener_evaluation.md`'s 2026-08-19 addendum):

- **PRIMARY**: the production probability rule,
  `home_cover_probability_at_open >= 0.5` -- what `pool.py`/`backtest.py`
  actually play.
- **Secondary**: the historical sign rule, `residual_at_open > 0` -- the
  original `docs/opener_evaluation.md` protocol, reported for comparability
  with prior opener-grade runs.

Close-graded accuracy on the SAME paired games is also reported (secondary;
per AGENTS.md, "a close-graded number may never veto a play" -- reported,
never a gate).

The primary quantity is the **paired candidate-minus-baseline** forced-pick
accuracy delta, in `accuracy_points`, `pick_correct` against the settlement
line, per `nfl_ats.clv.pick_correct` (the same function `opener_pick_evaluation`
itself calls).

## 4. Rotation mechanics: a NEW opener-graded family, inheriting the close-graded one

**A family's grade is fixed for its lifetime.** Read directly from
`src/nfl_ats/rotation.py`: `Family` carries one `grade` field, set once at
`declare_family` and never changed; `_validate` looks up
`GRADE_POOLS[family.grade]` for **every** window a family ever holds, so a
family declared `close` (as `fluview_elevated_on_production` already is,
`docs/fluview_on_production.md` section 5/7) cannot legally draw an opener
window under that same name -- there is no "change a family's grade" or
"draw a second grade" operation anywhere in the module, and `declare_family`
itself refuses to redeclare an existing name
(`"Family {name!r} is already declared; declarations are append-only"`).

**The registry's own established mechanism for exactly this situation --
an opener-graded confirmation of a close-graded family's finding -- is to
declare a new family at the opener grade that `--inherits` the close-graded
parent**, disclosing the lineage explicitly rather than drawing an
unrelated window. This is not improvised for this document: it is the
existing, already-used pattern in this repository --
`best_pick_ranker_opener` (grade `opener`) inherits `best_pick_ranker`, and
`combined_stacker` (grade `opener`) inherits `mod07_weak_signal_stack`
(measured, this session, `registry/rotation_registry.json`: both entries
exist with exactly this shape). This document follows it:

- **New family name: `fluview_home_elevated_opener`.**
- **Grade: `opener`** (`GRADE_POOLS["opener"] = (2020, 2025)`, the paired
  Tuesday-opener archive era).
- **`--inherits fluview_elevated_on_production`**, disclosing that this
  family's declaration was informed by, and its eligible-season pool is
  reduced by, the close-graded parent's already-spent window (legs 2011 and
  2025 -- `_touched_seasons` walks the `inherits` chain transitively, so
  season 2025 is excluded from this family's eligible opener blocks even
  though it was touched by a *stratified* window under a different grade).
- **`--acknowledge-mined`**: `GRADE_POOLS["opener"]` (2020-2025) sits
  entirely inside `MINED_SEASONS` (2018-2025), so every opener window
  intersects the mined-season ledger and the flag is required for any
  window to be assignable at all -- exactly why all three existing
  opener-graded families in the registry carry it.
- **Window size: the grade default (2 seasons)**, not overridden --
  matching `best_pick_ranker_opener`, `combined_stacker`, and
  `mod07_weak_signal_stack`, the three existing opener-graded precedents,
  none of which override `--size`.

**Predicted assignment, simulated against the live registry before
declaring for real (not asserted as fact until confirmed by the actual CLI
call in section 7):** this family's touched-season set (via inheritance)
is `{2011, 2025}`. `eligible_blocks` for a fresh opener-graded family with no
own windows yet, floor `max(2020, MIN_ELIGIBLE_START_SEASON=2011) = 2020`,
default size 2, excludes any block containing season 2025 -- i.e. `(2024,
2025)` is blocked, but `(2020, 2021)`, `(2021, 2022)`, `(2022, 2023)`, and
`(2023, 2024)` remain eligible. `assign_window` returns the earliest, so the
predicted assignment is **`(2020, 2021)`** -- the same block three
unrelated existing opener families already hold (retirement is per-family,
per AGENTS.md's "opener windows are not scarce" correction, so this is not
a conflict). **If the CLI's actual `rotation assign` call disagrees with
this prediction, section 7 reports what actually happened, not this
prediction.**

**If any step of this mechanism is refused by the CLI (a validation error,
a missing eligible block, or any other `RegistryError`), this document
STOPS and reports the blocker rather than improvising around it** -- per
the task's own explicit contingency.

## 5. Harness: reusing the sibling scripts, one new opener-grading adaptation

**Reused unchanged, not rebuilt:**
`nfl_ats.clv.opener_pick_evaluation`/`opener_evaluation_metrics`/`pick_correct`/
`week_blocked_bootstrap` (the exact opener-grade machinery every prior opener
confirmation in this repo uses); `nfl_ats.rotation.confirmation_split` (the
registry's own forward-chained (training, window) split for a **contiguous**
window -- this family's window is NOT stratified, so `confirmation_split`
applies, not `confirmation_split_legs`), exactly as
`scripts/mod07_weak_stack.py`'s `arm()` already does: hand the evaluator
`training + window` concatenated (so every earlier completed game remains
available to the walk-forward fit, while only the window's own seasons are
scored), then filter the scored frame down to the window's seasons.
`nfl_ats.fluview_production_feature.FLUVIEW_HOME_ELEVATED_COLUMN` and the
`weak_stack_fluview_home` profile (section 2, unchanged).

**New script**, `scripts/fluview_home_elevated_opener_look.py`, adapting the
close-graded harness's instrument-check discipline
(`scripts/fluview_elevated_on_production.py`) to the opener grade (which has
no precedent instrument-check implementation yet -- `mod07_weak_stack.py` and
`surface_profile_opener_eval.py` run only the real look, no null or positive
control). Three modes, mirroring the close-graded script's own:

- **`null`** -- the realized settlement outcome (`margin_vs_open`, the
  quantity `opener_pick_evaluation` grades against) is shuffled WITHIN each
  week, 200 draws, after the real models are fit once (only the grading
  outcome is permuted, so this costs no extra model fits, matching the
  close-graded script's `permuted_margins` design exactly). Not centred on
  zero by design (preserves each week's realized home-cover rate); reported
  alongside the bootstrap-vs-zero interval, never instead of it.
- **`positive-control`** -- `fluview_home_market_elevated` is temporarily
  REPLACED by the realized `ats_margin` (a deliberate, large leak) across the
  ENTIRE scoped table (training and scoring rows alike) before calling
  `opener_pick_evaluation`, mirroring the close-graded script's
  `leak_treatment` construction exactly (there, `candidate_source[column] =
  ats_margin` for training, `candidate_scoring` for the graded week; here,
  the single concatenated `training + window` frame plays both roles inside
  `opener_pick_evaluation` itself, so leaking the one column once over the
  whole frame reaches both). Proves the full-profile ridge fit CAN detect a
  real effect of meaningful size at this window's sample, before the real
  screen is trusted.
- **`screen`** -- the real look. Spends the family's assigned window.

**Uncertainty.** `week_blocked_bootstrap`, week-blocked primary (within-week
game correlation is zero by owner mandate), season-blocked secondary
(necessarily thin at a 2-season window -- reported anyway, never averaged
with the primary, same discipline every sibling document applies).
`OPENER_BOOTSTRAP_SAMPLES = 20_000`, `OPENER_BOOTSTRAP_SEED = 20260817` --
matching `scripts/surface_profile_opener_eval.py`'s and
`docs/opener_evaluation.md`'s own opener-grade convention (not the
close-graded sibling's `graph_input_screen` constants of 1000/20260826,
since this is now an opener-graded look and should be comparable to the
other opener confirmations already in this tree, not to the close-graded
family it inherits from).

## 6. Decision rule, frozen before scoring

FORCED-PICK pool: 285 cards must be submitted either way. The decision is
expected value, never a 0.90/95% threshold -- `probability_positive` above
0.5 (under the PRIMARY, production probability rule, week-blocked) favours
adding `fluview_home_market_elevated` to the played chain over declining to.
This is the deciding, opener-graded look for this cell: unlike the
close-graded look, a resolved wrong sign here (the whole week-blocked
interval below zero) WOULD be an admissible `wrong_sign_resolved` closure
(the close-graded restriction that "a resolved wrong sign at this grade
is... never a `refuted_mechanism` closure" applies only to the close grade,
per that document's own recording section) -- and a demonstrated
positive-control bound (the instrument proven able to detect an effect the
size of the close-graded reading, and finding nothing at the opener) would
likewise be admissible. Absent either, the result is
`unresolved_below_power`, exactly as the binding taxonomy requires; the
positive control's own detected magnitude is reported before any
classification is chosen, per "numbers and intervals before any verdict."

## Recording

One `nfl-ats weak-signals record` entry, `effect_units=accuracy_points`,
`--family fluview_elevated_on_production` (the same weak-signal pooling
bucket as the close-graded cells -- disclosed as opener-graded in
`--notes`, per the task's own instruction; this is the weak-signal
*pooling* family, a different concept from the *rotation* family name
`fluview_home_elevated_opener` declared in section 4).

One `nfl-ats rotation record --name fluview_home_elevated_opener` call,
spending the assigned contiguous opener window, carrying the primary
(production probability rule, week-blocked) paired effect, interval, and
`probability_positive`; the secondary sign-rule and close-graded readings
are disclosed in the same call's `--notes`.

## 7. Results (added after the look, 2026-08-31)

**Window, confirmed not asserted.** `nfl-ats rotation declare --name
fluview_home_elevated_opener --grade opener --inherits
fluview_elevated_on_production --acknowledge-mined` then `nfl-ats rotation
assign --name fluview_home_elevated_opener` returned window **(2020,
2021)**, exactly as section 4 predicted from simulating the assignment
against the live registry before declaring (`remaining_eligible_windows: 4`
right after declare, `2` right after assign, both read back from the CLI's
own response, not asserted). The family's touched-season set via
inheritance is `{2011, 2025}`; the earliest eligible size-2 opener block
excluding those seasons is `(2020, 2021)`, the same block three unrelated
existing opener families already hold (per-family retirement, not a
conflict).

**Instrument check 1 -- positive control** (`--mode positive-control`,
`fluview_home_market_elevated` replaced by the realized `ats_margin` across
the whole scoped table): production rule delta **+43.860** accuracy points,
week-blocked P+ **1.000**, 95% [+38.147, +49.672], n=456 games, 35 weeks.
Sign rule: delta +44.298 points, P+ 1.000, 95% [+39.446, +49.333]. The
full-profile ridge fit is not blind to a real effect of meaningful size on
this window.

**Instrument check 2 -- null** (`--mode null`, 200 within-week
permutations): production rule mean **+0.357** points, sd 1.036, 95%
[-1.541, +2.412], observed -0.439 (matches the real screen's own observed
value, since the null reuses the same fixed picks). Sign rule mean -0.054
points, sd 1.237, 95% [-2.412, +2.418], observed +0.219. Both sane, finite
distributions.

**The real screen** (`--mode screen`), 456 paired games, 35 weeks, seasons
2020-2021:

| rule | grade | pooled delta | week-blocked 95% CI | week-blocked P+ | season-blocked P+ |
|---|---|---|---|---|---|
| production (PRIMARY) | opener | **-0.439** pts | [-3.091, +2.198] | **0.341** | 0.000 (degenerate, 2 blocks) |
| sign (secondary) | opener | **+0.219** pts | [-3.153, +3.433] | **0.522** | 0.749 |
| production | close (secondary) | -0.651 pts | [-3.602, +2.361] | 0.305 | -- |
| sign | close (secondary) | +1.302 pts | [-1.075, +3.672] | 0.834 | 1.000 |

Permutation-null percentiles: the primary (production-rule) observed delta
of -0.439 sits at the **20.0th** percentile of its own null (a mild
negative lean, not extreme); the secondary sign-rule observed delta of
+0.219 sits at the **56.0th** percentile (essentially the null's centre).
Home-pick rate under the production rule: baseline 21.2%, candidate 15.9%
(the candidate picks home noticeably less often); under the sign rule,
baseline 42.1%, candidate 35.8%. 29 of 456 paired picks (6.4%) disagree
between arms under the production rule.

Artifacts: `artifacts/fluview_home_elevated_opener_look/20260831T164546Z/results.json`
(screen), `artifacts/fluview_home_elevated_opener_look/20260831T164207Z/results.json`
(positive-control), `artifacts/fluview_home_elevated_opener_look/20260831T164235Z/results.json`
(null).

### What this implies for the decision, before what is wrong with it

On EV grounds -- `probability_positive` above 0.5 favours playing the
candidate, the only decision rule this project uses -- **the PRIMARY
number (production probability rule, week-blocked, opener grade) is P+
0.341, below 0.5: at this look, EV leans AGAINST adding
`fluview_home_market_elevated` to the played chain.** This is a real
reversal from the close-graded lean for this same cell (P+ 0.792,
docs/fluview_on_production.md section 7) -- the "composition is not the
signal" pattern this project has seen before with the away-market cell
now also shows up for the home-market cell once graded at the rule and
line that actually decide anything. The secondary sign rule reads close to
a coin flip in the candidate's favour (P+ 0.522, barely above 0.5), and the
close-graded reads on this same 2020-2021 archive (secondary, never a
gate) split the same way the opener reads do: production rule negative
(P+ 0.305), sign rule positive (P+ 0.834).

This is **not** a closure. The primary week-blocked interval, [-3.091,
+2.198], crosses zero, so `wrong_sign_resolved` is unavailable (that
ground requires the WHOLE interval below zero -- the season-blocked
interval's upper bound touches exactly 0.0 but that reading is degenerate
at only 2 blocks and is reported, never treated as a gate, per this
document's own section 5 discipline). The positive control proved the
harness can detect a real effect (+43.860 points) at this window's sample,
but that leaked magnitude is roughly 100x either candidate reading being
tested (-0.439 opener, +0.969 close) -- it demonstrates the instrument is
not blind, not that an effect the SIZE actually in question would be
reliably detected here, so `bounded_by_control` is not available either.
Recorded `unresolved_below_power`, exactly as the predeclared taxonomy
requires, rotation verdict `unresolved`, family `fluview_home_elevated_opener`
stays **open** (status unchanged; only its window is spent).

The honest reading, stated plainly: the close-graded home-market finding
does not clearly survive being graded the way that actually decides
anything (the opener, under the rule production actually plays); it leans
the other way at this specific look, though both the lean against and the
sign rule's lean for are weak and each interval comfortably contains zero.
This is a below-power, single-window read at only 456 games -- it does not
refute the close-graded finding (no resolved wrong sign), and per the
binding taxonomy a below-power reversal of sign is exactly the expected
shape of noise around a small or absent true effect, not evidence of a
wrong-signed mechanism. The practical EV read for THIS decision, at THIS
sample: the number that matters (production probability rule, opener
grade) does not currently support adding this feature to the played
chain; the family remains open for a future, larger, or differently
constructed look rather than closed.

**Caveats, after the numbers above, not instead of them.** (1) This
window is thin -- 456 games, 35 weeks, 2 seasons -- and the season-blocked
read is acknowledged degenerate rather than trusted. (2) The two pick
rules disagree in direction here (production rule negative, sign rule
positive), both close to a coin flip; this is disclosed, not resolved, by
this look. (3) `fl`/`ny` coverage gaps (docs/fluview_battery.md section 1)
apply to this window's seasons the same as before. (4) The away-market
cell was not re-run at the opener grade in this document -- it remains
`unresolved_below_power` at the close grade only, per section 1.

### Registry, verified by reading it back (not by trusting the CLI's own echo)

`registry/weak_signals.json`: **608 -> 609** signals (measured before and
after via `python -c "json.load(...)['signals']"` length; the new entry
read back directly shows `classification=unresolved_below_power`,
`family=fluview_elevated_on_production`, `effect=-0.4386`,
`probability_positive=0.3411`).

`registry/rotation_registry.json`: **11 -> 12** families (the new
`fluview_home_elevated_opener` family). Its window `(2020, 2021)`,
`window_kind: "contiguous"`, is now `state: "spent"`, `verdict:
"unresolved"`, `status: "open"` (unchanged -- an `unresolved` verdict spends
the window without closing the family), carrying the primary
(production-rule) pooled effect/interval/`probability_positive`; the
secondary sign-rule and close-graded readings are disclosed in the same
entry's `notes`.

### Files touched

- `docs/fluview_opener_look.md` (this document).
- `scripts/fluview_home_elevated_opener_look.py` (new: the opener-grade
  harness, `--mode {null, positive-control, screen}`, reusing
  `nfl_ats.clv.opener_pick_evaluation`/`nfl_ats.rotation.confirmation_split`
  and the already-registered `weak_stack_fluview_home` profile / frozen
  FluView feature builder unchanged).
- `scripts/fluview_home_elevated_opener_record.py` (new: reads the screen/
  positive-control/null artifacts and records both `rotation record` and
  `weak-signals record`, no hand-typed numbers).
- `registry/weak_signals.json`, `registry/rotation_registry.json` (the two
  record calls above).

No `src/nfl_ats` file was touched by this document -- the candidate
profile, feature builder, and feature table were all already registered by
the close-graded sibling lane and reused unchanged.
