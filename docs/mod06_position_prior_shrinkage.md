# MOD-06's live arm as a feature_arm: position-prior shrinkage, NFL screen

Written 2026-08-19, **before** the enriched feature table is built or
`nfl-ats experiment run` is invoked (predeclaration required by
`docs/experiment_pipeline.md` and the "family must be declared before the
signs are seen" discipline in `AGENTS.md`). This tests MOD-06's one
surviving live arm -- unit-level shrinkage toward a position prior in the
player-value features -- as a training-time feature swap on the production
`weak_stack`/`market_residual` model.

## Binding closing-grounds taxonomy (verbatim; every verdict on this run must follow it)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

## What already exists: read this before treating this as a first look

**Read**, `registry/weak_signals.json` entry `mod06_js_shrinkage_position_prior_cfb`
(source `artifacts/cfb_james_stein_unit/20260818T213139Z`, script
`scripts/cfb_james_stein_unit_screen.py`): a full James-Stein (tau-derived
`b_i`, not the simplified design below) unit-value shrinkage screen already
ran on CFB on 2026-08-18. Its own recorded numbers: week-blocked 95% CI
**[-1.019, -0.043] accuracy points, `probability_positive` 0.0173**, 199
blocks, 8,933 games. A reviewer explicitly **refused terminal closure**
despite the whole interval sitting below zero: re-crossing zero needs only
a 1.082x widening, inside the documented 1.003-1.099x honest-refit band
(`docs/estimation_variance.md` Part II) -- so per the runner's own
mechanical classifier (`docs/experiment_pipeline.md`, "Mechanical
classification"), this does NOT clear `wrong_sign_resolved` (which requires
>1.099x). Classification stands as `unresolved_below_power`.

Per this file's own binding taxonomy above, that reading is **not** a
refutation and does not close this line of work. But ROADMAP's MOD-06 row
also declares a resource-discipline gate -- "Screen on CFB at
`probability_positive >= 0.75` before any NFL window" -- and the existing
CFB read is 43x short of that bar, in the wrong direction. Two things are
true at once and both are stated plainly rather than picking one:

1. This is **not** grounds to abandon or refuse to build the NFL arm (that
   would be exactly the crossing-zero-as-rejection error this file's own
   taxonomy exists to prevent, applied one level up).
2. It **is** a real, negative, measured prior data point that this
   predeclaration is obligated to carry forward rather than bury. The CFB
   screen's own `notes` field flags design issues specific to that attempt
   (the untuned production `alpha=10` confound; "single accuracy arm
   design: no direct old-vs-new accuracy comparison exists" -- it compared
   {baseline model with no skill-unit feature at all} against {baseline +
   NEW James-Stein feature}, conflating "add a feature" with "change the
   feature's shrinkage target"). The design below is deliberately built to
   **not** repeat that specific confound (see "Why this design differs from
   the CFB screen" below).

**Why CFB cannot be genuinely re-screened for the isolated question this
predeclaration asks:** the existing CFB `skill_unit_value` construct
(`scripts/cfb_james_stein_unit_screen.py`) has no OLD-shrink-to-zero
counterpart wired into `cfb_features.py`'s canonical build at all (its own
notes: "single-arm design"), and CFB has no defensive participant credits
wired into the canonical pipeline either (verified in that script's own
docstring, item 4). Re-running a clean isolated-variable CFB comparison
would mean building a second CFB feature pipeline from scratch, not
re-using either existing arm -- out of scope for a screen of the already-
implemented NFL production pipeline. This NFL screen proceeds directly,
carrying the CFB read above as acknowledged prior evidence, not as a
cleared gate.

## Why this design differs from the CFB screen

The CFB screen changed two things at once: it added a feature that did not
exist in the baseline at all, AND it used James-Stein shrinkage for it. A
negative or positive result there cannot tell you which change did the
work. This NFL screen isolates exactly one variable:

- **Baseline arm** already contains the metric (`injury_skill_epa_value_lost`,
  `injury_defense_disruption_value_lost` -- the `player_values` family,
  `PLAYER_VALUE_STATE_METRICS` in `src/nfl_ats/constants.py`), shrunk to
  zero, exactly as production computes it today.
- **Candidate arm** contains the SAME two underlying quantities (same
  numerator, denominator, career count, injury severity, role share --
  nothing else in the computation changes), with ONLY the shrinkage target
  swapped from zero to a position/channel prior.

This is the single-variable comparison the CFB screen's own notes say it
lacked.

## Exact formula

Production (`src/nfl_ats/players.py::_player_value_rate`, unchanged,
default `value_shrinkage_target="zero"`):

```
reliability_i(t) = career_i(t) / (career_i(t) + value_prior_snaps)
value_i(t) = 100 * (numerator_i(t) / denominator_i(t)) * reliability_i(t)
```

where `numerator`/`denominator`/`career` are one of two channels already
computed by the existing EWMA player-value state machine
(`_update_player_value_state`, `value_span=16` EWMA):

- **skill channel**: `skill_epa` (rushing_epa + receiving_epa, QB excluded)
  / `offense_snaps`, career = `career_offense_snaps`.
- **defense channel**: `defense_disruption` (weighted tackles-for-loss,
  forced fumbles, sacks, QB hits, INTs, passes defended) / `defense_snaps`,
  career = `career_defense_snaps`.

`value_prior_snaps = 200.0` (production default) is the reliability
half-life: at 200 career snaps a player's own rate and zero are weighted
equally.

**Candidate** (`src/nfl_ats/players.py::_player_value_rate_toward_prior`,
new, opt-in via `value_shrinkage_target="position_prior"`):

```
prior_c(t) = mean_{j in experienced_c(t)} [ 100 * numerator_j(t) / denominator_j(t) ]
value_i(t) = reliability_i(t) * raw_rate_i(t) + (1 - reliability_i(t)) * prior_c(t)
```

where `experienced_c(t) = { j : career_j(t) >= value_prior_snaps AND
denominator_j(t) > 0 }` -- the league-wide pool of players in channel `c`
(skill or defense) who have themselves cleared the same experience bar the
reliability weight already uses, evaluated from `player_value_states`
strictly before the current game's own snaps update it (the identical
point-in-time-safety property `_injury_value_features` already relies on
for the injured players themselves; see "Point-in-time safety" below).
`prior_c(t)` falls back to `0.0` -- bit-identical to the baseline -- when
`|experienced_c(t)| < value_js_prior_pool_minimum`.

`_player_value_rate` is the special case `prior_mean == 0`, so the
candidate is a strict generalization, not a different metric: setting the
prior to zero everywhere reproduces production exactly.

## Exact parameterization, and what is derived vs. reused

- **`value_prior_snaps = 200.0`** -- **reused unchanged**, deliberately not
  re-derived. This is the one thing this predeclaration does NOT touch,
  because touching it simultaneously with the shrinkage target would
  reintroduce the CFB screen's own two-variables-at-once confound. It gates
  both the reliability weight (unchanged, production value) and the
  "experienced" pool-membership threshold for `prior_c(t)` -- reusing the
  same constant for both is a deliberate choice, not an oversight: it means
  a player enters the prior-defining pool at exactly the point their own
  weight on zero-vs-prior first exceeds 50%.
- **`prior_c(t)` itself carries no hand-picked target constant.** This is
  the actual answer to "derive the shrinkage weight/target from the data
  rather than hand-picking a constant": the shrinkage TARGET (what MOD-06's
  hypothesis is actually about) is recomputed from data at every game, not
  fixed. The reliability WEIGHT formula is intentionally left as production
  computes it today (see above) to isolate the one variable under test.
- **`value_js_prior_pool_minimum = 20`** -- reused verbatim from the
  reviewed CFB precedent's `MIN_EXPERIENCED_POOL = 20`
  (`scripts/cfb_james_stein_unit_screen.py`), not re-derived for this run.
  Flagged as an underived constant, same as the CFB screen flagged it,
  rather than silently treated as settled.
- **Known simplification vs. the CFB precedent, stated plainly:** the CFB
  screen derives a full James-Stein shrinkage weight
  (`b_i = tau_between / (tau_between + tau_within/career_i)`, both tau
  terms estimated from data at a weekly cadence) instead of reusing the
  production `career/(career+200)` weight. This NFL screen does NOT do
  that -- it holds the weight fixed specifically so the isolated
  before/after delta is attributable to the target change alone. A
  full James-Stein weight re-derivation for NFL is future work, not
  bundled into this predeclaration.
- **Known simplification, no season/recency gate on the prior pool:**
  `player_value_states` carries no season-of-last-update field, so
  `experienced_c(t)` is not filtered for recency -- a player who cleared
  200 career snaps years ago and never appeared again remains eligible for
  the pool with their last-known (frozen) EWMA rate. Expected impact is
  small and diluting rather than sign-reversing: the pool is a MEAN over
  what is normally dozens to low hundreds of currently-active plus
  recently-retired players per channel by mid-season, and the CFB
  precedent's own `last_season`/`missed_streak` gating this deliberately
  skips is a materially larger engineering lift inside the shared,
  production-critical `enrich_with_player_features` function than this
  screen's scope justifies. Flagged, not fixed.

## Point-in-time safety

`prior_c(t)` is computed once per game (both home and away read the same
snapshot -- `player_value_states` does not change between them), from
`player_value_states` exactly as it stands when `_injury_value_features` is
called for that game -- i.e. strictly before that same game's own snaps/
production update `player_value_states` later in the same loop iteration.
This is the same ordering guarantee `_injury_value_features` already
provides for the injured players' own rates (see the existing
`test_player_value_uses_only_prior_game_stats`); the new
`test_position_prior_shrinkage_uses_only_prior_game_stats` in
`tests/test_players.py` tests it directly for the new prior mechanism:
modifying a week's own production leaves that SAME week's already-emitted
feature unchanged, while the NEXT week's feature (whose prior snapshot is
taken after that update) does move -- proving both the leak-safety and that
the modification was visible to the pipeline at all.

## Implementation: opt-in code path, default bit-identical

`src/nfl_ats/players.py::enrich_with_player_features` gained two new
keyword-only parameters, both defaulting to today's production behaviour:

- `value_shrinkage_target: Literal["zero", "position_prior"] = "zero"`
- `value_js_prior_pool_minimum: int = 20`

When `value_shrinkage_target="zero"` (the default, never overridden by any
existing caller), the code calls the SAME, untouched `_player_value_rate`
function production has always called -- no new code executes on that path
at all, so the default output is bit-identical by construction, not merely
"tested to match". `tests/test_players.py::test_value_shrinkage_target_zero_is_bit_identical_to_default`
asserts this directly (`pd.testing.assert_frame_equal`, not `approx`).

Registered as a new `MarginFeatureProfile`, `"weak_stack_js_prior"`
(`src/nfl_ats/margin.py`, `src/nfl_ats/constants.py`): `weak_stack` with the
`player_values` family (`diff_injury_skill_epa_value_lost`,
`diff_injury_defense_disruption_value_lost`) replaced by
`player_values_js_prior` (`diff_injury_skill_epa_value_lost_js_prior`,
`diff_injury_defense_disruption_value_lost_js_prior`) -- distinct column
names so both arms can be fit from ONE shared features file (the
`feature_arm` runner's own constraint; see
`docs/experiment_pipeline.md`, "the SAME enriched file... passed via
`experiment run`'s `--features` override", the precedent
`surface_switch_feature_arm` already established) without one arm
overwriting the other's values under an identical column name. Every other
family (bias, QB, injuries, continuity, market/context/elo/offense/defense)
is untouched between the two arms.

## Data source for this run

Mirrors `docs/surface_switch_feature_arm.md`'s precedent exactly: rather
than re-running the full base -> pbp -> learned-availability build chain,
`scripts/mod06_position_prior_shrinkage_build.py` re-runs ONLY
`enrich_with_player_features` a second time, with
`value_shrinkage_target="position_prior"`, using the IDENTICAL inputs
(**read**, `data/processed/game_features_weak_stack.manifest.json`):
`source_features=data/processed/game_features_pbp.parquet`,
`source_player_snapshot=20260812T200527Z`,
`source_pbp_snapshot=20260812T142851Z`,
`source_player_value_snapshot=20260813T121050Z`,
`data/processed/weak_stack_availability_rates.parquet` (the already-fitted
learned availability rates), `decision_hours_before_kickoff=24`, and every
other parameter at its CLI/function default (`role_span=8`, `qb_span=12`,
`qb_min_dropbacks=20`, `offseason_retention=0.75`, `value_span=16`,
`value_prior_snaps=200.0` -- all confirmed to match
`cli.py`'s `build-learned-availability-features` subcommand defaults). The
six new `_js_prior`-suffixed home/away/diff columns are merged by
`game_id` onto a COPY of the existing, already-built
`data/processed/game_features_weak_stack.parquet` (never overwritten),
written to a NEW file, `data/processed/game_features_weak_stack_js_prior.parquet`.
The build script also re-derives the `value_shrinkage_target="zero"`
columns from the same re-run and asserts them BYTE-IDENTICAL to the
existing table's own `injury_skill_epa_value_lost`/
`injury_defense_disruption_value_lost` columns before proceeding -- a
direct check that this reproduction is exact, not merely assumed.

## Declared exactly (read before any result exists)

- **Baseline arm**: `feature_profile="weak_stack"`, `ridge_alpha=10.0` --
  the active production configuration (**read**,
  `artifacts/active_ats_model.json`: `method="market_residual"`,
  `feature_profile="weak_stack"`, `ridge_alpha=10.0`).
- **Candidate arm**: `feature_profile="weak_stack_js_prior"`,
  `ridge_alpha=10.0`.
- **Grade**: `close` only. `feature_arm` has no `opener` implementation
  (**read**, `docs/experiment_pipeline.md`).
- **Population**: `seasons=[2018, 2025]`, league `nfl` -- **read** from this
  repo's established full-history `weak_stack`/`market_residual`
  close-grade convention (`scripts/ridge_alpha_promotion_eval.py`'s
  `--nflverse-start-season` default 2018;
  `surface_switch_feature_arm.json` uses the identical window), not a
  restriction invented for this run.
- **Endpoints**: primary `accuracy`; secondary `brier`, `logloss`
  (reported, never gating the registry `effect`).
- **Blocking**: primary `week`, secondary `season`.
- **Samples**: 20,000.
- **Seed**: `20260819`, fixed, no wall-clock nondeterminism.
- **Reliability check**: `method="not_applicable"` -- a feature-arm model
  comparison has no per-entity trait to split-half;
  `experiment_runner.py` refuses `split_half` for `feature_arm` outright.

## Reused-window acknowledgment (rotation_registry.md rule 6)

**Read**, `docs/rotation_registry.md` rule 6: a family that is a variant of
an existing line of work inherits that line's spent windows at
declaration; any window intersecting 2018-2025 requires acknowledging the
~130-150-look ledger. `experiment_runner.py`'s `feature_arm` path does not
route through `nfl_ats.rotation` at all (**measured**, `select:` grep of
`src/nfl_ats/experiment_runner.py` finds no `rotation` import), so there is
no formal rotation-registry window spent here -- but `[2018, 2025]` sits
squarely inside the acknowledged pool. **This run additionally carries the
CFB `mod06_js_shrinkage_position_prior_cfb` read as correlated prior
evidence for the same underlying mechanism** (different league, different
feature construct, but the same shrinkage-target hypothesis) -- per
AGENTS.md, a reused window/correlated prior "carries a stated discount, not
a ban."

## What this run does and does not decide

A positive, zero-excluding result promotes `weak_stack_js_prior` to a
serious production candidate (same bar as any other `feature_arm` win). A
`probability_positive` short of that, with an interval crossing zero, is
**not** a rejection -- it is recorded `unresolved_below_power` with its
`probability_positive`, per this file's own taxonomy above. A resolved
wrong-sign result on THIS isolated single-variable NFL comparison would be
a materially stronger closing signal than the CFB screen's own (which the
reviewer already declined to treat as closing, per the honest-refit-
widening rule) -- but even then, closure requires the runner's own
mechanical classifier to clear `wrong_sign_resolved` (>1.099x widening to
re-cross zero), not a human's reading of the point estimate. Promoting
`weak_stack_js_prior` into `artifacts/active_ats_model.json` is a separate,
later owner decision this predeclaration does not make. Per AGENTS.md's
"promotion bar is not a decision bar": the pool is forced picks, so
`probability_positive` above 0.5 favours playing the candidate, full stop
-- a promotion bar governs what documentation may CLAIM, never what gets
played.

## Result (measured 2026-08-19)

**Measured** (`nfl-ats experiment run registry/experiment_specs/mod06_position_prior_shrinkage.json
--features data/processed/game_features_weak_stack_js_prior.parquet`, recorded
to `registry/weak_signals.json` as `mod06_position_prior_shrinkage`, artifact
`artifacts/experiment_runner/20260819T234806Z`):

- **Population**: 2,075 paired games, 2018-2025, close grade.
- **Accuracy, week-blocked (primary, 141 blocks)**: estimate **-0.0482
  accuracy points**, 95% **[-0.7707, +0.6715]**, se=0.3679,
  **`probability_positive` = 0.4148**.
- **Accuracy, season-blocked (secondary, 8 blocks -- below the measured
  degeneracy floor, reported not gated)**: estimate -0.0482, 95%
  [-0.6305, +0.4888], P+ 0.4025.
- **Brier, week-blocked**: -0.0002, 95% [-0.0006, +0.0002], P+ 0.1445.
  **Log-loss, week-blocked**: -0.0004, 95% [-0.0011, +0.0003], P+ 0.1496.
  (Reported per `docs/experiment_pipeline.md`; never gate the registry
  `effect`, which stays the accuracy comparison.)
- **Classification**: `unresolved_below_power`, `closing_ground: null`.
  The primary interval crosses zero and sits nowhere near either admissible
  closing ground (not a resolved wrong sign, no positive control run) --
  per this file's own taxonomy above, this is the EXPECTED shape for a
  real small (or absent) effect at this evaluator's resolution, not a
  rejection.
- **Read against the pre-registration**: the reproduction-check step in
  `scripts/mod06_position_prior_shrinkage_build.py` confirmed the
  `value_shrinkage_target="zero"` re-run matches the existing production
  `game_features_weak_stack.parquet` columns exactly before the candidate
  was ever built, and the additive-merge check confirmed no pre-existing
  column was altered -- both printed PASSED in the build log before any
  accuracy number was computed.
- **Contrast with the CFB prior (both measured, not comparable in
  magnitude -- different design, different league, different feature
  construct)**: the CFB screen's two-variables-at-once comparison read
  -0.526 pts, P+ 0.0173 (interval fully below zero, closure refused on the
  honest-refit-widening rule). This isolated, single-variable NFL
  comparison reads far closer to a coin flip (-0.048 pts, P+ 0.4148,
  interval width driven almost entirely by noise, not a resolvable
  negative). The two reads are consistent with the CFB screen's own
  self-diagnosed confound (conflating "add a feature" with "change its
  shrinkage target") rather than with a genuine cross-league negative
  mechanism -- but this is **inferred**, not established; no test here
  disentangles the CFB confound directly.

### EV case for the forced-pick decision (AGENTS.md: promotion bar is not a decision bar)

`probability_positive` = 0.4148 is BELOW 0.5 -- on this isolated read, the
zero-shrinkage baseline is very slightly favoured over the position-prior
candidate, not the reverse. Per AGENTS.md's own framing ("P+ above 0.5
favours playing the candidate, full stop"), the symmetric reading applies:
P+ below 0.5 does not favour switching production to this candidate. This
is NOT a closure of the line of work (0.4148 is nowhere near either
admissible closing ground, and the interval is wide -- se=0.3679 on 141
blocks) -- it is `unresolved_below_power`, recorded as such, and remains
open to a future, better-powered or differently-parameterized look (e.g. a
full James-Stein weight re-derivation, or an opener-grade re-screen once
`feature_arm` supports it). But on the number in hand, there is no EV case
today for wiring `weak_stack_js_prior` into production ahead of
`weak_stack`.
