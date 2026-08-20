# Surface-switch tilt as a FEATURE, not an overlay: a feature_arm predeclaration

Written 2026-08-19, **before** `nfl-ats experiment run` is invoked (predeclaration
required by `docs/experiment_pipeline.md` and by the "family must be declared
before the signs are seen" discipline in `AGENTS.md`). This tests whether the
project's strongest prospective lead -- the surface-switch tilt, currently
wired in only as a post-prediction pick-level overlay
(`docs/surface_switch_tilt_overlay.md`, `src/nfl_ats/surface_switch_tilt_overlay.py`,
challenger `surface_switch_tilt_overlay`) -- also helps as a **training-time
ridge feature** on the production `weak_stack`/`market_residual` model.

## What is declared, exactly (read before any result exists)

- **Baseline arm**: `feature_profile="weak_stack"`, `ridge_alpha=10.0` -- the
  active production configuration (**read** from
  `artifacts/active_ats_model.json`: `method="market_residual"`,
  `feature_profile="weak_stack"`, `ridge_alpha=10.0`).
- **Candidate arm**: `feature_profile="weak_stack_surface"`, `ridge_alpha=10.0`
  -- `weak_stack` plus exactly **one** new column, `surface_switch_flag`.
- **The one column, exactly**: `1.0` when the AWAY team's modal home surface
  THIS SEASON normalizes to grass AND this game's own surface normalizes to
  turf, else `0.0`. This is the identical away-visitor flag
  `surface_switch_tilt_overlay.surface_switch_flag_by_game` already applies
  at pick level, now exposed to the ridge fit as a feature instead
  (`src/nfl_ats/features.py::add_surface_switch_features`,
  `constants.SURFACE_SWITCH_FEATURE_COLUMNS = ("surface_switch_flag",)`).
- **No signed home-minus-away / mirror column.** Considered and rejected: the
  registry's own grass-venue mirror reads
  (`surface_familiarity_r2_grass_venue_mirror` NFL -0.4995 pts, P+ 0.3205;
  `cfb_surface_familiarity_grass_venue_mirror` CFB -0.1218 pts, P+ 0.4291 --
  both **read** from `docs/surface_switch_tilt_overlay.md`) are near a coin
  flip leaning the WRONG way in both leagues. A `diff`-style column built
  from a null mirror would just be `0 - flag`, i.e. redundant with the flag
  itself, and folding an unmeasured "mirror direction" into the design before
  running risks smuggling a second, unsupported hypothesis into one
  predeclared family. One column, matching the one measured, tested
  mechanism -- not two.
- **Grade**: `close` only. `feature_arm` has no `opener` implementation
  (**read**, `docs/experiment_pipeline.md`: "`feature_arm` at
  `population.grade == "opener"`... not implemented here"); opener-side
  confirmation of the surface-switch construct already flows through the
  live `surface_switch_tilt_overlay` prospective challenger at pick level,
  which this experiment does not replace or duplicate.
- **Population**: `seasons=[2018, 2025]`, league `nfl`. 2018 is **read** from
  this repo's own established convention for full-history
  `weak_stack`/`market_residual` close-grade comparisons
  (`scripts/ridge_alpha_promotion_eval.py --nflverse-start-season` default
  `2018`; `scripts/weak_stack_v2_eval.py`'s `NFLVERSE_START_SEASON = 2018`),
  not a restriction invented for this run. 2025 excludes the in-progress,
  incomplete 2026 season from the paired comparison.
- **Endpoints**: primary `accuracy`; secondary `brier`, `logloss` (reported,
  never gating the registry `effect`, which stays the accuracy comparison
  per `docs/experiment_pipeline.md`).
- **Blocking**: primary `week`, secondary `season`.
- **Samples**: 20,000 (full-fidelity bootstrap, matching
  `penalty_discipline_reproduction.json`'s convention, not the 2,000-sample
  test-speed shortcut).
- **Seed**: `20260819`, fixed, no wall-clock nondeterminism.
- **Reliability check**: `method="not_applicable"` -- a feature-arm model
  comparison has no per-entity trait to split-half; `experiment_runner.py`
  refuses `split_half` for `feature_arm` outright (**read**, same doc).

## Reused-window acknowledgment (rotation_registry.md rule 6)

**Read**, `docs/rotation_registry.md` rule 6: "A family that is a variant of
an existing line of work inherits that line's spent windows at declaration...
Any window intersecting 2018-2025 additionally requires the declaration to
set an explicit `acknowledges_mined_2018_2025` flag -- the ~130-150-look
ledger means a result there carries a discount that the write-up must
state." `experiment_runner.py`'s `feature_arm` path does **not** route
through `nfl_ats.rotation` at all (**measured**: no `rotation` reference in
`src/nfl_ats/experiment_runner.py`), so there is no formal rotation-registry
family/window to spend here -- but the season range [2018, 2025] is squarely
inside the acknowledged ~130-150-look 2018-2025 pool, and the surface-switch
mechanism itself has already taken five prior registry looks this week
(**read**, `registry/weak_signals.json` via
`docs/surface_switch_tilt_overlay.md`):
`weather_battery_surface_switch_grass_to_turf`,
`surface_familiarity_r1_turf_venue_visitor_split`,
`surface_familiarity_r2_grass_venue_mirror`,
`surface_familiarity_r3_era_2009_2017`, `surface_familiarity_r3_era_2018_2025`
(plus two CFB replications on a separate benchmark). **This run is a reused
window with a stated discount, per rule 6, not a fresh independent look** --
its accuracy delta should be read as one more correlated measurement of the
same underlying construct, not as independent confirmation. This is a
caution stated up front, not a wall: per AGENTS.md, a reused window "carries
a stated discount, not a ban."

## Construct discrepancy, flagged rather than silently resolved

The task briefing that produced this predeclaration described the leak-safe
argument as "PRIOR-season modal surface for the visitor." **That does not
match the actual, already-tested construct.** Both
`surface_switch_tilt_overlay.surface_switch_flag_by_game` and its two
upstream sources (`scripts/nfl_weather_battery_screen.py`) use the modal
surface over the team's **CURRENT (THIS) season's** full REG schedule, not
the prior season's. `docs/surface_switch_tilt_overlay.md` and the overlay
module's own docstring defend this explicitly as pregame-safe: a team's
home-stadium surface is a structural, stadium-level fact fixed for
essentially the entire season and public before Week 1, and the derivation
never reads `result`/`spread_line` -- it is not derived from a prior
season's OUTCOMES, so "current-season" here is not a leak, it is a
different (and already the measured/tested) construct than "prior-season."
`add_surface_switch_features` (this feature) is implemented to be
**bit-identical** to that already-measured, already-tested construct --
matching the exact registry entries this experiment exists to evaluate as a
feature -- rather than inventing an untested prior-season variant. Two
leakage regression tests in `tests/test_features.py` (never-reads-outcome-
columns; a future season's surface data cannot move an earlier season's
flag) mirror the two the overlay module already carries in
`tests/test_surface_switch_tilt_overlay.py` for the identical construct.

## Feature-table provenance (data source for this run)

The production `weak_stack` table on disk, `data/processed/game_features_weak_stack.parquet`,
does not carry `surface_switch_flag` (**measured**: column list checked
directly). Rather than re-running the full base -> pbp -> learned-availability
build chain (three sequential CLI steps re-deriving Elo/graph/PBP/injury
enrichment, `~110` seconds per the existing manifest's own timing, and a risk
of introducing unrelated drift if any upstream snapshot resolves differently
today), `surface_switch_flag` is attached as a pure, additive merge-by-
`game_id` enrichment of the EXISTING, already-built weak_stack table --
computed by the same tested `add_surface_switch_features` function against
the latest local raw schedule snapshot, which every one of the table's 4,902
`game_id` values is confirmed (**measured**) to match 1:1. This mirrors the
established precedent in this repo for adding one stacker column to
`weak_stack` without a full rebuild
(`scripts/weak_stack_v2_eval.py::add_penalty_discipline_feature`, merge-by-
`game_id`/season onto the existing weak_stack table). Written to a NEW file,
`data/processed/game_features_weak_stack_surface.parquet`, so the original
`game_features_weak_stack.parquet` (used by the live production card path
and other concurrent sessions) is never touched. Both the baseline
(`weak_stack`) and candidate (`weak_stack_surface`) arms of this experiment
are fit from this SAME enriched file, passed via `experiment run`'s
`--features` override -- `weak_stack`'s own columns are byte-identical
between the original and enriched tables (purely additive merge), so the
baseline arm's numbers are unaffected by which of the two files it reads.

## Binding rules this run's verdict must follow (restated, not reargued)

- An interval or CI containing zero is never grounds to reject, fail, or
  close this experiment. At this evaluator's ~2-point resolution, that is
  the EXPECTED shape for a real small signal.
- Only two admissible closing grounds exist: (1) refuted mechanism --
  RESOLVED wrong sign, whole interval below zero, or zero split-half
  reliability (not applicable to a feature_arm comparison); (2) bounded by a
  positive control proven able to detect an effect this size. Anything else
  is `unresolved_below_power`, reported with `probability_positive` -- never
  the binary "contains zero."
- `experiment_runner.py`'s mechanical classifier is the only authority for a
  `wrong_sign_resolved` closure on this run (both conditions: primary
  interval entirely below zero, AND the honest-refit widening factor exceeds
  1.099x). If the record command errors, the verdict is wrong, not the
  validator.
- The pool is forced picks: `probability_positive` above 0.5 favours
  promoting the feature into production, full stop -- a promotion bar
  (predeclared thresholds like MOD-07's 0.90) governs what documentation may
  CLAIM, never what gets played.
- Every claim in the write-up that follows this run carries a provenance
  label: measured / read / reported / inferred.

## What this run does and does not decide

A positive, zero-excluding result promotes `weak_stack_surface` to a
serious production candidate (same bar as any other feature_arm win).
A `probability_positive` short of that, with an interval crossing zero, is
NOT a rejection -- it is recorded `unresolved_below_power` with its
`probability_positive`, exactly like the five prior surface-switch reads,
and the already-live `surface_switch_tilt_overlay` pick-level challenger
keeps accruing independent 2026 prospective evidence regardless of this
run's outcome. Promoting `weak_stack_surface` into `artifacts/active_ats_model.json`
is a separate, later owner decision this predeclaration does not make.
