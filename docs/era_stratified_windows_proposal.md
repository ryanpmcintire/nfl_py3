# Proposal: era-stratified confirmation windows

Status: **owner-approved 2026-08-19, implemented 2026-08-19.** Written
2026-08-19 during the owner's question about small, back-to-back evaluation
windows; the owner signed off the same day with one binding refinement: era
variation is expected to be a **change in effect magnitude, not binary
presence/absence** ("it's not that the signal is there in era-A but not
era-B, it's that it's less predictive"). Consequence: stratified-window
results must report per-leg magnitudes alongside the pooled read, and a
weaker-in-one-era reading is never itself evidence the mechanism is absent
there. The registry's validators remain owner-mandated to never be weakened.

## The problem it addresses

Rotation-registry confirmation windows are contiguous 2–3-season ranges
(e.g. `[2013, 2015]`, `[2020, 2021]`). Two costs, both measured elsewhere in
this repo:

1. **No valid season-blocked interval.** Coverage of the season-blocked
   bootstrap at k=2 blocks is 0.466 against a nominal 0.95
   (`docs/estimation_variance.md`, the D4 audit; hence
   `MIN_BLOCKS_FOR_INTERVAL = 10`). Every 2-season window therefore leans
   entirely on week blocking.
2. **Regime confounding.** Adjacent seasons share a rules era, carryover
   rosters (`DEFAULT_OFFSEASON_RETENTION = 0.67`), and a scoring environment.
   Era-localized effects are real here: the year-1 head-coach fade is null in
   2009–2017 and lives entirely in 2018–2025 (PER-07). A contiguous window
   cannot distinguish "real edge" from "era quirk."

## The proposal

Allow a family's one confirmation window to be composed of **non-adjacent
single-season legs**, each evaluated walk-forward with training strictly
prior to that leg (train ≤2012 → score 2013; train ≤2021 → score 2022),
pooled under week blocking across both legs.

- Same data budget per look (2 seasons), more regime diversity per look.
- Forward-chaining is preserved per leg; no leg ever trains on data at or
  after its own scoring season, so FND-05 semantics are untouched.
- Declare-before-look, per-family retirement, earliest-eligible assignment,
  and the closing-ground validators all carry over unchanged. The only new
  registry concept is a window whose `seasons` field is a list of legs
  rather than a contiguous range.

## Scope limits

- **Close-graded families only.** The eligible pool there is 2011–2025
  (warm-up floor 2011, rotation rule 9). Opener-graded families draw from
  2020–2025 — six adjacent seasons total (the purchased snapshot archive's
  coverage, MKT-02) — so stratification buys little and is not proposed there.
- Existing spent windows are untouched; this only affects future assignments.
- The earliest-eligible rule needs a deterministic extension for leg pairs
  (e.g. earliest untouched season + the untouched season maximally distant
  from it). That rule must be fixed in code before any family draws under it,
  so no window can be cherry-picked.

## What this does NOT change

The decision frame is unchanged: windows govern claims; plays are decided on
expected value over the full paired evidence. A stratified window still
resolves nothing smaller than its week-blocked power; its benefit is that the
one bit it does buy (direction, `probability_positive`) spans two regimes
instead of one.

## Implemented 2026-08-19

Shipped in `src/nfl_ats/rotation.py`, exercised by 21 new tests in
`tests/test_rotation.py` (55 passed, up from the pre-existing 34; the
original 34 pass **unchanged**, which is the regression proof described
below), and a minimal CLI surface in `src/nfl_ats/cli.py`
(`rotation assign --stratified`, `rotation record --leg-effects`).

### What shipped

- **A window can now be `"contiguous"` (unchanged) or `"stratified"`**
  (`Window.window_kind`, new field, default `"contiguous"`). A stratified
  window's `seasons` holds its two leg seasons, sorted ascending — not a
  range endpoint pair. `Window.covered_seasons` is the new general-purpose
  accessor (full range for contiguous, exactly the two legs for stratified)
  that every overlap, usage, and capacity computation in the module now uses
  instead of assuming `[min, max]` means "every season in between."
  `Window.season_range` is kept for contiguous windows only and now raises
  `RegistryError` if called on a stratified one, rather than silently
  returning a wrong range.
- **`eligible_stratified_seasons(registry, name)`** — every individual season
  a close-graded family may still draw as a leg: the pool floor
  (`MIN_ELIGIBLE_START_SEASON`) upward, minus everything touched by the
  family's own + inherited windows (via the real `covered_seasons` of each,
  not their endpoints), minus mined 2018-2025 seasons unless
  `acknowledges_mined_2018_2025`. Raises for a non-close grade.
- **`assign_stratified_window(registry, family)`** — assigns
  `(min(eligible), max(eligible))` as the leg pair (see "the deterministic
  leg-pair rule as implemented" below), refusing a second unspent window
  (same one-family-one-window invariant as `assign_window`) and refusing a
  non-close grade with a clear error.
- **`confirmation_split_legs(features, registry, family)`** — the stratified
  counterpart to `confirmation_split`. Returns one `LegSplit(season,
  training, scoring)` per leg; each leg's `training` is every completed
  regular-season game strictly before *that leg's* first gameday, computed
  independently per leg (not one shared cutoff). `confirmation_split` itself
  now refuses a stratified window with a redirect error, and
  `confirmation_split_legs` refuses a contiguous one the same way.
- **Per-leg magnitudes are a first-class, enforced field**, not a documentation
  convention: `LegResult(season, effect, probability_positive,
  sample_blocks)`, stored on `Window.leg_effects`. `record_look(...,
  leg_effects=...)` now:
  - **requires** `leg_effects` (one entry per leg, matching the window's own
    leg seasons exactly) when spending a stratified window, and
  - **rejects** `leg_effects` outright on a contiguous window.
  `_validate` additionally refuses to load a *spent* stratified window with
  no `leg_effects`, mirroring the pre-existing spent-without-artifact check.
  This is the code-level enforcement of the owner's binding refinement: era
  variation is a change in magnitude, so the per-leg numbers can never be
  silently collapsed into the pooled read.
- **Pooling under week blocking across legs** is left to the caller (as
  `confirmation_split`'s contiguous pooling already was — this module hands
  back frames and records verdicts, it does not fit or bootstrap models
  itself). What changed here is that the registry now *requires* the caller
  to also hand back each leg's own number, not just the pooled one.
- **`registry_status(...)["families"][i]["remaining_eligible_stratified_seasons"]`**
  — new status field, close-graded families only (0 for other grades),
  reporting `len(eligible_stratified_seasons(...))` for visibility, alongside
  the pre-existing `remaining_eligible_windows`.
- **CLI**: `rotation assign --stratified` (mutually exclusive with `--size`,
  which is meaningless for a fixed two-leg pair) and
  `rotation record --leg-effects '<json list>'`. Both are thin argument
  plumbing to the functions above; no new business logic lives in `cli.py`.

### The deterministic leg-pair rule, as implemented

The proposal's own stated rule — "earliest untouched season + the untouched
season maximally distant from it" — is implemented **exactly**, and reduces
algebraically to `(min(eligible), max(eligible))`:

Let `E` be the family's remaining eligible season set (pool floor upward,
minus touched, minus unacknowledged-mined) and `a = min(E)`. For every other
`s ∈ E`, `s ≥ a` by definition of `a` being the minimum, so the distance
`|s − a| = s − a`, a quantity that is monotonically increasing in `s`. Its
maximizer over `E \ {a}` is therefore always `max(E)`. No tie can arise
except when `|E| = 1`, which is refused before the pair is chosen (fewer than
`STRATIFIED_LEG_COUNT = 2` eligible seasons raises `RegistryError`). The pair
is thus fully determined by the ledger at draw time: no hidden choice,
nothing to tune, and — because it telescopes inward as legs are spent (a
family's second stratified draw picks the next-innermost min/max of what
remains, e.g. `(2011, 2025)` then `(2012, 2024)` then `(2013, 2023)`, pinned
in `test_stratified_assignment_leg_pair_is_earliest_and_maximally_distant`) —
no window can be cherry-picked by drawing in a different order.

### Resolution decisions (ambiguities not fully specified by the proposal text)

1. **Schema disambiguation.** A bare two-element `seasons` list is
   structurally identical whether it means "contiguous range `[start, end]`"
   or "leg pair `{a, b}`" — the proposal's own prose ("the `seasons` field
   is a list of legs rather than a contiguous range") does not by itself
   tell a loader which one it is looking at. Resolved by adding the explicit
   `window_kind` field (`"contiguous"` default, or `"stratified"`) rather
   than guessing from list shape or length. Old ledger entries have no
   `window_kind` key and default to `"contiguous"`, so every existing
   window's interpretation is unchanged.
2. **Grade scope: `nflverse_spread` excluded alongside `opener`.** The
   proposal's scope-limit section names only "close-graded families";
   `nflverse_spread` shares close's exact numeric season pool
   (`GRADE_POOLS["nflverse_spread"] == GRADE_POOLS["close"] == (2009, 2025)`,
   read from `src/nfl_ats/rotation.py`) but is never mentioned in that
   sentence. Rather than assume the proposal meant "any large pool," this
   implementation reads it literally: `assign_stratified_window` and
   `eligible_stratified_seasons` both raise for any grade other than
   `"close"` (`STRATIFIED_GRADE`), opener and nflverse_spread alike, with an
   error naming the scope-limit doc. If a future family wants stratification
   under `nflverse_spread`, that is a small, explicit follow-up (widen
   `STRATIFIED_GRADE` to a tuple), not something this change assumes.
3. **Leg count fixed at exactly two (`STRATIFIED_LEG_COUNT = 2`).** The
   proposal's worked example and its "same data budget per look (2 seasons)"
   framing both describe a pair, never a larger tuple, and the deterministic
   leg-pair rule above is defined in terms of "the earliest" and "the [one]
   maximally distant" — singular, not "the two next-most distant" or similar
   for a triple. Extending to three or more legs is out of scope here and
   would need its own deterministic tie-breaking rule the proposal does not
   specify.
4. **A later leg's training set may include an earlier leg's season.** The
   proposal forbids a leg "training on data at or after its own scoring
   season," not on data from another leg. `confirmation_split_legs` gives
   each leg an independent forward-chained cutoff (every completed game
   strictly before that leg's first gameday), so leg B's training naturally
   includes leg A's season when A is chronologically earlier — exactly the
   `train ≤2012 → score 2013; train ≤2021 → score 2022` example in the
   proposal, where 2013's own training would likewise include any season
   between it and 2022 if such a leg existed. This preserves FND-05
   semantics per leg, which is what the proposal states as the goal.
5. **`covered_seasons`, not `[min, max]`, is the correctness fix a stratified
   window forces everywhere.** This is mechanical, not a proposal design
   choice, but is recorded here because it touches shared code paths: the
   pre-existing `_overlaps` helper and every caller that treated a window's
   `seasons` endpoints as "every season in between" (`_validate`'s own- and
   inherited-window overlap checks, the mined-season acknowledgment check,
   `eligible_blocks`/`_blocked_seasons`, `season_usage`, and
   `grade_pool_capacity`) would silently over-block or over-count capacity
   for a stratified window — e.g. a family that spends legs `(2011, 2025)`
   would, under the old range-overlap logic, appear to have consumed every
   season from 2011 to 2025, blocking an inheriting family from ever drawing
   a contiguous `[2012, 2014]` block it never actually touched. Fixed by
   introducing `Window.covered_seasons` and `_touched_seasons`/
   `_windows_overlap` (season-set intersection) as the abstraction those
   call sites use instead; for an all-contiguous registry this is
   mathematically identical to the old range-overlap logic (a range's
   `covered_seasons` **is** the range), which is what the bit-identical
   regression tests below confirm. `test_touched_seasons_use_real_legs_not_the_span_between_them`
   and `test_grade_pool_capacity_uses_real_legs_not_the_span_for_a_stratified_window`
   pin the corrected behavior directly against what the old logic would have
   produced.

### Validator non-weakening: how it was verified

The registry's validators are owner-mandated to never be weakened. Verified
two ways:

1. **The full pre-existing rotation test suite (34 tests) passes unchanged,
   with zero edits to any pre-existing assertion.** Every contiguous-window
   behavior pinned before this change — earliest-eligible assignment,
   per-family retirement, forward-chaining refusal, the closing-ground
   taxonomy, the warm-up floor, effect-field validation, the CLI workflow —
   is exercised exactly as before and produces exactly the same results.
   This is the bit-identical regression proof for contiguous windows: the
   `covered_seasons` refactor described in resolution 5 above is provably a
   no-op for any window whose `window_kind` is `"contiguous"`, because
   `covered_seasons` reduces to the same range `_overlaps` was already
   checking.
2. **Every validator rule that applied to contiguous windows now applies
   identically to stratified ones, with no carve-out**: spent-without-
   artifact-or-verdict, the closing-ground taxonomy
   (`_validate_closing_ground`, unchanged — it does not look at
   `window_kind` at all), pool-bounds, mined-2018-2025 acknowledgment,
   own-window and inherited-chain overlap, and the one-assigned-window-per-
   family limit all fire the same way. One rule is *strictly additive*:
   `_validate` now also refuses to load a spent stratified window with no
   `leg_effects`, and `record_look` refuses to spend one without it — a new
   constraint, never a relaxation of an old one. `--closing-ground`'s
   admissible-grounds list (`wrong_sign_resolved`, `no_split_half_reliability`,
   `positive_control_bound`) is untouched, imported from `weak_signals.py`
   exactly as before; nothing about a stratified window's verdict path
   differs from a contiguous one's.

### Tests

`tests/test_rotation.py`, 21 new tests (55 total, up from 34): leg-pair
assignment determinism (single draw and three-draw telescoping), refusal for
non-close grades (opener and nflverse_spread) with the CLI's exit-code-2
path, refusal on fewer than two eligible seasons, refusal of a second unspent
window, per-leg forward-chaining strictness (`confirmation_split_legs`,
including the deliberate cross-leg training inclusion from resolution 4),
`confirmation_split`/`confirmation_split_legs` mutual redirect errors,
`leg_effects` required-on-spend / rejected-on-contiguous / must-match-legs /
round-trips-through-save-and-load, the `window_kind` schema-disambiguation
case (identical `seasons` list, different `covered_seasons`), the
touched-seasons and grade-pool-capacity correctness fixes from resolution 5,
and a CLI end-to-end workflow (`declare` → `assign --stratified` → `record
--leg-effects`).

### Verification run (measured this session)

```
.tools/uv.exe run --no-sync ruff format --check src/nfl_ats/rotation.py src/nfl_ats/cli.py tests/test_rotation.py
.tools/uv.exe run --no-sync ruff check src/nfl_ats/rotation.py src/nfl_ats/cli.py tests/test_rotation.py
.tools/uv.exe run --no-sync mypy src
.tools/uv.exe run --no-sync pytest tests/test_rotation.py -q
```

`ruff format --check` / `ruff check`: clean. `mypy src`: "Success: no issues
found in 89 source files." `pytest tests/test_rotation.py`: **55 passed**
(34 pre-existing + 21 new). A full-repo `pytest` run in this same working
tree also shows 21 unrelated failures in `tests/test_weekly.py` and
`tests/test_experiment_registry.py` ("No complete snapshots found under
.../data/raw"); those files are untouched by this change and the failure
traces to other concurrently in-progress, uncommitted work in this session
(`git diff --stat` shows large WIP diffs in `snapshots.py`,
`experiment_runner.py`, `public_board.py`, `features.py`,
`best_pick_nomination.py` from parallel agents), not to anything in this
proposal's implementation.
