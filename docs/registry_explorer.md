# Registry and overlap explorer (ENG-07)

`nfl_ats.registry_explorer` (`src/nfl_ats/registry_explorer.py`) is a
**read-only** reporting layer over `registry/weak_signals.json` and
`registry/rotation_registry.json`. It exists so research prioritization does
not depend on manually re-deriving `docs/pool_edge_plan.md`'s "2026-08-31
registry state and next shots" survey by hand every session — that section
is exactly the manual process this module mechanizes. The module never
calls either registry's writer (`save_registry` / `record_signal` /
`record_look`); `tests/test_registry_explorer.py` asserts the live registry
files are byte-identical before and after every view runs.

CLI: `scripts/registry_explore.py` (not a `nfl-ats registry explore`
subcommand — `src/nfl_ats/cli.py` is 6000+ lines and was being edited
concurrently by other sessions on other ROADMAP Phase 13 items while this
one was built, so a new top-level subparser risked a conflict for no
functional benefit; ENG-07's own task description names this script path as
the explicit fallback. Nothing stops a future session from wiring a thin
`nfl-ats registry explore` subcommand that imports this same module).

```
./.tools/uv.exe run --no-sync python scripts/registry_explore.py unresolved [--league nfl] [--units accuracy_points] [--family NAME] [--json]
./.tools/uv.exe run --no-sync python scripts/registry_explore.py repeated-windows [--json]
./.tools/uv.exe run --no-sync python scripts/registry_explore.py shared-populations [--league nfl] [--units accuracy_points] [--json]
./.tools/uv.exe run --no-sync python scripts/registry_explore.py source-availability [--league nfl] [--json]
./.tools/uv.exe run --no-sync python scripts/registry_explore.py next-shots [--league nfl] [--units accuracy_points] [--top 15] [--json]
```

Without `--json` each view prints a plain-text table; with `--json` it
prints the exact payload the Python functions return (sorted keys, indent
2), so a caller can pipe it into `jq` or another script.

## The five views

### (a) `unresolved`

Every `unresolved_below_power` entry in `registry/weak_signals.json`,
filtered by `--league` / `--units` / `--family`, sorted by
`probability_positive` descending (entries with no recorded value sort last
— never coerced to zero). `refuted_mechanism` and `bounded_by_control`
entries are excluded: those are closed lines, not open research questions.
Function: `registry_explorer.unresolved_signals`.

### (b) `repeated-windows`

Reports **cross-family** reuse of rotation-registry season blocks —
`docs/rotation_registry.md` rule 4 ("windows retire per-family, not
globally... two different families MAY draw overlapping seasons... but the
ledger records global usage per season so accumulating cross-family
multiplicity stays visible instead of silent") — plus every window that
intersects the mined 2018-2025 seasons, which rule 6 says "carries a
discount that the write-up must state" (a disclosed penalty, never a ban).
A single family cannot re-look at its own window (`record_look` spends it
forever, and `rotation._validate` refuses a family re-overlapping its own or
its inheritance chain's prior windows), so "repeated" here is necessarily
about *different* families, not a family repeating itself.
Function: `registry_explorer.repeated_windows`.

### (c) `shared-populations`

Groups of unresolved signals whose `[seasons[0], seasons[1]]` ranges
overlap, grouped the same way `weak_signals.family_overlap_warnings` groups
them (same `signal_family` key, same pairwise season-intersection test) —
reimplemented rather than called directly because this view additionally
needs each overlapping member's identity, which the existing function does
not expose (only group-level counts). The unmodified
`family_overlap_warnings` output is included verbatim as `pool_summary` so
every number is traceable back to the exact function `nfl-ats weak-signals
pool` already uses.

**Effective sample size is reported as a bound, not a point estimate.**
Members of one group are correlated decompositions of the same window
(AGENTS.md), so summing `sample_games`/`sample_blocks` across them
(`naive_sum_upper_bound`) treats them as independent information, which the
overlap makes false. The amount of information the single best-covered
member alone already carries (`max_single_member_lower_bound`) is reported
as the conservative floor. The true effective N sits somewhere between the
two; this module does not invent a single number for it, and it never
computes a "games needed" figure — that quantity is banned project-wide
(within-week correlation is fixed at zero by owner mandate).
Function: `registry_explorer.shared_population_groups`.

### (d) `source-availability`

One row per `(league, family)` pair, classifying whether the family's
underlying data source is currently captured by the live weekly scheduler
(`scripts/capture_scheduler.py`'s `SCHEDULE`), a manual/periodic bulk ingest
script that is *not* on that schedule, or built purely from data already in
the main feature pipeline with no separate capture at all. Every row carries
a `status`, a `detail`, and a `citation` (or an explicit `None` when there
is nothing to cite). **Three confidence tiers, and every row says which one
it is:**

1. **Direct rule match** (`captured_scheduled`, `paused_scheduled`,
   `derived_no_separate_capture`, `bulk_ingest_unscheduled`, `mixed`) — the
   family matched an entry in `registry_explorer.FAMILY_SOURCE_RULES`, and
   that entry's `citation` names the exact file (and, where useful, line
   range) read this session to establish it. `test_family_source_rules_all_carry_a_citation`
   pins that every hardcoded rule has one.
2. **Category-level inference** (`inferred_from_category`) — no family rule
   matched, but the registry's own `category` field (an existing structured
   field, not something this module infers) supports a lower-confidence
   general statement, e.g. "category='schedule' families verified so far
   (body_clock, bye_overval, roof, dst_transition) all read data already in
   the ingested schedule snapshot." Every such detail string ends with "not
   verified for this specific family" so it can never be mistaken for tier 1.
   `modeling` and `control` categories get a distinct
   `not_applicable_modeling` status: those are internal comparisons/
   constructions, not claims about an external source.
3. **`unknown`** — no family rule and no usable category. Never a guess.

Function: `registry_explorer.source_availability`; the rule table and its
citations live at the top of `src/nfl_ats/registry_explorer.py`.

### (e) `next-shots`

The ranked prioritisation output: `unresolved` sorted by
`probability_positive` descending, tie-broken by whether a rotation-registry
family with an unspent window exists for the entry's family, then by name.
Each row also carries `overlap_group_id` (from view (c), so a caller can see
at a glance whether a "top" result is actually one independent vote or one
member of a large correlated family) and `matching_rotation_families`.

**Family matching between the two registries is best-effort and says so.**
`weak_signals.signal_family` and the rotation registry's declared family
names are independent naming conventions with no guaranteed
correspondence — a name is treated as a match only if it equals or is a
prefix/suffix superstring of the other, every row reports the exact list it
matched against, and `unspent_rotation_window` is `None` (never a guessed
`False`) when nothing matched at all.
Function: `registry_explorer.next_shots`.

### ENG-27 addition: `coverage_plan` (rotation-registry coverage)

Read-only, like every view above: `registry_explorer.coverage_plan(weak_registry,
rotation_registry)` computes what `nfl-ats rotation declare-coverage` would
do, without writing anything. For every distinct `(league, family)` in
`weak_registry` that `matching_rotation_families` finds no rotation-family
match for, and that `rotation_registry.no_rotation_needed` does not already
excuse, it plans one of two actions:

- `declare_stub` — no admissible reason applies
  (`rotation.classify_no_rotation_reason` returned `None`), so the plan
  reserves a `declared_for_coverage` rotation-family stub (no window, no
  research commitment) named after the weak-signal family, falling back to a
  `<family>__<league>` suffix only on a name collision between two leagues
  sharing one bare family string (measured 2026-09-04: zero such collisions
  exist in the live registry; the fallback is defensive).
- `no_rotation_needed` — the classifier matched one of
  `rotation.NO_ROTATION_FIXED_REASONS` (`reliability_measurement`,
  `positive_control`, `oracle`, `retired_profile`) or a
  `decomposition_of_parent:<family>` tag, from the entry's `category` and
  name only — never guessed. See `rotation.classify_no_rotation_reason`'s
  docstring for the exact, citation-grounded rule (name contains "oracle" ->
  `oracle`; name contains "reliability" -> `reliability_measurement`; name
  contains "retired" -> `retired_profile`; `category == "control"` and none
  of the above -> `positive_control`).

The write path is `nfl-ats rotation declare-coverage [--dry-run|--apply]`
(`src/nfl_ats/cli_commands/registry.py`), driven by
`rotation.declare_coverage_stub` / `rotation.record_no_rotation_needed` — see
`docs/rotation_registry.md`'s "Coverage" section for the full mechanics,
including why the stub schema fields are omitted (not written as `null`) for
every family that does not carry them, so this command's writes are
byte-for-byte additive to the tracked ledger.

## What the live registry actually shows (read this before trusting "top 15")

Running `next-shots --league nfl --top 15` against the live registry
(measured this session; re-run for the current numbers) surfaces mostly
`probability_positive: 1.000` entries — several of which are named
`*_reliability`, `odds_microstructure_*_oracle_*`, or otherwise read as
split-half reliability measurements or deliberately-leaked positive-control
cells rather than genuine open research questions. This view sorts
*exactly* by the DoD's own spec (probability_positive descending), which is
mechanically correct; it does not — and should not — silently filter out
control/oracle/reliability rows, because doing so would be exactly the kind
of editorial judgment call this module exists to avoid making on a reader's
behalf. Treat the raw ranking as a starting list to be read with
`category` (`control` rows are visibly marked) rather than as a
pre-filtered final answer.

Separately, `unspent_rotation_window` was `None` for nearly every high-`PP`
entry in the live data: the rotation registry declared only ~29 families
total, against 350+ distinct weak-signal families (measured 2026-09-04, the
count that motivated ROADMAP.md ENG-27), so most strong reads had simply
never had a rotation family declared for them at all. That is a fact about
registry coverage, not a defect in the matcher — `matching_rotation_families`
on each row shows exactly what was (and was not) found.

**ENG-27 correction (2026-09-04, same session):** `nfl-ats rotation
declare-coverage --apply` has now been run for real (see the `coverage_plan`
section above), taking the rotation registry from 30 families to 395 (365
`declared_for_coverage` stubs) plus 15 `no_rotation_needed` records, so
`unspent_rotation_window` now reads `True` for essentially every top-15 row
instead of `None` — a stub carries no window, but its `close` grade
(2009-2025) starts fully unspent, so `_rotation_has_capacity` correctly
reports the whole pool as available rather than "no idea, nothing matched."
Re-run `next-shots` for the current numbers; the family MATCH problem this
section originally described is resolved, but a matched stub still holds no
actual confirmation window until someone runs `rotation assign` against it —
coverage and capacity are different facts, and this view still reports both
honestly rather than collapsing them into one.

## Tests

`tests/test_registry_explorer.py`: one test per view against small synthetic
registries built in `tmp_path` (covering filtering, overlap grouping,
bounded effective-N, the three source-availability confidence tiers, and the
`next_shots` tie-break), plus two tests against the live tracked registries
that assert only structural properties (non-empty, correctly shaped, valid
JSON) and that `registry/weak_signals.json` / `registry/rotation_registry.json`
are byte-identical before and after every view runs.
