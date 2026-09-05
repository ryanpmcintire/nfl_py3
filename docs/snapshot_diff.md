# Decision-time snapshot diff (ENG-18)

Written 2026-09-04, four days before Week 1 locks.

## What this is

`nfl_ats.snapshot_diff` builds a compact diff between one week's **Tuesday
lock** (the frozen grading line the pool actually settles on) and every
**later refresh pass** this project can currently reconstruct from
already-recorded evidence: the paper-decision ledger, the pick-revision
ledger (`nfl_ats.pick_refresh`, POL-11/MKT-08), later `margin_predictions`
forecast artifacts, each artifact's `lineage.json` (ENG-16), the ENG-08
refresh-trigger evidence log, and the ENG-14 source-freshness policy (for one
present-tense context section only).

**It is a diff, not a verdict.** It never adjudicates a signal, never writes
to `registry/`, and never calls `nfl-ats weak-signals record` or `nfl-ats
rotation record-look`. The binding closing-grounds taxonomy (AGENTS.md) does
not apply to this module's own output because it never scores or closes
anything — it reports what changed between two recorded snapshots.

Run it read-only:

```powershell
.\.tools\uv.exe run --no-sync python scripts/snapshot_diff.py --season 2026 --week 1
.\.tools\uv.exe run --no-sync python scripts/snapshot_diff.py --season 2026 --week 1 --json
.\.tools\uv.exe run --no-sync python scripts/snapshot_diff.py --season 2026 --week 1 --no-write
```

By default it writes `snapshot_diff.md` and `snapshot_diff.json` under
`artifacts/snapshot_diffs/<season>_wk<week>_<stamp>/` (gitignored, like every
other generated artifact) and prints the Markdown to stdout. `--no-write`
prints only.

## Two refresh channels, and why their diffs look different

1. **The pick-revision ledger.** `refresh-picks --record-decisions`
   recomputes the model at the FROZEN Tuesday `decision_home_spread` and
   appends only the games whose pick actually flipped, under one shared
   `refresh_run_id`. It never reloads or recomputes coach-fade,
   division-revenge, player-arrests, or spread-gap-zone inputs — they are
   copied verbatim from the Tuesday row — so a pick-revision-ledger pass's
   overlay diff is `unchanged` **by construction**, not a measurement, and
   its market-line diff is `unchanged` by the same frozen-line invariant.
   It also never persists a `lineage.json` (it recomputes in memory), so its
   per-source cells are always `no_data` with that reason stated.
2. **A later `margin_predictions` forecast artifact** for the same
   season/week. This channel carries real market-line/probability/pick data
   (`predictions.csv`), so those three diffs can be genuinely `changed`. It
   never carries an overlay decision (that composition step happens
   downstream of `margin-predict`), so its overlay diff is always `no_data`.
   Its per-source cells come from comparing the Tuesday lock's `lineage.json`
   against this artifact's own, field by field — when either side lacks one
   (true of every `margin_predictions` directory on this repo's disk as of
   2026-09-04, `measured` by `find ... -iname lineage.json` over the 19
   2026-week-1 artifacts: zero have one), the cells are `no_data`.

**Honesty note on channel 2, read from this session:** nothing on disk
distinguishes a genuine mid-week card republish from a repeated
`margin-predict` development/test invocation. The 19 real `margin_predictions`
artifacts this project has for 2026 Week 1 as of 2026-09-04 span 2026-08-12
through 2026-09-03 — clearly the latter, not a real late-week refresh
cadence. `render_markdown` still surfaces every one of them (the DoD asks for
"any later forecast artifacts," and the tool does not get to decide which
reruns "count"), but a reader should not mistake eighteen dev reruns for
eighteen real decision-time refreshes.

## Why a game absent from a pick-revision pass is not automatically "unchanged"

`refresh-picks` only appends CHANGED, eligible games to the ledger. A game
missing from one pass's rows could mean either "recomputed, pick did not
change" or "ineligible at this pass's instant (its own kickoff, or the
week's Sunday 4pm ET lock, had already passed)" — two very different facts.
`build_snapshot_diff` tells them apart using
`nfl_ats.pick_refresh.pick_deadline`/`sunday_pick_lock` — pure, already-
established functions, never a live model recompute — and falls back to an
explicit `no_data` only when it genuinely cannot know (no kickoff on record).
The resulting `refresh_pick_basis` string names exactly one of three cases:
`pick_revision_ledger` (present in this pass's rows), `inferred_unchanged`
(absent, but still eligible — the ledger's own recording contract means its
pick did not change), or `ineligible_at_this_pass` (absent, and its deadline
had already passed).

## Every cell carries an explicit state

Per the task contract, no cell is ever left blank for "no data." Numeric and
pick/overlay cells carry one of:

- `changed` / `unchanged` / `no_data` — market line, model probability,
  per-source timestamps.
- `same` / `flipped_<from>_to_<to>` / `no_data` — pick. `<from>`/`<to>` are
  lower-cased pick sides (`home`, `away`, or `pass` when the model recorded
  no edge), so a flip's direction is always legible, e.g.
  `flipped_pass_to_home`.
- `changed` / `unchanged` / `no_data` — overlays, with `overlays_added` /
  `overlays_removed` / `overlays_unchanged` naming which of the four
  individual flip flags (`coach_fade`, `division_revenge`, `player_arrests`,
  `spread_gap_zone`) moved.

Every one of those states is paired with a `*_basis` string on the
corresponding `GameSnapshot`/`GameDiffRow` field explaining **how** the tool
knows it — `paper_decision_ledger`, `forecast_artifact_raw`,
`pick_revision_ledger_post_coach_fade_pre_movement_policy`,
`frozen_by_design: ...`, `inferred_unchanged: ...`, `no_data: ...` — so a
reader never has to guess whether a number is authoritative, raw-model, or
inferred.

`tests/test_snapshot_diff.py::test_render_markdown_never_leaves_a_blank_cell`
pins this mechanically: it parses every Markdown table row in a synthetic
diff and asserts no cell is empty after stripping.

## Trigger provenance

A pick-revision-ledger row already carries MKT-08's own
`trigger_type`/`trigger_source`/`trigger_observed_at_utc`. When those are
blank or `unknown` (or for a forecast-artifact pass, which carries no trigger
fields at all), `build_snapshot_diff` looks up the nearest DEADLINE-VALID
entry at or before the pass's own instant in ENG-08's append-only evidence
log (`artifacts/refresh_triggers/<season>/week_<n>.jsonl`), preferring one
naming a game in this pass. `trigger_basis` on `RefreshPassDiff` names which
happened: `ledger_recorded`, `evidence_log_nearest`, or `unknown` — the
trigger is never defaulted to `clock_dispatch`, and never fabricated.

## The present-tense source-freshness section

`SnapshotDiff.current_source_freshness` is `nfl_ats.source_freshness_policy`'s
own `SourcePolicyReport.to_metadata()`, evaluated once, at the moment the
diff was generated. It is explicitly **not** wired into any per-pass
historical cell: `source_freshness_policy.observe_from_disk` can only ever
report the newest snapshot on disk *right now*, regardless of what instant is
passed to it, so using it for a historical "as of this pass" read would be
dishonest. It exists purely as present-tense operational context, rendered
under its own clearly-labelled heading.

## Read-only, and where it fits alongside ENG-08/ENG-14/ENG-15/ENG-16

This module builds on, and never duplicates, four already-shipped ENG
modules: `nfl_ats.refresh_triggers` (ENG-08, the evidence log),
`nfl_ats.source_freshness_policy` (ENG-14, the freshness state machine),
`nfl_ats.ledger_reconcile` (ENG-15, a different question — "did the right
ledger rows land," not "what changed between two snapshots" — and this
module reuses none of its machinery since the questions don't overlap), and
`nfl_ats.lineage` (ENG-16, `read_card_lineage`/`CardLineage.field`, reused
directly rather than re-parsing `lineage.json`).

It never runs `weekly-run`, `publish-predictions`, or `refresh-picks`, never
fits a model, and never writes to `registry/`.

`scripts/snapshot_diff.py` is listed in
`tests/test_experiment_registry.py::_ALLOWLISTED_UNSTAMPED_SCRIPTS` for the
same reason `ledger_reconcile.py` and `prospective_scorecard.py` are: it
writes JSON under `artifacts/` (its own diff, via `atomic_json`) but never to
`registry/`, and `write_experiment_artifact()` always writes a
`registry/experiments/...` row — wiring it in here would misrepresent a diff
as an adjudicated screen.
