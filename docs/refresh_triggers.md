# Refresh-trigger instrumentation (ENG-08)

Written 2026-09-04. Pure instrumentation: this reconstructs and records real
non-clock refresh triggers so they can eventually be compared prospectively
against the fixed-clock refresh checkpoints. It does not adjudicate anything
today, and it never writes to `registry/`.

## Why it exists

`nfl_ats.pick_refresh`'s late-week refresh (POL-11) is driven by a handful of
fixed clock checkpoints -- `refresh_thu`/`refresh_sat`/`refresh_sun` and the
seven `refresh_*_inactives_*` passes in `scripts/capture_scheduler.py`'s
`SCHEDULE`. The latter are named after inactives but are still purely
clock-driven: they fire on a fixed offset from their own capture window
closing, not on the inactives capture actually reporting anything. Nothing in
the project currently asks whether a refresh pass timed to a REAL event --
an inactives list posting, an injury report revising, a projected lineup
changing, the market moving -- would have done any better than waiting for
the next scheduled clock tick.

This module answers the "what data exists" half of that question:
`src/nfl_ats/refresh_triggers.py` reads the capture directories already on
disk and reconstructs every trigger event it can find, tagged with where it
came from, when it was captured, and whether it could have been acted on
before that game's own deadline. `scripts/refresh_trigger_log.py --scan`
appends those to an append-only evidence log. The "were they better" half --
actually pairing a trigger-time pick against a checkpoint-time pick and
scoring the difference -- needs prospective ledger rows that do not exist
yet; `compare_trigger_vs_checkpoint` is the scaffold for that, proven on
synthetic rows only (see "The comparison scaffold" below).

## Relationship to MKT-08

The project already has one refresh-trigger mechanism:
`nfl_ats.pick_refresh`'s `trigger_type`/`trigger_source`/
`trigger_observed_at_utc` columns on the pick-revision ledger
(`TRIGGER_CLOCK_DISPATCH` / `TRIGGER_NEWS_EVENT` / `TRIGGER_UNKNOWN`). That
records provenance for a refresh pass someone actually RAN
(`refresh-picks --trigger-type news_event --trigger-source ...`); it has no
detector of its own. This module is that detector, and its `trigger_source`
vocabulary is deliberately more granular than MKT-08's `trigger_type`:
`mkt08_trigger_type()` maps this module's four non-clock sources onto
MKT-08's coarser `TRIGGER_NEWS_EVENT`, `clock_checkpoint` onto
`TRIGGER_CLOCK_DISPATCH`, and `manual` onto `TRIGGER_UNKNOWN`, reusing those
constants rather than redefining them. A future step that records a
detected trigger onto the pick-revision ledger can carry this module's finer
`trigger_source` value straight through MKT-08's existing free-text
`--trigger-source` field.

## `RefreshTrigger`

One reconstructed event for one game:

| field | meaning |
|---|---|
| `trigger_source` | `clock_checkpoint` / `inactives_posted` / `injury_report_posted` / `lineup_change` / `line_move` / `manual` |
| `game_id`, `season`, `week` | which game |
| `observation_time` | when THIS SCAN reconstructed the trigger (the scan's own clock) |
| `source_capture_time` | when the underlying source was actually captured -- from the snapshot's own manifest/payload, **never** from `observation_time` |
| `checkpoint_name` | the scheduler job name for a `clock_checkpoint` trigger; `None` for every real non-clock trigger |
| `deadline` | this game's own `pick_refresh.pick_deadline` |
| `deadline_valid` | `source_capture_time` strictly before `deadline`? |
| `deadline_reason` | human-readable reason, `deadline_violation: ...` when invalid |
| `detail` | free-text context (snapshot id, changed side, line-move size) |

Every field other than `observation_time` is a fact read off a manifest or
payload already on disk -- nothing here is invented or estimated.

## Deadline validation

Owner rule, binding: a game's pick deadline is `min(own kickoff, that week's
Sunday 16:00 ET)`. Sunday-night and Monday games lock EARLY, at the same
Sunday-afternoon instant as the rest of the week, not at their own kickoff.
This module never redefines that arithmetic -- every deadline is
`nfl_ats.pick_refresh.pick_deadline(kickoff, nfl_ats.pick_refresh.sunday_pick_lock(...))`,
imported directly. A trigger captured at or after its game's deadline is
`deadline_valid=False` and is excluded from `compare_trigger_vs_checkpoint`'s
paired population -- a refresh this project could never actually have acted
on contributes no evidence, in either direction.

`tests/test_refresh_triggers.py` pins the rule against all four slot shapes
on one synthetic week: a Sunday 1pm game (deadline = its own kickoff), a
Thursday game (deadline = its own kickoff), an SNF game (deadline = the
Sunday 4pm ET lock, a full ~4.5 hours before its own 8:20pm kickoff), and an
MNF game (deadline = the SAME Sunday 4pm ET lock, a full day before its own
Monday kickoff). The SNF/MNF cases specifically assert that a capture AFTER
the 4pm lock but BEFORE the game's own kickoff is still a violation --
the early-lock rule, not a kickoff-relative one.

## The five detectors

Each is pure and read-only, taking a directory root and a set of
`GameWindow`s and returning `RefreshTrigger`s -- no ledger write, no network
call.

1. **`detect_clock_checkpoint_triggers`** -- reads `data/scheduler_state.json`
   (passed in, not read internally, so it stays testable on a synthetic
   dict) for `refresh_thu`/`refresh_sat`/`refresh_sun` and the seven
   `refresh_*_inactives_*` rows with status `OK`/`CAUGHT_UP`/
   `ALREADY-CAPTURED`. `source_capture_time` is the record's own `ran_at`
   (falling back to `window_start`) -- a scheduler-clock fact, never this
   scan's own clock.
2. **`detect_inactives_triggers`** -- reuses
   `nfl_ats.inactives_refresh_overlay.load_inactives_snapshots` /
   `inactives_rows_for_game` verbatim (the exact reader WP41's overlay
   already trusts) rather than re-parsing manifests. One trigger per game a
   snapshot actually names rows for.
3. **`detect_injury_report_triggers`** -- reads `data/players/raw/*/manifest.json`
   (the nflverse archive; season-wide, not week-specific, so a fresh pull is
   itself the event) and `data/raw/sportradar_injuries/*/manifest.json`
   (schema-tagged, `status == "complete"`, filtered to the scanned
   season/week).
4. **`detect_lineup_change_triggers`** -- diffs consecutive ARCHIVED copies
   of `artifacts/lineups/current/lineups.json`. That file is a
   REPLACEMENT artifact (`scripts/build_week_lineups.py` overwrites one
   stable path and deletes legacy stamped runs -- measured this session), so
   there is no on-disk history to diff directly; `archive_lineup_snapshot()`
   is the scan script's own bookkeeping, copying a dated snapshot into
   `artifacts/refresh_triggers/_lineup_archive/` keyed by the payload's own
   `generated_at` (idempotent: an unchanged file is never re-archived).
5. **`detect_line_move_triggers`** -- reuses `pick_refresh.original_card`
   (the frozen Tuesday `decision_home_spread`) and
   `pick_refresh.current_captured_home_spread` (a read-only local-store
   lookup, never a live fetch) against `MOVEMENT_POLICY_THRESHOLD`
   (imported from `nfl_ats.pick_refresh`, never redefined).

`detect_all_triggers(TriggerScanRoots, season=..., week=...)` runs all five
and returns everything reconstructed for one week, including the fixed
checkpoints.

## The evidence artifact

`scripts/refresh_trigger_log.py --scan [--current | --season S --week W]`
runs every detector and appends new rows to
`artifacts/refresh_triggers/<season>/week_<n>.jsonl` (gitignored, like every
other generated artifact -- `artifacts/**` is ignored except the tracked
`artifacts/prospective/challengers.json`). Appending is idempotent:
`append_triggers_to_evidence_log` de-duplicates by
`(trigger_source, source_capture_time, game_id)` against both the existing
file and the current batch, so re-running a scan never appends a second copy
of the same event, and nothing already on disk is ever rewritten, reordered,
or removed.

`--current` resolves the live (season, REG week) the same way
`scripts/capture_inactives.py` does
(`scripts.ingest_nflcom_injuries.resolve_current_reg_week`).

### Scheduler job

`refresh_trigger_log_sun` (`scripts/capture_scheduler.py`, `added_on
"2026-09-04"`) runs this scan at **18:00 ET Sunday**, grace 240 minutes --
after the last Sunday refresh-picks pass closes (`refresh_sun`'s window
closes 15:00 ET; `refresh_sun_inactives_late`'s closes 15:50 ET) and before
`backup_data` (22:00 ET), so the week's Sunday captures are already on disk
when this reconstructs them. `catch_up=True`: the job is idempotent by
construction and a late run reconstructs the same true history a late
capture would, matching `player_arrests_tue`/`referee_assignments_wed`'s
reasoning rather than `odds_sun_close`'s point-in-time one. It has no
`dedupe_dir`: its output is a JSONL file per week, not a UTC-stamped
snapshot directory, so the scheduler's snapshot-based dedupe does not apply
here -- the same shape as `refresh_thu`/`refresh_sat`/`refresh_sun`
themselves. This was an additive edit only; the daemon was not restarted, so
the job takes effect the next time the scheduler process is started.

## The comparison scaffold

**Binding closing-grounds taxonomy (AGENTS.md), restated verbatim, because
this scaffold classifies:** an interval or CI that contains zero is NEVER
grounds to reject, fail, or close an experiment. At this evaluator's
~2-point resolution, "contains zero" is the EXPECTED outcome for a real
small signal. Only two grounds ever close a line of work: (1) refuted
mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of
zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero." Within-week
correlation is ZERO by owner mandate, so pairing blocks by WEEK; this
scaffold never estimates or pads a within-week correlation and never
computes "games needed".

`compare_trigger_vs_checkpoint(ledger_rows, triggers, ...)` pairs, per game,
the pick taken at a fixed checkpoint against the pick a trigger-time refresh
would have taken, and scores the paired delta with
`nfl_ats.estimation_variance.naive_block_bootstrap_interval` -- the SAME
estimator every other paired comparison in this project already reports
from, blocked by week. Only games whose trigger is `deadline_valid=True` are
paired; excluded ids are reported, not silently dropped. Classification
defaults to `unresolved_below_power`
(`nfl_ats.weak_signals.POOLABLE_CLASSIFICATION`) and is only reclassified
`refuted_mechanism` / `wrong_sign_resolved` when the WHOLE interval sits
strictly below zero AND the interval is not itself degenerate (too few
week-blocks to trust its bounds at all). `bounded_by_control` is never
applied automatically -- it requires an external positive-control result
this scaffold is not given.

**There are no 2026 prospective ledger rows yet** (Week 1 locks
2026-09-08), so this scaffold has nothing real to run on today. It is
exercised in `tests/test_refresh_triggers.py` on synthetic rows only,
including the case that matters most: a population engineered so the paired
estimate is exactly zero and the bootstrap interval straddles it --
`classification` stays `unresolved_below_power`, never a rejection.

## Live scan, 2026-09-04 (measured)

Run read-only against the real repository tree, `--scan --current`:

```
season=2026 week=1
evidence log: artifacts/refresh_triggers/2026/week_1.jsonl
reconstructed 32 trigger(s); appended 32 new, skipped 0 already-logged duplicate(s)
deadline_valid=False (excluded from any future comparison): 0
  injury_report_posted: 32
```

32 = 16 Week-1 games x 2 nflverse player-archive snapshots on disk
(`data/players/raw/20260812T200527Z`, `data/players/raw/20260817T184901Z`).
All 32 are `deadline_valid=True` -- both snapshots were captured in
mid-August, weeks before any Week-1 deadline. Zero triggers from every other
detector, all for measured, disclosed reasons, not gaps in the code:

- **`clock_checkpoint`**: zero, because `refresh_thu`/`refresh_sat`/
  `refresh_sun`/`refresh_*_inactives_*` are `season_guarded` and the 2026
  season has not started (`data/scheduler_state.json` carries no entries for
  any of those job names yet).
- **`inactives_posted`**: zero, because `data/players/inactives/` is empty
  -- WP17's capture has never run against a real in-season report (the
  season has not started).
- **`sportradar`** (part of `injury_report_posted`): zero, because
  `data/raw/sportradar_injuries/` does not exist on disk (dormant without a
  configured credential in this environment).
- **`lineup_change`**: zero, because this was the FIRST scan --
  `archive_lineup_snapshot` created the archive's only entry
  (`20260903T165444Z.json`) this run, so there is no second snapshot yet to
  diff against. A second scan after the lineup forecast next refreshes will
  produce a real comparison.
- **`line_move`**: zero, because `artifacts/clv_ledger/decisions.parquet`
  has no Week 1 rows yet (the Tuesday paper-lock has not run) --
  `pick_refresh.original_card` returns empty, and the detector fails open
  by design rather than guessing an opener.

Re-running the scan immediately afterward (idempotency check, also
measured): `reconstructed 32 trigger(s); appended 0 new, skipped 32
already-logged duplicate(s)` -- confirmed no duplicate rows.

## Running it

```powershell
# what the scan can currently reconstruct, read-only, appended to the evidence log
.\.tools\uv.exe run --no-sync python scripts\refresh_trigger_log.py --scan --current

# a specific week
.\.tools\uv.exe run --no-sync python scripts\refresh_trigger_log.py --scan --season 2026 --week 1
```

Never runs `refresh-picks`, `publish-predictions`, or a
`weak-signals`/`rotation` recorder, and never writes to `registry/`.
