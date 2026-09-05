# Ledger reconciliation and recovery (ENG-15)

Written 2026-09-04, four days before Week 1 locks.

## The problem

`weekly-run --record-decisions` writes into up to seven append-only ledgers
through recorders that are deliberately **fail-open**
(`try/except -> {"recorded": 0, "error": ...}`, `src/nfl_ats/cli.py`): a
broken challenger must never un-publish the card. That is the right trade for
the card and the wrong one for the evidence — a recorder that silently wrote
nothing still produces a run that reports success, and the failure is buried
in one of twenty nested JSON keys nobody reads
(`scripts/lockday_verify.py`'s own docstring makes the same point about the
narrower question it answers).

`lockday_verify.py` already answers "did every registered challenger record
something, or a documented gate reason, this week?" This module answers a
different, complementary question for a **partial weekly failure**: *given
what actually landed on disk, what exactly is wrong, and what is the exact
command that fixes it?* It joins three things that are supposed to agree but
are written by different code paths:

1. **recorder result summaries** — a saved `weekly-run`/`publish-predictions`
   JSON blob (`--run-summary`), or the `recorders` section of an ENG-01
   lock-day decision package manifest (`--package`,
   `docs/lockday_package.md`). Both are optional; the reconciler works fully
   without either.
2. **each append-only ledger's own rows** for the requested season/week, read
   directly with `pandas.read_parquet` — never through the strict `load_*`
   loaders in `clv`/`prospective_scoring`/the five refresh modules, which
   raise `DataContractError` on *any* duplicate row anywhere in the whole
   file. That is correct for a recorder about to write; it would make a
   reconciler crash instead of reporting exactly the corruption it exists to
   find.
3. **the published card's picks** — parsed from the tracked
   `CURRENT_PREDICTIONS.md`, the externally-visible artefact a paper-decision
   row is supposed to match. Only the active model's paper-decision ledger is
   compared this way; challenger picks are never published to that file.

Everything here is **read-only**. Nothing in this module writes a ledger row,
deletes anything, or mutates a historical row. Running the reconciler twice
against an unchanged tree returns the same report and leaves every ledger's
bytes untouched (`tests/test_ledger_reconcile.py::test_reconcile_is_idempotent_and_never_writes`
pins this by comparing file bytes before and after two calls).

## The six classifications

Every registered recorder — the active model's paper-decision ledger (id
`active_model`) plus every `ACTIVE_PROSPECTIVE` challenger in
`artifacts/prospective/challengers.json` — is classified into exactly one of:

| status | meaning |
|---|---|
| `consistent` | rows exist for the week, no duplicate idempotency keys, and (when comparable) the ledger's pick agrees with the published card. |
| `missing_rows` | a result summary or package explicitly declared rows were recorded this run, but the ledger has **zero** rows for the week. The recorder claimed success and the ledger disagrees. |
| `orphan_rows` | rows exist in the shared challenger ledger under a `challenger_id` that is **not present in `challengers.json` at all** — a stray write (typo, deleted registration, hand-edited row). |
| `duplicate_rows` | the week's own slice of a ledger has more than one row under the same idempotency key (table below). |
| `card_mismatch` | (`active_model` only) the ledger's recorded `pick_side` disagrees with the same game's pick as parsed from the published card, for a week the card actually covers. |
| `not_run` | everything else: no ledger rows and no summary evidence of a legitimate gate/error, or a recorder with no CLI wiring at all (`PENDING_WIRING` in `lockday_verify.py`'s vocabulary). |

Priority order inside the classifier: **duplicates are checked first** (a
ledger that is internally inconsistent is reported as such regardless of what
a summary claims), then a summary-declared-but-absent write
(`missing_rows`), then a card disagreement, then plain presence
(`consistent`), then the various `not_run` reasons.

### Challenger ids registered but not `ACTIVE_PROSPECTIVE`

A challenger that recorded picks while it was active and was later
superseded (e.g. `SUPERSEDED_BY_PROMOTION`) is **not** `orphan_rows` — that
would misclassify legitimate history as corruption purely because time
passed. Its rows are surfaced separately, in the report's
`informational_inactive_challengers_with_rows` list, and excluded from the
six-way classification entirely.

## Idempotency keys, by ledger kind

The recovery plan's "is re-running safe" answer is read directly off each
recorder's own write path, not guessed:

| ledger kind | file(s) | idempotency key | re-run behaviour |
|---|---|---|---|
| paper ledger | `artifacts/clv_ledger/decisions.parquet` | `game_id` | `record_paper_decisions` (`src/nfl_ats/clv.py`) skips any `game_id` already present before appending — **safe, and a true no-op** for already-recorded games. |
| shared challenger ledger | `artifacts/prospective/challenger_decisions.parquet` | `(challenger_id, game_id)` | `record_challenger_decisions` (`src/nfl_ats/prospective_scoring.py`) skips any `(challenger_id, game_id)` already present — **safe, and a true no-op**. |
| dedicated refresh ledgers | `pick_revisions.parquet`, `injury_signal_refresh_decisions.parquet`, `nflcom_friday_refresh_decisions.parquet`, `inactives_refresh_decisions.parquet`, `crew_tilt_refresh_decisions.parquet` | `(game_id, refresh_run_id)` | **read**: none of the five `record_*` functions has an already-recorded check. Each is a revision log by design (that is what lets a late-week refresh record a changed pick), so re-running the documented command is safe from corruption but **always appends a new row** — never a no-op. |

The dedicated ledgers therefore use a looser duplicate key than the paper and
shared ledgers: two rows for the same `game_id` under two *different*
`refresh_run_id` values are normal (successive refresh passes); two rows for
the same `game_id` **and** the same `refresh_run_id` are not, and that *is*
flagged `duplicate_rows`
(`tests/test_ledger_reconcile.py::test_dedicated_ledger_different_refresh_run_id_is_not_a_duplicate`
and its `_same_refresh_run_id_twice` sibling pin both halves of this).

## Recovery commands

For every non-`consistent` recorder the report includes a `recovery_plan`
entry with the exact command to re-run, derived from the registry's own
`weekly_recording_command` field (never fabricated):

* If the field is directly runnable, optionally followed by a parenthetical
  explanation, the command up to `" ("` is used verbatim, with `--season N`
  and `--week N`/`--week <N>` substituted for the requested season/week.
* If the field is prose beginning `"N/A -- ..."` (some recorders have no
  standalone command — e.g. `injury_signal_refresh_tilt` and
  `model_only_refresh_incumbent` are only recordable as a side effect of
  `refresh-picks --record-decisions`), `rerun_command` is `None` and the raw
  explanatory text is returned unchanged as `rerun_command_raw` rather than
  guessing a command that does not exist.
* `active_model`'s recovery command is always
  `nfl-ats publish-predictions --record-decisions`, with a note that this
  command takes no `--season`/`--week` — it always acts on whichever
  forecast is currently linked as the active model.
* A `duplicate_rows` recorder is flagged `rerun_is_safe: false`: the strict
  `load_*` loader for that ledger will raise `DataContractError` on read
  until the duplicate is resolved, and re-running the recorder will not fix
  an existing duplicate (it may add another).
* A `card_mismatch` recorder's note explicitly flags that this **may be
  expected, not a defect** — republishing a card with a moved line never
  rewrites an already-recorded CLV anchor
  (`docs/prospective_evidence.md`, "The anti-backdating guarantee").

Nothing in `recovery_plan` executes anything. It is report-only, by design —
recovery is always a human decision.

## Reading the report

```powershell
.\.tools\uv.exe run --no-sync python scripts\ledger_reconcile.py --season 2026 --week 1
.\.tools\uv.exe run --no-sync python scripts\ledger_reconcile.py --season 2026 --week 1 --json
.\.tools\uv.exe run --no-sync python scripts\ledger_reconcile.py --season 2026 --week 1 `
    --package artifacts\lockday_packages\2026_wk01_<stamp>
.\.tools\uv.exe run --no-sync python scripts\ledger_reconcile.py --season 2026 --week 1 `
    --run-summary path\to\saved_weekly_run_summary.json
```

Exit code is `0` only when every recorder classifies `consistent`. The human
render additionally folds in one summary line from `lockday_verify.py` (the
existing wiring-level audit — is every `ACTIVE_PROSPECTIVE` challenger even
reachable from a CLI command) so a single invocation surfaces both questions;
`lockday_verify.py` still needs to be run directly for its own full detail.
No edit was made to `scripts/lockday_verify.py` to add this — the reconciler
only imports it (`nfl_ats.ledger_reconcile.load_lockday_verify_module`, the
same dynamic file-path import `nfl_ats.lockday_package` already uses) and
calls its existing `verify()` function.

`--package` accepts either an ENG-01 lock-day decision package directory or
its `manifest.json` directly (`nfl_ats.lockday_package.load_package`'s own
contract). Its `ledgers` section (measured row-count deltas, read *after* a
real run wrote) is preferred over a `--run-summary`'s self-reported
`"recorded"` count when both are supplied for the same recorder
(`build_declarations`, package wins). The reconciler was written and tested
before `lockday_package.py` existed in this session and was updated once it
landed; the integration is defensive (`_load_package_manifest` catches any
import or shape failure and degrades to "no package supplied") so a future
change to that module's schema cannot break this one's read-only-without-it
guarantee.

## Live read-only result (measured 2026-09-04)

The real `artifacts/clv_ledger/decisions.parquet` and
`artifacts/prospective/challenger_decisions.parquet` do not exist in this
repository yet: recording is scoped to 2026 onward
(`docs/prospective_evidence.md` — the paper ledger only ever records the
*current* published week, and `prospective-score` backtests before 2026 are
explicitly out of scope), and the real Week 1 2026 lock has not happened yet
(`CURRENT_PREDICTIONS.md` is dated 2026-09-03, a pre-lock preview; kickoffs
begin 2026-09-09). The only artefact under `artifacts/clv_ledger/` is an
archived `20260817T104601Z/scored_decisions.parquet` from a rehearsal that
was later reset (`docs/prospective_evidence.md`, "Known divergence").

The most recent season with any settled games is 2025. Measured directly
against `data/raw/20260824T115346Z/schedules.parquet`: every 2025 regular-season
week (1–18) has `result` populated; week 18 is the last `REG` week (weeks
19–22 are postseason `WC`/`DIV`/`CON`/`SB`).

```powershell
.\.tools\uv.exe run --no-sync python scripts\ledger_reconcile.py --season 2025 --week 18
```

reports, against the real repository, read-only:

* `registry: loaded` (29 `ACTIVE_PROSPECTIVE` challengers read from the real
  `artifacts/prospective/challengers.json`).
* `card: not published for the requested season/week` — the real
  `CURRENT_PREDICTIONS.md` is 2026 Week 1, so `card_mismatch` correctly
  cannot fire for this query.
* **30 of 30 recorders `not_run`** (`active_model` plus all 29 registered
  challengers) — because no real production ledger has ever recorded a row
  for 2025 at all, which is the exactly correct read of a system whose
  recording window has not opened yet, not a defect.
* Exit code `1` (correctly non-zero, since not every recorder is
  `consistent`).
* The folded-in `lockday_verify` line independently reports "0 recorded, 5
  skipped, 24 MISSING, 0 pending wiring of 29 active" for the same
  season/week — a different, wiring-focused verdict landing on the same
  underlying fact (nothing has ever been recorded for 2025).

This run created, deleted, or modified nothing: `git status` is unaffected by
it, and no file under `artifacts/` or `data/` changed.

## Known limitations

* **`missing_rows` needs a summary or package.** Without one, "the recorder
  ran and wrote nothing" and "the recorder was never invoked" are
  indistinguishable from the ledger alone, and both correctly classify
  `not_run` rather than a false `missing_rows`.
* **A stray row written into the shared challenger ledger under a
  dedicated-ledger challenger's id** (never expected from any production
  write path — the two ledgers are entirely separate files) is silently
  skipped rather than flagged, because that id's real evidence is read from
  its own ledger file, not the shared one. Documented in-line in
  `nfl_ats.ledger_reconcile.reconcile` where the skip happens.
* **`card_mismatch` only ever applies to `active_model`.** Challenger picks
  are never published to `CURRENT_PREDICTIONS.md`, so there is nothing to
  compare a challenger ledger row against.
* **The reconciler trusts the registry's `weekly_recording_command` text
  verbatim** for recovery commands. If that field in `challengers.json` is
  stale or wrong, the derived command inherits the mistake — this module
  does not independently verify that a command actually works.

## Where the code lives

| file | role |
|---|---|
| `src/nfl_ats/ledger_reconcile.py` | the read-only join/classify/recovery-plan logic |
| `scripts/ledger_reconcile.py` | thin CLI wrapper; `--json`, `--run-id`, `--run-summary`, `--package` |
| `tests/test_ledger_reconcile.py` | synthetic-tree tests covering all six classifications and the idempotency guarantee; never touches the real `artifacts/`/`data/` |

`scripts/ledger_reconcile.py` is allowlisted in
`tests/test_experiment_registry.py::_ALLOWLISTED_UNSTAMPED_SCRIPTS` for the
same reason `lockday_verify.py`/`lockday_package_verify.py` are: it is a
read-only operational audit that prints JSON to stdout and writes nothing
under `artifacts/`, never an experiment with a hypothesis, cell, or verdict.
