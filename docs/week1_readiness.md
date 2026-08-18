# Week 1 readiness check (2026-08-18, second pass)

Written 2026-08-18T12:10Z, 21 days before the pool locks Tuesday
2026-09-08T16:00Z (noon ET). This supersedes the 2026-08-18T00:10Z version of
this file. That version's steps 1-3 verification (CLI flags, doctor, the
mechanics of freeze/record/settle) reproduced cleanly and still hold; this
pass found two live problems that version did not catch, because both were
created by code and by a real ledger write made *during* that same prior
session, after its own checklist had already declared them clean. Everything
below was reproduced live in this session, not copied from the prior doc.

## Verdict

**NOT READY as of this writing, but every defect found is diagnosed, three of
four fixes are code-complete and verified live, and none is a data-loss risk
if handled before 2026-09-08.** The one open item — what to do with the 16
already-contaminated ledger rows — is now safe to leave open past this
session, because the guard that made deletion-alone insufficient the first
time is fixed: the same command cannot recontaminate the ledger a third time.

1. **`weekly-run`'s one-command card path was broken by the same promotion
   that made it necessary.** Three commits ago (`68b4dc0`) the active model
   was promoted from the `player` feature profile to `weak_stack`. Nobody
   updated `src/nfl_ats/weekly.py`, which still hardcodes `player` for the
   card path (steps 4-5). Run as-is, `weekly-run` would rebuild the `player`
   evaluation, silently reactivate it (overwriting `active_ats_model.json`),
   pass the synchronization check anyway (season/week still match), and
   publish the **demoted** model — reverting the promotion with no error, no
   warning, every week, forever. **Reproduced live** against the real
   `artifacts/active_ats_model.json`. **Fixed**: `weekly-run` now refuses to
   run at all when the active model's `feature_profile` disagrees with the
   hardcoded card-path profile, converting a silent revert into a loud abort
   — reproduced live against the real repo (see below). This makes Tuesday
   *safe*, not *runnable*: someone still has to decide how the card path
   should build `weak_stack` (see "What needs the owner's decision," #1).
2. **The real primary CLV ledger is not empty, and deletion alone would not
   have stayed fixed.** The safety brief for this task, and the
   just-refreshed `HANDOFF.md`, both assert `artifacts/clv_ledger/decisions.parquet`
   holds nothing so that Tuesday's publish is the first write. It is not
   empty: it holds 16 real 2026 Week 1 rows, `recorded_at_utc`
   **2026-08-18T01:24:56Z** — this morning, from model `4b01f055b684e27e`
   (the pre-promotion `player`-profile id, activated roughly seven minutes
   before the `weak_stack` promotion), with `is_best_pick=True` already
   locked onto `2026_01_ARI_LAC`. This was not written by this session; it
   predates this session's first command by hours and was written by an
   ordinary, real `publish-predictions` run during a prior session's own
   "live testing" — the same session that, in the same document, claimed the
   real ledger was still empty. **The root cause was that recording was
   opt-out** (`--skip-clv-ledger`): the default behavior of the exact command
   Tuesday uses was to write the real ledger, so testing it for real wrote
   it. Deleting the 2026-08-17 rehearsal rows (the "reset" already documented
   in `docs/prospective_evidence.md`) never touched that default, which is
   why the ledger was recontaminated within hours. **Fixed this session**
   (see "What was fixed" #3): recording is now opt-in everywhere, and a
   second, function-level guard refuses any recording whose week is not
   close to its own kickoff — reproduced against the actual incident's own
   data and confirmed it would have refused. The 16 existing rows themselves
   are still the owner's call; I did not touch the file.

## What was fixed this session (code + tests, all verified live)

### 1. `weekly-run` fails closed instead of silently reverting the promotion

`src/nfl_ats/weekly.py` gained `_check_active_model_profile`, called at the
top of `run_weekly` (before any step, including under `--dry-run`). It reads
`active_ats_model.json` if present; if its `feature_profile` differs from
`PLAYER_FEATURE_PROFILE` (`"player"`), it raises before touching anything:

```
error: Active model '118f31d9a98c815b' uses feature_profile='weak_stack', but
weekly-run's card path (steps 4-5) is hardcoded to feature_profile='player'
(nfl_ats.weekly.PLAYER_FEATURE_PROFILE/PLAYER_FEATURE_TABLE). Running would
rebuild and reactivate that profile and publish it, silently reverting
whatever promoted the currently active model. Update
PLAYER_FEATURE_PROFILE/PLAYER_FEATURE_TABLE (and the feature-table build step
feeding them) to match the active profile, or otherwise resolve the
mismatch, before running weekly-run.
```

Reproduced live: `weekly-run --season 2026 --week 1 --dry-run
--skip-prospective` against the real artifacts root prints exactly this and
exits nonzero, right now. Two tests added to `tests/test_weekly.py`
(`test_run_aborts_before_any_step_when_the_active_profile_disagrees_with_the_hardcoded_card_path`,
`test_run_proceeds_when_the_active_profile_already_matches_the_card_path`);
the full `test_weekly.py` (18 tests) and the wider prospective/clv/publish/
weekly/best_pick selection (103 tests) pass. `ruff format`, `ruff check`, and
`mypy src` (69 files) are clean.

This guard does **not** decide which profile is correct — that is a research
decision belonging to whoever owns the promotion, not to a readiness dry run.
It only converts "silently do the wrong thing" into "refuse and say why."

### 2. The published card now discloses a tied Best Pick (fold-in from a parallel finding)

The live Week 1 card is a two-way tie at the top of the `sweep_robustness`
signal (`2026_01_ARI_LAC` and `2026_01_WAS_PHI`, both 8.0; next best 5.5).
`select_best_pick` breaks it alphabetically onto `ARI_LAC`. The dashboard
(`src/nfl_ats/dashboard/app_pages/picks.py`) already disclosed ties; the
tracked, public card (`src/nfl_ats/publishing.py`, which writes
`CURRENT_PREDICTIONS.md`) had no tie logic at all and would have shown
`ARI +10.5` as an unqualified Best Pick.

Fixed by extracting the shared computation into
`nfl_ats.best_pick.best_pick_tie_count`/`best_pick_tie_note` (previously
`picks.py` computed the tie count inline; now both callers share one
definition of "tied"). `publishing.py` threads the note through
`_forecast_best_pick` -> `_publication_context` -> `_publication_header` /
`_best_pick_note`, and `publish_active_predictions`'s return payload now
carries `best_pick_tied: bool` so a script (or the Tuesday runbook) can check
it without parsing prose. Two tests added to `tests/test_publishing.py`
proving a tied week discloses and a clean week does not; both pass, along
with the rest of the publishing/best_pick suites.

**Verified against the real, live, tied Week 1 card**, not a synthetic
fixture: ran `nfl-ats publish-predictions --skip-clv-ledger --destination
<scratchpad>/CURRENT_PREDICTIONS_copy.md --readme <scratchpad>/README_copy.md`
against the real `artifacts/` root (at the time, before "What was fixed" #3
replaced `--skip-clv-ledger` with the opt-in `--record-decisions` — the same
verification today would need no ledger flag at all, since not recording is
now the default). Output: `"best_pick_tied": true`, and the rendered card
reads *"This week 2 games tie at the top of that signal, so choosing between
them is arbitrary -- reproducible, but not a lean."* right after the Best
Pick note. The redirected `--destination`/`--readme` and the (then-explicit,
now-default) skip of the ledger write mean this touched no tracked file and
wrote no ledger row (`git status` confirms; the real ledger's mtime is
unchanged).
**`CURRENT_PREDICTIONS.md` was deliberately left stale** — the fix is
code-complete and verified, but regenerating the real tracked file competes
with the ledger-contamination decision below (a real publish would also
just no-op against the 16 rows already there), so that regeneration should
happen once, deliberately, after the owner resolves both open items, not
twice.

### 3. Recording is now opt-in, and refuses to fire outside a real lock week

Two layered fixes, so no single missed flag can reach a real ledger again:

- **`--skip-clv-ledger` is gone.** `publish-predictions` and `weekly-run`
  both gained `--record-decisions` (default `False`). Neither writes a paper
  decision (or, for `weekly-run`, a challenger decision via step 10) unless
  it is passed explicitly. Without it, `weekly-run` still publishes the card
  and still builds/scores the challenger's own card for informational
  purposes (steps 8, 9, 11) — it just records nothing.
- **`nfl_ats.clv.refuse_if_outside_recording_lock_window`** (used by both
  `record_paper_decisions` and, via a shared import, `record_challenger_decisions`
  in `prospective_scoring.py`) refuses to write whenever a week's earliest
  kickoff is more than `RECORDING_LOCK_WINDOW` (7 days) away from the
  recording instant. This lives inside the recording functions themselves,
  not the CLI, so it also covers `clv-ledger` (whose own `--skip-record` is
  still opt-out, unchanged) and `prospective-record` (which has always
  recorded unconditionally when invoked) — every path into either ledger,
  not just the one that caused the incident.

**Verified against the actual incident, read-only:** replayed
`refuse_if_outside_recording_lock_window` against the real Week 1 forecast's
kickoffs and the incident's own recording instant
(`2026-08-18T01:24:56Z`) — refuses, naming the 22-day gap. Replayed again at
a real-Tuesday-lock instant (`2026-09-08T16:00:00Z`) against the same
kickoffs — does not refuse. Five new tests
(`tests/test_clv.py::test_record_paper_decisions_refuses_a_recording_weeks_before_kickoff`,
`tests/test_prospective_scoring.py::test_record_challenger_refuses_a_recording_weeks_before_kickoff`,
`tests/test_cli.py::test_publish_predictions_does_not_record_by_default` /
`test_publish_predictions_records_with_the_explicit_flag`,
`tests/test_weekly.py::test_record_decisions_defaults_to_false_and_does_not_reach_either_ledger`
/ `test_record_decisions_true_wires_both_ledger_writes` /
`test_run_weekly_forwards_record_decisions_and_reports_it`). Full suite: 655
tests pass. `ruff format --check`, `ruff check`, `mypy src` (69 files) clean.

`docs/prospective_evidence.md`'s "Known divergence" section now records this
correction in place — the original "RESOLVED 2026-08-17" claim is kept
verbatim, not reworded, with the correction stated plainly above it, matching
the project's rule that negative results are not silently removed.
`docs/ops_runbook.md` documents the new flag in both the one-command and
manual-fallback sections.

## What needs the owner's decision

1. **How should `weekly-run`'s card path build `weak_stack`?** It is not a
   drop-in constant swap: the main path currently only builds
   `game_features_player.parquet` (via `build-features` /
   `build-pbp-features` / `build-player-features`); `weak_stack` needs
   `build-learned-availability-features` run first, which today only exists
   as optional step 8 (the "challenger" step). Whoever resolves this should
   also decide the fate of the prospective-evidence tail (steps 8-11):
   `mod07_weak_signal_stack` is registered as a *challenger* to the active
   model, but it now scores the **identical** configuration the active model
   would run (same table, same profile, same alpha) — comparing weak_stack
   to itself. The natural replacement question — is `player` (the demoted
   baseline) now the thing worth challenging `weak_stack` with prospectively
   — requires a new registry entry, and `artifacts/prospective/challengers.json`
   plus `docs/pool_edge_plan.md` are both owned by a concurrent session this
   round.
2. **What to do about the 16 contaminated ledger rows.** Same two options
   `docs/prospective_evidence.md` already lays out for exactly this
   situation ("Known divergence"): reset again (delete the 16
   `2026`/week `1` rows so the real Tuesday publish is genuinely the first
   write, freeing the Best Pick nomination to be made from the Tuesday
   card), or accept these as a second rehearsal artifact and note that
   `is_best_pick` for Week 1 is permanently the arbitrary `ARI_LAC` tie-break
   from a picture of the model three weeks stale by kickoff. **I did not
   delete or modify the ledger** — that is exactly the kind of irreversible
   write/delete this dry run was told to avoid making unilaterally. A backup
   of the 16 rows exists outside the repo. Unlike 2026-08-17, whichever
   choice is made now **will hold**: recording is opt-in and
   kickoff-window-gated as of this session (see "What was fixed" #3), so the
   same ordinary command cannot silently repopulate the ledger a third time.
   `HANDOFF.md`'s "resolved" claim still needs correcting regardless of which
   option is chosen, which is outside this file's ownership;
   `docs/prospective_evidence.md`'s copy of the same claim has been corrected
   in place.
3. Both of the above should be resolved (and re-verified against this
   checklist's item 3, below) before 2026-09-08.

## Checklist (this session, live)

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | CLI commands match the docs | **PASS** | Ran `--help` for every command in `docs/ops_runbook.md` and `docs/prospective_evidence.md` (`doctor`, `weekly-run`, `publish-predictions`, `prospective-record`, `prospective-score`, `clv-ledger`, `margin-backtest`, `margin-predict`, `build-features`, `build-pbp-features`, `build-player-features`, `build-learned-availability-features`, `odds-summary`, `odds-ingest`, `ingest`). Every flag named in the docs exists and is spelled correctly; `weak_stack` is a valid `--feature-profile` choice on `margin-backtest`/`margin-predict`. No stale flags found. |
| 2 | Environment (`doctor`) | **PASS** | `nfl-ats` 0.2.0, `nflreadpy` 0.1.5, Python 3.12.13, scikit-learn 1.9.0. Latest raw snapshot `20260817T235649Z`, fetched same day, schedule/team-stat seasons run through 2026/2025 respectively — 2026 Week 1 schedule data is present. |
| 3 | `weekly-run`'s card path matches the active model | **FAIL, then FIXED** | See "What was fixed" #1. `weekly-run` would have silently reactivated the demoted `player` model on every future run, including Tuesday. Now aborts loudly instead. Does not by itself make the one-command path runnable — see owner decision #1. |
| 4 | Real primary ledger is empty (the stated precondition for a first Week-1 write on 2026-09-08) | **FAIL, recurrence now prevented** | 16 real rows already present, `recorded_at_utc` 2026-08-18T01:24:56Z, `model_id` `4b01f055b684e27e`, `is_best_pick=True` on `2026_01_ARI_LAC`. See owner decision #2 for the rows themselves. Recording is now opt-in (`--record-decisions`) and separately refuses outside a real lock week (`RECORDING_LOCK_WINDOW`) — reproduced against the incident's own data, it refuses. The **challenger** ledger (`artifacts/prospective/challenger_decisions.parquet`) genuinely does not exist — only the primary ledger was affected. |
| 5 | Fail-closed / anti-backdating guarantees have tests | **PASS** | Refuse-at-or-after-kickoff: `tests/test_prospective_scoring.py::test_scoring_refuses_a_pick_recorded_at_or_after_its_own_kickoff`, `test_record_challenger_records_dedupes_and_refuses_started_games`, `tests/test_clv.py::test_record_paper_decisions_records_dedupes_and_skips_started`. Never-rewrite/dedupe: the same tests plus `test_challenger_ledger_rejects_duplicate_rows`. Re-check on read: the scoring-refuses test above. Best Pick's three rules (whole-week pre-kickoff, first-write-wins, exactly one per week): `test_best_pick_is_never_nominated_once_any_game_of_the_week_has_started`, `test_best_pick_is_first_write_wins_across_republications`, `test_legacy_ledger_without_the_flag_loads_and_two_flags_a_week_is_rejected`. Retuned-configuration refusal: `test_record_challenger_refuses_a_retuned_configuration`. Fingerprint-not-recency artifact matching: `test_artifact_lookup_matches_on_fingerprint_not_on_recency`. No guarantee was found without a test. The one gap that *did* exist — nothing verified the card path's hardcoded profile against the live active model — is closed by this session's new tests. |
| 6 | Test suite | **PASS** | Full suite (`pytest tests`, no filter): **655 passed**, 0 failed, after both rounds of fixes this session (the profile guard, the Best Pick tie disclosure, and the recording-lock-window guard). The targeted `-k "prospective or clv or publish or weekly or best_pick or cli"` selection: 120 passed. `ruff format --check`, `ruff check`, `mypy src` (69 files) all clean on every file touched, both rounds. |
| 7 | Best Pick disclosure on the published card | **FAIL, then FIXED** | See "What was fixed" #2. |
| 8 | Odds/line capture health | **UNCHANGED FROM PRIOR SESSION, still worth watching** | `capture_log.txt` still shows only the two historical FAIL lines (2026-08-16, 2026-08-17T23:00:00Z); no capture has run since the prior session's fix to `scripts/odds_capture.ps1`, so that fix remains unverified against a real scheduled run (next: `Odds_ThuTNF` or `Odds_Sat`). All six scheduled tasks (`Odds_TueOpen`, `Odds_ThuTNF`, `Odds_MonMNF`, `Odds_Sat`, `Odds_SunClose`, `Odds_SunLate`) are `Ready`. `odds-summary` run live this session: 8,100 snapshots, last observation 2026-08-17T23:00:04Z (today), 5,946,403 quote rows — the pipeline is alive and current. |

## Corrected command sequence for Tuesday 2026-09-08

**Do not run `weekly-run` as-is until owner decision #1 (above) is
resolved — it will now abort at the pre-flight check rather than publish
the wrong model, but it still will not publish the right one.** Two paths,
depending on what has landed by then:

### If `weekly-run`'s card path has been updated to build/activate `weak_stack`

```powershell
git status --short
.\.tools\uv.exe run --no-sync nfl-ats doctor
.\.tools\uv.exe run --no-sync python -m nfl_ats weekly-run --season 2026 --week 1 --record-decisions
```
**`--record-decisions` is required** — without it this publishes the card but
records nothing to either ledger (safe default since this session; see "What
was fixed" #3). Read the JSON summary: `"published": true`, no
`"failed_step"`, `"record_decisions": true`, and check `"best_pick_tied"` in
the `publish-predictions` step's output — if `true`, the Best Pick nomination
is an arbitrary tie-break among that many games and the owner may prefer to
pick between the tied games by hand rather than accept alphabetical order.

### Manual fallback, valid today, reflecting the currently-promoted model

```powershell
git status --short
.\.tools\uv.exe run --no-sync nfl-ats doctor
.\.tools\uv.exe run --no-sync python -m nfl_ats ingest --start-season 2009 --end-season 2026 --stats-end-season 2025
.\.tools\uv.exe run --no-sync python -m nfl_ats build-features
.\.tools\uv.exe run --no-sync python -m nfl_ats build-pbp-features --snapshot <PBP_SNAPSHOT>
.\.tools\uv.exe run --no-sync python -m nfl_ats build-player-features --player-snapshot <PLAYER_SNAPSHOT> --player-value-snapshot <VALUE_SNAPSHOT> --pbp-snapshot <PBP_SNAPSHOT>
.\.tools\uv.exe run --no-sync python -m nfl_ats build-learned-availability-features --features data\processed\game_features_pbp.parquet --destination data\processed\game_features_weak_stack.parquet --rates-destination data\processed\weak_stack_availability_rates.parquet --evaluation-destination data\processed\weak_stack_availability_evaluation.csv --player-snapshot <PLAYER_SNAPSHOT> --player-value-snapshot <VALUE_SNAPSHOT> --pbp-snapshot <PBP_SNAPSHOT>
.\.tools\uv.exe run --no-sync python -m nfl_ats margin-backtest --features data\processed\game_features_weak_stack.parquet --feature-profile weak_stack
.\.tools\uv.exe run --no-sync python -m nfl_ats margin-predict --season 2026 --week 1 --features data\processed\game_features_weak_stack.parquet --feature-profile weak_stack
# Check by hand: active manifest reads SYNCHRONIZED, weekly_forecast season/week
# = 2026/1, AND feature_profile = weak_stack. Do not publish if any fail.
.\.tools\uv.exe run --no-sync python -m nfl_ats publish-predictions --with-board --record-decisions
```
**`--record-decisions` is required here too** — same reasoning as the
one-command path. Then check `best_pick_tied` the same way as above, and
confirm the ledger write recorded 16 (or however many games) new rows, not
"already_recorded" — if owner decision #2 was "accept," expect
`already_recorded` instead and treat that as correct, not a bug. If instead
it raises citing `RECORDING_LOCK_WINDOW`, the machine clock or the schedule
data is wrong; do not investigate a way around it, investigate the data.

**Timing unchanged from the prior measurement:** budget 15 minutes; the card
path alone measured ~4m21s in the last full rehearsal. Start any time after
the ~06:00-09:00 ET Tuesday-opener capture lands; finish comfortably before
11:30 ET so entries are in before the 12:00 lock.

## What was deliberately not done

- **The real `artifacts/clv_ledger/decisions.parquet` was read but never
  written or deleted.** It already held the 16 contaminating rows before
  this session started; they were not created by this session, and removing
  them is the owner's call (see decision #2), not a dry run's. A backup of
  the 16 rows was taken outside the repository before any further work
  (`...\scratchpad\ledger_backup_20260818\decisions.parquet`, 16 rows,
  verified) in case the owner wants to inspect or restore them.
- **`artifacts/prospective/challenger_decisions.parquet` still does not
  exist** and was not created.
- **`CURRENT_PREDICTIONS.md` and `README.md` were not modified.** The Best
  Pick tie-disclosure fix was verified against the real, live Week 1 card by
  redirecting `--destination`/`--readme` to the scratchpad and skipping the
  ledger write, so the fix is proven against real production data without
  touching any tracked file or writing a ledger row.
- **`artifacts/active_ats_model.json` was not modified** by this session.
  (`nfl-ats doctor` and `weekly-run --dry-run` only read it.)
- No commit, no push, no `git add`.
