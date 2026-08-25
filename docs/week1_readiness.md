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
   `artifacts/active_ats_model.json`. **Fixed, and corrected again 2026-08-18
   in place: the card path now follows the active profile instead of just
   detecting the mismatch.** An earlier draft of this fix (described below in
   "### 1", superseded within this same commit) only made `weekly-run` abort
   loudly on a mismatch. The version actually shipped goes further:
   `active_card_profile()` reads the active model's `feature_profile` and
   `CARD_PATH_TABLES` maps it to the right feature table, so step 3
   dynamically builds `weak_stack` (via `build-learned-availability-features`)
   whenever that is the active profile, and steps 4-5 score it — the card
   path now builds and publishes whatever is actually active, automatically.
   This makes Tuesday *runnable*, not merely *safe*: "What needs the owner's
   decision," #1 below is resolved, not open.
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

### 1. `weekly-run`'s card path now follows the active profile instead of reverting it

**Corrected 2026-08-18: this section originally described an intermediate,
abort-only draft of the fix (`_check_active_model_profile`, a hardcoded
`PLAYER_FEATURE_PROFILE` gate that raised on any mismatch). That function
does not exist in the shipped code — it was superseded within this same
commit by a fuller fix before anything was committed, and this section was
never updated to describe what actually shipped.**

`src/nfl_ats/weekly.py` gained `active_card_profile()` and
`CARD_PATH_TABLES`. `active_card_profile()` reads `active_ats_model.json`'s
`feature_profile` and looks it up in `CARD_PATH_TABLES` (currently `player`
and `weak_stack`); an unrecognised profile is fatal (guessing a feature table
for it would reintroduce the exact revert this exists to prevent), but a
*recognised* one is not an abort condition — the plan simply builds and
scores that profile. `plan_weekly_run` uses this to pick `card_profile`
dynamically, and when it resolves to `weak_stack`, step 3 gains an extra
`build-weak-stack-features` (`build-learned-availability-features`) entry
ahead of the scoring steps, and steps 4-5 (`margin-backtest`/`margin-predict`)
run against `game_features_weak_stack.parquet` with
`--feature-profile weak_stack` instead of `player`. The card path now
publishes whatever is actually active, automatically — it does not merely
detect and refuse a mismatch.

Reproduced live: `weekly-run --season 2026 --week 1 --dry-run
--skip-prospective` against the real artifacts root (active profile
`weak_stack`) prints a plan whose steps 3-5 build and score `weak_stack`,
not `player`. The real tests are `tests/test_weekly.py`
`test_the_card_path_follows_the_active_profile_instead_of_reverting_it`
(asserts the `weak_stack` table is built before scoring, and that scoring
uses `weak_stack`/`game_features_weak_stack.parquet`, never `player`) and
`test_an_unknown_active_profile_is_fatal_rather_than_guessed`
(the abort case, now scoped to genuinely unrecognised profiles only) plus
`test_run_proceeds_when_the_active_profile_already_matches_the_card_path`.
`tests/test_weekly.py` currently holds 22 tests, all passing; the wider
prospective/clv/publish/weekly/best_pick selection and the full suite were
also verified passing this session. `ruff format`, `ruff check`, and
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

1. ~~**How should `weekly-run`'s card path build `weak_stack`?**~~
   **RESOLVED — this was answered by shipped, tested code within the same
   session/commit that first raised it, not left to the owner.** See
   "What was fixed this session" #1, above: `active_card_profile()` +
   `CARD_PATH_TABLES` make the card path build whichever profile is active,
   dynamically, and step 3 gains a `build-weak-stack-features`
   (`build-learned-availability-features`) entry whenever that profile is
   `weak_stack`. `tests/test_weekly.py::test_the_card_path_follows_the_active_profile_instead_of_reverting_it`
   pins it. The owner is not owed this decision. A narrower question inside
   the original item is genuinely still open and still belongs to a
   concurrent session, not this one: the fate of the prospective-evidence
   tail (steps 8-11) — `mod07_weak_signal_stack` is registered as a
   *challenger* to the active model, but now that `weak_stack` is also the
   card-path profile, it scores the **identical** configuration the active
   model runs (same table, same profile, same alpha), comparing weak_stack
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

   **Owner decision, 2026-08-18: reset (option 1).** Delete the 16 rows so
   the real Tuesday publish on 2026-09-08 is genuinely the first write. A
   follow-up session was asked to execute that reset via the sanctioned
   "Known divergence" procedure in `docs/prospective_evidence.md`, backing
   up the ledger first. **It could not: the live ledger it was asked to
   edit does not exist.** A live check found `artifacts/clv_ledger/decisions.parquet`
   absent from the repo entirely — no file, not 16 rows, not 0 rows in an
   empty file, just no file at that path (`Get-ChildItem` on
   `artifacts/clv_ledger/` shows only the unrelated `20260817T104601Z/`
   scoring-run subdirectory). The 16-row state this item describes above is
   therefore stale as of 2026-08-18. `artifacts/prospective/challenger_decisions.parquet`
   is still genuinely absent, as before. A backup made by the session that
   found the 16 rows was located and verified byte-for-byte against this
   item's description — 16 rows, all season 2026/week 1, single
   `recorded_at_utc` batch `2026-08-18T01:24:56.231458Z`, `model_id`
   `4b01f055b684e27e`, `is_best_pick=True` on `2026_01_ARI_LAC` — at
   `...\56edf890-1650-456a-b560-8d8b00b374b6\scratchpad\ledger_backup_20260818\decisions.parquet`.
   Per this task's own stop condition (delete nothing if the live file
   doesn't hold exactly the described 16 rows), **no deletion was
   performed.** What is not known: whether another process already carried
   out the reset (in which case the empty state is correct and this item
   should be marked resolved once that is confirmed) or whether the local
   `artifacts/` tree was simply reset/lost between sessions (in which case
   the empty state is coincidental, not evidence of anything, and the
   underlying decision is still "open" in substance even though the file
   that would hold the contamination is gone). Next session: before
   treating Week 1 as clean, confirm which of those two it is — e.g. by
   checking whether any command that would produce this file ran in the
   interim — and only then update this item to **RESOLVED**.

   **RESOLVED, 2026-08-18.** `artifacts/clv_ledger/decisions.parquet` was
   re-checked (read-only) and is still absent — zero old-model rows.
   Marking this resolved rests on this reasoning: the end-state now matches
   the owner's chosen option (zero old-model rows; promoted model writes
   Week 1 fresh; refill guarded by opt-in recording + 7-day lock window with
   passing regression tests), the backup is preserved, and the undetermined
   cause is recorded honestly rather than resolved by assumption. That is,
   this does not claim to have determined *why* the file is absent — that
   question above is still genuinely open — only that the file's absence is
   itself sufficient to satisfy the owner's reset decision regardless of
   which of the two causes produced it, so the item no longer blocks
   2026-09-08.
3. **Both of the above are now resolved (2026-08-18)**, re-verified against
   this checklist's item 3, below, ahead of 2026-09-08. The `HANDOFF.md`
   half of item 2 is done: `ROADMAP.md`'s "Recommended execution order" item
   6, the source `HANDOFF.md` is generated from, no longer claims the
   2026-08-17 reset resolved anything, and `HANDOFF.md` was regenerated
   (`nfl_ats.handoff`) to carry the correction. The ledger-disposition half
   is also done — see the RESOLVED paragraph under item 2 above.

## Checklist (this session, live)

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | CLI commands match the docs | **PASS** | Ran `--help` for every command in `docs/ops_runbook.md` and `docs/prospective_evidence.md` (`doctor`, `weekly-run`, `publish-predictions`, `prospective-record`, `prospective-score`, `clv-ledger`, `margin-backtest`, `margin-predict`, `build-features`, `build-pbp-features`, `build-player-features`, `build-learned-availability-features`, `odds-summary`, `odds-ingest`, `ingest`). Every flag named in the docs exists and is spelled correctly; `weak_stack` is a valid `--feature-profile` choice on `margin-backtest`/`margin-predict`. No stale flags found. |
| 2 | Environment (`doctor`) | **PASS** | `nfl-ats` 0.2.0, `nflreadpy` 0.1.5, Python 3.12.13, scikit-learn 1.9.0. Latest raw snapshot `20260817T235649Z`, fetched same day, schedule/team-stat seasons run through 2026/2025 respectively — 2026 Week 1 schedule data is present. |
| 3 | `weekly-run`'s card path matches the active model | **FAIL, then FIXED** | See "What was fixed" #1. `weekly-run` would have silently reactivated the demoted `player` model on every future run, including Tuesday. **Corrected 2026-08-18: this row previously said it only "aborts loudly instead" and does not make the one-command path runnable.** The shipped fix goes further — `active_card_profile()`/`CARD_PATH_TABLES` make the card path build and score whichever profile is actually active, so the one-command path is runnable today, not merely safe. Owner decision #1 (the card-path question) is resolved, not open — see the corrected "What needs the owner's decision" section. |
| 4 | Real primary ledger is empty (the stated precondition for a first Week-1 write on 2026-09-08) | **FAIL, recurrence now prevented** | 16 real rows already present, `recorded_at_utc` 2026-08-18T01:24:56Z, `model_id` `4b01f055b684e27e`, `is_best_pick=True` on `2026_01_ARI_LAC`. See owner decision #2 for the rows themselves. Recording is now opt-in (`--record-decisions`) and separately refuses outside a real lock week (`RECORDING_LOCK_WINDOW`) — reproduced against the incident's own data, it refuses. The **challenger** ledger (`artifacts/prospective/challenger_decisions.parquet`) genuinely does not exist — only the primary ledger was affected. |
| 5 | Fail-closed / anti-backdating guarantees have tests | **PASS** | Refuse-at-or-after-kickoff: `tests/test_prospective_scoring.py::test_scoring_refuses_a_pick_recorded_at_or_after_its_own_kickoff`, `test_record_challenger_records_dedupes_and_refuses_started_games`, `tests/test_clv.py::test_record_paper_decisions_records_dedupes_and_skips_started`. Never-rewrite/dedupe: the same tests plus `test_challenger_ledger_rejects_duplicate_rows`. Re-check on read: the scoring-refuses test above. Best Pick's three rules (whole-week pre-kickoff, first-write-wins, exactly one per week): `test_best_pick_is_never_nominated_once_any_game_of_the_week_has_started`, `test_best_pick_is_first_write_wins_across_republications`, `test_legacy_ledger_without_the_flag_loads_and_two_flags_a_week_is_rejected`. Retuned-configuration refusal: `test_record_challenger_refuses_a_retuned_configuration`. Fingerprint-not-recency artifact matching: `test_artifact_lookup_matches_on_fingerprint_not_on_recency`. No guarantee was found without a test. The one gap that *did* exist — nothing verified the card path's hardcoded profile against the live active model — is closed by this session's new tests. |
| 6 | Test suite | **PASS** | Full suite (`pytest tests`, no filter): **655 passed**, 0 failed, after both rounds of fixes this session (the profile guard, the Best Pick tie disclosure, and the recording-lock-window guard). The targeted `-k "prospective or clv or publish or weekly or best_pick or cli"` selection: 120 passed. `ruff format --check`, `ruff check`, `mypy src` (69 files) all clean on every file touched, both rounds. |
| 7 | Best Pick disclosure on the published card | **FAIL, then FIXED** | See "What was fixed" #2. |
| 8 | Odds/line capture health | **UNCHANGED FROM PRIOR SESSION, still worth watching** | `capture_log.txt` still shows only the two historical FAIL lines (2026-08-16, 2026-08-17T23:00:00Z); no capture has run since the prior session's fix to `scripts/odds_capture.ps1`, so that fix remains unverified against a real scheduled run (next: `Odds_ThuTNF` or `Odds_Sat`). All six scheduled tasks (`Odds_TueOpen`, `Odds_ThuTNF`, `Odds_MonMNF`, `Odds_Sat`, `Odds_SunClose`, `Odds_SunLate`) are `Ready`. `odds-summary` run live this session: 8,100 snapshots, last observation 2026-08-17T23:00:04Z (today), 5,946,403 quote rows — the pipeline is alive and current. |

## Corrected command sequence for Tuesday 2026-09-08

**Corrected 2026-08-18: this used to say "do not run `weekly-run` as-is until
owner decision #1 is resolved." Owner decision #1 is resolved — see "What
needs the owner's decision" above — so the one-command path below is the
primary, preferred route today, not a conditional one.** The manual fallback
remains only for the week the one command breaks.

### `weekly-run`'s card path builds/activates `weak_stack` automatically

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

## 2026-08-24 rehearsal (operational dry run for the Tuesday 2026-09-08 lock)

Executed live by an agent session owning only this file and
`artifacts/rehearsal_lockday/`. Every number below is **measured** in that
session unless tagged otherwise; commands and artifact paths are named inline.
No ledger write occurred at any point, and `--record-decisions` was attempted
only as the step-4 guard probe. Nothing was committed or pushed.

### Verdict per step

| # | Step | Verdict | Evidence (all measured) |
|---|---|---|---|
| 1 | Ledger snapshot | **GO** | `artifacts/clv_ledger/decisions.parquet` ABSENT; `artifacts/prospective/challenger_decisions.parquet` ABSENT (python pandas existence check). |
| 2 | `nfl-ats doctor` | **GO** | 2.87s. Clean JSON: nfl_ats 0.2.0, nflreadpy 0.1.5, Python 3.12.13, scikit-learn 1.9.0; latest raw snapshot `20260817T235649Z`, schedule_seasons through 2026 (Week 1 schedule present). Nothing non-green. |
| 3 | One-command path without recording | **NO-GO** | 617.7s (~10m18s vs the documented ~4m21s budget). Steps 1–6 completed; **aborted at step 7 `ingest-player-arrests`**: `Expecting value: line 1 column 1 (char 0)`. The arrests snapshot itself completed fine — a full 56-page snapshot with manifest exists at `data/raw/player_arrests/20260824T110928Z`. Root cause read from source: `_cli_runner` (src/nfl_ats/weekly.py:610) does bare `json.loads()` on captured stdout, but `scripts/ingest_player_arrests.py:340` prints `Fetched page N/M` progress lines to stdout before the manifest JSON. **Every real lock-day run fetches fresh pages, so this crash is deterministic on Sept 8.** |
| 3b | Manual-fallback publish (no-record), to complete the card path | **PARTIAL** | 202.1s. Card + README written (both files' mtimes 11:15:53Z match), then **crashed** with `KeyError: 'surface_switch_flag'` in `build_public_site` â†’ `_challenger_week_previews` â†’ `apply_surface_switch_tilt_overlay` (src/nfl_ats/public_board.py:3314 â†’ src/nfl_ats/surface_switch_tilt_overlay.py:291). The publish JSON summary never printed. `best_pick_tied` was instead measured read-only via `_publication_context`: **false**, best pick `2026_01_MIA_LV` (MIA +3.5), rule v2, model `d1f07d773475dc58`, season/week 2026/1, 16 games. The card's `(?)` after "Best Pick of the week" is the marker column, not a tie flag (read: src/nfl_ats/publishing.py:140). |
| 4 | Recording guard refuses outside lock window | **GO** (guard proven; command itself currently broken) | Full `publish-predictions --record-decisions --no-board` ran 209.8s and crashed at src/nfl_ats/cli.py:650 (`record_surface_switch_tilt_challenger_decisions`) with the same `KeyError` — AFTER the caught-and-stored CLV-guard refusal, BEFORE its summary could print. The refusal text was therefore captured by calling `record_paper_decisions(Path("artifacts"), data_root=Path("data"), now=None)` directly (probe script kept at `artifacts/rehearsal_lockday/guard_probe_step4.py`): *"Refusing to record to the paper-decision ledger: this week's earliest kickoff (2026-09-10T00:20:00+00:00) is 16 days after the recording instant … more than RECORDING_LOCK_WINDOW (7 days)…"*. Ledger re-checked ABSENT immediately after both the crashed run and the probe. Deviation noted: `--no-board` was added because the board builder's KeyError aborts before reaching the guard block; on a fixed tree the plain command is expected to report the same refusal in `clv_ledger.error`. |
| 5 | Scheduled-capture freshness | **GO** | `data/market/capture_log.txt` exists (1,132 bytes; an earlier recursive search missed it because permission-denied `.promotion-*` dirs abort `Get-ChildItem -Recurse`; corrected by direct path). Two historical FAIL lines only (2026-08-16, 2026-08-17), then five consecutive OK: 08-18, 08-20, 08-22, 08-23 16:30, 08-23 20:15 — so both script fixes called "unverified" in checklist item 8 are now verified against real scheduled runs. Latest market snapshot `data/market/raw/20260823T201503Z`, rows=4580, quota_remaining=1496. All six tasks `Ready`: Odds_TueOpen next 08-25 09:00, Odds_ThuTNF last 08-20 result 0, Odds_Sat 08-22 result 0, Odds_SunClose 08-23 12:30 result 0, Odds_SunLate 08-23 16:15 result 0, Odds_MonMNF last 08-17 result **1** (next today 19:00 — watch it). `public_betting_live` latest `20260823T160001Z`; `public_betting` latest `20260820T111148Z`. |
| 6 | Challenger readiness (read-only) | **PARTIAL GO** | `artifacts/prospective/challengers.json` (unchanged since 08-22): 26 entries — 20 `ACTIVE_PROSPECTIVE`, 4 `SUPERSEDED_BY_PROMOTION`, 1 `CLOSED_BEFORE_ACTIVATION`, 1 `DEACTIVATED_STRUCTURAL_NO_OP`. All 20 active entries carry a non-empty config fingerprint (`bc77638d47e2748c…` ×18, `b53f07cf61b09b4b…` ×2). Per-challenger card generation deliberately skipped: two of the consumers of challenger overlays are exactly what is crashing (see 3b/4), and a second session was editing overlay code mid-rehearsal, so generated cards would be misleading. |
| 7 | Final ledger counts | **GO** | Both parquets ABSENT again; `artifacts/prospective/` contains only `challengers.json` (mtime 08-22). Identical to step 1. |

### Additional measured finding: silent model-identity swap under concurrent edits

At session start the active manifest was model `3083f6cbc5e45acb`
(feature_table_sha256 `0a18e2d9…`, activated 2026-08-20T00:50:17Z). After the
step-3 run it is **`d1f07d773475dc58`** (activated 2026-08-24T11:09:27Z,
feature_table_sha256 `853595a5…`; same evaluation configuration sha
`d5259477…`, same close-grade 1,081/2,075 = 52.10%). My own weekly-run's
feature rebuild produced a different feature-table hash and margin-predict
activated the matching fresh evaluation — while a second session was actively
editing `src/nfl_ats/cli.py` (07:11 local), `src/nfl_ats/prospective.py`
(07:15), and `scripts/vi_dispersion_screen.py` (07:20, two seconds before I
clocked it), plus new untracked overlay work. *Inferred*: their in-flight
changes altered the built feature table. Either way the mechanism is
measured: running the standard path on a non-quiescent tree changed the
public card's stated model identity with no explicit promotion decision. The
card diff beyond timestamps was exactly: model-id swap in the header lines of
both tracked files, and two probability cells moved 0.1pp (ATL@PIT 53.4â†’53.5,
CHI@CAR 53.5â†’53.4); no pick, policy, or Best Pick changes.

### Fix before Sept 8

1. **Fix `_cli_runner` stdout parsing (blocker, deterministic on lock day).**
   src/nfl_ats/weekly.py:610 must not bare-`json.loads()` output that contains
   progress lines; either route `Fetched page N/M` to stderr in
   scripts/ingest_player_arrests.py or parse only the final JSON document.
   Add a regression test that a fresh-fetch ingest step still yields one
   parseable summary.
2. **Fix the `surface_switch_flag` KeyError (blocker, two call sites).**
   src/nfl_ats/surface_switch_tilt_overlay.py:291 breaks both the public-site
   build (public_board.py:3314) and the `--record-decisions` challenger
   recorder (cli.py:650). Root-cause whether this comes from the concurrently
   edited overlay wiring landed 2026-08-24 morning or from a production-card
   frame genuinely lacking the column; add a regression test covering both
   call sites.
3. **Enforce a quiescent tree during the lock window.** No parallel editing
   session may run during the Sept 8 sequence; item 3b's evidence shows why.
4. **Re-time the card path after fixes** on an idle machine: this rehearsal
   measured ~10m18s for steps 1–7 versus the documented ~4m21s budget
   (confounded by concurrent load; treat the old budget as unverified until
   re-measured).
5. **Watch Odds_MonMNF tonight** (last result 1 on 08-17; next run 19:00
   local) and confirm `capture_log.txt` gains an OK line.
6. Re-run this rehearsal end-to-end once 1–2 land, including capturing a clean
   weekly-run JSON summary (`published`, `failed_step`, `best_pick_tied`,
   model id/season/week) — never obtained this session because both attempts
   crashed after the card write.

Informational, not blocking: margin-backtest emitted
`BootstrapDegeneracyWarning` (season-block bootstrap has <10 blocks; per the
warning's own text, report the estimate and probability_positive, not the
interval).

Nothing was written to `artifacts/clv_ledger/`,
`artifacts/prospective/challenger_decisions.parquet`, or any registry;
`CURRENT_PREDICTIONS.md` content changed only via the natural no-record
publish described above. No commit, no push.

---

## 2026-08-25 update: what actually records, and how it stays recorded

Added after a sweep found that the single most-cited command ("run
publish-predictions --record-decisions") does not by itself record every
active challenger. Verified against source and the live registry this session.

### The lock-day command

**Tuesday 2026-09-08:**

```powershell
.\.tools\uv.exe run --no-sync nfl-ats weekly-run --season 2026 --week 1 --record-decisions
```

It must be `weekly-run`, not `publish-predictions`. `mod07_weak_signal_stack`
records via `weekly-run` step 11 (`prospective-record`), which a bare
`publish-predictions --record-decisions` never invokes.

### Everything later in the week is scheduled

`model_only_refresh_incumbent` and `injury_signal_refresh_tilt` record ONLY
through `refresh-picks`, which `weekly-run` never calls; the NFL.com arm
additionally needs a live injury page captured inside a per-game window. Left
to anyone remembering a cadence three times a week for eighteen weeks, those
challengers record nothing and nobody notices until the season is over — the
identical silent-no-op this same sweep found in the gate itself.

Both are now jobs in `scripts/capture_scheduler.py` (`refresh_thu`,
`refresh_sat`, `refresh_sun`, plus the four `injuries_*` windows). See
`docs/capture_scheduling.md` for the mechanism. Any session can also force a
catch-up, idempotently:

```powershell
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --once
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --status
```

**Not yet exercised end-to-end:** the refresh jobs have never fired, because
the season guard keeps them dormant until the run-up to week 1 and the agent
harness blocked direct `refresh-picks` invocation. The injury capture branch
ran live
2026-08-25. Worth watching the first time the refresh branch reports DUE.

### Recording accept/refuse window (measured)

`RECORDING_LOCK_WINDOW = 7 days` (`src/nfl_ats/clv.py`) against Week 1's
earliest kickoff, `2026-09-10T00:20:00Z` (NE@SEA, Wednesday):

- **REFUSED** at any instant before **2026-09-03T00:20:00Z**.
- **ACCEPTED** from that instant onward.

The planned Tuesday lock (2026-09-08, ~noon ET) sits comfortably inside the
accept window. Running earlier than Sep 3 publishes the card but records
nothing, silently.

### State as of 2026-08-25

- 2026 Week 1 injury page captured (`data/raw/nflcom_injuries/20260825T191422Z`,
  0 rows — the league has not published it yet, which is expected this far out).
  The season-long 2026 data gap is closed as a repeating process.
- The refresh pass is correctly NOT due yet: the first pick deadline
  (2026-09-09 20:20 ET, NE@SEA) is more than six days out.
- Nothing here is waiting on a human. The catch-up command is the whole
  operating procedure; run it whenever, including right before the Tuesday
  lock-day command above.

---

## 2026-08-25 (evening): the recording chain rehearsed clean, end to end

The 2026-08-24 rehearsal above never obtained a clean run -- both attempts
crashed after the card write, and its own fix-list item 6 asked for a re-run.
This is that re-run, and it covers ground the earlier one could not: the
RECORDING chain, all the way through the late-week refresh pass.

Two new tracked scripts do it, and both are re-runnable:

* `scripts/lockday_rehearsal.py` -- drives every real recorder at a simulated
  lock instant against an isolated artifacts root.
* `scripts/lockday_verify.py` -- the aggregate check that did not exist. Run
  it right after the real Tuesday command and after each refresh pass.

### Why a rehearsal needed new machinery

Two guards make this chain unrehearsable at wall-clock time, and they pull in
opposite directions. `clv.refuse_if_outside_recording_lock_window` refuses any
write whose week's earliest kickoff is more than 7 days out, so nothing records
before 2026-09-03. `player_arrests_back_side_overlay.MAX_SNAPSHOT_AGE` refuses
any arrests snapshot more than 36 hours older than the recording instant, so
nothing fetched today is fresh relative to a simulated 2026-09-08. On the real
lock day weekly-run step 7 (`ingest-player-arrests`, **fatal**, measured via
`weekly-run --dry-run`) resolves this by fetching minutes before step 8
publishes.

The rehearsal reproduces that by shifting the CLOCK, not the data: every
recorder accepts a `now` override, and the data root is mirrored with hard
links (no extra disk, removing the mirror cannot touch the originals) with only
the 3.7 MB arrests tree real-copied so one snapshot can be restamped. Nothing
fabricated is written into the production data root.

### The finding: prospective evidence lives in FOUR ledgers, not one

This is what made a silent no-op invisible. Measured by enumerating every
parquet the rehearsal wrote:

| Ledger | Challengers | Written by |
|---|---|---|
| `prospective/challenger_decisions.parquet` | 16 | `publish-predictions --record-decisions` |
| `prospective/injury_signal_refresh_decisions.parquet` | `injury_signal_refresh_tilt` | `refresh-picks --record-decisions` |
| `prospective/pick_revisions.parquet` | `model_only_refresh_incumbent` | `refresh-picks --record-decisions` |
| `prospective/nflcom_friday_refresh_decisions.parquet` | `nflcom_friday_refresh_out2_starters_v1` | `refresh-picks --record-decisions` |

Any audit that reads only the shared challenger ledger reports four of the
twenty active challengers as missing when they are fine. The first version of
this rehearsal's own coverage check made exactly that mistake.

Compounding it: `cli._cmd_publish_predictions` wraps seventeen recorders in
`try/except -> {"recorded": 0, "error": ...}` so a broken challenger can never
un-publish the card. Correct for the card, wrong for the evidence -- zero rows
and a successful-looking run are indistinguishable without reading twenty
nested JSON keys. `lockday_verify.py` is that missing aggregate: it reads all
four ledgers, cross-references the run's JSON summary, and classifies every
active challenger as **recorded**, **skipped** (zero rows AND a named gate), or
**MISSING** (zero rows, no explanation).

### Result (measured, 2026-08-25)

Simulated lock 2026-09-08T16:00Z, refresh 2026-09-10T19:00Z, against the real
active model `d1f07d773475dc58` and its real Week 1 card:

**17 recorded, 3 skipped with a named gate, 0 MISSING, of 20 active.**
Paper ledger 16 rows; Best Pick `2026_01_MIA_LV` (rule v2, no tie).

The three gated skips are correct behaviour, not defects, and two of them are
operational facts worth knowing before Tuesday:

* `nflcom_friday_refresh_out2_starters_v1` **cannot record at the Tuesday
  lock** -- its gate needs a page fetched at or after Friday 16:00 ET. It
  records only on a Saturday/Sunday refresh pass. Judging it at the lock is
  judging it too early.
* `model_only_refresh_incumbent` records only games whose pick actually
  CHANGED. A week where nothing moves legitimately writes zero rows.
* `movement_rule_composed_v1` skipped on `latest_capture_not_from_today`,
  which is a rehearsal-clock artifact: the newest real capture is 2026-08-25
  and the simulated instant is 2026-09-08. On the real Tuesday the opener
  capture lands that morning (`capture_scheduler` job `odds_tue_open`), so
  this one should record -- and if it does not, the verifier will now say so
  instead of it passing unnoticed.

### Also measured

* `mod07_weak_signal_stack`'s configuration fingerprint resolves to
  `margin_predictions/2026-week-01-20260824T120725Z` -- **the active model's own
  linked weekly forecast**. It records 16 rows, but it is comparing
  `weak_stack` to itself, exactly the structural no-op flagged as open on
  2026-08-18. Its rows will carry no information. Registry disposition (a
  `DEACTIVATED_STRUCTURAL_NO_OP` status already exists for this) is an owner
  call, not a rehearsal's.
* Overlay recorders resolve some sibling paths from `artifacts_root.parent`
  (the interim-coach join reads `<repo>/data/raw/interim_coaches`). A rehearsal
  root parked anywhere but beside a `data/` directory fails those joins open to
  zero flags and records rows that never exercise the signal. The sim tree is
  laid out as `<sim>/artifacts` beside `<sim>/data` for this reason.
* `cli._cmd_publish_predictions` passes `now=publish_instant` to four recorders
  and lets the other thirteen read the wall clock. On lock day those agree to
  within seconds, so it is not a lock-day defect -- but the recorded instants
  will differ slightly between challengers.

### The Tuesday 2026-09-08 sequence, unchanged plus one line

```powershell
.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --once
.\.tools\uv.exe run --no-sync nfl-ats weekly-run --season 2026 --week 1 --record-decisions > lockday_summary.json
.\.tools\uv.exe run --no-sync python scripts\lockday_verify.py --season 2026 --week 1 --run-summary lockday_summary.json
```

The third line is new and is the point: it exits non-zero if any active
challenger recorded nothing without naming a reason, while every game of the
week is still ahead and the row can still be written.
