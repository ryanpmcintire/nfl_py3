# Prospective leads 2026-09-23

## Goal

Register the market-move-decomposition, opener-error-transfer, and
total-conditioned-lattice research candidates as prospective paired
challengers of the served four-term probability, recorded at each weekly
lock without changing the served card, where each has a genuine
point-in-time live input path.

## State

**Registration mechanism (read/measured):** `artifacts/prospective/challengers.json`
(`nfl_ats.prospective_scoring.challenger_registry_path`,
`src/nfl_ats/prospective_scoring.py:290-358`) lists challengers with
`status: ACTIVE_PROSPECTIVE`; `active_challenger_ids()` drives what
`scripts/lockday_verify.py` audits as wired vs `unwired` against
`PUBLISH_CHALLENGER_RESULT_KEYS`/`REFRESH_CHALLENGER_RESULT_KEYS`
(`lockday_verify.py:260-330`), which check that
`weekly_recording_command` corresponds to a real recorder called from
`nfl-ats publish-predictions --record-decisions` or `refresh-picks
--record-decisions`. `docs/lanes/week3-research-recorders.md` is the live
precedent (recorders wired this way, audited weekly).

**Root correction (mid-task):** the all-books median market move was
re-graded on the active Sunday-inclusive window and does NOT beat the
served move (-0.13 pts [-0.59, +0.33], P+ 0.22; `docs/lanes/market-move-decomposition.md`
Unit 3). Dropped from registration per root's instruction, though it did
have a live input path (odds-ingest already captures all
`LEADERSHIP_WEIGHTS` books multiple times daily via the scheduled
`odds_capture.ps1` -> `nfl-ats odds-ingest --markets spreads,h2h,totals` job,
`scripts/capture_scheduler.py:41,404` on).

**Registered (measured, prior session):** `total_conditioned_key_number_lattice_v1`
added to `artifacts/prospective/challengers.json`. Live-dry-run verified
against Week 3 2026 (16/16 games, point-in-time correct); see git history
of this lane for the sample rows.

**Wired and fully verified this session (DONE):**
- New module `src/nfl_ats/total_conditioned_lattice_challenger.py`:
  `total_band()`/`total_conditioned_read()`/`TOTAL_BAND_EDGES`/
  `TOTAL_BAND_LABELS`/`CHALLENGER_POLICY`/`ATOM_TOLERANCE` moved here
  verbatim from `scripts/total_conditioned_lattice.py` (which now imports
  them back, `scripts/total_conditioned_lattice.py:10-26`). New
  `record_total_conditioned_lattice_decisions(artifacts_root, data_root,
  *, now=None, forecast_artifact=None, replace_week=False)`: reads the
  active forecast card's `recommendations.csv` for the week's games,
  kickoffs, `spread_line`/`total_line`/served `home_cover_probability`,
  builds the point-in-time prior pool via
  `nfl_ats.mass_preserving_lattice.prior_pool()`/`prior_pool_for_week()`
  over `data/processed/game_features_weak_stack.parquet`, computes
  `total_conditioned_read()` per game (point = spread_line, matching the
  registered protocol), and writes one row per pre-kickoff game to a
  dedicated ledger `prospective/total_conditioned_lattice_decisions.parquet`.
- `src/nfl_ats/cli_commands/publishing.py`: import, `PUBLISH_CHALLENGER_RESULT_KEYS`
  entry, call site, and default skip stub all added (same shape as the
  `low_total_div_home_dog_challenger_ledger` block it sits next to).
- `artifacts/prospective/challengers.json`: `weekly_recording_command`
  rewritten to describe the real wiring. **JSON validity confirmed.**

**All 5 verification items run this session -- item 3 found and fixed a
real bug in the new recorder, item 4 found and fixed a real gap in the
audit script, everything else passed clean:**

1. **PASS (measured).** `json.load(open('artifacts/prospective/challengers.json'))`
   parses; `nfl_ats.prospective_scoring.active_challenger_ids(Path('artifacts'))`
   returns 63 ids including `total_conditioned_key_number_lattice_v1`;
   `find_challenger` returns the full entry with the rewritten
   `weekly_recording_command`. JSON not corrupted.
2. **PASS (measured).** `ruff format --check` + `ruff check` clean on
   `publishing.py` and `total_conditioned_lattice_challenger.py`. `mypy`
   clean on both files together (`Success: no issues found in 2 source
   files`) and on `mypy src` in full (`Success: no issues found in 240
   source files`).
3. **FIXED then PASS (measured).** First run raised
   `TypeError: Invalid comparison between dtype=datetime64[us] and
   Timestamp` inside `prior_pool_for_week` -- the recorder converted
   `features["gameday"]` to a **tz-naive** timestamp
   (`total_conditioned_lattice_challenger.py:205`, was
   `pd.to_datetime(features["gameday"], errors="raise")`) but compares it
   against the tz-aware `recorded_at` cutoff inside `prior_pool_for_week`.
   Fixed by adding `utc=True`, matching the sibling pattern in
   `lattice_centre_challenger.py:132` (`pd.to_datetime(schedules["gameday"],
   utc=True)`). Re-ran the recorder in-process against the REAL
   `artifacts/`/`data/` trees with only
   `total_conditioned_lattice_challenger.ledger_path` monkeypatched to a
   scratchpad file (never touched the live ledger): 16/16 Week 3 2026 rows
   written, `missing_total_line_skipped: 0`, `post_kickoff_skipped: 0`.
   Sample parity check against the earlier registration session's dry-run
   values: `2026_03_CIN_PIT` challenger_cover 0.496514 now vs 0.4965 then
   (match); `2026_03_SEA_WAS` challenger_cover 0.512463 now vs 0.5125 then
   (match) -- pool construction confirmed identical. (`served_cover_probability`
   drifted between the two samples because the active forecast card was
   regenerated between sessions -- expected, not a bug.)
4. **PASS after a fix (measured).** `scripts/lockday_verify.py --season
   2026 --week 3 --json`: `total_conditioned_key_number_lattice_v1` is NOT
   in `unwired_recorders`/`pending_wiring` (both empty for it), so the
   wiring itself is confirmed live. It DID initially surface a second real
   gap: the challenger wasn't in `lockday_verify.py`'s `DEDICATED_LEDGERS`
   map, so the audit fell back to checking the generic shared ledger
   (`prospective/challenger_decisions.parquet`) instead of this
   challenger's actual dedicated ledger
   (`prospective/total_conditioned_lattice_decisions.parquet`) -- it would
   have shown `MISSING` forever even after a real production recording
   pass. Fixed by adding a `total_conditioned_key_number_lattice_v1` entry
   to `DEDICATED_LEDGERS` (`scripts/lockday_verify.py`, using the module's
   own `load_decisions`), matching the `tiebreaker_lattice_centre`
   pattern. Re-ran: now shows `"ledger":
   "prospective/total_conditioned_lattice_decisions.parquet"`,
   `"recording_path": "publish/dedicated"`, `status: MISSING, rows: 0` --
   MISSING/0 is expected and correct because no real (non-dry)
   `--record-decisions` has been run against production this session per
   the task's constraint; it will resolve to `recorded` on the next real
   weekly-run pass. `ruff format --check`/`ruff check` clean on
   `lockday_verify.py`; `mypy scripts/lockday_verify.py` shows only
   pre-existing `import-untyped` noise identical to every other sibling
   import in that file (scripts/ isn't on `mypy src`'s configured path) --
   not a regression from this edit.
5. **PASS (measured), folded into item 3.** Confirmed directly:
   `resolve_recording_forecast`/`load_active_ats_model` against the real
   `artifacts/` resolves to
   `artifacts/margin_predictions/2026-week-03-20260923T161033Z`
   (`SYNCHRONIZED`), and its `recommendations.csv` has 16 rows with
   `total_line` present on every row (spot-checked all 16). The lane's
   worry that the served card might lack `total_line` was unfounded -- no
   fix needed there.

Tests: `pytest tests/test_cli.py tests/test_lockday_contract.py
tests/test_venue_flag_features.py -q` (the three live test files matching
`PUBLISH_CHALLENGER_RESULT_KEYS`/`low_total_div_home_dog` --
`tests/test_low_total_div_home_dog_challenger.py`,
`tests/test_lockday_verify.py`, `tests/test_prospective.py` only exist as
stale `.pyc` cache, source files are gone) -- **44 passed**, warnings are
pre-existing/unrelated (bootstrap degeneracy, an unrelated
`sharp_book_movement_features.py` deprecation warning).

This challenger is now fully wired and verified: registered, recorder
implemented and bug-fixed, `publishing.py` call site clean, and
`lockday_verify.py` correctly classifies it as wired with the right
dedicated ledger. Nothing further blocks it; the only remaining step is a
real production `--record-decisions` run, which is the primary
orchestrator's job, not this lane's.

**Not registered -- no live input path:** college-trained opener-error term
(`docs/lanes/opener-error-transfer.md`). Confirmed via
`grep -ic cfb scripts/capture_scheduler.py` = 0: no scheduled job pulls CFB
lines. `data/cfb/lines/raw/20260816T143907Z/manifest.json` is a single
historical snapshot from LEAD-53-era research, not a recurring capture.
`scripts/opener_error_transfer_unit3.py` is a manual/offline research
script, not wired to any weekly job. Nothing to register until a weekly
CFBD pull (CFBD_API_KEY already in user env) is added to
`capture_scheduler.py` and a live point-in-time CFB-to-NFL transfer scorer
is built.

## Tried

- Full read of `prospective_scoring.py`, `cli_commands/prospective.py`,
  `lockday_verify.py` challenger dicts, and an existing full registry entry
  (`pbp08_protection_mismatch_early_window_v1`) as the format precedent.
- Live dry run of total-conditioned-lattice against Week 3 2026 (script at
  `C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\...\scratchpad\total_conditioned_lattice_live_dry_run.py`,
  not in repo -- scratchpad only).
- Confirmed odds-ingest already captures all `LEADERSHIP_WEIGHTS` books
  (moot now given the market-move drop).
- This session: ran all 5 verification items end to end against the real
  `artifacts/`/`data/` trees (only the ledger write target was redirected
  to scratchpad via a monkeypatch of `ledger_path`, never the live
  ledger). Fixed a tz-naive/tz-aware comparison bug in the recorder and a
  missing `DEDICATED_LEDGERS` entry in `lockday_verify.py`. Ran the
  matching test files once (44 passed).

## Next (root)

1. Challenger is fully wired and verified. Nothing blocks it. Fold this
   lane's state into `docs/lanes/week3-research-recorders.md` or close it
   (move to `docs/lanes/done/`) per the session's own judgment.
2. When root next runs the real weekly `nfl-ats publish-predictions
   --record-decisions` (its own operational job, not a subagent action),
   confirm `total_conditioned_key_number_lattice_v1` flips from `MISSING`
   to `recorded` in `lockday_verify.py`'s output with 16 rows in
   `prospective/total_conditioned_lattice_decisions.parquet`.
3. If college opener-error is still wanted: schedule a weekly CFBD pull in
   `capture_scheduler.py`, then build a live scorer before registering.

## Open

- No served-path change, no commit/publish, no registry entries touched
  besides the one addition (`weekly_recording_command` text) from the
  prior session. No live ledger was written this session -- the dry run
  used a monkeypatched `ledger_path` writing to a scratchpad-only parquet.
- Two real bugs were found and fixed this session (both in code, not
  policy): the tz-naive/tz-aware `gameday` comparison in
  `total_conditioned_lattice_challenger.py`, and the missing
  `DEDICATED_LEDGERS` entry in `scripts/lockday_verify.py`. Both are
  verified fixed (ruff/mypy clean, re-run confirmed correct behavior).
- `git status` at session end also shows unrelated modified files
  (`docs/findings.html`, `docs/history.html`, `docs/index.html`,
  `docs/model.html`, `docs/lanes/line-move-regrade-legacy.md`,
  `scripts/roof_state_line_move_replication.py`) that this lane's session
  did not touch -- concurrent work from elsewhere, out of this lane's
  scope.
