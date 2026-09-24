# Coordinator adjudication 2026 (PER-07 / LEAD-29)

## Goal
Keep `data/raw/coordinators/` snapshots adjudicated so
`interim_playcaller_first_game_back_overlay.py` and `playcaller_change_screen.py`
(`screen`, `production-lead29`) resolve playcaller-change events against the
newest snapshot instead of falling back to a stale one.

## State
Captured a fresh snapshot and wrote its `change_validation.json`. Both readers
now resolve to `data/raw/coordinators/20260924T193505449093Z/`:
- `playcaller_change_screen._latest_coordinator_dir()` -> that dir; `load_counted_events()` returns 12 events.
- `interim_playcaller_first_game_back_overlay._latest_under(..., "*/change_validation.json")` -> same dir/file; `load_counted_playcaller_change_events()` returns 12 rows.

**Measured**: the new snapshot's `coverage_audit.json["changes"]` (16 raw
identity edits) is field-for-field identical, on every non-adjudicated column
(`previous_observed_at`, `source_url`, `after_first_kickoff`,
`revision_lag_days`), to the 16 edits already adjudicated in
`data/raw/coordinators/20260907T213814366437Z/change_validation.json`
(10 `staff_change`, 2 `playcaller_role`, 4 noise: 2 `reverted_identity_edit`,
1 `role_correction`, 1 `unverified_identity_edit`). Cause: `ingest_coordinator_history.py`'s
in-season identity-scan range is still hardcoded 2022-2025 (needs a completed
season schedule; 2026 isn't done), so this snapshot's in-season scan produced
the same 2022-2025 population, no new raw edit to adjudicate.

**No real 2026 in-season coordinator/HC change found.** Checked
`data/raw/pfr_transactions/20260923T110056Z/yearly/2026.parquet` (44
coordinator/playcaller-keyword rows, all dated January-February 2026 —
offseason hiring cycle, already reflected in the 2026 preseason row of
`coordinator_history.parquet`) and `data/raw/injury_news/20260924T162524Z`
(zero coordinator/playcaller keyword matches). Nothing mid-season yet — expected
this early (Week 3).

## Tried
1. Capture command (from `scripts/capture_scheduler.py` job `coordinators_tue`):
   `.tools/uv.exe run --no-sync python scripts/ingest_coordinator_history.py --league --inseason --max-requests 100 --delay 1.0`
   -> new snapshot `data/raw/coordinators/20260924T193505449093Z` (cache-backed,
   ran clean).
2. `.tools/uv.exe run --no-sync python scripts/ingest_coordinator_history.py --audit data/raw/coordinators/20260924T193505449093Z`
   -> wrote that snapshot's `coverage_audit.json`.
3. Diffed new `coverage_audit.json["changes"]` vs old adjudicated
   `change_validation.json["changes"]` by key (season/team/role/person/previous_person/revision_at)
   and by shared raw fields: exact match, 16/16, 0 mismatches outside the
   adjudicated-only fields (`event_date`, `lag_status`, `validation_status`,
   `validation_note`, `independent_source`, `revision_calendar_day_lag`).
   -> carried the prior adjudication forward verbatim into the new snapshot's
   `change_validation.json` (same schema: `_provenance_stamp`,
   `additional_playcaller_changes`, `calendar_lags`, `changes[]`,
   `confirmed_staff_changes`, `method`, `raw_identity_edits`), updating only
   `_provenance_stamp` (`code_revision` cb31da8, `recorded_at` 2026-09-24T19:37:50Z,
   `code_dirty: false`) and `method` (documents the carry-forward and the
   PFR/injury-news check for a new 2026 event).
4. Verified both consumer readers resolve to the new snapshot (see State).

## Next
- Weekly `coordinators_tue` re-run: if `coverage_audit.json["changes"]` grows
  past 16 or any row's season is 2026, that new row needs hand adjudication
  (public source, `event_date`, `lag_status`, `validation_status` in
  {`staff_change`,`playcaller_role`,noise statuses}) before
  `screen`/`production-lead29`/the overlay will see it.
- PER-07 itself needs no further action here; its coach-identity/aggressiveness
  findings are already closed/pooled per `ROADMAP.md` line 547 and untouched by
  this lane.

## Open
- None. This is a maintenance/adjudication unit, not a research finding; no
  card, grade, or page changed.

## 2026-09-24 update: hardcoded season ceiling removed
**Fixed** `scripts/ingest_coordinator_history.py` (3 edits, no comments):
`league_capture`'s inseason job list no longer hardcodes `range(2022, 2026)`;
it now derives `inseason_seasons` from `schedules.parquet` rows with a
non-null `home_score`/`away_score` (season >= 2022 floor kept, no upper
ceiling). Each team/season's Wikipedia `rvend` cutoff is now computed only
from that team's *completed* games (`home_score`/`away_score` not null)
instead of `gameday.max()` over the whole (possibly future) schedule; a
team/season with zero completed games is skipped (`continue`), so an
in-progress season only scans what's already been played —
`after_first_kickoff`/`revision_lag_days` semantics untouched.
`audit_capture`'s `changes` loop changed `history.season.between(2022, 2025)`
to `history.season.ge(2022)` (population is bounded by what was actually
captured, not a literal end-year).

**Ran** (real `coordinators_tue` argv): `.tools/uv.exe run --no-sync python
scripts/ingest_coordinator_history.py --league --inseason --max-requests 100
--delay 1.0` -> new snapshot
`data/raw/coordinators/20260924T194157799750Z` (**measured**: `manifest.json`
`network_requests: 32`, `stopped: null`, `assignments: 3107`; all 32 teams'
2026 inseason jobs fired for the first time, log line
`inseason 2026 WAS: 3107 assignments; 32 requests`). Then `--audit
data/raw/coordinators/20260924T194157799750Z` -> `coverage_audit.json`
(**measured**: `inseason_complete: 160` = 5 seasons x 32 teams, up from 128;
`changes` length still **16**, identical set to the prior adjudicated
snapshot; **zero rows have `season == 2026`**).

**No hand adjudication needed this run** — no 2026 in-season identity edit
appeared (still Week 3; consistent with the "nothing mid-season yet" finding
already on file). Carried the existing `change_validation.json` forward
verbatim (same 16 adjudicated rows, same schema, via
`nfl_ats.provenance.write_stamped_artifact`) onto
`data/raw/coordinators/20260924T194157799750Z/change_validation.json`
(`_provenance_stamp.code_revision` `b2e6f996`, `code_dirty: true` — working
tree has this unit's uncommitted script edit; `recorded_at`
2026-09-24T19:45:27Z). **Verified both consumer readers now resolve to this
newest snapshot**: `scripts/playcaller_change_screen._latest_coordinator_dir()`
-> that dir, `load_counted_events()` -> 12 rows;
`interim_playcaller_first_game_back_overlay.load_counted_playcaller_change_events(Path("data"))`
-> 12 rows (both glob-sort by directory name, so an older dir without
`change_validation.json` would otherwise have been picked or the newer dir
would have failed the glob).

**Verified**: `ruff check scripts/ingest_coordinator_history.py` — all
checks passed. `mypy scripts/ingest_coordinator_history.py` — 9 pre-existing
errors (missing stubs, var-annotations, Hashable/Generator typing), none on
the 3 edited lines (confirmed via `git diff`); no new errors introduced.
`pytest -k coordinator -q` — 0 tests collected (exit 5); no test file in
`tests/` mentions "coordinator" (`grep -rl coordinator tests/` empty) — this
script has no durable test today, consistent with the test moratorium
(maintenance/ingest script, not a served prediction path).

## Next (current)
- Re-run `coordinators_tue` weekly as scheduled; the ceiling fix means any
  real mid-season 2026 OC/DC/interim-playcaller change will now show up as a
  `season == 2026` row in `coverage_audit.json["changes"]` and must get hand
  adjudication (event_date, lag_status, validation_status) before
  `screen`/`production-lead29`/the overlay will see it — check
  `data/raw/pfr_transactions` and `data/raw/injury_news` for corroboration as
  before.
