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
- No adjudication action needed until `ingest_coordinator_history.py`'s
  in-season range is extended past 2025 (needs the 2026 regular season
  complete) or a real in-season 2026 change gets reported — re-run the same
  two capture/audit commands weekly (already scheduled Tuesday 07:30 as
  `coordinators_tue`) and re-diff against this snapshot's `change_validation.json`;
  if `coverage_audit.json["changes"]` grows past 16 or any row's season is
  2026, that new row needs hand adjudication (public source, `event_date`,
  `lag_status`, `validation_status` in {`staff_change`,`playcaller_role`,noise
  statuses}) before `screen`/`production-lead29`/the overlay will see it.
- PER-07 itself needs no further action here; its coach-identity/aggressiveness
  findings are already closed/pooled per `ROADMAP.md` line 547 and untouched by
  this lane.

## Open
- None. This is a maintenance/adjudication unit, not a research finding; no
  card, grade, or page changed.
