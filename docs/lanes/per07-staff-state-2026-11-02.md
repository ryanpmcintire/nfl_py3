# per07-staff-state-2026-11-02

## Goal
PER-07 data items: dated staff-state table (HC/OC/DC spells, vacancy/interim, playcaller separation, first public observation with revision id/URL) from cached Wikipedia staff revisions. No ATS look.

## State
Built scripts/build_staff_state.py (no network, reads cached responses in data/raw/coordinators/20261031T112746206891Z, adjudications from 20260907T213814366437Z/change_validation.json). Output artifacts/staff_state/20261102T221745192535Z/{staff_state.parquet, manifest.json}: 1,740 spells, 32 teams, 2009-2026 (2009-2021 preseason baseline only; 2022-2026 add every in-season revision). Vacant spells 2022-25 = 4+6+3+2 (2026: 0), interim spells 2022-26 = 6/7/14/7/1. Playcaller documented for 5 spells only (CAR22, IND22, LV25 title holder; PIT23, NYG25 differs); all else unknown. Cross-check: 12/12 known 2022-2025 events matched (spell starts at the revision instant and ends the prior spell). Ingest parity: 3,161 of 3,162 parquet rows reproduced.

## Tried
Searched all 1,114 cached revisions for play-call text on coordinator lines: 0 hits, so playcaller facts come only from dated, cited change_validation notes.

## Update (adjudication unit)
artifacts/staff_state/adjudication_20261102/adjudication.json: 26 unadjudicated spell starts (25 state-only flips, 1 not-a-change PHI Moore Feb 2025), 7 known events given playcaller + dated cited quote (BUF Brady, LV23 Hardegree, CHI Brown, CHI Beatty [playcaller stays Brown], LV24 Turner, NYG Bullen, NYJ Harris), NYJ24 Manuel left unknown, 2026 playcallers for 32 teams from one ABC list (2026-09-03, single source). build_staff_state.py takes --adjudication; run result: 42/42 in-season OC/DC spells adjudicated, 12/12 known matched. Usable first-game-under-new-playcaller events 2022-25: 11 of 12 (8 offence, 3 defence); Beatty is a title change only.

## Next
Consumers (playcaller_change_screen, overlay) not yet pointed at the table. Spells are keyed by state change, so a same-person interim-tag flip is its own spell (24 of 42 in-season OC/DC spells 2022-25 are state-only; 9 are person changes; 22 touch vacant/unlinked); 26 unadjudicated spell starts need review before counting as events.

## Open
Playcaller for interim OC/DC and HC-calls-plays cases needs dated team/news text (none on disk). 1 ingest row not reproduced by the scanner (not diagnosed). Effective start is the revision instant, not an employment date, except where an independent event_date is carried.

## Update (staff-state fitted-term rerun)
scripts/per07_first_game_fitted_term.py now takes --staff-state <parquet> (events = earliest playcaller-known spell per season/team/playcaller_name; first game whose Tuesday-noon-ET prediction time is after playcaller_usable_from, asserted at runtime; sets all and offence_only, looks=2; writes artifacts/per07/first_game_fitted_term_staff_state.json with difference_from_12_event_run). Launched in background (bash task bt1nxk2h5, 20,000 bootstrap draws x2 sets, slow). Not yet finished when written. Next: if the json exists, read it, then record cells under family per07_first_game_new_playcaller_fitted_term with weak-signals record (batch file, unresolved_below_power, no closing ground unless resolved wrong sign); if absent, rerun once in the background with the 4 thread env vars set to 1. Prior cells live in registry/weak_signals/per07_first_game_new_playcaller_fitted_term.json (2020-25, 1503 games).

## Result (staff-state rerun, done)
Run finished; artifacts/per07/first_game_fitted_term_staff_state.json. 6 cells PER07-ss-* recorded under per07_first_game_new_playcaller_fitted_term (unresolved_below_power). Next: none for this unit; consumers still not pointed at the table.

## 2026-11-03 coordinators_tue snapshot check (read-only)
Snapshot 20261103T123300Z vs 20261031T112746Z: coordinator_history.parquet identical (3,162 rows each, outer merge diff 0; newest 2026 revision 2026-10-23 SEA, unchanged). Within-2026 OC/DC person values: no team shows more than one person (no personnel change; only same-person revision re-captures). No firing, resignation or playcaller handoff, so nothing takes effect Week 9. Overlay (interim_playcaller_first_game_back_overlay) reads events from the latest */change_validation.json (20260924T194157799750Z: 16 changes, 0 in season 2026), counts only staff_change/playcaller_role; zero 2026 events, so it would not fire for Week 9. Caveat: staff_state table not yet wired to the overlay, and playcaller handoffs inside an unchanged OC are invisible to the snapshot (single 2026-09-03 source).

## 2026-11-03 change_validation wiring (LEAD-29 data path)
Overlay contract (read): load_counted_playcaller_change_events takes the lexically last data/raw/coordinators/*/change_validation.json, reads payload["changes"], counts only validation_status staff_change or playcaller_role, uses season, team, revision_at (naive UTC) as the event instant; no other field.
Writer: scripts/per07_write_change_validation.py (ruff clean). Carries the latest file's changes verbatim, drops any revision_at not before the write time, appends in-season OC/DC person changes from the newest staff_state parquet (status from the adjudication, else unadjudicated_staff_state_person_change, which the overlay ignores). Stamped dir = write time; never backdated.
Run 2026-11-03T16:06:24Z: data/raw/coordinators/20261103T160624919493Z/change_validation.json, 16 carried, 0 added; overlay parser on it: 12 counted events, 0 in 2026.
Job entry for the root to add to scripts/capture_scheduler.py (between coordinators_tue 07:30 and weekly_lock 12:20):
    Job("change_validation_tue", "tue", "07:45", 30, [str(UV), "run", "--no-sync", "python", str(REPO / "scripts" / "per07_write_change_validation.py")], True, "Writes a stamped change_validation.json from the staff-state table after the Tuesday coordinator ingest so the first-game-back overlay sees an in-season change before the noon lock.", added_on="2026-11-03", requires=("coordinators_tue",)),
Caveat: staff_state parquet is rebuilt only by scripts/build_staff_state.py; the job should be preceded by a rebuild (add that to the same argv chain or a prior Job) or new in-season changes are not seen.
