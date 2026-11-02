# per07-staff-state-2026-11-02

## Goal
PER-07 data items: dated staff-state table (HC/OC/DC spells, vacancy/interim, playcaller separation, first public observation with revision id/URL) from cached Wikipedia staff revisions. No ATS look.

## State
Built scripts/build_staff_state.py (no network, reads cached responses in data/raw/coordinators/20261031T112746206891Z, adjudications from 20260907T213814366437Z/change_validation.json). Output artifacts/staff_state/20261102T221745192535Z/{staff_state.parquet, manifest.json}: 1,740 spells, 32 teams, 2009-2026 (2009-2021 preseason baseline only; 2022-2026 add every in-season revision). Vacant spells 2022-25 = 4+6+3+2 (2026: 0), interim spells 2022-26 = 6/7/14/7/1. Playcaller documented for 5 spells only (CAR22, IND22, LV25 title holder; PIT23, NYG25 differs); all else unknown. Cross-check: 12/12 known 2022-2025 events matched (spell starts at the revision instant and ends the prior spell). Ingest parity: 3,161 of 3,162 parquet rows reproduced.

## Tried
Searched all 1,114 cached revisions for play-call text on coordinator lines: 0 hits, so playcaller facts come only from dated, cited change_validation notes.

## Next
Consumers (playcaller_change_screen, overlay) not yet pointed at the table. Spells are keyed by state change, so a same-person interim-tag flip is its own spell (24 of 42 in-season OC/DC spells 2022-25 are state-only; 9 are person changes; 22 touch vacant/unlinked); 26 unadjudicated spell starts need review before counting as events.

## Open
Playcaller for interim OC/DC and HC-calls-plays cases needs dated team/news text (none on disk). 1 ingest row not reproduced by the scanner (not diagnosed). Effective start is the revision instant, not an employment date, except where an independent event_date is carried.
