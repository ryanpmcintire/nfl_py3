# PER-03 free-source route

## Goal
Decide whether PER-03 (injury report history) can proceed without the owner.

## State
Measured from data/raw/nflverse_injuries/20261101T140009Z/injuries.parquet (93,211 rows):
- 2009-2024: one row per player-week, practice_status 100%, report_status 94-97% (2009-15) then 45-60% (2016-24; game status only for listed players), date_modified 98.6-100% from 2010 (2009: 0.4%). Timestamps are hour-granular UTC final modification per player-week, not per-day practice or revision history.
- 2025: date_modified 0% (upstream dropped it); 2026: 0%. practice_status 99%.
- Revision stream exists forward only: 136 nflverse snapshots since 2026-08-26 (daily, plus Thu/Sun extras); 2026 rows grow 0 -> 2,459 across snapshots, so capture time is a valid availability bound for 2026 designation changes within the week.
- Missing for 2025 and earlier: per-day practice participation (Wed/Thu/Fri), within-week designation revisions, report publication time.
- nflverse: policy green, CC-BY 4.0 with attribution (config/source_policies.json nflverse; LICENSE at nflverse-data). Free, no credential.
- NFL.com / team sites / ESPN: policy RED (express written consent required); not fetched.
- PFR/PFT transaction wire: yellow, live, private-only; gives dated injury-news events, not practice reports.

## Tried
No new fetch: the nflverse capture already runs daily and is the free sample; a new copy adds nothing.

## Next
Capture plan: keep the existing nflverse_injuries job (nflreadpy.load_injuries per season). Build 2026 revision history by diffing consecutive snapshots keyed (season, week, gsis_id); availability = snapshot fetched_utc. For 2009-2024 use date_modified as final-observation time with the 24-hour cutoff already ingested. Per-day practice and pre-2026 revisions have no free source.

## Open
Owner action narrowed to: SPORTRADAR_API_KEY (or written NFL.com consent) only for per-day practice history and pre-2026 revision replay. Forward 2026 revision work proceeds now without it.

## Unit 2 result (2026-11-03): 2026 revision history built
- scripts/per03_injury_revisions.py writes data/processed/injury_revisions_2026.parquet (+ .manifest.json with per-snapshot sha256): 4,467 rows, 136 snapshots, weeks 1-8. Columns: season, week, gsis_id, team, observed_at (snapshot fetch UTC), previous_snapshot_at, event (first_appearance/change/removed), changed_fields, current and prev_ values for report_status, practice_status, and the four injury fields. baseline_snapshot rows: 0.
- Validated: every observed_at is a snapshot fetch time and equals the next fetch after previous_snapshot_at (asserted in script); all 2026 games matched to kickoff.
- Final Out/Doubtful/Questionable first observed (non-Thursday games, share before Fri 16:00 ET / before Sat 00:00 ET / before kickoff): wk1 0/0/1.0 (n58), wk2 0/0/1.0 (99), wk3 .036/.036/1.0 (111), wk4 .015/.015/1.0 (131), wk5 0/0/1.0, wk6 0/0/1.0, wk7 .014/.014/1.0, wk8 0/0/1.0 (120).
- Cause is capture cadence, not publication: week-8 observation times (ET) cluster Thu 06:00, Fri 06:00, Sat 07:10 (201 rows), Sun 06:00, so the Friday report (afternoon) is first seen Saturday morning. Observed_at is an upper bound on true publication time; Sunday-morning inactive status is seen before kickoff only because of the Sun 06:00 snapshot.

## Feature-builder contract
As-of join: for prediction timestamp T take, per (season, week, gsis_id), the latest revision row with observed_at < T (strict), then fill forward its current values; absent row means no report. Enforce at runtime: assert max(observed_at used) < T and refuse any row with observed_at >= T; never use date_modified for 2026; never use final-snapshot values. Openers are graded before the week's reports, so for opener-grade features only prior-week state is admissible.

## PER-03 still needs
- A Friday ~20:00 ET snapshot (scheduler change, orchestrator) to separate Friday-report timing from Saturday capture; today pre-Saturday is unmeasurable.
- Per-day practice (Wed/Thu/Fri) history and pre-2026 revision replay: owner action SPORTRADAR_API_KEY or written NFL.com consent.
