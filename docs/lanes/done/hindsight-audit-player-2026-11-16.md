# Hindsight audit, player/availability/QB columns (2026-11-16)

## Goal
For the 24 player, availability and QB columns of the served full_weak_stack profile, state each training value's as-of time against the pick deadline min(kickoff, Sun 16:00 ET); rebuild and grade (v4_candidate stage, LOSO 2020-25, opener, pushes kept, QW = 836-667 reference) any column built from post-deadline information. Family hindsight_audit_player.

## State
2026-11-16T21:49Z declared: no column qualified for an as-of rebuild, so no arm was graded and no cell recorded (nothing measured to record).
Script scripts/hindsight_audit_player.py -> artifacts/hindsight_audit_player/timing.json (exit 0).
Timing per column (read players.py; measured timing.json):
- 9 diff_injury_* (7 unavailability + 2 value_lost): rows visible if effective_observed_at <= kickoff-24h (players.py:1824-1828, 1405-1424); kickoff-24h precedes the deadline. Basis measured: 2019-24 all date_modified (79% Friday), 2025 all week_proxy = max(kickoff-24h, Tue 00:00 ET) (players.py:216-250, 533), 2026 first_seen_capture. One row per player-week (final report). Role/value weights update only after each game (players.py:2013-2061). Learned rates use seasons < target (availability.py:320; source_end < target true). As-of; 2025 timing is imputed, not observed.
- 5 diff_qb_*: starter = previous game top passer before 2026 (depth charts only from 2026-03-22; players.py:1965-1973, 2062-2064); state = last QB state strictly before gameday (players.py:1394-1402); offseason shrink toward the completed prior season league mean (quarterbacks.py:529-532). As-of. History vs 2026 starter definition differs (depth chart), already covered by docs/lanes/wk10-qb-input-check-2026-11-14.md.
- 7 lineup continuity: overlap of the team's last two prior game snap lineups (players.py:1849-1868, appended after game 2015-2017). As-of.
- active_roster continuity/experience: weekly rosters with (season, week) strictly before the game (players.py:1308-1333, 1870-1877); years_exp constant within season in 21,692 of 21,694 player-seasons, +1 per consecutive season in 14,414 of 14,417. As-of.

## Tried
Read builder paths; one measured timing probe. No retrain run (no post-deadline column).

## Next
None for this unit. Possible follow-ups belong to other lanes: 2025 injury timestamps are a week proxy (no observed time on disk; no 2025 in-season capture exists).

## Open
- 2025 week_proxy asserts the final report was visible by kickoff-24h; unverifiable from disk (inferred true from the 2019-24 Friday timing).
- injury_timing_leakage_audit settled the kickoff-24h clock; not repeated.
