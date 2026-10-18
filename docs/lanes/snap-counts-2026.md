# snap-counts-2026

## Goal
Find whether 2026 snap counts reach the served card and the forward logs; make the forward logs use 2026 starters pregame-safely.

## State
- Served card NOT affected (read): snap counts feed training-side availability outcomes only (src/nfl_ats/cli_commands/features.py:441-464, availability.py:189-202, season-lagged rates); 2026 games have no outcomes there. weak_stack_v3_features.py reads schedules and pbp only, no snaps. roster_availability_flag_features.py:28 pins the 20260817T184901Z file; transaction_flag_features.py:90 default_snap_counts reads latest snapshot (ends 2025 wk22) for flag features, challengers only.
- Stale-snap consumers (challenger only, not the card): prospective.py:480-505 nflcom_starter_key_sets keys (season, week+1, team); with a snapshot ending 2025 wk22 no 2026 player is a starter, so the nflcom Out-2-starters challenger and nflcom_refresh_overlay.py:86 count 0 starters out for 2026 (never flips). Needs scheduler argv fix.
- Source (measured): nflreadpy 2026 snap_counts available via load_season_frame (nflverse_current_season.py:33,68-86; current season 2026). 7463 rows, weeks 1-5 near complete (1400-1502/wk), week 6 only 93 rows (DEN, SEA). Columns have no publication timestamp; week is the only proxy (rows exist only after a game). Lags a week or more behind the games.
- Forward logs fixed: mod24_forward_log.current_season_snaps(week) merges latest snapshot (seasons != 2026) with 2026 REG rows of week < target week (asserted) into data/players/forward_snaps/; draftcap_forward_log.py record() and armb_forward_log.py main path set U["SNAP_FILE"] / U6["SNAP_GLOB"] to it. Week 7 reruns: draftcap prior orders now 202604-202606 (was 2025 finals), 1 of 14 arm side differs from base; armb 0 of 14 differ (model and card). Rows: 14 each.

## Tried
- Week 7 reruns written: artifacts/draftcap_forward/runs/2026-week-07-20261018T225309Z.json, artifacts/armb_forward/runs/2026-week-07-20261018T225352Z.json (earlier stale-snap rows remain; canonical row = latest before deadline).

## Next
- Primary: change scheduler argv capture_scheduler.py:257 and :296 --snap-end-season 2025 to 2026. players.py:702 loads each season via load_season_frame (404 -> SeasonReleaseNotPublished handled for guarded seasons), so a partial in-season file is loaded as-is; verify canonicalize_snaps dedupe on one real player-ingest run and that week 6 partial rows do not break nflcom starter keys (week+1 of a partial week just leaves some teams unflagged).
- Once fixed, forward logs could read the latest snapshot directly; helper then redundant.

## Open
- Week 6 snap feed lags (93 rows); some teams use a week-4/5 prior game. Pregame-safe, stale by a week.
