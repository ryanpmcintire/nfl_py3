# fix-picks-u6 input freshness audit (2026-11-10, read-only)

## Goal
Test whether Week 8-10 forecasts were built on stale/truncated/duplicated/mis-joined inputs after the 2026-10-31 crash.

## State
Verdict: no input defect found for Weeks 8-9. Scratch: tests/scratch/fixpicks/u6_{b..j}.py, u6_A.pkl (first-kickoff forecast rows W1-10).
- Forecasts: artifacts/margin_predictions/2026-week-NN-*; W8 Thu 20261029T211056Z / Sun 20261101T170933Z; W9 20261105T221000Z / 20261108T120945Z; W10 20261111T011451Z. Team tables built 10-27T22:04Z (W8), 11-03T17:20Z (W9), 11-11T01:05Z (W10), each after the prior Monday game.
- Recency: home/away_team_games match schedule recompute, 0 mismatches W3-10; home_point_diff tracks mean-through-w-1 (r .91-.93 W7-10, same as W4-5). Games complete: 14/14/14/15/14 per week, no dup, no missing.
- Missingness: no |NaN shift|>5pt W8-10 vs W1-7 except week-1-design cols (returning snap share, prior_week_ats).
- Shift: W8-10 vs W1-7 only calendar cols (week_sin/cos, team_games) exceed |z|3; control W5-7 vs W1-4 had 23. W10 alone: pbp matchup/def and injury unavailability low (z -3..-5), inferred = Sunday injury snapshot (20261108T141611Z) is the newest for W10, pre-report.
- Forecast vs rebuild today: team-level features equal to <=4e-4; differing cells are injury/odds, qb_start_probability (W9 up to 17 sd, updated after) and surface_switch_flag.
- Scheduler 10-27..11-10: crash gap 10-31 07:17-07:29 MISSED lineups/refresh/injury_news (all later recovered: injury_news/player_snapshot OK 10-30..11-01; W8 Sunday forecast used injuries 11-01T11:00Z, qb 17:00Z). player_snapshot_thu_0915 11-05 FAIL download, recovered by thu_pm 16:46 and 22:00Z qb snapshot before W9 forecast. lineups_sun_am 11-08 manual FAIL (subprocess), W9 Sunday forecast still used injuries 11-08T11:00Z. FAIL independent_validation, draftcap_forward_log, refresh_sun 11-01 (pick_probability load) do not feed the base margin features.
- sync_captures conflicts=294: count frozen at 294 since 2026-10-27, grew 5->267 from 09-28 (scripts/sync_captures.py:420-440): a remote (second capture host) snapshot twin differs in content from local; remote kept, nothing pulled, local files never overwritten, so no stale/wrong local file. FAIL runs are ssh/hash staging failures of the feature-table push (remote copy), not local inputs.
- player_values input is the 2026-08-17 snapshot in every week (lineage.json), not crash related.

## Tried
No re-prediction: no corrected feature found.

## Next
Not done: raw-pbp recomputation of EPA features (only checked forecast==rebuild); W8-9 pick losses not input-driven by this evidence.

## Open
Is a Tuesday W10 forecast on Sunday injuries intended? Re-run after Wed player_snapshot.
