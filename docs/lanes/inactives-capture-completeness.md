# Inactives capture completeness

## Goal
Decide whether the 2026 inactives snapshots are complete enough to feed the served card, and what fails closed if not.

## State
Writer: scripts/capture_inactives.py -> src/nfl_ats/inactives_capture.py (main; manifest teams_seen at :526). Scheduler jobs: scripts/capture_scheduler.py:1455-1586 (inactives_sun_early/sun_late/thu_*/wed_primetime/sat_*), helper :373. Snapshots: data/players/inactives/<UTC id>/{inactives.parquet,manifest.json,primary.html,fallback.html}.

Measured (2026 snapshots with rows, best snapshot per team-week, via scratchpad a.py):
- Per-team-week coverage is the problem, not per-team list length. Week 3: 10 teams; Week 4: 11; Week 5: 5; Week 6: 7 (of ~28-32 playing). Median names per captured team 6-7, min 3 (Week 4 PHI, wed_primetime snap 20260930T225017Z), GB 4 in Weeks 4 and 5 (sun_early).
- Typical count: nflverse weekly_rosters 2025 REG status INA, 542 team-weeks, mean 6.3, median 6, 5th pct 5, min 3, max 8. So PHI Wk4 (3) and GB Wk4/5 (4) are below the 5th percentile; Week 5 min 4.
- source_used is "fallback" (rotowire) on all inspected snapshots; primary nfl.com showed the known placeholder (manifest 20261018T184017Z showed_known_placeholder true).
- Week 6 sun_late snapshot 20261018T184017Z: 7 rows, teams_seen ["ARI"] only, game 2026_06_ARI_LA; LA (home) absent although the game is listed.

Served path (read): publishing.py:1525 calls record_inactives_refresh_overlay on every refresh-picks; publishing.py:228 maps inactives_refresh_v1. Freshness gate src/nfl_ats/source_freshness_policy.py:520-544 fails only on row_count == 0 past T-90; row_count>0 from any one team gives COMPLETE (newest_snapshot_manifest_row_count at :653). Per-game acceptance inactives_refresh_overlay.py:214-264 requires only non-empty rows for the game_id, and reported_inactives (:113) only row_count>0, empty_reason None, recognized source. No team-count check anywhere. Inferred: a one-team or truncated list passes silently, giving one-sided or understated unavailability increments. Not verified: whether inactives_refresh_v1 currently changes a served side (check four_overlay_composition member state and off switches before calling it release-blocking).

## Tried
Local nflverse inactives file: none present; used weekly_rosters INA from GitHub release (downloaded to scratchpad only).

## Next
Done 2026-10-18 (uncommitted): inactives_refresh_overlay.py adds inactives_team_floor (5th pct of nflverse weekly_rosters INA per team-week, seasons before the plan season, local data/players/raw/<id>/weekly_rosters.parquet; 5 for 2026), inactives_game_complete, inactives_partial_detail; overlay tags incomplete games SOURCE_PARTIAL_LIST and records no increment. source_freshness_policy.py: SourceObservation.partial_detail, inactives DEGRADED with reason "partial inactives", summary_line adds "Partial lists". Freshness on the live data: COMPLETE -> DEGRADED, "Partial lists: inactives". Tests: test_source_freshness_policy + test_cli 92 passed.
Limit: freshness only sees games present in the newest snapshot; a scheduled game absent entirely is not counted. If floor cannot be derived (no rosters file) only presence of both teams is enforced.
Capture-side unit (not started, no capture changes yet): why rotowire yields ~10 teams per week. Measure (a) which teams the rotowire page lists at each capture time vs the schedule (page scope: only games already posted? only one slate/day?), (b) timing: sun_early vs sun_late windows and T-90 posting, (c) parser drops (primary.html/fallback.html in data/players/inactives/<id>/ vs inactives.parquet teams). Then add a manifest partial flag (teams_seen vs scheduled teams) in inactives_capture.py:526 and fix the cause. Why PHI Wk4 Wed capture had 3 names belongs here too.

## Open
Week 5 only 2 snapshots with rows; whether missing teams are absent from the page or dropped by the parser is unmeasured.

## Update 2026-10-18 (scoping)
Freshness completeness is now scoped to the current-week plan: source_freshness_policy.py _current_week_kickoffs (game_id->kickoff of the current slate), report_for_publication passes games whose T-90 has passed as inactives_due_game_ids to observe_from_disk; inactives_partial_detail(data_root, due_game_ids) uses the latest reported snapshot containing each due game; not-due games and other-week snapshots are ignored. Measured: Tue 2026-10-20 16:00Z -> inactives complete; Sun 2026-10-18 19:00Z -> degraded "3 of 5 games already due" partial. tests 92 passed. Limit: a due game absent from every snapshot is still not counted.

## Capture-side finding 2026-10-18 (measured, scratchpad m.py/n.py replaying saved fallback.html through _parse_rotowire_grid and the schedule)
Weeks 3-6, 122 team-weeks: 33 captured; 40 no capture inside [T-90, kickoff); 49 capture was inside the window but the rotowire page did not yet list the team; 0 dropped by the parser (kept teams == page teams with kickoff after capture, in every snapshot; rows at/after kickoff are rejected by design, inactives_capture.py:~470). Raw fallback.html and primary.html are saved per snapshot.
No-capture windows (all by schedule): Sun 16:25 slate (14:40 ET job is T-105), Sun 09:30 international, SNF 20:20, MNF 20:15 (no mon/sun-night jobs; Wed/Thu/Sat jobs exist). Page-lacked: rotowire posts per game as nfl.com does, staggered; 11:35 ET sun_early saw 7-9 teams of 24-28 pending, 14:40 sun_late saw 0-2 of 10-12 pending.
Smallest fix (scheduler.py is orchestrator-owned): add sun jobs at ~09:05 ET (intl), ~14:55 and ~16:00 ET (late-slate, before 16:00 lock caveat), 19:00 ET (SNF), and mon 18:50 ET (MNF); and add repeat captures at T-60 and T-30 for the 13:00 slate (sun 12:00, 12:30 ET). Same-slot dedupe_minutes=60 must be lowered to ~20 for the repeats. No parser fix needed; no NFL.com retrieval (MKT-09), rotowire only.

## Scheduler jobs added 2026-10-18 (uncommitted)
capture_scheduler.py: inactives_sun_intl sun 08:35; inactives_sun_early_second sun 12:20 (dedupe 30); inactives_sun_late_second sun 15:20 (dedupe 30); inactives_snf sun 19:15; inactives_mnf mon 19:05. Slot labels reuse sun_early/sun_late (SLOTS in src is fixed). Dry runs OK (each wrote a real snapshot dir 20261018T2353xxZ). tests/test_capture_scheduler.py: snf/mnf/second-late exempted in test_no_inactives_row_targets_snf_or_mnf; 57 passed. Next: after next Sunday, re-measure coverage per window.

## Update 2026-10-18 (SNF/MNF correction)
- Removed inactives_snf and inactives_mnf from scripts/capture_scheduler.py (owner design: SNF/MNF excluded, T-90 after Sunday 16:00 ET lock). Test restored to committed form (git diff empty). Kept inactives_sun_intl, _sun_early_second, _sun_late_second.
- source_freshness_policy._inactives_due_game_ids: due = T-90 passed AND T-90 < pick_deadline(kickoff, sunday_pick_lock). Verified: Sun 21:00 ET Wk6 excludes SNF (DAL_GB) and MNF (WAS_SF), inactives degraded (partial, honest); Tue Wk7 inactives complete.
- Pytest capture_scheduler + source_freshness_policy + cli: 149 passed.
- Dry-run snapshots data/players/inactives/20261018T23531{2,4,6,8}Z and 20261018T235320Z came from `--dry` ~19:53 ET; ordinary timestamped captures, no action.
