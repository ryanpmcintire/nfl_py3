# g55-u1b-live-inactives-2026-11-21

## Goal
Make the live gameday inactives capture (src/nfl_ats/inactives_capture.py; jobs scripts/capture_scheduler.py:463, :2604-2790) give the same construct as artifacts/g55_u1/deadline_actives.parquet: a per team-game active flag known before min(kickoff, Sun 16:00 ET). Parent: g55-u1-deadline-actives-2026-11-21.md. Not a look; reads no results. Patch only, src and scheduler untouched.

## State (start 2026-11-21 14:19 EST; stopped at the 50-call cap, end time not stamped)
- Diagnosis, measured from 137 archived snapshots, re-parsed in memory with no writes to data/. Scripts and outputs are in tests/scratch/g55_u1b/: inventory.py, diagnose.py (coverage.csv, reparsed_rows.parquet), agree.py (agree_before.log), verify.py (verify.log).
- Coverage of 260 eligible team-games: reproduced the U1 count of 122. Causes:
  - Weeks 1-2 (56 team-games): commit 6649499c skipped RotoWire whenever NFL.com showed its offseason placeholder (`need_fallback = not rows and empty_reason != ...`). NFL.com shows that placeholder all season. Fixed 2026-09-24 in 24849a81, now :423. No fallback html exists for those weeks, so they cannot be recovered.
  - Weeks 3-6, missing jobs: no 08:35 international pass (6 team-games) and no 15:20 pass (16:25 games 18, 16:05 games 5). Both jobs were added 2026-10-18.
  - Weeks 3-6, early slate: only the 11:35 pass existed, and RotoWire had not yet listed 43 team-games. The 12:20 pass was added in Week 7. Weeks 7-9 then show 0, 2 and 1 misses.
  - Week 10: inactives_sun_early_second ran at 13:14, after kickoff (scheduler_log.txt:7227), so 5 team-games were lost. The single-threaded daemon ran lineups_sun 12:00-12:24, then odds_tape_sun for 48 minutes. The tape now ends at 12:25 or skips (SUNDAY_TAPE_END_AT :158; capture_odds_tape.py:191-197).
  - Thursday: the single 18:50 pass caught 1 of 2 teams in Weeks 4, 6, 9 and 10, and 0 of 2 in Week 8.
- 159 captured-but-ACT rows: comparison artifact. The U1 player snapshot 20261108T141611Z was taken Sunday 09:16 ET, before Week 9 kicked off. With deduplication, 133 of the 136 ACT rows are Week 9. Against the post-game roster 20261121T181046Z, ACT falls to 3. Two are a stale PHI list captured Wednesday at T-90h, the previous game's list still on the page. One is Joe Fagnano, BAL Week 8 QB.
- Unmatched names: 73 of 761. Causes are suffixes (Jr, II, III, IV, V), short names (Mike for Michael, Cam, Rob, Nate) and nicknames (Kiko, Hollywood, Vega, CJ).
- Patch tests/scratch/g55_u1b/inactives_capture.patch (`git apply --check` passes; patched copy passes ruff check and format):
  - rejects rows captured before kickoff minus 90 minutes, the league posting time (reported, docs/inactives_channel.md:202-220);
  - adds attach_roster_ids, a cascade from exact name to suffix-stripped, initial plus last name, then a unique last name among ACT/INA/DEV on the latest week's roster at or before the capture week;
  - writes gsis_id and match_rule into the capture output and roster_snapshot and rows_matched_to_roster into the manifest.
- Verification, rows before the deadline and before kickoff, Weeks 3-10:
  - Before: 688 of 761 matched (90.4%); 685 of 688 are INA (99.56%); roster INA recall 685 of 768 (89.2%).
  - After, post-game roster: 758 of 758 matched; 757 INA (99.87%); recall 757 of 760 (99.6%). All 73 non-exact matches are INA.
  - After, forward roster 20261108: 756 of 758 (99.7%), recall 99.2%.
  - Full run_capture replay with a fake fetch, written to tests/scratch/g55_u1b/replay. Week 9 at 12:26: 93 of 93 rows matched. Week 4 PHI on Wednesday: 3 rows rejected (before_inactives_posting).
- Sunday 11-22 (Week 11): 7 games at 13:00, 1 at 16:05, 2 at 16:25; no international game. Jobs at 08:35, 11:35, 12:20, 14:40 and 15:20 are loaded (daemon started 13:03; file mtime 12:30). Late slate covered (Weeks 7-10 had 0 misses). The early slate depends on the 12:20 pass starting by 12:35 (15-minute grace):
  - It is queued behind the 12:00 batch: card_terms (about 0.5 min), card_fidelity (new), lineups_sun (24 min on 11-08 and 11-15), public_betting and injury_news.
  - Expected start is about 12:28 (inferred), about 7 minutes of margin. If lineups_sun overruns, the 13:00 teams get only the partial 11:35 pass.

## Tried
- One crash in agree.py: status column clash, fixed. The scheduler slot patch was not written because the cap was hit.

## Next
- 2026-11-24 08:40 EST: inactives_capture.patch applied and both slot changes below made directly in scripts/capture_scheduler.py (working tree, uncommitted; details in g55-u7-forward-logger-2026-11-21.md State). Remaining here: the deadline-actives rerun below.
- (done) Slot changes, originally planned as a patch:
  - (a) inactives_sun_early_second grace 15 -> 40, so the window closes at the 13:00 kickoff. run_capture already rejects rows captured after kickoff.
  - (b) A new job, inactives_thu_primetime_second, Thursday 19:35, grace 15, `_inactives_capture("thu_primetime")`, dedupe_dir data/players/inactives, dedupe 30, catch_up False, inserted before inactives_thu_primetime.
  - Then run `git apply --check`.
- Root: apply inactives_capture.patch, then rerun scripts/g55_u1_deadline_actives.py with the window [T-90, deadline) and the post-game roster.

## Open
- Week 12 has Wed 20:00, Thu 13:00/16:30/20:20 and Fri 15:00 games. There is no Friday inactives job; Thursday afternoon passes run at T-85 only.
- Fagnano (BAL Week 8, listed inactive, roster ACT) is probably the emergency third QB (inferred).
- Processes started: none persistent; all runs were foreground and exited. 0 left.
