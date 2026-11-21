# g55-u1-deadline-actives-2026-11-21

## Goal
Servable "who is active / who starts at QB" per team-game, known before the pool deadline min(kickoff, Sun 16:00 ET), to replace the P2-L O1/O3 realized-snap oracles (parent: goal-55-backwards-2026-11-21.md). Not a look: reads no scores, margins or ATS results.

## State (2026-11-21 14:16 start; DONE 14:50, measured)
- Script scripts/g55_u1_deadline_actives.py (ruff check + format clean, no comments). Outputs artifacts/g55_u1/: deadline_actives.parquet (285,009 rows, 244,773 known_at_deadline; active and starter_qb are NA when not known), starter_qb_team_games.parquet (3,774 team-games), summary.json (all numbers below), manifest.json (sha256, as-of basis).
- Same-week weekly_rosters status IS the gameday list (2020-25 REG, 3,230 team-games): ACT 48.0 per team-game every season (gameday 48), INA 5.8-6.3. INA with snaps 0 of 19,847. ACT without snaps 732-852 a season (about 3%, dressed but unused). Players with snaps whose status is not ACT: 6 of 150,302; missing from the week roster 2; snap rows with no gsis 44. Caveat: nflverse publishes it after the game (manifest contract: earlier weeks only), so historically it stands in for the T-90 list.
- O1 v servable (candidates = same-week roster players with an earlier off/def role, P2-L rule): agreement 98.27% of 186,745; servable catches 94.07% of oracle-out (3,221 misses = active but 0 snaps); precision 99.99% (5).
- Slots 2020-25 games / deadline after T-90: Thu 110/110, Sat 47/47, Sun early 819/819, Sun late 401/401, SNF 0/107, MNF 0/118, other 8/13 (1,385 of 1,615). 2026: 19/19, 2/2, 154/154, 57/57, 0/17, 0/17, 6/6 (238 of 272). Deadline >= kickoff only Thu/Sat/Sun early/other (Sun late deadline is before kickoff, after T-90).
- QB 2020-25: depth-chart rank-1 among ACT QBs v first-dropback QB 3,058/3,219 (95.0%); combined starter 3,066/3,227; prior-game incumbent 2,957/3,175 (93.1%); served projected_qb_id 2,844/3,230 (88.0%); projected QB inactive 219. O3 nonzero 605 team-games; servable flag (projected inactive or not starter) 390, overlap 326, flagged carries 67% of O3 (1-share) mass; share<0.5: 315 of 444 flagged. 2026 (no depth chart): incumbent 248/267 (92.9%), projected 265/272.
- Live capture src/nfl_ats/inactives_capture.py:33-35 (NFL.com, RotoWire fallback, parser :231), jobs scripts/capture_scheduler.py:463, :2604-2656. 2026 Wk1-10: 122 of 260 eligible team-games have a pre-deadline capture (Wk3-10 only; Wk7-10 18-25 each). Versus roster INA (Wk3-9, 104 team-games): 719/796 names matched; matched rows INA 559, ACT 159, DEV 1; roster INA recalled 451/505 (89.3%).

## Tried
- Runs 1-2 crashed (status column clash in live_captures); fixed.

## Next
- G55-U3 harness swap (copy scripts/mod57_p2l_availability_ceiling.py to a new g55_u3 script; never overwrite its LEAK table at :50, main() reuses it at :273):
  1. O1: in role_rows (:70-151) replace `real_any` (:104-108) with `active` keyed (game_id, team, gsis_id) and the roster gate (:81-86, :110) with this table's team-game rows. sev1 (:117) = 1 when active is False, else 0 (NA = SNF/MNF, no info). Keep the prior-role rule (:112-114; column prior_role mirrors it) and the synthetic stamp (:129-131) so enrich (:154-174, decision 24 h :56/:169) keeps the rows.
  2. O3: in build_oracle (:203-226) replace realized `share` (:216-220) with 1.0 when known and projected_active and starter_qb_id == projected_qb_id, 0.0 when known and not, 1.0 when not known. Keep gap (:221).
  3. O2 (sev2 :116) stays out until G55-U2.
- Forward unit (G55-U1b): make the live capture the same construct: keep captures taken at or after T-90 and before the deadline, fix name matching (77 unmatched), explain the 159 captured-but-ACT rows, backfill nothing.

## Open
- Depth chart (nflverse weekly 2013-2025, none for 2026) timing is reported, not verified pregame; incumbent rule is the 2026 fallback.
- 90-minute posting: reported (RotoWire FAQ, docs/inactives_channel.md:202-220), not timed.
- Served 2026 projected QB agrees 97% with first snap v 88% historically; check whether the 2026 column was refreshed after kickoff (inferred risk).
