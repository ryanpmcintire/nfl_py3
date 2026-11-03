# SIM-12 In-game starter exits

## Goal
Measure in-game starter exit hazard and its drive-level cost for the simulator (ROADMAP SIM-12). No engine edit.

## State
Family sim12_starter_exit declared before outcomes. Looks: 4 positions (QB first-snap QB, OL five first-snap starters, WR1 and CB1 by prior-games snap share) x 2 metrics (EPA/play, points/drive) x 4 cells (all, exit-wp strata <0.2, 0.2-0.8, >0.8) = 32 looks, 8 primary (all-strata). No fitted parameters; strata edges are descriptive.
Design: exit = last snap in Q1-Q3 and team has Q4 snaps; effect = (post-exit minus pre-exit drives, same game, opponent fixed) minus mean of same difference in no-exit team-games split at quarter midpoint, matched on position, quarter, wp stratum; game-cluster bootstrap 300; margin impact = hazard x slots x effect ppd x mean post drives (CB sign flipped).
Scripts (scratchpad): C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/7bc8aaac-35b2-44a8-ae86-7cba0db4ab6e/scratchpad/s12.py (build, writes ev/ct/hz.parquet there) and an12.py (cells.csv + hazard table). First build run crashed at save (path bug, fixed); rerun launched in background (background id b7bv4g6u5), about 4 minutes.
Data: participation data/players/participation/raw/20260813T131635Z, pbp data/pbp/raw/20260925T202544Z, positions from latest data/players/raw/*/weekly_rosters.parquet.
Engine hook locations (read): team strength enters via xq (sim10_drive.py:1468) into q[5] (1365) and the qm/strength_adjust path (1229-1252); main stream rng.random() at 1468/1499. Hook sketch: when flag off, no draw; when on, a separate Generator seeded from the game seed draws per-team exit time, and from that drive on q[5] shifts by the measured QB effect. Only write up if the QB effect is material.

## Results (measured, 2026-11-03, 685 exit events, 22260 control splits)
Hazard (exit before Q4, Wilson 95): QB 5.98% (4.93,7.24; n 1638), OL per starter 4.78% (4.32,5.29; n 7505), WR1 4.60% (4.05,5.22; n 4958).
EPA/play DiD (95% boot, P>0): QB +0.012 (-0.067,0.095; 0.63); OL -0.069 (-0.115,-0.026; 0.00); WR1 +0.013 (-0.024,0.066; 0.67). Points/drive: QB -0.154, OL -0.300, WR1 +0.058.
Margin impact per game: QB -0.048 (-0.19,+0.08), OL -0.44 over five slots (-0.80,-0.12), WR1 +0.015. QB not material: no hook written. OL only fully negative interval.
Gaps: CB1 0 events (defense id/position lookup failed, unmeasured); QB rows 1638 of ~5400 team-games (coverage unexplained). 12 cells recorded, family sim12_starter_exit.

## Tried
Row/prior-art check done; nothing else measured yet.

## Next
Debug CB1 and QB coverage in s12.py; rerun; consider OL hook spec.

## Open
Exits mix injury, ejection and blowout rest; wp stratum separates them only roughly.

## Unit 2 (measured 2026-11-03)
Scripts: scripts/sim12_starter_exits.py (build, --out, --legacy), scripts/sim12_analysis.py (cells, per-season LOSO, placebo); ruff clean; data in tests/scratch/sim12. Legacy mode reproduces unit 1 exactly (685 events, OL -0.069, QB +0.012).
Defects (file:line): starters were read from a team-game's first play, a kickoff (starter rows 162 of 512 in 2020), so QB/OL slots dropped (sim12_starter_exits.py first-play slot logic, now first pass/run play with non-scrimmage plays masked). CB1: rosters label most corners "DB", so CB lookup was empty; now {CB, DB}, renamed DB1 (includes safeties).
Fixed cells, EPA/play DiD (150 boots; look count now 4 groups x 4 cells = 16 more looks, same family): QB -0.016 (-0.063, 0.024) P>0 0.25, 297 events of 5275, hazard 5.63%; OL -0.038 (-0.057, -0.017) P>0 0.00, 1251 events, margin -0.30/game (-0.49, -0.13); WR1 +0.017 (-0.033, 0.067); DB1 -0.033 (-0.068, 0.011), 227 events. The unit 1 OL -0.069 was inflated by incomplete coverage (half-size).
OL placebo (pseudo exits at 0.25/0.75 quarter, disjoint halves, 150 reps): mean +0.0025, 95% (-0.018, +0.018); observed -0.038 lies outside. LOSO per fold dEPA 2016..2025: -0.078 -0.027 +0.012 -0.043 -0.007 -0.026 -0.084 -0.032 +0.017 -0.119; pooled -0.038; 8 of 10 negative. Non-injury exits unidentifiable (no reason field); wp>0.8 stratum is -0.089 (blowout rest not benign).
Recorded: 4 fixed all-cells in registry/weak_signals/sim12_starter_exit.json (unresolved_below_power).
## OL hook spec (not applied)
Where: game() at sim10_drive.py:1466 builds xq, 1468/1499 draw main rng; run_half q[5]=xq[pos] at 1363 feeds 1229-1252. Hazard 4.96% per starter (Wilson 4.70, 5.24) before Q4, 5 slots; draw per team from a separate Generator spawned from the game seed (no main rng draw; zero draws when flag off), exit time uniform over drives before Q4. From the next drive, offense q[5] shifts by the LOSO EPA/play effect (-0.038) converted through the existing EPA-to-strength scale; effect refit per fold. Open: DB1 sign and conversion unit need a mechanism read before wiring; defense side not specified.

## Next
Wire OL hook behind a flag after owner review; split exit reasons if an injury feed is joined.
