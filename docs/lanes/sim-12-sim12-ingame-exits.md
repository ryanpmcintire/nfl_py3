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

## Unit 3 OL exit hook (2026-11-03, agent)
Hook wired in scripts/sim10_drive.py behind `sim --olexit 1` (env SIM10_OLEXIT): ol_shift() method, game() draw block (separate Generator eng.orng, SeedSequence(wseed.entropy, spawn_key + stream_key)), q[5] shift in run_half, init and sim_world. Constants read from artifacts/sim12/ol_exit.json (source: tests/scratch/sim12/cells.csv row OL,all). Shift = epa_per_play_shift / epa_scale per exit already happened by drive start. Flag off: no draw. Smoke script in scratchpad smoke.sh (labels smk_head/smk_off/smk_on, seed 5, seasons 3); temp file scripts/_sim10_head.py (HEAD copy) must be deleted after smoke.
Next: Tier B labels ol_exit seeds 11,12,13, then tierb_cmp.py ol_exit 11,12,13.
IN FLIGHT 2026-11-03 ~16:10: smoke task bv5jkvy6x (smoke.sh in session scratchpad). Done marker artifacts/smk.done holds the three sha256 (smk_head, smk_off must match; smk_on must differ). Then delete scripts/_sim10_head.py, run Tier B (--olexit 1 --label ol_exit, seeds 11,12,13, workers 11, OMP 1, one at a time), adjudicate: .venv/Scripts/python.exe tests/scratch/tierb_cmp.py ol_exit 11,12,13. Append one result line to engine-patch-queue.
Smoke DONE (measured): smk_head = smk_off sha256 05e23e0c...fa592 (identical); smk_on bda55a03...4f22 (differs). _sim10_head.py deleted.
Tier B RUNNING ~16:20: background task bh2rbwvg5 (tb.sh in scratchpad: seeds 11,12,13 sequential, writes artifacts/olexit_cmp.txt at end). Adjudicate from that file / rerun tierb_cmp.py ol_exit 11,12,13; then append result to engine-patch-queue.
Tier B DONE 16:22: NOT ADOPTED (13->10, 14->11, 13->12; pts_game -0.50 away from real). Result line in engine-patch-queue. Hook stays, default off. Next: none queued; pts_game drop suggests the shift stacks on a sim already matching scoring (inferred).
- 2026-11-03 16:22 ET ROOT REVIEW: (1) the 2700 s exit window in artifacts/sim12/ol_exit.json is an underived constant (no-magic-numbers defect); next unit must measure the in-game exit-time distribution of OL starters from pbp (snap counts or last-snap time) and draw exit times from it. (2) Mechanism design defect (inferred): the engine's drive-outcome model was fitted on real games that already contain OL exits, so adding exits on top double-counts their mean effect (pts_game 44.86 -> 44.36 v real 44.83). The hook must redistribute: games with no exit get the measured no-exit EPA level and games with exits the exit level, with the league mean held at the fitted value by construction from the measured hazard (no tuned offset). Then re-run Tier B. Hook stays default off.
- 2026-11-03 Unit 4 redesign (agent): (1) exit times now drawn by inverse-CDF from the measured empirical distribution of OL starter last-snap elapsed game seconds (1251 exits of 25220 starter-games, 2016-2025, nflverse participation + pbp; quantiles 5/25/50/75/95% = 218/970/1716/2267/2639 s, mean 1582, max 2698; stored as exit_elapsed_seconds in artifacts/sim12/ol_exit.json; 2700 window removed). (2) Redistribution: shift at drive start with elapsed el = d * (k(el) - slots*h*F(el)), d = -0.038 EPA/play over scale, k = exits so far in the draw, F = measured CDF; E[k(el)] = slots*h*F(el) exactly, so the mean shift over the exit process is 0 at every clock time (OT/Q4: F=1). No tuned constant. Chain script tests/scratch/olx2_chain.sh (bg task bdwsjmiqr): smk2_off/smk2_on seed 5 seasons 3 (hashes artifacts/smk2.done; off must equal 05e23e0c...fa592), then Tier B ol_exit2 seeds 11,12,13, then artifacts/olexit2_cmp.txt; DONE marker artifacts/olx2.done.
- IN FLIGHT 16:24: bg task bdwsjmiqr running olx2_chain.sh; on artifacts/olx2.done read artifacts/smk2.done + artifacts/olexit2_cmp.txt, append one result line to engine-patch-queue-2026-11-03.md. Kill only the chain's PIDs if aborting (find by command line 'olx2_chain' / label ol_exit2).
- DONE 17:01: ol_exit2 NOT ADOPTED (see queue line). Mean preserved (pts_game +0.04); no gate beyond 2 SD.

## Unit 5 QB in-game exit (declared before any outcome read, stamp below)
Declaration (family sim12_starter_exit; prior looks 48 = 32 unit 1 + 16 unit 2; this unit adds 3 scored cells: hazard, effect, placebo+LOSO as one each).
- Population: regular-season games 2009-2025 pbp (data/pbp/raw/20260925T202544Z). Participation exists only 2016+, so the QB uses pbp passer ids, not snaps. Starter = passer_player_id on the team's first qb_dropback play (designed runs, kneels, spikes are not dropbacks; scrambles and sacks are dropbacks by the nflverse flag; a trick-play first dropback by a non-QB is not screened, stated limit).
- Exit: a different passer takes a later dropback for the team and the starter takes none after it, with at least one further team drive after the starter's last-dropback drive. Exit drive = drive of the starter's last dropback; post window = drives strictly after it. Hazard per team-game overall, by quarter of the first replacement dropback, and blowout split by |score_differential| >= 17 at the replacement dropback (three-score rule; the line unit had no such split, its split was wp strata).
- Effect: offence EPA per play (pass and run, no kneel/spike) post minus pre (same game, own-team plays up to and including exit drive), minus mean of the same difference in no-exit team-games split at quarter fractions .25/.5/.75, matched on quarter, non-blowout at the split only; game-cluster bootstrap 300; non-blowout exits only. Placebo: pseudo-exit at the same drive index in a random no-exit non-blowout team-game, 300 reps. LOSO: per-season effect against baseline from other seasons, sign count.
- Hook spec only; engine reading below.
Recorded in ROADMAP 2026-11-11

## Unit 5 results (measured 2026-11-11 10:39; scripts/sim12_qb_exit.py, artifacts/sim12/qb_exit.json, ruff clean; 3 cells recorded, shard sim12_starter_exit.json changed; family looks now 51)
Coordinator note: QB effect already in lane (unit 2: -0.016 (-0.063, 0.024), 297 events, 2016-2025 participation, hazard 5.63%); not re-claimed. This unit adds hazard by quarter, blowout split, placebo, LOSO, points, engine, hook, on pbp 2009-2025.
Hazard (exit = different passer later, starter never returns, a further drive exists): 958 exits of 8007 team-games 11.96% (Wilson 11.27, 12.69). Blowout >=17: 578 (7.22%); non-blowout 380 (4.75%). By quarter of replacement dropback, all: Q1 59, Q2 116, Q3 205, Q4 578; non-blowout: 58 / 97 / 114 / 111 (0.72/1.21/1.42/1.39% of team-games). Pbp-defined hazard is wider than the lane's participation one (includes Q4 and any two-passer game).
Effect non-blowout, 373 scored events: +0.0704 EPA/play (0.0005, 0.1428), P>0 0.973; OPPOSITE sign to the lane -0.016. Placebo (same drive index, no-exit games): mean +0.0476 (-0.005, 0.104); 76.3% of pseudo effects at or below observed, so exit not separated from placebo (placebo shares the baseline gap, a quarter-matched baseline does not remove a drive-index trend). LOSO: 5 of 17 seasons negative. Not interpretable as a backup helping; inferred artefact of exit timing (exits follow poor first halves, regression up) and the control baseline.
Engine (read): scripts/mod25_generator.py:304 backup_rate, 494-518 qb_out per team-week, 525-542 season_ratings adds backup_off_epa_effect to the week offence rating. A whole-game backup only; no in-game exit is represented. Hazard ~0.5-1.4% per team-game before Q4 non-blowout is inside the realised .165 week rate only for full weeks (inferred).
Hook spec (not built): sim10_drive.py game() ~1466 builds xq, run_half q[5]=xq[pos] ~1363 feeds 1229-1252; separate Generator from the game seed draws exit time from the measured quarter distribution above; from that drive q[5] shifts by backup_off_epa_effect (-0.0604 real, full-week) times the fraction of game left; zero draws when flag off. Adoption Tier B: 3 seeds, gate count not lower, none lost, a gate moved toward real beyond 2x default seed SD. Not warranted now: measured in-game effect is sign-unstable.
Points per game: 2 x 0.0475 x 0.0704 x 30.3 = +0.20 (wrong-sign, unreliable); using the lane -0.016: 2 x 0.0475 x -0.016 x 30.3 = -0.046. Either is tiny against 44.8 pts/game. Next: none for QB.
