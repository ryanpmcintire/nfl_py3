# SIM-21

## Goal
Drive length and scoring rate decomposition for trailing and leading offenses: plays per drive, seconds per play, and the uniform scoring-drive deficit (ROADMAP SIM-21).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: SIM-19 found the simulator trailing-offense scoring-drive rate 2 to 6 points low in every cell and drives 3 to 13 seconds short at deficit 14+. SIM-18 found sim drive length within 3 to 4 percent for leading drives. Decompose into plays per drive and seconds per play by deficit and clock to find which named mechanism is missing; the repair is a measured mechanism, not an offset.

## Tried
Novelty check only. Nearest: SIM-19 (gap measured, unit 1), SIM-18 (leading drive length), SIM-13 (snap load); grep 'drive length' found no decomposition.

## Next
First unit for one agent: Check whether the scoring-drive gap also exists for leading and tied offenses, then split the trailing gap into plays-per-drive and seconds-per-play shares with bootstrap intervals. Data: SIM-19 artifacts sim_vs_real_drives.json and instrument logs (local).

## Open
Instrument config of instr_s11 is unverified; confirm before attributing. Zero crossing closes nothing; one fitted probability selects the side.

## Unit 1 progress (2026-11-03 ~10:10 PM, agent)
Wrote scripts/sim21_gap_decomposition.py (ruff clean): A sim(instr_s11) vs pool real by trailing/leading/tied x deficit x time cells (score, el, start yl; plays/spp inferred via log-log beta), start-field counterfactual, B kernel-stage replication of draw() (sim10_drive.py 1206-1228; state dims yl, sd bw 37.6 flat, log secs, timeouts, x, season; sd/lt response slopes), C outcome-model oof bias (el/td/fg targets tilted at 1294-1321). Run launched in background, writes artifacts/sim21/gap_decomposition.json and run.log (ends with "done N" line). Results NOT yet read. Next: read the JSON, state shares, name the mechanism, finish lane.

## Unit 1 result (2026-11-03 ~10:25 PM) artifacts/sim21/gap_decomposition.json; looks: 9 pooled + 108 cell + 15 kernel + 6 oof + 6 other = 144; not recorded (no admissible effect unit)
measured, sim(instr_s11, 384900 drives) minus pool real (53030), standardized to real cell mix, game bootstrap: score rate trailing -.0187 [-.0254,-.0120], leading -.0146 [-.0203,-.0086], tied +.0014 [-.0084,.0112]. Drive length trailing -1.9 s [-3.1,-0.6], leading -3.1 s [-4.5,-1.6], tied -1.8 [-4.0,+0.5]. Start yl (yards to goal) trailing -0.71 (sim better), leading +0.69 (sim worse), tied -0.28.
inferred plays share of el gap (real log-log beta): trailing 71%, leading 60%, tied 86%; sim logs no plays so seconds/play not directly observable.
Start field explains ~18% of leading score gap (-.0027), none of trailing (sim starts nearer goal).
Kernel stage only (replicates draw() 1206-1228, 12000 queries, leave-game-out, base bandwidths; config unverified): score bias trailing -.0099, leading -.0049, tied -.0142; el +4.8/-4.0/+3.2 s (mixed sign, so kernel is not the systematic length cause); candidate yl +0.5..+1.05 worse than query. Kernel sd response slope 0.33 (bandwidth 28 pts vs sd of sd 11): score state only a third carried.
Outcome-model oof bias (targets for the tilt at 1294-1321): el <=0.6 s, score <=.004: not the source. Residual (about half trailing, two-thirds leading) is post-kernel: head reweight 1245, dr adj 1291, tilt 1306, q4dec/fgdec, needs a run with trace to split.
Tied has no gap => cause is tied to nonzero score state, both signs.
Next: instrumented run (after Tier B chain) with SIM10 trace recording target vs achieved tilt mean for td/fg/el by sign of sd; if achieved<target, fix is tilt convergence/clip (lo/hi clip at 957), else head reweight.

## Unit 2 progress (2026-11-03 ~5:15 PM, agent) HOLD
Instrumentation added to scripts/sim10_drive.py (ruff clean, logging only, no rng use): env SIM10_TILT=1 (with SIM10_INSTR=<label>) sets eng.tl; draw() appends per drive [hf,hm,sd,secs,yl,npool,kern mean(4),post-head mean(4),tg0(4),tg after dr/q4dec(4),achieved(4),clip flags td/fg/el(3),ess] (out cols el,td,fg,ds) and game log gets key "tl" aligned 1:1 with hf<3 "dr" rows. Analysis: scripts/sim21_tilt_trace.py <label> (writes artifacts/sim21/tilt_trace.json, stage deltas by sign).
Flag-off smoke (sim --realsched 1 --seed 5 --seasons 3 --workers 1 --label smk21_off, log artifacts/smk21_off.log) was still running (bash task bd6o1hfxw); hash artifacts/sim10/smk21_off_s5/games.parquet via sha256sum must equal 05e23e0cb3b3003413498c0263ebadbc2bad0b23662154f60af68e509e2fa592. Not yet verified.
Next: verify hash; then free-RAM check (41.8 GB free at 5:03 PM); run SIM10_INSTR=tl_s11 SIM10_TILT=1 sim --seed 11 --workers 11 --realsched 1 --label tl_s11 (same size as instr_s11, 384900 drives, stable cells) in background; then python scripts/sim21_tilt_trace.py tl_s11; attribute gaps, name defect (tilt clip tilt_weights line ~957 vs shortfall), propose fix.
Update ~5:40 PM: flag-off smoke hash = 05e23e0c...fa592, IDENTICAL to HEAD (measured). Traced run launched in background: SIM10_INSTR=tl_s11 SIM10_TILT=1 sim --realsched 1 --seed 11 --workers 11 --label tl_s11, log artifacts/sim21/tl_s11.log (ends with games/seconds line); output artifacts/sim10/instrument/tl_s11/*.jsonl. Then: .venv/Scripts/python.exe scripts/sim21_tilt_trace.py tl_s11. Kill only by PID (find via python.exe cmdline containing tl_s11).

## Unit 2 result (2026-11-03 ~6:10 PM, agent) artifacts/sim21/tilt_trace.json, run tl_s11 (seed 11, 384900 drives, 16384 games, 351 s)
measured: stage means of td+fg by sign (kern, head, tg0, tg, achieved): trail .3192/.3192/.3187/.3191/.3194; lead .3395/.3395/.3463/.3446/.3449; tied .3503/.3503/.3603/.3616/.3617. Kernel=head (head reweight no-op), tilt shortfall ach-tg +.0003/+.0003/+.0001 (converges), dr adj +.0004/-.0017/+.0013, clip overall td/fg 1.9%/2.3% trail, 2.5%/2.8% lead (H2 last 5 min 7-17%, contribution to lead gap about -.001), el ach-tg0 within 0.04 s. Pool median 5600.
REALIZED play-through: scored drive rate trail .3037 v ach .3194, lead .3286 v .3449, tied .3540 v .3617. Cause: truncation. Drives with he==1 or el>=secs (sim10_drive.py:1424) drop events after the clock (1432): truncated share trail 8.3%, lead 11.2%, tied 3.3%; in them achieved .174/.146/.236 but realized .039/.029/.068. Loss share*(ach-real): trail -.0113, lead -.0131, tied -.0056 v unit-1 gaps -.0187/-.0146/+.0014 (about 60% trail, 90% lead; tied does not fit, inferred).
Attribution (inferred, v unit-1 gap): truncation double count ~60% trail / ~90% lead; kernel (unit 1) ~half trail, 1/3 lead overlaps (the kernel stage shifts outcomes relative to target only via tg0; kern-tg0 trail +.0005, lead -.0068 here); clip, tilt shortfall, head, dr, q4dec ~0-10% each.
Defect: the tilt target (real full-drive scoring in the state) is matched on UNtruncated candidate outcomes, then the sampled drive is truncated by remaining clock (1424-1432), so clock-cut zero-scores are counted twice. Fix (measured, for review, not implemented): tilt on clock-truncated candidate outcomes (td/fg/el with events past secs removed for the query's secs) so achieved = played; or compare after truncation. Needs orchestrator review.

## Unit 3 progress (2026-11-03 ~5:30 PM, agent)
Implemented SIM10_TRUNCTILT=1 (default off) in scripts/sim10_drive.py: init builds padded per-drive event arrays (after pl_ptr line), trunc_out(ci, secs) before draw(), `oa` used in the three tilt_weights calls and the tl trace. Candidates with he==1 or el>=secs get el=min(el,secs), td/fg/ds recomputed from events with off<=secs (same rule as the post-draw cut). Target unchanged: outcome models are fit on real drives whose td/fg already include real clock truncation; only the candidate side was untruncated (measured, unit 2).
Smokes (background bnozd6x5i): artifacts/tt_off.log, tt_on.log; hashes in artifacts/tt_hash.txt (off must equal 05e23e0c...fa592).
Next: traced run SIM10_INSTR=tt_s11 SIM10_TILT=1 SIM10_TRUNCTILT=1 --seed 11 --workers 11 --realsched 1 --label tt_s11, then scripts/sim21_tilt_trace.py tt_s11; Tier B label trunc_tilt seeds 11,12,13; adjudicate: .venv/Scripts/python.exe tests/scratch/tierb_cmp.py trunc_tilt 11,12,13.
HOLD 5:31 PM: smoke chain running as background task bnozd6x5i (uv PID 33664, python 35060, worker 11336; tt_off then tt_on). Results artifacts/tt_hash.txt (ends "done"). Not yet run: traced run, Tier B.
Smoke result 5:45 PM (measured): tt_off hash 05e23e0c...fa592 = HEAD; tt_on 3f5ca40e...bd96 differs. Traced run tt_s11 launched (task bxyfbzkzy, log artifacts/sim21/tt_s11.log); then Tier B trunc_tilt.
Unit 3 trace 5:55 PM (measured, tt_s11, flag on, 380917 drives): achieved v realized td+fg trail .3178 v .3222, lead .3442 v .3453, tied .3608 v .3602 (before: .3194 v .3037, .3449 v .3286, .3617 v .3540). Tier B trunc_tilt seeds 11,12,13 chained in task bekftbgkd (logs artifacts/trunc_tilt_S.log) ending with tierb_cmp output.
- 2026-11-03 17:59 ET (root CORRECTION): kernel replications in SIM-21 unit 1 and SIM-26 unit 2 used the base bandwidths.json (sd 37.6 x g_sd, no week dimension, no h_reg), but the engine default is --h bandwidths_week.json with per-regime h_reg for halves 1-2 (sim10_drive.py 1239-1242, 2731; regime 3 sd bandwidth 18.8, regime 4 178). Their kernel-stage gap numbers (.09/.06/.03) and the '28 vs 11' bandwidth figure describe the wrong configuration and are NOT evidence that the engine's kernel smears trailing states. The SIM-21 unit 2 truncation finding stands (measured from the engine's own trace, tl_s11). Measurement with the engine's regime bandwidths is running (SIM-26 unit 3).
