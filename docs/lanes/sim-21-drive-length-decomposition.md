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
