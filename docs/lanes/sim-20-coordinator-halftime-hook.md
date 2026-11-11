# SIM-20

## Goal
Coordinator-keyed halftime adjustment inside the drive resampler (ROADMAP SIM-20).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: SIM-17 showed second-half success rate over first-half is carried by coordinators and an OC key predicts held-out halves (corr .062). In SIM-10 the second-half draw() kernel can reweight candidate drives by the offence's coordinator-keyed halftime shift, measured from play-by-play, not tuned. Checked against Tier B (Q4 slope, key-number mass) at the pinned flag-off hash.

## Tried
Novelty check only. Nearest: SIM-17 (measurement and pick term), SIM-09 (stakes arm, not adopted), SIM-19 (trailing risk, gap measured); grep 'halftime' found SIM-17 only.

## Next
First unit for one agent: Add the measured shift as a content-kernel reweight in the second-half draw behind a new flag defaulting off, run the 16384-game instrument once and report Tier B against the flag-off result. Data: SIM-17 shift table, SIM-10 resampler instrument logs (local).

## Open
Heavy run: single heavy job rule; queue behind the SIM-12 redesign chain. Zero crossing closes nothing; one fitted probability selects the side.

## Update 2026-11-11 10:44 EST
Done in SIM-17 unit 4 (docs/lanes/sim-17-halftime-adjustment.md): coordinator key beats head coach (OC offense held-out slope 0.40 [0.18, 0.61], HC 0.17 [-0.63, 1.03]; DC defense 0.44 [0.09, 0.77], HC -1.10 [-2.87, 0.68]); reliability by perm test below null for all four second-half keys, Q3-only inside the null. Built as a patch, not a content-kernel reweight: shift enters the xq quality feature (same channel as the OL exit hook), SIM10_HTADJ=1, patch tests/scratch/sim_finish_ht/sim17-halftime-hook.patch, apply-check OK, table artifacts/sim17/halftime_shift_h2.json. Run plan and adoption rule are in the SIM-17 lane Next (unit 4). Not run in the engine.
