# mod25e-e95-endspot

## Goal
Second leg of the persistence carrier: where side A's later drives END (b_ex,
A end spot on A's H1 residual), real vs sim. E91/E93 showed the start-on-end-spot
leg (b_se) is attenuated in the sim but repairing it closes only ~15-20% of the
E88 x-slope gap (real -.1326, base -.0556, replay -.066). Parent lanes
docs/lanes/mod25e-generator-fidelity.md, docs/lanes/mod25e-e91-startspot.md.

## State
2026-10-05 E95 run (scripts/mod25e_e95.py, artifacts/mod25e3/e95/e95.txt, 57 looks, 300 boots). b_ex gap is the A start spot, not drive gain.

## Tried
E95 nonscoring A-H2 drive end spot E on own xa: real -.0824 sim -.0458 (s-r +.037, P>0 .91); start s0 -.0586 vs -.0223 (s-r +.036, P .97) carries 99% of gap; gain G gap +.0002 (P .54); plays/ypp gaps ~0 unresolved. Score share slope equal (-.00228 vs -.00214); end mix equal; sim lacks real downs endings (.001 vs .054, classification). Stall distance not tested (not in drive table). Unresolved_below_power, not closed.

## Next
Orchestrator 2026-10-05: territory latent REJECTED (fitted shift with no named
mechanism). The own-start leg is the same damped field-position chain as E91;
test it through E94 yardline-matched flip draws in a full 3-seed e5, then rerun
mod25e_e95.py and mod25e_adj2.py on the new label. Lane done unless that fails.

## Open
None.
