# SIM-26

## Goal
Leading-defense shape late: yards per play and explosive-play rate for defenses protecting a lead (ROADMAP SIM-26).

## State
Unit 1 done Tue, Nov  3, 2026  5:46:33 PM. scripts/sim26_prevent_shape.py -> artifacts/sim26/prevent_shape.json (+run.log). Looks: 48 real state cells (4 metrics x 3 lead bins x 4 clock bins), 8 reliability, 24 sim cells. Bins inherited from SIM-19 quantile edges (lead 5-7/8-14/15+, neutral |diff|<=4 same clock bin; clock <=546,<=1432,<=2212,>2212 s). Explosive = yards >= 15 (0.9 quantile, measured). Baseline = same defense-season close plays, same clock bin; game-cluster bootstrap 300.
**measured** Prevent shape NOT present as declared. Final 9 min, leading 5-7 / 8-14: ypp +.44 [.30,.80] / +.35 [.21,.65]; explosive +.019 [.013,.032] / +.018 [.011,.028] (MORE, not fewer); drive TD rate +.15 / +.13 (CI >.10), score rate -.006 / +.05. Minutes 9-24: ypp +.15/+.24, explosive +.006/+.010, TD +.02/+.04. Lead 15+: no yardage shift; score rate -.04 to -.09 in all clocks (opponent offense quality uncontrolled, includes first quarter). Confound: leading defense = trailing offense hurry-up (SIM-19), prevent cannot be separated at play level with these data.
**measured** DC spell reliability (spells>=16 games, 209/304 pairs, lead>4 and clock<=1432 s, baseline within half): odd/even r ypp .03 [-.08,.10], expl .01, score -.11, TD -.12 (sign-flip p .13-.82); t vs t+1 r ypp -.05 [-.14,.06], expl .12 [.01,.22] p .033, score -.01, TD -.02. No stable DC trait.
**measured** Sim vs real drive level (instr_s11, 16384 games, drive points only; sim stores no yards): score-rate diff vs close matches in 11/12 cells (e.g. lead 15+ -.05..-.09 both). Sim understates final-9-min TD surge: TD diff real .148/.128/.113 vs sim .082/.086/.065 (real CIs exclude sim); also lead 8-14 at 9-24 min real .040 vs sim .015. Sim state kernel (sd, lt in DIMS) does draw real state-conditioned drives, so no neutral-state resampling issue; the gap is the clock-end hurry-up TD rate.

## Tried
Declared before signs: prevent = ypp up, explosive down, TD down late. Refuted as stated for this population (late leading defenses concede more of both).

## Next
Unit 2: split final-9-min drives by offense hurry-up (pass rate, no-huddle) to separate trailing-offense from defensive shape; check why sim TD rate in that cell is half real (SIM-19 depth/pass terms, end-of-game drives not truncated). Mechanism and entry point: trailing-offense end-of-game risk mixing at drive-draw (SIM-19), not a defensive prevent term.

## Open
Reliability/ypp not an admissible weak-signals effect unit (no epa_per_play run), so nothing recorded. Opponent offense quality not controlled. Sim yardage not stored.
