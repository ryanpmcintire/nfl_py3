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

## Next (unit 2 done below; remaining)
Old unit 2 text: split final-9-min drives by offense hurry-up (pass rate, no-huddle) to separate trailing-offense from defensive shape; check why sim TD rate in that cell is half real (SIM-19 depth/pass terms, end-of-game drives not truncated). Mechanism and entry point: trailing-offense end-of-game risk mixing at drive-draw (SIM-19), not a defensive prevent term.

## Open
Reliability/ypp not an admissible weak-signals effect unit (no epa_per_play run), so nothing recorded. Opponent offense quality not controlled. Sim yardage not stored.

## Unit 2 result Tue, Nov  3, 2026  5:52 PM, scripts/sim26_unit2_hurryup.py -> artifacts/sim26/unit2_hurryup.json (+unit2_run.log); ruff clean; looks 12 real cells + 9 decomposition cells; not recorded (rates, not admissible unit)
**measured** Hurry-up = drive no_huddle play share >=.5 OR seconds/play <= 7.0 (cut = best accuracy vs the no-huddle flag, acc .923; pbp row alignment .992). Real final 9 min share: neutral .075; trailing 5-7/8-14/15+ .120/.221/.205 [.096,.145]/[.195,.246]/[.184,.227]. TD rate hurry vs normal: .174/.251/.270 vs .258/.235/.208 (hurry NOT higher at 5-7). Neutral .110 vs .129.
**measured** TD surge vs neutral (real/sim/kernel-raw): 5-7 .121/.082/.030; 8-14 .111/.086/.048; 15+ .093/.065/.065. Share shift of hurry-up explains -.001 [-.003,.001], -.003 [-.009,.004], -.002 [-.008,.003] of the real surge: ~0. Kernel under-samples hurry share (8-14 -.071 [-.099,-.044]; 15+ -.035) but that costs only +.002/+.001 of TD gap. Truncation of drawn drives: .003-.004 (final 9), up to .008 second half: <=5% of gap.
**measured** Gap real-sim .039 [0,.074]/.025 [-.006,.057]/.028 [0,.056]. Entry: kernel stage real-kernel .091 [.055,.126]/.063 [.034,.092]/.029 [.005,.057]; downstream stages (outcome-model tilt, SIM-19 terms) add back +.056/+.041/+.004 so sim sits between. So the TD surge is a normal-pace phenomenon the pool kernel under-delivers (sd bandwidth smears -5..-14 states with neutral); hurry-up is not the mechanism. Caveats: kernel replicates SIM-21 base bandwidths (config unverified, no quarter strata); sim hurry class unobservable (no plays logged), so inferred via kernel weights.
**inferred** Mechanism: sd/clock kernel smoothing in draw() (sim10_drive.py 1206-1228) under-weights the real TD-heavy trailing drives; later offsets partly compensate. Next unit: measure kernel sd-bandwidth response (leave-game-out TD surge as function of sd bandwidth) and whether regime bandwidths (bandwidths_reg.json) change it.
