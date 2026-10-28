# SIM-11 strength transmission (closes SIM-10's two failing gates)

## Goal
Get strength variance (SIM-10 47.2 v real 58.0 [51.4, 64.9]) and weeks 10-18 R2 (.117 v .146
[.131, .161]) inside the real interval on the 3-seed mean without losing the other 12 gates.
Owner 2026-10-24: get them close to reality. ROADMAP SIM-11.

## State
LOCKED default (commit 0df0a23d): `python scripts/sim10_drive.py sim --seed S --workers 11`
= `--droof 1 --drmu 0 --mulabel 1 --strength artifacts/sim11/strength_terms.json`; bare seed 11
reproduces s11lab_s11 exactly. Random schedules (SIM-10 protocol), seeds 11-13: 13/14, strength
52.4 PASS, R2 .120 FAIL. Old config: `--strength "" --droof 0 --drmu 1 --mulabel 0`.
Flag `--realsched 1` (default 0): each sim season plays the matching real 2009-17 schedule
(16 games, 17 weeks), teams randomly mapped to slots: the gates' own measurement design.
3-seed means on real schedules (artifacts/sim10/rs_s1*, rs10_s1*): SIM-11 11/14 (fails strength
49.8, R2 .116, q4 -.0806 v floor -.0797); SIM-10 config 9/14 (strength 46.6, R2 .116, xq -20.6,
q4 -.084, mass14 .0510). Seed-level game-path noise ~1.5-2 strength points.

## Tried (measured; probes 11x2 seed 21; probe per-phase noise ~+-.08)
- Kept: label-matched strength target (tail effective sample 13-31 -> 75-232; TD target miss
  5-8 pp -> .3), per-phase strength terms, OOF and state-only DR residuals.
- Rejected: phase-scaled target; outcome-only tilt; drive-EPA and plays-per-drive targets;
  label tilt in end-of-half regimes too (end2h points 2.83 v real 1.80, EPA all .917 v .942);
  adjacent-season labels (label var +4.5% off, +17% def; not worth a rebuild); recalibrating
  latent variance to simulated EPA (would hide the .94 transmission; compensating knob).
- Form (weeks-scale team strength): opponent-adjusted offence lags 1-4 minus 9-15 +.97e-3
  [+.10,+1.98] P+ .965 (2009-25); exact-likelihood LOSO, net adjusted EPA: 2009-17 best decay .8
  (held-out +1.22 nats, form var 3.3e-3, season 8.9e-3); 2009-25 decay .8 +.27 nats, var 1.9e-3.
  Real schedules make consecutive opponents dissimilar (opp strength cov lags 1-4 -.84 v -.37,
  P .00), which hides form in raw EPA. Estimated R2 gain from form alone ~+6% (inferred).
- Diagnosis (measured, artifacts/sim11/draws_lab21.parquet): achieved drive mu on sim true strength slope .68 in every non-end-of-half phase (sd tgt .103 v a_mu .071); drive EPA slope already matches real (lab21 all .664 v real .658), so the .94 is the label-matched target's cx attenuation: real mu (posterior mean) has sd ~.07 < sim true sd .103, so deconvolved targets leave support (old mulabel 0 = TD miss, tail ESS 13-31). Resampling cannot repair it without new support or a joint outcome/mu target; no legit fix found.
- Rejected 2026-10-27: additive form latent (`--form 1`, SIM10_FORM; ratio 3.3e-3/8.9e-3, decay .8, seed 11 realsched): margin_sd 15.61 (fail), noise_re_var 191 (fail), q4 -.044, mass_3/7 fail, strength 53.0 pass, R2 .111 (no gain). Variance added on top inflates noise. Flag left default 0 (mod25_generator.py gen_world_latents form_ratio, sim10_drive.py FORM_*).

- 2026-10-27 unit: candidate (a) skipped as the same family as the rejected outcome-only tilt (a residual tilt still reweights drives by outcome, whose response to strength is the .664 already matched). (b) variance-preserving split added as `--form 2` (SIM10_FORM=2; mod25_generator.py `form_preserve`: persistent scaled by sqrt(1-ratio), form added); results below.

## Next
1. Adopt `--realsched 1` as the protocol for every new comparison (like-for-like with the
   real gates); random schedules read strength 2.6 higher for SIM-11 (0.6 for SIM-10).
2. Remaining persistent-strength deficit (real schedules): per-play EPA LOGO .88, margin LOGO
   .88 of real. Engine per-play EPA transmission .94 is the main piece (within-outcome drives
   come from teams ~2/3 as strong: shrunk labels); then excess catch-up (q4 -.076 v -.060),
   special teams (~3%), form (R2).

## Open
Persistent plays-per-game: real LOGO cov 3.56 v sim ~1.0; not a gate.

2026-10-27 result: `--form 2` seed 11 realsched (artifacts/sim10/rsfp_s11) vs default (rs_s11): strength 40.5 v 48.6 [51.4,64.9], R2 .092 v .117 [.131,.161], noise_re 182.7 v 170.3 FAIL, q4 -.083 v -.073, margin_sd 14.94, xq -17.5, mass_7 .079 fail. Worse on every gate; no 3-seed run. Variance moved to form is not seen as persistent strength through the .68 transmission. Not default. Next: (c) excess catch-up, or a new-support transmission fix.

2026-10-27 (c) excess catch-up FIXED: measured Q4 draws vs real (state-matched net points per drive match; clock-kill does not): last 120 s, leading offence (sd>8) runs out the clock in real 97% (40-80 s) / 84% (80-120 s) v sim 71% / 49%; trailing-by-9+ gets .105 v .061 drives per game at 40-80 s. Cause: Engine.draw (sim10_drive.py ~962) excluded clock-ended (he) candidates starting at lower secs than the state, which for a leader are the kill-the-clock drives. Fix `--hecarry 1` (now default): in half 2 with the offence ahead, he candidates with sd>0 are kept (they run out the clock). realsched 3-seed (rshc_s11-13) v rs: q4 -.0739 v -.0806 (real -.060 [-.0797,-.0393] PASS), strength 51.34 v 49.84 (floor 51.4, just short), R2 .1128 v .1158, mass_7 .0822, xq -13.1, all previously passing gates still pass: 12/14.
(d) net-label idea: already the design. seg x is the net label (sd .1135), posterior_table slope eiv_b=.6726 is the .68 transmission; netting adds no support. Not run.
Next: strength 51.34 v 51.4 and R2 .113 v .131 remain; form/new support.

2026-10-27 R2 decomposition (data only, tests/scratch/sim11_r2/{decomp,cf,lscale}.py): w10-18 R2 = slope^2 vx/vy. Sim rshc 3-seed .1128 (reproduced exactly) slope 26.7 vx .0351 vy 222.6; real 2009-25 slope 29.8 vx .0369 vy 221.9 R2 .147 (09-17: 30.7, .0360, 233, .1455). Gap is the margin-on-state slope (~80% of log gap), var_x ~20%, vy 0. Same x builder both sides (gen.autocorr_and_r2). Oracle: R2 of logged lat_d on y .236; x captures .49 of it. Persistent EPA lag plateau real 6.12e-3 v sim 5.70e-3 (+7%); same root as strength deficit. Drift: sim lag profile flatter than real (no positive share). Noise: q .975 (noise_re 173->165) +.004. Tail compression ruled out: draws_lab21 achieved/target ratio .625 in every decile. Counterfactual (logged lat, x and y scaled): c 1.019 R2 .119-.123, c 1.064 .132-.136, c 1.082 .138-.143. Proposal: proposal.patch flag --lscale lscale.json (c=1.036 from lag plateau ratio); expect R2 ~.125, strength ~55; it is latent recalibration in spirit (previously rejected), owner decides. Screen once v3 finishes: see report.
- Orchestrator 2026-10-27: the `--lscale` proposal (tests/scratch/sim11_r2/proposal.patch, c=1.036 from the real/sim lag-plateau ratio) is REJECTED as a compensating constant: it scales the latent to match an output aggregate (AGENTS.md no-bias/scale rule). Not applied. Root of both failing gates (measured, decomposition): persistent strength reaches outcomes 7% weaker (plateau 5.70e-3 v 6.12e-3; margin-on-state slope 26.7 v 29.8). Next named components to measure: special teams strength (~3% per earlier note), then turnovers and field position by team strength.

2026-10-27 component decomposition (data only, tests/scratch/sim11_st/{decomp,derive,mkpatch}.py; real 2011-25 REG 480 team-seasons v rshc_s11-13 6144; split-half odd/even games, 90% bootstrap):
- Persistent net scrimmage EPA/game var: real 38.07 [32.8,43.5] v sim 35.49 [33.7,36.9] (-6.8%): off 23.09 v 21.95, def 9.47 v 8.31, cross 5.51 v 5.23. EPA/play persistent var .0061 v .0057; margin per persistent EPA (b) real .840 v sim .861, so downstream conversion is NOT weaker; the deficit is in per-play persistent EPA itself (def -12%, off -5%).
- Persistent margin var 27.13 v 25.54. Non-scrimmage residual (margin minus scrimmage net EPA): real 1.23 [.67,1.79] v sim -0.06 [-.18,.07]: sim has none. Real ST EPA net var .93 (own punt .19, FG .11, kickoff .04, XP .002; return side rest), b on strength .019 [-.005,.042] (near independent of strength). Turnover margin persistent var .103 (b .043), start field position 3.84 yd^2 (b .24 yd/EPA); both already priced inside scrimmage EPA.
- Persistent plays/game var real 3.78 v sim .87 (4.3x); epp x 62 plays explains ~all net var, so volume is not the carrier (inferred).
- Proposal: tests/scratch/sim11_st/proposal.patch (git apply --check OK, NOT applied): flag --st st_latent.json, per team-season FG-make logit latent N(0, .0945 [.057,.138]) scaling the offence fg target by make-rate ratio (.845). Screen after v3: python scripts/sim10_drive.py sim --seed 11 --workers 11 --realsched 1 --st tests/scratch/sim11_st/st_latent.json --label rsst, compare gates v rshc_s11. Inferred: closes the missing 1.3 pts^2 residual only in part (FG is .11 of .93); does not touch the per-play EPA deficit.

2026-10-27 sim11_def unit (data only, tests/scratch/sim11_def/; no sim run, no tracked file edited):
- (A) Defence persistence. Measured (draws_lab21, ph0 sd0, two-way team fits): achieved/target slope off .690, def .691, so matching loses sides equally. Label reliability (2009-17 LOGO): off .74, def .47 (sum label .644 v eiv_b .673); conditional mean on the sum is unbiased, so unequal weighting is not the cause (inferred). Per-play persistent EPA var real 2011-25 off .00597 def .00292 v sim off .00560 (-6%) def .00236 (-19%). Cause: latent fit window 2009-17 (def logo_cov .002454 v 2009-25 .002725, +11%, 2018-25 minus 2009-17 P+ .84; off +1.4%). Fix def.patch: latfit --seasons/--out, default off; artifact latent_fit_2009_25.json (def var_mu .002525 -> .00293, off .00704 -> .00728). Screen: SIM10_LATENT=tests/scratch/sim11_def/latent_fit_2009_25.json python scripts/sim10_drive.py sim --seed 11 --workers 11 --realsched 1 --label rsdef. In-sample caveat: window overlaps the gates.
- (B) st_full.patch (flag --stfull tests/scratch/sim11_def/st_full.json): 5 per-team-season latents (punt net, punt return, kickoff net, kickoff return yards; FG logit), joint cov from split-half, shift next-drive start by kicker net minus receiver return (punt flag per segment seg_punt.npy, kickoffs after scores), FG scale as before. Check: predicted ST net persistent var .317 [.186,.462] v real .931 [.626,1.231]: the five channels reproduce about a third; real split: yard+FG+XP part .593, residual .077, cross .131 (volume and FG/XP EPA outside the latents; not filled). Screen: python scripts/sim10_drive.py sim --seed 11 --workers 11 --realsched 1 --stfull tests/scratch/sim11_def/st_full.json --label rsstf.
- Next after v3 ends: screen both, compare to rshc_s11; apply only what moves strength/R2 without losing gates.
- Orchestrator 2026-10-27 (read): era mismatch. Latent strength process fitted on sim10 SEASONS 2009-17 (scripts/sim10_drive.py:18) while realism gates are computed on generator REAL_SEASONS 2011-25 (scripts/mod25_generator.py:22). Defence persistent variance is +11% over 2009-25 (P late>early .84), so the refit to 2009-25 aligns the fitted era with the checked era; it is a likelihood fit of the real process, not a knob. Queue after the MOD-25 v3 run: seed-11 screens rsdef (SIM10_LATENT=tests/scratch/sim11_def/latent_fit_2009_25.json, def.patch) and rsstf (st_full.patch), then 3 seeds for any that improves without losses. Note: v3 distillation was produced on the 2009-17 latent; rerun it if rsdef is adopted.
- 2026-10-27 screens (measured, 3-seed means 11-13, real schedules; tests/scratch/sim11_def/three_seed.py over artifacts/sim10/<label>_s1N/gates.json):
  default rshc 12/14 (strength 51.34 F, R2 .1128 F, mass_3 .1365); rsdef (SIM10_LATENT=latent_fit_2009_25.json, era-aligned) 13/14: strength 54.19 PASS, R2 .1319 PASS, mass_3 .1321 FAIL (floor .1354), q4 -.064, noise 174.5; rsdefst (+ --stfull) 12/14: R2 .1254 F, mass_3 .1324 F. Seed 11 rsstf alone 13/14 (R2 .117 F).
  Not adopted: rsdef loses mass_3, a previously passing gate. The sim under-produces 3-point margins (real .1471); more strength spread exposes it. Next: key-number-3 mechanism (late close/tied game FG decisions, walk-off FGs, OT endings) on top of rsdef, then adopt.
  Patches applied to scripts/sim10_drive.py, default off: `latfit --seasons/--out`, SIM10_LATENT override, `--stfull`.

2026-10-27 mass_3 unit (SIM10_LATENT=latent_fit_2009_25.json; scripts tests/scratch/sim11_def/{m3dec,drivelog,drivecmp,probe_tied,fitdec}.py):
- Decomposition (rsdef s11, 2009-17 real): mass_3 sim .1364 v real .1471; reg-3 .0958 v .1029, OT-decided-by-3 .0406 v .0443, ot_share .0583 v .0616. P(final reg margin 3 | |m3|): tied .214 v .232; lead1-3 .161 v .178; 4-6 .124 v .102; 7-9 .128 v .130; 10-14 .075 v .096. Q4 net 0: sim .137 v real .160.
- Drive level: half-2 last 120 s, tied: real FG-only .224 v sim .116 (TD .019 v .041); sim also visits tied-late drives less (.071 v .118 per game). Replay of the 269 real tied-late states through Engine.draw: sim FG .171 v real .223; yl<=50 real .56-.62 v sim .30-.43. Cause: outcome model tg (regime 4 ignores yl/sd in kernel, h=69/178) under-predicts FG in states where a FG ties or takes the lead (sd in [-3,0]): pred .143 v actual .196 (regime 4), .173 v .199 (regime 3). Logit shift fit LOSO: regime 3 +.179 (range .157-.219), regime 4 +.409 (.325-.454), heldout dLL +2.8 / +4.8. TD shift ~0 (not used).
- Implemented: `--fgdec artifacts/sim10/fgdec.json` (sim10_drive.py Engine.draw, cmd_fgdec fits it; default off). Seed 11 realsched rsfgd: mass_3 .1304 v rsdef .1357 (worse; SE .0027), mass_7 .0787 v .0801, ot_share .0556 v .0583: more decisive FGs break ties. Not 3-seed run, not adopted.
- Open: remaining gap is spread over m3 classes (10-14 lead .075 v .096; Q4 net 0 .14 v .16), i.e. drive-sequence structure not single-state FG choice. Next: measure Q4 two-drive answer sequences (TD then answer) real v sim.
- 2026-10-27 (measured): real mass_3 is era-stable (2009-14 .1445, 2015-25 .1465, 2011-25 .1429; season sd ~.022), so the mass_3 shortfall is not era. --fgdec (late FG-decision logit shift, LOSO +.179/+.409) is real but alone lowers mass_3 (.1304 seed 11) by cutting OT; kept default off.

2026-10-27 Q4 sequence unit (tests/scratch/sim11_q4/{collect,ana}.py; 3 sim worlds seed 11 rsdef config v 2289 real 2009-17 games; Q4 drive starts, offence-side sd):
- Drives per Q4 match (real 5.94 v sim 6.01; last 300 s 2.42 v 2.45); answer-drive rates after a TD or FG match within noise. Misfits are outcome rates by score state. Trailing 4-8: FG real .057 v sim .094 (TD .252 v .228); tied: FG .202 v .176, TD .116 v .140; leading 4-8: TD .124 v .150; tied drive starts per Q4 .374 v .325, ties made from nonzero .120 v .106.
- Cause (measured, q4dec fit): outcome model overpredicts FG when trailing 4-8 (regime 3 pred .100 act .070, LOSO coef -.41, heldout dLL +7.9), underpredicts FG in decisive states (trail 1-3 +.21, tied +.14 reg 3, +.47 reg 4), leaders' TD overpredicted (-.10..-.14).
- Implemented `--q4dec artifacts/sim10/q4dec.json` (sim10_drive.py q4_bin, q4dec_fit, Engine.draw): per (regime 3/4, score-state bin) logit shifts on td and fg, a cell is kept only if LOSO heldout dLL > 0; subsumes --fgdec (use alone). Default off. Fit: `python scripts/sim10_drive.py q4dec --realsched 1`.
- Screens (realsched, SIM10_LATENT era-aligned, 3-seed means 11-13; three_seed.py): rsq4d (td+fg shifts, q4dec.json) 12/14: mass_3 .1432 PASS, q4 -.0833 F, R2 .1196 F (TD shifts make catch-up worse); rsq4fg (fg shifts only, q4dec_fg.json = q4dec.json with td {}) 13/14: mass_3 .1383 PASS, mass_7 .0865, q4 -.071 PASS, strength 54.07, noise 171.9, R2 .1301 F (floor .1313; rsdef .1319, within seed noise ~.005). Seed 11 rsq4d ot_share .0606 v .0616 (rsdef .0583).
- Next: R2 is the only miss on rsq4fg and sits at seed noise; run more seeds (14-16) or adopt rsq4fg as rsdef+fg-state shifts and re-judge R2 on 6 seeds. Orchestrator adopts.
