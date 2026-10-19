# mod25e-e180 final-drive turnovers (2026-10-18)

## Goal
Is the final-drive turnover deficit real on base crH...gx2faldrg (3 seeds s11-13, burn 2 seasons) vs real 2009-17, and what carries it?

## State
Measured (tests/scratch/e180/fd.py builds, an.py analyses; game bootstrap 400 reps, 90%; trailing 1-8, Q4 drives, down<=3 run/pass, INT+fumble lost; sim minus real):
- Start <=240s: TO/pass .0313 v .0609 (-2.96pp [-3.49,-2.43], P+ 0); TO/run .0088 v .0142 (-0.54pp [-1.35,+0.14], P+ .11); TO/drive .133 v .275.
- By clock: start 120-240 TO/pass .030 v .040 (-0.97pp [-1.58,-0.32]); <=120 .032 v .086 (-5.39pp [-6.45,-4.41]); <=60 .031 v .128 (-9.73pp [-11.6,-7.8]). Sim rate flat ~.031 at every clock, real rises 4x. TO/drive <=120 .115 v .316.
- By gap: 1-3 <=120 TO/pass .033 v .088 (-5.5pp), 4-8 <=120 .032 v .085 (-5.3pp): same, gap-independent. 1-3 run TO .005 v .048 (-4.3pp [-9.6,-0.7]).
- Pass share <=120 .931 v .945 (-1.4pp [-2.3,-0.5]); plays per drive equal (3.76 v 3.79). Air yards absent from sim logs and from raw pbp columns; no deep proxy.
- Game-end convention: last play run/pass turnover per game real .0369 (85/2304) v sim .0289 (1052/36402); trailing last rp play gsr<=60 TO share real .1285 v sim .0655. GFL relabel does not hide turnovers; the gap is not an end-flip artifact.
Carrier: the sim's pass-turnover probability is not conditioned on remaining clock (and gap). Existing RISK=1 tov (scripts/mod25e_risk.py:267-356, install_tov 478; fitted lead band x time-in-half, LOSO 9/9) is a candidate-row state-ratio reweight, off in base (label lacks r), barely moved trail>8 pass lost (-.0050 to -.0041, generator-fidelity lane line 29): consistent with a pool-limited reweight (inferred, not tested here).
Recorded weak-signals mod25e-e180-final-drive-pass-turnover-gap (rate_pp -5.39, unresolved_below_power, P+ 0.0).

Offline PTO (measured, 165770 real pass plays, 2304 games, 9 seasons LOSO, game bootstrap 1000, 90%): held-out LL/play full v pool served kernel share (best shrink) +0.00437 [+0.00397,+0.00477] P+ 1.0; v clock-free +0.00305 [+0.00271,+0.00338]; trail1-8 Q4 <=240 +0.04321 [+0.0367,+0.0498]. Trail1-8 Q4 <=120 rate real .1073, pool served .0459 (-6.14pp [-6.90,-5.37]), PTO .1039 (-0.33pp [-1.10,+0.42]); <=60 real .1642, pool .0448, PTO .1602. Pool is not the limit: only 26 of 165770 states lack turnover rows; pool served rate is clock-flat. Guards: non-trailing +0.01pp [-0.10,+0.10]; Q1-Q3 all +0.03pp [-0.06,+0.11]; Q1-Q3 trailing +0.13pp [0.00,+0.26] P+ .95 (base +0.14); non-turnover yards mean real 3.386 v draw 3.382 (identical to base conditional by construction, no fallbacks). Recorded weak-signal mod25e-e180-pto-loso-ll-vs-pool (unresolved_below_power).

## Tried
RISK tov (E78) reweight; GSTOP2 and GFL do not touch it. E79 "late turnovers ruled out" is contradicted for the current base.

## Next
PTO built default-off (scripts/mod25e_pto.py, env PTO=1, raises if CLSDRAW=1 also set; fit artifacts/mod25e3/pto/fit.json, loso_folds.json; scratch tests/scratch/e180/pto_fit.py, pto_replay.py, outputs .txt). Declared before fit: target P(turnover | pass, down 1-3, Q1-Q4) with turnover = (flip and points_off<6) or points_def>=6; 31 named features (down, distance band, yardline band, score band, seconds-left-in-half bins 60/120/240 x score band x half, off/def rating); no timeout feature (not in arrays). Family mod25e-e180, one look.
crH patch (not applied; crH untouched): after the clsdraw block at crH.py:206-209 add
    import mod25e_pto as ptm
    if ptm.enabled():
        ptm.install_pto()
and with the other label lines add: if os.environ.get("PTO") == "1": label = label + "pt"
Smoke (not launched): tests/scratch/e180/smoke_pt.sh (FD4=1, GSTOP2 unset, PTO=1, seed 31, 4 seasons). Gate: late TO/pass trail1-8 start <=120 target .086 (sim base .032), then margin SD, mass3, pts/g, catch-up checks; also check sim marks sack-fumbles on pass rows and count PTO fallbacks (STATE["fallback"]).
Side finding: possession_flip is true on scoring plays (pass flip .0728 = TD .0395 + turnover .0332); scripts/mod25e_clsdraw.py class_of treats all flips as other_turnover, so CLSDRAW class 1 contains TDs (unverified effect on its fit; review).

## Open
Real numerator includes sack-fumbles lost on pass plays; sim flip on pass includes the same class only if the engine marks it (unchecked). Whether the candidate pool holds enough turnover rows late (pool ceiling) is untested.

Update 2026-10-18: PTO crash fixed (mod25e_pto.py play_frame walks up to pol frame for qtr/clock_val/score_diff/distance; RG wrapper frame had no qtr). Tiny run (tests/scratch/e180/smoke_pt_tiny.sh, 1 world 1 season 1 worker) simulated all plays with PTO in 467s; died after sim in all_metrics (ZeroDivision, burn 2 > 1 season, no games kept), so no turnover rate. smoke_pt.sh appended to e178/queue.txt.

Update 2026-10-18 smoke scored (tests/scratch/e180/pt_an.py -> pt_an.txt; 1 world, 2 kept seasons, seed 31, 544 games, game bootstrap 1000, 90%). PTO v matched FD4 base v real, trailing 1-8 Q4 pass TO/pass: start<=120 .071 v .021 v .086 (PTO-base +4.96pp [+2.24,+7.61] P+ 1.0; PTO-real -1.53pp [-3.94,+0.99]); <=60 .068 v .039 v .128 (+2.90pp [-2.2,+8.0] P+ .83); 120-240 .053 v .064 v .040. Guards: non-trailing Q4 pass TO -0.54pp P+ .09; Q1-Q3 trailing +0.41pp [+0.05,+0.77] (offline +0.13, real .0339 v PTO .0332); Q1-Q3 all +0.12pp; non-TO pass yards +0.05 (identical by construction). Aggregates PTO v base: SD 15.85 v 15.15, mass3 .138 v .154, pts/g 44.40 v 44.16, noise var 182 v 167, q4_slope +.0002 v -.028. Final drives (start<=240, tiny n): scored .42 v .52 (real .26), clock-end .160 v .086 (real .177). STATE['fallback'] is not persisted in e5.json (no count). Verdict: passes smoke bar. run_pt_3seed.sh written (FD5 base, cand label ...gx2faldrgpt, seeds 11-13 base reused), not launched. Recorded mod25e-e180-pto-smoke-late-pass-to.

## PTO 3-seed, seed 11 only (measured 2026-10-19 00:10 ET; FD5 base ..._s11 v ...pt_s11 e5.json; provisional)
SD 15.040 -> 15.777, noise 168.9 -> 186.2, xq_cov 0.3 -> 17.8, Q4 slope -.046 -> -.020 (real -.060), mass3 .128 -> .121, late r2 flat, pts flat. All persistence checks move away.
- Seed 12 (measured): SD 15.235 -> 15.767, noise 172.2 -> 183.0, xq_cov 5.5 -> 17.3, Q4 slope -.044 -> -.018, mass3 .129 -> .122, late r2 .148 -> .153. Two seeds agree: PTO raises persistence strongly (xq_cov +12..+18). Expected verdict after s13 + boot: not adopted alone; candidate component once the missing catch-up channel is modelled (gap table E182: Q4 possessions -0.38/game).

## PTO 3-seed VERDICT (measured 2026-10-19 02:00 ET; artifacts/mod25e3/e180_pt_boot.log; FD5 base v FD5+PTO)
SD 15.179 -> 15.766 (+0.587 [+0.419,+0.743]); noise 170.8 -> 184.5 (+13.7 [+10.2,+17.1]); xq_cov 3.8 -> 17.6 (+13.8 [+9.7,+17.7]); Q4 slope -.042 -> -.018 (+.024, away from real -.060); mass3 -.0075 [-.0113,-.0036] (away); strength +4.5; late r2 flat; pts flat. Every persistence check moves away with whole intervals on the wrong side.
Decision: NOT adopted alone. The per-play target is correct (smoke: trailing <=120 s pass TO .071 v real .086, base .021) but adding real late-trailer turnovers widens the catch-up gap, like CTX F. Both stay candidate components for when the missing catch-up channel (E182 Q4 possessions -0.38/game and its unexplained remainder) is modelled.
