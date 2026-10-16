# mod25e E131 margin-SD gap decomposition (diagnosis only)

## Goal
Attribute the sim-real margin SD gap (15.38 v 14.63) on base ...kekudwtpq2ugky (s11-13, 39168 games) to quarter, possession-count, blowout and home-away covariance components. No sim run; nothing implemented.

## State (2026-10-16; measured; regulation-only points; game bootstrap 200 reps, 90%; ~60 looks, one family; scripts and outputs tests/scratch/e131/{build,an,an2}.py, an.txt, an2.txt)
- Var(M) real 221.3 v sim 235.6: +14.3 [+2.2,+24.6] P+ .97. Var(total pts) is flat: +1.1 [-9.1,+10.2]. Possessions/game 22.70 v 22.70.
- Var(H)+Var(A) +7.7 [+0.9,+15.3]; -2Cov(H,A) +6.6 [-1.7,+13.3] (corr -.071 real v -.100 sim, P(sim more negative) .885).
- Possession count: SD 3.28 real v 3.10 sim (sim narrower). Var(M) gap at the real N mix +14.6 [+3.1,+25.4]; N-mix effect +0.1 [-0.5,+0.8]. Explains none.
- Per quarter Cov(m_q,M) sim-real: Q1 +3.5, Q2 +3.0, Q3 +4.6, Q4 +3.1 (each interval spans 0, all P+ .84-.92). Per-possession points variance 8.83 v 9.00; P(0) and P(>=6) match; P(3) sim lower in Q2-Q4 (.136 v .152, .132 v .146, .096 v .104), as E118 (TD up, FG down).
- Var(M) = Var(M3) + 2Cov(M3,m4) + Var(m4): +8.7 [-3.1,+18.3] + 4.9 [-2.6,+12.3] + 0.7 [-1.7,+2.8]. So 61% accrues in Q1-Q3, 34% is weaker Q4 regression, Q4 own variance 5%.
- Blowout (margin at Q4 start): within-bin Var(M)*share, 17+ bin +15.0 [+3.3,+26.3] P .975; 8-16 +3.5; <=7 -2.1. In 17+ games (26% of both): leader Q4 pts 3.95 v 3.59 (+.36 [+.10,+.63] P .98); trailer Q4 pts var +2.4 [-.2,+5.0]; Q4 total pts +.51 [+.04,+1.0]; mean lead change same (+.2 [-.3,+.7]). Sim leaders keep scoring in garbage time; no extra trailer catch-up.
- Where it lives: score-state conditioning clips at +-24 and lumps leads 9+ (coarse) or 13+ (ipw): mod25e_sel.py:15-16, 27-41; late play-choice mod25e_desp.py:385-410 uses EDGES bands; mod25e_adj.py:311-316 only pushes leaders down; no starters-pulled / leader-clock mechanism by lead size exists.

## Tried
Possession-run tables from raw real pbp and sim play files (same definition both sides), game bootstrap of all components.

## Next
Proposed ONE mechanism, BLS (blowout leader selection): fit, from real 2009-17 pbp, the probability that a leading offense's play is drawn from the low-yield/run-heavy pool as a function of lead size in finer bins (9-12, 13-16, 17-23, 24+) x Q3/Q4 x time bucket, as a score-state reweight in the same form as SEL (mod25e_sel.py scheme_cells), cells chosen leave-one-season-out by held-out play log-likelihood (9 folds), no free scale. Out-of-sample plan: LOSO per-season fit, then ONE smoke seed (after E130 finishes) scored on: leader Q4 pts in 17+ (target 3.59), Var(m4|17+), Q4 total in 17+ (8.45), Var(M), plus noise, strength, pts/g, Q4 slope not worsening; confirm on 3 seeds. Inferred ceiling: the 17+ bin carries all of the gap, so an upper bound of about 15 of 14.3 (not claimed).

## Open
Which of leader run share, starters pulled or clock burn carries the +.36 leader pts is not split (needs play-level leader pass rate/EPA by lead size, sim v real). Q1-Q3 variance (+8.7) is unattributed and consistent with E118 TD dispersion; unresolved_below_power.

## Update 2026-10-16 (BLS unit; measured; no simulation run)
Channel split (tests/scratch/e131/ch.py, ch.txt; game bootstrap 200, 90%; real 2304 v sim 39168 games; ~230 cell-stat looks, one family). Leader Q4 pts in 17+ games: real 3.16 v sim 3.60 +.44 [+.16,+.69]; drives/game, plays/drive, sec/drive match (-.01, +.08, -.5 s), run share -.007: it is points per drive, +.16 [+.08,+.25]. Carrier is the 24+ bin: pts/drive +.27 [+.14,+.40], TD/drive +.040 [+.020,+.059] (Q4>600 +.40, Q4 300-600 +.45, Q4<300 +.21 pts/drive; 17-23 and 9-16 flat or small). Starters pulled: real backup-QB pass share 24+ = .25 overall, .20/.40/.82 by Q4 bucket (Q3 .13; 17-23 .08-.16). Not clock burn (sec/drive sim shorter by 7.6 in 24+, plays/drive same) and not run share (Q3 24+ only -5 pp). Real backup-minus-starter pass EPA at lead 17+: -.22 [-.33,-.09] (820 v 5530). Sim yards column is not comparable to real yards_gained (2.4 v 5.1); not used. Backup share in sim not measured (needs the engine idx map); the engine neighbour key clips score at +-21 (sim04_engine SCORE_CLIP) so 24+ states pull 9-21 neighbours (mod25e_sel clip 24).
BLS fit (scripts/mod25e_bls.py fit; artifacts/mod25e3/bls_fit.json, bls/isb.npy): P(backup passer | offense lead bin x Q1-2/Q3/Q4>600/Q4 300-600/Q4<300), 9 LOSO folds on 2009-17 pass rows, 16 looks (4 schemes x 4 prior strengths, one family). Chosen lead_t, prior 10: LOSO LL/row -.42277 v coarse_q -.42451 = +.00174, in-sample minus LOSO gap .0013. 24+ cells .087/.128/.200/.393/.778. No free scale: per state, backup pass rows are reweighted by p_fit/p_current and starter pass rows by (1-p_fit)/(1-p_current), p_current from the neighbour weights, so pass mass is preserved.
Hook: scripts/mod25e_bls.py install_bls, env BLS=1, outermost over pick_index_nn_conditioned (calls base first so ADJ state updates run; in applicable states Q3/Q4 with lead>=9 it resamples from the reweighted weights, bypassing the ADJ tilt for that draw), walks frames with `"offense" in fr.f_locals` and reads qtr without binding the name. Import-checked and cells() spot-checked only, install_bls NOT run (needs engine).

## Next (replaces the BLS plan above)
1. crH.py edit DEFERRED (E130 workers are spawn-started and could re-import it). After E130 finishes apply: after the `ad.install_adj()` block (crH.py:119-122) insert `import mod25e_bls as bl` / `if bl.enabled():` / `    bl.install_bls()`; in the label block after the URG2 lines (crH.py:325-328) add `if os.environ.get("BLS") == "1": label = label + "bl"`.
2. Smoke (one seed, s31, after E130): base env + BLS=1 (label ...ugkybl), `scripts/mod25e_crH.py e5 --workers 3 --seed 31`; confirm dv._G["bls_stats"] reweighted>0 in 17+ states, then score with mod25e_era.py and tests/scratch/e131/ch.py (point LAB at the smoke dir).
3. Targets: leader Q4 pts in 17+ real 3.59 (E131 definition; ch.py definition 3.16), Var(m4|17+) real, Var(M) real 221.3 (sim 235.6), noise 165 (178-181), strength 58.0, pts/g 45.21, Q4 slope -.060 not worse; also 24+ pts/drive real 1.55 and TD/drive .175, backup-share check by bucket. If smoke moves 24+ pts/drive toward real without worsening noise/strength/pts/g, run 3 seeds (11,12,13) and season-bootstrap vs base.
4. Open: rushing/receiver backups are not modelled (only passer id); sim backup share unmeasured; Q1-Q3 variance +8.7 remains unattributed (unresolved_below_power).
