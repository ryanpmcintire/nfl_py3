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
