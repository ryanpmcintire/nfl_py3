# mod25e E123 play volume deficit (diagnosis only; no sim run, no script edits)

## Goal
Name the plays/game deficit of base ...kekudwtpq2ugky (s11-13, 52224 games) v real 2009-17 REG (2304 games), by quarter and play class; specify LOSO-fittable fixes. Scratch tests/scratch/e123/ (load, tab, agg, budget, budget2, inc, ince .py; real/base2/clq1 .pkl, G.pkl).

## State (measured 2026-10-09; 90% game bootstrap both sides, 200 reps; ~250 cells looked at, families: budget decomposition, inc share by hs bin, el by class x hs bin x quarter group, inc el histogram)
- Accounting trap: sim play log has no kickoff or PAT rows and includes no-plays (code 6, 8.74/game; scripts/mod25e_clq.py CODES2, cmp3.py label "ko" was code 6 = no-play). Real 150.43 = run+pass+punt+FG+kneel+spike+kickoff 10.05, no no-plays. Like for like is real_fit (artifacts/sim09/f2/real_fit.parquet, incl no-play 9.22): 149.34 v sim 146.90 = -2.44, not -3.5.
- CLQ=2 is not the cost: v real per quarter dN Q1 -.26 [-.37,-.14], Q2 -.76 [-.91,-.59], Q3 -.22 [-.32,-.08], Q4 -1.15 [-1.36,-.95]. CLQ=1 smoke: Q1 +.66, Q3 +.78 (overshoot; run el 35.7 v 36.2, no-play el 13.1 v 16.0), Q2 -.77, Q4 -1.12 (same as CLQ=2). CLQ=2 corrects Q1/Q3 to real: mean el run 36.5/36.5 v 36.2/36.4, pass 34.5/34.4 v 34.6/34.7, no-play 16.1 v 16.0; P(el>40) run .310 v .298, pass .280 v .275, no-play .042 v .041, inc P(el>12) .054 v .044.
- Deficit sits in Q2/Q4 (-1.9 of -2.4), where CLQ2 does not act (scripts/mod25e_clq.py:571 qtr in 1,3).
- Channels, plays/game (mean-el mix + rate, centered, Q2+Q4 / Q1+Q3): inc -1.01 [mix -.49, rate -.52] / -.43 [-.33, -.10]; run -.37 / -.20; term -.36 / +.12; fg -.10 / -.02; punt +.15 / +.10; kneel +.15 / 0; no-play +.11 / -.06; pass +.05 / 0; sum -1.41 / -.49 (residual -.48 is real seconds < 3600 by first-snap offsets 7.3 s plus second order).
- Counts: passes -1.73 [-2.04,-1.32]/game, runs equal (51.91 v 51.95). Inc -1.52 [-1.71,-1.28]: Q1/Q3 -.51, Q2/Q4 hs<=120 -.44, (120,300] -.12, >300 -.45. Pass dropbacks hs<=120 Q2/Q4 -.92 [-1.04,-.76]. Inc share of nonend passes sim v real: Q1/Q3 .345 v .359; hs<=120 .357 v .385; (120,300] .344 v .357; >300 .346 v .362 (about -1.4 pts at every bin, -2.8 at hs<=120). By down 1/2/3: .3145/.3148/.3425 v .3283/.3276/.3596. 4th-down inc share sim .016 v 0 (fewer 4th-down flips).
- Inc el: Q2/Q4 hs (45,120] 6.84 v 5.54, (120,300] 7.86 v 6.12, (300,600] 7.64 v 6.67, >600 6.91 v 6.62 (Q1/Q3 matched, 6.98 v 6.77). Excess is a 20+ s tail (hs 120-420: el>20 sim 8% v real 2.7%, el>40 3.6% v 1.1%), i.e. completion-like clocks on rows whose final class is inc. Writer there is CDW (scripts/mod25e_cdw.py:177-215, chosen hz 420, w 20, used split, sign sign), keyed on class at draw time.
- No-play el Q2/Q4 hs>300: 13.6 v 16.1 (CLQ2 covers Q1/Q3 only), small, opposite sign (+.1 plays). Run Q2/Q4 hs<=45 el 14.1 v 11.2 (tail .031 v .006).
- Yard-shift lead refuted: DV crzhk yard_gain 0.0, ybias 0.0 (read, mod25d_variance DV, mod25e_crG.py:36); mod25e_scorestate.py:297-304 keeps b==0 passes at 0; sim nonint yards 0.24% of pass rows, |y|<1 nonint 8e-5. Misclassification < .01 plays/game. ck.klass (mod25e_clk.py:38-43) is not the carrier.

## Tried
Class tables per quarter real v sim v CLQ=1; seconds budget (real el has glitch rows, Q4 el==gsr mid-game, excluded); mix/rate decomposition; inc share and el by hs bin; el histograms; yard-shift trace.

## Next (specs, not written)
1. Inc el: stub the pol chain at Q2 hs 200 with class inc (E122 stub pattern), find the module that leaves el>20 on inc rows (suspects: a later writer or class change after CDW draw); fix by drawing the clock after the final class or CLK inc cells (E122 spec 2). Score: LOSO log-lik of el bin for inc rows, nested over hs edges; check P(el>20|inc) .027, mean 6.1.
2. Inc share: fit P(inc | nonend pass) with terms down, dist, hs bin, sd sign, qtr, LOSO log-lik v base row; first locate the loss (policy pass share .4838 v .4887 at 1st down, kNN condition tilt nn_weight_cache_cond, 4th-down flips). Extends E122 URG2(b) (hs<=120 only) to the whole game.
3. Q2/Q4 CLQ analogue (no-play el, hs<=45 run edge) via per-class censored cells, LOSO.
Re-check with budget.py after any run.

## Open
Real el sums per quarter are not exactly 900 (glitch rows); decomposition residual -.48 unexplained beyond that. Bootstrap CIs on el tables not computed. Total dN CI not computed (per quarter only).
