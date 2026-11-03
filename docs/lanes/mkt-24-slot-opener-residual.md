# MKT-24

## Goal
Kickoff-slot opener residual: opener error by Thursday, Sunday night, Monday, early, late, holiday and international slot (ROADMAP MKT-24).

## State
Unit 1 DONE 2026-11-03 16:23 (local). scripts/mkt24_slot_residual.py -> artifacts/mkt24_slot_residual/report.json; 20 cells recorded (family mkt24_slot_opener_residual, all unresolved_below_power). Slots from schedule: gametime gap cuts (Sunday early/late/night), Thursday, Monday, Saturday, holiday (Thanksgiving, Dec 25), international (Neutral REG). 4 other-weekday games dropped. Looks: 8 slots x 2 pops + LOSO 2 pops x (LL, Brier, acc) = 22.
Opener 2020-25 (n 1500): shrunk resid pp (tau2 .0005) sun early +1.5 P+ .85, sun late +1.8 P+ .85, Sat -1.1 P+ .31, others |<1|. LOSO slot term vs served: decisive flips 77-114 (slot right-wrong), dLL -0.00088 (season P+ .21), acc -2.5 pts, 2/6 seasons+. Proxy 2011-19 (n 2230): Thu +2.9 P+ .87, SNF +2.1 P+ .81; flips 164-159, dLL +0.00043 (P+ .74), acc +0.2, 7/9 seasons+. Populations not pooled; signs disagree (Thu, Sat, SNF).

## Tried
Ridge slot dummies (C chosen inner LOSO from 0.01-10) on logit(p_open); intercept-only control in report.

## Next
Unit 2: same-sign check (Sunday early/late shared +) and Thursday-only split-half reliability across 2011-19 vs 2020-25; slot-by-week interactions only if declared first.

## Open
Data: nflverse schedule slot and location fields (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
