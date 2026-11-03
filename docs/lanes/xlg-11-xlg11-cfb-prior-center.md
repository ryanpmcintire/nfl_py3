# XLG-11 College-centered ridge prior

## Goal
Test the college coefficients as a prior mean for the NFL margin ridge. ROADMAP row XLG-11.

## State
Unit 1 run 2026-11-03: `scripts/xlg11_cfb_centered.py`, report `artifacts/xlg11_cfb_centered/report.json`.
Family xlg11_cfb_centered_ridge, declared before outcomes: 2 populations (NFL 2020-25 Tuesday opener,
NFL 2011-19 SBR proxy) x 3 metrics (accuracy, log loss, Brier) = 6 looks; per-season cells descriptive.
Design: 14 shared features; base = zero-centered ridge alpha 10; arm = penalise (beta - beta_CFB), beta_CFB =
ridge alpha 10 on CFB games with gameday before the NFL training window end, in the NFL imputer/scaler space,
scaled by the measured ratio of NFL to CFB ats_margin sd. Strength from {1,10,100,1000,10000} by inner
leave-one-season-out SSE on NFL training seasons only, refit every week (walk-forward).

## Tried
Measured (6 looks, recorded 8701-8706, unresolved_below_power). Opener 2020-25 n=1503 decisive: base 786-717, arm 827-676, flips W-L 128-87; accuracy +2.73 pts [0.81,4.58] P+ 0.999; log loss gain +0.00075 [-0.00139,0.00276] P+ 0.77; Brier +0.00038 P+ 0.77. Proxy 2011-19 n=1747: base 939-808, arm 947-800, flips 113-105; accuracy +0.46 [-1.16,2.21] P+ 0.69; log loss +0.0016 [-0.00072,0.00405] P+ 0.91; Brier +0.00069 P+ 0.89. Per-season accuracy change positive in 9 of 14 seasons (2016, 2017, 2024 negative; 2015 -1.2).
Inner LOSO strength: 1000 for 2012-14, then 10000 (grid top) every season 2015-25: the arm is essentially the CFB coefficients with NFL intercept refit, so it is a transfer of the CFB fit, not a blend.

Unit 2 (scripts/xlg11_cfb_centered_u2.py, artifacts/xlg11_cfb_centered/report_u2.json, 15 cells *_u2 recorded). Audit: CFB fit uses games before train-window end (u2.py arms(): past = gameday < train max), sd ratio and NFL imputer/scaler computed in-window, 0 of 227 weeks violate; unit 1 reproduced exactly. Grid to 1e9: u1 picks 1e4 interior from 2015. Free in-window scale: median multiple of the ratio falls 1.0 (2013) to 0 (2023-25); opener 814 wins: +1.86 [0.13,3.56] P+ .982, flips 108-80, ll P+ .87; proxy +0.40 P+ .68. Vs served 88-input (794-709 on 1503): base14 -0.53 P+ .36, u1 +2.20 [-0.87,5.39] P+ .91 (flips 326-293), free +1.33 P+ .81; log loss P+ .97-.98 for all three incl. base14, so the ll edge is the Gaussian-vs-served probability, not CFB.

## Next
Control: very large strength with scale 0 (follow-the-line) to test if the gain is line-following, not CFB info. Declared, not run: CFB-centered prior on the 14-shared block inside the full 88-input ridge (same grids, inner LOSO), graded against served.

## Open
Scale shrinking to 0 as NFL data grows suggests the CFB prior fades; u1 gain may be inflated by a fixed ratio.
