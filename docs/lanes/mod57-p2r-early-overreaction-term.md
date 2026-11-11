# MOD-57 P2-R early-season over-reaction term (D2 in weeks 2-6 as a fifth fitted term)

## Goal
Test P2-Q's one lead (D2 = home-minus-away mean of margin-minus-close to date, negative slope weeks 2-6) as one fitted term inside the served four-term pick probability, LOSO 2020-25 at the opener, plus 2015-25 walk-forward power read and a frozen forward read on 2026 Weeks 2-6. Family mod57_how_wrong_is_the_line (shared; prior count 24 P2-Q cells read from the lane + 32 recorded by P2-Q per its Tried line, verify in shard). Script scripts/mod57_p2r_early_overreaction_term.py, outputs artifacts/mod57_p2r/.

## State
2026-11-11 10:44:30 DECLARED before any outcome read.
Novelty (read): the base model already carries ats_residual (point_diff minus team spread, features.py:313; STATE_METRICS constants.py:53, family "results" constants.py:361 inside FEATURE_SETS football/full) as smoothed home/away/diff state, linear, no week interaction. ats_streak_regress flag fitted at opener 2020-21: -0.66 accuracy points [-2.20,+0.89] (registry ats_streak_regress_on_production). Graph-input screen graph_input_screen_ats_residual is single-feature. The MOD-19 atlas split flags by week_in_season, not ats_residual. No LOSO-at-opener fit of D2 x early weeks found: not a duplicate.
- T = D2 (p2q surprise_values, pregame only) where week 2-6 else 0. T2 = D2 x linspace(1,0.2,5) over weeks 2..6 else 0. Primary: T as 5th term to BASE four, same ridge, LOSO 2020-25 on 1,503 games vs four-term 829-674. Controls: intercept-only; T with D2 permuted within season-week; T from weeks 7-11 (placebo). Power: walk-forward 2015-25 vs close, one-term logistic on D2 (weeks 2-6 only, trained on all earlier seasons from 2009), accuracy + log loss vs 0.5, season-block bootstrap, per week 2-6 all shown. Forward: coefficients frozen on 2020-25, 2026 W2-6, added to current structure logit in score.py harness (share 0).
- Looks: 12 + 2 + 3 = 17 plus 3 controls.

## Tried
2026-11-11 10:47:37 RUN DONE, 20 cells recorded (shard 72.8 KB to 102.3 KB, 56 to 76 signals; prior family count 56 recorded). Measured, artifacts/mod57_p2r/run.out. Truncation diff 0.0 on 10 games. Fitted T coefficient NEGATIVE in all 6 LOSO folds (-0.0172/pt mean, range -0.0201..-0.0136), matches the lead. Four-term 829-674; plus T 828-675 (139 changed picks, 69-70), ll -0.00098 P+ .78 (all games); weeks 2-6 (436 games) four-term 259-177, plus T 256-180 (91 changed, 44-47), ll gain 0.0030 P+ .76, acc -0.68 P+ .33. Taper: all 822-681, wk2-6 249-187 (60 changed 25-35), acc P+ .047. Controls: permuted, placebo, intercept all no better. Power 2015-25: 475-350 57.6% (+7.6 [+4.0,+10.6] P+ 1.00), ll gain .0055 P+ .978. Forward 2026 W2-6 77 games: current 50-27, +T 54-23 (20 changed, 12-8), ll gain .0088 P+ .89. Forward first run used mean-centering (T=0 games shifted); fixed to uncentered, numbers same within 1e-4 (first run kept as *_first_centered_forward).

## Next
Orchestrator decides: T is not served (no gain at opener beyond four-term, weeks 2-6 baseline already 59.4%). Could first act Week 2 of 2027 as an extra term only if a later LOSO beats four-term; hold for 2026 W7+ forward accrual (T zero outside W2-6).

## Open
Forward W2-6 is the same 77 games the current structure was tuned on (replay not held-out for current); T frozen on 2020-25 only.
