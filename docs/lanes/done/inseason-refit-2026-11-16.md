# inseason-refit-2026-11-16

## Goal
Test whether refitting the served four-term stage each week on prior seasons plus the current season's settled weeks (1..w-1) beats the prior-seasons-only stage, at the opener 2021-25 and at the pool line 2026 W2-10. Parent: goal-65-2026-11-16.md.

## State
- Read: stage fit uses whole seasons only. src/nfl_ats/pick_probability_fit.py:503-516 fits the served beta on the opener-evaluation population (fitted_seasons 2020-2025 per artifacts/active_pick_probability.json); LOSO folds :535-546 hold out whole seasons (train includes later seasons); chronological arm :570-583 trains on seasons < held. No 2026 week enters the served stage.
- Read: base ridge already refits in-season. src/nfl_ats/outcomes.py:477-480 trains on regular-season rows with gameday < the target week's first gameday, so weeks 1..w-1 of the current season are already in the base fit. Base arm not run (already production behaviour).
- Declared 2026-11-16T22:42:17Z before grading: arms P (stage fit on seasons < s; identical every week of s) and I (stage refit for week w of season s on seasons < s plus season s weeks < w). Inputs: QW opener model probability (artifacts/v4_candidate/opener_QW.parquet), served flag_sum, move_toward_home, move_available (pick_probability/20261116T171555Z per_game). Scored 2021-25, 2020 training only. Opener grade, pushes reported, not counted. Reproduction: P with stored model_logit must equal chronological_home_probability 2022-25. 2026: same arms, W1-10 inputs rebuilt with production flag and move functions, target and grade at the card's pool line, scored W2-10; hook term omitted in both arms. Cells: log loss, Brier, accuracy points (season and week block). Looks: 2 arms x 2 grades = 2 comparisons. Optional recency-weight arm not run.

## Tried
- 2026-11-16T22:45Z measured (scripts/inseason_refit.py, artifacts/inseason_refit/results.json, exit 0): reproduction P vs served chronological max abs 0.0.
- Opener 2021-25 (1283 decided, 27 pushes excluded): P 709-574, I 689-594; I minus P log loss -0.0034 [-0.0090, +0.0017] PP 0.11 (week block), accuracy -1.56 pts [-3.22, +0.08] PP 0.029 (week), season block [-2.63, -0.46] PP 0.002 (5 blocks); 138 changed picks, I 59-79. I worse in 4 of 5 seasons. Weekly I model_logit coef spans -0.07 to 0.54 (unstable early-season fits).
- 2026 W2-10 at pool line (129 graded, 0 pool pushes): P 67-62, I 71-58, card 68-61; log loss +0.028 [-0.008, +0.063] PP 0.947, accuracy +3.10 [-2.73, +8.21] PP 0.824; 10 changed picks, I 7-3. Mechanism: I shrinks move coef 0.220 to 0.081 and raises flag 0.234 to 0.295 by Week 10; P's 0.62+ band hits 30-29 at mean 0.739.
- Caveats: 2026 move read through Sunday prekick, not the deadline; P agrees with the card on 74/129 sides; hook omitted in both arms.
- Recorded family inseason_stage_refit, 4 cells, unresolved_below_power; pool --effect-units accuracy_points ran clean.

## Next
- If pursued: deadline-faithful 2026 move inputs (card-fidelity lane inputs) and the recency-weight arm with weight chosen by inner walk-forward on prior seasons; a 2026-only move-coefficient refit is the specific candidate (the shift is in the move term).

## Open
- History (opener) and 2026 (pool line) point opposite ways; not a serving candidate on this evidence.
