# Outside model gap 2026-11-16

## Goal
Test whether two independent outside models disagree with the market usefully: 538 QB Elo (starting-QB adjusted) and nfelo's model line, each as model spread minus the line (home minus away, points). Screen, then at most one declared served look (nfelo only).

## Declaration (stamped 2026-11-16 21:03 UTC, before any outcome was joined)
- Family outside_model_gap_2026_11_16. Terms, home-margin points, positive favours home:
  T1 qbelo_gap = 538 QB-Elo implied home margin minus the line. Implied margin = Elo difference / 25, Elo difference recovered from qbelo_prob1 by the Elo logistic (400 log10 odds); /25 is 538's published points-per-Elo (reported, 538 methodology), not tuned.
  T2 nfelo_gap = nfelo model home margin minus the line. Model margin = nfelo_dif_base times the points-per-Elo slope measured by least squares of nfelo's own published home_line_pre_regression on nfelo_dif_base (2021+, model outputs only, no outcomes). nfelo_dif_open/close and nfelo_home_line_* blend the market and are not used.
  T3 qb_adjust = 538 QB-Elo implied margin minus 538 plain-Elo implied margin (the quarterback adjustment alone; line-free).
- Lines: close = game_features spread_line; opener = per_game tue_open_home_spread (2020-2025 only).
- Looks (6): T1,T3 x {close 2011-2022, opener 2020-2022}; T2 x {close 2011-2025, opener 2020-2025}.
- Gate per look: Pearson r of term with ATS margin vs the same line (pushes dropped from accuracy, kept in r); 200 within-season shuffles of the term (seed fixed); survivor = r at or above the null 95th percentile AND positive in both chronological season halves. Expected sign positive.
- Recorded per look: forced-pick accuracy of sign(term) minus 50 (accuracy_points), 2000-draw game bootstrap interval and probability_positive, null percentile, halves.
- A T2 survivor gets ONE served look: offset logistic on out_of_season_home_probability, opener 2020-2025, LOSO, 200-shuffle null (mod36 harness). 538 ends 2022, so no served look for T1/T3.

## State (2026-11-16 21:06 UTC)
- Done. scripts/outside_model_gap.py (screen, served); outputs artifacts/outside_model_gap/ (screen.json, served.json, all_cells.json). 9 cells recorded (family outside_model_gap_2026_11_16), pool command ran clean.
- nfelo points per Elo measured 0.0365 from 1,564 of its own published lines (no outcomes).
- Screen (r null pct, halves r, accuracy pts [95%], P+): qbelo close 2011-22 98.5, +.057/+.025, +0.67 [-1.0,+2.4] .79 SURVIVOR; qbelo opener 20-22 91.0, +0.05/+0.06, +1.42 .78; qb_adjust close 39.5, +.020/-.019, -0.32 .37; qb_adjust opener 69.5, +1.28 .76; nfelo close 2011-25 100.0, +.091/+.011, +2.43 [+0.7,+4.1] .999 SURVIVOR; nfelo opener 20-25 93.5, +.085/+.006, +3.16 [+0.4,+5.8] .986 (acc null pct 98.5).
- Served look (nfelo gap vs Tuesday opener on out_of_season_home_probability, 1,503 games, 1,313 nonzero, LOSO): log-loss -0.00058 [-0.0016,+0.0004] P+ .12, null pct 38; coef/SD +0.031 [-0.073,+0.141] P+ .70, folds 5/6 positive; changed 55, record 28-27; acc +0.07 pp .54. Not confirmed; unresolved_below_power.

## Tried
- Pushes-kept served population: hook_rows has only 661 games x 2 pool lines; per_game drops 34 opener pushes, so the look ran on per_game (deviation from packet, named).

## Next
- None required. Optional: rerun served look on build_fit_population(keep_pushes=True) with hook_rows logits.

## Open
- Both outside models are strong in early seasons and near zero after (nfelo r .091 then .011; qbelo .057 then .025): consistent with retroactive tuning on history (inferred), and the served probability already absorbs the remainder.
