# pol-13-sunday-renomination-rule

## Goal
Decide, out of season, whether the Sunday Best Pick re-nomination (served since 2026-09-11) beats holding the Tuesday star, and whether a spread screen or an uncertainty gate helps. Measurement only.

## State
- Q1 read: the 57.3% vs 52.9% is S3 57.28% (59/103) vs T0 52.88% (55/104) in docs/best_pick_sunday_renomination.md, script scripts/best_pick_sunday_renomination_eval.py, artifact artifacts/best_pick_sunday_renomination/20260911T021349Z/. Frozen 2020-2025 opener archive, all seasons pooled, no leave-one-season-out, rule chosen after seeing 2026; P+ 0.814, interval [-5.83, +15.53].
- Q2 measured 2023-2025 (54 weeks; only seasons where Sunday information is in the fit): script tests/scratch/pol13/pol13.py, weekly rows tests/scratch/pol13/weekly.csv, input artifacts/pick_probability/20261108T171427Z/per_game.parquet. Graded at the Tuesday opener line, Sunday-served side.
  a hold 27/54; b current 34/54 (+12.96 pts, [-1.85,+27.78], P+ 0.951, 13-6 on 39 differing weeks); c screened 32/54 (+9.26, [-5.56,+24.07], 0.889, 11-6); d gated 31/54 (+7.41, [-5.56,+20.37], 0.858, 9-5).
  Per season a/b/c/d: 2023 10/12/13/11, 2024 8/9/9/7, 2025 9/13/10/13 (of 18).
- Forward 2026 W1-W9 (ledger reconstruction, not the live decisions): a 5/9, b 4/9, c 5/9, d 4/9. b differs from the actual ledger in W1.
- Recorded: family pol13_sunday_renomination_rule, 6 cells, all unresolved_below_power.

## Tried
- d gate = held star's delta-method SE from a logistic calibration fit on strictly earlier seasons (no tuned constant).
- Spread screen only (abs opener line <= 6.5); the cross-book dispersion half-screen is not in per_game, so c omits it.

## Next
- Re-run c with the dispersion screen from data/market/raw if wanted.
- Looks: 3 contrasts x 2 populations = 6, plus the 3 per-season splits unrecorded.

## Open
- 2023 move coefficient is 0, so Sunday and Tuesday differ little that year; Sunday information is mostly 2024-25.
- Playable approximated by game day (Sun/Mon), not kickoff times.
