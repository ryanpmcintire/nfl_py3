# MOD-09 drive-sequence model vs summary (2026-11-02)

## Goal
Row's declared step: small temporal model over each team's as-of drive sequence vs the summary
(per-game drive-outcome mean) features. Each enters the served four-term probability as ONE fitted
term, LOSO 2020-2025, graded at the opener on decisive games.

## State
- scripts/mod09_drive_sequence.py (imports sim10_drive read-only; drives from artifacts/sim06/sim10
  segments 2018-2025). Outputs artifacts/mod09/{results.json,run.log}.
- Sequence model: MLP over last K offensive + K defensive drives (net points, start yl, score diff)
  for both teams, K/hidden/L2 chosen by 3-block season CV on training seasons only. Summary: ridge on
  trailing-mean net points (K or all prior), K/alpha chosen the same way. Target: home-minus-away
  drive-attributed net points per game (all 2018-2025 labels). Term z = predicted margin minus opener.
- Runtime assert: every prior game used has strictly earlier (season, week) and gameday.
- Served base: out_of_season_tuesday_home_probability in latest artifacts/pick_probability per_game.

## Tried
Run done (215 s, 1 process; 1,499 decisive games 2020-25, no pushes). Log loss: served .69068, refit base .69109, +seq .69146, +summary .69185. Gain vs refit base: seq -0.0004 [-0.0015,+0.0008] P+ .27, summary -0.0008 [-0.0020,+0.0005] P+ .12, seq v summary +0.0004 [-0.0009,+0.0017] P+ .71; accuracy pts seq -1.53, summary -0.80. z coef right-signed all 6 folds (seq mean .0043, sum .0111). Drive-margin MSE: seq 215.3 and summary 178.5 vs mean-only 201.5 (sequence overfits). In-minus-out gain gap seq .0006, sum .0011. Recorded 3 cells (family mod09_drive_sequence), all unresolved_below_power.

## Next
Orchestrator decides MOD-09 row status; no further build justified by this unit alone.

## Open
Training-season z for the combiner uses models that exclude both the held-out and that season.
