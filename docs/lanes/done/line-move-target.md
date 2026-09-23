# line-move-target

## Goal
MKT-20: predict close minus open from opener-time information and grade the implied pick at the opener beside the served model.

## State
- 2026-09-16: queued by owner; ROADMAP row written.
- Unit 1 DONE 2026-09-16 (subagent-measured, parent-verified by rerun).
  Script `scripts/line_move_target.py`; artifact
  `artifacts/line_move_target/20260916T192806Z/` (`summary.json`,
  `per_game.parquet`, reliability tables). Served anchor reproduced
  841-662 in-script. Two cells recorded in family `line_move_target_v1`.

## Tried
- Label std move 1.485 vs margin 12.97; zero-move MAE 0.9686. Ridge MAE
  0.9877 (IS 0.9780), gain -0.0191 [-0.0363, -0.0047], P+ 0.003;
  GBR 1.0744 (IS 0.8693, gap -0.2051, overfit) — recorded
  refuted_mechanism / wrong_sign_resolved on predictability.
- Implied pick (predicted move + opener) vs served accuracy: ridge
  -3.66 pts [-5.73, -1.76], P+ 0.000 — recorded refuted_mechanism /
  wrong_sign_resolved (SE corrected to bootstrap width after the
  validator flagged the first offer). Brier/LL losses in the artifact.
- 8 looks (2 candidates x 4 metrics, both reported, no selection).

## Next
- Closed as a target. The move stays a fitted term (already served),
  never a forecast. No follow-up.

## Open
- None.
