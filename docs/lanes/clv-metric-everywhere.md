# CLV metric everywhere — ENG-47

## Goal
A signed "line move toward the pick" metric (points, opener to close, positive
= market moved toward our side) available from `backtest.summarize_predictions`
whenever a frame carries opener and close spreads, and reported in the opener
evaluation summary (mean, games, season-block bootstrap interval, per-season
values) beside a trivial baseline, on the same games as the served model.

## State
- `src/nfl_ats/backtest.py` `summarize_predictions` (~line 165-198): guarded
  block computes `line_move_toward_pick_mean/_games/_pushes/_no_close` when the
  frame carries `tue_open_home_spread`, `close_home_spread` and `pick`; `None`/0
  when absent, mirroring the existing `model_margin_mae` guard. No current call
  site (`walk_forward_backtest`, `evaluation.py`, `experiments.py`,
  `outcomes.py`) passes those columns yet, so this is a reusable hook, not a
  behavior change for existing callers.
- `src/nfl_ats/clv.py` `opener_pick_evaluation` (~line 2216): adds
  `result["line_move_toward_pick"] = where(pick_home_at_open, 1, -1) *
  open_move`, same sign convention as `score_clv`'s `clv_points`. Cache version
  bumped `_OPENER_EVAL_CACHE_VERSION = "3-line-move-toward-pick"` so stale
  cached frames don't hide the new column.
- `OPENER_EVALUATION_METRIC_COLUMNS` gains `"line_move_toward_pick"`;
  `opener_evaluation_metrics` and `opener_evaluation_metric_draws` both add
  `line_move_toward_pick_mean/_games/_pushes/_no_close`, guarded on column
  presence so `test_opener_evaluation_metrics_omits_probability_rule_keys_when_columns_absent`
  (which builds a bare frame without the column) still passes. This flows
  automatically through the existing `nfl-ats opener-evaluation` CLI: week- and
  season-block bootstrap (`uncertainty.csv`) and the per-season table
  (`season_summary.csv`) already call `opener_evaluation_metrics` /
  `opener_evaluation_metric_draws`.
- Gates measured this session: `ruff format --check` clean (2 files), `ruff
  check` clean, `mypy src` 239 files clean, `pytest tests/test_backtest.py
  tests/test_clv.py -q` 72 passed.
- **DONE this session.** Ran the real command via
  `F:/Repos/nfl_py3/.tools/uv.exe run --no-sync nfl-ats opener-evaluation
  --features data/processed/game_features_weak_stack.parquet` (background,
  Bash tool's own `run_in_background`, not manual `nohup &`/`disown` — that
  combination silently orphaned the process on Windows Git Bash on the first
  attempt, empty log, no output dir; the tool-managed background run
  completed normally). Artifact `artifacts/opener_evaluation/20260923T172849Z/`
  (feature_table_sha256 confirmed matching `artifacts/active_ats_model.json`),
  1,537 games. `per_game.parquet` carries `line_move_toward_pick` confirming
  the new code path fired (an earlier, pre-code-change run
  `20260923T161403Z` lacks the column entirely — do not read numbers from
  it).
  - Served (raw residual sign) mean 0.1313 pts, season-block 95% CI
    [0.089, 0.178], week-block [0.063, 0.209], `probability_positive` 1.0 on
    the bootstrap, per-game positive fraction 0.417, positive all 6 seasons
    (2020 0.225 .. 2022 low 0.066 .. 2025 0.194), 377 pushes (zero open_move),
    0 no-close.
  - Always-home baseline: mean 0.0991 [-0.007, 0.185], negative in 2020
    (-0.152).
  - Always-favorite baseline (`direction = -sign(tue_open_home_spread)`,
    1,525 games, 12 pick'em lines excluded as undefined): mean 0.1033
    [0.047, 0.166], low in 2024/2025 (0.021, 0.022).
  - Calibrated discrete-margin probability-rule pick
    (`pick_home_at_open_probability_rule`, closest column to "the served
    calibrated side" the harness exposes; not the four-term member-combined
    probability, which lives in `board_content.py`, out of scope): mean
    0.0942, season-block [0.021, 0.169], week-block [0.020, 0.174], per-game
    positive fraction 0.424, negative in 2021 (-0.056); differs from the raw
    residual-sign pick on 314/1,537 games.
  - Scripts used (scratchpad, not committed):
    `eng47_baselines.py`, `eng47_prob_rule.py`; full stdout appended to
    `tests/scratch/opener_evaluation_eng47_run.log`.
  - ROADMAP.md row `ENG-47` added right after `ENG-46` (before `MKT-20`),
    marked done (plumbing + measurement complete), framed as read-only
    diagnosis per AGENTS.md, no signal verdict.

## Tried
- Considered adding the metric to `evaluation.SELECTION_METRICS`; out of scope
  per the task (selection metrics gate model choice, this is a grading metric)
  and not requested, so left untouched.
- Considered computing the metric fresh in `opener_evaluation_metrics` from
  `pick_home_at_open` + `open_move` instead of a stored column; stored it on
  `result` instead so it lands in `per_game.parquet` for inspection and so the
  vectorized bootstrap draw function can slice it directly.

## Next
Lane complete: code, gates, real run, baselines, ROADMAP row, and this file
are all done as of 2026-09-23 (see State). Nothing outstanding for this lane.
Any further work (e.g. deciding whether line-move-toward-pick should join
`evaluation.SELECTION_METRICS`, or wiring the four-term member-combined
probability from `board_content.py` into a comparable harness column) is a
new, separately scoped lane — do not reopen this one for it.

No commit/push happened (out of scope per the task); ROADMAP.md,
docs/lanes/clv-metric-everywhere.md, and tests/scratch/*.log/*.py are the
only files touched this session, alongside the already-existing src/ diff
from the prior session.

## Open
- None. Resolved this session: always-favorite baseline computed ad hoc from
  `open_move` + `sign(-tue_open_home_spread)` in a scratch script, not added
  as a `src/` column (not requested; it's a one-off comparison, not a
  reusable metric).
