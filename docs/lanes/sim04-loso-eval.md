# SIM-04 held-out grade, what-if runs and dashboard

## Goal

Finish SIM-04 after the engine unit (`scripts/sim04_engine.py`, lane
`sim04-play-simulator.md`). Steps: add pregame team conditioning; grade
leave-one-season-out at the opener against the served discrete read and record
the result in the registry; run SIM-05 what-if runs; add a simulated-margin
chart to the dashboard game page, then publish and push.

## State

- Not started. It waits on the engine unit's result.
- The context packet was mapped on 2026-09-25 (read):
  - Discrete read: `DiscretePushReader.for_week` at
    `src/nfl_ats/mass_preserving_lattice.py:216`, fed from `prior_pool` (:163);
    `three_way(line, point)` is at :255. Production calls it from
    `fit_production_discrete_push_reader` (:529), which `pick_refresh.py:2029`
    calls. The opener grading loop is `clv.py:2083-2169`, with the reader built
    per week at :2105.
  - Openers and margins: `artifacts/opener_evaluation/20260925T161328Z/per_game.parquet`
    (1537 games, 2020-2025). Columns include `tue_open_home_spread`, `result`
    (home margin), `push_probability_at_open` and
    `home_cover_probability_excluding_push_at_open`. Team names are in
    `data/processed/game_features_pbp.parquet`.
  - Pregame team strength: `game_features_pbp.parquet`, with
    `{home,away}_off_epa_per_play`, `_def_epa_per_play`, pass rate, and pace
    (`_drive_seconds_per_drive`). Only prior games are used (`features.py:432`,
    `pbp.py:636`).
  - Registry: `nfl-ats weak-signals record`, parser at
    `cli_commands/registry.py:652`. Valid example: `docs/edge_audit_redteam.md:171`.
    Use `--effect-units log_loss_improvement` and `--probability-positive`.
  - Dashboard: the `GameDive` dataclass is at `board_content.py:442` and is
    filled by `_build_dive` (:2793). HTML goes next to `_game_dive_chart_html`
    in `board_terminal.py:1041`, with the slot at :1309. `publish-board` is at
    `cli_commands/publishing.py:1765`.

## Tried

- Nothing yet.

## Next

1. Team conditioning in the engine: shift play-outcome draws by pregame
   off/def EPA. Fit leave-one-season-out.
2. LOSO grade over 2020-2025 at the opener: simulator cover/push/loss vs the
   served discrete read. Report log loss and Brier, per-season deltas, a
   week-blocked CI and `probability_positive`, plus the positive control.
   Record the result.
3. SIM-05: a 4th-down policy swap and a QB-out what-if run.
4. Dashboard chart: simulated margin histogram with the line and key numbers
   marked. Then publish-board and push.

## Open

- None.
