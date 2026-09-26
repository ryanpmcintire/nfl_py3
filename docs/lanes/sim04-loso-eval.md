# SIM-04 held-out grade, what-if runs and dashboard

## Goal

Finish SIM-04 after the engine unit (`scripts/sim04_engine.py`, lane
`sim04-play-simulator.md`). Steps: add pregame team conditioning; grade
leave-one-season-out at the opener against the served discrete read and record
the result in the registry; run SIM-05 what-if runs; add a simulated-margin
chart to the dashboard game page, then publish and push.

## State

- Grading harness built and run once: `scripts/sim04_loso.py`. Output:
  `artifacts/sim04_loso/20260926T022317Z/report.json` +
  `per_game.parquet` (1537 games, 2020-2025 REG, Tuesday openers).
- Sign convention (measured): home covers when `result >
  tue_open_home_spread` (push if equal); confirmed both by clv.py
  (`prior_pool` uses the line with no negation; `margin.py:722-724`
  `predicted_margin = spread + raw` also unflipped) and by data
  (`frac(result>line)=0.4854` near 0.5 vs `frac(result>-line)=0.5511`;
  `mean(line)=1.4115` approx `mean(result)=1.6396`).
- Baseline reproduction (measured/read): `home_cover_probability_excluding_push_at_open`,
  `push_probability_at_open`, `home_loss_probability_at_open` sum to 1.0 for
  all 1537 games and are produced by the exact production loop
  (clv.py:2105 `DiscretePushReader.for_week`, clv.py:2154-2169 calling
  `serve_discrete_three_way`, mass_preserving_lattice.py:216/:394) — reused
  as-is, not re-derived.
- `predicted_margin_at_open` (measured): not stored directly; reconstructed
  as `tue_open_home_spread + residual_at_open_served`
  (`margin.py:722-724`, `clv.py:2184`). Omits the small per-week `location`
  shift `serve_discrete_three_way` adds (mass_preserving_lattice.py:407-409),
  which isn't stored per game — noted in report.json, not hidden.
- Positive control (measured): historical-margin shape blended
  `(1-alpha)*predicted_margin + alpha*result`. alpha=0.0: log-loss delta
  -0.0119, CI [-0.0223,-0.0026], probability_positive 0.008 (no leak, no
  false signal). alpha=0.05: delta +0.0168, CI [0.0069,0.0260],
  probability_positive 0.9985 — harness detects a 5%-toward-truth blend
  cleanly at this sample; alpha>=0.1 is unambiguous (pp=1.0).
- Provisional candidate (measured, engine's league-average margin shape,
  no team conditioning, re-centered on `predicted_margin_at_open`, trained
  on seasons strictly before the graded season, 20000 sim games per of the
  6 training windows): pooled log-loss delta vs baseline -0.00773, week-blocked
  bootstrap CI [-0.01579,-0.00020], probability_positive 0.0205 (candidate
  worse on log loss, whole interval on the adverse side). Brier delta
  +0.0000520, CI [-0.00452,0.00448], probability_positive 0.5155 (no signal
  on Brier). Per-season log-loss deltas mostly negative (2020 -0.0233, 2022
  -0.0212, 2023 -0.0071, 2024 -0.0002, 2025 -0.0025) except 2021 +0.0063.
  Expected: this is the untuned league-average shape, not the team-conditioned
  engine the plan calls for.
- Not run: `nfl-ats weak-signals record`. Draft command is in
  `report.json["record_command_draft"]`; do not run it against this
  provisional (no-team-conditioning) result — wait for the team-conditioned
  candidate.
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

- Built and ran `scripts/sim04_loso.py` once (2026-09-26). Confirmed
  `scripts/sim04_engine.py` is fast enough to reuse directly: `build_tables`
  on the largest window (2009-2024, 16 seasons, 619949 transition rows) took
  3.8s, `simulate(5000,...)` took 5.8s (measured), so 6 training windows x
  20000 games ran in a few minutes total, well inside the "minutes" budget.
  Did not edit `scripts/sim04_engine.py` (another agent owns it); imported
  `build_tables`/`simulate` read-only via `sys.path` (the script's own
  directory, since `scripts/` has no `__init__.py`).

## Next

1. Team conditioning in the engine (owned by the engine's lane/agent): shift
   play-outcome draws by pregame off/def EPA. Fit leave-one-season-out.
2. Re-run `scripts/sim04_loso.py`'s `provisional_candidate_hists` (or a new
   team-conditioned candidate function following the same
   histogram-per-game interface already built) once conditioning lands, and
   compare against this run's numbers above. Then run the drafted
   `nfl-ats weak-signals record` command with the real
   `probability_positive` filled in.
3. SIM-05: a 4th-down policy swap and a QB-out what-if run.
4. Dashboard chart: simulated margin histogram with the line and key numbers
   marked. Then publish-board and push.

## Open

- None.
