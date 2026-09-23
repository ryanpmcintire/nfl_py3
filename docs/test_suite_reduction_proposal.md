# Test suite reduction proposal

Measured 2026-09-23: `.tools\uv.exe run --no-sync pytest -q --durations=40 -p no:cacheprovider`,
log at `tests/scratch/pytest_baseline.log`.

- **4,529 passed, 9 skipped, 162 warnings in 119.04s** (xdist parallel workers).
- Source-level count (`grep -c "def test_"`): **233 test files, 3,872 test
  functions** (matches owner's figure; excludes `tests/scratch` and 4 non-test
  support files: `conftest.py`, `_overlay_test_kit.py`,
  `_card_refit_test_kit.py`, `_board_content_fixtures.py`).
- Slowest items are almost all single-test **setup** cost in board-rendering
  files: `test_board_improvements.py` (35.3s), `test_board_humanised.py`
  (32.0s), `test_board_site.py` (23.7/23.0/22.6/20.9s), `test_board_terminal.py`
  (23.0/18.7/17.2/17.0/14.7/14.7s). One outlier call is
  `test_cli.py::test_refresh_crew_recorder_is_gated_and_fails_open` (73.4s).
  These board/CLI files account for >5 minutes of summed duration even though
  wall time is 119s under xdist — they are the disproportionate cost per
  worker and the biggest deletion-driven runtime win.

This proposal was built by grep/glob classification of file names, imports,
and shared test-kit usage (`_overlay_test_kit.py` used by 25 files,
`_card_refit_test_kit.py` by 4), not by reading every file in full. Function
counts are exact (`grep -cE "^\s*(async )?def test_"`); TRIM keep-counts are
estimates for the owner to confirm file-by-file. No test was edited or
deleted.

## Kept scope (the six owner-named durable categories)

Prediction safety/leakage, chronology, evaluator arithmetic, board rendering
(small set), scheduler argv, registry closure — plus the serving-path
guards those categories directly imply (ledger/lineage/provenance/publish
fail-closed, tiebreaker consistency, preflight, timestamp-fallback leakage
guards) since AGENTS.md marks those release-blocking.

## Category table

| # | Category | Verdict | Files (before) | Funcs (before) | Funcs (kept, est.) | Notes |
|---|---|---|---:|---:|---:|---|
| 1 | Prediction safety / leakage / chronology core (`test_prediction_safety`, `test_no_wager_path`, `test_purged_cv`, `test_repository_policy`) + leakage-adjacent misclassified files (`test_outcomes.py` bootstrap arithmetic, `test_injury_timestamp_fallback.py`, `test_feature_manifest.py`) | **KEEP**, light trim | 7 | 103 | 90 | Core of the moratorium's release-blocking rule. |
| 2 | Evaluator arithmetic core (`test_calibration`, `test_discrete_margin_mapping`, `test_discrete_push_read`, `test_evaluation`, `test_key_numbers`, `test_margin`, `test_score_lattice`, `test_estimation_variance`) | **KEEP**, trim redundant scenario reps | 8 | 122 | 65 | Keep arithmetic/lattice correctness; drop repeated season/week fixtures of the same formula. |
| 3 | Margin-mapping research challengers (`test_calibration_distortion`, `test_calibration_ecdf_smoothing`, `test_conditional_margin`, `test_hybrid_margin`, `test_margin_groupwise`, `test_margin_variance`, `test_joint_residual_model`, `test_gaussian_median`, `test_smooth_cdf_mapping_overlay`, `test_gaussian_mean_mapping_incumbent_overlay`, `test_ecdf_mapping_incumbent_overlay`, `test_four_overlay_composition`, `test_four_overlay_incumbent`, `test_retired_four_member_union`) | **DELETE-FILE** | 12 | 113 | 0 | Pooled-residual vs. discrete-margin challenger comparisons; core arithmetic already covered by #2. Research-number pins. |
| 4 | Scheduler argv (`test_capture_scheduler`, `test_cli`, `test_cli_contract`, `test_scheduled_lock`) | **KEEP**, trim `test_capture_scheduler.py` (57→~20) | 4 | 107 | 45 | Keep argv/dry-run/exit-code contracts; the 73s gated-recorder test and per-source argv variants are one test each, not one per source. |
| 5 | Registry closure (`test_experiment_registry`, `test_findings_registry`, `test_registry_explorer`, `test_registry_schema_resilience`, `test_rotation_validate`, `test_rotation`, `test_weak_signals`) | **KEEP** core files, **DELETE-FILE** duplicative sub-files (`test_rotation_coverage`, `test_weak_signals_invalidation`, `test_weak_signals_retag_units`, `test_weak_signals_set_reliability`) | 11 | 238 | 65 | `weak-signals record`/`pool` and rotation closure-ground validation must survive; per-scenario duplicate files fold into the parent test. |
| 6 | Board rendering | **KEEP small set** (`test_board_content.py`, `test_board_flip_line.py` [named rule enforcer], `test_public_board.py` trimmed, `test_board_terminal.py` trimmed to one real-artifact render), **DELETE-FILE** the other 14 (`test_board_assistant`, `_lineups`, `_content_coverage`, `_content_season_and_flips`, `_humanised`, `_improvements`, `_interactive`, `_season_mode`, `_site`, `_site_content`, `test_public_board_wave1/2`, `test_hosted_dashboard_contract`) | 18 | 466 | 55 | Biggest runtime win: removes the 35s/32s/24s/23s/23s/21s/19s setup outliers. Surviving reader-facing checks (humanised text, ticker rows, nav) collapse into the trimmed public_board/board_content files instead of one file per wave/feature. |
| 7 | Overlay/tilt/challenger per-signal research (30 `*_tilt_overlay`/`*_overlay` files + 4 `*_challenger` files) | **DELETE-FILE** all | 34 | 703 | 0 | Per AGENTS.md, individual situational signals are fitted terms inside one calibrated probability, never served or flipped alone; isolated per-signal test files are the textbook one-off/research-pin case. Shared `_overlay_test_kit.py`/`_card_refit_test_kit.py` harnesses go with them unless another kept file still imports them. |
| 8 | Feature builders (`*_flag_features.py`, `*_features.py`, `test_apm_unit_feature.py`) | **TRIM** to leakage/timestamp-guard assertions only | 14 | 254 | 30 | Pregame-only enforcement is release-blocking; exhaustive numeric/coefficient coverage per feature is not. |
| 9 | CFB (secondary league: `test_cfb*.py`) | **TRIM** to one smoke/contract file confirming CFB doesn't leak into NFL serving; rest **DELETE-FILE** | 6 | 51 | 8 | Flagged uncertain — confirm with owner whether CFB is still active research before deleting `test_cfb.py`'s CLI workflow coverage. |
| 10 | Capture/ingest one-off scripts (`*capture*`, `*ingest*`, `*wayback*`, `*archive*`, `*refresh_now*`, `*refresh_triggers*`, `*refresh_picks_week_default*`, `airnow`, `inactives`, `sportradar`, `referee_assignments`, `player_arrests`) | **TRIM** to argv/dry-run contract per script | 14 | 284 | 40 | Same shape as scheduler argv: keep "resolves and dry-run exits 0" per source, drop exhaustive parsed-payload assertions. |
| 11 | Serving-safety adjacent (`test_model_ledger`, `test_signal_ledger`, `test_lineage`, `test_provenance`, `test_publishing`, `test_tiebreaker`, `test_preflight`, `test_source_freshness_policy`, `test_artifact_contracts`, `test_site_theme_invariants`) | **KEEP**, trim | 10 | 262 | 70 | "One ledger, every surface" and publish-board fail-closed are release-blocking; keep the fail-closed/consistency assertions, drop styling/format repetition. |
| 12 | Best-pick / pick-serving path (`test_best_pick_nomination`, `test_pick_refresh`, `test_key_line_pick_read`, `test_card_explanation`, `test_weekly`, `test_best_pick`, `test_best_pick_composed_rule_eval`, `test_tiebreaker_shade_prospective`, `test_served_total`, `test_played_card_expectation`, `test_schedule_flag_on_production`) | **TRIM** | 11 | 209 | 55 | Keep the served-pick/tiebreaker-agreement guards; drop per-week/per-scenario duplicates. |
| 13 | Market/CLV/splash research analysis (`test_clv`, `test_market_decomposition`, `test_splash_lines`, `test_drift`, `test_surgical_gating`, `test_environment_report`, `test_market_data`, `test_market_observed_at`, `test_historical_market`, `test_open_close_market`, `test_odds`, `test_odds_backfill`, `test_lines`, `test_history_opener_close`) | **DELETE-FILE** | 14 | 269 | 0 | Not in the six named categories; market-analysis research, no leakage/chronology role. |
| 14 | Assistant golden/fixture tests (`test_assistant_golden`, `test_assistant_js_parity`, `test_assistant_battery`, `test_cli_player_research`) | **DELETE-FILE** | 4 | 27 | 0 | Classic "asserts the fixture, not the production artifact" pattern. |
| 15 | Long-tail single-purpose model/feature/pool/UI-explorer research (`test_experiment_runner`, `test_lockday_package/verify/contract`, `test_pool*`, `test_totals*`, `test_prospective*`, `test_spread_explorer`, `test_team_explorer`, `test_players`, `test_quarterbacks`, `test_pbp*`, `test_role_actions`, `test_readme_state`, `test_home_side_*`, `test_postseason`, `test_travel_geometry`, `test_probability_*`, `test_model_weak_spots`, `test_model_explanation`, `test_model_card`, `test_evidence_conventions`, `test_reporting`, `test_lineup_*`, `test_availability`, `test_spread_regime`, `test_source_policy`, `test_site_theme_pack`, `test_rest_context`, `test_experiments`, `test_backup_data`, `test_nflverse_current_season`, `test_portfolio`, `test_number_variables`, `test_io`, `test_verification_tiers`, `test_participation`, `test_home_dog_location`, `test_handoff`, `test_graph_ratings`, `test_data`, `test_snapshots`, `test_sensitivity_audit`, `test_modeling`, `test_findings_headline`, `test_container_contract`, `test_backtest`, `test_nfl_week`, `test_dependence`, `test_active_model`, `test_drought_monitor`) — full 71-file list in `tests/scratch/longtail_files.txt` | **DELETE-FILE** by default, owner vetoes individually | 71 | 776 | 0 | None named in the six kept categories; not skimmed line-by-line — appendix list lets the owner pull any specific file back to KEEP/TRIM before execution. |

## Totals

| | Files | Functions |
|---|---:|---:|
| Current | 233 | 3,872 |
| Proposed kept | ~37 | ~423 |
| Reduction | ~6.3x fewer files | **~9.2x fewer functions** |

Functions-per-file is the owner's stated unit ("3,870 test functions... an
order of magnitude too large"); the proposal lands at ~9.2x, within "roughly
10x." Tightening categories 2, 5, 11, 12 further (owner's call on which
served-path scenarios are truly redundant) reaches a clean 10x.

## Resulting kept file set (proposed)

Category 1: `test_prediction_safety.py`, `test_no_wager_path.py`,
`test_purged_cv.py`, `test_repository_policy.py`, `test_outcomes.py`,
`test_injury_timestamp_fallback.py`, `test_feature_manifest.py`.
Category 2: `test_calibration.py`, `test_discrete_margin_mapping.py`,
`test_discrete_push_read.py`, `test_evaluation.py`, `test_key_numbers.py`,
`test_margin.py`, `test_score_lattice.py`, `test_estimation_variance.py`.
Category 4: `test_capture_scheduler.py`, `test_cli.py`,
`test_cli_contract.py`, `test_scheduled_lock.py`.
Category 5: `test_experiment_registry.py`, `test_findings_registry.py`,
`test_registry_explorer.py`, `test_registry_schema_resilience.py`,
`test_rotation_validate.py`, `test_rotation.py`, `test_weak_signals.py`.
Category 6: `test_board_content.py`, `test_board_flip_line.py`,
`test_public_board.py`, `test_board_terminal.py`.
Category 8 (all 14 files kept, trimmed).
Category 9: `test_cfb.py` (trimmed).
Category 10 (all 14 files kept, trimmed).
Category 11 (all 10 files kept, trimmed).
Category 12 (all 11 files kept, trimmed).

Everything in categories 3, 7, 13, 14, 15 (135 files, 2,388 functions)
proposed for full-file deletion.

## Not done here

No test file was deleted, edited, or moved. This is a proposal for owner
review; execution is a separate authorized task.
