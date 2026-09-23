# Line-move regrade of legacy (pre-line-move-yardstick) registry families

## Goal
Today's positive-control-power lane (docs/lanes/positive-control-power.md, unit 2)
measured that paired line movement toward the pick resolves effects of
~0.08-0.21 line-move points (~0.27-0.70 accuracy points) at 80% power on
2020-2025, 7-18x finer than the accuracy yardstick used to grade most of the
registry. docs/lanes/done/tuesday-terms-line-move.md already re-graded 5
terms (diff_lineup_total, diff_divergence, cfb_transfer_logit,
reddit_home_comment_ratio_elevated, gated_flag_sum) on the line-move
yardstick. This lane re-grades 8 MORE Tuesday-knowable, rebuildable
registry families that were only ever graded on accuracy, using the same
pattern (scripts/tuesday_terms_line_move.py): one term added at a time to
the Tuesday-knowable base (model_logit + composition_flag_sum), LOSO
2020-2025, season-block bootstrap, accuracy companion.

## State (2026-09-23, session 1 — INCOMPLETE, hit 50-tool-call cap)
Selection is DONE and predeclared (see Open for the full ranking derivation
and caveats). New script `scripts/line_move_regrade_legacy.py` is WRITTEN
(mirrors scripts/tuesday_terms_line_move.py structure exactly: 8
`TERM_DECLARATIONS`, one builder function per term, `loso()`,
`variant_report()`, writes `artifacts/line_move_regrade_legacy/<ts>/results.json`).

**NOT YET DONE this session:**
1. `ruff check scripts/line_move_regrade_legacy.py` (WITHOUT --fix) — command
   was issued but the tool-call cap hit before output returned. Read the
   output and fix any violations (no code comments/docstrings rule already
   respected in the file as written — no `#` comment lines were added).
2. Script has NOT been run yet. Run it once in the foreground:
   `/f/Repos/nfl_py3/.venv/Scripts/python scripts/line_move_regrade_legacy.py`
   (or via `.tools/uv.exe run python scripts/line_move_regrade_legacy.py`).
   Expect it to take a comparable order of magnitude to
   scripts/tuesday_terms_line_move.py (that ran in well under a minute for 5
   terms on population n=1503; this one has 8 terms, some with heavier I/O
   — the precip term reads a 4,431-row forecast parquet, the deadline-drag
   term reads a PFR transactions snapshot + snap-count parquet, the roof
   term calls `roof_state_screen.build_prediction_table()` which itself
   rebuilds a walk-forward roof predictor — could take a few minutes; if it
   looks like it will exceed ~120s move it to background per the harness's
   normal long-command handling, do not kill it).
3. Each of the 8 terms wraps its `variant_report()` call in a bare
   `try/except Exception` in `main()` — if any builder fails (missing data
   file, column-name mismatch, join producing all-NaN, etc.) that term's
   entry in `results.json["terms"]` will be `{"label": ..., "error": "..."}`
   instead of real numbers. CHECK EVERY TERM'S OUTPUT for an `error` key
   before treating results as final; do not silently drop a failed term —
   report the failure and either fix the builder or drop that term from the
   8 (documenting why) rather than padding with a 9th untested term.
4. Once results.json exists: read it, pull each term's
   `line_move_toward_pick_cell` (mean_points, season_block_interval_low/high,
   season_block_probability_positive) and `accuracy_companion_cell` (same
   fields), and draft (or the orchestrator runs) 8
   `nfl-ats weak-signals record` commands, one per term, following the exact
   flag pattern already used in docs/lanes/done/tuesday-terms-line-move.md's
   Next section (`--effect <line-move mean_points> --effect-units
   ats_points --classification unresolved_below_power --interval-low
   --interval-high --probability-positive --sample-games --sample-blocks 6
   --category <see Open per-term category> --plain-summary "..." --source
   artifacts/line_move_regrade_legacy/<ts>/results.json`). Do NOT run the
   record commands yourself unless the orchestrator says to — this agent's
   packet said "put exact commands in the lane Next", not execute them.
   WHOLE-interval-negative-or-positive results (candidate wrong_sign_resolved)
   need the same orchestrator-adjudication treatment the reddit term got in
   the done lane (flag, do not auto-close; no split-half reliability check
   was run here either).

## Tried (research done this session, all of it still valid, do not redo)
- Read docs/lanes/positive-control-power.md and
  docs/lanes/done/tuesday-terms-line-move.md in full for the MDE numbers and
  the already-graded exclusion set (diff_lineup_total, diff_divergence,
  cfb_transfer_logit, reddit_home_comment_ratio_elevated, gated_flag_sum /
  pbp08_protection_mismatch_gated_flag_sum_fit family).
- Read scripts/tuesday_terms_line_move.py in full — this is the exact
  pattern line_move_regrade_legacy.py copies (base features, LOSO, cell_stats
  from scripts/line_move_yardstick_paired_eval.py, season-block bootstrap,
  2000 draws, seed 20260923).
- Enumerated all 258 non-null `family` values in registry/weak_signals.json
  (7,034 total signal entries) via
  `.venv/Scripts/python` one-off scripts (not saved as repo files, ran from
  the scratchpad dir C:\Users\Ryan\AppData\Local\Temp\claude\
  F--Repos-nfl-py3\3de407a1-acbe-4e06-9e02-f857ffb073d6\scratchpad\
  rank_families.py and rank_themed.py and show_selected.py — these are
  scratch, not part of the repo, safe to ignore/regenerate).
- Registry categories present: modeling(5191), market(646), onfield(349),
  control(235), health(207), schedule(180), environment(119), offfield(44),
  None(34), attention(29). Filtered to schedule/environment/health/offfield
  (plus onfield for the LOO-marginal situational-rule families, which live
  under onfield in this registry) for the "situational/schedule/travel/rest/
  weather-forecast/referee/coach/injury" theme the task named.
- Ranked distinct families by |effect|/standard_error (SE field, else
  (interval_high-interval_low)/(2*1.96)) among classification=
  unresolved_below_power, effect_units=accuracy_points entries only (so
  already-closed refuted_mechanism/positive_control entries are excluded by
  construction, e.g. coach_speak's 45x ratio is a positive control and was
  discarded; weak_stack_v4_forecast_weather's 2.75 is already
  refuted_mechanism and was discarded; officials_archive_battery's 2.07 top
  entry is also refuted_mechanism and was discarded).
- Excluded CFB-only/benchmark-transfer families (cfb_body_clock_replication,
  cfb_post_bye_home_on_benchmark, cfb_rest_bye_replication,
  cfb_altitude_cold_home_on_benchmark, cfb_served_tilts_replication, etc.) —
  out of NFL 2020-2025 scope per the task.
- Excluded referee/crew families (late_season_crews, officials_archive_battery,
  referee_assignments_crew_tilt, rookie_crew_reconciliation,
  penalty_crew_tendencies, served_refresh_card's rookie-crew entry) on the
  finding that this repo's OWN registry entry `crew_tilt_stacked_on_production`
  describes the underlying flag construction as a "Late-week officiating-crew
  tilt" — i.e. referee-crew assignment is not Tuesday-knowable in this
  repo's own established convention, so no referee family qualifies as
  Tuesday-knowable no matter its ratio.
- Excluded health-category families whose construction depends on in-week
  injury/practice-report data (follow_news_gate, expected_lineup_loss_on_card,
  nflcom_friday_refresh, unserved_tilt_marginal_on_played_card injury_value_tilt,
  injury_news_vs_level, sim03_lineup_mixture_cx21, veteran_rest_day_tell
  [explicitly Wednesday DNP], injury_trajectory_refresh_cx14,
  injury_signal_on_refresh_card, illness_on_production, per13_durability_*,
  specialist_absence_fade, inactives_channel_historical_proxy_v1 [kickoff-90],
  fluview_* families [CDC surveillance timing not verified Tuesday-safe,
  deprioritized rather than confirmed excluded — see Open]).
- Excluded ablations of already-served composition members rather than fresh
  rebuildable primitives (overlay_leave_one_out_2026_08_26,
  graph_ratings_v2_team_stat injury_defense_disruption — a complex graph-model
  pipeline output, not a simple situational primitive).
- Excluded conditional_situational_probability_v1 (top-ranked at 2.513) after
  confirming by grep that no script anywhere in the repo builds a column
  literally named `conditional_situational_probability` — its source is
  `artifacts/signal_atlas/20260921T231249832601Z/report.json`, a composite/
  pooled atlas score across many splits, not one rebuildable feature column,
  so "one added term" is ill-defined for it.
- For the remaining candidates, confirmed each has an existing, reusable
  builder already in src/nfl_ats (not re-derived from scratch):
  - tank_zone_fade_tilt_overlay.tank_zone_flag_by_game(schedule)
  - bye_edge_fade_overlay.bye_edge_flag_by_game(schedule)
  - roof_state_screen.build_prediction_table() (scripts/, walk-forward roof
    predictor, reused via sys.path exactly like tuesday_terms_line_move.py
    reuses other scripts/ modules)
  - forecast_weather_kn_precip_high_total_tilt_overlay.
    precip_high_total_flag_by_game(schedules, forecasts, total_lines) —
    forecasts from data/raw/forecast_archive/pool_decision_2009_2025/
    forecasts.parquet (confirmed columns include game_id,
    forecast_precip_prob_pct; 4,431 rows); total_lines from nflverse
    schedule's own total_line column (confirmed present via
    default_schedule()). Overlay direction confirmed by reading
    apply_precip_high_total_tilt_overlay: flag TRUE flips an away pick to
    home, i.e. it is a toward-home push, not home/away symmetric — term
    built as a plain 0/1 flag, not signed.
  - schedule_flag_features.derive_week1_dog_features /
    derive_division_dog_features(schedule, opener_lines) — opener_lines
    passed as population[["game_id","tue_open_home_spread"]] directly
    (population's own Tuesday-open price, not a separate snapshot fetch).
  - interim_hc_first_game_tilt_overlay.
    interim_first_game_flag_by_game_fail_open(repo_root) — returns
    game_id+team rows where a team is in its first game under an interim HC;
    script merges onto schedule home_team/away_team to build a signed
    +1 home / -1 away / 0 neither term (confirmed by reading
    apply_interim_hc_first_game_tilt_overlay's own home_first_game/
    away_first_game merge pattern).
  - transaction_flag_features.attach_deadline_integration_drag_features(
    features, schedule=schedule) — internally loads
    default_transactions_index() (latest data/raw/pfr_transactions/*/
    index.parquet) and default_snap_counts() itself; confirmed sign
    convention in transaction_flag_features._attach_qualifying_sides: +1.0
    when only the AWAY team qualifies (just acquired a confirmed high-snap
    player within its first 3 games), -1.0 when only home qualifies, 0
    otherwise. Wrapped in try/except DataContractError in the script (falls
    back to an all-zero term) in case the transactions/snap-count snapshot
    columns have drifted since LEAD-23 was last run — CHECK for this
    fallback firing when reading results.json (a real 0.0 nonzero-rate on
    this term likely means the fallback fired, not that the true rate is 0).
  - Confirmed nflverse schedule (default_schedule(), 4,902 rows) carries all
    needed raw columns directly: game_type, home_team, away_team, result,
    spread_line, total_line, div_game, roof, gameday, weekday, overtime,
    home_rest, away_rest, referee, home_coach, away_coach — no missing local
    data for any of the 8 builders.

## Next
1. Run `ruff check scripts/line_move_regrade_legacy.py` (no --fix) and fix
   any reported issues (expect at most import-order/unused-import nits; the
   file has zero `#` comments/docstrings by construction, matching the
   packet's "no code comments" instruction).
2. Run the script once in the foreground:
   `/f/Repos/nfl_py3/.venv/Scripts/python scripts/line_move_regrade_legacy.py`
3. Read the resulting `artifacts/line_move_regrade_legacy/<ts>/results.json`.
   For each of the 8 terms, check for an `error` key first. For terms that
   ran clean, pull `line_move_toward_pick_cell` (mean_points,
   season_block_interval_low, season_block_interval_high,
   season_block_probability_positive) and `accuracy_companion_cell` (same
   4 fields) plus `paired_games` and `term_nonzero_rate`.
4. Draft 8 `nfl-ats weak-signals record` commands (one per successful term),
   copying the exact flag set used in
   docs/lanes/done/tuesday-terms-line-move.md's Next section:
   `--name <label>_line_move_regrade_legacy --description "..." --source
   artifacts/line_move_regrade_legacy/<ts>/results.json --effect
   <line_move mean_points> --effect-units ats_points --classification
   unresolved_below_power --league nfl --season-start 2020 --season-end 2025
   --interval-low <low> --interval-high <high> --probability-positive <p+>
   --sample-games <paired_games> --sample-blocks 6 --category <see below>
   --plain-summary "..."`. Suggested `--category` per term: tank_zone_fade_tilt
   onfield, bye_edge_fade onfield, roof_state_predicted_open environment,
   precip_high_total_tilt environment, week1_dog schedule,
   interim_hc_first_game_tilt offfield, division_dog schedule,
   deadline_integration_drag offfield. Flag (do not auto-classify) any term
   whose BOTH line-move and accuracy intervals sit wholly on one side of
   zero for orchestrator adjudication (same treatment as the reddit term in
   the done lane) rather than recording it as a terminal classification.
5. Report the 8 line-move deltas with intervals and P+, plus the 8 record
   commands, to the orchestrator in one message; do not run the record
   commands without explicit authorization.
6. Move this lane to docs/lanes/done/ once the run, the read, and the
   command drafts are complete and reported.

## Open
Final 8 predeclared terms (ranked by prior |effect|/SE on accuracy_points,
registry name / ratio / category / builder used):
1. `deadline_integration_drag_on_production` 2.218 offfield —
   transaction_flag_features.attach_deadline_integration_drag_features
2. `tank_zone_fade_tilt__spread_band_short_2020_2025` 2.098 onfield —
   tank_zone_fade_tilt_overlay.tank_zone_flag_by_game
3. `bye_edge_fade__spread_band_long_2020_2025` 1.940 onfield —
   bye_edge_fade_overlay.bye_edge_flag_by_game
4. `roof_state_predicted_open_fade_on_production` 1.572 environment —
   roof_state_screen.build_prediction_table
5. `precip_high_total_tilt__week_in_season_weeks_1_4_2020_2025` 1.481 onfield —
   forecast_weather_kn_precip_high_total_tilt_overlay.precip_high_total_flag_by_game
6. `week1_dog_on_production` 1.473 schedule —
   schedule_flag_features.derive_week1_dog_features
7. `interim_hc_first_game_tilt__week_in_season_weeks_5_12_2020_2025` 1.449 offfield —
   interim_hc_first_game_tilt_overlay.interim_first_game_flag_by_game_fail_open
8. `division_dog_on_production` 1.338 schedule —
   schedule_flag_features.derive_division_dog_features

Deprioritized-but-not-selected (ranked above #8 or thematically relevant,
kept here in case any of the 8 fails and a replacement is needed):
- `dst_transition_battery` (dst_arizona_away_shield, ratio 1.412, schedule/
  travel — Arizona DST-exception travel-body-clock proxy) — no dedicated
  builder script found anywhere in the repo (only registry_explorer.py
  references the name, which just reads the registry, not a feature
  builder); would need fresh construction, not "rebuildable" in the
  low-effort sense the other 8 are. Next-best travel-themed candidate if a
  substitute is needed.
- `lead_sweep_on_card_v1` fluview_home_elevated_tilt (1.829) and
  `weak_stack_v5_fluview_away` (1.622) and `fluview_elevated_on_production`
  (0.89) — FluView (CDC influenza surveillance) families; Tuesday-safety
  not verified this session (CDC FluView reports are typically released
  Fridays with roughly a one-week reporting lag — plausibly Tuesday-knowable
  using the PRIOR week's release, but this was not confirmed against the
  actual as-of construction in weak_stack_v5_fluview_away's source before
  the cap hit). Do not use without first confirming the as-of date logic in
  whichever fluview builder module produces these columns.
- `altitude_fourth_quarter` lead43_altitude_4q_mexico_city_on_production_opener
  (1.285, environment/travel) — narrow (Mexico City games only), builder
  location not confirmed this session.
- `primetime_cells` pt_post_mnf_sunday_changepoint (1.454, schedule) — no
  dedicated builder module found near the top-level grep; likely
  constructible directly from schedule weekday/gametime/week columns without
  a helper, but not confirmed.

Every ratio above is a PRIOR from the original (mostly non-line-move,
mostly non-2020-2025-only) accuracy grading — it is used only to rank
selection order, not carried into the new measurement. The new script
re-measures each term from scratch on the served 2020-2025 population with
its own LOSO fit and bootstrap.

No results yet — the run has not happened this session. Nothing here
depends on conversation-only state: the selection rule, the ranked-family
derivation, the script, and this lane file are all on disk. A fresh agent
can resume at Next step 1 (ruff check) purely from this file.
