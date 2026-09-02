"""Command-line entry point for the NFL ATS research pipeline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import pandas as pd

from nfl_ats import __version__
from nfl_ats.active_model import activate_matching_ats_model
from nfl_ats.anytime import (
    ANYTIME_METRICS,
    DEFAULT_ALPHA,
    DEFAULT_TARGET_GAMES,
    anytime_summary,
    paired_anytime_comparisons,
)
from nfl_ats.availability import (
    AVAILABILITY_COMBINATION_PRIOR,
    AVAILABILITY_POSITION_PRIOR,
    AVAILABILITY_RATE_VERSION,
    build_availability_outcomes,
    build_season_lagged_availability_rates,
    score_availability_rates,
    summarize_availability_scores,
)
from nfl_ats.backtest import score_week, walk_forward_backtest
from nfl_ats.backup_qb_fade_overlay import record_backup_qb_fade_challenger_decisions
from nfl_ats.best_pick_big_spread_challenger import (
    record_big_spread_nomination_challenger_decisions,
)
from nfl_ats.best_pick_nomination import (
    record_nomination_challenger_decisions,
    record_nomination_v3_challenger_decisions,
)
from nfl_ats.board_site import build_site
from nfl_ats.bye_edge_fade_overlay import record_bye_edge_fade_challenger_decisions
from nfl_ats.calibration import RESIDUAL_SMOOTHING_METHODS
from nfl_ats.cfb import (
    cfb_source_spec,
    fetch_cfb_snapshot,
    plan_cfb_ingest,
    summarize_cfb_snapshots,
)
from nfl_ats.cfb_audit import (
    CFB_AUDIT_BOOTSTRAP_SAMPLES,
    CFB_AUDIT_REPLICAS,
    CFB_AUDIT_SEED,
    run_cfb_sensitivity_audit,
)
from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_BOOTSTRAP_SAMPLES,
    CFB_BENCHMARK_BOOTSTRAP_SEED,
    CFB_BENCHMARK_CALIBRATION,
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_REGRESSOR,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    CFB_BENCHMARK_TARGET,
    cfb_benchmark_uncertainty,
    cfb_walk_forward_benchmark,
)
from nfl_ats.cfb_features import (
    CFB_FEATURE_VERSION,
    build_cfb_game_features,
    load_cfb_benchmark_inputs,
    load_cfb_seasons,
)
from nfl_ats.cfb_role_features import (
    CFB_ROLE_FEATURE_COLUMNS,
    CONTINUITY_NEUTRAL,
    absence_separation_study,
    attach_role_continuity,
    build_role_continuity,
    cfb_role_benchmark,
)
from nfl_ats.cfb_roles import (
    CFB_ROLE_PBP_LOAD_COLUMNS,
    FROZEN_ROLE_SEASONS,
    cfb_role_actions,
    run_role_replication,
    summarize_absences,
)
from nfl_ats.clv import (
    FROZEN_PILOT_PROTOCOL,
    ClosePredictionUnavailable,
    PilotProtocolBlocked,
    build_pairing_table,
    close_reference_table,
    clv_summary,
    live_close_reference,
    load_paper_decisions,
    opener_evaluation_metrics,
    opener_pick_evaluation,
    predict_close_for_week,
    record_paper_decisions,
    resolve_active_model_config,
    run_predeclared_pilot,
    score_clv,
    score_paper_ledger,
    sign_test_pilot_b,
    upcoming_week,
    week_blocked_bootstrap,
)
from nfl_ats.coach_fade_overlay import record_overlay_challenger_decisions
from nfl_ats.constants import (
    DEFAULT_MIN_TRAIN_GAMES,
    DEFAULT_OFFSEASON_RETENTION,
    FEATURE_SETS,
)
from nfl_ats.crew_tilt_refresh_overlay import record_crew_tilt_refresh_overlay
from nfl_ats.data import DataContractError, check_nflverse_contract, fetch_nflverse
from nfl_ats.dependence import prediction_dependence_audit
from nfl_ats.division_revenge_tilt_overlay import record_division_revenge_tilt_challenger_decisions
from nfl_ats.drift import (
    build_drift_report,
    write_drift_artifacts,
)
from nfl_ats.ecdf_mapping_incumbent_overlay import (
    record_ecdf_mapping_incumbent_challenger_decisions,
)
from nfl_ats.era_weighted_half_life_8_overlay import (
    record_era_weighted_half_life_8_challenger_decisions,
)
from nfl_ats.evaluation import (
    DEFAULT_EVALUATION_CANDIDATES,
    SELECTION_METRICS,
    format_candidates,
    nested_walk_forward_evaluation,
    parse_candidates,
)
from nfl_ats.experiment_runner import run_experiment_cli
from nfl_ats.experiments import (
    DEFAULT_EXPERIMENT_SETS,
    DEFAULT_PLAYER_PROFILE_SETS,
    FROZEN_AVAILABILITY_MIN_TRAIN_GAMES,
    FROZEN_AVAILABILITY_PROFILE,
    FROZEN_AVAILABILITY_RIDGE_ALPHA,
    FROZEN_AVAILABILITY_START_SEASON,
    FROZEN_PARTICIPATION_BASELINE_PROFILE,
    FROZEN_PARTICIPATION_CANDIDATE_PROFILE,
    FROZEN_PARTICIPATION_MIN_TRAIN_GAMES,
    FROZEN_PARTICIPATION_RIDGE_ALPHA,
    FROZEN_PARTICIPATION_START_SEASON,
    FROZEN_PLAYER_CALIBRATIONS,
    FROZEN_PLAYER_EVALUATION_START_SEASON,
    FROZEN_PLAYER_FIRST_TEST_SEASON,
    FROZEN_PLAYER_MIN_CALIBRATION_GAMES,
    FROZEN_PLAYER_MIN_TRAIN_GAMES,
    FROZEN_PLAYER_MODEL_PROFILES,
    FROZEN_PLAYER_RAW_START_SEASON,
    FROZEN_PLAYER_RIDGE_ALPHAS,
    FROZEN_PLAYER_VALIDATION_SEASONS,
    nested_outcome_profile_selection,
    paired_feature_comparisons,
    player_model_candidate_id,
    run_feature_set_experiment,
    run_frozen_player_model_selection,
    run_outcome_profile_experiment,
)
from nfl_ats.features import build_game_features
from nfl_ats.forecast_cold_visitor_tilt_overlay import (
    record_forecast_cold_visitor_tilt_challenger_decisions,
)
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (
    record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions,
)
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (
    fetch_shared_kickoff_nearest_forecasts_fail_open,
    record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions,
)
from nfl_ats.four_overlay_incumbent import record_former_production_incumbent_decisions
from nfl_ats.handoff import check_session_handoff, write_session_handoff
from nfl_ats.historical_market import fetch_historical_market_snapshot
from nfl_ats.inactives_refresh_overlay import record_inactives_refresh_overlay
from nfl_ats.injury_signal_refresh_tilt import record_injury_signal_refresh_tilt
from nfl_ats.injury_value_tilt_overlay import record_injury_value_tilt_challenger_decisions
from nfl_ats.interim_hc_first_game_tilt_overlay import (
    record_interim_hc_first_game_tilt_challenger_decisions,
)
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, atomic_text, run_id
from nfl_ats.key_numbers import (
    DEFAULT_KEY_NUMBERS,
    cover_reliability_by_line_bucket,
    summarize_key_number_calibration,
)
from nfl_ats.lines import (
    PUSH_RULES,
    build_ats_pool_card_at_lines,
    load_lines_file,
    pool_card_at_lines_markdown,
    rescore_at_lines,
)
from nfl_ats.margin import DEFAULT_LINE_SWEEP_OFFSETS, MARGIN_FEATURE_PROFILES
from nfl_ats.margin_variance import cfb_variance_benchmark
from nfl_ats.market_data import (
    attach_nflverse_game_ids,
    fetch_odds_api_from_environment,
    load_quote_history,
    parse_odds_api_response,
    spread_consensus,
    tuesday_opener_quotes,
    write_market_snapshot,
)
from nfl_ats.market_decomposition import (
    DEFAULT_END_SEASON,
    DEFAULT_NOISE_SHARE_THRESHOLD,
    DEFAULT_OVERPRICED_RATIO_THRESHOLD,
    DEFAULT_RIDGE_ALPHA,
    DEFAULT_START_SEASON,
    attribute_predictions,
    decomposition_feature_columns,
    latest_open_close_games_path,
    market_decomposition_markdown,
    run_market_decomposition,
)
from nfl_ats.model_card import build_model_card, model_card_markdown
from nfl_ats.modeling import MODEL_NAMES, logistic_coefficients, model_metadata
from nfl_ats.nflcom_refresh_overlay import record_nflcom_refresh_overlay
from nfl_ats.odds_backfill import (
    DECISION_LABELS,
    DEFAULT_QUOTA_FLOOR,
    DEFAULT_SLEEP_SECONDS,
    HISTORICAL_CAPTURE_KIND,
    execute_backfill,
    plan_backfill,
    summarize_backfill_plan,
)
from nfl_ats.open_close_market import fetch_open_close_snapshot
from nfl_ats.outcomes import (
    MARGIN_DISTRIBUTION_METHODS,
    OUTCOME_METHODS,
    fit_margin_models_for_week,
    outcome_bootstrap_intervals,
    score_outcome_week,
    score_outcome_week_line_sweep,
    walk_forward_key_number_mass,
    walk_forward_outcomes,
)
from nfl_ats.pace_mismatch_dog_tilt_overlay import (
    record_pace_mismatch_dog_tilt_challenger_decisions,
)
from nfl_ats.participation import (
    PARTICIPATION_RATING_EPA_CLIP,
    PARTICIPATION_RATING_LOOKBACK_SEASONS,
    PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
    PARTICIPATION_RATING_RIDGE_ALPHA,
    PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    build_season_lagged_player_ratings,
    fetch_participation_snapshot,
    latest_participation_snapshot,
    load_participation_snapshot,
    participation_snapshot_from_root,
)
from nfl_ats.pbp import (
    PBP_FEATURE_VERSION,
    enrich_with_pbp_features,
    fetch_pbp_snapshot,
    latest_pbp_snapshot,
    load_pbp_snapshot,
)
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (
    record_pbp08_protection_mismatch_tilt_challenger_decisions,
)
from nfl_ats.pick_refresh import (
    append_refresh_to_card,
    plan_refresh,
    record_plan,
    refresh_summary,
)
from nfl_ats.players import (
    PLAYER_AVAILABILITY_FEATURE_VERSION,
    PLAYER_FEATURE_VERSION,
    PLAYER_PARTICIPATION_FEATURE_VERSION,
    attach_snap_player_ids,
    canonicalize_injuries,
    canonicalize_rosters,
    canonicalize_snaps,
    enrich_with_player_features,
    fetch_player_snapshot,
    fetch_player_value_snapshot,
    latest_player_snapshot,
    latest_player_value_snapshot,
    load_player_snapshot,
    load_player_value_snapshot,
    player_snapshot_from_root,
    player_value_snapshot_from_root,
)
from nfl_ats.pool import (
    build_ats_pool_card,
    build_straight_up_pool_card,
    pool_card_markdown,
    straight_up_pool_markdown,
)
from nfl_ats.portfolio import simulate_bankroll_paths, simulate_paper_bankroll
from nfl_ats.prediction_safety import (
    validate_outcome_prediction_card,
    validate_prediction_card,
)
from nfl_ats.prospective import (
    freeze_forecast,
    record_movement_rule_composed_challenger_decisions,
    record_nflcom_refresh_out2_starters_challenger_decisions,
)
from nfl_ats.prospective_scoring import (
    active_challenger_ids,
    find_challenger,
    find_challenger_artifact,
    load_challenger_decisions,
    prospective_accuracy,
    prospective_accuracy_metrics,
    prospective_week_summary,
    record_challenger_decisions,
    settle_prospective_picks,
)
from nfl_ats.provenance import (
    artifact_provenance,
    sha256_file,
    verify_experiment_links,
    write_experiment_artifact,
)
from nfl_ats.publishing import publish_active_predictions
from nfl_ats.quarterbacks import (
    depth_snapshot_from_root,
    enrich_with_qb_features,
    fetch_depth_snapshot,
    latest_depth_snapshot,
    load_depth_snapshot,
)
from nfl_ats.reporting import block_bootstrap_intervals
from nfl_ats.role_actions import (
    RoleActionsSnapshot,
    fetch_role_actions_snapshot,
    latest_role_actions_snapshot,
    load_role_actions_snapshot,
    role_actions_snapshot_from_root,
)
from nfl_ats.rotation import (
    GRADE_POOLS,
    MAX_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    MINED_SEASONS,
    STRATIFIED_GRADE,
    VERDICTS,
    Registry,
    assign_stratified_window,
    assign_window,
    declare_family,
    default_registry_path,
    load_registry,
    record_look,
    registry_status,
    save_registry,
)
from nfl_ats.snapshots import (
    Snapshot,
    describe_snapshot,
    latest_snapshot,
    load_snapshot,
    snapshot_from_root,
)
from nfl_ats.special_teams_return_tilt_overlay import (
    record_special_teams_return_tilt_challenger_decisions,
)
from nfl_ats.spread_gap_zone_fade_overlay import (
    record_spread_gap_zone_fade_challenger_decisions,
)
from nfl_ats.surface_switch_tilt_overlay import record_surface_switch_tilt_challenger_decisions
from nfl_ats.tank_zone_fade_tilt_overlay import record_tank_zone_fade_tilt_challenger_decisions
from nfl_ats.third_down_reversion_fade_overlay import (
    record_third_down_reversion_fade_challenger_decisions,
)
from nfl_ats.tiebreaker import format_report as format_tiebreaker_report
from nfl_ats.tiebreaker import tiebreaker_report
from nfl_ats.totals import format_results as format_totals_results
from nfl_ats.totals import run_backtest as run_totals_backtest
from nfl_ats.turnover_luck_rebound_tilt_overlay import (
    record_turnover_luck_rebound_tilt_challenger_decisions,
)
from nfl_ats.weak_signals import (
    CATEGORIES as WEAK_SIGNAL_CATEGORIES,
)
from nfl_ats.weak_signals import (
    CLASSIFICATIONS,
    EFFECT_UNITS,
    LEAGUES,
    WeakSignal,
    combination_report,
    family_overlap_warnings,
    record_signal,
    retag_effect_units,
    set_reliability,
)
from nfl_ats.weak_signals import (
    CLOSING_GROUNDS as WEAK_SIGNAL_CLOSING_GROUNDS,
)
from nfl_ats.weak_signals import (
    coherence_problems as weak_signal_coherence_problems,
)
from nfl_ats.weak_signals import (
    default_registry_path as weak_signal_registry_path,
)
from nfl_ats.weak_signals import (
    load_registry as load_weak_signals,
)
from nfl_ats.weak_signals import (
    save_registry as save_weak_signals,
)
from nfl_ats.weekly import run_weekly

# Every ACTIVE_PROSPECTIVE registry entry whose documented recording command is
# ``nfl-ats publish-predictions --record-decisions`` must have one result
# channel here.  ``tests/test_cli.py`` compares this map to the live registry,
# so registering a challenger without wiring (or retiring one without updating
# the command surface) fails before lock day instead of silently losing a
# season of prospective evidence.
PUBLISH_CHALLENGER_RESULT_KEYS: dict[str, str] = {
    "hc_year_one_fade_overlay": "overlay_challenger_ledger",
    "bye_edge_fade_overlay": "bye_edge_fade_challenger_ledger",
    "best_pick_nomination_v2": "nomination_challenger_ledger",
    "best_pick_nomination_v3": "nomination_v3_challenger_ledger",
    "best_pick_big_spread_eligibility": "big_spread_nomination_challenger_ledger",
    "injury_value_lost_tilt_overlay": "injury_value_tilt_challenger_ledger",
    "division_revenge_tilt_overlay": "division_revenge_tilt_challenger_ledger",
    "surface_switch_tilt_overlay": "surface_switch_tilt_challenger_ledger",
    "spread_gap_zone_fade_overlay": "spread_gap_zone_fade_challenger_ledger",
    "overlay_production_chain_coach_arrest_incumbent": ("four_overlay_incumbent_challenger_ledger"),
    "ecdf_mapping_incumbent": "ecdf_mapping_incumbent_challenger_ledger",
    "era_weighted_half_life_8": "era_weighted_half_life_8_challenger_ledger",
    "forecast_cold_visitor_tilt": "forecast_cold_visitor_tilt_challenger_ledger",
    "interim_hc_first_game_tilt_overlay": "interim_hc_first_game_tilt_challenger_ledger",
    "forecast_weather_kn_warm_team_cold_late_tilt": (
        "forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"
    ),
    "forecast_weather_kn_precip_high_total_tilt": (
        "forecast_weather_kn_precip_high_total_tilt_challenger_ledger"
    ),
    "movement_rule_composed_v1": "movement_rule_composed_challenger_ledger",
    "nflcom_friday_refresh_out2_starters_v1": "nflcom_refresh_out2_starters_challenger_ledger",
    "pbp08_protection_mismatch_tilt_overlay": ("pbp08_protection_mismatch_tilt_challenger_ledger"),
    "tank_zone_fade_tilt_overlay": "tank_zone_fade_tilt_challenger_ledger",
    "third_down_reversion_fade_overlay": ("third_down_reversion_fade_challenger_ledger"),
    "turnover_luck_rebound_tilt_overlay": ("turnover_luck_rebound_tilt_challenger_ledger"),
    "special_teams_return_tilt_overlay": ("special_teams_return_tilt_challenger_ledger"),
    "pace_mismatch_dog_tilt_overlay": "pace_mismatch_dog_tilt_challenger_ledger",
}


def _data_root() -> Path:
    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


def _artifacts_root() -> Path:
    return Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts"))


def _registry_root() -> Path:
    return Path(os.environ.get("NFL_ATS_REGISTRY_DIR", "registry"))


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_features(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. Run `nfl-ats build-features` first."
        )
    return pd.read_parquet(path)


def _cmd_doctor(_: argparse.Namespace) -> None:
    import nflreadpy
    import sklearn

    data_root = _data_root()
    payload: dict[str, Any] = {
        "nfl_ats_version": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "nflreadpy": getattr(nflreadpy, "__version__", "unknown"),
        "scikit_learn": sklearn.__version__,
        "data_root": str(data_root.resolve()),
        "artifacts_root": str(_artifacts_root().resolve()),
    }
    try:
        payload["latest_snapshot"] = describe_snapshot(latest_snapshot(data_root / "raw"))
    except FileNotFoundError:
        payload["latest_snapshot"] = None
    _print_json(payload)


def _repo_root_on_path() -> None:
    """Make ``scripts.*`` importable however this process was launched.

    ``scripts`` is not part of the installed package, so it resolves only when
    the repository root happens to be on ``sys.path``. ``python -m nfl_ats``
    puts the working directory there and the console script does NOT, so
    ``nfl-ats ingest-player-arrests`` raised ``ModuleNotFoundError: No module
    named 'scripts'`` while ``python -m nfl_ats ingest-player-arrests``
    succeeded from the same directory.

    That is a lock-day abort, not a cosmetic difference.
    ``nfl_ats.weekly._cli_runner`` dispatches every step IN-PROCESS, so
    ``weekly-run`` step 7 (``ingest-player-arrests``, fail-closed) inherits
    whatever ``sys.path`` launched it -- and the documented Tuesday command in
    ``docs/week1_readiness.md`` is the console script. Left alone, the real
    2026-09-08 run would have aborted before publishing anything.

    Resolved from this file's own location rather than the working directory,
    so it holds no matter where the command is invoked from.
    """

    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _cmd_ingest_player_arrests(args: argparse.Namespace) -> None:
    """Refresh the production arrest snapshot through the audited ingester."""

    _repo_root_on_path()
    from scripts.ingest_player_arrests import (
        DEFAULT_DELAY_SECONDS,
        PlayerArrestsIngestError,
        ingest,
        new_snapshot_dir,
    )

    delay = args.delay_seconds if args.delay_seconds is not None else DEFAULT_DELAY_SECONDS
    if args.max_pages is not None and args.max_pages < 1:
        raise ValueError("--max-pages must be >= 1")
    if delay < 0:
        raise ValueError("--delay-seconds must be >= 0")
    snapshot_dir = new_snapshot_dir(_data_root() / "raw" / "player_arrests", args.snapshot)
    try:
        manifest = ingest(
            snapshot_dir,
            max_pages=args.max_pages,
            delay_seconds=delay,
        )
    except PlayerArrestsIngestError as error:
        raise ValueError(str(error)) from error
    _print_json(manifest)


def _site_directory(destination: Path) -> Path:
    """The directory a public-site flag points at.

    ``--destination``/``--board-destination`` historically named the single
    board FILE (``docs/index.html``); the site is now three pages, so a path
    that looks like a file is reduced to its parent directory. That keeps every
    existing invocation working while ``--site-destination docs`` says what is
    actually meant.
    """

    return destination.parent if destination.suffix else destination


def _write_public_site(destination: Path) -> dict[str, Any]:
    """Write the real ATS Terminal site (:func:`nfl_ats.board_site.build_site`)
    to ``destination``'s directory.

    2026-08-31 full-site conversion: this used to call
    ``public_board.build_public_site`` (a single skin, one file per the old
    seven-entry ``SITE_PAGES``, written flat into ``directory``). It briefly
    called a two-skin ``build_two_skin_site`` (a ``terminal/``/``desk/``
    directory split behind a top-level redirect) before the owner dropped
    the Cover Desk skin entirely. It now calls
    :func:`~nfl_ats.board_site.build_site`, which returns exactly THREE
    pages -- ``"index.html"``, ``"model.html"``, ``"findings.html"`` -- each
    a bare, site-root relative path, same flat layout as the original
    single-skin site. Nothing else about this function's contract (loaders,
    guards, fail-open behavior -- all owned by
    ``build_site``/``board_site_content.load_site_content``) changed.
    """

    directory = _site_directory(destination)
    pages = build_site(_artifacts_root(), require_fresh_arrest_overlay=True)
    written = []
    for relative_path, html in pages.items():
        path = directory / relative_path
        atomic_text(html, path)
        written.append(str(path))
    nojekyll = directory / ".nojekyll"
    if not nojekyll.is_file():
        atomic_text("", nojekyll)
    return {
        "site_destination": str(directory),
        "pages_written": written,
        # Retained for callers that parsed the single-page output -- the
        # redirect page still lives at exactly this path.
        "board_destination": str(directory / "index.html"),
        "nojekyll": str(nojekyll),
    }


def _cmd_publish_predictions(args: argparse.Namespace) -> None:
    publish_instant = datetime.now(UTC)
    result = publish_active_predictions(
        _artifacts_root(),
        destination=args.destination,
        readme_path=args.readme,
        data_root=_data_root(),
        published_at=publish_instant,
        registry_root=_registry_root(),
    )
    if args.with_board:
        # Default-on since 2026-08-19: the public site is THE dashboard, and a
        # publish that skips regeneration is how docs/ served picks that
        # disagreed with the published card (owner-observed: the site showed
        # the pre-overlay BAL pick and the v1 ARI Best Pick for hours). A
        # rehearsal publish that must not touch docs/ passes --no-board.
        # Fail-open like the ledger recorders below: a site-build failure must
        # stay visible in the result but never un-publish the card.
        try:
            result.update(_write_public_site(args.site_destination or args.board_destination))
        except (ValueError, FileNotFoundError) as error:
            result["public_site"] = {"written": False, "error": str(error)}
    if args.record_decisions:
        # MKT-04 routine wiring: every published card's pre-kickoff picks are
        # appended to the paper-decision CLV ledger. A failure here must stay
        # visible but not un-publish the files already written above.
        # ``record_paper_decisions`` itself refuses to write when this week's
        # earliest kickoff is more than RECORDING_LOCK_WINDOW away, so passing
        # this flag on a rehearsal run still does not reach the ledger.
        try:
            result["clv_ledger"] = record_paper_decisions(
                _artifacts_root(),
                data_root=_data_root(),
                now=publish_instant,
                require_fresh_arrest_overlay=True,
            )
        except (ValueError, FileNotFoundError) as error:
            result["clv_ledger"] = {"recorded": 0, "error": str(error)}
        # The year-1-coach fade overlay arm (PER-07,
        # docs/coach_fade_overlay.md): the paper ledger stores both the raw
        # model side and the final played policy side; this appends the
        # coach-only arm for every game to the SEPARATE prospective ledger.
        # ``prospective-score`` derives the raw-model control from the frozen
        # ``model_pick_side`` column. A failure here must not un-publish the
        # card either.
        try:
            result["overlay_challenger_ledger"] = record_overlay_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["overlay_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # POL-09's v2 Best Pick nomination rule (nfl_ats.best_pick_nomination):
        # v1's nomination is already in clv_ledger above (unchanged, via the
        # active model's own is_best_pick flag); this appends v2's weekly
        # nominee to the SEPARATE prospective challenger ledger, so the
        # season scores v1 against v2 weekly. A failure here must not
        # un-publish the card either.
        try:
            result["nomination_challenger_ledger"] = record_nomination_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["nomination_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # POL-09's v3 Best Pick nomination challenger (docs/best_pick_ranker.md
        # "v3 audit", 2026-08-19): SIDE-LEDGER-ONLY, never read by publishing.py
        # and never touches NOMINATION_V2_ENABLED or is_best_pick. Same filter
        # and primary ranking as v2, but ties break on game_id alone (no
        # dispersion tie-break) -- the historical head-to-head against v2 leaned
        # positive (P+ 0.631) but traced to a single diverging week of 103, so
        # this accrues independent 2026 evidence rather than resting on that.
        # A failure here must not un-publish the card either.
        try:
            result["nomination_v3_challenger_ledger"] = record_nomination_v3_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["nomination_v3_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # Prospective-only 10+ point spread eligibility screen for Best Pick
        # (docs/best_pick_big_spread_challenger.md). It composes with v2 and
        # records one alternative nominee; publishing.py never imports it, so
        # the played/published Best Pick remains untouched. A failure here
        # must not un-publish the card either.
        try:
            result["big_spread_nomination_challenger_ledger"] = (
                record_big_spread_nomination_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["big_spread_nomination_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Injury value-lost tilt overlay (docs/injury_value_lost_tilt_overlay.md):
        # a parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["injury_value_tilt_challenger_ledger"] = (
                record_injury_value_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["injury_value_tilt_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # Division-revenge tilt overlay (docs/division_revenge_tilt_overlay.md):
        # a parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["division_revenge_tilt_challenger_ledger"] = (
                record_division_revenge_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["division_revenge_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Backup-QB fade overlay (docs/backup_qb_fade_overlay.md): a
        # parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["backup_qb_fade_challenger_ledger"] = record_backup_qb_fade_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["backup_qb_fade_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # Surface-switch tilt overlay (docs/surface_switch_tilt_overlay.md): a
        # parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["surface_switch_tilt_challenger_ledger"] = (
                record_surface_switch_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["surface_switch_tilt_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        # Spread-gap-zone fade overlay (docs/spread_gap_zone_fade_overlay.md):
        # a parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. A failure here must not
        # un-publish the card either.
        try:
            result["spread_gap_zone_fade_challenger_ledger"] = (
                record_spread_gap_zone_fade_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["spread_gap_zone_fade_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # The six 2026-09-01 parameter-free overlays are prospective-only.
        # Each records its own forced-pick arm and cannot affect the published
        # card; preserve an individual failure in the result without undoing
        # the publish that already completed above.
        try:
            result["bye_edge_fade_challenger_ledger"] = record_bye_edge_fade_challenger_decisions(
                _artifacts_root(), _data_root()
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["bye_edge_fade_challenger_ledger"] = {"recorded": 0, "error": str(error)}
        try:
            result["tank_zone_fade_tilt_challenger_ledger"] = (
                record_tank_zone_fade_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["tank_zone_fade_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["third_down_reversion_fade_challenger_ledger"] = (
                record_third_down_reversion_fade_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["third_down_reversion_fade_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["turnover_luck_rebound_tilt_challenger_ledger"] = (
                record_turnover_luck_rebound_tilt_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["turnover_luck_rebound_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["special_teams_return_tilt_challenger_ledger"] = (
                record_special_teams_return_tilt_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["special_teams_return_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        try:
            result["pace_mismatch_dog_tilt_challenger_ledger"] = (
                record_pace_mismatch_dog_tilt_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["pace_mismatch_dog_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # PBP-08 protection-mismatch tilt (docs/pbp08_matchup_screen.md): back
        # the defense when one side's offense carries a top-quartile four-game
        # pressure-allowed window against a top-quartile pressure-generating
        # defense. The strongest mined mean-edge cell in the project (+0.336
        # points, both blockings excluding zero, mirror controls clean), wired
        # as a dual-tracked challenger only -- never applied to the published
        # card, and costing no rotation-registry window. The flag build is
        # FAIL-OPEN (absent snapshot -> zero flags), but this outer try/except
        # still guards every other failure mode so a failure here must not
        # un-publish the card either.
        try:
            result["pbp08_protection_mismatch_tilt_challenger_ledger"] = (
                record_pbp08_protection_mismatch_tilt_challenger_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["pbp08_protection_mismatch_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Paired incumbent for the four-member production policy: record the
        # exact former coach->arrests chain frozen by the primary ledger.
        try:
            result["four_overlay_incumbent_challenger_ledger"] = (
                record_former_production_incumbent_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["four_overlay_incumbent_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # ECDF-mapping-incumbent overlay (docs/smooth_cdf_mapping.md, MOD-08
        # promotion, 2026-08-19): the published card's own probability read
        # IS now the Gaussian mapping (score_outcome_week's promoted
        # default); this challenger tracks the FORMER production ECDF read
        # off the SAME out-of-time residual sample, dual-tracked against the
        # active model in the SEPARATE prospective challenger ledger only --
        # it is never applied to the published card. Supersedes the retired
        # smooth_cdf_mapping challenger (artifacts/prospective/challengers.json),
        # which tracked the mapping in the opposite direction while the
        # published card was still ECDF-native. A failure here must not
        # un-publish the card either.
        try:
            result["ecdf_mapping_incumbent_challenger_ledger"] = (
                record_ecdf_mapping_incumbent_challenger_decisions(_artifacts_root(), _data_root())
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["ecdf_mapping_incumbent_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Era-weighted (half-life 8) challenger (docs/era_weighting_screen.md,
        # MOD-14): refits the active recipe weekly with exponential
        # season-decay sample weights, dual-tracked against the active model
        # in the SEPARATE prospective challenger ledger only -- it is never
        # applied to the published card. A failure here must not un-publish
        # the card either.
        try:
            result["era_weighted_half_life_8_challenger_ledger"] = (
                record_era_weighted_half_life_8_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["era_weighted_half_life_8_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Forecast cold-visitor tilt overlay (docs/forecast_weather_screen.md,
        # ENV-01): a parameter-free pick-level nudge using the live
        # Tuesday-noon GFS-MOS forecast, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. The live forecast fetch is
        # FAIL-OPEN (network/station-mapping failures fold into zero flags,
        # never an exception), but this outer try/except still guards
        # against every other failure mode (registry/fingerprint/ledger
        # errors) so a failure here must not un-publish the card either.
        try:
            result["forecast_cold_visitor_tilt_challenger_ledger"] = (
                record_forecast_cold_visitor_tilt_challenger_decisions(
                    _artifacts_root(), _data_root(), _registry_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["forecast_cold_visitor_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Interim head-coach first-game tilt overlay (docs/interim_coach_screen.md):
        # a parameter-free pick-level nudge, dual-tracked against the active
        # model in the SEPARATE prospective challenger ledger only -- it is
        # never applied to the published card. The interim-coach join is
        # FAIL-OPEN (missing/unavailable source data folds into zero flags,
        # never an exception), but this outer try/except still guards against
        # every other failure mode (registry/fingerprint/ledger errors) so a
        # failure here must not un-publish the card either.
        try:
            result["interim_hc_first_game_tilt_challenger_ledger"] = (
                record_interim_hc_first_game_tilt_challenger_decisions(
                    _artifacts_root(), _data_root()
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["interim_hc_first_game_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Shared live kickoff-nearest GFS-MOS fetch (docs/forecast_weather_screen.md,
        # "Wiring recommendations": "one fetch, several consumers") for the two
        # challengers below -- fetched ONCE here and passed to both via their
        # forecasts= parameter, rather than each making its own outbound
        # network call for the identical (game, station, cutoff) set. Never
        # raises (see its own docstring); None just means each recorder falls
        # back to fetching for itself.
        shared_kn_forecasts = fetch_shared_kickoff_nearest_forecasts_fail_open(
            _artifacts_root(), _data_root(), _registry_root()
        )
        # Forecast (kickoff-nearest) warm-team-cold-late tilt overlay
        # (docs/forecast_weather_screen.md, highest-EV wiring recommendation
        # after the archive's 2009-2025 fetch completed -- both windows'
        # registered intervals exclude zero): a parameter-free pick-level
        # nudge using a LIVE kickoff-nearest GFS-MOS forecast, dual-tracked
        # against the active model in the SEPARATE prospective challenger
        # ledger only -- it is never applied to the published card. The live
        # forecast fetch is FAIL-OPEN, but this outer try/except still guards
        # against every other failure mode so a failure here must not
        # un-publish the card either.
        try:
            result["forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"] = (
                record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    _registry_root(),
                    forecasts=shared_kn_forecasts,
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Forecast (kickoff-nearest) precip-high-total tilt overlay
        # (docs/forecast_weather_screen.md): a parameter-free pick-level
        # nudge sharing the SAME live kickoff-nearest GFS-MOS fetch as the
        # warm-team-cold-late challenger above (one fetch, several
        # consumers), dual-tracked against the active model in the SEPARATE
        # prospective challenger ledger only -- it is never applied to the
        # published card. The live forecast fetch is FAIL-OPEN, but this
        # outer try/except still guards against every other failure mode so
        # a failure here must not un-publish the card either.
        try:
            result["forecast_weather_kn_precip_high_total_tilt_challenger_ledger"] = (
                record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
                    _artifacts_root(),
                    _data_root(),
                    _registry_root(),
                    forecasts=shared_kn_forecasts,
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["forecast_weather_kn_precip_high_total_tilt_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # Movement-rule-on-composed-chain challenger (2026-08-22 registration,
        # docs/movement_composition_eval.md): flips the PLAYED chain pick to the
        # market side whenever the latest captured line moved >=1.0 pt off the
        # frozen Tuesday line, reusing nfl_ats.pick_refresh's own read-only
        # captured-line read. Dual-tracked only -- never applied to the
        # published card. Runs after record_paper_decisions above, whose rows
        # are its base card. A failure here must not un-publish the card.
        try:
            result["movement_rule_composed_challenger_ledger"] = (
                record_movement_rule_composed_challenger_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["movement_rule_composed_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
        # NFL.com Friday out>=2-starters refresh fade on the chain (2026-08-22
        # registration, docs/nflcom_friday_refresh.md frozen rule text):
        # freshness-gated injury-page flags flip the PLAYED chain pick,
        # dual-tracked only. FAIL-OPEN by design: absent/stale inputs come back
        # as a skipped week, and any other failure is caught here so it must
        # not un-publish the card either.
        try:
            result["nflcom_refresh_out2_starters_challenger_ledger"] = (
                record_nflcom_refresh_out2_starters_challenger_decisions(
                    _artifacts_root(), _data_root(), now=publish_instant
                )
            )
        except (ValueError, FileNotFoundError, DataContractError) as error:
            result["nflcom_refresh_out2_starters_challenger_ledger"] = {
                "recorded": 0,
                "error": str(error),
            }
    else:
        # Safe by default: an ordinary publish does not touch the ledger.
        # Recording is a deliberate act (--record-decisions), because an
        # ordinary command silently reaching the real ledger during
        # rehearsal/testing is exactly how it was contaminated on 2026-08-18
        # (docs/prospective_evidence.md, "Known divergence").
        result["clv_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append this card's picks to the "
            "paper-decision ledger",
        }
        result["overlay_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the overlay's picks to the "
            "prospective challenger ledger",
        }
        result["nomination_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the v2 Best Pick nomination to "
            "the prospective challenger ledger",
        }
        result["nomination_v3_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the v3 Best Pick nomination to "
            "the prospective challenger ledger",
        }
        result["big_spread_nomination_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the big-spread-screened "
            "Best Pick nomination to the prospective challenger ledger",
        }
        result["injury_value_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the injury value-lost tilt's "
            "picks to the prospective challenger ledger",
        }
        result["division_revenge_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the division-revenge tilt's "
            "picks to the prospective challenger ledger",
        }
        result["backup_qb_fade_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the backup-QB fade's picks to "
            "the prospective challenger ledger",
        }
        result["surface_switch_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the surface-switch tilt's "
            "picks to the prospective challenger ledger",
        }
        result["spread_gap_zone_fade_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the spread-gap-zone fade's "
            "picks to the prospective challenger ledger",
        }
        result["bye_edge_fade_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the bye-edge fade's picks to the "
            "prospective challenger ledger",
        }
        result["tank_zone_fade_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the tank-zone fade tilt's picks to the "
            "prospective challenger ledger",
        }
        result["third_down_reversion_fade_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the third-down reversion fade's "
            "picks to the prospective challenger ledger",
        }
        result["turnover_luck_rebound_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the turnover-luck rebound tilt's "
            "picks to the prospective challenger ledger",
        }
        result["special_teams_return_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the special-teams return tilt's "
            "picks to the prospective challenger ledger",
        }
        result["pace_mismatch_dog_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the pace-mismatch dog tilt's picks to the "
            "prospective challenger ledger",
        }
        result["pbp08_protection_mismatch_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the PBP-08 protection-mismatch "
            "tilt's picks to the prospective challenger ledger",
        }
        result["four_overlay_incumbent_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the former coach-to-arrests "
            "incumbent's picks to the prospective challenger ledger",
        }
        result["ecdf_mapping_incumbent_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the ECDF-mapping-incumbent "
            "overlay's picks to the prospective challenger ledger",
        }
        result["era_weighted_half_life_8_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the era-weighted (half-life 8) "
            "refit's picks to the prospective challenger ledger",
        }
        result["forecast_cold_visitor_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the forecast cold-visitor "
            "tilt's picks to the prospective challenger ledger",
        }
        result["interim_hc_first_game_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the interim head-coach "
            "first-game tilt's picks to the prospective challenger ledger",
        }
        result["forecast_weather_kn_warm_team_cold_late_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the forecast (kickoff-nearest) "
            "warm-team-cold-late tilt's picks to the prospective challenger ledger",
        }
        result["forecast_weather_kn_precip_high_total_tilt_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the forecast (kickoff-nearest) "
            "precip-high-total tilt's picks to the prospective challenger ledger",
        }
        result["movement_rule_composed_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the movement-rule-on-chain "
            "challenger's picks to the prospective challenger ledger",
        }
        result["nflcom_refresh_out2_starters_challenger_ledger"] = {
            "recorded": 0,
            "skipped": True,
            "reason": "pass --record-decisions to append the NFL.com Friday out>=2-starters "
            "refresh fade's picks to the prospective challenger ledger",
        }
    _print_json(result)


def _cmd_publish_board(args: argparse.Namespace) -> None:
    _print_json(_write_public_site(args.site_destination or args.destination))


def _cmd_refresh_picks(args: argparse.Namespace) -> None:
    plan = plan_refresh(
        _artifacts_root(),
        _data_root(),
        season=args.season,
        week=args.week,
        features_path=args.features,
        min_train_games=args.min_train_games,
    )
    result = refresh_summary(plan, record_decisions=args.record_decisions)
    result["ledger"] = record_plan(
        _artifacts_root(), plan, note=args.note, record_decisions=args.record_decisions
    )
    result["injury_signal_refresh_tilt"] = record_injury_signal_refresh_tilt(
        _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
    )
    # NFL.com Friday out>=2-starters fade (docs/nflcom_friday_refresh.md frozen
    # rule), CHALLENGER-TRACKED at refresh time: computes the WOULD-BE pick on
    # top of the post-market-follow played side and records it to a SEPARATE
    # append-only ledger. It can never alter the played pick -- plan is
    # consumed read-only -- and every absent-input path inside the recorder is
    # a documented no-op; this guard additionally keeps any unexpected failure
    # here from breaking the production refresh pass.
    try:
        result["nflcom_refresh_out2_starters_overlay"] = record_nflcom_refresh_overlay(
            _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
        )
    except (ValueError, FileNotFoundError, DataContractError) as error:
        result["nflcom_refresh_out2_starters_overlay"] = {
            "recorded": 0,
            "error": str(error),
        }
    # Official T-90 inactives are a prospective challenger only. This recorder
    # consumes `plan` read-only and writes its separate ledger; it cannot alter
    # the refresh ledger, published card, or played pick.
    try:
        result["inactives_refresh_overlay"] = record_inactives_refresh_overlay(
            _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
        )
    except (ValueError, FileNotFoundError, DataContractError) as error:
        result["inactives_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    # Officiating crews are published after Tuesday, so this prospective arm
    # belongs to each late refresh rather than the Tuesday publish. It consumes
    # the plan read-only and writes only its own ledger; unexpected recorder
    # failures remain visible but cannot break a production refresh or card
    # append.
    try:
        result["crew_tilt_refresh_overlay"] = record_crew_tilt_refresh_overlay(
            _artifacts_root(),
            _data_root(),
            plan,
            repo_root=Path.cwd(),
            record_decisions=args.record_decisions,
        )
    except (ValueError, FileNotFoundError, DataContractError) as error:
        result["crew_tilt_refresh_overlay"] = {"recorded": 0, "error": str(error)}
    if args.publish_card:
        if not plan.changed_games:
            result["card"] = {
                "written": False,
                "reason": "no eligible picks changed; nothing to append",
            }
        else:
            try:
                append_refresh_to_card(args.destination, plan, note=args.note)
                result["card"] = {"written": True, "destination": str(args.destination)}
            except (ValueError, FileNotFoundError) as error:
                result["card"] = {"written": False, "error": str(error)}
    _print_json(result)


def _cmd_handoff(args: argparse.Namespace) -> None:
    if args.check:
        result = check_session_handoff(
            Path.cwd(), _artifacts_root(), args.destination, registry_root=_registry_root()
        )
    else:
        result = write_session_handoff(
            Path.cwd(),
            _artifacts_root(),
            args.destination,
            registry_root=_registry_root(),
        )
    _print_json(result)


def _cmd_ingest(args: argparse.Namespace) -> None:
    seasons = list(range(args.start_season, args.end_season + 1))
    if args.end_season < args.start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    stats_end_season = args.stats_end_season or args.end_season
    if stats_end_season < args.start_season or stats_end_season > args.end_season:
        raise ValueError("stats-end-season must be within the requested schedule seasons")
    team_stat_seasons = list(range(args.start_season, stats_end_season + 1))
    snapshot = fetch_nflverse(
        seasons,
        _data_root() / "raw",
        team_stat_seasons=team_stat_seasons,
    )
    _print_json(describe_snapshot(snapshot))


def _cmd_smoke_source(args: argparse.Namespace) -> None:
    _print_json(check_nflverse_contract(args.schedule_season, args.stats_season))


_INCLUDE_POSTSEASON_HELP = (
    "also store postseason rows (WC/DIV/CON/SB, spelled POST in the play-by-play "
    "and player-stat feeds). Off by default: every feature build re-filters to the "
    "regular season on read, so this only widens what a future snapshot can serve. "
    "Pass it when building snapshots intended for playoff-game predictions."
)


def _cmd_pbp_ingest(args: argparse.Namespace) -> None:
    if args.end_season < args.start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    snapshot = fetch_pbp_snapshot(
        list(range(args.start_season, args.end_season + 1)),
        _data_root() / "pbp" / "raw",
        include_postseason=args.include_postseason,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "seasons": list(snapshot.seasons),
            "rows": manifest["rows"],
            "include_postseason": manifest["include_postseason"],
            "filter_version": manifest["filter_version"],
        }
    )


def _cmd_depth_ingest(args: argparse.Namespace) -> None:
    if args.end_season < args.start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    snapshot = fetch_depth_snapshot(
        list(range(args.start_season, args.end_season + 1)),
        _data_root() / "quarterbacks" / "depth" / "raw",
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "rows": manifest["rows"],
            "teams": manifest["teams"],
            "first_observation": manifest["first_observation"],
            "last_observation": manifest["last_observation"],
            "contract_version": manifest["contract_version"],
        }
    )


def _cmd_player_ingest(args: argparse.Namespace) -> None:
    ranges = {
        "injury": (args.injury_start_season, args.injury_end_season),
        "roster": (args.roster_start_season, args.roster_end_season),
        "snap": (args.snap_start_season, args.snap_end_season),
    }
    for label, (start, end) in ranges.items():
        if end < start:
            raise ValueError(f"{label}-end-season cannot be earlier than {label}-start-season")
    snapshot = fetch_player_snapshot(
        list(range(args.injury_start_season, args.injury_end_season + 1)),
        list(range(args.roster_start_season, args.roster_end_season + 1)),
        list(range(args.snap_start_season, args.snap_end_season + 1)),
        _data_root() / "players" / "raw",
        include_postseason=args.include_postseason,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "contract_version": manifest["contract_version"],
            "include_postseason": manifest["include_postseason"],
            "files": manifest["files"],
            "availability_contract": manifest["availability_contract"],
        }
    )


def _cmd_player_value_ingest(args: argparse.Namespace) -> None:
    if args.end_season < args.start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    snapshot = fetch_player_value_snapshot(
        list(range(args.start_season, args.end_season + 1)),
        _data_root() / "players" / "values" / "raw",
        include_postseason=args.include_postseason,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "contract_version": manifest["contract_version"],
            "include_postseason": manifest["include_postseason"],
            "file": manifest["file"],
            "availability_contract": manifest["availability_contract"],
        }
    )


def _cmd_role_actions_fetch(args: argparse.Namespace) -> None:
    snapshot = fetch_role_actions_snapshot(
        _data_root() / "players" / "role_actions" / "raw",
        args.seasons,
        include_postseason=args.include_postseason,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "seasons": manifest["seasons"],
            "include_postseason": manifest["include_postseason"],
            "rows": manifest["rows"],
            "sha256": manifest["sha256"],
            "source": manifest["source"],
        }
    )


def _cmd_participation_ingest(args: argparse.Namespace) -> None:
    if args.end_season < args.start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    snapshot = fetch_participation_snapshot(
        list(range(args.start_season, args.end_season + 1)),
        _data_root() / "players" / "participation" / "raw",
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "contract_version": manifest["contract_version"],
            "seasons": manifest["seasons"],
            "rows": manifest["rows"],
            "partitions": manifest["partitions"],
            "availability_contract": manifest["availability_contract"],
        }
    )


def _cmd_cfb_ingest(args: argparse.Namespace) -> None:
    spec = cfb_source_spec(args.source)
    start_season = args.start_season or spec.default_start_season
    if args.end_season < start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    seasons = list(range(start_season, args.end_season + 1))
    if args.dry_run:
        _print_json(plan_cfb_ingest(args.source, seasons))
        return
    snapshot = fetch_cfb_snapshot(args.source, seasons, _data_root() / "cfb")
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "cfb_source": manifest["cfb_source"],
            "contract_version": manifest["contract_version"],
            "seasons": manifest["seasons"],
            "rows": manifest["rows"],
            "partitions": manifest["partitions"],
            "source": manifest["source"],
        }
    )


def _cmd_cfb_summary(_: argparse.Namespace) -> None:
    _print_json(summarize_cfb_snapshots(_data_root() / "cfb"))


def _cmd_cfb_build_features(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    schedules, lines, pbp = load_cfb_benchmark_inputs(
        _data_root() / "cfb", args.start_season, args.end_season
    )
    features, audit = build_cfb_game_features(
        schedules,
        lines,
        pbp,
        start_season=args.start_season,
        end_season=args.end_season,
        span=args.ewm_span,
        min_periods=args.min_periods,
        offseason_retention=args.offseason_retention,
    )
    destination = _data_root() / "processed" / "cfb_game_features.parquet"
    atomic_parquet(features, destination)
    metadata = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "cfb_feature_version": CFB_FEATURE_VERSION,
        "start_season": args.start_season,
        "end_season": args.end_season,
        "ewm_span": args.ewm_span,
        "min_periods": args.min_periods,
        "offseason_retention": args.offseason_retention,
        "rows": len(features),
        "audit": audit,
        "destination": str(destination),
        "timing": {"total_seconds": perf_counter() - command_started},
    }
    atomic_json(metadata, destination.with_name("cfb_game_features.manifest.json"))
    _print_json(metadata)


def _cmd_cfb_benchmark(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    modeling_started = perf_counter()
    result = cfb_walk_forward_benchmark(
        features,
        start_season=args.start_season,
        end_season=args.end_season,
        min_train_games=args.min_train_games,
        ridge_alpha=args.ridge_alpha,
    )
    modeling_seconds = perf_counter() - modeling_started
    output = _artifacts_root() / "cfb_benchmark" / run_id()
    atomic_parquet(result.predictions, output / "predictions.parquet")
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    uncertainty_started = perf_counter()
    uncertainty = cfb_benchmark_uncertainty(
        result.predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    uncertainty_seconds = perf_counter() - uncertainty_started
    atomic_csv(uncertainty, output / "uncertainty.csv")
    configuration = {
        "command": "cfb-benchmark",
        "league": "cfb_only",
        "target": CFB_BENCHMARK_TARGET,
        "regressor": CFB_BENCHMARK_REGRESSOR,
        "ridge_alpha": args.ridge_alpha,
        "calibration_method": CFB_BENCHMARK_CALIBRATION,
        "start_season": args.start_season,
        "end_season": args.end_season,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    clean = result.summary.loc[result.summary["evaluation_window"].eq("clean_core")]
    headline = {
        str(row["method"]): {
            "cover_games": int(row["cover_games"]),
            "cover_accuracy": float(row["cover_accuracy"]),
            "cover_brier_score": float(row["cover_brier_score"]),
            "margin_mae": float(row["margin_mae"]),
        }
        for _, row in clean.iterrows()
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "clean_core_headline": headline,
        "timing": {
            "modeling_seconds": modeling_seconds,
            "uncertainty_seconds": uncertainty_seconds,
            "total_seconds": perf_counter() - command_started,
        },
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="cfb-benchmark",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_cfb_sensitivity_audit(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    benchmark_predictions = pd.read_parquet(args.benchmark_predictions)
    result = run_cfb_sensitivity_audit(
        features,
        benchmark_predictions,
        replicas=args.replicas,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = _artifacts_root() / "cfb_sensitivity_audits" / run_id()
    atomic_csv(result.details, output / "replica_results.csv")
    atomic_csv(result.summary, output / "summary.csv")
    configuration = {
        "command": "cfb-sensitivity-audit",
        "features": str(args.features),
        "benchmark_predictions": str(args.benchmark_predictions),
        "replicas": args.replicas,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **result.metadata,
        "benchmark_predictions": str(args.benchmark_predictions),
        "benchmark_predictions_sha256": sha256_file(args.benchmark_predictions),
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="cfb-sensitivity-audit",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})
    print(result.summary.to_string(index=False))


def _resolve_role_actions_snapshot(identifier: str | None) -> RoleActionsSnapshot:
    raw_root = _data_root() / "players" / "role_actions" / "raw"
    return (
        role_actions_snapshot_from_root(raw_root / identifier)
        if identifier
        else latest_role_actions_snapshot(raw_root)
    )


def _cmd_cfb_role_replication(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    seasons = list(range(FROZEN_ROLE_SEASONS[0], FROZEN_ROLE_SEASONS[1] + 1))
    cfb_pbp = load_cfb_seasons(
        _data_root() / "cfb", "pbp", seasons, columns=list(CFB_ROLE_PBP_LOAD_COLUMNS)
    )
    canonical_games = pd.read_parquet(args.cfb_features)
    role_snapshot = _resolve_role_actions_snapshot(args.role_actions_snapshot)
    nfl_role_stats = load_role_actions_snapshot(role_snapshot)

    result = run_role_replication(cfb_pbp, canonical_games, nfl_role_stats)

    output = _artifacts_root() / "cfb_role_experiments" / run_id()
    delivery_summary = pd.concat(
        [
            pd.DataFrame(result["cfb_summary"]).assign(league="cfb"),
            pd.DataFrame(result["nfl_summary"]).assign(league="nfl"),
        ],
        ignore_index=True,
    )
    atomic_csv(delivery_summary, output / "delivery_summary.csv")
    absence_summary = summarize_absences(result["cfb_absences"], result["nfl_absences"])
    atomic_csv(absence_summary, output / "absence_summary.csv")
    atomic_parquet(result["cfb_delivery"], output / "cfb_delivery.parquet")
    atomic_parquet(result["nfl_delivery"], output / "nfl_delivery.parquet")
    atomic_parquet(result["cfb_absences"], output / "cfb_absences.parquet")
    atomic_parquet(result["nfl_absences"], output / "nfl_absences.parquet")

    configuration = {
        "command": "cfb-role-replication",
        **result["configuration"],
        "cfb_features": str(args.cfb_features),
        "role_actions_snapshot": role_snapshot.snapshot_id,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": "cfb-role-replication",
        "configuration": configuration,
        "gates": result["gates"],
        "cfb_summary": result["cfb_summary"],
        "nfl_summary": result["nfl_summary"],
        "coverage": result["coverage"],
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.cfb_features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="cfb-role-replication",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _load_cfb_role_inputs(
    cfb_features_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CFB actions/team-games plus the pbp team-id map (shared loader).

    The team-id map exists because pbp names teams by display name
    ("Minnesota Golden Gophers") while the canonical table uses schedule
    names ("Minnesota"): joining continuity onto games must go through
    ESPN team ids, never names.
    """

    seasons = list(range(FROZEN_ROLE_SEASONS[0], FROZEN_ROLE_SEASONS[1] + 1))
    cfb_pbp = load_cfb_seasons(
        _data_root() / "cfb", "pbp", seasons, columns=[*CFB_ROLE_PBP_LOAD_COLUMNS, "pos_team_id"]
    )
    canonical_games = pd.read_parquet(cfb_features_path)
    actions, team_games, _ = cfb_role_actions(cfb_pbp, canonical_games)
    team_ids = (
        cfb_pbp.loc[
            cfb_pbp["pos_team"].notna() & cfb_pbp["pos_team_id"].notna(),
            ["game_id", "pos_team", "pos_team_id"],
        ]
        .rename(columns={"pos_team": "team", "pos_team_id": "team_id"})
        .drop_duplicates(["game_id", "team"])
    )
    return actions, team_games, team_ids


def _cmd_cfb_absence_separation(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    actions, team_games, _ = _load_cfb_role_inputs(args.cfb_features)
    study = absence_separation_study(actions, team_games)

    output = _artifacts_root() / "cfb_role_experiments" / run_id()
    atomic_parquet(study["episodes"], output / "absence_episodes.parquet")
    atomic_parquet(study["carryover"], output / "carryover.parquet")
    atomic_csv(study["episode_summary"], output / "episode_summary.csv")
    atomic_csv(study["carryover_summary"], output / "carryover_summary.csv")
    configuration = {"command": "cfb-absence-separation", "cfb_features": str(args.cfb_features)}
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "episodes": len(study["episodes"]),
        "carryover_rows": len(study["carryover"]),
        "episode_summary": study["episode_summary"].to_dict(orient="records"),
        "carryover_summary": study["carryover_summary"].to_dict(orient="records"),
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.cfb_features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="cfb-absence-separation",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})
    print(study["episode_summary"].to_string(index=False))
    print(study["carryover_summary"].to_string(index=False))


def _cmd_cfb_role_benchmark(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    actions, team_games, team_ids = _load_cfb_role_inputs(args.cfb_features)
    canonical_games = pd.read_parquet(args.cfb_features)
    continuity = build_role_continuity(actions, team_games)
    features = attach_role_continuity(canonical_games, continuity, team_ids)
    side_columns = [column for column in CFB_ROLE_FEATURE_COLUMNS if not column.startswith("diff_")]
    non_neutral = features.loc[:, side_columns].ne(CONTINUITY_NEUTRAL).any(axis=1)

    result = cfb_role_benchmark(
        features,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    output = _artifacts_root() / "cfb_role_experiments" / run_id()
    atomic_parquet(result.predictions, output / "predictions.parquet")
    atomic_parquet(continuity, output / "role_continuity.parquet")
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    atomic_csv(result.paired, output / "paired_comparisons.csv")
    configuration = {
        "command": "cfb-role-benchmark",
        "cfb_features": str(args.cfb_features),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "hypothesis_frozen_before_scoring": True,
        "predeclaration": "docs/cfb_role_features.md",
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "games_scored": int(result.predictions["game_id"].nunique()),
        "role_feature_games_non_neutral": int(non_neutral.sum()),
        "role_feature_non_neutral_fraction": float(non_neutral.mean()),
        "paired_comparisons": result.paired.to_dict(orient="records"),
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.cfb_features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="cfb-role-benchmark",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})
    print(result.summary.to_string(index=False))
    print(result.paired.to_string(index=False))


def _cmd_cfb_variance_benchmark(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = pd.read_parquet(args.cfb_features)
    result = cfb_variance_benchmark(
        features,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    output = _artifacts_root() / "cfb_variance_experiments" / run_id()
    atomic_parquet(result.predictions, output / "predictions.parquet")
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    atomic_csv(result.paired, output / "paired_comparisons.csv")
    configuration = {
        "command": "cfb-variance-benchmark",
        "cfb_features": str(args.cfb_features),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "hypothesis_frozen_before_scoring": True,
        "predeclaration": "docs/margin_variance.md",
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "games_scored": int(result.predictions["game_id"].nunique()),
        "scale_ratio_summary": result.scale_ratio_summary,
        "paired_comparisons": result.paired.to_dict(orient="records"),
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.cfb_features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="cfb-variance-benchmark",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})
    print(result.summary.to_string(index=False))
    print(result.paired.to_string(index=False))


def _cmd_odds_ingest(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    payload, quota = fetch_odds_api_from_environment(
        regions=args.regions,
        markets=args.markets,
        bookmakers=args.bookmakers,
    )
    observed_at = datetime.now(UTC)
    quotes = parse_odds_api_response(payload, observed_at=observed_at)
    quotes = attach_nflverse_game_ids(quotes, features)
    request_metadata = {
        "sport": "americanfootball_nfl",
        "regions": args.regions,
        "markets": args.markets,
        "bookmakers": args.bookmakers,
        "odds_format": "american",
    }
    snapshot = write_market_snapshot(
        payload,
        quotes,
        _data_root() / "market" / "raw",
        observed_at=observed_at,
        request_metadata=request_metadata,
        quota=quota,
    )
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "quotes": len(quotes),
            "events": int(quotes["provider_event_id"].nunique()),
            "matched_events": int(
                quotes.loc[quotes["nflverse_game_id"].notna(), "provider_event_id"].nunique()
            ),
            "quota": quota,
        }
    )


def _cmd_odds_backfill(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    weeks = [int(week) for week in args.weeks.split(",")] if args.weeks else None
    labels = args.labels.split(",") if args.labels else None
    targets = plan_backfill(
        features,
        args.start_season,
        args.end_season,
        regions=args.regions,
        weeks=weeks,
        labels=labels,
    )
    plan = summarize_backfill_plan(targets)
    if args.dry_run:
        _print_json({"dry_run": True, "regions": args.regions, "plan": plan})
        return
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise ValueError("Set THE_ODDS_API_KEY before running the historical backfill")
    result = execute_backfill(
        targets,
        _data_root() / "market" / "raw",
        features,
        api_key=api_key,
        budget=args.budget,
        quota_floor=args.quota_floor,
        resume=args.resume,
        sleep_seconds=args.sleep_seconds,
    )
    _print_json({"regions": args.regions, "plan": plan, "result": result})


def _cmd_odds_summary(_: argparse.Namespace) -> None:
    history = load_quote_history(_data_root() / "market" / "raw")
    if history.empty:
        raise ValueError("No point-in-time market snapshots are available")
    consensus = spread_consensus(history)
    _print_json(
        {
            "quote_rows": len(history),
            "snapshots": int(history["observed_at_utc"].nunique()),
            "events": int(history["provider_event_id"].nunique()),
            "bookmakers": int(history["bookmaker_key"].nunique()),
            "matched_events": int(history["nflverse_game_id"].nunique()),
            "consensus_games": len(consensus),
            "first_observation": pd.to_datetime(history["observed_at_utc"], utc=True)
            .min()
            .isoformat(),
            "last_observation": pd.to_datetime(history["observed_at_utc"], utc=True)
            .max()
            .isoformat(),
        }
    )


def _cmd_tiebreaker(args: argparse.Namespace) -> None:
    report = tiebreaker_report(
        _data_root(),
        artifacts_root=_artifacts_root(),
        season=args.season,
        week=args.week,
        game_id=args.game_id,
    )
    print(format_tiebreaker_report(report))


def _cmd_totals_backtest(args: argparse.Namespace) -> None:
    results = run_totals_backtest(
        _data_root(),
        args.features,
        _artifacts_root(),
        min_train_games=args.min_train_games,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(format_totals_results(results))


def _cmd_market_backfill(args: argparse.Namespace) -> None:
    reference_games = _load_features(args.features)
    snapshot = fetch_historical_market_snapshot(
        _data_root() / "market" / "historical" / "raw",
        reference_games=reference_games,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "source": manifest["source"],
            "rows": manifest["rows"],
            "semantics": manifest["semantics"],
            "audit": manifest["audit"],
        }
    )


def _cmd_market_open_close_backfill(args: argparse.Namespace) -> None:
    reference_games = _load_features(args.features)
    snapshot = fetch_open_close_snapshot(
        _data_root() / "market" / "historical" / "open_close" / "raw",
        reference_games=reference_games,
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "source": manifest["source"],
            "rows": manifest["rows"],
            "semantics": manifest["semantics"],
            "audit": manifest["audit"],
        }
    )


def _cmd_clv_score(args: argparse.Namespace) -> None:
    predictions = pd.read_parquet(args.predictions)
    features = _load_features(args.features)
    market_root = _data_root() / "market" / "raw"
    pairing = build_pairing_table(market_root, capture_kind=args.capture_kind, schedule=features)
    if pairing.empty:
        raise ValueError(
            f"No {args.capture_kind!r} snapshots with decision quotes were found under "
            f"{market_root}"
        )
    close_reference = close_reference_table(pairing, features)
    scored = score_clv(predictions, pairing, close_reference)
    output = _artifacts_root() / "clv" / run_id()
    atomic_parquet(scored, output / "scored_picks.parquet")
    summary = clv_summary(scored)
    uncertainty = pd.concat(
        [
            week_blocked_bootstrap(
                scored,
                clv_summary,
                block="week",
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
            week_blocked_bootstrap(
                scored,
                clv_summary,
                block="season",
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    atomic_csv(uncertainty, output / "uncertainty.csv")
    configuration = {
        "command": "clv-score",
        "capture_kind": args.capture_kind,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "picks": len(predictions),
        "scored_picks": int(scored["clv_points"].notna().sum()),
        "summary": summary,
        "close_source_counts": close_reference["close_source"].value_counts().to_dict(),
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="clv-score",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_clv_pilot(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    market_root = _data_root() / "market" / "raw"
    active_model_config = (
        {
            "feature_profile": args.feature_profile,
            "regressor": args.regressor,
            "ridge_alpha": args.ridge_alpha,
            "target": "market_residual",
        }
        if args.feature_profile
        else resolve_active_model_config(_artifacts_root())
    )
    protocol = FROZEN_PILOT_PROTOCOL
    try:
        result = run_predeclared_pilot(
            market_root,
            features,
            protocol=protocol,
            capture_kind=args.capture_kind,
            active_model_config=active_model_config,
            min_train_games=args.min_train_games,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            threshold=args.threshold,
        )
    except PilotProtocolBlocked as blocked:
        _print_json(
            {
                "command": "clv-pilot",
                "blocked": True,
                "protocol": {
                    "train_start_season": protocol.train_start_season,
                    "train_end_season": protocol.train_end_season,
                    "validate_season": protocol.validate_season,
                    "test_season": protocol.test_season,
                },
                "reason": str(blocked),
            }
        )
        return
    output = _artifacts_root() / "clv_pilot" / run_id()
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": "clv-pilot",
        "active_model_config": active_model_config,
        **result,
        "provenance": artifact_provenance(
            {"command": "clv-pilot", **result["protocol"]}, args.features
        ),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="clv-pilot",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_clv_sign_test(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    market_root = _data_root() / "market" / "raw"
    active_model_config = (
        {
            "feature_profile": args.feature_profile,
            "regressor": args.regressor,
            "ridge_alpha": args.ridge_alpha,
            "target": "market_residual",
        }
        if args.feature_profile
        else resolve_active_model_config(_artifacts_root())
    )
    result = sign_test_pilot_b(
        market_root,
        features,
        capture_kind=args.capture_kind,
        active_model_config=active_model_config,
        min_train_games=args.min_train_games,
    )
    output = _artifacts_root() / "clv_sign_test" / run_id()
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": "clv-sign-test",
        "active_model_config": active_model_config,
        **result,
    }
    atomic_json(metadata, output / "metadata.json")
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_clv_ledger(args: argparse.Namespace) -> None:
    now = datetime.now(UTC)
    if args.skip_record:
        record: dict[str, Any] = {"skipped": True}
    else:
        try:
            record = record_paper_decisions(_artifacts_root(), data_root=_data_root(), now=now)
        except (ValueError, FileNotFoundError) as error:
            record = {"recorded": 0, "error": str(error)}
    decisions = load_paper_decisions(_artifacts_root())
    if decisions.empty:
        raise ValueError(
            "The paper-decision ledger is empty and nothing could be recorded; publish a "
            "weekly forecast first (`nfl-ats publish-predictions`). "
            f"Recording reported: {record}"
        )
    features = _load_features(args.features)
    close_reference = live_close_reference(_data_root() / "market" / "raw", features, as_of=now)
    scored = score_paper_ledger(decisions, close_reference)

    output = _artifacts_root() / "clv_ledger" / run_id()
    atomic_parquet(scored, output / "scored_decisions.parquet")
    picks_summary = clv_summary(scored)
    bets = scored.loc[scored["bet_side"].ne("PASS")].copy()
    bets["clv_points"] = bets["bet_clv_points"]
    if picks_summary["n"] > 0:
        uncertainty = pd.concat(
            [
                week_blocked_bootstrap(
                    scored,
                    clv_summary,
                    block="week",
                    samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed,
                ),
                week_blocked_bootstrap(
                    scored,
                    clv_summary,
                    block="season",
                    samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed,
                ),
            ],
            ignore_index=True,
        )
        atomic_csv(uncertainty, output / "uncertainty.csv")
    configuration = {
        "command": "clv-ledger",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": now.isoformat(),
        **configuration,
        "recording": record,
        "decisions": len(scored),
        "scored_decisions": int(scored["clv_status"].eq("scored").sum()),
        "pending_decisions": int(scored["clv_status"].eq("pending").sum()),
        "pick_summary": picks_summary,
        "bet_summary": clv_summary(bets),
        "close_source_counts": scored["close_source"].value_counts().to_dict(),
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="clv-ledger",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_prospective_record(args: argparse.Namespace) -> None:
    artifacts = _artifacts_root()
    entry = find_challenger(artifacts, args.challenger)
    artifact = args.artifact
    if artifact is None:
        artifact = find_challenger_artifact(artifacts, entry, season=args.season, week=args.week)
        if artifact is None:
            raise ValueError(
                f"No margin-predict artifact for {args.season} week {args.week} matches "
                f"challenger {args.challenger!r}. Generate it first with the challenger's "
                "registered weekly_generation_command, then re-run."
            )
    _print_json(
        record_challenger_decisions(artifacts, args.challenger, artifact, now=datetime.now(UTC))
    )


def _prospective_entrant_report(
    name: str,
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    close_reference: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Settle one entrant's ledger slice and summarize it (with intervals when settled)."""

    settled = settle_prospective_picks(decisions, outcomes, close_reference=close_reference)
    settled.insert(0, "entrant", name)
    report: dict[str, Any] = {
        "entrant": name,
        **prospective_accuracy(settled),
        "weeks": prospective_week_summary(settled).to_dict(orient="records"),
    }
    resolved = settled.dropna(subset=["correct_at_decision_line"])
    if not resolved.empty:
        report["uncertainty"] = week_blocked_bootstrap(
            resolved,
            prospective_accuracy_metrics,
            block="week",
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ).to_dict(orient="records")
    return settled, report


def _prospective_primary_entrants(active: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Expose the played policy and its frozen raw-model control."""

    entrants = [("active_model", active)]
    if active.empty or "model_pick_side" not in active.columns:
        return entrants
    raw_incumbent = active.copy()
    raw_incumbent["pick_side"] = raw_incumbent["model_pick_side"].astype(str)
    raw_incumbent["bet_side"] = "PASS"
    raw_incumbent["edge"] = float("nan")
    entrants.append(("base_model_no_pick_overlays", raw_incumbent))
    return entrants


def _cmd_prospective_score(args: argparse.Namespace) -> None:
    now = datetime.now(UTC)
    artifacts = _artifacts_root()
    features = _load_features(args.features)
    outcomes = features.loc[:, ["game_id", "result"]].copy()
    close_reference = live_close_reference(_data_root() / "market" / "raw", features, as_of=now)

    active = load_paper_decisions(artifacts)
    if not active.empty:
        active = active.loc[active["season"].astype(int).ge(args.start_season)]
    entrants = _prospective_primary_entrants(active)
    if not args.skip_challengers:
        challengers = load_challenger_decisions(artifacts)
        if not challengers.empty:
            challengers = challengers.loc[challengers["season"].astype(int).ge(args.start_season)]
        for challenger_id in sorted(set(challengers["challenger_id"].astype(str))):
            entrants.append(
                (challenger_id, challengers.loc[challengers["challenger_id"].eq(challenger_id)])
            )

    frames: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for name, decisions in entrants:
        settled, report = _prospective_entrant_report(
            name,
            decisions.reset_index(drop=True),
            outcomes,
            close_reference,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        frames.append(settled)
        reports.append(report)

    output = _artifacts_root() / "prospective_scoring" / run_id(now)
    combined = (
        pd.concat(frames, ignore_index=True)
        if any(not frame.empty for frame in frames)
        else pd.DataFrame()
    )
    if not combined.empty:
        atomic_parquet(combined, output / "settled_decisions.parquet")
        atomic_csv(
            pd.concat(
                [
                    prospective_week_summary(frame).assign(entrant=frame["entrant"].iloc[0])
                    for frame in frames
                    if not frame.empty
                ],
                ignore_index=True,
            ),
            output / "week_summary.csv",
        )
    configuration = {
        "command": "prospective-score",
        "start_season": args.start_season,
        "skip_challengers": args.skip_challengers,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    try:
        registered = [] if args.skip_challengers else active_challenger_ids(artifacts)
    except FileNotFoundError:
        registered = []
    metadata = {
        "created_at_utc": now.isoformat(),
        **configuration,
        "registered_challengers": registered,
        "entrants": reports,
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="prospective-score",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _find_drift_cards(
    artifacts_root: Path,
    *,
    season: int,
    week: int,
    feature_profile: str,
    probability_method: str,
) -> list[tuple[dict[str, Any], Path]]:
    """Every margin-predict card matching this drift query, oldest first.

    Cards are matched on configuration -- feature profile and probability
    method, not directory name or recency -- because the active model's card
    and every challenger's card share one ``margin_predictions`` namespace and
    picking the newest would silently monitor the wrong model (the same
    fingerprint lesson ``prospective-record`` learned; see
    ``docs/prospective_evidence.md``).
    """

    root = artifacts_root / "margin_predictions"
    if not root.is_dir():
        return []
    matches: list[tuple[dict[str, Any], Path]] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(metadata, dict):
            continue
        try:
            card_season = int(metadata.get("season", -1))
            card_week = int(metadata.get("week", -1))
        except (TypeError, ValueError):
            continue
        if card_season != season or card_week != week:
            continue
        if metadata.get("feature_profile") != feature_profile:
            continue
        if metadata.get("probability_method") != probability_method:
            continue
        predictions_path = metadata_path.parent / "predictions.csv"
        if not predictions_path.is_file():
            continue
        matches.append((metadata, predictions_path))
    matches.sort(key=lambda item: str(item[0].get("created_at_utc", "")))
    return matches


def _cmd_drift_report(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    cards = _find_drift_cards(
        _artifacts_root(),
        season=args.season,
        week=args.week,
        feature_profile=args.feature_profile,
        probability_method=args.probability_method,
    )
    if not cards:
        raise ValueError(
            f"No margin-predict card found for {args.season} week {args.week} "
            f"with feature profile {args.feature_profile!r} and probability method "
            f"{args.probability_method!r}. Run margin-predict first."
        )
    current_entry = cards[-1]
    history_entries = [entry for entry in cards if entry is not current_entry]
    history: pd.DataFrame | None = None
    if history_entries:
        # Oldest first so per-game dedupe keeps each game's FIRST published
        # probability -- the ledger convention: a republished or re-tuned card
        # never rewrites what an earlier card already said.
        history = pd.concat(
            [pd.read_csv(path) for _, path in history_entries], ignore_index=True
        ).drop_duplicates(subset=["game_id"], keep="first")
    report, drift_table = build_drift_report(
        features,
        pd.read_csv(current_entry[1]),
        history,
        season=args.season,
        week=args.week,
        feature_profile=args.feature_profile,
        probability_method=args.probability_method,
        reference_weeks=args.reference_weeks,
        calibration_recent_weeks=args.calibration_recent_weeks,
    )
    output = write_drift_artifacts(report, drift_table, _artifacts_root() / "drift")
    _print_json({**report, "artifact_directory": str(output), "card_used": str(current_entry[1])})


def _cmd_opener_evaluation(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    market_root = _data_root() / "market" / "raw"
    active_model_config = (
        {
            "feature_profile": args.feature_profile,
            "regressor": args.regressor,
            "ridge_alpha": args.ridge_alpha,
            "target": "market_residual",
        }
        if args.feature_profile
        else resolve_active_model_config(_artifacts_root())
    )
    scored = opener_pick_evaluation(
        market_root,
        features,
        active_model_config=active_model_config,
        min_train_games=args.min_train_games,
    )
    metrics = opener_evaluation_metrics(scored)
    uncertainty = pd.concat(
        [
            week_blocked_bootstrap(
                scored,
                opener_evaluation_metrics,
                block="week",
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
            week_blocked_bootstrap(
                scored,
                opener_evaluation_metrics,
                block="season",
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    season_rows: list[dict[str, Any]] = []
    for season, group in scored.groupby("season", sort=True):
        season_row: dict[str, Any] = {"season": int(str(season)), "games": len(group)}
        season_row.update(opener_evaluation_metrics(group))
        season_rows.append(season_row)
    season_summary = pd.DataFrame(season_rows)

    output = _artifacts_root() / "opener_evaluation" / run_id()
    atomic_parquet(scored, output / "per_game.parquet")
    atomic_csv(uncertainty, output / "uncertainty.csv")
    atomic_csv(season_summary, output / "season_summary.csv")
    configuration = {
        "command": "opener-evaluation",
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "hypothesis_frozen_before_scoring": True,
        "predeclaration": "docs/opener_evaluation.md",
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "active_model_config": active_model_config,
        "games": len(scored),
        "mean_absolute_open_to_close_move": float(scored["open_move"].abs().mean()),
        "metrics": metrics,
        "uncertainty": uncertainty.to_dict(orient="records"),
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="opener-evaluation",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})
    print(season_summary.to_string(index=False))
    print(uncertainty.to_string(index=False))


def _cmd_predict_close(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    market_root = _data_root() / "market" / "raw"
    active_model_config = (
        {
            "feature_profile": args.feature_profile,
            "regressor": args.regressor,
            "ridge_alpha": args.ridge_alpha,
            "target": "market_residual",
        }
        if args.feature_profile
        else resolve_active_model_config(_artifacts_root())
    )
    if args.season is not None and args.week is not None:
        season, week = args.season, args.week
    elif args.season is None and args.week is None:
        season, week = upcoming_week(features)
    else:
        raise ValueError("Pass --season and --week together, or neither")
    try:
        result = predict_close_for_week(
            market_root,
            features,
            season=season,
            week=week,
            active_model_config=active_model_config,
            min_train_games=args.min_train_games,
        )
    except (PilotProtocolBlocked, ClosePredictionUnavailable) as blocked:
        _print_json(
            {
                "command": "predict-close",
                "blocked": True,
                "season": season,
                "week": week,
                "reason": str(blocked),
            }
        )
        return
    predictions = result["predictions"]
    output = _artifacts_root() / "close_predictions" / run_id()
    atomic_parquet(predictions, output / "predictions.parquet")
    configuration = {
        "command": "predict-close",
        "season": season,
        "week": week,
        "min_train_games": args.min_train_games,
        "train_start_season": result["train_start_season"],
        "train_end_season": result["train_end_season"],
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "active_model_config": active_model_config,
        "train_games": result["train_games"],
        "games_predicted": len(predictions),
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="predict-close",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _resolve_snapshot(identifier: str | None) -> Snapshot:
    raw_root = _data_root() / "raw"
    return snapshot_from_root(raw_root / identifier) if identifier else latest_snapshot(raw_root)


def _cmd_build_features(args: argparse.Namespace) -> None:
    snapshot = _resolve_snapshot(args.snapshot)
    schedules, team_stats = load_snapshot(snapshot)
    features = build_game_features(
        schedules,
        team_stats,
        span=args.ewm_span,
        min_periods=args.min_periods,
        offseason_retention=args.offseason_retention,
        graph_half_life_weeks=args.graph_half_life,
        graph_ridge_alpha=args.graph_ridge_alpha,
        graph_min_games=args.graph_min_games,
        include_postseason=args.include_postseason,
    )
    destination = _data_root() / "processed" / "game_features.parquet"
    atomic_parquet(features, destination)
    completed = int(features["home_cover"].notna().sum())
    metadata = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source_snapshot": snapshot.snapshot_id,
        "ewm_span": args.ewm_span,
        "min_periods": args.min_periods,
        "offseason_retention": args.offseason_retention,
        "graph_half_life_weeks": args.graph_half_life,
        "graph_ridge_alpha": args.graph_ridge_alpha,
        "graph_min_games": args.graph_min_games,
        "include_postseason": args.include_postseason,
        "rows": len(features),
        "postseason_rows": int(features["game_type"].ne("REG").sum()),
        "completed_non_push_rows": completed,
        "first_season": int(features["season"].min()),
        "last_season": int(features["season"].max()),
        "destination": str(destination),
    }
    atomic_json(metadata, destination.with_name("game_features.manifest.json"))
    _print_json(metadata)


def _cmd_build_pbp_features(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    raw_root = _data_root() / "pbp" / "raw"
    snapshot = (
        pbp_snapshot_from_root(raw_root / args.snapshot)
        if args.snapshot
        else latest_pbp_snapshot(raw_root)
    )
    pbp = load_pbp_snapshot(snapshot)
    enriched = enrich_with_pbp_features(
        features,
        pbp,
        span=args.ewm_span,
        min_periods=args.min_periods,
        offseason_retention=args.offseason_retention,
        opponent_half_life_weeks=args.opponent_half_life,
        opponent_ridge_alpha=args.opponent_ridge_alpha,
        opponent_min_team_games=args.opponent_min_games,
    )
    destination = _data_root() / "processed" / "game_features_pbp.parquet"
    atomic_parquet(enriched, destination)
    metadata = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source_pbp_snapshot": snapshot.snapshot_id,
        "source_features": str(args.features),
        "ewm_span": args.ewm_span,
        "min_periods": args.min_periods,
        "offseason_retention": args.offseason_retention,
        "pbp_feature_version": PBP_FEATURE_VERSION,
        "opponent_half_life_weeks": args.opponent_half_life,
        "opponent_ridge_alpha": args.opponent_ridge_alpha,
        "opponent_min_team_games": args.opponent_min_games,
        "rows": len(enriched),
        "pbp_rows": len(pbp),
        "destination": str(destination),
    }
    atomic_json(metadata, destination.with_name("game_features_pbp.manifest.json"))
    _print_json(metadata)


def _cmd_build_qb_features(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    pbp_root = _data_root() / "pbp" / "raw"
    pbp_snapshot = (
        pbp_snapshot_from_root(pbp_root / args.pbp_snapshot)
        if args.pbp_snapshot
        else latest_pbp_snapshot(pbp_root)
    )
    depth_root = _data_root() / "quarterbacks" / "depth" / "raw"
    depth_snapshot = (
        depth_snapshot_from_root(depth_root / args.depth_snapshot)
        if args.depth_snapshot
        else latest_depth_snapshot(depth_root)
    )
    enriched = enrich_with_qb_features(
        features,
        load_pbp_snapshot(pbp_snapshot),
        load_depth_snapshot(depth_snapshot),
        decision_hours_before_kickoff=args.decision_hours,
        max_depth_age_days=args.max_depth_age_days,
        span=args.ewm_span,
        min_dropbacks=args.min_dropbacks,
        offseason_retention=args.offseason_retention,
    )
    destination = _data_root() / "processed" / "game_features_qb.parquet"
    atomic_parquet(enriched, destination)
    both_qbs = enriched["home_qb_id"].notna() & enriched["away_qb_id"].notna()
    both_states = (
        enriched["home_qb_epa_per_dropback"].notna() & enriched["away_qb_epa_per_dropback"].notna()
    )
    metadata = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source_features": str(args.features),
        "source_pbp_snapshot": pbp_snapshot.snapshot_id,
        "source_depth_snapshot": depth_snapshot.snapshot_id,
        "decision_hours_before_kickoff": args.decision_hours,
        "max_depth_age_days": args.max_depth_age_days,
        "ewm_span": args.ewm_span,
        "min_dropbacks": args.min_dropbacks,
        "offseason_retention": args.offseason_retention,
        "rows": len(enriched),
        "games_with_both_expected_qbs": int(both_qbs.sum()),
        "games_with_both_qb_states": int(both_states.sum()),
        "destination": str(destination),
    }
    atomic_json(metadata, destination.with_name("game_features_qb.manifest.json"))
    _print_json(metadata)


def _cmd_build_player_features(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    player_root = _data_root() / "players" / "raw"
    player_snapshot = (
        player_snapshot_from_root(player_root / args.player_snapshot)
        if args.player_snapshot
        else latest_player_snapshot(player_root)
    )
    pbp_root = _data_root() / "pbp" / "raw"
    pbp_snapshot = (
        pbp_snapshot_from_root(pbp_root / args.pbp_snapshot)
        if args.pbp_snapshot
        else latest_pbp_snapshot(pbp_root)
    )
    player_value_root = _data_root() / "players" / "values" / "raw"
    player_value_snapshot = (
        player_value_snapshot_from_root(player_value_root / args.player_value_snapshot)
        if args.player_value_snapshot
        else latest_player_value_snapshot(player_value_root)
    )
    injuries, rosters, snaps = load_player_snapshot(player_snapshot)
    enriched = enrich_with_player_features(
        features,
        injuries,
        rosters,
        snaps,
        load_pbp_snapshot(pbp_snapshot),
        load_player_value_snapshot(player_value_snapshot),
        decision_hours_before_kickoff=args.decision_hours,
        role_span=args.role_span,
        qb_span=args.qb_span,
        qb_min_dropbacks=args.qb_min_dropbacks,
        offseason_retention=args.offseason_retention,
        value_span=args.value_span,
        value_prior_snaps=args.value_prior_snaps,
    )
    destination = args.destination
    atomic_parquet(enriched, destination)
    both_qbs = enriched["home_projected_qb_id"].notna() & enriched["away_projected_qb_id"].notna()
    both_injuries = (
        enriched["home_injury_offense_unavailability"].notna()
        & enriched["away_injury_offense_unavailability"].notna()
    )
    both_continuity = (
        enriched["home_offense_lineup_continuity"].notna()
        & enriched["away_offense_lineup_continuity"].notna()
    )
    both_player_values = (
        enriched["home_injury_skill_epa_value_lost"].notna()
        & enriched["away_injury_skill_epa_value_lost"].notna()
    )
    metadata = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source_features": str(args.features),
        "source_player_snapshot": player_snapshot.snapshot_id,
        "source_pbp_snapshot": pbp_snapshot.snapshot_id,
        "source_player_value_snapshot": player_value_snapshot.snapshot_id,
        "player_feature_version": PLAYER_FEATURE_VERSION,
        "decision_hours_before_kickoff": args.decision_hours,
        "role_span": args.role_span,
        "qb_span": args.qb_span,
        "qb_min_dropbacks": args.qb_min_dropbacks,
        "offseason_retention": args.offseason_retention,
        "value_span": args.value_span,
        "value_prior_snaps": args.value_prior_snaps,
        "rows": len(enriched),
        "games_with_both_projected_qbs": int(both_qbs.sum()),
        "games_with_both_injury_states": int(both_injuries.sum()),
        "games_with_both_lineup_continuity_states": int(both_continuity.sum()),
        "games_with_both_player_value_states": int(both_player_values.sum()),
        "destination": str(destination),
    }
    atomic_json(metadata, destination.with_name(f"{destination.stem}.manifest.json"))
    _print_json(metadata)


def _cmd_build_participation_features(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    player_root = _data_root() / "players" / "raw"
    player_snapshot = (
        player_snapshot_from_root(player_root / args.player_snapshot)
        if args.player_snapshot
        else latest_player_snapshot(player_root)
    )
    pbp_root = _data_root() / "pbp" / "raw"
    pbp_snapshot = (
        pbp_snapshot_from_root(pbp_root / args.pbp_snapshot)
        if args.pbp_snapshot
        else latest_pbp_snapshot(pbp_root)
    )
    player_value_root = _data_root() / "players" / "values" / "raw"
    player_value_snapshot = (
        player_value_snapshot_from_root(player_value_root / args.player_value_snapshot)
        if args.player_value_snapshot
        else latest_player_value_snapshot(player_value_root)
    )
    participation_root = _data_root() / "players" / "participation" / "raw"
    participation_snapshot = (
        participation_snapshot_from_root(participation_root / args.participation_snapshot)
        if args.participation_snapshot
        else latest_participation_snapshot(participation_root)
    )

    pbp = load_pbp_snapshot(pbp_snapshot)
    rating_started = perf_counter()
    ratings = build_season_lagged_player_ratings(
        load_participation_snapshot(participation_snapshot),
        pbp,
        target_seasons=sorted(features["season"].astype(int).unique()),
    )
    rating_seconds = perf_counter() - rating_started
    injuries, rosters, snaps = load_player_snapshot(player_snapshot)
    enrichment_started = perf_counter()
    enriched = enrich_with_player_features(
        features,
        injuries,
        rosters,
        snaps,
        pbp,
        load_player_value_snapshot(player_value_snapshot),
        ratings,
        decision_hours_before_kickoff=args.decision_hours,
        role_span=args.role_span,
        qb_span=args.qb_span,
        qb_min_dropbacks=args.qb_min_dropbacks,
        offseason_retention=args.offseason_retention,
        value_span=args.value_span,
        value_prior_snaps=args.value_prior_snaps,
    )
    enrichment_seconds = perf_counter() - enrichment_started
    atomic_parquet(ratings, args.ratings_destination)
    atomic_parquet(enriched, args.destination)
    both_participation_values = (
        enriched["home_injury_offense_participation_value_lost"].notna()
        & enriched["away_injury_offense_participation_value_lost"].notna()
    )
    target_summary = (
        ratings.groupby(
            ["target_season", "source_start_season", "source_end_season", "source_plays"],
            sort=True,
        )
        .size()
        .rename("rated_players")
        .reset_index()
        .to_dict(orient="records")
    )
    metadata = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source_features": str(args.features),
        "source_player_snapshot": player_snapshot.snapshot_id,
        "source_player_value_snapshot": player_value_snapshot.snapshot_id,
        "source_pbp_snapshot": pbp_snapshot.snapshot_id,
        "source_participation_snapshot": participation_snapshot.snapshot_id,
        "player_feature_version": PLAYER_PARTICIPATION_FEATURE_VERSION,
        "rating_configuration": {
            "lookback_seasons": PARTICIPATION_RATING_LOOKBACK_SEASONS,
            "ridge_alpha": PARTICIPATION_RATING_RIDGE_ALPHA,
            "team_feature_scale": PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
            "reliability_prior_plays": PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
            "epa_clip": PARTICIPATION_RATING_EPA_CLIP,
            "eligible_plays": "competitive valid 11-on-11 v1 PBP plays",
            "availability": "only seasons strictly before each target season",
        },
        "target_seasons": target_summary,
        "ratings_rows": len(ratings),
        "ratings_sha256": sha256_file(args.ratings_destination),
        "rows": len(enriched),
        "games_with_both_participation_value_states": int(both_participation_values.sum()),
        "ratings_destination": str(args.ratings_destination),
        "destination": str(args.destination),
        "timing": {
            "rating_seconds": rating_seconds,
            "enrichment_seconds": enrichment_seconds,
            "total_seconds": perf_counter() - command_started,
        },
    }
    atomic_json(metadata, args.destination.with_name(f"{args.destination.stem}.manifest.json"))
    _print_json(metadata)


def _cmd_build_learned_availability_features(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    player_root = _data_root() / "players" / "raw"
    player_snapshot = (
        player_snapshot_from_root(player_root / args.player_snapshot)
        if args.player_snapshot
        else latest_player_snapshot(player_root)
    )
    pbp_root = _data_root() / "pbp" / "raw"
    pbp_snapshot = (
        pbp_snapshot_from_root(pbp_root / args.pbp_snapshot)
        if args.pbp_snapshot
        else latest_pbp_snapshot(pbp_root)
    )
    player_value_root = _data_root() / "players" / "values" / "raw"
    player_value_snapshot = (
        player_value_snapshot_from_root(player_value_root / args.player_value_snapshot)
        if args.player_value_snapshot
        else latest_player_value_snapshot(player_value_root)
    )
    injuries, rosters, snaps = load_player_snapshot(player_snapshot)
    canonical_injury_rows = canonicalize_injuries(injuries)
    canonical_roster_rows = canonicalize_rosters(rosters)
    snaps_with_ids = attach_snap_player_ids(canonicalize_snaps(snaps), canonical_roster_rows)

    availability_started = perf_counter()
    outcomes = build_availability_outcomes(
        canonical_injury_rows,
        snaps_with_ids,
        features,
        decision_hours_before_kickoff=args.decision_hours,
    )
    rates = build_season_lagged_availability_rates(
        outcomes,
        target_seasons=sorted(features["season"].astype(int).unique()),
    )
    scored = score_availability_rates(outcomes, rates)
    availability_summary = summarize_availability_scores(scored)
    availability_seconds = perf_counter() - availability_started

    enrichment_started = perf_counter()
    enriched = enrich_with_player_features(
        features,
        injuries,
        rosters,
        snaps,
        load_pbp_snapshot(pbp_snapshot),
        player_stats=load_player_value_snapshot(player_value_snapshot),
        availability_rates=rates,
        decision_hours_before_kickoff=args.decision_hours,
        role_span=args.role_span,
        qb_span=args.qb_span,
        qb_min_dropbacks=args.qb_min_dropbacks,
        offseason_retention=args.offseason_retention,
        value_span=args.value_span,
        value_prior_snaps=args.value_prior_snaps,
    )
    enrichment_seconds = perf_counter() - enrichment_started
    atomic_parquet(rates, args.rates_destination)
    atomic_csv(availability_summary, args.evaluation_destination)
    atomic_parquet(enriched, args.destination)
    fixed = availability_summary.loc[availability_summary["method"].eq("fixed")].iloc[0]
    learned = availability_summary.loc[availability_summary["method"].eq("learned")].iloc[0]
    metadata = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source_features": str(args.features),
        "source_player_snapshot": player_snapshot.snapshot_id,
        "source_player_value_snapshot": player_value_snapshot.snapshot_id,
        "source_pbp_snapshot": pbp_snapshot.snapshot_id,
        "player_feature_version": PLAYER_AVAILABILITY_FEATURE_VERSION,
        "availability_configuration": {
            "rate_version": AVAILABILITY_RATE_VERSION,
            "combination": "report category x practice category",
            "position_refinement": True,
            "combination_prior": AVAILABILITY_COMBINATION_PRIOR,
            "position_prior": AVAILABILITY_POSITION_PRIOR,
            "training_window": "expanding completed prior seasons only",
            "target": "player logged any offense, defense, or special-teams snap",
            "decision_hours_before_kickoff": args.decision_hours,
        },
        "availability_outcomes": len(outcomes),
        "availability_evaluation_player_games": int(learned["player_games"]),
        "fixed_availability_brier": float(fixed["brier_score"]),
        "learned_availability_brier": float(learned["brier_score"]),
        "availability_brier_improvement": float(fixed["brier_score"] - learned["brier_score"]),
        "rates_rows": len(rates),
        "rates_sha256": sha256_file(args.rates_destination),
        "rows": len(enriched),
        "rates_destination": str(args.rates_destination),
        "evaluation_destination": str(args.evaluation_destination),
        "destination": str(args.destination),
        "timing": {
            "availability_seconds": availability_seconds,
            "enrichment_seconds": enrichment_seconds,
            "total_seconds": perf_counter() - command_started,
        },
    }
    atomic_json(metadata, args.destination.with_name(f"{args.destination.stem}.manifest.json"))
    _print_json(metadata)


def _cmd_backtest(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    result = walk_forward_backtest(
        features,
        start_season=args.start_season,
        end_season=args.end_season,
        model_name=args.model,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        feature_set=args.feature_set,
    )
    output = _artifacts_root() / "backtests" / run_id()
    atomic_parquet(result.predictions, output / "predictions.parquet")
    portfolio = simulate_paper_bankroll(
        result.predictions,
        initial_bankroll=args.initial_bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bet_fraction=args.max_bet_fraction,
        max_week_fraction=args.max_week_fraction,
        probability_haircut=args.probability_haircut,
    )
    atomic_parquet(portfolio.ledger, output / "paper_ledger.parquet")
    atomic_json(portfolio.metrics, output / "portfolio_metrics.json")
    simulation = simulate_bankroll_paths(
        result.predictions,
        paths=args.bankroll_paths,
        seed=args.bankroll_seed,
        initial_bankroll=args.initial_bankroll,
        kelly_multiplier=args.kelly_multiplier,
        max_bet_fraction=args.max_bet_fraction,
        max_week_fraction=args.max_week_fraction,
        probability_haircut=args.probability_haircut,
    )
    atomic_json(simulation.metrics, output / "bankroll_simulation.json")
    atomic_parquet(simulation.paths, output / "bankroll_paths.parquet")
    metrics = {**result.metrics, "paper_portfolio": portfolio.metrics}
    atomic_json(metrics, output / "metrics.json")
    uncertainty = pd.concat(
        [
            block_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            block_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    atomic_csv(uncertainty, output / "uncertainty.csv")
    configuration = {
        "command": "backtest",
        "start_season": args.start_season,
        "end_season": args.end_season,
        "model": args.model,
        "feature_set": args.feature_set,
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "initial_bankroll": args.initial_bankroll,
        "kelly_multiplier": args.kelly_multiplier,
        "max_bet_fraction": args.max_bet_fraction,
        "max_week_fraction": args.max_week_fraction,
        "probability_haircut": args.probability_haircut,
        "bankroll_paths": args.bankroll_paths,
        "bankroll_seed": args.bankroll_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    provenance = artifact_provenance(configuration, args.features)
    write_experiment_artifact(
        output,
        "run.json",
        provenance,
        command="backtest",
        metrics=metrics,
        provenance_key=None,
        registry_root=_registry_root(),
    )
    card = build_model_card(metrics, provenance, result.predictions)
    atomic_json(card, output / "model_card.json")
    (output / "model_card.md").write_text(model_card_markdown(card), encoding="utf-8")
    _print_json({**metrics, "artifact_directory": str(output)})


def _cmd_nested_evaluate(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    candidates = parse_candidates(args.candidates)
    result = nested_walk_forward_evaluation(
        features,
        first_test_season=args.first_test_season,
        last_test_season=args.last_test_season,
        validation_seasons=args.validation_seasons,
        candidates=candidates,
        selection_metric=args.selection_metric,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
    )
    output = _artifacts_root() / "nested_evaluations" / run_id()
    atomic_csv(result.candidate_validation, output / "candidate_validation.csv")
    atomic_csv(result.fold_summary, output / "fold_summary.csv")
    atomic_parquet(result.predictions, output / "predictions.parquet")
    atomic_json(result.metrics, output / "metrics.json")
    uncertainty = pd.concat(
        [
            block_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            block_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    atomic_csv(uncertainty, output / "uncertainty.csv")
    dependence = prediction_dependence_audit(
        result.predictions,
        permutations=args.dependence_permutations,
        seed=args.dependence_seed,
    )
    atomic_json(dependence.summary, output / "dependence_summary.json")
    atomic_csv(dependence.team_summary, output / "dependence_by_team.csv")
    configuration = {
        "command": "nested-evaluate",
        "first_test_season": args.first_test_season,
        "last_test_season": args.last_test_season,
        "validation_seasons": args.validation_seasons,
        "selection_metric": args.selection_metric,
        "candidates": [candidate.candidate_id for candidate in candidates],
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "dependence_permutations": args.dependence_permutations,
        "dependence_seed": args.dependence_seed,
    }
    write_experiment_artifact(
        output,
        "run.json",
        artifact_provenance(configuration, args.features),
        command="nested-evaluate",
        metrics=result.metrics,
        provenance_key=None,
        registry_root=_registry_root(),
    )
    _print_json({**result.metrics, "artifact_directory": str(output)})


def _cmd_dependence_audit(args: argparse.Namespace) -> None:
    if not args.predictions.is_file():
        raise FileNotFoundError(f"Prediction table not found: {args.predictions}")
    predictions = pd.read_parquet(args.predictions)
    audit = prediction_dependence_audit(
        predictions,
        permutations=args.permutations,
        seed=args.seed,
    )
    output = _artifacts_root() / "dependence" / run_id()
    summary = {
        **audit.summary,
        "source_predictions": str(args.predictions.resolve()),
        "source_predictions_sha256": sha256_file(args.predictions),
    }
    atomic_json(summary, output / "summary.json")
    atomic_csv(audit.team_summary, output / "by_team.csv")
    atomic_parquet(audit.team_residuals, output / "team_residuals.parquet")
    _print_json({**summary, "artifact_directory": str(output)})


def _cmd_experiment(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    feature_sets = tuple(part.strip() for part in args.feature_sets.split(",") if part.strip())
    result = run_feature_set_experiment(
        features,
        start_season=args.start_season,
        model_name=args.model,
        feature_sets=feature_sets,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
    )
    output = _artifacts_root() / "experiments" / run_id()
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    atomic_parquet(result.predictions, output / "predictions.parquet")
    baseline_feature_set = args.baseline_feature_set or feature_sets[0]
    paired = pd.concat(
        [
            paired_feature_comparisons(
                result.predictions,
                baseline_feature_set=baseline_feature_set,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                result.predictions,
                baseline_feature_set=baseline_feature_set,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    atomic_csv(paired, output / "paired_comparisons.csv")

    configuration = {
        "command": "experiment",
        "start_season": args.start_season,
        "model": args.model,
        "feature_sets": list(feature_sets),
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "baseline_feature_set": baseline_feature_set,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="experiment",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_experiment_run(args: argparse.Namespace) -> None:
    """The declarative pipeline: spec in, screen -> bootstrap -> classify ->
    registry record -> provenance stamp out, with zero hand-transcription.
    See ``nfl_ats.experiment_runner`` and ``docs/experiment_pipeline.md``.
    """

    outcome = run_experiment_cli(
        args.spec,
        repo_root=Path.cwd(),
        dry_run=args.dry_run,
        replace=args.replace,
        features_path=args.features,
        market_root=args.market_root,
        artifacts_root=_artifacts_root(),
        registry_root=_registry_root(),
        registry_path=weak_signal_registry_path(_registry_root()),
    )
    if outcome.dry_run:
        _print_json({"dry_run": True, "would_record": outcome.preview})
        return
    _print_json(
        {
            "dry_run": False,
            "artifact_directory": outcome.artifact_directory,
            "registry_record": outcome.registry_record,
            "recorded": outcome.preview,
        }
    )


def _cmd_experiment_verify(args: argparse.Namespace) -> None:
    """Report which registry rows still resolve to a real artifact on disk.

    Read-only audit of ``registry/experiments/``: every row's
    ``artifact_directory`` is resolved (absolute stored paths as-is; relative
    ones against each ``--artifacts-root`` and the repo root) and checked for
    existence, and path/identity inconsistencies are flagged. See
    :func:`nfl_ats.provenance.verify_experiment_links` for the flag meanings.
    """

    verifications = verify_experiment_links(
        artifacts_roots=list(args.artifacts_root) if args.artifacts_root else None
    )
    if args.as_json:
        _print_json(
            {
                "rows": [vars(verification) for verification in verifications],
                "total": len(verifications),
                "present": sum(1 for v in verifications if v.exists),
                "missing": sum(1 for v in verifications if not v.exists),
            }
        )
        if args.require_links and any(not v.exists for v in verifications):
            raise SystemExit(1)
        return

    present = sum(1 for v in verifications if v.exists)
    missing = len(verifications) - present
    flag_counts: dict[str, int] = {}
    for v in verifications:
        for flag in v.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    print(f"Scanned {len(verifications)} experiment-registry rows.")
    print(f"  linked artifact present: {present}")
    print(
        f"  linked artifact missing: {missing} (expected on a fresh clone: "
        f"artifacts/ is gitignored and local)"
    )
    if flag_counts:
        print("  consistency flags:")
        for flag, count in sorted(flag_counts.items()):
            print(f"    {flag}: {count}")
    if missing:
        print("\nMissing links (measured on this machine, not inferred):")
        for v in verifications:
            if not v.exists:
                tried = ", ".join(v.candidate_paths) or "(no artifact_directory stored)"
                print(f"  - {v.experiment_id}\n      tried: {tried}")
    if args.require_links and missing:
        raise SystemExit(1)


def _cmd_margin_backtest(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    modeling_started = perf_counter()
    result = walk_forward_outcomes(
        features,
        start_season=args.start_season,
        regressor=args.regressor,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        feature_profile=args.feature_profile,
        methods=methods,
        ridge_alpha=args.ridge_alpha,
        probability_method=args.probability_method,
    )
    modeling_seconds = perf_counter() - modeling_started
    output = _artifacts_root() / "margins" / run_id()
    atomic_parquet(result.predictions, output / "predictions.parquet")
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    uncertainty_started = perf_counter()
    uncertainty = pd.concat(
        [
            outcome_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            outcome_bootstrap_intervals(
                result.predictions,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    uncertainty_seconds = perf_counter() - uncertainty_started
    atomic_csv(uncertainty, output / "uncertainty.csv")
    configuration = {
        "command": "margin-backtest",
        "start_season": args.start_season,
        "regressor": args.regressor,
        "ridge_alpha": args.ridge_alpha,
        "calibration_method": "none",
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "feature_profile": args.feature_profile,
        "probability_method": args.probability_method,
        "methods": list(methods),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "methods": result.summary["method"].tolist(),
        "timing": {
            "modeling_seconds": modeling_seconds,
            "uncertainty_seconds": uncertainty_seconds,
            "total_seconds": perf_counter() - command_started,
        },
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="margin-backtest",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_player_ablation(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    profiles = tuple(part.strip() for part in args.profiles.split(",") if part.strip())
    result = run_outcome_profile_experiment(
        features,
        start_season=args.start_season,
        profiles=profiles,
        regressor=args.regressor,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        ridge_alpha=args.ridge_alpha,
    )
    output = _artifacts_root() / "player_experiments" / run_id()
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    atomic_parquet(result.predictions, output / "predictions.parquet")

    comparison_input = result.predictions.rename(columns={"feature_profile": "feature_set"})
    paired = pd.concat(
        [
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=args.baseline_profile,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=args.baseline_profile,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "baseline_feature_set": "baseline_feature_profile",
            "candidate_feature_set": "candidate_feature_profile",
        }
    )
    atomic_csv(paired, output / "paired_comparisons.csv")
    nested = nested_outcome_profile_selection(
        result.predictions,
        first_test_season=args.first_nested_test_season,
        validation_seasons=args.validation_seasons,
    )
    atomic_csv(nested.candidate_validation, output / "nested_candidate_validation.csv")
    atomic_csv(nested.fold_summary, output / "nested_fold_summary.csv")
    atomic_csv(nested.summary, output / "nested_summary.csv")
    atomic_csv(nested.season_summary, output / "nested_season_summary.csv")
    atomic_parquet(nested.predictions, output / "nested_predictions.parquet")
    nested_baseline = result.predictions.loc[
        result.predictions["feature_profile"].eq(args.baseline_profile)
        & result.predictions["season"].isin(nested.predictions["season"].unique())
    ].copy()
    nested_baseline["feature_set"] = args.baseline_profile
    nested_selected = nested.predictions.copy()
    nested_selected["feature_set"] = "nested_selected"
    nested_comparison_input = pd.concat([nested_baseline, nested_selected], ignore_index=True)
    nested_paired = pd.concat(
        [
            paired_feature_comparisons(
                nested_comparison_input,
                baseline_feature_set=args.baseline_profile,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                nested_comparison_input,
                baseline_feature_set=args.baseline_profile,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "baseline_feature_set": "baseline_feature_profile",
            "candidate_feature_set": "candidate_feature_profile",
        }
    )
    atomic_csv(nested_paired, output / "nested_paired_comparisons.csv")
    configuration = {
        "command": "player-ablation",
        "start_season": args.start_season,
        "regressor": args.regressor,
        "ridge_alpha": args.ridge_alpha,
        "calibration_method": "none",
        "profiles": list(profiles),
        "baseline_profile": args.baseline_profile,
        "method": "market_residual",
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "first_nested_test_season": args.first_nested_test_season,
        "validation_seasons": args.validation_seasons,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="player-ablation",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_participation_ablation(args: argparse.Namespace) -> None:
    """Run the predeclared one-candidate participation-value comparison."""

    command_started = perf_counter()
    features = _load_features(args.features)
    profiles = (
        FROZEN_PARTICIPATION_BASELINE_PROFILE,
        FROZEN_PARTICIPATION_CANDIDATE_PROFILE,
    )
    result = run_outcome_profile_experiment(
        features,
        start_season=FROZEN_PARTICIPATION_START_SEASON,
        profiles=profiles,
        regressor="ridge",
        min_edge=0.02,
        min_train_games=FROZEN_PARTICIPATION_MIN_TRAIN_GAMES,
        ridge_alpha=FROZEN_PARTICIPATION_RIDGE_ALPHA,
    )
    output = _artifacts_root() / "participation_experiments" / run_id()
    atomic_csv(result.summary, output / "summary.csv")
    atomic_csv(result.season_summary, output / "season_summary.csv")
    atomic_parquet(result.predictions, output / "predictions.parquet")
    comparison_input = result.predictions.rename(columns={"feature_profile": "feature_set"})
    paired = pd.concat(
        [
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=FROZEN_PARTICIPATION_BASELINE_PROFILE,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=FROZEN_PARTICIPATION_BASELINE_PROFILE,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "baseline_feature_set": "baseline_feature_profile",
            "candidate_feature_set": "candidate_feature_profile",
        }
    )
    atomic_csv(paired, output / "paired_comparisons.csv")
    configuration = {
        "command": "participation-ablation",
        "hypothesis_frozen_before_scoring": True,
        "start_season": FROZEN_PARTICIPATION_START_SEASON,
        "regressor": "ridge",
        "ridge_alpha": FROZEN_PARTICIPATION_RIDGE_ALPHA,
        "calibration_method": "none",
        "profiles": list(profiles),
        "baseline_profile": FROZEN_PARTICIPATION_BASELINE_PROFILE,
        "candidate_profile": FROZEN_PARTICIPATION_CANDIDATE_PROFILE,
        "method": "market_residual",
        "min_edge": 0.02,
        "min_train_games": FROZEN_PARTICIPATION_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="participation-ablation",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_availability_ablation(args: argparse.Namespace) -> None:
    """Run the predeclared learned-versus-fixed availability comparison."""

    command_started = perf_counter()
    baseline_features = _load_features(args.baseline_features)
    learned_features = _load_features(args.learned_features)
    results = []
    for method, features in (
        ("fixed", baseline_features),
        ("learned", learned_features),
    ):
        result = run_outcome_profile_experiment(
            features,
            start_season=FROZEN_AVAILABILITY_START_SEASON,
            profiles=(FROZEN_AVAILABILITY_PROFILE,),
            regressor="ridge",
            min_edge=0.02,
            min_train_games=FROZEN_AVAILABILITY_MIN_TRAIN_GAMES,
            ridge_alpha=FROZEN_AVAILABILITY_RIDGE_ALPHA,
        )
        result.summary.insert(0, "availability_method", method)
        result.season_summary.insert(0, "availability_method", method)
        result.predictions.insert(0, "availability_method", method)
        results.append(result)
    summary = pd.concat([result.summary for result in results], ignore_index=True)
    season_summary = pd.concat([result.season_summary for result in results], ignore_index=True)
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    comparison_input = predictions.rename(columns={"availability_method": "feature_set"})
    paired = pd.concat(
        [
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set="fixed",
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set="fixed",
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "baseline_feature_set": "baseline_availability_method",
            "candidate_feature_set": "candidate_availability_method",
        }
    )
    output = _artifacts_root() / "availability_experiments" / run_id()
    atomic_csv(summary, output / "summary.csv")
    atomic_csv(season_summary, output / "season_summary.csv")
    atomic_parquet(predictions, output / "predictions.parquet")
    atomic_csv(paired, output / "paired_comparisons.csv")
    configuration = {
        "command": "availability-ablation",
        "hypothesis_frozen_before_scoring": True,
        "start_season": FROZEN_AVAILABILITY_START_SEASON,
        "regressor": "ridge",
        "ridge_alpha": FROZEN_AVAILABILITY_RIDGE_ALPHA,
        "calibration_method": "none",
        "feature_profile": FROZEN_AVAILABILITY_PROFILE,
        "baseline_availability_method": "fixed",
        "candidate_availability_method": "learned",
        "method": "market_residual",
        "min_edge": 0.02,
        "min_train_games": FROZEN_AVAILABILITY_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "baseline_provenance": artifact_provenance(configuration, args.baseline_features),
        "learned_provenance": artifact_provenance(configuration, args.learned_features),
        "timing": {"total_seconds": perf_counter() - command_started},
    }
    atomic_json(metadata, output / "metadata.json")
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_player_model_selection(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    result = run_frozen_player_model_selection(
        features,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
    )
    output = _artifacts_root() / "player_model_selection" / run_id()
    atomic_parquet(result.raw_predictions, output / "raw_predictions.parquet")
    atomic_parquet(result.predictions, output / "candidate_predictions.parquet")
    atomic_csv(result.summary, output / "candidate_summary.csv")
    atomic_csv(result.season_summary, output / "candidate_season_summary.csv")
    atomic_csv(result.nested.candidate_validation, output / "nested_candidate_validation.csv")
    atomic_csv(result.nested.fold_summary, output / "nested_fold_summary.csv")
    atomic_csv(result.nested.summary, output / "nested_summary.csv")
    atomic_csv(result.nested.season_summary, output / "nested_season_summary.csv")
    atomic_parquet(result.nested.predictions, output / "nested_predictions.parquet")

    baseline_id = player_model_candidate_id("base", 10.0, "none")
    nested_seasons = result.nested.predictions["season"].unique()
    nested_baseline = result.predictions.loc[
        result.predictions["candidate_id"].eq(baseline_id)
        & result.predictions["season"].isin(nested_seasons)
    ].copy()
    nested_baseline["feature_set"] = baseline_id
    nested_selected = result.nested.predictions.copy()
    nested_selected["feature_set"] = "nested_selected"
    comparison_input = pd.concat([nested_baseline, nested_selected], ignore_index=True)
    nested_paired = pd.concat(
        [
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=baseline_id,
                samples=args.bootstrap_samples,
                block="week",
                seed=args.bootstrap_seed,
            ),
            paired_feature_comparisons(
                comparison_input,
                baseline_feature_set=baseline_id,
                samples=args.bootstrap_samples,
                block="season",
                seed=args.bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    atomic_csv(nested_paired, output / "nested_paired_comparisons.csv")

    configuration = {
        "command": "player-model-selection",
        "budget_status": "FROZEN_BEFORE_EVALUATION",
        "raw_start_season": FROZEN_PLAYER_RAW_START_SEASON,
        "evaluation_start_season": FROZEN_PLAYER_EVALUATION_START_SEASON,
        "first_nested_test_season": FROZEN_PLAYER_FIRST_TEST_SEASON,
        "validation_seasons": FROZEN_PLAYER_VALIDATION_SEASONS,
        "profiles": list(FROZEN_PLAYER_MODEL_PROFILES),
        "ridge_alphas": list(FROZEN_PLAYER_RIDGE_ALPHAS),
        "calibration_methods": list(FROZEN_PLAYER_CALIBRATIONS),
        "min_calibration_games": FROZEN_PLAYER_MIN_CALIBRATION_GAMES,
        "baseline_candidate_id": baseline_id,
        "selection_rule": "validation_accuracy_desc,brier_score_asc,candidate_id_asc",
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "raw_configurations": int(result.raw_predictions["raw_candidate_id"].nunique()),
        "candidate_configurations": int(result.predictions["candidate_id"].nunique()),
        "outer_test_games": int(result.nested.predictions["home_cover"].notna().sum()),
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="player-model-selection",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_margin_predict(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    predictions = score_outcome_week(
        features,
        season=args.season,
        week=args.week,
        regressor=args.regressor,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        feature_profile=args.feature_profile,
        ridge_alpha=args.ridge_alpha,
        probability_method=args.probability_method,
    )
    safety = validate_outcome_prediction_card(
        predictions,
        min_edge=args.min_edge,
        expected_methods=OUTCOME_METHODS,
        expected_season=args.season,
        expected_week=args.week,
    )
    output = (
        _artifacts_root() / "margin_predictions" / f"{args.season}-week-{args.week:02d}-{run_id()}"
    )
    configuration = {
        "command": "margin-predict",
        "season": args.season,
        "week": args.week,
        "regressor": args.regressor,
        "ridge_alpha": args.ridge_alpha,
        "calibration_method": "none",
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
        "feature_profile": args.feature_profile,
        "probability_method": args.probability_method,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "games": int(predictions["game_id"].nunique()),
        "methods": sorted(predictions["method"].unique().tolist()),
        "ats_method": "market_residual",
        "prediction_safety": safety.to_dict(),
        "provenance": artifact_provenance(configuration, args.features),
    }
    atomic_csv(predictions, output / "predictions.csv")
    atomic_json(safety.to_dict(), output / "prediction_safety.json")
    ats_predictions = predictions.loc[predictions["method"].eq(metadata["ats_method"])].copy()
    if ats_predictions.empty:
        raise ValueError(f"Outcome card is missing ATS method {metadata['ats_method']!r}")
    atomic_csv(ats_predictions, output / "recommendations.csv")
    atomic_csv(build_ats_pool_card(ats_predictions), output / "pool_card.csv")
    pool_methods: list[str] = []
    for method in ("market", "fair_margin", "market_residual", "straight_up"):
        try:
            pool_card = build_straight_up_pool_card(predictions, method=method)
        except ValueError:
            continue
        atomic_csv(pool_card, output / f"straight_up_pool_{method}.csv")
        (output / f"straight_up_pool_{method}.md").write_text(
            straight_up_pool_markdown(pool_card, args.season, args.week), encoding="utf-8"
        )
        pool_methods.append(method)
    metadata["straight_up_pool_methods"] = pool_methods
    if args.line_sweep:
        sweep = score_outcome_week_line_sweep(
            features,
            season=args.season,
            week=args.week,
            regressor=args.regressor,
            min_train_games=args.min_train_games,
            feature_profile=args.feature_profile,
            ridge_alpha=args.ridge_alpha,
        )
        atomic_parquet(sweep, output / "line_sweep.parquet")
        metadata["line_sweep"] = {
            "path": "line_sweep.parquet",
            "rows": len(sweep),
            "methods": sorted(sweep["method"].unique().tolist()),
            "offsets": list(DEFAULT_LINE_SWEEP_OFFSETS),
        }
    active_model = activate_matching_ats_model(_artifacts_root(), output, metadata)
    if active_model is None:
        metadata["synchronization_status"] = "UNLINKED"
    else:
        metadata["synchronization_status"] = "SYNCHRONIZED"
        metadata["active_model_id"] = active_model["model_id"]
        metadata["historical_evaluation"] = active_model["historical_evaluation"]
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="margin-predict",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _latest_margin_prediction_dir(artifacts_root: Path) -> Path | None:
    """Most recent ``margin-predict`` artifact directory, or ``None``.

    Directory names are ``{season}-week-{week:02d}-{run_id}``, so a
    lexicographic sort is also a chronological one.
    """

    predictions_root = artifacts_root / "margin_predictions"
    if not predictions_root.is_dir():
        return None
    candidates = sorted(entry for entry in predictions_root.iterdir() if entry.is_dir())
    return candidates[-1] if candidates else None


def _cmd_market_decomposition(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    feature_columns = decomposition_feature_columns(args.feature_profile)

    opener_root = _data_root() / "market" / "historical" / "open_close" / "raw"
    opener_games_path = args.opener_games or latest_open_close_games_path(opener_root)
    opener_games = (
        pd.read_parquet(opener_games_path)
        if opener_games_path is not None and opener_games_path.is_file()
        else None
    )

    modeling_started = perf_counter()
    result = run_market_decomposition(
        features,
        feature_profile=args.feature_profile,
        start_season=args.start_season,
        end_season=args.end_season,
        ridge_alpha=args.ridge_alpha,
        min_train_games=args.min_train_games,
        noise_share_threshold=args.noise_share_threshold,
        overpriced_ratio_threshold=args.overpriced_ratio_threshold,
        opener_games=opener_games,
    )
    modeling_seconds = perf_counter() - modeling_started

    attribution: pd.DataFrame | None = None
    attribution_status = "skipped: --no-attribution"
    prediction_dir: Path | None = None
    if not args.no_attribution:
        prediction_dir = _latest_margin_prediction_dir(_artifacts_root())
        if prediction_dir is None:
            attribution_status = "skipped: no margin_predictions artifact found"
        else:
            prediction_metadata = json.loads(
                (prediction_dir / "metadata.json").read_text(encoding="utf-8")
            )
            try:
                attribution = attribute_predictions(
                    features,
                    season=int(prediction_metadata["season"]),
                    week=int(prediction_metadata["week"]),
                    feature_columns=feature_columns,
                    ridge_alpha=args.ridge_alpha,
                    min_train_games=args.min_train_games,
                )
                attribution_status = f"ok: {prediction_dir.name}"
            except (ValueError, DataContractError) as error:
                attribution_status = f"skipped: {error}"

    output = _artifacts_root() / "market_decomposition" / run_id()
    atomic_csv(result.coefficients, output / "coefficients.csv")
    atomic_csv(result.family_weights, output / "family_weights.csv")
    atomic_csv(result.classification, output / "classification.csv")
    atomic_csv(result.r_squared, output / "r_squared.csv")
    if result.opener_variant.available:
        if result.opener_variant.coefficients is not None:
            atomic_csv(result.opener_variant.coefficients, output / "opener_coefficients.csv")
        if result.opener_variant.family_weights is not None:
            atomic_csv(result.opener_variant.family_weights, output / "opener_family_weights.csv")
        if result.opener_variant.r_squared is not None:
            atomic_csv(result.opener_variant.r_squared, output / "opener_r_squared.csv")
    if attribution is not None:
        atomic_parquet(attribution, output / "attribution.parquet")

    markdown = market_decomposition_markdown(result, attribution=attribution)
    (output / "summary.md").write_text(markdown, encoding="utf-8")

    configuration = {
        "command": "market-decomposition",
        "feature_profile": args.feature_profile,
        "start_season": args.start_season,
        "end_season": args.end_season,
        "ridge_alpha": args.ridge_alpha,
        "min_train_games": args.min_train_games,
        "noise_share_threshold": args.noise_share_threshold,
        "overpriced_ratio_threshold": args.overpriced_ratio_threshold,
        "opener_games_path": str(opener_games_path) if opener_games_path else None,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "feature_count": len(feature_columns),
        "refit_weeks": result.refit_weeks,
        "thresholds": result.thresholds,
        "reconciliation": result.reconciliation,
        "opener_variant_available": result.opener_variant.available,
        "opener_variant_reason": result.opener_variant.reason,
        "opener_variant_games": result.opener_variant.games,
        "attribution_status": attribution_status,
        "attribution_source_artifact": str(prediction_dir) if prediction_dir else None,
        "timing": {
            "modeling_seconds": modeling_seconds,
            "total_seconds": perf_counter() - command_started,
        },
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="market-decomposition",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_pool_card_at_lines(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    target, margin_models = fit_margin_models_for_week(
        features,
        season=args.season,
        week=args.week,
        regressor=args.regressor,
        min_train_games=args.min_train_games,
        feature_profile=args.feature_profile,
        ridge_alpha=args.ridge_alpha,
        methods=(args.method,),
    )
    model = margin_models[args.method]

    if args.lines_file is not None:
        lines = load_lines_file(args.lines_file)
        line_source = f"lines_file:{args.lines_file}"
    elif args.use_tuesday_opener:
        quotes = load_quote_history(_data_root() / "market" / "raw")
        opener = tuesday_opener_quotes(quotes)
        if opener.empty:
            raise ValueError("No Tuesday opener quotes found in data/market/raw")
        lines = opener.rename(
            columns={"nflverse_game_id": "game_id", "opener_home_spread": "home_spread"}
        ).loc[:, ["game_id", "home_spread"]]
        lines = lines.dropna(subset=["game_id", "home_spread"])
        if lines.empty:
            raise ValueError("Tuesday opener quotes could not be matched to any game_id")
        line_source = "tuesday_opener"
    else:
        raise ValueError("Provide --lines-file or --use-tuesday-opener")

    rescored = rescore_at_lines(model, target, lines)
    card = build_ats_pool_card_at_lines(rescored, push_rule=args.push_rule)

    output = (
        _artifacts_root()
        / "pool_at_lines"
        / f"{args.season}-week-{args.week:02d}-{args.method}-{run_id()}"
    )
    atomic_csv(rescored, output / "predictions_at_lines.csv")
    atomic_csv(card, output / "pool_card.csv")
    (output / "pool_card.md").write_text(
        pool_card_at_lines_markdown(card, args.season, args.week), encoding="utf-8"
    )
    configuration = {
        "command": "pool-card-at-lines",
        "season": args.season,
        "week": args.week,
        "method": args.method,
        "regressor": args.regressor,
        "ridge_alpha": args.ridge_alpha,
        "feature_profile": args.feature_profile,
        "min_train_games": args.min_train_games,
        "push_rule": args.push_rule,
        "line_source": line_source,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "games": int(rescored["game_id"].nunique()),
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="pool-card-at-lines",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_key_number_calibration(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    mass = walk_forward_key_number_mass(
        features,
        start_season=args.start_season,
        end_season=args.end_season,
        regressor=args.regressor,
        min_train_games=args.min_train_games,
        feature_profile=args.feature_profile,
        ridge_alpha=args.ridge_alpha,
    )
    key_number_summary = summarize_key_number_calibration(mass)

    outcomes = walk_forward_outcomes(
        features,
        start_season=args.start_season,
        end_season=args.end_season,
        regressor=args.regressor,
        min_train_games=args.min_train_games,
        feature_profile=args.feature_profile,
        methods=MARGIN_DISTRIBUTION_METHODS,
        ridge_alpha=args.ridge_alpha,
    )
    reliability_frames = []
    for method, group in outcomes.predictions.groupby("method", sort=True):
        table = cover_reliability_by_line_bucket(group)
        table.insert(0, "method", method)
        reliability_frames.append(table)
    reliability = pd.concat(reliability_frames, ignore_index=True)

    output = _artifacts_root() / "key_number_calibration" / run_id()
    atomic_csv(mass, output / "key_number_mass.csv")
    atomic_csv(key_number_summary, output / "key_number_summary.csv")
    atomic_csv(reliability, output / "line_bucket_reliability.csv")
    configuration = {
        "command": "key-number-calibration",
        "start_season": args.start_season,
        "end_season": args.end_season,
        "regressor": args.regressor,
        "ridge_alpha": args.ridge_alpha,
        "feature_profile": args.feature_profile,
        "min_train_games": args.min_train_games,
        "key_numbers": list(DEFAULT_KEY_NUMBERS),
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "games": int(mass["game_id"].nunique()),
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="key-number-calibration",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    _print_json({**metadata, "artifact_directory": str(output)})


def _recommendation_markdown(predictions: pd.DataFrame, metadata: dict[str, Any]) -> str:
    table = predictions.copy()
    home_pick = table["home_cover_probability"].ge(0.5)
    table["matchup"] = table["away_team"] + " at " + table["home_team"]
    table["ats_pick"] = table["home_team"].where(home_pick, table["away_team"])
    table["ats_line"] = (-table["spread_line"]).where(home_pick, table["spread_line"])
    table["pick_probability"] = table["home_cover_probability"].where(
        home_pick, 1.0 - table["home_cover_probability"]
    )
    selected_team = table["home_team"].where(table["bet_side"].eq("HOME"), table["away_team"])
    table["paper_action"] = selected_team.where(table["bet_side"].ne("PASS"), "PASS")
    columns = [
        "gameday",
        "matchup",
        "ats_pick",
        "ats_line",
        "pick_probability",
        "paper_action",
        "edge",
    ]
    table = table.loc[:, columns].copy()
    table["gameday"] = pd.to_datetime(table["gameday"]).dt.date.astype(str)
    table["pick_probability"] = table["pick_probability"].map(lambda x: f"{x:.1%}")
    table["edge"] = table["edge"].map(lambda x: f"{x:.1%}")
    table = table.rename(
        columns={
            "gameday": "Date",
            "matchup": "Matchup",
            "ats_pick": "ATS pick",
            "ats_line": "Line",
            "pick_probability": "Model probability",
            "paper_action": "Paper action",
            "edge": "Model edge",
        }
    )
    heading = (
        f"# NFL ATS recommendations: {metadata['season']} week {metadata['week']}\n\n"
        f"Model: `{metadata['model_name']}`  \n"
        f"Training data through: `{metadata['training_max_gameday']}`  \n"
        f"Minimum edge: `{metadata['min_edge']:.1%}`\n\n"
    )
    disclaimer = (
        "\n\nProbabilities are research outputs, not guarantees. A PASS is an intentional "
        "decision when neither side clears its vig-adjusted break-even probability.\n"
    )
    return heading + table.to_markdown(index=False) + disclaimer


def _cmd_predict(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    predictions, model = score_week(
        features,
        season=args.season,
        week=args.week,
        model_name=args.model,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        feature_set=args.feature_set,
    )
    safety = validate_prediction_card(
        predictions,
        min_edge=args.min_edge,
        expected_season=args.season,
        expected_week=args.week,
        feature_columns=model.feature_columns,
    )
    created_at = datetime.now(UTC)
    identifier = f"{args.season}-week-{args.week:02d}-{run_id(created_at)}"
    output = _artifacts_root() / "predictions" / identifier
    metadata = {
        **model_metadata(model),
        "created_at_utc": created_at.isoformat(),
        "season": args.season,
        "week": args.week,
        "min_edge": args.min_edge,
        "games": len(predictions),
        "prediction_safety": safety.to_dict(),
    }
    configuration = {
        "command": "predict",
        "season": args.season,
        "week": args.week,
        "model": args.model,
        "feature_set": args.feature_set,
        "min_edge": args.min_edge,
        "min_train_games": args.min_train_games,
    }
    metadata["provenance"] = artifact_provenance(configuration, args.features)
    if args.freeze:
        frozen = freeze_forecast(
            predictions,
            metadata,
            _artifacts_root() / "prospective",
            created_at=created_at,
        )
        metadata["prospective_forecast_id"] = frozen.forecast_id
        metadata["prospective_directory"] = str(frozen.directory)
    atomic_csv(predictions, output / "recommendations.csv")
    atomic_json(safety.to_dict(), output / "prediction_safety.json")
    pool_card = build_ats_pool_card(predictions)
    atomic_csv(pool_card, output / "pool_card.csv")
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command="predict",
        metrics=metadata,
        registry_root=_registry_root(),
    )
    markdown_path = output / "recommendations.md"
    markdown_path.write_text(_recommendation_markdown(predictions, metadata), encoding="utf-8")
    (output / "pool_card.md").write_text(
        pool_card_markdown(pool_card, args.season, args.week), encoding="utf-8"
    )
    model_path = output / "model.joblib"
    temporary_model = model_path.with_suffix(".joblib.tmp")
    joblib.dump(model, temporary_model)
    temporary_model.replace(model_path)
    if model.model_name == "logistic":
        atomic_csv(logistic_coefficients(model), output / "coefficients.csv")
    _print_json({**metadata, "artifact_directory": str(output)})


def _rotation_family_payload(registry: Registry, name: str) -> dict[str, Any]:
    status = registry_status(registry)
    families = [family for family in status["families"] if family["name"] == name]
    return {
        "registry": str(default_registry_path()),
        "family": families[0],
        "grade_pools": status["grade_pools"],
        "season_usage": status["season_usage"],
    }


def _cmd_rotation_status(_: argparse.Namespace) -> None:
    registry = load_registry()
    _print_json({"registry": str(default_registry_path()), **registry_status(registry)})


def _cmd_weak_signals_status(args: argparse.Namespace) -> None:
    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    signals = sorted(registry.signals.values(), key=lambda signal: signal.name)
    filtered = [
        signal for signal in signals if args.classification in (None, signal.classification)
    ]
    families = family_overlap_warnings(filtered)
    _print_json(
        {
            "registry": str(path),
            "recorded": len(signals),
            "families": families["families"],
            "overlap_warnings": families,
            "measurement_coherence_problems": weak_signal_coherence_problems(filtered),
            "signals": [
                {
                    "name": signal.name,
                    "classification": signal.classification,
                    "league": signal.league,
                    "seasons": list(signal.seasons),
                    "effect": signal.effect,
                    "effect_units": signal.effect_units,
                    "standard_error": signal.resolved_standard_error(),
                    "favours_candidate": signal.favours_candidate,
                    "family": signal.family,
                    "source": signal.source,
                }
                for signal in filtered
            ],
        }
    )


def _cmd_weak_signals_record(args: argparse.Namespace) -> None:
    """Record one below-power result so it stops being re-litigated in prose.

    This command exists because its absence was the actual defect. The registry
    had ``status`` and ``pool`` but no way in, so recording a signal meant
    hand-writing Python against the internal API -- and every session took the
    cheaper path of writing a prose verdict instead. A standing rule with no
    ergonomic path is a rule that silently stops being followed: the ledger sat
    at three entries while a documented 13 of 27 discarded families belonged in
    it.
    """

    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    interval = None
    if args.interval_low is not None and args.interval_high is not None:
        interval = (float(args.interval_low), float(args.interval_high))
    signal = WeakSignal(
        name=args.name,
        recorded_at=args.recorded_at or datetime.now(UTC).date().isoformat(),
        description=args.description,
        source=args.source,
        effect=float(args.effect),
        effect_units=args.effect_units,
        classification=args.classification,
        league=args.league,
        seasons=(int(args.season_start), int(args.season_end)),
        standard_error=args.standard_error,
        interval=interval,
        probability_positive=args.probability_positive,
        sample_games=args.sample_games,
        sample_blocks=args.sample_blocks,
        reliability=args.reliability,
        family=args.family,
        classification_evidence=args.classification_evidence,
        closing_ground=args.closing_ground,
        notes=args.notes,
        plain_summary=args.plain_summary,
        category=args.category,
    )
    registry = record_signal(registry, signal, replace=args.replace)
    save_weak_signals(registry, path)
    # Both fields are optional (475 pre-existing rows carry neither), but a
    # NEW record that skips them is the ledger's raw-description/Uncategorised
    # fallback silently choosing itself -- warn out loud on stderr so this
    # never gets buried in the JSON stdout a caller might be parsing.
    if not args.plain_summary:
        print(
            f"warning: {signal.name!r} recorded with no --plain-summary; the public "
            "Signal Ledger page will show its raw description instead of plain "
            "English for this row",
            file=sys.stderr,
        )
    if not args.category:
        print(
            f"warning: {signal.name!r} recorded with no --category; it will render "
            "under 'Uncategorised' on the public Signal Ledger page",
            file=sys.stderr,
        )
    _print_json(
        {
            "registry": str(path),
            "recorded": signal.name,
            "classification": signal.classification,
            "effect": signal.effect,
            "effect_units": signal.effect_units,
            "favours_candidate": signal.favours_candidate,
            "total_signals": len(registry.signals),
        }
    )


def _cmd_weak_signals_pool(args: argparse.Namespace) -> None:
    """Ask whether the accumulated below-power pile is worth one combined look."""

    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    report = combination_report(
        registry,
        league=args.league,
        effect_units=args.effect_units,
        method=args.method,
    )
    _print_json({"registry": str(path), **report})


def _cmd_weak_signals_retag_units(args: argparse.Namespace) -> None:
    """Correct a mis-tagged ``effect_units`` on one entry without touching anything else.

    Exists because some entries were forced into a unit that did not match
    what was measured (a correlation coefficient, an MAE/Brier/log-loss
    *improvement*), with the true sign convention explained only in prose
    inside ``notes`` -- exactly the note a pooler will not read. This changes
    only the unit and appends one audit line; effect, interval,
    classification, and closing_ground are untouched (AGENTS.md forbids
    silently rewriting a recorded measurement, and a unit correction is not a
    new one).
    """

    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    previous_units = (
        registry.signals[args.name].effect_units if args.name in registry.signals else None
    )
    registry = retag_effect_units(
        registry,
        args.name,
        effect_units=args.effect_units,
        reason=args.reason,
    )
    save_weak_signals(registry, path)
    signal = registry.signals[args.name]
    _print_json(
        {
            "registry": str(path),
            "retagged": signal.name,
            "previous_effect_units": previous_units,
            "effect_units": signal.effect_units,
            "notes": signal.notes,
        }
    )


def _cmd_weak_signals_set_reliability(args: argparse.Namespace) -> None:
    """Attach a measured split-half reliability to one entry, touching nothing else.

    Most entries carry ``reliability: null``, which leaves one of only two
    admissible closing grounds neither usable nor rulable-out. This writes the
    measured number (plus its interval, method and artifact path, as one audit
    line in ``notes`` -- the schema has no interval field) and leaves effect,
    interval, classification, closing_ground and source byte-identical. It
    does NOT reclassify: a low reliability is a candidate for the
    ``no_split_half_reliability`` ground, and acting on it stays a separate,
    explicit decision.
    """

    path = weak_signal_registry_path()
    registry = load_weak_signals(path)
    previous = registry.signals[args.name].reliability if args.name in registry.signals else None
    registry = set_reliability(
        registry,
        args.name,
        reliability=args.reliability,
        reliability_low=args.reliability_low,
        reliability_high=args.reliability_high,
        method=args.method,
        source=args.source,
        reason=args.reason,
    )
    save_weak_signals(registry, path)
    signal = registry.signals[args.name]
    _print_json(
        {
            "registry": str(path),
            "name": signal.name,
            "previous_reliability": previous,
            "reliability": signal.reliability,
            "reliability_interval": [args.reliability_low, args.reliability_high],
            "method": args.method,
            "measured_from": args.source,
            "classification": signal.classification,
            "closing_ground": signal.closing_ground,
            "notes": signal.notes,
        }
    )


def _cmd_rotation_declare(args: argparse.Namespace) -> None:
    path = default_registry_path()
    inherits = tuple(part.strip() for part in str(args.inherits or "").split(",") if part.strip())
    registry = declare_family(
        load_registry(path),
        args.name,
        description=args.description,
        grade=args.grade,
        inherits=inherits,
        acknowledges_mined_2018_2025=args.acknowledge_mined,
    )
    save_registry(registry, path)
    _print_json({"declared": args.name, **_rotation_family_payload(registry, args.name)})


def _cmd_rotation_assign(args: argparse.Namespace) -> None:
    path = default_registry_path()
    if args.stratified:
        if args.size is not None:
            raise ValueError(
                "--size does not apply to --stratified windows; a stratified "
                "window is always a two-leg pair (docs/era_stratified_windows_proposal.md)"
            )
        registry = assign_stratified_window(load_registry(path), args.name)
    else:
        registry = assign_window(load_registry(path), args.name, size=args.size)
    save_registry(registry, path)
    _print_json({"assigned": args.name, **_rotation_family_payload(registry, args.name)})


def _cmd_rotation_record(args: argparse.Namespace) -> None:
    path = default_registry_path()
    interval = None
    if args.interval_low is not None and args.interval_high is not None:
        interval = (float(args.interval_low), float(args.interval_high))
    leg_effects = None if args.leg_effects is None else json.loads(args.leg_effects)
    registry = record_look(
        load_registry(path),
        args.name,
        artifact=args.artifact,
        verdict=args.verdict,
        probability_positive=args.probability_positive,
        closing_ground=args.closing_ground,
        effect=args.effect,
        effect_units=args.effect_units,
        interval=interval,
        standard_error=args.standard_error,
        sample_blocks=args.sample_blocks,
        leg_effects=leg_effects,
        notes=args.notes,
        replace_existing=args.replace,
    )
    save_registry(registry, path)
    _print_json({"recorded": args.name, **_rotation_family_payload(registry, args.name)})


def _cmd_anytime_compare(args: argparse.Namespace) -> None:
    predictions = pd.read_parquet(args.predictions)
    if args.feature_set_column != "feature_set":
        predictions = predictions.rename(columns={args.feature_set_column: "feature_set"})
    trace = paired_anytime_comparisons(
        predictions,
        baseline_feature_set=args.baseline_feature_set,
        metric=args.metric,
        block=args.block,
        alpha=args.alpha,
        prior_variance=args.prior_variance,
        target_games=args.target_games,
        per_game_variance_proxy=args.per_game_variance_proxy,
        intraclass_correlation=args.intraclass_correlation,
    )
    summary = anytime_summary(trace)
    output = _artifacts_root() / "anytime" / run_id()
    atomic_csv(trace, output / "trace.csv")
    atomic_csv(summary, output / "summary.csv")
    metadata = {
        "command": "anytime compare",
        "predictions": str(args.predictions),
        "baseline_feature_set": args.baseline_feature_set,
        "metric": args.metric,
        "block": args.block,
        "alpha": args.alpha,
        "target_games": args.target_games,
        "per_game_variance_proxy": args.per_game_variance_proxy,
        "intraclass_correlation": args.intraclass_correlation,
        "artifact": str(output),
        "summary": summary.to_dict(orient="records"),
    }
    atomic_json(metadata, output / "metadata.json")
    _print_json(metadata)


def _cmd_weekly_run(args: argparse.Namespace) -> None:
    _print_json(
        run_weekly(
            season=args.season,
            week=args.week,
            data_root=_data_root(),
            artifacts_root=_artifacts_root(),
            refresh_player_data=args.refresh_player_data,
            skip_ingest=args.skip_ingest,
            skip_prospective=args.skip_prospective,
            skip_drift=args.skip_drift,
            record_decisions=args.record_decisions,
            dry_run=args.dry_run,
        )
    )


# --- shared argument-family helpers (hyg-cli phase 1) ----------------------
#
# build_parser used to repeat identical add_argument blocks across dozens of
# commands. These helpers register the repeated families verbatim at their
# original call sites inside build_parser, so per-command flag order -- and
# therefore --help output and parse results -- is unchanged. Commands whose
# defaults differ pass them through explicitly.


def _add_features_arg(
    parser: argparse.ArgumentParser,
    filename: str = "game_features.parquet",
    *,
    help_text: str | None = None,
) -> None:
    """Register the shared --features feature-table flag under data/processed."""
    parser.add_argument(
        "--features",
        type=Path,
        default=_data_root() / "processed" / filename,
        help=help_text,
    )


def _add_bootstrap_args(
    parser: argparse.ArgumentParser,
    samples: int = 2_000,
    seed: int = 20260812,
) -> None:
    """Register the shared bootstrap-uncertainty pair."""
    parser.add_argument("--bootstrap-samples", type=int, default=samples)
    parser.add_argument("--bootstrap-seed", type=int, default=seed)


def _add_season_range_args(
    parser: argparse.ArgumentParser,
    start_default: int | None,
    end_default: int | None,
) -> None:
    """Register the shared --start-season/--end-season pair."""
    parser.add_argument("--start-season", type=int, default=start_default)
    parser.add_argument("--end-season", type=int, default=end_default)


def _add_season_week_args(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    """Register the shared --season/--week pair (required, or prospective defaults)."""
    if required:
        parser.add_argument("--season", type=int, required=True)
        parser.add_argument("--week", type=int, required=True)
    else:
        parser.add_argument("--season", type=int, default=2026)
        parser.add_argument("--week", type=int, default=1)


def _add_snapshot_args(parser: argparse.ArgumentParser, *specs: tuple[str, str]) -> None:
    """Register "(label) snapshot ID; defaults to latest" flags as (flag, label) pairs."""
    for flag, label in specs:
        head = f"{label} snapshot ID" if label else "snapshot ID"
        parser.add_argument(flag, help=f"{head}; defaults to latest")


def _add_include_postseason_arg(parser: argparse.ArgumentParser) -> None:
    """Register the shared --include-postseason flag."""
    parser.add_argument(
        "--include-postseason",
        action="store_true",
        help=_INCLUDE_POSTSEASON_HELP,
    )


def _add_ewm_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared EWM smoothing trio."""
    parser.add_argument("--ewm-span", type=int, default=8)
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--offseason-retention", type=float, default=DEFAULT_OFFSEASON_RETENTION)


def _add_regressor_args(parser: argparse.ArgumentParser, *, choices: bool = True) -> None:
    """Register the shared --regressor/--ridge-alpha pair."""
    if choices:
        parser.add_argument("--regressor", choices=("ridge", "hgb"), default="ridge")
    else:
        parser.add_argument("--regressor", default="ridge")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)


def _add_feature_profile_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    help_text: str | None = None,
) -> None:
    """Register the shared --feature-profile choice over MARGIN_FEATURE_PROFILES."""
    parser.add_argument(
        "--feature-profile",
        choices=MARGIN_FEATURE_PROFILES,
        default=default,
        help=help_text,
    )


def _add_board_destination_args(
    parser: argparse.ArgumentParser,
    *,
    legacy_flag: str,
) -> None:
    """Register the duplicated board/site destination pair for the publish commands."""
    parser.add_argument(
        legacy_flag,
        type=Path,
        default=Path("docs/index.html"),
        help="deprecated alias for --site-destination; a file path is reduced to its directory",
    )
    parser.add_argument(
        "--site-destination",
        type=Path,
        default=None,
        help="directory to write the three public pages into (default: docs/)",
    )


def _add_player_feature_tuning_args(parser: argparse.ArgumentParser) -> None:
    """Register the seven tuning flags shared by the three player-feature builders."""
    parser.add_argument("--decision-hours", type=int, default=24)
    parser.add_argument("--role-span", type=int, default=8)
    parser.add_argument("--qb-span", type=int, default=12)
    parser.add_argument("--qb-min-dropbacks", type=int, default=20)
    parser.add_argument("--offseason-retention", type=float, default=0.75)
    parser.add_argument("--value-span", type=int, default=16)
    parser.add_argument("--value-prior-snaps", type=float, default=200.0)


def build_parser() -> argparse.ArgumentParser:
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(
        prog="nfl-ats",
        description="Leak-safe NFL against-the-spread research pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="show runtime and data health")
    doctor.set_defaults(handler=_cmd_doctor)

    arrests_ingest = subparsers.add_parser(
        "ingest-player-arrests",
        help="build a fresh, complete point-in-time player-arrests snapshot",
    )
    arrests_ingest.add_argument("--snapshot", type=str, default=None)
    arrests_ingest.add_argument("--max-pages", type=int, default=None)
    arrests_ingest.add_argument("--delay-seconds", type=float, default=None)
    arrests_ingest.set_defaults(handler=_cmd_ingest_player_arrests)

    publish = subparsers.add_parser(
        "publish-predictions",
        help="write the synchronized active weekly ATS card into GitHub Markdown",
    )
    publish.add_argument("--destination", type=Path, default=Path("CURRENT_PREDICTIONS.md"))
    publish.add_argument("--readme", type=Path, default=Path("README.md"))
    publish.add_argument(
        "--with-board",
        action="store_true",
        default=True,
        help=(
            "regenerate the public GitHub Pages site into docs/. ON by default since "
            "2026-08-19 so the "
            "served site can never lag the published card; retained as an explicit "
            "flag only so existing invocations keep working"
        ),
    )
    publish.add_argument(
        "--no-board",
        dest="with_board",
        action="store_false",
        help="skip regenerating the public site (rehearsal publishes that must not touch docs/)",
    )
    _add_board_destination_args(publish, legacy_flag="--board-destination")
    publish.add_argument(
        "--record-decisions",
        action="store_true",
        help=(
            "append this card's pre-kickoff picks to the paper-decision CLV ledger. Off "
            "by default -- recording is a deliberate act for the real weekly lock, not "
            "something an ordinary/rehearsal publish should do. record_paper_decisions "
            "also refuses to write when this week's earliest kickoff is more than "
            "RECORDING_LOCK_WINDOW away, so passing this flag outside the real lock "
            "week still does not reach the ledger."
        ),
    )
    publish.set_defaults(handler=_cmd_publish_predictions)

    refresh_picks = subparsers.add_parser(
        "refresh-picks",
        help=(
            "recompute one week's picks with current data at the frozen Tuesday grading "
            "lines (POL-11, docs/late_week_refresh.md); a second, opt-in step run any "
            "time between the Tuesday publish and each game's own deadline"
        ),
    )
    _add_season_week_args(refresh_picks, required=True)
    refresh_picks.add_argument(
        "--features",
        type=Path,
        default=None,
        help=(
            "current-week feature table (default: the active model's own card-path "
            "table under data/processed/, matching weekly-run's card path)"
        ),
    )
    refresh_picks.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    refresh_picks.add_argument(
        "--record-decisions",
        action="store_true",
        help=(
            "append this pass's changed, eligible picks to the append-only "
            "pick-revision ledger, AND every eligible game's injury_signal_refresh_tilt "
            "challenger reading (both arms) to its own ledger. Off by default, exactly "
            "like publish-predictions --record-decisions. Both recorders also refuse to "
            "write when this week's earliest kickoff is more than RECORDING_LOCK_WINDOW "
            "away, and record_plan never revises a game whose own deadline (its kickoff, "
            "or the week's Sunday 4:00 PM ET cap if earlier) has already passed."
        ),
    )
    refresh_picks.add_argument(
        "--publish-card",
        action="store_true",
        help=(
            "additively label a 'Late-week refresh' section onto the published card "
            "(--destination) listing only changed picks; never rewrites the Tuesday "
            "section publish-predictions wrote"
        ),
    )
    refresh_picks.add_argument("--destination", type=Path, default=Path("CURRENT_PREDICTIONS.md"))
    refresh_picks.add_argument(
        "--note",
        type=str,
        default="",
        help=(
            "free-text label for which weekly pass this is (e.g. 'thursday_afternoon', "
            "'sunday_morning_final'), stored in each revision's reason field and shown "
            "in the card section"
        ),
    )
    refresh_picks.set_defaults(handler=_cmd_refresh_picks)

    publish_board = subparsers.add_parser(
        "publish-board",
        help=(
            "render the public GitHub Pages site (ATS Terminal): index.html, model.html, "
            "and findings.html into docs/"
        ),
    )
    _add_board_destination_args(publish_board, legacy_flag="--destination")
    publish_board.set_defaults(handler=_cmd_publish_board)

    handoff = subparsers.add_parser(
        "handoff",
        help="refresh the tracked new-session handoff from Git and local model state",
    )
    handoff.add_argument("--destination", type=Path, default=Path("HANDOFF.md"))
    handoff.add_argument(
        "--check",
        action="store_true",
        help="verify tracked handoff freshness without changing files",
    )
    handoff.set_defaults(handler=_cmd_handoff)

    ingest = subparsers.add_parser("ingest", help="download an immutable nflverse snapshot")
    _add_season_range_args(ingest, 2009, current_year)
    ingest.add_argument(
        "--stats-end-season",
        type=int,
        help="last team-stat season; use the prior year before current-season stats exist",
    )
    ingest.set_defaults(handler=_cmd_ingest)

    smoke = subparsers.add_parser(
        "smoke-source", help="check current nflverse source availability and schemas"
    )
    smoke.add_argument("--schedule-season", type=int, default=current_year)
    smoke.add_argument("--stats-season", type=int, default=current_year - 1)
    smoke.set_defaults(handler=_cmd_smoke_source)

    pbp_ingest = subparsers.add_parser(
        "pbp-ingest", help="download a versioned, season-partitioned nflverse PBP snapshot"
    )
    _add_season_range_args(pbp_ingest, 2009, current_year - 1)
    _add_include_postseason_arg(pbp_ingest)
    pbp_ingest.set_defaults(handler=_cmd_pbp_ingest)

    depth_ingest = subparsers.add_parser(
        "depth-ingest", help="archive timestamped nflverse quarterback depth charts"
    )
    _add_season_range_args(depth_ingest, current_year - 1, current_year - 1)
    depth_ingest.set_defaults(handler=_cmd_depth_ingest)

    player_ingest = subparsers.add_parser(
        "player-ingest",
        help="archive injuries, earlier-week rosters, and lagged player snaps",
    )
    player_ingest.add_argument("--injury-start-season", type=int, default=2009)
    player_ingest.add_argument("--injury-end-season", type=int, default=2024)
    player_ingest.add_argument("--roster-start-season", type=int, default=2009)
    player_ingest.add_argument("--roster-end-season", type=int, default=current_year - 1)
    player_ingest.add_argument("--snap-start-season", type=int, default=2013)
    player_ingest.add_argument("--snap-end-season", type=int, default=current_year - 1)
    _add_include_postseason_arg(player_ingest)
    player_ingest.set_defaults(handler=_cmd_player_ingest)

    player_value_ingest = subparsers.add_parser(
        "player-value-ingest",
        help="archive weekly nflverse player production for lagged value estimates",
    )
    _add_season_range_args(player_value_ingest, 2009, current_year - 1)
    _add_include_postseason_arg(player_value_ingest)
    player_value_ingest.set_defaults(handler=_cmd_player_value_ingest)

    participation_ingest = subparsers.add_parser(
        "participation-ingest",
        help="archive season-partitioned nflverse player participation",
    )
    _add_season_range_args(participation_ingest, 2016, current_year - 1)
    participation_ingest.set_defaults(handler=_cmd_participation_ingest)

    role_actions_fetch = subparsers.add_parser(
        "role-actions-fetch",
        help="archive nflverse weekly player action counts for the XLG-04 replication",
    )
    role_actions_fetch.add_argument(
        "--seasons", type=int, nargs="+", default=list(range(2013, 2026))
    )
    _add_include_postseason_arg(role_actions_fetch)
    role_actions_fetch.set_defaults(handler=_cmd_role_actions_fetch)

    cfb_ingest = subparsers.add_parser(
        "cfb-ingest",
        help="download an immutable college-football source snapshot (XLG-02)",
    )
    cfb_ingest.add_argument(
        "--source",
        required=True,
        choices=(
            "schedules",
            "lines",
            "pbp",
            "rosters",
            "participants",
            "espn-betting",
            "draft-picks",
            "returning-production",
            "recruiting-teams",
            "recruiting-players",
            "usage",
            "portal",
        ),
        help="which audited CFB source to snapshot (CFBD API sources need CFBD_API_KEY)",
    )
    cfb_ingest.add_argument(
        "--start-season",
        type=int,
        help="defaults to the source's first usable audited season",
    )
    cfb_ingest.add_argument("--end-season", type=int, default=current_year - 1)
    cfb_ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve pinned upstream files and sizes without downloading data",
    )
    cfb_ingest.set_defaults(handler=_cmd_cfb_ingest)

    cfb_summary = subparsers.add_parser(
        "cfb-summary", help="summarize the latest local CFB source snapshots"
    )
    cfb_summary.set_defaults(handler=_cmd_cfb_summary)

    cfb_build_features = subparsers.add_parser(
        "cfb-build-features",
        help="build the canonical CFB benchmark game table with pregame state (XLG-03)",
    )
    _add_season_range_args(cfb_build_features, CFB_BENCHMARK_START_SEASON, CFB_BENCHMARK_END_SEASON)
    _add_ewm_args(cfb_build_features)
    cfb_build_features.set_defaults(handler=_cmd_cfb_build_features)

    cfb_benchmark = subparsers.add_parser(
        "cfb-benchmark",
        help="run the frozen CFB-only market-residual walk-forward benchmark (XLG-03)",
    )
    _add_features_arg(cfb_benchmark, "cfb_game_features.parquet")
    cfb_benchmark.add_argument("--start-season", type=int, default=CFB_BENCHMARK_START_SEASON)
    cfb_benchmark.add_argument("--end-season", type=int, default=CFB_BENCHMARK_END_SEASON)
    cfb_benchmark.add_argument("--min-train-games", type=int, default=CFB_BENCHMARK_MIN_TRAIN_GAMES)
    cfb_benchmark.add_argument("--ridge-alpha", type=float, default=CFB_BENCHMARK_RIDGE_ALPHA)
    _add_bootstrap_args(
        cfb_benchmark,
        samples=CFB_BENCHMARK_BOOTSTRAP_SAMPLES,
        seed=CFB_BENCHMARK_BOOTSTRAP_SEED,
    )
    cfb_benchmark.set_defaults(handler=_cmd_cfb_benchmark)

    cfb_sensitivity = subparsers.add_parser(
        "cfb-sensitivity-audit",
        help="positive-control sensitivity audit of the CFB benchmark evaluator (XLG-03)",
    )
    _add_features_arg(cfb_sensitivity, "cfb_game_features.parquet")
    cfb_sensitivity.add_argument("--benchmark-predictions", type=Path, required=True)
    cfb_sensitivity.add_argument("--replicas", type=int, default=CFB_AUDIT_REPLICAS)
    cfb_sensitivity.add_argument(
        "--bootstrap-samples", type=int, default=CFB_AUDIT_BOOTSTRAP_SAMPLES
    )
    cfb_sensitivity.add_argument("--seed", type=int, default=CFB_AUDIT_SEED)
    cfb_sensitivity.set_defaults(handler=_cmd_cfb_sensitivity_audit)

    cfb_role_replication = subparsers.add_parser(
        "cfb-role-replication",
        help="run the predeclared XLG-04 cross-league role-delivery replication",
    )
    cfb_role_replication.add_argument(
        "--cfb-features",
        type=Path,
        default=_data_root() / "processed" / "cfb_game_features.parquet",
    )
    cfb_role_replication.add_argument(
        "--role-actions-snapshot", help="role-actions snapshot ID; defaults to latest"
    )
    cfb_role_replication.set_defaults(handler=_cmd_cfb_role_replication)

    cfb_absence_separation = subparsers.add_parser(
        "cfb-absence-separation",
        help="descriptive departure-vs-temporary-absence study on CFB role holders "
        "(participation only; informs the role-feature predeclaration)",
    )
    cfb_absence_separation.add_argument(
        "--cfb-features",
        type=Path,
        default=_data_root() / "processed" / "cfb_game_features.parquet",
    )
    cfb_absence_separation.set_defaults(handler=_cmd_cfb_absence_separation)

    cfb_role_benchmark_parser = subparsers.add_parser(
        "cfb-role-benchmark",
        help="score the predeclared role-continuity family against the frozen XLG-03 "
        "benchmark (three matched arms, paired week/season-blocked intervals)",
    )
    cfb_role_benchmark_parser.add_argument(
        "--cfb-features",
        type=Path,
        default=_data_root() / "processed" / "cfb_game_features.parquet",
    )
    _add_bootstrap_args(cfb_role_benchmark_parser, seed=20260817)
    cfb_role_benchmark_parser.set_defaults(handler=_cmd_cfb_role_benchmark)

    cfb_variance_parser = subparsers.add_parser(
        "cfb-variance-benchmark",
        help="score the predeclared MOD-16 conditional-variance distribution against the "
        "pooled residual distribution on the frozen XLG-03 benchmark (same picks, "
        "paired probability-calibration intervals)",
    )
    cfb_variance_parser.add_argument(
        "--cfb-features",
        type=Path,
        default=_data_root() / "processed" / "cfb_game_features.parquet",
    )
    _add_bootstrap_args(cfb_variance_parser, seed=20260817)
    cfb_variance_parser.set_defaults(handler=_cmd_cfb_variance_benchmark)

    odds_ingest = subparsers.add_parser(
        "odds-ingest", help="archive timestamped NFL quotes from The Odds API"
    )
    _add_features_arg(odds_ingest)
    odds_ingest.add_argument("--regions", default="us")
    odds_ingest.add_argument("--markets", default="spreads,h2h")
    odds_ingest.add_argument("--bookmakers")
    odds_ingest.set_defaults(handler=_cmd_odds_ingest)

    odds_summary = subparsers.add_parser(
        "odds-summary", help="summarize locally archived point-in-time quotes"
    )
    odds_summary.set_defaults(handler=_cmd_odds_summary)

    tiebreaker = subparsers.add_parser(
        "tiebreaker",
        help="final-score guess for the pool's tiebreaker game (the week's last kickoff)",
    )
    tiebreaker.add_argument("--season", type=int, help="default: the next upcoming week's season")
    tiebreaker.add_argument("--week", type=int, help="default: the next upcoming week")
    tiebreaker.add_argument(
        "--game-id", dest="game_id", help="explicit nflverse game id, overrides season/week"
    )
    tiebreaker.set_defaults(handler=_cmd_tiebreaker)

    totals_backtest = subparsers.add_parser(
        "totals-backtest",
        help="walk-forward over/under regime: ridge on the market total's residual",
    )
    _add_features_arg(totals_backtest)
    totals_backtest.add_argument(
        "--min-train-games",
        dest="min_train_games",
        type=int,
        default=DEFAULT_MIN_TRAIN_GAMES,
        help="warm-up floor before a week is scored (default: production's constant)",
    )
    _add_bootstrap_args(totals_backtest, seed=20260901)
    totals_backtest.set_defaults(handler=_cmd_totals_backtest)

    odds_backfill = subparsers.add_parser(
        "odds-backfill",
        help="backfill historical point-in-time NFL snapshots from The Odds API",
    )
    _add_features_arg(odds_backfill)
    odds_backfill.add_argument("--start-season", type=int, required=True)
    odds_backfill.add_argument("--end-season", type=int, required=True)
    odds_backfill.add_argument("--regions", default="us")
    odds_backfill.add_argument(
        "--weeks", help="comma-separated week filter, e.g. 1,2 (default: every scheduled week)"
    )
    odds_backfill.add_argument(
        "--labels",
        help=f"comma-separated decision labels from: {', '.join(DECISION_LABELS)}",
    )
    odds_backfill.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact call and credit plan without spending any credits",
    )
    odds_backfill.add_argument(
        "--budget",
        type=int,
        help="refuse to start when the planned cost exceeds this many credits",
    )
    odds_backfill.add_argument(
        "--quota-floor",
        type=int,
        default=DEFAULT_QUOTA_FLOOR,
        help="stop before any call that would leave fewer provider credits than this",
    )
    odds_backfill.add_argument(
        "--resume",
        action="store_true",
        help="skip planned snapshots already present in the store and continue",
    )
    odds_backfill.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    odds_backfill.set_defaults(handler=_cmd_odds_backfill)

    market_backfill = subparsers.add_parser(
        "market-backfill",
        help="download and audit the free historical NFL closing-line archive",
    )
    _add_features_arg(market_backfill)
    market_backfill.set_defaults(handler=_cmd_market_backfill)

    open_close_backfill = subparsers.add_parser(
        "market-open-close-backfill",
        help="download the free 2025 NFL opener and multi-book closing sample",
    )
    _add_features_arg(open_close_backfill)
    open_close_backfill.set_defaults(handler=_cmd_market_open_close_backfill)

    clv_score = subparsers.add_parser(
        "clv-score", help="score a predictions parquet for closing-line value (CLV)"
    )
    clv_score.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="parquet with columns game_id, side (HOME/AWAY), decision_label",
    )
    _add_features_arg(clv_score)
    clv_score.add_argument(
        "--capture-kind",
        default="live",
        help="market store capture_kind to score against (live or historical_backfill)",
    )
    _add_bootstrap_args(clv_score, seed=20260816)
    clv_score.set_defaults(handler=_cmd_clv_score)

    clv_ledger = subparsers.add_parser(
        "clv-ledger",
        help="record the published weekly card's paper decisions and score the ledger's "
        "closing-line value (MKT-04); pending games score once their close exists",
    )
    _add_features_arg(clv_ledger)
    clv_ledger.add_argument(
        "--skip-record",
        action="store_true",
        help="score the existing ledger without recording the currently published card",
    )
    _add_bootstrap_args(clv_ledger, seed=20260816)
    clv_ledger.set_defaults(handler=_cmd_clv_ledger)

    prospective_record = subparsers.add_parser(
        "prospective-record",
        help="append a registered challenger's pre-kickoff weekly picks to the prospective "
        "ledger (POL-10); the active model's own picks are recorded by publish-predictions",
    )
    prospective_record.add_argument(
        "--challenger",
        required=True,
        help="challenger_id from artifacts/prospective/challengers.json",
    )
    _add_season_week_args(prospective_record)
    prospective_record.add_argument(
        "--artifact",
        type=Path,
        help="margin-predict artifact directory to record from; by default the newest card "
        "for the season/week whose configuration fingerprint matches the registration",
    )
    prospective_record.set_defaults(handler=_cmd_prospective_record)

    prospective_score = subparsers.add_parser(
        "prospective-score",
        help="settle every recorded prospective pick against results and report forced-pick "
        "ATS accuracy at the recorded line (primary) and the close (secondary)",
    )
    _add_features_arg(prospective_score)
    prospective_score.add_argument(
        "--start-season",
        type=int,
        default=2026,
        help="first season to score; defaults to the prospective era (2026+), because "
        "earlier seasons are historical backtests, not pre-kickoff decisions",
    )
    prospective_score.add_argument(
        "--skip-challengers",
        action="store_true",
        help="score only the active model's ledger",
    )
    _add_bootstrap_args(prospective_score, seed=20260817)
    prospective_score.set_defaults(handler=_cmd_prospective_score)

    drift_report = subparsers.add_parser(
        "drift-report",
        help="RWB-12 drift monitoring: feature, missingness, probability and calibration "
        "drift for one published week versus recent history (read-only telemetry)",
    )
    _add_season_week_args(drift_report, required=True)
    _add_features_arg(drift_report)
    _add_feature_profile_arg(
        drift_report,
        default="player",
        help_text=(
            "which card namespace to monitor; must match the margin-predict run being "
            "monitored, since challenger cards share the same artifacts tree"
        ),
    )
    drift_report.add_argument("--probability-method", default="gaussian")
    drift_report.add_argument(
        "--reference-weeks",
        type=int,
        default=6,
        help="completed weeks strictly before the target week used as the reference window",
    )
    drift_report.add_argument(
        "--calibration-recent-weeks",
        type=int,
        default=4,
        help="most recent settled weeks compared against prior settled history",
    )
    drift_report.set_defaults(handler=_cmd_drift_report)

    clv_pilot = subparsers.add_parser(
        "clv-pilot",
        help="run the predeclared MKT-06 close-prediction pilot (frozen train/validate/test split)",
    )
    _add_features_arg(clv_pilot)
    clv_pilot.add_argument("--capture-kind", default=HISTORICAL_CAPTURE_KIND)
    clv_pilot.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    _add_bootstrap_args(clv_pilot, seed=20260816)
    clv_pilot.add_argument("--threshold", type=float, default=0.5)
    _add_feature_profile_arg(
        clv_pilot,
        help_text=(
            "override the active-model feature profile used for the residual-at-opener feature "
            "(default: read artifacts/active_ats_model.json, or feature_profile=player if absent)"
        ),
    )
    _add_regressor_args(clv_pilot, choices=False)
    clv_pilot.set_defaults(handler=_cmd_clv_pilot)

    clv_sign_test = subparsers.add_parser(
        "clv-sign-test",
        help="sign(active-model fair margin - opener) vs sign(close - opener), all seasons",
    )
    _add_features_arg(clv_sign_test)
    clv_sign_test.add_argument("--capture-kind", default=HISTORICAL_CAPTURE_KIND)
    clv_sign_test.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    _add_feature_profile_arg(clv_sign_test)
    _add_regressor_args(clv_sign_test, choices=False)
    clv_sign_test.set_defaults(handler=_cmd_clv_sign_test)

    opener_evaluation_parser = subparsers.add_parser(
        "opener-evaluation",
        help="grade the frozen active model against Tuesday openers vs closes on every "
        "archived paired game (the pool primary-goal measurement; one predeclared look)",
    )
    _add_features_arg(
        opener_evaluation_parser,
        "game_features_player.parquet",
        help_text="must match the active model's feature profile (player)",
    )
    opener_evaluation_parser.add_argument(
        "--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES
    )
    _add_feature_profile_arg(
        opener_evaluation_parser,
        help_text=("override the active-model feature profile (default: read the active manifest)"),
    )
    _add_regressor_args(opener_evaluation_parser, choices=False)
    _add_bootstrap_args(opener_evaluation_parser, seed=20260817)
    opener_evaluation_parser.set_defaults(handler=_cmd_opener_evaluation)

    predict_close = subparsers.add_parser(
        "predict-close",
        help="predict one week's closing spreads with the frozen MKT-06 pilot model "
        "(writes the Week Board's close_predictions artifact; reports blocked and writes "
        "nothing until that week's live Tuesday opener capture exists)",
    )
    _add_features_arg(
        predict_close,
        "game_features_player.parquet",
        help_text=(
            "must match the active model's feature profile (player) so the "
            "opener-time residual feature can be rebuilt"
        ),
    )
    predict_close.add_argument(
        "--season", type=int, help="target season; defaults to the earliest unplayed week"
    )
    predict_close.add_argument(
        "--week", type=int, help="target week; defaults to the earliest unplayed week"
    )
    predict_close.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    _add_feature_profile_arg(
        predict_close,
        help_text=(
            "override the active-model feature profile used for the residual-at-opener feature "
            "(default: read artifacts/active_ats_model.json, or feature_profile=player if absent)"
        ),
    )
    _add_regressor_args(predict_close, choices=False)
    predict_close.set_defaults(handler=_cmd_predict_close)

    feature_parser = subparsers.add_parser(
        "build-features", help="build the canonical pregame feature table"
    )
    _add_snapshot_args(feature_parser, ("--snapshot", ""))
    _add_ewm_args(feature_parser)
    feature_parser.add_argument("--graph-half-life", type=float, default=8.0)
    feature_parser.add_argument("--graph-ridge-alpha", type=float, default=8.0)
    feature_parser.add_argument("--graph-min-games", type=int, default=16)
    feature_parser.add_argument(
        "--include-postseason",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "add WC/DIV/CON/SB rows for weekly playoff serving; regular-season "
            "rows are bit-identical either way and training stays REG-only"
        ),
    )
    feature_parser.set_defaults(handler=_cmd_build_features)

    pbp_features = subparsers.add_parser(
        "build-pbp-features", help="add leak-safe PBP states to the canonical feature table"
    )
    _add_snapshot_args(pbp_features, ("--snapshot", "PBP"))
    _add_features_arg(pbp_features)
    _add_ewm_args(pbp_features)
    pbp_features.add_argument("--opponent-half-life", type=float, default=16.0)
    pbp_features.add_argument("--opponent-ridge-alpha", type=float, default=10.0)
    pbp_features.add_argument("--opponent-min-games", type=int, default=64)
    pbp_features.set_defaults(handler=_cmd_build_pbp_features)

    qb_features = subparsers.add_parser(
        "build-qb-features",
        help="attach point-in-time expected starters and strictly prior QB states",
    )
    _add_snapshot_args(qb_features, ("--pbp-snapshot", "PBP"), ("--depth-snapshot", "depth-chart"))
    _add_features_arg(qb_features, "game_features_pbp.parquet")
    qb_features.add_argument("--decision-hours", type=int, default=24)
    qb_features.add_argument("--max-depth-age-days", type=int, default=14)
    qb_features.add_argument("--ewm-span", type=int, default=12)
    qb_features.add_argument("--min-dropbacks", type=int, default=50)
    qb_features.add_argument("--offseason-retention", type=float, default=0.75)
    qb_features.set_defaults(handler=_cmd_build_qb_features)

    player_features = subparsers.add_parser(
        "build-player-features",
        help="add leak-safe expected-lineup, injury, QB, and continuity states",
    )
    _add_snapshot_args(
        player_features,
        ("--player-snapshot", "player"),
        ("--player-value-snapshot", "player-value"),
        ("--pbp-snapshot", "PBP"),
    )
    _add_features_arg(player_features, "game_features_pbp.parquet")
    player_features.add_argument(
        "--destination",
        type=Path,
        default=_data_root() / "processed" / "game_features_player.parquet",
    )
    _add_player_feature_tuning_args(player_features)
    player_features.set_defaults(handler=_cmd_build_player_features)

    participation_features = subparsers.add_parser(
        "build-participation-features",
        help="add frozen season-lagged player participation values to injury states",
    )
    _add_snapshot_args(
        participation_features,
        ("--player-snapshot", "player"),
        ("--player-value-snapshot", "player-value"),
        ("--participation-snapshot", "participation"),
        ("--pbp-snapshot", "PBP"),
    )
    _add_features_arg(participation_features, "game_features_pbp.parquet")
    participation_features.add_argument(
        "--ratings-destination",
        type=Path,
        default=_data_root() / "processed" / "player_participation_ratings.parquet",
    )
    participation_features.add_argument(
        "--destination",
        type=Path,
        default=_data_root() / "processed" / "game_features_player_participation.parquet",
    )
    _add_player_feature_tuning_args(participation_features)
    participation_features.set_defaults(handler=_cmd_build_participation_features)

    availability_features = subparsers.add_parser(
        "build-learned-availability-features",
        help="replace hand-authored injury weights with season-lagged empirical rates",
    )
    _add_snapshot_args(
        availability_features,
        ("--player-snapshot", "player"),
        ("--player-value-snapshot", "player-value"),
        ("--pbp-snapshot", "PBP"),
    )
    _add_features_arg(availability_features, "game_features_pbp.parquet")
    availability_features.add_argument(
        "--rates-destination",
        type=Path,
        default=_data_root() / "processed" / "player_availability_rates.parquet",
    )
    availability_features.add_argument(
        "--evaluation-destination",
        type=Path,
        default=_data_root() / "processed" / "player_availability_evaluation.csv",
    )
    availability_features.add_argument(
        "--destination",
        type=Path,
        default=_data_root() / "processed" / "game_features_player_learned_availability.parquet",
    )
    _add_player_feature_tuning_args(availability_features)
    availability_features.set_defaults(handler=_cmd_build_learned_availability_features)

    backtest = subparsers.add_parser("backtest", help="run expanding weekly evaluation")
    _add_features_arg(backtest)
    _add_season_range_args(backtest, 2018, None)
    backtest.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    backtest.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="full")
    backtest.add_argument("--min-edge", type=float, default=0.02)
    backtest.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    backtest.add_argument("--initial-bankroll", type=float, default=100.0)
    backtest.add_argument("--kelly-multiplier", type=float, default=0.25)
    backtest.add_argument("--max-bet-fraction", type=float, default=0.02)
    backtest.add_argument("--max-week-fraction", type=float, default=0.10)
    backtest.add_argument("--probability-haircut", type=float, default=0.0)
    backtest.add_argument("--bankroll-paths", type=int, default=5_000)
    backtest.add_argument("--bankroll-seed", type=int, default=20260812)
    _add_bootstrap_args(backtest)
    backtest.set_defaults(handler=_cmd_backtest)

    nested = subparsers.add_parser(
        "nested-evaluate",
        help="select configurations on prior seasons and score untouched outer seasons",
    )
    _add_features_arg(nested)
    nested.add_argument("--first-test-season", type=int, default=2018)
    nested.add_argument("--last-test-season", type=int, default=current_year - 1)
    nested.add_argument("--validation-seasons", type=int, default=2)
    nested.add_argument("--selection-metric", choices=SELECTION_METRICS, default="brier_score")
    nested.add_argument(
        "--candidates",
        default=format_candidates(DEFAULT_EVALUATION_CANDIDATES),
        help="comma-separated frozen search budget of model:feature_set entries",
    )
    nested.add_argument("--min-edge", type=float, default=0.02)
    nested.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    _add_bootstrap_args(nested)
    nested.add_argument("--dependence-permutations", type=int, default=1_000)
    nested.add_argument("--dependence-seed", type=int, default=20260812)
    nested.set_defaults(handler=_cmd_nested_evaluate)

    dependence = subparsers.add_parser(
        "dependence-audit", help="measure serial dependence in saved prediction errors"
    )
    dependence.add_argument("--predictions", type=Path, required=True)
    dependence.add_argument("--permutations", type=int, default=1_000)
    dependence.add_argument("--seed", type=int, default=20260812)
    dependence.set_defaults(handler=_cmd_dependence_audit)

    experiment = subparsers.add_parser(
        "experiment", help="feature-set comparisons and the declarative experiment pipeline"
    )
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)

    experiment_compare = experiment_commands.add_parser(
        "compare", help="compare feature sets with identical walk-forward windows"
    )
    _add_features_arg(experiment_compare)
    experiment_compare.add_argument("--start-season", type=int, default=2022)
    experiment_compare.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    experiment_compare.add_argument("--feature-sets", default=",".join(DEFAULT_EXPERIMENT_SETS))
    experiment_compare.add_argument(
        "--baseline-feature-set",
        choices=tuple(FEATURE_SETS),
        help="paired comparison baseline; defaults to the first requested feature set",
    )
    _add_bootstrap_args(experiment_compare)
    experiment_compare.add_argument("--min-edge", type=float, default=0.02)
    experiment_compare.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    experiment_compare.set_defaults(handler=_cmd_experiment)

    experiment_run = experiment_commands.add_parser(
        "run",
        help=(
            "run a declarative experiment spec: reliability check -> screen -> bootstrap -> "
            "mechanical classification -> registry record -> provenance stamp"
        ),
    )
    experiment_run.add_argument("spec", type=Path, help="path to a JSON experiment spec")
    _add_features_arg(experiment_run, help_text="override the feature table")
    experiment_run.add_argument(
        "--market-root",
        type=Path,
        default=_data_root() / "market" / "raw",
        help="historical odds snapshot root used by population.grade='opener'",
    )
    experiment_run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the would-be registry record and exit; writes nothing to disk",
    )
    experiment_run.add_argument(
        "--replace",
        action="store_true",
        help="allow overwriting an existing registry entry of the same name",
    )
    experiment_run.set_defaults(handler=_cmd_experiment_run)

    experiment_verify = experiment_commands.add_parser(
        "verify",
        help=(
            "check each registry row's linked artifact_directory resolves on disk "
            "and surface path/identity inconsistencies (read-only)"
        ),
    )
    experiment_verify.add_argument(
        "--artifacts-root",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "extra root to resolve relative artifact_directory paths against "
            "(repeatable); defaults to $NFL_ATS_ARTIFACTS_DIR or ./artifacts. "
            "Absolute stored paths are always checked as-is."
        ),
    )
    experiment_verify.add_argument(
        "--require-links",
        action="store_true",
        help=(
            "exit non-zero if any linked artifact is missing. Leave unset in CI "
            "on a fresh clone: artifacts/ is gitignored and legitimately absent, "
            "so a missing link is not by itself a defect -- the row's hashes are."
        ),
    )
    experiment_verify.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit machine-readable JSON instead of a human table",
    )
    experiment_verify.set_defaults(handler=_cmd_experiment_verify)

    margin_backtest = subparsers.add_parser(
        "margin-backtest",
        help="compare market, fair-margin, residual-margin, straight-up, and direct ATS models",
    )
    _add_features_arg(margin_backtest)
    margin_backtest.add_argument("--start-season", type=int, default=2018)
    _add_regressor_args(margin_backtest)
    _add_feature_profile_arg(margin_backtest, default="base")
    margin_backtest.add_argument(
        "--methods",
        default=",".join(OUTCOME_METHODS),
        help=f"comma-separated subset of: {','.join(OUTCOME_METHODS)}",
    )
    margin_backtest.add_argument("--min-edge", type=float, default=0.02)
    margin_backtest.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    margin_backtest.add_argument(
        "--probability-method",
        choices=RESIDUAL_SMOOTHING_METHODS,
        # Default UNCHANGED (2026-08-19): this backs historical/research
        # backtests broadly, so it stays "ecdf" unless a caller explicitly
        # asks for "gaussian" -- e.g. to build the matching margins/
        # evaluation a --probability-method gaussian margin-predict forecast
        # needs to activate SYNCHRONIZED (nfl_ats.active_model).
        default="ecdf",
        help="how home_cover_probability is read off the out-of-time residual "
        "sample for every margin-model method scored (market_residual, fair_margin)",
    )
    _add_bootstrap_args(margin_backtest, samples=1_000)
    margin_backtest.set_defaults(handler=_cmd_margin_backtest)

    player_ablation = subparsers.add_parser(
        "player-ablation",
        help="compare player feature families on the residual-margin ATS model",
    )
    _add_features_arg(player_ablation, "game_features_player.parquet")
    player_ablation.add_argument("--start-season", type=int, default=2018)
    _add_regressor_args(player_ablation)
    player_ablation.add_argument("--profiles", default=",".join(DEFAULT_PLAYER_PROFILE_SETS))
    player_ablation.add_argument(
        "--baseline-profile", choices=MARGIN_FEATURE_PROFILES, default="base"
    )
    player_ablation.add_argument("--min-edge", type=float, default=0.02)
    player_ablation.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    _add_bootstrap_args(player_ablation)
    player_ablation.add_argument("--first-nested-test-season", type=int, default=2020)
    player_ablation.add_argument("--validation-seasons", type=int, default=2)
    player_ablation.set_defaults(handler=_cmd_player_ablation)

    participation_ablation = subparsers.add_parser(
        "participation-ablation",
        help="run the frozen player-value versus participation-value comparison",
    )
    _add_features_arg(participation_ablation, "game_features_player_participation.parquet")
    _add_bootstrap_args(participation_ablation, seed=20260813)
    participation_ablation.set_defaults(handler=_cmd_participation_ablation)

    availability_ablation = subparsers.add_parser(
        "availability-ablation",
        help="run the frozen learned-versus-fixed availability comparison",
    )
    availability_ablation.add_argument(
        "--baseline-features",
        type=Path,
        default=_data_root() / "processed" / "game_features_player_value.parquet",
    )
    availability_ablation.add_argument(
        "--learned-features",
        type=Path,
        default=_data_root() / "processed" / "game_features_player_learned_availability.parquet",
    )
    _add_bootstrap_args(availability_ablation, seed=20260813)
    availability_ablation.set_defaults(handler=_cmd_availability_ablation)

    player_selection = subparsers.add_parser(
        "player-model-selection",
        help="run the frozen nested player profile, Ridge, and calibration budget",
    )
    player_selection.add_argument(
        "--features",
        type=Path,
        default=_data_root() / "processed" / "game_features_player_value.parquet",
    )
    player_selection.add_argument("--min-edge", type=float, default=0.02)
    # This command runs the FROZEN selection, so its default is pinned to the
    # value that selection was scored under. It must not drift with the live
    # default, or re-running it would stop reproducing the recorded artifact.
    player_selection.add_argument(
        "--min-train-games", type=int, default=FROZEN_PLAYER_MIN_TRAIN_GAMES
    )
    _add_bootstrap_args(player_selection, seed=20260813)
    player_selection.set_defaults(handler=_cmd_player_model_selection)

    margin_predict = subparsers.add_parser(
        "margin-predict", help="score one week with fair-margin and outcome models"
    )
    _add_season_week_args(margin_predict, required=True)
    _add_features_arg(margin_predict)
    _add_regressor_args(margin_predict)
    _add_feature_profile_arg(margin_predict, default="base")
    margin_predict.add_argument("--min-edge", type=float, default=0.02)
    margin_predict.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    margin_predict.add_argument(
        "--probability-method",
        choices=RESIDUAL_SMOOTHING_METHODS,
        # PROMOTED DEFAULT (MOD-08, 2026-08-19, docs/smooth_cdf_mapping.md):
        # this is the SOLE production weekly-forecast entry point, so its
        # default must match nfl_ats.outcomes.score_outcome_week's own
        # promoted default -- pinned together by
        # tests/test_probability_method_promotion.py.
        default="gaussian",
        help="how home_cover_probability is read off the out-of-time residual "
        "sample: 'ecdf' is the pre-2026-08-19 raw empirical CDF, 'gaussian' is "
        "the promoted MOD-08 default",
    )
    margin_predict.add_argument(
        "--line-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write a per-game line-sweep confidence table (line_sweep.parquet)",
    )
    margin_predict.set_defaults(handler=_cmd_margin_predict)

    market_decomposition = subparsers.add_parser(
        "market-decomposition",
        help=(
            "decompose margin/spread/residual ridge weights by feature family "
            "-- what the market prices vs. what reality prices"
        ),
    )
    _add_features_arg(market_decomposition, "game_features_player.parquet")
    _add_feature_profile_arg(market_decomposition, default="player")
    _add_season_range_args(market_decomposition, DEFAULT_START_SEASON, DEFAULT_END_SEASON)
    market_decomposition.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    market_decomposition.add_argument(
        "--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES
    )
    market_decomposition.add_argument(
        "--noise-share-threshold", type=float, default=DEFAULT_NOISE_SHARE_THRESHOLD
    )
    market_decomposition.add_argument(
        "--overpriced-ratio-threshold", type=float, default=DEFAULT_OVERPRICED_RATIO_THRESHOLD
    )
    market_decomposition.add_argument(
        "--opener-games",
        type=Path,
        default=None,
        help=(
            "games.parquet from an open/close snapshot (default: feature-detect the latest "
            "under data/market/historical/open_close/raw)"
        ),
    )
    market_decomposition.add_argument(
        "--no-attribution",
        action="store_true",
        help="skip per-game attribution for the latest margin-predict artifact",
    )
    market_decomposition.set_defaults(handler=_cmd_market_decomposition)

    pool_card_at_lines = subparsers.add_parser(
        "pool-card-at-lines",
        help="score an ATS pool card at externally supplied home spreads",
    )
    _add_season_week_args(pool_card_at_lines, required=True)
    _add_features_arg(pool_card_at_lines)
    pool_card_at_lines.add_argument(
        "--method", choices=MARGIN_DISTRIBUTION_METHODS, default="fair_margin"
    )
    _add_regressor_args(pool_card_at_lines)
    _add_feature_profile_arg(pool_card_at_lines, default="base")
    pool_card_at_lines.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    pool_card_at_lines.add_argument("--lines-file", type=Path, default=None)
    pool_card_at_lines.add_argument(
        "--use-tuesday-opener",
        action="store_true",
        help="derive lines from the latest captured Tuesday opener in data/market/raw",
    )
    pool_card_at_lines.add_argument("--push-rule", choices=PUSH_RULES, default="loss")
    pool_card_at_lines.set_defaults(handler=_cmd_pool_card_at_lines)

    key_number_calibration = subparsers.add_parser(
        "key-number-calibration",
        help="leak-safe walk-forward report on key-number mass and line-bucket reliability",
    )
    key_number_calibration.add_argument("--start-season", type=int, required=True)
    key_number_calibration.add_argument("--end-season", type=int, default=None)
    _add_features_arg(key_number_calibration)
    _add_regressor_args(key_number_calibration)
    _add_feature_profile_arg(key_number_calibration, default="base")
    key_number_calibration.add_argument(
        "--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES
    )
    key_number_calibration.set_defaults(handler=_cmd_key_number_calibration)

    predict = subparsers.add_parser("predict", help="score one season/week")
    _add_season_week_args(predict, required=True)
    _add_features_arg(predict)
    predict.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    predict.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="market_context")
    predict.add_argument("--min-edge", type=float, default=0.02)
    predict.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    predict.add_argument(
        "--freeze",
        action="store_true",
        help="archive an immutable forecast after verifying every game is pre-kickoff",
    )
    predict.set_defaults(handler=_cmd_predict)

    rotation = subparsers.add_parser(
        "rotation",
        help="manage the per-family confirmation-window registry "
        "(docs/rotation_registry.md); a look is one look and is always recorded",
    )
    rotation_commands = rotation.add_subparsers(dest="rotation_command", required=True)

    weak_signals = subparsers.add_parser(
        "weak-signals",
        help="track effects too small for their own test to resolve, and pool them "
        "(docs/pool_edge_plan.md); a below-power result is kept, never deleted",
    )
    weak_signal_commands = weak_signals.add_subparsers(dest="weak_signal_command", required=True)

    weak_signals_status = weak_signal_commands.add_parser(
        "status", help="list every recorded signal, its effect, direction and classification"
    )
    weak_signals_status.add_argument(
        "--classification",
        choices=tuple(CLASSIFICATIONS),
        default=None,
        help="show only signals of one kind (default: all)",
    )
    weak_signals_status.set_defaults(handler=_cmd_weak_signals_status)

    weak_signals_record = weak_signal_commands.add_parser(
        "record",
        help="record one below-power result so it is kept instead of re-litigated; "
        "an interval containing zero is NOT a negative and belongs here",
    )
    weak_signals_record.add_argument("--name", required=True)
    weak_signals_record.add_argument("--description", required=True)
    weak_signals_record.add_argument(
        "--source", required=True, help="artifact path or doc that records the measurement"
    )
    weak_signals_record.add_argument("--effect", type=float, required=True)
    weak_signals_record.add_argument("--effect-units", choices=tuple(EFFECT_UNITS), required=True)
    weak_signals_record.add_argument(
        "--classification", choices=tuple(CLASSIFICATIONS), required=True
    )
    weak_signals_record.add_argument("--league", choices=tuple(LEAGUES), required=True)
    weak_signals_record.add_argument("--season-start", type=int, required=True)
    weak_signals_record.add_argument("--season-end", type=int, required=True)
    weak_signals_record.add_argument("--standard-error", type=float, default=None)
    weak_signals_record.add_argument("--interval-low", type=float, default=None)
    weak_signals_record.add_argument("--interval-high", type=float, default=None)
    weak_signals_record.add_argument("--probability-positive", type=float, default=None)
    weak_signals_record.add_argument("--sample-games", type=int, default=None)
    weak_signals_record.add_argument("--sample-blocks", type=int, default=None)
    weak_signals_record.add_argument(
        "--reliability",
        type=float,
        default=None,
        help=(
            "split-half reliability of the underlying trait. AGENTS.md makes "
            "this the decisive field: an unreliable trait is refuted because no "
            "sample size rescues it, so a signal recorded without it cannot be "
            "adjudicated later"
        ),
    )
    weak_signals_record.add_argument(
        "--family",
        default=None,
        help=(
            "measurement family this cell belongs to (e.g. its screening battery). "
            "Family members share windows and are correlated, not independent votes; "
            "omit to have one inferred from the name"
        ),
    )
    weak_signals_record.add_argument(
        "--classification-evidence",
        default="",
        help="why this classification and not one of the other two",
    )
    weak_signals_record.add_argument(
        "--closing-ground",
        choices=tuple(
            ground for grounds in WEAK_SIGNAL_CLOSING_GROUNDS.values() for ground in grounds
        ),
        default=None,
        help="required for a terminal classification: the admissible AGENTS.md "
        "ground the closure stands on. An interval containing zero is NOT one; "
        "that outcome is unresolved_below_power",
    )
    weak_signals_record.add_argument(
        "--plain-summary",
        default=None,
        help=(
            "one or two sentences a football fan with no statistics background "
            "can read on their own, naming the situation AND what the rule does "
            "about it. Optional but recorded rows without one render their raw "
            "--description on the public Signal Ledger page instead"
        ),
    )
    weak_signals_record.add_argument(
        "--category",
        choices=tuple(WEAK_SIGNAL_CATEGORIES),
        default=None,
        help=(
            "one reader-facing bucket for the public Signal Ledger page. "
            "Optional but an omitted category renders under 'Uncategorised'"
        ),
    )
    weak_signals_record.add_argument("--notes", default="")
    weak_signals_record.add_argument("--recorded-at", default=None, help="default: today")
    weak_signals_record.add_argument(
        "--replace", action="store_true", help="overwrite an existing signal of this name"
    )
    weak_signals_record.set_defaults(handler=_cmd_weak_signals_record)

    weak_signals_pool = weak_signal_commands.add_parser(
        "pool",
        help="sign test plus inverse-variance pooling across the unresolved pile, "
        "with shared-season warnings; says whether a combined look is worth a window",
    )
    weak_signals_pool.add_argument("--league", choices=tuple(LEAGUES), default=None)
    weak_signals_pool.add_argument("--effect-units", choices=tuple(EFFECT_UNITS), default=None)
    weak_signals_pool.add_argument(
        "--method",
        choices=("random", "fixed"),
        default="random",
        help="random effects (default) inflates the variance by observed heterogeneity",
    )
    weak_signals_pool.set_defaults(handler=_cmd_weak_signals_pool)

    weak_signals_retag_units = weak_signal_commands.add_parser(
        "retag-units",
        help="correct a mis-tagged effect_units on one existing entry; changes ONLY "
        "the unit and appends an audit note to it, nothing else",
    )
    weak_signals_retag_units.add_argument(
        "--name", required=True, help="the recorded signal's name"
    )
    weak_signals_retag_units.add_argument(
        "--effect-units", choices=tuple(EFFECT_UNITS), required=True
    )
    weak_signals_retag_units.add_argument(
        "--reason", required=True, help="why the original unit was wrong"
    )
    weak_signals_retag_units.set_defaults(handler=_cmd_weak_signals_retag_units)

    weak_signals_set_reliability = weak_signal_commands.add_parser(
        "set-reliability",
        help="attach a measured split-half reliability (plus its interval, method and "
        "artifact) to one existing entry; changes ONLY the reliability field and "
        "appends an audit note, and never reclassifies the entry",
    )
    weak_signals_set_reliability.add_argument(
        "--name", required=True, help="the recorded signal's name"
    )
    weak_signals_set_reliability.add_argument(
        "--reliability",
        type=float,
        required=True,
        help="point estimate, a correlation in [-1, 1]; an unmeasurable reliability is "
        "reported as unmeasured, never written here as a number",
    )
    weak_signals_set_reliability.add_argument(
        "--reliability-low", type=float, required=True, help="95%% interval lower bound"
    )
    weak_signals_set_reliability.add_argument(
        "--reliability-high", type=float, required=True, help="95%% interval upper bound"
    )
    weak_signals_set_reliability.add_argument(
        "--method",
        required=True,
        help="which quantity was measured (e.g. 'team-season odd/even-week split-half of "
        "<trait>, Spearman-Brown corrected'); a trait's reliability and a flag's exposure "
        "reliability are different quantities and must not be compared",
    )
    weak_signals_set_reliability.add_argument(
        "--source", required=True, help="artifact path holding the measurement"
    )
    weak_signals_set_reliability.add_argument(
        "--reason", required=True, help="why this reliability applies to this entry"
    )
    weak_signals_set_reliability.set_defaults(handler=_cmd_weak_signals_set_reliability)

    rotation_status = rotation_commands.add_parser(
        "status", help="print every family, its windows, remaining pool capacity, and usage"
    )
    rotation_status.set_defaults(handler=_cmd_rotation_status)

    rotation_declare = rotation_commands.add_parser(
        "declare", help="declare a family BEFORE any confirmation run"
    )
    rotation_declare.add_argument("--name", required=True)
    rotation_declare.add_argument("--description", required=True)
    rotation_declare.add_argument("--grade", choices=tuple(GRADE_POOLS), required=True)
    rotation_declare.add_argument(
        "--inherits", help="comma-separated families whose spent windows this family inherits"
    )
    rotation_declare.add_argument(
        "--acknowledge-mined",
        action="store_true",
        help=f"acknowledge the {MINED_SEASONS[0]}-{MINED_SEASONS[1]} multiplicity ledger; "
        "required for any window intersecting those seasons",
    )
    rotation_declare.set_defaults(handler=_cmd_rotation_declare)

    rotation_assign = rotation_commands.add_parser(
        "assign", help="assign the earliest eligible window block to a family"
    )
    rotation_assign.add_argument("--name", required=True)
    rotation_assign.add_argument(
        "--size",
        type=int,
        help=f"window size in seasons ({MIN_WINDOW_SIZE}-{MAX_WINDOW_SIZE}); "
        "defaults to the grade's default; not valid with --stratified",
    )
    rotation_assign.add_argument(
        "--stratified",
        action="store_true",
        help="assign a two-leg era-stratified window instead of a contiguous block "
        f"({STRATIFIED_GRADE}-graded families only; "
        "docs/era_stratified_windows_proposal.md)",
    )
    rotation_assign.set_defaults(handler=_cmd_rotation_assign)

    rotation_record = rotation_commands.add_parser(
        "record", help="record the look and spend the family's assigned window"
    )
    rotation_record.add_argument("--name", required=True)
    rotation_record.add_argument("--artifact", required=True)
    rotation_record.add_argument("--verdict", choices=VERDICTS, required=True)
    rotation_record.add_argument(
        "--probability-positive",
        type=float,
        required=True,
        help="fraction of blocked resamples favouring the candidate",
    )
    rotation_record.add_argument(
        "--closing-ground",
        choices=tuple(
            ground for grounds in WEAK_SIGNAL_CLOSING_GROUNDS.values() for ground in grounds
        ),
        default=None,
        help="required for closed_negative: the admissible AGENTS.md ground the "
        "closure stands on; an interval containing zero is NOT one and that "
        "verdict is 'unresolved'",
    )
    rotation_record.add_argument(
        "--effect",
        type=float,
        default=None,
        help="point estimate, positive favours the candidate (requires --effect-units)",
    )
    rotation_record.add_argument("--effect-units", choices=tuple(EFFECT_UNITS), default=None)
    rotation_record.add_argument("--interval-low", type=float, default=None)
    rotation_record.add_argument("--interval-high", type=float, default=None)
    rotation_record.add_argument("--standard-error", type=float, default=None)
    rotation_record.add_argument("--sample-blocks", type=int, default=None)
    rotation_record.add_argument(
        "--leg-effects",
        default=None,
        help="JSON list of per-leg magnitudes, required for a stratified window: "
        '\'[{"season": 2013, "effect": 1.2, "probability_positive": 0.7, '
        '"sample_blocks": 12}, ...]\' -- one entry per leg, sharing --effect-units '
        "(owner's binding refinement: era variation is a change in magnitude, "
        "never collapsed into the pooled read alone)",
    )
    rotation_record.add_argument("--notes", default="")
    rotation_record.add_argument(
        "--replace",
        action="store_true",
        help=(
            "correct the latest spent window only; requires its exact existing artifact and "
            "preserves assignment/spend provenance"
        ),
    )
    rotation_record.set_defaults(handler=_cmd_rotation_record)

    anytime = subparsers.add_parser(
        "anytime", help="anytime-valid (continuous-monitoring) paired comparisons"
    )
    anytime_commands = anytime.add_subparsers(dest="anytime_command", required=True)
    anytime_compare = anytime_commands.add_parser(
        "compare",
        help=(
            "confidence-sequence/e-value trace for a paired feature-set comparison; same "
            "input contract as experiments.paired_feature_comparisons, valid under peeking "
            "after every week/season instead of only at a single predeclared sample size"
        ),
    )
    anytime_compare.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="parquet with feature_set/game_id/season/week/home_cover/"
        "home_cover_probability columns (or --feature-set-column to rename one)",
    )
    anytime_compare.add_argument(
        "--feature-set-column",
        default="feature_set",
        help="column to rename to 'feature_set' first, e.g. 'method' for a cfb-benchmark "
        "predictions.parquet",
    )
    anytime_compare.add_argument("--baseline-feature-set", required=True)
    anytime_compare.add_argument(
        "--metric", choices=ANYTIME_METRICS, default="accuracy_improvement"
    )
    anytime_compare.add_argument("--block", choices=("week", "season"), default="week")
    anytime_compare.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    anytime_compare.add_argument(
        "--target-games",
        type=int,
        default=DEFAULT_TARGET_GAMES,
        help="horizon the default mixing variance is tuned for (ignored if --prior-variance "
        "is passed); default matches the rotation registry's own 3-season window",
    )
    anytime_compare.add_argument(
        "--prior-variance",
        type=float,
        default=None,
        help="override the mixing variance directly instead of deriving it from --target-games",
    )
    anytime_compare.add_argument(
        "--per-game-variance-proxy",
        type=float,
        default=1.0,
        help="upper bound on one game's own variance; 1.0 is Hoeffding's worst case for a "
        "[-1, 1] variable and needs no assumption. See docs/anytime_valid.md for the "
        "measured, less-conservative value this project uses (0.55) and why",
    )
    anytime_compare.add_argument(
        "--intraclass-correlation",
        type=float,
        default=0.0,
        help="Kish design-effect correlation, 0-1; 0.0 (independence -- disjoint teams, no "
        "shared outcome mechanism) is this project's standing decision, not an estimate. "
        "1.0 (every game in a block moves together) is the assumption-free worst case, "
        "kept available for stress-testing. See docs/anytime_valid.md for the argument "
        "and why an unmeasured 0.10 pad and an auto-estimated value were both rejected",
    )
    anytime_compare.set_defaults(handler=_cmd_anytime_compare)

    weekly = subparsers.add_parser(
        "weekly-run",
        help="run the whole Tuesday sequence in order, fail-closed, and publish",
    )
    _add_season_week_args(weekly, required=True)
    weekly.add_argument(
        "--refresh-player-data",
        action="store_true",
        help="build the enriched tables from the latest player/PBP snapshots instead of "
        "the snapshot ids pinned in the current production manifests",
    )
    weekly.add_argument(
        "--skip-ingest",
        action="store_true",
        help="reuse the latest nflverse snapshot instead of downloading a fresh one",
    )
    weekly.add_argument(
        "--skip-prospective",
        action="store_true",
        help="skip steps 8-11, which produce, record and settle the prospective 2026 "
        "challenger evidence; they run after the publish and never block the card",
    )
    weekly.add_argument(
        "--skip-drift",
        action="store_true",
        help="skip step 13 (drift-report), which writes a read-only drift-monitoring "
        "telemetry artifact after the publish; it never blocks the card",
    )
    weekly.add_argument(
        "--record-decisions",
        action="store_true",
        help=(
            "the real weekly lock: append this card's picks to the paper-decision ledger "
            "(step 7) and the challenger's picks to the prospective ledger (step 10). Off "
            "by default so an ordinary/rehearsal weekly-run does not touch either ledger; "
            "pass this only for the actual Tuesday lock. Both underlying recorders also "
            "refuse to write when this week's earliest kickoff is more than "
            "RECORDING_LOCK_WINDOW away, so this flag alone cannot reach the ledger outside "
            "the real lock week either."
        ),
    )
    weekly.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan and run nothing",
    )
    weekly.set_defaults(handler=_cmd_weekly_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
