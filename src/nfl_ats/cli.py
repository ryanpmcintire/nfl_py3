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
from nfl_ats.backtest import score_week, walk_forward_backtest
from nfl_ats.constants import FEATURE_SETS
from nfl_ats.data import check_nflverse_contract, fetch_nflverse
from nfl_ats.dependence import prediction_dependence_audit
from nfl_ats.evaluation import (
    DEFAULT_EVALUATION_CANDIDATES,
    SELECTION_METRICS,
    format_candidates,
    nested_walk_forward_evaluation,
    parse_candidates,
)
from nfl_ats.experiments import (
    DEFAULT_EXPERIMENT_SETS,
    DEFAULT_PLAYER_PROFILE_SETS,
    FROZEN_PLAYER_CALIBRATIONS,
    FROZEN_PLAYER_EVALUATION_START_SEASON,
    FROZEN_PLAYER_FIRST_TEST_SEASON,
    FROZEN_PLAYER_MIN_CALIBRATION_GAMES,
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
from nfl_ats.handoff import check_session_handoff, write_session_handoff
from nfl_ats.historical_market import fetch_historical_market_snapshot
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id
from nfl_ats.margin import MARGIN_FEATURE_PROFILES
from nfl_ats.market_data import (
    attach_nflverse_game_ids,
    fetch_odds_api_from_environment,
    load_quote_history,
    parse_odds_api_response,
    spread_consensus,
    write_market_snapshot,
)
from nfl_ats.model_card import build_model_card, model_card_markdown
from nfl_ats.modeling import MODEL_NAMES, logistic_coefficients, model_metadata
from nfl_ats.open_close_market import fetch_open_close_snapshot
from nfl_ats.outcomes import (
    OUTCOME_METHODS,
    outcome_bootstrap_intervals,
    score_outcome_week,
    walk_forward_outcomes,
)
from nfl_ats.pbp import (
    PBP_FEATURE_VERSION,
    enrich_with_pbp_features,
    fetch_pbp_snapshot,
    latest_pbp_snapshot,
    load_pbp_snapshot,
)
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
from nfl_ats.players import (
    PLAYER_FEATURE_VERSION,
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
from nfl_ats.prospective import freeze_forecast
from nfl_ats.provenance import artifact_provenance, sha256_file
from nfl_ats.publishing import publish_active_predictions
from nfl_ats.quarterbacks import (
    depth_snapshot_from_root,
    enrich_with_qb_features,
    fetch_depth_snapshot,
    latest_depth_snapshot,
    load_depth_snapshot,
)
from nfl_ats.reporting import block_bootstrap_intervals
from nfl_ats.snapshots import (
    Snapshot,
    describe_snapshot,
    latest_snapshot,
    load_snapshot,
    snapshot_from_root,
)


def _data_root() -> Path:
    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


def _artifacts_root() -> Path:
    return Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts"))


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


def _cmd_dashboard(args: argparse.Namespace) -> None:
    from nfl_ats.dashboard import launch_dashboard

    raise SystemExit(
        launch_dashboard(address=args.address, port=args.port, headless=args.no_browser)
    )


def _cmd_publish_predictions(args: argparse.Namespace) -> None:
    result = publish_active_predictions(
        _artifacts_root(),
        destination=args.destination,
        readme_path=args.readme,
    )
    _print_json(result)


def _cmd_handoff(args: argparse.Namespace) -> None:
    if args.check:
        result = check_session_handoff(Path.cwd(), _artifacts_root(), args.destination)
    else:
        result = write_session_handoff(
            Path.cwd(),
            _artifacts_root(),
            args.destination,
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


def _cmd_pbp_ingest(args: argparse.Namespace) -> None:
    if args.end_season < args.start_season:
        raise ValueError("end-season cannot be earlier than start-season")
    snapshot = fetch_pbp_snapshot(
        list(range(args.start_season, args.end_season + 1)),
        _data_root() / "pbp" / "raw",
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "seasons": list(snapshot.seasons),
            "rows": manifest["rows"],
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
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "contract_version": manifest["contract_version"],
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
    )
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "snapshot_id": snapshot.snapshot_id,
            "directory": str(snapshot.root),
            "contract_version": manifest["contract_version"],
            "file": manifest["file"],
            "availability_contract": manifest["availability_contract"],
        }
    )


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
        "rows": len(features),
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
    atomic_json(provenance, output / "run.json")
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
    atomic_json(artifact_provenance(configuration, args.features), output / "run.json")
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
    atomic_json(metadata, output / "metadata.json")
    _print_json({**metadata, "artifact_directory": str(output)})


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
    atomic_json(metadata, output / "metadata.json")
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
    atomic_json(metadata, output / "metadata.json")
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
    active_model = activate_matching_ats_model(_artifacts_root(), output, metadata)
    if active_model is None:
        metadata["synchronization_status"] = "UNLINKED"
    else:
        metadata["synchronization_status"] = "SYNCHRONIZED"
        metadata["active_model_id"] = active_model["model_id"]
        metadata["historical_evaluation"] = active_model["historical_evaluation"]
    atomic_json(metadata, output / "metadata.json")
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
    atomic_json(metadata, output / "metadata.json")
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


def build_parser() -> argparse.ArgumentParser:
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(
        prog="nfl-ats",
        description="Leak-safe NFL against-the-spread research pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="show runtime and data health")
    doctor.set_defaults(handler=_cmd_doctor)

    dashboard = subparsers.add_parser("dashboard", help="open the local research dashboard")
    dashboard.add_argument("--address", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument("--no-browser", action="store_true")
    dashboard.set_defaults(handler=_cmd_dashboard)

    publish = subparsers.add_parser(
        "publish-predictions",
        help="write the synchronized active weekly ATS card into GitHub Markdown",
    )
    publish.add_argument("--destination", type=Path, default=Path("CURRENT_PREDICTIONS.md"))
    publish.add_argument("--readme", type=Path, default=Path("README.md"))
    publish.set_defaults(handler=_cmd_publish_predictions)

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
    ingest.add_argument("--start-season", type=int, default=2009)
    ingest.add_argument("--end-season", type=int, default=current_year)
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
    pbp_ingest.add_argument("--start-season", type=int, default=2009)
    pbp_ingest.add_argument("--end-season", type=int, default=current_year - 1)
    pbp_ingest.set_defaults(handler=_cmd_pbp_ingest)

    depth_ingest = subparsers.add_parser(
        "depth-ingest", help="archive timestamped nflverse quarterback depth charts"
    )
    depth_ingest.add_argument("--start-season", type=int, default=current_year - 1)
    depth_ingest.add_argument("--end-season", type=int, default=current_year - 1)
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
    player_ingest.set_defaults(handler=_cmd_player_ingest)

    player_value_ingest = subparsers.add_parser(
        "player-value-ingest",
        help="archive weekly nflverse player production for lagged value estimates",
    )
    player_value_ingest.add_argument("--start-season", type=int, default=2009)
    player_value_ingest.add_argument("--end-season", type=int, default=current_year - 1)
    player_value_ingest.set_defaults(handler=_cmd_player_value_ingest)

    odds_ingest = subparsers.add_parser(
        "odds-ingest", help="archive timestamped NFL quotes from The Odds API"
    )
    odds_ingest.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
    odds_ingest.add_argument("--regions", default="us")
    odds_ingest.add_argument("--markets", default="spreads,h2h")
    odds_ingest.add_argument("--bookmakers")
    odds_ingest.set_defaults(handler=_cmd_odds_ingest)

    odds_summary = subparsers.add_parser(
        "odds-summary", help="summarize locally archived point-in-time quotes"
    )
    odds_summary.set_defaults(handler=_cmd_odds_summary)

    market_backfill = subparsers.add_parser(
        "market-backfill",
        help="download and audit the free historical NFL closing-line archive",
    )
    market_backfill.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
    market_backfill.set_defaults(handler=_cmd_market_backfill)

    open_close_backfill = subparsers.add_parser(
        "market-open-close-backfill",
        help="download the free 2025 NFL opener and multi-book closing sample",
    )
    open_close_backfill.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
    open_close_backfill.set_defaults(handler=_cmd_market_open_close_backfill)

    feature_parser = subparsers.add_parser(
        "build-features", help="build the canonical pregame feature table"
    )
    feature_parser.add_argument("--snapshot", help="snapshot ID; defaults to latest")
    feature_parser.add_argument("--ewm-span", type=int, default=8)
    feature_parser.add_argument("--min-periods", type=int, default=3)
    feature_parser.add_argument("--offseason-retention", type=float, default=0.67)
    feature_parser.add_argument("--graph-half-life", type=float, default=8.0)
    feature_parser.add_argument("--graph-ridge-alpha", type=float, default=8.0)
    feature_parser.add_argument("--graph-min-games", type=int, default=16)
    feature_parser.set_defaults(handler=_cmd_build_features)

    pbp_features = subparsers.add_parser(
        "build-pbp-features", help="add leak-safe PBP states to the canonical feature table"
    )
    pbp_features.add_argument("--snapshot", help="PBP snapshot ID; defaults to latest")
    pbp_features.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
    pbp_features.add_argument("--ewm-span", type=int, default=8)
    pbp_features.add_argument("--min-periods", type=int, default=3)
    pbp_features.add_argument("--offseason-retention", type=float, default=0.67)
    pbp_features.add_argument("--opponent-half-life", type=float, default=16.0)
    pbp_features.add_argument("--opponent-ridge-alpha", type=float, default=10.0)
    pbp_features.add_argument("--opponent-min-games", type=int, default=64)
    pbp_features.set_defaults(handler=_cmd_build_pbp_features)

    qb_features = subparsers.add_parser(
        "build-qb-features",
        help="attach point-in-time expected starters and strictly prior QB states",
    )
    qb_features.add_argument("--pbp-snapshot", help="PBP snapshot ID; defaults to latest")
    qb_features.add_argument("--depth-snapshot", help="depth-chart snapshot ID; defaults to latest")
    qb_features.add_argument(
        "--features",
        type=Path,
        default=_data_root() / "processed" / "game_features_pbp.parquet",
    )
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
    player_features.add_argument("--player-snapshot", help="player snapshot ID; defaults to latest")
    player_features.add_argument(
        "--player-value-snapshot", help="player-value snapshot ID; defaults to latest"
    )
    player_features.add_argument("--pbp-snapshot", help="PBP snapshot ID; defaults to latest")
    player_features.add_argument(
        "--features",
        type=Path,
        default=_data_root() / "processed" / "game_features_pbp.parquet",
    )
    player_features.add_argument(
        "--destination",
        type=Path,
        default=_data_root() / "processed" / "game_features_player.parquet",
    )
    player_features.add_argument("--decision-hours", type=int, default=24)
    player_features.add_argument("--role-span", type=int, default=8)
    player_features.add_argument("--qb-span", type=int, default=12)
    player_features.add_argument("--qb-min-dropbacks", type=int, default=20)
    player_features.add_argument("--offseason-retention", type=float, default=0.75)
    player_features.add_argument("--value-span", type=int, default=16)
    player_features.add_argument("--value-prior-snaps", type=float, default=200.0)
    player_features.set_defaults(handler=_cmd_build_player_features)

    backtest = subparsers.add_parser("backtest", help="run expanding weekly evaluation")
    backtest.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
    backtest.add_argument("--start-season", type=int, default=2018)
    backtest.add_argument("--end-season", type=int)
    backtest.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    backtest.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="full")
    backtest.add_argument("--min-edge", type=float, default=0.02)
    backtest.add_argument("--min-train-games", type=int, default=500)
    backtest.add_argument("--initial-bankroll", type=float, default=100.0)
    backtest.add_argument("--kelly-multiplier", type=float, default=0.25)
    backtest.add_argument("--max-bet-fraction", type=float, default=0.02)
    backtest.add_argument("--max-week-fraction", type=float, default=0.10)
    backtest.add_argument("--probability-haircut", type=float, default=0.0)
    backtest.add_argument("--bankroll-paths", type=int, default=5_000)
    backtest.add_argument("--bankroll-seed", type=int, default=20260812)
    backtest.add_argument("--bootstrap-samples", type=int, default=2_000)
    backtest.add_argument("--bootstrap-seed", type=int, default=20260812)
    backtest.set_defaults(handler=_cmd_backtest)

    nested = subparsers.add_parser(
        "nested-evaluate",
        help="select configurations on prior seasons and score untouched outer seasons",
    )
    nested.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
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
    nested.add_argument("--min-train-games", type=int, default=500)
    nested.add_argument("--bootstrap-samples", type=int, default=2_000)
    nested.add_argument("--bootstrap-seed", type=int, default=20260812)
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
        "experiment", help="compare feature sets with identical walk-forward windows"
    )
    experiment.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
    experiment.add_argument("--start-season", type=int, default=2022)
    experiment.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    experiment.add_argument("--feature-sets", default=",".join(DEFAULT_EXPERIMENT_SETS))
    experiment.add_argument(
        "--baseline-feature-set",
        choices=tuple(FEATURE_SETS),
        help="paired comparison baseline; defaults to the first requested feature set",
    )
    experiment.add_argument("--bootstrap-samples", type=int, default=2_000)
    experiment.add_argument("--bootstrap-seed", type=int, default=20260812)
    experiment.add_argument("--min-edge", type=float, default=0.02)
    experiment.add_argument("--min-train-games", type=int, default=500)
    experiment.set_defaults(handler=_cmd_experiment)

    margin_backtest = subparsers.add_parser(
        "margin-backtest",
        help="compare market, fair-margin, residual-margin, straight-up, and direct ATS models",
    )
    margin_backtest.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
    margin_backtest.add_argument("--start-season", type=int, default=2018)
    margin_backtest.add_argument("--regressor", choices=("ridge", "hgb"), default="ridge")
    margin_backtest.add_argument("--ridge-alpha", type=float, default=10.0)
    margin_backtest.add_argument(
        "--feature-profile", choices=MARGIN_FEATURE_PROFILES, default="base"
    )
    margin_backtest.add_argument(
        "--methods",
        default=",".join(OUTCOME_METHODS),
        help=f"comma-separated subset of: {','.join(OUTCOME_METHODS)}",
    )
    margin_backtest.add_argument("--min-edge", type=float, default=0.02)
    margin_backtest.add_argument("--min-train-games", type=int, default=500)
    margin_backtest.add_argument("--bootstrap-samples", type=int, default=1_000)
    margin_backtest.add_argument("--bootstrap-seed", type=int, default=20260812)
    margin_backtest.set_defaults(handler=_cmd_margin_backtest)

    player_ablation = subparsers.add_parser(
        "player-ablation",
        help="compare player feature families on the residual-margin ATS model",
    )
    player_ablation.add_argument(
        "--features",
        type=Path,
        default=_data_root() / "processed" / "game_features_player.parquet",
    )
    player_ablation.add_argument("--start-season", type=int, default=2018)
    player_ablation.add_argument("--regressor", choices=("ridge", "hgb"), default="ridge")
    player_ablation.add_argument("--ridge-alpha", type=float, default=10.0)
    player_ablation.add_argument("--profiles", default=",".join(DEFAULT_PLAYER_PROFILE_SETS))
    player_ablation.add_argument(
        "--baseline-profile", choices=MARGIN_FEATURE_PROFILES, default="base"
    )
    player_ablation.add_argument("--min-edge", type=float, default=0.02)
    player_ablation.add_argument("--min-train-games", type=int, default=500)
    player_ablation.add_argument("--bootstrap-samples", type=int, default=2_000)
    player_ablation.add_argument("--bootstrap-seed", type=int, default=20260812)
    player_ablation.add_argument("--first-nested-test-season", type=int, default=2020)
    player_ablation.add_argument("--validation-seasons", type=int, default=2)
    player_ablation.set_defaults(handler=_cmd_player_ablation)

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
    player_selection.add_argument("--min-train-games", type=int, default=500)
    player_selection.add_argument("--bootstrap-samples", type=int, default=2_000)
    player_selection.add_argument("--bootstrap-seed", type=int, default=20260813)
    player_selection.set_defaults(handler=_cmd_player_model_selection)

    margin_predict = subparsers.add_parser(
        "margin-predict", help="score one week with fair-margin and outcome models"
    )
    margin_predict.add_argument("--season", type=int, required=True)
    margin_predict.add_argument("--week", type=int, required=True)
    margin_predict.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
    margin_predict.add_argument("--regressor", choices=("ridge", "hgb"), default="ridge")
    margin_predict.add_argument("--ridge-alpha", type=float, default=10.0)
    margin_predict.add_argument(
        "--feature-profile", choices=MARGIN_FEATURE_PROFILES, default="base"
    )
    margin_predict.add_argument("--min-edge", type=float, default=0.02)
    margin_predict.add_argument("--min-train-games", type=int, default=500)
    margin_predict.set_defaults(handler=_cmd_margin_predict)

    predict = subparsers.add_parser("predict", help="score one season/week")
    predict.add_argument("--season", type=int, required=True)
    predict.add_argument("--week", type=int, required=True)
    predict.add_argument(
        "--features", type=Path, default=_data_root() / "processed" / "game_features.parquet"
    )
    predict.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    predict.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="market_context")
    predict.add_argument("--min-edge", type=float, default=0.02)
    predict.add_argument("--min-train-games", type=int, default=500)
    predict.add_argument(
        "--freeze",
        action="store_true",
        help="archive an immutable forecast after verifying every game is pre-kickoff",
    )
    predict.set_defaults(handler=_cmd_predict)
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
