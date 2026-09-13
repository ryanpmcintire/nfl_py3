from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from nfl_ats.calibration import RESIDUAL_SMOOTHING_METHODS
from nfl_ats.cli_common import (
    _add_bootstrap_args,
    _add_feature_profile_arg,
    _add_features_arg,
    _add_regressor_args,
    _add_season_week_args,
    _artifacts_root,
    _data_root,
    _load_features,
    _print_json,
    _registry_root,
)
from nfl_ats.clv import (
    FROZEN_PILOT_PROTOCOL,
    OPENER_EVALUATION_METRIC_COLUMNS,
    ClosePredictionUnavailable,
    PilotProtocolBlocked,
    build_pairing_table,
    close_reference_table,
    clv_summary,
    live_close_reference,
    load_paper_decisions,
    opener_evaluation_home_side_offset_summary,
    opener_evaluation_metric_draws,
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
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.drift import build_drift_report, write_drift_artifacts
from nfl_ats.home_side_location import HOME_SIDE_OFFSET_SERVED
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact
from nfl_ats.served_refresh_card import reuse_or_measure as served_refresh_card_reuse_or_measure


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


def _find_drift_cards(
    artifacts_root: Path,
    *,
    season: int,
    week: int,
    feature_profile: str,
    probability_method: str,
) -> list[tuple[dict[str, Any], Path]]:

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
    probability_method = getattr(args, "probability_method", None)
    if probability_method is not None:
        active_model_config = dict(active_model_config)
        if probability_method != active_model_config.get("probability_method", "ecdf"):
            active_model_config["comparison_baseline_model_id"] = active_model_config.pop(
                "model_id", None
            )
        active_model_config["probability_method"] = probability_method
    provenance = artifact_provenance(active_model_config, args.features)
    feature_sha = provenance["feature_table"]["sha256"]
    expected_sha = active_model_config.get("feature_table_sha256")
    if expected_sha is not None and expected_sha != feature_sha:
        raise ValueError("Opener evaluation feature table does not match the active model")
    active_model_config = {
        "probability_method": "ecdf",
        "calibration_method": "none",
        **active_model_config,
        "feature_table_sha256": feature_sha,
    }
    without_offset = bool(getattr(args, "no_home_side_offset", False))
    serve_offset = HOME_SIDE_OFFSET_SERVED and not without_offset
    if without_offset and HOME_SIDE_OFFSET_SERVED and "model_id" in active_model_config:
        active_model_config = dict(active_model_config)
        active_model_config["comparison_baseline_model_id"] = active_model_config.pop("model_id")
    line_source_path = getattr(args, "opener_line_source", None)
    line_source: dict[str, Any] | None = None
    override = None
    if line_source_path is not None:
        override = pd.read_parquet(line_source_path)
        line_source = {
            "path": str(line_source_path),
            "sha256": sha256_file(Path(line_source_path)),
            "games": len(override),
            "label": str(override["line_source"].iloc[0])
            if "line_source" in override.columns and len(override)
            else "unnamed",
        }
        if "model_id" in active_model_config:
            active_model_config = dict(active_model_config)
            active_model_config["comparison_baseline_model_id"] = active_model_config.pop(
                "model_id"
            )
    scored = opener_pick_evaluation(
        market_root,
        features,
        active_model_config=active_model_config,
        min_train_games=args.min_train_games,
        home_side_offset=serve_offset,
        opener_line_override=override,
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
                metric_columns=OPENER_EVALUATION_METRIC_COLUMNS,
                metric_draw_factory=opener_evaluation_metric_draws,
            ),
            week_blocked_bootstrap(
                scored,
                opener_evaluation_metrics,
                block="season",
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
                metric_columns=OPENER_EVALUATION_METRIC_COLUMNS,
                metric_draw_factory=opener_evaluation_metric_draws,
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

    root_name = "opener_evaluation" if line_source is None else "opener_evaluation_line_source"
    command_name = "opener-evaluation" if line_source is None else "opener-evaluation-line-source"
    output = _artifacts_root() / root_name / run_id()
    atomic_parquet(scored, output / "per_game.parquet")
    atomic_csv(uncertainty, output / "uncertainty.csv")
    atomic_csv(season_summary, output / "season_summary.csv")
    configuration = {
        "command": command_name,
        "min_train_games": args.min_train_games,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "hypothesis_frozen_before_scoring": True,
        "predeclaration": (
            "docs/opener_evaluation.md"
            if line_source is None
            else "docs/single_book_opener_grade.md"
        ),
        "active_model_config": active_model_config,
        "opener_line_source": line_source,
    }
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "active_model_config": active_model_config,
        "active_model_id": active_model_config.get("model_id"),
        **{
            key: active_model_config[key]
            for key in (
                "probability_method",
                "calibration_method",
                "feature_table_sha256",
                "feature_profile",
                "regressor",
                "ridge_alpha",
            )
        },
        "games": len(scored),
        "mean_absolute_open_to_close_move": float(scored["open_move"].abs().mean()),
        "home_side_offset": opener_evaluation_home_side_offset_summary(scored, served=serve_offset),
        "metrics": metrics,
        "uncertainty": uncertainty.to_dict(orient="records"),
        "timing": {"total_seconds": perf_counter() - command_started},
        "provenance": artifact_provenance(configuration, args.features),
    }
    write_experiment_artifact(
        output,
        "metadata.json",
        metadata,
        command=command_name,
        metrics=metadata,
        registry_root=_registry_root(),
    )
    print(season_summary.to_string(index=False))
    print(uncertainty.to_string(index=False))
    _print_json({**metadata, "artifact_directory": str(output)})


def _cmd_opener_line_series(args: argparse.Namespace) -> None:
    from nfl_ats.single_book_opener import (
        book_coverage,
        choose_book,
        half_point_median_series,
        opener_book_quotes,
        series_agreement,
        single_book_series,
    )

    features = _load_features(args.features)
    market_root = _data_root() / "market" / "raw"
    quotes = opener_book_quotes(market_root, schedule=features)
    coverage = book_coverage(quotes)
    book = args.book or choose_book(coverage)
    series = (
        single_book_series(quotes, book)
        if args.series == "book"
        else half_point_median_series(quotes)
    )
    output = args.output or (_artifacts_root() / "opener_line_series" / run_id())
    atomic_parquet(series, output / f"{args.series}.parquet")
    atomic_csv(coverage, output / "book_coverage.csv")
    consensus = build_pairing_table(
        market_root, capture_kind=HISTORICAL_CAPTURE_KIND, labels=("tue_open",), schedule=features
    )
    consensus = consensus.rename(columns={"home_spread": "tue_open_home_spread"})
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "series": args.series,
        "book": book if args.series == "book" else None,
        "book_chosen_by": "most games among the predeclared candidate books"
        if args.book is None
        else "explicit --book",
        "opener_quote_rows": len(quotes),
        "opener_quote_games": int(quotes["game_id"].nunique()) if len(quotes) else 0,
        "books_in_archive": int(quotes["bookmaker_key"].nunique()) if len(quotes) else 0,
        "coverage_top": coverage.head(12).to_dict(orient="records"),
        "agreement": series_agreement(series, consensus),
        "artifact_directory": str(output),
    }
    atomic_json(summary, output / "summary.json")
    _print_json(summary)


def _cmd_overlay_composition(args: argparse.Namespace) -> None:
    from nfl_ats.overlay_composition import DEFAULT_INCIDENTS, run_overlay_composition
    from nfl_ats.public_board import find_matching_opener_evaluation

    per_game = args.per_game_artifact
    if per_game is None:
        match = find_matching_opener_evaluation(_artifacts_root())
        if match is None:
            raise ValueError("No opener-evaluation matches the active model; run opener-evaluation")
        per_game = match[1] / "per_game.parquet"
    _print_json(
        run_overlay_composition(
            per_game_artifact=per_game,
            data_root=_data_root(),
            features=args.features or _data_root() / "processed" / "game_features_pbp.parquet",
            incidents=args.incidents or _data_root() / DEFAULT_INCIDENTS.relative_to("data"),
            output_root=_artifacts_root() / "overlay_subset_composition",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    )


def _cmd_served_refresh_card(args: argparse.Namespace) -> None:
    from nfl_ats.served_refresh_card import measure as served_refresh_card_measure

    artifacts_root = _artifacts_root()
    data_root = _data_root()
    repo_root = Path.cwd()
    registry_dir = args.registry_dir or _registry_root()
    if args.force:
        result = served_refresh_card_measure(
            artifacts_root=artifacts_root,
            data_root=data_root,
            repo_root=repo_root,
            record=args.record,
            registry_dir=registry_dir,
        )
    else:
        result = served_refresh_card_reuse_or_measure(
            artifacts_root=artifacts_root,
            data_root=data_root,
            repo_root=repo_root,
            record=args.record,
            registry_dir=registry_dir,
        )
    _print_json(result)


def _cmd_served_card_archive(args: argparse.Namespace) -> None:
    from nfl_ats.overlay_composition import DEFAULT_FEATURES, DEFAULT_INCIDENTS
    from nfl_ats.public_board import (
        find_matching_opener_evaluation,
        load_active_ats_model,
        load_served_union_measurement,
    )
    from nfl_ats.unserved_tilt_marginals import run_unserved_tilt_marginals

    artifacts_root = _artifacts_root()
    active = load_active_ats_model(artifacts_root) or {}
    existing = None if args.force else load_served_union_measurement(artifacts_root, active)
    if existing is not None:
        _print_json(
            {
                "status": "reused",
                "active_model_id": active.get("model_id"),
                "served_card_accuracy": existing.accuracy,
                "n_scored_games": existing.scored_games,
            }
        )
        return
    match = find_matching_opener_evaluation(artifacts_root, active)
    if match is None:
        raise ValueError("No opener-evaluation matches the active model; run opener-evaluation")
    data_root = _data_root()
    result = run_unserved_tilt_marginals(
        per_game_artifact=match[1] / "per_game.parquet",
        data_root=data_root,
        repo_root=Path.cwd(),
        features=args.features or data_root / DEFAULT_FEATURES.relative_to("data"),
        incidents=args.incidents or data_root / DEFAULT_INCIDENTS.relative_to("data"),
        output_root=artifacts_root / "unserved_tilt_marginals",
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    _print_json(
        {
            "status": "measured",
            "active_model_id": result.get("active_model_id"),
            "served_card_accuracy": result.get("served_card_accuracy"),
            "n_scored_games": result.get("n_scored_games"),
            "output_dir": result.get("artifact_directory"),
        }
    )


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


def _cmd_injury_headlines(args: argparse.Namespace) -> None:
    from datetime import date, timedelta

    from nfl_ats.injury_headlines import run_injury_headlines

    data_root = args.data_root or _data_root()
    artifacts_root = args.artifacts_root or _artifacts_root()
    since = date.fromisoformat(args.since) if args.since else date.today() - timedelta(days=14)
    result = run_injury_headlines(
        data_root=data_root,
        artifacts_root=artifacts_root,
        since=since,
        dry=args.dry,
    )
    _print_json(result)


def register_scoring(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:

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


def register_diagnostics(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:

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
    drift_report.add_argument("--probability-method", default="gaussian_median")
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
    opener_evaluation_parser.add_argument(
        "--probability-method",
        choices=RESIDUAL_SMOOTHING_METHODS,
        default=None,
        help="override the active model probability mapping for this evaluation only",
    )
    opener_evaluation_parser.add_argument(
        "--opener-line-source",
        type=Path,
        default=None,
        help="grade at an alternative opener line instead of the captured cross-book consensus: "
        "a parquet with game_id, home_spread and optional books columns (see "
        "nfl-ats opener-line-series). The run writes to artifacts/opener_evaluation_line_source "
        "and never identifies itself as the active model's opener evaluation",
    )
    opener_evaluation_parser.add_argument(
        "--no-home-side-offset",
        action="store_true",
        help="score the raw model without the served walk-forward home-side offset "
        "(a comparison run; it never identifies itself as the active model)",
    )
    _add_bootstrap_args(opener_evaluation_parser, seed=20260817)
    opener_evaluation_parser.set_defaults(handler=_cmd_opener_evaluation)

    line_series_parser = subparsers.add_parser(
        "opener-line-series",
        help="build a Tuesday-opener line series from the per-book archive: one named book's "
        "posted spread, or the median of the half-point quotes only",
    )
    _add_features_arg(line_series_parser, "game_features_weak_stack.parquet")
    line_series_parser.add_argument(
        "--series", choices=("book", "halfpoint_median"), default="book"
    )
    line_series_parser.add_argument(
        "--book", default=None, help="bookmaker key; default picks the best-covered candidate book"
    )
    line_series_parser.add_argument("--output", type=Path, default=None)
    line_series_parser.set_defaults(handler=_cmd_opener_line_series)

    composition_parser = subparsers.add_parser(
        "overlay-composition",
        help="recompute overlay subset scores from the active model's matching opener evaluation",
    )
    composition_parser.add_argument("--per-game-artifact", type=Path)
    composition_parser.add_argument("--features", type=Path)
    composition_parser.add_argument("--incidents", type=Path)
    _add_bootstrap_args(composition_parser, samples=20_000, seed=20260821)
    composition_parser.set_defaults(handler=_cmd_overlay_composition)

    served_refresh_card_parser = subparsers.add_parser(
        "served-refresh-card",
        help="score the whole served refresh chain (Tuesday card plus every through-the-week "
        "rule) against the active model's matching opener evaluation; reuses the latest "
        "measurement when its archive still matches the active model and re-measures otherwise",
    )
    served_refresh_card_parser.add_argument(
        "--force",
        action="store_true",
        help="re-measure even when the existing artifact already matches the active model",
    )
    served_refresh_card_parser.add_argument(
        "--record",
        action="store_true",
        help="record every chain cell through weak-signals record (family served_refresh_card)",
    )
    served_refresh_card_parser.add_argument("--registry-dir", type=Path, default=None)
    served_refresh_card_parser.set_defaults(handler=_cmd_served_refresh_card)

    served_card_archive_parser = subparsers.add_parser(
        "served-card-archive",
        help="score the card as played (every served adjustment) against the active model's "
        "matching opener evaluation: the board's headline archive number; reuses the latest "
        "measurement when it already names the active model and re-measures otherwise",
    )
    served_card_archive_parser.add_argument("--features", type=Path)
    served_card_archive_parser.add_argument("--incidents", type=Path)
    served_card_archive_parser.add_argument(
        "--force",
        action="store_true",
        help="re-measure even when the existing artifact already matches the active model",
    )
    _add_bootstrap_args(served_card_archive_parser, samples=20_000, seed=20260821)
    served_card_archive_parser.set_defaults(handler=_cmd_served_card_archive)

    injury_headlines_parser = subparsers.add_parser(
        "injury-headlines",
        help="parse PFT injury headlines into a timestamped designation table "
        "(fallback source for mornings when the official league feed is late)",
    )
    injury_headlines_parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="data directory (default: env NFL_ATS_DATA or ./data)",
    )
    injury_headlines_parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="artifacts directory (default: env NFL_ATS_ARTIFACTS or ./artifacts)",
    )
    injury_headlines_parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO date; only include captures on or after this date (default: 14 days ago)",
    )
    injury_headlines_parser.add_argument(
        "--dry",
        action="store_true",
        help="print summary and write nothing",
    )
    injury_headlines_parser.set_defaults(handler=_cmd_injury_headlines)

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
