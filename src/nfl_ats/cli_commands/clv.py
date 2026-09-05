"""Closing-line-value scoring, the predeclared pilot and drift diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

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
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.drift import build_drift_report, write_drift_artifacts
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id
from nfl_ats.odds_backfill import HISTORICAL_CAPTURE_KIND
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact


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
        "active_model_config": active_model_config,
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
    print(season_summary.to_string(index=False))
    print(uncertainty.to_string(index=False))
    _print_json({**metadata, "artifact_directory": str(output)})


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


def register_scoring(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the CLV scoring and paper-ledger commands."""

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
    """Register the drift, pilot and opener/close diagnostic commands."""

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

    composition_parser = subparsers.add_parser(
        "overlay-composition",
        help="recompute overlay subset scores from the active model's matching opener evaluation",
    )
    composition_parser.add_argument("--per-game-artifact", type=Path)
    composition_parser.add_argument("--features", type=Path)
    composition_parser.add_argument("--incidents", type=Path)
    _add_bootstrap_args(composition_parser, samples=20_000, seed=20260821)
    composition_parser.set_defaults(handler=_cmd_overlay_composition)

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
