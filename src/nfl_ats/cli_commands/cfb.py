"""College-football ingest, feature build, benchmark and role commands."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

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
from nfl_ats.cli_common import (
    _add_bootstrap_args,
    _add_ewm_args,
    _add_features_arg,
    _add_season_range_args,
    _artifacts_root,
    _data_root,
    _load_features,
    _print_json,
    _registry_root,
    _season_range,
)
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id
from nfl_ats.margin_variance import cfb_variance_benchmark
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact
from nfl_ats.role_actions import (
    RoleActionsSnapshot,
    latest_role_actions_snapshot,
    load_role_actions_snapshot,
    role_actions_snapshot_from_root,
)


def _cmd_cfb_ingest(args: argparse.Namespace) -> None:
    spec = cfb_source_spec(args.source)
    start_season = args.start_season or spec.default_start_season
    seasons = _season_range(start_season, args.end_season)
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


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the ``cfb-*`` commands."""

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
