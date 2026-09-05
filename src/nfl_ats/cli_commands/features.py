"""Feature-table build commands."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from nfl_ats.artifact_contracts import KIND_FEATURE_TABLE, stamp
from nfl_ats.availability import (
    AVAILABILITY_COMBINATION_PRIOR,
    AVAILABILITY_POSITION_PRIOR,
    AVAILABILITY_RATE_VERSION,
    build_availability_outcomes,
    build_season_lagged_availability_rates,
    score_availability_rates,
    summarize_availability_scores,
)
from nfl_ats.cli_common import (
    _add_ewm_args,
    _add_features_arg,
    _add_player_feature_tuning_args,
    _add_snapshot_args,
    _data_root,
    _load_features,
    _print_json,
    _resolve_pbp_snapshot,
    _resolve_player_snapshot,
    _resolve_player_value_snapshot,
    _resolve_snapshot,
)
from nfl_ats.feature_manifest import inherit_source_snapshots, manifest_path_for
from nfl_ats.features import build_game_features
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet
from nfl_ats.lineage import parse_snapshot_capture
from nfl_ats.participation import (
    PARTICIPATION_RATING_EPA_CLIP,
    PARTICIPATION_RATING_LOOKBACK_SEASONS,
    PARTICIPATION_RATING_RELIABILITY_PRIOR_PLAYS,
    PARTICIPATION_RATING_RIDGE_ALPHA,
    PARTICIPATION_RATING_TEAM_FEATURE_SCALE,
    build_season_lagged_player_ratings,
    latest_participation_snapshot,
    load_participation_snapshot,
    participation_snapshot_from_root,
)
from nfl_ats.pbp import PBP_FEATURE_VERSION, enrich_with_pbp_features, load_pbp_snapshot
from nfl_ats.players import (
    PLAYER_AVAILABILITY_FEATURE_VERSION,
    PLAYER_FEATURE_VERSION,
    PLAYER_PARTICIPATION_FEATURE_VERSION,
    attach_snap_player_ids,
    canonicalize_injuries,
    canonicalize_rosters,
    canonicalize_snaps,
    enrich_with_player_features,
    load_player_snapshot,
    load_player_value_snapshot,
)
from nfl_ats.provenance import sha256_file
from nfl_ats.quarterbacks import (
    depth_snapshot_from_root,
    enrich_with_qb_features,
    latest_depth_snapshot,
    load_depth_snapshot,
)
from nfl_ats.snapshots import load_snapshot


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
    # ENG-09: stamp the manifest with this contract layer's schema/builder
    # version so a later check_compatible() call has something to compare
    # against; additive, never changes an existing manifest key.
    metadata = stamp(KIND_FEATURE_TABLE, metadata)
    atomic_json(metadata, destination.with_name("game_features.manifest.json"))
    _print_json(metadata)


def _cmd_build_pbp_features(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    snapshot = _resolve_pbp_snapshot(args.snapshot)
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
    # ENG-22: carry the base nflverse source_snapshot (and anything else
    # game_features.manifest.json already inherited) forward, since this
    # manifest otherwise only records source_features -- a path.
    source_snapshots = inherit_source_snapshots([manifest_path_for(args.features)])
    if source_snapshots:
        metadata["source_snapshots"] = source_snapshots
    # ENG-09: see the comment at _cmd_build_features's stamp() call.
    metadata = stamp(KIND_FEATURE_TABLE, metadata)
    atomic_json(metadata, destination.with_name("game_features_pbp.manifest.json"))
    _print_json(metadata)


def _cmd_build_qb_features(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    pbp_snapshot = _resolve_pbp_snapshot(args.pbp_snapshot)
    player_snapshot = _resolve_player_snapshot(args.player_snapshot)
    depth_root = args.depth_root or (_data_root() / "quarterbacks" / "depth" / "raw")
    depth_snapshot = (
        depth_snapshot_from_root(depth_root / args.depth_snapshot)
        if args.depth_snapshot
        else latest_depth_snapshot(depth_root)
    )
    injuries, _, _ = load_player_snapshot(player_snapshot)
    availability_rates = (
        pd.read_parquet(args.availability_rates) if args.availability_rates is not None else None
    )
    enriched = enrich_with_qb_features(
        features,
        load_pbp_snapshot(pbp_snapshot),
        load_depth_snapshot(depth_snapshot),
        injuries,
        availability_rates,
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
        "source_player_snapshot": player_snapshot.snapshot_id,
        "source_availability_rates": (
            {
                "path": str(args.availability_rates),
                "sha256": sha256_file(args.availability_rates),
            }
            if args.availability_rates is not None
            else {"mode": "fixed_status_prior"}
        ),
        "decision_hours_before_kickoff": args.decision_hours,
        "max_depth_age_days": args.max_depth_age_days,
        "ewm_span": args.ewm_span,
        "min_dropbacks": args.min_dropbacks,
        "offseason_retention": args.offseason_retention,
        "qb_feature_version": str(enriched["qb_feature_version"].iloc[0]),
        "rows": len(enriched),
        "games_with_both_expected_qbs": int(both_qbs.sum()),
        "games_with_both_qb_states": int(both_states.sum()),
        "games_with_both_named_backups": int(
            (
                enriched["home_depth_qb_backup_id"].notna()
                & enriched["away_depth_qb_backup_id"].notna()
            ).sum()
        ),
        "games_with_both_start_probabilities": int(
            (
                enriched["home_depth_qb_start_probability"].notna()
                & enriched["away_depth_qb_start_probability"].notna()
            ).sum()
        ),
        "destination": str(destination),
    }
    # ENG-22: see the comment at _cmd_build_pbp_features's inherit_source_snapshots call.
    source_snapshots = inherit_source_snapshots([manifest_path_for(args.features)])
    if source_snapshots:
        metadata["source_snapshots"] = source_snapshots
    # ENG-09: see the comment at _cmd_build_features's stamp() call.
    metadata = stamp(KIND_FEATURE_TABLE, metadata)
    atomic_json(metadata, destination.with_name("game_features_qb.manifest.json"))
    _print_json(metadata)


def _cmd_build_player_features(args: argparse.Namespace) -> None:
    features = _load_features(args.features)
    player_snapshot = _resolve_player_snapshot(args.player_snapshot)
    pbp_snapshot = _resolve_pbp_snapshot(args.pbp_snapshot)
    player_value_snapshot = _resolve_player_value_snapshot(args.player_value_snapshot)
    depth_root = args.depth_root or (_data_root() / "quarterbacks" / "depth" / "raw")
    try:
        depth_snapshot = latest_depth_snapshot(depth_root)
        depth_charts = load_depth_snapshot(depth_snapshot)
    except FileNotFoundError:
        depth_snapshot = None
        depth_charts = None
    injuries, rosters, snaps = load_player_snapshot(player_snapshot)
    enriched = enrich_with_player_features(
        features,
        injuries,
        rosters,
        snaps,
        load_pbp_snapshot(pbp_snapshot),
        load_player_value_snapshot(player_value_snapshot),
        depth_charts=depth_charts,
        decision_hours_before_kickoff=args.decision_hours,
        role_span=args.role_span,
        qb_span=args.qb_span,
        qb_min_dropbacks=args.qb_min_dropbacks,
        offseason_retention=args.offseason_retention,
        value_span=args.value_span,
        value_prior_snaps=args.value_prior_snaps,
        # ENG-23: fills {side}_injury_observed_at when no team-specific
        # revision is visible, instead of leaving it null forever.
        injury_snapshot_captured_at=parse_snapshot_capture(player_snapshot.snapshot_id),
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
        "source_depth_snapshot": depth_snapshot.snapshot_id if depth_snapshot else None,
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
    # ENG-22: see the comment at _cmd_build_pbp_features's inherit_source_snapshots call.
    source_snapshots = inherit_source_snapshots([manifest_path_for(args.features)])
    if source_snapshots:
        metadata["source_snapshots"] = source_snapshots
    # ENG-09: see the comment at _cmd_build_features's stamp() call.
    metadata = stamp(KIND_FEATURE_TABLE, metadata)
    atomic_json(metadata, destination.with_name(f"{destination.stem}.manifest.json"))
    _print_json(metadata)


def _cmd_build_participation_features(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    player_snapshot = _resolve_player_snapshot(args.player_snapshot)
    pbp_snapshot = _resolve_pbp_snapshot(args.pbp_snapshot)
    player_value_snapshot = _resolve_player_value_snapshot(args.player_value_snapshot)
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
        # ENG-23: see the identical comment at _cmd_build_player_features's call.
        injury_snapshot_captured_at=parse_snapshot_capture(player_snapshot.snapshot_id),
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
    # ENG-22: see the comment at _cmd_build_pbp_features's inherit_source_snapshots call.
    source_snapshots = inherit_source_snapshots([manifest_path_for(args.features)])
    if source_snapshots:
        metadata["source_snapshots"] = source_snapshots
    # ENG-09: see the comment at _cmd_build_features's stamp() call.
    metadata = stamp(KIND_FEATURE_TABLE, metadata)
    atomic_json(metadata, args.destination.with_name(f"{args.destination.stem}.manifest.json"))
    _print_json(metadata)


def _cmd_build_learned_availability_features(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    features = _load_features(args.features)
    player_snapshot = _resolve_player_snapshot(args.player_snapshot)
    pbp_snapshot = _resolve_pbp_snapshot(args.pbp_snapshot)
    player_value_snapshot = _resolve_player_value_snapshot(args.player_value_snapshot)
    depth_root = args.depth_root or (_data_root() / "quarterbacks" / "depth" / "raw")
    try:
        depth_snapshot = latest_depth_snapshot(depth_root)
        depth_charts = load_depth_snapshot(depth_snapshot)
    except FileNotFoundError:
        depth_snapshot = None
        depth_charts = None
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
        depth_charts=depth_charts,
        decision_hours_before_kickoff=args.decision_hours,
        role_span=args.role_span,
        qb_span=args.qb_span,
        qb_min_dropbacks=args.qb_min_dropbacks,
        offseason_retention=args.offseason_retention,
        value_span=args.value_span,
        value_prior_snaps=args.value_prior_snaps,
        # ENG-23: see the identical comment at _cmd_build_player_features's call.
        injury_snapshot_captured_at=parse_snapshot_capture(player_snapshot.snapshot_id),
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
        "source_depth_snapshot": depth_snapshot.snapshot_id if depth_snapshot else None,
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
    # ENG-22: see the comment at _cmd_build_pbp_features's inherit_source_snapshots call.
    source_snapshots = inherit_source_snapshots([manifest_path_for(args.features)])
    if source_snapshots:
        metadata["source_snapshots"] = source_snapshots
    # ENG-09: see the comment at _cmd_build_features's stamp() call.
    metadata = stamp(KIND_FEATURE_TABLE, metadata)
    atomic_json(metadata, args.destination.with_name(f"{args.destination.stem}.manifest.json"))
    _print_json(metadata)


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the ``build-*-features`` commands."""

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
    _add_snapshot_args(qb_features, ("--player-snapshot", "player"))
    qb_features.add_argument(
        "--depth-root",
        type=Path,
        help="depth snapshot root; defaults to timestamped data/quarterbacks/depth/raw",
    )
    _add_features_arg(qb_features, "game_features_pbp.parquet")
    qb_features.add_argument("--decision-hours", type=int, default=24)
    qb_features.add_argument("--max-depth-age-days", type=int, default=14)
    qb_features.add_argument("--ewm-span", type=int, default=12)
    qb_features.add_argument("--min-dropbacks", type=int, default=50)
    qb_features.add_argument("--offseason-retention", type=float, default=0.75)
    qb_features.add_argument(
        "--availability-rates",
        type=Path,
        help="optional season-lagged availability-rate parquet; fixed status priors otherwise",
    )
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
        "--depth-root",
        type=Path,
        default=None,
        help="timestamped quarterback depth snapshots (defaults to the latest local snapshot)",
    )
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
    availability_features.add_argument("--depth-root", type=Path, default=None)
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
