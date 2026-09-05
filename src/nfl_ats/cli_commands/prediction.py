"""Scoring commands: margin prediction, decomposition and the weekly card."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import pandas as pd

from nfl_ats.active_model import activate_matching_ats_model, load_active_ats_model
from nfl_ats.artifact_contracts import KIND_FORECAST, check_compatible, stamp
from nfl_ats.backtest import score_week
from nfl_ats.calibration import RESIDUAL_SMOOTHING_METHODS, ResidualSmoothingMethod
from nfl_ats.cli_common import (
    _add_feature_profile_arg,
    _add_features_arg,
    _add_regressor_args,
    _add_season_range_args,
    _add_season_week_args,
    _artifacts_root,
    _data_root,
    _load_features,
    _print_json,
    _registry_root,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_csv, atomic_json, atomic_parquet, run_id
from nfl_ats.key_numbers import (
    DEFAULT_KEY_NUMBERS,
    cover_reliability_by_line_bucket,
    summarize_key_number_calibration,
)
from nfl_ats.lineage import (
    LINEAGE_FILENAME,
    PUBLISHED_DISPLAY_FIELDS,
    build_card_lineage,
    write_card_lineage,
)
from nfl_ats.lines import (
    PUSH_RULES,
    build_ats_pool_card_at_lines,
    load_lines_file,
    pool_card_at_lines_markdown,
    rescore_at_lines,
)
from nfl_ats.margin import (
    DEFAULT_LINE_SWEEP_OFFSETS,
    MarginFeatureProfile,
    margin_feature_columns,
)
from nfl_ats.market_data import load_quote_history, tuesday_opener_quotes
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
from nfl_ats.market_observation import attach_market_observed_at
from nfl_ats.modeling import MODEL_NAMES, logistic_coefficients, model_metadata
from nfl_ats.outcomes import (
    MARGIN_DISTRIBUTION_METHODS,
    OUTCOME_METHODS,
    fit_margin_models_for_week,
    score_outcome_week,
    score_outcome_week_line_sweep,
    walk_forward_key_number_mass,
    walk_forward_outcomes,
)
from nfl_ats.pool import (
    build_ats_pool_card,
    build_straight_up_pool_card,
    pool_card_markdown,
    straight_up_pool_markdown,
)
from nfl_ats.prediction_safety import (
    validate_outcome_prediction_card,
    validate_prediction_card,
    validate_prediction_lineage,
)
from nfl_ats.prospective import freeze_forecast
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact


@dataclass(frozen=True)
class PredictionArtifacts:
    """What a scoring command wrote: the metadata document and its directory.

    ``metadata`` is the same dict the handler used to print inline, and
    ``output`` the artifact directory it was written to."""

    metadata: dict[str, Any]
    output: Path


def _feature_table_manifest_for(path: Path) -> dict[str, Any] | None:
    """ENG-09: the ``*.manifest.json`` sibling of a feature-table parquet, if any."""

    manifest_path = path.with_name(f"{path.stem}.manifest.json")
    if not manifest_path.is_file():
        return None
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload


def _active_model_for_compatibility() -> dict[str, Any] | None:
    """ENG-09: the active model manifest, or ``None`` when absent/unreadable.

    A malformed ``active_ats_model.json`` is a pre-existing, separately
    surfaced problem (``nfl_ats.publishing._publication_context`` already
    raises loudly on it at publish time); this pre-fit compatibility check is
    an additional safety net, not the primary guard, so it degrades to "no
    active model to compare against" rather than blocking every command.
    """

    try:
        return load_active_ats_model(_artifacts_root())
    except ValueError:
        return None


@dataclass(frozen=True)
class MarginPredictRequest:
    """Everything ``nfl-ats margin-predict`` needs from the command line."""

    features: Path
    season: int
    week: int
    regressor: str
    min_edge: float
    min_train_games: int
    feature_profile: MarginFeatureProfile
    ridge_alpha: float
    probability_method: ResidualSmoothingMethod
    line_sweep: bool


def parse_margin_predict_request(args: argparse.Namespace) -> MarginPredictRequest:
    """Validate the parsed namespace into a MarginPredictRequest.

    Pure: reads only ``args`` and raises exactly what reading a missing or
    ill-typed attribute raises today."""

    return MarginPredictRequest(
        features=Path(args.features),
        season=args.season,
        week=args.week,
        regressor=args.regressor,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        feature_profile=args.feature_profile,
        ridge_alpha=args.ridge_alpha,
        probability_method=args.probability_method,
        line_sweep=bool(args.line_sweep),
    )


def orchestrate_margin_predict(request: MarginPredictRequest) -> PredictionArtifacts:
    """Fit, score, validate and write the outcome card for one season/week.

    Returns the metadata document and artifact directory the handler prints."""

    features = _load_features(request.features)
    # ENG-09: refuse before fitting when the feature table this run would fit
    # on carries an explicit version that contradicts the active model's own
    # record of what it was fit on. legacy_unversioned (either side has no
    # stamp yet) is a warning, not a refusal -- see nfl_ats.artifact_contracts.
    fit_compatibility = check_compatible(
        _active_model_for_compatibility(), _feature_table_manifest_for(request.features)
    )
    fit_compatibility.refuse_if_incompatible(action="fit a model on this feature table")
    predictions = score_outcome_week(
        features,
        season=request.season,
        week=request.week,
        regressor=request.regressor,
        min_edge=request.min_edge,
        min_train_games=request.min_train_games,
        feature_profile=request.feature_profile,
        ridge_alpha=request.ridge_alpha,
        probability_method=request.probability_method,
    )
    # ENG-23: join the point-in-time odds capture's observation instant onto
    # the forecast frame; never touches spread_line or which side is picked.
    predictions = attach_market_observed_at(
        predictions, market_raw_root=_data_root() / "market" / "raw"
    )
    safety = validate_outcome_prediction_card(
        predictions,
        min_edge=request.min_edge,
        expected_methods=OUTCOME_METHODS,
        expected_season=request.season,
        expected_week=request.week,
        compatibility=fit_compatibility,
    )
    output = (
        _artifacts_root()
        / "margin_predictions"
        / f"{request.season}-week-{request.week:02d}-{run_id()}"
    )
    configuration = {
        "command": "margin-predict",
        "season": request.season,
        "week": request.week,
        "regressor": request.regressor,
        "ridge_alpha": request.ridge_alpha,
        "calibration_method": "none",
        "min_edge": request.min_edge,
        "min_train_games": request.min_train_games,
        "feature_profile": request.feature_profile,
        "probability_method": request.probability_method,
    }
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **configuration,
        "game_type": (
            str(predictions["game_type"].iloc[0]) if "game_type" in predictions.columns else None
        ),
        "games": int(predictions["game_id"].nunique()),
        "methods": sorted(predictions["method"].unique().tolist()),
        "ats_method": "market_residual",
        "prediction_safety": safety.to_dict(),
        "provenance": artifact_provenance(configuration, request.features),
    }
    # ENG-09: this forecast's own schema/builder-version contract block,
    # independent of the feature table's (stamped separately, at build time).
    metadata = stamp(KIND_FORECAST, metadata)
    # ENG-21: top-level convenience mirror of the same dict artifact_provenance()
    # already computed under "provenance" -- a reference, not a recomputation --
    # so weekly-run's margin-predict step metadata carries the environment lock
    # report without digging into provenance.
    metadata["environment"] = metadata["provenance"]["environment"]
    # ENG-16: every decision-bearing card field says where it came from, and
    # the lineage audit is release-blocking -- a card that cannot answer that
    # never reaches the artifact directory.
    card_lineage = build_card_lineage(
        predictions,
        metadata,
        feature_columns=margin_feature_columns("market_residual", request.feature_profile),
        display_fields=PUBLISHED_DISPLAY_FIELDS,
    )
    lineage_audit = validate_prediction_lineage(card_lineage)
    metadata["lineage"] = {
        "path": LINEAGE_FILENAME,
        "schema_version": card_lineage.schema_version,
        "builder_version": card_lineage.builder_version,
        "checks_passed": list(lineage_audit.checks_passed),
        "decision_bearing_fields": list(card_lineage.decision_bearing_fields()),
    }
    atomic_csv(predictions, output / "predictions.csv")
    atomic_json(safety.to_dict(), output / "prediction_safety.json")
    write_card_lineage(card_lineage, output)
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
            straight_up_pool_markdown(pool_card, request.season, request.week), encoding="utf-8"
        )
        pool_methods.append(method)
    metadata["straight_up_pool_methods"] = pool_methods
    if request.line_sweep:
        sweep = score_outcome_week_line_sweep(
            features,
            season=request.season,
            week=request.week,
            regressor=request.regressor,
            min_train_games=request.min_train_games,
            feature_profile=request.feature_profile,
            ridge_alpha=request.ridge_alpha,
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
    return PredictionArtifacts(metadata=metadata, output=output)


def _cmd_margin_predict(args: argparse.Namespace) -> None:
    result = orchestrate_margin_predict(parse_margin_predict_request(args))
    _print_json({**result.metadata, "artifact_directory": str(result.output)})


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
    metadata: dict[str, Any] = {
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
    metadata: dict[str, Any] = {
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
    metadata: dict[str, Any] = {
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


@dataclass(frozen=True)
class PredictRequest:
    """Everything ``nfl-ats predict`` needs from the command line."""

    features: Path
    season: int
    week: int
    model: str
    feature_set: str
    min_edge: float
    min_train_games: int
    freeze: bool


def parse_predict_request(args: argparse.Namespace) -> PredictRequest:
    """Validate the parsed namespace into a PredictRequest.

    Pure: reads only ``args`` and raises exactly what reading a missing or
    ill-typed attribute raises today."""

    return PredictRequest(
        features=Path(args.features),
        season=args.season,
        week=args.week,
        model=args.model,
        feature_set=args.feature_set,
        min_edge=args.min_edge,
        min_train_games=args.min_train_games,
        freeze=bool(args.freeze),
    )


def orchestrate_predict(request: PredictRequest) -> PredictionArtifacts:
    """Fit, score, validate and write the direct ATS card for one season/week.

    Returns the metadata document and artifact directory the handler prints."""

    features = _load_features(request.features)
    # ENG-09: see the identical comment in _cmd_margin_predict.
    fit_compatibility = check_compatible(
        _active_model_for_compatibility(), _feature_table_manifest_for(request.features)
    )
    fit_compatibility.refuse_if_incompatible(action="fit a model on this feature table")
    predictions, model = score_week(
        features,
        season=request.season,
        week=request.week,
        model_name=request.model,
        min_edge=request.min_edge,
        min_train_games=request.min_train_games,
        feature_set=request.feature_set,
    )
    # ENG-23: see the identical comment in orchestrate_margin_predict.
    predictions = attach_market_observed_at(
        predictions, market_raw_root=_data_root() / "market" / "raw"
    )
    safety = validate_prediction_card(
        predictions,
        min_edge=request.min_edge,
        expected_season=request.season,
        expected_week=request.week,
        feature_columns=model.feature_columns,
        compatibility=fit_compatibility,
    )
    created_at = datetime.now(UTC)
    identifier = f"{request.season}-week-{request.week:02d}-{run_id(created_at)}"
    output = _artifacts_root() / "predictions" / identifier
    metadata: dict[str, Any] = {
        **model_metadata(model),
        "created_at_utc": created_at.isoformat(),
        "season": request.season,
        "week": request.week,
        "min_edge": request.min_edge,
        "games": len(predictions),
        "prediction_safety": safety.to_dict(),
    }
    configuration = {
        "command": "predict",
        "season": request.season,
        "week": request.week,
        "model": request.model,
        "feature_set": request.feature_set,
        "min_edge": request.min_edge,
        "min_train_games": request.min_train_games,
    }
    metadata["provenance"] = artifact_provenance(configuration, request.features)
    # ENG-09: this forecast's own schema/builder-version contract block.
    metadata = stamp(KIND_FORECAST, metadata)
    # ENG-16: same lineage contract as margin-predict, on the direct card.
    card_lineage = build_card_lineage(
        predictions,
        metadata,
        feature_columns=model.feature_columns,
        prediction_timestamp=created_at,
        display_fields=PUBLISHED_DISPLAY_FIELDS,
    )
    lineage_audit = validate_prediction_lineage(card_lineage, prediction_timestamp=created_at)
    metadata["lineage"] = {
        "path": LINEAGE_FILENAME,
        "schema_version": card_lineage.schema_version,
        "builder_version": card_lineage.builder_version,
        "checks_passed": list(lineage_audit.checks_passed),
        "decision_bearing_fields": list(card_lineage.decision_bearing_fields()),
    }
    write_card_lineage(card_lineage, output)
    if request.freeze:
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
        pool_card_markdown(pool_card, request.season, request.week), encoding="utf-8"
    )
    model_path = output / "model.joblib"
    temporary_model = model_path.with_suffix(".joblib.tmp")
    joblib.dump(model, temporary_model)
    temporary_model.replace(model_path)
    if model.model_name == "logistic":
        atomic_csv(logistic_coefficients(model), output / "coefficients.csv")
    return PredictionArtifacts(metadata=metadata, output=output)


def _cmd_predict(args: argparse.Namespace) -> None:
    result = orchestrate_predict(parse_predict_request(args))
    _print_json({**result.metadata, "artifact_directory": str(result.output)})


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    current_year: int,
) -> None:
    """Register the prediction and decomposition commands."""

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
