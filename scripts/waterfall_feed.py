from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.attribution_waterfall import (
    KIND_FAMILY,
    WaterfallInputError,
    build_game_waterfall,
    family_contributions_from_ridge,
    key_number_distance,
    probability_rule_offset,
)
from nfl_ats.card_view import resolve_card_view
from nfl_ats.home_side_location import HOME_SIDE_OFFSET_FILENAME
from nfl_ats.io import atomic_json, run_id
from nfl_ats.market_decomposition import (
    FAMILY_PHRASES,
    INTERCEPT_FAMILY,
    WEEKLY_CONTEXT_FAMILY,
)
from nfl_ats.outcomes import fit_margin_models_for_week
from nfl_ats.provenance import (
    artifact_provenance,
    sha256_file,
    write_experiment_artifact,
)

WATERFALL_FEED_SCHEMA_VERSION = 1
ARTIFACT_DIRNAME = "waterfall_feed"

MAX_INPUT_STALENESS_HOURS = 36.0

REPRODUCTION_ATOL = 1e-6

PROBABILITY_TOLERANCE = 1e-6

OFFSET_SENTENCE_THRESHOLD_POINTS = 0.25

KIND_HOME_SIDE_OFFSET = "home_side_offset"

HOME_SIDE_STEP_ID = "home_side_offset"

HOME_SIDE_STEP_LABEL = "Home-side correction on big spreads"

_SENTINEL_FAMILIES = frozenset({INTERCEPT_FAMILY, WEEKLY_CONTEXT_FAMILY})

OVERLAY_PHRASES: dict[str, str] = {
    "coach_fade": "Year-one coach fade",
    "division_revenge_tilt": "Division revenge tilt",
    "player_arrests_back_side_policy": "Player-arrests back-side",
    "spread_gap_zone_fade": "Spread-gap zone fade",
}


class WaterfallFeedError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeedContext:
    artifacts_root: Path
    data_root: Path
    forecast_directory: Path
    active: dict[str, Any]
    metadata: dict[str, Any]
    season: int
    week: int
    method: str
    card: pd.DataFrame
    recommendations: pd.DataFrame
    sweep: pd.DataFrame
    features_path: Path
    features: pd.DataFrame
    feature_table_sha256: str
    feature_table_age_hours: float
    feature_table_hash_verified: bool
    home_side_offsets: dict[str, float]
    home_side_offset_policy: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WaterfallFeedError(f"Required JSON input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WaterfallFeedError(message)


def load_feed_context(
    artifacts_root: Path,
    data_root: Path,
    *,
    max_staleness_hours: float = MAX_INPUT_STALENESS_HOURS,
) -> FeedContext:

    artifacts_root = artifacts_root.resolve()
    data_root = data_root.resolve()
    active = load_active_ats_model(artifacts_root)
    _require(active is not None, f"No synchronized active ATS model under {artifacts_root}")
    assert active is not None
    forecast_directory = active_artifact_path(artifacts_root, active, "weekly_forecast")
    _require(
        forecast_directory is not None,
        "Active ATS model has no linked weekly forecast",
    )
    assert forecast_directory is not None
    metadata = _load_json(forecast_directory / "metadata.json")
    _require(
        metadata.get("active_model_id") == active.get("model_id"),
        "Weekly forecast model ID does not match the active model",
    )
    _require(
        metadata.get("synchronization_status") == "SYNCHRONIZED",
        "Weekly forecast is not synchronized with an evaluation",
    )
    method = str(active["method"])
    card = pd.read_csv(forecast_directory / "predictions.csv")
    card = card.loc[card["method"].eq(method)].reset_index(drop=True)
    _require(
        not card.empty,
        f"Published card carries no {method!r} prediction rows",
    )
    recommendations = (
        pd.read_csv(forecast_directory / "recommendations.csv")
        .loc[lambda frame: frame["method"].eq(method)]
        .reset_index(drop=True)
    )
    _require(not recommendations.empty, "Published card has no recommendation rows")
    sweep_path = forecast_directory / "line_sweep.parquet"
    sweep = pd.read_parquet(sweep_path) if sweep_path.is_file() else pd.DataFrame()
    if "method" in sweep.columns:
        sweep = sweep.loc[sweep["method"].eq(method)]

    provenance = metadata.get("provenance") or {}
    feature_record = provenance.get("feature_table") or {}
    recorded_sha = feature_record.get("sha256")
    recorded_path = feature_record.get("path")
    candidates = []
    if isinstance(recorded_path, str):
        candidates.append(Path(recorded_path))
    candidates.append(
        data_root / "processed" / f"game_features_{active.get('feature_profile')}.parquet"
    )
    features_path = next((path for path in candidates if path.is_file()), None)
    _require(
        features_path is not None,
        f"Feature table not found at any recorded location: {[str(path) for path in candidates]}",
    )
    assert features_path is not None
    actual_sha = sha256_file(features_path)
    hash_verified = isinstance(recorded_sha, str) and actual_sha == recorded_sha
    if isinstance(recorded_sha, str):
        _require(
            hash_verified,
            f"Feature table {features_path} failed its recorded sha256 check "
            f"(stale or rebuilt since the card was published)",
        )
    manifest_record = feature_record.get("manifest") or {}
    built_at_raw = manifest_record.get("built_at_utc")
    created_raw = metadata.get("created_at_utc")
    _require(
        isinstance(built_at_raw, str) and isinstance(created_raw, str),
        "Card metadata lacks built-at/created-at timestamps for the staleness guard",
    )
    built_at = pd.Timestamp(built_at_raw).to_pydatetime()
    card_created = pd.Timestamp(created_raw).to_pydatetime()
    age_hours = (card_created - built_at).total_seconds() / 3600.0
    _require(
        hash_verified or age_hours <= max_staleness_hours,
        f"Feature table behind the published card by {age_hours:.1f}h, exceeding the "
        f"{max_staleness_hours:.1f}h freshness budget, and its sha256 cannot be verified "
        "against the card record -- refusing to attribute an unverifiable stale card",
    )

    home_side_offsets, home_side_policy = _load_home_side_offsets(forecast_directory)

    return FeedContext(
        artifacts_root=artifacts_root,
        data_root=data_root,
        forecast_directory=forecast_directory,
        active=active,
        metadata=metadata,
        season=int(metadata["season"]),
        week=int(metadata["week"]),
        method=method,
        card=card,
        recommendations=recommendations,
        sweep=sweep.reset_index(drop=True),
        features_path=features_path,
        features=pd.read_parquet(features_path),
        feature_table_sha256=actual_sha,
        feature_table_age_hours=age_hours,
        feature_table_hash_verified=hash_verified,
        home_side_offsets=home_side_offsets,
        home_side_offset_policy=home_side_policy,
    )


def _load_home_side_offsets(forecast_directory: Path) -> tuple[dict[str, float], str]:

    path = forecast_directory / HOME_SIDE_OFFSET_FILENAME
    if not path.is_file():
        return {}, ""
    payload = _load_json(path)
    if not payload.get("served"):
        return {}, ""
    fit = payload.get("fit") or {}
    rows = payload.get("games")
    _require(
        isinstance(rows, list),
        f"{path} is served but carries no per-game offsets; refusing to attribute a card "
        "whose served centre correction cannot be reproduced",
    )
    assert isinstance(rows, list)
    offsets: dict[str, float] = {}
    for row in rows:
        _require(
            isinstance(row, dict) and "game_id" in row,
            f"{path} carries a per-game row without a game_id",
        )
        value = row.get("home_side_offset")
        offsets[str(row["game_id"])] = 0.0 if value is None else float(value)
    return offsets, str(fit.get("policy") or "")


def _production_resolver(
    card: pd.DataFrame, sweep: pd.DataFrame, metadata: dict[str, Any], *, data_root: Path
) -> Any:
    return resolve_card_view(
        card,
        sweep,
        metadata,
        data_root=data_root,
        require_fresh_arrest_overlay=True,
    )


def _overlay_flip_map(view: Any) -> dict[str, list[str]]:
    composition = getattr(view, "production_overlay", None)
    if composition is None:
        raise WaterfallFeedError(
            "Production four-overlay composition unavailable; refusing to emit "
            "flip events without attribution"
        )
    flipped = set(composition.union_flipped_game_ids)
    flip_map: dict[str, list[str]] = {}
    for game in composition.games:
        members = sorted(str(member) for member in game.member_ids)
        if members and str(game.game_id) in flipped:
            flip_map[str(game.game_id)] = members
    return flip_map


def _phrase_for_family(family: str) -> str:
    return FAMILY_PHRASES.get(family, family.replace("_", " "))


def _split_home_side_step(
    steps: list[dict[str, Any]],
    *,
    home_side_offset: float,
    probability_rule_offset_points: float,
    picked_side: str,
) -> list[dict[str, Any]]:

    if home_side_offset == 0.0:
        return steps
    sign = 1.0 if picked_side == "HOME" else -1.0

    def direction(delta: float) -> int:
        return 0 if delta == 0.0 else int(math.copysign(1.0, delta * sign))

    split: list[dict[str, Any]] = []
    for step in steps:
        if step["kind"] != "probability_rule":
            split.append(step)
            continue
        after_families = float(step["cumulative_points"]) - float(step["delta_points"])
        after_home_side = after_families + home_side_offset
        split.append(
            {
                "step_id": HOME_SIDE_STEP_ID,
                "label": HOME_SIDE_STEP_LABEL,
                "family": None,
                "kind": KIND_HOME_SIDE_OFFSET,
                "delta_points": home_side_offset,
                "cumulative_points": after_home_side,
                "direction": direction(home_side_offset),
            }
        )
        split.append(
            {
                **step,
                "delta_points": probability_rule_offset_points,
                "cumulative_points": after_home_side + probability_rule_offset_points,
                "direction": direction(probability_rule_offset_points),
            }
        )
    return split


def rationale_sentences(
    *,
    family_contributions: dict[str, float],
    steps_flip_overlays: list[str],
    picked_side: str,
    picked_team: str,
    probability_rule_offset_points: float,
    home_side_offset_points: float,
    predicted_residual: float,
    market_line: float,
    final_probability: float,
) -> list[str]:

    pick_sign = 1.0 if picked_side == "HOME" else -1.0
    ranked = sorted(
        (
            (family, value)
            for family, value in family_contributions.items()
            if family not in _SENTINEL_FAMILIES
        ),
        key=lambda item: (-abs(item[1]), item[0]),
    )[:2]
    sentences: list[str] = []
    for family, value in ranked:
        direction = "toward" if pick_sign * value >= 0.0 else "against"
        sentences.append(
            f"{_phrase_for_family(family)} moves this {abs(float(value)):.1f} points "
            f"{direction} {picked_team}"
        )
    for overlay in steps_flip_overlays:
        label = OVERLAY_PHRASES.get(overlay, overlay.replace("_", " ").capitalize())
        sentences.append(f"{label} flips this pick")
    if abs(home_side_offset_points) > OFFSET_SENTENCE_THRESHOLD_POINTS:
        home_direction = "toward" if pick_sign * home_side_offset_points >= 0.0 else "against"
        sentences.append(
            f"The big-spread home-side correction moves this "
            f"{abs(home_side_offset_points):.1f} points {home_direction} {picked_team}"
        )
    if abs(probability_rule_offset_points) > OFFSET_SENTENCE_THRESHOLD_POINTS:
        sentences.append(
            f"The residual-sample probability rule shifts every pick "
            f"{probability_rule_offset_points:+.1f} points"
        )
    sentences.append(
        f"The model breaks from the market by {abs(predicted_residual):.2f} points on a "
        f"line of {market_line:+.1f}; the card takes {picked_team} at {final_probability:.3f}"
    )
    return sentences


def allowed_rationale_numbers(
    game: dict[str, Any], probability_rule_offset_points: float
) -> set[str]:

    def plain(value: str) -> str:
        return value.lstrip("+-")

    allowed: set[str] = set()
    for step in game["steps"]:
        if step["kind"] == KIND_FAMILY and step["family"] not in _SENTINEL_FAMILIES:
            allowed.add(f"{abs(float(step['delta_points'])):.1f}")
    offset = float(probability_rule_offset_points)
    allowed.add(plain(f"{offset:+.1f}"))
    allowed.add(f"{abs(float(game.get('home_side_offset_points', 0.0))):.1f}")
    allowed.add(f"{abs(float(game['predicted_residual'])):.2f}")
    allowed.add(plain(f"{float(game['market_line']):+.1f}"))
    allowed.add(f"{float(game['final_probability']):.3f}")
    return allowed


def audit_rationale_numbers(feed: dict[str, Any]) -> None:

    pattern = r"\d+\.\d+"
    offset = float(feed["probability_rule_offset_points"])
    for game in feed["games"]:
        allowed = allowed_rationale_numbers(game, offset)
        for sentence in game["rationale_sentences"]:
            for token in re.findall(pattern, str(sentence)):
                if token not in allowed:
                    raise WaterfallFeedError(
                        f"Rationale sentence contains a non-field-sourced number "
                        f"{token!r}: {sentence!r}"
                    )


def build_feed(
    context: FeedContext,
    *,
    now: datetime | None = None,
    fit_fn: Callable[..., tuple[pd.DataFrame, dict[str, Any]]] = fit_margin_models_for_week,
    resolve_card_fn: Callable[..., Any] = _production_resolver,
) -> dict[str, Any]:

    instant = now or datetime.now(UTC)
    target_games, models = fit_fn(
        context.features,
        season=context.season,
        week=context.week,
        regressor=str(context.active.get("regressor", "ridge")),
        min_train_games=int(context.metadata["min_train_games"]),
        feature_profile=str(context.active["feature_profile"]),
        ridge_alpha=float(context.active["ridge_alpha"]),
        methods=(context.method,),
    )
    model = models.get(context.method)
    if model is None:
        raise WaterfallFeedError(f"Fitted models carry no {context.method!r} entry")

    scored = model.predict(target_games)
    scored = scored.assign(game_id=target_games["game_id"].to_numpy()).set_index("game_id")
    card_indexed = context.card.set_index("game_id")
    missing = sorted(set(card_indexed.index).difference(scored.index))
    extra = sorted(set(scored.index).difference(card_indexed.index))
    _require(
        not missing and not extra,
        f"Refit games do not match the published card (missing={missing}, extra={extra})",
    )
    served_offsets = (
        pd.Series(context.home_side_offsets, dtype="float64")
        .reindex(scored.index)
        .fillna(0.0)
        .astype(float)
    )
    center_gap = float(
        (scored["predicted_margin"] + served_offsets - card_indexed["predicted_margin"]).abs().max()
    )
    residual_gap = float(
        (
            scored["predicted_market_residual"]
            + served_offsets
            - card_indexed["predicted_market_residual"]
        )
        .abs()
        .max()
    )
    _require(
        center_gap <= REPRODUCTION_ATOL and residual_gap <= REPRODUCTION_ATOL,
        f"Refit does not reproduce the published card "
        f"(max |center| delta {center_gap:.3g}, max |residual| delta {residual_gap:.3g}); "
        "the card or feature table has moved since publication",
    )

    feature_frame = target_games.loc[:, list(model.feature_columns)]
    contributions = family_contributions_from_ridge(
        model.estimator,
        feature_frame,
        feature_columns=model.feature_columns,
    )
    contributions_by_game = {
        str(gid): totals for gid, totals in zip(target_games["game_id"], contributions, strict=True)
    }

    offset = probability_rule_offset(model.residuals)
    probability_method = str(context.active.get("probability_method", "ecdf"))
    if probability_method == "gaussian":
        offset = float(np.mean(np.asarray(model.residuals, dtype=np.float64)))
    elif probability_method == "gaussian_median":
        offset = float(np.median(np.asarray(model.residuals, dtype=np.float64)))
    elif probability_method != "ecdf":
        raise WaterfallFeedError(
            f"Unsupported deployed probability method {probability_method!r}; "
            "the pick-space waterfall needs 'ecdf', 'gaussian' or 'gaussian_median'"
        )

    view = resolve_card_fn(
        context.recommendations,
        context.sweep,
        context.metadata,
        data_root=context.data_root,
    )
    played = view.predictions
    played_probabilities = {
        str(gid): float(prob)
        for gid, prob in zip(played["game_id"], played["home_cover_probability"], strict=True)
    }
    flip_map = _overlay_flip_map(view)

    recs_indexed = context.recommendations.set_index("game_id")
    games: list[dict[str, Any]] = []
    for game_id, row in recs_indexed.iterrows():
        game_id = str(game_id)
        residual = float(row["predicted_market_residual"])
        home_side_offset = float(context.home_side_offsets.get(game_id, 0.0))
        model_residual = residual - home_side_offset
        line = float(row["spread_line"])
        raw_probability = float(row["home_cover_probability"])
        firing = [{"overlay": member, "fires": True} for member in flip_map.get(game_id, [])]
        try:
            waterfall = build_game_waterfall(
                game_id=game_id,
                market_line=0.0,
                predicted_residual=model_residual,
                family_contributions=dict(contributions_by_game[game_id]),
                probability_rule_offset_points=float(offset) + home_side_offset,
                raw_home_cover_probability=raw_probability,
                overlays=firing,
            )
        except WaterfallInputError as error:
            raise WaterfallFeedError(
                f"Waterfall reconciliation failed for {game_id}: {error}"
            ) from error
        played_probability = played_probabilities[game_id]
        played_side = "HOME" if played_probability >= 0.5 else "AWAY"
        if waterfall.picked_side != played_side:
            raise WaterfallFeedError(
                f"Reconstructed pick {waterfall.picked_side} disagrees with the played "
                f"card pick {played_side} for {game_id}"
            )
        if abs(waterfall.final_probability - played_probability) > PROBABILITY_TOLERANCE:
            raise WaterfallFeedError(
                f"Reconstructed final probability {waterfall.final_probability:.6f} "
                f"disagrees with the played card {played_probability:.6f} for {game_id}"
            )
        kickoff = pd.Timestamp(row["kickoff"]).isoformat()
        picked_team = (
            str(row["home_team"]) if waterfall.picked_side == "HOME" else str(row["away_team"])
        )
        games.append(
            {
                "game_id": game_id,
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "kickoff": kickoff,
                "market_line": line,
                "predicted_residual": residual,
                "final_probability": waterfall.final_probability,
                "picked_side": waterfall.picked_side,
                "edge_vs_spread": abs(residual),
                "key_number_distance": key_number_distance(line + residual),
                "home_side_offset_points": home_side_offset,
                "steps": _split_home_side_step(
                    [
                        {
                            **asdict(step),
                            "delta_points": step.delta_points + 0.0,
                            "cumulative_points": step.cumulative_points + 0.0,
                        }
                        for step in waterfall.steps
                    ],
                    home_side_offset=home_side_offset,
                    probability_rule_offset_points=float(offset),
                    picked_side=waterfall.picked_side,
                ),
                "flip_events": [asdict(event) for event in waterfall.flip_events],
                "rationale_sentences": rationale_sentences(
                    family_contributions=dict(contributions_by_game[game_id]),
                    steps_flip_overlays=[event.overlay for event in waterfall.flip_events],
                    picked_side=waterfall.picked_side,
                    picked_team=picked_team,
                    probability_rule_offset_points=float(offset),
                    home_side_offset_points=home_side_offset,
                    predicted_residual=residual,
                    market_line=line,
                    final_probability=waterfall.final_probability,
                ),
            }
        )

    games.sort(key=lambda game: (game["kickoff"], game["game_id"]))
    feed: dict[str, Any] = {
        "schema_version": WATERFALL_FEED_SCHEMA_VERSION,
        "created_at_utc": instant.isoformat(),
        "source_forecast": context.forecast_directory.relative_to(
            context.artifacts_root
        ).as_posix(),
        "active_model_id": str(context.active["model_id"]),
        "season": context.season,
        "week": context.week,
        "method": context.method,
        "feature_table_sha256": context.feature_table_sha256,
        "feature_table_hash_verified": context.feature_table_hash_verified,
        "feature_table_age_hours": round(context.feature_table_age_hours, 1),
        "probability_rule_offset_points": float(offset),
        "probability_method": probability_method,
        "home_side_offset_policy": context.home_side_offset_policy,
        "games": games,
    }
    audit_rationale_numbers(feed)
    return feed


def write_feed(
    feed: dict[str, Any],
    artifacts_root: Path,
    *,
    now: datetime | None = None,
    features_path: Path | None = None,
    registry_root: Path | None = None,
) -> Path:

    instant = now or datetime.now(UTC)
    directory = artifacts_root / ARTIFACT_DIRNAME / run_id(instant)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(feed, directory / "feed.json")
    configuration = {
        "command": "waterfall-feed",
        "source_forecast": feed["source_forecast"],
        "season": feed["season"],
        "week": feed["week"],
        "method": feed["method"],
        "feature_table_sha256": feed["feature_table_sha256"],
        "feature_table_hash_verified": feed["feature_table_hash_verified"],
    }
    manifest = {
        "artifact": ARTIFACT_DIRNAME,
        "schema_version": WATERFALL_FEED_SCHEMA_VERSION,
        "created_at_utc": feed["created_at_utc"],
        "source_forecast": feed["source_forecast"],
        "season": feed["season"],
        "week": feed["week"],
        "games": len(feed["games"]),
        "files": [{"name": "feed.json", "sha256": sha256_file(directory / "feed.json")}],
    }
    metrics = {
        "games": len(feed["games"]),
        "season": feed["season"],
        "week": feed["week"],
        "probability_rule_offset_points": feed["probability_rule_offset_points"],
        "feature_table_age_hours": feed["feature_table_age_hours"],
    }
    if features_path is not None:
        manifest["provenance"] = artifact_provenance(configuration, features_path)
        write_experiment_artifact(
            directory,
            "manifest.json",
            manifest,
            command="waterfall-feed",
            metrics=metrics,
            registry_root=registry_root,
        )
    else:
        atomic_json(manifest, directory / "manifest.json")
    pointer = {
        "latest": directory.name,
        "created_at_utc": feed["created_at_utc"],
        "manifest_sha256": sha256_file(directory / "manifest.json"),
    }
    atomic_json(pointer, artifacts_root / ARTIFACT_DIRNAME / "latest.json")
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-game waterfall feed for the Week Board, built from the "
        "active model's published weekly card"
    )
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--max-staleness-hours",
        type=float,
        default=MAX_INPUT_STALENESS_HOURS,
        help="maximum hours a hashed input may trail the published card",
    )
    args = parser.parse_args(argv)
    context = load_feed_context(
        args.artifacts_root,
        args.data_root,
        max_staleness_hours=args.max_staleness_hours,
    )
    feed = build_feed(context)
    directory = write_feed(
        feed,
        args.artifacts_root,
        features_path=context.features_path,
    )
    print(
        json.dumps(
            {
                "artifact_directory": str(directory),
                "games": len(feed["games"]),
                "season": feed["season"],
                "week": feed["week"],
                "probability_rule_offset_points": feed["probability_rule_offset_points"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
