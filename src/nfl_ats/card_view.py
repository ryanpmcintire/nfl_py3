from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.best_pick import best_pick_tie_note, select_best_pick
from nfl_ats.best_pick_nomination import (
    NOMINATION_V2_ENABLED,
    NominationV2Result,
    nominate_v2_small_spread,
    nomination_v2_disclosure_note,
    nomination_v2_tie_note,
    select_served_nominee,
)
from nfl_ats.coach_fade_overlay import (
    OVERLAY_WEEK_MAX,
    OverlayResult,
    apply_coach_fade_overlay,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.displayed_confidence import (
    ProductionDisplayedConfidence,
    attach_displayed_confidence,
)
from nfl_ats.four_overlay_composition import (
    FAIL_CLOSED_MEMBERS,
    FourOverlayCompositionResult,
    apply_four_overlay_composition_for_publication,
    forecasts_for_card,
    protection_flags_for_card,
)
from nfl_ats.mass_preserving_lattice import THREE_WAY_COLUMNS
from nfl_ats.pick_probability import (
    CALIBRATED_PICK_PROBABILITY_COLUMN,
    MARKET_MOVE_FEATURE_LEGACY,
    PickProbabilityModel,
    attach_pick_probability,
    load_pick_probability_model_or_none,
    market_move_toward_home,
    signed_composition_flags_fail_open,
)
from nfl_ats.player_arrests_back_side_overlay import (
    ArrestOverlayResult,
    apply_player_arrests_back_side_overlay,
    load_latest_complete_arrest_snapshot,
)
from nfl_ats.prospective_scoring import artifact_model_config
from nfl_ats.published_picks import game_deadlines
from nfl_ats.snapshots import latest_snapshot, load_snapshot, load_verified_snapshot


def _disabled_overlay(predictions: pd.DataFrame) -> OverlayResult:

    return OverlayResult(predictions.reset_index(drop=True).copy(), (), (), OVERLAY_WEEK_MAX, False)


def resolve_overlay(predictions: pd.DataFrame, data_root: Path | None) -> OverlayResult:

    if data_root is None:
        return _disabled_overlay(predictions)
    try:
        schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    except FileNotFoundError:
        return _disabled_overlay(predictions)
    try:
        return apply_coach_fade_overlay(predictions, schedules)
    except DataContractError:
        return _disabled_overlay(predictions)


def _disabled_arrest_overlay(predictions: pd.DataFrame) -> ArrestOverlayResult:
    disabled = pd.Series(False, index=predictions.reset_index(drop=True).index, dtype=bool)
    return ArrestOverlayResult(
        overlaid_predictions=predictions.reset_index(drop=True).copy(),
        flips=(),
        home_flags=disabled.copy(),
        away_flags=disabled.copy(),
        enabled=False,
    )


def resolve_player_arrests_overlay(
    predictions: pd.DataFrame,
    data_root: Path | None,
    *,
    now: datetime | None = None,
    require_fresh: bool = False,
) -> ArrestOverlayResult:

    if data_root is None:
        if require_fresh:
            raise FileNotFoundError("Player-arrests production overlay requires a data root")
        return _disabled_arrest_overlay(predictions)
    try:
        snapshot = load_latest_complete_arrest_snapshot(data_root, now=now)
        incidents = pd.read_parquet(
            snapshot.safe_index_path, columns=["record_id", "incident_date", "team"]
        )
        result = apply_player_arrests_back_side_overlay(predictions, incidents)
        return ArrestOverlayResult(
            overlaid_predictions=result.overlaid_predictions,
            flips=result.flips,
            home_flags=result.home_flags,
            away_flags=result.away_flags,
            enabled=result.enabled,
            snapshot_id=snapshot.snapshot_id,
            snapshot_fetched_at_utc=snapshot.fetched_at_utc,
            safe_index_sha256=snapshot.safe_index_sha256,
        )
    except (FileNotFoundError, OSError, ValueError, DataContractError):
        if require_fresh:
            raise
        return _disabled_arrest_overlay(predictions)


@dataclass(frozen=True)
class BestPickNomination:
    v1_game_id: str | None
    v1_tie_note: str
    v2_result: NominationV2Result | None
    active_rule: str
    active_game_id: str | None
    active_tie_note: str
    method_note: str


def _v1_nomination(predictions: pd.DataFrame, sweep: pd.DataFrame) -> tuple[str | None, str]:

    if (
        "game_type" in predictions.columns
        and not predictions["game_type"].astype(str).eq("REG").all()
    ):
        return None, ""
    if sweep is None or sweep.empty:
        return None, ""
    if "method" in sweep.columns and "method" in predictions.columns:
        sweep = sweep.loc[sweep["method"].isin(set(predictions["method"].astype(str)))]
    best_pick_id = select_best_pick(predictions, sweep)
    if best_pick_id is None:
        return None, ""
    return best_pick_id, best_pick_tie_note(predictions, sweep)


@dataclass(frozen=True)
class V2NominationInputs:
    feature_table: str
    market_root: str
    season: int
    week: int
    regressor: str
    feature_profile: Any
    min_train_games: int


def v2_nomination_inputs(
    metadata: Mapping[str, Any], data_root: Path | None
) -> V2NominationInputs | None:

    if data_root is None:
        return None
    season = metadata.get("season")
    week = metadata.get("week")
    feature_profile = metadata.get("feature_profile")
    if season is None or week is None or not feature_profile:
        return None
    feature_table = artifact_model_config(metadata).get("feature_table")
    if not feature_table:
        return None
    min_train_games = metadata.get("min_train_games")
    return V2NominationInputs(
        feature_table=str(feature_table),
        market_root=str(data_root / "market" / "raw"),
        season=int(season),
        week=int(week),
        regressor=str(metadata.get("regressor", "ridge")),
        feature_profile=feature_profile,
        min_train_games=int(min_train_games) if min_train_games else DEFAULT_MIN_TRAIN_GAMES,
    )


def compute_v2_nomination(
    predictions: pd.DataFrame,
    metadata: Mapping[str, Any],
    data_root: Path | None,
    *,
    nominate_v2_fn: Callable[..., NominationV2Result | None] = nominate_v2_small_spread,
) -> NominationV2Result | None:

    inputs = v2_nomination_inputs(metadata, data_root)
    if inputs is None:
        return None
    try:
        features = pd.read_parquet(inputs.feature_table)
    except (OSError, ValueError):
        return None
    try:
        return nominate_v2_fn(
            predictions,
            features,
            market_root=Path(inputs.market_root),
            season=inputs.season,
            week=inputs.week,
            regressor=inputs.regressor,
            feature_profile=inputs.feature_profile,
            min_train_games=inputs.min_train_games,
        )
    except (ValueError, DataContractError):
        return None


_UNSET: Any = object()

LOCKED_BEST_PICK_NOTE = (
    "the star this card carried when the picks locked; once a game is past its deadline the "
    "star stays where the pool saw it."
)

SUNDAY_RENOMINATION_NOTE = (
    "re-nominated on Sunday morning: of the games that had not kicked off yet, this is the "
    "one the model was most sure about at the spread the pool locked on Tuesday."
)


def apply_sunday_renomination(
    base: BestPickNomination, predictions: pd.DataFrame, renominated_game_id: str | None
) -> BestPickNomination:

    if not renominated_game_id or base.active_game_id is None:
        return base
    if renominated_game_id == base.active_game_id:
        return base
    if "game_id" not in predictions.columns:
        return base
    if renominated_game_id not in set(predictions["game_id"].astype(str)):
        return base
    return replace(
        base,
        active_game_id=renominated_game_id,
        active_tie_note="",
        method_note=SUNDAY_RENOMINATION_NOTE,
    )


def apply_locked_best_pick(
    base: BestPickNomination, predictions: pd.DataFrame, locked_game_id: str | None
) -> BestPickNomination:

    if not locked_game_id or base.active_game_id is None:
        return base
    if "game_id" not in predictions.columns:
        return base
    if locked_game_id not in set(predictions["game_id"].astype(str)):
        return base
    if locked_game_id == base.active_game_id:
        if base.active_rule == "served_probability":
            return replace(base, active_tie_note="", method_note=LOCKED_BEST_PICK_NOTE)
        return base
    return replace(
        base,
        active_game_id=locked_game_id,
        active_tie_note="",
        method_note=LOCKED_BEST_PICK_NOTE,
    )


def resolve_nomination(
    predictions: pd.DataFrame,
    sweep: pd.DataFrame,
    metadata: Mapping[str, Any],
    data_root: Path | None,
    *,
    v2_result: NominationV2Result | None = _UNSET,
    nominate_v2_fn: Callable[..., NominationV2Result | None] = nominate_v2_small_spread,
    renominated_game_id: str | None = None,
    locked_game_id: str | None = None,
    as_of: datetime | None = None,
) -> BestPickNomination:

    if CALIBRATED_PICK_PROBABILITY_COLUMN in predictions.columns:
        if predictions.empty or (
            "game_type" in predictions.columns
            and not predictions["game_type"].astype(str).eq("REG").all()
        ):
            return BestPickNomination(None, "", None, "served_probability", None, "", "")
        candidates = predictions
        if as_of is not None and locked_game_id is None:
            deadlines = game_deadlines(predictions)
            if len(deadlines) != len(predictions):
                raise DataContractError(
                    "Served Best Pick requires a kickoff deadline for every game"
                )
            instant = pd.Timestamp(as_of).tz_convert("UTC")
            eligible = pd.to_datetime(
                predictions["game_id"].astype(str).map(deadlines), utc=True, errors="coerce"
            ).gt(instant)
            candidates = predictions.loc[eligible]
            if candidates.empty:
                return BestPickNomination(None, "", None, "served_probability", None, "", "")
        served_id, n_tied, _tie_break = select_served_nominee(
            candidates, probability_column=CALIBRATED_PICK_PROBABILITY_COLUMN
        )
        tie_note = (
            f"{n_tied} games share the highest estimate; the tie is resolved consistently."
            if n_tied > 1
            else ""
        )
        method_note = (
            "provisionally chosen because it has the highest estimated chance to cover "
            "among eligible games. Its estimated lead over the other picks is uncertain."
        )
        if tie_note:
            method_note += " " + tie_note
        base = BestPickNomination(
            v1_game_id=None,
            v1_tie_note="",
            v2_result=None,
            active_rule="served_probability",
            active_game_id=served_id,
            active_tie_note=tie_note,
            method_note=method_note,
        )
        return apply_locked_best_pick(base, predictions, locked_game_id)

    v1_id, v1_tie = _v1_nomination(predictions, sweep)
    resolved_v2 = (
        compute_v2_nomination(predictions, metadata, data_root, nominate_v2_fn=nominate_v2_fn)
        if v2_result is _UNSET
        else v2_result
    )

    if NOMINATION_V2_ENABLED and resolved_v2 is not None:
        base = BestPickNomination(
            v1_game_id=v1_id,
            v1_tie_note=v1_tie,
            v2_result=resolved_v2,
            active_rule="v2",
            active_game_id=resolved_v2.game_id,
            active_tie_note=nomination_v2_tie_note(resolved_v2),
            method_note=nomination_v2_disclosure_note(resolved_v2),
        )
    else:
        base = BestPickNomination(
            v1_game_id=v1_id,
            v1_tie_note=v1_tie,
            v2_result=resolved_v2,
            active_rule="v1",
            active_game_id=v1_id,
            active_tie_note=v1_tie,
            method_note="",
        )
    renominated = apply_sunday_renomination(base, predictions, renominated_game_id)
    return apply_locked_best_pick(renominated, predictions, locked_game_id)


@dataclass(frozen=True)
class CardProbabilities:
    predictions: pd.DataFrame
    overlay: OverlayResult
    arrest_overlay: ArrestOverlayResult
    production_overlay: FourOverlayCompositionResult | None
    pick_probability: PickProbabilityModel | None


@dataclass(frozen=True)
class CardView:
    predictions: pd.DataFrame
    overlay: OverlayResult
    arrest_overlay: ArrestOverlayResult
    nomination: BestPickNomination
    production_overlay: FourOverlayCompositionResult | None = None
    pick_probability: PickProbabilityModel | None = None

    @property
    def best_pick_game_id(self) -> str | None:
        return self.nomination.active_game_id


def _arrest_incidents(data_root: Path, now: datetime | None) -> pd.DataFrame:

    try:
        snapshot = load_latest_complete_arrest_snapshot(data_root, now=now)
        return pd.read_parquet(
            snapshot.safe_index_path, columns=["record_id", "incident_date", "team"]
        )
    except (FileNotFoundError, OSError, ValueError, DataContractError):
        return pd.DataFrame(columns=["record_id", "incident_date", "team"])


def _served_probability_frame(
    final_predictions: pd.DataFrame,
    predictions: pd.DataFrame,
    model: PickProbabilityModel,
    *,
    flags: pd.DataFrame | None,
    move: pd.DataFrame | None,
) -> pd.DataFrame:

    frame = final_predictions.reset_index(drop=True).copy()
    raw = predictions.reset_index(drop=True).copy()
    raw["game_id"] = raw["game_id"].astype(str)
    lookup = raw.set_index("game_id")
    frame["game_id"] = frame["game_id"].astype(str)
    for column in ("home_cover_probability", *THREE_WAY_COLUMNS):
        if column in lookup.columns:
            frame[column] = frame["game_id"].map(lookup[column]).astype(float)
    if "base_probability_policy" in lookup.columns:
        frame["base_probability_policy"] = frame["game_id"].map(lookup["base_probability_policy"])
    return attach_pick_probability(frame, model, flags=flags, move=move)


def resolve_card_probabilities(
    predictions: pd.DataFrame,
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
    displayed_confidence: ProductionDisplayedConfidence | None = None,
    artifacts_root: Path | None = None,
) -> CardProbabilities:

    pick_probability = (
        load_pick_probability_model_or_none(artifacts_root) if artifacts_root is not None else None
    )
    overlay = resolve_overlay(predictions, data_root)
    arrest_overlay = resolve_player_arrests_overlay(
        overlay.overlaid_predictions,
        data_root,
        now=now,
        require_fresh=require_fresh_arrest_overlay,
    )
    production_overlay: FourOverlayCompositionResult | None = None
    served_flags: pd.DataFrame | None = None
    served_move: pd.DataFrame | None = None
    if data_root is None:
        if require_fresh_arrest_overlay:
            raise FileNotFoundError("Four-overlay production policy requires a data root")
    else:
        try:
            schedules, _team_stats = load_verified_snapshot(latest_snapshot(data_root / "raw"))
            repo_root = data_root.resolve().parent
            forecasts_tuesday, forecasts_kickoff = forecasts_for_card(
                predictions, schedules, repo_root / "registry"
            )
            production_overlay = apply_four_overlay_composition_for_publication(
                predictions,
                schedules,
                data_root,
                now=now,
                forecasts_tuesday_noon=forecasts_tuesday,
                forecasts_kickoff_nearest=forecasts_kickoff,
            )
            disabled = [
                member.member_id
                for member in production_overlay.members
                if member.member_id in FAIL_CLOSED_MEMBERS
                and (member.status != "applied" or not member.enabled)
            ]
            if disabled:
                raise DataContractError(
                    "Four-overlay production policy has disabled members: " + ", ".join(disabled)
                )
            served_flags = signed_composition_flags_fail_open(
                predictions,
                schedules,
                incidents=_arrest_incidents(data_root, now),
                forecasts_tuesday_noon=forecasts_tuesday,
                protection_back_side=protection_flags_for_card(predictions, data_root),
            )
            served_move = market_move_toward_home(
                predictions,
                data_root,
                now=now,
                feature_version=(
                    pick_probability.market_move_feature_version
                    if pick_probability is not None
                    else MARKET_MOVE_FEATURE_LEGACY
                ),
            )
        except (FileNotFoundError, OSError, ValueError, DataContractError):
            if require_fresh_arrest_overlay:
                raise
            production_overlay = None
    final_predictions = (
        production_overlay.overlaid_predictions
        if production_overlay is not None
        else arrest_overlay.overlaid_predictions
    )
    if pick_probability is not None:
        final_predictions = _served_probability_frame(
            final_predictions,
            predictions,
            pick_probability,
            flags=served_flags,
            move=served_move,
        )
    else:
        final_predictions = attach_displayed_confidence(final_predictions, displayed_confidence)
    return CardProbabilities(
        final_predictions,
        overlay,
        arrest_overlay,
        production_overlay,
        pick_probability,
    )


def resolve_card_view(
    predictions: pd.DataFrame,
    sweep: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
    nominate_v2_fn: Callable[..., NominationV2Result | None] = nominate_v2_small_spread,
    renominated_game_id: str | None = None,
    locked_game_id: str | None = None,
    displayed_confidence: ProductionDisplayedConfidence | None = None,
    artifacts_root: Path | None = None,
) -> CardView:

    probabilities = resolve_card_probabilities(
        predictions,
        data_root=data_root,
        now=now,
        require_fresh_arrest_overlay=require_fresh_arrest_overlay,
        displayed_confidence=displayed_confidence,
        artifacts_root=artifacts_root,
    )
    nomination = resolve_nomination(
        probabilities.predictions,
        sweep,
        metadata,
        data_root,
        nominate_v2_fn=nominate_v2_fn,
        renominated_game_id=renominated_game_id,
        locked_game_id=locked_game_id,
        as_of=now,
    )
    return CardView(
        probabilities.predictions,
        probabilities.overlay,
        probabilities.arrest_overlay,
        nomination,
        probabilities.production_overlay,
        probabilities.pick_probability,
    )


__all__ = [
    "BestPickNomination",
    "CardProbabilities",
    "CardView",
    "V2NominationInputs",
    "apply_locked_best_pick",
    "apply_sunday_renomination",
    "compute_v2_nomination",
    "resolve_card_probabilities",
    "resolve_card_view",
    "resolve_nomination",
    "resolve_overlay",
    "resolve_player_arrests_overlay",
    "v2_nomination_inputs",
]
