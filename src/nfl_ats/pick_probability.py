from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.bye_edge_fade_overlay import bye_edge_flag_by_game
from nfl_ats.coach_fade_overlay import OVERLAY_WEEK_MAX as COACH_WEEK_MAX
from nfl_ats.coach_fade_overlay import year_one_by_game
from nfl_ats.data import DataContractError
from nfl_ats.division_revenge_tilt_overlay import division_revenge_side_by_game
from nfl_ats.forecast_cold_visitor_tilt_overlay import forecast_cold_visitor_flag_by_game
from nfl_ats.four_overlay_composition import (
    BYE_EDGE_FADE,
    COACH_FADE,
    COMPOSITION_ORDER,
    DIVISION_REVENGE_TILT,
    FORECAST_COLD_VISITOR_TILT,
    INTERIM_HC_FIRST_GAME_TILT,
    OWNER_HELD_MEMBERS,
    PBP08_PROTECTION_MISMATCH_TILT,
    PLAYER_ARRESTS_BACK_SIDE_POLICY,
    PRECIP_HIGH_TOTAL_TILT,
    TANK_ZONE_FADE_TILT,
)
from nfl_ats.player_arrests_back_side_overlay import _broad_side_flags
from nfl_ats.tank_zone_fade_tilt_overlay import OVERLAY_WEEK_MAX as TANK_ZONE_WEEK_MAX
from nfl_ats.tank_zone_fade_tilt_overlay import OVERLAY_WEEK_MIN as TANK_ZONE_WEEK_MIN
from nfl_ats.tank_zone_fade_tilt_overlay import tank_zone_flag_by_game

PICK_PROBABILITY_ARTIFACT_ROOT = "pick_probability"
ACTIVE_PICK_PROBABILITY_FILENAME = "active_pick_probability.json"
COEFFICIENTS_FILENAME = "coefficients.json"
METADATA_FILENAME = "metadata.json"
SCHEMA_VERSION = 1
PICK_PROBABILITY_POLICY = "four_term_pick_probability_v1"

MODEL_LOGIT_TERM = "model_logit"
FLAG_SUM_TERM = "flag_sum"
MOVE_TERM = "move_toward_home"
MOVE_AVAILABLE_TERM = "move_available"
INTERCEPT_TERM = "intercept"
COEFFICIENT_TERMS = (
    INTERCEPT_TERM,
    MODEL_LOGIT_TERM,
    FLAG_SUM_TERM,
    MOVE_TERM,
    MOVE_AVAILABLE_TERM,
)

MEMBER_FLAG_COLUMNS: dict[str, str] = {
    COACH_FADE: "flag_coach",
    DIVISION_REVENGE_TILT: "flag_division",
    PLAYER_ARRESTS_BACK_SIDE_POLICY: "flag_arrests",
    BYE_EDGE_FADE: "flag_bye",
    FORECAST_COLD_VISITOR_TILT: "flag_cold_visitor",
    PBP08_PROTECTION_MISMATCH_TILT: "flag_protection",
    INTERIM_HC_FIRST_GAME_TILT: "flag_interim_hc",
    TANK_ZONE_FADE_TILT: "flag_tank_zone",
    PRECIP_HIGH_TOTAL_TILT: "flag_precip",
}
FLAG_COLUMNS: tuple[str, ...] = tuple(MEMBER_FLAG_COLUMNS[member] for member in COMPOSITION_ORDER)
COUNTED_FLAG_COLUMNS: tuple[str, ...] = tuple(
    MEMBER_FLAG_COLUMNS[member] for member in COMPOSITION_ORDER if member not in OWNER_HELD_MEMBERS
)

FLAG_SUM_COLUMN = "composition_flag_sum"
MOVE_COLUMN = "market_move_toward_home"
MOVE_AVAILABLE_COLUMN = "market_move_available"
MODEL_HOME_PROBABILITY_COLUMN = "model_home_cover_probability"
CALIBRATED_HOME_PROBABILITY_COLUMN = "calibrated_home_cover_probability"
CALIBRATED_PICK_PROBABILITY_COLUMN = "calibrated_pick_probability"
CALIBRATED_PICK_SIDE_COLUMN = "calibrated_pick_side"
CALIBRATED_STRENGTH_WORD_COLUMN = "calibrated_strength_word"

STRENGTH_WORDS = ("slight", "lean", "strong")
STRENGTH_BAND_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
STRENGTH_ROUNDING_PLACES = 3
CONFIDENCE_BAND_EDGES = (0.50, 0.52, 0.55, 0.58, 0.62, 1.0)
PICK_SIDE_FLOOR = 0.5
PROBABILITY_EPSILON = 1e-6


class PickProbabilitySourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConfidenceBand:
    lower: float
    upper: float
    games: int
    accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "games": self.games,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConfidenceBand:
        accuracy = payload.get("accuracy")
        return cls(
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
            games=int(payload["games"]),
            accuracy=None if accuracy is None else float(accuracy),
        )


@dataclass(frozen=True)
class StrengthBand:
    word: str
    minimum: float
    games: int
    accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "minimum": self.minimum,
            "games": self.games,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StrengthBand:
        accuracy = payload.get("accuracy")
        return cls(
            word=str(payload["word"]),
            minimum=float(payload["minimum"]),
            games=int(payload["games"]),
            accuracy=None if accuracy is None else float(accuracy),
        )


@dataclass(frozen=True)
class PickProbabilityModel:
    intercept: float
    model_logit: float
    flag_sum: float
    move_toward_home: float
    move_available: float
    strength_bands: tuple[StrengthBand, ...]
    confidence_bands: tuple[ConfidenceBand, ...]
    fitted_games: int
    fitted_seasons: tuple[int, ...]
    artifact: str
    policy: str = PICK_PROBABILITY_POLICY

    @property
    def lean_minimum(self) -> float:
        return self._minimum_for("lean")

    @property
    def strong_minimum(self) -> float:
        return self._minimum_for("strong")

    def _minimum_for(self, word: str) -> float:
        for band in self.strength_bands:
            if band.word == word:
                return band.minimum
        return PICK_SIDE_FLOOR

    def home_probability(
        self,
        model_home_probability: pd.Series,
        flag_sum: pd.Series,
        move_toward_home: pd.Series,
        move_available: pd.Series,
    ) -> pd.Series:

        stated = pd.to_numeric(model_home_probability, errors="coerce").clip(
            PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
        )
        logit = np.log(stated / (1.0 - stated))
        z = (
            self.intercept
            + self.model_logit * logit
            + self.flag_sum * pd.to_numeric(flag_sum, errors="coerce").fillna(0.0)
            + self.move_toward_home * pd.to_numeric(move_toward_home, errors="coerce").fillna(0.0)
            + self.move_available * pd.to_numeric(move_available, errors="coerce").fillna(0.0)
        )
        bounded = np.clip(z.to_numpy(dtype=float), -35.0, 35.0)
        return pd.Series(1.0 / (1.0 + np.exp(-bounded)), index=stated.index, dtype=float)

    def strength_word(self, pick_probability: float) -> str:
        shown = round(float(pick_probability), STRENGTH_ROUNDING_PLACES)
        if shown >= self.strong_minimum:
            return STRENGTH_WORDS[2]
        if shown >= self.lean_minimum:
            return STRENGTH_WORDS[1]
        return STRENGTH_WORDS[0]

    def strength_words(self, pick_probability: pd.Series) -> pd.Series:
        value = pd.to_numeric(pick_probability, errors="coerce").round(STRENGTH_ROUNDING_PLACES)
        return pd.Series(
            np.select(
                [value.ge(self.strong_minimum), value.ge(self.lean_minimum)],
                [STRENGTH_WORDS[2], STRENGTH_WORDS[1]],
                default=STRENGTH_WORDS[0],
            ),
            index=pick_probability.index,
            dtype=object,
        ).where(value.notna())

    def landing_rate(self, word: str) -> float | None:
        for band in self.strength_bands:
            if band.word == word:
                return band.accuracy
        return None

    def graded_games(self) -> int:
        return sum(band.games for band in self.strength_bands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": self.policy,
            "coefficients": {
                INTERCEPT_TERM: self.intercept,
                MODEL_LOGIT_TERM: self.model_logit,
                FLAG_SUM_TERM: self.flag_sum,
                MOVE_TERM: self.move_toward_home,
                MOVE_AVAILABLE_TERM: self.move_available,
            },
            "strength_bands": [band.to_dict() for band in self.strength_bands],
            "confidence_bands": [band.to_dict() for band in self.confidence_bands],
            "counted_flag_columns": list(COUNTED_FLAG_COLUMNS),
            "held_members": sorted(OWNER_HELD_MEMBERS),
            "fitted_games": self.fitted_games,
            "fitted_seasons": list(self.fitted_seasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, artifact: str) -> PickProbabilityModel:
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise PickProbabilitySourceError(
                f"pick probability artifact {artifact} has schema version "
                f"{payload.get('schema_version')!r}, expected {SCHEMA_VERSION}"
            )
        raw = payload.get("coefficients")
        if not isinstance(raw, Mapping):
            raise PickProbabilitySourceError(
                f"pick probability artifact {artifact} has no coefficients block"
            )
        missing = [term for term in COEFFICIENT_TERMS if term not in raw]
        if missing:
            raise PickProbabilitySourceError(
                f"pick probability artifact {artifact} is missing coefficients: "
                + ", ".join(missing)
            )
        try:
            values = {term: float(raw[term]) for term in COEFFICIENT_TERMS}
        except (TypeError, ValueError) as error:
            raise PickProbabilitySourceError(
                f"pick probability artifact {artifact} has a non-numeric coefficient: {error}"
            ) from error
        if not all(np.isfinite(value) for value in values.values()):
            raise PickProbabilitySourceError(
                f"pick probability artifact {artifact} has a non-finite coefficient"
            )
        strength_raw = payload.get("strength_bands")
        if not isinstance(strength_raw, Sequence) or len(strength_raw) != len(STRENGTH_WORDS):
            raise PickProbabilitySourceError(
                f"pick probability artifact {artifact} must carry "
                f"{len(STRENGTH_WORDS)} strength bands"
            )
        strength = tuple(ConfidenceLoader.strength(entry) for entry in strength_raw)
        if [band.word for band in strength] != list(STRENGTH_WORDS):
            raise PickProbabilitySourceError(
                f"pick probability artifact {artifact} names unexpected strength words"
            )
        confidence_raw = payload.get("confidence_bands")
        confidence = (
            tuple(ConfidenceLoader.confidence(entry) for entry in confidence_raw)
            if isinstance(confidence_raw, Sequence)
            else ()
        )
        return cls(
            intercept=values[INTERCEPT_TERM],
            model_logit=values[MODEL_LOGIT_TERM],
            flag_sum=values[FLAG_SUM_TERM],
            move_toward_home=values[MOVE_TERM],
            move_available=values[MOVE_AVAILABLE_TERM],
            strength_bands=strength,
            confidence_bands=confidence,
            fitted_games=int(payload.get("fitted_games") or 0),
            fitted_seasons=tuple(int(value) for value in payload.get("fitted_seasons") or ()),
            artifact=artifact,
            policy=str(payload.get("policy") or PICK_PROBABILITY_POLICY),
        )


class ConfidenceLoader:
    @staticmethod
    def strength(entry: Any) -> StrengthBand:
        if not isinstance(entry, Mapping):
            raise PickProbabilitySourceError("strength band entries must be objects")
        try:
            return StrengthBand.from_dict(entry)
        except (KeyError, TypeError, ValueError) as error:
            raise PickProbabilitySourceError(f"unusable strength band: {error}") from error

    @staticmethod
    def confidence(entry: Any) -> ConfidenceBand:
        if not isinstance(entry, Mapping):
            raise PickProbabilitySourceError("confidence band entries must be objects")
        try:
            return ConfidenceBand.from_dict(entry)
        except (KeyError, TypeError, ValueError) as error:
            raise PickProbabilitySourceError(f"unusable confidence band: {error}") from error


def active_pick_probability_path(artifacts_root: Path) -> Path:
    return artifacts_root / ACTIVE_PICK_PROBABILITY_FILENAME


def load_pick_probability_model(artifacts_root: Path) -> PickProbabilityModel:

    pointer_path = active_pick_probability_path(artifacts_root)
    if not pointer_path.is_file():
        raise PickProbabilitySourceError(
            f"no active pick probability artifact at {pointer_path}; run "
            "`nfl-ats fit-pick-probability` before serving a per-game chance"
        )
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PickProbabilitySourceError(
            f"active pick probability pointer {pointer_path} is unreadable: {error}"
        ) from error
    if not isinstance(pointer, Mapping):
        raise PickProbabilitySourceError(
            f"active pick probability pointer {pointer_path} is not an object"
        )
    relative = str(pointer.get("artifact") or "")
    if not relative:
        raise PickProbabilitySourceError(
            f"active pick probability pointer {pointer_path} names no artifact"
        )
    coefficients_path = artifacts_root / relative / COEFFICIENTS_FILENAME
    if not coefficients_path.is_file():
        raise PickProbabilitySourceError(
            f"active pick probability artifact {relative} has no {COEFFICIENTS_FILENAME}"
        )
    try:
        payload = json.loads(coefficients_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PickProbabilitySourceError(
            f"pick probability artifact {relative} is unreadable: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise PickProbabilitySourceError(f"pick probability artifact {relative} is not an object")
    return PickProbabilityModel.from_dict(payload, artifact=relative)


def load_pick_probability_model_or_none(artifacts_root: Path) -> PickProbabilityModel | None:

    if not active_pick_probability_path(artifacts_root).is_file():
        return None
    return load_pick_probability_model(artifacts_root)


def _empty_flags(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"game_id": predictions["game_id"].astype(str)})
    for column in FLAG_COLUMNS:
        frame[column] = 0
    frame[FLAG_SUM_COLUMN] = 0.0
    return frame.reset_index(drop=True)


def signed_composition_flags(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    incidents: pd.DataFrame | None = None,
    forecasts_tuesday_noon: pd.DataFrame | None = None,
    protection_back_side: pd.DataFrame | None = None,
) -> pd.DataFrame:

    required = {"game_id", "season", "week", "gameday", "home_team", "away_team"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(
            f"Predictions are missing composition-flag columns: {', '.join(missing)}"
        )
    frame = predictions.reset_index(drop=True).copy()
    frame["game_id"] = frame["game_id"].astype(str)
    flags = pd.DataFrame({"game_id": frame["game_id"]})
    week = pd.to_numeric(frame["week"], errors="coerce").fillna(0).astype(int)

    coach = year_one_by_game(schedules)
    merged = frame.merge(coach, on=["game_id", "season"], how="left")
    year_one_home = merged["year_one_home"].fillna(False).to_numpy()
    year_one_away = merged["year_one_away"].fillna(False).to_numpy()
    eligible_coach = week.le(COACH_WEEK_MAX).to_numpy()
    flags["flag_coach"] = np.where(
        eligible_coach & year_one_away & ~year_one_home,
        1,
        np.where(eligible_coach & year_one_home & ~year_one_away, -1, 0),
    )

    revenge = division_revenge_side_by_game(schedules)
    merged = frame.merge(revenge, on=["game_id", "season"], how="left")
    revenge_home = merged["revenge_home"].fillna(False).to_numpy()
    revenge_away = merged["revenge_away"].fillna(False).to_numpy()
    flags["flag_division"] = np.where(
        revenge_home & ~revenge_away, 1, np.where(revenge_away & ~revenge_home, -1, 0)
    )

    if incidents is None or incidents.empty:
        flags["flag_arrests"] = 0
    else:
        home_flags, away_flags = _broad_side_flags(
            frame[["game_id", "gameday", "home_team", "away_team"]].copy(), incidents
        )
        home_side = home_flags.to_numpy()
        away_side = away_flags.to_numpy()
        flags["flag_arrests"] = np.where(
            home_side & ~away_side, 1, np.where(away_side & ~home_side, -1, 0)
        )

    bye = bye_edge_flag_by_game(schedules)
    merged = frame.merge(bye, on=["game_id", "season"], how="left")
    home_off_bye = merged["home_off_bye"].fillna(False).to_numpy()
    away_off_bye = merged["away_off_bye"].fillna(False).to_numpy()
    flags["flag_bye"] = np.where(
        away_off_bye & ~home_off_bye, 1, np.where(home_off_bye & ~away_off_bye, -1, 0)
    )

    if forecasts_tuesday_noon is None or forecasts_tuesday_noon.empty:
        flags["flag_cold_visitor"] = 0
    else:
        cold = forecast_cold_visitor_flag_by_game(schedules, forecasts_tuesday_noon)
        merged = frame.merge(
            cold[["game_id", "forecast_cold_visitor_flag"]], on="game_id", how="left"
        )
        flags["flag_cold_visitor"] = (
            merged["forecast_cold_visitor_flag"].fillna(False).astype(int).to_numpy()
        )

    if protection_back_side is None or protection_back_side.empty:
        flags["flag_protection"] = 0
    else:
        table = protection_back_side[["game_id", "back_side"]].copy()
        table["game_id"] = table["game_id"].astype(str)
        merged = frame.merge(table.drop_duplicates("game_id"), on="game_id", how="left")
        back_side = merged["back_side"].fillna("").astype(str).to_numpy()
        flags["flag_protection"] = np.where(
            back_side == "HOME", 1, np.where(back_side == "AWAY", -1, 0)
        )

    tank = tank_zone_flag_by_game(schedules)
    merged = frame.merge(tank, on=["game_id", "season"], how="left")
    tank_home = merged["tank_zone_home"].fillna(False).to_numpy()
    tank_away = merged["tank_zone_away"].fillna(False).to_numpy()
    eligible_tank = week.between(TANK_ZONE_WEEK_MIN, TANK_ZONE_WEEK_MAX).to_numpy()
    flags["flag_tank_zone"] = np.where(
        eligible_tank & tank_away & ~tank_home,
        1,
        np.where(eligible_tank & tank_home & ~tank_away, -1, 0),
    )

    flags["flag_interim_hc"] = 0
    flags["flag_precip"] = 0
    flags[FLAG_SUM_COLUMN] = (
        flags[list(COUNTED_FLAG_COLUMNS)].astype(float).sum(axis=1).astype(float)
    )
    return flags.reset_index(drop=True)


def signed_composition_flags_fail_open(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    incidents: pd.DataFrame | None = None,
    forecasts_tuesday_noon: pd.DataFrame | None = None,
    protection_back_side: pd.DataFrame | None = None,
) -> pd.DataFrame:

    try:
        return signed_composition_flags(
            predictions,
            schedules,
            incidents=incidents,
            forecasts_tuesday_noon=forecasts_tuesday_noon,
            protection_back_side=protection_back_side,
        )
    except (KeyError, OSError, ValueError, DataContractError):
        return _empty_flags(predictions)


def _empty_move(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": predictions["game_id"].astype(str),
            MOVE_COLUMN: 0.0,
            MOVE_AVAILABLE_COLUMN: 0.0,
        }
    ).reset_index(drop=True)


def market_move_toward_home(
    predictions: pd.DataFrame, data_root: Path | None, *, now: datetime | None = None
) -> pd.DataFrame:

    if data_root is None or "kickoff" not in predictions.columns:
        return _empty_move(predictions)
    try:
        from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes
        from nfl_ats.sharp_book_movement_features import sharp_book_movement_features

        quotes = load_decision_quotes(data_root / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
        if quotes.empty:
            return _empty_move(predictions)
        kickoff = pd.to_datetime(predictions["kickoff"], utc=True, errors="coerce")
        if kickoff.isna().all():
            return _empty_move(predictions)
        anchor = kickoff.min()
        games = pd.DataFrame(
            {
                "game_id": predictions["game_id"].astype(str),
                "commence_time_utc": kickoff,
                "week_first_commence_utc": anchor,
            }
        ).dropna(subset=["commence_time_utc"])
        if now is not None:
            games["cutoff_utc"] = pd.Timestamp(now).tz_convert("UTC")
        exposure = sharp_book_movement_features(quotes, games.drop_duplicates("game_id"))
    except (ImportError, KeyError, OSError, ValueError, DataContractError):
        return _empty_move(predictions)
    move = pd.to_numeric(exposure["leader_median_net_move"], errors="coerce").fillna(0.0)
    available = pd.to_numeric(exposure["leader_books"], errors="coerce").fillna(0.0).gt(0)
    table = pd.DataFrame(
        {
            "game_id": exposure["game_id"].astype(str),
            MOVE_COLUMN: move.where(available, 0.0),
            MOVE_AVAILABLE_COLUMN: available.astype(float),
        }
    )
    merged = (
        _empty_move(predictions)
        .drop(columns=[MOVE_COLUMN, MOVE_AVAILABLE_COLUMN])
        .merge(table, on="game_id", how="left")
    )
    merged[MOVE_COLUMN] = merged[MOVE_COLUMN].fillna(0.0)
    merged[MOVE_AVAILABLE_COLUMN] = merged[MOVE_AVAILABLE_COLUMN].fillna(0.0)
    return merged


def calibrated_pick_probability(
    predictions: pd.DataFrame,
    model: PickProbabilityModel,
    *,
    flags: pd.DataFrame | None = None,
    move: pd.DataFrame | None = None,
) -> pd.DataFrame:

    required = {"game_id", "home_team", "away_team", "home_cover_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(
            f"Predictions are missing pick-probability columns: {', '.join(missing)}"
        )
    frame = predictions.reset_index(drop=True).copy()
    frame["game_id"] = frame["game_id"].astype(str)
    resolved_flags = _empty_flags(frame) if flags is None else flags.reset_index(drop=True).copy()
    resolved_flags["game_id"] = resolved_flags["game_id"].astype(str)
    resolved_move = _empty_move(frame) if move is None else move.reset_index(drop=True).copy()
    resolved_move["game_id"] = resolved_move["game_id"].astype(str)

    joined = frame[["game_id", "home_team", "away_team", "home_cover_probability"]].merge(
        resolved_flags[["game_id", FLAG_SUM_COLUMN]], on="game_id", how="left"
    )
    joined = joined.merge(
        resolved_move[["game_id", MOVE_COLUMN, MOVE_AVAILABLE_COLUMN]], on="game_id", how="left"
    )
    if len(joined) != len(frame):
        raise DataContractError("Pick-probability inputs dropped or duplicated games")
    joined[FLAG_SUM_COLUMN] = joined[FLAG_SUM_COLUMN].fillna(0.0)
    joined[MOVE_COLUMN] = joined[MOVE_COLUMN].fillna(0.0)
    joined[MOVE_AVAILABLE_COLUMN] = joined[MOVE_AVAILABLE_COLUMN].fillna(0.0)

    model_home = pd.to_numeric(joined["home_cover_probability"], errors="coerce")
    home_probability = model.home_probability(
        model_home,
        joined[FLAG_SUM_COLUMN],
        joined[MOVE_COLUMN],
        joined[MOVE_AVAILABLE_COLUMN],
    )
    pick_home = home_probability.ge(0.5)
    pick_probability = home_probability.where(pick_home, 1.0 - home_probability)
    result = pd.DataFrame(
        {
            "game_id": joined["game_id"],
            MODEL_HOME_PROBABILITY_COLUMN: model_home,
            FLAG_SUM_COLUMN: joined[FLAG_SUM_COLUMN],
            MOVE_COLUMN: joined[MOVE_COLUMN],
            MOVE_AVAILABLE_COLUMN: joined[MOVE_AVAILABLE_COLUMN],
            CALIBRATED_HOME_PROBABILITY_COLUMN: home_probability,
            CALIBRATED_PICK_PROBABILITY_COLUMN: pick_probability.clip(lower=PICK_SIDE_FLOOR),
            CALIBRATED_PICK_SIDE_COLUMN: np.where(pick_home, "HOME", "AWAY"),
        }
    )
    result[CALIBRATED_STRENGTH_WORD_COLUMN] = model.strength_words(
        result[CALIBRATED_PICK_PROBABILITY_COLUMN]
    )
    result["pick_team"] = np.where(pick_home, joined["home_team"], joined["away_team"])
    return result


def attach_pick_probability(
    predictions: pd.DataFrame,
    model: PickProbabilityModel | None,
    *,
    flags: pd.DataFrame | None = None,
    move: pd.DataFrame | None = None,
) -> pd.DataFrame:

    if model is None:
        return predictions
    from nfl_ats.displayed_confidence import (
        DISPLAYED_PICK_PROBABILITY_COLUMN,
        DISPLAYED_STRENGTH_WORD_COLUMN,
    )

    served = calibrated_pick_probability(predictions, model, flags=flags, move=move)
    frame = predictions.reset_index(drop=True).copy()
    frame["game_id"] = frame["game_id"].astype(str)
    lookup = served.set_index("game_id")
    frame[MODEL_HOME_PROBABILITY_COLUMN] = (
        frame["game_id"].map(lookup[MODEL_HOME_PROBABILITY_COLUMN]).astype(float)
    )
    for column in (
        FLAG_SUM_COLUMN,
        MOVE_COLUMN,
        MOVE_AVAILABLE_COLUMN,
        CALIBRATED_HOME_PROBABILITY_COLUMN,
        CALIBRATED_PICK_PROBABILITY_COLUMN,
    ):
        frame[column] = frame["game_id"].map(lookup[column]).astype(float)
    frame[CALIBRATED_PICK_SIDE_COLUMN] = frame["game_id"].map(lookup[CALIBRATED_PICK_SIDE_COLUMN])
    frame[CALIBRATED_STRENGTH_WORD_COLUMN] = frame["game_id"].map(
        lookup[CALIBRATED_STRENGTH_WORD_COLUMN]
    )
    frame["home_cover_probability"] = frame[CALIBRATED_HOME_PROBABILITY_COLUMN]
    frame[DISPLAYED_PICK_PROBABILITY_COLUMN] = frame[CALIBRATED_PICK_PROBABILITY_COLUMN]
    frame[DISPLAYED_STRENGTH_WORD_COLUMN] = frame[CALIBRATED_STRENGTH_WORD_COLUMN]
    return frame


def landing_rate_sentence(model: PickProbabilityModel | None) -> str:

    if model is None:
        return ""
    graded = model.graded_games()
    strong = model.landing_rate("strong")
    slight = model.landing_rate("slight")
    if not graded or strong is None or slight is None:
        return (
            "The chance beside each pick is how often picks like this one have actually landed, "
            "not a feeling about the game."
        )
    return (
        "The chance beside each pick is how often picks like this one have actually landed: "
        f"across {graded:,} past games scored the same way, the picks this card called strong "
        f"won {strong:.0%} of the time and the ones it called slight won {slight:.0%}."
    )


__all__ = [
    "ACTIVE_PICK_PROBABILITY_FILENAME",
    "CALIBRATED_HOME_PROBABILITY_COLUMN",
    "CALIBRATED_PICK_PROBABILITY_COLUMN",
    "CALIBRATED_PICK_SIDE_COLUMN",
    "CALIBRATED_STRENGTH_WORD_COLUMN",
    "COEFFICIENTS_FILENAME",
    "COEFFICIENT_TERMS",
    "CONFIDENCE_BAND_EDGES",
    "COUNTED_FLAG_COLUMNS",
    "FLAG_COLUMNS",
    "FLAG_SUM_COLUMN",
    "METADATA_FILENAME",
    "MODEL_HOME_PROBABILITY_COLUMN",
    "MOVE_AVAILABLE_COLUMN",
    "MOVE_COLUMN",
    "PICK_PROBABILITY_ARTIFACT_ROOT",
    "PICK_PROBABILITY_POLICY",
    "PICK_SIDE_FLOOR",
    "SCHEMA_VERSION",
    "STRENGTH_BAND_QUANTILES",
    "STRENGTH_WORDS",
    "ConfidenceBand",
    "PickProbabilityModel",
    "PickProbabilitySourceError",
    "StrengthBand",
    "active_pick_probability_path",
    "attach_pick_probability",
    "calibrated_pick_probability",
    "landing_rate_sentence",
    "load_pick_probability_model",
    "load_pick_probability_model_or_none",
    "market_move_toward_home",
    "signed_composition_flags",
    "signed_composition_flags_fail_open",
]
