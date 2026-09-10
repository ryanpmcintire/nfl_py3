"""MOD-18 lane AH: the displayed score, calibrated to how the model has actually done.

Predeclared in ``docs/displayed_confidence.md``. The served model states one
confidence and reuses it at every line size -- 55.6% to 56.5% across the four
buckets -- while its realised accuracy runs 56.2% / 51.4% / 51.6% / 47.3%
(``docs/spread_hole_diagnosis.md``). This module replaces the number a READER
sees with the shrunken realised correctness of that pick's own cell, four line
buckets by three stated-probability bands, fitted only on games completed
before the card's Tuesday.

Nothing here touches a side. The pick is fixed before the transform runs and
the transform is applied to the pick's own oriented probability, so a
calibrated value below 0.5 is displayed as it is rather than flipping anything
(AGENTS.md's ban on unexplained threshold flips). The estimator -- 20 fixed
pseudo-observations shrinking toward the game's own stated probability -- is
MOD-18 C3's, carried over rather than re-tuned
(``docs/spread_regime_program.md``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DISPLAY_BUCKETS = ("0-6.5", "7", "7.5-10", "10.5+")
PROBABILITY_BANDS = ("[0.50, 0.55)", "[0.55, 0.60)", "[0.60, 1]")
PSEUDO_OBSERVATIONS = 20.0

DISPLAYED_CONFIDENCE_POLICY = "displayed_confidence_reliability_v1"
DISPLAYED_CONFIDENCE_SERVED = True
DISPLAYED_CONFIDENCE_FILENAME = "displayed_confidence.json"
DISPLAYED_PICK_PROBABILITY_COLUMN = "displayed_pick_probability"
DISPLAYED_STRENGTH_WORD_COLUMN = "displayed_strength_word"

STRENGTH_WORDS = ("slight", "lean", "strong")
STRENGTH_BAND_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
STRENGTH_ROUNDING_PLACES = 3


def display_spread_bucket(spread: pd.Series) -> pd.Series:
    """The four line-size buckets the diagnosis and the Model page already report."""

    size = pd.to_numeric(spread, errors="raise").abs()
    return pd.Series(
        np.select(
            [size.le(6.5), size.lt(7.5), size.le(10)],
            DISPLAY_BUCKETS[:3],
            default=DISPLAY_BUCKETS[3],
        ),
        index=spread.index,
        dtype=object,
    ).where(size.notna())


def probability_band(stated: pd.Series) -> pd.Series:
    """MOD-18 C3's three stated-probability bands, verbatim."""

    value = pd.to_numeric(stated, errors="raise")
    return pd.Series(
        np.select(
            [value.lt(0.55), value.lt(0.60)],
            PROBABILITY_BANDS[:2],
            default=PROBABILITY_BANDS[2],
        ),
        index=stated.index,
        dtype=object,
    ).where(value.notna())


def cell_keys(spread: pd.Series, stated: pd.Series) -> pd.Series:
    bucket = display_spread_bucket(spread)
    band = probability_band(stated)
    return (bucket.astype(str) + "/" + band.astype(str)).where(bucket.notna() & band.notna())


@dataclass(frozen=True)
class ReliabilityCells:
    """Prior games and prior correct picks per (bucket, band) cell."""

    games: dict[str, int]
    wins: dict[str, float]

    def calibrate(self, stated: pd.Series, spread: pd.Series) -> pd.Series:
        keys = cell_keys(spread, stated)
        prior_games = keys.map(self.games).astype(float).fillna(0.0)
        prior_wins = keys.map(self.wins).astype(float).fillna(0.0)
        value = pd.to_numeric(stated, errors="raise")
        calibrated = (prior_wins + PSEUDO_OBSERVATIONS * value) / (
            prior_games + PSEUDO_OBSERVATIONS
        )
        return calibrated.where(keys.notna(), value)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for bucket in DISPLAY_BUCKETS:
            for band in PROBABILITY_BANDS:
                key = f"{bucket}/{band}"
                games = int(self.games.get(key, 0))
                wins = float(self.wins.get(key, 0.0))
                rows.append(
                    {
                        "bucket": bucket,
                        "band": band,
                        "prior_games": games,
                        "prior_wins": wins,
                        "realised": wins / games if games else float("nan"),
                    }
                )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "games": dict(self.games),
            "wins": dict(self.wins),
            "pseudo_observations": PSEUDO_OBSERVATIONS,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReliabilityCells:
        stored_games: Mapping[str, Any] = payload.get("games") or {}
        stored_wins: Mapping[str, Any] = payload.get("wins") or {}
        return cls(
            games={str(key): int(value) for key, value in stored_games.items()},
            wins={str(key): float(value) for key, value in stored_wins.items()},
        )


def fit_reliability_cells(prior: pd.DataFrame) -> ReliabilityCells:
    """Count games and correct picks per cell over completed rows of ``prior``."""

    completed = prior.loc[prior["correct"].notna()]
    keys = cell_keys(completed["spread_line"], completed["stated"])
    correct = pd.to_numeric(completed["correct"], errors="raise")
    games: dict[str, int] = {}
    wins: dict[str, float] = {}
    for key, group in correct.groupby(keys):
        games[str(key)] = int(group.size)
        wins[str(key)] = float(group.sum())
    return ReliabilityCells(games=games, wins=wins)


def archive_display_stream(per_game: pd.DataFrame) -> pd.DataFrame:
    """The opener-evaluation archive as the frame this module calibrates on.

    ``stated`` is the served opener probability oriented to the model's own
    probability-rule pick -- the number the card prints -- and ``correct`` is
    whether that pick was right at the opener.
    """

    required = {
        "game_id",
        "season",
        "week",
        "tue_open_home_spread",
        "home_cover_probability_at_open",
        "pick_home_at_open_probability_rule",
        "correct_at_open_probability_rule",
    }
    missing = sorted(required.difference(per_game.columns))
    if missing:
        raise ValueError(f"Opener archive is missing columns: {', '.join(missing)}")
    home_probability = pd.to_numeric(per_game["home_cover_probability_at_open"], errors="coerce")
    pick_home = per_game["pick_home_at_open_probability_rule"].astype("boolean")
    stated = home_probability.where(pick_home.fillna(True), 1.0 - home_probability)
    stream = pd.DataFrame(
        {
            "game_id": per_game["game_id"].astype(str),
            "season": per_game["season"].astype(int),
            "week": per_game["week"].astype(int),
            "spread_line": pd.to_numeric(per_game["tue_open_home_spread"], errors="coerce"),
            "stated": stated,
            "correct": pd.to_numeric(per_game["correct_at_open_probability_rule"], errors="coerce"),
        }
    )
    stream["bucket"] = display_spread_bucket(stream["spread_line"])
    stream["band"] = probability_band(stream["stated"])
    return stream.loc[stream["spread_line"].notna() & stream["stated"].notna()].reset_index(
        drop=True
    )


def prior_rows_before(stream: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Completed archive rows from a strictly earlier week, expanding from 2020."""

    earlier_season = stream["season"].lt(season)
    earlier_week = stream["season"].eq(season) & stream["week"].lt(week)
    return stream.loc[(earlier_season | earlier_week) & stream["correct"].notna()]


def walk_forward_displayed_confidence(stream: pd.DataFrame) -> pd.Series:
    """Each row's calibrated display, fitted only on strictly earlier weeks."""

    calibrated = pd.Series(np.nan, index=stream.index, dtype=float)
    for _, group in stream.groupby(["season", "week"], sort=True):
        season = int(group["season"].iloc[0])
        week = int(group["week"].iloc[0])
        cells = fit_reliability_cells(prior_rows_before(stream, season, week))
        calibrated.loc[group.index] = cells.calibrate(group["stated"], group["spread_line"])
    return calibrated


@dataclass(frozen=True)
class StrengthBands:
    """Where the board's slight/lean/strong meter cuts the displayed score."""

    lean_min: float
    strong_min: float

    def word(self, probability: float) -> str:
        shown = round(float(probability), STRENGTH_ROUNDING_PLACES)
        if shown >= self.strong_min:
            return STRENGTH_WORDS[2]
        if shown >= self.lean_min:
            return STRENGTH_WORDS[1]
        return STRENGTH_WORDS[0]

    def words(self, probability: pd.Series) -> pd.Series:
        value = pd.to_numeric(probability, errors="coerce").round(STRENGTH_ROUNDING_PLACES)
        return pd.Series(
            np.select(
                [value.ge(self.strong_min), value.ge(self.lean_min)],
                [STRENGTH_WORDS[2], STRENGTH_WORDS[1]],
                default=STRENGTH_WORDS[0],
            ),
            index=probability.index,
            dtype=object,
        ).where(value.notna())

    def to_dict(self) -> dict[str, float]:
        return {"lean_min": self.lean_min, "strong_min": self.strong_min}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> StrengthBands | None:
        if not isinstance(payload, Mapping):
            return None
        try:
            lean = float(payload["lean_min"])
            strong = float(payload["strong_min"])
        except (KeyError, TypeError, ValueError):
            return None
        return cls(lean_min=lean, strong_min=strong) if lean <= strong else None


def derive_strength_bands(stream: pd.DataFrame) -> StrengthBands | None:
    """Terciles of the archive's own walk-forward calibrated scores.

    The meter says where a pick sits among the reads this model actually
    produces, so its two edges are quantiles of that distribution rather than
    round numbers. Measured on the 2020-2025 opener archive, realised accuracy
    does NOT rise across the three bands, so the bands are relative standing
    and never a promised hit rate -- see ``docs/displayed_confidence.md``.
    """

    scored = stream.loc[stream["correct"].notna()]
    if scored.empty:
        return None
    calibrated = walk_forward_displayed_confidence(scored.copy()).dropna()
    if calibrated.empty:
        return None
    lower, upper = np.quantile(calibrated.to_numpy(dtype=float), STRENGTH_BAND_QUANTILES)
    return StrengthBands(
        lean_min=round(float(lower), STRENGTH_ROUNDING_PLACES),
        strong_min=round(float(upper), STRENGTH_ROUNDING_PLACES),
    )


@dataclass(frozen=True)
class ProductionDisplayedConfidence:
    """Cells fitted for one target week from an archived out-of-time stream."""

    policy: str
    cells: ReliabilityCells
    source_path: str | None
    source_model_id: str | None
    active_model_id: str | None
    prior_rows: int
    warnings: tuple[str, ...]
    bands: StrengthBands | None = None

    @property
    def served(self) -> bool:
        return DISPLAYED_CONFIDENCE_SERVED and self.prior_rows > 0

    def calibrate(self, stated: pd.Series, spread: pd.Series) -> pd.Series:
        if not self.served:
            return pd.to_numeric(stated, errors="raise")
        return self.cells.calibrate(stated, spread)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "served": self.served,
            "cells": self.cells.to_dict(),
            "table": self.cells.to_frame().to_dict(orient="records"),
            "strength_bands": self.bands.to_dict() if self.bands is not None else None,
            "strength_band_quantiles": list(STRENGTH_BAND_QUANTILES),
            "source_path": self.source_path,
            "source_model_id": self.source_model_id,
            "active_model_id": self.active_model_id,
            "prior_rows": self.prior_rows,
            "warnings": list(self.warnings),
        }


def _newest_opener_evaluation(artifacts_root: Path) -> Path | None:
    root = artifacts_root / "opener_evaluation"
    if not root.is_dir():
        return None
    candidates = sorted(
        (path for path in root.iterdir() if (path / "per_game.parquet").is_file()),
        key=lambda path: path.name,
    )
    return candidates[-1] if candidates else None


def _evaluation_model_id(evaluation: Path) -> str | None:
    metadata = evaluation / "metadata.json"
    if not metadata.is_file():
        return None
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except ValueError:
        return None
    for key in ("active_model_id", "model_id"):
        value = payload.get(key)
        if value:
            return str(value)
    return None


def _empty_production(
    active_model_id: str | None, warnings: list[str]
) -> ProductionDisplayedConfidence:
    return ProductionDisplayedConfidence(
        policy=DISPLAYED_CONFIDENCE_POLICY,
        cells=ReliabilityCells(games={}, wins={}),
        source_path=None,
        source_model_id=None,
        active_model_id=active_model_id,
        prior_rows=0,
        warnings=tuple(warnings),
    )


def fit_production_displayed_confidence(
    artifacts_root: Path,
    active: Mapping[str, object] | None,
    *,
    season: int,
    week: int,
) -> ProductionDisplayedConfidence:
    """Fit the served display cells for one week from archived out-of-time picks.

    History precedence, all read-only and mirroring
    :func:`nfl_ats.home_side_location.fit_production_home_side_offsets`: the
    opener evaluation matched to the active model, else the newest evaluation
    of any model with the mismatch recorded as a warning, else no cells at all
    (the stated probability is then displayed unchanged). The card is never
    blocked by this layer.
    """

    warnings: list[str] = []
    active_model_id = str(active.get("model_id")) if active and active.get("model_id") else None
    evaluation: Path | None = None
    if active is not None:
        try:
            from nfl_ats.public_board import find_matching_opener_evaluation

            matched = find_matching_opener_evaluation(artifacts_root, dict(active))
            evaluation = matched[1] if matched is not None else None
        except Exception as error:
            warnings.append(f"matching opener evaluation lookup failed: {error}")
            evaluation = None
    if evaluation is None:
        evaluation = _newest_opener_evaluation(artifacts_root)
        if evaluation is not None:
            warnings.append(
                "no opener evaluation matches the active model; display cells fitted from the "
                f"newest evaluation {evaluation.name} instead"
            )
    if evaluation is None or not (evaluation / "per_game.parquet").is_file():
        warnings.append("no opener evaluation archive found; displaying the stated probability")
        return _empty_production(active_model_id, warnings)
    try:
        stream = archive_display_stream(pd.read_parquet(evaluation / "per_game.parquet"))
    except (OSError, ValueError) as error:
        warnings.append(f"opener evaluation archive unusable: {error}")
        return _empty_production(active_model_id, warnings)
    prior = prior_rows_before(stream, season, week)
    source_model_id = _evaluation_model_id(evaluation)
    if active_model_id and source_model_id and source_model_id != active_model_id:
        warnings.append(
            f"display history comes from model {source_model_id}, the active model is "
            f"{active_model_id}"
        )
    try:
        bands = derive_strength_bands(stream)
    except (KeyError, ValueError) as error:
        warnings.append(f"strength bands could not be derived: {error}")
        bands = None
    if bands is None:
        warnings.append("no strength bands derived; the board shows no strength word")
    return ProductionDisplayedConfidence(
        policy=DISPLAYED_CONFIDENCE_POLICY,
        cells=fit_reliability_cells(prior),
        source_path=str(evaluation.relative_to(artifacts_root)).replace("\\", "/"),
        source_model_id=source_model_id,
        active_model_id=active_model_id,
        prior_rows=len(prior),
        warnings=tuple(warnings),
        bands=bands,
    )


def served_strength_bands(
    artifacts_root: Path, active: Mapping[str, object] | None
) -> StrengthBands | None:
    """The meter edges the board serves, from the active model's own archive."""

    return fit_production_displayed_confidence(artifacts_root, active, season=0, week=0).bands


def attach_displayed_confidence(
    predictions: pd.DataFrame, calibration: ProductionDisplayedConfidence | None
) -> pd.DataFrame:
    """Add the pick-oriented displayed probability without touching any side.

    The pick is read from ``home_cover_probability`` and never re-derived from
    the calibrated value, so a cell whose realised correctness sits below 0.5
    lowers the printed number and changes nothing else.
    """

    frame = predictions.copy()
    if "home_cover_probability" not in frame or "spread_line" not in frame:
        return frame
    home_probability = pd.to_numeric(frame["home_cover_probability"], errors="coerce")
    pick_home = home_probability.ge(0.5)
    stated = home_probability.where(pick_home, 1.0 - home_probability)
    if calibration is None or not calibration.served:
        frame[DISPLAYED_PICK_PROBABILITY_COLUMN] = stated
        return frame
    frame[DISPLAYED_PICK_PROBABILITY_COLUMN] = calibration.calibrate(stated, frame["spread_line"])
    if calibration.bands is not None:
        frame[DISPLAYED_STRENGTH_WORD_COLUMN] = calibration.bands.words(
            frame[DISPLAYED_PICK_PROBABILITY_COLUMN]
        )
    return frame


def displayed_pick_probability(row: Mapping[str, Any] | pd.Series) -> float | None:
    """One row's displayed score, or ``None`` when the row does not carry one."""

    value: Any = row.get(DISPLAYED_PICK_PROBABILITY_COLUMN)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(number) else number


def displayed_strength_word(row: Mapping[str, Any] | pd.Series) -> str | None:
    """One row's derived strength word, or ``None`` when the row has none."""

    value: Any = row.get(DISPLAYED_STRENGTH_WORD_COLUMN)
    word = str(value).strip().lower() if isinstance(value, str) else ""
    return word if word in STRENGTH_WORDS else None


__all__ = [
    "DISPLAYED_CONFIDENCE_FILENAME",
    "DISPLAYED_CONFIDENCE_POLICY",
    "DISPLAYED_CONFIDENCE_SERVED",
    "DISPLAYED_PICK_PROBABILITY_COLUMN",
    "DISPLAYED_STRENGTH_WORD_COLUMN",
    "DISPLAY_BUCKETS",
    "PROBABILITY_BANDS",
    "PSEUDO_OBSERVATIONS",
    "STRENGTH_BAND_QUANTILES",
    "STRENGTH_ROUNDING_PLACES",
    "STRENGTH_WORDS",
    "ProductionDisplayedConfidence",
    "ReliabilityCells",
    "StrengthBands",
    "archive_display_stream",
    "attach_displayed_confidence",
    "cell_keys",
    "derive_strength_bands",
    "display_spread_bucket",
    "displayed_pick_probability",
    "displayed_strength_word",
    "fit_production_displayed_confidence",
    "fit_reliability_cells",
    "prior_rows_before",
    "probability_band",
    "served_strength_bands",
    "walk_forward_displayed_confidence",
]
