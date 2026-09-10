from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nfl_ats.spread_regime import BUCKETS, spread_bucket

HOME_SIDE_HINGE_COLUMNS = ("home_side_hinge_7",)
HINGE_POINTS = 7.0

PRIOR_WEIGHT_GAMES = 100.0
TRAILING_SEASONS = 5
COMPLETION_ALLOWANCE_DAYS = 1
HOME_SIDE_OFFSET_BUCKETS = ("7", "7.5-10", "10.5+")


def attach_home_side_location(frame: pd.DataFrame) -> pd.DataFrame:

    result = frame.copy()
    line = pd.to_numeric(result["spread_line"], errors="raise")
    result["home_side_hinge_7"] = (line.abs() - HINGE_POINTS).clip(lower=0.0).where(line.notna())
    return result


@dataclass(frozen=True)
class HomeSideOffsets:
    offsets: dict[str, float]
    prior_games: dict[str, int]

    def offset_for(self, spread: pd.Series) -> pd.Series:
        buckets = spread_bucket(spread)
        return buckets.map(self.offsets).astype(float).where(buckets.notna())


def fit_home_side_offsets(prior: pd.DataFrame, *, all_buckets: bool = False) -> HomeSideOffsets:

    completed = prior.loc[prior["result"].notna() & prior["point_incumbent"].notna()]
    error = completed["result"] - completed["point_incumbent"]
    buckets = spread_bucket(completed["spread_line"])
    offsets: dict[str, float] = {}
    counts: dict[str, int] = {}
    for bucket in BUCKETS:
        mask = buckets.eq(bucket)
        n = int(mask.sum())
        counts[bucket] = n
        fitted = float(error.loc[mask].sum() / (n + PRIOR_WEIGHT_GAMES)) if n else 0.0
        offsets[bucket] = fitted if all_buckets or bucket in HOME_SIDE_OFFSET_BUCKETS else 0.0
    return HomeSideOffsets(offsets=offsets, prior_games=counts)


def prior_games_for_week(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:

    gameday = pd.to_datetime(frame["gameday"])
    target = frame["season"].eq(season) & frame["week"].eq(week)
    cutoff = gameday.loc[target].min()
    completed_by = gameday + pd.Timedelta(days=COMPLETION_ALLOWANCE_DAYS)
    eligible = (
        ~target
        & completed_by.lt(cutoff)
        & frame["season"].ge(season - TRAILING_SEASONS)
        & frame["result"].notna()
    )
    return frame.loc[eligible]


def walk_forward_home_offsets(frame: pd.DataFrame, *, all_buckets: bool = False) -> pd.DataFrame:

    out = pd.DataFrame(index=frame.index)
    out["bucket"] = spread_bucket(frame["spread_line"])
    out["home_side_offset"] = np.nan
    out["prior_games_in_bucket"] = np.nan
    for _, group in frame.groupby(["season", "week"], sort=True):
        season, week = int(group["season"].iloc[0]), int(group["week"].iloc[0])
        fitted = fit_home_side_offsets(
            prior_games_for_week(frame, season, week), all_buckets=all_buckets
        )
        out.loc[group.index, "home_side_offset"] = fitted.offset_for(group["spread_line"])
        out.loc[group.index, "prior_games_in_bucket"] = (
            out.loc[group.index, "bucket"].map(fitted.prior_games).astype(float)
        )
    out["point_corrected"] = frame["point_incumbent"] + out["home_side_offset"]
    return out


def gaussian_median_cover_probability(
    lines: pd.Series | np.ndarray,
    points: pd.Series | np.ndarray,
    residual_median: pd.Series | np.ndarray | float,
    residual_std: pd.Series | np.ndarray | float,
) -> np.ndarray:

    thresholds = np.asarray(lines, dtype=float) - np.asarray(points, dtype=float)
    return np.asarray(
        stats.norm.sf(
            thresholds,
            loc=np.asarray(residual_median, dtype=float),
            scale=np.asarray(residual_std, dtype=float),
        ),
        dtype=np.float64,
    )


HOME_SIDE_OFFSET_POLICY = "home_side_offset_big_spreads_v2"
HOME_SIDE_OFFSET_SERVED = True
HOME_SIDE_OFFSET_FILENAME = "home_side_offset.json"


@dataclass(frozen=True)
class ProductionHomeSideOffsets:
    policy: str
    offsets: dict[str, float]
    prior_games: dict[str, int]
    source_path: str | None
    source_model_id: str | None
    active_model_id: str | None
    prior_rows: int
    warnings: tuple[str, ...]

    def offset_for(self, spread: pd.Series) -> pd.Series:
        return HomeSideOffsets(self.offsets, self.prior_games).offset_for(spread)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "offsets": dict(self.offsets),
            "prior_games": dict(self.prior_games),
            "prior_weight_games": PRIOR_WEIGHT_GAMES,
            "trailing_seasons": TRAILING_SEASONS,
            "source_path": self.source_path,
            "source_model_id": self.source_model_id,
            "active_model_id": self.active_model_id,
            "prior_rows": self.prior_rows,
            "warnings": list(self.warnings),
        }


def archive_prior_stream(per_game: pd.DataFrame) -> pd.DataFrame:

    required = {"game_id", "season", "week", "tue_open_home_spread", "residual_at_open", "result"}
    missing = sorted(required.difference(per_game.columns))
    if missing:
        raise ValueError(f"Opener archive is missing columns: {', '.join(missing)}")
    line = pd.to_numeric(per_game["tue_open_home_spread"], errors="coerce")
    residual = pd.to_numeric(per_game["residual_at_open"], errors="coerce")
    return pd.DataFrame(
        {
            "game_id": per_game["game_id"].astype(str),
            "season": per_game["season"].astype(int),
            "week": per_game["week"].astype(int),
            "spread_line": line,
            "point_incumbent": line + residual,
            "result": pd.to_numeric(per_game["result"], errors="coerce"),
        }
    )


def prior_rows_before(stream: pd.DataFrame, season: int, week: int) -> pd.DataFrame:

    earlier_season = stream["season"].lt(season)
    earlier_week = stream["season"].eq(season) & stream["week"].lt(week)
    eligible = (
        (earlier_season | earlier_week)
        & stream["season"].ge(season - TRAILING_SEASONS)
        & stream["result"].notna()
        & stream["point_incumbent"].notna()
    )
    return stream.loc[eligible]


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


def fit_production_home_side_offsets(
    artifacts_root: Path,
    active: Mapping[str, object] | None,
    *,
    season: int,
    week: int,
) -> ProductionHomeSideOffsets:

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
                "no opener evaluation matches the active model; offsets fitted from the newest "
                f"evaluation {evaluation.name} instead"
            )
    if evaluation is None or not (evaluation / "per_game.parquet").is_file():
        warnings.append("no opener evaluation archive found; serving zero offsets")
        return ProductionHomeSideOffsets(
            policy=HOME_SIDE_OFFSET_POLICY,
            offsets=dict.fromkeys(BUCKETS, 0.0),
            prior_games=dict.fromkeys(BUCKETS, 0),
            source_path=None,
            source_model_id=None,
            active_model_id=active_model_id,
            prior_rows=0,
            warnings=tuple(warnings),
        )
    stream = archive_prior_stream(pd.read_parquet(evaluation / "per_game.parquet"))
    prior = prior_rows_before(stream, season, week)
    fitted = fit_home_side_offsets(prior)
    source_model_id = _evaluation_model_id(evaluation)
    if active_model_id and source_model_id and source_model_id != active_model_id:
        warnings.append(
            f"offset history comes from model {source_model_id}, the active model is "
            f"{active_model_id}"
        )
    return ProductionHomeSideOffsets(
        policy=HOME_SIDE_OFFSET_POLICY,
        offsets=fitted.offsets,
        prior_games=fitted.prior_games,
        source_path=str(evaluation.relative_to(artifacts_root)).replace("\\", "/"),
        source_model_id=source_model_id,
        active_model_id=active_model_id,
        prior_rows=len(prior),
        warnings=tuple(warnings),
    )


def load_forecast_home_side_offsets(forecast_dir: Path) -> dict[str, object] | None:

    path = forecast_dir / HOME_SIDE_OFFSET_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def served_center_offsets(forecast_dir: Path | None) -> dict[str, float] | None:

    if forecast_dir is None:
        return None
    sidecar = load_forecast_home_side_offsets(forecast_dir)
    if sidecar is None or not sidecar.get("served"):
        return None
    games = sidecar.get("games")
    if not isinstance(games, list):
        return None
    return {
        str(row.get("game_id")): float(row.get("home_side_offset", 0.0) or 0.0)
        for row in games
        if isinstance(row, dict)
    }


def center_offset_for_frame(
    frame: pd.DataFrame, center_offsets: Mapping[str, float] | None
) -> np.ndarray | None:

    if center_offsets is None:
        return None
    ids = frame["game_id"].astype(str)
    return np.asarray([float(center_offsets.get(game_id, 0.0)) for game_id in ids], dtype=float)


def center_offsets_from_metadata(
    metadata: Mapping[str, object], predictions: pd.DataFrame
) -> dict[str, float] | None:

    block = metadata.get("home_side_offset")
    if not isinstance(block, Mapping):
        return None
    offsets = block.get("offsets")
    if not isinstance(offsets, Mapping) or not offsets:
        return None
    table = {str(key): float(value) for key, value in offsets.items()}
    spread = pd.to_numeric(predictions["spread_line"], errors="coerce")
    buckets = spread_bucket(spread)
    ids = predictions["game_id"].astype(str)
    return {
        game_id: float(table.get(str(bucket), 0.0)) if pd.notna(bucket) else 0.0
        for game_id, bucket in zip(ids, buckets, strict=True)
    }
