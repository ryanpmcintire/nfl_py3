"""MOD-18 lane S: symmetric home-side location correction (docs/home_side_location.md).

The active model's point forecast under-locates the HOME team on big spreads
on both sides (lane Q's true home split of Diagnosis D). Two frozen shapes:

* S1 -- one row-local ridge feature, ``max(0, |spread_line| - 7)``, so a
  home-side location term may grow with the spread size on either side.
* S2 -- a correction OUTSIDE the ridge: a per-spread-bucket offset added to
  the incumbent's point forecast, equal to the shrunken mean of (actual home
  margin minus the incumbent's out-of-time point) over PRIOR completed games
  only, then mapped with the unchanged gaussian_median read.

Both use only information available before each row's kickoff.
"""

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
    """Add ``home_side_hinge_7`` = max(0, |spread_line| - 7) from this row's line only."""

    result = frame.copy()
    line = pd.to_numeric(result["spread_line"], errors="raise")
    result["home_side_hinge_7"] = (line.abs() - HINGE_POINTS).clip(lower=0.0).where(line.notna())
    return result


@dataclass(frozen=True)
class HomeSideOffsets:
    """Per-bucket shrunken home-side offsets fitted on prior games."""

    offsets: dict[str, float]
    prior_games: dict[str, int]

    def offset_for(self, spread: pd.Series) -> pd.Series:
        buckets = spread_bucket(spread)
        return buckets.map(self.offsets).astype(float).where(buckets.notna())


def fit_home_side_offsets(prior: pd.DataFrame, *, all_buckets: bool = False) -> HomeSideOffsets:
    """Shrunken mean of ``result - point_incumbent`` per spread bucket.

    ``prior`` carries ``spread_line`` (the line each row was scored at),
    ``point_incumbent`` and ``result`` (actual home margin); rows without a
    result are ignored. The prior weight is a fixed 100 games toward zero.

    The SERVED policy (S3) zeroes the offset outside ``HOME_SIDE_OFFSET_BUCKETS``;
    ``all_buckets=True`` is the research replay of lane S's S2 (every bucket
    served), kept so the frozen experiment still reproduces its definition.
    """

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
    """Rows usable to fit the offsets for one target week.

    Excludes the whole target week, every row whose gameday plus the one-day
    completion allowance is not strictly before the target week's earliest
    gameday, and seasons earlier than the target season minus five.
    """

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
    """Per-row offset for every (season, week) in ``frame`` from prior rows only.

    Returns a frame indexed like ``frame`` with ``bucket``, ``home_side_offset``,
    ``prior_games_in_bucket`` and ``point_corrected``.
    """

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
    """The unchanged gaussian_median read at a (possibly corrected) point.

    Mirrors ``ResidualSmoother.survival`` for ``gaussian_median``: the
    probability that ``point + residual`` exceeds ``line`` where the residual
    is Normal with the fitted week's median and sample standard deviation.
    """

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
    """Offsets fitted for one target week from an archived out-of-time stream."""

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
    """The opener-evaluation archive as the ``fit_home_side_offsets`` frame.

    ``point_incumbent`` is the archived Tuesday opener plus the model's
    out-of-time residual at that line, the same archive points lanes L and S
    scored, so the offset corrects the point the model actually produced
    before each game, never a refit.
    """

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
    """Archive rows strictly before (``season``, ``week``), five trailing seasons.

    A superset of the exclusions ``prior_games_for_week`` applies when a
    gameday is available: the whole target week and every later week are out,
    seasons earlier than ``season - 5`` are out, and only rows with a result
    count. Rows of the target season and an EARLIER week enter only once their
    result is recorded, which is the completion allowance expressed without
    a gameday column.
    """

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
    """Fit the served offsets for one week from archived out-of-time points.

    History precedence, all read-only: (a) the newest opener evaluation
    matched to the active model; (b) otherwise the newest opener evaluation of
    any model id, with the mismatch recorded as a warning (the Tuesday lock
    activates a new model id BEFORE its own evaluation runs, so this is the
    normal lock-day path); (c) otherwise zero offsets with a warning. The
    forecast is never blocked by this layer.
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
    """The sidecar a served forecast wrote, or ``None`` for pre-promotion cards."""

    path = forecast_dir / HOME_SIDE_OFFSET_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def served_center_offsets(forecast_dir: Path | None) -> dict[str, float] | None:
    """game_id -> served point offset from a forecast's sidecar; ``None`` when
    the card was produced without the promotion (no sidecar, or not served).

    Every module that REFITS the active recipe to reproduce or extend the
    served card (spread explorer, mapping-incumbent recorders, late-week
    refresh) applies this so its refit centre matches the point the card was
    actually built on. Missing games get 0.0 by ``center_offset_for_frame``.
    """

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
    """Row-aligned point shifts for ``frame`` (by ``game_id``), or ``None``."""

    if center_offsets is None:
        return None
    ids = frame["game_id"].astype(str)
    return np.asarray([float(center_offsets.get(game_id, 0.0)) for game_id in ids], dtype=float)


def center_offsets_from_metadata(
    metadata: Mapping[str, object], predictions: pd.DataFrame
) -> dict[str, float] | None:
    """Per-game served offsets rebuilt from a forecast's ``metadata.json``.

    ``margin-predict`` records the fitted per-bucket offsets under
    ``home_side_offset``; each game's offset is its spread bucket's value, so
    any module that has the card and its metadata (but not the artifact
    directory) can reproduce the served centre without file access.
    ``None`` for a card produced without the promotion.
    """

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
