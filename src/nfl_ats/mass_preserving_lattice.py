from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize

from nfl_ats.conditional_margin import BANDWIDTH
from nfl_ats.home_side_location import COMPLETION_ALLOWANCE_DAYS, TRAILING_SEASONS
from nfl_ats.modeling import regular_season_rows

FloatArray = npt.NDArray[np.float64]

DISCRETE_PUSH_READ_SERVED = True
DISCRETE_PUSH_READ_POLICY = "mass_preserving_lattice_mp1_v1"
DISCRETE_PUSH_READ_FILENAME = "discrete_push_read.json"

BAND_HALF_WIDTH = float(BANDWIDTH)
MIN_BAND_GAMES = 200
MAX_BAND = 20.0
BAND_STEP = 0.5
THETA_BRACKET = 1.0
KEY_NUMBERS: tuple[int, ...] = (3, 7, 10, 14)
_ATOM_TOLERANCE = 1e-9


def tilted_atoms(
    margins: FloatArray, counts: FloatArray, line: float, target: float
) -> tuple[FloatArray, float]:

    base = counts / counts.sum()
    if margins.size == 1:
        return base, 0.0
    offsets = margins - line

    def weights(theta: float) -> FloatArray:
        logits = theta * offsets
        mass = base * np.exp(logits - logits.max())
        return np.asarray(mass / mass.sum(), dtype=np.float64)

    def mean_at(theta: float) -> float:
        return float((margins * weights(theta)).sum())

    low, high = -THETA_BRACKET, THETA_BRACKET
    if target <= mean_at(low):
        theta = low
    elif target >= mean_at(high):
        theta = high
    else:
        theta = float(optimize.brentq(lambda t: mean_at(t) - target, low, high, xtol=1e-13))
    return weights(theta), theta


@dataclass(frozen=True)
class MassPreservingRead:
    cover: float
    push: float
    loss: float
    theta: float
    band: float
    band_games: int
    atoms: int
    key_mass_3: float

    @property
    def home_cover_probability(self) -> float:

        return self.cover + 0.5 * self.push

    @property
    def conditional_cover_probability(self) -> float:

        total = self.cover + self.loss
        return self.cover / total if total > 0 else 0.5

    def three_way(self) -> tuple[float, float, float]:
        return self.cover, self.push, self.loss

    def as_dict(self) -> dict[str, float | int]:

        return {
            "cover": self.cover,
            "push": self.push,
            "loss": self.loss,
            "home_cover_probability": self.home_cover_probability,
            "conditional_cover_probability": self.conditional_cover_probability,
            "theta": self.theta,
            "band": self.band,
            "band_games": self.band_games,
            "atoms": self.atoms,
            "key_mass_3": self.key_mass_3,
        }


def key_number_mass(
    margins: FloatArray, mass: FloatArray, keys: Iterable[int] = KEY_NUMBERS
) -> dict[int, float]:

    return {
        int(key): float(mass[np.abs(np.abs(margins) - float(key)) < _ATOM_TOLERANCE].sum())
        for key in keys
    }


def band_read(
    pool_line: FloatArray,
    pool_margin: FloatArray,
    line: float,
    point: float,
    half_width: float = BAND_HALF_WIDTH,
    min_band_games: int = MIN_BAND_GAMES,
    *,
    conditioning_line: float | None = None,
) -> MassPreservingRead:

    if not np.isfinite([line, point]).all():
        raise ValueError("The discrete read needs a finite line and point")
    anchor = line if conditioning_line is None else float(conditioning_line)
    if not np.isfinite(anchor):
        raise ValueError("The discrete read needs a finite conditioning line")
    band = half_width
    while True:
        selected = np.abs(pool_line - anchor) <= band
        if int(selected.sum()) >= min_band_games or band >= MAX_BAND:
            break
        band = min(band + BAND_STEP, MAX_BAND)
    margins = pool_margin[selected]
    if margins.size == 0:
        raise ValueError(f"No prior games within {band} points of line {anchor}")
    values, counts = np.unique(margins, return_counts=True)
    mass, theta = tilted_atoms(values, counts.astype(float), anchor, point)
    is_push = np.abs(values - line) < _ATOM_TOLERANCE
    return MassPreservingRead(
        cover=float(mass[values > line + _ATOM_TOLERANCE].sum()),
        push=float(mass[is_push].sum()),
        loss=float(mass[values < line - _ATOM_TOLERANCE].sum()),
        theta=theta,
        band=band,
        band_games=int(selected.sum()),
        atoms=int(values.size),
        key_mass_3=key_number_mass(values, mass, (3,))[3],
    )


def residual_location(residuals: FloatArray, probability_method: str) -> float:

    values = np.asarray(residuals, dtype=float)
    if values.size == 0:
        raise ValueError("A fitted residual sample is required for the served point")
    return float(np.mean(values)) if probability_method == "gaussian" else float(np.median(values))


_POOL_COLUMNS = ("game_id", "season", "week", "gameday", "spread_line", "result")


def prior_pool(
    features: pd.DataFrame, opener_lines: Mapping[str, float] | pd.Series | None = None
) -> pd.DataFrame:

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    pool = frame.loc[frame["result"].notna(), list(_POOL_COLUMNS)].copy()
    pool["game_id"] = pool["game_id"].astype(str)
    opener = (
        pd.Series(dtype=float)
        if opener_lines is None
        else pd.Series(
            dict(opener_lines) if not isinstance(opener_lines, pd.Series) else opener_lines
        )
    )
    opener.index = opener.index.astype(str)
    pool["line"] = pool["game_id"].map(opener).fillna(pool["spread_line"])
    pool = pool.loc[pool["line"].notna()].reset_index(drop=True)
    pool["line"] = pool["line"].astype(float)
    pool["result"] = np.rint(pool["result"].to_numpy(dtype=float))
    return pool


def prior_pool_for_week(
    pool: pd.DataFrame,
    *,
    season: int,
    week: int,
    cutoff: pd.Timestamp,
    exclude_game_ids: Iterable[str] = (),
) -> pd.DataFrame:

    gameday = pd.to_datetime(pool["gameday"])
    eligible = (
        (gameday + pd.Timedelta(days=COMPLETION_ALLOWANCE_DAYS)).lt(pd.Timestamp(cutoff))
        & pool["season"].ge(int(season) - TRAILING_SEASONS)
        & ~(pool["season"].eq(int(season)) & pool["week"].eq(int(week)))
        & ~pool["game_id"].astype(str).isin({str(g) for g in exclude_game_ids})
    )
    return pool.loc[eligible]


@dataclass(frozen=True)
class DiscretePushReader:
    lines: FloatArray
    margins: FloatArray
    half_width: float = BAND_HALF_WIDTH
    min_band_games: int = MIN_BAND_GAMES
    prior_rows: int = 0
    max_gameday: str | None = None
    policy: str = DISCRETE_PUSH_READ_POLICY

    @classmethod
    def for_week(
        cls,
        pool: pd.DataFrame,
        *,
        season: int,
        week: int,
        cutoff: pd.Timestamp,
        exclude_game_ids: Iterable[str] = (),
        half_width: float = BAND_HALF_WIDTH,
        min_band_games: int = MIN_BAND_GAMES,
    ) -> DiscretePushReader:
        eligible = prior_pool_for_week(
            pool, season=season, week=week, cutoff=cutoff, exclude_game_ids=exclude_game_ids
        )
        if eligible.empty:
            raise ValueError(f"No prior games precede {season} week {week} for the discrete read")
        newest = pd.to_datetime(eligible["gameday"]).max()
        return cls(
            lines=eligible["line"].to_numpy(dtype=float),
            margins=eligible["result"].to_numpy(dtype=float),
            half_width=float(half_width),
            min_band_games=int(min_band_games),
            prior_rows=len(eligible),
            max_gameday=None if pd.isna(newest) else str(pd.Timestamp(newest).date()),
        )

    def read(
        self, line: float, point: float, *, conditioning_line: float | None = None
    ) -> MassPreservingRead:
        return band_read(
            self.lines,
            self.margins,
            float(line),
            float(point),
            self.half_width,
            self.min_band_games,
            conditioning_line=conditioning_line,
        )

    def three_way(
        self, line: float, point: float, *, conditioning_line: float | None = None
    ) -> tuple[float, float, float]:

        return self.read(line, point, conditioning_line=conditioning_line).three_way()


DiscreteRead = MassPreservingRead


def discrete_read(
    prior_stream: pd.DataFrame,
    line: float,
    served_point: float,
    *,
    season: int,
    week: int,
    cutoff: pd.Timestamp,
    game_id: str | None = None,
    band: float = BAND_HALF_WIDTH,
    prior: int = MIN_BAND_GAMES,
) -> MassPreservingRead:

    eligible = prior_pool_for_week(
        prior_stream,
        season=season,
        week=week,
        cutoff=cutoff,
        exclude_game_ids=() if game_id is None else (str(game_id),),
    )
    if eligible.empty:
        raise ValueError(f"No prior games precede {season} week {week} for the discrete read")
    return band_read(
        eligible["line"].to_numpy(dtype=float),
        eligible["result"].to_numpy(dtype=float),
        float(line),
        float(served_point),
        band,
        prior,
    )


def discrete_reads_for_frame(
    prior_stream: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    line_column: str = "spread_line",
    point_column: str = "point",
    band: float = BAND_HALF_WIDTH,
    prior: int = MIN_BAND_GAMES,
) -> pd.DataFrame:

    out = pd.DataFrame(index=frame.index)
    columns = ("cover", "push", "loss", "home_cover_probability", "theta", "band")
    for column in (*columns, "band_games", "atoms", "key_mass_3", "prior_rows"):
        out[column] = np.nan
    for _, group in frame.groupby(["season", "week"], sort=True):
        reader = DiscretePushReader.for_week(
            prior_stream,
            season=int(group["season"].iloc[0]),
            week=int(group["week"].iloc[0]),
            cutoff=pd.Timestamp(pd.to_datetime(group["gameday"]).min()),
            exclude_game_ids=set(group["game_id"].astype(str)),
            half_width=band,
            min_band_games=prior,
        )
        for index, row in group.iterrows():
            read = reader.read(float(row[line_column]), float(row[point_column]))
            values = read.as_dict()
            for column in (*columns, "band_games", "atoms", "key_mass_3"):
                out.loc[index, column] = values[column]
            out.loc[index, "prior_rows"] = reader.prior_rows
    for column in ("band_games", "atoms", "prior_rows"):
        out[column] = out[column].astype(int)
    return out


def walk_forward_reads(
    pool: pd.DataFrame, targets: pd.DataFrame, half_width: float
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []
    for _, batch in targets.groupby(["season", "week"], sort=True):
        reader = DiscretePushReader.for_week(
            pool,
            season=int(batch["season"].iloc[0]),
            week=int(batch["week"].iloc[0]),
            cutoff=pd.Timestamp(batch["gameday"].min()),
            exclude_game_ids=set(batch["game_id"].astype(str)),
            half_width=half_width,
        )
        for _, row in batch.iterrows():
            read = reader.read(float(row["line"]), float(row["point"]))
            rows.append(
                {"game_id": row["game_id"], "prior_rows": reader.prior_rows, **read.as_dict()}
            )
    return pd.DataFrame(rows)


THREE_WAY_COLUMNS = (
    "home_cover_probability_excluding_push",
    "push_probability",
    "home_loss_probability",
)


@dataclass(frozen=True)
class ServedPushRead:
    game_id: str
    line: float
    point: float
    discrete: MassPreservingRead
    smooth: tuple[float, float, float]
    home_cover_probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "spread_line": self.line,
            "point": self.point,
            "home_cover_probability": self.home_cover_probability,
            "served": {
                "cover": self.discrete.cover,
                "push": self.discrete.push,
                "loss": self.discrete.loss,
            },
            "smooth": {
                "cover": self.smooth[0],
                "push": self.smooth[1],
                "loss": self.smooth[2],
            },
            "theta": self.discrete.theta,
            "band": self.discrete.band,
            "band_games": self.discrete.band_games,
            "atoms": self.discrete.atoms,
            "key_mass_3": self.discrete.key_mass_3,
        }


def serve_discrete_three_way(
    forecasts: pd.DataFrame,
    games: pd.DataFrame,
    reader: DiscretePushReader,
    *,
    residuals: FloatArray,
    probability_method: str,
    log: MutableMapping[str, ServedPushRead] | None = None,
) -> pd.DataFrame:

    if len(forecasts) != len(games):
        raise ValueError("forecasts and games must be row-aligned")
    result = forecasts.copy()
    location = residual_location(residuals, probability_method)
    lines = pd.to_numeric(games["spread_line"], errors="raise").to_numpy(dtype=float)
    points = result["predicted_margin"].to_numpy(dtype=float) + location
    ids = games["game_id"].astype(str).to_numpy()
    cover = np.empty(len(result), dtype=float)
    push = np.empty(len(result), dtype=float)
    loss = np.empty(len(result), dtype=float)
    for index, (game_id, line, point) in enumerate(zip(ids, lines, points, strict=True)):
        read = reader.read(line, point)
        cover[index], push[index], loss[index] = read.three_way()
        if log is not None:
            log[str(game_id)] = ServedPushRead(
                game_id=str(game_id),
                line=float(line),
                point=float(point),
                discrete=read,
                smooth=(
                    float(forecasts["home_cover_probability_excluding_push"].iloc[index]),
                    float(forecasts["push_probability"].iloc[index]),
                    float(forecasts["home_loss_probability"].iloc[index]),
                ),
                home_cover_probability=float(forecasts["home_cover_probability"].iloc[index]),
            )
    result["home_cover_probability_excluding_push"] = cover
    result["push_probability"] = push
    result["home_loss_probability"] = loss
    return result


def serve_discrete_sweep(
    sweep: pd.DataFrame,
    reader: DiscretePushReader,
    *,
    points_by_game: Mapping[str, float],
    quoted_lines_by_game: Mapping[str, float] | None = None,
) -> pd.DataFrame:

    result = sweep.copy()
    cover = np.empty(len(result), dtype=float)
    push = np.empty(len(result), dtype=float)
    loss = np.empty(len(result), dtype=float)
    ids = result["game_id"].astype(str).to_numpy()
    lines = result["alternative_line"].to_numpy(dtype=float)
    for index, (game_id, line) in enumerate(zip(ids, lines, strict=True)):
        if game_id not in points_by_game:
            raise ValueError(f"No served point for game {game_id!r} in the line sweep")
        quoted = None if quoted_lines_by_game is None else quoted_lines_by_game.get(game_id)
        cover[index], push[index], loss[index] = reader.three_way(
            line, points_by_game[game_id], conditioning_line=quoted
        )
    result["home_cover_probability_excluding_push"] = cover
    result["push_probability"] = push
    result["home_loss_probability"] = loss
    return result


@dataclass(frozen=True)
class ProductionDiscretePushRead:
    policy: str
    reader: DiscretePushReader | None
    source_path: str | None
    source_model_id: str | None
    active_model_id: str | None
    opener_lines_matched: int
    warnings: tuple[str, ...]
    error: str | None = None

    @property
    def served(self) -> bool:
        return self.reader is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "served": self.served,
            "error": self.error,
            "band_half_width": BAND_HALF_WIDTH,
            "min_band_games": MIN_BAND_GAMES,
            "max_band": MAX_BAND,
            "trailing_seasons": TRAILING_SEASONS,
            "prior_rows": self.reader.prior_rows if self.reader is not None else 0,
            "prior_max_gameday": self.reader.max_gameday if self.reader is not None else None,
            "source_path": self.source_path,
            "source_model_id": self.source_model_id,
            "active_model_id": self.active_model_id,
            "opener_lines_matched": self.opener_lines_matched,
            "warnings": list(self.warnings),
        }


def _opener_lines_from_evaluation(evaluation: Path) -> pd.Series:
    per_game = pd.read_parquet(evaluation / "per_game.parquet")
    if "tue_open_home_spread" not in per_game.columns or "game_id" not in per_game.columns:
        return pd.Series(dtype=float)
    lines = pd.to_numeric(per_game["tue_open_home_spread"], errors="coerce")
    series = pd.Series(lines.to_numpy(dtype=float), index=per_game["game_id"].astype(str))
    return series.loc[series.notna()]


def fit_production_discrete_push_reader(
    features: pd.DataFrame,
    artifacts_root: Path,
    active: Mapping[str, object] | None,
    *,
    season: int,
    week: int,
) -> ProductionDiscretePushRead:

    from nfl_ats.home_side_location import _evaluation_model_id, _newest_opener_evaluation

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
    if evaluation is None:
        evaluation = _newest_opener_evaluation(artifacts_root)
        if evaluation is not None:
            warnings.append(
                "no opener evaluation matches the active model; opener lines read from the "
                f"newest evaluation {evaluation.name} instead"
            )
    opener_lines = pd.Series(dtype=float)
    source_path: str | None = None
    source_model_id: str | None = None
    if evaluation is not None and (evaluation / "per_game.parquet").is_file():
        try:
            opener_lines = _opener_lines_from_evaluation(evaluation)
            source_path = str(evaluation.relative_to(artifacts_root)).replace("\\", "/")
            source_model_id = _evaluation_model_id(evaluation)
        except Exception as error:
            warnings.append(f"opener archive unreadable, feature-table lines used: {error}")
    else:
        warnings.append("no opener evaluation archive found; feature-table lines used")

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    target = frame.loc[frame["season"].eq(int(season)) & frame["week"].eq(int(week))]
    if target.empty:
        raise ValueError(f"No games found for {season} week {week}")
    pool = prior_pool(frame, opener_lines)
    reader = DiscretePushReader.for_week(
        pool,
        season=int(season),
        week=int(week),
        cutoff=pd.Timestamp(target["gameday"].min()),
        exclude_game_ids=set(target["game_id"].astype(str)),
    )
    matched_lines = (
        int(pool["game_id"].isin(set(opener_lines.index)).sum()) if len(opener_lines) else 0
    )
    return ProductionDiscretePushRead(
        policy=DISCRETE_PUSH_READ_POLICY,
        reader=reader,
        source_path=source_path,
        source_model_id=source_model_id,
        active_model_id=active_model_id,
        opener_lines_matched=matched_lines,
        warnings=tuple(warnings),
    )


def load_forecast_discrete_push_read(forecast_dir: Path) -> dict[str, Any] | None:

    path = forecast_dir / DISCRETE_PUSH_READ_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


__all__ = [
    "BAND_HALF_WIDTH",
    "BAND_STEP",
    "DISCRETE_PUSH_READ_FILENAME",
    "DISCRETE_PUSH_READ_POLICY",
    "DISCRETE_PUSH_READ_SERVED",
    "KEY_NUMBERS",
    "MAX_BAND",
    "MIN_BAND_GAMES",
    "THETA_BRACKET",
    "THREE_WAY_COLUMNS",
    "DiscretePushReader",
    "DiscreteRead",
    "MassPreservingRead",
    "ProductionDiscretePushRead",
    "ServedPushRead",
    "band_read",
    "discrete_read",
    "discrete_reads_for_frame",
    "fit_production_discrete_push_reader",
    "key_number_mass",
    "load_forecast_discrete_push_read",
    "prior_pool",
    "prior_pool_for_week",
    "residual_location",
    "serve_discrete_sweep",
    "serve_discrete_three_way",
    "tilted_atoms",
    "walk_forward_reads",
]
