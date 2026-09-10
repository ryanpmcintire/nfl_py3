from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from nfl_ats.data import require_columns
from nfl_ats.margin import MarginModel

TEAM_STATE_SPAN = 8

OPPONENT_ADJUSTMENT_HALF_LIFE_WEEKS = 16.0
GRAPH_RATING_HALF_LIFE_WEEKS = 8.0


def half_life_contamination_weeks(half_life_weeks: float, weight_threshold: float) -> float:

    if half_life_weeks <= 0.0:
        raise ValueError("half_life_weeks must be positive")
    if not 0.0 < weight_threshold < 1.0:
        raise ValueError("weight_threshold must be strictly between 0 and 1")
    return half_life_weeks * math.log2(1.0 / weight_threshold)


PURGE_WEIGHT_THRESHOLD = 0.05
EMBARGO_WEIGHT_THRESHOLD = 0.01


def ewma_retained_weight(games_elapsed: int, span: int) -> float:

    if games_elapsed < 0:
        raise ValueError("games_elapsed cannot be negative")
    if span < 2:
        raise ValueError("span must be at least 2")
    alpha = 2.0 / (span + 1.0)
    return float((1.0 - alpha) ** games_elapsed)


def ewma_contamination_games(span: int, weight_threshold: float) -> int:

    if not 0.0 < weight_threshold < 1.0:
        raise ValueError("weight_threshold must be strictly between 0 and 1")
    alpha = 2.0 / (span + 1.0)
    games = math.log(weight_threshold) / math.log(1.0 - alpha)
    return math.ceil(games)


DEFAULT_PURGE_WEEKS = ewma_contamination_games(TEAM_STATE_SPAN, PURGE_WEIGHT_THRESHOLD)

DEFAULT_EMBARGO_WEEKS = (
    ewma_contamination_games(TEAM_STATE_SPAN, EMBARGO_WEIGHT_THRESHOLD) - DEFAULT_PURGE_WEEKS
)


def assign_week_order(frame: pd.DataFrame) -> pd.DataFrame:

    require_columns(frame, ("season", "week", "gameday"), "purged CV frame")
    working = frame.drop(columns=["week_order"], errors="ignore").copy()
    working["gameday"] = pd.to_datetime(working["gameday"], errors="raise")
    week_starts = (
        working.groupby(["season", "week"], sort=False)["gameday"]
        .min()
        .reset_index()
        .sort_values("gameday")
        .reset_index(drop=True)
    )
    week_starts["week_order"] = np.arange(len(week_starts), dtype="int64")
    working = working.merge(
        week_starts[["season", "week", "week_order"]],
        on=["season", "week"],
        how="left",
        validate="many_to_one",
    )
    return working


def partition_week_blocks(n_weeks: int, n_blocks: int) -> npt.NDArray[np.int64]:

    if n_weeks < 1:
        raise ValueError("n_weeks must be positive")
    if not 1 <= n_blocks <= n_weeks:
        raise ValueError("n_blocks must be between 1 and n_weeks")
    chunks = np.array_split(np.arange(n_weeks, dtype="int64"), n_blocks)
    block_of_week = np.empty(n_weeks, dtype="int64")
    for block_id, chunk in enumerate(chunks):
        block_of_week[chunk] = block_id
    return block_of_week


@dataclass(frozen=True)
class PurgedFold:
    path_id: int
    test_blocks: tuple[int, ...]
    test_week_range: tuple[int, int]
    train_index: npt.NDArray[np.int64]
    test_index: npt.NDArray[np.int64]


def purged_embargoed_folds(
    frame: pd.DataFrame,
    *,
    n_blocks: int,
    test_group_size: int = 1,
    purge_weeks: int = DEFAULT_PURGE_WEEKS,
    embargo_weeks: int = DEFAULT_EMBARGO_WEEKS,
    max_paths: int | None = None,
    path_seed: int = 20260818,
) -> list[PurgedFold]:

    if purge_weeks < 0:
        raise ValueError("purge_weeks cannot be negative")
    if embargo_weeks < 0:
        raise ValueError("embargo_weeks cannot be negative")
    if test_group_size < 1:
        raise ValueError("test_group_size must be positive")
    if test_group_size > n_blocks:
        raise ValueError("test_group_size cannot exceed n_blocks")

    working = assign_week_order(frame)
    n_weeks = int(working["week_order"].max()) + 1
    block_of_week = partition_week_blocks(n_weeks, n_blocks)
    row_week = working["week_order"].to_numpy(dtype="int64")
    row_block = block_of_week[row_week]

    combos = list(itertools.combinations(range(n_blocks), test_group_size))
    if max_paths is not None and len(combos) > max_paths:
        rng = np.random.default_rng(path_seed)
        keep = np.sort(rng.choice(len(combos), size=max_paths, replace=False))
        combos = [combos[i] for i in keep]

    folds: list[PurgedFold] = []
    for path_id, combo in enumerate(combos):
        test_mask = np.isin(row_block, combo)
        block_ranges = [
            (
                int(np.flatnonzero(block_of_week == block_id).min()),
                int(np.flatnonzero(block_of_week == block_id).max()),
            )
            for block_id in combo
        ]
        purge_mask = np.zeros(len(working), dtype=bool)
        embargo_mask = np.zeros(len(working), dtype=bool)
        for lo, hi in block_ranges:
            purge_mask |= (row_week >= lo - purge_weeks) & (row_week <= hi + purge_weeks)
            embargo_mask |= (row_week > hi + purge_weeks) & (
                row_week <= hi + purge_weeks + embargo_weeks
            )
        train_mask = (~test_mask) & (~purge_mask) & (~embargo_mask)
        folds.append(
            PurgedFold(
                path_id=path_id,
                test_blocks=combo,
                test_week_range=(
                    min(lo for lo, _ in block_ranges),
                    max(hi for _, hi in block_ranges),
                ),
                train_index=np.flatnonzero(train_mask),
                test_index=np.flatnonzero(test_mask),
            )
        )
    return folds


_PASSTHROUGH_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "spread_line",
    "home_spread_odds",
    "away_spread_odds",
    "result",
    "ats_margin",
    "home_cover",
)


@dataclass(frozen=True)
class PurgedCVResult:
    predictions: pd.DataFrame
    fold_summary: pd.DataFrame
    config: dict[str, Any] = field(default_factory=dict)


def _score_fold(
    test: pd.DataFrame,
    models: dict[str, MarginModel],
    fold: PurgedFold,
) -> list[pd.DataFrame]:
    batches: list[pd.DataFrame] = []
    for method, model in models.items():
        batch = test.loc[:, list(_PASSTHROUGH_COLUMNS)].copy()
        forecasts = model.predict(test)
        for column in forecasts:
            batch[column] = forecasts[column].to_numpy()
        batch["method"] = method
        batch["model_name"] = model.model_name
        batch["train_rows"] = model.training_rows
        batch["distribution_rows"] = model.distribution_rows
        batch["train_max_gameday"] = model.training_max_gameday
        batch["path_id"] = fold.path_id
        batch["test_blocks"] = str(fold.test_blocks)
        batch["bet_side"] = "PASS"
        batch["bet_odds"] = np.nan
        batches.append(batch)
    return batches


def permute_target(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:

    require_columns(frame, ("ats_margin", "spread_line"), "purged CV frame")
    working = frame.copy()
    rng = np.random.default_rng(seed)
    permuted_ats_margin = rng.permutation(
        pd.to_numeric(working["ats_margin"], errors="raise").to_numpy(dtype=float)
    )
    working["ats_margin"] = permuted_ats_margin
    spread = pd.to_numeric(working["spread_line"], errors="raise").to_numpy(dtype=float)
    working["result"] = spread + permuted_ats_margin
    working["home_cover"] = np.select(
        [permuted_ats_margin > 0, permuted_ats_margin < 0], [1.0, 0.0], default=np.nan
    )
    return working


def team_persistent_null(
    frame: pd.DataFrame,
    *,
    team_sigma: float,
    noise_sigma: float,
    seed: int,
) -> pd.DataFrame:

    require_columns(frame, ("home_id", "away_id", "season", "spread_line"), "purged CV frame")
    working = frame.copy()
    rng = np.random.default_rng(seed)
    team_seasons = pd.concat(
        [
            working[["home_id", "season"]].rename(columns={"home_id": "team_id"}),
            working[["away_id", "season"]].rename(columns={"away_id": "team_id"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    quality = pd.Series(
        rng.normal(0.0, team_sigma, size=len(team_seasons)),
        index=pd.MultiIndex.from_frame(team_seasons),
    )
    home_quality = quality.reindex(
        pd.MultiIndex.from_frame(
            working[["home_id", "season"]].rename(columns={"home_id": "team_id"})
        )
    ).to_numpy()
    away_quality = quality.reindex(
        pd.MultiIndex.from_frame(
            working[["away_id", "season"]].rename(columns={"away_id": "team_id"})
        )
    ).to_numpy()
    noise = rng.normal(0.0, noise_sigma, size=len(working))
    fake_ats_margin = home_quality - away_quality + noise

    spread = pd.to_numeric(working["spread_line"], errors="raise").to_numpy(dtype=float)
    working["ats_margin"] = fake_ats_margin
    working["result"] = spread + fake_ats_margin
    working["home_cover"] = np.select(
        [fake_ats_margin > 0, fake_ats_margin < 0], [1.0, 0.0], default=np.nan
    )
    return working


def synthetic_signal_accuracy(beta: float, noise_std: float) -> float:

    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive")
    return float(stats.norm.cdf(beta / noise_std))


def synthetic_signal_beta(target_accuracy: float, noise_std: float) -> float:

    if not 0.5 < target_accuracy < 1.0:
        raise ValueError("target_accuracy must be strictly between 0.5 and 1.0")
    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive")
    return float(noise_std * stats.norm.ppf(target_accuracy))


def inject_synthetic_signal(
    frame: pd.DataFrame,
    *,
    target_accuracy: float,
    noise_std: float | None = None,
    seed: int,
    column: str = "synthetic_signal",
) -> pd.DataFrame:

    require_columns(frame, ("spread_line", "ats_margin"), "purged CV frame")
    if noise_std is None:
        noise_std = float(pd.to_numeric(frame["ats_margin"], errors="coerce").std())
    beta = synthetic_signal_beta(target_accuracy, noise_std)
    rng = np.random.default_rng(seed)
    n = len(frame)
    signal = rng.choice(np.array([-1.0, 1.0]), size=n)
    noise = rng.normal(0.0, noise_std, size=n)
    synthetic_margin = beta * signal + noise

    working = frame.copy()
    working[column] = signal
    spread = pd.to_numeric(working["spread_line"], errors="raise").to_numpy(dtype=float)
    working["result"] = spread + synthetic_margin
    working["ats_margin"] = synthetic_margin
    working["home_cover"] = np.select(
        [synthetic_margin > 0, synthetic_margin < 0], [1.0, 0.0], default=np.nan
    )
    return working
