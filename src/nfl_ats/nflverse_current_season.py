from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

__all__ = [
    "SEASON_DATASETS",
    "SeasonDataset",
    "SeasonReleaseNotPublished",
    "guard_blocks_season",
    "load_season_frame",
    "load_seasons_frame",
    "nflreadpy_current_season",
]


class SeasonReleaseNotPublished(RuntimeError):
    pass


@dataclass(frozen=True)
class SeasonDataset:
    release: str
    stem: str

    def path(self, season: int) -> str:
        return f"{self.release}/{self.stem}_{season}"


SEASON_DATASETS: dict[str, SeasonDataset] = {
    "injuries": SeasonDataset("injuries", "injuries"),
    "rosters_weekly": SeasonDataset("weekly_rosters", "roster_weekly"),
    "snap_counts": SeasonDataset("snap_counts", "snap_counts"),
    "participation": SeasonDataset("pbp_participation", "pbp_participation"),
}


def nflreadpy_current_season() -> int:

    from nflreadpy.utils_date import get_current_season

    return int(get_current_season())


def guard_blocks_season(season: int) -> bool:

    return int(season) > nflreadpy_current_season()


def _to_pandas(frame: Any) -> pd.DataFrame:

    to_pandas = getattr(frame, "to_pandas", None)
    converted = frame if to_pandas is None else to_pandas()
    if not isinstance(converted, pd.DataFrame):
        raise TypeError(f"expected a DataFrame from nflverse, got {type(converted).__name__}")
    return converted


def load_season_frame(dataset: str, season: int) -> pd.DataFrame:

    entry = SEASON_DATASETS[dataset]
    if not guard_blocks_season(season):
        import nflreadpy as nfl

        loader = getattr(nfl, f"load_{dataset}")
        return _to_pandas(loader(seasons=[int(season)]))

    from nflreadpy.downloader import get_downloader

    try:
        downloaded = get_downloader().download(
            "nflverse-data", entry.path(int(season)), season=int(season)
        )
    except ConnectionError as exc:
        if "404" not in str(exc):
            raise
        raise SeasonReleaseNotPublished(
            f"nflverse has published no {dataset} release for season {season} yet"
        ) from exc
    return _to_pandas(downloaded)


def load_seasons_frame(dataset: str, seasons: list[int]) -> pd.DataFrame:

    guarded = [int(season) for season in seasons if guard_blocks_season(season)]
    ordinary = [int(season) for season in seasons if not guard_blocks_season(season)]
    frames: list[pd.DataFrame] = []
    if ordinary:
        import nflreadpy as nfl

        loader = getattr(nfl, f"load_{dataset}")
        frames.append(_to_pandas(loader(seasons=ordinary)))
    frames.extend(load_season_frame(dataset, season) for season in guarded)
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)
