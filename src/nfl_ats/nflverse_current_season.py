"""Reach the current season's nflverse releases before nflreadpy admits it exists.

**The defect this fixes (measured 2026-09-08).** ``nflreadpy`` guards every
seasonal loader with its own ``get_current_season()``, which rolls over on
"the Thursday following Labor Day". Labor Day 2026 was Monday 2026-09-07, so
that function returns **2025** until Thursday 2026-09-10 -- and
``load_injuries(seasons=[2026])`` raises
``ValueError: Season must be between 2009 and 2025``.

2026 Week 1 opens on **Wednesday 2026-09-09** (NE at SEA, 8:20 PM ET). The
league's first Week 1 injury report therefore lands, and the pool's first
pick comes due, one day BEFORE the library will serve the season's data.
Measured the same day, the data was already published and reachable:
``injuries_2026.parquet`` returned HTTP 200 with **11 rows** -- three New
England and seven Seattle players carrying practice designations for exactly
that opener.

This is the same failure shape as the missing Wednesday capture window fixed
2026-09-07: a rule that assumes the season starts on a Thursday, applied to a
season that starts on a Wednesday.

**Why the caller's fallback was not enough.** ``build_week_lineups.py``
already caught the ``ValueError`` and carried on, so nothing crashed -- but it
reported the cause as "nflverse has not published season 2026 injuries yet",
and the card told readers no injury reports existed. Both statements were
false; the publisher had the rows and the client refused to ask for them. A
graceful degradation that states the wrong reason is worse than a crash,
because nobody goes looking for it.

**How the bypass works.** The season guard lives in the ``load_*`` wrappers,
not in the transport: ``nflreadpy.load_injuries`` validates the season and
then calls ``get_downloader().download("nflverse-data", f"injuries/injuries_
{season}")``. Calling that downloader directly reuses nflreadpy's own URL
resolution, HTTP session, caching and parquet parsing -- everything except the
date rule -- so this module hand-rolls no networking of its own.

**This does not paper over missing data.** A season nflverse genuinely has not
published still fails: the downloader raises ``ConnectionError`` on the 404,
and :func:`load_season_frame` turns that into
:class:`SeasonReleaseNotPublished` so callers can say the honest thing.
Measured 2026-09-08, that is exactly the state of ``snap_counts_2026`` (no
games have been played), while ``injuries_2026`` and ``roster_weekly_2026``
are both live.
"""

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
    """nflverse has no release file for this dataset and season yet.

    Distinct from the client-side season guard on purpose: this one means the
    data really does not exist, and a caller may honestly tell a reader so.
    """


@dataclass(frozen=True)
class SeasonDataset:
    """One nflverse release, named the way its own download path is built.

    ``release`` is the release tag and ``stem`` the file stem, so the asset is
    ``{release}/{stem}_{season}.parquet`` -- the exact shape ``nflreadpy``'s
    loaders assemble.
    """

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
    """The season ``nflreadpy`` currently admits to, by its own rule."""

    from nflreadpy.utils_date import get_current_season

    return int(get_current_season())


def guard_blocks_season(season: int) -> bool:
    """Would ``nflreadpy``'s season guard refuse ``season`` right now?

    True only while the calendar sits between a season's real first kickoff
    and nflreadpy's Thursday-after-Labor-Day rollover.
    """

    return int(season) > nflreadpy_current_season()


def _to_pandas(frame: Any) -> pd.DataFrame:
    """Accept either a polars frame (what the downloader returns) or pandas."""

    to_pandas = getattr(frame, "to_pandas", None)
    converted = frame if to_pandas is None else to_pandas()
    if not isinstance(converted, pd.DataFrame):
        raise TypeError(f"expected a DataFrame from nflverse, got {type(converted).__name__}")
    return converted


def load_season_frame(dataset: str, season: int) -> pd.DataFrame:
    """One season of an nflverse release, guard or no guard.

    Defers to ``nflreadpy``'s public loader whenever it will answer, so normal
    seasons take the ordinary, cached, fully-supported path and this module
    changes nothing about them. Only when the client-side guard is the sole
    obstacle does it drop to the same downloader that loader would have used.

    Raises :class:`SeasonReleaseNotPublished` when nflverse has no file for
    this season, and :class:`KeyError` for a dataset this repo does not pull.
    """

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
    """Several seasons at once, guarded ones included.

    Deliberately splits the request: every season ``nflreadpy`` will serve
    goes through ONE ordinary bulk call, exactly as before, and only the
    guarded tail drops to the bypass. When nothing is guarded -- which is
    every season the archive is built from -- this returns precisely what
    ``nflreadpy.load_<dataset>(seasons=seasons)`` returned before, with no
    concatenation in the path. That matters more here than anywhere else:
    this feeds the model's feature table, and the project's methodology
    rests on those artifacts reproducing exactly.
    """

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
