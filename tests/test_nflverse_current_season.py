"""The current season must be reachable before nflreadpy admits it exists.

2026 Week 1 opened Wednesday 2026-09-09; nflreadpy's season guard rolled over
Thursday 2026-09-10. Everything here pins the one-day gap shut.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats import nflverse_current_season as ncs


class _FakeDownloader:
    def __init__(self, table: dict[str, pd.DataFrame]) -> None:
        self.table = table
        self.calls: list[str] = []

    def download(self, repo: str, path: str, season: int | None = None) -> pd.DataFrame:
        self.calls.append(path)
        if path not in self.table:
            raise ConnectionError(
                f"Failed to download https://example.invalid/{path}.parquet: "
                "404 Client Error: Not Found for url: https://example.invalid"
            )
        return self.table[path]


@pytest.fixture
def guard_at_2025(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze nflreadpy's rollover where it sat on 2026-09-08."""

    monkeypatch.setattr(ncs, "nflreadpy_current_season", lambda: 2025)


def test_release_paths_match_the_loaders_they_stand_in_for() -> None:
    """The bypass must build the SAME asset path nflreadpy would have."""

    assert ncs.SEASON_DATASETS["injuries"].path(2026) == "injuries/injuries_2026"
    assert ncs.SEASON_DATASETS["rosters_weekly"].path(2026) == "weekly_rosters/roster_weekly_2026"
    assert ncs.SEASON_DATASETS["snap_counts"].path(2026) == "snap_counts/snap_counts_2026"


def test_guard_blocks_only_seasons_past_the_rollover(guard_at_2025: None) -> None:
    assert ncs.guard_blocks_season(2026) is True
    assert ncs.guard_blocks_season(2025) is False
    assert ncs.guard_blocks_season(2019) is False


def test_a_guarded_season_is_served_from_the_release(
    guard_at_2025: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact failure of 2026-09-08: the guard says no, the data exists."""

    rows = pd.DataFrame({"season": [2026] * 11, "week": [1] * 11, "team": ["NE"] * 3 + ["SEA"] * 8})
    fake = _FakeDownloader({"injuries/injuries_2026": rows})
    monkeypatch.setattr("nflreadpy.downloader.get_downloader", lambda: fake)

    frame = ncs.load_season_frame("injuries", 2026)

    assert len(frame) == 11
    assert fake.calls == ["injuries/injuries_2026"]


def test_an_unguarded_season_still_goes_through_nflreadpy(
    guard_at_2025: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal seasons must not change path: same loader, same caching."""

    import nflreadpy as nfl

    seen: list[list[int]] = []

    def _loader(seasons: list[int]) -> pd.DataFrame:
        seen.append(seasons)
        return pd.DataFrame({"season": seasons})

    monkeypatch.setattr(nfl, "load_injuries", _loader, raising=False)
    fake = _FakeDownloader({})
    monkeypatch.setattr("nflreadpy.downloader.get_downloader", lambda: fake)

    frame = ncs.load_season_frame("injuries", 2024)

    assert seen == [[2024]]
    assert list(frame["season"]) == [2024]
    assert fake.calls == [], "an unguarded season must never reach the bypass"


def test_a_genuinely_missing_release_is_reported_as_missing(
    guard_at_2025: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured 2026-09-08: snap counts for 2026 really did 404 (no games yet).

    That is the one case where "nflverse has not published this yet" is the
    true sentence, and callers must still be able to say it.
    """

    fake = _FakeDownloader({})
    monkeypatch.setattr("nflreadpy.downloader.get_downloader", lambda: fake)

    with pytest.raises(ncs.SeasonReleaseNotPublished) as caught:
        ncs.load_season_frame("snap_counts", 2026)
    assert "snap_counts" in str(caught.value)
    assert "2026" in str(caught.value)


def test_a_real_network_fault_is_never_reported_as_missing_data(
    guard_at_2025: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "The site is down" must not become "the season does not exist"."""

    class _Broken:
        def download(self, repo: str, path: str, season: int | None = None) -> pd.DataFrame:
            raise ConnectionError("Failed to download: connection reset by peer")

    monkeypatch.setattr("nflreadpy.downloader.get_downloader", _Broken)

    with pytest.raises(ConnectionError):
        ncs.load_season_frame("injuries", 2026)


def test_polars_frames_are_converted(guard_at_2025: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The downloader hands back polars; every caller here expects pandas."""

    polars = pytest.importorskip("polars")
    fake = _FakeDownloader({"injuries/injuries_2026": polars.DataFrame({"season": [2026]})})
    monkeypatch.setattr("nflreadpy.downloader.get_downloader", lambda: fake)

    frame = ncs.load_season_frame("injuries", 2026)

    assert isinstance(frame, pd.DataFrame)
    assert list(frame["season"]) == [2026]
