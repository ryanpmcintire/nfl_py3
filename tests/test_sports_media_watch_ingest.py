from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest_sports_media_watch import (
    SportsMediaWatchIngestError,
    ingest,
    parse_archive_html,
    point_in_time_view,
)

ARCHIVE_HTML = b"""
<html><body>
<table id="smwtable5"><tbody>
<tr><td>Window</td><th>Featured Game</th><th>Net</th><th>Rtg.</th>
    <th>+/-</th><th>Vwrs.</th><th>+/-</th></tr>
<tr><td colspan="7"><span>WEEK ONE</span></td></tr>
<tr><td colspan="7"><span>Sunday, September 7</span></td></tr>
<tr><td>National</td><td>49ers/Cowboys (89%)</td><td>FOX</td>
    <td>15.7</td><td>-5%</td><td>28.00M</td><td>-2%</td></tr>
</tbody></table>
<span class="sectionhed">Week 2 games</span>
<img alt="week 2 nfl ratings" data-src="https://www.sportsmediawatch.com/wp-content/uploads/2022/09/week2nflratings.png">
<img alt="advertisement" data-src="https://ads.example/image.png">
</body></html>
"""

SIX_AND_EIGHT_COLUMN_HTML = b"""
<html><body>
<table id="smwtable5"><tbody>
<tr><th>Window/Game</th><th>Net</th><th>Rtg.</th><th>+/-</th><th>Vwrs.</th><th>+/-</th></tr>
<tr><th colspan="6">WEEK TWO</th></tr>
<tr><td>Cowboys/Eagles Doubleheader Gm. 2</td><td>FOX</td><td>15.5</td><td>+8%</td>
    <td>27.2M</td><td>+9%</td></tr>
</tbody></table>
<h2><span class="subhed">NFL Week Three Ratings</span></h2>
<table id="olytable"><tbody>
<tr><td>Window</td><td>Game</td><td>Net</td><td>Rtg.</td><td>+/-</td><td>Vwrs.</td><td>+/-</td><td>A18-49</td></tr>
<tr><td>SNF</td><td>PHI-NYG</td><td>NBC</td><td>12.0</td><td>+1%</td><td>20.0M</td><td>+2%</td><td>6.0</td></tr>
</tbody></table>
</body></html>
"""


def test_archive_parser_extracts_structured_rows_and_relevant_assets() -> None:
    rows, assets = parse_archive_html(
        ARCHIVE_HTML,
        season=2014,
        page_url="https://www.sportsmediawatch.com/example/",
        observed_at="2026-08-20T17:00:00+00:00",
    )
    assert len(rows) == 1
    assert rows.iloc[0].to_dict()["away_team"] == "SF"
    assert rows.iloc[0].to_dict()["home_team"] == "DAL"
    assert rows.iloc[0].to_dict()["viewers"] == 28_000_000
    assert rows.iloc[0].to_dict()["week"] == 1
    assert rows.iloc[0].to_dict()["source_published_at"] is None
    assert len(assets) == 1
    assert assets.iloc[0]["week"] == 2


def test_archive_parser_supports_six_and_eight_column_primary_schemas() -> None:
    rows, _assets = parse_archive_html(
        SIX_AND_EIGHT_COLUMN_HTML,
        season=2017,
        page_url="https://www.sportsmediawatch.com/example/",
        observed_at="2026-08-20T17:00:00+00:00",
    )
    assert rows[["week", "away_team", "home_team", "viewers"]].to_dict("records") == [
        {"week": 2, "away_team": "DAL", "home_team": "PHI", "viewers": 27_200_000},
        {"week": 3, "away_team": "PHI", "home_team": "NYG", "viewers": 20_000_000},
    ]


def test_archive_parser_accepts_unidentified_ten_column_tables() -> None:
    payload = b"""
    <h4>Week 4 games</h4><table>
    <tr><th>Window</th><th>Game</th><th>Net</th><th>Rtg.</th><th>+/-</th>
        <th>Vwrs.</th><th>+/-</th><th>A18-49</th><th>A18-34</th><th>A25-54</th></tr>
    <tr><td>SNF</td><td>TB/KC</td><td>NBC</td><td>10.0</td><td>+1%</td>
        <td>18.5M</td><td>+2%</td><td>5.0</td><td>4.0</td><td>6.0</td></tr>
    </table>
    """
    rows, _assets = parse_archive_html(
        payload,
        season=2020,
        page_url="https://www.sportsmediawatch.com/example/",
        observed_at="2026-08-20T17:00:00+00:00",
    )
    assert rows[["week", "away_team", "home_team", "viewers"]].to_dict("records") == [
        {"week": 4, "away_team": "TB", "home_team": "KC", "viewers": 18_500_000}
    ]


def test_postseason_heading_clears_prior_regular_season_week() -> None:
    payload = b"""
    <h4>Week 17 games</h4><table id="olytable">
    <tr><td>SNF</td><td>PHI-NYG</td><td>NBC</td><td>10</td><td>0%</td><td>18M</td><td>0%</td><td>5</td></tr>
    </table>
    <h4>Wild Card</h4><table id="olytable">
    <tr><td>Early</td><td>TB-WAS</td><td>FOX</td><td>12</td><td>0%</td><td>20M</td><td>0%</td><td>6</td></tr>
    </table>
    """
    rows, _assets = parse_archive_html(
        payload,
        season=2020,
        page_url="https://www.sportsmediawatch.com/example/",
        observed_at="2026-08-20T17:00:00+00:00",
    )
    assert rows["week"].tolist()[0] == 17
    assert pd.isna(rows["week"].tolist()[1])


def test_point_in_time_view_refuses_undated_archive_revision() -> None:
    rows, _assets = parse_archive_html(
        ARCHIVE_HTML,
        season=2014,
        page_url="https://www.sportsmediawatch.com/example/",
        observed_at="2026-08-20T17:00:00+00:00",
    )
    with pytest.raises(SportsMediaWatchIngestError, match="publication timestamp"):
        point_in_time_view(rows, decision_at=pd.Timestamp("2014-09-09T12:00:00Z"))


def test_point_in_time_view_excludes_same_day_and_late_publications() -> None:
    rows = pd.DataFrame(
        [
            {
                "id": "prior_available",
                "event_date": "2024-09-08",
                "source_published_at": "2024-09-09T10:00:00Z",
            },
            {
                "id": "prior_late",
                "event_date": "2024-09-08",
                "source_published_at": "2024-09-10T13:00:00Z",
            },
            {
                "id": "same_game_outcome",
                "event_date": "2024-09-10",
                "source_published_at": "2024-09-10T10:00:00Z",
            },
        ]
    )
    visible = point_in_time_view(rows, decision_at=pd.Timestamp("2024-09-10T12:00:00Z"))
    assert visible["id"].tolist() == ["prior_available"]


def test_ingest_resumes_cached_pages_and_assets(tmp_path: Path) -> None:
    calls: list[str] = []
    image = b"fake-png"

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return image if url.endswith(".png") else ARCHIVE_HTML

    first = ingest(
        tmp_path,
        seasons=[2014],
        max_assets=1,
        fetcher=fetcher,
        sleeper=lambda _seconds: None,
    )
    assert first["pages_fetched_this_run"] == [2014]
    assert first["assets_downloaded_this_run"] == 1
    assert len(calls) == 2
    first_observed = pd.read_parquet(tmp_path / "ratings_rows.parquet").iloc[0][
        "source_observed_at"
    ]

    calls.clear()
    second = ingest(
        tmp_path,
        seasons=[2014],
        max_assets=None,
        fetcher=fetcher,
        sleeper=lambda _seconds: None,
    )
    assert second["pages_cached_before_run"] == [2014]
    assert second["assets_cached_before_run"] == 1
    assert calls == []
    second_observed = pd.read_parquet(tmp_path / "ratings_rows.parquet").iloc[0][
        "source_observed_at"
    ]
    assert second_observed == first_observed
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["point_in_time_contract"]["same_game_viewership_allowed"] is False
    assert set(manifest["output_sha256"]) == {"ratings_rows.parquet", "source_index.parquet"}
