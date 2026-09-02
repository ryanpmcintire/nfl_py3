from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backfill_sports_media_watch_timestamps import (
    PublicationBackfillError,
    _audience_values,
    _finish_snapshot,
    _prepare_snapshot,
    enrich_assets,
    enrich_rows,
)
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


def test_publication_match_requires_postgame_exact_audience_and_matchup() -> None:
    rows = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 9,
                "event_date": None,
                "away_team": "NO",
                "home_team": "TB",
                "network": "NBC",
                "viewers": 16_880_000,
                "source_published_at": None,
                "point_in_time_usable": False,
            }
        ]
    )
    schedules = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 9,
                "gameday": "2020-11-08",
                "gametime": "20:20",
                "away_team": "NO",
                "home_team": "TB",
                "game_type": "REG",
            }
        ]
    )
    posts = pd.DataFrame(
        [
            {
                "post_id": 1,
                "source_published_at": "2020-11-08T12:00:00Z",
                "source_modified_at": "2020-11-08T12:30:00Z",
                "source_url": "https://www.sportsmediawatch.com/pregame/",
                "title": "Saints Buccaneers preview",
                "content_text": "Projected audience 16.88 million.",
            },
            {
                "post_id": 2,
                "source_published_at": "2020-11-09T03:00:00Z",
                "source_modified_at": "2020-11-09T03:01:00Z",
                "source_url": "https://www.sportsmediawatch.com/in-game/",
                "title": "Saints Buccaneers update",
                "content_text": "The Saints and Buccaneers drew 16.88 million viewers so far.",
            },
            {
                "post_id": 3,
                "source_published_at": "2020-11-11T05:27:09Z",
                "source_modified_at": "2020-11-11T05:28:27Z",
                "source_url": "https://www.sportsmediawatch.com/week-9/",
                "title": "Quiet Week 9",
                "content_text": "The Saints routed the Buccaneers before 16.88 million viewers.",
            },
        ]
    )

    enriched = enrich_rows(rows, posts, schedules).iloc[0]

    assert enriched["source_published_at"] == "2020-11-11T05:27:09Z"
    assert enriched["source_modified_at"] == "2020-11-11T05:28:27Z"
    assert enriched["timestamp_source_id"] == "3"
    assert bool(enriched["point_in_time_usable"])


def test_publication_match_fails_closed_without_exact_identity() -> None:
    rows = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 9,
                "event_date": None,
                "away_team": "NO",
                "home_team": "TB",
                "network": "NBC",
                "viewers": 16_880_000,
                "source_published_at": None,
                "point_in_time_usable": False,
            }
        ]
    )
    schedules = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 9,
                "gameday": "2020-11-08",
                "away_team": "NO",
                "home_team": "TB",
                "game_type": "REG",
            }
        ]
    )
    posts = pd.DataFrame(
        [
            {
                "post_id": 3,
                "source_published_at": "2020-11-11T05:27:09Z",
                "source_modified_at": "2020-11-11T05:28:27Z",
                "source_url": "https://www.sportsmediawatch.com/other/",
                "title": "Unrelated audience",
                "content_text": "The Eagles and Cowboys averaged 16.88 million viewers.",
            }
        ]
    )

    enriched = enrich_rows(rows, posts, schedules).iloc[0]

    assert pd.isna(enriched["source_published_at"])
    assert not bool(enriched["point_in_time_usable"])


def test_event_date_fallback_rejects_same_day_publication() -> None:
    rows = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 9,
                "event_date": "2020-11-08",
                "away_team": "NO",
                "home_team": "TB",
                "network": "NBC",
                "viewers": 16_880_000,
            }
        ]
    )
    schedules = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 9,
                "gameday": "2020-11-08",
                "away_team": "SEA",
                "home_team": "BUF",
                "game_type": "REG",
            }
        ]
    )
    posts = pd.DataFrame(
        [
            {
                "post_id": 1,
                "source_published_at": "2020-11-08T23:59:00Z",
                "source_modified_at": "2020-11-08T23:59:30Z",
                "source_url": "https://www.sportsmediawatch.com/same-day/",
                "title": "Saints Buccaneers",
                "content_text": "16.88 million viewers",
            },
            {
                "post_id": 2,
                "source_published_at": "2020-11-09T01:00:00Z",
                "source_modified_at": "2020-11-09T01:01:00Z",
                "source_url": "https://www.sportsmediawatch.com/next-day/",
                "title": "Saints Buccaneers",
                "content_text": "16.88 million viewers",
            },
        ]
    )

    enriched = enrich_rows(rows, posts, schedules).iloc[0]

    assert enriched["timestamp_source_id"] == "2"


def test_future_and_postgame_mutations_do_not_change_prior_publication_match() -> None:
    rows = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 9,
                "event_date": None,
                "away_team": "NO",
                "home_team": "TB",
                "network": "NBC",
                "viewers": 16_880_000,
            }
        ]
    )
    schedules = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 9,
                "gameday": "2020-11-08",
                "gametime": "20:20",
                "away_team": "NO",
                "home_team": "TB",
                "game_type": "REG",
                "away_score": 38,
                "home_score": 3,
            }
        ]
    )
    posts = pd.DataFrame(
        [
            {
                "post_id": 3,
                "source_published_at": "2020-11-11T05:27:09Z",
                "source_modified_at": "2020-11-11T05:28:27Z",
                "source_url": "https://www.sportsmediawatch.com/week-9/",
                "title": "Week 9 ratings",
                "content_text": "The Saints and Buccaneers averaged 16.88 million viewers.",
            }
        ]
    )
    expected = enrich_rows(rows, posts, schedules).iloc[0]

    mutated_schedules = schedules.assign(away_score=0, home_score=99)
    mutated_schedules = pd.concat(
        [
            mutated_schedules,
            pd.DataFrame(
                [
                    {
                        "season": 2020,
                        "week": 10,
                        "gameday": "2020-11-15",
                        "gametime": "20:20",
                        "away_team": "BAL",
                        "home_team": "NE",
                        "game_type": "REG",
                        "away_score": 17,
                        "home_score": 23,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    mutated_posts = pd.concat(
        [
            posts,
            pd.DataFrame(
                [
                    {
                        "post_id": 99,
                        "source_published_at": "2020-11-20T05:00:00Z",
                        "source_modified_at": "2026-01-01T00:00:00Z",
                        "source_url": "https://www.sportsmediawatch.com/future/",
                        "title": "Future revision",
                        "content_text": "The Saints and Buccaneers averaged 16.88 million viewers.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    actual = enrich_rows(rows, mutated_posts, mutated_schedules).iloc[0]

    assert actual["source_published_at"] == expected["source_published_at"]
    assert actual["source_modified_at"] == expected["source_modified_at"]
    assert actual["timestamp_source_id"] == expected["timestamp_source_id"]


def test_media_match_requires_one_exact_filename_and_keeps_both_timestamps() -> None:
    assets = pd.DataFrame(
        [
            {
                "season": 2023,
                "week": 9,
                "asset_url": (
                    "https://www.sportsmediawatch.com/wp-content/uploads/2023/11/"
                    "nflweek9update-1024x600.png"
                ),
                "source_published_at": None,
                "point_in_time_usable": False,
            }
        ]
    )
    media = pd.DataFrame(
        [
            {
                "asset_key": "nflweek9update",
                "asset_identity": "/wp-content/uploads/2022/11/nflweek9update.png",
                "media_id": 101,
                "source_published_at": "2022-11-10T01:00:19Z",
                "source_modified_at": "2022-11-10T02:00:19Z",
                "attachment_url": "https://www.sportsmediawatch.com/2022-week-9/",
            },
            {
                "asset_key": "nflweek9update",
                "asset_identity": "/wp-content/uploads/2023/11/nflweek9update.png",
                "media_id": 112594,
                "source_published_at": "2023-11-09T01:00:19Z",
                "source_modified_at": "2023-11-09T02:00:19Z",
                "attachment_url": "https://www.sportsmediawatch.com/nflweek9update/",
            },
        ]
    )

    enriched = enrich_assets(assets, media).iloc[0]

    assert enriched["source_published_at"] == "2023-11-09T01:00:19Z"
    assert enriched["source_modified_at"] == "2023-11-09T02:00:19Z"
    assert enriched["timestamp_source_id"] == "112594"
    assert bool(enriched["point_in_time_usable"])


def test_snapshot_resume_requires_exact_config_and_verifies_complete_hash(tmp_path: Path) -> None:
    output = tmp_path / "publication-snapshot"
    config = {"schema": "test/1", "source_sha256": {"rows": "abc"}}
    started = _prepare_snapshot(output, config)
    assert started["status"] == "IN_PROGRESS"

    with pytest.raises(PublicationBackfillError, match="configuration is incompatible"):
        _prepare_snapshot(output, {**config, "source_sha256": {"rows": "changed"}})

    payload = b"sealed output"
    member = output / "rows.parquet"
    member.write_bytes(payload)
    manifest = {
        **started,
        "status": "COMPLETE",
        "output_sha256": {"rows.parquet": hashlib.sha256(payload).hexdigest()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _prepare_snapshot(output, config)["status"] == "COMPLETE"

    member.write_bytes(b"mutated")
    with pytest.raises(PublicationBackfillError, match="output hash mismatch"):
        _prepare_snapshot(output, config)


def test_existing_unconfigured_output_is_not_treated_as_resumable(tmp_path: Path) -> None:
    output = tmp_path / "not-a-snapshot"
    output.mkdir()

    with pytest.raises(PublicationBackfillError, match="is not a resumable"):
        _prepare_snapshot(output, {"schema": "test/1"})


def test_finalizing_snapshot_promotes_only_hash_pinned_staging_member(tmp_path: Path) -> None:
    output = tmp_path / "publication-snapshot"
    staged = output / "staging" / "rows.parquet"
    staged.parent.mkdir(parents=True)
    payload = b"complete staged parquet"
    staged.write_bytes(payload)
    manifest = {
        "schema": "test/1",
        "status": "FINALIZING",
        "output_sha256": {"rows.parquet": hashlib.sha256(payload).hexdigest()},
    }

    completed = _finish_snapshot(output, manifest)

    assert completed["status"] == "COMPLETE"
    assert (output / "rows.parquet").read_bytes() == payload
    assert not staged.exists()


def test_audience_parser_does_not_confuse_unscaled_ratings_with_viewers() -> None:
    assert _audience_values("a 9.5 rating and 16.88 million; cable drew 802,000") == {
        16_880_000,
        802_000,
    }
