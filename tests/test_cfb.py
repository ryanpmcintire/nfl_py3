from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats import cfb, cli
from nfl_ats.cfb import summarize_cfb_snapshots

COMMIT_SHA = "f" * 40
CFBD_TEST_KEY = "unit-test-key"


def _schedules_frame(season: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(3):
        rows.append(
            {
                "game_id": season * 1000 + index,
                "season": season,
                "week": index + 1,
                "season_type": "regular",
                "start_date": f"{season}-09-0{index + 1}T16:00:00.000Z",
                "start_time_tbd": False,
                "completed": True,
                "neutral_site": False,
                "conference_game": index > 0,
                "attendance": 50000,
                "venue_id": 10,
                "venue": "Stadium",
                "home_id": 1,
                "home_team": f"Home{index}",
                "home_division": "fbs",
                "home_conference": "Big Ten",
                "home_points": 28,
                "home_post_win_prob": 0.9,
                "home_pregame_elo": 1500,
                "home_postgame_elo": 1510,
                "away_id": 2,
                "away_team": f"Away{index}",
                "away_division": "fbs" if index else "fcs",
                "away_conference": "SEC",
                "away_points": 21,
                "away_post_win_prob": 0.1,
                "away_pregame_elo": 1490,
                "away_postgame_elo": 1480,
                "excitement_index": 5.0,
                "notes": None,
            }
        )
    return pd.DataFrame(rows)


def _lines_frame(seasons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in seasons:
        has_openers = season >= 2012 and season != 2020
        for game in range(2):
            game_id = season * 100 + game
            for book in ("PINNACLE", "5Dimes & sportbet"):
                for market, abbr, line in (
                    ("spread", "HOM", -3.5),
                    ("spread", "AWY", 3.5),
                    ("total", "over", 55.5),
                    ("money_line", "HOM", None),
                ):
                    rows.append(
                        {
                            "game_id": game_id,
                            "season": season,
                            "date_time": f"{season}-09-01 19:00:00",
                            "market_type": market,
                            "abbr": abbr,
                            "lines": line,
                            "odds": -110,
                            "opening_lines": 3.0 if has_openers and market == "spread" else None,
                            "opening_odds": -110 if has_openers and market == "spread" else None,
                            "book": book,
                            "season_type": "regular",
                            "week": 1,
                            "home_team_id": 11,
                            "away_team_id": 22,
                        }
                    )
    return pd.DataFrame(rows)


def _pbp_frame(season: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for play in range(1, 7):
        rows.append(
            {
                "game_id": season * 10 + 1,
                "game_play_number": play,
                "season": season,
                "week": 1,
                "seasonType": 2 if play < 6 else 3,
                "pos_team": "HOM",
                "def_pos_team": "AWY",
                "homeTeamId": 11,
                "awayTeamId": 22,
                "period": 1,
                "down": 1,
                "distance": 10,
                "text": "rush for 5 yards",
                "EPA": 0.1 * play,
                "wpa": 0.01,
                "rush": True,
                "pass": False,
                "pos_team_id": 11,
                "def_pos_team_id": 22,
                "statYardage": 5,
                "type.text": "Rush",
            }
        )
    return pd.DataFrame(rows)


def _rosters_frame(season: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game in range(2):
        for athlete in range(3):
            rows.append(
                {
                    "game_id": season * 10 + game,
                    "season": season,
                    "week": game + 1,
                    "athlete_id": 1000 + athlete,
                    "team_id": 5,
                    "home_away": "home",
                    "full_name": f"Player {athlete}",
                    "jersey": str(athlete),
                    "team_abbreviation": "HOM",
                    "first_name": "Player",
                    "last_name": str(athlete),
                    "weight": 100.0,
                    "height": 75.0,
                    "date_of_birth": "2004-01-01",
                    "experience_years": 2.0,
                    "team_display_name": "Home Team",
                    "order": athlete,
                    "active": True,
                    "is_active": True,
                    "did_not_play": False,
                    "starter": False,
                    "valid": False,
                    "status_id": "1",
                    "status_name": "Active",
                    "status_type": "active",
                    "status_abbreviation": "A",
                }
            )
    return pd.DataFrame(rows)


def _participants_frame(season: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for play in range(1, 5):
        rows.append(
            {
                "game_id": season * 10 + 1,
                "play_id": play,
                "season": season,
                "week": 1,
                "passer_player_id": "4430000",
                "rusher_player_id": None,
                "receiver_player_id": "4430001",
                "tackler_player_id": "4430002",
            }
        )
    return pd.DataFrame(rows)


def _espn_betting_frame(season: int, placeholder: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game in range(4):
        junk = placeholder or game == 3
        rows.append(
            {
                "game_id": season * 10 + game,
                "season": season,
                "week": game + 1,
                "game_spread": 2.5 if junk else 3.0 + game,
                "over_under": 55.5 if junk else 48.0 + game,
                "home_favorite": True,
                "home_team_spread": -2.5 if junk else -(3.0 + game),
                "game_spread_available": not junk,
                "odds_source": "default" if junk else "core_odds_api",
            }
        )
    return pd.DataFrame(rows)


def _draft_pick_records(years: tuple[int, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for year in years:
        for overall in (1, 2):
            records.append(
                {
                    "collegeAthleteId": None if overall == 2 else 4430000 + year,
                    "nflAthleteId": 5540000 + year * 10 + overall,
                    "collegeId": 99,
                    "collegeTeam": "Alma Mater",
                    "collegeConference": "Big Ten",
                    "nflTeamId": 12,
                    "nflTeam": "Testers",
                    "year": year,
                    "overall": overall,
                    "round": 1,
                    "pick": overall,
                    "name": f"Pick {year}-{overall}",
                    "position": "QB",
                    "height": 75.0,
                    "weight": 210,
                    "preDraftRanking": overall,
                    "preDraftPositionRanking": 1,
                    "preDraftGrade": 90,
                    "hometownInfo": {
                        "city": "Springfield",
                        "state": "OH",
                        "country": "USA",
                        "latitude": None,
                        "longitude": None,
                        "countyFips": None,
                    },
                }
            )
    return records


def _returning_records(season: int) -> list[dict[str, Any]]:
    return [
        {
            "season": season,
            "team": team,
            "conference": "B1G",
            "totalPPA": 100.0,
            "totalPassingPPA": 60.0,
            "totalReceivingPPA": 30.0,
            "totalRushingPPA": 10.0,
            "percentPPA": 0.6,
            "percentPassingPPA": 0.5,
            "percentReceivingPPA": 0.4,
            "percentRushingPPA": 0.7,
            "usage": 0.55,
            "passingUsage": 0.5,
            "receivingUsage": 0.45,
            "rushingUsage": 0.6,
        }
        for team in ("Aardvark State", "Badger Tech")
    ]


def _recruiting_team_records(season: int) -> list[dict[str, Any]]:
    teams = ("Aardvark State", "Badger Tech", "Capybara U")
    return [
        {"year": season, "rank": index + 1, "team": team, "points": 300.0 - index}
        for index, team in enumerate(teams)
    ]


def _recruit_records(season: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{season}-{index}",
            "athleteId": str(4560000 + index) if index else None,
            "recruitType": "HighSchool",
            "year": season,
            "ranking": index + 1,
            "name": f"Recruit {index}",
            "school": "Prep High",
            "committedTo": "Aardvark State",
            "position": "WR",
            "height": 74.0,
            "weight": 190,
            "stars": 4,
            "rating": 0.95,
            "city": "Springfield",
            "stateProvince": "OH",
            "country": "USA",
            "hometownInfo": {"fipsCode": None, "latitude": None, "longitude": None},
        }
        for index in range(3)
    ]


def _usage_records(season: int) -> list[dict[str, Any]]:
    return [
        {
            "season": season,
            "id": str(4430000 + index),
            "name": f"Player {index}",
            "position": "RB",
            "team": "Aardvark State",
            "conference": "B1G",
            "usage": {
                "overall": 0.5,
                "pass": 0.2,
                "rush": 0.7,
                "firstDown": 0.5,
                "secondDown": 0.5,
                "thirdDown": 0.4,
                "standardDowns": 0.55,
                "passingDowns": 0.3,
            },
        }
        for index in range(3)
    ]


def _portal_records(season: int) -> list[dict[str, Any]]:
    return [
        {
            "season": season,
            "firstName": "First",
            "lastName": f"Mover{index}",
            "position": "QB",
            "origin": "Aardvark State",
            "destination": "Badger Tech" if index else None,
            "transferDate": f"{season}-01-0{index + 1}T00:00:00.000Z",
            "rating": 0.9,
            "stars": 3,
            "eligibility": "Immediate",
        }
        for index in range(3)
    ]


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _json_bytes(records: object) -> bytes:
    return json.dumps(records).encode("utf-8")


def _release_payload(tag: str, assets: dict[str, bytes]) -> bytes:
    return json.dumps(
        {
            "assets": [
                {
                    "name": name,
                    "size": len(payload),
                    "updated_at": "2026-08-03T05:31:04Z",
                    "browser_download_url": f"https://dl.test/{tag}/{name}",
                }
                for name, payload in assets.items()
            ]
        }
    ).encode("utf-8")


def _fake_http(pages: dict[str, bytes]) -> Callable[[str], bytes]:
    def fetch(url: str) -> bytes:
        if url not in pages:
            raise urllib.error.HTTPError(url, 404, "missing", None, None)  # type: ignore[arg-type]
        return pages[url]

    return fetch


def _fake_upstream(monkeypatch: pytest.MonkeyPatch, seasons: tuple[int, ...]) -> dict[str, bytes]:

    pages: dict[str, bytes] = {}
    api = cfb.GITHUB_API_ROOT
    raw = cfb.RAW_CONTENT_ROOT
    cfbfastr = cfb.CFBFASTR_DATA_REPOSITORY
    releases = cfb.SPORTSDATAVERSE_DATA_REPOSITORY
    pages[f"{api}/repos/{cfbfastr}/branches/main"] = json.dumps(
        {"commit": {"sha": COMMIT_SHA}}
    ).encode("utf-8")
    schedule_files = {
        f"cfb_schedules_{season}.parquet": _parquet_bytes(_schedules_frame(season))
        for season in seasons
    }
    for name, payload in schedule_files.items():
        pages[f"{raw}/{cfbfastr}/{COMMIT_SHA}/schedules/parquet/{name}"] = payload
    lines_payload = _parquet_bytes(_lines_frame(seasons))
    pages[f"{raw}/{cfbfastr}/{COMMIT_SHA}/betting/parquet/cfb_line_odds.parquet"] = lines_payload
    pages[f"{api}/repos/{cfbfastr}/contents/schedules/parquet?ref={COMMIT_SHA}"] = json.dumps(
        [{"name": name, "size": len(payload)} for name, payload in schedule_files.items()]
    ).encode("utf-8")
    pages[f"{api}/repos/{cfbfastr}/contents/betting/parquet?ref={COMMIT_SHA}"] = json.dumps(
        [{"name": "cfb_line_odds.parquet", "size": len(lines_payload)}]
    ).encode("utf-8")
    release_sources = {
        "espn_cfb_pbp": ("play_by_play", _pbp_frame),
        "espn_cfb_game_rosters": ("game_rosters", _rosters_frame),
        "espn_cfb_play_participants": ("play_participants", _participants_frame),
        "espn_cfb_betting": ("betting", _espn_betting_frame),
    }
    for tag, (prefix, builder) in release_sources.items():
        assets = {
            f"{prefix}_{season}.parquet": _parquet_bytes(builder(season)) for season in seasons
        }
        pages[f"{api}/repos/{releases}/releases/tags/{tag}"] = _release_payload(tag, assets)
        for name, payload in assets.items():
            pages[f"https://dl.test/{tag}/{name}"] = payload
    monkeypatch.setattr(cfb, "_http_bytes", _fake_http(pages))
    return pages


def _fake_cfbd_upstream(
    monkeypatch: pytest.MonkeyPatch,
    seasons: tuple[int, ...],
    pages: dict[str, bytes] | None = None,
) -> list[str]:

    pages = pages if pages is not None else {}
    pages[cfb.CFBD_API_DOCS_URL] = _json_bytes({"info": {"version": "5.99.0-test"}})
    monkeypatch.setattr(cfb, "_http_bytes", _fake_http(pages))
    root = cfb.CFBD_API_ROOT
    draft_years = tuple(sorted({*seasons, min(seasons) - 1}))
    cfbd_pages: dict[str, bytes] = {
        f"{root}/draft/picks": _json_bytes(_draft_pick_records(draft_years))
    }
    for season in seasons:
        cfbd_pages[f"{root}/player/returning?year={season}"] = _json_bytes(
            _returning_records(season)
        )
        cfbd_pages[f"{root}/recruiting/teams?year={season}"] = _json_bytes(
            _recruiting_team_records(season)
        )
        cfbd_pages[f"{root}/recruiting/players?year={season}"] = _json_bytes(
            _recruit_records(season)
        )
        cfbd_pages[f"{root}/player/usage?year={season}"] = _json_bytes(_usage_records(season))
        cfbd_pages[f"{root}/player/portal?year={season}"] = _json_bytes(_portal_records(season))
    calls: list[str] = []

    def fetch(url: str, api_key: str) -> tuple[bytes, dict[str, str]]:
        assert api_key == CFBD_TEST_KEY
        calls.append(url)
        if url not in cfbd_pages:
            raise urllib.error.HTTPError(url, 404, "missing", None, None)  # type: ignore[arg-type]
        return cfbd_pages[url], {"X-CallLimit-Remaining": "941"}

    monkeypatch.setattr(cfb, "_cfbd_http", fetch)
    monkeypatch.setenv("CFBD_API_KEY", CFBD_TEST_KEY)
    return calls


@pytest.mark.full
def test_cli_cfb_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(tmp_path / "data"))
    pages = _fake_upstream(monkeypatch, (2024,))
    _fake_cfbd_upstream(monkeypatch, (2024,), pages=pages)

    assert cli.main(["cfb-summary"]) == 0
    summary: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert set(summary) == set(cfb.CFB_SOURCES)
    assert all(value is None for value in summary.values())

    assert (
        cli.main(
            [
                "cfb-ingest",
                "--source",
                "schedules",
                "--start-season",
                "2024",
                "--end-season",
                "2024",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert not (tmp_path / "data" / "cfb").exists()

    for source in (
        "schedules",
        "lines",
        "pbp",
        "rosters",
        "participants",
        "espn-betting",
        "draft-picks",
        "returning-production",
        "recruiting-teams",
        "recruiting-players",
        "usage",
        "portal",
    ):
        assert (
            cli.main(
                [
                    "cfb-ingest",
                    "--source",
                    source,
                    "--start-season",
                    "2024",
                    "--end-season",
                    "2024",
                ]
            )
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["seasons"] == [2024]
        assert payload["rows"] > 0
        assert payload["partitions"][0]["source_file"]["sha256"]

    assert cli.main(["cfb-summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert all(value is not None for value in summary.values())
    assert summary["lines"]["rows"] > 0

    direct = summarize_cfb_snapshots(tmp_path / "data" / "cfb")
    assert direct["schedules"]["seasons"] == [2024]

    with pytest.raises(SystemExit):
        cli.main(
            [
                "cfb-ingest",
                "--source",
                "espn-betting",
                "--start-season",
                "2010",
                "--end-season",
                "2010",
            ]
        )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "cfb-ingest",
                "--source",
                "pbp",
                "--start-season",
                "2024",
                "--end-season",
                "2023",
            ]
        )
