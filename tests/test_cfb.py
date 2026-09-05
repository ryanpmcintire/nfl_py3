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
from nfl_ats.cfb import (
    CFB_LINE_SNAPSHOT_COLUMNS,
    CFB_PARTICIPANT_SNAPSHOT_COLUMNS,
    CFB_ROSTER_QUARANTINED_COLUMNS,
    CFB_ROSTER_SNAPSHOT_COLUMNS,
    assert_no_quarantined_roster_columns,
    canonicalize_cfb_game_rosters,
    canonicalize_cfb_lines,
    canonicalize_cfb_pbp,
    canonicalize_cfb_play_participants,
    canonicalize_cfb_schedules,
    canonicalize_cfbd_draft_picks,
    canonicalize_cfbd_portal,
    canonicalize_cfbd_recruiting_players,
    canonicalize_cfbd_recruiting_teams,
    canonicalize_cfbd_returning_production,
    canonicalize_cfbd_usage,
    canonicalize_espn_cfb_betting,
    cfb_line_source_regime,
    cfb_snapshot_from_root,
    cfb_source_spec,
    fetch_cfb_snapshot,
    latest_cfb_snapshot,
    load_cfb_snapshot,
    plan_cfb_ingest,
    summarize_cfb_snapshots,
)
from nfl_ats.data import DataContractError

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
    """Serve every CFB source for the requested seasons from in-memory bytes."""

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
    """Serve CFBD endpoints from memory; returns the authenticated-call log."""

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


def test_source_specs_and_requested_season_guards(tmp_path: Path) -> None:
    assert cfb_source_spec("espn-betting").key == "espn_betting"
    with pytest.raises(ValueError, match="Unknown CFB source"):
        cfb_source_spec("hockey")
    with pytest.raises(ValueError, match="non-empty"):
        fetch_cfb_snapshot("pbp", [], tmp_path)
    with pytest.raises(DataContractError, match="no data before 2004"):
        fetch_cfb_snapshot("pbp", [2003], tmp_path)
    with pytest.raises(DataContractError, match="placeholder"):
        fetch_cfb_snapshot("espn_betting", [2010], tmp_path)
    assert cfb_source_spec("draft-picks").single_call is True
    with pytest.raises(DataContractError, match="no data before 2021"):
        fetch_cfb_snapshot("portal", [2020], tmp_path)
    with pytest.raises(DataContractError, match="no data before 2013"):
        fetch_cfb_snapshot("usage", [2012], tmp_path)


def test_schedule_contract_rejects_structural_breaks() -> None:
    canonical, audit = canonicalize_cfb_schedules(_schedules_frame(2024), 2024)
    assert audit == {
        "rows": 3,
        "fbs_home_games": 3,
        "season_types": {"regular": 3},
    }
    assert canonical["game_id"].tolist() == [2024000, 2024001, 2024002]
    with pytest.raises(DataContractError, match="missing required columns"):
        canonicalize_cfb_schedules(_schedules_frame(2024).drop(columns=["home_division"]), 2024)
    duplicated = pd.concat([_schedules_frame(2024)] * 2, ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate game_id"):
        canonicalize_cfb_schedules(duplicated, 2024)
    with pytest.raises(DataContractError, match="contains seasons"):
        canonicalize_cfb_schedules(_schedules_frame(2023), 2024)
    mirror = _schedules_frame(2024)
    mirror["away_team"] = mirror["home_team"]
    with pytest.raises(DataContractError, match="identical teams"):
        canonicalize_cfb_schedules(mirror, 2024)


def test_line_contract_regimes_openers_and_junk_rows() -> None:
    source = _lines_frame((2013, 2020, 2024))
    junk = source.iloc[[0]].copy()
    junk["game_id"] = None
    source = pd.concat([source, source.iloc[[0]], junk], ignore_index=True)
    canonical, audit = canonicalize_cfb_lines(source, [2013, 2020, 2024])
    assert list(canonical.columns) == list(CFB_LINE_SNAPSHOT_COLUMNS)
    assert audit["full_row_duplicates_dropped"] == 1
    assert audit["unjoinable_rows_dropped"] == 1
    assert audit["per_season"]["2013"]["source_regime"] == "sbr_multibook"
    assert audit["per_season"]["2020"]["source_regime"] == "cfbd_provider_sparse"
    assert audit["per_season"]["2024"]["source_regime"] == "espn_book_era"
    assert audit["per_season"]["2020"]["opener_rows"] == 0
    assert audit["per_season"]["2020"]["unexpected_openers"] is False
    assert audit["per_season"]["2024"]["opener_rows"] > 0
    regimes = canonical.groupby("season")["source_regime"].first()
    assert regimes.loc[2013] == "sbr_multibook"
    assert regimes.loc[2020] == "cfbd_provider_sparse"
    assert regimes.loc[2024] == "espn_book_era"

    with pytest.raises(DataContractError, match="no rows for season 2019"):
        canonicalize_cfb_lines(source, [2013, 2019])
    stripped = _lines_frame((2013,))
    stripped["opening_lines"] = None
    with pytest.raises(DataContractError, match="zero openers"):
        canonicalize_cfb_lines(stripped, [2013])
    drifted = _lines_frame((2024,))
    drifted.loc[0, "market_type"] = "team_total"
    with pytest.raises(DataContractError, match="unknown market types"):
        canonicalize_cfb_lines(drifted, [2024])
    with pytest.raises(DataContractError, match="No CFB line source regime"):
        cfb_line_source_regime(2005)


def test_pbp_contract_rejects_structural_breaks() -> None:
    canonical, audit = canonicalize_cfb_pbp(_pbp_frame(2024), 2024)
    assert audit["rows"] == 6
    assert audit["games"] == 1
    assert audit["season_types"] == {"regular": 5, "postseason": 1}
    assert canonical["game_play_number"].tolist() == [1, 2, 3, 4, 5, 6]
    with pytest.raises(DataContractError, match="missing required columns"):
        canonicalize_cfb_pbp(_pbp_frame(2024).drop(columns=["EPA"]), 2024)
    duplicated = pd.concat([_pbp_frame(2024)] * 2, ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate game_id/game_play_number"):
        canonicalize_cfb_pbp(duplicated, 2024)
    with pytest.raises(DataContractError, match="contains seasons"):
        canonicalize_cfb_pbp(_pbp_frame(2023), 2024)
    drifted = _pbp_frame(2024)
    drifted["seasonType"] = 9
    with pytest.raises(DataContractError, match="unknown seasonType"):
        canonicalize_cfb_pbp(drifted, 2024)


def test_pbp_contract_drops_and_audits_known_upstream_defects() -> None:
    source = _pbp_frame(2024)
    null_key = source.iloc[[0]].copy()
    null_key["game_play_number"] = None
    allstar = source.iloc[[1]].copy()
    allstar["game_id"] = 999
    allstar["seasonType"] = 4
    combined = pd.concat([source, null_key, allstar], ignore_index=True)
    canonical, audit = canonicalize_cfb_pbp(combined, 2024)
    assert audit["rows"] == 6
    assert audit["games"] == 1
    assert audit["null_key_rows_dropped"] == 1
    assert audit["excluded_offseason_rows"] == {"rows": 1, "games": 1, "codes": {"4": 1}}
    assert canonical["game_play_number"].tolist() == [1, 2, 3, 4, 5, 6]
    all_excluded = _pbp_frame(2024)
    all_excluded["seasonType"] = 5
    with pytest.raises(DataContractError, match="contains no rows"):
        canonicalize_cfb_pbp(all_excluded, 2024)


def test_roster_contract_quarantines_availability_flags() -> None:
    source = _rosters_frame(2024)
    with pytest.raises(DataContractError, match="quarantined availability columns"):
        assert_no_quarantined_roster_columns(source)
    duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    canonical, audit = canonicalize_cfb_game_rosters(duplicated, 2024)
    assert list(canonical.columns) == list(CFB_ROSTER_SNAPSHOT_COLUMNS)
    assert not set(CFB_ROSTER_QUARANTINED_COLUMNS).intersection(canonical.columns)
    assert audit["duplicate_rows_dropped"] == 1
    assert audit["rows"] == 6
    assert audit["quarantined_columns_excluded"] == sorted(CFB_ROSTER_QUARANTINED_COLUMNS)
    with pytest.raises(DataContractError, match="missing required columns"):
        canonicalize_cfb_game_rosters(source.drop(columns=["athlete_id"]), 2024)
    with pytest.raises(DataContractError, match="contains seasons"):
        canonicalize_cfb_game_rosters(_rosters_frame(2023), 2024)


def test_participant_contract_fills_roles_and_rejects_duplicates() -> None:
    canonical, audit = canonicalize_cfb_play_participants(_participants_frame(2024), 2024)
    assert list(canonical.columns) == list(CFB_PARTICIPANT_SNAPSHOT_COLUMNS)
    assert audit == {"rows": 4, "games": 1}
    assert canonical["kicker_player_id"].isna().all()
    duplicated = pd.concat([_participants_frame(2024)] * 2, ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate game_id/play_id"):
        canonicalize_cfb_play_participants(duplicated, 2024)
    with pytest.raises(DataContractError, match="missing required columns"):
        canonicalize_cfb_play_participants(
            _participants_frame(2024).drop(columns=["tackler_player_id"]), 2024
        )


def test_espn_betting_contract_refuses_placeholder_content() -> None:
    canonical, audit = canonicalize_espn_cfb_betting(_espn_betting_frame(2024), 2024)
    assert audit["rows"] == 3
    assert audit["placeholder_rows_dropped"] == 1
    assert audit["odds_sources"] == {"core_odds_api": 3}
    assert canonical["game_spread"].nunique() == 3
    with pytest.raises(DataContractError, match="only placeholder default rows"):
        canonicalize_espn_cfb_betting(_espn_betting_frame(2024, placeholder=True), 2024)
    duplicated = pd.concat([_espn_betting_frame(2024)] * 2, ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate game_id"):
        canonicalize_espn_cfb_betting(duplicated, 2024)


def test_fetch_release_source_writes_immutable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_upstream(monkeypatch, (2024,))
    snapshot = fetch_cfb_snapshot("rosters", [2024], tmp_path, "fixed")
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["release_tag"] == "espn_cfb_game_rosters"
    assert manifest["source"]["license"].startswith("MIT")
    assert manifest["quarantined_columns"] == CFB_ROSTER_QUARANTINED_COLUMNS
    assert "availability_contract" in manifest
    assert manifest["rows"] == 6
    partition = manifest["partitions"][0]
    assert partition["sha256"]
    assert partition["source_file"]["sha256"]
    assert partition["source_file"]["updated_at"] == "2026-08-03T05:31:04Z"
    assert (snapshot.root / "source" / "game_rosters_2024.parquet").is_file()
    loaded = load_cfb_snapshot(snapshot)
    assert len(loaded) == 6
    assert not set(CFB_ROSTER_QUARANTINED_COLUMNS).intersection(loaded.columns)
    assert latest_cfb_snapshot(tmp_path, "rosters") == snapshot
    assert cfb_snapshot_from_root(snapshot.root) == snapshot
    with pytest.raises(FileExistsError):
        fetch_cfb_snapshot("rosters", [2024], tmp_path, "fixed")
    with pytest.raises(DataContractError, match="no asset"):
        fetch_cfb_snapshot("rosters", [2024, 2025], tmp_path)
    with pytest.raises(FileNotFoundError, match="No CFB pbp snapshots"):
        latest_cfb_snapshot(tmp_path, "pbp")
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        cfb_snapshot_from_root(tmp_path / "nowhere")


def test_fetch_pinned_cfbfastr_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_upstream(monkeypatch, (2013, 2020, 2024))
    schedules = fetch_cfb_snapshot("schedules", [2024], tmp_path)
    schedule_manifest = json.loads(schedules.manifest_path.read_text(encoding="utf-8"))
    assert schedule_manifest["source"]["commit_sha"] == COMMIT_SHA
    assert schedule_manifest["source"]["license"].startswith("CC BY 4.0")
    assert schedule_manifest["rows"] == 3
    assert schedule_manifest["audit"]["per_season"]["2024"]["fbs_home_games"] == 3

    lines = fetch_cfb_snapshot("lines", [2013, 2020, 2024], tmp_path)
    manifest = json.loads(lines.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["commit_sha"] == COMMIT_SHA
    assert manifest["semantics"]["timestamped_quotes"] is False
    assert set(manifest["source_regimes"]) == {
        "sbr_multibook",
        "cfbd_provider_sparse",
        "espn_book_era",
    }
    assert [partition["season"] for partition in manifest["partitions"]] == [2013, 2020, 2024]
    assert (lines.root / "source" / "cfb_line_odds.parquet").is_file()
    loaded = load_cfb_snapshot(lines)
    assert set(loaded["source_regime"]) == {
        "sbr_multibook",
        "cfbd_provider_sparse",
        "espn_book_era",
    }
    with pytest.raises(DataContractError, match="unavailable \\(404\\)"):
        fetch_cfb_snapshot("schedules", [2019], tmp_path, "missing-season")


def test_plan_cfb_ingest_resolves_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = _fake_upstream(monkeypatch, (2024,))
    plan = plan_cfb_ingest("schedules", [2024])
    assert plan["dry_run"] is True
    assert plan["provenance"]["commit_sha"] == COMMIT_SHA
    assert plan["files"][0]["name"] == "cfb_schedules_2024.parquet"
    assert plan["total_bytes"] > 0
    lines_plan = plan_cfb_ingest("lines", [2024])
    assert lines_plan["files"][0]["name"] == "cfb_line_odds.parquet"
    assert lines_plan["files"][0]["season"] is None
    pbp_plan = plan_cfb_ingest("pbp", [2024])
    assert pbp_plan["provenance"]["release_tag"] == "espn_cfb_pbp"
    assert pbp_plan["files"][0]["updated_at"] == "2026-08-03T05:31:04Z"
    with pytest.raises(DataContractError, match="no asset"):
        plan_cfb_ingest("pbp", [2024, 2025])
    with pytest.raises(DataContractError, match="missing files at the pinned commit"):
        plan_cfb_ingest("schedules", [2019])
    assert pages
    assert not any(tmp_path.iterdir())


def test_cfbd_draft_pick_contract_and_crosswalk_rates() -> None:
    frame = pd.json_normalize(_draft_pick_records((2024,)))
    canonical, audit = canonicalize_cfbd_draft_picks(frame, 2024)
    assert list(canonical.columns) == list(cfb.CFBD_DRAFT_PICK_SNAPSHOT_COLUMNS)
    assert audit["rows"] == 2
    assert audit["college_athlete_id_nonnull_rate"] == 0.5
    assert audit["nfl_athlete_id_nonnull_rate"] == 1.0
    assert audit["both_ids_nonnull_rate"] == 0.5
    assert audit["zero_value_ids"] == {"collegeAthleteId": 0, "nflAthleteId": 0}
    assert canonical["hometownInfo.city"].tolist() == ["Springfield", "Springfield"]
    with pytest.raises(DataContractError, match="missing required columns"):
        canonicalize_cfbd_draft_picks(frame.drop(columns=["collegeAthleteId"]), 2024)
    with pytest.raises(DataContractError, match="missing required columns"):
        canonicalize_cfbd_draft_picks(frame.drop(columns=["nflAthleteId"]), 2024)
    duplicated = pd.concat([frame] * 2, ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate year/overall"):
        canonicalize_cfbd_draft_picks(duplicated, 2024)
    with pytest.raises(DataContractError, match="contains seasons"):
        canonicalize_cfbd_draft_picks(frame, 2025)
    with pytest.raises(DataContractError, match="contains no rows"):
        canonicalize_cfbd_draft_picks(frame.iloc[0:0], 2024)


def test_cfbd_team_grain_contracts() -> None:
    returning = pd.json_normalize(_returning_records(2024))
    canonical, audit = canonicalize_cfbd_returning_production(returning, 2024)
    assert audit == {"rows": 2, "conferences": 1, "null_percent_ppa_rows": 0}
    assert canonical["team"].tolist() == ["Aardvark State", "Badger Tech"]
    with pytest.raises(DataContractError, match="duplicate teams"):
        canonicalize_cfbd_returning_production(pd.concat([returning] * 2, ignore_index=True), 2024)
    with pytest.raises(DataContractError, match="missing required columns"):
        canonicalize_cfbd_returning_production(returning.drop(columns=["percentPPA"]), 2024)
    with pytest.raises(DataContractError, match="contains seasons"):
        canonicalize_cfbd_returning_production(returning, 2023)

    teams = pd.json_normalize(_recruiting_team_records(2024))
    canonical_teams, team_audit = canonicalize_cfbd_recruiting_teams(teams, 2024)
    assert team_audit == {"rows": 3, "max_rank": 3}
    assert canonical_teams["rank"].tolist() == [1, 2, 3]
    with pytest.raises(DataContractError, match="duplicate teams"):
        canonicalize_cfbd_recruiting_teams(pd.concat([teams] * 2, ignore_index=True), 2024)
    with pytest.raises(DataContractError, match="contains no rows"):
        canonicalize_cfbd_recruiting_teams(teams.iloc[0:0], 2024)


def test_cfbd_recruit_usage_and_portal_contracts() -> None:
    recruits = pd.json_normalize(_recruit_records(2025))
    canonical, audit = canonicalize_cfbd_recruiting_players(recruits, 2025)
    assert list(canonical.columns) == list(cfb.CFBD_RECRUIT_SNAPSHOT_COLUMNS)
    assert audit["rows"] == 3
    assert audit["athlete_id_nonnull_rate"] == round(2 / 3, 4)
    assert audit["recruit_types"] == {"HighSchool": 3}
    with pytest.raises(DataContractError, match="duplicate recruit ids"):
        canonicalize_cfbd_recruiting_players(pd.concat([recruits] * 2, ignore_index=True), 2025)
    nameless = recruits.copy()
    nameless["id"] = None
    with pytest.raises(DataContractError, match="null recruit ids"):
        canonicalize_cfbd_recruiting_players(nameless, 2025)
    drifted = recruits.copy()
    drifted.loc[0, "recruitType"] = "MiddleSchool"
    with pytest.raises(DataContractError, match="unknown recruit types"):
        canonicalize_cfbd_recruiting_players(drifted, 2025)

    usage = pd.json_normalize(_usage_records(2024))
    canonical_usage, usage_audit = canonicalize_cfbd_usage(usage, 2024)
    assert usage_audit == {
        "rows": 3,
        "players": 3,
        "teams": 1,
        "shadow_duplicate_rows_dropped": 0,
    }
    assert canonical_usage["usage.overall"].tolist() == [0.5, 0.5, 0.5]
    shadowed = pd.concat([usage, usage.iloc[[0]]], ignore_index=True)
    shadowed.loc[shadowed.index[-1], "conference"] = ""
    deduped, shadow_audit = canonicalize_cfbd_usage(shadowed, 2024)
    assert shadow_audit["rows"] == 3
    assert shadow_audit["shadow_duplicate_rows_dropped"] == 1
    assert deduped["conference"].notna().all()
    conflicting = pd.concat([usage, usage.iloc[[0]]], ignore_index=True)
    conflicting.loc[conflicting.index[-1], "usage.overall"] = 0.9
    with pytest.raises(DataContractError, match="conflicting usage shares"):
        canonicalize_cfbd_usage(conflicting, 2024)
    anonymous = usage.copy()
    anonymous["id"] = None
    with pytest.raises(DataContractError, match="null athlete ids"):
        canonicalize_cfbd_usage(anonymous, 2024)

    portal = pd.json_normalize(_portal_records(2023))
    canonical_portal, portal_audit = canonicalize_cfbd_portal(
        pd.concat([portal, portal.iloc[[0]]], ignore_index=True), 2023
    )
    assert portal_audit["rows"] == 3
    assert portal_audit["duplicate_rows_dropped"] == 1
    assert portal_audit["eligibility"] == {"Immediate": 3}
    assert portal_audit["identity_contract"] == cfb.CFBD_PORTAL_IDENTITY_CONTRACT
    assert "athleteId" not in canonical_portal.columns
    drifted_portal = portal.copy()
    drifted_portal.loc[0, "eligibility"] = "Probably"
    with pytest.raises(DataContractError, match="unknown eligibility values"):
        canonicalize_cfbd_portal(drifted_portal, 2023)
    with pytest.raises(DataContractError, match="contains no rows"):
        canonicalize_cfbd_portal(portal.iloc[0:0], 2023)


def test_fetch_cfbd_sources_snapshot_endpoint_params_and_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_cfbd_upstream(monkeypatch, (2024,))

    draft = fetch_cfb_snapshot("draft-picks", [2024], tmp_path, "fixed")
    manifest = json.loads(draft.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["origin"] == "cfbd_api"
    assert manifest["source"]["license"].startswith("CollegeFootballData")
    assert manifest["source"]["api_version"] == "5.99.0-test"
    assert CFBD_TEST_KEY not in draft.manifest_path.read_text(encoding="utf-8")
    assert manifest["audit"]["api_calls"] == 1
    assert manifest["audit"]["quota_headers_last"] == {"X-CallLimit-Remaining": "941"}
    assert manifest["audit"]["seasons_in_response"] == [2023, 2024]
    assert manifest["crosswalk_contract"] == cfb.CFBD_DRAFT_CROSSWALK_CONTRACT
    partition = manifest["partitions"][0]
    assert partition["source_file"]["endpoint"] == "/draft/picks"
    assert partition["source_file"]["params"] == {}
    assert partition["source_file"]["sha256"]
    assert (draft.root / "source" / "draft_picks.json").is_file()
    loaded = load_cfb_snapshot(draft)
    assert loaded["year"].tolist() == [2024, 2024]
    assert latest_cfb_snapshot(tmp_path, "draft-picks") == draft

    usage = fetch_cfb_snapshot("usage", [2024], tmp_path)
    usage_manifest = json.loads(usage.manifest_path.read_text(encoding="utf-8"))
    assert usage_manifest["audit"]["api_calls"] == 1
    assert usage_manifest["partitions"][0]["source_file"]["params"] == {"year": 2024}
    assert (usage.root / "source" / "usage_2024.json").is_file()

    assert calls == [
        f"{cfb.CFBD_API_ROOT}/draft/picks",
        f"{cfb.CFBD_API_ROOT}/player/usage?year=2024",
    ]
    with pytest.raises(DataContractError, match="HTTP 404"):
        fetch_cfb_snapshot("portal", [2023], tmp_path)


def test_cfbd_get_retries_burst_rate_limit_then_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    naps: list[float] = []

    def flaky(url: str, api_key: str) -> tuple[bytes, dict[str, str]]:
        attempts.append(url)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(url, 429, "burst", None, None)  # type: ignore[arg-type]
        return b"[]", {"X-CallLimit-Remaining": "900"}

    monkeypatch.setattr(cfb, "_cfbd_http", flaky)
    monkeypatch.setattr(cfb, "_sleep", lambda seconds: naps.append(seconds))
    payload, quota, url = cfb._cfbd_get("/player/portal", {"year": 2021}, "k")
    assert payload == b"[]"
    assert quota == {"X-CallLimit-Remaining": "900"}
    assert url == f"{cfb.CFBD_API_ROOT}/player/portal?year=2021"
    assert naps == list(cfb.CFBD_RETRY_WAITS)
    assert len(attempts) == 3

    def always_burst(url: str, api_key: str) -> tuple[bytes, dict[str, str]]:
        raise urllib.error.HTTPError(url, 429, "burst", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(cfb, "_cfbd_http", always_burst)
    with pytest.raises(DataContractError, match="HTTP 429"):
        cfb._cfbd_get("/player/portal", {"year": 2021}, "k")


def test_cfbd_ingest_fails_closed_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_network(url: str, api_key: str = "") -> bytes:
        raise AssertionError(f"network attempted: {url}")

    monkeypatch.setattr(cfb, "_http_bytes", refuse_network)
    monkeypatch.setattr(cfb, "_cfbd_http", refuse_network)
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(DataContractError, match="CFBD_API_KEY"):
        fetch_cfb_snapshot("portal", [2021], tmp_path)
    monkeypatch.setenv("CFBD_API_KEY", "   ")
    with pytest.raises(DataContractError, match="CFBD_API_KEY"):
        fetch_cfb_snapshot("returning-production", [2024], tmp_path)
    assert not any(tmp_path.iterdir())


def test_plan_cfbd_ingest_spends_no_calls_and_reports_key_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_cfbd_upstream(monkeypatch, (2024,))
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    plan = plan_cfb_ingest("usage", [2023, 2024])
    assert plan["dry_run"] is True
    assert plan["api_calls_required"] == 2
    assert plan["api_key_configured"] is False
    assert plan["provenance"]["api_version"] == "5.99.0-test"
    assert plan["files"][0] == {
        "season": 2023,
        "endpoint": "/player/usage",
        "params": {"year": 2023},
        "url": f"{cfb.CFBD_API_ROOT}/player/usage?year=2023",
    }
    draft_plan = plan_cfb_ingest("draft-picks", [2004, 2005])
    assert draft_plan["api_calls_required"] == 1
    assert draft_plan["files"][0]["season"] is None
    monkeypatch.setenv("CFBD_API_KEY", CFBD_TEST_KEY)
    assert plan_cfb_ingest("portal", [2021])["api_key_configured"] is True
    assert calls == []
    assert not any(tmp_path.iterdir())


@pytest.mark.full  # ENG-11: end-to-end CLI CFB build/model workflow; dominates --durations
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
