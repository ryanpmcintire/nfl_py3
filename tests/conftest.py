from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nfl_ats.board_site_content import SiteContent, load_site_content
from nfl_ats.constants import GRAPH_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from nfl_ats.market_data import QUOTE_COLUMNS

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def _shared_real_site_content() -> SiteContent:
    """Loads real repo artifacts into a :class:`SiteContent` ONCE for the
    whole test session (WP51, test-suite speed).

    ``load_site_content(artifacts_root, require_fresh_arrest_overlay=False)``
    is a pure, read-only function of its arguments -- see its own docstring
    in ``nfl_ats/board_site_content.py`` ("this module itself never opens an
    artifact" / loaders only read). ``tests/test_board_improvements.py`` and
    ``tests/test_board_terminal.py`` each called it with these exact
    arguments in their own module-scoped ``site_content`` fixture, and
    ``tests/test_board_site.py``'s ``site`` fixture triggered a THIRD,
    identical call indirectly via ``build_site``. Each call cost ~44-54s of
    real I/O (measured), so three modules paid it three times for the same
    result. This fixture computes it once; the three test files' own
    fixtures now return this shared, frozen (immutable) object instead of
    recomputing it. Nothing may mutate it -- every dataclass it references is
    ``@dataclass(frozen=True)`` with tuple fields, and every test that reads
    it only reads or calls ``dataclasses.replace`` to produce an unrelated
    copy.
    """

    # The dashboard contracts exercise artifact -> view model -> HTML.  They
    # do not exercise the market snapshot reader, which has dedicated tests.
    # Avoid scanning every historical quote parquet merely to derive the
    # current card's optional dispersion pool; an empty, schema-correct quote
    # frame takes the production missing-data fallback through the same code.
    empty_quotes = pd.DataFrame(columns=QUOTE_COLUMNS)
    with patch(
        "nfl_ats.best_pick_nomination.load_quote_history",
        return_value=empty_quotes,
    ):
        return load_site_content(_REPO_ROOT / "artifacts", require_fresh_arrest_overlay=False)


# ---------------------------------------------------------------------------
# Synthetic college-football universe for the XLG-03 benchmark tests
# ---------------------------------------------------------------------------

CFB_FIXTURE_SEASONS = (2013, 2014)
CFB_FIXTURE_WEEKS = 14
CFB_FIXTURE_TEAMS = tuple(range(1, 9))
CFB_FIXTURE_STRENGTH = {team: (8 - team) * 1.5 for team in CFB_FIXTURE_TEAMS}
# Crafted irregularities exercised by the feature-table contract tests.
CFB_GAME_UNRESOLVED = 20130302  # both line abbrs unique to this game
CFB_GAME_REPAIRED = 20140202  # home abbr unique, away abbr resolvable
CFB_GAME_NO_LINES = 20130502
CFB_GAME_NO_PBP = 20140402


def _cfb_round_robin() -> list[list[tuple[int, int]]]:
    teams = list(CFB_FIXTURE_TEAMS)
    rounds: list[list[tuple[int, int]]] = []
    for round_index in range(len(teams) - 1):
        pairs = []
        for index in range(len(teams) // 2):
            first, second = teams[index], teams[len(teams) - 1 - index]
            pairs.append((first, second) if round_index % 2 == 0 else (second, first))
        rounds.append(pairs)
        teams = [teams[0], teams[-1], *teams[1:-1]]
    return rounds


def _cfb_conference(team: int) -> str:
    return "SEC" if team <= 4 else "Mid-American"


def cfb_true_spread(home: int, away: int) -> float:
    return 2.0 + CFB_FIXTURE_STRENGTH[home] - CFB_FIXTURE_STRENGTH[away]


def _cfb_schedule_rows() -> list[dict[str, object]]:
    rounds = _cfb_round_robin()
    rows: list[dict[str, object]] = []
    for season in CFB_FIXTURE_SEASONS:
        for week in range(1, CFB_FIXTURE_WEEKS + 1):
            pairs = rounds[(week - 1) % len(rounds)]
            swap = week > len(rounds)
            gameday = pd.Timestamp(f"{season}-08-30") + pd.Timedelta(days=7 * (week - 1))
            for pair_index, (first, second) in enumerate(pairs):
                home, away = (second, first) if swap else (first, second)
                game_id = season * 10000 + week * 100 + pair_index + 1
                date_value = gameday
                if season == 2014 and week == 8 and pair_index == 0:
                    date_value = gameday - pd.Timedelta(days=1)
                margin = round(
                    2
                    + CFB_FIXTURE_STRENGTH[home]
                    - CFB_FIXTURE_STRENGTH[away]
                    + ((game_id % 5) - 2)
                )
                away_points = 24 + (game_id % 7)
                rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "season_type": "regular",
                        "start_date": f"{date_value:%Y-%m-%d}T19:00:00.000Z",
                        "completed": True,
                        "neutral_site": False,
                        "conference_game": _cfb_conference(home) == _cfb_conference(away),
                        "home_id": home,
                        "home_team": f"Team {home}",
                        "home_division": "fbs",
                        "home_conference": _cfb_conference(home),
                        "home_points": away_points + margin,
                        "away_id": away,
                        "away_team": f"Team {away}",
                        "away_division": "fbs",
                        "away_conference": _cfb_conference(away),
                        "away_points": away_points,
                    }
                )
    rows.append(
        {
            "game_id": 20139901,
            "season": 2013,
            "week": 1,
            "season_type": "regular",
            "start_date": "2013-08-30T19:00:00.000Z",
            "completed": True,
            "neutral_site": False,
            "conference_game": False,
            "home_id": 99,
            "home_team": "FCS Home",
            "home_division": "fcs",
            "home_conference": "MEAC",
            "home_points": 21,
            "away_id": 98,
            "away_team": "FCS Away",
            "away_division": "fcs",
            "away_conference": "MEAC",
            "away_points": 14,
        }
    )
    rows.append(
        {
            "game_id": 20149901,
            "season": 2014,
            "week": 15,
            "season_type": "postseason",
            "start_date": "2014-12-20T19:00:00.000Z",
            "completed": True,
            "neutral_site": True,
            "conference_game": False,
            "home_id": 1,
            "home_team": "Team 1",
            "home_division": "fbs",
            "home_conference": "SEC",
            "home_points": 30,
            "away_id": 2,
            "away_team": "Team 2",
            "away_division": "fbs",
            "away_conference": "SEC",
            "away_points": 20,
        }
    )
    rows.append(
        {
            "game_id": 20149902,
            "season": 2014,
            "week": 14,
            "season_type": "regular",
            "start_date": "2014-12-06T19:00:00.000Z",
            "completed": False,
            "neutral_site": False,
            "conference_game": False,
            "home_id": 1,
            "home_team": "Team 1",
            "home_division": "fbs",
            "home_conference": "SEC",
            "home_points": 0,
            "away_id": 5,
            "away_team": "Team 5",
            "away_division": "fbs",
            "away_conference": "Mid-American",
            "away_points": 0,
        }
    )
    return rows


def _cfb_line_rows(schedule: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eligible = schedule.loc[
        schedule["season_type"].eq("regular")
        & schedule["completed"].astype(bool)
        & schedule["home_division"].eq("fbs")
        & schedule["away_division"].eq("fbs")
    ]
    for row in eligible.itertuples():
        game_id = int(row.game_id)
        if game_id == CFB_GAME_NO_LINES:
            continue
        home, away = int(row.home_id), int(row.away_id)
        home_abbr, away_abbr = f"T{home}", f"T{away}"
        if game_id == CFB_GAME_UNRESOLVED:
            home_abbr, away_abbr = "XU1", "XU2"
        if game_id == CFB_GAME_REPAIRED:
            home_abbr = f"Q{home}"
        spread = cfb_true_spread(home, away)
        for book, offset in (("B1", 0.5), ("B2", -0.5)):
            oriented = spread + offset
            opener = spread + 1.5 if book == "B1" else np.nan
            rows.append(
                {
                    "game_id": game_id,
                    "season": int(row.season),
                    "market_type": "spread",
                    "abbr": home_abbr,
                    "book": book,
                    "lines": -oriented,
                    "odds": -105.0,
                    "opening_lines": -opener if book == "B1" else np.nan,
                }
            )
            rows.append(
                {
                    "game_id": game_id,
                    "season": int(row.season),
                    "market_type": "spread",
                    "abbr": away_abbr,
                    "book": book,
                    "lines": oriented,
                    "odds": -115.0,
                    "opening_lines": opener if book == "B1" else np.nan,
                }
            )
        for book, total_offset in (("B1", 1.0), ("B2", -1.0)):
            for total_side in ("over", "under"):
                rows.append(
                    {
                        "game_id": game_id,
                        "season": int(row.season),
                        "market_type": "total",
                        "abbr": total_side,
                        "book": book,
                        "lines": 50.0 + total_offset,
                        "odds": -110.0,
                        "opening_lines": np.nan,
                    }
                )
    return rows


def _cfb_pbp_rows(schedule: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eligible = schedule.loc[
        schedule["season_type"].eq("regular")
        & schedule["completed"].astype(bool)
        & schedule["home_division"].eq("fbs")
        & schedule["away_division"].eq("fbs")
    ]
    for row in eligible.itertuples():
        game_id = int(row.game_id)
        if game_id == CFB_GAME_NO_PBP:
            continue
        home, away = int(row.home_id), int(row.away_id)
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            base = 0.2 * (CFB_FIXTURE_STRENGTH[team] - CFB_FIXTURE_STRENGTH[opponent])
            for play in range(8):
                epa = base + (play - 3.5) * 0.15
                home_wp = 0.97 if play == 1 else 0.5 + 0.01 * play
                rows.append(
                    {
                        "game_id": game_id,
                        "season": int(row.season),
                        "week": int(row.week),
                        "seasonType": 2,
                        "pos_team_id": team,
                        "homeTeamId": home,
                        "awayTeamId": away,
                        "is_home": is_home,
                        "EPA": epa,
                        "EPA_success": epa > 0,
                        "rush": play % 2 == 0,
                        "pass": play % 2 == 1,
                        "kneel_down": is_home and play == 0,
                        "statYardage": 25 if play == 7 else 12 if play == 6 else 4,
                        "home_wp_before": home_wp,
                        "away_wp_before": 1.0 - home_wp,
                    }
                )
    rows.append(
        {
            "game_id": 20149901,
            "season": 2014,
            "week": 15,
            "seasonType": 3,
            "pos_team_id": 1,
            "homeTeamId": 1,
            "awayTeamId": 2,
            "is_home": True,
            "EPA": 0.4,
            "EPA_success": True,
            "rush": True,
            "pass": False,
            "kneel_down": False,
            "statYardage": 6,
            "home_wp_before": 0.6,
            "away_wp_before": 0.4,
        }
    )
    return rows


@pytest.fixture(scope="session")
def _shared_cfb_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the deterministic CFB source fixture once per test session."""

    schedules = pd.DataFrame(_cfb_schedule_rows())
    lines = pd.DataFrame(_cfb_line_rows(schedules))
    pbp = pd.DataFrame(_cfb_pbp_rows(schedules))
    return schedules, lines, pbp


@pytest.fixture
def cfb_inputs(
    _shared_cfb_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return isolated copies so a test cannot mutate the shared templates."""

    schedules, lines, pbp = _shared_cfb_inputs
    return schedules.copy(deep=True), lines.copy(deep=True), pbp.copy(deep=True)


@pytest.fixture(scope="session")
def _shared_cfb_features_frame(
    _shared_cfb_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    from nfl_ats.cfb_features import build_cfb_game_features

    schedules, lines, pbp = _shared_cfb_inputs
    features, _ = build_cfb_game_features(schedules, lines, pbp, start_season=2013, end_season=2014)
    return features


@pytest.fixture
def cfb_features_frame(_shared_cfb_features_frame: pd.DataFrame) -> pd.DataFrame:
    """Return a deep copy of the session-cached deterministic feature table."""

    return _shared_cfb_features_frame.copy(deep=True)


@pytest.fixture
def schedules_and_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2022-09-11", periods=5, freq="7D")
    schedules = pd.DataFrame(
        {
            "game_id": [f"2022_{week:02d}_B_A" for week in range(1, 6)],
            "season": 2022,
            "game_type": "REG",
            "week": range(1, 6),
            "gameday": dates,
            "away_team": "B",
            "home_team": "A",
            "away_score": [17, 24, 14, 27, 20],
            "home_score": [20, 20, 21, 25, 20],
            "result": [3, -4, 7, -2, 0],
            "spread_line": [1.0, 2.0, 3.0, -3.0, 0.0],
            "total_line": [42.5, 44.0, 41.5, 47.0, 43.0],
            "home_rest": 7,
            "away_rest": 7,
            "location": "Home",
            "div_game": 1,
            "temp": 65.0,
            "wind": 8.0,
            "home_spread_odds": -110.0,
            "away_spread_odds": -110.0,
        }
    )
    rows: list[dict[str, object]] = []
    for game_index, game in schedules.iterrows():
        for team, direction in (("A", 1.0), ("B", -1.0)):
            rows.append(
                {
                    "game_id": game["game_id"],
                    "season": 2022,
                    "season_type": "REG",
                    "week": game["week"],
                    "team": team,
                    "attempts": 30 + game_index,
                    "carries": 25 - game_index,
                    "passing_epa": direction * (3.0 + game_index),
                    "rushing_epa": direction * (1.0 + game_index / 2),
                    "passing_cpoe": direction * (2.0 + game_index),
                    "passing_yards": 220 + direction * 10 + game_index,
                    "rushing_yards": 105 + direction * 5 + game_index,
                    "sacks_suffered": 2,
                    "passing_interceptions": int(direction < 0),
                    "sack_fumbles_lost": 0,
                    "rushing_fumbles_lost": 0,
                    "receiving_fumbles_lost": 0,
                }
            )
    return schedules, pd.DataFrame(rows)


@pytest.fixture
def model_frame() -> pd.DataFrame:
    rows = 160
    start = date(2018, 9, 1)
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "game_id": [f"game_{value:03d}" for value in index],
            "season": np.where(index < 100, 2019, 2020),
            "week": np.where(index < 100, (index // 10) + 1, ((index - 100) // 15) + 1),
            "gameday": [start + timedelta(days=int(value)) for value in index],
            "away_team": "AWY",
            "home_team": "HME",
            "home_spread_odds": -110.0,
            "away_spread_odds": -110.0,
        }
    )
    all_features = (*MODEL_FEATURE_COLUMNS, *GRAPH_FEATURE_COLUMNS)
    for feature_index, column in enumerate(all_features, start=1):
        frame[column] = np.sin(index / feature_index) + (index % 5) / 10.0
    frame["spread_line"] = np.where(index % 2 == 0, 2.5, -2.5)
    frame["home_cover"] = (index % 3 != 0).astype(float)
    frame["ats_margin"] = np.where(frame["home_cover"].eq(1), 3.0, -3.0)
    frame["result"] = frame["spread_line"] + frame["ats_margin"]
    return frame
