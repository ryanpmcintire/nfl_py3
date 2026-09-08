"""Contract tests for the Splash Sports board capture (the lines the pool grades).

The whole point of ``nfl_ats.splash_lines`` is to fail closed, so most of these
tests assert that a malformed capture raises rather than that a good one loads.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_ats.data import DataContractError
from nfl_ats.splash_lines import (
    SPLASH_SUBDIRECTORY,
    SplashCapture,
    SplashGame,
    build_game_id,
    capture_age,
    default_picks_lock_et,
    is_half_point,
    is_stale,
    load_splash_capture,
    normalize_team,
    parse_splash_board,
    picks_locked,
    splash_capture_paths,
    splash_decision_lines,
    validate_splash_capture,
)
from scripts import capture_splash_lines

ET = ZoneInfo("America/New_York")
REPO = Path(__file__).resolve().parents[1]

#: Byte-for-byte copy of the first real capture,
#: ``data/splash/2026_week01_20260908_noon.json``, hand-read off the board by
#: an agent at the 2026-09-08 noon lock. Kept under ``tests/fixtures`` so these
#: tests hold in a fresh clone, where ``data/`` is empty.
WEEK1_CAPTURE_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "splash" / "2026_week01_20260908_noon.json"
)

#: A browser read of the same Week 1 board, in the repeated block shape the
#: page renders. Two abbreviations here are Splash's rather than nflverse's
#: ("JAC" for Jacksonville, "LAR" for the Rams) so the parser's alias fold is
#: exercised by the realistic text, not only by a unit test.
WEEK1_BOARD_TEXT = """
NFL Week 1 - Winner (ATS)

NE   Wed, Sep 9 8:20 PM   SEA
Winner (ATS)
Patriots     NE +3.5
Seahawks     SEA -3.5

SF   Thu, Sep 10 8:35 PM   LAR
Winner (ATS)
49ers        SF +3.5
Rams         LAR -3.5

CHI   Sun, Sep 13 1:00 PM   CAR
Winner (ATS)
Bears        CHI -2.5
Panthers     CAR +2.5

CLE   Sun, Sep 13 1:00 PM   JAC
Winner (ATS)
Browns       CLE +8.5
Jaguars      JAC -8.5

NO   Sun, Sep 13 1:00 PM   DET
Winner (ATS)
Saints       NO +6.5
Lions        DET -6.5

ATL   Sun, Sep 13 1:00 PM   PIT
Winner (ATS)
Falcons      ATL +3.5
Steelers     PIT -3.5

NYJ   Sun, Sep 13 1:00 PM   TEN
Winner (ATS)
Jets         NYJ +1.5
Titans       TEN -1.5

TB   Sun, Sep 13 1:00 PM   CIN
Winner (ATS)
Buccaneers   TB +3.5
Bengals      CIN -3.5

BAL   Sun, Sep 13 1:00 PM   IND
Winner (ATS)
Ravens       BAL -3.5
Colts        IND +3.5

BUF   Sun, Sep 13 1:00 PM   HOU
Winner (ATS)
Bills        BUF -1.5
Texans       HOU +1.5

GB   Sun, Sep 13 4:25 PM   MIN
Winner (ATS)
Packers      GB +1.5
Vikings      MIN -1.5

ARI   Sun, Sep 13 4:25 PM   LAC
Winner (ATS)
Cardinals    ARI +9.5
Chargers     LAC -9.5

MIA   Sun, Sep 13 4:25 PM   LV
Winner (ATS)
Dolphins     MIA +3.5
Raiders      LV -3.5

WAS   Sun, Sep 13 4:25 PM   PHI
Winner (ATS)
Commanders   WAS +5.5
Eagles       PHI -5.5

DAL   Sun, Sep 13 8:20 PM   NYG
Winner (ATS)
Cowboys      DAL -2.5
Giants       NYG +2.5

DEN   Mon, Sep 14 8:15 PM   KC
Winner (ATS)
Broncos      DEN +2.5
Chiefs       KC -2.5
"""

ONE_GAME_BOARD = """
CHI   Sun, Sep 13 1:00 PM   CAR
Winner (ATS)
Bears        CHI -2.5
Panthers     CAR +2.5
"""


def _write_capture(root: Path, name: str, payload: dict[str, object]) -> Path:
    directory = root / SPLASH_SUBDIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _week1_payload() -> dict[str, object]:
    payload = json.loads(WEEK1_CAPTURE_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sample_capture(**overrides: object) -> SplashCapture:
    game = SplashGame(
        game_id="2026_01_NE_SEA",
        away="NE",
        home="SEA",
        away_line=3.5,
        home_spread=3.5,
        kickoff_et=datetime(2026, 9, 9, 20, 20, tzinfo=ET),
    )
    base: dict[str, object] = {
        "season": 2026,
        "week": 1,
        "captured_at_et": datetime(2026, 9, 8, 12, 45, tzinfo=ET),
        "games": (game,),
    }
    base.update(overrides)
    return SplashCapture(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Load / round trip
# ---------------------------------------------------------------------------


def test_real_week1_capture_round_trips(tmp_path: Path) -> None:
    payload = _week1_payload()
    _write_capture(tmp_path, "2026_week01_20260908_noon.json", payload)

    capture = load_splash_capture(tmp_path, 2026, 1)

    assert capture is not None
    assert len(capture.games) == 16
    assert capture.season == 2026 and capture.week == 1
    assert capture.captured_at_et == datetime(2026, 9, 8, 12, 45, tzinfo=ET)
    assert capture.picks_lock_et == datetime(2026, 9, 13, 16, 0, tzinfo=ET)
    assert capture.contest is not None and capture.contest["channel"] == "EXAMPLE-0000"
    # The public fixture carries no contest identifiers and no submitted entry.
    assert capture.submitted_entry is None
    # The schema survives a full round trip, key for key and value for value.
    assert capture.to_dict() == payload


def test_every_real_week1_line_is_a_half_point(tmp_path: Path) -> None:
    _write_capture(tmp_path, "2026_week01_20260908_noon.json", _week1_payload())
    capture = load_splash_capture(tmp_path, 2026, 1)
    assert capture is not None
    assert all(is_half_point(game.home_spread) for game in capture.games)


def test_decision_lines_are_home_spreads(tmp_path: Path) -> None:
    _write_capture(tmp_path, "2026_week01_20260908_noon.json", _week1_payload())
    capture = load_splash_capture(tmp_path, 2026, 1)
    assert capture is not None

    lines = splash_decision_lines(capture)

    assert len(lines) == 16
    assert lines["2026_01_NE_SEA"] == 3.5  # SEA favored by 3.5
    assert lines["2026_01_CHI_CAR"] == -2.5  # CHI favored, so the home number is negative
    assert lines["2026_01_ARI_LAC"] == 9.5


def test_absent_capture_returns_none(tmp_path: Path) -> None:
    assert load_splash_capture(tmp_path, 2026, 1) is None
    assert splash_capture_paths(tmp_path, 2026, 1) == []

    (tmp_path / SPLASH_SUBDIRECTORY).mkdir()
    assert load_splash_capture(tmp_path, 2026, 1) is None
    assert load_splash_capture(tmp_path, 2026, 7) is None


def test_newest_capture_wins_by_capture_time_not_filename(tmp_path: Path) -> None:
    early = _week1_payload()
    early["captured_at_et"] = "2026-09-08T09:05:00-04:00"

    late = _week1_payload()
    late["captured_at_et"] = "2026-09-08T18:00:00-04:00"
    games = late["games"]
    assert isinstance(games, list)
    games[0]["home_spread"] = 4.5
    games[0]["away_line"] = 4.5

    # Filenames deliberately sort the newer capture FIRST, so a lexicographic
    # "newest" would return the stale board.
    _write_capture(tmp_path, "2026_week01_20260908_aaa.json", late)
    _write_capture(tmp_path, "2026_week01_20260908_zzz.json", early)

    capture = load_splash_capture(tmp_path, 2026, 1)

    assert capture is not None
    assert capture.captured_at_et == datetime(2026, 9, 8, 18, 0, tzinfo=ET)
    assert splash_decision_lines(capture)["2026_01_NE_SEA"] == 4.5


def test_capture_for_a_different_week_is_refused(tmp_path: Path) -> None:
    payload = _week1_payload()
    payload["week"] = 2
    _write_capture(tmp_path, "2026_week01_20260908_noon.json", payload)

    with pytest.raises(DataContractError, match="week 2, not the requested 1"):
        load_splash_capture(tmp_path, 2026, 1)


def test_malformed_json_raises(tmp_path: Path) -> None:
    directory = tmp_path / SPLASH_SUBDIRECTORY
    directory.mkdir(parents=True)
    (directory / "2026_week01_20260908_noon.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(DataContractError, match="not valid JSON"):
        load_splash_capture(tmp_path, 2026, 1)


def test_missing_required_key_raises(tmp_path: Path) -> None:
    payload = _week1_payload()
    del payload["games"]
    _write_capture(tmp_path, "2026_week01_20260908_noon.json", payload)

    with pytest.raises(DataContractError, match="missing required key 'games'"):
        load_splash_capture(tmp_path, 2026, 1)


def test_missing_game_field_raises(tmp_path: Path) -> None:
    payload = _week1_payload()
    games = payload["games"]
    assert isinstance(games, list)
    del games[3]["home_spread"]
    _write_capture(tmp_path, "2026_week01_20260908_noon.json", payload)

    with pytest.raises(DataContractError, match="missing required keys: home_spread"):
        load_splash_capture(tmp_path, 2026, 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_is_half_point() -> None:
    assert is_half_point(3.5)
    assert is_half_point(-2.5)
    assert is_half_point(0.5)
    assert is_half_point(13.5)
    assert not is_half_point(3.0)
    assert not is_half_point(3)
    assert not is_half_point(0.0)
    assert not is_half_point(2.25)
    assert not is_half_point(-7.0)


def test_whole_number_line_is_a_contract_violation() -> None:
    capture = _sample_capture(
        games=(
            SplashGame(
                game_id="2026_01_NE_SEA",
                away="NE",
                home="SEA",
                away_line=3.0,
                home_spread=3.0,
                kickoff_et=datetime(2026, 9, 9, 20, 20, tzinfo=ET),
            ),
        )
    )

    with pytest.raises(DataContractError) as error:
        validate_splash_capture(capture)

    message = str(error.value)
    assert "2026_01_NE_SEA home_spread is 3.0" in message
    assert "not a half point" in message
    assert "no push is possible" in message
    assert "silently re-enable push machinery that cannot apply" in message


def test_away_line_must_agree_with_home_spread() -> None:
    capture = _sample_capture(
        games=(
            SplashGame(
                game_id="2026_01_NE_SEA",
                away="NE",
                home="SEA",
                away_line=-3.5,
                home_spread=3.5,
                kickoff_et=datetime(2026, 9, 9, 20, 20, tzinfo=ET),
            ),
        )
    )

    with pytest.raises(DataContractError, match="disagrees with home_spread"):
        validate_splash_capture(capture)


def test_game_id_must_match_season_week_and_teams() -> None:
    capture = _sample_capture(
        games=(
            SplashGame(
                game_id="2026_02_NE_SEA",
                away="NE",
                home="SEA",
                away_line=3.5,
                home_spread=3.5,
                kickoff_et=datetime(2026, 9, 9, 20, 20, tzinfo=ET),
            ),
        )
    )

    with pytest.raises(DataContractError, match="expected '2026_01_NE_SEA'"):
        validate_splash_capture(capture)


def test_reversed_teams_in_game_id_are_refused() -> None:
    capture = _sample_capture(
        games=(
            SplashGame(
                game_id="2026_01_SEA_NE",
                away="NE",
                home="SEA",
                away_line=3.5,
                home_spread=3.5,
                kickoff_et=datetime(2026, 9, 9, 20, 20, tzinfo=ET),
            ),
        )
    )

    with pytest.raises(DataContractError, match="does not match the nflverse form"):
        validate_splash_capture(capture)


def test_duplicate_game_ids_are_refused() -> None:
    game = SplashGame(
        game_id="2026_01_NE_SEA",
        away="NE",
        home="SEA",
        away_line=3.5,
        home_spread=3.5,
        kickoff_et=datetime(2026, 9, 9, 20, 20, tzinfo=ET),
    )
    capture = _sample_capture(games=(game, game))

    with pytest.raises(DataContractError, match="duplicate game_id"):
        validate_splash_capture(capture)


def test_empty_capture_is_refused() -> None:
    with pytest.raises(DataContractError, match="holds no games"):
        validate_splash_capture(_sample_capture(games=()))


def test_absurd_magnitude_is_refused_as_a_misread() -> None:
    capture = _sample_capture(
        games=(
            SplashGame(
                game_id="2026_01_NE_SEA",
                away="NE",
                home="SEA",
                away_line=93.5,
                home_spread=93.5,
                kickoff_et=datetime(2026, 9, 9, 20, 20, tzinfo=ET),
            ),
        )
    )

    with pytest.raises(DataContractError, match="sanity bound"):
        validate_splash_capture(capture)


def test_unknown_team_abbreviation_is_refused() -> None:
    with pytest.raises(DataContractError, match="not an NFL team abbreviation"):
        normalize_team("XYZ", context="test")


def test_board_abbreviations_fold_onto_nflverse_identity() -> None:
    assert normalize_team("JAC", context="test") == "JAX"
    assert normalize_team("LAR", context="test") == "LA"
    assert normalize_team("WSH", context="test") == "WAS"
    assert normalize_team("sea", context="test") == "SEA"


def test_build_game_id_zero_pads_the_week() -> None:
    assert build_game_id(2026, 1, "NE", "SEA") == "2026_01_NE_SEA"
    assert build_game_id(2026, 12, "NE", "SEA") == "2026_12_NE_SEA"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parses_the_real_week1_board_text() -> None:
    games = parse_splash_board(WEEK1_BOARD_TEXT, 2026, 1)

    expected = _week1_payload()["games"]
    assert isinstance(expected, list)
    assert len(games) == len(expected)
    for parsed, row in zip(games, expected, strict=True):
        assert parsed.game_id == row["game_id"]
        assert parsed.away == row["away"]
        assert parsed.home == row["home"]
        assert parsed.away_line == row["away_line"]
        assert parsed.home_spread == row["home_spread"]
        assert parsed.kickoff_et == datetime.fromisoformat(str(row["kickoff_et"]))


def test_parser_folds_splash_abbreviations() -> None:
    games = {game.game_id: game for game in parse_splash_board(WEEK1_BOARD_TEXT, 2026, 1)}

    assert "2026_01_CLE_JAX" in games  # board printed JAC
    assert "2026_01_SF_LA" in games  # board printed LAR
    assert games["2026_01_CLE_JAX"].home == "JAX"
    assert games["2026_01_SF_LA"].home == "LA"


def test_parser_orients_the_home_side() -> None:
    (game,) = parse_splash_board(ONE_GAME_BOARD, 2026, 1)

    # Board printed "CHI -2.5 / CAR +2.5": the road team is favored, so the
    # home spread is negative under this repository's convention.
    assert game.home_spread == -2.5
    assert game.away_line == -2.5
    assert game.kickoff_et == datetime(2026, 9, 13, 13, 0, tzinfo=ET)


def test_parser_rejects_a_whole_number_line() -> None:
    text = ONE_GAME_BOARD.replace("-2.5", "-2").replace("+2.5", "+2")

    with pytest.raises(DataContractError, match="not a half point"):
        parse_splash_board(text, 2026, 1)


def test_parser_rejects_a_block_with_one_option() -> None:
    text = "\n".join(ONE_GAME_BOARD.strip().splitlines()[:-1])

    with pytest.raises(DataContractError, match="has 1 spread option"):
        parse_splash_board(text, 2026, 1)


def test_parser_rejects_sides_that_are_not_opposites() -> None:
    text = ONE_GAME_BOARD.replace("CAR +2.5", "CAR +3.5")

    with pytest.raises(DataContractError, match="must be exact opposites"):
        parse_splash_board(text, 2026, 1)


def test_parser_rejects_an_option_naming_a_team_not_in_the_matchup() -> None:
    text = ONE_GAME_BOARD.replace("Panthers     CAR +2.5", "Packers      GB +2.5")

    with pytest.raises(DataContractError, match="does not match the matchup header"):
        parse_splash_board(text, 2026, 1)


def test_parser_rejects_an_unknown_abbreviation() -> None:
    text = ONE_GAME_BOARD.replace("CAR", "ZZZ")

    with pytest.raises(DataContractError, match="not an NFL team abbreviation"):
        parse_splash_board(text, 2026, 1)


def test_parser_rejects_text_with_no_matchup_header() -> None:
    with pytest.raises(DataContractError, match="no matchup headers found"):
        parse_splash_board("Winner (ATS)\nno games here\n", 2026, 1)


def test_parser_rejects_an_option_row_before_any_header() -> None:
    text = "Bears        CHI -2.5\n" + ONE_GAME_BOARD

    with pytest.raises(DataContractError, match="before any matchup header"):
        parse_splash_board(text, 2026, 1)


def test_parser_rejects_a_weekday_that_contradicts_the_date() -> None:
    text = ONE_GAME_BOARD.replace("Sun, Sep 13", "Mon, Sep 13")

    with pytest.raises(DataContractError, match="the board says Mon"):
        parse_splash_board(text, 2026, 1)


def test_parser_puts_january_games_in_the_following_calendar_year() -> None:
    text = ONE_GAME_BOARD.replace("Sun, Sep 13 1:00 PM", "Sun, Jan 3 1:00 PM")

    (game,) = parse_splash_board(text, 2026, 18)

    assert game.kickoff_et == datetime(2027, 1, 3, 13, 0, tzinfo=ET)
    assert game.game_id == "2026_18_CHI_CAR"


def test_parser_ignores_page_chrome() -> None:
    noisy = "Entries: 250\nSpreads lock Tue 12:00 PM\n" + ONE_GAME_BOARD + "\nBest Pick\n"

    assert len(parse_splash_board(noisy, 2026, 1)) == 1


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def test_capture_age_and_staleness() -> None:
    capture = _sample_capture()
    captured = datetime(2026, 9, 8, 12, 45, tzinfo=ET)

    assert capture_age(capture, captured + timedelta(hours=3)) == timedelta(hours=3)
    assert not is_stale(capture, captured + timedelta(days=6, hours=23))
    assert is_stale(capture, captured + timedelta(days=7, seconds=1))
    # A caller may tighten the horizon; the same-day capture is fresh, the
    # previous board is not.
    assert not is_stale(capture, captured + timedelta(hours=2), max_age=timedelta(days=1))
    assert is_stale(capture, captured + timedelta(days=2), max_age=timedelta(days=1))


def test_freshness_helpers_require_a_timezone() -> None:
    capture = _sample_capture()
    with pytest.raises(ValueError, match="timezone-aware"):
        capture_age(capture, datetime(2026, 9, 8, 12, 45))
    with pytest.raises(ValueError, match="timezone-aware"):
        picks_locked(capture, datetime(2026, 9, 8, 12, 45))


def test_picks_locked() -> None:
    capture = _sample_capture(picks_lock_et=datetime(2026, 9, 13, 16, 0, tzinfo=ET))

    assert picks_locked(capture, datetime(2026, 9, 13, 15, 59, tzinfo=ET)) is False
    assert picks_locked(capture, datetime(2026, 9, 13, 16, 0, tzinfo=ET)) is True
    assert picks_locked(_sample_capture(), datetime(2026, 9, 13, 16, 0, tzinfo=ET)) is None


def test_default_picks_lock_is_the_slate_sunday_at_four() -> None:
    games = parse_splash_board(WEEK1_BOARD_TEXT, 2026, 1)

    assert default_picks_lock_et(games) == datetime(2026, 9, 13, 16, 0, tzinfo=ET)
    assert default_picks_lock_et(()) is None


# ---------------------------------------------------------------------------
# Capture CLI
# ---------------------------------------------------------------------------


def test_cli_writes_a_validated_capture(tmp_path: Path) -> None:
    board = tmp_path / "board.txt"
    board.write_text(WEEK1_BOARD_TEXT, encoding="utf-8")

    exit_code = capture_splash_lines.main(
        [
            "--season",
            "2026",
            "--week",
            "1",
            "--text-file",
            str(board),
            "--data-root",
            str(tmp_path / "data"),
            "--captured-at",
            "2026-09-08T12:45:00",
            "--contest-id",
            "contest_EXAMPLE",
            "--slate-id",
            "slate_EXAMPLE",
            "--entries",
            "250",
        ]
    )

    assert exit_code == 0
    written = splash_capture_paths(tmp_path / "data", 2026, 1)
    assert [path.name for path in written] == ["2026_week01_20260908_noon.json"]

    capture = load_splash_capture(tmp_path / "data", 2026, 1)
    assert capture is not None
    assert capture.contest is not None and capture.contest["entries"] == 250
    assert capture.picks_lock_et == datetime(2026, 9, 13, 16, 0, tzinfo=ET)
    assert splash_decision_lines(capture) == splash_decision_lines(
        SplashCapture(
            season=2026,
            week=1,
            captured_at_et=datetime(2026, 9, 8, 12, 45, tzinfo=ET),
            games=parse_splash_board(WEEK1_BOARD_TEXT, 2026, 1),
        )
    )


def test_cli_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    board = tmp_path / "board.txt"
    board.write_text(ONE_GAME_BOARD, encoding="utf-8")

    exit_code = capture_splash_lines.main(
        [
            "--season",
            "2026",
            "--week",
            "1",
            "--text-file",
            str(board),
            "--data-root",
            str(tmp_path / "data"),
            "--dry",
        ]
    )

    assert exit_code == 0
    assert splash_capture_paths(tmp_path / "data", 2026, 1) == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["games"][0]["game_id"] == "2026_01_CHI_CAR"


def test_cli_refuses_to_overwrite_without_replace(tmp_path: Path) -> None:
    board = tmp_path / "board.txt"
    board.write_text(ONE_GAME_BOARD, encoding="utf-8")
    data_root = tmp_path / "data"
    _write_capture(data_root, "2026_week01_20260908_noon.json", _week1_payload())

    argv = [
        "--season",
        "2026",
        "--week",
        "1",
        "--text-file",
        str(board),
        "--data-root",
        str(data_root),
        "--captured-at",
        "2026-09-09T09:00:00",
    ]

    assert capture_splash_lines.main(argv) == 2
    assert len(splash_capture_paths(data_root, 2026, 1)) == 1

    assert capture_splash_lines.main([*argv, "--replace"]) == 0
    assert len(splash_capture_paths(data_root, 2026, 1)) == 2


def test_cli_refuses_an_unparseable_board(tmp_path: Path) -> None:
    board = tmp_path / "board.txt"
    board.write_text("CHI   Sun, Sep 13 1:00 PM   CAR\nBears   CHI -2\nPanthers   CAR +2\n")

    exit_code = capture_splash_lines.main(
        [
            "--season",
            "2026",
            "--week",
            "1",
            "--text-file",
            str(board),
            "--data-root",
            str(tmp_path / "data"),
        ]
    )

    assert exit_code == 2
    assert splash_capture_paths(tmp_path / "data", 2026, 1) == []


def test_cli_label_defaults() -> None:
    noon = datetime(2026, 9, 8, 12, 45, tzinfo=ET)
    morning = datetime(2026, 9, 8, 9, 5, tzinfo=ET)

    assert capture_splash_lines.capture_label(noon) == "noon"
    assert capture_splash_lines.capture_label(morning) == "0905"
    assert (
        capture_splash_lines.capture_filename(2026, 1, noon, "noon")
        == "2026_week01_20260908_noon.json"
    )


# ---------------------------------------------------------------------------
# The real file on disk (skipped in a fresh clone, where data/ is empty)
# ---------------------------------------------------------------------------


@pytest.mark.full  # reads the real on-disk capture rather than a fixture
@pytest.mark.skipif(
    not (REPO / "data" / SPLASH_SUBDIRECTORY).is_dir(),
    reason="no local Splash captures (data/ is not in version control)",
)
def test_on_disk_captures_satisfy_the_contract() -> None:
    directory = REPO / "data" / SPLASH_SUBDIRECTORY
    files = sorted(directory.glob("*.json"))
    if not files:
        pytest.skip("no local Splash captures")

    for path in files:
        season, week_token = path.stem.split("_", 2)[:2]
        capture = load_splash_capture(REPO / "data", int(season), int(week_token[4:]))
        assert capture is not None
        assert capture.games
        assert all(is_half_point(game.home_spread) for game in capture.games)
