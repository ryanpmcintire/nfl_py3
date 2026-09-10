from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from _board_content_fixtures import build_fixture_content

from nfl_ats import board_site_content as bsc
from nfl_ats import board_terminal
from nfl_ats.board_site_content import (
    ARCHIVE_CLOSE_CORRECT_COLUMN,
    ARCHIVE_OPENER_CORRECT_COLUMN,
    HISTORY_GRADE_CAPTION,
    HISTORY_WEEK_REPLAY_CAPTION,
    HistoryPageContent,
    HistoryWeekGrade,
)
from nfl_ats.public_board import find_matching_opener_evaluation

_REPO_ROOT = Path(__file__).resolve().parents[1]

_ACTIVE = {
    "model_id": "abc123",
    "feature_table_sha256": "f" * 64,
    "feature_profile": "weak_stack",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "method": "market_residual",
    "probability_method": "gaussian_median",
    "calibration_method": "none",
}


def _write_opener_evaluation(
    artifacts_root: Path,
    per_game: pd.DataFrame,
    *,
    stamp: str = "20260908T184350Z",
    feature_table_sha256: str = "f" * 64,
    model_id: str = "abc123",
) -> Path:

    directory = artifacts_root / "opener_evaluation" / stamp
    directory.mkdir(parents=True)
    metadata = {
        "active_model_id": model_id,
        "active_model_config": {
            "model_id": model_id,
            "feature_profile": "weak_stack",
            "regressor": "ridge",
            "ridge_alpha": 10.0,
            "target": "market_residual",
            "probability_method": "gaussian_median",
            "calibration_method": "none",
        },
        "provenance": {"feature_table": {"sha256": feature_table_sha256}},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    per_game.to_parquet(directory / "per_game.parquet")
    return directory


def _archive_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _game(season: int, week: int, opener: float | None, close: float | None) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        ARCHIVE_OPENER_CORRECT_COLUMN: opener,
        ARCHIVE_CLOSE_CORRECT_COLUMN: close,
    }


def test_week_counter_reports_both_records_and_their_difference() -> None:
    graded = _archive_frame(
        [
            _game(2025, 4, 1.0, 1.0),
            _game(2025, 4, 1.0, 0.0),
            _game(2025, 4, 0.0, 0.0),
            _game(2025, 4, 0.0, 0.0),
        ]
    )
    (row,) = bsc._week_grades_from_graded_games(
        graded,
        opener_column=ARCHIVE_OPENER_CORRECT_COLUMN,
        close_column=ARCHIVE_CLOSE_CORRECT_COLUMN,
    )
    assert (row.season, row.week, row.picks) == (2025, 4, 4)
    assert row.opener_record_text == "2-2 (50.0%)"
    assert row.close_record_text == "1-3 (25.0%)"
    assert row.delta_text == "+25.0%"
    assert row.note == ""


def test_week_counter_leaves_pushes_out_of_both_records() -> None:

    graded = _archive_frame(
        [
            _game(2025, 5, 1.0, 1.0),
            _game(2025, 5, None, 0.0),
            _game(2025, 5, 0.0, None),
        ]
    )
    (row,) = bsc._week_grades_from_graded_games(
        graded,
        opener_column=ARCHIVE_OPENER_CORRECT_COLUMN,
        close_column=ARCHIVE_CLOSE_CORRECT_COLUMN,
    )
    assert row.picks == 3
    assert (row.opener_settled, row.opener_wins) == (2, 1)
    assert (row.close_settled, row.close_wins) == (2, 1)
    assert row.opener_record_text == "1-1 (50.0%)"


def test_week_counter_says_so_when_a_whole_week_has_no_close_line() -> None:
    graded = _archive_frame([_game(2024, 9, 1.0, None), _game(2024, 9, 0.0, None)])
    (row,) = bsc._week_grades_from_graded_games(
        graded,
        opener_column=ARCHIVE_OPENER_CORRECT_COLUMN,
        close_column=ARCHIVE_CLOSE_CORRECT_COLUMN,
    )
    assert row.close_accuracy is None
    assert row.note == bsc.NO_CLOSE_LINE_ARCHIVED_WEEK_NOTE


def test_week_counter_ignores_a_table_without_the_grade_columns() -> None:
    graded = pd.DataFrame([{"season": 2025, "week": 1, "something_else": 1.0}])
    assert (
        bsc._week_grades_from_graded_games(
            graded,
            opener_column=ARCHIVE_OPENER_CORRECT_COLUMN,
            close_column=ARCHIVE_CLOSE_CORRECT_COLUMN,
        )
        == ()
    )


def test_archive_week_grades_reads_every_finished_week(tmp_path: Path) -> None:
    _write_opener_evaluation(
        tmp_path,
        _archive_frame(
            [
                _game(2024, 1, 1.0, 0.0),
                _game(2024, 1, 1.0, 1.0),
                _game(2025, 2, 0.0, 0.0),
            ]
        ),
    )
    rows = bsc._archive_week_grades(tmp_path, _ACTIVE)
    assert [(row.season, row.week, row.picks) for row in rows] == [(2024, 1, 2), (2025, 2, 1)]
    assert rows[0].opener_record_text == "2-0 (100.0%)"
    assert rows[0].close_record_text == "1-1 (50.0%)"


def test_archive_week_grades_refuses_another_models_run(tmp_path: Path) -> None:

    _write_opener_evaluation(
        tmp_path,
        _archive_frame([_game(2024, 1, 1.0, 1.0)]),
        feature_table_sha256="a" * 64,
    )
    assert find_matching_opener_evaluation(tmp_path, _ACTIVE) is None
    assert bsc._archive_week_grades(tmp_path, _ACTIVE) == ()


def test_archive_week_grades_degrades_when_the_run_has_no_per_game_table(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "opener_evaluation" / "20260908T184350Z"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "active_model_id": "abc123",
                "active_model_config": {
                    "model_id": "abc123",
                    "feature_profile": "weak_stack",
                    "regressor": "ridge",
                    "ridge_alpha": 10.0,
                    "target": "market_residual",
                    "probability_method": "gaussian_median",
                    "calibration_method": "none",
                },
                "provenance": {"feature_table": {"sha256": "f" * 64}},
            }
        ),
        encoding="utf-8",
    )
    assert bsc._archive_week_grades(tmp_path, _ACTIVE) == ()


def test_archive_week_grades_empty_without_an_active_model(tmp_path: Path) -> None:
    assert bsc._archive_week_grades(tmp_path, {}) == ()


def _grade(season: int, week: int, opener_wins: int, settled: int) -> HistoryWeekGrade:
    return HistoryWeekGrade(
        season=season,
        week=week,
        picks=settled,
        opener_settled=settled,
        opener_wins=opener_wins,
        opener_accuracy=opener_wins / settled,
        close_settled=settled,
        close_wins=opener_wins,
        close_accuracy=opener_wins / settled,
    )


def test_combined_week_grades_puts_the_newest_week_first() -> None:
    recorded = (_grade(2026, 1, 9, 16),)
    archived = (_grade(2024, 3, 5, 10), _grade(2025, 18, 6, 10), _grade(2025, 2, 4, 10))
    rows = bsc._combined_week_grades(recorded, archived)
    assert [(row.season, row.week) for row in rows] == [(2026, 1), (2025, 18), (2025, 2), (2024, 3)]


def test_combined_week_grades_lets_the_recorded_week_win_its_slot() -> None:

    recorded = (_grade(2026, 1, 12, 16),)
    archived = (_grade(2026, 1, 3, 16),)
    (row,) = bsc._combined_week_grades(recorded, archived)
    assert row.opener_wins == 12


def test_combined_week_grades_with_nothing_recorded_yet() -> None:
    archived = (_grade(2025, 1, 5, 10),)
    assert bsc._combined_week_grades((), archived) == archived


def _history_content(week_grades: tuple[HistoryWeekGrade, ...], caption: str) -> HistoryPageContent:
    board = build_fixture_content()
    return HistoryPageContent(
        generated_at_text="2026-09-08 12:00:00 UTC",
        picks=(),
        primary_available=False,
        primary_error=None,
        challenger_assessments=(),
        ticker_chrome=board.ticker_chrome,
        link_preview=board.link_preview,
        week_grades=week_grades,
        grade_caption=caption,
    )


def test_history_page_shows_both_records_week_by_week() -> None:
    rows = (
        _grade(2026, 1, 9, 16),
        HistoryWeekGrade(
            season=2025,
            week=22,
            picks=1,
            opener_settled=1,
            opener_wins=1,
            opener_accuracy=1.0,
            close_settled=1,
            close_wins=0,
            close_accuracy=0.0,
        ),
    )
    caption = f"{HISTORY_GRADE_CAPTION} {HISTORY_WEEK_REPLAY_CAPTION}"
    html = board_terminal.render_history_page(_history_content(rows, caption))
    assert "Opener vs close, side by side" in html
    assert "2026 / W1" in html
    assert "2025 / W22" in html
    assert "1-0 (100.0%)" in html
    assert "0-1 (0.0%)" in html
    assert "+100.0%" in html
    assert "run the model again on the opening" in html


def test_the_week_caption_stays_in_pool_player_words() -> None:

    text = HISTORY_WEEK_REPLAY_CAPTION
    assert not re.search(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text)
    assert not re.search(r"[0-9a-f]{8,}", text)
    for token in ("P+", "week-blocked", "raw model", "probability rule", "opener-graded"):
        assert token not in text


def _real_active() -> dict[str, object] | None:
    path = _REPO_ROOT / "artifacts" / "active_ats_model.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def test_archived_weeks_add_up_to_the_season_row_above_them() -> None:

    active = _real_active()
    if active is None:
        pytest.skip("no active model artifact in this clone")
    match = find_matching_opener_evaluation(_REPO_ROOT / "artifacts", active)
    if match is None or not (match[1] / "season_summary.csv").is_file():
        pytest.skip("no opener evaluation matches the active model in this clone")
    seasons = pd.read_csv(match[1] / "season_summary.csv")
    if "opener_accuracy_probability_rule" not in seasons.columns:
        pytest.skip("this evaluation predates the served-pick season columns")

    rows = bsc._archive_week_grades(_REPO_ROOT / "artifacts", active)
    assert rows, "the matched evaluation should grade at least one week"
    for _, season_row in seasons.iterrows():
        season = int(str(season_row["season"]).split(".")[0])
        weeks = [row for row in rows if row.season == season]
        assert weeks, f"no week rows for archived season {season}"
        opener_wins = sum(row.opener_wins for row in weeks)
        opener_settled = sum(row.opener_settled for row in weeks)
        close_wins = sum(row.close_wins for row in weeks)
        close_settled = sum(row.close_settled for row in weeks)
        assert opener_wins / opener_settled == pytest.approx(
            float(season_row["opener_accuracy_probability_rule"])
        )
        assert close_wins / close_settled == pytest.approx(
            float(season_row["close_accuracy_probability_rule"])
        )
        assert sum(row.picks for row in weeks) == int(season_row["games"])
