"""Unit tests for :mod:`nfl_ats.board_site_content`'s new pure helpers
(owner-approved improvement batch):

* item 2 -- :func:`nfl_ats.board_site_content._finding_trace`;
* item 9 -- :func:`nfl_ats.board_site_content._evidence_strength` and
  :func:`nfl_ats.board_site_content._grouped_ledger_rows`.

These sit alongside real-artifact integration coverage in
``tests/test_board_improvements.py``; this file exercises the exact
tie-break/ordering rules with small, synthetic inputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_ats import board_site_content as bsc
from nfl_ats.board_site_content import ModelLedgerRowView
from nfl_ats.dashboard.findings_content import Finding
from nfl_ats.findings_registry import RegistryEntry


def _row(
    arm_id: str,
    *,
    games: int | None,
    own_probability_positive: float | None,
    is_promoted: bool = False,
) -> ModelLedgerRowView:
    return ModelLedgerRowView(
        arm_id=arm_id,
        display_name=arm_id,
        status_badge="PROMOTED" if is_promoted else "CHALLENGER",
        is_promoted=is_promoted,
        games=games,
        accuracy=None,
        interval_low=None,
        interval_high=None,
        interval_unit="accuracy_points",
        grade="",
        summary_sentence="",
        own_probability_positive=own_probability_positive,
        evidence=(),
        agreement_text=None,
        artifact_ref=None,
    )


def test_evidence_strength_ranks_by_distance_from_a_coin_flip() -> None:
    assert bsc._evidence_strength(0.95) == pytest.approx(bsc._evidence_strength(0.05))
    assert bsc._evidence_strength(0.9) > bsc._evidence_strength(0.6)
    assert bsc._evidence_strength(0.5) == 0.0


def test_evidence_strength_unmeasured_sorts_last() -> None:
    assert bsc._evidence_strength(None) < bsc._evidence_strength(0.5)
    assert bsc._evidence_strength(None) < bsc._evidence_strength(0.01)


def test_grouped_ledger_rows_splits_on_games_is_none() -> None:
    rows = (
        _row("a", games=100, own_probability_positive=0.9),
        _row("b", games=None, own_probability_positive=0.8),
        _row("c", games=50, own_probability_positive=None),
    )
    graded, waiting = bsc._grouped_ledger_rows(rows)
    assert {row.arm_id for row in graded} == {"a", "c"}
    assert {row.arm_id for row in waiting} == {"b"}


def test_grouped_ledger_rows_sorts_each_group_by_evidence_strength_descending() -> None:
    rows = (
        _row("weak", games=10, own_probability_positive=0.55),
        _row("strong", games=10, own_probability_positive=0.98),
        _row("unmeasured", games=10, own_probability_positive=None),
    )
    graded, _waiting = bsc._grouped_ledger_rows(rows)
    assert [row.arm_id for row in graded] == ["strong", "weak", "unmeasured"]


def test_grouped_ledger_rows_pins_the_promoted_row_first_regardless_of_evidence() -> None:
    rows = (
        _row("challenger_strong", games=10, own_probability_positive=0.99),
        _row("promoted", games=2000, own_probability_positive=None, is_promoted=True),
    )
    graded, _waiting = bsc._grouped_ledger_rows(rows)
    assert graded[0].arm_id == "promoted"
    assert graded[1].arm_id == "challenger_strong"


def test_grouped_ledger_rows_handles_empty_groups() -> None:
    graded, waiting = bsc._grouped_ledger_rows(())
    assert graded == ()
    assert waiting == ()


def _entry(key: str, name: str, probability_positive: float | None) -> RegistryEntry:
    return RegistryEntry(
        key=key,
        store=key.split(":")[0],
        name=name,
        description="",
        classification=None,
        effect=None,
        effect_units=None,
        probability_positive=probability_positive,
        interval=None,
        seasons=None,
        league=None,
        recorded_at=None,
        fingerprint="",
    )


def _finding(registry_keys: tuple[str, ...]) -> Finding:
    return Finding(
        question="Q",
        verdict="unresolved",
        plain_answer="A",
        detail="D",
        source="test",
        registry_keys=registry_keys,
    )


def test_finding_trace_uses_the_first_key_with_a_measured_probability() -> None:
    entries = {
        "weak_signal:no_pp": _entry("weak_signal:no_pp", "no_pp", None),
        "weak_signal:has_pp": _entry("weak_signal:has_pp", "has_pp", 0.93),
    }
    finding = _finding(("weak_signal:no_pp", "weak_signal:has_pp"))
    name, probability = bsc._finding_trace(finding, entries)
    assert name == "has_pp"
    assert probability == 0.93


def test_finding_trace_is_none_when_no_key_resolves() -> None:
    finding = _finding(("weak_signal:missing",))
    name, probability = bsc._finding_trace(finding, {})
    assert name is None
    assert probability is None


def test_finding_trace_is_none_for_an_evergreen_finding_with_no_keys() -> None:
    finding = _finding(())
    name, probability = bsc._finding_trace(
        finding, {"weak_signal:x": _entry("weak_signal:x", "x", 0.9)}
    )
    assert name is None
    assert probability is None


def _prospective_decisions(game_id: str, pick_side: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "challenger_id": "challenger-x",
                "season": 2026,
                "week": 1,
                "kickoff": "2026-09-10T20:00:00Z",
                "recorded_at_utc": "2026-09-09T12:00:00Z",
                "pick_side": pick_side,
                "decision_home_spread": -3.5,
            }
        ]
    )


def test_history_challenger_uses_matching_latest_score_report_not_registry_evidence() -> None:
    challenger = _prospective_decisions("g1", "HOME")
    active = _prospective_decisions("g1", "AWAY").drop(columns="challenger_id")
    outcomes = pd.DataFrame([{"game_id": "g1", "result": 4.0}])
    report = {
        "forced_picks": {"decision_line": {"games": 1}},
        "uncertainty": [
            {
                "metric": "decision_line_accuracy",
                "block": "week",
                "probability_positive": 0.88,
                "lower": 0.45,
                "upper": 0.95,
            }
        ],
    }
    rows = bsc._history_challenger_assessments(
        [{"challenger_id": "challenger-x", "evidence": {"probability_positive": 0.12}}],
        challenger,
        active,
        outcomes,
        {"challenger-x": report},
    )
    assert rows[0].probability_positive == pytest.approx(0.88)
    assert rows[0].interval_low == pytest.approx(0.45)
    assert "latest prospective-score report" in rows[0].grading_basis


def test_history_challenger_labels_registry_evidence_when_score_report_is_not_paired() -> None:
    challenger = _prospective_decisions("g1", "HOME")
    active = _prospective_decisions("g1", "AWAY").drop(columns="challenger_id")
    outcomes = pd.DataFrame([{"game_id": "g1", "result": 4.0}])
    rows = bsc._history_challenger_assessments(
        [{"challenger_id": "challenger-x", "evidence": {"probability_positive": 0.12}}],
        challenger,
        active,
        outcomes,
        {
            "challenger-x": {
                "forced_picks": {"decision_line": {"games": 0}},
                "uncertainty": [],
            }
        },
    )
    assert rows[0].probability_positive == pytest.approx(0.12)
    assert "pre-registration/historical evidence" in rows[0].grading_basis


# ---------------------------------------------------------------------------
# UI-20(h): _season_grade_rows / _history_week_grades / _load_close_schedule
# ---------------------------------------------------------------------------


def _season_row(season: str, games: int, opener: float, close: float | None) -> bsc.SeasonRowView:
    return bsc.SeasonRowView(
        season=season, games=games, opener_accuracy=opener, close_accuracy=close
    )


def test_season_grade_rows_reuses_the_model_page_pairs_unchanged() -> None:
    seasons = (
        _season_row("2024", 272, 0.545, 0.553),
        _season_row("2025", 272, 0.532, 0.509),
    )
    rows = bsc._season_grade_rows(seasons, active={})
    assert [row.season_label for row in rows] == ["2024", "2025"]
    assert all(row.note == "" for row in rows)
    assert rows[0].delta == pytest.approx(0.545 - 0.553)
    assert rows[0].delta_text == "-0.8%"


def test_season_grade_rows_adds_a_dynamic_gap_row_when_the_archive_is_narrower() -> None:
    """The archive's population (here: 100 games) is narrower than the
    model's own long-run evaluation (here: 638 games) -- the 538-game gap
    must appear as an explicit row, never silently dropped, and its size is
    COMPUTED from the two live totals, never a hardcoded figure."""

    seasons = (_season_row("2025", 100, 0.53, 0.51),)
    active = {"historical_evaluation": {"games": 638}}
    rows = bsc._season_grade_rows(seasons, active)
    assert len(rows) == 2
    gap_row, season_row = rows
    assert gap_row.games == 538
    assert gap_row.opener_accuracy is None
    assert gap_row.close_accuracy is None
    assert gap_row.note == bsc.NO_OPENER_LINE_ARCHIVED_SEASON_NOTE
    assert season_row.season_label == "2025"


def test_season_grade_rows_omits_the_gap_row_when_totals_already_agree() -> None:
    seasons = (_season_row("2025", 272, 0.53, 0.51),)
    active = {"historical_evaluation": {"games": 272}}
    rows = bsc._season_grade_rows(seasons, active)
    assert len(rows) == 1
    assert rows[0].note == ""


def test_season_grade_rows_omits_the_gap_row_when_no_historical_total_exists() -> None:
    rows = bsc._season_grade_rows((), active={})
    assert rows == ()


def _decision_row(
    game_id: str, season: int, week: int, *, pick_side: str, decision_home_spread: float
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "kickoff": pd.Timestamp("2026-09-10T20:00:00Z"),
        "recorded_at_utc": pd.Timestamp("2026-09-09T00:00:00Z"),
        "pick_side": pick_side,
        "decision_home_spread": decision_home_spread,
    }


def test_history_week_grades_empty_ledger_returns_no_rows() -> None:
    assert bsc._history_week_grades(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()) == ()


def test_history_week_grades_reports_both_records_and_their_delta() -> None:
    decisions = pd.DataFrame(
        [
            # Home picked, home wins by 4 vs a -3 decision line -> covers.
            _decision_row("G1", 2026, 1, pick_side="HOME", decision_home_spread=-3.0),
        ]
    )
    outcomes = pd.DataFrame([{"game_id": "G1", "result": 4.0}])
    close_reference = pd.DataFrame([{"game_id": "G1", "close_home_spread": -2.5}])
    rows = bsc._history_week_grades(decisions, outcomes, close_reference)
    assert len(rows) == 1
    row = rows[0]
    assert (row.season, row.week, row.picks) == (2026, 1, 1)
    assert row.opener_settled == 1 and row.opener_wins == 1
    assert row.close_settled == 1 and row.close_wins == 1
    assert row.note == ""
    assert row.opener_record_text == "1-0 (100.0%)"
    assert row.delta_text == "+0.0%"


def test_history_week_grades_no_close_reference_renders_an_explicit_note() -> None:
    """A week with a recorded opener pick but NO resolvable close line
    (``close_reference`` has no row for its games) must say so explicitly,
    never leave a blank cell."""

    decisions = pd.DataFrame(
        [_decision_row("G1", 2026, 2, pick_side="AWAY", decision_home_spread=2.5)]
    )
    outcomes = pd.DataFrame([{"game_id": "G1", "result": -1.0}])
    rows = bsc._history_week_grades(decisions, outcomes, pd.DataFrame())
    assert len(rows) == 1
    row = rows[0]
    assert row.opener_settled == 1
    assert row.close_settled == 0
    assert row.close_accuracy is None
    assert row.note == bsc.NO_CLOSE_LINE_ARCHIVED_WEEK_NOTE
    assert row.close_record_text == "--"


def test_history_week_grades_unplayed_week_says_not_yet_settled() -> None:
    """No outcome recorded at all (the game has not been played) must read
    as "not yet settled", never as "no opener line archived" -- the two
    are different facts and must not be conflated."""

    decisions = pd.DataFrame(
        [_decision_row("G1", 2026, 3, pick_side="HOME", decision_home_spread=-1.0)]
    )
    outcomes = pd.DataFrame(columns=["game_id", "result"])
    rows = bsc._history_week_grades(decisions, outcomes, pd.DataFrame())
    assert len(rows) == 1
    row = rows[0]
    assert row.opener_settled == 0 and row.close_settled == 0
    assert row.note == bsc.HISTORY_WEEK_NOT_SETTLED_NOTE


def test_load_close_schedule_missing_table_degrades_to_empty(tmp_path: Path) -> None:
    schedule = bsc._load_close_schedule(tmp_path)
    assert list(schedule.columns) == list(bsc._CLOSE_SCHEDULE_COLUMNS)
    assert schedule.empty
