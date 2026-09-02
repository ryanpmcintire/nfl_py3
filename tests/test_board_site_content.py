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
