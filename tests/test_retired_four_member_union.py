"""The retired union stays paired without toggling overlapping member flips."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from test_four_overlay_incumbent import _write_fixture

from nfl_ats.clv import paper_decision_ledger_path
from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import load_challenger_decisions
from nfl_ats.retired_four_member_union import (
    INCUMBENT_CHALLENGER_ID,
    record_retired_four_member_union_decisions,
    retired_union_sides,
)


def test_retired_union_complements_model_once_at_both_zone_edges() -> None:
    primary = pd.DataFrame(
        {
            "decision_home_spread": [7.0, 7.5, -10.0, 10.5, -8.0],
            "model_pick_side": ["HOME", "HOME", "AWAY", "AWAY", "HOME"],
            "composed_overlay_flip": [False, False, False, False, True],
        }
    )
    assert retired_union_sides(primary).tolist() == ["HOME", "AWAY", "HOME", "AWAY", "AWAY"]


def _fixture(root: Path, *, complete: bool = True) -> tuple[Path, Path]:
    artifacts, data = _write_fixture(root, complete_primary=complete)
    registry = artifacts / "prospective/challengers.json"
    payload = json.loads(registry.read_text())
    payload["challengers"][0]["challenger_id"] = INCUMBENT_CHALLENGER_ID
    registry.write_text(json.dumps(payload))
    path = paper_decision_ledger_path(artifacts)
    frame = pd.read_parquet(path)
    frame["decision_home_spread"] = 7.5
    frame.to_parquet(path, index=False)
    return artifacts, data


def test_recorder_pairs_frozen_sides_and_is_first_write_wins(tmp_path: Path) -> None:
    artifacts, data = _fixture(tmp_path)
    now = datetime(2026, 9, 8, 16, 30, tzinfo=UTC)
    before = pd.read_parquet(paper_decision_ledger_path(artifacts))
    result = record_retired_four_member_union_decisions(artifacts, data, now=now)
    assert result["recorded"] == 2
    ledger = load_challenger_decisions(artifacts)
    assert ledger["pick_side"].tolist() == ["AWAY", "AWAY"]
    assert ledger["decision_home_spread"].tolist() == [7.5, 7.5]
    assert ledger["challenger_id"].eq(INCUMBENT_CHALLENGER_ID).all()
    pd.testing.assert_frame_equal(before, pd.read_parquet(paper_decision_ledger_path(artifacts)))
    again = record_retired_four_member_union_decisions(artifacts, data, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_recorder_refuses_incomplete_primary_card(tmp_path: Path) -> None:
    artifacts, data = _fixture(tmp_path, complete=False)
    with pytest.raises(DataContractError, match="complete current"):
        record_retired_four_member_union_decisions(
            artifacts, data, now=datetime(2026, 9, 8, 16, 30, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty
