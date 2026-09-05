from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats import tiebreaker_shade_prospective as shade


@pytest.fixture
def inputs(tmp_path: Path):
    history = pd.DataFrame(
        [
            {
                "game_id": f"old-{i}",
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "gameday": "2025-09-07",
                "gametime": "13:00",
                "home_score": h,
                "away_score": a,
                "spread_line": 3.0,
                "total_line": 44.5,
            }
            for i, (h, a) in enumerate([(23, 20), (24, 20), (24, 21), (21, 20), (23, 21), (27, 17)])
        ]
    )
    current = pd.DataFrame(
        [
            {
                "game_id": "last",
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "gameday": "2026-09-14",
                "gametime": "20:15",
                "home_score": None,
                "away_score": None,
                "spread_line": 3.0,
                "total_line": 44.5,
            }
        ]
    )
    schedules = pd.concat([history, current], ignore_index=True)
    data = tmp_path / "data"
    snapshot = data / "raw" / "20260908" / "schedules.parquet"
    snapshot.parent.mkdir(parents=True)
    schedules.to_parquet(snapshot)
    payload = {
        "game_id": "last",
        "season": 2026,
        "week": 1,
        "guess_home": 24,
        "guess_away": 20,
        "market_total": 44.5,
        "lattice_centre_margin": 4.0,
        "pick_side": "HOME",
        "pick_spread_line": 3.0,
        "generated_at_utc": "2026-09-08T16:00:00+00:00",
    }
    published = tmp_path / "tiebreaker.json"
    published.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path / "artifacts", data, published, payload, schedules, snapshot


def test_pair_is_frozen_and_settled_on_later_pass(inputs):
    artifacts, data, published, payload, schedules, snapshot = inputs
    instant = datetime.fromisoformat(payload["generated_at_utc"])
    result = shade.record_tiebreaker_shade_decisions(
        artifacts, data, published_path=published, now=instant
    )
    assert result["recorded"] == 1
    original = shade.load_decisions(artifacts)
    assert original.iloc[0]["served_total"] == 44
    assert original.iloc[0]["shade_target"] == 43.5
    assert original.iloc[0]["shaded_home"] - original.iloc[0]["shaded_away"] > 3
    payload["guess_home"] = 90
    published.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        shade.record_tiebreaker_shade_decisions(
            artifacts, data, published_path=published, now=instant
        )["already_recorded"]
        == 1
    )
    pd.testing.assert_frame_equal(original, shade.load_decisions(artifacts))
    schedules.loc[schedules["game_id"].eq("last"), ["home_score", "away_score"]] = [23, 20]
    schedules.to_parquet(snapshot)
    # Settlement precedes missing-current-publication handling.
    result = shade.record_tiebreaker_shade_decisions(
        artifacts, data, now=datetime.fromisoformat("2026-09-15T16:00:00+00:00")
    )
    assert result["skipped"]
    settled = shade.load_decisions(artifacts).iloc[0]
    assert settled["actual_total"] == 43
    assert settled["served_absolute_error"] == 1
    assert settled["shaded_absolute_error"] == abs(settled["shaded_total"] - 43)
    assert settled["closer_arm"] in {"tie", "shaded", "served"}


@pytest.mark.parametrize(
    "instant",
    ["2026-08-01T16:00:00+00:00", "2026-09-13T20:00:00+00:00", "2026-09-15T00:15:00+00:00"],
)
def test_cannot_record_outside_playable_window(inputs, instant):
    artifacts, data, published, payload, _, _ = inputs
    payload["generated_at_utc"] = instant
    published.write_text(json.dumps(payload), encoding="utf-8")
    result = shade.record_tiebreaker_shade_decisions(
        artifacts, data, published_path=published, now=datetime.fromisoformat(instant)
    )
    assert result["recorded"] == 0
    assert result["reason"]
    assert not shade.ledger_path(artifacts).exists()


def test_future_finals_cannot_change_shaded_cell(inputs):
    _, _, _, payload, schedules, _ = inputs
    expected = shade.shaded_score(payload, schedules)
    changed = schedules.copy()
    changed.loc[changed["season"].ge(2026), ["home_score", "away_score"]] = [99, 1]
    future = changed.iloc[[-1]].copy()
    future["season"] = 2027
    future["game_id"] = "future"
    assert shade.shaded_score(payload, pd.concat([changed, future])) == expected


def test_stale_payload_and_missing_inputs_skip(inputs):
    artifacts, data, published, _, _, _ = inputs
    result = shade.record_tiebreaker_shade_decisions(
        artifacts,
        data,
        published_path=published,
        now=datetime.fromisoformat("2026-09-08T17:00:00+00:00"),
    )
    assert "not from this publication" in result["reason"]
    assert shade.record_tiebreaker_shade_decisions(artifacts, data / "missing")["skipped"]


@pytest.mark.parametrize(
    ("served", "shaded", "winner"), [(44, 43, "shaded"), (43, 44, "served"), (42, 44, "tie")]
)
def test_absolute_error_winner(served, shaded, winner):
    decisions = pd.DataFrame(
        [
            {
                "game_id": "g",
                "served_total": served,
                "shaded_total": shaded,
                "actual_total": float("nan"),
            }
        ]
    )
    schedules = pd.DataFrame([{"game_id": "g", "home_score": 23, "away_score": 20}])
    settled = shade.settle_decisions(decisions, schedules)
    assert settled.iloc[0]["closer_arm"] == winner
    schedules["home_score"] = 100
    assert shade.settle_decisions(settled, schedules).iloc[0]["actual_total"] == 43
