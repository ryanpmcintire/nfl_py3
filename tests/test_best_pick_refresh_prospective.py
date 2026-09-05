from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from nfl_ats import best_pick_refresh_prospective as best


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch):
    artifacts, data = tmp_path / "artifacts", tmp_path / "data"
    instant = pd.Timestamp("2026-09-08T16:00:00+00:00")
    original = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2026,
                "week": 1,
                "kickoff": pd.Timestamp(kickoff),
                "recorded_at_utc": instant,
                "pick_side": "HOME",
                "decision_home_spread": 3.0,
                "is_best_pick": game_id == "a",
            }
            for game_id, kickoff in [
                ("a", "2026-09-13T17:00:00+00:00"),
                ("b", "2026-09-14T00:20:00+00:00"),
            ]
        ]
    )
    monkeypatch.setattr(best, "original_card", lambda *args, **kwargs: original.copy())
    publication = {
        "season": 2026,
        "week": 1,
        "best_pick_game_id": "a",
        "best_pick_prospective_input": {
            "predictions": [
                {"game_id": "a", "home_cover_probability": 0.6},
                {"game_id": "b", "home_cover_probability": 0.55},
            ],
            "pool": [
                {"game_id": game_id, "pool_pass": True, "spread_std": 0.5} for game_id in ("a", "b")
            ],
        },
    }
    snapshot = data / "raw" / "20260908" / "schedules.parquet"
    snapshot.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"game_id": game_id, "home_score": float("nan"), "away_score": float("nan")}
            for game_id in ("a", "b")
        ]
    ).to_parquet(snapshot)
    games = tuple(
        SimpleNamespace(
            game_id=row.game_id,
            kickoff=row.kickoff,
            deadline=min(row.kickoff, pd.Timestamp("2026-09-13T20:00:00+00:00")),
            eligible=True,
            original_recorded_at_utc=instant,
            new_home_cover_probability=0.59 if row.game_id == "a" else 0.7,
            new_pick_side="HOME",
            decision_home_spread=3.0,
        )
        for row in original.itertuples()
    )
    plan = SimpleNamespace(
        season=2026, week=1, computed_at_utc=pd.Timestamp("2026-09-13T14:00:00+00:00"), games=games
    )
    return artifacts, data, publication, original, plan, snapshot


def freeze(inputs):
    artifacts, data, publication, _, _, _ = inputs
    result = best.record_best_pick_tuesday(
        artifacts, data, publication, now=datetime.fromisoformat("2026-09-08T16:00:00+00:00")
    )
    assert result["recorded"] == 1


def test_pair_freezes_both_probabilities_and_settles(inputs):
    freeze(inputs)
    artifacts, data, publication, _, plan, snapshot = inputs
    assert (
        best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)["recorded"] == 1
    )
    ledger = best.load_decisions(artifacts)
    row = ledger.iloc[0]
    assert (row["tuesday_game_id"], row["sunday_game_id"]) == ("a", "b")
    assert row["tuesday_probability"] == 0.6
    assert row["sunday_probability"] == 0.7
    assert row["nominees_differ"]
    plan.games[0].new_home_cover_probability = 0.99
    assert (
        best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)[
            "already_recorded"
        ]
        == 1
    )
    assert best.record_best_pick_tuesday(artifacts, data, publication)["already_recorded"] == 1
    pd.testing.assert_frame_equal(ledger, best.load_decisions(artifacts))
    pd.DataFrame(
        [
            {"game_id": "a", "home_score": 20, "away_score": 24},
            {"game_id": "b", "home_score": 28, "away_score": 20},
        ]
    ).to_parquet(snapshot)
    best.record_best_pick_tuesday(artifacts, data, publication)
    row = best.load_decisions(artifacts).iloc[0]
    assert row["tuesday_cover"] == 0
    assert row["sunday_cover"] == 1
    assert row["paired_cover_delta"] == 1


@pytest.mark.parametrize(
    "instant",
    [
        "2026-09-12T14:00:00Z",
        "2026-09-13T16:00:00Z",
        "2026-09-13T20:00:00Z",
        "2026-09-20T14:00:00Z",
    ],
)
def test_only_this_sunday_morning_is_recordable(inputs, instant):
    freeze(inputs)
    artifacts, data, _, _, plan, _ = inputs
    plan.computed_at_utc = pd.Timestamp(instant)
    assert (
        best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)["recorded"] == 0
    )
    assert best.load_decisions(artifacts)["sunday_game_id"].isna().all()


def test_missing_or_disabled_refresh_skips(inputs):
    artifacts, data, _, _, plan, _ = inputs
    assert best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)["skipped"]
    freeze(inputs)
    assert best.record_best_pick_refresh(artifacts, data, plan)["skipped"]
    plan.games = plan.games[:1]
    assert (
        "missing"
        in best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)["reason"]
    )


@pytest.mark.parametrize("home_score", [0, 50])
def test_locked_tuesday_nominee_cannot_be_replaced_using_known_outcome(inputs, home_score):
    artifacts, data, _, original, plan, snapshot = inputs
    original.loc[original["game_id"].eq("a"), "kickoff"] = pd.Timestamp("2026-09-11T00:15:00Z")
    freeze(inputs)
    pd.DataFrame(
        [
            {"game_id": "a", "home_score": home_score, "away_score": 20},
            {"game_id": "b", "home_score": None, "away_score": None},
        ]
    ).to_parquet(snapshot)
    result = best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)
    assert result["recorded"] == 1
    row = best.load_decisions(artifacts).iloc[0]
    assert row["sunday_game_id"] == row["tuesday_game_id"] == "a"
    assert row["sunday_probability"] == row["tuesday_probability"]
    assert not row["nominees_differ"]


def test_invalid_probability_and_future_original_timestamp_skip(inputs):
    freeze(inputs)
    artifacts, data, _, _, plan, _ = inputs
    plan.games[1].new_home_cover_probability = float("nan")
    assert (
        "invalid"
        in best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)["reason"]
    )
    plan.games[1].new_home_cover_probability = 0.7
    plan.games[1].original_recorded_at_utc = plan.computed_at_utc + pd.Timedelta(days=1)
    assert (
        "missing"
        in best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)["reason"]
    )


def test_same_nominee_is_still_a_pair(inputs):
    freeze(inputs)
    artifacts, data, _, _, plan, _ = inputs
    plan.games[0].new_home_cover_probability = 0.8
    result = best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)
    assert result["recorded"] == 1
    assert not result["nominees_differ"]


def test_push_is_retained_separately(inputs):
    freeze(inputs)
    artifacts, data, _, _, plan, snapshot = inputs
    best.record_best_pick_refresh(artifacts, data, plan, record_decisions=True)
    pd.DataFrame(
        [
            {"game_id": "a", "home_score": 23, "away_score": 20},
            {"game_id": "b", "home_score": None, "away_score": None},
        ]
    ).to_parquet(snapshot)
    row = best.settle_ledger(artifacts, data).iloc[0]
    assert row["tuesday_status"] == "push"
    assert pd.isna(row["tuesday_cover"])
    assert pd.isna(row["paired_cover_delta"])
