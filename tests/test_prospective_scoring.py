"""POL-10: prospective pick recording and forced-pick ATS settlement.

The load-bearing tests here are the anti-backdating ones. Prospective evidence
is only worth anything if the picks provably existed before kickoff, so both
the write path and the read path are tested for refusing a post-kickoff pick,
and the fingerprint guard is tested for refusing a silently re-tuned challenger.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_parquet
from nfl_ats.prospective_scoring import (
    ACTIVE_CHALLENGER_STATUS,
    CHALLENGER_DECISION_COLUMNS,
    challenger_ledger_path,
    challenger_registry_path,
    config_fingerprint,
    find_challenger,
    find_challenger_artifact,
    load_challenger_decisions,
    prospective_accuracy,
    prospective_accuracy_metrics,
    prospective_week_summary,
    record_challenger_decisions,
    settle_prospective_picks,
)

KICKOFF = pd.Timestamp("2026-09-13T17:00:00Z")
RECORDED = pd.Timestamp("2026-09-08T16:00:00Z")


def _decisions(**overrides: Any) -> pd.DataFrame:
    """Four picks with hand-chosen lines: one home win, one away win, two pushes."""

    frame = pd.DataFrame(
        {
            "game_id": ["g_home", "g_away", "g_push_dec", "g_push_close"],
            "season": 2026,
            "week": 1,
            "kickoff": KICKOFF,
            "recorded_at_utc": RECORDED,
            "pick_side": ["HOME", "AWAY", "HOME", "AWAY"],
            "bet_side": ["HOME", "PASS", "HOME", "AWAY"],
            "decision_home_spread": [-3.5, 6.5, 7.0, 2.5],
            "is_best_pick": [True, False, False, False],
        }
    )
    for column, value in overrides.items():
        frame[column] = value
    return frame


def _outcomes() -> pd.DataFrame:
    # result is home minus away points.
    return pd.DataFrame(
        {
            "game_id": ["g_home", "g_away", "g_push_dec", "g_push_close"],
            "result": [7.0, 3.0, 7.0, -1.0],
        }
    )


def _close_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g_home", "g_away", "g_push_dec", "g_push_close"],
            "close_home_spread": [-6.5, 6.5, 3.0, -1.0],
            "close_source": "live_store_close",
        }
    )


# ---------------------------------------------------------------------------
# 1. Settlement, hand computed at both grades
# ---------------------------------------------------------------------------


def test_settlement_is_hand_computable_at_both_grades_including_pushes() -> None:
    settled = settle_prospective_picks(
        _decisions(), _outcomes(), close_reference=_close_reference()
    ).set_index("game_id")

    # HOME pick at -3.5, result +7 -> margin +10.5 -> home covered -> correct.
    assert settled.loc["g_home", "ats_margin_at_decision_line"] == pytest.approx(10.5)
    assert settled.loc["g_home", "correct_at_decision_line"] == pytest.approx(1.0)
    # Same pick at the close of -6.5 -> margin +13.5 -> still correct.
    assert settled.loc["g_home", "correct_at_close_line"] == pytest.approx(1.0)

    # AWAY pick at +6.5, result +3 -> margin -3.5 -> home did NOT cover -> correct.
    assert settled.loc["g_away", "ats_margin_at_decision_line"] == pytest.approx(-3.5)
    assert settled.loc["g_away", "correct_at_decision_line"] == pytest.approx(1.0)

    # A zero margin is a push at that grade: excluded, never scored as a loss.
    assert settled.loc["g_push_dec", "ats_margin_at_decision_line"] == pytest.approx(0.0)
    assert pd.isna(settled.loc["g_push_dec", "correct_at_decision_line"])
    assert settled.loc["g_push_dec", "status_at_decision_line"] == "push"
    # ...and the same game is NOT a push at the close (line 3.0, margin +4.0).
    assert settled.loc["g_push_dec", "status_at_close_line"] == "settled"
    assert settled.loc["g_push_dec", "correct_at_close_line"] == pytest.approx(1.0)

    # The mirror case: settled at the decision line, push at the close.
    assert settled.loc["g_push_close", "status_at_decision_line"] == "settled"
    assert settled.loc["g_push_close", "status_at_close_line"] == "push"
    assert pd.isna(settled.loc["g_push_close", "correct_at_close_line"])


def test_games_without_a_result_or_close_stay_pending_not_wrong() -> None:
    outcomes = _outcomes()
    outcomes.loc[outcomes["game_id"].eq("g_home"), "result"] = np.nan
    settled = settle_prospective_picks(_decisions(), outcomes).set_index("game_id")

    assert settled.loc["g_home", "status_at_decision_line"] == "pending"
    assert pd.isna(settled.loc["g_home", "correct_at_decision_line"])
    # No close reference was supplied at all, so every close grade is pending.
    assert set(settled["status_at_close_line"]) == {"pending"}
    assert settled["correct_at_close_line"].isna().all()


def test_accuracy_reports_both_grades_plus_best_pick_and_bet_subsets() -> None:
    settled = settle_prospective_picks(
        _decisions(), _outcomes(), close_reference=_close_reference()
    )
    summary = prospective_accuracy(settled)

    assert summary["decisions"] == 4
    decision = summary["forced_picks"]["decision_line"]
    # Three non-push games at the decision line; all three picks were correct.
    assert decision["games"] == 3
    assert decision["correct"] == 3
    assert decision["accuracy"] == pytest.approx(1.0)
    assert decision["pushes"] == 1
    assert decision["vs_coin_flip"] == pytest.approx(0.5)
    assert summary["forced_picks"]["close_line"]["pushes"] == 1

    # The Best Pick subset is exactly the one flagged game.
    assert summary["best_pick"]["weeks_nominated"] == 1
    assert summary["best_pick"]["decision_line"]["games"] == 1
    # PASS rows are excluded from the paper-bet subset.
    assert summary["paper_bets"]["bets"] == 3


def test_metrics_have_the_bootstrap_metric_fn_shape() -> None:
    settled = settle_prospective_picks(
        _decisions(), _outcomes(), close_reference=_close_reference()
    )
    metrics = prospective_accuracy_metrics(settled)
    assert set(metrics) == {
        "decision_line_accuracy",
        "close_line_accuracy",
        "decision_vs_coin_flip",
        "decision_minus_close",
    }
    assert metrics["decision_vs_coin_flip"] == pytest.approx(
        metrics["decision_line_accuracy"] - 0.5
    )

    resolved = settled.dropna(subset=["correct_at_decision_line"])
    bootstrap = week_blocked_bootstrap(
        resolved, prospective_accuracy_metrics, block="week", samples=20, seed=7
    )
    assert "probability_positive" in bootstrap.columns


def test_week_summary_carries_the_weekly_record_and_the_best_pick() -> None:
    settled = settle_prospective_picks(
        _decisions(), _outcomes(), close_reference=_close_reference()
    )
    summary = prospective_week_summary(settled)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert (row["season"], row["week"]) == (2026, 1)
    assert row["picks"] == 4
    assert row["settled"] == 3
    assert row["pushes"] == 1
    assert row["best_pick_game_id"] == "g_home"
    assert row["best_pick_correct"] == pytest.approx(1.0)
    assert prospective_week_summary(settled.iloc[0:0]).empty


def test_settlement_contract_guards() -> None:
    outcomes = _outcomes()
    with pytest.raises(DataContractError, match="missing columns"):
        settle_prospective_picks(_decisions().drop(columns=["pick_side"]), outcomes)
    duplicated = pd.concat([_decisions(), _decisions()], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate game rows"):
        settle_prospective_picks(duplicated, outcomes)
    with pytest.raises(ValueError, match="unsupported pick sides"):
        settle_prospective_picks(_decisions(pick_side="PUSH"), outcomes)
    with pytest.raises(ValueError, match="unsupported bet sides"):
        settle_prospective_picks(_decisions(bet_side="MAYBE"), outcomes)
    with pytest.raises(DataContractError, match="outcomes are missing"):
        settle_prospective_picks(_decisions(), outcomes.drop(columns=["result"]))


# ---------------------------------------------------------------------------
# 2. Anti-backdating: the guarantee the whole exercise rests on
# ---------------------------------------------------------------------------


def test_scoring_refuses_a_pick_recorded_at_or_after_its_own_kickoff() -> None:
    """The read-side half of the guarantee.

    A hand-edited ledger row claiming a pick that was really chosen after the
    game started must not be launderable into "prospective" evidence just by
    running the scorer over it.
    """

    backdated = _decisions()
    backdated.loc[0, "recorded_at_utc"] = KICKOFF + pd.Timedelta(hours=3)
    with pytest.raises(DataContractError, match="recorded at or after kickoff"):
        settle_prospective_picks(backdated, _outcomes())

    # Exactly at kickoff is late too -- the pool locks strictly before.
    at_kickoff = _decisions()
    at_kickoff.loc[0, "recorded_at_utc"] = KICKOFF
    with pytest.raises(DataContractError, match="recorded at or after kickoff"):
        settle_prospective_picks(at_kickoff, _outcomes())

    # A row that cannot prove its timing at all is refused rather than assumed.
    unknown = _decisions()
    unknown.loc[0, "kickoff"] = pd.NaT
    with pytest.raises(DataContractError, match="cannot prove pre-kickoff timing"):
        settle_prospective_picks(unknown, _outcomes())


# ---------------------------------------------------------------------------
# 3. The challenger ledger
# ---------------------------------------------------------------------------

_MODEL = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "min_edge": 0.02,
    "min_train_games": 500,
    "feature_table": "data/processed/game_features_weak_stack.parquet",
}


def _write_registry(artifacts: Path, *, status: str = ACTIVE_CHALLENGER_STATUS) -> None:
    payload = {
        "ledger": "prospective_challengers",
        "schema_version": 1,
        "challengers": [
            {
                "challenger_id": "stack",
                "status": status,
                "model": dict(_MODEL),
            }
        ],
    }
    path = challenger_registry_path(artifacts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_card(
    artifacts: Path,
    name: str,
    *,
    kickoffs: list[str],
    probabilities: list[float],
    profile: str = "weak_stack",
    spread_lines: list[float] | None = None,
) -> Path:
    directory = artifacts / "margin_predictions" / name
    directory.mkdir(parents=True, exist_ok=True)
    count = len(kickoffs)
    pd.DataFrame(
        {
            "game_id": [f"2026_01_A{index}_H{index}" for index in range(count)],
            "season": 2026,
            "week": 1,
            "kickoff": kickoffs,
            "away_team": [f"A{index}" for index in range(count)],
            "home_team": [f"H{index}" for index in range(count)],
            "spread_line": spread_lines or [2.5] * count,
            "home_cover_probability": probabilities,
            "bet_side": ["HOME"] * count,
            "edge": 0.05,
        }
    ).to_csv(directory / "recommendations.csv", index=False)
    metadata = {
        "created_at_utc": "2026-09-08T15:00:00+00:00",
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "feature_profile": profile,
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {
            "feature_table": {
                "path": f"F:/repo/data/processed/game_features_{profile}.parquet",
                "sha256": "abc123",
            }
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def test_record_challenger_records_dedupes_and_refuses_started_games(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    card = _write_card(
        artifacts,
        "2026-week-01-a",
        kickoffs=["2026-09-13T17:00:00+00:00", "2026-09-08T00:15:00+00:00"],
        probabilities=[0.61, 0.40],
    )
    now = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)

    result = record_challenger_decisions(artifacts, "stack", card, now=now)
    assert result["recorded"] == 1
    assert result["post_kickoff_skipped"] == 1
    assert result["config_fingerprint"] == config_fingerprint(_MODEL)

    ledger = load_challenger_decisions(artifacts)
    assert list(ledger.columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert ledger["pick_side"].tolist() == ["HOME"]
    assert ledger["source_sha256"].str.len().iloc[0] == 64

    # Re-running records nothing new, and a card with moved lines and a flipped
    # pick cannot rewrite what is already on the ledger.
    moved = _write_card(
        artifacts,
        "2026-week-01-b",
        kickoffs=["2026-09-13T17:00:00+00:00", "2026-09-08T00:15:00+00:00"],
        probabilities=[0.10, 0.40],
        spread_lines=[9.5, 2.5],
    )
    again = record_challenger_decisions(artifacts, "stack", moved, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 1
    anchored = load_challenger_decisions(artifacts)
    assert anchored["pick_side"].tolist() == ["HOME"]
    assert anchored["decision_home_spread"].tolist() == [pytest.approx(2.5)]


def test_record_challenger_refuses_a_recording_weeks_before_kickoff(tmp_path: Path) -> None:
    """Same guard as the paper-decision ledger, and the same shared function
    (nfl_ats.clv.refuse_if_outside_recording_lock_window) -- a rehearsal run
    weeks before a week's real kickoff must not reach the challenger ledger
    either."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    card = _write_card(
        artifacts,
        "2026-week-01-a",
        kickoffs=["2026-09-13T17:00:00+00:00", "2026-09-08T00:15:00+00:00"],
        probabilities=[0.61, 0.40],
    )
    now = datetime(2026, 8, 18, 1, 24, 56, tzinfo=UTC)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_challenger_decisions(artifacts, "stack", card, now=now)

    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_a_retuned_configuration(tmp_path: Path) -> None:
    """A different profile under the same challenger id is a different hypothesis."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    baseline = _write_card(
        artifacts,
        "2026-week-01-player",
        kickoffs=["2026-09-13T17:00:00+00:00"],
        probabilities=[0.61],
        profile="player",
    )
    with pytest.raises(DataContractError, match="different configuration"):
        record_challenger_decisions(
            artifacts, "stack", baseline, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    card = _write_card(
        artifacts,
        "2026-week-01-a",
        kickoffs=["2026-09-13T17:00:00+00:00"],
        probabilities=[0.61],
    )
    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_challenger_decisions(
            artifacts, "stack", card, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )


def test_unknown_challenger_names_the_registered_ones(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    with pytest.raises(ValueError, match="registered: stack"):
        find_challenger(artifacts, "nope")


def test_artifact_lookup_matches_on_fingerprint_not_on_recency(tmp_path: Path) -> None:
    """The active model's card and the challenger's share one directory namespace.

    Picking the newest directory would silently record the BASELINE's picks as
    the challenger's, which is the failure mode this lookup exists to prevent.
    """

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_card(
        artifacts,
        "2026-week-01-20260908T100000Z",
        kickoffs=["2026-09-13T17:00:00+00:00"],
        probabilities=[0.61],
        profile="weak_stack",
    )
    # A NEWER card from a different (baseline) configuration.
    _write_card(
        artifacts,
        "2026-week-01-20260908T110000Z",
        kickoffs=["2026-09-13T17:00:00+00:00"],
        probabilities=[0.61],
        profile="player",
    )
    entry = find_challenger(artifacts, "stack")
    found = find_challenger_artifact(artifacts, entry, season=2026, week=1)
    assert found is not None and found.name == "2026-week-01-20260908T100000Z"
    assert find_challenger_artifact(artifacts, entry, season=2026, week=2) is None


def test_challenger_ledger_rejects_duplicate_rows(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    row = dict.fromkeys(CHALLENGER_DECISION_COLUMNS, "x")
    frame = pd.DataFrame([row, row])
    atomic_parquet(frame, challenger_ledger_path(artifacts))
    with pytest.raises(DataContractError, match="duplicate rows"):
        load_challenger_decisions(artifacts)


def test_fingerprint_is_stable_across_int_float_and_path_style() -> None:
    other = dict(_MODEL)
    other["ridge_alpha"] = 10
    other["min_train_games"] = 500.0
    other["feature_table"] = "F:\\repo\\data\\processed\\game_features_weak_stack.parquet"
    assert config_fingerprint(other) == config_fingerprint(_MODEL)
    # But a real configuration change moves it.
    assert config_fingerprint({**_MODEL, "ridge_alpha": 1.0}) != config_fingerprint(_MODEL)


def test_a_ledger_without_the_best_pick_column_marks_no_best_pick() -> None:
    """The challenger ledger has no Best Pick flag, and must not grow a phantom one.

    Regression: a boolean column built with ``pd.Series(dtype=bool, index=...)``
    fills with NaN, which reads back True, so every row of a flagless ledger
    looked like the week's Best Pick.
    """

    flagless = _decisions().drop(columns=["is_best_pick"])
    settled = settle_prospective_picks(flagless, _outcomes())
    assert "best_pick" not in prospective_accuracy(settled)
    assert prospective_week_summary(settled).iloc[0]["best_pick_game_id"] is None
