"""Identity and leakage contracts for the shared opener-confirmation runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import on_production_opener_confirmation as confirmation  # noqa: E402

from nfl_ats.margin import margin_feature_columns  # noqa: E402


def test_profile_identity_is_production_plus_the_declared_one_column() -> None:
    candidate = confirmation.CANDIDATES["pace"]
    columns = set(margin_feature_columns("market_residual", candidate.profile))
    frame = pd.DataFrame({column: [0.0] for column in columns})

    observed = confirmation.profile_identity(candidate, frame)

    assert observed == {
        "baseline_columns": len(margin_feature_columns("market_residual", "weak_stack")),
        "candidate_columns": len(columns),
        "only_added_column": "team_style_pace_mismatch_flag",
    }


def test_scoped_window_refuses_training_that_reaches_the_screen(monkeypatch) -> None:
    training = pd.DataFrame({"season": [2019], "gameday": ["2020-09-01"], "result": [1]})
    window = pd.DataFrame({"season": [2020], "gameday": ["2020-09-01"], "result": [1]})
    monkeypatch.setattr(confirmation, "confirmation_split", lambda *_: (training, window))

    try:
        confirmation.scoped_window_frame(pd.DataFrame(), object(), "family")
    except ValueError as error:
        assert "leaked" in str(error)
    else:  # pragma: no cover - an assertion gives a clearer failure than a silent pass
        raise AssertionError("same-day training must be rejected")


def test_only_positive_control_replaces_the_candidate_column(monkeypatch) -> None:
    candidate = confirmation.CANDIDATES["illness"]
    source = pd.DataFrame({"season": [2020], "ats_margin": [7.0], candidate.column: [0.0]})
    captured: list[pd.DataFrame] = []

    def fake_evaluation(_market_root, frame, **_kwargs):
        captured.append(frame.copy())
        return pd.DataFrame({"season": [2020]})

    monkeypatch.setattr(confirmation, "opener_pick_evaluation", fake_evaluation)
    confirmation.run_arm(
        source,
        candidate,
        market_root=Path("market"),
        profile=candidate.profile,
        seasons=(2020,),
        min_train_games=1,
        leak=False,
    )
    confirmation.run_arm(
        source,
        candidate,
        market_root=Path("market"),
        profile=candidate.profile,
        seasons=(2020,),
        min_train_games=1,
        leak=True,
    )

    assert captured[0][candidate.column].tolist() == [0.0]
    assert captured[1][candidate.column].tolist() == [7.0]
    assert source[candidate.column].tolist() == [0.0]
