from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.injury_scenarios import build_injury_scenario_margin_mixture


def _revision(
    revision_id: str = "r1",
    observed_at: str = "2026-09-10T12:00:00Z",
    *,
    probabilities: tuple[float, float] = (0.25, 0.75),
    centers: tuple[float, float] = (0.0, 4.0),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["game", "game"],
            "revision_id": [revision_id, revision_id],
            "scenario_id": ["starter_active", "starter_inactive"],
            "probability": probabilities,
            "predicted_margin": centers,
            "active_player_ids": [("QB1",), ("QB2",)],
            "inactive_player_ids": [("QB2",), ("QB1",)],
            "observed_at_utc": [observed_at, observed_at],
            "effective_at_utc": ["2026-09-10T11:00:00Z"] * 2,
            "source_id": ["injury-snapshot-sha256"] * 2,
        }
    )


def _build(revisions: pd.DataFrame, decision_at: str = "2026-09-10T17:00:00Z"):
    return build_injury_scenario_margin_mixture(
        revisions,
        game_id="game",
        decision_at_utc=decision_at,
        spread_line=3.0,
        residuals=[-1.0, 1.0],
    )


def test_exact_mixture_moments_and_cdf() -> None:
    mixture = _build(_revision())

    # Component samples are [-1, 1] at p=.25 and [3, 5] at p=.75.
    assert mixture.mean == pytest.approx(3.0)
    assert mixture.variance == pytest.approx(4.0)
    assert mixture.home_win_probability == pytest.approx(0.75)
    assert mixture.home_cover_probability == pytest.approx(5.0 / 12.0)
    assert mixture.home_cover_probability_excluding_push == pytest.approx(0.375)
    assert mixture.push_probability == pytest.approx(0.375)
    assert mixture.home_loss_probability == pytest.approx(0.25)
    assert (
        mixture.home_cover_probability_excluding_push
        + mixture.push_probability
        + mixture.home_loss_probability
    ) == pytest.approx(1.0)
    assert mixture.revision_id == "r1"
    assert mixture.source_id == "injury-snapshot-sha256"
    assert mixture.components[0].active_player_ids == ("QB1",)
    assert mixture.components[1].inactive_player_ids == ("QB1",)


def test_future_revision_cannot_change_decision_time_mixture() -> None:
    current = _revision()
    future = _revision(
        "r2",
        "2026-09-10T18:00:00Z",
        probabilities=(0.9, 0.1),
        centers=(100.0, -100.0),
    )
    revisions = pd.concat([current, future], ignore_index=True)

    assert _build(revisions) == _build(current)
    later = _build(revisions, "2026-09-10T19:00:00Z")
    assert later.revision_id == "r2"
    assert later.mean != pytest.approx(_build(current).mean)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("probability", "sum to one"),
        ("duplicate_signature", "duplicate state signature"),
        ("incomplete_partition", "full partition"),
        ("active_inactive_overlap", "active and inactive"),
    ),
)
def test_malformed_or_overlapping_scenario_sets_fail_closed(case: str, message: str) -> None:
    revisions = _revision()
    if case == "probability":
        revisions["probability"] = [0.2, 0.2]
    elif case == "duplicate_signature":
        revisions.at[1, "active_player_ids"] = ("QB1",)
        revisions.at[1, "inactive_player_ids"] = ("QB2",)
    elif case == "incomplete_partition":
        revisions.at[1, "inactive_player_ids"] = ()
    else:
        revisions.at[0, "inactive_player_ids"] = ("QB1", "QB2")

    with pytest.raises(DataContractError, match=message):
        _build(revisions)


def test_missing_or_ambiguous_visible_revision_fails_closed() -> None:
    future = _revision(observed_at="2026-09-11T12:00:00Z")
    with pytest.raises(DataContractError, match="No injury scenario revision is visible"):
        _build(future)

    duplicate = _revision("r2")
    ambiguous = pd.concat([_revision(), duplicate], ignore_index=True)
    with pytest.raises(DataContractError, match="ambiguous latest provenance"):
        _build(ambiguous)
