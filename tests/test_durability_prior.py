"""Leakage, shrinkage and join contracts for the PER-13 durability prior.

AGENTS.md requires a leakage regression test for every new pregame feature
family. The first four tests below are that regression: they assert the prior
for a game can never move when information that did not exist at that game's
decision time changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.durability_prior import (
    DURABILITY_COLUMNS,
    LOGIT_CLIP,
    MAX_PRIOR_STRENGTH,
    RESERVE_STATUSES,
    DurabilityCalibration,
    DurabilityHistory,
    attach_durability_prior,
    beta_binomial_prior_strength,
    clipped_logit,
    durability_prior_columns,
    residual_prior_strength,
    split_half_reliability,
)

KICKOFF = pd.Timestamp("2016-09-11 17:00", tz="UTC")


def _outcome_rows(
    player: str,
    *,
    seasons: list[int],
    weeks: list[int],
    unavailable: list[float],
    report: list[str] | None = None,
    cell: float = 0.3,
) -> pd.DataFrame:
    kickoffs = [
        pd.Timestamp("2013-09-08 17:00", tz="UTC")
        + pd.Timedelta(days=365 * (season - 2013) + 7 * (week - 1))
        for season, week in zip(seasons, weeks, strict=True)
    ]
    return pd.DataFrame(
        {
            "gsis_id": player,
            "season": seasons,
            "week": weeks,
            "kickoff": kickoffs,
            "position_group": "skill",
            "report_category": report if report is not None else ["questionable"] * len(seasons),
            "unavailable": unavailable,
            "cell_probability": cell,
        }
    )


def _roster_rows(player: str, seasons: list[int], weeks: list[int], status: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": player,
            "season": seasons,
            "week": weeks,
            "position_group": "skill",
            "status": status,
            "played": [status == "ACT"] * len(seasons),
            "snap_covered": True,
        }
    )


def _history(outcomes: pd.DataFrame, rosters: pd.DataFrame | None = None) -> DurabilityHistory:
    if rosters is None:
        rosters = pd.DataFrame(
            {
                "gsis_id": pd.Series(dtype="string"),
                "season": pd.Series(dtype="int64"),
                "week": pd.Series(dtype="int64"),
                "position_group": pd.Series(dtype="string"),
                "status": pd.Series(dtype="string"),
                "played": pd.Series(dtype="bool"),
                "snap_covered": pd.Series(dtype="bool"),
            }
        )
    return DurabilityHistory(outcomes=outcomes, rosters=rosters)


def _targets(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.loc[:, ["gsis_id", "season", "week", "position_group", "kickoff"]].copy()
    rows["decision_cutoff"] = pd.to_datetime(rows["kickoff"], utc=True) - pd.Timedelta(hours=24)
    return rows.drop(columns="kickoff").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1-4: leakage
# ---------------------------------------------------------------------------


def test_future_season_outcomes_never_move_an_earlier_prior() -> None:
    """The frozen leakage contract: rewrite the future, the past must not move."""

    seasons = [2013, 2014, 2015, 2016, 2017, 2018]
    weeks = [1, 1, 1, 1, 1, 1]
    baseline = _outcome_rows("A", seasons=seasons, weeks=weeks, unavailable=[0, 0, 1, 0, 0, 0])
    rewritten = baseline.copy()
    future = rewritten["season"] > 2016
    rewritten.loc[future, "unavailable"] = 1.0 - rewritten.loc[future, "unavailable"]
    assert not baseline.equals(rewritten)

    targets = _targets(baseline)
    before_or_at = targets["season"] <= 2016

    original = durability_prior_columns(
        _history(baseline).aggregates(targets), _history(baseline).calibration(before_season=2016)
    )
    flipped = durability_prior_columns(
        _history(rewritten).aggregates(targets),
        _history(baseline).calibration(before_season=2016),
    )
    pd.testing.assert_frame_equal(original.loc[before_or_at], flipped.loc[before_or_at])


def test_a_row_never_sees_its_own_outcome() -> None:
    """A debutant is scored by the designation cell alone: all six columns are 0."""

    frame = _outcome_rows("A", seasons=[2016], weeks=[1], unavailable=[1.0])
    history = _history(frame)
    aggregates = history.aggregates(_targets(frame))
    assert float(aggregates.loc[0, "rate_n"]) == 0.0
    columns = durability_prior_columns(aggregates, history.calibration(before_season=2016))
    assert list(columns.columns) == list(DURABILITY_COLUMNS)
    assert columns.iloc[0].abs().max() == pytest.approx(0.0)


def test_history_boundary_is_the_kickoff_not_the_season() -> None:
    """A game inside the 24-hour decision window is excluded even in the same week."""

    frame = pd.DataFrame(
        {
            "gsis_id": ["A", "A"],
            "season": [2016, 2016],
            "week": [3, 3],
            "kickoff": [KICKOFF, KICKOFF + pd.Timedelta(hours=12)],
            "position_group": "skill",
            "report_category": "questionable",
            "unavailable": [1.0, 0.0],
            "cell_probability": 0.3,
        }
    )
    history = _history(frame)
    targets = _targets(frame)
    aggregates = history.aggregates(targets)
    # The second row's cutoff is 12 hours BEFORE the first row's kickoff, so
    # neither row may see the other.
    assert aggregates["rate_n"].tolist() == [0.0, 0.0]

    later = targets.iloc[[1]].copy()
    later["decision_cutoff"] = KICKOFF + pd.Timedelta(days=7)
    assert float(history.aggregates(later).iloc[0]["rate_n"]) == 2.0


def test_roster_history_is_strictly_earlier_season_week() -> None:
    rosters = pd.concat(
        [
            _roster_rows("A", [2016, 2016, 2016], [1, 2, 3], "RES"),
            _roster_rows("A", [2016], [4], "ACT"),
        ],
        ignore_index=True,
    )
    outcomes = _outcome_rows("A", seasons=[2016], weeks=[4], unavailable=[0.0])
    history = DurabilityHistory(outcomes=outcomes, rosters=rosters)
    aggregates = history.aggregates(_targets(outcomes))
    # Weeks 1-3 count; the target's own week 4 row does not.
    assert float(aggregates.loc[0, "reserve_n"]) == 3.0
    assert float(aggregates.loc[0, "reserve_k"]) == 3.0
    assert float(aggregates.loc[0, "roster_absent"]) == 3.0


def test_reserve_statuses_carry_suspensions() -> None:
    assert "SUS" in RESERVE_STATUSES
    assert "ACT" not in RESERVE_STATUSES
    assert "DEV" not in RESERVE_STATUSES


# ---------------------------------------------------------------------------
# 5: shrinkage math
# ---------------------------------------------------------------------------


def test_beta_binomial_prior_strength_recovers_a_planted_dispersion() -> None:
    generator = np.random.default_rng(11)
    trials = np.full(4_000, 40.0)
    rates = generator.beta(2.0, 6.0, size=trials.size)  # mean 0.25, strength 8
    successes = generator.binomial(trials.astype(int), rates).astype(float)
    strength = beta_binomial_prior_strength(successes, trials)
    assert 6.0 < strength < 11.0


def test_beta_binomial_prior_strength_caps_when_dispersion_is_binomial() -> None:
    generator = np.random.default_rng(3)
    trials = np.full(2_000, 30.0)
    successes = generator.binomial(30, 0.25, size=trials.size).astype(float)
    assert beta_binomial_prior_strength(successes, trials) == MAX_PRIOR_STRENGTH


def test_residual_prior_strength_matches_the_variance_ratio() -> None:
    generator = np.random.default_rng(5)
    players = np.repeat(np.arange(1_500), 30)
    effects = generator.normal(0.0, 0.10, size=1_500)[players]
    residuals = effects + generator.normal(0.0, 0.40, size=players.size)
    strength = residual_prior_strength(residuals, players)
    expected = 0.40**2 / 0.10**2
    assert 0.7 * expected < strength < 1.4 * expected


def test_zero_history_shrinks_exactly_to_the_group_and_long_history_to_the_rate() -> None:
    aggregates = pd.DataFrame(
        {
            "position_group": ["skill", "skill"],
            "rate_n": [0.0, 100_000.0],
            "rate_k": [0.0, 90_000.0],
            "resid_n": [0.0, 0.0],
            "resid_sum": [0.0, 0.0],
            "active_n": [0.0, 0.0],
            "active_sum": [0.0, 0.0],
            "roster_n": [0.0, 0.0],
            "roster_absent": [0.0, 0.0],
            "reserve_n": [0.0, 0.0],
            "reserve_k": [0.0, 0.0],
        }
    )
    calibration = DurabilityCalibration(
        before_season=2016,
        residual_strength=5.0,
        active_residual_strength=5.0,
        rate_strength={"skill": 5.0, "__pooled__": 5.0},
        rate_group_rate={"skill": 0.30, "__pooled__": 0.30},
        absence_strength={"__pooled__": 5.0},
        absence_group_rate={"__pooled__": 0.10},
        reserve_strength={"__pooled__": 5.0},
        reserve_group_rate={"__pooled__": 0.10},
    )
    columns = durability_prior_columns(aggregates, calibration)
    assert columns.loc[0, "durability_rate_logit_offset"] == pytest.approx(0.0)
    expected = float(clipped_logit(np.array([0.9]))[0] - clipped_logit(np.array([0.30]))[0])
    assert columns.loc[1, "durability_rate_logit_offset"] == pytest.approx(expected, abs=1e-3)


def test_clipped_logit_never_returns_an_infinity() -> None:
    values = clipped_logit(np.array([0.0, 1.0, 0.5]))
    assert np.isfinite(values).all()
    assert values[0] == pytest.approx(np.log(LOGIT_CLIP / (1 - LOGIT_CLIP)))


# ---------------------------------------------------------------------------
# 6: join correctness
# ---------------------------------------------------------------------------


def test_attach_preserves_rows_order_and_index() -> None:
    frame = _outcome_rows(
        "A", seasons=[2013, 2014, 2015], weeks=[1, 1, 1], unavailable=[1.0, 0.0, 1.0]
    )
    history = _history(frame)
    targets = _targets(frame).sample(frac=1.0, random_state=2)
    attached = attach_durability_prior(targets, history, history.calibration(before_season=2016))
    assert len(attached) == len(targets)
    pd.testing.assert_index_equal(attached.index, targets.index)
    pd.testing.assert_series_equal(attached["gsis_id"], targets["gsis_id"])
    assert set(DURABILITY_COLUMNS).issubset(attached.columns)


def test_attach_refuses_to_overwrite_existing_columns() -> None:
    frame = _outcome_rows("A", seasons=[2013], weeks=[1], unavailable=[0.0])
    history = _history(frame)
    targets = _targets(frame)
    targets["durability_residual"] = 0.0
    with pytest.raises(DataContractError, match="already present"):
        attach_durability_prior(targets, history, history.calibration(before_season=2016))


def test_two_players_do_not_share_history() -> None:
    frame = pd.concat(
        [
            _outcome_rows("A", seasons=[2013, 2014], weeks=[1, 1], unavailable=[1.0, 1.0]),
            _outcome_rows("B", seasons=[2013, 2014], weeks=[1, 1], unavailable=[0.0, 0.0]),
        ],
        ignore_index=True,
    )
    later = pd.concat(
        [
            _outcome_rows("A", seasons=[2015], weeks=[1], unavailable=[0.0]),
            _outcome_rows("B", seasons=[2015], weeks=[1], unavailable=[0.0]),
        ],
        ignore_index=True,
    )
    history = _history(pd.concat([frame, later], ignore_index=True))
    aggregates = history.aggregates(_targets(later))
    assert aggregates["rate_k"].tolist() == [2.0, 0.0]
    columns = durability_prior_columns(aggregates, history.calibration(before_season=2015))
    offsets = columns["durability_rate_logit_offset"]
    assert offsets.iloc[0] > offsets.iloc[1]


def test_history_rejects_a_missing_kickoff() -> None:
    frame = _outcome_rows("A", seasons=[2013], weeks=[1], unavailable=[0.0])
    frame.loc[0, "kickoff"] = pd.NaT
    with pytest.raises(DataContractError, match="kickoff"):
        _history(frame)


# ---------------------------------------------------------------------------
# Trait reliability helper
# ---------------------------------------------------------------------------


def test_split_half_reliability_is_high_for_a_planted_trait() -> None:
    generator = np.random.default_rng(9)
    players = np.repeat([f"P{index}" for index in range(400)], 40)
    effects = np.repeat(generator.normal(0.0, 0.30, size=400), 40)
    frame = pd.DataFrame(
        {"gsis_id": players, "value": effects + generator.normal(0.0, 0.20, size=players.size)}
    )
    result = split_half_reliability(frame, "value", minimum_per_half=10)
    assert result["players"] == 400.0
    assert result["spearman_brown"] > 0.9


def test_split_half_reliability_is_near_zero_for_pure_noise() -> None:
    generator = np.random.default_rng(4)
    players = np.repeat([f"P{index}" for index in range(400)], 40)
    frame = pd.DataFrame({"gsis_id": players, "value": generator.normal(size=players.size)})
    result = split_half_reliability(frame, "value", minimum_per_half=10)
    assert abs(result["spearman_brown"]) < 0.25
