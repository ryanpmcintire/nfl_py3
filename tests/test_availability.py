from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.availability import (
    build_availability_outcomes,
    build_season_lagged_availability_rates,
    canonicalize_availability_rates,
    fixed_unavailability,
    learned_unavailability,
    score_availability_rates,
    summarize_availability_scores,
)
from nfl_ats.data import DataContractError


def _outcomes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2013, 2014, 2015):
        for index in range(40):
            dnp = index < 20
            unavailable = dnp and index % 5 != 0
            rows.append(
                {
                    "season": season,
                    "report_category": "questionable",
                    "practice_category": "dnp" if dnp else "full",
                    "position_group": "skill" if index % 2 else "front",
                    "position": "WR" if index % 2 else "LB",
                    "unavailable": float(unavailable),
                    "fixed_unavailability": 0.35,
                }
            )
    return pd.DataFrame(rows)


def test_season_lagged_rates_ignore_target_season_outcomes() -> None:
    outcomes = _outcomes()
    baseline = build_season_lagged_availability_rates(
        outcomes,
        target_seasons=(2014, 2015),
        combination_prior=20,
        position_prior=100,
    )
    changed_outcomes = outcomes.copy()
    changed_outcomes.loc[changed_outcomes["season"].eq(2015), "unavailable"] = 1.0
    changed = build_season_lagged_availability_rates(
        changed_outcomes,
        target_seasons=(2014, 2015),
        combination_prior=20,
        position_prior=100,
    )
    pd.testing.assert_frame_equal(baseline, changed)
    assert baseline.groupby("target_season")["source_end_season"].max().to_dict() == {
        2014: 2013,
        2015: 2014,
    }


def test_learned_rates_improve_the_synthetic_availability_target() -> None:
    outcomes = _outcomes()
    rates = build_season_lagged_availability_rates(
        outcomes,
        target_seasons=(2014, 2015),
    )
    scored = score_availability_rates(outcomes.loc[outcomes["season"].ge(2014)], rates)
    summary = summarize_availability_scores(scored).set_index("method")
    assert summary.loc["learned", "brier_score"] < summary.loc["fixed", "brier_score"]
    lookup = {
        (
            int(row.target_season),
            str(row.report_category),
            str(row.practice_category),
            str(row.position_group),
        ): float(row.unavailability_probability)
        for row in rates.itertuples(index=False)
    }
    dnp = learned_unavailability(
        lookup,
        target_season=2015,
        report_status="Questionable",
        practice_status="Did Not Participate In Practice",
        position="WR",
    )
    full = learned_unavailability(
        lookup,
        target_season=2015,
        report_status="Questionable",
        practice_status="Full Participation in Practice",
        position="WR",
    )
    assert dnp is not None and full is not None and dnp > full
    assert fixed_unavailability("Questionable", "Full Participation in Practice") == 0.35


def test_availability_outcomes_use_cutoff_and_missing_snap_as_unavailable() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["2022_01_B_A"],
            "season": [2022],
            "week": [1],
            "home_team": ["A"],
            "away_team": ["B"],
            "kickoff": ["2022-09-11T17:00:00Z"],
        }
    )
    injuries = pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "week": [1, 1, 1],
            "team": ["A", "A", "B"],
            "gsis_id": ["P1", "P1", "P2"],
            "position": ["WR", "WR", "LB"],
            "report_status": ["Questionable", "Out", "Questionable"],
            "practice_status": ["Limited", "DNP", "Full"],
            "date_modified": [
                "2022-09-09T12:00:00Z",
                "2022-09-11T16:00:00Z",
                "2022-09-09T12:00:00Z",
            ],
        }
    )
    snaps = pd.DataFrame(
        {
            "season": [2022],
            "week": [1],
            "team": ["A"],
            "gsis_id": ["P1"],
            "offense_snaps": [10],
            "defense_snaps": [0],
            "st_snaps": [0],
        }
    )
    outcomes = build_availability_outcomes(injuries, snaps, games)
    assert len(outcomes) == 2
    assert not bool(outcomes.loc[outcomes["gsis_id"].eq("P1"), "unavailable"].iloc[0])
    assert bool(outcomes.loc[outcomes["gsis_id"].eq("P2"), "unavailable"].iloc[0])
    assert outcomes.loc[outcomes["gsis_id"].eq("P1"), "report_category"].iloc[0] == ("questionable")

    older_injury = injuries.iloc[[0]].copy()
    older_injury["season"] = 2021
    older_injury["date_modified"] = "2021-09-09T12:00:00Z"
    older_game = games.copy()
    older_game["game_id"] = "2021_01_B_A"
    older_game["season"] = 2021
    older_game["kickoff"] = "2021-09-12T17:00:00Z"
    covered_only = build_availability_outcomes(
        pd.concat([injuries, older_injury], ignore_index=True),
        snaps,
        pd.concat([games, older_game], ignore_index=True),
    )
    assert covered_only["season"].eq(2022).all()


def test_availability_rate_contract_rejects_leakage() -> None:
    rates = build_season_lagged_availability_rates(_outcomes(), target_seasons=(2014,))
    leaked = rates.copy()
    leaked["source_end_season"] = leaked["target_season"]
    with pytest.raises(DataContractError, match="earlier"):
        canonicalize_availability_rates(leaked)
    with pytest.raises(ValueError, match="position_prior"):
        build_season_lagged_availability_rates(
            _outcomes(), target_seasons=(2014,), position_prior=-1
        )


def test_fixed_unavailability_is_bit_faithful_to_the_original_heuristic() -> None:
    """The frozen active model's injury prior: legacy strings must keep their
    original (substring-matched) meanings. Routing this through the
    categorized parser silently changed 18 historical games in 2010-2015."""

    assert fixed_unavailability("Out", "Full Participation in Practice") == 1.0
    assert fixed_unavailability("Doubtful", None) == 0.85
    assert fixed_unavailability("Questionable", None) == 0.35
    assert fixed_unavailability("Probable", None) == 0.05

    assert fixed_unavailability(None, "Did Not Participate In Practice") == 0.25
    assert fixed_unavailability(None, "Limited Participation in Practice") == 0.10
    assert fixed_unavailability(None, "Full Participation in Practice") == 0.0

    # Regression: the categorized parser recognizes these; the original
    # heuristic never did, and the frozen model's features depend on that.
    assert fixed_unavailability(None, "Out") == 0.0
    assert fixed_unavailability(None, "Out (Definitely Will Not Play)") == 0.0
    assert fixed_unavailability(None, "DNP") == 0.0
    assert fixed_unavailability("Suspension", "Suspension") == 0.0
    assert fixed_unavailability(None, None) == 0.0
