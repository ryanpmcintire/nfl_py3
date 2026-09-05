from __future__ import annotations

import hashlib

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
from nfl_ats.players import canonicalize_injuries


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


# ---------------------------------------------------------------------------
# ENG-39 follow-up: build_availability_outcomes must use the same
# effective_observed_at-else-date_modified visibility rule as
# nfl_ats.players._injury_rows_asof, so a season only made visible via the
# leakage-safe week_proxy fallback (e.g. 2025, where nflverse drops
# date_modified entirely) is not silently excluded from the learned
# availability rates this function feeds. See
# docs/injury_timestamp_fallback.md and the lane S2 report referenced there.
# ---------------------------------------------------------------------------


def _proxy_schedule(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "season": 2025,
        "week": 2,
        "home_team": "A",
        "away_team": "B",
        "kickoff": pd.Timestamp("2025-09-15T17:00:00Z"),
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _proxy_games(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "game_id": ["2025_02_B_A"],
        "season": [2025],
        "week": [2],
        "home_team": ["A"],
        "away_team": ["B"],
        "kickoff": ["2025-09-15T17:00:00Z"],
    }
    row.update(overrides)
    return pd.DataFrame(row)


def _2025_shaped_injury_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2025,
        "game_type": "REG",
        "team": "A",
        "week": 2,
        "gsis_id": "P1",
        "position": "WR",
        "report_status": "Questionable",
        "practice_status": "Limited Participation in Practice",
        # No "date_modified" column at all -- the real 2025 nflverse shape
        # (docs/injury_timestamp_fallback.md M1).
    }
    row.update(overrides)
    return row


def test_availability_outcomes_plain_frame_is_byte_identical_to_pre_eng39() -> None:
    """A frame with no ``effective_observed_at`` column (every snapshot built
    before this change, and any frame canonicalized with the default
    ``timestamp_fallback="drop"``) must keep filtering on ``date_modified``
    exactly as before -- pinned with a hash so an accidental regression on
    this path fails loudly here."""

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
    assert "effective_observed_at" not in injuries.columns
    outcomes = build_availability_outcomes(injuries, snaps, games)
    assert len(outcomes) == 2  # unchanged from the pre-existing cutoff test above

    digest = hashlib.sha256(
        pd.util.hash_pandas_object(outcomes, index=True).to_numpy().tobytes()
    ).hexdigest()
    assert digest == "ca5f6e5cbdda0c5b63984b68b04712fa75fa80629e40650b6e5e14264c1c4649"


def test_availability_outcomes_prefers_effective_observed_at_for_a_proxied_2025_row() -> None:
    """THE GAP (lane S2, docs/injury_timestamp_fallback.md 'separate, unfixed
    gap'): a season made visible only via the week_proxy fallback must still
    produce an availability outcome, not be silently excluded."""

    schedule = _proxy_schedule()
    games = _proxy_games()
    raw_injuries = pd.DataFrame([_2025_shaped_injury_row()])
    assert "date_modified" not in raw_injuries.columns

    canonical = canonicalize_injuries(
        raw_injuries, timestamp_fallback="week_proxy", schedule=schedule
    )
    assert canonical.loc[0, "observed_at_basis"] == "week_proxy"

    snaps = pd.DataFrame(
        {
            "season": [2025],
            "week": [2],
            "team": ["A"],
            "gsis_id": ["P1"],
            "offense_snaps": [0],
            "defense_snaps": [0],
            "st_snaps": [0],
        }
    )
    outcomes = build_availability_outcomes(
        canonical, snaps, games, decision_hours_before_kickoff=24
    )
    assert len(outcomes) == 1
    assert outcomes.loc[0, "gsis_id"] == "P1"
    assert bool(outcomes.loc[0, "unavailable"])  # zero snaps logged -> unavailable

    # The default "drop" mode never sees this row: canonicalize_injuries
    # itself already raises before build_availability_outcomes is reached
    # (M1's exact failure mode), reproduced here for contrast.
    with pytest.raises(DataContractError):
        canonicalize_injuries(raw_injuries)


def test_availability_outcomes_leakage_proxied_row_invisible_before_its_proxy_time() -> None:
    """AGENTS.md: a proxied row must never become an outcome before its own
    effective_observed_at. The proxy here is exactly kickoff-24h; a cutoff
    computed from decision_hours_before_kickoff=48 sits BEFORE that proxy
    time and must see nothing, while decision_hours_before_kickoff=1 sits
    AFTER it and must see the row."""

    schedule = _proxy_schedule()
    games = _proxy_games()
    raw_injuries = pd.DataFrame([_2025_shaped_injury_row()])
    canonical = canonicalize_injuries(
        raw_injuries, timestamp_fallback="week_proxy", schedule=schedule
    )
    proxy_at = pd.Timestamp("2025-09-15T17:00:00Z") - pd.Timedelta(hours=24)
    assert canonical.loc[0, "effective_observed_at"] == proxy_at

    snaps = pd.DataFrame(
        {
            "season": [2025],
            "week": [2],
            "team": ["A"],
            "gsis_id": ["P1"],
            "offense_snaps": [10],
            "defense_snaps": [0],
            "st_snaps": [0],
        }
    )

    too_early = build_availability_outcomes(
        canonical, snaps, games, decision_hours_before_kickoff=48
    )
    assert too_early.empty

    visible = build_availability_outcomes(canonical, snaps, games, decision_hours_before_kickoff=1)
    assert len(visible) == 1
    assert not bool(visible.loc[0, "unavailable"])  # logged snaps -> played


def test_availability_outcomes_never_overwrites_a_real_date_modified() -> None:
    """A row with a real ``date_modified`` (``observed_at_basis ==
    "date_modified"``) must keep using that real timestamp, not the proxy
    machinery, even when ``effective_observed_at`` is present on the frame.

    Chosen so the real timestamp and the would-be proxy time
    (``kickoff - 24h == 2025-09-14T17:00:00Z``) fall on opposite sides of
    the cutoff below: visible only if the real ``date_modified`` -- not a
    proxy -- is what actually gates visibility.
    """

    schedule = _proxy_schedule()
    games = _proxy_games()
    real_timestamp = "2025-09-14T12:00:00Z"
    raw_injuries = pd.DataFrame([_2025_shaped_injury_row(date_modified=real_timestamp)])
    canonical = canonicalize_injuries(
        raw_injuries, timestamp_fallback="week_proxy", schedule=schedule
    )
    assert canonical.loc[0, "observed_at_basis"] == "date_modified"
    assert canonical.loc[0, "effective_observed_at"] == pd.Timestamp(real_timestamp)

    snaps = pd.DataFrame(
        {
            "season": [2025],
            "week": [2],
            "team": ["A"],
            "gsis_id": ["P1"],
            "offense_snaps": [10],
            "defense_snaps": [0],
            "st_snaps": [0],
        }
    )
    # cutoff = kickoff - 25h = 2025-09-14T16:00:00Z: after the real
    # timestamp (12:00) but before the would-be proxy time (17:00).
    outcomes = build_availability_outcomes(
        canonical, snaps, games, decision_hours_before_kickoff=25
    )
    assert len(outcomes) == 1


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
