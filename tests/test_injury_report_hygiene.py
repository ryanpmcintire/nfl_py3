"""Release-blocking tests for the LEAD-18/LEAD-19 injury-report hygiene module.

No network access; every frame is synthetic. Mirrors
docs/injury_report_hygiene.md's frozen population definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.injury_report_hygiene import (
    COACH_TEAM_DECISION_DESIGNATIONS,
    CONCUSSION_LINE_POSITIONS,
    CONCUSSION_SKILL_POSITIONS,
    DID_NOT_PARTICIPATE_STATUS,
    ILLNESS_DESIGNATIONS,
    OTHER_NON_INJURY_DESIGNATIONS,
    PERSONAL_MATTER_DESIGNATIONS,
    REST_DAY_DESIGNATIONS,
    attach_played_outcome,
    build_player_week_frame,
    canonical_text,
    canonicalize_injury_text_rows,
    classify_designation,
    concussion_position_group,
    rate_gap_to_dict,
    season_block_bootstrap_gap,
    select_earliest_revision_per_player_week,
)

# ---------------------------------------------------------------------------
# Frozen designation string classification
# ---------------------------------------------------------------------------


def test_every_frozen_string_classifies_into_its_own_bucket() -> None:
    for text in PERSONAL_MATTER_DESIGNATIONS:
        assert classify_designation(text, None) == "personal_matter"
        assert classify_designation(None, text) == "personal_matter"
    for text in REST_DAY_DESIGNATIONS:
        assert classify_designation(text, None) == "rest_day"
    for text in ILLNESS_DESIGNATIONS:
        assert classify_designation(text, None) == "illness"
    for text in COACH_TEAM_DECISION_DESIGNATIONS:
        assert classify_designation(text, None) == "coach_team_decision"
    for text in OTHER_NON_INJURY_DESIGNATIONS:
        assert classify_designation(text, None) == "other_non_injury"


def test_frozen_designation_buckets_are_disjoint() -> None:
    buckets = (
        PERSONAL_MATTER_DESIGNATIONS,
        REST_DAY_DESIGNATIONS,
        ILLNESS_DESIGNATIONS,
        COACH_TEAM_DECISION_DESIGNATIONS,
        OTHER_NON_INJURY_DESIGNATIONS,
    )
    for i, first in enumerate(buckets):
        for second in buckets[i + 1 :]:
            assert not (first & second)


def test_genuine_injury_text_classifies_as_injury() -> None:
    assert classify_designation("hamstring", None) == "injury"
    assert classify_designation("concussion", "concussion") == "injury"
    assert classify_designation(None, None) == "injury"


def test_compound_and_narrative_strings_are_not_guessed_into_personal_matter() -> None:
    """Exact-match only: a body-part/personal compound or a narrative sentence

    is deliberately NOT claimed by any frozen bucket (docs/injury_report_hygiene.md
    section 2) -- it falls through to "injury", the conservative direction for
    an exclusion audit.
    """

    compound = "ankle [not injury related - personal, thursday only]"
    narrative = "did not travel to brazil due to a personal matter and will not play in the game."
    assert classify_designation(compound, None) == "injury"
    assert classify_designation(narrative, None) == "injury"


def test_canonical_text_lowercases_strips_and_handles_missing() -> None:
    assert canonical_text("  Not Injury Related - Personal Matter  ") == (
        "not injury related - personal matter"
    )
    assert canonical_text(None) is None
    assert canonical_text(pd.NA) is None
    assert canonical_text(float("nan")) is None
    assert canonical_text("") is None


# ---------------------------------------------------------------------------
# Position grouping
# ---------------------------------------------------------------------------


def test_concussion_position_group_matches_frozen_predeclared_sets() -> None:
    for position in CONCUSSION_SKILL_POSITIONS:
        assert concussion_position_group(position) == "skill"
        assert concussion_position_group(position.lower()) == "skill"
    for position in CONCUSSION_LINE_POSITIONS:
        assert concussion_position_group(position) == "line"
    for position in ("CB", "LB", "S", "SAF", "FS", "SS"):
        assert concussion_position_group(position) == "other"


# ---------------------------------------------------------------------------
# Canonicalization + earliest-revision selection
# ---------------------------------------------------------------------------

_INJURY_COLUMNS = (
    "season",
    "game_type",
    "team",
    "week",
    "gsis_id",
    "position",
    "report_status",
    "report_primary_injury",
    "practice_status",
    "practice_primary_injury",
    "date_modified",
)


def _injury_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2013,
        "game_type": "REG",
        "team": "AAA",
        "week": 1,
        "gsis_id": "P1",
        "position": "QB",
        "report_status": "Out",
        "report_primary_injury": "Concussion",
        "practice_status": DID_NOT_PARTICIPATE_STATUS,
        "practice_primary_injury": "Concussion",
        "date_modified": pd.Timestamp("2013-09-04", tz="UTC"),
    }
    row.update(overrides)
    return row


def test_canonicalize_injury_text_rows_keeps_regular_season_only_by_default() -> None:
    frame = pd.DataFrame(
        [
            _injury_row(),
            _injury_row(gsis_id="P2", game_type="WC"),
        ]
    )
    canonical = canonicalize_injury_text_rows(frame)
    assert list(canonical["gsis_id"]) == ["P1"]


def test_canonicalize_injury_text_rows_requires_all_columns() -> None:
    frame = pd.DataFrame([_injury_row()]).drop(columns=["practice_primary_injury"])
    with pytest.raises(DataContractError):
        canonicalize_injury_text_rows(frame)


def test_select_earliest_revision_per_player_week_keeps_earliest_and_counts_multi() -> None:
    frame = pd.DataFrame(
        [
            _injury_row(report_status="Out", date_modified=pd.Timestamp("2013-09-04", tz="UTC")),
            _injury_row(
                report_status="Doubtful", date_modified=pd.Timestamp("2013-09-05", tz="UTC")
            ),
            _injury_row(gsis_id="P2", date_modified=pd.Timestamp("2013-09-04", tz="UTC")),
        ]
    )
    earliest, multi_revision_count = select_earliest_revision_per_player_week(frame)
    assert multi_revision_count == 1
    assert len(earliest) == 2
    p1_row = earliest.loc[earliest["gsis_id"] == "P1"].iloc[0]
    assert p1_row["report_status"] == "Out"  # earliest revision kept, not the later "Doubtful"


# ---------------------------------------------------------------------------
# Sunday-action join
# ---------------------------------------------------------------------------

_ROSTER_COLUMNS = (
    "season",
    "team",
    "position",
    "status",
    "full_name",
    "gsis_id",
    "pfr_id",
    "years_exp",
    "week",
    "game_type",
)
_SNAP_COLUMNS = (
    "game_id",
    "season",
    "game_type",
    "week",
    "player",
    "pfr_player_id",
    "position",
    "team",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
)


def _roster_row(gsis_id: str, pfr_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2013,
        "team": "AAA",
        "position": "QB",
        "status": "ACT",
        "full_name": f"Player {gsis_id}",
        "gsis_id": gsis_id,
        "pfr_id": pfr_id,
        "years_exp": 3,
        "week": 1,
        "game_type": "REG",
    }
    row.update(overrides)
    return row


def _snap_row(pfr_player_id: str, offense_snaps: float, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": "2013_01_AAA_XXX",
        "season": 2013,
        "game_type": "REG",
        "week": 1,
        "player": f"Player {pfr_player_id}",
        "pfr_player_id": pfr_player_id,
        "position": "QB",
        "team": "AAA",
        "offense_snaps": offense_snaps,
        "offense_pct": 0.5,
        "defense_snaps": 0.0,
        "defense_pct": 0.0,
        "st_snaps": 0.0,
        "st_pct": 0.0,
    }
    row.update(overrides)
    return row


def test_attach_played_outcome_true_when_snaps_present_false_when_absent() -> None:
    population = pd.DataFrame(
        [
            _injury_row(gsis_id="P1"),
            _injury_row(gsis_id="P2"),
        ]
    )
    rosters = pd.DataFrame([_roster_row("P1", "PFR_P1"), _roster_row("P2", "PFR_P2")])
    snaps = pd.DataFrame([_snap_row("PFR_P1", 10.0)])  # only P1 recorded snaps

    result = attach_played_outcome(population, snaps, rosters)
    played_by_player = dict(zip(result["gsis_id"], result["played"], strict=True))
    assert played_by_player["P1"] is True
    assert played_by_player["P2"] is False


# ---------------------------------------------------------------------------
# Season-blocked bootstrap
# ---------------------------------------------------------------------------


def _rate_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for season in (2013, 2014, 2015, 2016):
        for index in range(30):
            rows.append(
                {
                    "season": season,
                    "group": "a" if index < 15 else "b",
                    # group "a" outcomes average ~0.8, group "b" ~0.3
                    "outcome": float(rng.random() < (0.8 if index < 15 else 0.3)),
                }
            )
    return pd.DataFrame(rows)


def test_season_block_bootstrap_gap_is_deterministic() -> None:
    frame = _rate_frame()
    first = season_block_bootstrap_gap(
        frame, group_column="group", outcome_column="outcome", group_a="a", group_b="b", seed=5
    )
    second = season_block_bootstrap_gap(
        frame, group_column="group", outcome_column="outcome", group_a="a", group_b="b", seed=5
    )
    assert first == second


def test_season_block_bootstrap_gap_recovers_known_positive_gap() -> None:
    frame = _rate_frame()
    result = season_block_bootstrap_gap(
        frame, group_column="group", outcome_column="outcome", group_a="a", group_b="b", seed=5
    )
    assert result.estimate > 0
    assert result.probability_positive > 0.5
    assert result.block_count == 4
    assert result.n_a + result.n_b == len(frame)
    payload = rate_gap_to_dict(result)
    assert payload["ci95"][0] <= payload["estimate"] <= payload["ci95"][1]


def test_season_block_bootstrap_gap_raises_on_empty_group() -> None:
    frame = _rate_frame()
    frame = frame.loc[frame["group"] != "b"].copy()
    with pytest.raises(DataContractError):
        season_block_bootstrap_gap(
            frame, group_column="group", outcome_column="outcome", group_a="a", group_b="b"
        )


# ---------------------------------------------------------------------------
# Leakage: the outcome must never change population/classification columns
# ---------------------------------------------------------------------------


def _leakage_population() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _injury_row(gsis_id="P1", position="QB", report_primary_injury="Concussion"),
            _injury_row(gsis_id="P2", position="T", report_primary_injury="Concussion"),
            _injury_row(
                gsis_id="P3",
                position="WR",
                report_primary_injury="Not injury related - personal matter",
            ),
            _injury_row(gsis_id="P4", position="RB", report_primary_injury="Hamstring"),
            _injury_row(gsis_id="P5", position="LB", report_primary_injury="Rest"),
        ]
    )


def _leakage_rosters() -> pd.DataFrame:
    return pd.DataFrame(
        [_roster_row(gsis_id, f"PFR_{gsis_id}") for gsis_id in ("P1", "P2", "P3", "P4", "P5")]
    )


def test_outcome_never_changes_the_population_or_designation_columns() -> None:
    injuries = _leakage_population()
    rosters = _leakage_rosters()
    all_played_snaps = pd.DataFrame(
        [
            _snap_row(f"PFR_{gsis_id}", 10.0, player=f"Player {gsis_id}")
            for gsis_id in ("P1", "P2", "P3", "P4", "P5")
        ]
    )
    # A snap table for an unrelated player: every population row is absent
    # from it, so every population row resolves played=False.
    none_played_snaps = pd.DataFrame([_snap_row("PFR_UNRELATED", 10.0, player="Unrelated")])

    frame_all_played, _ = build_player_week_frame(
        injuries, all_played_snaps, rosters, season_start=2013, season_end=2013
    )
    frame_none_played, _ = build_player_week_frame(
        injuries, none_played_snaps, rosters, season_start=2013, season_end=2013
    )

    assert bool(frame_all_played["played"].all())
    assert not bool(frame_none_played["played"].any())

    population_columns = [
        "gsis_id",
        "position",
        "practice_status",
        "report_canon",
        "practice_canon",
        "designation",
        "is_concussion_report",
        "concussion_group",
    ]
    left = frame_all_played.sort_values("gsis_id")[population_columns].reset_index(drop=True)
    right = frame_none_played.sort_values("gsis_id")[population_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)

    designations = dict(zip(left["gsis_id"], left["designation"], strict=True))
    assert designations == {
        "P1": "injury",  # concussion is a genuine injury designation
        "P2": "injury",
        "P3": "personal_matter",
        "P4": "injury",
        "P5": "rest_day",
    }
