"""Playcaller-change screen (docs/playcaller_change_leads.md): helper
contracts and the mandatory leakage regressions."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.coordinator_changes import (
    GAMES_AFTER_CHANGE_COLUMNS,
    OC_TENURE_COLUMNS,
    games_after_coordinator_change,
    oc_tenure_at_season_start,
)
from nfl_ats.data import DataContractError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.playcaller_change_screen import (
    FAMILY,
    load_counted_events,
    opener_graded_features,
    r1_flags,
    record_argv,
    score_cell,
    team_game_table,
)


def _games(weeks: int = 6) -> pd.DataFrame:
    rows = []
    for week in range(1, weeks + 1):
        rows.append(
            {
                "game_id": f"2024_{week:02d}_B_A",
                "season": 2024,
                "week": week,
                "kickoff": pd.Timestamp("2024-09-08T17:00:00Z") + pd.Timedelta(days=7 * (week - 1)),
                "home_team": "A",
                "away_team": "B",
            }
        )
    return pd.DataFrame(rows)


def _event(event_id: str, revision_at: str, team: str = "A", role: str = "OC") -> dict[str, object]:
    return {
        "event_id": event_id,
        "season": 2024,
        "team": team,
        "role": role,
        "revision_at": revision_at,
    }


def _history_rows(team: str, names: dict[int, str], *, sample_mode: str = "preseason") -> list:
    return [
        {
            "season": season,
            "team": team,
            "role": "OC",
            "person": name,
            "effective_observed_at": f"{season}-07-15T12:00:00Z",
            "sampled_as_of": f"{season}-09-01T00:00:00Z",
            "sample_mode": sample_mode,
            "observed_at_basis": "wikipedia_revision",
        }
        for season, name in names.items()
    ]


def test_games_after_change_numbers_games_and_restarts_at_second_event() -> None:
    events = pd.DataFrame(
        [
            _event("e1", "2024-09-17T15:00:00Z"),
            _event("e2", "2024-10-01T15:00:00Z", role="DC"),
        ]
    )
    result = games_after_coordinator_change(_games(), events)

    assert list(result.columns) == list(GAMES_AFTER_CHANGE_COLUMNS)
    numbered = result.set_index("game_id")["game_number_after_change"].to_dict()
    assert numbered == {
        "2024_03_B_A": 1,
        "2024_04_B_A": 2,
        "2024_05_B_A": 1,
        "2024_06_B_A": 2,
    }
    by_game = result.set_index("game_id")
    assert by_game.loc["2024_04_B_A", "event_id"] == "e1"
    assert by_game.loc["2024_05_B_A", "event_id"] == "e2"
    assert by_game.loc["2024_05_B_A", "role"] == "DC"
    assert (result["team"] == "A").all()
    assert result["revision_at"].lt(result["kickoff"]).all()


def test_revision_at_or_after_kickoff_never_flags_that_game() -> None:
    at_kickoff = pd.DataFrame([_event("e1", "2024-09-22T17:00:00Z")])
    result = games_after_coordinator_change(_games(), at_kickoff)
    assert "2024_03_B_A" not in set(result["game_id"])
    assert result.set_index("game_id").loc["2024_04_B_A", "game_number_after_change"] == 1

    after_kickoff = pd.DataFrame([_event("e1", "2024-09-22T17:00:01Z")])
    later = games_after_coordinator_change(_games(), after_kickoff)
    assert "2024_03_B_A" not in set(later["game_id"])


def test_later_revision_cannot_change_earlier_rows() -> None:
    """Leakage regression: a revision recorded after games have been played
    (a later correction, or a second change) must leave every earlier
    game's flag, event attribution and number exactly as first computed."""

    baseline = games_after_coordinator_change(
        _games(), pd.DataFrame([_event("e1", "2024-09-17T15:00:00Z")])
    )
    later_events = pd.DataFrame(
        [
            _event("e1", "2024-09-17T15:00:00Z"),
            _event("late_correction", "2024-10-08T15:00:00Z"),
        ]
    )
    with_later = games_after_coordinator_change(_games(), later_events)

    earlier_rows = with_later.loc[with_later["kickoff"].le(pd.Timestamp("2024-10-08T15:00:00Z"))]
    assert set(earlier_rows["game_id"]) == {"2024_03_B_A", "2024_04_B_A", "2024_05_B_A"}
    pd.testing.assert_frame_equal(
        earlier_rows.reset_index(drop=True),
        baseline.loc[baseline["game_id"].isin(earlier_rows["game_id"])].reset_index(drop=True),
        check_exact=True,
    )
    late_rows = with_later.loc[with_later["event_id"].eq("late_correction")]
    assert set(late_rows["game_id"]) == {"2024_06_B_A"}
    assert late_rows["game_number_after_change"].tolist() == [1]


def test_event_with_no_following_game_fails_closed() -> None:
    events = pd.DataFrame([_event("e1", "2024-12-25T00:00:00Z")])
    with pytest.raises(DataContractError, match="no game with a kickoff after"):
        games_after_coordinator_change(_games(), events)


def test_events_only_number_their_own_season_and_team() -> None:
    games = pd.concat(
        [
            _games(),
            _games().assign(
                season=2023, game_id=lambda f: f["game_id"].str.replace("2024", "2023")
            ),
        ],
        ignore_index=True,
    )
    games.loc[games["season"].eq(2023), "kickoff"] -= pd.Timedelta(days=364)
    events = pd.DataFrame([_event("e1", "2024-09-17T15:00:00Z")])
    result = games_after_coordinator_change(games, events)
    assert (result["season"] == 2024).all()
    assert (result["team"] == "A").all()
    with pytest.raises(DataContractError, match="roles"):
        games_after_coordinator_change(
            games, pd.DataFrame([_event("e2", "2024-09-17T15:00:00Z", role="HC")])
        )


def test_oc_tenure_buckets_and_unknowns() -> None:
    history = pd.DataFrame(
        _history_rows("A", {2019: "X", 2020: "X", 2021: "X", 2022: "Y"})
        + _history_rows("B", {2020: "P", 2021: "Q (American football)", 2022: "Q"})
    )
    result = oc_tenure_at_season_start(history)
    assert list(result.columns) == list(OC_TENURE_COLUMNS)
    tenure = {(row.season, row.team): row.oc_tenure_years for row in result.itertuples()}
    assert tenure == {
        (2021, "A"): 3,
        (2022, "A"): 1,
        (2021, "B"): 1,
        (2022, "B"): 2,
    }
    assert (2020, "A") not in tenure
    assert (2019, "A") not in tenure
    assert (2020, "B") not in tenure
    observed = result.set_index(["season", "team"])["observed_at"]
    assert observed.loc[(2022, "A")] == pd.Timestamp("2022-07-15T12:00:00Z")
    assert observed.loc[(2021, "A")] == pd.Timestamp("2021-07-15T12:00:00Z")


def test_oc_tenure_ignores_inseason_rows_and_fails_closed_after_cutoff() -> None:
    """Leakage regression: an in-season revision (the firing that happens in
    November) must not change the season-start tenure, and a preseason row
    observed after its own cutoff is a contract violation, not data."""

    history = pd.DataFrame(_history_rows("A", {2019: "X", 2020: "X", 2021: "X", 2022: "Y"}))
    baseline = oc_tenure_at_season_start(history)

    inseason = pd.DataFrame(
        _history_rows("A", {2021: "Interim Z", 2022: "Interim W"}, sample_mode="inseason")
    )
    inseason["effective_observed_at"] = ["2021-11-10T15:00:00Z", "2022-11-10T15:00:00Z"]
    pd.testing.assert_frame_equal(
        oc_tenure_at_season_start(pd.concat([history, inseason], ignore_index=True)),
        baseline,
    )

    post_cutoff = history.copy()
    post_cutoff.loc[post_cutoff["season"].eq(2021), "effective_observed_at"] = (
        "2021-09-02T00:00:00Z"
    )
    with pytest.raises(DataContractError, match="after requested cutoff"):
        oc_tenure_at_season_start(post_cutoff)


def test_oc_tenure_ambiguous_team_season_is_unknown() -> None:
    history = pd.DataFrame(_history_rows("A", {2019: "X", 2020: "X", 2021: "X"}))
    ambiguous = pd.concat(
        [history, pd.DataFrame(_history_rows("A", {2021: "Someone Else"}))], ignore_index=True
    )
    result = oc_tenure_at_season_start(ambiguous)
    assert result.empty


def test_load_counted_events_keeps_only_supported_adjudications(tmp_path: Path) -> None:
    payload = {
        "changes": [
            {
                "season": 2023,
                "team": "BUF",
                "role": "OC",
                "person": "New",
                "previous_person": "Old",
                "revision_at": "2023-11-14 16:40:27+00:00",
                "validation_status": "staff_change",
            },
            {
                "season": 2022,
                "team": "IND",
                "role": "OC",
                "person": "New",
                "previous_person": "Old",
                "revision_at": "2022-11-09 15:46:43+00:00",
                "validation_status": "playcaller_role",
            },
            {
                "season": 2022,
                "team": "SF",
                "role": "DC",
                "person": "X",
                "previous_person": "Y",
                "revision_at": "2022-12-26 21:58:39+00:00",
                "validation_status": "reverted_identity_edit",
            },
            {
                "season": 2024,
                "team": "NYJ",
                "role": "DC",
                "person": "X",
                "previous_person": "Y",
                "revision_at": "2024-11-19 19:55:22+00:00",
                "validation_status": "unverified_identity_edit",
            },
            {
                "season": 2023,
                "team": "PIT",
                "role": "OC",
                "person": "X",
                "previous_person": "Y",
                "revision_at": "2023-11-21 13:58:36+00:00",
                "validation_status": "role_correction",
            },
        ]
    }
    path = tmp_path / "change_validation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    events = load_counted_events(path)
    assert events["validation_status"].tolist() == ["playcaller_role", "staff_change"]
    assert events["side"].tolist() == ["offence", "offence"]
    assert events["revision_at"].dt.tz is not None
    assert events["event_id"].is_unique


def _features(n_weeks: int = 14) -> pd.DataFrame:
    rows = []
    teams = ["A", "B", "C", "D"]
    for week in range(1, n_weeks + 1):
        for pair in ((0, 1), (2, 3)):
            home, away = teams[pair[0]], teams[pair[1]]
            rows.append(
                {
                    "game_id": f"2024_{week:02d}_{away}_{home}",
                    "season": 2024,
                    "week": week,
                    "game_type": "REG",
                    "kickoff": pd.Timestamp("2024-09-08T17:00:00Z")
                    + pd.Timedelta(days=7 * (week - 1)),
                    "home_team": home,
                    "away_team": away,
                    "spread_line": -3.0,
                    "result": 7.0 if week % 2 else -7.0,
                    "home_cover": 1.0 if week % 2 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_r1_flags_use_kickoff_boundary_and_first_game_only() -> None:
    features = _features()
    table = team_game_table(features, (2024, 2024))
    events = pd.DataFrame([_event("e1", "2024-09-24T15:00:00Z")])
    flagged = r1_flags(table, features, events)
    first = flagged.loc[flagged["first_game_after_change"]]
    assert first["game_id"].tolist() == ["2024_04_B_A"]
    assert first["team"].tolist() == ["A"]
    assert flagged.loc[flagged["games_2_to_4_after_change"], "week"].tolist() == [5, 6, 7]

    late = pd.DataFrame([_event("e1", "2024-09-29T17:00:00Z")])
    later = r1_flags(table, features, late)
    assert later.loc[later["first_game_after_change"], "game_id"].tolist() == ["2024_05_B_A"]


def test_opener_grading_replaces_line_and_drops_pushes() -> None:
    features = _features(4)
    per_game = pd.DataFrame(
        {
            "game_id": features["game_id"].iloc[:4],
            "tue_open_home_spread": [7.0, -7.0, 10.0, -1.0],
            "result": features["result"].iloc[:4],
        }
    )
    graded = opener_graded_features(features, per_game)
    assert len(graded) == 4
    assert graded["spread_line"].tolist() == [7.0, -7.0, 10.0, -1.0]
    covers = graded["home_cover"].tolist()
    assert np.isnan(covers[0])
    assert covers[1:] == [1.0, 0.0, 0.0]
    table = team_game_table(graded, (2024, 2024))
    assert not table["team_covered"].isna().any()
    assert set(table["game_id"]) == set(features["game_id"].iloc[1:4])
    assert len(table) == 6


def test_score_cell_reports_raw_gap_and_scaled_effect() -> None:
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {
            "team_covered": rng.integers(0, 2, size=400).astype(float),
            "week_block": np.repeat(np.arange(20), 20),
            "season": np.repeat(np.arange(4), 100),
        }
    )
    flag = pd.Series(np.arange(400) % 10 == 0)
    cell = score_cell(
        name="synthetic",
        table=table,
        flag=flag,
        eligible=None,
        sign=1,
        grade="close",
        seasons=(2009, 2012),
    )
    expected_gap = (
        table.loc[flag, "team_covered"].mean() - table.loc[~flag, "team_covered"].mean()
    ) * 100.0
    assert cell["raw_gap_points"] == pytest.approx(expected_gap)
    assert cell["fraction_of_slate"] == pytest.approx(0.1)
    assert cell["effect_full_slate_points"] == pytest.approx(expected_gap * 0.1)
    week = cell["week_blocked"]
    assert week["raw_gap"]["lower"] <= cell["raw_gap_points"] <= week["raw_gap"]["upper"]
    assert 0.0 <= week["probability_positive"] <= 1.0
    assert cell["classification"] in {"unresolved_below_power", "refuted_mechanism"}
    assert cell["flag_record"]["games"] == 40


_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def test_record_argv_carries_family_prefix_and_human_plain_summary() -> None:
    table = pd.DataFrame(
        {
            "team_covered": [1.0, 0.0] * 60,
            "week_block": np.repeat(np.arange(12), 10),
            "season": np.repeat(np.arange(3), 40),
        }
    )
    flag = pd.Series([True] * 12 + [False] * 108)
    cell = score_cell(
        name=f"{FAMILY}_first_game",
        table=table,
        flag=flag,
        eligible=None,
        sign=1,
        grade="opener",
        seasons=(2020, 2025),
    )
    plain = "A team's first game after a mid-season coordinator change. Back that team."
    argv = record_argv(
        cell,
        description="desc",
        plain_summary=plain,
        source="docs/playcaller_change_leads.md",
        extra_notes="",
    )
    assert argv[:2] == ["weak-signals", "record"]
    assert argv[argv.index("--name") + 1].startswith(f"{FAMILY}_")
    assert argv[argv.index("--family") + 1] == FAMILY
    summary = argv[argv.index("--plain-summary") + 1]
    assert not _SNAKE_CASE_RE.search(summary)
    assert "P+" not in summary and "week-blocked" not in summary.lower()
    assert argv[argv.index("--classification") + 1] == cell["classification"]
    if cell["closing_ground"] is None:
        assert "--closing-ground" not in argv
