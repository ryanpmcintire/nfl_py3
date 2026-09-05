"""Construction, sign-convention, censoring and leakage contracts for the
officiating-crew leads (LEAD-31/32/34; LEAD-33 is skipped for a documented
source gap -- see ``docs/officials_crew_leads.md``).

Most tests inject an already-built ``home_away_penalty_game_table``-shaped
(or ``rookie_crew_table``-shaped) DataFrame directly via each function's
``table``/``trait`` keyword, so no real ``data/raw/officials/*`` snapshot is
read. One integration test
(``test_home_away_penalty_game_table_crosswalks_officials_to_game_penalties``)
builds a tiny on-disk fixture tree mirroring
``tests/test_experiment_runner.py``'s own officials/game_penalties/schedules
fixture convention, to pin the crosswalk join itself.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.officials_flag_features import (
    CREW_HOME_BIAS_COLUMN,
    ROOKIE_CREW_UNDERDOG_COLUMN,
    ROOKIE_ELIGIBLE_SEASON_FLOOR,
    SECOND_MEETING_FAVORITE_COLUMN,
    TRAILING_HOME_BIAS_MIN_GAMES,
    crew_familiarity_table,
    derive_crew_home_bias_features,
    derive_rookie_crew_underdog_features,
    derive_second_meeting_favorite_features,
    describe_crew_familiarity,
    describe_referee_left_censoring,
    home_away_penalty_game_table,
    officials_home_bias_reliability,
    rookie_crew_table,
    trailing_home_bias_table,
)


def _game_row(
    game_id: str,
    official: str,
    season: int,
    week: int,
    home: str,
    away: str,
    penalties_on_home: float,
    penalties_on_away: float,
) -> dict:
    return {
        "game_id": game_id,
        "official_name": official,
        "season": season,
        "week": week,
        "home_team": home,
        "away_team": away,
        "penalties_total": penalties_on_home + penalties_on_away,
        "penalties_on_home": penalties_on_home,
        "penalties_on_away": penalties_on_away,
        "home_minus_away": penalties_on_home - penalties_on_away,
    }


def _table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _lines(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["game_id", "tue_open_home_spread"])


# ---------------------------------------------------------------------------
# Loader / crosswalk integration test (real on-disk fixture, mirrors
# tests/test_experiment_runner.py's own officials-fixture convention)
# ---------------------------------------------------------------------------


def test_home_away_penalty_game_table_crosswalks_officials_to_game_penalties(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "data" / "raw" / "20200101T000000Z"
    raw_dir.mkdir(parents=True)
    pd.DataFrame([{"game_id": "2020_01_AAA_BBB", "old_game_id": "2020090100"}]).to_parquet(
        raw_dir / "schedules.parquet"
    )

    officials_dir = tmp_path / "data" / "raw" / "officials" / "20200101T000000Z"
    officials_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "game_id": "2020090100",
                "official_name": "REF_A",
                "position": "Referee",
                "season": 2020,
                "season_type": "REG",
            },
            # A non-Referee position row for the same game must be excluded.
            {
                "game_id": "2020090100",
                "official_name": "UMP_X",
                "position": "Umpire",
                "season": 2020,
                "season_type": "REG",
            },
        ]
    ).to_parquet(officials_dir / "officials.parquet")
    pd.DataFrame(
        [
            {
                "game_id": "2020_01_AAA_BBB",
                "season": 2020,
                "week": 1,
                "home_team": "BBB",
                "away_team": "AAA",
                "penalties_total": 10.0,
                "penalties_on_home": 7.0,
                "penalties_on_away": 3.0,
            }
        ]
    ).to_parquet(officials_dir / "game_penalties.parquet")

    table = home_away_penalty_game_table(tmp_path)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["game_id"] == "2020_01_AAA_BBB"
    assert row["official_name"] == "REF_A"
    assert row["home_minus_away"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# LEAD-32: trailing (prior-games-only) home-bias quartile flag, leakage
# ---------------------------------------------------------------------------


def _trailing_bias_fixture() -> pd.DataFrame:
    """One crew-season (REF_A, 2020) with 5 games, ``home_minus_away``
    strictly increasing (1, 2, 3, 4, 100) so the trailing mean of the first 3
    games (1,2,3 -> avg 2.0) is a small value while later games' trailing
    means rise. A second crew-season (REF_B, 2021, 5 games, diffs
    10/20/30/40/50) supplies two MORE distinct eligible trailing values (20,
    25) so the combined eligible population has >= 4 distinct values and
    ``pd.qcut(4)`` can form real quartiles rather than skipping (this
    module's own ``>= 4`` guard)."""

    rows = [
        _game_row("gA1", "REF_A", 2020, 1, "H", "A", 5.0, 4.0),  # diff 1
        _game_row("gA2", "REF_A", 2020, 2, "H", "A", 6.0, 4.0),  # diff 2
        _game_row("gA3", "REF_A", 2020, 3, "H", "A", 7.0, 4.0),  # diff 3
        _game_row(
            "gA4", "REF_A", 2020, 4, "H", "A", 8.0, 4.0
        ),  # diff 4 (trailing so far: (1+2+3)/3=2)
        _game_row(
            "gA5", "REF_A", 2020, 5, "H", "A", 104.0, 4.0
        ),  # diff 100 (trailing: (1+2+3+4)/4=2.5)
    ]
    for week, diff in enumerate((10.0, 20.0, 30.0, 40.0, 50.0), start=1):
        rows.append(_game_row(f"gB{week}", "REF_B", 2021, week, "H", "A", 4.0 + diff, 4.0))
    return _table(rows)


def test_trailing_home_bias_requires_minimum_prior_games() -> None:
    table = _trailing_bias_fixture()
    trailing = trailing_home_bias_table(table=table)
    by_game = trailing.set_index("game_id")["trailing_home_bias"]
    # First TRAILING_HOME_BIAS_MIN_GAMES (3) games of REF_A/2020 have no
    # valid trailing value yet.
    assert TRAILING_HOME_BIAS_MIN_GAMES == 3
    assert pd.isna(by_game["gA1"])
    assert pd.isna(by_game["gA2"])
    assert pd.isna(by_game["gA3"])
    # gA4 is REF_A's 4th game: trailing mean of games 1-3 (diffs 1,2,3) = 2.0.
    assert by_game["gA4"] == pytest.approx(2.0)
    # gA5 is REF_A's 5th game: trailing mean of games 1-4 (diffs 1,2,3,4) = 2.5.
    assert by_game["gA5"] == pytest.approx(2.5)


def test_trailing_home_bias_never_uses_this_games_own_penalty_count() -> None:
    """Mutating game gA4's OWN penalty counts must not change gA4's own
    trailing value, but legitimately DOES change gA5's (a later game in the
    same crew-season) -- both directions asserted, matching the schedule-flag
    battery's own leakage-test shape."""

    table = _trailing_bias_fixture()
    baseline = trailing_home_bias_table(table=table).set_index("game_id")["trailing_home_bias"]

    mutated = table.copy()
    mutated.loc[mutated["game_id"] == "gA4", "home_minus_away"] = -999.0
    mutated_trailing = trailing_home_bias_table(table=mutated).set_index("game_id")[
        "trailing_home_bias"
    ]

    # gA4's own trailing value (built from games 1-3 only) is unchanged.
    assert mutated_trailing["gA4"] == baseline["gA4"]
    # gA5's trailing value (built from games 1-4, which now includes the
    # mutated gA4) DOES change.
    assert mutated_trailing["gA5"] != baseline["gA5"]


def test_crew_home_bias_flag_is_unsigned_top_quartile_only() -> None:
    table = _trailing_bias_fixture()
    trailing = trailing_home_bias_table(table=table)
    flags = derive_crew_home_bias_features(table=table).set_index("game_id")[CREW_HOME_BIAS_COLUMN]
    assert set(flags.unique()).issubset({0.0, 1.0})
    # Games with no valid trailing value (not yet 3 prior games) are ALWAYS
    # 0.0, regardless of the quartile cut.
    for game_id in ("gA1", "gA2", "gA3", "gB1", "gB2", "gB3"):
        assert flags[game_id] == 0.0

    # Among the ELIGIBLE (non-NaN trailing) games, the flagged set must be
    # EXACTLY the top quartile by pd.qcut(4) -- reproduced independently here
    # rather than hand-guessed, so the test pins the mechanism, not one
    # fixture's specific numbers.
    eligible = trailing.dropna(subset=["trailing_home_bias"]).set_index("game_id")
    assert len(eligible) >= 4
    expected_top = set(
        pd.qcut(eligible["trailing_home_bias"], 4, labels=[1, 2, 3, 4])
        .astype(int)
        .loc[lambda s: s == 4]
        .index
    )
    flagged_ids = set(flags.loc[flags == 1.0].index)
    assert flagged_ids == expected_top
    assert flagged_ids  # the fixture must actually produce a non-empty top quartile


def test_crew_home_bias_missing_from_features_raises() -> None:
    with pytest.raises(DataContractError):
        derive_crew_home_bias_features(table=pd.DataFrame(columns=["game_id"]))


# ---------------------------------------------------------------------------
# LEAD-34: crew-familiarity second meetings
# ---------------------------------------------------------------------------


def _familiarity_fixture() -> pd.DataFrame:
    return _table(
        [
            # REF_A's first game this season involving H/A -> not a second
            # meeting.
            _game_row("g1", "REF_A", 2020, 1, "H", "A", 5.0, 5.0),
            # REF_A again, same season, week 3, H now plays a NEW opponent C
            # -> H was seen before (in g1) -> second meeting.
            _game_row("g2", "REF_A", 2020, 3, "H", "C", 5.0, 5.0),
            # REF_A again, week 5, neither team seen before by this crew this
            # season -> not a second meeting.
            _game_row("g3", "REF_A", 2020, 5, "D", "E", 5.0, 5.0),
            # A different official, same season: no shared history with
            # REF_A -> not a second meeting even though H/A repeat.
            _game_row("g4", "REF_B", 2020, 1, "H", "A", 5.0, 5.0),
            # REF_A next SEASON: the season boundary resets -- H/A meeting
            # REF_A again in a new season is NOT a second meeting.
            _game_row("g5", "REF_A", 2021, 1, "H", "A", 5.0, 5.0),
        ]
    )


def test_second_meeting_flag_resets_at_season_boundary_and_requires_same_crew() -> None:
    table = _familiarity_fixture()
    familiarity = crew_familiarity_table(table=table).set_index("game_id")["second_meeting"]
    assert bool(familiarity["g1"]) is False
    assert bool(familiarity["g2"]) is True
    assert bool(familiarity["g3"]) is False
    assert bool(familiarity["g4"]) is False
    assert bool(familiarity["g5"]) is False


def test_second_meeting_flag_never_uses_this_games_own_penalty_count() -> None:
    table = _familiarity_fixture()
    baseline = crew_familiarity_table(table=table).set_index("game_id")["second_meeting"]

    mutated = table.copy()
    mutated.loc[mutated["game_id"] == "g2", "home_minus_away"] = -999.0
    mutated.loc[mutated["game_id"] == "g2", "penalties_on_home"] = 0.0
    mutated_familiarity = crew_familiarity_table(table=mutated).set_index("game_id")[
        "second_meeting"
    ]
    pd.testing.assert_series_equal(baseline, mutated_familiarity)


def test_describe_crew_familiarity_reports_frequency_and_gap() -> None:
    table = _familiarity_fixture()
    stats = describe_crew_familiarity(table=table)
    assert stats["n_games_with_referee"] == 5
    assert stats["n_second_meeting"] == 1
    assert stats["pct_second_meeting"] == pytest.approx(1 / 5)


def test_second_meeting_favorite_sign_convention() -> None:
    table = _familiarity_fixture()
    lines = _lines(
        [
            ("g1", 3.0),  # not a second meeting -> 0 regardless of spread
            ("g2", 3.0),  # second meeting, home favored (positive) -> +1
            ("g3", -3.0),  # not a second meeting -> 0
            ("g5", -3.0),  # season boundary reset -> not second meeting -> 0
        ]
    )
    flags = derive_second_meeting_favorite_features(None, lines, table=table).set_index("game_id")[
        SECOND_MEETING_FAVORITE_COLUMN
    ]
    assert flags["g1"] == 0.0
    assert flags["g2"] == 1.0
    assert flags["g3"] == 0.0
    assert flags["g5"] == 0.0
    # g4 has no opener line row -> missing spread -> 0.
    assert flags["g4"] == 0.0


def test_second_meeting_favorite_away_favorite_sign() -> None:
    table = _table(
        [
            _game_row("h1", "REF_C", 2022, 1, "H", "A", 5.0, 5.0),
            _game_row("h2", "REF_C", 2022, 3, "H", "Z", 5.0, 5.0),
        ]
    )
    lines = _lines([("h2", -3.0)])  # away (Z) favored
    flags = derive_second_meeting_favorite_features(None, lines, table=table).set_index("game_id")[
        SECOND_MEETING_FAVORITE_COLUMN
    ]
    assert flags["h2"] == -1.0


def test_second_meeting_favorite_requires_game_id_column() -> None:
    with pytest.raises(DataContractError):
        derive_second_meeting_favorite_features(
            None, pd.DataFrame({"tue_open_home_spread": [1.0]}), table=_familiarity_fixture()
        )


# ---------------------------------------------------------------------------
# LEAD-31: rookie-referee censoring and underdog sign convention
# ---------------------------------------------------------------------------


def _rookie_trait_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Left-censored: season 2015 debut, excluded from eligibility
            # regardless of prior_seasons_experience.
            {
                "game_id": "r2015",
                "official_name": "REF_CENSORED",
                "season": 2015,
                "prior_seasons_experience": 0,
            },
            # Genuine rookie: first eligible season, 0 prior seasons.
            {
                "game_id": "r_rookie",
                "official_name": "REF_ROOKIE",
                "season": 2020,
                "prior_seasons_experience": 0,
            },
            # Genuine second-year: 1 prior season -- still "rookie" per the
            # task's own {0,1} definition.
            {
                "game_id": "r_second_year",
                "official_name": "REF_SECOND_YEAR",
                "season": 2020,
                "prior_seasons_experience": 1,
            },
            # Veteran: 2 prior seasons -- NOT a rookie crew.
            {
                "game_id": "r_veteran",
                "official_name": "REF_VETERAN",
                "season": 2020,
                "prior_seasons_experience": 2,
            },
        ]
    )


def test_rookie_eligible_season_floor_excludes_2015() -> None:
    assert ROOKIE_ELIGIBLE_SEASON_FLOOR == 2016
    trait = _rookie_trait_fixture()
    lines = _lines([("r2015", -3.0), ("r_rookie", -3.0)])
    flags = derive_rookie_crew_underdog_features(None, lines, trait=trait).set_index("game_id")[
        ROOKIE_CREW_UNDERDOG_COLUMN
    ]
    # r2015 would otherwise flag (prior_seasons_experience 0, home underdog)
    # but season 2015 is excluded from the rookie-eligible population.
    assert flags["r2015"] == 0.0
    assert flags["r_rookie"] == 1.0


def test_rookie_crew_underdog_sign_convention() -> None:
    trait = _rookie_trait_fixture()
    lines = _lines(
        [
            ("r_rookie", -3.0),  # rookie crew, home underdog -> +1
            ("r_second_year", 3.0),  # rookie crew (2nd yr), away underdog -> -1
            ("r_veteran", -3.0),  # veteran crew, home underdog -> 0
        ]
    )
    flags = derive_rookie_crew_underdog_features(None, lines, trait=trait).set_index("game_id")[
        ROOKIE_CREW_UNDERDOG_COLUMN
    ]
    assert flags["r_rookie"] == 1.0
    assert flags["r_second_year"] == -1.0
    assert flags["r_veteran"] == 0.0


def test_rookie_crew_table_returns_only_the_four_columns() -> None:
    trait = _rookie_trait_fixture()
    out = rookie_crew_table(trait=trait)
    assert list(out.columns) == ["game_id", "official_name", "season", "prior_seasons_experience"]
    assert len(out) == len(trait)


# ---------------------------------------------------------------------------
# LEAD-32: reliability harness wiring (season-blocked bootstrap, P+,
# Spearman-Brown, label-shuffle null all present)
# ---------------------------------------------------------------------------


def _reliability_fixture() -> pd.DataFrame:
    """Several referees whose home_minus_away is a season-and-official
    fixed constant plus small week-level noise, so a real, positive
    odd/even split-half correlation is present and testable end to end."""

    rows: list[dict] = []
    base = {"REF_1": 5.0, "REF_2": -3.0, "REF_3": 1.0, "REF_4": -1.0, "REF_5": 4.0}
    noise_by_week = {1: 0.1, 2: -0.1, 3: 0.2, 4: -0.2, 5: 0.0, 6: 0.1}
    for season in (2020, 2021, 2022):
        for official, level in base.items():
            for week in range(1, 7):
                rows.append(
                    _game_row(
                        f"{official}_{season}_{week}",
                        official,
                        season,
                        week,
                        "H",
                        "A",
                        max(level + noise_by_week[week], 0.0),
                        max(-(level + noise_by_week[week]), 0.0)
                        if level + noise_by_week[week] < 0
                        else 0.0,
                    )
                )
    table = _table(rows)
    # home_minus_away must equal the constructed level+noise directly
    # (the helper above only fabricates plausible non-negative counts);
    # overwrite it explicitly so the fixture's intent is unambiguous.
    parts = table["game_id"].str.rsplit("_", n=2, expand=True)
    table["official_name"] = parts[0]
    table["week"] = parts[2].astype(int)
    table["home_minus_away"] = [
        base[row.official_name] + noise_by_week[row.week] for row in table.itertuples()
    ]
    return table


def test_officials_home_bias_reliability_returns_full_harness_output() -> None:
    table = _reliability_fixture()
    result = officials_home_bias_reliability(table=table, n_boot=200, n_null=200)
    within = result["within_season_odd_even_week"]
    across = result["season_to_season_same_referee"]
    for section in (within, across):
        assert section["status"] == "measured"
        assert "pearson_r" in section
        assert "pearson_r_ci95" in section
        assert "pearson_probability_positive" in section
        assert "null_mean_r" in section
        assert "null_sd_r" in section
    # Spearman-Brown correction is only meaningful for the within-season
    # split.
    assert within["spearman_brown_full_length_reliability"] is not None
    assert across["spearman_brown_full_length_reliability"] is None
    # The fixture's construction (a strong, stable per-official level) should
    # produce a strongly positive, non-degenerate reliability read.
    assert within["pearson_r"] > 0.9
    assert within["pearson_probability_positive"] > 0.5


# ---------------------------------------------------------------------------
# Left-censoring disclosure
# ---------------------------------------------------------------------------


def test_describe_referee_left_censoring_counts_2015_debuts(tmp_path: Path) -> None:
    officials_dir = tmp_path / "data" / "raw" / "officials" / "20200101T000000Z"
    officials_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "game_id": "g1",
                "official_name": "REF_CENSORED",
                "position": "Referee",
                "season": 2015,
                "season_type": "REG",
            },
            {
                "game_id": "g2",
                "official_name": "REF_GENUINE",
                "position": "Referee",
                "season": 2018,
                "season_type": "REG",
            },
        ]
    ).to_parquet(officials_dir / "officials.parquet")
    pd.DataFrame(
        [
            {
                "game_id": "g1",
                "penalties_total": 1.0,
                "penalties_on_home": 1.0,
                "penalties_on_away": 0.0,
            }
        ]
    ).to_parquet(officials_dir / "game_penalties.parquet")

    stats = describe_referee_left_censoring(tmp_path)
    assert stats["n_officials_total"] == 2
    assert stats["n_censored_2015_debut"] == 1
    assert stats["n_genuine_debut_2016_2025"] == 1
