"""The opener_error_mining reliability sweep's slice reproduction and split arithmetic.

Two guarantees, both of which a silent drift in the slicing logic or the
estimator would break:

1. ``scripts/reliability_opener_eval.py`` rebuilds the exact push-excluded
   n=1,503 population every ``opener_error_mining_*`` registry cell's
   ``sample_games`` was measured against, and its 28 boolean slice masks
   reproduce every one of those counts exactly. If the join, the push
   exclusion, or a single bucket boundary drifted, a cell's reliability
   would be measured on a different population than the effect recorded
   beside it in the registry.
2. ``scripts/reliability_lib.measure_reliability`` (imported, never
   reimplemented) does the split arithmetic the sweep claims it does, on a
   frame whose answer is computable by hand -- and a construct restricted
   to a single season, or flagged by the compositional-artifact diagnostic,
   comes back UNMEASURED rather than as a manufactured number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.append(str(REPO / "scripts"))

import reliability_lib as rlib  # noqa: E402
import reliability_opener_eval as sweep  # noqa: E402

from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

# ---------------------------------------------------------------------------
# 1. The 28 slices reproduce their registry-recorded sample_games exactly
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def population() -> pd.DataFrame:
    return sweep.add_derived_columns(sweep.load_population())


@pytest.fixture(scope="module")
def masks(population: pd.DataFrame) -> dict[str, pd.Series]:
    return sweep.entry_slice_masks(population)


@pytest.fixture(scope="module")
def recorded_sample_games() -> dict[str, int | None]:
    registry = load_registry(default_registry_path(REPO / "registry"))
    return {
        name: signal.sample_games
        for name, signal in registry.signals.items()
        if name.startswith(sweep.PREFIX)
    }


def test_population_is_the_push_excluded_1503_archive(population: pd.DataFrame) -> None:
    assert len(population) == 1503
    assert population["correct_at_open_probability_rule"].notna().all()


@pytest.mark.parametrize(
    "entry",
    [
        f"{sweep.PREFIX}confidence_bucket_lt0p02",
        f"{sweep.PREFIX}division_game_yes",
        f"{sweep.PREFIX}movement_agreement_disagrees_overlay_paired_delta_move_ge_1_0",
        f"{sweep.PREFIX}total_bucket_below_42",
        f"{sweep.PREFIX}week_third_early",
        f"{sweep.PREFIX}season_2025",
    ],
)
def test_reconstructed_slice_matches_the_registrys_recorded_sample_games(
    entry: str,
    masks: dict[str, pd.Series],
    recorded_sample_games: dict[str, int | None],
) -> None:
    recorded = recorded_sample_games[entry]
    assert recorded is not None, f"{entry} carries no sample_games in the registry to check against"
    assert int(masks[entry].sum()) == recorded


def test_every_registry_cell_reproduces_its_sample_games(
    masks: dict[str, pd.Series], recorded_sample_games: dict[str, int | None]
) -> None:
    """Every one of the 28 cells, not just the sampled subset above."""

    assert set(masks) == set(recorded_sample_games)
    mismatches = {
        name: (int(masks[name].sum()), recorded_sample_games[name])
        for name in masks
        if recorded_sample_games[name] is not None
        and int(masks[name].sum()) != recorded_sample_games[name]
    }
    assert mismatches == {}


def test_entry_construct_covers_every_registry_cell_with_a_valid_kind(
    recorded_sample_games: dict[str, int | None],
) -> None:
    assert len(recorded_sample_games) == 28
    for name in recorded_sample_games:
        assert name in sweep.ENTRY_CONSTRUCT
        kind = sweep.ENTRY_CONSTRUCT[name]["kind"]
        assert kind in ("trait", "venue", "exposure")
        quantity = sweep.ENTRY_CONSTRUCT[name]["quantity"]
        assert quantity in sweep.QUANTITY_NOTE


def test_fallback_construct_only_covers_the_two_rest_diff_entries() -> None:
    assert set(sweep.FALLBACK_CONSTRUCT) == {
        f"{sweep.PREFIX}rest_diff_away_more_rested",
        f"{sweep.PREFIX}rest_diff_even",
    }
    for spec in sweep.FALLBACK_CONSTRUCT.values():
        assert spec["kind"] == "exposure"


# ---------------------------------------------------------------------------
# 2. The split arithmetic, on an answer computable by hand
# ---------------------------------------------------------------------------


def _synthetic_games(n_teams: int = 8) -> pd.DataFrame:
    """One game per (team pair, week) for 4 weeks, 2020 only -- enough units
    for measure_reliability's MIN_UNITS=20 floor once home+away both count."""

    rows = []
    for week in range(1, 5):
        for i in range(n_teams):
            home = f"H{i}"
            away = f"A{i}"
            # A deterministic per-pair spread and rest_diff so the resulting
            # team-week values are hand-computable.
            spread = float(i) - 3.5
            rest_diff = float(i % 3) - 1.0
            rows.append(
                {
                    "season": 2020,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "tue_open_home_spread": spread,
                    "rest_diff": rest_diff,
                    "open_move": spread * 0.1,
                    "home_cover_probability_at_open": 0.5 + spread * 0.01,
                    "total_line": 44.0 + i,
                }
            )
    return pd.DataFrame(rows)


def test_build_trait_frame_mirrors_the_away_row_sign() -> None:
    """favorite_spread_own: home row IS the spread, away row is its negation --
    the own-side convention documented in the module docstring."""

    games = _synthetic_games()
    long = sweep.build_trait_frame(games, "favorite_spread_own")

    assert len(long) == 2 * len(games)
    home_rows = long.loc[long["team_id"].isin(games["home_team"])]
    away_rows = long.loc[long["team_id"].isin(games["away_team"])]
    # Team H0's spread is -3.5 (row-for-row identical across all 4 weeks);
    # team A0 (its away-side mirror pairing) must see +3.5.
    assert home_rows.loc[home_rows["team_id"] == "H0", "value"].unique().tolist() == [-3.5]
    assert away_rows.loc[away_rows["team_id"] == "A0", "value"].unique().tolist() == [3.5]


def test_build_venue_frame_is_one_row_per_game_keyed_on_home_team() -> None:
    games = _synthetic_games()
    long = sweep.build_venue_frame(games, "total_line")

    assert len(long) == len(games)
    assert set(long["team_id"]) == set(games["home_team"])
    assert long.loc[long["team_id"] == "H0", "value"].unique().tolist() == [44.0]


def test_recovers_a_hand_computed_correlation_from_a_trait_frame() -> None:
    # 6 team-seasons, 4 weeks each: odd half = weeks 1,3; even half = weeks 2,4.
    # Values set directly so the odd/even half-MEANS have a known Pearson r.
    odd_means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    even_means = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
    rows = []
    for index, (odd, even) in enumerate(zip(odd_means, even_means, strict=True)):
        for week, value in ((1, odd), (2, even), (3, odd), (4, even)):
            rows.append({"team_id": f"T{index}", "season": 2020, "week": week, "value": value})
    long = pd.DataFrame(rows)

    import numpy as np

    expected_r = float(np.corrcoef(odd_means, even_means)[0, 1])
    expected_sb = 2.0 * expected_r / (1.0 + expected_r)

    result = rlib.measure_reliability(
        long, "value", method=rlib.METHOD_TRAIT, n_boot=200, min_units=3
    )

    assert result["status"] == rlib.STATUS_MEASURED
    assert result["n_units"] == len(odd_means)
    assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)
    assert result["spearman_brown_full_length_reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert result["reliability"] == pytest.approx(expected_sb, abs=1e-9)
    assert result["reliability_low"] <= result["reliability"] <= result["reliability_high"]


# ---------------------------------------------------------------------------
# 3. A single-season-restricted flag is UNMEASURED, never a manufactured number
# ---------------------------------------------------------------------------


def test_single_season_flag_restricted_to_that_season_is_constant_not_measured() -> None:
    """Mirrors opener_error_mining_season_2025 exactly: a flag defined as
    ``season == 2025`` on the full archive becomes a CONSTANT the instant the
    frame is restricted to season 2025 alone (every surviving row already
    satisfies the flag), so it must come back unmeasured, never as a
    manufactured no_split_half_reliability candidate."""

    rows = []
    for i in range(30):
        for season in (2024, 2025):
            for week in range(1, 5):
                rows.append(
                    {
                        "team_id": f"T{i}",
                        "season": season,
                        "week": week,
                        "flag": 1.0 if season == 2025 else 0.0,
                    }
                )
    long = pd.DataFrame(rows)

    result = rlib.measure_reliability(
        long, "flag", method=rlib.METHOD_EXPOSURE, seasons=(2025, 2025), n_boot=100
    )

    assert result["status"] != rlib.STATUS_MEASURED
    assert result["reliability"] is None


def test_season_2025_entry_reproduces_that_same_unmeasured_status(
    population: pd.DataFrame,
) -> None:
    """The actual production construction for opener_error_mining_season_2025."""

    flag = sweep.build_exposure_flag(population, "season_2025_flag")
    long = rlib.game_flag_to_team_week(population, flag)
    result = rlib.measure_reliability(
        long, "exposure", method=rlib.METHOD_EXPOSURE, seasons=(2025, 2025), n_boot=100
    )

    assert result["status"] != rlib.STATUS_MEASURED
    assert result["reliability"] is None


# ---------------------------------------------------------------------------
# 4. The compositional-artifact diagnostic's decision rule
# ---------------------------------------------------------------------------


def test_compositional_artifact_flagged_when_a_negative_survives_randomization() -> None:
    # rest_diff_own's actual real/diagnostic pair (measured this session):
    # real -0.7012, random-half mean -0.7067 -- both comfortably past the
    # 0.30 magnitude floor.
    assert sweep.is_compositional_artifact(-0.7012, -0.7067) is True


def test_compositional_artifact_flagged_on_a_large_real_vs_diagnostic_gap() -> None:
    # week_third_early's actual pair: real +0.2655, random-half mean -0.6796;
    # the sign flip alone clears the 0.5 gap threshold.
    assert sweep.is_compositional_artifact(0.2655, -0.6796) is True


def test_sound_measurement_is_not_flagged() -> None:
    # confidence_distance's actual pair: real +0.6549, random-half +0.5931.
    assert sweep.is_compositional_artifact(0.6549, 0.5931) is False
    # A small, consistent near-zero pair (rest_diff<0 flag) is not flagged either.
    assert sweep.is_compositional_artifact(-0.0682, -0.0697) is False


def test_compositional_artifact_ignores_unmeasured_constructs() -> None:
    assert sweep.is_compositional_artifact(None, -0.5) is False
    assert sweep.is_compositional_artifact(-0.5, None) is False
