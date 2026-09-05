"""Tests for LEAD-58 (src/nfl_ats/age_curves.py): snap-weighted career-age curves.

Closing-grounds taxonomy (verbatim, per CLAUDE.md/AGENTS.md -- pasted here
because this file adjudicates measured reliability numbers): an interval or
CI that contains zero is NEVER grounds to reject, fail, or close an
experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line
of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is ``unresolved_below_power``: report ``probability_positive``, never the
binary "contains zero". This module does not itself adjudicate a signal --
it is QUALITY infrastructure with no ATS direction -- so no test here closes
anything; the tests below check the MACHINERY (bounds, leakage safety,
exclusion rules, and that a known strong signal is actually detected).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.age_curves import (
    DELTA_METHOD_SNAP_FLOOR,
    METRIC_BY_GROUP,
    NO_LOCAL_METRIC_GROUPS,
    POSITION_GROUPS,
    build_age_curves,
    build_career_age_panel,
    cross_sectional_curve,
    delta_curve,
    local_linear_smooth,
    player_age_cells,
    shrink_cells,
    smooth_curve,
    split_half_reliability,
)
from nfl_ats.players import write_player_snapshot, write_player_value_snapshot

# ---------------------------------------------------------------------------
# Small raw-frame builders (canonicalize_* coerces dtypes, so plain python
# values are fine here -- same convention as tests/test_players.py).
# ---------------------------------------------------------------------------


def _roster_row(
    gsis_id: str,
    pfr_id: str,
    team: str,
    position: str,
    season: int,
    years_exp: int,
    week: int = 1,
) -> dict[str, object]:
    return {
        "season": season,
        "team": team,
        "position": position,
        "status": "ACT",
        "full_name": gsis_id,
        "gsis_id": gsis_id,
        "pfr_id": pfr_id,
        "years_exp": years_exp,
        "week": week,
        "game_type": "REG",
    }


def _snap_row(
    pfr_id: str,
    team: str,
    position: str,
    season: int,
    week: int,
    *,
    offense: float = 0.0,
    defense: float = 0.0,
    st: float = 0.0,
    game_id: str | None = None,
    player_name: str | None = None,
) -> dict[str, object]:
    return {
        "game_id": game_id or f"{season}_{week:02d}_{team}_OPP",
        "season": season,
        "game_type": "REG",
        "week": week,
        "player": player_name or pfr_id,
        "pfr_player_id": pfr_id,
        "position": position,
        "team": team,
        "offense_snaps": offense,
        "offense_pct": min(offense / 60.0, 1.0),
        "defense_snaps": defense,
        "defense_pct": min(defense / 60.0, 1.0),
        "st_snaps": st,
        "st_pct": min(st / 20.0, 1.0) if st else 0.0,
    }


def _stats_row(
    player_id: str,
    team: str,
    season: int,
    week: int,
    *,
    position: str = "WR",
    rushing_epa: float = 0.0,
    receiving_epa: float = 0.0,
    tfl: float = 0.0,
    ff: float = 0.0,
    sacks: float = 0.0,
    qb_hits: float = 0.0,
    ints: float = 0.0,
    passes_defended: float = 0.0,
    game_id: str | None = None,
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "season": season,
        "week": week,
        "season_type": "REG",
        "game_id": game_id or f"{season}_{week:02d}_{team}_OPP",
        "team": team,
        "position": position,
        "rushing_epa": rushing_epa,
        "receiving_epa": receiving_epa,
        "def_tackles_for_loss": tfl,
        "def_fumbles_forced": ff,
        "def_sacks": sacks,
        "def_qb_hits": qb_hits,
        "def_interceptions": ints,
        "def_pass_defended": passes_defended,
    }


def _pbp_row(
    game_id: str, passer_player_id: str, epa: float, *, season_type: str = "REG"
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season_type": season_type,
        "qb_dropback": 1,
        "passer_player_id": passer_player_id,
        "epa": epa,
    }


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------


def test_position_group_mapping_and_metric_labels_agree() -> None:
    assert set(POSITION_GROUPS.values()) == set(METRIC_BY_GROUP)
    for group in NO_LOCAL_METRIC_GROUPS:
        assert METRIC_BY_GROUP[group] == "no_local_metric"
    for group, metric in METRIC_BY_GROUP.items():
        if group not in NO_LOCAL_METRIC_GROUPS:
            assert metric != "no_local_metric"


def test_panel_drops_unmapped_positions_and_counts_them() -> None:
    rosters = pd.DataFrame(
        [
            _roster_row("W1", "PW1", "A", "WR", 2021, 2),
            _roster_row("O1", "PO1", "A", "T", 2021, 3),
            _roster_row("J1", "PJ1", "A", "C/G", 2021, 1),
        ]
    )
    snaps = pd.DataFrame(
        [
            _snap_row("PW1", "A", "WR", 2021, 1, offense=40.0),
            _snap_row("PO1", "A", "T", 2021, 1, offense=50.0),
            _snap_row("PJ1", "A", "C/G", 2021, 1, offense=10.0),
        ]
    )
    stats = pd.DataFrame([_stats_row("W1", "A", 2021, 1, receiving_epa=2.0)])

    panel, diagnostics = build_career_age_panel(snaps, rosters, stats, {})

    assert diagnostics["snap_rows_total"] == 3
    assert diagnostics["snap_rows_unlinked_to_gsis"] == 0
    assert diagnostics["gsis_match_rate"] == 1.0
    assert diagnostics["snap_rows_unmapped_position"] == 1
    assert diagnostics["snap_rows_missing_career_age"] == 0
    assert len(panel) == 2
    assert set(panel["gsis_id"]) == {"W1", "O1"}

    wr_row = panel.loc[panel["gsis_id"] == "W1"].iloc[0]
    assert wr_row["pos_group"] == "WR"
    assert wr_row["career_age"] == 2
    assert wr_row["metric_numerator"] == pytest.approx(2.0)
    assert wr_row["metric_denominator"] == pytest.approx(40.0)
    assert wr_row["coverage_status"] == "metric"

    ol_row = panel.loc[panel["gsis_id"] == "O1"].iloc[0]
    assert ol_row["pos_group"] == "OL"
    assert ol_row["career_age"] == 3
    assert math.isnan(ol_row["metric_numerator"])
    assert math.isnan(ol_row["metric_denominator"])
    assert ol_row["primary_snaps"] == pytest.approx(50.0)
    assert ol_row["coverage_status"] == "no_local_metric"


def test_skill_and_defense_weeks_without_a_stats_row_fill_zero_numerator() -> None:
    rosters = pd.DataFrame(
        [
            _roster_row("W2", "PW2", "A", "WR", 2021, 1),
            _roster_row("C1", "PC1", "A", "CB", 2021, 1),
        ]
    )
    snaps = pd.DataFrame(
        [
            _snap_row("PW2", "A", "WR", 2021, 1, offense=30.0),
            _snap_row("PC1", "A", "CB", 2021, 1, defense=45.0),
        ]
    )
    panel, _ = build_career_age_panel(snaps, rosters, pd.DataFrame(), {})

    wr_row = panel.loc[panel["gsis_id"] == "W2"].iloc[0]
    assert wr_row["metric_numerator"] == pytest.approx(0.0)
    assert wr_row["metric_denominator"] == pytest.approx(30.0)

    cb_row = panel.loc[panel["gsis_id"] == "C1"].iloc[0]
    assert cb_row["metric_numerator"] == pytest.approx(0.0)
    assert cb_row["metric_denominator"] == pytest.approx(45.0)


def test_qb_week_without_a_linked_dropback_row_is_excluded_not_zeroed() -> None:
    rosters = pd.DataFrame([_roster_row("Q1", "PQ1", "A", "QB", 2021, 5)])
    snaps = pd.DataFrame(
        [
            _snap_row("PQ1", "A", "QB", 2021, 1, offense=55.0, game_id="2021_01_A_OPP"),
            _snap_row("PQ1", "A", "QB", 2021, 2, offense=50.0, game_id="2021_02_A_OPP"),
        ]
    )
    pbp_frames = {
        2021: pd.DataFrame(
            [
                _pbp_row("2021_01_A_OPP", "Q1", 1.0),
                _pbp_row("2021_01_A_OPP", "Q1", 0.5),
                _pbp_row("2021_01_A_OPP", "Q1", -0.2, season_type="POST"),
            ]
        )
    }
    panel, _ = build_career_age_panel(snaps, rosters, pd.DataFrame(), pbp_frames)

    week1 = panel.loc[panel["week"] == 1].iloc[0]
    assert week1["metric_numerator"] == pytest.approx(1.5)
    assert week1["metric_denominator"] == pytest.approx(2.0)

    week2 = panel.loc[panel["week"] == 2].iloc[0]
    assert math.isnan(week2["metric_numerator"])
    assert math.isnan(week2["metric_denominator"])


# ---------------------------------------------------------------------------
# Point-in-time / leakage safety
# ---------------------------------------------------------------------------


def _build_multi_season_sources(
    *, include_future_rows: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    roster_rows = [
        _roster_row("W3", "PW3", "A", "WR", 2019, 0),
        _roster_row("W3", "PW3", "A", "WR", 2020, 1),
        _roster_row("Q2", "PQ2", "A", "QB", 2019, 4),
        _roster_row("Q2", "PQ2", "A", "QB", 2020, 5),
    ]
    snap_rows = [
        _snap_row("PW3", "A", "WR", 2019, 1, offense=40.0, game_id="2019_01_A_OPP"),
        _snap_row("PW3", "A", "WR", 2020, 1, offense=42.0, game_id="2020_01_A_OPP"),
        _snap_row("PQ2", "A", "QB", 2019, 1, offense=55.0, game_id="2019_01_A_OPP"),
        _snap_row("PQ2", "A", "QB", 2020, 1, offense=56.0, game_id="2020_01_A_OPP"),
    ]
    stats_rows = [
        _stats_row("W3", "A", 2019, 1, receiving_epa=1.0, game_id="2019_01_A_OPP"),
        _stats_row("W3", "A", 2020, 1, receiving_epa=1.2, game_id="2020_01_A_OPP"),
    ]
    pbp_frames = {
        2019: pd.DataFrame(
            [_pbp_row("2019_01_A_OPP", "Q2", 0.3), _pbp_row("2019_01_A_OPP", "Q2", 0.1)]
        ),
        2020: pd.DataFrame(
            [_pbp_row("2020_01_A_OPP", "Q2", 0.4), _pbp_row("2020_01_A_OPP", "Q2", 0.2)]
        ),
    }
    if include_future_rows:
        roster_rows += [
            _roster_row("W3", "PW3", "A", "WR", 2021, 2),
            _roster_row("Q2", "PQ2", "A", "QB", 2021, 6),
            _roster_row("FUTURE", "PFUT", "A", "TE", 2021, 0),
        ]
        snap_rows += [
            _snap_row("PW3", "A", "WR", 2021, 1, offense=999.0, game_id="2021_01_A_OPP"),
            _snap_row("PQ2", "A", "QB", 2021, 1, offense=999.0, game_id="2021_01_A_OPP"),
            _snap_row("PFUT", "A", "TE", 2021, 1, offense=999.0, game_id="2021_01_A_OPP"),
        ]
        stats_rows += [_stats_row("W3", "A", 2021, 1, receiving_epa=999.0, game_id="2021_01_A_OPP")]
        pbp_frames = dict(pbp_frames)
        pbp_frames[2021] = pd.DataFrame(
            [_pbp_row("2021_01_A_OPP", "Q2", 999.0), _pbp_row("2021_01_A_OPP", "Q2", 999.0)]
        )
    return (
        pd.DataFrame(snap_rows),
        pd.DataFrame(roster_rows),
        pd.DataFrame(stats_rows),
        pbp_frames,
    )


def test_as_of_season_uses_only_strictly_earlier_seasons() -> None:
    snaps, rosters, stats, pbp_frames = _build_multi_season_sources(include_future_rows=False)
    panel, _ = build_career_age_panel(snaps, rosters, stats, pbp_frames, as_of_season=2020)
    assert (panel["season"] < 2020).all()
    assert set(panel["season"].unique()) == {2019}


def test_as_of_season_is_bit_identical_regardless_of_later_rows_on_disk() -> None:
    snaps_a, rosters_a, stats_a, pbp_a = _build_multi_season_sources(include_future_rows=False)
    panel_a, diagnostics_a = build_career_age_panel(
        snaps_a, rosters_a, stats_a, pbp_a, as_of_season=2021
    )

    snaps_b, rosters_b, stats_b, pbp_b = _build_multi_season_sources(include_future_rows=True)
    panel_b, diagnostics_b = build_career_age_panel(
        snaps_b, rosters_b, stats_b, pbp_b, as_of_season=2021
    )

    pd.testing.assert_frame_equal(panel_a, panel_b)
    assert diagnostics_a["snap_rows_in_panel"] == diagnostics_b["snap_rows_in_panel"]
    assert (panel_b["season"] < 2021).all()


def test_player_age_cells_collapses_weeks_within_a_season() -> None:
    rosters = pd.DataFrame([_roster_row("W4", "PW4", "A", "WR", 2021, 1)])
    snaps = pd.DataFrame(
        [
            _snap_row("PW4", "A", "WR", 2021, 1, offense=30.0, game_id="2021_01_A_OPP"),
            _snap_row("PW4", "A", "WR", 2021, 2, offense=20.0, game_id="2021_02_A_OPP"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stats_row("W4", "A", 2021, 1, receiving_epa=1.0, game_id="2021_01_A_OPP"),
            _stats_row("W4", "A", 2021, 2, receiving_epa=0.5, game_id="2021_02_A_OPP"),
        ]
    )
    panel, _ = build_career_age_panel(snaps, rosters, stats, {})
    cells = player_age_cells(panel)

    assert len(cells) == 1
    row = cells.iloc[0]
    assert row["gsis_id"] == "W4"
    assert row["career_age"] == 1
    assert row["metric_numerator"] == pytest.approx(1.5)
    assert row["metric_denominator"] == pytest.approx(50.0)
    assert row["n_weeks"] == 2


# ---------------------------------------------------------------------------
# Cross-sectional curve
# ---------------------------------------------------------------------------


def test_cross_sectional_curve_is_snap_weighted() -> None:
    cells = pd.DataFrame(
        [
            {
                "gsis_id": "P1",
                "pos_group": "WR",
                "season": 2020,
                "career_age": 0,
                "metric_numerator": 4.0,
                "metric_denominator": 100.0,
                "primary_snaps": 100.0,
                "n_weeks": 4,
            },
            {
                "gsis_id": "P2",
                "pos_group": "WR",
                "season": 2020,
                "career_age": 0,
                "metric_numerator": 1.0,
                "metric_denominator": 50.0,
                "primary_snaps": 50.0,
                "n_weeks": 4,
            },
            {
                "gsis_id": "P3",
                "pos_group": "OL",
                "season": 2020,
                "career_age": 0,
                "metric_numerator": np.nan,
                "metric_denominator": np.nan,
                "primary_snaps": 80.0,
                "n_weeks": 4,
            },
        ]
    )
    curve = cross_sectional_curve(cells)

    wr = curve.loc[(curve["pos_group"] == "WR") & (curve["career_age"] == 0)].iloc[0]
    assert wr["raw_rate"] == pytest.approx(5.0 / 150.0)
    assert wr["snaps"] == pytest.approx(150.0)
    assert wr["n_players"] == 2
    assert wr["coverage_status"] == "metric"
    assert wr["sparse"]

    ol = curve.loc[(curve["pos_group"] == "OL") & (curve["career_age"] == 0)].iloc[0]
    assert math.isnan(ol["raw_rate"])
    assert ol["snaps"] == pytest.approx(80.0)
    assert ol["coverage_status"] == "no_local_metric"


# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage
# ---------------------------------------------------------------------------


def _dispersed_wr_cells() -> pd.DataFrame:
    """Two low-snap outlier players at age 0; a stable, high-snap age 1 and 2."""

    rows = []
    rows.append(
        {
            "gsis_id": "OUT1",
            "pos_group": "WR",
            "season": 2020,
            "career_age": 0,
            "metric_numerator": 10.0,
            "metric_denominator": 10.0,
            "primary_snaps": 10.0,
            "n_weeks": 1,
        }
    )
    for age in (1, 2):
        for player, rate in ((f"S{age}A", 0.018), (f"S{age}B", 0.022)):
            rows.append(
                {
                    "gsis_id": player,
                    "pos_group": "WR",
                    "season": 2020 + age,
                    "career_age": age,
                    "metric_numerator": rate * 1000.0,
                    "metric_denominator": 1000.0,
                    "primary_snaps": 1000.0,
                    "n_weeks": 16,
                }
            )
    return pd.DataFrame(rows)


def test_shrunk_rate_always_lies_between_raw_rate_and_grand_mean() -> None:
    cells = _dispersed_wr_cells()
    curve = cross_sectional_curve(cells)
    curve = shrink_cells(curve, cells)

    metric_rows = curve.loc[curve["coverage_status"] == "metric"]
    for _, row in metric_rows.iterrows():
        if math.isnan(row["shrunk_rate"]) or math.isnan(row["grand_mean"]):
            continue
        low, high = sorted((row["raw_rate"], row["grand_mean"]))
        assert low - 1e-9 <= row["shrunk_rate"] <= high + 1e-9

    age0 = metric_rows.loc[metric_rows["career_age"] == 0].iloc[0]
    assert age0["raw_rate"] == pytest.approx(1.0)
    assert age0["shrunk_rate"] < age0["raw_rate"]
    assert abs(age0["shrunk_rate"] - age0["grand_mean"]) < abs(
        age0["raw_rate"] - age0["grand_mean"]
    )


def test_no_local_metric_groups_are_never_shrunk() -> None:
    cells = pd.DataFrame(
        [
            {
                "gsis_id": "K1",
                "pos_group": "K",
                "season": 2020,
                "career_age": 0,
                "metric_numerator": np.nan,
                "metric_denominator": np.nan,
                "primary_snaps": 60.0,
                "n_weeks": 4,
            }
        ]
    )
    curve = cross_sectional_curve(cells)
    curve = shrink_cells(curve, cells)
    row = curve.iloc[0]
    assert math.isnan(row["shrunk_rate"])
    assert math.isnan(row["shrinkage_k"])


# ---------------------------------------------------------------------------
# Local-linear smooth
# ---------------------------------------------------------------------------


def test_local_linear_smooth_returns_the_point_itself_when_isolated() -> None:
    ages = np.array([0.0, 5.0])
    rates = np.array([0.02, 0.09])
    weights = np.array([500.0, 500.0])
    smoothed = local_linear_smooth(ages, rates, weights)
    assert smoothed[0] == pytest.approx(0.02)
    assert smoothed[1] == pytest.approx(0.09)


def test_smooth_curve_skips_no_local_metric_groups() -> None:
    cells = _dispersed_wr_cells()
    curve = cross_sectional_curve(cells)
    curve = smooth_curve(curve)
    assert curve["smoothed_rate"].notna().all()

    ol_cells = pd.DataFrame(
        [
            {
                "gsis_id": "O1",
                "pos_group": "OL",
                "season": 2020,
                "career_age": 0,
                "metric_numerator": np.nan,
                "metric_denominator": np.nan,
                "primary_snaps": 60.0,
                "n_weeks": 4,
            }
        ]
    )
    ol_curve = smooth_curve(cross_sectional_curve(ol_cells))
    assert math.isnan(ol_curve["smoothed_rate"].iloc[0])


# ---------------------------------------------------------------------------
# Delta-method curve
# ---------------------------------------------------------------------------


def test_delta_curve_only_pairs_strictly_consecutive_ages_above_the_floor() -> None:
    floor = DELTA_METHOD_SNAP_FLOOR
    cells = pd.DataFrame(
        [
            # Player A: ages 0, 1, 3 (age 2 missing) -- only 0->1 is a valid pair.
            {
                "gsis_id": "A",
                "pos_group": "WR",
                "season": 2020,
                "career_age": 0,
                "metric_numerator": 0.01 * floor,
                "metric_denominator": floor,
            },
            {
                "gsis_id": "A",
                "pos_group": "WR",
                "season": 2021,
                "career_age": 1,
                "metric_numerator": 0.03 * floor,
                "metric_denominator": floor,
            },
            {
                "gsis_id": "A",
                "pos_group": "WR",
                "season": 2023,
                "career_age": 3,
                "metric_numerator": 0.05 * floor,
                "metric_denominator": floor,
            },
            # Player B: age 0 below the snap floor -- excluded even though age 1 exists.
            {
                "gsis_id": "B",
                "pos_group": "WR",
                "season": 2020,
                "career_age": 0,
                "metric_numerator": 0.01 * (floor / 4),
                "metric_denominator": floor / 4,
            },
            {
                "gsis_id": "B",
                "pos_group": "WR",
                "season": 2021,
                "career_age": 1,
                "metric_numerator": 0.10 * floor,
                "metric_denominator": floor,
            },
        ]
    )
    curve = pd.DataFrame(
        [
            {"pos_group": "WR", "career_age": 0, "n_players": 2, "coverage_status": "metric"},
            {"pos_group": "WR", "career_age": 1, "n_players": 1, "coverage_status": "metric"},
        ]
    )
    delta = delta_curve(cells, curve, snap_floor=floor)

    assert list(delta["career_age_from"]) == [0]
    row = delta.iloc[0]
    assert row["n_pairs"] == 1
    assert row["mean_delta"] == pytest.approx(0.02 * floor / floor)
    # career_age 0 is the modal entry age here (2 players), so its own
    # cumulative delta is the zero baseline.
    assert row["cumulative_delta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Split-half reliability
# ---------------------------------------------------------------------------


def _synthetic_reliability_cells(seed: int = 7, n_players: int = 40) -> pd.DataFrame:
    """A clean linear age signal (rate = 0.01 * age) plus small noise.

    Spread across many distinct seasons (players enter in staggered years)
    so both the odd/even-season and random-player-half schemes have enough
    independent blocks to resample.
    """

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for player in range(n_players):
        gsis_id = f"P{player:03d}"
        start_season = 2000 + (player % 8)
        for age in range(6):
            season = start_season + age
            true_rate = 0.01 * age
            denom = 200.0
            noise = rng.normal(0.0, 0.004)
            numerator = (true_rate + noise) * denom
            rows.append(
                {
                    "gsis_id": gsis_id,
                    "pos_group": "WR",
                    "season": season,
                    "career_age": age,
                    "metric_numerator": numerator,
                    "metric_denominator": denom,
                    "primary_snaps": denom,
                    "n_weeks": 16,
                }
            )
    return pd.DataFrame(rows)


def test_split_half_reliability_stats_are_bounded() -> None:
    cells = _synthetic_reliability_cells()
    curve = cross_sectional_curve(cells)
    reliability = split_half_reliability(cells, curve, samples=300, seed=11)

    assert not reliability.empty
    assert set(reliability["scheme"]) == {"odd_even_seasons", "random_player_halves"}
    for _, row in reliability.iterrows():
        if not math.isnan(row["pearson_r"]):
            assert -1.0 - 1e-9 <= row["pearson_r"] <= 1.0 + 1e-9
        if not math.isnan(row["probability_positive"]):
            assert 0.0 <= row["probability_positive"] <= 1.0


def test_split_half_reliability_detects_a_real_age_signal() -> None:
    cells = _synthetic_reliability_cells()
    curve = cross_sectional_curve(cells)
    reliability = split_half_reliability(cells, curve, samples=500, seed=123)

    wr_rows = reliability.loc[reliability["pos_group"] == "WR"]
    assert len(wr_rows) == 2
    # A clean, strongly age-dependent signal should show up as a strongly
    # positive probability_positive under BOTH schemes -- never rejected for
    # "containing zero" (it should not even come close in this synthetic
    # case), but this is a power check on the machinery, not a claim about
    # any real position group.
    assert (wr_rows["probability_positive"] > 0.9).all()
    assert (wr_rows["pearson_r"] > 0.5).all()


def test_split_half_reliability_never_scores_no_local_metric_groups() -> None:
    cells = _synthetic_reliability_cells()
    ol_cells = pd.DataFrame(
        [
            {
                "gsis_id": "O1",
                "pos_group": "OL",
                "season": 2020,
                "career_age": 0,
                "metric_numerator": 5.0,  # deliberately non-null: exclusion must be structural
                "metric_denominator": 50.0,
                "primary_snaps": 50.0,
                "n_weeks": 4,
            }
        ]
    )
    combined = pd.concat([cells, ol_cells], ignore_index=True)
    curve = cross_sectional_curve(combined)
    reliability = split_half_reliability(combined, curve, samples=50, seed=1)
    assert "OL" not in set(reliability["pos_group"])


# ---------------------------------------------------------------------------
# End-to-end orchestrator (small, real snapshots on disk)
# ---------------------------------------------------------------------------


def _write_pbp_snapshot(root: Path, frames: dict[int, pd.DataFrame]) -> None:
    snapshot_dir = root / "20200101T000000Z"
    partitions = []
    for season, frame in frames.items():
        season_dir = snapshot_dir / f"season={season}"
        season_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(season_dir / "plays.parquet", index=False)
        partitions.append({"season": season})
    manifest = {"partitions": partitions, "seasons": sorted(frames)}
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_build_age_curves_end_to_end_smoke(tmp_path: Path) -> None:
    players_raw_root = tmp_path / "players_raw"
    values_raw_root = tmp_path / "values_raw"
    pbp_raw_root = tmp_path / "pbp_raw"

    injuries = pd.DataFrame(
        columns=[
            "season",
            "game_type",
            "team",
            "week",
            "gsis_id",
            "position",
            "report_status",
            "practice_status",
            "date_modified",
        ]
    )
    rosters = pd.DataFrame(
        [
            _roster_row("W9", "PW9", "A", "WR", 2019, 0),
            _roster_row("W9", "PW9", "A", "WR", 2020, 1),
            _roster_row("Q9", "PQ9", "A", "QB", 2019, 3),
            _roster_row("Q9", "PQ9", "A", "QB", 2020, 4),
            _roster_row("O9", "PO9", "A", "T", 2019, 2),
            _roster_row("O9", "PO9", "A", "T", 2020, 3),
        ]
    )
    snaps = pd.DataFrame(
        [
            _snap_row("PW9", "A", "WR", 2019, 1, offense=40.0, game_id="2019_01_A_OPP"),
            _snap_row("PW9", "A", "WR", 2020, 1, offense=42.0, game_id="2020_01_A_OPP"),
            _snap_row("PQ9", "A", "QB", 2019, 1, offense=55.0, game_id="2019_01_A_OPP"),
            _snap_row("PQ9", "A", "QB", 2020, 1, offense=56.0, game_id="2020_01_A_OPP"),
            _snap_row("PO9", "A", "T", 2019, 1, offense=50.0, game_id="2019_01_A_OPP"),
            _snap_row("PO9", "A", "T", 2020, 1, offense=51.0, game_id="2020_01_A_OPP"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stats_row("W9", "A", 2019, 1, receiving_epa=1.0, game_id="2019_01_A_OPP"),
            _stats_row("W9", "A", 2020, 1, receiving_epa=1.4, game_id="2020_01_A_OPP"),
        ]
    )
    pbp_frames = {
        2019: pd.DataFrame(
            [_pbp_row("2019_01_A_OPP", "Q9", 0.2), _pbp_row("2019_01_A_OPP", "Q9", 0.1)]
        ),
        2020: pd.DataFrame(
            [_pbp_row("2020_01_A_OPP", "Q9", 0.3), _pbp_row("2020_01_A_OPP", "Q9", 0.2)]
        ),
    }

    write_player_snapshot(
        injuries, rosters, snaps, players_raw_root, [2019, 2020], [2019, 2020], [2019, 2020]
    )
    write_player_value_snapshot(stats, values_raw_root, [2019, 2020])
    _write_pbp_snapshot(pbp_raw_root, pbp_frames)

    result = build_age_curves(players_raw_root, values_raw_root, pbp_raw_root, bootstrap_samples=20)

    assert not result.curve.empty
    assert {"WR", "QB", "OL"}.issubset(set(result.curve["pos_group"]))
    assert result.manifest["resolved_snapshots"]["players"]
    assert result.manifest["resolved_snapshots"]["player_values"]
    assert result.manifest["resolved_snapshots"]["pbp"]
    assert result.manifest["diagnostics"]["snap_rows_total"] == 6

    ol_rows = result.curve.loc[result.curve["pos_group"] == "OL"]
    assert (ol_rows["coverage_status"] == "no_local_metric").all()
    assert ol_rows["raw_rate"].isna().all()
