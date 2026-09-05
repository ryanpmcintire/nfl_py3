"""Tests for LEAD-24 Stage 1 (src/nfl_ats/rookie_wall.py): rookie workload wall.

Closing-grounds taxonomy (verbatim, per CLAUDE.md/AGENTS.md -- pasted here
because this file exercises measured reliability/bootstrap machinery an
adjudicating session might later read): an interval or CI that contains zero
is NEVER grounds to reject, fail, or close an experiment. At this
evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for
a real small signal. Only two grounds ever close a line of work: (1) refuted
mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of
zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
``unresolved_below_power``: report ``probability_positive``, never the
binary "contains zero". This module is QUALITY infrastructure with no ATS
direction of its own, so no test here closes anything -- the tests below
check the MACHINERY (the top-50 gate, the 70%-snap gate, trailing-window
leakage safety, and split-half correlation math on a known frame).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.rookie_wall import (
    EARLY_WEEKS,
    HIGH_WORKLOAD_SNAP_SHARE,
    LATE_WEEKS,
    TOP50_DRAFT_OVR_MAX,
    dependence_split_half_reliability,
    high_workload_player_seasons,
    late_season_high_dependence_flag,
    season_to_season_pairs,
    team_season_split_half,
    team_week_dependence_shares,
    top50_pick_lookup,
    trailing_dependence_feature,
    wall_candidates,
    within_player_half_season_delta,
)

# ---------------------------------------------------------------------------
# Small raw-frame builders
# ---------------------------------------------------------------------------


def _roster_row(gsis_id: str, pfr_id: str, season: int) -> dict[str, object]:
    return {
        "season": season,
        "team": "ANY",
        "position": "WR",
        "status": "ACT",
        "full_name": gsis_id,
        "gsis_id": gsis_id,
        "pfr_id": pfr_id,
        "years_exp": 0,
        "week": 1,
        "game_type": "REG",
    }


def _combine_row(
    pfr_id: str | None, draft_ovr: float | None, draft_year: float | None
) -> dict[str, object]:
    return {"pfr_id": pfr_id, "draft_ovr": draft_ovr, "draft_year": draft_year}


def _panel_row(
    gsis_id: str,
    season: int,
    week: int,
    *,
    team: str = "KC",
    pos_group: str = "WR",
    career_age: int = 0,
    offense_pct: float = 0.0,
    defense_pct: float = 0.0,
    is_top50_pick: bool = True,
    is_rookie: bool | None = None,
    metric_numerator: float | None = 0.0,
    metric_denominator: float | None = 60.0,
) -> dict[str, object]:
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "team": team,
        "position": pos_group,
        "pos_group": pos_group,
        "career_age": career_age,
        "offense_pct": offense_pct,
        "defense_pct": defense_pct,
        "offense_snaps": offense_pct * 60,
        "defense_snaps": defense_pct * 60,
        "st_snaps": 0.0,
        "metric_numerator": metric_numerator,
        "metric_denominator": metric_denominator,
        "primary_snaps": metric_denominator,
        "coverage_status": "metric",
        "is_top50_pick": is_top50_pick,
        "is_rookie": (career_age == 0) if is_rookie is None else is_rookie,
    }


# ---------------------------------------------------------------------------
# Top-50-pick gate
# ---------------------------------------------------------------------------


def test_top50_pick_lookup_flags_only_resolved_top50_picks() -> None:
    rosters = pd.DataFrame(
        [
            _roster_row("00-p1", "pfr1", 2020),
            _roster_row("00-p2", "pfr2", 2020),
            # pfr3 never appears in rosters -> unjoinable.
        ]
    )
    combine = pd.DataFrame(
        [
            _combine_row("pfr1", 1.0, 2020.0),  # top-50
            _combine_row("pfr2", TOP50_DRAFT_OVR_MAX + 1.0, 2020.0),  # just outside top-50
            _combine_row("pfr3", 5.0, 2020.0),  # unjoinable pfr_id
            _combine_row("pfr4", None, 2020.0),  # no draft_ovr at all
            _combine_row(None, 3.0, 2020.0),  # no pfr_id at all
        ]
    )

    lookup, diagnostics = top50_pick_lookup(combine, rosters)

    assert lookup == {"00-p1": True, "00-p2": False}
    assert diagnostics["n_combine_rows"] == 5
    assert diagnostics["n_with_draft_ovr"] == 4  # pfr4's row (None draft_ovr) excluded
    assert diagnostics["n_with_draft_ovr_and_pfr_id"] == 3  # pfr1, pfr2, pfr3
    assert diagnostics["n_joined_to_gsis"] == 2  # pfr3 never resolves
    assert diagnostics["n_unique_players_top50"] == 1


def test_top50_pick_lookup_keeps_earliest_draft_year_on_duplicate_pfr_id() -> None:
    rosters = pd.DataFrame([_roster_row("00-p1", "pfr1", 2019)])
    combine = pd.DataFrame(
        [
            _combine_row("pfr1", 60.0, 2020.0),  # later, not-top-50 combine invite
            _combine_row("pfr1", 2.0, 2019.0),  # earlier, real draft: top-50
        ]
    )

    lookup, _ = top50_pick_lookup(combine, rosters)

    assert lookup["00-p1"] is True


def test_top50_pick_lookup_requires_draft_columns() -> None:
    rosters = pd.DataFrame([_roster_row("00-p1", "pfr1", 2020)])
    combine = pd.DataFrame({"pfr_id": ["pfr1"]})

    with pytest.raises(Exception, match="missing columns"):
        top50_pick_lookup(combine, rosters)


# ---------------------------------------------------------------------------
# 70%-snap-share gate (weeks 1-11 only)
# ---------------------------------------------------------------------------


def test_high_workload_gate_uses_only_weeks_1_through_11() -> None:
    rows = []
    # Player A: 0.80 offense_pct every week 1-11 (qualifies), then a
    # deliberately huge value in a LATE week that must NOT move the gate.
    for week in EARLY_WEEKS:
        rows.append(_panel_row("00-a", 2020, week, offense_pct=0.80))
    rows.append(_panel_row("00-a", 2020, 15, offense_pct=0.01))

    # Player B: only 0.50 offense_pct all weeks 1-11 -- does not qualify.
    for week in EARLY_WEEKS:
        rows.append(_panel_row("00-b", 2020, week, offense_pct=0.50))

    # Player C: qualifies via DEFENSE share instead of offense.
    for week in EARLY_WEEKS:
        rows.append(
            _panel_row("00-c", 2020, week, pos_group="LB", offense_pct=0.0, defense_pct=0.75)
        )

    panel = pd.DataFrame(rows)
    gate = high_workload_player_seasons(panel, threshold=HIGH_WORKLOAD_SNAP_SHARE)
    gate = gate.set_index("gsis_id")

    assert bool(gate.loc["00-a", "high_workload"]) is True
    assert gate.loc["00-a", "mean_offense_pct"] == pytest.approx(0.80)
    assert bool(gate.loc["00-b", "high_workload"]) is False
    assert bool(gate.loc["00-c", "high_workload"]) is True


# ---------------------------------------------------------------------------
# Within-player half-season delta and the snap floor
# ---------------------------------------------------------------------------


def test_within_player_half_season_delta_requires_snap_floor_in_both_halves() -> None:
    rows = []
    # Player A: 150 snaps/week both halves -- clears the floor both sides.
    for week in EARLY_WEEKS:
        rows.append(_panel_row("00-a", 2020, week, metric_numerator=3.0, metric_denominator=150.0))
    for week in LATE_WEEKS:
        rows.append(_panel_row("00-a", 2020, week, metric_numerator=1.5, metric_denominator=150.0))

    # Player B: plenty of snaps early, but a garbage-time-only late half
    # (well under the 100-snap floor) -- must be EXCLUDED entirely.
    for week in EARLY_WEEKS:
        rows.append(_panel_row("00-b", 2020, week, metric_numerator=3.0, metric_denominator=150.0))
    for week in LATE_WEEKS:
        rows.append(_panel_row("00-b", 2020, week, metric_numerator=0.1, metric_denominator=5.0))

    panel = pd.DataFrame(rows)
    delta = within_player_half_season_delta(panel).set_index("gsis_id")

    assert "00-a" in delta.index
    assert "00-b" not in delta.index
    # rate_early = 3.0/150 = 0.02 per week -> summed over 11 weeks: 33/1650=0.02
    # rate_late = 1.5/150 = 0.01 per week -> summed over 6 weeks: 9/900=0.01
    assert delta.loc["00-a", "rate_early"] == pytest.approx(0.02)
    assert delta.loc["00-a", "rate_late"] == pytest.approx(0.01)
    assert delta.loc["00-a", "delta"] == pytest.approx(-0.01)


def test_wall_candidates_separates_rookie_top50_and_veteran_populations() -> None:
    rows = []

    def add_player(gsis_id: str, career_age: int, is_top50: bool, workload: float) -> None:
        for week in EARLY_WEEKS:
            rows.append(
                _panel_row(
                    gsis_id,
                    2020,
                    week,
                    career_age=career_age,
                    is_top50_pick=is_top50,
                    offense_pct=workload,
                    metric_numerator=2.0,
                    metric_denominator=150.0,
                )
            )
        for week in LATE_WEEKS:
            rows.append(
                _panel_row(
                    gsis_id,
                    2020,
                    week,
                    career_age=career_age,
                    is_top50_pick=is_top50,
                    offense_pct=workload,
                    metric_numerator=1.0,
                    metric_denominator=150.0,
                )
            )

    add_player("00-rookie-top50", career_age=0, is_top50=True, workload=0.80)  # rookie population
    add_player("00-rookie-late", career_age=0, is_top50=False, workload=0.80)  # excluded: not top50
    add_player("00-vet-heavy", career_age=5, is_top50=False, workload=0.80)  # veteran control
    add_player(
        "00-vet-light", career_age=5, is_top50=False, workload=0.20
    )  # excluded: not high workload
    add_player(
        "00-vet-young", career_age=1, is_top50=False, workload=0.80
    )  # excluded: career_age<3, not rookie either

    panel = pd.DataFrame(rows)
    candidates = wall_candidates(panel)
    by_population = candidates.groupby("population")["gsis_id"].apply(set)

    assert by_population["rookie_top50_high_workload"] == {"00-rookie-top50"}
    assert by_population["veteran_high_workload_control"] == {"00-vet-heavy"}


# ---------------------------------------------------------------------------
# Dependence metric: aggregation, trailing-window leakage, and its threshold
# ---------------------------------------------------------------------------


def test_team_week_dependence_shares_sums_qualifying_rookies_and_covers_empty_weeks() -> None:
    rows = [
        _panel_row("00-r1", 2020, 1, team="KC", offense_pct=0.60, is_top50_pick=True, career_age=0),
        _panel_row("00-r2", 2020, 1, team="KC", offense_pct=0.30, is_top50_pick=True, career_age=0),
        # A non-top50 rookie on the same team-week must NOT be counted.
        _panel_row(
            "00-r3", 2020, 1, team="KC", offense_pct=0.90, is_top50_pick=False, career_age=0
        ),
        # week 2 has snap rows but no qualifying rookie at all -- must show 0.0, not be absent.
        _panel_row(
            "00-vet", 2020, 2, team="KC", offense_pct=0.90, is_top50_pick=False, career_age=6
        ),
    ]
    panel = pd.DataFrame(rows)
    shares = team_week_dependence_shares(panel).set_index("week")

    assert shares.loc[1, "offense_share"] == pytest.approx(0.90)
    assert shares.loc[1, "n_top50_rookies"] == 2
    assert shares.loc[2, "offense_share"] == pytest.approx(0.0)
    assert shares.loc[2, "share_sum"] == pytest.approx(0.0)


def test_trailing_dependence_feature_excludes_the_current_weeks_own_value() -> None:
    shares = pd.DataFrame(
        {
            "team": ["KC"] * 5,
            "season": [2020] * 5,
            "week": [1, 2, 3, 4, 5],
            "offense_share": [0.60, 0.60, 0.60, 0.60, 0.60],
            "defense_share": [0.0] * 5,
            "n_top50_rookies": [1] * 5,
            "share_sum": [0.60] * 5,
        }
    )
    baseline = trailing_dependence_feature(shares)

    perturbed = shares.copy()
    perturbed.loc[perturbed["week"] == 3, "offense_share"] = 999.0
    perturbed.loc[perturbed["week"] == 3, "share_sum"] = 999.0
    perturbed_trailing = trailing_dependence_feature(perturbed)

    week3_before = baseline.loc[baseline["week"] == 3, "trailing_offense_share"].iloc[0]
    week3_after = perturbed_trailing.loc[
        perturbed_trailing["week"] == 3, "trailing_offense_share"
    ].iloc[0]
    assert week3_before == pytest.approx(
        week3_after
    )  # own current value never leaks into own trailing

    # A LATER week must pick up the perturbation -- proves the window is a
    # real rolling-prior-games mean, not a no-op.
    week4_after = perturbed_trailing.loc[
        perturbed_trailing["week"] == 4, "trailing_offense_share"
    ].iloc[0]
    assert week4_after > 100.0

    # week 1 has no prior game at all -- must be NaN, never a fabricated 0.
    week1_trailing = baseline.loc[baseline["week"] == 1, "trailing_offense_share"].iloc[0]
    assert np.isnan(week1_trailing)


def test_late_season_high_dependence_flag_needs_both_the_percentile_and_the_week_gate() -> None:
    # Two teams, same season/week: KC trails much higher than NE every week.
    rows = []
    for week in range(1, 14):
        rows.append(
            {
                "team": "KC",
                "season": 2020,
                "week": week,
                "offense_share": 1.0,
                "defense_share": 0.0,
                "n_top50_rookies": 1,
                "share_sum": 1.0,
            }
        )
        rows.append(
            {
                "team": "NE",
                "season": 2020,
                "week": week,
                "offense_share": 0.0,
                "defense_share": 0.0,
                "n_top50_rookies": 0,
                "share_sum": 0.0,
            }
        )
    shares = pd.DataFrame(rows)
    trailing = trailing_dependence_feature(shares)
    flagged = late_season_high_dependence_flag(trailing, percentile=0.80, late_week_min=12)

    kc = flagged.loc[flagged["team"] == "KC"].set_index("week")
    # KC clears the 80th percentile every week it has a trailing value, but
    # week < 12 must never be flagged regardless.
    assert bool(kc.loc[5, "late_season_high_dependence"]) is False
    assert bool(kc.loc[13, "late_season_high_dependence"]) is True
    ne = flagged.loc[flagged["team"] == "NE"].set_index("week")
    assert bool(ne.loc[13, "late_season_high_dependence"]) is False


# ---------------------------------------------------------------------------
# Split-half reliability math on a known frame
# ---------------------------------------------------------------------------


def test_team_season_split_half_pairs_odd_and_even_weeks_per_team_season() -> None:
    shares = pd.DataFrame(
        {
            "team": ["KC", "KC", "KC", "KC"],
            "season": [2020, 2020, 2020, 2020],
            "week": [1, 2, 3, 4],
            "share_sum": [1.0, 0.0, 1.0, 0.0],
        }
    )
    pivot = team_season_split_half(shares).set_index(["team", "season"])
    assert pivot.loc[("KC", 2020), "odd"] == pytest.approx(1.0)  # weeks 1,3
    assert pivot.loc[("KC", 2020), "even"] == pytest.approx(0.0)  # weeks 2,4


def test_season_to_season_pairs_only_the_same_team_adjacent_seasons() -> None:
    shares = pd.DataFrame(
        {
            "team": ["KC", "KC", "NE"],
            "season": [2020, 2021, 2020],
            "week": [1, 1, 1],
            "share_sum": [0.5, 0.8, 0.2],
        }
    )
    pairs = season_to_season_pairs(shares)
    # Only KC 2020->2021 has both seasons present for the SAME team; NE 2020
    # has no NE 2021 row to pair with.
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["team"] == "KC"
    assert row["share_this_season"] == pytest.approx(0.5)
    assert row["share_next_season"] == pytest.approx(0.8)


def test_dependence_split_half_reliability_recovers_a_known_perfect_correlation() -> None:
    # 6 team-seasons across 3 seasons (2 teams/season) with odd == even
    # EXACTLY -- a hand-computable r=1.0 for the within-season scheme.
    rows = []
    for season in (2020, 2021, 2022):
        for team, level in (("KC", 0.9), ("NE", 0.1)):
            for week in range(1, 9):
                rows.append(
                    {
                        "team": team,
                        "season": season,
                        "week": week,
                        "share_sum": level,
                    }
                )
    shares = pd.DataFrame(rows)

    reliability = dependence_split_half_reliability(shares, samples=500, seed=1).set_index("scheme")
    within = reliability.loc["odd_even_weeks_team_season"]
    assert within["pearson_r"] == pytest.approx(1.0, abs=1e-9)
    assert within["spearman_rho"] == pytest.approx(1.0, abs=1e-9)
    assert within["bootstrap_ci95_low"] > 0.99
    assert within["probability_positive"] == pytest.approx(1.0)
    # The observed r=1.0 must sit above (or at) essentially the whole
    # team-label shuffle null, since the null destroys the team-specific
    # pairing that produces the perfect correlation here.
    assert within["shuffle_null_percentile_of_observed"] > 0.9


def test_reliability_row_reports_nan_not_a_crash_below_the_minimum_unit_count() -> None:
    shares = pd.DataFrame(
        {
            "team": ["KC", "KC"],
            "season": [2020, 2021],
            "week": [1, 1],
            "share_sum": [0.5, 0.6],
        }
    )
    reliability = dependence_split_half_reliability(shares, samples=200, seed=1)
    # Only one team-season pair for the within-season scheme (dropna on
    # odd/even leaves nothing without both parities) and one team-to-team
    # pair for season-to-season -- both well under the minimum unit count.
    assert reliability.empty or reliability["pearson_r"].isna().all()
