"""Tests for the team-explorer page logic and rendering.

Driven entirely with the deterministic :func:`make_schema_fixture` (and a
small hand-built feature table) so the contract holds with no on-disk parquet.
Every function degrades to a clean empty state when its input is absent, and
that path is asserted directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats import public_board, team_explorer
from nfl_ats.constants import STATE_METRICS


def _feature_table() -> pd.DataFrame:
    """A tiny canonical per-game feature table with home_/away_ state columns."""

    rows = {
        "game_id": ["2025_01_ARI_BUF"],
        "season": [2025],
        "gameday": [pd.Timestamp("2025-09-07")],
        "home_team": ["ARI"],
        "away_team": ["BUF"],
    }
    for side, value in (("home", 0.12), ("away", -0.08)):
        for metric in STATE_METRICS:
            rows[f"{side}_{metric}"] = [value]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema fixture + schema contract
# ---------------------------------------------------------------------------


def test_make_schema_fixture_carries_canonical_columns_and_is_deterministic() -> None:
    first = team_explorer.make_schema_fixture(seed=7)
    second = team_explorer.make_schema_fixture(seed=7)
    assert list(first.columns) == [
        *team_explorer.IDENTIFIER_COLUMNS,
        *team_explorer.STATE_COLUMNS,
    ]
    assert len(first) == 4 * 3 * 9  # teams x seasons x games
    # Deterministic for a fixed seed.
    pd.testing.assert_frame_equal(first, second)


def test_make_schema_fixture_differs_by_seed() -> None:
    a = team_explorer.make_schema_fixture(seed=1)
    b = team_explorer.make_schema_fixture(seed=2)
    assert not a.equals(b)


# ---------------------------------------------------------------------------
# Coercion / validation
# ---------------------------------------------------------------------------


def test_coerce_state_table_rejects_missing_identifier() -> None:
    bad = team_explorer.make_schema_fixture().drop(columns=["team"])
    with pytest.raises(ValueError, match="identifier"):
        team_explorer.coerce_state_table(bad)


def test_coerce_state_table_rejects_missing_metric() -> None:
    bad = team_explorer.make_schema_fixture().drop(columns=["state_off_epa_per_play"])
    with pytest.raises(ValueError, match="state columns"):
        team_explorer.coerce_state_table(bad)


def test_coerce_state_table_returns_empty_frame_for_none() -> None:
    frame = team_explorer.coerce_state_table(None)
    assert frame.empty
    assert list(frame.columns) == [
        *team_explorer.IDENTIFIER_COLUMNS,
        *team_explorer.STATE_COLUMNS,
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_team_trends_empty_input_is_empty() -> None:
    trends = team_explorer.aggregate_team_trends(pd.DataFrame())
    assert trends.latest_season is None
    assert trends.teams == []
    assert trends.latest.empty
    assert trends.trend.empty


def test_aggregate_team_trends_computes_latest_season_and_z() -> None:
    df = team_explorer.make_schema_fixture()
    trends = team_explorer.aggregate_team_trends(df)
    assert trends.latest_season == 2025
    assert set(trends.teams) == {"ARI", "BUF", "KC", "SF"}
    # One row per (team, metric) in the latest season.
    assert len(trends.latest) == len(trends.teams) * len(STATE_METRICS)
    # z is value minus the league mean for that season/metric -> sums to ~0.
    for metric in STATE_METRICS[:3]:
        col = trends.latest.loc[trends.latest["metric"] == metric, "z"]
        assert col.sum() == pytest.approx(0.0, abs=1e-9)
    # Trend has one row per (team, metric, season).
    assert len(trends.trend) == len(trends.teams) * len(STATE_METRICS) * 3


def test_aggregate_team_trends_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        team_explorer.aggregate_team_trends(
            team_explorer.make_schema_fixture(), metrics=["not_real"]
        )


def test_team_state_payload_maps_team_to_metric_z() -> None:
    df = team_explorer.make_schema_fixture()
    trends = team_explorer.aggregate_team_trends(df)
    payload = team_explorer.team_state_payload(trends)
    assert set(payload) == set(trends.teams)
    assert "off_epa_per_play" in payload["ARI"]
    # Empty trends -> empty payload.
    assert team_explorer.team_state_payload(team_explorer.TeamTrends.empty()) == {}


# ---------------------------------------------------------------------------
# Feature-table conversion
# ---------------------------------------------------------------------------


def test_feature_table_to_team_states_melts_home_and_away() -> None:
    converted = team_explorer.feature_table_to_team_states(_feature_table())
    assert converted is not None
    assert set(converted["team"]) == {"ARI", "BUF"}
    # Two rows (one per side) and the canonical state columns survive.
    assert len(converted) == 2
    assert "state_off_epa_per_play" in converted.columns
    ari = converted.loc[converted["team"] == "ARI", "state_off_epa_per_play"].iloc[0]
    assert ari == pytest.approx(0.12)


def test_feature_table_to_team_states_returns_none_without_state_columns() -> None:
    assert (
        team_explorer.feature_table_to_team_states(
            pd.DataFrame({"home_team": ["ARI"], "away_team": ["BUF"]})
        )
        is None
    )
    assert team_explorer.feature_table_to_team_states(None) is None


# ---------------------------------------------------------------------------
# Rendering (design-system + fail-open contract)
# ---------------------------------------------------------------------------


def _assert_public_safe(page: str) -> None:
    assert page.startswith("<!doctype html>")
    assert '<div class="ats">' in page
    assert public_board.DISCLAIMER_SHORT in page
    assert public_board.DISCLAIMER_FULL in page
    assert 'aria-current="page"' in page


def test_render_team_explorer_page_empty_state_without_data() -> None:
    page = public_board.render_team_explorer_page(
        pd.DataFrame(), generated_at=pd.Timestamp("2026-08-24", tz="UTC")
    )
    _assert_public_safe(page)
    assert "No team-state data yet" in page
    assert "Latest season shown" not in page


def test_render_team_explorer_page_renders_all_sections_from_fixture() -> None:
    page = public_board.render_team_explorer_page(
        team_explorer.make_schema_fixture(),
        generated_at=pd.Timestamp("2026-08-24", tz="UTC"),
    )
    _assert_public_safe(page)
    assert "Per-team pregame state, by season" in page
    assert "Per-team season trend" in page
    assert "Matchup comparison" in page
    assert "Latest season shown: 2025" in page
    # Interactive comparer payload + controls present.
    assert 'id="ats-te-data"' in page
    assert 'id="ats-te-a"' in page
    assert 'id="ats-te-b"' in page
    # Honesty footnote about rate-stat direction is present.
    assert "not necessarily better" in page


def test_render_team_explorer_page_respects_custom_metrics() -> None:
    page = public_board.render_team_explorer_page(
        team_explorer.make_schema_fixture(),
        metrics=["off_epa_per_play", "def_epa_per_play"],
        generated_at=pd.Timestamp("2026-08-24", tz="UTC"),
    )
    # Only the two requested metrics appear as overview headers.
    assert "Offense EPA/play" in page
    assert "Defense EPA/play allowed" in page
    assert "Point differential" not in page
