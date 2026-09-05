"""UI-20: every listed player gets a play probability, not only the QB.

Owner complaint: "why is that only quarterbacks have the lineup percentage
filled in?" (measured cause: ``scripts/build_week_lineups.py`` only ever set
``play_probability`` for the one QB the active model consumed). This file
covers:

* ``nfl_ats.lineup_availability``'s no-designation base-rate derivation and
  per-player resolver, on a tiny synthetic roster/injury/snap fixture (unit
  math, not the real multi-season snapshot).
* ``scripts.build_week_lineups._team_payload``'s every-player coverage, the
  bit-identical QB guarantee, and the point-in-time leakage rule (a report
  dated after the artifact's ``generated_at`` must not change a player's
  number).
* The render legend/em-dash rule in ``nfl_ats.board_terminal``.
* The lineup-aware assistant's availability answer for a non-QB player.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_ats import board_terminal
from nfl_ats.board_assistant import _tokens
from nfl_ats.board_assistant_lineups import build_lineup_knowledge, player_availability_answer
from nfl_ats.board_content import AttributionPanel, GameDive
from nfl_ats.lineup_availability import (
    RECENT_ROLE_NO_RECENT_ROLE,
    RECENT_ROLE_RETURNING_CONTRIBUTOR,
    RECENT_ROLE_UNKNOWN_NO_HISTORY,
    build_no_designation_outcomes,
    build_no_designation_rates,
    depth_chart_position_group,
    latest_recent_roles,
    no_designation_rate_lookup,
    no_designation_unavailability,
    resolve_play_probability,
)
from nfl_ats.lineup_view import team_lineup
from nfl_ats.players import canonicalize_injuries
from scripts.build_week_lineups import _team_payload, _visible_injuries_by_team

# ---------------------------------------------------------------------------
# Synthetic roster/injury/snap fixture shared by the lineup_availability tests
# ---------------------------------------------------------------------------
#
# Two "eligible" (status == ACT) WRs on team AAA across 2024 weeks 1-3:
#   p1 plays every week (snaps > 0) -> unavailable=0.0 every week.
#   p2 never plays (no snap rows at all) -> unavailable=1.0 every week.
# p3 is INA (must be EXCLUDED -- "INA" is nflverse's own weekly inactive
# tag, so folding it in would make "unavailable" tautological).
# p4 is listed on the week-2 injury report (must be EXCLUDED from the
# not-listed pool for that week only).


def _rosters() -> pd.DataFrame:
    rows = []
    for player_id, name in (("p1", "P One"), ("p2", "P Two")):
        for week in (1, 2, 3):
            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "team": "AAA",
                    "gsis_id": player_id,
                    "position": "WR",
                    "status": "ACT",
                    "pfr_id": f"{player_id}_pfr",
                    "full_name": name,
                }
            )
    rows.append(
        {
            "season": 2024,
            "week": 2,
            "team": "AAA",
            "gsis_id": "p3",
            "position": "WR",
            "status": "INA",
            "pfr_id": "p3_pfr",
            "full_name": "P Three",
        }
    )
    rows.append(
        {
            "season": 2024,
            "week": 2,
            "team": "AAA",
            "gsis_id": "p4",
            "position": "WR",
            "status": "ACT",
            "pfr_id": "p4_pfr",
            "full_name": "P Four",
        }
    )
    return pd.DataFrame(rows)


def _snaps() -> pd.DataFrame:
    # Only p1 ever records a snap row; p2/p3/p4 are absent entirely (the
    # left-join in _active_roster_snap_timeline fills their `played` as
    # False, exactly like a real snap_counts table with no row for a
    # player who never took the field).
    rows = [
        {
            "season": 2024,
            "week": week,
            "team": "AAA",
            "pfr_player_id": "p1_pfr",
            "player": "P One",
            "offense_snaps": 35,
            "defense_snaps": 0,
            "st_snaps": 0,
        }
        for week in (1, 2, 3)
    ]
    return pd.DataFrame(rows)


def _injuries() -> pd.DataFrame:
    return pd.DataFrame([{"season": 2024, "week": 2, "team": "AAA", "gsis_id": "p4"}])


def test_no_designation_outcomes_exclude_ina_and_listed_rows() -> None:
    outcomes = build_no_designation_outcomes(_injuries(), _rosters(), _snaps())
    # p4 (listed week 2) and p3 (INA) never appear; p1/p2 appear for all
    # three weeks each -- six rows total.
    assert set(outcomes["gsis_id"]) == {"p1", "p2"}
    assert len(outcomes) == 6
    p1_rows = outcomes.loc[outcomes["gsis_id"] == "p1"].sort_values("week")
    p2_rows = outcomes.loc[outcomes["gsis_id"] == "p2"].sort_values("week")
    assert p1_rows["unavailable"].tolist() == [0.0, 0.0, 0.0]
    assert p2_rows["unavailable"].tolist() == [1.0, 1.0, 1.0]
    # recent_role: week 1 has no earlier ACT appearance for either player;
    # week 2/3 look at that SAME player's own immediately preceding week.
    assert p1_rows["recent_role"].tolist() == [
        RECENT_ROLE_UNKNOWN_NO_HISTORY,
        RECENT_ROLE_RETURNING_CONTRIBUTOR,
        RECENT_ROLE_RETURNING_CONTRIBUTOR,
    ]
    assert p2_rows["recent_role"].tolist() == [
        RECENT_ROLE_UNKNOWN_NO_HISTORY,
        RECENT_ROLE_NO_RECENT_ROLE,
        RECENT_ROLE_NO_RECENT_ROLE,
    ]


def test_no_designation_rates_shrinkage_math() -> None:
    outcomes = build_no_designation_outcomes(_injuries(), _rosters(), _snaps())
    rates = build_no_designation_rates(outcomes, target_seasons=[2025])
    lookup = no_designation_rate_lookup(rates)
    # Global: 3 of 6 rows unavailable -> exactly 0.5, and the "skill"
    # (WR) group already sits at the global rate, so shrinking it toward
    # 0.5 leaves it at 0.5 too.
    assert lookup[(2025, "__all__", "__all__")] == pytest.approx(0.5)
    assert lookup[(2025, "skill", "__all__")] == pytest.approx(0.5)
    # unknown_no_history: 1 of 2 rows unavailable, shrunk toward the
    # group rate (0.5) with role_prior=20 -> (1 + 20*0.5) / (2 + 20).
    assert lookup[(2025, "skill", RECENT_ROLE_UNKNOWN_NO_HISTORY)] == pytest.approx(
        (1 + 20 * 0.5) / 22
    )
    # returning_contributor: 0 of 2 unavailable -> pulled UP toward 0.5,
    # never all the way to 0 -- the shrinkage is doing real work here.
    returning_rate = lookup[(2025, "skill", RECENT_ROLE_RETURNING_CONTRIBUTOR)]
    assert returning_rate == pytest.approx((0 + 20 * 0.5) / 22)
    assert 0.0 < returning_rate < 0.5
    # no_recent_role: 2 of 2 unavailable -> pulled DOWN toward 0.5.
    no_role_rate = lookup[(2025, "skill", RECENT_ROLE_NO_RECENT_ROLE)]
    assert no_role_rate == pytest.approx((2 + 20 * 0.5) / 22)
    assert 0.5 < no_role_rate < 1.0
    # A position group absent from training (offensive_line) falls all
    # the way back to the global rate rather than returning None.
    fallback = no_designation_unavailability(
        lookup, target_season=2025, position="LT", recent_role="anything"
    )
    assert fallback == pytest.approx(0.5)
    # An unseen target season returns None rather than inventing a number.
    assert (
        no_designation_unavailability(lookup, target_season=1999, position="WR", recent_role="x")
        is None
    )


def test_latest_recent_roles_reflects_each_players_own_last_act_appearance() -> None:
    roles = latest_recent_roles(_rosters(), _snaps(), before_season=2025)
    assert roles["p1"] == RECENT_ROLE_RETURNING_CONTRIBUTOR  # played week 3
    assert roles["p2"] == RECENT_ROLE_NO_RECENT_ROLE  # never played
    assert roles["p4"] == RECENT_ROLE_NO_RECENT_ROLE  # ACT week 2, no snaps
    assert "p3" not in roles  # INA is never eligible, so it has no role at all


def test_depth_chart_position_group_understands_side_specific_tags() -> None:
    # nflverse depth charts (unlike injuries/rosters) use side-specific
    # tags the bare availability.position_group cannot see.
    assert depth_chart_position_group("LDE") == "front"
    assert depth_chart_position_group("RCB") == "secondary"
    assert depth_chart_position_group("LT") == "offensive_line"
    assert depth_chart_position_group("QB") == "skill"


def test_resolve_play_probability_three_paths() -> None:
    lookup = no_designation_rate_lookup(
        build_no_designation_rates(
            build_no_designation_outcomes(_injuries(), _rosters(), _snaps()),
            target_seasons=[2025],
        )
    )
    # No gsis_id at all: never invents a number.
    probability, source, reason = resolve_play_probability(
        gsis_id=None,
        position="WR",
        target_season=2025,
        current_injury=None,
        learned_lookup=None,
        no_designation_lookup=lookup,
    )
    assert probability is None
    assert source == "unavailable"
    assert "gsis_id" in reason

    # Listed this week: uses the fixed prior (learned_lookup=None) on the
    # player's OWN current-week report/practice status.
    current_injury = pd.Series({"report_status": "Out", "practice_status": None, "position": "WR"})
    probability, source, reason = resolve_play_probability(
        gsis_id="wr-1",
        position="WR",
        target_season=2025,
        current_injury=current_injury,
        learned_lookup=None,
        no_designation_lookup=lookup,
    )
    assert probability == pytest.approx(0.0)  # "Out" -> fixed_unavailability == 1.0
    assert source == "availability_model"
    assert "listed" in reason

    # No designation this week: falls back to the position's no-designation
    # base rate, conditioned on recent_role -- a real number, never None.
    probability, source, reason = resolve_play_probability(
        gsis_id="wr-1",
        position="WR",
        target_season=2025,
        current_injury=None,
        learned_lookup=None,
        no_designation_lookup=lookup,
        recent_role=RECENT_ROLE_RETURNING_CONTRIBUTOR,
    )
    assert probability is not None
    assert 0.0 < probability < 1.0
    assert source == "availability_model"
    assert "no injury designation" in reason

    # No designation AND no lookup available at all: honestly None.
    probability, source, reason = resolve_play_probability(
        gsis_id="wr-1",
        position="WR",
        target_season=2025,
        current_injury=None,
        learned_lookup=None,
        no_designation_lookup=None,
    )
    assert probability is None
    assert source == "unavailable"


# ---------------------------------------------------------------------------
# scripts.build_week_lineups._team_payload: every-player coverage, the QB
# bit-identical guarantee, and point-in-time leakage.
# ---------------------------------------------------------------------------


def _synthetic_depth(team: str, players: list[dict]) -> pd.DataFrame:
    rows = []
    for player in players:
        rows.append(
            {
                "team": team,
                "pos_abb": player["position"],
                "pos_rank": player.get("depth", 1),
                "player_name": player["name"],
                "gsis_id": player.get("gsis_id"),
                "observed_at_utc": "2026-09-01T00:00:00Z",
            }
        )
    return pd.DataFrame(rows)


def test_every_player_with_a_gsis_id_gets_a_real_probability() -> None:
    depth = _synthetic_depth(
        "KC",
        [
            {"name": "QB Model", "position": "QB", "depth": 1, "gsis_id": "qb-model"},
            {"name": "QB Backup", "position": "QB", "depth": 2, "gsis_id": "qb-backup"},
            {"name": "WR One", "position": "WR", "depth": 1, "gsis_id": "wr-1"},
            {"name": "OL Deep", "position": "LT", "depth": 3, "gsis_id": "ol-deep"},
            {"name": "No Gsis", "position": "TE", "depth": 4, "gsis_id": None},
        ],
    )
    no_designation_lookup = {(2026, "__all__", "__all__"): 0.30}
    payload = _team_payload(
        depth,
        "KC",
        "qb-model",
        0.9123456789,
        target_season=2026,
        current_injuries=None,
        learned_lookup=None,
        no_designation_lookup=no_designation_lookup,
        recent_roles={},
    )
    by_gsis = {player["gsis_id"]: player for player in payload["players"]}

    # The base-model QB: bit-identical to the forecast's own input, never
    # recomputed by the availability model.
    assert by_gsis["qb-model"]["play_probability"] == 0.9123456789
    assert by_gsis["qb-model"]["probability_source"] == "base_model_qb"

    # Every other player with a gsis_id -- including a backup QB and a
    # deep-bench offensive lineman -- gets a real, non-None probability
    # from the availability model, not left blank.
    for key in ("qb-backup", "wr-1", "ol-deep"):
        player = by_gsis[key]
        assert player["play_probability"] is not None
        assert 0.0 <= player["play_probability"] <= 1.0
        assert player["probability_source"] == "availability_model"
        assert player["probability_reason"]

    # A row with no gsis_id at all is the only one left blank, and it says
    # exactly why.
    none_row = next(player for player in payload["players"] if player["gsis_id"] is None)
    assert none_row["play_probability"] is None
    assert none_row["probability_source"] == "unavailable"
    assert "gsis_id" in none_row["probability_reason"]


def test_qb_probability_stays_bit_identical_to_the_forecast_input() -> None:
    depth = _synthetic_depth(
        "KC", [{"name": "QB Model", "position": "QB", "depth": 1, "gsis_id": "qb-model"}]
    )
    for forecast_probability in (0.0, 1.0, 0.123456789012345, 0.5):
        payload = _team_payload(
            depth,
            "KC",
            "qb-model",
            forecast_probability,
            target_season=2026,
            current_injuries=None,
            learned_lookup=None,
            no_designation_lookup=None,
            recent_roles={},
        )
        qb = payload["players"][0]
        assert qb["play_probability"] == forecast_probability
        assert qb["probability_source"] == "base_model_qb"


def test_qb_probability_is_none_when_the_forecast_never_supplied_one() -> None:
    depth = _synthetic_depth(
        "KC", [{"name": "QB Model", "position": "QB", "depth": 1, "gsis_id": "qb-model"}]
    )
    payload = _team_payload(
        depth,
        "KC",
        "qb-model",
        None,
        target_season=2026,
        current_injuries=None,
        learned_lookup=None,
        no_designation_lookup=None,
        recent_roles={},
    )
    qb = payload["players"][0]
    assert qb["play_probability"] is None
    assert qb["probability_source"] == "base_model_qb"
    assert "unavailable" in qb["probability_reason"]


def test_a_later_dated_injury_report_never_changes_a_players_number() -> None:
    """Point-in-time discipline: only injury reports observed strictly
    before the artifact's own ``generated_at`` may move a player's number.
    """

    generated_at = pd.Timestamp("2026-09-10T12:00:00Z")
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "home_team": "KC",
                "away_team": "DEN",
                "kickoff": pd.Timestamp("2026-09-14T20:00:00Z"),
            }
        ]
    )
    depth = _synthetic_depth(
        "KC", [{"name": "WR One", "position": "WR", "depth": 1, "gsis_id": "wr-1"}]
    )
    no_designation_lookup = {(2026, "__all__", "__all__"): 0.10}

    def probability_for(raw_rows: list[dict]) -> dict:
        if raw_rows:
            canonical = canonicalize_injuries(
                pd.DataFrame(raw_rows),
                include_postseason=False,
                timestamp_fallback="week_proxy",
                schedule=schedule,
            )
            visible = canonical.loc[canonical["effective_observed_at"] <= generated_at].copy()
        else:
            visible = pd.DataFrame()
        current_injuries_by_team = _visible_injuries_by_team(visible)
        payload = _team_payload(
            depth,
            "KC",
            None,
            None,
            target_season=2026,
            current_injuries=current_injuries_by_team.get("KC"),
            learned_lookup=None,
            no_designation_lookup=no_designation_lookup,
            recent_roles={},
        )
        return payload["players"][0]

    base_row = {
        "season": 2026,
        "game_type": "REG",
        "team": "KC",
        "week": 1,
        "gsis_id": "wr-1",
        "position": "WR",
        "report_status": "Out",
        "practice_status": "Did Not Participate In Practice",
    }

    no_report = probability_for([])
    late_report = probability_for([{**base_row, "date_modified": "2026-09-11T12:00:00Z"}])
    early_report = probability_for([{**base_row, "date_modified": "2026-09-08T12:00:00Z"}])

    # A report dated AFTER generated_at must be invisible -- identical to
    # there being no report at all.
    assert late_report["play_probability"] == no_report["play_probability"]
    assert (
        late_report["probability_source"] == no_report["probability_source"] == "availability_model"
    )
    assert "no injury designation" in late_report["probability_reason"]
    assert no_report["play_probability"] == pytest.approx(0.90)

    # The SAME report, dated before generated_at, IS visible and changes
    # the number -- proving the leakage filter (not some other bug) is
    # what suppressed the late one.
    assert early_report["play_probability"] == pytest.approx(0.0)
    assert early_report["probability_source"] == "availability_model"
    assert "listed" in early_report["probability_reason"]


# ---------------------------------------------------------------------------
# Render: the legend line, and the em dash reserved for genuinely
# unscoreable rows.
# ---------------------------------------------------------------------------


def _lineup_for_render() -> object:
    return team_lineup(
        {
            "team": "KC",
            "as_of": "2026-09-10T00:00:00Z",
            "source": "nflverse depth charts",
            "injury_status": "nflverse injury report attached (0 player(s) listed); "
            "per-player probabilities from the availability model",
            "note": None,
            "players": [
                {
                    "name": "QB Model",
                    "position": "QB",
                    "slot": "QB1",
                    "depth": 1,
                    "unit": "offense",
                    "gsis_id": "qb-model",
                    "play_probability": 0.9,
                    "model_role": "base_model",
                    "probability_source": "base_model_qb",
                    "probability_reason": "forecast input",
                },
                {
                    "name": "WR One",
                    "position": "WR",
                    "slot": "WR1",
                    "depth": 1,
                    "unit": "offense",
                    "gsis_id": "wr-1",
                    "play_probability": 0.62,
                    "model_role": "context_only",
                    "probability_source": "availability_model",
                    "probability_reason": "no injury designation this week; using the "
                    "position's historical no-designation base rate (recent role: "
                    "no_recent_role)",
                },
                {
                    "name": "No Gsis Guy",
                    "position": "TE",
                    "slot": "TE1",
                    "depth": 1,
                    "unit": "offense",
                    "gsis_id": None,
                    "play_probability": None,
                    "model_role": "context_only",
                    "probability_source": "unavailable",
                    "probability_reason": "no gsis_id on this depth-chart row",
                },
            ],
        }
    )


def _dive_for_render() -> GameDive:
    lineup = _lineup_for_render()
    return GameDive(
        game_id="2026_01_KC_DEN",
        matchup_label="KC -3.0 at KC",
        pick_team="KC",
        pick_spread_text="-3.0",
        home="KC",
        kickoff_group_label="Sun 1:00 ET",
        probability_text="55%",
        is_best=False,
        attribution=AttributionPanel(available=False),
        cover_curve=(),
        cover_curve_offset_zero_note=None,
        adjuster=None,
        home_lineup=lineup,
        away_lineup=lineup,
    )


def test_lineups_html_prints_a_probability_legend_and_reserves_the_dash() -> None:
    html = board_terminal._lineups_html(_dive_for_render())
    assert "chance the player is active" in html
    assert "90%" in html
    assert "62%" in html
    # The em dash appears exactly once per team block -- only for the
    # player the model genuinely could not score.
    assert html.count("—") == 2  # once for the away team's block, once for home


def test_lineup_probability_cell_carries_the_reason_as_a_tooltip() -> None:
    html = board_terminal._lineup_team_html(_lineup_for_render())
    assert 'title="no gsis_id on this depth-chart row"' in html
    assert "no injury designation this week" in html


# ---------------------------------------------------------------------------
# The lineup-aware assistant now answers availability for non-QB players.
# ---------------------------------------------------------------------------


def test_player_availability_answer_covers_a_non_qb_player() -> None:
    lineup = _lineup_for_render()
    knowledge = build_lineup_knowledge(
        {"2026_01_KC_DEN": (lineup, lineup)},
        reference=datetime(2026, 9, 10, 1, 0, tzinfo=UTC),
    )
    tokens = frozenset(_tokens("is wr one playing this week"))
    answer = player_availability_answer(tokens, knowledge)
    assert answer is not None
    assert "62%" in answer.text
    assert "availability model" in answer.text
    assert "WR One" in answer.text
