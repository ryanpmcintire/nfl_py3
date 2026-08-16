"""Smoke tests for the redesigned dashboard: every page, with data and empty.

Fixtures write small synthetic artifact/data trees (no junction dependence)
and drive each page script directly through ``streamlit.testing.v1.AppTest``,
matching how Streamlit executes a multipage app's individual page files.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nfl_ats.dashboard.board import pick_side_cells, render_picks_board, tier_counts
from nfl_ats.dashboard.data import (
    ArtifactSelection,
    describe_artifact_source,
    explanations_by_game,
    select_research_artifact,
)
from nfl_ats.dashboard.ui import (
    FairLine,
    canonical_fair_line,
    confidence_tier,
    fair_line_gap,
    favorite_label,
    format_line_journey,
    format_value_framing,
    implied_fair_line,
    line_movement_signal,
    line_sweep_ribbon,
    pick_side_and_line,
)
from nfl_ats.snapshots import write_snapshot

FUTURE_KICKOFF = "2030-09-08T17:00:00Z"
FUTURE_GAMEDAY = "2030-09-08"
PAST_KICKOFF = "2023-09-10T17:00:00Z"
PAST_GAMEDAY = "2023-09-10"


# ---------------------------------------------------------------------------
# Pure-function unit tests (no Streamlit)
# ---------------------------------------------------------------------------


def test_confidence_tier_bands() -> None:
    assert confidence_tier(0.50)[0] == "Coin flip"
    assert confidence_tier(0.515)[0] == "Coin flip"
    assert confidence_tier(0.52)[0] == "Lean"
    assert confidence_tier(0.569)[0] == "Lean"
    assert confidence_tier(0.57)[0] == "Strong lean"
    assert confidence_tier(0.90)[0] == "Strong lean"


def test_pick_side_and_line_picks_the_more_likely_side() -> None:
    home_favored = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": -3.5, "home_cover_probability": 0.62}
    )
    team, line, probability = pick_side_and_line(home_favored)
    assert team == "SEA"
    assert line == 3.5
    assert probability == pytest.approx(0.62)

    away_favored = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": 3.5, "home_cover_probability": 0.4}
    )
    team, line, probability = pick_side_and_line(away_favored)
    assert team == "NE"
    assert line == 3.5
    assert probability == pytest.approx(0.6)


def test_favorite_label_reads_the_market_spread() -> None:
    # Positive spread_line means the home team is favored: home_cover requires
    # (home_score - away_score) > spread_line, so a positive line makes the
    # home side "give" points.
    row = pd.Series({"home_team": "SEA", "away_team": "NE", "spread_line": 3.5})
    assert favorite_label(row) == "SEA -3.5"
    row = pd.Series({"home_team": "SEA", "away_team": "NE", "spread_line": -3.5})
    assert favorite_label(row) == "NE -3.5"
    row = pd.Series({"home_team": "SEA", "away_team": "NE", "spread_line": 0.0})
    assert favorite_label(row) == "Pick'em"


# ---------------------------------------------------------------------------
# Confidence ribbon: sign-convention and fair-line unit tests
# ---------------------------------------------------------------------------


def test_line_sweep_ribbon_orients_to_a_home_pick() -> None:
    # home_cover_probability=0.58 >= 0.5 -> the pick is home (SEA); by
    # pick_side_and_line's convention that reads as "SEA +3" (spread_line
    # -3.0 negated).
    game = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": -3.0, "home_cover_probability": 0.58}
    )
    game_sweep = pd.DataFrame(
        {
            "line_offset": [-1.0, 0.0, 1.0],
            "alternative_line": [-4.0, -3.0, -2.0],
            "home_cover_probability": [0.66, 0.58, 0.50],
            "push_probability": [0.0, 0.0, 0.0],
        }
    )
    ribbon = line_sweep_ribbon(game, game_sweep, perspective="pick")
    assert ribbon["line"].tolist() == [2.0, 3.0, 4.0]
    assert ribbon["probability"].tolist() == pytest.approx([0.50, 0.58, 0.66])
    assert ribbon["label"].tolist() == ["+2", "+3", "+4"]
    assert ribbon.loc[ribbon["line"].eq(3.0), "is_market"].item()
    assert not ribbon.loc[ribbon["line"].eq(2.0), "is_market"].item()


def test_line_sweep_ribbon_orients_to_an_away_pick() -> None:
    # home_cover_probability=0.40 < 0.5 -> the pick is away (NE); by
    # pick_side_and_line's convention that reads as "NE +3" (spread_line
    # kept as-is).
    game = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": 3.0, "home_cover_probability": 0.40}
    )
    game_sweep = pd.DataFrame(
        {
            "line_offset": [-1.0, 0.0, 1.0],
            "alternative_line": [2.0, 3.0, 4.0],
            "home_cover_probability": [0.46, 0.40, 0.34],
            "push_probability": [0.0, 0.0, 0.0],
        }
    )
    ribbon = line_sweep_ribbon(game, game_sweep, perspective="pick")
    assert ribbon["line"].tolist() == [2.0, 3.0, 4.0]
    # Away pick reads 1 - home_cover_probability.
    assert ribbon["probability"].tolist() == pytest.approx([0.54, 0.60, 0.66])
    assert ribbon["label"].tolist() == ["+2", "+3", "+4"]
    assert ribbon.loc[ribbon["line"].eq(3.0), "is_market"].item()


def test_line_sweep_ribbon_flags_integer_lines_with_push_mass() -> None:
    game = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": -3.0, "home_cover_probability": 0.55}
    )
    game_sweep = pd.DataFrame(
        {
            "line_offset": [-0.5, 0.0, 0.5],
            "alternative_line": [-3.5, -3.0, -2.5],
            "home_cover_probability": [0.60, 0.55, 0.50],
            "push_probability": [0.0, 0.03, 0.0],
        }
    )
    ribbon = line_sweep_ribbon(game, game_sweep, perspective="pick")
    integer_row = ribbon.loc[ribbon["line"].eq(3.0)].iloc[0]
    assert bool(integer_row["is_integer"])
    assert integer_row["push_probability"] == pytest.approx(0.03)
    half_point_row = ribbon.loc[ribbon["line"].eq(2.5)].iloc[0]
    assert not bool(half_point_row["is_integer"])
    assert half_point_row["push_probability"] == pytest.approx(0.0)


def test_line_sweep_ribbon_deduplicates_stacked_methods() -> None:
    # score_outcome_week_line_sweep() (nfl-ats margin-predict --line-sweep)
    # stacks every margin-distribution method into one file with a "method"
    # column and no other per-game uniqueness; a caller that forgets to
    # filter to one method hands line_sweep_ribbon 2-3x duplicate rows per
    # line_offset. It must not crash (pandas Styler outright refuses
    # non-unique columns) and should keep exactly one row per line.
    game = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": -3.0, "home_cover_probability": 0.58}
    )
    game_sweep = pd.DataFrame(
        {
            "line_offset": [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
            "alternative_line": [-4.0, -4.0, -3.0, -3.0, -2.0, -2.0],
            "home_cover_probability": [0.66, 0.40, 0.58, 0.20, 0.50, 0.10],
            "push_probability": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    ribbon = line_sweep_ribbon(game, game_sweep, perspective="pick")
    assert ribbon["line"].tolist() == [2.0, 3.0, 4.0]
    assert not ribbon["line"].duplicated().any()


def test_line_sweep_ribbon_home_perspective_is_unconverted() -> None:
    game = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": -3.0, "home_cover_probability": 0.58}
    )
    game_sweep = pd.DataFrame(
        {
            "line_offset": [-1.0, 0.0, 1.0],
            "alternative_line": [-4.0, -3.0, -2.0],
            "home_cover_probability": [0.66, 0.58, 0.50],
            "push_probability": [0.0, 0.0, 0.0],
        }
    )
    ribbon = line_sweep_ribbon(game, game_sweep, perspective="home")
    assert ribbon["line"].tolist() == [-4.0, -3.0, -2.0]
    assert ribbon["probability"].tolist() == pytest.approx([0.66, 0.58, 0.50])


def test_implied_fair_line_interpolates_the_50_percent_crossing() -> None:
    exact = implied_fair_line(
        pd.DataFrame({"line": [1.0, 2.0, 3.0], "probability": [0.60, 0.50, 0.40]})
    )
    assert exact.value == pytest.approx(2.0)
    assert exact.label == "+2"

    between = implied_fair_line(pd.DataFrame({"line": [1.0, 2.0], "probability": [0.55, 0.45]}))
    assert between.value == pytest.approx(1.5)
    assert between.label == "+1½"


def test_implied_fair_line_clamps_outside_the_swept_range() -> None:
    always_favored = implied_fair_line(
        pd.DataFrame({"line": [-4.0, 0.0, 4.0], "probability": [0.70, 0.65, 0.60]})
    )
    assert always_favored.value is None
    assert always_favored.label == "< -4"

    always_underdog = implied_fair_line(
        pd.DataFrame({"line": [-4.0, 0.0, 4.0], "probability": [0.30, 0.35, 0.40]})
    )
    assert always_underdog.value is None
    assert always_underdog.label == "> +4"


def test_implied_fair_line_empty_ribbon() -> None:
    empty = implied_fair_line(pd.DataFrame(columns=["line", "probability"]))
    assert empty.value is None
    assert empty.label == "—"


# ---------------------------------------------------------------------------
# Canonical fair line and bettor-facing value framing (defects #3 and #4)
# ---------------------------------------------------------------------------


def test_canonical_fair_line_prefers_the_residual_over_the_sweep() -> None:
    # spread_line 3.5 + residual -0.3 = 3.2, home-oriented -- deliberately
    # different from what a sweep crossing might interpolate, to prove the
    # residual wins even when a (differently-shaped) ribbon is also passed.
    game = pd.Series({"spread_line": 3.5, "predicted_market_residual": -0.3})
    misleading_ribbon = pd.DataFrame({"line": [1.0, 2.0], "probability": [0.6, 0.4]})
    fair = canonical_fair_line(game, misleading_ribbon)
    assert fair.value == pytest.approx(3.2)
    assert fair.label == "+3.2"


def test_canonical_fair_line_falls_back_to_the_sweep_without_a_residual() -> None:
    game = pd.Series({"spread_line": 3.5, "predicted_market_residual": float("nan")})
    ribbon = pd.DataFrame({"line": [1.0, 2.0, 3.0], "probability": [0.60, 0.50, 0.40]})
    fair = canonical_fair_line(game, ribbon)
    assert fair.value == pytest.approx(2.0)
    assert fair.label == "+2"


def test_canonical_fair_line_falls_back_when_residual_column_is_missing() -> None:
    # A legacy card that predates predicted_market_residual: Series.get
    # returns None for a missing key, same as no residual at all.
    game = pd.Series({"spread_line": 3.5})
    fair = canonical_fair_line(game, pd.DataFrame(columns=["line", "probability"]))
    assert fair.value is None
    assert fair.label == "—"


def test_fair_line_gap_home_pick_offered_better_than_fair() -> None:
    # Home pick displayed as "-3.5"; the model's tougher home-oriented fair
    # line (+3.6, i.e. home should be favored by more) makes the offered
    # -3.5 an easier ask than fair -- good for the bettor.
    gap = fair_line_gap(-3.5, FairLine(3.6, "+3.6"), home_pick=True)
    assert gap == pytest.approx(0.1)


def test_fair_line_gap_home_pick_offered_worse_than_fair() -> None:
    # Fair line only +3.2 (home should be favored by less than market says)
    # -- the offered -3.5 asks the home pick to cover more than fair.
    gap = fair_line_gap(-3.5, FairLine(3.2, "+3.2"), home_pick=True)
    assert gap == pytest.approx(-0.3)


def test_fair_line_gap_away_pick_offered_better_than_fair() -> None:
    # Away pick displayed as "+3.5"; fair only +3.2 -- the offered line
    # hands the dog more points than fair says it deserves.
    gap = fair_line_gap(3.5, FairLine(3.2, "+3.2"), home_pick=False)
    assert gap == pytest.approx(0.3)


def test_fair_line_gap_away_pick_offered_worse_than_fair() -> None:
    # Fair says the dog deserves +4.0, but the market only offers +3.5.
    gap = fair_line_gap(3.5, FairLine(4.0, "+4"), home_pick=False)
    assert gap == pytest.approx(-0.5)


def test_fair_line_gap_none_when_fair_has_no_concrete_value() -> None:
    assert fair_line_gap(3.5, FairLine(None, "> +4"), home_pick=False) is None


def test_format_value_framing_wording_and_color_cues() -> None:
    assert format_value_framing(0.1) == ":green[getting 0.1 pt better than fair]"
    assert format_value_framing(0.3) == ":green[getting 0.3 pt better than fair]"
    assert format_value_framing(-0.3) == ":orange[0.3 pt worse than fair]"
    assert format_value_framing(-0.5) == ":orange[0.5 pt worse than fair]"
    assert format_value_framing(0.0) == "right at fair"
    assert format_value_framing(None) == "fair line beyond the swept window"


def test_line_movement_signal_detects_direction() -> None:
    assert line_movement_signal(None, -3.0, 1.0) == "unknown"
    assert line_movement_signal(-5.0, -3.0, -3.0) == "toward"
    assert line_movement_signal(-3.0, -6.0, -3.0) == "away"
    assert line_movement_signal(-3.0, -3.0, 0.0) == "flat"


def test_format_line_journey_formats_and_colors_movement() -> None:
    fair = FairLine(-4.5, "-4½")
    text = format_line_journey(-2.5, -3.0, None, fair)
    assert text == "Open -2½ → Now :green[-3] → Close (pred) — → Fair -4½"

    away_text = format_line_journey(-3.0, -6.0, -3.0, FairLine(-3.0, "-3"))
    assert ":orange[-6]" in away_text

    missing = format_line_journey(None, None, None, FairLine(None, "—"))
    assert missing == "Open — → Now — → Close (pred) — → Fair —"


def test_format_line_journey_appends_framing_as_one_consolidated_line() -> None:
    # The card shows exactly one fair/journey caption -- the picked side's
    # value framing (see format_value_framing) is appended to the same
    # line, not rendered as a separate caption.
    fair = FairLine(3.2, "+3.2")
    text = format_line_journey(
        None, None, None, fair, framing=":green[getting 0.3 pt better than fair]"
    )
    assert text == (
        "Open — → Now — → Close (pred) — → Fair +3.2 · :green[getting 0.3 pt better than fair]"
    )

    without_framing = format_line_journey(None, None, None, fair, framing=None)
    assert without_framing == "Open — → Now — → Close (pred) — → Fair +3.2"
    assert "·" not in without_framing


# ---------------------------------------------------------------------------
# board.py: pure-function tests (no Streamlit)
# ---------------------------------------------------------------------------


def test_tier_counts_buckets_by_picked_side_probability() -> None:
    predictions = pd.DataFrame(
        {"home_cover_probability": [0.60, 0.53, 0.48, 0.90, 0.40]}
    )  # picked-side: .60 .53 .52 .90 .60 -> strong/lean/lean/strong/strong
    assert tier_counts(predictions) == (3, 2, 0)
    assert tier_counts(pd.DataFrame(columns=["home_cover_probability"])) == (0, 0, 0)


def test_pick_side_cells_home_pick_orders_descending_by_pick_side_line() -> None:
    # Same fixture as ui.test_line_sweep_ribbon_orients_to_a_home_pick: home
    # (SEA) is picked. line_sweep_ribbon itself returns ascending [2, 3, 4];
    # the approved mock's own (unreversed, for a home pick) loop instead
    # builds the row in descending pick-side-line order -- this is the "subtle
    # reversal" the mock generator relies on, verified explicitly here rather
    # than trusted by inspection.
    game = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": -3.0, "home_cover_probability": 0.58}
    )
    game_sweep = pd.DataFrame(
        {
            "line_offset": [-1.0, 0.0, 1.0],
            "alternative_line": [-4.0, -3.0, -2.0],
            "home_cover_probability": [0.66, 0.58, 0.50],
            "push_probability": [0.0, 0.0, 0.0],
        }
    )
    cells = pick_side_cells(game, game_sweep)
    assert cells["line"].tolist() == [4.0, 3.0, 2.0]
    assert cells["probability"].tolist() == pytest.approx([0.66, 0.58, 0.50])
    assert cells.loc[cells["line"].eq(3.0), "is_market"].item()


def test_pick_side_cells_away_pick_orders_descending_by_pick_side_line() -> None:
    # Same fixture as ui.test_line_sweep_ribbon_orients_to_an_away_pick: away
    # (NE) is picked. Despite the opposite home/away orientation, the board's
    # display order is the same convention -- descending by pick-side line --
    # which is exactly why the mock generator's loop reverses the away case
    # but not the home case: both must land in the same final order.
    game = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": 3.0, "home_cover_probability": 0.40}
    )
    game_sweep = pd.DataFrame(
        {
            "line_offset": [-1.0, 0.0, 1.0],
            "alternative_line": [2.0, 3.0, 4.0],
            "home_cover_probability": [0.46, 0.40, 0.34],
            "push_probability": [0.0, 0.0, 0.0],
        }
    )
    cells = pick_side_cells(game, game_sweep)
    assert cells["line"].tolist() == [4.0, 3.0, 2.0]
    assert cells["probability"].tolist() == pytest.approx([0.66, 0.60, 0.54])
    assert cells.loc[cells["line"].eq(3.0), "is_market"].item()


def test_pick_side_cells_empty_sweep_returns_empty_frame() -> None:
    game = pd.Series(
        {"home_team": "SEA", "away_team": "NE", "spread_line": -3.0, "home_cover_probability": 0.58}
    )
    cells = pick_side_cells(game, pd.DataFrame())
    assert cells.empty


def _board_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2030_01_NE_SEA", "2030_01_SF_LA"],
            "home_team": ["SEA", "LA"],
            "away_team": ["NE", "SF"],
            "spread_line": [3.5, -3.5],
            "home_cover_probability": [0.485, 0.90],
            "predicted_market_residual": [-0.3, 5.0],
            "weekday": ["Sunday", "Sunday"],
            "gameday": ["2030-09-08", "2030-09-08"],
            "gametime": ["13:00:00", "16:25:00"],
            "method": ["market_residual", "market_residual"],
        }
    )


def test_render_picks_board_sorts_by_confidence_descending() -> None:
    html = render_picks_board(_board_predictions(), pd.DataFrame())
    # LA (0.90 confidence) must render before NE (0.515 confidence).
    assert html.index("LA") < html.index("NE @ SEA")


def test_render_picks_board_no_sweep_omits_numbered_ribbon() -> None:
    html = render_picks_board(_board_predictions(), pd.DataFrame())
    assert 'class="pb-ribbon-full"' not in html
    assert html.count('<span class="pb-no-sweep">') == 2


def test_render_picks_board_theme_attribute_scoped_to_wrapper_not_root() -> None:
    light = render_picks_board(_board_predictions(), pd.DataFrame(), theme="light")
    dark = render_picks_board(_board_predictions(), pd.DataFrame(), theme="dark")
    neither = render_picks_board(_board_predictions(), pd.DataFrame(), theme=None)
    assert '<div class="pb" data-theme="light">' in light
    assert '<div class="pb" data-theme="dark">' in dark
    assert '<div class="pb">' in neither
    # Never on the real document root -- this fragment shares a DOM with the
    # rest of the app (st.html is not iframed in this Streamlit version).
    assert ":root[data-theme=" not in light
    assert ":root[data-theme=" not in dark


def test_render_picks_board_empty_predictions_does_not_crash() -> None:
    html = render_picks_board(pd.DataFrame(), pd.DataFrame())
    assert "No games this week" in html


def test_render_picks_board_explanations_and_fallback() -> None:
    html = render_picks_board(
        _board_predictions(), pd.DataFrame(), {"2030_01_SF_LA": "Because of a big residual."}
    )
    assert "Because of a big residual." in html
    assert "Model and market are close on this one." in html


def test_render_picks_board_toggle_uses_addeventlistener_not_inline_onclick() -> None:
    html = render_picks_board(_board_predictions(), pd.DataFrame())
    assert 'id="toggle-all"' in html
    assert "addEventListener" in html
    assert "onclick" not in html.lower()


def test_render_picks_board_metadata_footer_is_optional() -> None:
    with_metadata = render_picks_board(
        _board_predictions(), pd.DataFrame(), metadata={"active_model_id": "abc123"}
    )
    without_metadata = render_picks_board(_board_predictions(), pd.DataFrame())
    assert "abc123" in with_metadata
    assert '<p class="pb-foot">' not in without_metadata


# ---------------------------------------------------------------------------
# data.py: research-artifact staleness-fix helpers (pure, no Streamlit)
# ---------------------------------------------------------------------------


def test_select_research_artifact_features_the_active_path(tmp_path: Path) -> None:
    older = tmp_path / "run-b-newer-name"
    active = tmp_path / "run-a-active"
    older.mkdir()
    active.mkdir()
    selection = select_research_artifact([older, active], active)
    assert selection.featured == active
    assert selection.featured_is_active
    assert not selection.active_declared_but_missing
    assert selection.older == (older,)


def test_select_research_artifact_reports_missing_active_path_explicitly(tmp_path: Path) -> None:
    present = tmp_path / "present"
    present.mkdir()
    missing = tmp_path / "missing-does-not-exist"
    selection = select_research_artifact([present], missing)
    assert selection.featured is None
    assert not selection.featured_is_active
    assert selection.active_declared_but_missing
    assert selection.older == (present,)


def test_select_research_artifact_without_active_path_features_the_newest() -> None:
    # artifact_directories() sorts descending by name; select_research_artifact
    # trusts that ordering when there is no active-model concept at all.
    directories = [Path("b-newest"), Path("a-older")]
    selection = select_research_artifact(directories, None)
    assert selection.featured == Path("b-newest")
    assert not selection.featured_is_active
    assert not selection.active_declared_but_missing
    assert selection.older == (Path("a-older"),)


def test_select_research_artifact_empty_directories() -> None:
    assert select_research_artifact([], None) == ArtifactSelection(None, False, False, ())
    # Zero saved runs on disk, but an active model declares one -- that is the
    # "missing" case (never silently treated as "no active-model concept").
    assert select_research_artifact([], Path("x")) == ArtifactSelection(None, False, True, ())


def test_describe_artifact_source_relativizes_and_stamps_date(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    directory = root / "margins" / "20260816T184528Z"
    directory.mkdir(parents=True)
    stamp = describe_artifact_source(directory, root)
    assert stamp.startswith("Reading `margins/20260816T184528Z` · created ")
    assert "2026-08-16 18:45 UTC" in stamp


def test_explanations_by_game_dedupes_and_falls_back_gracefully() -> None:
    attribution = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "family": ["offense", "intercept", "offense"],
            "explanation": ["Because of offense.", "Because of offense.", None],
        }
    )
    explanations = explanations_by_game(attribution)
    assert explanations == {"g1": "Because of offense."}
    assert explanations_by_game(pd.DataFrame()) == {}
    assert explanations_by_game(pd.DataFrame({"game_id": ["g1"]})) == {}


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _weekly_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2030_01_NE_SEA", "2030_01_SF_LA", "2030_01_ATL_PIT", "2023_01_BUF_MIA"],
            "season": [2030, 2030, 2030, 2023],
            "week": [1, 1, 1, 1],
            "gameday": [FUTURE_GAMEDAY, FUTURE_GAMEDAY, FUTURE_GAMEDAY, PAST_GAMEDAY],
            "weekday": ["Sunday", "Sunday", "Sunday", "Sunday"],
            "gametime": ["13:00:00", "16:25:00", "20:20:00", "13:00:00"],
            "kickoff": [FUTURE_KICKOFF, FUTURE_KICKOFF, FUTURE_KICKOFF, PAST_KICKOFF],
            "away_team": ["NE", "SF", "ATL", "MIA"],
            "home_team": ["SEA", "LA", "PIT", "BUF"],
            "spread_line": [3.5, -3.5, -3.0, -2.5],
            "total_line": [44.5, 47.0, 42.5, 45.0],
            "home_cover_probability": [0.485, 0.616, 0.512, 0.70],
            "home_cover": [float("nan"), float("nan"), float("nan"), 1.0],
            "ats_margin": [float("nan"), float("nan"), float("nan"), 5.0],
            "predicted_market_residual": [-0.3, 4.1, 0.4, 3.0],
            "edge": [-0.015, 0.116, 0.012, 0.2],
            "bet_side": ["PASS", "AWAY", "PASS", "HOME"],
            "bet_odds": [float("nan"), -110.0, float("nan"), -110.0],
            "method": ["market_residual", "market_residual", "market_residual", "market_residual"],
        }
    )


def _write_weekly_forecast(artifacts: Path, run_id: str) -> Path:
    directory = artifacts / "margin_predictions" / run_id
    recommendations = _weekly_forecast_frame()
    _write_csv(recommendations, directory / "recommendations.csv")
    # A real margin-predict run always writes predictions.csv alongside
    # recommendations.csv; artifact discovery for this directory looks for
    # predictions.csv, so it must exist even though recommendations.csv (the
    # file actually read) takes priority once the directory is found.
    _write_csv(recommendations.assign(method="market_residual"), directory / "predictions.csv")
    _write_json(
        {
            "season": 2030,
            "week": 1,
            "feature_profile": "player",
            "ats_method": "market_residual",
            "created_at_utc": "2030-08-12T21:15:33Z",
            "prediction_safety": {"status": "PASS", "warnings": []},
            "historical_evaluation": {
                "accuracy": 0.5205,
                "games": 2075,
                "correct": 1080,
                "intervals": {
                    "week": {"lower": 0.4985, "upper": 0.5425},
                    "season": {"lower": 0.5019, "upper": 0.5414},
                },
            },
        },
        directory / "metadata.json",
    )
    return directory


def _write_active_model(
    artifacts: Path, *, evaluation_relative: str, forecast_relative: str
) -> None:
    _write_json(
        {
            "version": 1,
            "status": "SYNCHRONIZED",
            "target": "ats_classification",
            "model_id": "testmodel01",
            "method": "market_residual",
            "feature_profile": "player",
            "historical_evaluation": {
                "artifact": evaluation_relative,
                "accuracy": 0.5205,
                "correct": 1080,
                "games": 2075,
                "intervals": {
                    "week": {"lower": 0.4985, "upper": 0.5425},
                    "season": {"lower": 0.5019, "upper": 0.5414},
                },
            },
            "weekly_forecast": {
                "artifact": forecast_relative,
                "season": 2030,
                "week": 1,
                "created_at_utc": "2030-08-12T21:15:33Z",
            },
        },
        artifacts / "active_ats_model.json",
    )


def _evaluation_predictions_frame() -> pd.DataFrame:
    rows = []
    game = 0
    for season in (2022, 2023):
        for week in range(1, 6):
            game += 1
            probability = 0.5 + 0.01 * (game % 6)
            covered = float(game % 5 != 0)
            rows.append(
                {
                    "game_id": f"g{game}",
                    "season": season,
                    "week": week,
                    "method": "market_residual",
                    "home_cover": covered,
                    "home_cover_probability": probability,
                }
            )
    return pd.DataFrame(rows)


def _write_margins_evaluation(artifacts: Path, run_id: str) -> Path:
    directory = artifacts / "margins" / run_id
    predictions = _evaluation_predictions_frame()
    _write_parquet(predictions, directory / "predictions.parquet")
    summary = pd.DataFrame(
        [
            {
                "method": method,
                "win_accuracy": 0.55,
                "win_brier_score": 0.24,
                "margin_mae": 9.5,
                "cover_accuracy": 0.5205 if method != "market" else None,
                "cover_brier_score": 0.2495,
                "cover_games": 2075,
                "roi": 0.01,
            }
            for method in ("market", "fair_margin", "market_residual", "straight_up", "direct_ats")
        ]
    )
    _write_csv(summary, directory / "summary.csv")
    _write_json(
        {"regressor": "ridge", "feature_profile": "player", "start_season": 2018},
        directory / "metadata.json",
    )
    uncertainty_rows = []
    for block in ("week", "season"):
        for method in ("fair_margin", "market_residual", "straight_up"):
            for metric in ("cover_accuracy", "win_accuracy", "margin_mae", "cover_brier_score"):
                uncertainty_rows.append(
                    {
                        "block": block,
                        "method": method,
                        "metric": metric,
                        "estimate": 0.01,
                        "lower": -0.01,
                        "upper": 0.03,
                    }
                )
    _write_csv(pd.DataFrame(uncertainty_rows), directory / "uncertainty.csv")
    return directory


def _write_backtest_run(artifacts: Path, run_id: str) -> Path:
    directory = artifacts / "backtests" / run_id
    predictions = _evaluation_predictions_frame().drop(columns="method")
    _write_parquet(predictions, directory / "predictions.parquet")
    _write_json(
        {
            "accuracy": 0.512,
            "brier_score": 0.249,
            "roi": 0.01,
            "games_evaluated": len(predictions),
            "model_name": "logistic",
            "feature_set": "full",
        },
        directory / "metrics.json",
    )
    _write_json(
        {"configuration": {"model": "logistic", "feature_set": "full"}}, directory / "run.json"
    )
    uncertainty_rows = [
        {
            "metric": "accuracy",
            "estimate": 0.512,
            "lower": 0.49,
            "upper": 0.53,
            "block": "week",
            "samples": 500,
        },
        {
            "metric": "brier_score",
            "estimate": 0.249,
            "lower": 0.24,
            "upper": 0.26,
            "block": "week",
            "samples": 500,
        },
    ]
    _write_csv(pd.DataFrame(uncertainty_rows), directory / "uncertainty.csv")

    ledger = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2022, 2022, 2023],
            "week": [1, 2, 1],
            "gameday": ["2022-09-11", "2022-09-18", "2023-09-10"],
            "away_team": ["B", "B", "MIA"],
            "home_team": ["A", "A", "BUF"],
            "bet_side": ["HOME", "AWAY", "HOME"],
            "bet_probability": [0.55, 0.53, 0.58],
            "stake": [20.0, 18.0, 22.0],
            "profit": [18.2, -18.0, 20.0],
            "bankroll_before_week": [1000.0, 1018.2, 1000.2],
            "bankroll_after_week": [1018.2, 1000.2, 1020.2],
        }
    )
    _write_parquet(ledger, directory / "paper_ledger.parquet")
    _write_json(
        {
            "initial_bankroll": 1000.0,
            "final_bankroll": 1020.2,
            "return": 0.0202,
            "max_drawdown": -0.02,
        },
        directory / "portfolio_metrics.json",
    )
    _write_json(
        {
            "terminal_bankroll_median": 1050.0,
            "terminal_bankroll_p05": 900.0,
            "probability_of_loss": 0.3,
            "median_max_drawdown": -0.05,
        },
        directory / "bankroll_simulation.json",
    )
    paths = pd.DataFrame(
        {"start": [1000.0, 1000.0], "2022-W01": [1020.0, 980.0], "2022-W02": [1040.0, 950.0]}
    )
    _write_parquet(paths, directory / "bankroll_paths.parquet")
    return directory


def _write_nested_evaluation(artifacts: Path, run_id: str) -> None:
    directory = artifacts / "nested_evaluations" / run_id
    _write_json(
        {
            "outer_folds": 2,
            "accuracy": 0.513,
            "brier_score": 0.2498,
            "roi": 0.01,
            "candidate_count": 6,
            "selection_metric": "brier_score",
        },
        directory / "metrics.json",
    )
    _write_csv(
        pd.DataFrame(
            [
                {
                    "outer_test_season": 2022,
                    "selected_model_name": "logistic",
                    "selected_feature_set": "full",
                    "validation_selection_score": 0.249,
                    "test_accuracy": 0.51,
                    "test_brier_score": 0.2497,
                }
            ]
        ),
        directory / "fold_summary.csv",
    )
    _write_csv(
        pd.DataFrame(
            [
                {
                    "outer_test_season": 2022,
                    "candidate_id": "full|logistic",
                    "selection_rank": 1,
                    "selected": True,
                    "brier_score": 0.249,
                    "accuracy": 0.51,
                }
            ]
        ),
        directory / "candidate_validation.csv",
    )


def _write_player_model_selection(artifacts: Path, run_id: str) -> None:
    directory = artifacts / "player_model_selection" / run_id
    _write_csv(
        pd.DataFrame(
            [
                {
                    "feature_profile": "player_qb_continuity",
                    "ridge_alpha": 1.0,
                    "calibration_method": "none",
                    "cover_games": 2075,
                    "cover_accuracy": 0.5263,
                    "cover_brier_score": 0.2531,
                    "cover_ece": 0.0405,
                },
                {
                    "feature_profile": "base",
                    "ridge_alpha": 10.0,
                    "calibration_method": "none",
                    "cover_games": 2075,
                    "cover_accuracy": 0.5108,
                    "cover_brier_score": 0.2521,
                    "cover_ece": 0.0456,
                },
            ]
        ),
        directory / "candidate_summary.csv",
    )
    _write_csv(pd.DataFrame([{"cover_accuracy": 0.5070}]), directory / "nested_summary.csv")
    _write_csv(
        pd.DataFrame(
            [
                {
                    "metric": "accuracy_improvement",
                    "block": "week",
                    "estimate": -0.0019,
                    "lower": -0.0248,
                    "upper": 0.0221,
                }
            ]
        ),
        directory / "nested_paired_comparisons.csv",
    )


def _write_participation_experiment(artifacts: Path, run_id: str) -> None:
    directory = artifacts / "participation_experiments" / run_id
    _write_csv(
        pd.DataFrame(
            [
                {
                    "feature_profile": "player_value",
                    "cover_accuracy": 0.5214,
                    "cover_brier_score": 0.2530,
                },
                {
                    "feature_profile": "player_participation",
                    "cover_accuracy": 0.5171,
                    "cover_brier_score": 0.2538,
                },
            ]
        ),
        directory / "summary.csv",
    )
    _write_csv(
        pd.DataFrame(
            [
                {
                    "metric": "accuracy_improvement",
                    "block": "week",
                    "estimate": -0.0043,
                    "lower": -0.0153,
                    "upper": 0.0063,
                }
            ]
        ),
        directory / "paired_comparisons.csv",
    )


def _write_availability_experiment(artifacts: Path, run_id: str) -> None:
    directory = artifacts / "availability_experiments" / run_id
    _write_csv(
        pd.DataFrame(
            [
                {
                    "availability_method": "fixed",
                    "cover_accuracy": 0.5214,
                    "cover_brier_score": 0.2531,
                },
                {
                    "availability_method": "learned",
                    "cover_accuracy": 0.5224,
                    "cover_brier_score": 0.2530,
                },
            ]
        ),
        directory / "summary.csv",
    )
    _write_csv(
        pd.DataFrame(
            [
                {
                    "metric": "accuracy_improvement",
                    "block": "week",
                    "estimate": 0.0010,
                    "lower": -0.0063,
                    "upper": 0.0078,
                }
            ]
        ),
        directory / "paired_comparisons.csv",
    )
    _write_json(
        {
            "learned_provenance": {
                "feature_table": {"manifest": {"availability_brier_improvement": 0.00444}}
            }
        },
        directory / "metadata.json",
    )


def _write_feature_experiment(artifacts: Path, run_id: str) -> None:
    directory = artifacts / "experiments" / run_id
    _write_csv(
        pd.DataFrame(
            [
                {
                    "feature_set": "market",
                    "feature_count": 2,
                    "games_evaluated": 2075,
                    "accuracy": 0.505,
                    "brier_score": 0.2499,
                    "expected_calibration_error": 0.02,
                    "roi": 0.0,
                },
                {
                    "feature_set": "full",
                    "feature_count": 58,
                    "games_evaluated": 2075,
                    "accuracy": 0.5205,
                    "brier_score": 0.2495,
                    "expected_calibration_error": 0.018,
                    "roi": 0.01,
                },
            ]
        ),
        directory / "summary.csv",
    )
    _write_csv(
        pd.DataFrame(
            [
                {
                    "candidate_feature_set": "full",
                    "metric": "accuracy_improvement",
                    "block": "week",
                    "estimate": 0.015,
                    "lower": -0.01,
                    "upper": 0.04,
                    "paired_games": 2075,
                }
            ]
        ),
        directory / "paired_comparisons.csv",
    )


def _write_player_experiment(artifacts: Path, run_id: str) -> None:
    directory = artifacts / "player_experiments" / run_id
    _write_csv(
        pd.DataFrame(
            [
                {
                    "feature_profile": "base",
                    "cover_games": 2075,
                    "cover_accuracy": 0.5108,
                    "cover_brier_score": 0.2521,
                    "margin_mae": 9.6,
                },
                {
                    "feature_profile": "player_qb",
                    "cover_games": 2075,
                    "cover_accuracy": 0.5205,
                    "cover_brier_score": 0.2495,
                    "margin_mae": 9.4,
                },
            ]
        ),
        directory / "summary.csv",
    )


def _write_market_snapshots(data_root: Path) -> None:
    now = pd.Timestamp.now(tz="UTC")
    for offset_hours, spread in ((48, -3.5), (24, -3.0)):
        observed_at = now - pd.Timedelta(hours=offset_hours)
        snapshot_id = observed_at.strftime("%Y%m%dT%H%M%SZ")
        directory = data_root / "market" / "raw" / snapshot_id
        directory.mkdir(parents=True, exist_ok=True)
        quotes = pd.DataFrame(
            {
                "observed_at_utc": [observed_at] * 4,
                "provider": ["the-odds-api"] * 4,
                "provider_event_id": ["evt-1", "evt-1", "evt-2", "evt-2"],
                "sport_key": ["americanfootball_nfl"] * 4,
                "commence_time_utc": [pd.Timestamp(FUTURE_KICKOFF)] * 4,
                "home_team_name": ["Seattle Seahawks"] * 2 + ["Los Angeles Rams"] * 2,
                "away_team_name": ["New England Patriots"] * 2 + ["San Francisco 49ers"] * 2,
                "home_team": ["SEA", "SEA", "LA", "LA"],
                "away_team": ["NE", "NE", "SF", "SF"],
                "nflverse_game_id": ["2030_01_NE_SEA"] * 2 + ["2030_01_SF_LA"] * 2,
                "bookmaker_key": ["draftkings", "betus", "draftkings", "betus"],
                "bookmaker_title": ["DraftKings", "BetUS", "DraftKings", "BetUS"],
                "bookmaker_last_update_utc": [observed_at] * 4,
                "market": ["spreads"] * 4,
                "market_last_update_utc": [observed_at] * 4,
                "outcome_name": [
                    "Seattle Seahawks",
                    "Seattle Seahawks",
                    "Los Angeles Rams",
                    "Los Angeles Rams",
                ],
                "outcome_side": ["HOME"] * 4,
                "line": [spread, spread - 0.5, -3.5, -4.0],
                "price": [-110, -108, -110, -105],
                "home_spread_line": [spread, spread - 0.5, -3.5, -4.0],
                "raw_response_sha256": ["abc123"] * 4,
            }
        )
        _write_parquet(quotes, directory / "quotes.parquet")
        _write_json(
            {
                "schema_version": 1,
                "snapshot_id": snapshot_id,
                "observed_at_utc": observed_at.isoformat(),
                "provider": "the-odds-api",
                "request": {"markets": "spreads", "regions": "us"},
                "quota": {},
                "files": {},
            },
            directory / "manifest.json",
        )


def _sweep_offsets() -> list[float]:
    return [round(-4.0 + 0.5 * step, 1) for step in range(17)]


def _synthetic_sweep_rows(
    game_id: str,
    quoted_line: float,
    market_home_cover_probability: float,
    *,
    method: str = "market_residual",
) -> list[dict[str, Any]]:
    """A roughly linear line-sweep table, anchored at the game's real market line.

    ``market_home_cover_probability`` matches the game's ``home_cover_probability``
    in ``_weekly_forecast_frame`` at ``line_offset == 0``, so the ribbon's
    market-line cell and the card's existing confidence figure agree.
    """

    rows: list[dict[str, Any]] = []
    for offset in _sweep_offsets():
        alternative_line = round(quoted_line + offset, 4)
        probability = min(max(market_home_cover_probability - 0.05 * offset, 0.01), 0.99)
        is_integer_line = float(alternative_line).is_integer()
        push = 0.03 if is_integer_line else 0.0
        loss = max(0.0, 1.0 - probability - push)
        rows.append(
            {
                "method": method,
                "game_id": game_id,
                "quoted_line": quoted_line,
                "line_offset": offset,
                "alternative_line": alternative_line,
                "home_cover_probability": probability,
                "home_cover_probability_excluding_push": max(probability - push, 0.0),
                "push_probability": push,
                "home_loss_probability": loss,
                "pick_probability": probability if probability >= 0.5 else 1.0 - probability,
                "confidence": abs(probability - 0.5),
            }
        )
    return rows


def _write_line_sweep(forecast_directory: Path) -> None:
    """A synthetic ``line_sweep.parquet`` covering every game in ``_weekly_forecast_frame``.

    Mirrors the real ``nfl-ats margin-predict --line-sweep`` artifact shape:
    ``score_outcome_week_line_sweep`` stacks every margin-distribution
    method (``market``, ``fair_margin``, ``market_residual``) into one file
    with a ``method`` column and no other per-game uniqueness, so a naive
    reader sees 2-3x duplicate rows per game. The non-``market_residual``
    rows here use deliberately different probabilities so a test can catch
    a dashboard regression that forgets to filter down to the active
    ``ats_method`` (``market_residual``, per ``_write_weekly_forecast``).
    """

    rows: list[dict[str, Any]] = []
    for game_id, quoted_line, probability in (
        ("2030_01_NE_SEA", 3.5, 0.485),
        ("2030_01_SF_LA", -3.5, 0.616),
        ("2030_01_ATL_PIT", -3.0, 0.512),
        ("2023_01_BUF_MIA", -2.5, 0.70),
    ):
        rows += _synthetic_sweep_rows(game_id, quoted_line, probability, method="market_residual")
        rows += _synthetic_sweep_rows(game_id, quoted_line, probability - 0.10, method="market")
        rows += _synthetic_sweep_rows(
            game_id, quoted_line, probability + 0.05, method="fair_margin"
        )
    _write_parquet(pd.DataFrame(rows), forecast_directory / "line_sweep.parquet")


def _write_close_predictions(artifacts_root: Path, run_id: str) -> None:
    """A predicted-close artifact for exactly one game, to exercise per-game feature-detection."""

    directory = artifacts_root / "close_predictions" / run_id
    _write_parquet(
        pd.DataFrame(
            {
                "game_id": ["2030_01_NE_SEA"],
                "predicted_close_home_spread": [4.0],
                "model_id": ["mkt06-test"],
                "created_at_utc": ["2030-08-13T00:00:00Z"],
            }
        ),
        directory / "predictions.parquet",
    )


def _write_attribution(artifacts_root: Path, run_id: str) -> None:
    """A market-decomposition attribution artifact for exactly one game.

    Mirrors ``nfl_ats.market_decomposition.attribute_predictions``'s schema: one
    row per ``(game_id, family)`` with the same ``explanation`` repeated on every
    row for a game. Only ``2030_01_SF_LA`` gets rows here, so a test can exercise
    both the "has an explanation" path and the "falls back to the generic
    sentence" path for the other games in ``_weekly_forecast_frame``.
    """

    directory = artifacts_root / "market_decomposition" / run_id
    explanation = (
        "The model leans LA 4.1 points more than the market mainly because of recent "
        "offensive performance (+3.2)."
    )
    rows = [
        {
            "game_id": "2030_01_SF_LA",
            "season": 2030,
            "week": 1,
            "home_team": "LA",
            "away_team": "SF",
            "family": family,
            "contribution": contribution,
            "predicted_residual": 4.1,
            "actual_residual": float("nan"),
            "explanation": explanation,
        }
        for family, contribution in (("offense", 3.2), ("defense", 0.9), ("intercept", 0.0))
    ]
    _write_parquet(pd.DataFrame(rows), directory / "attribution.parquet")


@pytest.fixture
def populated_env(
    tmp_path: Path, schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame]
) -> tuple[Path, Path]:
    """A dashboard data/artifacts tree with realistic content for every page."""

    from nfl_ats.features import build_game_features

    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"

    schedules, stats = schedules_and_stats
    write_snapshot(schedules, stats, seasons=[2022], raw_root=data_root / "raw")
    features = build_game_features(schedules, stats, span=3, min_periods=1)
    _write_parquet(features, data_root / "processed" / "game_features.parquet")
    _write_json(
        {
            "built_at_utc": "2022-09-01T00:00:00Z",
            "source_snapshot": "20220901T000000Z",
            "ewm_span": 3,
            "offseason_retention": 0.75,
            "rows": len(features),
            "first_season": 2022,
            "last_season": 2022,
        },
        data_root / "processed" / "game_features.manifest.json",
    )

    forecast_directory = _write_weekly_forecast(artifacts_root, "2030-week-01-run")
    evaluation_directory = _write_margins_evaluation(artifacts_root, "eval-run")
    _write_active_model(
        artifacts_root,
        evaluation_relative=str(evaluation_directory.relative_to(artifacts_root).as_posix()),
        forecast_relative=str(forecast_directory.relative_to(artifacts_root).as_posix()),
    )
    _write_backtest_run(artifacts_root, "backtest-run")
    _write_nested_evaluation(artifacts_root, "nested-run")
    _write_player_model_selection(artifacts_root, "player-selection-run")
    _write_participation_experiment(artifacts_root, "participation-run")
    _write_availability_experiment(artifacts_root, "availability-run")
    _write_feature_experiment(artifacts_root, "feature-run")
    _write_player_experiment(artifacts_root, "player-run")
    _write_market_snapshots(data_root)
    _write_attribution(artifacts_root, "decomp-run")

    return data_root, artifacts_root


@pytest.fixture
def empty_env(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    return data_root, artifacts_root


@pytest.fixture
def sweep_env(populated_env: tuple[Path, Path]) -> tuple[Path, Path]:
    """``populated_env`` plus a line-sweep table and a predicted-close artifact.

    Layered on the same forecast directory and market snapshots
    ``populated_env`` already writes, so this exercises the "everything
    present" path: confidence ribbon + fair-line callout, a line journey row
    with real opener/latest quotes, and a predicted close for one game.
    """

    data_root, artifacts_root = populated_env
    forecast_directory = artifacts_root / "margin_predictions" / "2030-week-01-run"
    _write_line_sweep(forecast_directory)
    _write_close_predictions(artifacts_root, "close-run")
    return data_root, artifacts_root


@pytest.fixture
def multi_run_env(populated_env: tuple[Path, Path]) -> tuple[Path, Path]:
    """``populated_env`` plus a second, unlinked saved run for two active-chain artifact types.

    ``populated_env``'s active model manifest links ``margins/eval-run``. This adds
    ``margins/eval-run-older``, whose directory name sorts *ahead* of it
    (``artifact_directories`` sorts descending by name) -- so a page that still
    picked "the latest artifact of its own type" would show the unlinked run
    instead of the active one. Also adds a second, unlinked ``nested_evaluations``
    run to exercise the same "featured run + older runs" split on a page with no
    active-model concept at all.
    """

    data_root, artifacts_root = populated_env
    _write_margins_evaluation(artifacts_root, "eval-run-older")
    _write_nested_evaluation(artifacts_root, "nested-run-older")
    return data_root, artifacts_root


@pytest.fixture
def missing_active_evaluation_env(populated_env: tuple[Path, Path]) -> tuple[Path, Path]:
    """``populated_env`` with the active model's evaluation artifact deleted from disk.

    The manifest still declares ``margins/eval-run`` (unedited), but the directory
    itself is gone, and it was the *only* saved run -- the "active-declared-but-
    missing, and nothing else to fall back to" case.
    """

    data_root, artifacts_root = populated_env
    shutil.rmtree(artifacts_root / "margins" / "eval-run")
    return data_root, artifacts_root


@pytest.fixture
def missing_active_with_other_runs_env(multi_run_env: tuple[Path, Path]) -> tuple[Path, Path]:
    """``multi_run_env`` with the active model's evaluation artifact deleted from disk.

    Unlike ``missing_active_evaluation_env``, ``margins/eval-run-older`` still
    exists -- the "active-declared-but-missing, but other runs remain browsable"
    case: the page must say the active artifact is missing *and* still let the
    owner look at what is on disk, clearly marked as not the active model.
    """

    data_root, artifacts_root = multi_run_env
    shutil.rmtree(artifacts_root / "margins" / "eval-run")
    return data_root, artifacts_root


@pytest.fixture
def home_env_no_quotes(tmp_path: Path) -> tuple[Path, Path]:
    """A forecast card with no active-model link, no sweep, and no live quotes.

    The minimal env for the home page's "nothing captured yet" line-journey
    case -- deliberately skips ``_write_market_snapshots`` (unlike
    ``populated_env``), matching the real first-render state where the live
    capture archive is empty.
    """

    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    _write_weekly_forecast(artifacts_root, "2030-week-01-run")
    return data_root, artifacts_root


def _run_page(
    page: str, data_root: Path, artifacts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> AppTest:
    """Run one page through the real app shell so ``st.navigation``/``st.page_link`` work.

    Individual page scripts use ``st.page_link``, which requires a live page
    registry from ``st.navigation`` -- running a page file in isolation (not
    through ``app.py``) raises. Booting the real app and switching pages
    matches how Streamlit actually serves a multipage app.
    """

    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))
    app = AppTest.from_file("src/nfl_ats/dashboard/app.py")
    app.run(timeout=30)
    if page != "home.py":
        app.switch_page(f"app_pages/{page}")
        app.run(timeout=30)
    return app


# ---------------------------------------------------------------------------
# Home page: "This week's picks" (the picks board)
# ---------------------------------------------------------------------------


def test_home_page_renders_board_with_data(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = populated_env
    app = _run_page("home.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception
    assert app.title[0].value == "This week's picks"

    badge_text = " ".join(m.value for m in app.markdown)
    assert "-badge[" in badge_text
    assert "Synchronized with the active model" in badge_text

    captions = [c.value for c in app.caption]
    assert any("Conf = chance the pick covers" in text for text in captions)  # legend, once

    html_elements = app.get("html")
    assert len(html_elements) == 1  # the whole board is one st.html call
    board_html = html_elements[0].body
    assert "SEA" in board_html and "LA" in board_html and "PIT" in board_html
    assert board_html.count('class="pb-row"') == len(_weekly_forecast_frame())


def test_home_page_empty_state_names_next_capture(
    empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = empty_env
    app = _run_page("home.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception
    assert any("No pick card yet" in info.value for info in app.info)
    assert any("Tuesday" in text.value for text in app.markdown)


def test_home_page_without_sweep_renders_rows_without_a_strip(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``line_sweep.parquet`` at all -- every row renders without a confidence
    strip, but the fair line still comes from ``predicted_market_residual``, which
    needs no sweep at all (see ``canonical_fair_line``)."""

    data_root, artifacts_root = populated_env
    app = _run_page("home.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception

    board_html = app.get("html")[0].body
    assert 'class="pb-ribbon-full"' not in board_html  # no sweep -> no numbered ribbon
    assert board_html.count('<span class="pb-no-sweep">') == len(_weekly_forecast_frame())

    # populated_env has live quotes but no sweep, so the journey text (now
    # rendered inside the board's own HTML, not a separate caption) shows a
    # real Open/Now but no predicted close.
    assert "Open -3.75" in board_html
    # NE @ SEA: spread_line 3.5, residual -0.3 -> fair +3.2 (home-oriented);
    # NE is picked (away) and reads 0.3 pt better than that fair line --
    # both computed and shown without any sweep artifact.
    assert "Fair +3.2" in board_html
    assert "getting 0.3 pt better than fair" in board_html


def test_home_page_rows_carry_tier_css_classes(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = populated_env
    app = _run_page("home.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception
    badge_text = " ".join(m.value for m in app.markdown)
    assert "strong" in badge_text and "leans" in badge_text and "coin flips" in badge_text

    board_html = app.get("html")[0].body
    # SF @ LA (0.616) is a strong lean; NE @ SEA (0.515) is a coin flip -- both
    # tiers, driven by ui.confidence_tier's 57%/52% thresholds, must show up as
    # the pick chip's own CSS class, not just in the (always-present) stylesheet.
    assert 'class="pb-pick pb-strong"' in board_html
    assert 'class="pb-pick pb-flip"' in board_html


def test_home_page_renders_confidence_ribbon_and_fair_line(
    sweep_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = sweep_env
    app = _run_page("home.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception

    board_html = app.get("html")[0].body
    assert "Open" in board_html and "Fair" in board_html and "→" in board_html
    captions = [c.value for c in app.caption]
    assert any("amber box = market" in text for text in captions)  # legend, written once

    games = len(_weekly_forecast_frame())
    offsets = len(_sweep_offsets())
    # Two separate tables per row (a numberless mini strip in the summary, plus
    # a numbered ribbon in the detail), each with one <td> per swept line.
    assert board_html.count('class="pb-ribbon-mini"') == games
    assert board_html.count('class="pb-ribbon-full"') == games
    assert board_html.count('<td class="pb-rm') == games * offsets
    assert board_html.count('<td class="pb-rc') == games * offsets
    assert "table-layout:fixed" in board_html
    assert "pb-mkt" in board_html  # the amber-boxed market column


def test_home_page_predicted_close_is_feature_detected(
    sweep_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = sweep_env
    app = _run_page("home.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception

    board_html = app.get("html")[0].body
    # The one game with a close_predictions row shows a real number...
    assert "Close (pred) +4" in board_html
    # ...while every other game still shows an em dash.
    assert "Close (pred) —" in board_html


def test_home_page_line_journey_em_dash_without_quotes(
    home_env_no_quotes: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No live captures, no sweep, no predicted close: the market-quote fields
    (Open/Now/Close) are all em dashes, but Fair and its framing still come
    from predicted_market_residual, which needs neither quotes nor a sweep."""

    data_root, artifacts_root = home_env_no_quotes
    app = _run_page("home.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception

    board_html = app.get("html")[0].body
    assert "Open — → Now — → Close (pred) — → Fair +3.2" in board_html
    assert "getting 0.3 pt better than fair" in board_html
    captions = [c.value for c in app.caption]
    assert any("No live line captures yet this week" in text for text in captions)
    assert any("Predicted close isn't wired up yet" in text for text in captions)


def test_home_page_explanations_are_feature_detected(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The market-decomposition attribution artifact (see ``_write_attribution``)
    only covers one game; every other game falls back to the generic sentence."""

    data_root, artifacts_root = populated_env
    app = _run_page("home.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception

    board_html = app.get("html")[0].body
    assert "recent offensive performance" in board_html
    assert "Model and market are close on this one." in board_html


def test_home_page_toggle_all_button_is_present_and_progressively_enhanced(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner-requested expand/collapse-all control: an addEventListener bound
    to a button that starts hidden (revealed only once its script actually runs),
    never an inline onclick attribute (stripped by Streamlit's HTML sanitizer)."""

    data_root, artifacts_root = populated_env
    app = _run_page("home.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception

    board_html = app.get("html")[0].body
    assert '<button class="pb-toggle" id="toggle-all" type="button" style="display:none">' in (
        board_html
    )
    assert "expand all</button>" in board_html
    assert "<script>" in board_html
    assert 'document.getElementById("toggle-all")' in board_html
    assert 'button.style.display = "";' in board_html
    assert 'document.querySelectorAll("details.pb-row")' in board_html
    assert "onclick" not in board_html.lower()


# ---------------------------------------------------------------------------
# Track record: "Is this thing good?"
# ---------------------------------------------------------------------------


def test_track_record_with_data(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = populated_env
    app = _run_page("track_record.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception
    assert app.title[0].value == "Is this thing actually good?"
    assert any("52.1%" in m.value or "52.0%" in m.value for m in app.markdown)
    assert any("not proof of a profitable edge" in w.value for w in app.warning)


def test_track_record_empty_state(
    empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = empty_env
    app = _run_page("track_record.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception
    assert any("No synchronized active model" in info.value for info in app.info)


def test_track_record_reports_missing_evaluation_artifact_explicitly(
    missing_active_evaluation_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest still declares margins/eval-run, but the directory is gone.

    The headline accuracy (baked into the manifest itself) still shows, but the
    season-by-season detail that depends on re-reading the artifact must say so
    explicitly rather than rendering an empty "no games settled yet" state that
    would misattribute a missing artifact to an empty season.
    """

    data_root, artifacts_root = missing_active_evaluation_env
    app = _run_page("track_record.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception
    assert any("52.1%" in m.value or "52.0%" in m.value for m in app.markdown)
    error_text = " ".join(e.value for e in app.error)
    assert "active model's evaluation artifact is missing locally" in error_text
    assert "margin-backtest" in error_text


# ---------------------------------------------------------------------------
# Market: "What's happening with the lines?"
# ---------------------------------------------------------------------------


def test_market_page_with_data(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = populated_env
    app = _run_page("market.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception
    assert app.title[0].value == "What's happening with the lines?"
    assert len(app.metric) >= 2


def test_market_page_empty_state(
    empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = empty_env
    app = _run_page("market.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception
    assert any("No live captures yet this week" in info.value for info in app.info)


# ---------------------------------------------------------------------------
# Research pages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "page",
    [
        "research_backtests.py",
        "research_validation.py",
        "research_experiments.py",
        "research_bankroll.py",
        "research_data_health.py",
        "research_explorer.py",
        "research_glossary.py",
    ],
)
def test_research_pages_render_without_data(
    page: str, empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = empty_env
    app = _run_page(page, data_root, artifacts_root, monkeypatch)
    assert not app.exception


@pytest.mark.parametrize(
    "page",
    [
        "research_backtests.py",
        "research_validation.py",
        "research_experiments.py",
        "research_bankroll.py",
        "research_data_health.py",
        "research_explorer.py",
        "research_glossary.py",
    ],
)
def test_research_pages_render_with_data(
    page: str, populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = populated_env
    app = _run_page(page, data_root, artifacts_root, monkeypatch)
    assert not app.exception


# ---------------------------------------------------------------------------
# Research-tab staleness fix: active model featured, older runs explicit,
# missing artifacts reported rather than silently substituted
# ---------------------------------------------------------------------------


def test_backtests_outcome_lab_features_the_active_model_over_a_newer_named_run(
    multi_run_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = multi_run_env
    app = _run_page("research_backtests.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception

    subheaders = [s.value for s in app.subheader]
    assert "The active model" in subheaders

    badge_text = " ".join(m.value for m in app.markdown)
    assert "Linked to the active model" in badge_text

    captions = [c.value for c in app.caption]
    # eval-run is the active model's declared evaluation; eval-run-older sorts
    # ahead of it alphabetically but must not be what's featured by default.
    assert any("margins/eval-run`" in text for text in captions)

    expander_labels = [e.label for e in app.expander]
    assert any(label.startswith("Older research runs") for label in expander_labels)


def test_backtests_outcome_lab_reports_missing_active_artifact_explicitly(
    missing_active_evaluation_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = missing_active_evaluation_env
    app = _run_page("research_backtests.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception

    error_text = " ".join(e.value for e in app.error)
    assert "active model's evaluation artifact is missing locally" in error_text
    assert "margin-backtest" in error_text


def test_backtests_outcome_lab_missing_active_artifact_still_shows_other_runs(
    missing_active_with_other_runs_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = missing_active_with_other_runs_env
    app = _run_page("research_backtests.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception

    error_text = " ".join(e.value for e in app.error)
    assert "active model's evaluation artifact is missing locally" in error_text
    # Does not silently substitute -- but does not strand the user either: the
    # other saved run is still shown, explicitly labeled as not the active model.
    assert any("Showing other saved runs instead" in c.value for c in app.caption)
    assert any("margins/eval-run-older`" in c.value for c in app.caption)


def test_nested_evaluation_features_the_newest_run_with_older_runs_explicit(
    multi_run_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = multi_run_env
    app = _run_page("research_validation.py", data_root, artifacts_root, monkeypatch)
    assert not app.exception
    # "Nested evaluation" is the default selectbox choice, so its picker renders
    # without further interaction.
    subheaders = [s.value for s in app.subheader]
    assert "Latest run" in subheaders  # no active-model concept for this artifact type
    expander_labels = [e.label for e in app.expander]
    assert any(label.startswith("Older research runs") for label in expander_labels)


# ---------------------------------------------------------------------------
# App entry point (navigation shell)
# ---------------------------------------------------------------------------


def test_app_entry_boots_and_defaults_to_home(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = populated_env
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))
    app = AppTest.from_file("src/nfl_ats/dashboard/app.py")
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "This week's picks"
    assert any("no wagering integration" in caption.value for caption in app.sidebar.caption)
