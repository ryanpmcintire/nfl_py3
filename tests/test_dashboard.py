"""Tests for the rebuilt dashboard: components, tokens, the state anchor, pages.

Four layers, cheapest first:

* :mod:`nfl_ats.dashboard.viz` is pure -- every component takes data and
  returns an HTML string, so its marks, labels, table-view twins and
  color-never-alone invariants are unit-tested with no Streamlit at all.
  Note that ``st.html``'s sanitizer strips ``<svg>``, so the components draw
  with positioned ``<div>``s and ``clip-path``; the assertions below check the
  marks and labels those produce, never SVG markup.
* :mod:`nfl_ats.dashboard.theme` owns the token contract. The tests below pin
  it from both ends: the stylesheet defines the light palette and *both* dark
  scopes, and every ``var(--...)`` any component emits is defined there.
* :mod:`nfl_ats.dashboard.state` is the single anchor. It is exercised against
  temporary artifact trees -- absent, synchronized, and provenance-stale --
  plus the modification-time cache keying that keeps a rebuild visible.
* Each of the four pages is driven through ``streamlit.testing.v1.AppTest``
  against the real app shell (``st.navigation`` needs it), with a realistic
  tree and with an empty one, asserting that neither raises and that the page
  renders the content the owner reads.

Fixtures write small synthetic artifact/data trees under ``tmp_path``; the
dashboard is pointed at them through ``NFL_ATS_DATA_DIR`` /
``NFL_ATS_ARTIFACTS_DIR`` (see ``nfl_ats.dashboard.data.data_root`` and
``artifacts_root``).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nfl_ats.dashboard import theme, viz
from nfl_ats.dashboard.findings_content import DETAIL_SUMMARY_LABEL, FINDINGS, GROUPS
from nfl_ats.dashboard.state import load_csv, load_parquet, project_state

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "src" / "nfl_ats" / "dashboard" / "app.py"
PAGES = (
    "picks.py",
    "findings.py",
    "model_explanation.py",
    "track_record.py",
    "engine_room.py",
    "workbench.py",
)

# The feature-table digest the active manifest records, and a different one an
# artifact built before the last feature rebuild would have recorded.
ACTIVE_FEATURE_SHA = "a" * 64
EARLIER_FEATURE_SHA = "b" * 64

FORECAST_RUN = "2030-week-01-20300812T211533Z"
EVALUATION_RUN = "eval-20300810T120000Z"
OPENER_RUN = "opener-20300813T010000Z"
LEDGER_RUN = "clv-20300813T020000Z"
CLOSE_RUN = "close-20300813T030000Z"
DECOMP_RUN = "decomp-20300811T120000Z"
SNAPSHOT_ID = "20300810T060000Z"

FUTURE_GAMEDAY = "2030-09-08"
# Distinct kickoffs, so the card order the page renders is the kickoff order.
FUTURE_KICKOFFS = [
    "2030-09-08T17:00:00Z",
    "2030-09-08T20:25:00Z",
    "2030-09-09T00:20:00Z",
    "2030-09-10T00:15:00Z",
]

# Exactly one game in the fixture card clears this, so "strong lean" copy can be
# asserted to appear on exactly one card (see picks.py's STRONG_LEAN_POINTS).
STRONG_LEAN_GAME = "2030_01_SF_LA"
STRONG_LEAN_MATCHUP = "SF at LA"
SWEEP_OFFSETS = [round(-4.0 + 0.5 * step, 1) for step in range(17)]


# ---------------------------------------------------------------------------
# viz.py -- pure component unit tests (no Streamlit)
# ---------------------------------------------------------------------------


def _sweep_points() -> list[tuple[float, float]]:
    return [(-1.0, 0.55), (-0.5, 0.565), (0.0, 0.58), (0.5, 0.60), (1.0, 0.62)]


def test_sweep_curve_direct_labels_the_quoted_line() -> None:
    html = viz.sweep_curve(
        "sweep-1", _sweep_points(), quoted_line=0.0, pick_text="SEA to cover", quote_label="SEA -3"
    )
    # Selective direct labelling: the quoted line carries the only value label,
    # and the coin-flip baseline is named rather than left to the reader.
    assert "58% at the line" in html
    assert html.count("at the line") == 1
    assert ">50%</span>" in html
    assert 'aria-label="Confidence in SEA to cover across alternative lines"' in html
    # The x axis names the quote in the pool's own terms instead of "0".
    assert ">SEA -3</span>" in html
    assert ">+0</span>" not in html


def test_sweep_curve_ships_a_table_view_twin() -> None:
    html = viz.sweep_curve("sweep-1", _sweep_points(), quoted_line=0.0, pick_text="SEA")
    assert '<details class="table-view"><summary>View as table</summary>' in html
    assert "<th>Line vs. quote</th><th>Confidence</th>" in html
    assert html.count("<tr><td>") == len(_sweep_points())
    assert "<td>-1</td><td>55%</td>" in html
    assert "<td>+1</td><td>62%</td>" in html


def test_sweep_curve_carries_its_geometry_in_data_attributes() -> None:
    # Hover wiring is delegated (Streamlit strips inline on* handlers), so the
    # script recomputes geometry from these attributes -- they are the contract.
    html = viz.sweep_curve("sweep-2030_01_SF_LA", _sweep_points(), quoted_line=0.0, pick_text="LA")
    for attribute in ("data-points", "data-xmin", "data-xmax", "data-ymin", "data-ymax"):
        assert f"{attribute}=" in html
    assert 'id="sweep-2030_01_SF_LA"' in html
    payload = re.search(r'data-points="([^"]+)"', html)
    assert payload is not None
    assert json.loads(payload.group(1).replace("&quot;", '"')) == [
        [line, probability] for line, probability in _sweep_points()
    ]


def test_sweep_curve_without_points_falls_back_to_an_empty_state() -> None:
    html = viz.sweep_curve("sweep-1", [], quoted_line=0.0, pick_text="SEA")
    assert html == viz.empty_state(
        "No line sweep saved", "This card predates the line-sweep artifact."
    )
    assert "<svg" not in html


MARKER = "top:50%;width:9px;height:9px;"


def test_probability_meter_clamps_out_of_range_probabilities() -> None:
    high = viz.probability_meter(1.4, label="Chance the pick covers", width=200)
    assert ">100%</span>" in high
    assert 'aria-label="Chance the pick covers: 100%"' in high
    assert f"left:100.0%;{MARKER}" in high  # the marker stops at the track's end

    low = viz.probability_meter(-0.3, label="Chance the pick covers", width=200)
    assert ">0%</span>" in low
    assert f"left:0.0%;{MARKER}" in low

    # The coin flip is the anchor of the scale, always drawn and always named.
    assert ">coin flip</span>" in high
    assert "background:var(--baseline);" in high


def test_line_journey_needs_two_numbers_before_it_draws() -> None:
    sentence = "Line comparison appears once two of the three numbers exist."
    assert viz.line_journey(opener=None, fair=None, predicted_close=None) == (
        f'<p class="fine">{sentence}</p>'
    )
    assert viz.line_journey(opener=-3.5, fair=None, predicted_close=None) == (
        f'<p class="fine">{sentence}</p>'
    )


def test_line_journey_labels_every_mark_it_draws() -> None:
    # Identity is carried by color + shape + a labelled legend row, never color
    # alone: two values give two legend entries, three give three.
    two = viz.line_journey(opener=-3.5, fair=-2.9, predicted_close=None)
    assert 'opened <span class="num">-3.5</span>' in two
    assert 'our number <span class="num">-2.9</span>' in two
    assert "close guess" not in two

    three = viz.line_journey(opener=-3.5, fair=-2.9, predicted_close=-3.1)
    assert "opened" in three and "our number" in three and "close guess" in three
    assert three.count('class="fine"') == 3
    # Shape distinguishes the three marks, so the legend reads without color:
    # square opener, filled circle for our number, hollow circle for the close.
    assert three.count("border-radius:2px;") == 1
    assert three.count("&#9632;") == 1
    assert three.count("&#9679;") == 1
    assert three.count("&#9675;") == 1


def test_season_bars_draws_one_bar_per_row_and_a_reference_line() -> None:
    rows = [("2020", 0.474), ("2021", 0.523), ("2022", 0.531)]
    html = viz.season_bars(rows)

    assert html.count("border-radius:4px;background:var(--series-model);") == len(rows)
    for label, value in rows:
        assert f">{label}</span>" in html  # direct-labelled season
        assert f">{value:.1%}</span>" in html  # direct-labelled value
    # The reference guide runs through every bar and is named once, at the foot.
    assert html.count("dashed var(--baseline);") == len(rows)
    assert html.count(">coin flip</span>") == 1
    assert 'aria-label="By season vs. coin flip"' in html
    # The losing season is drawn recessive rather than dropped.
    assert "opacity:0.55;" in html
    assert html.count("opacity:1.0;") == 2


def test_season_bars_without_rows_falls_back_to_an_empty_state() -> None:
    assert viz.season_bars([]) == viz.empty_state(
        "Nothing to chart yet", "This fills in once seasons are scored."
    )


def test_stat_tile_delta_arrow_follows_the_delta_direction() -> None:
    good = viz.stat_tile("k", "52.5%", "c", delta_text="+2.5 points", delta_good=True)
    assert "&#9650; +2.5 points" in good
    assert "var(--good-text)" in good

    bad = viz.stat_tile("k", "48.0%", "c", delta_text="-2.0 points", delta_good=False)
    assert "&#9660; -2.0 points" in bad
    assert "var(--critical)" in bad

    neutral = viz.stat_tile("k", "52.5%", "c", delta_text="1,537 games", delta_good=None)
    assert "1,537 games" in neutral
    assert "&#9650;" not in neutral and "&#9660;" not in neutral
    assert "var(--ink-2)" in neutral

    bare = viz.stat_tile("k", "52.5%", "c")
    assert "&#9650;" not in bare and "&#9660;" not in bare
    assert '<div class="hero num">52.5%</div><p class="fine"' in bare


@pytest.mark.parametrize("kind", ["good", "warning", "critical", "unrecognized"])
def test_status_line_never_carries_meaning_by_color_alone(kind: str) -> None:
    html = viz.status_line(kind, "Synchronized with the active model")
    # A glyph badge plus the words, every time: the class only tints them.
    # (An icon, not an <svg> -- st.html's sanitizer strips SVG entirely.)
    assert "border:1.5px solid currentColor;" in html
    assert "<span>Synchronized with the active model</span>" in html
    assert f'class="status {kind}"' in html


def test_status_line_glyphs_differ_by_kind() -> None:
    glyphs = {kind: viz.status_line(kind, "x") for kind in ("good", "warning", "critical")}
    assert len(set(glyphs.values())) == 3
    assert "&#10003;" in glyphs["good"]
    assert "&#215;" in glyphs["critical"]
    # An unknown kind still ships a glyph rather than degrading to color alone.
    assert "border:1.5px solid currentColor;" in viz.status_line("mystery", "x")


# ---------------------------------------------------------------------------
# theme.py -- the token contract
# ---------------------------------------------------------------------------

TOKEN_DECLARATION = re.compile(r"--([a-z0-9-]+)\s*:")
TOKEN_REFERENCE = re.compile(r"var\(--([a-z0-9-]+)\)")


def _block_after(stylesheet: str, selector: str) -> str:
    """The declarations of the first rule opened by ``selector``."""

    assert selector in stylesheet, f"missing selector: {selector}"
    return stylesheet.split(selector, 1)[1].split("}", 1)[0]


def _viz_sample_html() -> str:
    """One rendering of every component, for parsing the tokens they reference."""

    return "".join(
        (
            viz.page_header("Track record", "How often the picks landed", "Two lines."),
            viz.card("<p>plain</p>"),
            viz.card("<p>accented</p>", accent=True),
            viz.stat_tile("k", "52.5%", "c", delta_text="up", delta_good=True),
            viz.stat_tile("k", "48.0%", "c", delta_text="down", delta_good=False),
            viz.stat_tile("k", "52.5%", "c", delta_text="flat", delta_good=None),
            viz.status_line("good", "ok"),
            viz.status_line("warning", "stale"),
            viz.status_line("critical", "broken"),
            viz.empty_state("Nothing yet", "It fills in later."),
            viz.probability_meter(0.62, label="Chance the pick covers"),
            viz.sweep_curve("s", _sweep_points(), quoted_line=0.0, pick_text="LA to cover"),
            viz.line_journey(opener=-3.5, fair=-2.9, predicted_close=-3.1),
            viz.season_bars([("2024", 0.53), ("2025", 0.48)]),
        )
    )


def test_stylesheet_defines_light_tokens_and_both_dark_scopes() -> None:
    stylesheet = theme.stylesheet()
    light = _block_after(stylesheet, ".ats {")
    media_dark = _block_after(stylesheet, '.ats:not([data-theme="light"]) {')
    explicit_dark = _block_after(stylesheet, '.ats[data-theme="dark"] {')

    assert "@media (prefers-color-scheme: dark)" in stylesheet
    # The system-preference scope must stay guarded, or an explicit light choice
    # loses to the OS on a dark desktop.
    assert '.ats:not([data-theme="light"])' in stylesheet

    expected = set(theme.TOKENS_LIGHT)
    assert expected == set(theme.TOKENS_DARK)
    for block in (light, media_dark, explicit_dark):
        assert set(TOKEN_DECLARATION.findall(block)) >= expected

    assert f"--surface: {theme.TOKENS_LIGHT['surface']};" in light
    for block in (media_dark, explicit_dark):
        assert f"--surface: {theme.TOKENS_DARK['surface']};" in block
        assert "color-scheme: dark;" in block


def test_every_token_the_components_reference_is_defined_in_the_stylesheet() -> None:
    referenced = set(TOKEN_REFERENCE.findall(_viz_sample_html()))
    defined = set(TOKEN_DECLARATION.findall(theme.stylesheet()))
    assert referenced, "the sample rendered no role tokens at all"
    assert referenced <= defined, f"undefined tokens: {sorted(referenced - defined)}"


def test_stacked_card_margin_is_scoped_to_the_stack_not_global() -> None:
    # The regression the old design shipped: a global `.card + .card` margin also
    # fires between side-by-side cards in a flex `.row`, dropping every tile
    # after the first by 14px. The rule must be opt-in under `.stack`.
    stylesheet = theme.stylesheet()
    assert ".stack > .card + .card" in stylesheet
    assert ".ats .card + .card" not in stylesheet


# ---------------------------------------------------------------------------
# Fixture tree: a realistic artifacts/data pair for every page
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
    """Four games, of which exactly one is a strong lean.

    ``fair_spread`` shares ``spread_line``'s home-oriented sign convention, so
    ``fair_spread == spread_line + predicted_market_residual`` by construction --
    the same relationship ``margin.py`` writes.
    """

    return pd.DataFrame(
        {
            "game_id": [
                "2030_01_NE_SEA",
                STRONG_LEAN_GAME,
                "2030_01_ATL_PIT",
                "2030_01_MIA_BUF",
            ],
            "season": [2030, 2030, 2030, 2030],
            "week": [1, 1, 1, 1],
            "game_type": ["REG", "REG", "REG", "REG"],
            "gameday": [FUTURE_GAMEDAY, FUTURE_GAMEDAY, FUTURE_GAMEDAY, "2030-09-09"],
            "weekday": ["Sunday", "Sunday", "Sunday", "Monday"],
            "gametime": ["13:00:00", "16:25:00", "20:20:00", "20:15:00"],
            "kickoff": FUTURE_KICKOFFS,
            "away_team": ["NE", "SF", "ATL", "MIA"],
            "home_team": ["SEA", "LA", "PIT", "BUF"],
            "spread_line": [3.5, -3.5, -3.0, -2.5],
            "total_line": [44.5, 47.0, 42.5, 45.0],
            "fair_spread": [3.2, 0.6, -2.6, -1.5],
            "predicted_market_residual": [-0.3, 4.1, 0.4, 1.0],
            "home_cover_probability": [0.485, 0.616, 0.512, 0.545],
            "edge": [-0.015, 0.116, 0.012, 0.045],
            "bet_side": ["PASS", "AWAY", "PASS", "PASS"],
            "method": ["market_residual"] * 4,
        }
    )


def _write_weekly_forecast(artifacts: Path, run_id: str) -> Path:
    directory = artifacts / "margin_predictions" / run_id
    recommendations = _weekly_forecast_frame()
    _write_csv(recommendations, directory / "recommendations.csv")
    # A real margin-predict run writes both; artifact discovery for this
    # directory requires predictions.csv even though recommendations.csv is the
    # file actually read once the directory is found.
    _write_csv(recommendations, directory / "predictions.csv")
    _write_json(
        {
            "season": 2030,
            "week": 1,
            "feature_profile": "player",
            "ats_method": "market_residual",
            "regressor": "ridge",
            "ridge_alpha": 10.0,
            "calibration_method": "isotonic",
            "created_at_utc": "2030-08-12T21:15:33+00:00",
            "prediction_safety": {"status": "PASS", "warnings": []},
            "provenance": {"feature_table": {"sha256": ACTIVE_FEATURE_SHA}},
        },
        directory / "metadata.json",
    )
    return directory


def _write_line_sweep(forecast_directory: Path) -> None:
    """A line sweep for every game, stacked across methods like the real artifact.

    ``score_outcome_week_line_sweep`` writes every margin-distribution method
    into one file keyed only by ``method``, so a reader that forgets to filter
    down to the card's ``ats_method`` sees each game twice. The non-active rows
    here carry deliberately different probabilities so that regression shows up
    as a doubled table-view twin.
    """

    rows: list[dict[str, Any]] = []
    for game_id, base in (
        ("2030_01_NE_SEA", 0.515),
        (STRONG_LEAN_GAME, 0.616),
        ("2030_01_ATL_PIT", 0.512),
        ("2030_01_MIA_BUF", 0.545),
    ):
        for method, shift in (("market_residual", 0.0), ("market", -0.10)):
            for offset in SWEEP_OFFSETS:
                probability = min(max(base + shift - 0.03 * offset, 0.02), 0.98)
                rows.append(
                    {
                        "method": method,
                        "game_id": game_id,
                        "quoted_line": 0.0,
                        "line_offset": offset,
                        "alternative_line": offset,
                        "home_cover_probability": probability,
                        "push_probability": 0.0,
                        "pick_probability": max(probability, 1.0 - probability),
                    }
                )
    _write_parquet(pd.DataFrame(rows), forecast_directory / "line_sweep.parquet")


def _write_margins_evaluation(artifacts: Path, run_id: str) -> Path:
    directory = artifacts / "margins" / run_id
    _write_csv(
        pd.DataFrame(
            [
                {
                    "method": method,
                    "cover_accuracy": 0.5205,
                    "cover_brier_score": 0.2495,
                    "cover_games": 2075,
                    "margin_mae": 9.9,
                }
                for method in ("market", "fair_margin", "market_residual")
            ]
        ),
        directory / "summary.csv",
    )
    _write_json(
        {
            "regressor": "ridge",
            "feature_profile": "player",
            "provenance": {"feature_table": {"sha256": ACTIVE_FEATURE_SHA}},
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
            "model_id": "d41d8cd98f00b204",
            "activated_at_utc": "2030-08-12T21:15:33+00:00",
            "method": "market_residual",
            "feature_profile": "player",
            "regressor": "ridge",
            "ridge_alpha": 10.0,
            "calibration_method": "isotonic",
            "feature_table_sha256": ACTIVE_FEATURE_SHA,
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
                "created_at_utc": "2030-08-12T21:15:33+00:00",
            },
        },
        artifacts / "active_ats_model.json",
    )


def _opener_season_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": 2020, "games": 227, "opener_accuracy": 0.4758, "close_accuracy": 0.4713},
            {"season": 2021, "games": 265, "opener_accuracy": 0.5321, "close_accuracy": 0.5170},
            {"season": 2022, "games": 262, "opener_accuracy": 0.5267, "close_accuracy": 0.5153},
            {"season": 2023, "games": 261, "opener_accuracy": 0.5326, "close_accuracy": 0.5211},
            {"season": 2024, "games": 259, "opener_accuracy": 0.5290, "close_accuracy": 0.5097},
            {"season": 2025, "games": 263, "opener_accuracy": 0.5361, "close_accuracy": 0.5285},
        ]
    )


def _write_opener_evaluation(artifacts: Path, run_id: str, *, feature_sha: str) -> Path:
    """The pool-relevant grade: metrics, season table, and its own provenance.

    ``feature_sha`` is what the run recorded for the feature table it read.
    When it differs from the active manifest's, ``project_state`` must report
    the grade as inconsistent rather than quietly showing it as current.
    """

    directory = artifacts / "opener_evaluation" / run_id
    _write_parquet(
        pd.DataFrame(
            {
                "game_id": ["2024_01_A_B", "2024_01_C_D", "2024_02_A_C"],
                "season": [2024, 2024, 2024],
                "week": [1, 1, 2],
                "correct_at_open": [1.0, 0.0, 1.0],
                "correct_at_close": [1.0, 0.0, 0.0],
                "oracle_correct_at_open": [1.0, 1.0, 1.0],
                "open_move": [0.5, -1.0, 0.25],
            }
        ),
        directory / "per_game.parquet",
    )
    _write_csv(_opener_season_summary(), directory / "season_summary.csv")
    uncertainty = [
        {
            "metric": metric,
            "block": block,
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
        }
        for block in ("week", "season")
        for metric, estimate, lower, upper in (
            ("opener_accuracy", 0.5250, 0.5019, 0.5432),
            ("close_accuracy", 0.5112, 0.4901, 0.5309),
        )
    ]
    _write_csv(pd.DataFrame(uncertainty), directory / "uncertainty.csv")
    _write_json(
        {
            "created_at_utc": "2030-08-13T01:00:00+00:00",
            "command": "opener-evaluation",
            "games": 1537,
            "mean_absolute_open_to_close_move": 1.02,
            "metrics": {
                "opener_accuracy": 0.5250,
                "close_accuracy": 0.5112,
                "opener_minus_close": 0.0138,
                "opener_vs_coin_flip": 0.0250,
                "movement_oracle_accuracy": 0.5510,
            },
            "uncertainty": uncertainty,
            "provenance": {
                "configuration_sha256": "c" * 64,
                "feature_table": {"sha256": feature_sha},
            },
        },
        directory / "metadata.json",
    )
    return directory


def _write_clv_ledger(artifacts: Path, run_id: str) -> Path:
    directory = artifacts / "clv_ledger" / run_id
    _write_parquet(
        pd.DataFrame(
            {
                "game_id": [
                    "2030_01_NE_SEA",
                    STRONG_LEAN_GAME,
                    "2030_01_ATL_PIT",
                    "2030_01_MIA_BUF",
                ],
                "season": [2030, 2030, 2030, 2030],
                "week": [1, 1, 1, 1],
                "pick_side": ["AWAY", "HOME", "HOME", "HOME"],
                "bet_side": ["PASS", "AWAY", "PASS", "PASS"],
                "clv_points": [0.5, 1.0, -0.5, float("nan")],
                "bet_clv_points": [float("nan"), 1.0, float("nan"), float("nan")],
                "clv_status": ["scored", "scored", "scored", "pending"],
                "close_source": ["live", "live", "live", "none"],
            }
        ),
        directory / "scored_decisions.parquet",
    )
    return directory


def _write_close_predictions(artifacts: Path, run_id: str) -> Path:
    """A predicted close for exactly one game, exercising per-game feature detection."""

    directory = artifacts / "close_predictions" / run_id
    _write_parquet(
        pd.DataFrame(
            {
                "game_id": [STRONG_LEAN_GAME],
                "predicted_close_home_spread": [-2.5],
                "model_id": ["mkt06-test"],
            }
        ),
        directory / "predictions.parquet",
    )
    return directory


@dataclass(frozen=True)
class _DecompFamily:
    family: str
    margin_share: float
    spread_share: float
    classification: str
    weight_in_spread: float
    refit_std_in_spread: float


# One family per classification bucket, with a refit_std_in_spread /
# weight_in_spread ratio deliberately on each side of
# model_explanation.STABILITY_JUMPY_RATIO (0.15).
DECOMP_FAMILIES: tuple[_DecompFamily, ...] = (
    _DecompFamily("results", 0.30, 0.28, "priced", 7.0, 0.35),  # ratio 0.05 -> steady
    _DecompFamily("player_continuity", 0.22, 0.02, "unpriced_predictive", 0.30, 0.10),  # 0.33
    _DecompFamily("elo", 0.05, 0.15, "overpriced", 2.0, 0.10),  # ratio 0.05 -> steady
    _DecompFamily("context", 0.02, 0.01, "noise", 0.05, 0.005),  # ratio 0.10 -> steady
)


def _write_attribution(
    artifacts: Path, run_id: str, *, feature_sha: str = ACTIVE_FEATURE_SHA
) -> Path:
    """A market-decomposition run: per-game attribution (strong-lean game only)
    plus the family-weight/classification tables and metadata the
    model-explanation page reads.

    ``feature_sha`` is what this run recorded for the feature table it read;
    a value other than ``ACTIVE_FEATURE_SHA`` exercises the same
    active-model-pin staleness path :func:`_write_opener_evaluation` does.
    """

    directory = artifacts / "market_decomposition" / run_id
    explanation = (
        "The model leans LA 4.1 points more than the market mainly because of recent "
        "offensive performance (+3.2)."
    )
    _write_parquet(
        pd.DataFrame(
            [
                {
                    "game_id": STRONG_LEAN_GAME,
                    "season": 2030,
                    "week": 1,
                    "family": family,
                    "contribution": contribution,
                    "predicted_residual": 4.1,
                    "explanation": explanation,
                }
                for family, contribution in (("offense", 3.2), ("defense", 0.9))
            ]
        ),
        directory / "attribution.parquet",
    )
    _write_csv(
        pd.DataFrame(
            [
                {
                    "family": row.family,
                    "weight_in_margin": row.margin_share * 50.0,
                    "margin_share": row.margin_share,
                    "net_signed_in_margin": 0.5,
                    "refit_std_in_margin": row.refit_std_in_spread,
                    "season_std_in_margin": row.refit_std_in_spread,
                    "weight_in_spread": row.weight_in_spread,
                    "spread_share": row.spread_share,
                    "net_signed_in_spread": 0.2,
                    "refit_std_in_spread": row.refit_std_in_spread,
                    "season_std_in_spread": row.refit_std_in_spread,
                    "weight_in_residual": row.margin_share * 40.0,
                    "residual_share": row.margin_share,
                    "net_signed_in_residual": 0.1,
                    "refit_std_in_residual": row.refit_std_in_spread,
                    "season_std_in_residual": row.refit_std_in_spread,
                    "classification": row.classification,
                }
                for row in DECOMP_FAMILIES
            ]
        ),
        directory / "classification.csv",
    )
    _write_csv(
        pd.DataFrame(
            [
                {
                    "target": target,
                    "family": row.family,
                    "mean_abs_weight": (
                        row.weight_in_spread if target == "spread" else row.margin_share * 50.0
                    ),
                    "refit_std_abs_weight": row.refit_std_in_spread,
                    "mean_signed_weight": 0.2,
                    "refits": 141,
                    "season_std_abs_weight": row.refit_std_in_spread,
                    "seasons": 8,
                    "share": row.spread_share if target == "spread" else row.margin_share,
                }
                for target in ("margin", "spread", "residual")
                for row in DECOMP_FAMILIES
            ]
        ),
        directory / "family_weights.csv",
    )
    _write_json(
        {
            "command": "market-decomposition",
            "created_at_utc": "2030-08-11T12:00:00+00:00",
            "feature_profile": "player",
            "provenance": {"feature_table": {"sha256": feature_sha}},
        },
        directory / "metadata.json",
    )
    return directory


def _write_data_tree(data_root: Path) -> None:
    """A raw nflverse snapshot and the feature-table manifest built from it.

    Only the manifests are read by the dashboard (``data_summary`` describes the
    snapshot from ``manifest.json``), so the parquet payloads stay tiny.
    """

    snapshot = data_root / "raw" / SNAPSHOT_ID
    _write_parquet(pd.DataFrame({"game_id": ["2030_01_NE_SEA"]}), snapshot / "schedules.parquet")
    _write_parquet(pd.DataFrame({"team": ["SEA"]}), snapshot / "team_stats.parquet")
    _write_json(
        {
            "snapshot_id": SNAPSHOT_ID,
            "fetched_at_utc": "2030-08-10T06:00:00+00:00",
            "seasons": [2030],
            "files": {
                "schedules.parquet": {"rows": 6924, "sha256": "d" * 64},
                "team_stats.parquet": {"rows": 13848, "sha256": "e" * 64},
            },
        },
        snapshot / "manifest.json",
    )
    _write_json(
        {
            "built_at_utc": "2030-08-10T06:30:00+00:00",
            "source_snapshot": SNAPSHOT_ID,
            "rows": 6924,
            "postseason_rows": 285,
            "completed_non_push_rows": 6510,
            "first_season": 2009,
            "last_season": 2030,
        },
        data_root / "processed" / "game_features.manifest.json",
    )


def _write_market_snapshots(data_root: Path) -> None:
    """Two live odds captures, in the archive layout every capture command writes.

    NOTE: ``picks.py`` currently reads ``data/odds`` rather than this
    ``data/market/raw`` root, so the opener lookup it drives is not reached from
    this tree. The fixture stays at the real location on purpose -- pointing it
    at ``data/odds`` would encode the mismatch as if it were the contract.
    """

    for offset_hours, spread in ((48, -3.5), (24, -3.0)):
        observed_at = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=offset_hours)
        directory = data_root / "market" / "raw" / observed_at.strftime("%Y%m%dT%H%M%SZ")
        _write_parquet(
            pd.DataFrame(
                {
                    "observed_at_utc": [observed_at] * 2,
                    "nflverse_game_id": ["2030_01_NE_SEA", STRONG_LEAN_GAME],
                    "bookmaker_key": ["draftkings", "draftkings"],
                    "market": ["spreads", "spreads"],
                    "outcome_side": ["HOME", "HOME"],
                    "home_spread_line": [spread, -3.5],
                    "capture_kind": ["live", "live"],
                }
            ),
            directory / "quotes.parquet",
        )


def _write_playoff_forecast(artifacts: Path, run_id: str) -> Path:
    """A single-game postseason card -- the model has no playoff feature-row
    coverage yet (FND-15), but the dashboard must still degrade honestly if a
    ``game_type != REG`` card ever exists on disk."""

    directory = artifacts / "margin_predictions" / run_id
    recommendations = pd.DataFrame(
        {
            "game_id": ["2030_19_KC_BUF"],
            "season": [2030],
            "week": [19],
            "game_type": ["WC"],
            "gameday": ["2031-01-11"],
            "weekday": ["Saturday"],
            "gametime": ["16:30:00"],
            "kickoff": ["2031-01-11T21:30:00Z"],
            "away_team": ["KC"],
            "home_team": ["BUF"],
            "spread_line": [-2.5],
            "total_line": [48.5],
            "fair_spread": [-2.0],
            "predicted_market_residual": [0.5],
            "home_cover_probability": [0.53],
            "edge": [0.03],
            "bet_side": ["PASS"],
            "method": ["market_residual"],
        }
    )
    _write_csv(recommendations, directory / "predictions.csv")
    _write_json(
        {
            "season": 2030,
            "week": 19,
            "feature_profile": "player",
            "ats_method": "market_residual",
            "created_at_utc": "2031-01-08T12:00:00+00:00",
        },
        directory / "metadata.json",
    )
    return directory


@pytest.fixture
def playoff_env(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal tree whose only forecast is a playoff week -- no REG card at all."""

    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    _write_playoff_forecast(artifacts_root, "playoff-20310108T120000Z")
    return data_root, artifacts_root


def _tuesday_before(
    now: pd.Timestamp, *, min_days_back: int = 2, max_days_back: int = 9
) -> pd.Timestamp:
    """A UTC Tuesday at least ``min_days_back`` days before ``now``.

    Deterministic regardless of which real-world weekday the suite happens to
    run on, and comfortably inside ``list_recent_market_snapshots``'s 10-day
    recency window (``max_days_back`` stays under 10).
    """

    for days_back in range(min_days_back, max_days_back + 1):
        candidate = (now - pd.Timedelta(days=days_back)).normalize() + pd.Timedelta(hours=15)
        if candidate.weekday() == 1:
            return candidate
    raise AssertionError("no Tuesday found in the lookback window")


# (game_id, kickoff, Tuesday-open home spread, latest home spread). Deliberately
# NOT NE_SEA/STRONG_LEAN_GAME: those two are the games ``_write_market_snapshots``
# above already covers, in a lighter shape (no ``provider_event_id`` /
# ``commence_time_utc``) built only for ``game_opening_lines``. Reusing them here
# would let that fixture's own randomly-Tuesday-observed row collide with these.
_WORKBENCH_QUOTE_GAMES: tuple[tuple[str, str, float, float], ...] = (
    ("2030_01_ATL_PIT", FUTURE_KICKOFFS[2], -3.0, -2.5),
    ("2030_01_MIA_BUF", FUTURE_KICKOFFS[3], -2.5, -3.5),
)


def _write_workbench_market_snapshots(data_root: Path) -> None:
    """A Tuesday-captured opener and a later quote, for the workbench's
    "what changed since Tuesday open" table.

    Written with the real ``nfl_ats.market_data`` quote shape --
    ``provider_event_id`` and ``commence_time_utc`` included -- which
    ``tuesday_opener_quotes``/``spread_consensus`` need but the lighter
    ``_write_market_snapshots`` fixture above does not carry.
    """

    now = pd.Timestamp.now(tz="UTC")
    tuesday = _tuesday_before(now)
    later = tuesday + pd.Timedelta(hours=20)

    def _quotes(observed_at: pd.Timestamp, *, is_tuesday: bool) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "observed_at_utc": observed_at,
                    "provider_event_id": f"event-{game_id}",
                    "commence_time_utc": pd.Timestamp(kickoff),
                    "nflverse_game_id": game_id,
                    "bookmaker_key": "draftkings",
                    "market": "spreads",
                    "outcome_side": "HOME",
                    "home_spread_line": tuesday_spread if is_tuesday else latest_spread,
                    "capture_kind": "live",
                }
                for game_id, kickoff, tuesday_spread, latest_spread in _WORKBENCH_QUOTE_GAMES
            ]
        )

    for observed_at, is_tuesday in ((tuesday, True), (later, False)):
        directory = data_root / "market" / "raw" / observed_at.strftime("%Y%m%dT%H%M%SZ")
        _write_parquet(_quotes(observed_at, is_tuesday=is_tuesday), directory / "quotes.parquet")


def _write_pool_levers(artifacts: Path) -> None:
    """A small, real-shaped POL-05 field-size simulation (``scripts/pool_levers.py``).

    Carries just the fields ``workbench.py`` reads -- enough of
    ``accuracy_scaling``, ``edge_value`` and ``best_pick`` to exercise the
    chart and the three live-computed narrative numbers -- rather than the
    full 40,000-sample-per-cell artifact.
    """

    payload = {
        "baseline_accuracy": 0.525,
        "season_games": 285,
        "best_pick_weeks": 18,
        "accuracy_scaling": [
            {"accuracy": 0.50, "entrants": 100.0, "probability_first": 0.0119},
            {"accuracy": 0.525, "entrants": 100.0, "probability_first": 0.0656},
            {"accuracy": 0.545, "entrants": 100.0, "probability_first": 0.1834},
        ],
        "edge_value": [
            {"entrants": 5.0, "entry": "coin_flip", "probability_first": 0.1710},
            {"entrants": 5.0, "entry": "model", "probability_first": 0.3942},
            {"entrants": 1000.0, "entry": "coin_flip", "probability_first": 0.00106},
            {"entrants": 1000.0, "entry": "model", "probability_first": 0.01346},
        ],
        "best_pick": [
            {
                "ranker_effect": "none",
                "best_pick_bonus": 1.0,
                "entrants": 100.0,
                "strategy": "arbitrary_nomination",
                "probability_first": 0.07131,
            },
            {
                "ranker_effect": "tie_break_agnostic_+0.9pts",
                "best_pick_bonus": 1.0,
                "entrants": 100.0,
                "strategy": "ranker_nominates_the_high_game",
                "probability_first": 0.07360,
            },
        ],
    }
    _write_json(payload, artifacts / "pool_levers" / "levers.json")


# The forced pool pick per game on the standard forecast fixture (derived the
# same way workbench.py derives it: home_cover_probability >= 0.5 -> HOME).
_FORCED_SIDES: dict[str, str] = {
    "2030_01_NE_SEA": "AWAY",
    STRONG_LEAN_GAME: "HOME",
    "2030_01_ATL_PIT": "HOME",
    "2030_01_MIA_BUF": "HOME",
}


def _paper_decision_rows(*, mismatched_game_id: str | None = None) -> pd.DataFrame:
    """One paper-decision row per game on the standard forecast fixture.

    Matches the live forecast's forced picks by default; pass
    ``mismatched_game_id`` to flip that one game's recorded side, simulating
    the model having changed after the lock.
    """

    sides = dict(_FORCED_SIDES)
    if mismatched_game_id is not None:
        sides[mismatched_game_id] = "AWAY" if sides[mismatched_game_id] == "HOME" else "HOME"
    game_ids = list(sides)
    count = len(game_ids)
    return pd.DataFrame(
        {
            "recorded_at_utc": ["2030-09-06T12:00:00+00:00"] * count,
            "forecast_artifact": [FORECAST_RUN] * count,
            "forecast_created_at_utc": ["2030-08-12T21:15:33+00:00"] * count,
            "model_id": ["d41d8cd98f00b204"] * count,
            "method": ["market_residual"] * count,
            "game_id": game_ids,
            "season": [2030] * count,
            "week": [1] * count,
            "kickoff": FUTURE_KICKOFFS,
            "away_team": ["NE", "SF", "ATL", "MIA"],
            "home_team": ["SEA", "LA", "PIT", "BUF"],
            "pick_side": [sides[game_id] for game_id in game_ids],
            "bet_side": [sides[game_id] for game_id in game_ids],
            "decision_home_spread": [3.5, -3.5, -3.0, -2.5],
            "edge": [-0.015, 0.116, 0.012, 0.045],
            "is_best_pick": [False, False, False, False],
        }
    )


def _write_paper_decisions(artifacts: Path, rows: pd.DataFrame) -> None:
    """The append-only paper-decision ledger workbench.py reads live.

    Lives at a fixed top-level path (``clv_ledger/decisions.parquet``), never
    inside a run-id directory -- distinct from the *scored* CLV ledger
    ``_write_clv_ledger`` writes at ``clv_ledger/<run_id>/scored_decisions.parquet``.
    """

    _write_parquet(rows, artifacts / "clv_ledger" / "decisions.parquet")


def _build_tree(
    tmp_path: Path,
    *,
    opener_feature_sha: str,
    decomposition_feature_sha: str = ACTIVE_FEATURE_SHA,
    paper_decisions: pd.DataFrame | None = None,
    include_pool_levers: bool = True,
    include_workbench_market_snapshots: bool = True,
) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    forecast = _write_weekly_forecast(artifacts_root, FORECAST_RUN)
    _write_line_sweep(forecast)
    evaluation = _write_margins_evaluation(artifacts_root, EVALUATION_RUN)
    _write_active_model(
        artifacts_root,
        evaluation_relative=evaluation.relative_to(artifacts_root).as_posix(),
        forecast_relative=forecast.relative_to(artifacts_root).as_posix(),
    )
    _write_opener_evaluation(artifacts_root, OPENER_RUN, feature_sha=opener_feature_sha)
    _write_clv_ledger(artifacts_root, LEDGER_RUN)
    _write_close_predictions(artifacts_root, CLOSE_RUN)
    _write_attribution(artifacts_root, DECOMP_RUN, feature_sha=decomposition_feature_sha)
    _write_data_tree(data_root)
    _write_market_snapshots(data_root)
    if include_workbench_market_snapshots:
        _write_workbench_market_snapshots(data_root)
    if paper_decisions is not None:
        _write_paper_decisions(artifacts_root, paper_decisions)
    if include_pool_levers:
        _write_pool_levers(artifacts_root)
    return data_root, artifacts_root


@pytest.fixture
def populated_env(tmp_path: Path) -> tuple[Path, Path]:
    """A dashboard data/artifacts tree with realistic content for every page.

    No paper decisions are recorded -- matching the real repo's current state
    (the ledger file is absent) -- so this fixture exercises the workbench's
    "not yet recorded" submission status, not a locked one.
    """

    return _build_tree(tmp_path, opener_feature_sha=ACTIVE_FEATURE_SHA)


@pytest.fixture
def locked_matching_env(tmp_path: Path) -> tuple[Path, Path]:
    """``populated_env`` with this week locked and matching the live forecast."""

    return _build_tree(
        tmp_path, opener_feature_sha=ACTIVE_FEATURE_SHA, paper_decisions=_paper_decision_rows()
    )


@pytest.fixture
def locked_mismatched_env(tmp_path: Path) -> tuple[Path, Path]:
    """``populated_env`` with a locked pick that no longer matches the live forecast."""

    return _build_tree(
        tmp_path,
        opener_feature_sha=ACTIVE_FEATURE_SHA,
        paper_decisions=_paper_decision_rows(mismatched_game_id="2030_01_NE_SEA"),
    )


@pytest.fixture
def no_pool_levers_env(tmp_path: Path) -> tuple[Path, Path]:
    """``populated_env`` without a saved pool-format simulation."""

    return _build_tree(tmp_path, opener_feature_sha=ACTIVE_FEATURE_SHA, include_pool_levers=False)


@pytest.fixture
def no_market_movement_env(tmp_path: Path) -> tuple[Path, Path]:
    """``populated_env`` without the Tuesday-shaped market snapshots the
    workbench's "what changed" table needs (the legacy ``_write_market_snapshots``
    quotes lack the columns that table's functions require, so this fixture
    deterministically has no Tuesday-open-vs-latest data to pair)."""

    return _build_tree(
        tmp_path, opener_feature_sha=ACTIVE_FEATURE_SHA, include_workbench_market_snapshots=False
    )


@pytest.fixture
def stale_opener_env(tmp_path: Path) -> tuple[Path, Path]:
    """``populated_env`` with the opening-line grade measured on earlier inputs."""

    return _build_tree(tmp_path, opener_feature_sha=EARLIER_FEATURE_SHA)


@pytest.fixture
def stale_decomposition_env(tmp_path: Path) -> tuple[Path, Path]:
    """``populated_env`` with the model-explanation breakdown measured on earlier inputs."""

    return _build_tree(
        tmp_path,
        opener_feature_sha=ACTIVE_FEATURE_SHA,
        decomposition_feature_sha=EARLIER_FEATURE_SHA,
    )


@pytest.fixture
def empty_env(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    return data_root, artifacts_root


@pytest.fixture
def point_dashboard_at(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point ``data_root``/``artifacts_root`` at a fixture tree for this test."""

    def apply(env: tuple[Path, Path]) -> None:
        data_root, artifacts_root = env
        monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
        monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))

    return apply


# ---------------------------------------------------------------------------
# state.py -- the single anchor
# ---------------------------------------------------------------------------


def test_project_state_without_artifacts_names_the_missing_model(
    empty_env: tuple[Path, Path], point_dashboard_at: Any
) -> None:
    point_dashboard_at(empty_env)
    state = project_state()

    assert state.active is None
    assert state.forecast is None
    assert state.synchronized is False
    assert state.opener_evaluation.directory is None
    assert state.clv_ledger.directory is None
    assert state.close_predictions.directory is None
    assert state.market_decomposition.directory is None
    assert len(state.issues) == 1
    assert "No synchronized active model" in state.issues[0]
    assert "nfl-ats margin-predict" in state.issues[0]


def test_project_state_on_a_synchronized_tree_reports_no_issues(
    populated_env: tuple[Path, Path], point_dashboard_at: Any
) -> None:
    point_dashboard_at(populated_env)
    state = project_state()

    assert state.active is not None
    assert state.forecast is not None
    assert state.forecast.is_active
    assert state.synchronized is True
    assert state.issues == ()
    assert state.opener_evaluation.consistent_with_active is True
    assert state.opener_evaluation.directory is not None
    assert state.opener_evaluation.directory.name == OPENER_RUN
    assert state.clv_ledger.directory is not None
    assert state.close_predictions.directory is not None
    assert state.market_decomposition.consistent_with_active is True
    assert state.market_decomposition.directory is not None
    assert state.market_decomposition.directory.name == DECOMP_RUN
    # The card the picks page renders is the one the manifest declares.
    assert state.forecast.directory.name == FORECAST_RUN
    assert len(state.forecast.recommendations) == 4


def test_project_state_flags_a_decomposition_grade_from_an_earlier_feature_build(
    stale_decomposition_env: tuple[Path, Path], point_dashboard_at: Any
) -> None:
    point_dashboard_at(stale_decomposition_env)
    state = project_state()

    # The picks/model pair is still sound -- only the supporting breakdown is behind.
    assert state.synchronized is True
    assert state.market_decomposition.consistent_with_active is False
    assert len(state.issues) == 1
    issue = state.issues[0]
    assert "earlier build" in issue
    assert "nfl-ats market-decomposition" in issue


def test_project_state_flags_an_opener_grade_from_an_earlier_feature_build(
    stale_opener_env: tuple[Path, Path], point_dashboard_at: Any
) -> None:
    point_dashboard_at(stale_opener_env)
    state = project_state()

    # The picks/model pair is still sound -- only the supporting grade is behind.
    assert state.synchronized is True
    assert state.opener_evaluation.consistent_with_active is False
    assert len(state.issues) == 1
    issue = state.issues[0]
    assert "earlier build" in issue
    assert "nfl-ats opener-evaluation" in issue


def test_load_parquet_is_keyed_on_modification_time_not_a_ttl(tmp_path: Path) -> None:
    path = tmp_path / "sweep.parquet"
    pd.DataFrame({"pick_probability": [0.51]}).to_parquet(path, index=False)
    assert load_parquet(path)["pick_probability"].tolist() == [0.51]

    pd.DataFrame({"pick_probability": [0.62]}).to_parquet(path, index=False)
    # Force a distinct mtime: a same-second rewrite must not be able to serve a
    # stale cache entry, and the filesystem's own clock is too coarse to trust.
    stamp = path.stat().st_mtime + 10.0
    os.utime(path, (stamp, stamp))
    assert load_parquet(path)["pick_probability"].tolist() == [0.62]


def test_load_csv_is_keyed_on_modification_time_too(tmp_path: Path) -> None:
    # The track record's season chart reads season_summary.csv through this, so
    # a re-run grade has to be visible on the next rerun.
    path = tmp_path / "season_summary.csv"
    pd.DataFrame({"opener_accuracy": [0.521]}).to_csv(path, index=False)
    assert load_csv(path)["opener_accuracy"].tolist() == [0.521]

    pd.DataFrame({"opener_accuracy": [0.525]}).to_csv(path, index=False)
    stamp = path.stat().st_mtime + 10.0
    os.utime(path, (stamp, stamp))
    assert load_csv(path)["opener_accuracy"].tolist() == [0.525]


def test_loaders_return_empty_frames_for_missing_files(tmp_path: Path) -> None:
    assert load_parquet(tmp_path / "absent.parquet").empty
    assert load_csv(tmp_path / "absent.csv").empty


# ---------------------------------------------------------------------------
# AppTest harness
# ---------------------------------------------------------------------------


def _run_page(page: str, env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Run one page through the real app shell so ``st.navigation`` is live.

    Page scripts are registered by ``app.py``'s ``st.navigation`` call; booting
    the shell and switching pages is how Streamlit actually serves them. The
    working directory is pinned to the repository root because AppTest resolves
    the script and its page paths relative to it.
    """

    data_root, artifacts_root = env
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))
    monkeypatch.chdir(REPO_ROOT)
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=60)
    if page != "picks.py":
        app.switch_page(f"app_pages/{page}")
        app.run(timeout=60)
    return app


def _page_html(app: AppTest) -> str:
    """Every ``st.html`` block the page emitted, concatenated."""

    # AppTest.get() is typed as Element | Block; every html element carries .body.
    return "".join(str(getattr(element, "body", "")) for element in app.get("html"))


def _t(text: str) -> str:
    """A plain-English string as it appears in page HTML (apostrophes escaped)."""

    return escape(text)


def _pick_cards(body: str) -> list[str]:
    """One HTML segment per pick card, split on each card's matchup heading."""

    return body.split('<h3 class="title" style="font-size:19px;">')[1:]


def _matchup(segment: str) -> str:
    return segment.split("</h3>", 1)[0]


# ---------------------------------------------------------------------------
# Page: this week's picks
# ---------------------------------------------------------------------------


def test_picks_page_renders_a_card_per_game_with_its_forced_pick(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("picks.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert _t("This week's picks") in body
    assert "2030 · Week 1 · 4 games" in body

    # One card per game, in kickoff order.
    cards = _pick_cards(body)
    assert [_matchup(card) for card in cards] == [
        "NE at SEA",
        STRONG_LEAN_MATCHUP,
        "ATL at PIT",
        "MIA at BUF",
    ]
    # The forced pick is the side the calibrated probability favors, and the
    # team's name is on the card -- not just a probability.
    picked = re.findall(r'Our pick</p><div class="hero num"[^>]*>([A-Z]+)</div>', body)
    assert picked == ["NE", "LA", "PIT", "BUF"]
    assert "Synchronized with the active model" in body
    assert "1 strong lean<" in body


def test_picks_page_explains_the_market_only_on_the_strong_lean(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("picks.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert body.count("What we think the market is missing") == 1

    explained = [
        card for card in _pick_cards(body) if "What we think the market is missing" in card
    ]
    assert [_matchup(card) for card in explained] == [STRONG_LEAN_MATCHUP]
    assert "We make this line 4.1 points different from the market" in explained[0]
    assert "recent offensive performance" in explained[0]

    # Every other card says plainly that we are with the market.
    quiet = [card for card in _pick_cards(body) if card not in explained]
    assert len(quiet) == 3
    for card in quiet:
        # Literal page copy, not an interpolated value -- not escaped.
        assert "We land close to the market's number here" in card


def test_picks_page_sweep_ships_its_table_view_for_the_active_method_only(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("picks.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert body.count("<summary>View as table</summary>") == 4
    assert "Confidence if the line moves (four points either side)" in body

    # The saved sweep stacks two methods; the card must filter to its own, or
    # every table-view twin doubles up.
    for card in _pick_cards(body):
        assert card.count("<tr><td>") == len(SWEEP_OFFSETS)
        assert 'class="ats-sweep"' in card
        assert "data-points=" in card
    assert 'id="sweep-2030_01_SF_LA"' in body


def test_picks_page_line_journey_and_predicted_close_are_feature_detected(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("picks.py", populated_env, monkeypatch)
    assert not app.exception

    cards = {_matchup(card): card for card in _pick_cards(_page_html(app))}
    # Only one game has a predicted close saved.
    assert "close guess" in cards[STRONG_LEAN_MATCHUP]
    assert "close guess" not in cards["NE at SEA"]
    for card in cards.values():
        assert "our number" in card  # fair_spread needs no optional artifact


def test_picks_page_without_artifacts_renders_its_empty_state(
    empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("picks.py", empty_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "No pick card yet" in body
    assert "Nothing has been generated since the last kickoff" in body
    assert "nfl-ats margin-predict" in body
    assert '<h3 class="title"' not in body  # no pick cards at all


# ---------------------------------------------------------------------------
# Page: what we've learned
# ---------------------------------------------------------------------------


def _findings_in(verdict: str) -> tuple[Any, ...]:
    return tuple(finding for finding in FINDINGS if finding.verdict == verdict)


def test_findings_page_renders_every_finding_under_its_verdict(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("findings.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert len(FINDINGS) >= 20
    assert body.count(f"<summary>{escape(DETAIL_SUMMARY_LABEL)}</summary>") == len(FINDINGS)
    for finding in FINDINGS:
        assert escape(finding.question) in body

    assert len(GROUPS) == 4
    for group in GROUPS:
        assert escape(group.title) in body
        assert f"{escape(group.kicker)} · {len(_findings_in(group.verdict))}" in body


def test_findings_page_is_static_content_and_needs_no_artifacts(
    empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("findings.py", empty_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert body.count(f"<summary>{escape(DETAIL_SUMMARY_LABEL)}</summary>") == len(FINDINGS)
    assert "How to read any number on this dashboard" in body


# ---------------------------------------------------------------------------
# Page: how the model decides
# ---------------------------------------------------------------------------

STRONG_LEAN_SENTENCE = (
    "The model leans LA 4.1 points more than the market mainly because of recent offensive "
    "performance (opponent-adjusted efficiency) (+3.2) and recent defensive performance "
    "(opponent-adjusted efficiency) (+0.9)."
)


def test_model_explanation_page_weighs_every_family_in_plain_english(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("model_explanation.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert _t("How the model decides") in body
    assert "What the model leans on, and what the market seems to miss" in body

    # Every family gets its plain-English family name, never the raw column name.
    assert "recent scoring margin and against-the-spread trend" in body  # results
    assert "lineup continuity" in body  # player_continuity
    assert "overall team strength ratings" in body  # elo
    assert "rest and schedule spots" in body  # context
    assert "player_continuity" not in body
    assert "spread_share" not in body

    # The four classification buckets, translated -- never the raw enum, and
    # "unpriced_predictive" reads as unconfirmed, never as a found edge.
    assert "the model leans on this and the market does not seem to price it -- unconfirmed" in (
        body
    )
    assert "the market leans on this more than reality rewards it" in body
    assert "market and reality roughly agree here" in body
    assert "neither says much about this" in body
    assert "unpriced_predictive" not in body
    assert "overpriced" not in body  # no raw enum leaks even inside its own caption

    # The shares themselves are direct-labelled.
    for pct in ("30%", "28%", "22%", "5%", "15%", "1%"):
        assert f">{pct}<" in body

    # Stability reads as a plain word, with the exact numbers off the headline
    # (available as a hover/secondary detail, not printed as a statistic).
    assert "steady across refits" in body
    assert "jumps around between refits" in body
    assert "Brier" not in body
    assert "bootstrap" not in body
    assert "confidence interval" not in body


def test_model_explanation_page_breaks_down_every_game_not_just_strong_leans(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("model_explanation.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert _t("This week's breakdown, game by game") in body
    # All four games appear, not just the one strong lean (contrast picks.py).
    for matchup in ("NE at SEA", STRONG_LEAN_MATCHUP, "ATL at PIT", "MIA at BUF"):
        assert matchup in body

    assert STRONG_LEAN_SENTENCE in body
    assert "supports the pick" in body
    assert "opposes the pick" in body
    assert "+3.2" in body
    assert "+0.9" in body

    # Games without a saved attribution row say so honestly rather than guessing.
    assert body.count("No per-family attribution saved for this game yet.") == 3


def test_model_explanation_page_says_a_stale_breakdown_out_loud(
    stale_decomposition_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("model_explanation.py", stale_decomposition_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert _t("earlier build of the model's inputs") in body
    assert "nfl-ats market-decomposition" in body
    # The measurement is not withdrawn, only qualified.
    assert "recent scoring margin and against-the-spread trend" in body
    assert ">30%<" in body


def test_model_explanation_page_without_artifacts_renders_its_empty_state(
    empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("model_explanation.py", empty_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "No model-explanation run saved yet" in body
    assert "nfl-ats market-decomposition" in body
    assert '<h3 class="title"' not in body  # no family or game cards at all


# ---------------------------------------------------------------------------
# Page: track record
# ---------------------------------------------------------------------------


def test_track_record_leads_with_the_opener_grade_and_charts_every_season(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("track_record.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert _t("Against the pool's line") in body
    assert "52.5%" in body  # metrics.opener_accuracy
    assert "51.1%" in body  # metrics.close_accuracy
    assert "+2.5 points vs. a coin flip" in body
    assert "1,537 games" in body

    assert 'aria-label="By season vs. coin flip"' in body
    for _, row in _opener_season_summary().iterrows():
        assert f"{row['opener_accuracy']:.1%}" in body
    # The losing season stays on the chart, with its explanation.
    assert "5 of the 6 seasons finished above the coin flip" in body
    assert "2020 at 47.6%" in body
    assert "COVID season" in body

    # Paper picks: 3 scored (2 of them positive), 1 pending.
    assert "The market moved our way" in body
    assert "67%" in body
    assert "Still waiting" in body


def test_track_record_says_a_stale_grade_out_loud_next_to_the_number(
    stale_opener_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("track_record.py", stale_opener_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert _t("This grade and this week's picks come from different builds") in body
    assert "opener-evaluation" in body
    # The measurement is not withdrawn, only qualified.
    assert "52.5%" in body


def test_track_record_without_artifacts_renders_its_empty_states(
    empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("track_record.py", empty_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "The pool grade has not been measured yet" in body
    assert "nfl-ats opener-evaluation" in body
    assert "No active model is linked yet" in body
    assert "No paper picks recorded yet" in body
    assert 'aria-label="By season' not in body


# ---------------------------------------------------------------------------
# Page: engine room
# ---------------------------------------------------------------------------


def test_engine_room_reports_synchronized_and_lists_artifact_currency(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("engine_room.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert '<div class="hero">Synchronized</div>' in body
    assert "Every page is reading the same pipeline state" in body
    assert "Nothing to re-run" in body
    assert "&#10007;" not in body  # every readiness check satisfied

    assert "Artifact currency" in body
    for name, run_id in (
        ("This week's pick card", FORECAST_RUN),
        ("Model evaluation", EVALUATION_RUN),
        ("Opening-line grade", OPENER_RUN),
        ("Paper-picks ledger", LEDGER_RUN),
        ("Predicted close", CLOSE_RUN),
        ("Model explanation", DECOMP_RUN),
    ):
        assert _t(name) in body
        assert run_id in body
    assert "not built yet" not in body

    # Data freshness, from the snapshot and feature-table manifests.
    assert SNAPSHOT_ID in body
    assert "Includes playoff rows: yes" in body
    assert "6,924" in body


def test_engine_room_warns_when_a_supporting_artifact_is_behind(
    stale_opener_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("engine_room.py", stale_opener_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    # The picks/model pair is sound, so the verdict stays "Synchronized" -- but
    # it must not read as "nothing to do".
    assert '<div class="hero">Synchronized</div>' in body
    assert "1 supporting artifact still needs a re-run" in body
    assert "Nothing to re-run" not in body
    assert "Out of sync" in body
    assert "&#10007; The opening-line grade matches it" in body


def test_engine_room_without_artifacts_reports_not_synchronized(
    empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("engine_room.py", empty_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert '<div class="hero">Not synchronized</div>' in body
    assert "Pages may be describing different builds" in body
    assert "No active model" in body
    assert "No complete raw snapshot" in body
    assert body.count("not built yet") == 6  # every artifact row, said explicitly


# ---------------------------------------------------------------------------
# Page: pool workbench
# ---------------------------------------------------------------------------


def test_workbench_page_shows_the_rules_and_an_unrecorded_submission_status(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert _t("Pool workbench") in body

    # populated_env carries no paper-decision ledger at all -- matching the
    # real repo's current state -- so the status must read as unrecorded,
    # never as a hardcoded claim about any particular week.
    assert "Not yet recorded" in body
    assert "first genuine write happens at the Tuesday lock" in body
    # "Locked picks" is the (always-present) stat-tile kicker; the actual
    # status word is what must be absent when nothing has been recorded.
    assert "Locked --" not in body

    # The rules card: static prose, sourced from docs/pool_edge_plan.md.
    assert "272 regular-season games" in body
    assert "13 playoff games" in body
    assert "one Best Pick each regular-season week" in body
    assert "Picks lock Tuesday at noon" in body

    # No invented Best Pick bonus number -- the honest placeholder, verbatim.
    assert "bonus size not configured" in body
    assert "the pool&#x27;s rules aren&#x27;t recorded here" in body or (
        "the pool's rules aren't recorded here" in body
    )


def test_workbench_page_locked_status_matches_the_live_forecast(
    locked_matching_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", locked_matching_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "Locked -- matches the live forecast" in body
    assert "All 4 picks recorded for this week match" in body
    assert "Not yet recorded" not in body
    assert "differs from the live forecast" not in body


def test_workbench_page_locked_status_flags_a_pick_that_no_longer_matches(
    locked_mismatched_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", locked_mismatched_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "Locked -- differs from the live forecast" in body
    assert "1 of 4 locked picks no longer match" in body
    assert "NE at SEA" in body
    assert "model changed after the lock" in body


def test_workbench_page_entries_table_ranks_by_confidence_and_flags_the_lean(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "Entries, ranked by confidence" in body
    assert "48.6% over 107 weeks" in body
    # Exactly one row carries the strong-lean badge -- the same game and the
    # same STRONG_LEAN_POINTS threshold picks.py's own card uses.
    assert body.count('<span class="chip model">strong lean</span>') == 1
    assert _t(STRONG_LEAN_MATCHUP) in body
    for matchup in ("NE at SEA", STRONG_LEAN_MATCHUP, "ATL at PIT", "MIA at BUF"):
        assert matchup in body


def test_workbench_page_best_pick_nomination_matches_the_confirmed_signal(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "BEST PICK OF THE WEEK" in body
    # sweep_robustness ranks 2030_01_SF_LA (STRONG_LEAN_GAME) highest and
    # unambiguously on this fixture (verified against nfl_ats.best_pick
    # directly): the home team, LA, is the forced pick.
    assert '<div class="hero num" style="font-size:26px;color:var(--good-text);">LA</div>' in body
    assert "budgeted at roughly +0.9 points, not the +8.7" in body


def test_workbench_page_market_movement_pairs_tuesday_open_with_the_latest_quote(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "What changed since Tuesday open" in body
    assert "this pool locks at the opener" in body
    # ATL at PIT: -3.0 -> -2.5 (+0.5); MIA at BUF: -2.5 -> -3.5 (-1.0).
    assert "ATL at PIT" in body
    assert "MIA at BUF" in body
    assert "PIT -3<" in body
    assert "PIT -2.5<" in body
    assert "BUF -2.5<" in body
    assert "BUF -3.5<" in body
    assert "<td>+0.5</td>" in body
    assert "<td>-1.0</td>" in body


def test_workbench_page_market_movement_is_an_honest_empty_state_without_a_pairing(
    no_market_movement_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", no_market_movement_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "Nothing has moved to compare yet" in body


def test_workbench_page_decision_levers_are_computed_live_from_the_artifact(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "Decision levers" in body
    assert "never real per-game ownership data" in body
    assert "simulation" in body

    # The chart: three accuracy tiers at 100 entrants, anchored at the
    # coin-flip row -- all read live from the fixture's levers.json, not
    # hardcoded page copy.
    assert 'aria-label="By season vs. at a coin flip"' in body
    assert ">50.0%</span>" in body
    assert ">52.5%</span>" in body
    assert ">54.5%</span>" in body

    # Every quantitative narrative claim is computed from the same artifact,
    # not typed in: 5-rival vs. 1,000-rival multiples from edge_value, the
    # two-accuracy-point delta from accuracy_scaling, and the honest Best
    # Pick ranker's own delta from best_pick -- all at entrants=100.
    assert "2.3&times; to 12.7&times;" in body
    assert "+11.8 percentage points" in body
    assert "<b>+0.23 points</b>" in body
    assert "multiplier to protect, not a lever to pull" in body
    assert "loses monotonically at every field size" in body
    # The bonus placeholder is disclosed a second time here, not just in the
    # rules card, since the ranker scenario above depends on it.
    assert body.count("bonus size not configured") == 2

    assert "pool-format simulation built" in body


def test_workbench_page_decision_levers_empty_state_without_the_artifact(
    no_pool_levers_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", no_pool_levers_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "No pool-format simulation saved yet" in body
    assert "scripts/pool_levers.py" in body
    assert "pool-format simulation built" not in body


def test_workbench_page_shows_an_honest_playoff_empty_state(
    playoff_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", playoff_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "No playoff model coverage yet" in body
    assert "No Best Pick in the postseason" in body
    assert "docs/postseason_support.md" in body


def test_workbench_page_without_artifacts_renders_its_empty_state(
    empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page("workbench.py", empty_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert "No pick card yet" in body
    assert "Nothing has been generated since the last kickoff" in body
    assert "nfl-ats margin-predict" in body


# ---------------------------------------------------------------------------
# Cross-page invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page", PAGES)
def test_every_page_runs_clean_on_an_empty_tree(
    page: str, empty_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page(page, empty_env, monkeypatch)
    assert not app.exception
    body = _page_html(app)
    assert body.strip(), f"{page} rendered nothing at all"
    assert '<div class="ats">' in body


@pytest.mark.parametrize("page", PAGES)
def test_every_page_ships_the_stylesheet_once_and_scopes_its_blocks(
    page: str, populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_page(page, populated_env, monkeypatch)
    assert not app.exception

    body = _page_html(app)
    assert body.count("color-scheme: light;") == 1  # the stylesheet, exactly once
    assert '<div class="ats">' in body
    # Streamlit strips inline handlers, so no component may rely on one.
    assert "onclick" not in body.lower()
    assert "onmousemove" not in body.lower()


def test_app_shell_boots_and_defaults_to_the_picks_page(
    populated_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, artifacts_root = populated_env
    monkeypatch.setenv("NFL_ATS_DATA_DIR", str(data_root))
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts_root))
    monkeypatch.chdir(REPO_ROOT)
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=60)

    assert not app.exception
    assert _t("This week's picks") in _page_html(app)
    captions = [caption.value for caption in app.caption]
    assert any("no wagering integration" in text for text in captions)
    assert any(str(artifacts_root.resolve()) in text for text in captions)
