"""Static invariants for the shared design system (theme + viz), no host app.

Ported from the retired ``tests/test_dashboard.py`` when the internal
dashboard runtime was deleted: these pin the token contract and the
color-never-alone component guarantees that the public GitHub Pages site
inherits through :mod:`nfl_ats.public_board`.
"""

from __future__ import annotations

import json
import re

import pytest

from nfl_ats.dashboard import theme, viz


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
    assert "<td>-1.0</td><td>55.0%</td>" in html
    assert "<td>+1.0</td><td>62.0%</td>" in html
    assert "<td>0.0</td>" in html
    assert "<td>+0.0</td>" not in html


def test_sweep_curve_carries_its_geometry_in_data_attributes() -> None:
    # Hover wiring is delegated, so the script recomputes geometry from these
    # attributes -- they are the contract.
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
