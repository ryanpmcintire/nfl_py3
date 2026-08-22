"""Static invariants for the toggleable Gridiron Observatory skin pack.

The skin must be inert by default (nothing outside ``.theme-obs`` scopes),
define every token it references, and keep the shipped toggle script inside
the public-board safety guards (no inline handlers, idempotent, bounded size).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nfl_ats import site_theme

CSS = site_theme.OBSERVATORY_CSS_PATH.read_text(encoding="utf-8")
JS = site_theme.TOGGLE_JS_PATH.read_text(encoding="utf-8")

TOKEN_DECLARATION = re.compile(r"--([a-z0-9-]+)\s*:")
TOKEN_REFERENCE = re.compile(r"var\(--([a-z0-9-]+)")


def _top_level_blocks(css: str) -> list[tuple[str, str]]:
    """(selector, body) for each top-level rule; @media bodies stay raw."""

    blocks: list[tuple[str, str]] = []
    selector: str | None = None
    buffer: list[str] = []
    depth = 0
    for char in css:
        if char == "{":
            if depth == 0:
                selector = "".join(buffer).strip()
                buffer = []
            else:
                buffer.append(char)
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                assert selector is not None
                blocks.append((selector, "".join(buffer)))
                buffer = []
                selector = None
            else:
                buffer.append(char)
        else:
            buffer.append(char)
    assert depth == 0 and not "".join(buffer).strip(), "unbalanced braces in observatory.css"
    return blocks


def _block(css: str, needle: str) -> str:
    matches = [body for selector, body in _top_level_blocks(css) if selector == needle]
    assert len(matches) == 1, f"expected exactly one block matching {needle!r}"
    return matches[0]


def test_both_mode_scopes_exist_and_carry_the_full_palette() -> None:
    night = _block(CSS, "body.theme-obs .ats")
    day = _block(CSS, 'body.theme-obs[data-mode="day"] .ats')
    palette = [
        "surface",
        "plane",
        "surface-raised",
        "turf-a",
        "turf-b",
        "chalk",
        "chalk-dim",
        "chalk-faint",
        "ink",
        "ink-2",
        "muted",
        "grid",
        "baseline",
        "border",
        "series-model",
        "series-market",
        "series-third",
        "seq-100",
        "seq-400",
        "seq-700",
        "div-neg",
        "div-pos",
        "good",
        "good-text",
        "warning",
        "serious",
        "critical",
        "bulb-core",
        "bulb-glow-rgb",
        "field-grass",
        "field-line",
        "field-hash",
        "accent-flag",
        "stub-home",
        "stub-away",
        "paper",
        "paper-ink",
        "paper-muted",
        "shadow-card",
    ]
    night_tokens = set(TOKEN_DECLARATION.findall(night))
    day_tokens = set(TOKEN_DECLARATION.findall(day))
    assert set(palette) <= night_tokens
    assert set(palette) <= day_tokens
    assert "color-scheme: dark" in night
    assert "color-scheme: light" in day


def test_every_referenced_token_is_defined_in_a_theme_obs_scope() -> None:
    scoped_css = "\n".join(
        body for selector, body in _top_level_blocks(CSS) if ".theme-obs" in selector
    )
    referenced = set(TOKEN_REFERENCE.findall(scoped_css))
    defined = set(TOKEN_DECLARATION.findall(scoped_css))
    assert referenced, "the skin references no role tokens at all"
    assert referenced <= defined, f"undefined tokens: {sorted(referenced - defined)}"


def test_everything_except_the_toggle_button_is_scoped_under_theme_obs() -> None:
    unscoped_allowlist = {"#theme-toggle-mount", ".theme-toggle-button"}
    for selector, _body in _top_level_blocks(CSS):
        if selector.startswith("@media"):
            continue
        assert ".theme-obs" in selector or any(
            selector.startswith(allowed) for allowed in unscoped_allowlist
        ), selector


def test_signature_elements_ship_post_revision_styles() -> None:
    for primitive in (
        ".ticket-stub",
        ".ticket-torn",
        ".marginalia::before",
        ".fieldstrip-range",
        ".fieldstrip-marker",
        ".fieldstrip-flag::after",
        ".bulb-dot",
        ".chalkable",
    ):
        assert f".theme-obs .ats {primitive}" in CSS, primitive
    track = _block(CSS, "body.theme-obs .ats .fieldstrip-range")
    assert "border-radius: 8px" in track
    assert "color-mix(in srgb, var(--series-model)" in track


def test_toggle_button_has_no_inline_handlers_and_bounded_size() -> None:
    assert len(JS.splitlines()) <= 60
    lowered = JS.lower()
    assert "onclick" not in lowered
    assert "onload" not in lowered
    assert "<script" not in lowered
    assert "addeventlistener(" in lowered
    assert site_theme.STORAGE_KEY in JS
    assert "__atsThemeToggleLoaded" in JS


def test_toggle_script_touches_nothing_but_the_documented_hooks() -> None:
    for forbidden in ("stApp", "__atsThemeInterval", "fetch(", "XMLHttpRequest"):
        assert forbidden not in JS
    assert 'getElementById("theme-toggle-mount")' in JS
    assert '"theme-toggle"' in JS
    assert 'classList.toggle("theme-obs"' in JS
    assert 'setAttribute("data-mode", "day")' in JS


def test_head_snippet_carries_the_required_integration_hooks() -> None:
    head = site_theme.render_theme_toggle_head()
    assert '<link rel="stylesheet" href="./observatory.css">' in head
    assert '<script src="./toggle.js" defer></script>' in head
    assert f'<div id="{site_theme.TOGGLE_MOUNT_ID}"' in head
    assert "onclick" not in head.lower()
    custom = site_theme.render_theme_toggle_head(asset_prefix="assets/theme")
    assert 'href="assets/theme/observatory.css"' in custom
    assert 'src="assets/theme/toggle.js"' in custom


@pytest.mark.parametrize("asset", [site_theme.OBSERVATORY_CSS_PATH, site_theme.TOGGLE_JS_PATH])
def test_asset_paths_resolve_inside_the_package(asset: Path) -> None:
    assert asset.is_file()
    assert asset.parent == site_theme.PACKAGE_DIR
