"""Tests for :mod:`nfl_ats.board_site` -- the real site builder.

This is the function ``cli._write_public_site`` calls for the actual publish
path (``nfl-ats publish-board`` / ``publish-predictions --with-board``), so
these tests exercise its CONTRACT (the exact set of pages it returns, the
flat site layout, parameter pass-through) against real repo artifacts -- the
same integration-level coverage ``tests/test_board_terminal.py`` already
uses for the individual pages.

2026-08-31 owner redirect: the Cover Desk skin (and its ``terminal/``/
``desk/`` directory split, top-level redirect page, and header skin-toggle)
is dropped entirely. ``build_two_skin_site`` is renamed ``build_site`` and
now returns exactly four bare, site-root filenames including History.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from nfl_ats import board_site, board_terminal
from nfl_ats.board_site import build_site
from nfl_ats.board_site_content import SiteContent
from nfl_ats.io import atomic_text

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site(_shared_real_site_content: SiteContent) -> dict[str, str]:
    """The real, unchanged ``build_site()`` still runs end-to-end here (this
    is the actual publish-path function, and ``build_site`` itself is a thin
    ``load_site_content`` + four ``board_terminal.render*`` calls -- see
    its docstring) -- only its internal ``load_site_content`` call is
    patched to return the session-shared, already-loaded content instead of
    re-reading real repo artifacts a second time (WP51, test-suite speed;
    identical args to ``tests/conftest.py::_shared_real_site_content``:
    ``_REPO_ROOT / "artifacts"``, ``require_fresh_arrest_overlay=False``).
    ``test_build_site_passes_require_fresh_arrest_overlay_through`` below
    separately covers ``build_site``'s real kwarg pass-through to
    ``load_site_content`` with its own lightweight fake."""

    with patch.object(board_site, "load_site_content", return_value=_shared_real_site_content):
        return build_site(
            _REPO_ROOT / "artifacts",
            require_fresh_arrest_overlay=False,
        )


def test_site_has_exactly_the_expected_pages(site: dict[str, str]) -> None:
    assert set(site.keys()) == {"index.html", "model.html", "history.html", "findings.html"}


def test_site_pages_match_board_terminal_registry(site: dict[str, str]) -> None:
    expected = {filename for filename, _label, _title in board_terminal.SITE_PAGES}
    assert set(site.keys()) == expected


def test_every_page_is_a_complete_html_document(site: dict[str, str]) -> None:
    for relative_path, html in site.items():
        assert html.startswith("<!doctype html>"), f"{relative_path} missing doctype"
        assert "<title>" in html, f"{relative_path} missing a title"


def test_no_page_is_nested_in_a_skin_subdirectory(site: dict[str, str]) -> None:
    """Regression guard: every page is a bare filename at the site root --
    no ``terminal/`` or ``desk/`` prefix from the retired two-skin layout."""

    for relative_path in site:
        assert "/" not in relative_path
        assert "\\" not in relative_path


def test_no_page_carries_a_skin_toggle_or_desk_reference(site: dict[str, str]) -> None:
    for relative_path, html in site.items():
        assert "skin-toggle" not in html, relative_path
        assert "Cover Desk" not in html, relative_path
        assert "ats-board-skin" not in html, relative_path


@pytest.mark.full  # ENG-11: triggers the real-artifact site-content build (dominates --durations)
def test_nav_is_the_same_four_pages_on_every_page(site: dict[str, str]) -> None:
    for filename, html in site.items():
        for other_filename, _label, _title in board_terminal.SITE_PAGES:
            assert f'href="{other_filename}"' in html, (
                f"{filename} missing nav link to {other_filename}"
            )


def test_build_site_passes_require_fresh_arrest_overlay_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_load_site_content(artifacts_root: Path, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise ValueError("stop before doing real work -- pass-through is what's under test")

    monkeypatch.setattr(board_site, "load_site_content", _fake_load_site_content)
    with pytest.raises(ValueError):
        build_site(Path("unused"), require_fresh_arrest_overlay=True)
    assert captured["require_fresh_arrest_overlay"] is True


def test_writing_the_site_to_disk_creates_a_flat_directory(
    tmp_path: Path, site: dict[str, str]
) -> None:
    """Mirrors exactly what ``cli._write_public_site`` does with the dict
    ``build_site`` returns."""

    for relative_path, html in site.items():
        atomic_text(html, tmp_path / relative_path)

    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "model.html").is_file()
    assert (tmp_path / "history.html").is_file()
    assert (tmp_path / "findings.html").is_file()
    # No stale terminal/ or desk/ subdirectories from the retired two-skin
    # layout should ever be created by this function.
    assert not (tmp_path / "terminal").exists()
    assert not (tmp_path / "desk").exists()


# ---------------------------------------------------------------------------
# Mobile-width overflow fix (2026-08-31 390px-iframe audit): the owner
# reproduced a real document scrollWidth overflow on every one of the four
# pages (index.html's policy-note policy id, model.html's challenger-ledger
# evidence-pill registry keys, findings.html's trace chips / watching-lead
# channel names & artifact paths / signal-registry names). These two tests
# scan the REAL built pages (the ``site`` fixture above, real repo
# artifacts -- exactly what a publish would ship) rather than a hand-built
# fixture, since the bug was in real content shape, not the renderer's logic.
# Pragmatic, class-based checks, not a browser measurement -- see
# ``tests/test_board_terminal.py`` for the stylesheet-contract half of this
# fix, and the owner's own 390px-iframe re-measurement for ground truth.
# ---------------------------------------------------------------------------

#: Classes the appended "mobile-width overflow fix" CSS block gives
#: ``overflow-wrap:anywhere`` (see board_terminal_style.css and
#: test_board_terminal.py's ``test_mobile_overflow_fix_css_covers_every_
#: long_identifier_class``).
_OVERFLOW_WRAP_CLASSES = {
    "policy-note",
    "evidence-pill",
    "trace-chip",
    "chan",
    "chan-sub",
    "mono-id",
    "game-sub",
    "sub",
    "gen",
}
#: Classes whose element scrolls its own overflow rather than relying on
#: wrapping (the board table's own scroll container, and the ticker, which
#: clips via ``overflow:hidden``).
_OVERFLOW_CONTAINER_CLASSES = {"board-scroll", "ticker", "ticker-track"}
_LONG_UNBROKEN_TOKEN = re.compile(r"\S{40,}")
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class _OverflowStructureScanner(HTMLParser):
    """Walks a rendered page tracking the class-attribute ancestor chain.
    Records (a) every ``<table>`` with no ``board-scroll``-carrying ancestor
    and (b) every 40+ character unbroken text run outside script/style whose
    ancestor chain carries neither an overflow-wrap class nor an overflow
    container class."""

    def __init__(self) -> None:
        super().__init__()
        self._stack: list[set[str]] = []
        self._skip_depth = 0
        self.unwrapped_tables: list[str] = []
        self.unwrapped_long_tokens: list[str] = []

    def _ancestor_classes(self) -> set[str]:
        return {cls for frame in self._stack for cls in frame}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes: set[str] = set()
        for name, value in attrs:
            if name == "class" and value:
                classes.update(value.split())
        if tag == "table" and not (self._ancestor_classes() & _OVERFLOW_CONTAINER_CLASSES):
            self.unwrapped_tables.append(str(attrs))
        if tag in _VOID_TAGS:
            return
        self._stack.append(classes)
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        pass  # self-closed elements carry no text content of their own

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag not in _VOID_TAGS and self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        ancestors = self._ancestor_classes()
        if ancestors & _OVERFLOW_WRAP_CLASSES or ancestors & _OVERFLOW_CONTAINER_CLASSES:
            return
        for match in _LONG_UNBROKEN_TOKEN.finditer(data):
            self.unwrapped_long_tokens.append(match.group(0))


def test_every_table_is_wrapped_in_an_overflow_container(site: dict[str, str]) -> None:
    """Every ``<table>`` on every page sits inside a ``board-scroll``
    ancestor -- the container that gives it ``overflow-x:auto`` at every
    width the mobile board-collapse doesn't otherwise handle."""

    for relative_path, html in site.items():
        scanner = _OverflowStructureScanner()
        scanner.feed(html)
        assert not scanner.unwrapped_tables, (
            f"{relative_path} has a <table> with no board-scroll ancestor: "
            f"{scanner.unwrapped_tables}"
        )


def test_no_long_identifier_escapes_its_overflow_wrap_or_scroll_container(
    site: dict[str, str],
) -> None:
    """No emitted leaf text node of 40+ unbroken characters lives outside an
    element covered by the mobile-overflow-fix's overflow-wrap rule or an
    overflow-scrolling container, on any of the four real, live-artifact
    pages."""

    for relative_path, html in site.items():
        scanner = _OverflowStructureScanner()
        scanner.feed(html)
        assert not scanner.unwrapped_long_tokens, (
            f"{relative_path} has an unwrapped 40+ char token: {scanner.unwrapped_long_tokens}"
        )
