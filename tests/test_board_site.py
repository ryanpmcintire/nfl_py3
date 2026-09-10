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

    for relative_path in site:
        assert "/" not in relative_path
        assert "\\" not in relative_path


def test_no_page_carries_a_skin_toggle_or_desk_reference(site: dict[str, str]) -> None:
    for relative_path, html in site.items():
        assert "skin-toggle" not in html, relative_path
        assert "Cover Desk" not in html, relative_path
        assert "ats-board-skin" not in html, relative_path


@pytest.mark.full
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

    for relative_path, html in site.items():
        atomic_text(html, tmp_path / relative_path)

    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "model.html").is_file()
    assert (tmp_path / "history.html").is_file()
    assert (tmp_path / "findings.html").is_file()
    assert not (tmp_path / "terminal").exists()
    assert not (tmp_path / "desk").exists()


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
        pass

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

    for relative_path, html in site.items():
        scanner = _OverflowStructureScanner()
        scanner.feed(html)
        assert not scanner.unwrapped_long_tokens, (
            f"{relative_path} has an unwrapped 40+ char token: {scanner.unwrapped_long_tokens}"
        )
