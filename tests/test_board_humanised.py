from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_publishing import _publish_with_fresh_empty_arrest, _write_overlay_publication_fixture

from nfl_ats import board_terminal
from nfl_ats.board_content import BANNED_BOILERPLATE
from nfl_ats.board_site_content import PLAIN_SUMMARY_PENDING, SiteContent

_HEX_RE = re.compile(r"(?<![a-z0-9])[0-9a-f]{8,}(?![a-z0-9])", re.IGNORECASE)
_VERSIONED_SLUG_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_v\d\b")
_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_SNAPSHOT_STAMP_RE = re.compile(r"\b\d{8}T\d{6}Z\b")
_ISO_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_P_PLUS_RE = re.compile(r"\bP\+")
_WEEK_BLOCKED_RE = re.compile(r"week-blocked", re.IGNORECASE)

_EXEMPT_TAG_BLOCK_RE = re.compile(
    r"<(script|style|code)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_BACKTICK_SPAN_RE = re.compile(r"`[^`\n]+`")
_TAG_RE = re.compile(r"<[^>]+>")

_FILE_PATH_RE = re.compile(
    r"\b(?:[a-zA-Z][\w-]*/)+[\w.-]+\.(?:md|py|json|csv|parquet|txt|yaml|yml|css)\b"
)


def _visible_text(markup: str) -> str:

    text = _EXEMPT_TAG_BLOCK_RE.sub(" ", markup)
    text = _BACKTICK_SPAN_RE.sub(" ", text)
    text = _FILE_PATH_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return text


def _assert_humanised(label: str, markup: str) -> None:
    text = _visible_text(markup)
    lowered = text.lower()

    for phrase in BANNED_BOILERPLATE:
        assert phrase not in lowered, f"{label}: banned boilerplate phrase {phrase!r}"

    hex_hit = _HEX_RE.search(text)
    if hex_hit is not None:
        assert hex_hit.group(0).isdigit(), (
            f"{label}: hex-looking token {hex_hit.group(0)!r} (fingerprint/model id "
            "leaked into reader text)"
        )

    slug_hit = _VERSIONED_SLUG_RE.search(text)
    assert slug_hit is None, f"{label}: versioned slug {slug_hit.group(0)!r}"

    stamp_hit = _SNAPSHOT_STAMP_RE.search(text)
    assert stamp_hit is None, f"{label}: raw artifact stamp {stamp_hit.group(0)!r}"

    iso_hit = _ISO_TIMESTAMP_RE.search(text)
    assert iso_hit is None, f"{label}: raw ISO timestamp {iso_hit.group(0)!r}"

    pplus_hit = _P_PLUS_RE.search(text)
    assert pplus_hit is None, f'{label}: literal "P+" notation'

    week_blocked_hit = _WEEK_BLOCKED_RE.search(text)
    assert week_blocked_hit is None, f'{label}: literal "week-blocked" phrase'

    snake_hit = _SNAKE_CASE_RE.search(text)
    assert snake_hit is None, f"{label}: bare snake_case identifier {snake_hit.group(0)!r}"


@pytest.fixture(scope="module")
def site_content(_shared_real_site_content: SiteContent) -> SiteContent:

    return _shared_real_site_content


def test_this_week_page_is_humanised(site_content: SiteContent) -> None:
    html = board_terminal.render(site_content.board)
    _assert_humanised("index.html (This Week)", html)


def test_model_page_is_humanised(site_content: SiteContent) -> None:
    html = board_terminal.render_model_page(site_content.model)
    _assert_humanised("model.html (The Model)", html)


def test_history_page_is_humanised(site_content: SiteContent) -> None:
    html = board_terminal.render_history_page(site_content.history)
    _assert_humanised("history.html (History)", html)


def test_findings_page_is_humanised(site_content: SiteContent) -> None:
    html = board_terminal.render_findings_page(site_content.findings)
    _assert_humanised("findings.html (What We've Learned)", html)


_MISSING_PLAIN_SUMMARY_HINT = (
    "-- run .tools\\uv.exe run --no-sync python scripts\\backfill_plain_summaries.py "
    "--missing-plain-summary to see the current backlog, then record a --plain-summary "
    "via `nfl-ats weak-signals record --replace` (every other field unchanged)"
)


def test_watching_leads_have_no_plain_summary_backlog(site_content: SiteContent) -> None:

    for lead in site_content.findings.watching_leads:
        assert lead.description != PLAIN_SUMMARY_PENDING, (
            f"watching lead {lead.name!r} has no plain_summary {_MISSING_PLAIN_SUMMARY_HINT}"
        )


def test_signal_registry_notable_rows_have_no_plain_summary_backlog(
    site_content: SiteContent,
) -> None:

    for row in site_content.findings.ledger_summary.notable:
        assert row.idea != PLAIN_SUMMARY_PENDING, (
            f"signal registry row {row.name!r} has no plain_summary {_MISSING_PLAIN_SUMMARY_HINT}"
        )


def test_recent_activity_weak_signal_entries_have_no_plain_summary_backlog() -> None:

    from nfl_ats.findings_registry import (
        STORE_WEAK_SIGNAL,
        load_rotation_registry,
        load_weak_signal_registry,
        recent_registry_activity,
    )

    registry = load_weak_signal_registry()
    rotation_registry = load_rotation_registry()
    activity = recent_registry_activity(registry, rotation_registry, datetime.now(UTC))
    missing = [
        entry.key
        for _category, entries in activity.entries_by_category
        for entry in entries
        if entry.store == STORE_WEAK_SIGNAL and not entry.plain_summary
    ]
    assert not missing, (
        f"'Research this week' weak-signal row(s) with no plain_summary: {missing!r} "
        f"{_MISSING_PLAIN_SUMMARY_HINT}"
    )


_ASSISTANT_DATA_RE = re.compile(
    r'<script type="application/json" class="assistant-data">(.*?)</script>', re.DOTALL
)


def _assistant_watching_bodies(markup: str) -> list[str]:

    match = _ASSISTANT_DATA_RE.search(markup)
    if match is None:
        return []
    payload = json.loads(match.group(1))
    return [
        str(entry.get("body", ""))
        for entry in payload.get("entries", [])
        if isinstance(entry, dict) and str(entry.get("id", "")).startswith("watching:")
    ]


def test_this_week_page_assistant_watching_answers_are_humanised(
    site_content: SiteContent,
) -> None:
    bodies = _assistant_watching_bodies(board_terminal.render(site_content.board))
    for body in bodies:
        _assert_humanised("index.html assistant watching answer", body)


def test_findings_page_assistant_watching_answers_are_humanised(
    site_content: SiteContent,
) -> None:
    bodies = _assistant_watching_bodies(board_terminal.render_findings_page(site_content.findings))
    assert bodies, "fixture regression: findings.html has no watching-lead assistant entries"
    for body in bodies:
        _assert_humanised("findings.html assistant watching answer", body)


def test_published_card_is_humanised(tmp_path: Path) -> None:

    _, readme, data_root = _write_overlay_publication_fixture(tmp_path)
    destination = tmp_path / "CURRENT_PREDICTIONS.md"

    _publish_with_fresh_empty_arrest(
        tmp_path,
        destination=destination,
        readme_path=readme,
        data_root=data_root,
        published_at=datetime(2026, 9, 8, 16, 0, tzinfo=UTC),
    )

    card = destination.read_text(encoding="utf-8")
    assert "**Production policy active:**" in card, (
        "fixture regression: the composition-note branch this test exists to cover did not fire"
    )
    _assert_humanised("CURRENT_PREDICTIONS.md", card)
