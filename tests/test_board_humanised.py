"""Render-contract test: no reader-visible surface may leak the research
machinery's own vocabulary.

Owner, verbatim, on a live panel (2026-09-05), reacting to the picks page's
policy-overlay panel showing a raw slug and hash: "whats the point of
showing this anywhere? ... remember when i said this is for humans not the
opus autist... theres lots of other shit like this on the dashboard...".
Also that day, on the same theme: "please do not let those percentages get
out of date anymore" (the number-provenance contract this suite's sibling
work wired into ``board_content.verify_number_provenance``).

This is the ONE shared scan: every rendered page (This Week, The Model,
History, What We've Learned) plus the published card
(``CURRENT_PREDICTIONS.md``'s own generator), checked for:

- a hex-looking token 8+ characters long (a fingerprint/hash/model id)
- a ``..._v1``/``..._v2`` style versioned slug
- a raw ``YYYYMMDDTHHMMSSZ`` artifact-directory stamp
- a raw ISO datetime (``YYYY-MM-DDTHH:MM``)
- the literal "P+" notation (the registry's own probability-positive
  shorthand -- plain-English "68% likely real" is required instead)
- the literal phrase "week-blocked" (a bootstrap-method name, not football)
- every phrase in :data:`nfl_ats.board_content.BANNED_BOILERPLATE` (the
  same constant :mod:`nfl_ats.card_explanation`'s own per-pick language
  contract already enforces on ``explain_pick``'s text -- shared, not
  duplicated, so the two checks can never quietly disagree)
- a bare snake_case identifier (a registry/policy/challenger slug)

Two narrow, DELIBERATE exemptions, matching the render-contract's own
stated carve-out for literal/technical text (``<script>``/``<style>``/
``<code>``, extended here to backtick-quoted spans, which render as plain
backtick characters rather than an HTML ``<code>`` tag but carry the exact
same "this is literal, not prose" meaning -- e.g. a CLI example like
`` `nfl-ats weak-signals pool --effect-units accuracy_points` ``, which
would be a WRONG example if "accuracy_points" were rewritten with a
space) and file-path citations (``docs/opener_evaluation.md``,
``scripts/overlay_subset_composition.py``) -- footnote-style references
to the codebase, not registry identifiers, and not something this project
could stop doing without gutting its own "label how you know it" culture.
Registry-sourced free-form research prose (the Watching Leads'/Signal
Registry's/Research-this-week's own methodology notes -- data this lane's
file list does not include, and would take a rewrite of
``registry/weak_signals.json`` itself to fully re-author) is marked up in
``<code>`` at the render layer for exactly this reason; this test does not
special-case it further, it relies on that markup.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_publishing import _publish_with_fresh_empty_arrest, _write_overlay_publication_fixture

from nfl_ats import board_terminal
from nfl_ats.board_content import BANNED_BOILERPLATE
from nfl_ats.board_site_content import SiteContent

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"(?<![a-z0-9])[0-9a-f]{8,}(?![a-z0-9])", re.IGNORECASE)
_VERSIONED_SLUG_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_v\d\b")
_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_SNAPSHOT_STAMP_RE = re.compile(r"\b\d{8}T\d{6}Z\b")
#: The raw ISO-8601 ``T`` separator specifically -- NOT the human
#: "YYYY-MM-DD HH:MM UTC" (space-separated) format this site's OWN "updated
#: .../Generated ..." footers use on purpose (``board_content
#: ._build_headline_stats``'s ``synced_at_text``,
#: ``board_site_content._generated_at_text``): that format is the fix, not
#: a violation of the render contract it is checked against.
_ISO_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_P_PLUS_RE = re.compile(r"\bP\+")
_WEEK_BLOCKED_RE = re.compile(r"week-blocked", re.IGNORECASE)

# Exempt blocks: script/style/code tags, and backtick-quoted spans (the
# markdown card's own "code span" convention -- see module docstring).
_EXEMPT_TAG_BLOCK_RE = re.compile(
    r"<(script|style|code)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_BACKTICK_SPAN_RE = re.compile(r"`[^`\n]+`")
_TAG_RE = re.compile(r"<[^>]+>")

# File-path citations (``docs/opener_evaluation.md``,
# ``scripts/overlay_subset_composition.py``, ``registry/weak_signals.json``)
# -- footnote-style references, not registry identifiers.
_FILE_PATH_RE = re.compile(
    r"\b(?:[a-zA-Z][\w-]*/)+[\w.-]+\.(?:md|py|json|csv|parquet|txt|yaml|yml|css)\b"
)


def _visible_text(markup: str) -> str:
    """Reduce HTML (or the plain-text markdown card) to the words a reader
    actually sees flow past: script/style/code blocks, backtick-quoted code
    spans, and file-path citations are removed FIRST (so a slug living only
    inside one of those never reaches the checks below); every remaining
    tag is then replaced with a single space (never concatenated bare --
    two adjacent table cells must not glue into one false hex/snake-case
    token) and HTML entities are left as-is (``escape()`` already turned
    every literal ``<``/``>``/``&`` in real content into entities, so a
    banned phrase hiding behind ``&amp;`` would still read as one word)."""

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
        # An 8+ digit run with no letters is virtually always two adjacent
        # numbers (a season, a game count) rather than a real fingerprint --
        # still worth a real assertion, not a silent pass, so only an
        # ALL-DIGIT token is forgiven.
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


# ---------------------------------------------------------------------------
# The four rendered pages
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def site_content(_shared_real_site_content: SiteContent) -> SiteContent:
    """Real repo artifacts, loaded once for the whole test session -- see
    ``tests/conftest.py::_shared_real_site_content`` and
    ``tests/test_board_terminal.py``'s identically-named fixture, which
    this one deliberately mirrors so both modules share the same cached
    object rather than paying the ~44-54s real-I/O cost twice."""

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


# ---------------------------------------------------------------------------
# The published card
# ---------------------------------------------------------------------------


def test_published_card_is_humanised(tmp_path: Path) -> None:
    """The same scan against ``publishing.py``'s own generated card text --
    reuses ``tests/test_publishing.py``'s overlay-composition fixture (not
    duplicated here) specifically because it is the ONE existing fixture
    that exercises ``_composition_note`` (the "Production policy active"
    paragraph the owner's complaint was literally about), so this is the
    branch of card-generation code most likely to regress."""

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
