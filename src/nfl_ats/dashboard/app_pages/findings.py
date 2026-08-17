"""What we've learned -- every finding in the project's record, in plain English.

Layout only. Every word on this page lives in
:mod:`nfl_ats.dashboard.findings_content`; this module turns that content into
the dashboard's design system (:mod:`nfl_ats.dashboard.theme` tokens,
:mod:`nfl_ats.dashboard.viz` components) and hands Streamlit the HTML.

Rendering contract (verified in ``board.py``): ``st.html`` inserts into the app
DOM rather than an iframe, so the stylesheet ships once at the top of the page
and every block is wrapped in ``.ats`` for scoping. Inline ``on*`` attributes
are stripped by Streamlit's sanitizer, so interactivity here is limited to
native ``<details>`` disclosure -- no JavaScript is needed beyond the theme sync
that closes the page.

One layout quirk worth knowing: the stylesheet gives ``.card + .card`` a top
margin, which is right for stacked cards and wrong inside a flex row. Cards laid
out side by side are therefore each wrapped in their own grid cell, so no two
``.card`` elements are ever siblings.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import streamlit as st

from nfl_ats.dashboard import theme, viz
from nfl_ats.dashboard.findings_content import (
    CLOSING_NOTE,
    DETAIL_SUMMARY_LABEL,
    GROUPS,
    HERO_KICKER,
    HERO_PARAGRAPHS,
    HERO_SUB,
    HERO_TILES,
    HERO_TITLE,
    HONESTY_KICKER,
    HONESTY_RULES,
    HONESTY_SUB,
    HONESTY_TITLE,
    LEGEND_KICKER,
    SOURCE_LABEL,
    Finding,
    VerdictGroup,
    findings_for,
)

# ---------------------------------------------------------------------------
# Small layout helpers
# ---------------------------------------------------------------------------


def _block(inner: str) -> str:
    """One ``.ats``-scoped HTML block, ready for ``st.html``."""

    return f'<div class="ats">{inner}</div>'


def _rows(cards: Sequence[str], *, per_row: int = 2) -> str:
    """Lay cards out ``per_row`` across, wrapping on narrow screens.

    Each card gets its own grid cell so the stylesheet's ``.card + .card``
    stacking margin never fires between side-by-side cards.
    """

    chunks = [cards[index : index + per_row] for index in range(0, len(cards), per_row)]
    return "".join(
        '<div class="row" style="margin-bottom:14px;">'
        + "".join(f'<div style="display:grid;">{card}</div>' for card in chunk)
        + "</div>"
        for chunk in chunks
    )


def _section_header(kicker: str, title: str, sub: str, *, top: int = 34) -> str:
    return (
        f'<div style="margin:{top}px 0 16px;max-width:70ch;">'
        f'<p class="kicker">{escape(kicker)}</p>'
        f'<h3 class="title" style="font-size:22px;margin-bottom:6px;">{escape(title)}</h3>'
        f'<p class="sub">{escape(sub)}</p></div>'
    )


def _verdict_chip(group: VerdictGroup) -> str:
    """The verdict badge: icon + label for state, a muted pill otherwise."""

    if group.chip_kind in {"good", "warning"}:
        return viz.status_line(group.chip_kind, group.chip_label)
    dot = (
        '<span class="dot" style="background:var(--muted);"></span>'
        if group.chip_kind == "muted"
        else ""
    )
    return f'<span class="chip">{dot}{escape(group.chip_label)}</span>'


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def _hero() -> str:
    tiles = _rows(
        [
            viz.stat_tile(
                tile.kicker,
                tile.value,
                tile.context,
                delta_text=tile.delta_text,
                delta_good=tile.delta_good,
            )
            for tile in HERO_TILES
        ],
        per_row=3,
    )
    story = viz.card(
        '<div class="prose">'
        + "".join(f"<p>{escape(paragraph)}</p>" for paragraph in HERO_PARAGRAPHS)
        + "</div>",
        accent=True,
    )
    legend_items = "".join(
        f'<div><div style="margin-bottom:6px;">{_verdict_chip(group)}</div>'
        f'<p class="fine">{escape(group.legend)}</p></div>'
        for group in GROUPS
    )
    legend = viz.card(
        f'<p class="kicker">{escape(LEGEND_KICKER)}</p>'
        f'<div class="row" style="margin-top:4px;">{legend_items}</div>'
    )
    return (
        viz.page_header(HERO_KICKER, HERO_TITLE, HERO_SUB)
        + tiles
        + f'<div style="margin-bottom:14px;">{story}</div>'
        + legend
    )


def _finding_card(finding: Finding, group: VerdictGroup) -> str:
    inner = (
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        'gap:14px;margin-bottom:10px;">'
        f'<p class="title" style="max-width:38ch;">{escape(finding.question)}</p>'
        f'<span style="flex:none;">{_verdict_chip(group)}</span>'
        "</div>"
        f'<div class="prose"><p>{escape(finding.plain_answer)}</p></div>'
        '<details class="table-view"><summary>'
        f"{escape(DETAIL_SUMMARY_LABEL)}</summary>"
        f'<p class="fine" style="margin:10px 0 0;max-width:68ch;">{escape(finding.detail)}</p>'
        f'<p class="fine" style="margin:8px 0 0;">{escape(SOURCE_LABEL)}: '
        f'<span style="color:var(--ink-2);">{escape(finding.source)}</span></p>'
        "</details>"
    )
    return viz.card(inner, accent=group.verdict == "helps")


def _group_section(group: VerdictGroup) -> str:
    findings = findings_for(group.verdict)
    header = _section_header(
        f"{group.kicker} · {len(findings)}",
        group.title,
        group.blurb,
    )
    return header + _rows([_finding_card(finding, group) for finding in findings])


def _honesty() -> str:
    rules = _rows(
        [
            viz.card(
                f'<p class="title" style="margin-bottom:8px;">{escape(rule.title)}</p>'
                f'<div class="prose"><p>{escape(rule.body)}</p></div>'
            )
            for rule in HONESTY_RULES
        ]
    )
    closing = f'<p class="fine" style="max-width:68ch;margin-top:4px;">{escape(CLOSING_NOTE)}</p>'
    return _section_header(HONESTY_KICKER, HONESTY_TITLE, HONESTY_SUB, top=42) + rules + closing


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

st.html(theme.stylesheet() + _block(_hero()), unsafe_allow_javascript=True)

for verdict_group in GROUPS:
    st.html(_block(_group_section(verdict_group)), unsafe_allow_javascript=True)

st.html(_block(_honesty()) + theme.theme_sync_script(), unsafe_allow_javascript=True)
