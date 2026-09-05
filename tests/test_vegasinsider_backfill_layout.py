"""ENG-40: VegasInsider board cells that render spread/total in EITHER
vertical order (see ``scripts/backfill_vegasinsider.py::classify_line_tokens``
docstring).

Measured 2026-09-05: 155 rows across the 2005-2016 VegasInsider tidy archive
(``artifacts/vegasinsider_backfill/<run-id>/season_<year>.parquet``) carried
a POSITIVE, oversized ``spread_line`` -- a game TOTAL misfiled into the
spread column. Root cause, read directly from the cached board HTML
(``data/raw/vegasinsider/20260822T033952Z/snapshots/20091216095259.html``,
the Dallas @ New Orleans row, and ``.../20091026145946.html``): VegasInsider
renders a book cell's two lines (spread, total) in either vertical order --
the away team's row shows its own line only when it is the favorite, and
the underdog's row shows the total instead. When the total carries no o/u
price marker it renders as a bare "+"-signed number (e.g. ``"+54"``), which
used to satisfy the spread-token regex before the real (later, negative)
spread token was ever read.

The fix is a SIGN-CONVENTION rule (a "+"-prefixed token can never be this
archive's favorite-only spread_line, which is always <= 0), not a
magnitude/range filter -- see ``classify_line_tokens``'s own docstring.
These tests reproduce the bug at both the token-classification level and
the full HTML-parse level, using the real cell layout as rendered in the
cached source (verbatim token shapes measured from the DAL@NO and
DEN@BAL/SF@IND rows on 2026-09-05).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.backfill_vegasinsider as biv

# ---------------------------------------------------------------------------
# classify_line_tokens: token-level regression coverage
# ---------------------------------------------------------------------------


def test_total_first_plus_signed_no_price_is_classified_as_total_not_spread() -> None:
    """ENG-40's exact bug shape: measured verbatim from the DAL@NO book cell
    (capture 20091216095259, book anchor 'J') -- '+54<br />-7&nbsp;-110'
    tokenizes to ["+54", "-7 -110"]. Before the fix, "+54" satisfied the
    spread-token regex first and became spread_line=54.0 (positive,
    impossible under this archive's favorite-only convention); the real
    spread ("-7 -110") was then silently ignored because spread was already
    set."""

    spread, total = biv.classify_line_tokens(["+54", "-7 -110"])
    assert spread == -7.0
    assert total == 54.0


def test_spread_first_with_price_still_correct_and_total_now_recovered() -> None:
    """The other cell order (spread first, with its price glued on;
    measured verbatim from the IND@JAX cell: '-2.5-110<br />+41&frac12;' ->
    ["-2.5-110", "+41.5"]). Pre-fix, spread was already correct (-2.5) but
    the "+41.5" total token was silently dropped (total stayed None) because
    the old code only ever populated total via the OU/bare-number branches,
    never after a signed-token match. This is one of the 54 rows the ENG-40
    rebuild additionally recovered (total_line filled in, spread_line
    unchanged) -- see docs/vi_half_lines.md's ENG-40 note."""

    spread, total = biv.classify_line_tokens(["-2.5-110", "+41.5"])
    assert spread == -2.5
    assert total == 41.5


def test_bare_number_total_first_regression_unaffected() -> None:
    """The 'consensus' column (no book price shown at all) already worked
    before this fix, and must keep working identically: a bare (unsigned)
    total token followed by a signed spread token -- measured verbatim from
    the DAL@NO consensus cell: '54.5 | -7.5'."""

    spread, total = biv.classify_line_tokens(["54.5", "-7.5"])
    assert spread == -7.5
    assert total == 54.5


def test_bare_number_spread_first_regression_unaffected() -> None:
    """Consensus-cell order the other way round, measured verbatim from the
    IND@JAX consensus cell: '-6.5 | 46.5'."""

    spread, total = biv.classify_line_tokens(["-6.5", "46.5"])
    assert spread == -6.5
    assert total == 46.5


def test_pk_and_ou_marker_paths_unaffected() -> None:
    """PK (pick'em) and an explicit o/u-marked total must still classify
    exactly as before -- the fix only changes how a bare "+"-signed token is
    routed, nothing else in this function."""

    spread, total = biv.classify_line_tokens(["PK", "45u-110"])
    assert spread == 0.0
    assert total == 45.0


def test_positive_signed_token_never_becomes_spread_even_when_seen_first() -> None:
    """Direct assertion of the archive's own documented invariant
    (docs/vegasinsider_pilot.md: spread_line is a favorite-side quote,
    always <= 0): no sequence of tokens can make classify_line_tokens return
    a positive spread."""

    for tokens in (["+54", "-7 -110"], ["+41.5", "-6.5-110"], ["+.5", "-9.5 -110"]):
        spread, _total = biv.classify_line_tokens(tokens)
        assert spread is None or spread <= 0


# ---------------------------------------------------------------------------
# Full HTML fixture: the real "modern" board layout, one game row per cell
# order, mirroring the exact markup measured in the cached snapshot
# ---------------------------------------------------------------------------


def _game_row_html(
    *,
    row_class: str,
    away_rot: str,
    away_name: str,
    home_rot: str,
    home_name: str,
    kickoff: str,
    consensus_cell: str,
    book_cell: str,
    book_anchor: str,
) -> str:
    """One <tr> game row in the real board's markup shape (measured 2026-09-05
    from data/raw/vegasinsider/20260822T033952Z/snapshots/20091216095259.html
    and .../20091026145946.html): a nested team-info table followed by one
    oddsText <td> per column -- a "consensus" cell with no anchor/link, and
    one book cell whose value is wrapped in a line-movement anchor link
    (the real source for the anchor ANCHOR_RE reads off href="...#<code>")."""

    return f"""<tr class='{row_class}'>
<td valign="bottom" width="100%" style="position: relative;">
<table border="0" cellpadding="0" cellspacing="0" width="100%">
<tr><td height="13"><b>{kickoff}</b></td></tr>
<tr><td><b>{away_rot}&nbsp;<a href="/nfl/teams/team-page.cfm/team/x">{away_name}</a></b></td></tr>
<tr><td><b>{home_rot}&nbsp;<a href="/nfl/teams/team-page.cfm/team/y">{home_name}</a></b></td></tr>
</table>
</td>
<td width="58" class="oddsText_even" nowrap valign="bottom"
    align="center"><nobr>&nbsp;{consensus_cell}&nbsp;</nobr></td>
<td width="61" class="oddsText_even" nowrap valign="bottom"
    align="center"><nobr>&nbsp;<a
    href="/nfl/odds/las-vegas/line-movement/x-@-y.cfm/date/1-1-09/time/2020#{book_anchor}"
    target=_blank>{book_cell}</a></nobr></td>
</tr>"""


def _board_page_html(*rows: str) -> str:
    return f"""<html><body>
<table>
{"".join(rows)}
</table>
</body></html>"""


def test_board_parse_total_first_book_cell_is_not_misfiled_as_spread() -> None:
    """ENG-40's exact reported symptom at the full parse_board() level: a
    book cell rendered total-first ('+54<br />-7&nbsp;-110', the DAL@NO
    shape) must parse to a negative spread and a plausible total, never a
    positive/oversized spread_line."""

    html = _board_page_html(
        _game_row_html(
            row_class="oddsText_even",
            away_rot="303",
            away_name="Dallas",
            home_rot="304",
            home_name="New Orleans",
            kickoff="12/19  8:20 PM",
            consensus_cell="54&frac12;<br />-7.5&nbsp;",
            book_cell="+54<br />-7&nbsp;-110",
            book_anchor="J",
        )
    )
    parsed = biv.parse_board("20091216095259", html)
    assert parsed.error is None
    assert len(parsed.games) == 1
    game = parsed.games[0]
    assert game.away_name == "Dallas"
    assert game.home_name == "New Orleans"
    consensus, book = game.cells
    assert consensus.spread_line == -7.5
    assert consensus.total_line == 54.5
    assert book.spread_line == -7.0
    assert book.total_line == 54.0
    # The archive-wide invariant this bug violated (docs/vegasinsider_pilot.md):
    # spread_line is a favorite-side quote and must never be positive.
    for cell in game.cells:
        assert cell.spread_line is None or cell.spread_line <= 0


def test_board_parse_spread_first_book_cell_regression() -> None:
    """The other real order (IND@JAX shape: '-2.5-110<br />+41&frac12;') must
    keep parsing correctly end-to-end through parse_board(), including the
    total value this fix additionally recovers."""

    html = _board_page_html(
        _game_row_html(
            row_class="oddsText_odd",
            away_rot="301",
            away_name="Indianapolis",
            home_rot="302",
            home_name="Jacksonville",
            kickoff="12/17  8:20 PM",
            consensus_cell="-6.5<br />46&frac12;",
            book_cell="-2.5-110<br />+41&frac12;",
            book_anchor="J",
        )
    )
    parsed = biv.parse_board("20091217000000", html)
    assert parsed.error is None
    game = parsed.games[0]
    consensus, book = game.cells
    assert consensus.spread_line == -6.5
    assert consensus.total_line == 46.5
    assert book.spread_line == -2.5
    assert book.total_line == 41.5


def test_build_tidy_end_to_end_never_produces_a_positive_or_oversized_spread() -> None:
    """The same invariant, checked past build_tidy() (the function that
    actually writes season_<year>.parquet), using both real cell orders in
    one page -- the level closest to the original 155-row symptom."""

    html = _board_page_html(
        _game_row_html(
            row_class="oddsText_even",
            away_rot="303",
            away_name="Dallas",
            home_rot="304",
            home_name="New Orleans",
            kickoff="12/19  8:20 PM",
            consensus_cell="54&frac12;<br />-7.5&nbsp;",
            book_cell="+54<br />-7&nbsp;-110",
            book_anchor="J",
        ),
        _game_row_html(
            row_class="oddsText_odd",
            away_rot="301",
            away_name="Indianapolis",
            home_rot="302",
            home_name="Jacksonville",
            kickoff="12/17  8:20 PM",
            consensus_cell="-6.5<br />46&frac12;",
            book_cell="-2.5-110<br />+41&frac12;",
            book_anchor="J",
        ),
    )
    parsed = biv.parse_board("20091216095259", html)
    book_maps = {"20091216095259": {"J": "HILTON"}}
    tidy = biv.build_tidy([parsed], book_maps)
    # The unanchored consensus cell in each row is excluded by build_tidy
    # (no anchor and no book_name) -- only the two named-book cells (one per
    # game) survive into the tidy table.
    assert len(tidy) == 2
    assert (tidy["book"] == "HILTON").all()
    assert (tidy["spread_line"] > 0).sum() == 0
    assert (tidy["spread_line"].abs() > 30).sum() == 0
    # And the totals this fix recovers are present, not silently dropped.
    assert tidy["total_line"].notna().all()
    assert set(tidy["spread_line"]) == {-7.0, -2.5}
    assert set(tidy["total_line"]) == {54.0, 41.5}
