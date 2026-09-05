"""The "ATS Terminal" design: a pure renderer over :mod:`nfl_ats.board_content`
and :mod:`nfl_ats.board_site_content`.

This module owns STRUCTURE and STYLE only -- the mockup's DOM, its class
names, and its stylesheet, transplanted verbatim (see ``board_terminal_style
.css``, loaded below as a module constant). It must never contain a content
literal: no number, no sentence, no policy id. Every fact rendered here is a
field read off a content dataclass -- built once, upstream, by
:mod:`nfl_ats.board_content` / :mod:`nfl_ats.board_site_content`. A future
number change (a new experiment, a refreshed interval, updated findings)
touches those modules exactly once and this page picks it up automatically;
see ``tests/test_board_content_coverage.py`` for the coverage test that
guarantees it.

Site (2026-09-02): exactly FOUR pages, at the site root (no
skin subdirectory, no toggle -- the Cover Desk skin was dropped entirely).
``index.html`` (This Week) is the approved mockup, restructured 2026-09-05
into UI-20 layout A ("board + inspector", owner: "layout A is definitely
the best. lets go with that.") -- see :func:`render`'s docstring for the
two-column shape and :func:`_inspector_section`'s docstring for why the
old standalone "Why this pick" tab-strip section is gone (its content, the
old per-game deep dive -- attribution, cover curve, the folded-in
"spread explorer" line-offset adjuster, and the projected lineups -- now
lives in that column, selected by the board's own rows instead of a
second, redundant selector). ``model.html`` (The Model) and
``findings.html`` (What We've Learned) are original extensions of the same
visual system -- see each ``render_*_page`` function's docstring for what
it merges and why.

The only markup here that is NOT part of the approved mockup is (a) the
small degraded-state blocks the mockup's own CSS sheet reserves space for
(delimited in ``board_terminal_style.css`` with a
``/* degraded states -- ... */`` comment), (b) the game-selector/adjuster
markup in the inspector, plus its own inline script, (c) the
retrieval-only board-assistant panel (UI-16, :mod:`nfl_ats.board_assistant`),
plus its own inline script, and (d) the ``.week-grid`` two-column layout
(2026-09-05, layout A) -- none changes the mockup's own DOM elsewhere, all
are additive.
"""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from itertools import groupby
from pathlib import Path

from nfl_ats import board_assistant
from nfl_ats.board_content import (
    CADENCE_NOTE,
    SOURCE_POLICY_COMPUTED_LIVE_NOTE,
    BoardContent,
    GameDive,
    GameRow,
    HeadlineStats,
    LinkPreview,
    SourcePolicyView,
    TickerChrome,
    TiebreakerView,
)
from nfl_ats.board_site_content import (
    ChallengerAssessment,
    FamilyWeightRow,
    FindingItemView,
    FindingsPageContent,
    HistoryPageContent,
    HistoryPickRow,
    HistoryWeekGrade,
    LedgerEvidenceItem,
    ModelLedgerRowView,
    ModelPageContent,
    RecentActivityCategoryView,
    RecentActivityView,
    SeasonGradeRow,
    SignalNotableRow,
    VerdictGroupView,
    WatchingLeadView,
)
from nfl_ats.lineup_view import TeamLineup
from nfl_ats.public_board import humanize_identifier
from nfl_ats.spread_explorer import SPREAD_EXPLORER_STEP

_STYLE_PATH = Path(__file__).with_name("board_terminal_style.css")
TERMINAL_STYLE_CSS = _STYLE_PATH.read_text(encoding="utf-8")

_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700'
    "&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700;800"
    '&display=swap" rel="stylesheet">'
)

# ---------------------------------------------------------------------------
# Site page registry -- four pages, at the site root. Deliberately a LOCAL
# constant, never ``nfl_ats.public_board.SITE_PAGES``: the public-board
# legacy Track Record entry is retired, while this site's nav is the current
# four-page shape.
# ---------------------------------------------------------------------------

PICKS_PAGE = "index.html"
MODEL_PAGE = "model.html"
FINDINGS_PAGE = "findings.html"
HISTORY_PAGE = "history.html"

#: (file name, nav label, browser title) in nav order.
SITE_PAGES: tuple[tuple[str, str, str], ...] = (
    (PICKS_PAGE, "This week", "This week's picks"),
    (MODEL_PAGE, "The model", "The model"),
    (HISTORY_PAGE, "History", "History"),
    (FINDINGS_PAGE, "What we've learned", "What we've learned"),
)

#: Shared script for the This Week page's per-game selector (layout A,
#: 2026-09-05: the board's own rows, not a separate tab strip -- see
#: :func:`_inspector_section`'s docstring) and its line-offset adjuster.
#: One click handler for every ``table.board tr.game`` row plus one
#: drag/click handler for every ``.ats-adjuster`` widget on the page,
#: mirroring the erf approximation
#: ``nfl_ats.spread_explorer.widget_home_cover_probability`` mirrors in
#: Python (kept in lock-step by that module's own tests) -- the formula is
#: never invented here, and every widget's ``center``/``mean``/``std``/
#: ``card-line`` data attribute is a guard-proven field off a
#: :class:`~nfl_ats.board_content.SpreadAdjusterParams`, never a literal.
_DIVE_SCRIPT = """
<script>
(function () {
  function erf(x) {
    var sign = x < 0 ? -1 : 1; x = Math.abs(x);
    var a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    var t = 1 / (1 + p * x);
    var y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }
  function homeCoverProbability(line, center, mean, std) {
    var z = ((line - center) - mean) / (std * Math.SQRT2);
    return 1 - 0.5 * (1 + erf(z));
  }

  function selectGame(gameId) {
    document.querySelectorAll('.dive-panel').forEach(function (panel) {
      panel.hidden = panel.dataset.gameId !== gameId;
    });
    document.querySelectorAll('table.board tr.game').forEach(function (row) {
      row.classList.toggle('is-selected', row.dataset.gameId === gameId);
    });
  }
  // Exposed so the shared ticker script (_TICKER_SCRIPT) can select a game
  // by id too, without needing a second copy of this logic or a
  // now-removed ``.dive-tab`` to click through.
  window.atsSelectGame = selectGame;

  document.querySelectorAll('table.board tr.game').forEach(function (row) {
    row.addEventListener('click', function (evt) {
      var gameId = row.dataset.gameId;
      if (!gameId) return;
      // Keeps the board in place: a row click swaps the inspector panel
      // in-page rather than following the row-link anchor's own
      // #<game_id> href (that href is the no-JS fallback, shown via
      // :target in the stylesheet).
      evt.preventDefault();
      selectGame(gameId);
    });
  });

  document.querySelectorAll('.ats-adjuster').forEach(function (widget) {
    var slider = widget.querySelector('.adjuster-slider');
    if (!slider) return;
    var lineOut = widget.querySelector('.adjuster-line');
    var pctOut = widget.querySelector('.adjuster-pct');
    var marker = widget.querySelector('.adjuster-marker');
    var center = parseFloat(widget.dataset.center);
    var mean = parseFloat(widget.dataset.mean);
    var std = parseFloat(widget.dataset.std);
    var cardLine = parseFloat(widget.dataset.cardLine);
    var pickIsHome = widget.dataset.pickIsHome === '1';
    var xMin = parseFloat(widget.dataset.xMin);
    var xMax = parseFloat(widget.dataset.xMax);
    var yMin = parseFloat(widget.dataset.yMin);
    var yMax = parseFloat(widget.dataset.yMax);
    var xSpan = (xMax - xMin) || 1;
    var ySpan = (yMax - yMin) || 1;
    function px(offset) { return 20 + (offset - xMin) / xSpan * 240; }
    function py(probability) {
      var clamped = Math.min(Math.max(probability, yMin), yMax);
      return 85 - (clamped - yMin) / ySpan * 75;
    }
    function update() {
      var offset = parseFloat(slider.value);
      var homeP = homeCoverProbability(cardLine + offset, center, mean, std);
      var pickP = pickIsHome ? homeP : 1 - homeP;
      var line = cardLine + offset;
      if (lineOut) lineOut.textContent = (line > 0 ? '+' : '') + line.toFixed(1);
      if (pctOut) pctOut.textContent = (pickP * 100).toFixed(1) + '%';
      if (marker) {
        marker.setAttribute('cx', px(offset).toFixed(1));
        marker.setAttribute('cy', py(pickP).toFixed(1));
      }
    }
    slider.addEventListener('input', update);
    update();
  });
})();
</script>
"""

#: The board's KICKOFF | CONFIDENCE sort toggle (item 5) -- vanilla JS,
#: static-safe, no dependency on any other inline script on the page.
#: "Kickoff" restores the exact original DOM order (group headers and all,
#: captured once on load); "Confidence" hides the day-group headers (a
#: confidence sort flattens across days) and re-orders the game rows by
#: descending ``data-prob``.
_SORT_SCRIPT = """
<script>
(function () {
  document.querySelectorAll('.board-scroll table.board').forEach(function (table) {
    var tbody = table.tBodies[0];
    var toggle = table.closest('section').querySelector('.sort-toggle');
    if (!tbody || !toggle) return;
    var originalOrder = Array.prototype.slice.call(tbody.children);
    function sortByKickoff() {
      originalOrder.forEach(function (row) { tbody.appendChild(row); });
      tbody.querySelectorAll('tr.grp').forEach(function (row) { row.hidden = false; });
    }
    function sortByConfidence() {
      tbody.querySelectorAll('tr.grp').forEach(function (row) { row.hidden = true; });
      var games = Array.prototype.slice.call(tbody.querySelectorAll('tr.game'));
      games.sort(function (a, b) {
        return parseFloat(b.dataset.prob) - parseFloat(a.dataset.prob);
      });
      games.forEach(function (row) { tbody.appendChild(row); });
    }
    toggle.querySelectorAll('.sort-btn').forEach(function (button) {
      button.addEventListener('click', function () {
        toggle.querySelectorAll('.sort-btn').forEach(function (other) {
          other.classList.toggle('is-active', other === button);
          other.setAttribute('aria-pressed', other === button ? 'true' : 'false');
        });
        if (button.dataset.sort === 'confidence') { sortByConfidence(); } else { sortByKickoff(); }
      });
    });
  });
})();
</script>
"""

#: Lineup unit filters are progressive enhancement; all players remain in the
#: HTML and the buttons only change visibility for the current reader.
_LINEUP_SCRIPT = """
<script>
(function () {
  document.querySelectorAll('.lineups-block').forEach(function (block) {
    var buttons = block.querySelectorAll('[data-lineup-toggle]');
    buttons.forEach(function (button) {
      button.addEventListener('click', function () {
        var unit = button.dataset.lineupToggle;
        var active = !button.classList.contains('is-active');
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', active ? 'true' : 'false');
        block.querySelectorAll('[data-lineup-unit="' + unit + '"]').forEach(function (section) {
          section.hidden = !active;
        });
      });
    });
  });
})();
</script>
"""

#: Shared ticker-click behaviour for every page (items 6-7): each tick is a
#: real ``index.html#<game_id>`` link. On This Week itself (where
#: ``.dive-panel`` inspector panels exist), a click selects that game (via
#: ``window.atsSelectGame``, defined by :data:`_DIVE_SCRIPT` -- see layout
#: A, 2026-09-05: the board's own rows are the selector, so there is no
#: ``.dive-tab`` to click through any more) and scrolls to its panel, and
#: page load re-selects from ``location.hash`` -- both without a page
#: reload. On The Model/Findings (no inspector on the page), the click is
#: left alone and the browser's ordinary anchor navigation carries the
#: reader to This Week with the hash already set, where the same on-load
#: handler takes over.
_TICKER_SCRIPT = """
<script>
(function () {
  function activateGame(gameId) {
    var panel = document.querySelector('.dive-panel[data-game-id="' + gameId + '"]');
    if (!panel) return null;
    if (window.atsSelectGame) { window.atsSelectGame(gameId); }
    return panel;
  }
  document.querySelectorAll('.tick-link').forEach(function (link) {
    link.addEventListener('click', function (evt) {
      var gameId = link.dataset.gameId;
      var panel = activateGame(gameId);
      if (panel) {
        evt.preventDefault();
        history.pushState(null, '', 'index.html#' + gameId);
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
  if (document.querySelector('.dive-panel') && location.hash) {
    var hashGameId = decodeURIComponent(location.hash.slice(1));
    var panel = activateGame(hashGameId);
    if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
})();
</script>
"""

#: UI-20 (2026-09-05, owner: "i absolutely hate this dynamic page load
#: thing where elements only appear once you scroll down far enough").
#: The scroll-gated IntersectionObserver reveal and the KPI number
#: roll-up (formerly here as ``_MOTION_SCRIPT``) are REMOVED entirely:
#: every element is visible at load, with no scroll dependency, and every
#: KPI value renders its final number immediately (it always was already
#: in the static HTML; only the animation is gone).


def _nav_links(page: str) -> str:
    """Real links to every SITE_PAGES page, in nav order, as bare
    filenames. The current page is marked both with ``.is-active`` and
    ``aria-current="page"``."""

    links = []
    for filename, label, _title in SITE_PAGES:
        if filename == page:
            links.append(
                f'<a href="{escape(filename)}" class="is-active" aria-current="page">'
                f"{escape(label)}</a>"
            )
        else:
            links.append(f'<a href="{escape(filename)}">{escape(label)}</a>')
    return "".join(links)


def _ticker(chrome: TickerChrome) -> str:
    """The scrolling ticker, shared on every page (owner-approved
    improvement batch, item 7). Each tick is a real link to
    ``index.html#<game_id>`` (item 6): on This Week itself, the shared
    ``_TICKER_SCRIPT`` intercepts the click and selects/scrolls to that
    game's dive in place; on The Model/Findings, the browser's ordinary
    anchor navigation takes over and This Week's own selector script reads
    the hash on load. The whole track also gets ``tabindex`` via its own
    focusable ticks, so keyboard focus (not just hover) pauses the
    animation -- see the ``.ticker:hover, .ticker:focus-within`` CSS rule.
    """

    def tick(game: GameRow) -> str:
        href = f"index.html#{escape(game.game_id)}"
        if chrome.best_pick_game_id is not None and game.game_id == chrome.best_pick_game_id:
            inner = (
                '<span class="best">&#9733; '
                f"{escape(game.ticker_text)} {escape(game.pick_team)} "
                f"{escape(game.pick_spread_text)} {escape(game.probability_text)}</span>"
            )
        else:
            inner = (
                f"{escape(game.ticker_text)} <b>{escape(game.pick_team)} "
                f"{escape(game.pick_spread_text)}</b> "
                f'<span class="up">{escape(game.probability_text)}</span>'
            )
        return (
            f'<a class="tick tick-link" href="{href}" data-game-id="{escape(game.game_id)}">'
            f"{inner}</a>"
        )

    ticks = "".join(tick(game) for game in chrome.games)
    # Doubled for the seamless CSS-animation loop, exactly as the mockup does.
    return (
        '<div class="ticker" role="marquee" aria-label="This week\'s board, scrolling summary">'
        f'<div class="ticker-track">{ticks}{ticks}</div></div>'
    )


def _header(
    *,
    page: str,
    season: int | None = None,
    week: int | None = None,
    game_type: str = "REG",
    week_label: str = "",
) -> str:
    week_tag = ""
    if season is not None or week is not None:
        season_text = f"SEASON {season} &middot; " if season is not None else ""
        week_text = f"WEEK {week}" if game_type == "REG" else week_label.upper()
        week_tag = f'<span class="week-tag">{season_text}{escape(week_text)}</span>'
    return (
        '<header class="bar">'
        '<div class="brand"><span class="dot"></span>'
        '<span class="brand-word">ATS<span>::</span>TERM</span></div>'
        f'<nav class="links">{_nav_links(page)}</nav>'
        f'<div class="session-meta">{week_tag}</div></header>'
    )


def _cmd_row(chrome: TickerChrome) -> str:
    """The command row, shared on every page (item 7). ``page_command_suffix``
    is the one field that varies per page (item 7: "vary the command row
    text per page ... via the content layer") -- e.g. ``--page model``."""

    method_arg = escape(chrome.model_method_label.split(" ", 1)[0])
    suffix = f" {escape(chrome.page_command_suffix)}" if chrome.page_command_suffix else ""
    return (
        '<div class="cmd-row"><span class="prompt">&gt;</span><span>nfl-ats board</span>'
        f'<span class="arg">--season {chrome.season} --week {chrome.week} '
        f"--model {method_arg}{suffix}</span>"
        '<span class="cursor"></span></div>'
    )


def _motion_status_rail(chrome: TickerChrome) -> str:
    """Rotate only facts already present in the rendered board snapshot."""

    game_count = len(chrome.games)
    strong_count = sum(game.confidence_word == "strong" for game in chrome.games)
    best_game = next(
        (game for game in chrome.games if game.game_id == chrome.best_pick_game_id), None
    )
    if best_game is None:
        best_text = "NO UNIQUE BEST PICK"
        best_html = escape(best_text)
    else:
        best_text = f"BEST PICK {best_game.pick_team} {best_game.pick_spread_text}"
        best_html = (
            'BEST PICK <span class="rail-accent">'
            f"{escape(best_game.pick_team)} {escape(best_game.pick_spread_text)}</span>"
        )
    season_week = "SCHEDULE CONTEXT UNAVAILABLE"
    if chrome.season is not None and chrome.week is not None:
        season_week = f"SEASON {chrome.season} / WEEK {chrome.week}"
    frames = (
        f'<span class="status-frame" style="--frame-index:0">'
        f'<span class="rail-number" data-roll-to="{game_count}">{game_count}</span> '
        "GAMES / "
        f'<span class="rail-number" data-roll-to="{strong_count}">{strong_count}</span> '
        "STRONG READS</span>",
        f'<span class="status-frame" style="--frame-index:1">MODEL '
        f'<span class="rail-accent">{escape(chrome.model_method_label)}</span></span>',
        f'<span class="status-frame" style="--frame-index:2">{best_html}</span>',
        f'<span class="status-frame" style="--frame-index:3">{escape(season_week)}</span>',
    )
    accessible_summary = (
        f"Board snapshot: {game_count} games, {strong_count} strong reads. "
        f"Model {chrome.model_method_label}. "
        f"{best_text}. {season_week}."
    )
    return (
        f'<div class="motion-status" aria-label="{escape(accessible_summary)}">'
        '<span class="motion-beacon" aria-hidden="true"></span>'
        '<span class="motion-label">BOARD SNAPSHOT</span>'
        f'<span class="status-rotator" aria-hidden="true">{"".join(frames)}</span>'
        '<span class="motion-bars" aria-hidden="true"><i></i><i></i><i></i><i></i></span>'
        "</div>"
    )


def _terminal_chrome(
    chrome: TickerChrome,
    *,
    page: str,
    season: int | None = None,
    week: int | None = None,
    game_type: str = "REG",
    week_label: str = "",
) -> str:
    """Render the shared ticker/nav/command/status stack as one sticky unit."""

    return (
        '<div class="terminal-chrome">'
        + _ticker(chrome)
        + _header(
            page=page,
            season=season,
            week=week,
            game_type=game_type,
            week_label=week_label,
        )
        + _cmd_row(chrome)
        + _motion_status_rail(chrome)
        + "</div>"
    )


def _season_shape_html(headline: HeadlineStats) -> str:
    if headline.raw_model_season_count <= 0:
        return ""
    ticks = "".join("<i></i>" for _ in range(headline.raw_model_season_count))
    caption = (
        escape(headline.raw_model_season_note.upper()) if headline.raw_model_season_note else ""
    )
    baseline = f'<span class="baseline">{caption}</span>' if caption else ""
    return f'<div class="season-shape">{ticks}{baseline}</div>'


def _prospective_scoreboard_html(headline: HeadlineStats) -> str:
    """The paired prospective record beside the "tracked prospectively"
    caveat (owner-approved improvement batch, item 3). Renders the designed
    dormant state ("prospective tracking begins Week 1") until either
    ledger holds a row -- never a raise, never an invented number."""

    scoreboard = headline.prospective_scoreboard
    classes = "prospective-scoreboard dormant" if scoreboard.dormant else "prospective-scoreboard"
    detail_html = (
        f'<p class="ps-detail">{escape(scoreboard.detail_text)}</p>'
        if scoreboard.detail_text
        else ""
    )
    return (
        f'<div class="{classes}"><span class="ps-flag">Prospective scoreboard</span>'
        f"<p>{escape(scoreboard.headline_text)}</p>{detail_html}</div>"
    )


def _headline_section(headline: HeadlineStats) -> str:
    """The four-stat headline block. Shared verbatim by the This Week page
    and The Model page -- see :class:`~nfl_ats.board_site_content
    .ModelPageContent`'s docstring for why this is the one deliberate
    cross-page dedup exception rather than two copies."""

    raw_model_ci = (
        f"95% CI <b>[{headline.raw_model_ci[0]:.2f}%, {headline.raw_model_ci[1]:.2f}%]</b>"
        if headline.raw_model_ci is not None
        else "interval not yet published"
    )
    return (
        '<section aria-labelledby="stats-h"><div class="section-head">'
        '<h2 id="stats-h">Headline accuracy</h2>'
        '<span class="sub">source: this week\'s published forecast</span></div>'
        '<div class="headline-block"><div class="headline-main">'
        '<span class="label">Played policy &middot; archive score</span>'
        f'<span class="value">{headline.played_card_value_text}</span>'
        f'<span class="foot">{escape(headline.played_card_foot_text)}</span>'
        f'<span class="foot">{escape(headline.model_method_label)}</span>'
        "</div>"
        '<div class="caveat">'
        '<span class="caveat-flag">How to read this number'
        "</span>"
        f"<p>{escape(headline.selection_caveat_text)}</p>"
        f"{_prospective_scoreboard_html(headline)}"
        "</div></div>"
        '<div class="kpi-grid">'
        '<div class="kpi"><span class="label">Prior chain &middot; coach &rarr; arrests</span>'
        f'<span class="value muted">{headline.prior_chain_value_text}</span>'
        f'<span class="foot">{escape(headline.prior_chain_caption)}</span></div>'
        '<div class="kpi"><span class="label">Model alone &middot; opener grade baseline</span>'
        f'<span class="value good">{headline.raw_model_value_text}</span>'
        f"{_season_shape_html(headline)}"
        f'<span class="foot">{raw_model_ci}</span>'
        "</div>"
        '<div class="kpi"><span class="label">Active model &middot; close grade</span>'
        f'<span class="value muted">{headline.close_grade_value_text}</span>'
        f'<span class="foot">{escape(headline.close_grade_caption)}</span></div>'
        "</div>"
        '<div class="policy-note" style="margin-top:1px;border-left-color:var(--line);">'
        f"Active model <b>{escape(headline.model_method_label)}</b>"
        ". Four stats, four roles: headline archive score, the prior chain it's tracked "
        "against, the model-alone baseline it's built on, and the model's own close-graded "
        "classification. Full source-and-date detail is at the bottom of this page.</div>"
        "</section>"
    )


def _confidence_meter_html(game: GameRow) -> str:
    segments = "".join(
        f'<i class="{"on" if index < game.confidence_fill else ""}"></i>' for index in range(3)
    )
    return (
        f'<div class="meter"><div class="segs">{segments}</div>'
        f'<span class="word">{escape(game.confidence_label)}</span></div>'
    )


def _final_outcome_html(game: GameRow) -> str:
    """Replaces the confidence meter for a FINAL game (season mode, item 4):
    the played pick's own final score and cover result, never the raw
    model's meter -- the meter measured a forecast, this reports what
    happened."""

    return (
        f'<div class="outcome outcome-{escape(game.cover_result or "")}">'
        f'<span class="outcome-score">{escape(game.final_score_text or "Final")}</span>'
        f'<span class="outcome-word">{escape(game.cover_result_label)}</span></div>'
    )


def _flip_pill_html(game: GameRow) -> str:
    """The flip pill (owner-approved improvement batch, item 1): the swap
    glyph "⇄" -- never the word "FLIPPED", per the owner's explicit
    instruction -- plus the member name(s) that fired. Empty when the game
    was not flipped."""

    if not game.flip_member_labels:
        return ""
    return f'<span class="pill flip-pill">{escape(game.flip_pill_text)}</span>'


def _flip_line_html(game: GameRow) -> str:
    """The "Flips at" cell (owner request, 2026-09-01): the pick's own
    handicap at the first half-point line that changes the mind, then the
    team it switches to -- ``NYJ +2.5 → TEN`` (see
    ``GameRow.flip_line_text`` for why the pick's orientation, not the
    flipped-to team's). Policy members are re-evaluated at the hypothetical
    line, so a spread-gap-zone game shows its zone exit; a pick nothing
    switches inside the adjuster's own ±4 span says "within ±4" -- a
    bounded claim on purpose, never "at any line" (owner catches,
    2026-09-01, both rounds). An em-dash only when the game is final (a
    flip line on a settled row is stale noise) or no source exists
    (degraded artifacts)."""

    if game.final or not game.flip_line_text:
        return "<span class='flip-none'>&mdash;</span>"
    if game.flip_line is None and game.flip_held:
        return (
            "<span class='flip-none' title='No line inside the adjuster&#39;s explored "
            f"range changes this pick'>{escape(game.flip_line_text)}</span>"
        )
    return escape(game.flip_line_text)


def _board_sort_toggle_html() -> str:
    """The KICKOFF | CONFIDENCE sort toggle (item 5) -- vanilla JS,
    static-safe (see :data:`_SORT_SCRIPT`), 44px touch targets via the
    ``.sort-btn`` CSS, and native ``<button>``s so keyboard users get the
    toggle for free."""

    return (
        '<div class="sort-toggle" role="group" aria-label="Sort the board">'
        '<button type="button" class="sort-btn is-active" data-sort="kickoff" '
        'aria-pressed="true">Kickoff</button>'
        '<button type="button" class="sort-btn" data-sort="confidence" '
        'aria-pressed="false">Confidence</button>'
        "</div>"
    )


def _source_policy_panel_html(view: SourcePolicyView) -> str:
    """The SOURCES panel (ENG-34): the worst-wins card state in the panel's
    own header line, one dot-leader line per source, and the plain-English
    legend -- after the board's picks and supporting notes (see
    :func:`_board_section`). Reuses ``.policy-note`` for the panel frame and
    only the small ``.src-*`` rules added to ``board_terminal_style.css`` for
    the per-source rows; pure HTML/CSS, no script, so it renders identically
    with JS disabled like the rest of the board.

    UI-20(c): when nothing was persisted for this forecast, ``view.rows`` may
    still carry a REAL, just-computed report (``view.computed_live``) rather
    than being empty -- rendered exactly like a recorded report, plus one
    extra disclosure sentence so a reader never mistakes "computed now" for
    "locked at Tuesday's publish".
    """

    header = (
        '<div class="sources-panel policy-note" aria-labelledby="sources-h">'
        '<b id="sources-h">Sources &mdash; card state: '
        f'<span class="src-state {escape(view.card_state)}">{escape(view.card_state_label)}'
        "</span></b>"
    )
    if not view.rows:
        body = (
            '<p class="src-empty">No source-freshness block is recorded for this forecast '
            "(an older artifact that predates the ENG-14 policy being persisted to "
            "metadata.json).</p>"
        )
    else:
        rows_html = "".join(
            '<div class="src-row">'
            f'<span class="src-name">{escape(humanize_identifier(row.source_id))}</span>'
            '<span class="src-leader" aria-hidden="true"></span>'
            f'<span class="src-state {escape(row.state)}" title="{escape(row.detail_text)}">'
            f"{escape(row.state_label)}</span>"
            '<span class="src-asof">'
            f"{escape(_relative_update(row.observed_at, view.evaluated_at))}</span>"
            "</div>"
            for row in view.rows
        )
        live_note = f" {escape(SOURCE_POLICY_COMPUTED_LIVE_NOTE)}" if view.computed_live else ""
        body = (
            f'<div class="src-rows">{rows_html}</div>'
            f'<p class="src-evaluated">Checked {escape(_humanize_timestamp(view.evaluated_at))}. '
            "Source ages are measured from that check."
            f"{live_note}</p>"
        )
    legend = (
        '<p class="src-legend">complete: fresh enough to use; degraded: we fell back '
        "to an older copy; blocked: we refused to publish</p>"
    )
    return header + body + legend + "</div>"


def _tiebreaker_panel_html(view: TiebreakerView) -> str:
    """UI-20(g): the pool's tiebreaker guess for the week's last game.
    Collapsed by default (a ``<details>`` block, matching the "Why this
    pick" disclosure) so its point totals never inflate This Week's
    default-visible-percentage budget -- not that a point total is a
    percentage, but the same de-firehose discipline applies to every
    numeric block on this board. Reuses ``.policy-note``/``.micro``/
    ``.game-sub``; zero new CSS."""

    if not view.recorded:
        body = f'<p class="game-sub">{escape(view.note)}</p>'
    else:
        guess_line = f", guess {escape(view.guess_score_text)}" if view.guess_score_text else ""
        body = (
            f'<p class="game-sub">{escape(view.matchup_text)}: market total '
            f"{escape(view.market_total_text)}, blended total "
            f"{escape(view.blended_total_text)}, implied margin "
            f"{escape(view.implied_margin_text)}{guess_line}.</p>"
            f'<p class="micro">{escape(view.note)}</p>'
        )
    return (
        '<details class="policy-note"><summary class="micro" style="cursor:pointer;">'
        "Tiebreaker guess</summary>" + body + "</details>"
    )


def _why_this_pick_html(explanation_text: str) -> str:
    """ "Why this pick" (dashboard queue, ROADMAP.md UI-20(a); relocated
    2026-09-05 for layout A, "board + inspector"): the ENG-12 explanation
    text (market line, this game's own model probability, fired overlays,
    per-source freshness, and Tuesday-to-refresh status -- already composed
    and language-contract-checked by ``nfl_ats.card_explanation``) or the
    explicit not-recorded sentence, rendered once, inside the game's own
    inspector panel (see :func:`_dive_panel_html`). This used to be a
    collapsed disclosure printed under EVERY board row; it now prints once
    per game, inside that game's own mostly-hidden ``.dive-panel`` (only the
    selected game's panel lacks the ``hidden`` attribute), which is the same
    "adds nothing to the default-visible count for any other game"
    discipline the old collapsed-by-default row followed, applied through
    panel visibility instead of ``<details>``. Reuses the existing
    ``.policy-note`` boxed-fact styling already used for the policy-overlay
    note directly above the board; no new CSS."""

    return (
        '<div class="policy-note" style="margin:0 18px 14px;">'
        f"<b>Why this pick</b> &mdash; {escape(explanation_text)}</div>"
    )


# ---------------------------------------------------------------------------
# Reader-facing number/identifier formatting (owner mandate, 2026-09-05,
# verbatim on a live panel: "whats the point of showing this anywhere? ...
# remember when i said this is for humans not the opus autist"). The
# research registries stay snake_case, hashed, and P+-notated internally --
# still queryable via ``nfl-ats weak-signals``/``rotation`` -- these three
# helpers are the ONE place that vocabulary gets translated before it
# reaches a rendered page. Kept together so every render function below
# reaches for the same words rather than inventing its own phrasing.
# ---------------------------------------------------------------------------

#: The weak-signal registry's three closing classifications
#: (``nfl_ats.weak_signals.POOLABLE_CLASSIFICATION`` /
#: ``TERMINAL_CLASSIFICATIONS``), in words. Anything else (a rotation-
#: registry status, etc.) falls back to :func:`humanize_identifier`.
_CLASSIFICATION_WORDS: dict[str, str] = {
    "unresolved_below_power": "not enough evidence yet",
    "refuted_mechanism": "ruled out",
    "bounded_by_control": "ruled out by a control test",
}


def _humanize_classification(value: str) -> str:
    return _CLASSIFICATION_WORDS.get(value, humanize_identifier(value))


def _humanize_probability_positive(value: float) -> str:
    """Plain-English rendering of a weak-signal registry's
    ``probability_positive`` -- the chance the effect is genuinely
    positive, not a certainty about its SIZE, and not "contains zero" --
    see AGENTS.md's closing-grounds taxonomy. Replaces the registry's own
    "P+ 0.79" shorthand, which is machine notation, not football."""

    return f"{value:.0%} likely real"


def _humanize_artifact_ref(ref: str) -> str:
    """An artifact reference like ``"margins/20260905T133348Z"`` -- kind
    plus a bare reader-facing date, never the raw stamp (owner mandate,
    2026-09-05). Falls back to the raw text when it doesn't parse as
    ``<kind>/<stamp>`` -- never hides real data behind a formatting bug."""

    kind, _, stamp = ref.rpartition("/")
    if not kind or not stamp:
        return f"via {ref}"
    humanized = _humanize_timestamp(stamp)
    if humanized == stamp:
        return f"via {ref}"
    return f"({kind}, {humanized})"


def _parse_render_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace(" UTC", "+00:00").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _relative_update(raw: str | None, evaluated_at: str | None) -> str:
    observed = _parse_render_time(raw)
    evaluated = _parse_render_time(evaluated_at)
    if observed is None:
        return "no snapshot" if not raw else "update time unavailable"
    if evaluated is None:
        return f"updated {_humanize_timestamp(raw)}"
    minutes = (evaluated - observed).total_seconds() / 60
    if minutes < 0:
        return "updated after this card was checked"
    if minutes < 60:
        return "updated less than an hour ago"
    if minutes < 24 * 60:
        hours = int(minutes // 60)
        return f"updated {hours} hour{'s' if hours != 1 else ''} ago"
    return f"updated {_humanize_timestamp(raw)}"


def _humanize_timestamp(raw: str | None) -> str:
    """Render a source instant as a weekday and part of day, with no raw stamp."""

    parsed = _parse_render_time(raw)
    if parsed is None:
        return "time unavailable"
    period = "morning" if parsed.hour < 12 else "afternoon" if parsed.hour < 18 else "evening"
    return f"{parsed:%A} {period}"


def _default_game_id(content: BoardContent) -> str:
    """The game the board pre-selects: the Best Pick when one exists, else
    the week's first game (chronological, matching ``content.games``'
    own order) -- shared by the board table (which row is ``is-selected``)
    and the inspector (which panel is visible without ``hidden``), so the
    two never disagree about which game is "current"."""

    if content.best_pick_game_id is not None:
        return content.best_pick_game_id
    if content.dives:
        return content.dives[0].game_id
    if content.games:
        return content.games[0].game_id
    return ""


def _board_section(content: BoardContent) -> str:
    """UI-20 layout A (2026-09-05 owner-approved mockup, "board + inspector"):
    the LEFT column of the This Week two-column grid. Every row's matchup
    cell is now a real ``href="#<game_id>"`` anchor into the matching
    ``.dive-panel`` in the RIGHT column's inspector (see
    :func:`_inspector_section`) -- a plain in-page link with JavaScript off,
    and the same click the shared selector script (``_DIVE_SCRIPT``) wires
    to a no-scroll panel swap plus the ``is-selected`` row highlight when
    JavaScript runs. The per-row "Why this pick" disclosure this section
    used to render directly under each pick row is gone from here -- its
    content (``GameRow.explanation_text``) now lives once, in the selected
    game's own inspector panel, never printed twice on the page."""

    policy = content.policy
    if policy.rich_narrative:
        policy_html = escape(policy.rich_narrative)
    else:
        policy_html = escape(policy.composition_text) + "."

    default_game_id = _default_game_id(content)
    rows: list[str] = []
    for day, day_games in groupby(content.games, key=lambda game: game.kickoff_group_label):
        rows.append(f'<tr class="grp"><td colspan="6">{escape(day)}</td></tr>')
        for game in day_games:
            pick_text = f"{escape(game.pick_team)} {escape(game.pick_spread_text)}"
            if game.is_best:
                pick_cell = (
                    f'<span class="star">&#9733;</span>{pick_text}'
                    '<span class="best-flag">Best pick</span>'
                )
            else:
                pick_cell = pick_text
            pick_cell += _flip_pill_html(game)
            row_classes = ["game"]
            if game.is_best:
                row_classes.append("is-best")
            if game.game_id == default_game_id:
                row_classes.append("is-selected")
            if game.final and game.cover_result:
                row_classes.append(f"final-{game.cover_result}")
            conf_cell = _final_outcome_html(game) if game.final else _confidence_meter_html(game)
            matchup_cell = (
                f'<a class="row-link" href="#{escape(game.game_id)}" '
                f'data-game-id="{escape(game.game_id)}" '
                f'aria-label="Inspect {escape(game.away)} at {escape(game.home)}">'
                f"{escape(game.away)} at <b>{escape(game.home)}</b></a>"
            )
            rows.append(
                f'<tr class="{" ".join(row_classes)}" data-game-id="{escape(game.game_id)}" '
                f'data-prob="{game.pick_probability:.6f}">'
                f'<td class="kickoff" data-label="Kickoff">{escape(game.kickoff_short_label)}</td>'
                f'<td class="matchup" data-label="Matchup">{matchup_cell}</td>'
                f'<td class="pick" data-label="Pick">{pick_cell}</td>'
                f'<td class="prob" data-label="Cover prob">{escape(game.probability_text)}</td>'
                f'<td class="flipline" data-label="Flips at">{_flip_line_html(game)}</td>'
                f'<td class="conf" data-label="Confidence">{conf_cell}</td>'
                "</tr>"
            )

    table = (
        '<table class="board"><thead><tr>'
        "<th>Kickoff</th><th>Matchup</th><th>Pick</th>"
        "<th><abbr title=\"The computer's own probability, oriented to the final pick. On a "
        'flip this is a mirrored decision-strength score, not a freshly calibrated probability.">'
        "Cover&nbsp;prob</abbr></th>"
        "<th><abbr title=\"Read it as: if the pick's own line reaches this number, the card "
        "switches to the team after the arrow. E.g. a NYJ +3 pick with NYJ +2.5 → TEN "
        "flips to TEN once NYJ gets only +2.5. Uses the spread adjuster's math plus the "
        "fix-up rules re-checked within a point of the quoted line -- crossing the "
        "7.5-10 fade zone's edge flips a pick too when the edge is that close. "
        "'Within ±4' means nothing in the adjuster's explored range changes the "
        'pick.">'
        "Flips&nbsp;at</abbr></th><th>Confidence</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return (
        '<section aria-labelledby="board-h" class="board-col"><div class="section-head">'
        f'<h2 id="board-h">{escape(content.week_label)} board &middot; forced picks</h2>'
        f'<span class="sub">{len(content.games)} games &middot; every pool card played '
        "&middot; click a row to inspect</span>"
        "</div>"
        f"{_board_sort_toggle_html()}"
        f'<div class="board-scroll">{table}</div>'
        f'<p class="policy-note">{escape(content.injury_note)}</p>'
        f"{_tiebreaker_panel_html(content.tiebreaker)}"
        f'<div class="policy-note"><b>Policy overlay</b> &mdash; {policy_html}</div>'
        f"{_source_policy_panel_html(content.source_policy)}</section>"
    )


def _adjuster_html(
    dive: GameDive, *, x_min: float, x_max: float, y_min: float, y_max: float
) -> str:
    """The restored spread-explorer widget, folded into this game's deep
    dive: a line-offset slider driven by the SAME published-fields-only
    Gaussian read :func:`nfl_ats.spread_explorer.widget_home_cover_probability`
    computes in Python, mirrored in :data:`_DIVE_SCRIPT`'s JS. ``None``
    only when this build's active model has no closed-form probability
    method for this game -- never an invented formula."""

    if dive.adjuster is None:
        return (
            '<p class="adjuster-empty">Line-offset adjuster unavailable for this game -- this '
            "build's active model has no closed-form probability read.</p>"
        )
    adjuster = dive.adjuster
    return (
        f'<div class="ats-adjuster" data-center="{adjuster.center:.6f}" '
        f'data-mean="{adjuster.residual_mean:.6f}" data-std="{adjuster.residual_std:.6f}" '
        f'data-card-line="{adjuster.card_line:.3f}" '
        f'data-pick-is-home="{"1" if adjuster.pick_is_home else "0"}" '
        f'data-x-min="{x_min:g}" data-x-max="{x_max:g}" '
        f'data-y-min="{y_min:g}" data-y-max="{y_max:g}">'
        '<span class="adjuster-label">Try a different line</span>'
        f'<input type="range" class="adjuster-slider" min="{x_min:g}" max="{x_max:g}" '
        f'step="{SPREAD_EXPLORER_STEP:g}" value="0" '
        f'aria-label="Hypothetical line offset for {escape(dive.pick_team)} '
        f'{escape(dive.pick_spread_text)}">'
        f'<p class="adjuster-sentence">At <span class="num adjuster-line">'
        f"{adjuster.card_line:+g}</span>, {escape(dive.pick_team)} covers "
        f'<span class="num adjuster-pct">{escape(dive.probability_text)}</span>.</p>'
        "</div>"
    )


def _game_dive_chart_html(dive: GameDive) -> str:
    """The cover-probability curve plus (when available) its line-offset
    adjuster, for one game. Real swept model output wherever the build has
    it; a designed empty state when it does not."""

    if not dive.cover_curve:
        return (
            '<div class="chart-empty">Cover curve not published for this game on this '
            "artifact tree.</div>"
        )
    # Plot box matches the mockup's own viewBox exactly (0 0 280 100, x:20-260, y:10-85).
    offsets = [point.offset for point in dive.cover_curve]
    probabilities = [point.probability for point in dive.cover_curve]
    x_min, x_max = min(offsets), max(offsets)
    x_span = (x_max - x_min) or 1.0
    pad = max(0.03, (max(probabilities) - min(probabilities)) * 0.15)
    y_min = max(0.0, min(probabilities) - pad)
    y_max = min(1.0, max(probabilities) + pad)
    y_span = (y_max - y_min) or 1.0

    def px(offset: float) -> float:
        return 20 + (offset - x_min) / x_span * 240

    def py(probability: float) -> float:
        return 85 - (probability - y_min) / y_span * 75

    path_points = [
        f"{px(point.offset):.1f},{py(point.probability):.1f}" for point in dive.cover_curve
    ]
    path = "M" + " L".join(path_points)
    current = next((point for point in dive.cover_curve if point.offset == 0.0), None)
    pick_label = f"{dive.pick_team} {dive.pick_spread_text}"
    marker = ""
    if current is not None:
        cx, cy = px(0.0), py(current.probability)
        marker = (
            f'<circle class="marker" cx="{cx:.1f}" cy="{cy:.1f}" r="3.4"></circle>'
            f'<text x="{cx + 6:.1f}" y="{cy - 4:.1f}">'
            f"{escape(pick_label)} / {escape(f'{current.probability:.1%}')}</text>"
        )
    ref_line = ""
    if y_min <= 0.5 <= y_max:
        ref_y = py(0.5)
        ref_line = (
            f'<line class="ref" x1="20" y1="{ref_y:.1f}" x2="260" y2="{ref_y:.1f}"></line>'
            f'<text x="222" y="{ref_y - 2:.1f}">50% ref</text>'
        )
    adjuster_marker = (
        '<circle class="adjuster-marker" cx="0" cy="0" r="4"></circle>' if dive.adjuster else ""
    )
    svg = (
        '<svg class="curve" viewBox="0 0 280 100" width="100%" height="140" role="img" '
        f'aria-label="Cover probability across hypothetical lines for {escape(dive.pick_team)} '
        f'to cover, current line marked at {escape(pick_label)}">'
        '<line class="grid" x1="20" y1="10" x2="20" y2="85"></line>'
        '<line class="grid" x1="20" y1="85" x2="260" y2="85"></line>'
        f"{ref_line}"
        f'<path class="curve-path" d="{path}"></path>'
        f"{marker}{adjuster_marker}"
        f'<text x="16" y="94">{x_min:+g}</text><text x="136" y="94">0</text>'
        f'<text x="250" y="94">{x_max:+g}</text>'
        "</svg>"
    )
    note_html = ""
    if dive.cover_curve_offset_zero_note:
        note_html = (
            '<p style="margin-top:8px;font-family:var(--font-mono);font-size:10.5px;'
            f'color:var(--text-faint);">{escape(dive.cover_curve_offset_zero_note)}</p>'
        )
    return (
        f"{svg}"
        '<div class="curve-legend"><span>x &middot; spread offset from the quoted line</span>'
        "<span><b>&#9679;</b> current line</span></div>"
        f"{note_html}"
        f"{_adjuster_html(dive, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)}"
    )


def _attribution_html(dive: GameDive) -> str:
    attribution = dive.attribution
    if not attribution.available:
        return (
            '<div class="attr-empty"><div class="dash">&#8212;</div>'
            f'<div class="note">{escape(attribution.unavailable_note)}</div></div>'
        )
    rows_html = []
    for row in attribution.rows:
        tone = "pos" if row.is_positive else "neg"
        rows_html.append(
            '<div class="attr-row"><div>'
            f'<span class="chan">{escape(row.label)}</span></div>'
            f'<div class="bar"><i class="{tone}" style="width:{row.bar_width_pct:.1f}%;"></i></div>'
            f'<div class="pts {tone}">{escape(row.delta_text)}</div></div>'
        )
    total = (
        f'<div class="attr-total"><span>{escape(attribution.net_label or "Net")}</span>'
        f"<b>{attribution.net_points:+.2f} pts</b></div>"
        if attribution.net_points is not None
        else ""
    )
    return "".join(rows_html) + total


#: UI-20-AB retirement (2026-09-05): the 2026-09-05 "no designation" marker
#: hid the percentage for any player without a visible injury designation,
#: because that number used to be a constant position-level base rate that
#: carried no information about that specific player. It no longer applies:
#: ``play_probability`` is now a real per-player, per-game forecast from
#: ``nfl_ats.play_probability`` (depth-chart rank, this week's own injury
#: report, recent snaps, roster status) for every scored player, so the
#: owner's directive -- "it needs to be a forecast about the game and it
#: needs to consider depth chart" -- is met for every row, designated or
#: not. Every player with a model probability now shows it as a percentage;
#: only ``probability_source == "unavailable"`` (no gsis_id / no predictor)
#: still shows the em dash.


def _lineup_team_html(lineup: TeamLineup | None) -> str:
    if lineup is None:
        return '<div class="lineup-empty">Projected lineup artifact not published yet.</div>'
    rows_by_unit: dict[str, list[str]] = {"offense": [], "defense": [], "special_teams": []}
    for player in lineup.players:
        # UI-20-AB: every player with a model probability shows it as a
        # percentage (see the retirement note above) -- the em dash stays
        # reserved for a row the model genuinely could not score at all
        # (``probability_source == "unavailable"``, i.e. no gsis_id or no
        # predictor this run). UI-20 colour-coding fix (2026-09-05, owner
        # complaint: "what is the color coding? green vs red? some aren't
        # coloured"): the probability cell's tone is AVAILABILITY RISK ON
        # THE NUMBER ITSELF -- green >=85%, amber 50-85%, red <50% -- never
        # the sign of the scored QB's matchup impact, which has nothing to
        # do with whether a player is expected to play (that tone lives on
        # the name-line impact text instead).
        if player.play_probability is None:
            probability = "—"
            risk_tone = ""
        else:
            probability = f"{player.play_probability:.0%}"
            risk_tone = (
                "risk-high"
                if player.play_probability >= 0.85
                else "risk-low"
                if player.play_probability < 0.50
                else "risk-mid"
            )
        # The play model forecasts a starting slot by playing time for
        # every position; display it whenever the saved lineup supplies it.
        start_html = ""
        if player.start_probability is not None:
            start_html = (
                '<span title="Fills a starting slot by playing time" '
                'style="display:block;font-weight:400;font-size:9px;'
                f'color:var(--text-faint);">starts {player.start_probability:.0%}</span>'
            )
        injury = player.injury_status or "no report"
        is_base_model_qb = player.model_role == "base_model"
        impact = player.model_impact_note or (
            "model's starter"
            if is_base_model_qb
            else "not scored by the active model"
            if player.model_role == "context_only"
            else "model input"
        )
        # The matchup-impact tone (positive/negative effect on the model's
        # margin) now lives on the impact-note TEXT, not the probability
        # cell -- it only ever applies to the model's own scored QB input.
        impact_tone = (
            "impact-pos"
            if player.model_impact_points is not None and player.model_impact_points >= 0
            else "impact-neg"
            if player.model_impact_points is not None
            else ""
        )
        prob_title = escape("Plays = takes at least one snap. " + (player.probability_reason or ""))
        row = (
            '<div class="lineup-row">'
            f'<div class="lineup-pos">{escape(player.slot)}</div>'
            f'<div class="lineup-player"><b>{escape(player.name)}</b>'
            f'<span class="{impact_tone}">{escape(injury)} &middot; {escape(impact)}</span></div>'
            f'<div class="lineup-prob {risk_tone}" title="{prob_title}">plays {escape(probability)}'
            f"{start_html}</div>"
            "</div>"
        )
        rows_by_unit.setdefault(player.unit, []).append(row)
    sections = []
    labels = {"offense": "Offense", "defense": "Defense", "special_teams": "Special teams"}
    for unit, label in labels.items():
        rows = rows_by_unit[unit]
        if rows:
            sections.append(
                f'<div class="lineup-unit" data-lineup-unit="{unit}">'
                f'<div class="lineup-unit-head">{label}<span>{len(rows)} players</span></div>'
                f"{''.join(rows)}</div>"
            )
    source = (
        "depth chart"
        if "depth" in (lineup.source or "").lower()
        else escape(lineup.source or "source unavailable")
    )
    as_of = escape(_humanize_timestamp(lineup.as_of))
    injury_status = lineup.injury_status
    if injury_status.startswith("no players listed on this week's injury report"):
        injury_status = (
            "No one from this team is on this week's injury report yet, so these chances "
            "come from each player's recent playing time and roster status."
        )
    note = f'<div class="lineup-note">{escape(lineup.note)}</div>' if lineup.note else ""
    return (
        f'<div class="lineup-team-head"><b>{escape(lineup.team)}</b>'
        f"<span>{source} from {as_of}</span></div>"
        f'<div class="lineup-status">{escape(injury_status)}</div>'
        f"{note}{''.join(sections)}"
    )


#: UI-20-AB fine print (rewritten 2026-09-05, owner directive via the
#: coordinator: "it needs to be a forecast about the game and it needs to
#: consider depth chart" -- retires the 2026-09-05 "no designation" stopgap
#: now that every player's percentage is a real forecast, not a position
#: base rate). States what the number is, where the QB's second number
#: comes from, and how the colour is chosen.
_LINEUP_PROBABILITY_LEGEND = (
    "plays = takes at least one snap; starts = fills a starting slot by playing time. "
    "Colour shows availability risk: green is low, amber is medium, red is high. "
    "Hover for the basis; a dash means no estimate is available."
)


def _lineups_html(dive: GameDive) -> str:
    return (
        '<div class="lineups-block"><div class="lineups-head">'
        '<div><div class="chart-cap">Projected lineups &amp; model impact</div>'
        '<div class="lineups-sub">depth-chart starters, play likelihood, and the active model\'s '
        "scored player state</div></div>"
        '<span class="sample-tag">source-aware</span></div>'
        '<div class="lineup-toggles" role="group" aria-label="Lineup units">'
        '<button type="button" class="lineup-toggle is-active" '
        'data-lineup-toggle="offense">Offense</button>'
        '<button type="button" class="lineup-toggle is-active" '
        'data-lineup-toggle="defense">Defense</button>'
        '<button type="button" class="lineup-toggle is-active" '
        'data-lineup-toggle="special_teams">Special teams</button>'
        "</div>"
        '<div class="lineup-grid">'
        f'<div class="lineup-team">{_lineup_team_html(dive.away_lineup)}</div>'
        f'<div class="lineup-team">{_lineup_team_html(dive.home_lineup)}</div>'
        "</div>"
        '<p style="margin:10px 18px 14px;font-family:var(--font-mono);font-size:10.5px;'
        f'color:var(--text-faint);">{escape(_LINEUP_PROBABILITY_LEGEND)}</p>'
        "</div>"
    )


def _dive_panel_html(
    content: BoardContent,
    dive: GameDive,
    *,
    default_game_id: str,
    explanation_text: str,
    is_last_game: bool,
) -> str:
    """One game's inspector panel (layout A, "board + inspector"): header
    (Best Pick tag when applicable, matchup, kickoff, pick + cover prob),
    why this pick, attribution, the cover-probability curve plus its
    line-offset spread explorer, and the two lineup blocks -- everything
    the RIGHT column shows for whichever game the LEFT column's board rows
    select (see :func:`_board_section`/:func:`_inspector_section`). All 16
    panels render unconditionally; every panel but ``default_game_id``'s
    carries ``hidden`` so only the selected game is visible at load --
    :data:`_DIVE_SCRIPT` toggles that attribute with no page reload, and
    ``:target`` in the stylesheet shows the right one when JavaScript
    cannot run at all (see the board row's own ``.row-link`` anchor)."""

    hidden_attr = "" if dive.game_id == default_game_id else " hidden"
    note_html = ""
    if dive.is_best and content.best_pick_note:
        note_html = (
            f'<div class="game-sub" style="margin-top:4px;">{escape(content.best_pick_note)}</div>'
        )
    flip_note_html = ""
    if dive.flip_note:
        flip_note_html = (
            '<div class="game-sub"><span class="pill flip-pill">&#8644;</span> '
            f"{escape(dive.flip_note)}</div>"
        )
    tiebreaker_note_html = ""
    if is_last_game:
        tiebreaker_note_html = (
            '<div class="game-sub"><b>Tiebreaker game</b> &mdash; this week\'s last game; the '
            "pool's tiebreaker guess for it is on the board, to the left.</div>"
        )
    star = "&#9733; " if dive.is_best else ""
    best_suffix = " &middot; Best Pick of the week" if dive.is_best else ""
    return (
        f'<div class="dive-panel" id="{escape(dive.game_id)}" '
        f'data-game-id="{escape(dive.game_id)}"{hidden_attr}>'
        '<div class="dive"><div class="dive-head"><div>'
        f'<div class="game-id">{star}{escape(dive.pick_team)} {escape(dive.pick_spread_text)} '
        '<span style="color:var(--text-faint);font-weight:400;">at '
        f"{escape(dive.home)}</span></div>"
        f'<div class="game-sub">{escape(dive.kickoff_group_label)} &middot; cover prob '
        f'<b class="num" style="color:var(--green);">{escape(dive.probability_text)}</b>'
        f"{best_suffix}</div>"
        f"{note_html}"
        f"{flip_note_html}"
        f"{tiebreaker_note_html}"
        "</div>"
        '<span class="sample-tag">Real attribution</span>'
        "</div>"
        f"{_why_this_pick_html(explanation_text)}"
        '<div class="dive-body"><div>'
        f"{_attribution_html(dive)}"
        "</div><div>"
        f'<div class="chart-cap">Cover probability vs. spread &middot; '
        f"{escape(dive.matchup_label)}</div>"
        f"{_game_dive_chart_html(dive)}"
        "</div></div>"
        f"{_lineups_html(dive)}</div></div>"
    )


def _inspector_section(content: BoardContent) -> str:
    """UI-20 layout A: the RIGHT column of the This Week two-column grid --
    one inspector panel per game (see :func:`_dive_panel_html`), all
    present in the markup, selected by the LEFT column's board rows (see
    :func:`_board_section`) rather than by a second, redundant selector of
    its own. This replaces the old standalone "Why this pick" section,
    which carried its own tab-strip selector (``.dive-selector``/
    ``.dive-tab``) below the board -- a second control for the exact same
    choice the board rows already make, and (via that section's 16 panels)
    a second full-width block for content that now belongs beside the
    board instead of under it. Nothing here is rendered a second time
    elsewhere: the explanation text this function feeds each panel used to
    print under every board row (see :func:`_why_this_pick_html`'s
    docstring); it prints once now, inside the matching game's own panel."""

    if not content.dives:
        return ""
    default_game_id = _default_game_id(content)
    row_by_id = {game.game_id: game for game in content.games}
    last_game_id = content.games[-1].game_id if content.games else None
    panels = "".join(
        _dive_panel_html(
            content,
            dive,
            default_game_id=default_game_id,
            explanation_text=(
                row_by_id[dive.game_id].explanation_text if dive.game_id in row_by_id else ""
            ),
            is_last_game=dive.game_id == last_game_id,
        )
        for dive in content.dives
    )
    return (
        '<section aria-labelledby="dive-h" class="inspector-col"><div class="section-head">'
        '<h2 id="dive-h">Game inspector</h2>'
        '<span class="sub">shows the board\'s selected game &middot; why this pick, '
        "lineups, and the spread explorer</span></div>"
        f'<div class="dive-panels">{panels}</div></section>'
    )


def _findings_teaser_section(content: BoardContent) -> str:
    if not content.findings:
        return ""
    cards = "".join(
        f'<div class="find-card"><span class="tag">{escape(finding.tag)}</span>'
        f"<p>{escape(finding.text)}</p>"
        '<a class="more" href="findings.html">Read the writeup</a></div>'
        for finding in content.findings
    )
    return (
        '<section aria-labelledby="find-h"><div class="section-head">'
        '<h2 id="find-h">Findings desk</h2>'
        '<span class="sub"><a href="findings.html">full findings log &rarr;</a></span></div>'
        f'<div class="find-grid">{cards}</div></section>'
    )


def _footer_html(generated_at_text: str, *, model_bit: str) -> str:
    """2026-09-05 (owner, verbatim: "ive told you repeatedly to drop these
    fucking legal bullshit words"): the compliance disclaimer block and the
    gambling-helpline line are REMOVED from every page's footer. The footer
    states only plain, honest facts: when the page was generated and the
    pool's own lock cadence."""

    tail = f" &middot; {escape(model_bit)}" if model_bit else ""
    return (
        "<footer>"
        f'<div class="gen">Generated {escape(_humanize_timestamp(generated_at_text))}{tail} '
        f"&middot; ATS Terminal &middot; {escape(CADENCE_NOTE)}</div>"
        "</footer>"
    )


def _generic_footer(generated_at_text: str, *, model_bit: str = "") -> str:
    """Footer for the two pages that have no per-page fact of their own to
    add beyond the generated-at stamp and model id."""

    return _footer_html(generated_at_text, model_bit=model_bit)


def _footer(content: BoardContent) -> str:
    """The This Week page's own footer."""

    model_bit = f"source model {content.headline.model_method_label}"
    return _footer_html(content.generated_at_text, model_bit=model_bit)


def _link_preview_meta_html(link_preview: LinkPreview) -> str:
    """``og:title``/``og:description``/``og:site_name``/``twitter:card``
    (owner-approved improvement batch, item 10) -- every value but the site
    name (this site's own, already-literal brand, see ``_page_shell``'s
    ``title`` logic) comes off ``link_preview``, the content layer."""

    return (
        f'<meta property="og:title" content="{escape(link_preview.title)}">\n'
        f'<meta property="og:description" content="{escape(link_preview.description)}">\n'
        '<meta property="og:site_name" content="ATS Terminal">\n'
        '<meta name="twitter:card" content="summary">'
    )


def _page_shell(*, page: str, body: str, link_preview: LinkPreview, extra_script: str = "") -> str:
    label = next((label for filename, label, _title in SITE_PAGES if filename == page), page)
    title = "ATS Terminal" if page == PICKS_PAGE else f"ATS Terminal — {label}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
{_link_preview_meta_html(link_preview)}
{_FONT_LINKS}
<style>
{TERMINAL_STYLE_CSS}
</style>
</head>
<body>
{body}
{extra_script}
</body>
</html>
"""


def _page_lead(kicker: str, title: str, sub: str) -> str:
    """The top-of-``<main>`` block every non-index page opens with, since
    these pages are not part of the This Week page's one continuous scroll."""

    return (
        '<div class="page-lead">'
        f'<span class="micro">{escape(kicker)}</span>'
        f"<h1>{escape(title)}</h1>"
        f'<p class="sub">{escape(sub)}</p>'
        "</div>"
    )


def _season_record_strip_html(content: BoardContent) -> str:
    """The hero's running record strip (season mode, item 4): this week,
    season to date, and the Best Pick tracked separately. Renders nothing
    until at least one graded game exists this season -- matches today's
    all-upcoming rendering exactly until then."""

    record = content.season_record
    if record is None:
        return ""
    best_pick_html = (
        f'<span class="record-chip">{escape(record.best_pick_record_text)}</span>'
        if record.best_pick_record_text
        else ""
    )
    return (
        '<div class="season-record-strip">'
        f'<span class="record-chip">{escape(record.week_record_text)}</span>'
        f'<span class="record-chip">{escape(record.season_record_text)}</span>'
        f"{best_pick_html}"
        "</div>"
    )


def render(content: BoardContent, *, page: str = PICKS_PAGE) -> str:
    """Render the full This Week page for ``content``.

    UI-20 layout A (owner, 2026-09-05, choosing it over two other mockups:
    "layout A is definitely the best. lets go with that."): under the
    pinned chrome and the headline strip, the board (:func:`_board_section`)
    and the selected game's inspector (:func:`_inspector_section`) sit side
    by side in one ``.week-grid`` -- at least 700px for the board at
    >=1100px, stacked board-then-inspector below that width (see the
    ``.week-grid`` rules appended to ``board_terminal_style.css``). Findings
    and the assistant panel stay exactly as they were, below the grid.
    """

    body = (
        _terminal_chrome(
            content.ticker_chrome,
            page=page,
            season=content.season,
            week=content.week,
            game_type=content.game_type,
            week_label=content.week_label,
        )
        + '<main class="week-page">'
        + _season_record_strip_html(content)
        + _headline_section(content.headline)
        + '<div class="week-grid">'
        + _board_section(content)
        + _inspector_section(content)
        + "</div>"
        + _findings_teaser_section(content)
        + board_assistant.assistant_section(board_assistant.build_knowledge_for_board(content))
        + "</main>"
        + _footer(content)
    )
    return _page_shell(
        page=page,
        body=body,
        link_preview=content.link_preview,
        extra_script=_DIVE_SCRIPT
        + _SORT_SCRIPT
        + _LINEUP_SCRIPT
        + _TICKER_SCRIPT
        + board_assistant.assistant_script(),
    )


# ---------------------------------------------------------------------------
# The Model page -- merges what used to be two separate pages (Models,
# the earlier model-page draft) into one story: what we play, how it's done, what's
# challenging it. See ``board_site_content.ModelPageContent``'s docstring
# for exactly which duplicate facts were dropped in the merge.
# ---------------------------------------------------------------------------


def _grading_rule_kpi(label: str, value: float | None) -> str:
    text = f"{value:.1%}" if value is not None else "not yet measured"
    return (
        f'<div class="kpi"><span class="label">{escape(label)}</span>'
        f'<span class="value">{escape(text)}</span></div>'
    )


def _ledger_evidence_html(row: ModelLedgerRowView) -> str:
    """Every arm's evidence cell -- fixed to address two 2026-08-31
    browser-QA findings: (1) a promoted row (which never cites outside
    registry evidence -- its own track record above IS its evidence, see
    ``model_ledger._promoted_row``) rendered ``NO CITED EVIDENCE`` in the
    ``.micro`` class's forced uppercase, which read like an indictment
    rather than a fact about the row shape; (2) a challenger row with many
    evidence entries rendered every one of them inline with no cap, so that
    row's height was driven by column-width squeeze rather than its own
    content -- several times taller than any other row on the same table.
    Both are fixed here: a promoted row gets a real provenance line instead,
    and any row with more than :data:`_LEDGER_EVIDENCE_INLINE_LIMIT`
    entries collapses the rest behind a ``<details>`` toggle."""

    if not row.evidence:
        if row.is_promoted:
            provenance = (
                f"Evaluated {_humanize_artifact_ref(row.artifact_ref)}"
                if row.artifact_ref
                else "Its own historical evaluation is above -- see the season-by-season "
                "record and grading rule."
            )
            return f'<span class="game-sub">{escape(provenance)}</span>'
        return '<span class="game-sub">No registry evidence linked yet.</span>'

    def chip(item: LedgerEvidenceItem) -> str:
        pp = (
            _humanize_probability_positive(item.probability_positive)
            if item.probability_positive is not None
            else "not yet scored"
        )
        classification = (
            f" &middot; {escape(_humanize_classification(item.classification))}"
            if item.classification
            else ""
        )
        return (
            f'<span class="pill evidence-pill">{escape(humanize_identifier(item.registry_key))} '
            f"&middot; {pp}{classification}</span>"
        )

    shown = row.evidence[:_LEDGER_EVIDENCE_INLINE_LIMIT]
    rest = row.evidence[_LEDGER_EVIDENCE_INLINE_LIMIT:]
    chips = "".join(chip(item) for item in shown)
    if rest:
        chips += (
            f'<details class="evidence-more"><summary>+{len(rest)} more</summary>'
            f"{''.join(chip(item) for item in rest)}</details>"
        )
    return chips


#: Evidence chips shown inline before collapsing the rest -- see
#: :func:`_ledger_evidence_html`'s docstring for the layout bug this caps.
_LEDGER_EVIDENCE_INLINE_LIMIT = 3


def _ledger_interval_text(row: ModelLedgerRowView) -> str:
    """The ledger's interval cell, unit-aware (2026-08-31 browser-QA fix):
    ``row.interval_unit`` names which of the two units this row's interval
    actually is -- never guessed from the numbers' magnitude. A rate (the
    promoted row's season accuracy-proportion CI) is percent-formatted; an
    accuracy-points effect delta (every challenger row) is rendered as
    signed points, e.g. ``+0.29 to +2.04 pts``, never with a ``%`` sign --
    the bug this guards against rendered that same interval as
    ``[29.0%, 203.8%]``."""

    if row.interval_low is None or row.interval_high is None:
        return "--"
    if row.interval_unit == "accuracy_points":
        return f"{row.interval_low:+.2f} to {row.interval_high:+.2f} pts"
    return f"[{row.interval_low:.1%}, {row.interval_high:.1%}]"


def _model_ledger_row_html(row: ModelLedgerRowView) -> str:
    interval = _ledger_interval_text(row)
    accuracy = f"{row.accuracy:.1%}" if row.accuracy is not None else "--"
    games = f"{row.games:,}" if row.games is not None else "--"
    agreement = (
        f'<div class="game-sub">{escape(row.agreement_text)}</div>' if row.agreement_text else ""
    )
    badge_class = "pill preview" if row.is_promoted else "pill"
    row_class = "game is-best" if row.is_promoted else "game"
    # ``mono-id``: defensive -- most arms have a short mapped display name,
    # but a not-yet-mapped challenger falls back to its raw registry id
    # (``CHALLENGER_DISPLAY_NAMES.get(challenger_id, challenger_id)`` in
    # model_ledger.py), an unbroken mono identifier the same shape as the
    # ones this fix's CSS block wraps.
    return (
        f'<tr class="{row_class}">'
        f'<td data-label="Arm"><b class="mono-id">{escape(row.display_name)}</b><br>'
        f'<span class="{badge_class}">{escape(row.status_badge)}</span></td>'
        f'<td data-label="Grade">{escape(row.grade)}</td>'
        f'<td data-label="Games">{games}</td>'
        f'<td data-label="Accuracy" class="prob">{accuracy}</td>'
        f'<td data-label="Interval">{interval}</td>'
        f'<td data-label="Evidence">{_ledger_evidence_html(row)}</td>'
        f'<td data-label="Summary"><p class="game-sub" style="margin:0;">'
        f"{escape(row.summary_sentence)}</p>{agreement}</td>"
        "</tr>"
    )


def _model_family_row_html(family: FamilyWeightRow) -> str:
    return (
        '<tr class="game">'
        f'<td data-label="Family"><b>{escape(family.label)}</b></td>'
        f'<td data-label="Margin share" class="num">{family.margin_share:.2f}</td>'
        f'<td data-label="Spread share" class="num">{family.spread_share:.2f}</td>'
        f'<td data-label="Weight" class="num">{family.weight_in_spread:.2f}</td>'
        f'<td data-label="Stability">{escape(family.stability_word)}'
        f'<div class="game-sub">{escape(family.stability_detail)}</div></td>'
        f'<td data-label="Classification">{escape(family.classification)}'
        f'<div class="game-sub">{escape(family.caption)}</div></td>'
        "</tr>"
    )


def _season_honesty_sentence(content: ModelPageContent) -> str:
    total = len(content.seasons)
    if total == 0:
        return ""
    sentence = f"{content.seasons_above_coin_flip} of {total} seasons finished above the coin flip"
    if content.seasons_even:
        even_listed = ", ".join(content.seasons_even)
        sentence += f", {len(content.seasons_even)} landed exactly at it ({even_listed})"
    sentence += "."
    if content.seasons_below:
        listed = ", ".join(f"{label} at {value:.1%}" for label, value in content.seasons_below)
        word = "One did not" if len(content.seasons_below) == 1 else "Some did not"
        sentence += f" {word}: {listed}."
    return sentence


def _season_dot_chart_svg(content: ModelPageContent) -> str:
    """Six seasons' opener-graded accuracy as direct-labeled dots on a
    shared axis, with the 50% coin-flip line and the season-blocked
    interval band -- "all six above the coin flip" in one glance (item 8).
    Never hue-only: every dot's own value is printed beside it regardless
    of the tone color, and the season label sits directly under its dot.
    ``""`` when there are no season rows to plot."""

    seasons = content.seasons
    if not seasons:
        return ""
    band = content.long_run_range
    values = [row.opener_accuracy for row in seasons]
    low = min([*values, 0.5, *([band[0]] if band else [])])
    high = max([*values, 0.5, *([band[1]] if band else [])])
    pad = max(0.015, (high - low) * 0.25)
    y_min, y_max = max(0.0, low - pad), min(1.0, high + pad)
    y_span = (y_max - y_min) or 1.0
    count = len(seasons)

    def px(index: int) -> float:
        return 20 + (index + 0.5) / count * 240

    def py(value: float) -> float:
        return 85 - (value - y_min) / y_span * 75

    band_html = ""
    if band is not None:
        y_top, y_bottom = py(band[1]), py(band[0])
        band_html = (
            f'<rect class="season-band" x="20" y="{y_top:.1f}" width="240" '
            f'height="{max(0.0, y_bottom - y_top):.1f}"></rect>'
        )
    ref_y = py(0.5)
    ref_line = (
        f'<line class="ref" x1="20" y1="{ref_y:.1f}" x2="260" y2="{ref_y:.1f}"></line>'
        f'<text x="196" y="{ref_y - 3:.1f}">50% coin flip</text>'
    )
    marks = []
    for index, row in enumerate(seasons):
        cx, cy = px(index), py(row.opener_accuracy)
        tone = "good" if row.opener_accuracy >= 0.5 else "bad"
        marks.append(
            f'<circle class="season-dot {tone}" cx="{cx:.1f}" cy="{cy:.1f}" r="4.2"></circle>'
            f'<text class="season-value" x="{cx:.1f}" y="{cy - 8:.1f}" text-anchor="middle">'
            f"{row.opener_accuracy:.1%}</text>"
            f'<text class="season-label" x="{cx:.1f}" y="96" text-anchor="middle">'
            f"{escape(row.season)}</text>"
        )
    return (
        '<svg class="curve season-chart" viewBox="0 0 280 100" width="100%" height="160" '
        'role="img" aria-label="Opener-graded accuracy by season, each season shown against '
        'the 50% coin flip and the season-by-season confidence band">'
        '<line class="grid" x1="20" y1="10" x2="20" y2="85"></line>'
        '<line class="grid" x1="20" y1="85" x2="260" y2="85"></line>'
        f"{band_html}{ref_line}{''.join(marks)}</svg>"
    )


def _ledger_table_body_html(rows: tuple[ModelLedgerRowView, ...]) -> str:
    rows_html = "".join(_model_ledger_row_html(row) for row in rows)
    return (
        '<div class="board-scroll"><table class="board ledger-fixed"><colgroup>'
        '<col style="width:17%"><col style="width:7%"><col style="width:8%">'
        '<col style="width:9%"><col style="width:12%"><col style="width:22%">'
        '<col style="width:25%"></colgroup><thead><tr>'
        "<th>Arm</th><th>Grade</th><th>Games</th><th>Accuracy</th>"
        "<th>Interval</th><th>Evidence</th><th>Summary</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></div>"
    )


def _grouped_ledger_group_html(title: str, rows: tuple[ModelLedgerRowView, ...]) -> str:
    """One "graded" or "waiting on the season" ledger group (item 9) --
    ``""`` when the group is empty (e.g. a very early season with nothing
    graded yet)."""

    if not rows:
        return ""
    return (
        '<div class="section-head ledger-group-head">'
        f"<h3>{escape(title)}</h3>"
        f'<span class="sub">{len(rows)} arm{"s" if len(rows) != 1 else ""}</span></div>'
        f"{_ledger_table_body_html(rows)}"
    )


def _number_provenance_html(content: ModelPageContent) -> str:
    """The model page's "where these numbers come from" fine print (owner
    mandate, 2026-09-05: "please do not let those percentages get out of
    date anymore") -- one row per headline number
    ``verify_number_provenance`` checked, each dated and model-labeled,
    never fingerprinted (no hashes, per the same mandate). Collapsed by
    default like every other technical aside on this page (the selection
    discount, the model ledger's evidence chips) -- de-firehose discipline,
    not concealment: a reader who wants to check the numbers match opens
    it, everyone else never sees a hash."""

    if content.number_provenance:
        rows_html = "".join(
            '<tr class="game">'
            f'<td data-label="Number">{escape(row.label)}</td>'
            f'<td data-label="Source">{escape(row.artifact_kind)}</td>'
            f'<td data-label="Dated">{escape(row.date_text)}</td>'
            f'<td data-label="Model">{escape(row.model_text)}</td>'
            "</tr>"
            for row in content.number_provenance
        )
        body = (
            '<div class="board-scroll"><table class="board"><thead><tr>'
            "<th>Number</th><th>Source</th><th>Dated</th><th>Model</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table></div>"
        )
    else:
        note = content.number_provenance_note or "not verified for this build."
        body = f'<p class="game-sub">{escape(note)}</p>'
    return (
        '<details class="line-tools" style="margin-top:12px;">'
        "<summary>Where these numbers come from</summary>"
        f'<div style="margin-top:8px;">{body}</div></details>'
    )


def render_model_page(content: ModelPageContent) -> str:
    """Render ``model.html``: what we play (the headline strip, reused
    verbatim from the This Week page), how it's done (season-by-season
    record and grading-rule comparison), and what's challenging it (the
    model ledger and family-weight explanation) -- see
    :class:`~nfl_ats.board_site_content.ModelPageContent`'s docstring for
    exactly which duplicate facts this merge dropped.
    """

    long_run_html = ""
    if content.long_run_range is not None:
        correct_text = (
            f"{content.long_run_correct:,} of {content.long_run_games:,} games"
            if content.long_run_correct is not None and content.long_run_games is not None
            else ""
        )
        long_run_html = (
            '<p class="policy-note">Measured a second way, over full seasons at a time: '
            "95% range "
            f"[{content.long_run_range[0]:.1%}, {content.long_run_range[1]:.1%}]"
            f"{f' over {escape(correct_text)}' if correct_text else ''}.</p>"
        )

    ladder_html = "".join(f"<p>{escape(rung)}</p>" for rung in content.ladder_rungs)

    season_rows_html = "".join(
        '<tr class="game">'
        f'<td data-label="Season">{escape(row.season)}</td>'
        f'<td data-label="Games">{row.games if row.games is not None else "--"}</td>'
        f'<td data-label="Opener" class="prob">{row.opener_accuracy:.1%}</td>'
        '<td data-label="Close">'
        f"{f'{row.close_accuracy:.1%}' if row.close_accuracy is not None else '--'}</td>"
        "</tr>"
        for row in content.seasons
    )

    grading_html = (
        _grading_rule_kpi("Sign rule, opener", content.grading.protocol_opener)
        + _grading_rule_kpi("Sign rule, close", content.grading.protocol_close)
        + _grading_rule_kpi("Rule we play, opener", content.grading.production_opener)
        + _grading_rule_kpi("Rule we play, close", content.grading.production_close)
    )

    if content.ledger_available:
        # Grouped by record status (item 9), each group re-sorted by
        # evidence strength -- see ``board_site_content._grouped_ledger_rows``.
        # table-layout:fixed + an explicit colgroup (2026-08-31 browser-QA
        # fix): auto layout let the Evidence column's long unbreakable
        # registry-key text squeeze Summary down to a sliver, wrapping its
        # prose across a dozen one-word lines and making that row several
        # times taller than any other -- see .ledger-fixed's CSS docstring.
        ledger_body = _grouped_ledger_group_html(
            "Tracked against a record", content.graded_rows
        ) + _grouped_ledger_group_html("Waiting on the season", content.waiting_rows)
    elif content.ledger_error:
        # ``content.ledger_error`` is a raised validator's own exception
        # text (e.g. a challenger's registration failing
        # ``model_ledger.validate_ledger``) -- diagnostic detail for
        # whoever fixes the registration, not reader prose, so it is
        # wrapped in ``<code>`` rather than rewritten (owner mandate,
        # 2026-09-05).
        ledger_body = (
            '<div class="caveat"><span class="caveat-flag">&sect; model ledger unavailable'
            f'</span><p class="game-sub"><code>{escape(content.ledger_error)}</code></p></div>'
        )
    else:
        ledger_body = (
            '<div class="chart-empty">No prospective challenger ledger recorded yet.</div>'
        )

    families_section = ""
    if content.explanation_available and content.families:
        families_rows_html = "".join(_model_family_row_html(family) for family in content.families)
        run_sub = (
            f"measured {escape(_humanize_timestamp(content.run_directory))}"
            if content.run_directory
            else ""
        )
        families_section = (
            '<section aria-labelledby="families-h"><div class="section-head">'
            '<h2 id="families-h">How the model decides</h2>'
            f'<span class="sub">{run_sub}</span></div>'
            '<div class="board-scroll"><table class="board"><thead><tr>'
            "<th>Family</th><th>Margin share</th><th>Spread share</th><th>Weight</th>"
            "<th>Stability</th><th>Classification</th>"
            f"</tr></thead><tbody>{families_rows_html}</tbody></table></div></section>"
        )

    season_chart_html = _season_dot_chart_svg(content)

    body = (
        _terminal_chrome(content.ticker_chrome, page=MODEL_PAGE)
        + "<main>"
        + _page_lead(
            "THE MODEL",
            "What we play, how it's done, what's challenging it",
            "One story: the played policy, its measured record, and every arm tracked against it.",
        )
        + _headline_section(content.headline)
        + '<details class="line-tools" style="margin-top:12px;"><summary>The selection discount, '
        f'in numbers</summary><div style="margin-top:8px;">{ladder_html}</div></details>'
        + f'<p class="policy-note">Realistic ceiling: {escape(content.ceiling_text)}</p>'
        + '<section aria-labelledby="howgood-h"><div class="section-head">'
        '<h2 id="howgood-h">How it&#39;s done</h2>'
        '<span class="sub">season by season, the simple sign rule vs. the rule we actually '
        "play</span></div>"
        f'<div class="kpi-grid">{grading_html}</div>'
        f"{long_run_html}"
        f"{season_chart_html}"
        '<div class="board-scroll"><table class="board"><thead><tr>'
        "<th>Season</th><th>Games</th><th>Opener</th><th>Close</th>"
        f"</tr></thead><tbody>{season_rows_html}</tbody></table></div>"
        f'<p class="policy-note">{escape(_season_honesty_sentence(content))}</p></section>'
        + '<section aria-labelledby="ledger-h"><div class="section-head">'
        '<h2 id="ledger-h">What&#39;s challenging it</h2>'
        f'<span class="sub">{len(content.rows)} arms</span></div>'
        f"{ledger_body}</section>"
        + families_section
        + board_assistant.assistant_section(board_assistant.build_knowledge_for_model(content))
        + _number_provenance_html(content)
        + "</main>"
        + _generic_footer(content.generated_at_text)
    )
    return _page_shell(
        page=MODEL_PAGE,
        body=body,
        link_preview=content.link_preview,
        extra_script=_TICKER_SCRIPT + board_assistant.assistant_script(),
    )


def _history_status_html(row: HistoryPickRow) -> str:
    labels = {
        "settled": "Correct" if row.correct else "Incorrect",
        "push": "Push",
        "pending": "Pending",
    }
    label = labels.get(row.status, row.status.title())
    score = (
        f'<span class="game-sub">{escape(row.score_text)}</span>'
        if row.score_text and row.status in {"settled", "push"}
        else ""
    )
    return f'<span class="outcome outcome-{escape(row.status)}">{escape(label)}</span>{score}'


def _history_pick_row_html(row: HistoryPickRow) -> str:
    season_week = (
        f"{row.season if row.season is not None else '--'} / "
        f"W{row.week if row.week is not None else '--'}"
    )
    best = '<span class="best-flag">Best pick</span>' if row.best_pick else ""
    confidence = f"{row.confidence:.1%}" if row.confidence is not None else "--"
    line = f"{row.decision_home_spread:+g}" if row.decision_home_spread is not None else "--"
    # Wrapped in ``<code>`` (not prose): a raw hash has no natural words to
    # translate, so this stays a literal, technical identifier for anyone
    # cross-checking a specific pick against ``artifacts/active_ats_model
    # .json`` -- shortened so the column doesn't dominate the row (owner
    # mandate, 2026-09-05: no raw hashes in reader-facing text).
    model_id = f"<code>{escape(row.model_id[:8])}</code>" if row.model_id else "--"
    row_class = "game is-best" if row.best_pick else "game"
    return (
        f'<tr class="{row_class}">'
        f'<td data-label="Season / week">{escape(season_week)}</td>'
        f'<td data-label="Matchup">{escape(row.away_team)} at '
        f"<b>{escape(row.home_team)}</b></td>"
        f'<td data-label="Pick"><b>{escape(row.pick_side)}</b> {escape(line)} {best}</td>'
        f'<td data-label="Confidence" class="prob">{confidence}</td>'
        f'<td data-label="Outcome">{_history_status_html(row)}</td>'
        f'<td data-label="Model id"><span class="mono-id">{model_id}</span></td>'
        "</tr>"
    )


def _history_season_grade_row_html(row: SeasonGradeRow) -> str:
    """UI-20(h): one season's opener-vs-close grading. ``row.note`` (only
    set for the archive-gap sentinel row -- see
    ``board_site_content._season_grade_rows``) spans the grade columns with
    an explicit sentence instead of leaving them blank."""

    lead_cells = (
        f'<td data-label="Season">{escape(row.season_label)}</td>'
        f'<td data-label="Games">{row.games if row.games is not None else "--"}</td>'
    )
    if row.note:
        grade_cells = (
            f'<td data-label="Opener vs close" colspan="3">'
            f'<span class="game-sub">{escape(row.note)}</span></td>'
        )
    else:
        grade_cells = (
            f'<td data-label="Opener accuracy" class="prob">{escape(row.opener_text)}</td>'
            f'<td data-label="Close accuracy" class="prob">{escape(row.close_text)}</td>'
            f'<td data-label="Opener minus close" class="prob">{escape(row.delta_text)}</td>'
        )
    return f'<tr class="game">{lead_cells}{grade_cells}</tr>'


def _history_week_grade_row_html(row: HistoryWeekGrade) -> str:
    """UI-20(h): one recorded week's opener-vs-close grading. ``row.note``
    (set whenever one grade -- or, for an unplayed week, neither -- could
    not be computed) spans the grade columns rather than leaving them
    blank."""

    lead_cells = (
        f'<td data-label="Season / week">{row.season} / W{row.week}</td>'
        f'<td data-label="Picks">{row.picks}</td>'
    )
    if row.note:
        grade_cells = (
            f'<td data-label="Opener vs close" colspan="3">'
            f'<span class="game-sub">{escape(row.note)}</span></td>'
        )
    else:
        grade_cells = (
            f'<td data-label="Opener record" class="prob">'
            f"{escape(row.opener_record_text)}</td>"
            f'<td data-label="Close record" class="prob">{escape(row.close_record_text)}</td>'
            f'<td data-label="Opener minus close" class="prob">{escape(row.delta_text)}</td>'
        )
    return f'<tr class="game">{lead_cells}{grade_cells}</tr>'


def _history_grading_section_html(content: HistoryPageContent) -> str:
    """UI-20(h): season- and week-level opener-vs-close grading, side by
    side, with the caption explaining why the two differ and which one the
    pool settles on. Empty (no section at all) until at least one of the
    two tables has a row -- the same dormant-until-real-data discipline
    every other season-mode field on this site already follows."""

    if not content.season_grades and not content.week_grades:
        return ""
    parts = [
        '<section aria-labelledby="history-grading-h"><div class="section-head">'
        '<h2 id="history-grading-h">Opener vs close, side by side</h2>'
        '<span class="sub">the pool\'s decision line compared to the close</span></div>'
    ]
    if content.season_grades:
        season_body = "".join(_history_season_grade_row_html(row) for row in content.season_grades)
        parts.append(
            '<div class="board-scroll"><table class="board"><thead><tr>'
            "<th>Season</th><th>Games</th><th>Opener accuracy</th><th>Close accuracy</th>"
            "<th>Opener minus close</th></tr></thead><tbody>"
            f"{season_body}</tbody></table></div>"
        )
    if content.week_grades:
        week_body = "".join(_history_week_grade_row_html(row) for row in content.week_grades)
        parts.append(
            '<div class="board-scroll"><table class="board"><thead><tr>'
            "<th>Season / week</th><th>Picks</th><th>Opener record</th><th>Close record</th>"
            "<th>Opener minus close</th></tr></thead><tbody>"
            f"{week_body}</tbody></table></div>"
        )
    if content.grade_caption:
        parts.append(f'<p class="policy-note">{escape(content.grade_caption)}</p>')
    parts.append("</section>")
    return "".join(parts)


def _history_assessment_html(row: ChallengerAssessment) -> str:
    record = f"{row.wins}-{row.losses}-{row.pushes}"
    accuracy = f"{row.accuracy:.1%}" if row.accuracy is not None else "--"
    delta = (
        f"{row.delta_accuracy_points:+.2f} pts" if row.delta_accuracy_points is not None else "--"
    )
    if row.probability_positive is not None:
        uncertainty = f"probability_positive {row.probability_positive:.2f}"
    elif row.interval_low is not None and row.interval_high is not None:
        uncertainty = f"uncertainty [{row.interval_low:+.2f}, {row.interval_high:+.2f}] pts"
    else:
        uncertainty = "uncertainty not recorded"
    return (
        '<tr class="game">'
        f'<td data-label="Challenger"><b class="mono-id">{escape(row.display_name)}</b></td>'
        f'<td data-label="Paired games">{row.paired_games:,}</td>'
        f'<td data-label="Record">{record} '
        f'<span class="game-sub">{row.pending:,} pending</span></td>'
        f'<td data-label="Accuracy" class="prob">{accuracy}</td>'
        f'<td data-label="Delta vs active" class="prob">{delta}</td>'
        f'<td data-label="Uncertainty">{escape(uncertainty)}</td>'
        f'<td data-label="Grading basis"><span class="game-sub">'
        f"{escape(row.grading_basis)}</span></td>"
        "</tr>"
    )


def render_history_page(content: HistoryPageContent) -> str:
    """Render ``history.html`` from the primary and prospective ledgers.

    The primary ledger is deliberately allowed to be empty.  Pending rows
    render no scores or result detail; only settled rows can expose outcomes.
    """

    if content.picks:
        picks_body = "".join(_history_pick_row_html(row) for row in content.picks)
        picks_section = (
            '<div class="board-scroll"><table class="board"><thead><tr>'
            "<th>Season / week</th><th>Matchup</th><th>Pick at frozen line</th>"
            "<th>Chosen-side confidence</th><th>Outcome</th><th>Model id</th>"
            f"</tr></thead><tbody>{picks_body}</tbody></table></div>"
        )
    elif content.primary_error:
        picks_section = (
            '<div class="caveat"><span class="caveat-flag">&sect; primary ledger unavailable</span>'
            f"<p>{escape(content.primary_error)}</p></div>"
        )
    else:
        picks_section = (
            '<div class="chart-empty">No recorded model picks yet. The primary '
            "paper-decision ledger currently has 0 rows.</div>"
        )

    if content.challenger_assessments:
        assessment_body = "".join(
            _history_assessment_html(row) for row in content.challenger_assessments
        )
        assessments_section = (
            '<div class="board-scroll"><table class="board"><thead><tr>'
            "<th>Challenger</th><th>Paired games</th><th>Record</th><th>Accuracy</th>"
            "<th>Delta vs active</th><th>Probability / uncertainty</th><th>Grading basis</th>"
            f"</tr></thead><tbody>{assessment_body}</tbody></table></div>"
        )
    else:
        assessments_section = (
            '<div class="chart-empty">No settled prospective challenger games yet. '
            "Assessments will appear after both ledgers contain recorded picks and "
            "outcomes settle.</div>"
        )

    body = (
        _terminal_chrome(content.ticker_chrome, page=HISTORY_PAGE)
        + "<main>"
        + _page_lead(
            "HISTORY",
            "Recorded picks, settled honestly",
            "The primary ledger at its frozen decision/opener line, plus running "
            "prospective challenger assessments.",
        )
        + '<section aria-labelledby="history-picks-h"><div class="section-head">'
        '<h2 id="history-picks-h">Model picks</h2>'
        f'<span class="sub">{len(content.picks)} recorded rows</span></div>'
        f"{picks_section}</section>"
        + _history_grading_section_html(content)
        + '<section aria-labelledby="history-challengers-h"><div class="section-head">'
        '<h2 id="history-challengers-h">Challenger assessment</h2>'
        '<span class="sub">settled prospective scoring</span></div>'
        f"{assessments_section}"
        '<p class="policy-note">Accuracy and deltas use the frozen decision/opener line. '
        "Probability and uncertainty describe evidence; they do not set the played "
        "card. A promotion threshold is a claims bar, not a play decision.</p></section>"
        + board_assistant.assistant_section(board_assistant.build_knowledge_for_history(content))
        + "</main>"
        + _generic_footer(content.generated_at_text)
    )
    return _page_shell(
        page=HISTORY_PAGE,
        body=body,
        link_preview=content.link_preview,
        extra_script=_TICKER_SCRIPT + board_assistant.assistant_script(),
    )


# ---------------------------------------------------------------------------
# What We've Learned -- plain-English findings by verdict, open leads,
# honesty rules, then ONE dense secondary section summarizing the weak-
# signal registry (2026-08-31: replaces the old standalone Signal Ledger
# page -- see ``board_site_content.SignalLedgerSummary``'s docstring).
# ---------------------------------------------------------------------------


def _trace_chip_html(finding: FindingItemView) -> str:
    """The findings trace chip (owner-approved improvement batch, item 2):
    the registry signal a finding traces to, plus its recorded P+ -- e.g.
    ``injury_value_lost · P+ 0.98``. ``""`` when the finding carries no
    trace (evergreen, or no cited key has a measured P+). Deliberately
    isolated to this one helper plus the ``.trace-chip`` CSS block so the
    whole feature can be deleted in one commit if the owner vetoes it."""

    if finding.trace_signal_name is None or finding.trace_probability_positive is None:
        return ""
    return (
        '<span class="trace-chip">'
        f"{escape(humanize_identifier(finding.trace_signal_name))} &middot; "
        f"{_humanize_probability_positive(finding.trace_probability_positive)}</span>"
    )


def _findings_group_html(group: VerdictGroupView) -> str:
    cards = "".join(
        '<div class="find-card">'
        f'<span class="tag">{escape(finding.question)}</span>'
        f"<p><b>{escape(finding.plain_answer)}</b></p>"
        f"<p>{escape(finding.detail)}</p>"
        f"{_trace_chip_html(finding)}"
        "</div>"
        for finding in group.findings
    )
    verdict_id = escape(group.verdict)
    return (
        f'<section aria-labelledby="group-{verdict_id}-h">'
        f'<div class="section-head"><h2 id="group-{verdict_id}-h">{escape(group.title)}</h2>'
        f'<span class="sub">{escape(group.chip_label)}</span></div>'
        f'<p class="policy-note">{escape(group.blurb)}</p>'
        f'<div class="find-grid">{cards}</div></section>'
    )


def _watching_lead_html(lead: WatchingLeadView) -> str:
    # ``lead.description`` is ALREADY a genuine plain-English summary by the
    # time it reaches this view (a curated blurb, a recorded
    # WeakSignal.plain_summary, or the PLAIN_SUMMARY_PENDING placeholder --
    # see ``board_site_content._watching_lead_view``), never the registry's
    # raw research note; rendered as plain prose, not wrapped in ``<code>``
    # (2026-09-05 fix, dashboard humanising follow-up to lane AH's audit:
    # AH's own fix wrapped the raw text in ``<code>`` rather than replacing
    # it, which still reads as machine text -- "this is for humans not the
    # opus autist").
    return (
        '<div class="attr-row"><div><span class="chan">'
        f"{escape(humanize_identifier(lead.name))} &middot; {escape(lead.league)} &middot; "
        f"{escape(lead.seasons_text)}"
        f'</span><div class="chan-sub">{escape(lead.description)}</div></div>'
        f'<div class="pts">{escape(lead.effect_text)}</div>'
        f'<div class="pts">{_humanize_probability_positive(lead.probability_positive)}</div></div>'
    )


def _recent_activity_category_html(group: RecentActivityCategoryView) -> str:
    """One category's worth of "Research this week" lines (dashboard queue
    UI-20(b)), collapsed behind a ``<details>`` toggle -- de-firehose
    discipline again: 100+ entries can land in a single week's window, so
    the category header states the count and the reader opens what they
    want to read, rather than the page dumping every line by default.

    ``entry.plain_summary`` is ALREADY a genuine plain-English summary (or
    the PLAIN_SUMMARY_PENDING placeholder) by the time it reaches this view
    -- ``board_site_content._recent_activity_entry_view`` never falls back
    to the registry's raw description any more (2026-09-05 fix, dashboard
    humanising follow-up to lane AH's audit: AH's own fix wrapped that raw
    text in ``<code>`` rather than replacing it). Rendered as plain prose,
    not ``<code>``, like the other two research-log sections on this page
    (Signal registry, Watching leads)."""

    lines = "".join(
        '<p class="game-sub" style="margin:6px 0;">'
        f"{escape(entry.plain_summary)} &mdash; {escape(entry.effect_text)}. "
        f"{escape(entry.direction_sentence)}"
        + (f' <span class="pill">{escape(entry.closed_label)}</span>' if entry.closed_label else "")
        + "</p>"
        for entry in group.entries
    )
    return (
        '<details class="table-view">'
        f'<summary class="micro" style="cursor:pointer;">'
        f"{escape(humanize_identifier(group.category))} ({len(group.entries)})</summary>"
        f"{lines}</details>"
    )


def _recent_activity_section_html(activity: RecentActivityView) -> str:
    """ "Research this week" (dashboard queue UI-20(b)): everything recorded
    or screened in the registries' own last-``window_days``-days window,
    grouped by category, with the count of entries screened and how many
    resolved. Renders a plain "no new screens" line when the window is
    empty -- never an empty section with nothing to explain."""

    header = (
        '<section aria-labelledby="recentactivity-h"><div class="section-head">'
        '<h2 id="recentactivity-h">Research this week</h2>'
        f'<span class="sub">{activity.screened_count} screened &middot; '
        f"{activity.resolved_count} resolved &middot; last {activity.window_days} days</span>"
        "</div>"
    )
    if activity.is_empty:
        body = '<p class="policy-note">No new screens recorded this week.</p>'
    else:
        body = "".join(_recent_activity_category_html(group) for group in activity.categories)
    return header + body + "</section>"


def _notable_signal_row_html(row: SignalNotableRow) -> str:
    # ``row.name`` is the registry's own machine id, still queryable via
    # ``nfl-ats weak-signals``; ``humanize_identifier`` keeps it out of the
    # rendered text (owner mandate, 2026-09-05). ``row.idea`` is ALREADY a
    # genuine plain-English summary (or the PLAIN_SUMMARY_PENDING
    # placeholder) by the time it reaches this view --
    # ``board_site_content._load_signal_ledger_summary`` never passes the
    # raw registry description any more (2026-09-05 fix, dashboard
    # humanising follow-up to lane AH's audit: AH's own fix wrapped that raw
    # text in ``<code>`` rather than replacing it), so it renders as plain
    # prose, not ``<code>``.
    return (
        '<tr class="game">'
        f'<td data-label="Signal"><b class="mono-id">{escape(humanize_identifier(row.name))}</b>'
        f'<div class="game-sub">{escape(row.idea)}</div></td>'
        f'<td data-label="Effect" class="prob">{escape(row.effect_text)}</td>'
        f'<td data-label="Likely real">{_humanize_probability_positive(row.probability_positive)}'
        "</td>"
        f'<td data-label="Status">{escape(_humanize_classification(row.status))}</td>'
        "</tr>"
    )


def _ledger_summary_section_html(content: FindingsPageContent) -> str:
    summary = content.ledger_summary
    counts_html = "".join(
        f'<div class="kpi"><span class="label">{escape(_humanize_classification(status))}</span>'
        f'<span class="value">{count}</span></div>'
        for status, count in sorted(summary.counts_by_status.items())
    )
    notable_html = "".join(_notable_signal_row_html(row) for row in summary.notable)
    table = (
        '<div class="board-scroll"><table class="board"><thead><tr>'
        "<th>Signal</th><th>Effect</th><th>Likely real</th><th>Status</th>"
        f"</tr></thead><tbody>{notable_html}</tbody></table></div>"
        if summary.notable
        else '<div class="chart-empty">No signal has a recorded confidence figure yet.</div>'
    )
    return (
        '<section aria-labelledby="ledgersummary-h"><div class="section-head">'
        '<h2 id="ledgersummary-h">Signal registry</h2>'
        f'<span class="sub">{summary.total_signals} signals recorded</span></div>'
        f'<div class="kpi-grid">{counts_html}</div>'
        f"{table}"
        '<p class="policy-note">Highest-confidence entries shown above; every recorded '
        "signal (including ones not listed here, whether promising or refuted) stays "
        "queryable via <code>nfl-ats weak-signals</code>. An interval crossing zero is "
        "never grounds to call a signal settled.</p></section>"
    )


def render_findings_page(content: FindingsPageContent) -> str:
    """Render ``findings.html``: hero tiles, curated findings by verdict,
    open leads, honesty rules, and a compact signal-registry summary --
    all off :class:`~nfl_ats.board_site_content.FindingsPageContent`."""

    tiles = "".join(
        '<div class="kpi"><span class="label">'
        f'{escape(tile.kicker)}</span><span class="value">{escape(tile.value)}</span>'
        f'<span class="foot">{escape(tile.context)}</span></div>'
        for tile in content.hero_tiles
    )
    groups_html = "".join(_findings_group_html(group) for group in content.groups)
    leads_html = "".join(_watching_lead_html(lead) for lead in content.watching_leads)
    honesty_html = "".join(
        f'<div class="find-card"><span class="tag">{escape(rule.title)}</span>'
        f"<p>{escape(rule.body)}</p></div>"
        for rule in content.honesty_rules
    )

    body = (
        _terminal_chrome(content.ticker_chrome, page=FINDINGS_PAGE)
        + "<main>"
        + _page_lead(
            "WHAT WE'VE LEARNED",
            "Every finding, in plain words",
            "Each answer traces to a registry entry or is declared evergreen.",
        )
        + f'<div class="kpi-grid">{tiles}</div>'
        + groups_html
        + '<section aria-labelledby="watching-h"><div class="section-head">'
        '<h2 id="watching-h">What we&#39;re watching</h2>'
        f'<span class="sub">{len(content.watching_leads)} of '
        f"{content.ledger_summary.total_signals} recorded signals</span></div>"
        f"{leads_html}</section>"
        + _recent_activity_section_html(content.recent_activity)
        + '<section aria-labelledby="honesty-h"><div class="section-head">'
        '<h2 id="honesty-h">How we keep ourselves honest</h2></div>'
        f'<div class="find-grid">{honesty_html}</div></section>'
        + _ledger_summary_section_html(content)
        + board_assistant.assistant_section(board_assistant.build_knowledge_for_findings(content))
        + "</main>"
        + _generic_footer(content.generated_at_text)
    )
    return _page_shell(
        page=FINDINGS_PAGE,
        body=body,
        link_preview=content.link_preview,
        extra_script=_TICKER_SCRIPT + board_assistant.assistant_script(),
    )


__all__ = [
    "FINDINGS_PAGE",
    "MODEL_PAGE",
    "PICKS_PAGE",
    "SITE_PAGES",
    "TERMINAL_STYLE_CSS",
    "render",
    "render_findings_page",
    "render_model_page",
]
