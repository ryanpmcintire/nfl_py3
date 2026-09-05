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
``index.html`` (This Week) is the approved mockup, unchanged in spirit; its
one content change is folding the old standalone "spread explorer" into
each game's deep dive as a line-offset adjuster, reachable for any game via
a selector defaulting to the Best Pick. ``model.html`` (The Model) and
``findings.html`` (What We've Learned) are original extensions of the same
visual system -- see each ``render_*_page`` function's docstring for what
it merges and why.

The only markup here that is NOT part of the approved mockup is (a) the
small degraded-state blocks the mockup's own CSS sheet reserves space for
(delimited in ``board_terminal_style.css`` with a
``/* degraded states -- ... */`` comment), (b) the game-selector/adjuster
markup added to the deep-dive section, plus its own inline script, and
(c) the retrieval-only board-assistant panel (UI-16,
:mod:`nfl_ats.board_assistant`), plus its own inline script -- none changes
the mockup's own DOM elsewhere, all are additive.
"""

from __future__ import annotations

from html import escape
from itertools import groupby
from pathlib import Path

from nfl_ats import board_assistant
from nfl_ats.board_content import (
    CADENCE_NOTE,
    SOURCE_POLICY_COMPUTED_LIVE_NOTE,
    SOURCE_POLICY_LEGEND,
    BoardContent,
    GameDive,
    GameRow,
    HeadlineStats,
    LinkPreview,
    SourcePolicyView,
    TickerChrome,
)
from nfl_ats.board_site_content import (
    ChallengerAssessment,
    FamilyWeightRow,
    FindingItemView,
    FindingsPageContent,
    HistoryPageContent,
    HistoryPickRow,
    LedgerEvidenceItem,
    ModelLedgerRowView,
    ModelPageContent,
    RecentActivityCategoryView,
    RecentActivityView,
    SignalNotableRow,
    VerdictGroupView,
    WatchingLeadView,
)
from nfl_ats.lineup_view import TeamLineup
from nfl_ats.public_board import DISCLAIMER_FULL, DISCLAIMER_SHORT
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

#: Shared script for the This Week page's per-game selector and its
#: line-offset adjuster. One drag/click handler for every ``.dive-tab`` /
#: ``.ats-adjuster`` widget on the page, mirroring the erf approximation
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
    document.querySelectorAll('.dive-tab').forEach(function (tab) {
      var active = tab.dataset.gameId === gameId;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
    });
  }
  document.querySelectorAll('.dive-tab').forEach(function (tab) {
    tab.addEventListener('click', function () { selectGame(tab.dataset.gameId); });
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
#: ``.dive-tab``/``.dive-panel`` exist), a click selects that game in the
#: deep-dive selector and scrolls to it, and page load re-selects from
#: ``location.hash`` -- both without a page reload. On The Model/Findings
#: (no dive selector on the page), the click is left alone and the browser's
#: ordinary anchor navigation carries the reader to This Week with the hash
#: already set, where the same on-load handler takes over.
_TICKER_SCRIPT = """
<script>
(function () {
  function activateGame(gameId) {
    var tab = document.querySelector('.dive-tab[data-game-id="' + gameId + '"]');
    if (!tab) return null;
    tab.click();
    return document.querySelector('.dive-panel[data-game-id="' + gameId + '"]');
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
  if (document.querySelector('.dive-selector') && location.hash) {
    var hashGameId = decodeURIComponent(location.hash.slice(1));
    var panel = activateGame(hashGameId);
    if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
})();
</script>
"""

#: Progressive enhancement for the compact status rail below the command
#: row. The HTML always starts with the final repository-derived values; this
#: script only rolls integer counters up once for sighted users who have not
#: requested reduced motion. It performs no fetches and invents no live state.
_MOTION_SCRIPT = r"""
<script>
(function () {
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced) return;

  function formatMotionNumber(value, decimals, grouped) {
    var fixed = value.toFixed(decimals);
    if (!grouped) return fixed;
    var parts = fixed.split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return parts.join('.');
  }

  function rollContentNumber(node) {
    if (node.dataset.motionRolled === 'true') return;
    var original = node.textContent.trim();
    var match = original.match(/^([+-]?)([\d,]+)(?:\.(\d+))?(\s*(?:%|pts|x))?$/i);
    if (!match) return;
    var digits = match[2].replace(/,/g, '');
    var decimals = match[3] ? match[3].length : 0;
    var target = Number((match[1] || '') + digits + (match[3] ? '.' + match[3] : ''));
    if (!Number.isFinite(target) || target === 0) return;
    var suffix = match[4] || '';
    var grouped = match[2].indexOf(',') !== -1;
    var distance = Math.max(Math.pow(10, -decimals), Math.abs(target) * 0.06);
    var start = target >= 0 ? target - distance : target + distance;
    var started = null;
    node.dataset.motionRolled = 'true';
    node.setAttribute('aria-label', original);
    function frame(now) {
      if (started === null) started = now;
      var progress = Math.min(1, (now - started) / 720);
      var eased = 1 - Math.pow(1 - progress, 3);
      var current = start + ((target - start) * eased);
      node.textContent = formatMotionNumber(current, decimals, grouped) + suffix;
      if (progress < 1) requestAnimationFrame(frame);
      else node.textContent = original;
    }
    requestAnimationFrame(frame);
  }

  document.querySelectorAll('[data-roll-to]').forEach(function (node) {
    var target = parseInt(node.dataset.rollTo, 10);
    if (!Number.isFinite(target) || target < 0) return;
    var started = null;
    function frame(now) {
      if (started === null) started = now;
      var progress = Math.min(1, (now - started) / 850);
      var eased = 1 - Math.pow(1 - progress, 3);
      node.textContent = String(Math.round(target * eased));
      if (progress < 1) requestAnimationFrame(frame);
      else node.classList.add('roll-complete');
    }
    requestAnimationFrame(frame);
  });

  var motionTargets = document.querySelectorAll(
    'main > .page-lead, main > .season-record-strip, main > .kpi-grid, main > section'
  );
  motionTargets.forEach(function (target) {
    var items = target.querySelectorAll(
      '.kpi, .find-card, table.board tbody tr, .attr-row, .dive-tab'
    );
    items.forEach(function (item, index) {
      item.classList.add('motion-item');
      item.style.setProperty('--motion-delay', String(110 + (Math.min(index, 12) * 38)) + 'ms');
    });
  });

  function reveal(target) {
    target.classList.add('content-motion-visible');
    target.querySelectorAll('.kpi .value, .headline-main .value').forEach(rollContentNumber);
  }
  if (!('IntersectionObserver' in window)) {
    motionTargets.forEach(function (node) {
      reveal(node);
      node.classList.add('content-motion-active');
    });
    return;
  }
  motionTargets.forEach(function (node) { node.classList.add('content-motion-ready'); });
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      reveal(entry.target);
      observer.unobserve(entry.target);
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
  motionTargets.forEach(function (node) { observer.observe(node); });

  var ambientObserver = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      entry.target.classList.toggle('content-motion-active', entry.isIntersecting);
    });
  }, { rootMargin: '8% 0px 8% 0px', threshold: 0 });
  motionTargets.forEach(function (node) { ambientObserver.observe(node); });
})();
</script>
"""


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
        f'<div class="session-meta">{week_tag}'
        '<span class="pill preview">Research preview</span>'
        "</div></header>"
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

    model_id_text = f"{headline.model_id[:8]}&hellip;" if headline.model_id else "unknown"
    synced_text = f" &middot; synced {headline.synced_at_text}" if headline.synced_at_text else ""
    raw_model_ci = (
        f"95% CI <b>[{headline.raw_model_ci[0]:.2f}%, {headline.raw_model_ci[1]:.2f}%]</b>"
        if headline.raw_model_ci is not None
        else "interval not yet published"
    )
    return (
        '<section aria-labelledby="stats-h"><div class="section-head">'
        '<h2 id="stats-h">Headline accuracy</h2>'
        '<span class="sub">source: CURRENT_PREDICTIONS.md &middot; artifacts/active_ats_model.json'
        "</span></div>"
        '<div class="headline-block"><div class="headline-main">'
        '<span class="label">Played policy &middot; archive score</span>'
        f'<span class="value">{headline.played_card_value_text}</span>'
        f'<span class="foot">{escape(headline.played_card_foot_text)}</span>'
        f'<span class="foot">model <b>{model_id_text}</b>{synced_text}</span>'
        "</div>"
        '<div class="caveat">'
        '<span class="caveat-flag">&sect; selection caveat &mdash; read before citing this number'
        "</span>"
        f"<p>{escape(headline.selection_caveat_text)}</p>"
        f"{_prospective_scoreboard_html(headline)}"
        "</div></div>"
        '<div class="kpi-grid">'
        '<div class="kpi"><span class="label">Prior chain &middot; coach &rarr; arrests</span>'
        f'<span class="value muted">{headline.prior_chain_value_text}</span>'
        f'<span class="foot">{escape(headline.prior_chain_caption)}</span></div>'
        '<div class="kpi"><span class="label">Raw model &middot; opener grade baseline</span>'
        f'<span class="value good">{headline.raw_model_value_text}</span>'
        f"{_season_shape_html(headline)}"
        f'<span class="foot">{raw_model_ci} &middot; season-blocked</span>'
        "</div>"
        '<div class="kpi"><span class="label">Active model &middot; close grade</span>'
        f'<span class="value muted">{headline.close_grade_value_text}</span>'
        f'<span class="foot">{escape(headline.close_grade_caption)}</span></div>'
        "</div>"
        '<div class="policy-note" style="margin-top:1px;border-left-color:var(--line);">'
        f"Active model <b>{escape(headline.model_method_label)}</b>"
        + (f", id <b>{headline.model_id}</b>" if headline.model_id else "")
        + (f", synced {escape(headline.synced_at_text)}" if headline.synced_at_text else "")
        + ". Four stats, four roles: headline archive score, the prior chain it's tracked "
        "against, the raw-model baseline it's built on, and the model's own close-graded "
        "classification.</div></section>"
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
    legend -- directly beneath the board's card header (see
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
            f'<span class="src-name">{escape(row.source_id)}</span>'
            '<span class="src-leader" aria-hidden="true"></span>'
            f'<span class="src-state {escape(row.state)}" title="{escape(row.detail_text)}">'
            f"{escape(row.state_label)}</span>"
            f'<span class="src-asof">{escape(row.observed_at_text)}</span>'
            "</div>"
            for row in view.rows
        )
        evaluated_suffix = (
            f" as of {escape(view.evaluated_at_text)}" if view.evaluated_at_text else ""
        )
        live_note = f" {escape(SOURCE_POLICY_COMPUTED_LIVE_NOTE)}" if view.computed_live else ""
        body = (
            f'<div class="src-rows">{rows_html}</div>'
            f'<p class="src-evaluated">Evaluated{evaluated_suffix}.{live_note}</p>'
        )
    legend = f'<p class="src-legend">{escape(SOURCE_POLICY_LEGEND)}</p>'
    return header + body + legend + "</div>"


def _why_this_pick_row_html(game: GameRow) -> str:
    """ "Why this pick" (dashboard queue, ROADMAP.md UI-20(a)): a collapsed
    disclosure directly under each pick row, carrying the ENG-12 explanation
    text (market line, this game's own model probability, fired overlays,
    per-source freshness, and Tuesday-to-refresh status -- already composed
    and language-contract-checked by ``nfl_ats.card_explanation``) or the
    explicit not-recorded sentence. ``<details>`` is collapsed by default,
    so this adds ZERO default-visible percentages to the page -- the same
    de-firehose discipline every other numeric block on this board already
    follows (evidence chips, the spread adjuster). Reuses the existing
    ``.micro``/``.game-sub`` fine-print typography; no new CSS."""

    return (
        f'<tr class="explain" data-game-id="{escape(game.game_id)}">'
        '<td colspan="6">'
        '<details><summary class="micro" style="cursor:pointer;">Why this pick</summary>'
        f'<p class="game-sub" style="margin:8px 0 0;max-width:820px;white-space:normal;">'
        f"{escape(game.explanation_text)}</p>"
        "</details></td></tr>"
    )


def _board_section(content: BoardContent) -> str:
    policy = content.policy
    if policy.rich_narrative:
        policy_html = escape(policy.rich_narrative)
        if policy.policy_id or policy.policy_fingerprint:
            policy_html += f" Policy <b>{escape(policy.policy_id or '')}</b>"
            if policy.policy_fingerprint:
                policy_html += f" ({escape(policy.policy_fingerprint[:8])}&hellip;)."
    else:
        policy_html = escape(policy.composition_text) + "."

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
            if game.final and game.cover_result:
                row_classes.append(f"final-{game.cover_result}")
            conf_cell = _final_outcome_html(game) if game.final else _confidence_meter_html(game)
            rows.append(
                f'<tr class="{" ".join(row_classes)}" data-game-id="{escape(game.game_id)}" '
                f'data-prob="{game.pick_probability:.6f}">'
                f'<td class="kickoff" data-label="Kickoff">{escape(game.kickoff_short_label)}</td>'
                f'<td class="matchup" data-label="Matchup">{escape(game.away)} at '
                f"<b>{escape(game.home)}</b></td>"
                f'<td class="pick" data-label="Pick">{pick_cell}</td>'
                f'<td class="prob" data-label="Cover prob">{escape(game.probability_text)}</td>'
                f'<td class="flipline" data-label="Flips at">{_flip_line_html(game)}</td>'
                f'<td class="conf" data-label="Confidence">{conf_cell}</td>'
                "</tr>"
            )
            rows.append(_why_this_pick_row_html(game))

    table = (
        '<table class="board"><thead><tr>'
        "<th>Kickoff</th><th>Matchup</th><th>Pick</th>"
        '<th><abbr title="Raw model probability oriented to the final policy side. On a flip '
        'this is a mirrored decision-strength score, not a freshly calibrated probability.">'
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
        '<section aria-labelledby="board-h"><div class="section-head">'
        f'<h2 id="board-h">{escape(content.week_label)} board &middot; forced picks</h2>'
        f'<span class="sub">{len(content.games)} games &middot; every pool card played</span>'
        "</div>"
        f"{_source_policy_panel_html(content.source_policy)}"
        f'<div class="policy-note"><b>Policy overlay</b> &mdash; {policy_html}</div>'
        f"{_board_sort_toggle_html()}"
        f'<div class="board-scroll">{table}</div></section>'
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


def _lineup_team_html(lineup: TeamLineup | None) -> str:
    if lineup is None:
        return '<div class="lineup-empty">Projected lineup artifact not published yet.</div>'
    rows_by_unit: dict[str, list[str]] = {"offense": [], "defense": [], "special_teams": []}
    for player in lineup.players:
        # UI-20: the em dash is reserved for rows the availability model
        # genuinely could not score (``probability_source == "unavailable"``,
        # i.e. no gsis_id or no rate) -- every other player, QB or not, now
        # carries a real number.
        probability = (
            f"{player.play_probability:.0%}" if player.play_probability is not None else "—"
        )
        injury = player.injury_status or "no report"
        impact = player.model_impact_note or (
            "not scored by the active model"
            if player.model_role == "context_only"
            else "model input"
        )
        tone = (
            "impact-pos"
            if player.model_impact_points is not None and player.model_impact_points >= 0
            else "impact-neg"
            if player.model_impact_points is not None
            else ""
        )
        # No new colour: an "impact-pos"/"impact-neg" tone already only ever
        # applies to the model's own scored QB input, so the plain,
        # text-faint ".lineup-prob" style a non-tone row falls back to is
        # already the distinct, muted look for every ``availability_model``
        # (or unavailable) row. ``title`` carries the full, honest reason on
        # hover without adding any new visual token.
        prob_title = escape(player.probability_reason or "")
        row = (
            '<div class="lineup-row">'
            f'<div class="lineup-pos">{escape(player.slot)}</div>'
            f'<div class="lineup-player"><b>{escape(player.name)}</b>'
            f"<span>{escape(injury)} &middot; {escape(impact)}</span></div>"
            f'<div class="lineup-prob {tone}" title="{prob_title}">{escape(probability)}</div>'
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
    source = escape(lineup.source or "source unavailable")
    as_of = escape(lineup.as_of or "time unavailable")
    note = f'<div class="lineup-note">{escape(lineup.note)}</div>' if lineup.note else ""
    return (
        f'<div class="lineup-team-head"><b>{escape(lineup.team)}</b>'
        f"<span>{source} &middot; {as_of}</span></div>"
        f'<div class="lineup-status">injury feed: {escape(lineup.injury_status)}</div>'
        f"{note}{''.join(sections)}"
    )


#: UI-20 fine print: what the lineup panel's percentage actually is, and
#: why the QB's own number looks different (colour) from everyone else's
#: (plain). Kept short and printed once per lineup block, not per row.
_LINEUP_PROBABILITY_LEGEND = (
    "% = chance the player is active, from the same availability model the picks use: this "
    "week's own injury designation where one exists, else the position's historical "
    "no-designation base rate. The active model's QB input (coloured by matchup impact) comes "
    "from the forecast instead. Hover a percentage for its exact basis."
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


def _dive_selector_html(dives: tuple[GameDive, ...], default_game_id: str) -> str:
    buttons = []
    for dive in dives:
        active = dive.game_id == default_game_id
        classes = "dive-tab is-active" if active else "dive-tab"
        star = "&#9733; " if dive.is_best else ""
        buttons.append(
            f'<button type="button" class="{classes}" role="tab" '
            f'aria-selected="{"true" if active else "false"}" '
            f'data-game-id="{escape(dive.game_id)}">{star}{escape(dive.pick_team)} '
            f"{escape(dive.pick_spread_text)}</button>"
        )
    return (
        '<div class="dive-selector" role="tablist" aria-label="Choose a game to examine">'
        f"{''.join(buttons)}</div>"
    )


def _dive_panel_html(content: BoardContent, dive: GameDive, *, default_game_id: str) -> str:
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
    star = "&#9733; " if dive.is_best else ""
    best_suffix = " &middot; Best Pick of the week" if dive.is_best else ""
    return (
        f'<div class="dive-panel" id="{escape(dive.game_id)}" '
        f'data-game-id="{escape(dive.game_id)}"{hidden_attr} '
        'role="tabpanel">'
        '<div class="dive"><div class="dive-head"><div>'
        f'<div class="game-id">{star}{escape(dive.pick_team)} {escape(dive.pick_spread_text)} '
        '<span style="color:var(--text-faint);font-weight:400;">at '
        f"{escape(dive.home)}</span></div>"
        f'<div class="game-sub">{escape(dive.kickoff_group_label)} &middot; cover prob '
        f'<b class="num" style="color:var(--green);">{escape(dive.probability_text)}</b>'
        f"{best_suffix}</div>"
        f"{note_html}"
        f"{flip_note_html}"
        "</div>"
        '<span class="sample-tag">Real attribution</span>'
        '</div><div class="dive-body"><div>'
        f"{_attribution_html(dive)}"
        "</div><div>"
        f'<div class="chart-cap">Cover probability vs. spread &middot; '
        f"{escape(dive.matchup_label)}</div>"
        f"{_game_dive_chart_html(dive)}"
        "</div></div>"
        f"{_lineups_html(dive)}</div></div>"
    )


def _dive_section(content: BoardContent) -> str:
    if not content.dives:
        return ""
    default_game_id = content.best_pick_game_id or content.dives[0].game_id
    panels = "".join(
        _dive_panel_html(content, dive, default_game_id=default_game_id) for dive in content.dives
    )
    return (
        '<section aria-labelledby="dive-h"><div class="section-head">'
        '<h2 id="dive-h">Why this pick</h2>'
        '<span class="sub">select a game to examine &middot; attribution, cover curve, '
        "and a line-offset adjuster</span></div>"
        f"{_dive_selector_html(content.dives, default_game_id)}"
        f"{panels}</section>"
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


def _footer_html(
    generated_at_text: str, *, model_bit: str, disclaimer_short: str, disclaimer_full: str
) -> str:
    tail = f" &middot; {escape(model_bit)}" if model_bit else ""
    return (
        "<footer>"
        f'<div class="gen">Generated {escape(generated_at_text)}{tail} '
        f"&middot; ATS Terminal &middot; {escape(CADENCE_NOTE)}</div>"
        f'<div class="disclaimer"><b>{escape(disclaimer_short)}</b><br>'
        f"{escape(disclaimer_full)}</div>"
        "</footer>"
    )


def _generic_footer(generated_at_text: str, *, model_bit: str = "") -> str:
    """Footer for the two pages that have no ``Disclaimer`` of their own on
    their content dataclass: the SAME public disclaimer text the This Week
    page's footer carries, imported directly (``public_board
    .DISCLAIMER_SHORT``/``DISCLAIMER_FULL`` -- never re-typed prose), since
    the disclaimer is site-wide legal boilerplate, not a per-week fact. The
    This Week page itself uses :func:`_footer` below, which reads
    ``content.disclaimer`` instead."""

    return _footer_html(
        generated_at_text,
        model_bit=model_bit,
        disclaimer_short=DISCLAIMER_SHORT,
        disclaimer_full=DISCLAIMER_FULL,
    )


def _footer(content: BoardContent) -> str:
    """The This Week page's own footer. Unlike :func:`_generic_footer`,
    this reads ``content.disclaimer`` -- never the imported
    ``DISCLAIMER_SHORT``/``DISCLAIMER_FULL`` constants directly -- so a
    hand-built :class:`~nfl_ats.board_content.BoardContent` fixture (e.g. in
    tests) can exercise a disclaimer that differs from production's, exactly
    like every other field on this page."""

    model_bit = f"source model {content.headline.model_id}" if content.headline.model_id else ""
    return _footer_html(
        content.generated_at_text,
        model_bit=model_bit or "source model unknown",
        disclaimer_short=content.disclaimer.short,
        disclaimer_full=content.disclaimer.full,
    )


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
    """Render the full This Week page for ``content``."""

    body = (
        _terminal_chrome(
            content.ticker_chrome,
            page=page,
            season=content.season,
            week=content.week,
            game_type=content.game_type,
            week_label=content.week_label,
        )
        + "<main>"
        + _season_record_strip_html(content)
        + _headline_section(content.headline)
        + _board_section(content)
        + _dive_section(content)
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
        + _MOTION_SCRIPT
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
                f"Evaluated via {escape(row.artifact_ref)}"
                if row.artifact_ref
                else "Its own historical evaluation is above -- see the season-by-season "
                "record and grading rule."
            )
            return f'<span class="game-sub">{provenance}</span>'
        return '<span class="game-sub">No registry evidence linked yet.</span>'

    def chip(item: LedgerEvidenceItem) -> str:
        pp = (
            f"P+ {item.probability_positive:.2f}"
            if item.probability_positive is not None
            else "P+ --"
        )
        classification = f" &middot; {escape(item.classification)}" if item.classification else ""
        return (
            f'<span class="pill evidence-pill">{escape(item.registry_key)} '
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
        'the 50% coin flip and the season-blocked confidence band">'
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
            '<p class="policy-note">Also season-blocked (fewer, wider blocks): 95% CI '
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
        + _grading_rule_kpi("Probability rule, opener", content.grading.production_opener)
        + _grading_rule_kpi("Probability rule, close", content.grading.production_close)
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
        ledger_body = (
            '<div class="caveat"><span class="caveat-flag">&sect; model ledger unavailable'
            f"</span><p>{escape(content.ledger_error)}</p></div>"
        )
    else:
        ledger_body = (
            '<div class="chart-empty">No prospective challenger ledger recorded yet.</div>'
        )

    families_section = ""
    if content.explanation_available and content.families:
        families_rows_html = "".join(_model_family_row_html(family) for family in content.families)
        run_sub = f"run {escape(content.run_directory)}" if content.run_directory else ""
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
        '<span class="sub">season by season, sign rule vs. probability rule</span></div>'
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
        + "</main>"
        + _generic_footer(content.generated_at_text)
    )
    return _page_shell(
        page=MODEL_PAGE,
        body=body,
        link_preview=content.link_preview,
        extra_script=_TICKER_SCRIPT + _MOTION_SCRIPT + board_assistant.assistant_script(),
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
    model_id = escape(row.model_id or "--")
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
        extra_script=_TICKER_SCRIPT + _MOTION_SCRIPT + board_assistant.assistant_script(),
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
        f"{escape(finding.trace_signal_name)} &middot; P+ "
        f"{finding.trace_probability_positive:.2f}</span>"
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
    return (
        '<div class="attr-row"><div><span class="chan">'
        f"{escape(lead.name)} &middot; {escape(lead.league)} &middot; {escape(lead.seasons_text)}"
        f'</span><div class="chan-sub">{escape(lead.description)}</div></div>'
        f'<div class="pts">{escape(lead.effect_text)}</div>'
        f'<div class="pts">P+ {lead.probability_positive:.2f}</div></div>'
    )


def _recent_activity_category_html(group: RecentActivityCategoryView) -> str:
    """One category's worth of "Research this week" lines (dashboard queue
    UI-20(b)), collapsed behind a ``<details>`` toggle -- de-firehose
    discipline again: 100+ entries can land in a single week's window, so
    the category header states the count and the reader opens what they
    want to read, rather than the page dumping every line by default."""

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
        f"{escape(group.category)} ({len(group.entries)})</summary>"
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
    # registry signal names are unbroken mono identifiers with no natural
    # wrap point (e.g. "odds_microstructure_H3_3_0a_full_week_oracle_
    # 2020_2025_sanity_check") -- ``mono-id`` carries the mobile-overflow
    # wrap rule (see the "mobile-width overflow fix" CSS block).
    return (
        '<tr class="game">'
        f'<td data-label="Signal"><b class="mono-id">{escape(row.name)}</b>'
        f'<div class="game-sub">{escape(row.idea)}</div></td>'
        f'<td data-label="Effect" class="prob">{escape(row.effect_text)}</td>'
        f'<td data-label="P+">{row.probability_positive:.2f}</td>'
        f'<td data-label="Status">{escape(row.status)}</td>'
        "</tr>"
    )


def _ledger_summary_section_html(content: FindingsPageContent) -> str:
    summary = content.ledger_summary
    counts_html = "".join(
        f'<div class="kpi"><span class="label">{escape(status.replace("_", " "))}</span>'
        f'<span class="value">{count}</span></div>'
        for status, count in sorted(summary.counts_by_status.items())
    )
    notable_html = "".join(_notable_signal_row_html(row) for row in summary.notable)
    table = (
        '<div class="board-scroll"><table class="board"><thead><tr>'
        "<th>Signal</th><th>Effect</th><th>P+</th><th>Status</th>"
        f"</tr></thead><tbody>{notable_html}</tbody></table></div>"
        if summary.notable
        else '<div class="chart-empty">No signal has a recorded P+ yet.</div>'
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
        extra_script=_TICKER_SCRIPT + _MOTION_SCRIPT + board_assistant.assistant_script(),
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
