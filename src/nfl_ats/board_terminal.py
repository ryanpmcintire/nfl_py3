from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from itertools import groupby
from pathlib import Path

from nfl_ats import board_assistant
from nfl_ats.board_content import (
    CADENCE_NOTE,
    INJURY_STATE_NAME,
    REFRESH_POLICY_NOTE,
    RIVAL_RULES_TITLE,
    SOURCE_POLICY_COMPUTED_LIVE_NOTE,
    SOURCE_POLICY_LEGEND,
    WEEK_CHANGES_TITLE,
    BoardContent,
    GameDive,
    GameRow,
    HeadlineStats,
    LinkPreview,
    RivalRulesPanel,
    SourcePolicyView,
    TickerChrome,
    TiebreakerView,
)
from nfl_ats.board_site_content import (
    SEASON_SO_FAR_TITLE,
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
    RecentActivityEntryView,
    RecentActivityView,
    SeasonChallengerRecord,
    SeasonGradeRow,
    SeasonSoFar,
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


PICKS_PAGE = "index.html"
MODEL_PAGE = "model.html"
FINDINGS_PAGE = "findings.html"
HISTORY_PAGE = "history.html"

SITE_PAGES: tuple[tuple[str, str, str], ...] = (
    (PICKS_PAGE, "This week", "This week's picks"),
    (MODEL_PAGE, "The model", "The model"),
    (HISTORY_PAGE, "History", "History"),
    (FINDINGS_PAGE, "What we've learned", "What we've learned"),
)

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
    var marker = widget.parentElement.querySelector('.adjuster-marker');
    var center = parseFloat(widget.dataset.center);
    var mean = parseFloat(widget.dataset.mean);
    var std = parseFloat(widget.dataset.std);
    var cardLine = parseFloat(widget.dataset.cardLine);
    var pickIsHome = widget.dataset.pickIsHome === '1';
    // A line sitting exactly on 3 or 7 is read off how games like it really
    // finished, not the curve: at the quoted line show that served number.
    var pinnedP = widget.dataset.pinnedP === undefined ? NaN : parseFloat(widget.dataset.pinnedP);
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
      if (offset === 0 && !isNaN(pinnedP)) homeP = pinnedP;
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


def _nav_links(page: str) -> str:

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

    method_arg = escape(chrome.model_method_label.split(" ", 1)[0])
    suffix = f" {escape(chrome.page_command_suffix)}" if chrome.page_command_suffix else ""
    return (
        '<div class="cmd-row"><span class="prompt">&gt;</span><span>nfl-ats board</span>'
        f'<span class="arg">--season {chrome.season} --week {chrome.week} '
        f"--model {method_arg}{suffix}</span>"
        '<span class="cursor"></span></div>'
    )


def _motion_status_rail(chrome: TickerChrome) -> str:

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
        '<div class="headline-block" style="margin-top:1px;">'
        '<div class="headline-main">'
        '<span class="label">With the line-move and injury rules applied through the week</span>'
        f'<span class="value">{headline.refresh_chain_value_text}</span>'
        f'<span class="foot">{escape(headline.refresh_chain_foot_text)}</span></div>'
        '<div class="caveat"><span class="caveat-flag">What this second number is</span>'
        f"<p>{escape(headline.refresh_chain_caption)}</p></div></div>"
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

    return (
        f'<div class="outcome outcome-{escape(game.cover_result or "")}">'
        f'<span class="outcome-score">{escape(game.final_score_text or "Final")}</span>'
        f'<span class="outcome-word">{escape(game.cover_result_label)}</span></div>'
    )


def _flip_pill_html(game: GameRow) -> str:

    if not game.flip_member_labels:
        return ""
    return f'<span class="pill flip-pill">{escape(game.flip_pill_text)}</span>'


def _lock_html(lock_text: str | None) -> str:

    if not lock_text:
        return ""
    return f'<span class="lock">{escape(lock_text)}</span>'


def _lock_sub_html(lock_text: str | None) -> str:
    if not lock_text:
        return ""
    return f" &middot; {escape(lock_text)}"


def _flip_line_html(game: GameRow) -> str:

    if game.final or not game.flip_line_text:
        return "<span class='flip-none'>&mdash;</span>"
    if game.flip_line is None and game.flip_held:
        pinned_by_rule = [label for label in game.flip_member_labels if label != "spread-gap zone"]
        if pinned_by_rule:
            reason = (
                f"The {' + '.join(pinned_by_rule)} rule backs {game.pick_team} whichever side "
                "the model leans, so no spread between those two lines changes this pick"
            )
        else:
            reason = "No spread between those two lines changes this pick"
        return (
            f"<span class='flip-none' title='{escape(reason, quote=True)}'>"
            f"{escape(game.flip_line_text)}</span>"
        )
    if game.flip_reason == "spread-gap zone":
        reason = (
            "Rule-driven, not the model changing its mind: the spread-gap rule fades the "
            "model's pick whenever the spread is between 7.5 and 10, and this line is where "
            "it starts or stops applying"
        )
        return (
            f"<span class='flip-rule' title='{escape(reason, quote=True)}'>"
            f"{escape(game.flip_line_text)}</span>"
        )
    return escape(game.flip_line_text)


def _board_sort_toggle_html() -> str:

    return (
        '<div class="sort-toggle" role="group" aria-label="Sort the board">'
        '<button type="button" class="sort-btn is-active" data-sort="kickoff" '
        'aria-pressed="true">Kickoff</button>'
        '<button type="button" class="sort-btn" data-sort="confidence" '
        'aria-pressed="false">Confidence</button>'
        "</div>"
    )


def _source_policy_panel_html(view: SourcePolicyView) -> str:

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
            f'<span class="src-state {escape(row.state)}" '
            f'title="{escape(row.neutral_note or row.detail_text)}">'
            f"{escape(row.state_label)}</span>"
            '<span class="src-asof">'
            f"{escape(row.neutral_note or _relative_update(row.observed_at, view.evaluated_at))}"
            "</span>"
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
    legend = f'<p class="src-legend">{escape(SOURCE_POLICY_LEGEND)}</p>'
    return header + body + legend + "</div>"


def _tiebreaker_panel_html(view: TiebreakerView) -> str:

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


def _injury_state_html(content: BoardContent) -> str:

    return (
        f'<p class="policy-note"><b>{escape(INJURY_STATE_NAME)}</b> '
        f'<span class="src-state {escape(content.injury_state_class)}">'
        f"{escape(content.injury_state_label)}</span> &mdash; "
        f"{escape(content.injury_note)}"
        + (f" {escape(content.injury_coverage_note)}" if content.injury_coverage_note else "")
        + "</p>"
    )


def _why_this_pick_html(explanation_text: str) -> str:

    return (
        '<div class="policy-note" style="margin:0 18px 14px;">'
        f"<b>Why this pick</b> &mdash; {escape(explanation_text)}</div>"
    )


_CLASSIFICATION_WORDS: dict[str, str] = {
    "unresolved_below_power": "not enough evidence yet",
    "refuted_mechanism": "ruled out",
    "bounded_by_control": "ruled out by a control test",
}


def _humanize_classification(value: str) -> str:
    return _CLASSIFICATION_WORDS.get(value, humanize_identifier(value))


def _humanize_probability_positive(value: float) -> str:

    return f"{value:.0%} likely real"


def _humanize_artifact_ref(ref: str) -> str:

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

    parsed = _parse_render_time(raw)
    if parsed is None:
        return "time unavailable"
    period = "morning" if parsed.hour < 12 else "afternoon" if parsed.hour < 18 else "evening"
    return f"{parsed:%A} {period}"


def _default_game_id(content: BoardContent) -> str:

    if content.best_pick_game_id is not None:
        return content.best_pick_game_id
    if content.dives:
        return content.dives[0].game_id
    if content.games:
        return content.games[0].game_id
    return ""


def _board_section(content: BoardContent) -> str:

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
                f'<td class="kickoff" data-label="Kickoff">{escape(game.kickoff_short_label)}'
                f"{_lock_html(game.lock_text)}</td>"
                f'<td class="matchup" data-label="Matchup">{matchup_cell}</td>'
                f'<td class="pick" data-label="Pick">{pick_cell}</td>'
                f'<td class="prob" data-label="Cover chance">{escape(game.probability_text)}</td>'
                f'<td class="flipline" data-label="Flips at">{_flip_line_html(game)}</td>'
                f'<td class="conf" data-label="Confidence">{conf_cell}</td>'
                "</tr>"
            )

    table = (
        '<table class="board"><thead><tr>'
        "<th>Kickoff</th><th>Matchup</th><th>Pick</th>"
        "<th><abbr title=\"The computer's own chance that this side covers, adjusted for how "
        "it has actually done on spreads this size. Big favourites and big underdogs have "
        "been its weak spot, so a very confident-looking number there is pulled back toward "
        'what it has really hit.">'
        "Cover&nbsp;chance</abbr></th>"
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
        f"{_injury_state_html(content)}"
        f"{_tiebreaker_panel_html(content.tiebreaker)}"
        f'<div class="policy-note"><b>Policy overlay</b> &mdash; {policy_html}</div>'
        f'<p class="policy-note">{escape(REFRESH_POLICY_NOTE)}</p>'
        + (
            f'<p class="policy-note pick-lock-note">{escape(content.pick_lock_note)}</p>'
            if content.pick_lock_note
            else ""
        )
        + f"{_source_policy_panel_html(content.source_policy)}</section>"
    )


def _adjuster_html(
    dive: GameDive, *, x_min: float, x_max: float, y_min: float, y_max: float
) -> str:

    if dive.adjuster is None:
        return (
            '<p class="adjuster-empty">Line-offset adjuster unavailable for this game -- this '
            "build's active model has no closed-form probability read.</p>"
        )
    adjuster = dive.adjuster
    pinned = (
        f'data-pinned-p="{adjuster.pinned_home_cover_probability:.6f}" '
        if adjuster.pinned_home_cover_probability is not None
        else ""
    )
    return (
        f'<div class="ats-adjuster" data-center="{adjuster.center:.6f}" '
        f'data-mean="{adjuster.residual_mean:.6f}" data-std="{adjuster.residual_std:.6f}" '
        f'data-card-line="{adjuster.card_line:.3f}" {pinned}'
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

    if not dive.cover_curve:
        return (
            '<div class="chart-empty">Cover curve not published for this game on this '
            "artifact tree.</div>"
        )
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
        f"{best_suffix}{_lock_sub_html(dive.lock_text)}</div>"
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


def _week_change_side_html(team: str, best: bool, *, ahead: bool = False) -> str:

    star = '<span class="star">&#9733;</span>' if best else ""
    flag = '<span class="flip-pill">not in the picks table yet</span>' if ahead else ""
    return f"{star}{escape(team)}{flag}"


def _week_changes_section(content: BoardContent) -> str:

    panel = content.week_changes
    count = f'<span class="sub">{escape(panel.count_text)}</span>' if panel.count_text else ""
    head = (
        '<section aria-labelledby="changed-h"><div class="section-head">'
        f'<h2 id="changed-h">{escape(WEEK_CHANGES_TITLE)}</h2>{count}</div>'
        f'<p class="policy-note">{escape(panel.summary)}</p>'
    )
    if not panel.rows:
        return head + "</section>"
    body = "".join(
        '<tr class="change-row">'
        f'<td class="kickoff" data-label="When">{escape(row.when_text)}</td>'
        f'<td class="matchup" data-label="Game"><b>{escape(row.matchup)}</b></td>'
        f'<td class="pick" data-label="Was">'
        f"{_week_change_side_html(row.was_team, row.was_best)}</td>"
        f'<td class="pick" data-label="Now">'
        f"{_week_change_side_html(row.now_team, row.now_best, ahead=row.ahead_of_board)}</td>"
        f'<td data-label="Why"><span class="game-sub">{escape(row.reason)}</span></td>'
        "</tr>"
        for row in panel.rows
    )
    catch_up = (
        f'<p class="policy-note">{escape(panel.catch_up_note)}</p>' if panel.catch_up_note else ""
    )
    return (
        head + '<div class="board-scroll"><table class="board"><thead><tr>'
        "<th>When</th><th>Game</th><th>Was</th><th>Now</th><th>Why</th>"
        f"</tr></thead><tbody>{body}</tbody></table></div>{catch_up}"
        f'<p class="micro">{escape(panel.method_note)}</p></section>'
    )


def _rival_rules_section(content: BoardContent) -> str:

    panel: RivalRulesPanel = content.rivals
    if not panel.recorded:
        return (
            '<section aria-labelledby="rivals-h"><div class="section-head">'
            f'<h2 id="rivals-h">{escape(RIVAL_RULES_TITLE)}</h2></div>'
            f'<p class="policy-note">{escape(panel.summary)}</p></section>'
        )
    body = "".join(
        '<tr class="game">'
        f'<td data-label="Rule"><b>{escape(row.name)}</b></td>'
        f'<td data-label="Picks differently on" class="num">{escape(row.differs_text)}</td>'
        f'<td data-label="Games">{escape(row.games_text)}</td>'
        f'<td data-label="Record so far">{escape(row.record_text)}</td>'
        "</tr>"
        for row in panel.rows
    )
    table = (
        '<div class="board-scroll"><table class="board">'
        "<thead><tr><th>Rule</th><th>Picks differently on</th><th>Games</th>"
        "<th>Record so far</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )
    single = (
        f'<p class="micro">{escape(panel.single_game_line)}</p>' if panel.single_game_line else ""
    )
    return (
        '<section aria-labelledby="rivals-h"><div class="section-head">'
        f'<h2 id="rivals-h">{escape(RIVAL_RULES_TITLE)}</h2>'
        f'<span class="sub">{escape(panel.count_text)}</span></div>'
        f'<p class="policy-note">{escape(panel.summary)}</p>'
        '<details class="policy-note"><summary class="micro" style="cursor:pointer;">'
        "Where each one differs</summary>"
        f"{table}{single}"
        f'<p class="micro">{escape(panel.method_note)}</p>'
        "</details></section>"
    )


def _footer_html(generated_at_text: str, *, model_bit: str) -> str:
    tail = f" &middot; {escape(model_bit)}" if model_bit else ""
    return (
        "<footer>"
        f'<div class="gen">Generated {escape(_humanize_timestamp(generated_at_text))}{tail} '
        f"&middot; ATS Terminal &middot; {escape(CADENCE_NOTE)}</div>"
        "</footer>"
    )


def _generic_footer(generated_at_text: str, *, model_bit: str = "") -> str:

    return _footer_html(generated_at_text, model_bit=model_bit)


def _footer(content: BoardContent) -> str:

    model_bit = f"source model {content.headline.model_method_label}"
    return _footer_html(content.generated_at_text, model_bit=model_bit)


def _link_preview_meta_html(link_preview: LinkPreview) -> str:

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

    return (
        '<div class="page-lead">'
        f'<span class="micro">{escape(kicker)}</span>'
        f"<h1>{escape(title)}</h1>"
        f'<p class="sub">{escape(sub)}</p>'
        "</div>"
    )


def _season_record_strip_html(content: BoardContent) -> str:

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


def _week_timeline_panel(content: BoardContent) -> str:

    return (
        '<p class="policy-note week-line" id="week-timeline-h">'
        f"{escape(content.week_timeline.summary)}</p>"
    )


def render(content: BoardContent, *, page: str = PICKS_PAGE) -> str:

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
        + _week_timeline_panel(content)
        + '<div class="week-grid">'
        + _board_section(content)
        + _inspector_section(content)
        + "</div>"
        + _week_changes_section(content)
        + _rival_rules_section(content)
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


def _grading_rule_kpi(label: str, value: float | None) -> str:
    text = f"{value:.1%}" if value is not None else "not yet measured"
    return (
        f'<div class="kpi"><span class="label">{escape(label)}</span>'
        f'<span class="value">{escape(text)}</span></div>'
    )


def _ledger_evidence_html(row: ModelLedgerRowView) -> str:

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
            '<span class="pill evidence-pill">Supporting comparison '
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


_LEDGER_EVIDENCE_INLINE_LIMIT = 3


def _ledger_interval_text(row: ModelLedgerRowView) -> str:

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
    season_record = (
        f'<div class="game-sub">{escape(row.season_record_text)}</div>'
        if row.season_record_text
        else ""
    )
    return (
        f'<tr class="{row_class}">'
        f'<td data-label="Arm"><b class="mono-id">{escape(row.display_name)}</b><br>'
        f'<span class="{badge_class}">{escape(row.status_badge)}</span>'
        f"{season_record}</td>"
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

    if not rows:
        return ""
    return (
        '<div class="section-head ledger-group-head">'
        f"<h3>{escape(title)}</h3>"
        f'<span class="sub">{len(rows)} arm{"s" if len(rows) != 1 else ""}</span></div>'
        f"{_ledger_table_body_html(rows)}"
    )


def _number_provenance_html(content: ModelPageContent) -> str:

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
        ledger_body = _grouped_ledger_group_html(
            "Tracked against a record", content.graded_rows
        ) + _grouped_ledger_group_html("Waiting on the season", content.waiting_rows)
    elif content.ledger_error:
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

    from nfl_ats.model_weak_spots import (
        BUCKET_NOTE,
        DISPLAYED_CONFIDENCE_NOTE,
        HOME_CORRECTION_LEAD,
        HOME_CORRECTION_NOTE,
        HOME_CORRECTION_UNAVAILABLE,
        HOME_SPLIT_NOTE,
        SEASON_TIMING_LEAD,
        SEASON_TIMING_UNAVAILABLE,
        UNAVAILABLE,
    )

    weak_spots_html = (
        '<section aria-labelledby="weak-spots-h"><div class="section-head">'
        '<h2 id="weak-spots-h">Where the model is weak</h2></div>'
    )
    if content.weak_spots.rows:
        headers = (
            "Spread size",
            "Games",
            "Model right",
            "Took favourite",
            "Favourite covered",
            "Stated confidence",
            "Right: favourite picks",
            "Right: underdog picks",
        )
        weak_spots_html += (
            f'<p class="policy-note">{escape(content.weak_spots.explanation)}</p>'
            '<div class="board-scroll"><table class="board"><thead><tr>'
            + "".join(f"<th>{escape(header)}</th>" for header in headers)
            + "</tr></thead><tbody>"
            + "".join(
                '<tr class="game">'
                + "".join(
                    f'<td data-label="{escape(header)}">{escape(cell)}</td>'
                    for header, cell in zip(headers, row.cells, strict=True)
                )
                + "</tr>"
                for row in content.weak_spots.rows
            )
            + "</tbody></table></div>"
            + f'<p class="policy-note">{escape(BUCKET_NOTE)}</p>'
            + f'<p class="policy-note">{escape(DISPLAYED_CONFIDENCE_NOTE)}</p>'
            + "".join(
                f'<p class="policy-note">{escape(row.reliability)}</p>'
                for row in content.weak_spots.rows
            )
        )
    else:
        weak_spots_html += f'<p class="policy-note">{escape(UNAVAILABLE)}</p>'
    weak_spots_html += (
        '<div class="section-head ledger-group-head">'
        '<h3 id="weak-spots-home-h">Home favourite or home underdog</h3>'
        f'<span class="sub">{len(content.weak_spots.home_split)} rows</span></div>'
    )
    if content.weak_spots.home_split:
        split_headers = (
            "Spread size",
            "Home team was",
            "Games",
            "Home team covered",
            "Model expected home to cover",
            "Model right",
        )
        weak_spots_html += (
            f'<p class="policy-note">{escape(content.weak_spots.home_split_lead)}</p>'
            '<div class="board-scroll"><table class="board"><thead><tr>'
            + "".join(f"<th>{escape(header)}</th>" for header in split_headers)
            + "</tr></thead><tbody>"
            + "".join(
                '<tr class="game">'
                + "".join(
                    f'<td data-label="{escape(header)}">{escape(cell)}</td>'
                    for header, cell in zip(split_headers, row.cells, strict=True)
                )
                + "</tr>"
                for row in content.weak_spots.home_split
            )
            + "</tbody></table></div>"
            + f'<p class="policy-note">{escape(HOME_SPLIT_NOTE)}</p>'
        )
    else:
        weak_spots_html += f'<p class="policy-note">{escape(UNAVAILABLE)}</p>'
    correction = content.weak_spots.home_correction
    weak_spots_html += (
        '<div class="section-head ledger-group-head">'
        '<h3 id="weak-spots-push-h">The home-side push</h3>'
        f'<span class="sub">{len(correction.rows) if correction else 0} rows</span></div>'
    )
    if correction is not None and correction.rows:
        push_headers = (
            "Spread size",
            "This week's push (points)",
            "Learned from games",
            "Archive games",
            "Picks it changed",
            "Right with the push",
            "Right without it",
        )
        weak_spots_html += (
            f'<p class="policy-note">{escape(HOME_CORRECTION_LEAD)}</p>'
            '<div class="board-scroll"><table class="board"><thead><tr>'
            + "".join(f"<th>{escape(header)}</th>" for header in push_headers)
            + "</tr></thead><tbody>"
            + "".join(
                '<tr class="game">'
                + "".join(
                    f'<td data-label="{escape(header)}">{escape(cell)}</td>'
                    for header, cell in zip(push_headers, row.cells, strict=True)
                )
                + "</tr>"
                for row in correction.rows
            )
            + "</tbody></table></div>"
            + '<p class="policy-note">'
            + f"{escape(correction.summary)} {escape(HOME_CORRECTION_NOTE)}</p>"
        )
    else:
        weak_spots_html += f'<p class="policy-note">{escape(HOME_CORRECTION_UNAVAILABLE)}</p>'
    season_timing = content.weak_spots.season_timing
    weak_spots_html += (
        '<div class="section-head ledger-group-head">'
        '<h3 id="weak-spots-season-h">Early season vs. late season</h3>'
        f'<span class="sub">{len(season_timing.rows)} rows</span></div>'
    )
    if season_timing.available:
        season_headers = (
            "Time of season",
            "Games",
            "Model right",
            "Resampled range",
            "Stated confidence",
        )
        weak_spots_html += (
            f'<p class="policy-note">{escape(SEASON_TIMING_LEAD)}</p>'
            '<div class="board-scroll"><table class="board"><thead><tr>'
            + "".join(f"<th>{escape(header)}</th>" for header in season_headers)
            + "</tr></thead><tbody>"
            + "".join(
                '<tr class="game">'
                + "".join(
                    f'<td data-label="{escape(header)}">{escape(cell)}</td>'
                    for header, cell in zip(season_headers, row.cells, strict=True)
                )
                + "</tr>"
                for row in season_timing.rows
            )
            + "</tbody></table></div>"
            + f'<p class="policy-note">{escape(season_timing.summary)}</p>'
            + f'<p class="policy-note">{escape(season_timing.plain)}</p>'
        )
    else:
        weak_spots_html += f'<p class="policy-note">{escape(SEASON_TIMING_UNAVAILABLE)}</p>'
    weak_spots_html += "</section>"

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
        + weak_spots_html
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
    line = row.pick_line_text
    row_class = "game is-best" if row.best_pick else "game"
    return (
        f'<tr class="{row_class}">'
        f'<td data-label="Season / week">{escape(season_week)}</td>'
        f'<td data-label="Matchup">{escape(row.away_team)} at '
        f"<b>{escape(row.home_team)}</b></td>"
        f'<td data-label="Pick"><b>{escape(row.pick_team)}</b> {escape(line)} {best}</td>'
        f'<td data-label="Confidence" class="prob">{confidence}</td>'
        f'<td data-label="Outcome">{_history_status_html(row)}</td>'
        "</tr>"
    )


def _history_season_grade_row_html(row: SeasonGradeRow) -> str:

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


def _season_so_far_challenger_row_html(row: SeasonChallengerRecord) -> str:
    return (
        '<tr class="game">'
        f'<td data-label="Rule"><b>{escape(row.display_name)}</b></td>'
        f'<td data-label="This season">{escape(row.record_text)}</td>'
        f'<td data-label="Against the card">'
        f'<span class="game-sub">{escape(row.versus_card_text)}</span></td>'
        "</tr>"
    )


def _season_so_far_section_html(block: SeasonSoFar | None) -> str:

    if block is None:
        return ""
    season_label = str(block.season) if block.season is not None else "this season"
    head = (
        '<section aria-labelledby="season-so-far-h"><div class="section-head">'
        f'<h2 id="season-so-far-h">{escape(SEASON_SO_FAR_TITLE)}</h2>'
        f'<span class="sub">{escape(season_label)}, counting only games that have '
        "finished</span></div>"
    )
    if not block.has_rows:
        return (
            f"{head}"
            f'<div class="chart-empty">{escape(block.nothing_settled_text or "")}</div>'
            "</section>"
        )
    games = "game" if block.settled_games == 1 else "games"
    card_foot = f"{block.settled_games} {games} finished"
    if block.accuracy_text:
        card_foot = f"{block.accuracy_text} &middot; {card_foot}"
    parts = [
        head,
        '<div class="kpi-grid">'
        '<div class="kpi"><span class="label">The card&rsquo;s picks</span>'
        f'<span class="value">{escape(block.card_record_text)}</span>'
        f'<span class="foot">{card_foot}</span></div>'
        '<div class="kpi"><span class="label">Strongest pick of the week</span>'
        f'<span class="value">{escape(block.best_pick_record_text)}</span>'
        f'<span class="foot">{escape(block.best_pick_text)}</span></div>'
        "</div>",
        f'<p class="policy-note">{escape(block.caveat_text)}</p>',
        f'<p class="policy-note">{escape(block.tiebreaker_text)}</p>',
    ]
    if block.challengers:
        body = "".join(_season_so_far_challenger_row_html(row) for row in block.challengers)
        parts.append(
            '<div class="section-head ledger-group-head">'
            "<h3>Rules that came out differently from the card</h3>"
            f'<span class="sub">{len(block.challengers)} of them</span></div>'
            '<div class="board-scroll"><table class="board"><thead><tr>'
            "<th>Rule being tried out</th><th>This season</th><th>Against the card</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>"
        )
    parts.append(f'<p class="policy-note">{escape(block.challenger_summary_text)}</p>')
    parts.append("</section>")
    return "".join(parts)


def _history_assessment_html(row: ChallengerAssessment) -> str:
    record = f"{row.wins}-{row.losses}-{row.pushes}"
    accuracy = f"{row.accuracy:.1%}" if row.accuracy is not None else "--"
    delta = (
        f"{row.delta_accuracy_points:+.2f} pts" if row.delta_accuracy_points is not None else "--"
    )
    if row.probability_positive is not None:
        uncertainty = f"{row.probability_positive:.0%} likely better"
    elif row.interval_low is not None and row.interval_high is not None:
        uncertainty = (
            f"somewhere between {row.interval_low:+.2f} and {row.interval_high:+.2f} points"
        )
    else:
        uncertainty = "not measured yet"
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

    if content.picks:
        picks_body = "".join(_history_pick_row_html(row) for row in content.picks)
        picks_section = (
            '<div class="board-scroll"><table class="board"><thead><tr>'
            "<th>Season / week</th><th>Matchup</th><th>Pick at frozen line</th>"
            "<th>Chosen-side confidence</th><th>Outcome</th>"
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
        + _season_so_far_section_html(content.season_so_far)
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


def _trace_chip_html(finding: FindingItemView) -> str:

    if finding.trace_signal_name is None or finding.trace_probability_positive is None:
        return ""
    label = humanize_identifier(finding.trace_signal_name)
    if finding.trace_signal_name == "mod18_home_side_location_v1_s3_through_card_vs_s2":
        label = "Big-spread push versus all-spread push"
    return (
        '<span class="trace-chip">'
        f"{escape(label)} &middot; "
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
    return (
        '<div class="attr-row"><div><span class="chan">'
        f"Open question &middot; {escape(lead.league)} &middot; "
        f"{escape(lead.seasons_text)}"
        f'</span><div class="chan-sub">{escape(lead.description)}</div></div>'
        f'<div class="pts">{escape(lead.effect_text)}</div>'
        f'<div class="pts">{_humanize_probability_positive(lead.probability_positive)}</div></div>'
    )


def _recent_activity_entry_html(entry: RecentActivityEntryView) -> str:

    return (
        '<p class="game-sub" style="margin:6px 0;">'
        f"{escape(entry.plain_summary)} &mdash; {escape(entry.effect_text)}, "
        f"{escape(entry.chance_it_helps_text)}."
        + (f' <span class="pill">{escape(entry.closed_label)}</span>' if entry.closed_label else "")
        + "</p>"
    )


def _recent_activity_section_html(activity: RecentActivityView) -> str:

    header = (
        '<section aria-labelledby="recentactivity-h"><div class="section-head">'
        '<h2 id="recentactivity-h">Research this week</h2>'
        f'<span class="sub">{activity.screened_count} signals looked at &middot; '
        f"{activity.resolved_count} resolved either way &middot; "
        f"{activity.still_open_count} still open</span>"
        "</div>"
    )
    if activity.is_empty:
        body = '<p class="policy-note">No new screens recorded this week.</p>'
    elif not activity.entries:
        body = (
            '<p class="policy-note">Nothing worth a one-line summary '
            "from this week&#x27;s screens yet.</p>"
        )
    else:
        body = "".join(_recent_activity_entry_html(entry) for entry in activity.entries)
    return header + body + "</section>"


def _notable_signal_row_html(row: SignalNotableRow) -> str:
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
