"""The weekly picks board: one self-contained HTML fragment, one row per game.

This is a faithful port of the owner-approved design mock (``build_picks_board.py`` /
``picks-board.html``, built from the real synchronized Week 1 artifact) into a pure
function over dataframes, so it can be unit-tested and rendered from live dashboard
data via ``st.html`` rather than regenerated as a standalone file.

What is ported verbatim from the mock: the CSS design tokens and both light/dark
palettes, the per-row grid layout and its mobile media query, the numberless 17-cell
gradient strip with the amber-boxed market column, the confidence-descending sort,
tier thresholds (57%/52%), and the client-side "expand all / collapse all" toggle
(now wired via ``addEventListener`` rather than an inline ``onclick`` attribute --
inline event-handler attributes are stripped by Streamlit's HTML sanitizer even with
``unsafe_allow_javascript=True``, which only allows ``<script>`` tags to execute).

What is deliberately *not* ported, because it is now rendered natively by
``app_pages/home.py`` instead (see that module for why): the ``<h1>`` title, the
season/week byline, the honesty note, and the tier-count chips. Only the toggle
button stays inside this HTML fragment -- it must live in the same DOM subtree as
the ``<details>`` rows it controls for the client-side script to reach them without
a Streamlit rerun.

Sign conventions and formatting are reused from :mod:`nfl_ats.dashboard.ui` rather
than re-derived here: :func:`nfl_ats.dashboard.ui.pick_side_and_line` for which side
is "the pick" and its line, :func:`nfl_ats.dashboard.ui.confidence_tier` for the
57%/52% tier thresholds, :func:`nfl_ats.dashboard.ui.spread_label` for half-point
("-7½") formatting, :func:`nfl_ats.dashboard.ui.line_sweep_ribbon` for reorienting a
game's swept lines to the picked side, and :func:`nfl_ats.dashboard.ui.canonical_fair_line`
/ :func:`nfl_ats.dashboard.ui.fair_line_gap` / :func:`nfl_ats.dashboard.ui.format_line_journey`
/ :func:`nfl_ats.dashboard.ui.format_value_framing` for the fair line, the edge, and the
detail row's Open/Now/Close/Fair line journey. The journey helpers render Streamlit's
``:green[...]``/``:orange[...]`` markdown color syntax, which does nothing inside raw
HTML, so :func:`_markdown_color_to_html` converts their output to real ``<span>``
elements rather than re-deriving the wording.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from nfl_ats.dashboard.ui import (
    canonical_fair_line,
    confidence_tier,
    fair_line_gap,
    format_line_journey,
    format_value_framing,
    line_sweep_ribbon,
    pick_side_and_line,
    spread_label,
)

# ---------------------------------------------------------------------------
# Tier CSS-class mapping: reuses ui.confidence_tier's 57%/52% thresholds so the
# native chips (rendered by home.py) and the HTML board's pick-chip colors can
# never drift out of sync with each other.
# ---------------------------------------------------------------------------

_TIER_CSS_KEY = {"Strong lean": "pb-strong", "Lean": "pb-lean", "Coin flip": "pb-flip"}


def tier_counts(predictions: pd.DataFrame) -> tuple[int, int, int]:
    """(#strong leans, #leans, #coin flips) by picked-side probability.

    Routed entirely through :func:`nfl_ats.dashboard.ui.confidence_tier` so the
    native tier-count chips home.py renders above the board always agree with
    the tier coloring used inside the board's own HTML.
    """

    if predictions.empty:
        return 0, 0, 0
    probabilities = predictions["home_cover_probability"].map(
        lambda value: max(float(value), 1.0 - float(value))
    )
    labels = probabilities.map(lambda p: confidence_tier(p)[0])
    return (
        int((labels == "Strong lean").sum()),
        int((labels == "Lean").sum()),
        int((labels == "Coin flip").sum()),
    )


# ---------------------------------------------------------------------------
# Markdown-color -> HTML adapter for the reused line-journey helpers
# ---------------------------------------------------------------------------

_COLOR_TAG_RE = re.compile(r":(green|orange)\[([^\]]*)\]")
_COLOR_CSS = {"green": "pb-good", "orange": "pb-bad"}


def _markdown_color_to_html(markdown_text: str) -> str:
    """Convert ``:green[...]``/``:orange[...]`` (Streamlit markdown) into HTML spans.

    ``format_line_journey``/``format_value_framing`` are reused verbatim for their
    wording and color *decisions*; this only translates the syntax those decisions
    are expressed in, since ``st.html`` does not interpret Streamlit's colored-
    markdown extension. The non-tagged text is never escaped here because every
    caller only ever feeds it output built from ``spread_label`` (digits, ``+``,
    ``-``, ``½``, em dash) and fixed literal words -- there is no path for raw user
    text to reach this string.
    """

    return _COLOR_TAG_RE.sub(
        lambda match: f'<span class="{_COLOR_CSS[match.group(1)]}">{match.group(2)}</span>',
        markdown_text,
    )


# ---------------------------------------------------------------------------
# Pick-side ribbon ordering (the mock's subtle reversal, reused + re-sorted)
# ---------------------------------------------------------------------------


def pick_side_cells(row: pd.Series, game_sweep: pd.DataFrame) -> pd.DataFrame:
    """One game's swept lines, pick-side oriented, in the board's display order.

    :func:`nfl_ats.dashboard.ui.line_sweep_ribbon` (``perspective="pick"``) already
    does the sign-convention work of reorienting a home-oriented sweep to whichever
    side the model picked -- that is reused here, not re-derived. It returns rows
    ascending by pick-side line; the approved mock instead displays every row
    descending by pick-side line regardless of whether the pick is home or away
    (verified against the mock generator: a home pick's cells are built in
    home-line-ascending order and *not* reversed, while an away pick's cells are
    built in home-line-ascending order and *are* reversed -- both end up descending
    by pick-side line). This just re-sorts the ribbon's own output to match that,
    rather than re-implementing the home/away sign flip.
    """

    if game_sweep.empty:
        return pd.DataFrame(columns=["line", "label", "probability", "is_market"])
    ribbon = line_sweep_ribbon(row, game_sweep, perspective="pick")
    return ribbon.sort_values("line", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ribbon cell coloring (the mock's own translucent, theme-adaptive scale --
# deliberately not ui.py's render_confidence_ribbon, whose fixed absolute RGB
# blend does not adapt to the surrounding page's light/dark tokens)
# ---------------------------------------------------------------------------


def _cell_background(probability: float) -> str:
    percent = round(probability * 100)
    if percent >= 50:
        alpha_step = min(1.0, (percent - 50) / 18)
        return f"rgba(46,107,79,{0.12 + 0.55 * alpha_step:.3f})"
    alpha_step = min(1.0, (50 - percent) / 18)
    return f"rgba(140,90,60,{0.06 + 0.30 * alpha_step:.3f})"


def _mini_ribbon_html(cells: pd.DataFrame) -> str:
    if cells.empty:
        return '<span class="pb-no-sweep">&mdash;</span>'
    parts = []
    for _, cell in cells.iterrows():
        percent = round(float(cell["probability"]) * 100)
        label = html.escape(str(cell["label"]))
        border = "pb-mkt" if bool(cell["is_market"]) else ""
        style = f"background:{_cell_background(float(cell['probability']))};"
        parts.append(
            f'<td class="pb-rm {border}" style="{style}" title="{label}: {percent}%"></td>'
        )
    return f'<table class="pb-ribbon-mini"><tr>{"".join(parts)}</tr></table>'


def _full_ribbon_html(cells: pd.DataFrame) -> str:
    if cells.empty:
        return ""
    parts = []
    for _, cell in cells.iterrows():
        percent = round(float(cell["probability"]) * 100)
        label = html.escape(str(cell["label"]))
        border = "pb-mkt" if bool(cell["is_market"]) else ""
        style = f"background:{_cell_background(float(cell['probability']))};"
        parts.append(
            f'<td class="pb-rc {border}" style="{style}">'
            f'<span class="pb-rl">{label}</span><br>{percent}%</td>'
        )
    return f'<table class="pb-ribbon-full"><tr>{"".join(parts)}</tr></table>'


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


def _row_html(
    row: pd.Series,
    game_sweep: pd.DataFrame,
    explanation: str,
    opener: float | None,
    latest: float | None,
    predicted_close: float | None,
) -> str:
    team, pick_line, pick_probability = pick_side_and_line(row)
    home_pick = team == row["home_team"]
    has_sweep = not game_sweep.empty

    home_ribbon = (
        line_sweep_ribbon(row, game_sweep, perspective="home") if has_sweep else pd.DataFrame()
    )
    fair = canonical_fair_line(row, home_ribbon)
    edge = fair_line_gap(pick_line, fair, home_pick=home_pick)

    tier_label = confidence_tier(pick_probability)[0]
    tier_css = _TIER_CSS_KEY[tier_label]

    if edge is None:
        edge_text, edge_css = "&mdash;", "pb-flat"
    else:
        edge_text = f"{edge:+.1f}"
        edge_css = "pb-good" if edge > 0.05 else ("pb-bad" if edge < -0.05 else "pb-flat")

    cells = pick_side_cells(row, game_sweep)

    weekday = str(row.get("weekday") or "")[:3]
    gameday = str(row.get("gameday") or "")[5:].replace("-", "/")
    gametime = str(row.get("gametime") or "")[:5]
    kickoff = html.escape(f"{weekday} {gameday} {gametime}".strip())

    matchup = html.escape(f"{row['away_team']} @ {row['home_team']}")
    pick_text = html.escape(f"{team} {spread_label(pick_line)}")
    explanation_text = (
        html.escape(explanation) if explanation else "Model and market are close on this one."
    )

    journey_text = format_line_journey(
        opener, latest, predicted_close, fair, format_value_framing(edge)
    )
    journey_html = _markdown_color_to_html(journey_text)

    market_line_text = html.escape(spread_label(float(row["spread_line"])))
    detail = (
        f'<div class="pb-detail">{_full_ribbon_html(cells)}'
        f'<p class="pb-expl">{explanation_text}</p>'
        f'<p class="pb-mini">Market line {market_line_text} (home) &middot; {journey_html}</p>'
        "</div>"
    )
    summary = (
        f'<summary><span class="pb-kick">{kickoff}</span>'
        f'<span class="pb-match">{matchup}</span>'
        f'<span class="pb-pick {tier_css}">{pick_text}</span>'
        f'<span class="pb-conf">{round(pick_probability * 100)}%</span>'
        f'<span class="pb-edge {edge_css}">{edge_text}</span>'
        f'<span class="pb-rib">{_mini_ribbon_html(cells)}</span>'
        '<span class="pb-chev">&#9662;</span></summary>'
    )
    return f'<details class="pb-row">{summary}{detail}</details>'


# ---------------------------------------------------------------------------
# CSS (ported design tokens; classes prefixed pb- because st.html inserts this
# fragment directly into the app's shared document -- it is not sandboxed in
# an iframe in this Streamlit version, so unprefixed generic class names like
# ".row" or ".detail" could collide with other markup on the same page)
# ---------------------------------------------------------------------------

_CSS = """
.pb { --ground:#FAFAF7; --surface:#F1F2ED; --ink:#1C2420; --muted:#5C6862;
  --line:#D9DDD6; --accent:#2E6B4F; --amber:#B98A2E; --brick:#A8503C;
  --strong:#2E6B4F; --lean:#4A7A96; --flip:#8A8F8A;
  background:var(--ground); color:var(--ink); margin:0;
  font-family:"Segoe UI",system-ui,sans-serif; line-height:1.45;
  max-width:56rem; box-sizing:border-box; }
.pb *, .pb *::before, .pb *::after { box-sizing:border-box; }
@media (prefers-color-scheme: dark) { .pb:not([data-theme="light"]) {
  --ground:#141816; --surface:#1C2220; --ink:#E8ECE9; --muted:#9AA69F;
  --line:#2E3733; --accent:#5FA983; --amber:#C9A75A; --brick:#C97B66;
  --strong:#5FA983; --lean:#7FA8C4; --flip:#7A807A;
} }
.pb[data-theme="dark"] { --ground:#141816; --surface:#1C2220; --ink:#E8ECE9; --muted:#9AA69F;
  --line:#2E3733; --accent:#5FA983; --amber:#C9A75A; --brick:#C97B66;
  --strong:#5FA983; --lean:#7FA8C4; --flip:#7A807A; }
.pb-toolbar { display:flex; justify-content:flex-end; margin:0 0 .4rem; }
.pb-toggle { cursor:pointer; color:var(--accent); font:inherit; font-size:.75rem;
  font-weight:700; padding:.15rem .6rem; border-radius:999px; border:1px solid var(--line);
  background:var(--surface); }
.pb-toggle:hover { border-color:var(--accent); }
.pb-toggle:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.pb-hdr, .pb-row summary { display:grid; align-items:center; gap:.6rem;
  grid-template-columns: 6.2rem 6.5rem 6.5rem 3.2rem 3.2rem 1fr 1rem; }
.pb-hdr { font-size:.68rem; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); padding:.25rem .5rem; }
.pb-row { border:1px solid var(--line); border-radius:6px; background:var(--surface);
  margin-bottom:.35rem; }
.pb-row summary { padding:.45rem .5rem; cursor:pointer; list-style:none; }
.pb-row summary::-webkit-details-marker { display:none; }
.pb-kick { color:var(--muted); font-size:.78rem; }
.pb-match { font-weight:600; font-size:.9rem; }
.pb-pick { font-weight:700; font-family:ui-monospace,Consolas,monospace; font-size:.9rem;
  padding:.1rem .45rem; border-radius:4px; justify-self:start; }
.pb-pick.pb-strong { color:#fff; background:var(--strong); }
.pb-pick.pb-lean { color:#fff; background:var(--lean); }
.pb-pick.pb-flip { color:var(--ink); background:transparent; border:1px solid var(--line); }
.pb-conf { font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums;
  font-weight:700; font-size:1rem; text-align:right; }
.pb-edge { font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums;
  font-size:.85rem; text-align:right; }
.pb-good { color:var(--accent); } .pb-bad { color:var(--brick); } .pb-flat { color:var(--muted); }
.pb-ribbon-mini { width:100%; border-collapse:collapse; table-layout:fixed; height:14px; }
.pb-ribbon-mini td.pb-rm { padding:0; height:14px; border:2px solid transparent; }
.pb-ribbon-mini td.pb-rm.pb-mkt, .pb-ribbon-full td.pb-rc.pb-mkt { border:2px solid var(--amber); }
.pb-no-sweep { color:var(--muted); font-size:.78rem; }
.pb-chev { color:var(--muted); font-size:.8rem; }
.pb-detail { padding:.3rem .6rem .7rem; border-top:1px solid var(--line); }
.pb-ribbon-full { width:100%; border-collapse:collapse; table-layout:fixed;
  font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums;
  font-size:.68rem; margin:.4rem 0; }
.pb-ribbon-full td.pb-rc { text-align:center; padding:2px 0; border:2px solid transparent; }
.pb-ribbon-full .pb-rl { color:var(--muted); font-size:.62rem; }
.pb-expl { font-size:.85rem; margin:.4rem 0 .2rem; }
.pb-mini { color:var(--muted); font-size:.75rem; margin:.2rem 0 0; }
.pb-foot { color:var(--muted); font-size:.72rem; margin-top:.6rem;
  border-top:1px solid var(--line); padding-top:.4rem; }
.pb-empty { color:var(--muted); font-size:.85rem; padding:.5rem; }
@media (max-width:640px) {
  .pb-hdr { display:none; }
  .pb-row summary { grid-template-columns: 1fr auto auto 1rem; grid-template-areas:
    "match pick conf chev" "kick kick kick kick" "rib rib rib rib"; }
  .pb-match { grid-area:match; } .pb-pick { grid-area:pick; } .pb-conf { grid-area:conf; }
  .pb-chev { grid-area:chev; } .pb-kick { grid-area:kick; } .pb-rib { grid-area:rib; }
  .pb-edge { display:none; }
}
"""

# Client-side "expand all / collapse all": deliberately an addEventListener bound
# to #toggle-all rather than an inline onclick attribute. Streamlit's HTML
# sanitizer strips inline on* event-handler attributes unconditionally (verified
# live), even with unsafe_allow_javascript=True on the st.html call -- that flag
# only permits <script> tags to run, so the listener has to be wired from one.
#
# The button itself starts hidden (inline style="display:none" on the element)
# and the trailing <script> is what reveals it, alongside wiring the listener --
# progressive enhancement: every <details>/<summary> row already expands/collapses
# natively with zero JS (that's plain HTML), so if a rendering context ever runs
# without scripts, there is no dead, non-functional button left visible; the
# button only appears where it will actually work.
#
# The same script also keeps the board's own data-theme attribute in sync with
# Streamlit's *actual live* theme, not just its theme at the moment this HTML was
# rendered. Verified live: st.html's content is inserted directly into the app's
# DOM in this Streamlit version (not iframed -- see HtmlMixin.html's own
# docstring), so it inherits the surrounding page's CSS cascade, but picking a
# theme from Streamlit's Settings menu is a client-only preference change that
# does not necessarily trigger a script rerun -- server-computed
# st.context.theme.type (the `theme` argument below) can go stale the moment the
# user opens that menu, leaving the board on whatever theme was active at the
# last rerun while the rest of the app has already switched. A short poll reading
# a live Streamlit-owned element's actual rendered background color is what
# stays correct regardless of *why* the surrounding theme changed (menu
# selection, OS-level scheme change while set to "System", etc.); the
# prefers-color-scheme media-query change listener just makes the OS-level case
# instant instead of waiting for the next poll tick.
_TOGGLE_SCRIPT = """
<script>
(function () {
  var board = document.querySelector(".pb");
  var button = document.getElementById("toggle-all");
  if (button) {
    button.style.display = "";
    button.addEventListener("click", function () {
      var rows = document.querySelectorAll("details.pb-row");
      var shouldOpen = Array.prototype.some.call(rows, function (r) { return !r.open; });
      rows.forEach(function (r) { r.open = shouldOpen; });
      button.textContent = shouldOpen ? "collapse all" : "expand all";
    });
  }

  if (!board) return;
  function isDarkBackground(el) {
    while (el) {
      var bg = getComputedStyle(el).backgroundColor;
      var match = bg.match(/rgba?\\((\\d+), *(\\d+), *(\\d+)(?:, *([\\d.]+))?\\)/);
      if (match) {
        var alpha = match[4] === undefined ? 1 : parseFloat(match[4]);
        if (alpha > 0) {
          var luma = 0.299 * match[1] + 0.587 * match[2] + 0.114 * match[3];
          return luma < 128;
        }
      }
      el = el.parentElement;
    }
    return false;
  }
  function syncTheme() {
    var appEl = document.querySelector('[data-testid="stApp"]') || document.body;
    board.setAttribute("data-theme", isDarkBackground(appEl) ? "dark" : "light");
  }
  syncTheme();
  setInterval(syncTheme, 500);
  var media = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
  if (media && media.addEventListener) {
    media.addEventListener("change", syncTheme);
  }
})();
</script>
"""

_HEADER_ROW = (
    '<div class="pb-hdr"><span>Kickoff</span><span>Game</span><span>Pick</span><span>Conf</span>'
    "<span>Edge</span><span>Confidence by line</span><span></span></div>"
)


def render_picks_board(
    predictions: pd.DataFrame,
    sweep: pd.DataFrame | None = None,
    explanations: Mapping[str, str] | None = None,
    *,
    openers: Mapping[str, float] | None = None,
    latest_lines: Mapping[str, float] | None = None,
    predicted_close: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
    theme: Literal["light", "dark"] | None = None,
) -> str:
    """Render the picks board as one self-contained HTML fragment for ``st.html``.

    ``predictions`` is one row per game (the market_residual-method rows for the
    active weekly forecast, e.g. ``WeeklyForecast.recommendations``). ``sweep`` is
    the matching line-sweep table (``game_id`` + the columns
    :func:`nfl_ats.dashboard.ui.line_sweep_ribbon` needs); pass an empty or ``None``
    frame to render every row without a confidence strip -- no sweep is required.
    ``explanations`` maps ``game_id`` to a plain-English sentence (from the
    market-decomposition attribution artifact); a game without one falls back to
    "Model and market are close on this one." ``openers``/``latest_lines``/
    ``predicted_close`` map ``game_id`` to a home-oriented spread for the detail
    row's line journey; any missing entry renders as an em dash. ``metadata`` is
    the forecast card's own ``metadata.json`` (used only for its
    ``active_model_id``, in a one-line footer below the rows) -- everything else
    the mock's footer said (season/week, "synchronized with the active model",
    the honesty note) is rendered natively by ``home.py`` above the board instead,
    and the mock's "openers/live line/predicted close appear as captures land"
    caveat is dropped rather than ported: it would be actively wrong here, since
    this board's own detail row already shows real Open/Now/Close values whenever
    ``openers``/``latest_lines``/``predicted_close`` have them.

    ``theme`` should be ``st.context.theme.type`` ("light"/"dark"/``None``). It
    only sets the *initial* ``data-theme`` attribute on the board's own wrapper
    element (never on the real document root -- this fragment shares a DOM with
    the rest of the app, unlike the standalone mock file, so its CSS custom
    properties are scoped to ``.pb`` rather than ``:root``). The trailing
    ``<script>`` immediately overrides that initial guess with a live read of
    Streamlit's actual rendered theme, and keeps polling: picking a theme from
    Streamlit's Settings menu is a client-only change that does not reliably
    trigger a script rerun, so a value baked in at render time can go stale the
    moment the user opens that menu -- verified live, not assumed (see the
    comment above ``_TOGGLE_SCRIPT``).
    """

    explanations = explanations or {}
    openers = openers or {}
    latest_lines = latest_lines or {}
    predicted_close = predicted_close or {}
    theme_attr = f' data-theme="{theme}"' if theme in ("light", "dark") else ""
    model_id = (metadata or {}).get("active_model_id")
    foot = f'<p class="pb-foot">Model {html.escape(str(model_id))}</p>' if model_id else ""

    if predictions.empty:
        return (
            f'<div class="pb"{theme_attr}><style>{_CSS}</style>'
            f'<p class="pb-empty">No games this week.</p>{foot}</div>'
        )

    sweep = sweep if sweep is not None else pd.DataFrame()
    if not sweep.empty and "method" in sweep.columns and "method" in predictions.columns:
        method = str(predictions["method"].iloc[0])
        sweep = sweep.loc[sweep["method"].eq(method)]
    sweep_game_key = "game_id" if "game_id" in sweep.columns else None

    pick_probability = predictions["home_cover_probability"].map(
        lambda value: max(float(value), 1.0 - float(value))
    )
    ordered = (
        predictions.assign(_pick_probability=pick_probability)
        .sort_values("_pick_probability", ascending=False, kind="stable")
        .drop(columns="_pick_probability")
    )

    rows_html = []
    for _, row in ordered.iterrows():
        game_id = str(row["game_id"])
        game_sweep = (
            sweep.loc[sweep[sweep_game_key].eq(row["game_id"])]
            if sweep_game_key is not None
            else pd.DataFrame()
        )
        rows_html.append(
            _row_html(
                row,
                game_sweep,
                explanations.get(game_id, ""),
                openers.get(game_id),
                latest_lines.get(game_id),
                predicted_close.get(game_id),
            )
        )

    toolbar = (
        '<div class="pb-toolbar">'
        '<button class="pb-toggle" id="toggle-all" type="button" style="display:none">'
        "expand all</button>"
        "</div>"
    )
    return (
        f'<div class="pb"{theme_attr}><style>{_CSS}</style>{toolbar}{_HEADER_ROW}'
        f"{''.join(rows_html)}{foot}</div>{_TOGGLE_SCRIPT}"
    )


__all__ = ["pick_side_cells", "render_picks_board", "tier_counts"]
