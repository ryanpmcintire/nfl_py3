"""Render the public GitHub Pages site: three static pages in the site's design.

This module imports the shared pure presentation modules directly and composes
them into self-contained static HTML:

* :mod:`nfl_ats.dashboard.theme` -- ``stylesheet()`` (role tokens, light + dark).
* :mod:`nfl_ats.dashboard.viz` -- ``probability_meter``, ``line_journey``,
  ``sweep_curve``, ``season_bars``, ``stat_tile``, ``status_line``, ``card``,
  ``page_header``, ``empty_state``, ``interaction_script``.
* :mod:`nfl_ats.dashboard.findings_content` -- the findings text model.

None of those three imports a web-framework runtime, so the CLI publish path
(``nfl-ats publish-board``, and ``nfl-ats publish-predictions --with-board``)
stays free of one. This module keeps its own artifact loading (below) rather
than importing any page layer.

Notes for a static page with no host application:

* The stylesheet's bare ``prefers-color-scheme`` media query handles light/dark
  on its own; its ``:not([data-theme="light"])`` guard is simply inert without
  an external stamper.
* ``viz.interaction_script()`` ships as its own ``<script>`` tag (picks page
  only).

The components' no-SVG / no-tag-inside-JS discipline is preserved regardless:
one implementation serves every surface, so a "static pages could use SVG"
divergence would immediately rot the shared design system.

Public-audience guardrail (licensing/ethics constraint, not a style choice)
--------------------------------------------------------------------------
These pages render only fields already published in the repo's tracked public
markdown card (see :func:`nfl_ats.publishing._published_card`):

* the pick and its decision-strength label,
* the model's own fair line (pure model output),
* ONE consensus market line per game -- ``spread_line`` from the synchronized
  weekly forecast, never a per-book quote,
* kickoff, and the plain-English market-decomposition explanation,
* aggregate accuracy statistics (opener/close grades, per-season accuracy).

The line-sweep curve is model output evaluated at OFFSETS from that single
published line: its axis is ``line_offset`` (-4 to +4) with the one consensus
line as the origin label, so it exposes no market number the card did not
already carry. The internal dashboard additionally shows an archive-derived
opener consensus and a predicted close; both are withheld here pending the
MKT-09 provider licensing/quota audit (see ROADMAP.md) -- see the
``line_journey`` call in :func:`render_picks_page`.

Book names, per-book prices, and every other raw market-feed field
(``home_spread_odds``, ``away_spread_odds``, ``total_line``, ...) must never
appear on any generated page. ``tests/test_public_board.py`` enforces the
blocklist across ALL THREE pages.

``DISCLAIMER_SHORT`` appears once near the top of every page;
``DISCLAIMER_FULL`` appears once, in the footer.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.backup_qb_fade_overlay import apply_backup_qb_fade_overlay
from nfl_ats.best_pick_nomination import nominate_v3
from nfl_ats.card_view import (
    BestPickNomination,
    resolve_card_view,
    resolve_nomination,
    resolve_overlay,
    resolve_player_arrests_overlay,
    v2_nomination_inputs,
)
from nfl_ats.coach_fade_overlay import OverlayFlip, OverlayResult
from nfl_ats.dashboard import theme, viz
from nfl_ats.dashboard.findings_content import (
    CEILING_BUG_MARK_PCT,
    CHALLENGER_DISPLAY_NAMES,
    CLOSING_NOTE,
    DETAIL_SUMMARY_LABEL,
    FINDINGS,
    GROUPS,
    HEADLINE,
    HERO_KICKER,
    HERO_PARAGRAPHS,
    HERO_SUB,
    HERO_TILES,
    HERO_TITLE,
    HONESTY_KICKER,
    HONESTY_RULES,
    HONESTY_SUB,
    HONESTY_TITLE,
    LEAD_BLURBS,
    LEGEND_KICKER,
    PLAYED_CARD_EXPECTATION_HERO,
    PREMEASUREMENT_GUESS_BAND,
    SOURCE_LABEL,
    Finding,
    LeadBlurb,
    VerdictGroup,
    findings_for,
    ladder_rungs,
)
from nfl_ats.data import DataContractError
from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay
from nfl_ats.findings_registry import (
    WatchingLead,
    load_all_entries,
    load_weak_signal_registry,
    top_open_leads,
    validate_curation,
)
from nfl_ats.four_overlay_composition import FourOverlayCompositionResult
from nfl_ats.injury_value_tilt_overlay import (
    PLAYER_FEATURE_TABLE_NAME,
    apply_injury_value_tilt_overlay,
)
from nfl_ats.interim_hc_first_game_tilt_overlay import (
    apply_interim_hc_first_game_tilt_overlay,
)
from nfl_ats.model_ledger import build_and_render
from nfl_ats.player_arrests_back_side_overlay import (
    POLICY_BASELINE_OPENER_ACCURACY,
    POLICY_EFFECT_ACCURACY_POINTS,
    POLICY_GRADED_GAMES,
    POLICY_OPENER_ACCURACY,
    POLICY_PROBABILITY_POSITIVE,
    ArrestFlip,
    ArrestOverlayResult,
)
from nfl_ats.pool_workbench import PoolRules, build_pool_workbench_body
from nfl_ats.reporting import artifact_directories, read_json
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.spread_explorer import (
    SPREAD_EXPLORER_MAX_LINE,
    SPREAD_EXPLORER_MIN_LINE,
    SPREAD_EXPLORER_STEP,
    SpreadExplorerGameParams,
    compute_spread_explorer_params,
    load_feature_table_for_forecast,
    spread_explorer_payload,
    widget_home_cover_probability,
)
from nfl_ats.spread_gap_zone_fade_overlay import apply_spread_gap_zone_fade_overlay
from nfl_ats.surface_switch_tilt_overlay import apply_surface_switch_tilt_overlay
from nfl_ats.surgical_gating import VALUE_LOST_DIFF_COLUMNS
from nfl_ats.team_explorer import (
    DEFAULT_TREND_METRICS,
    TeamTrends,
    aggregate_team_trends,
    feature_table_to_team_states,
    metric_label,
    team_state_payload,
)
from nfl_ats.weak_signals import Registry as WeakSignalRegistry
from nfl_ats.weak_signals import default_registry_path as _default_weak_signals_registry_path

# ---------------------------------------------------------------------------
# Mandatory public-audience disclaimer text
#
# Embedded verbatim (not escaped) wherever it appears in a page: both strings
# are static, hardcoded, developer-authored constants -- never artifact or user
# data -- so there is no injection risk, and escaping would only mangle the
# apostrophe into "&#x27;" for no benefit.
# ---------------------------------------------------------------------------

DISCLAIMER_SHORT = (
    "Research project — simulated, paper picks only. Not betting advice. A small "
    "historical edge is not proof of a profitable one."
)

DISCLAIMER_FULL = (
    "This page is the output of a personal research project. Every pick shown is a "
    "simulated, paper pick made to evaluate a forecasting model — it is not betting "
    "advice, not a recommendation to wager, and no real money is risked on these picks "
    "by the author. The model's historical accuracy sits close to a coin flip "
    "and is not proof of a profitable edge -- sportsbook vig alone would likely erase an "
    "edge that size over the long run. If you choose to gamble, please do so responsibly. "
    "21+ where applicable. If you or someone you know has a gambling "
    "problem, call 1-800-GAMBLER (US) or visit ncpgambling.org."
)


# ---------------------------------------------------------------------------
# The shared page shell
# ---------------------------------------------------------------------------

PICKS_PAGE = "index.html"
FINDINGS_PAGE = "findings.html"
TRACK_RECORD_PAGE = "track_record.html"
MODELS_PAGE = "models.html"
TEAM_EXPLORER_PAGE = "team_explorer.html"
POOL_PAGE = "pool.html"

# (file name, nav label, browser title) in nav order.
SITE_PAGES: tuple[tuple[str, str, str], ...] = (
    (PICKS_PAGE, "This week", "This week's picks"),
    (MODELS_PAGE, "Models", "Model ledger"),
    (TEAM_EXPLORER_PAGE, "Team trends", "Team pregame-state trends"),
    (FINDINGS_PAGE, "What we've learned", "What we've learned"),
    (TRACK_RECORD_PAGE, "Track record", "Track record"),
    (POOL_PAGE, "Pool workbench", "Pool workbench"),
)

# Page chrome only: the "Ledger base + Terminal layout" design system. It rides
# AFTER theme.stylesheet() so its token remaps win on equal specificity, and the
# dark theme is a second DESIGNED palette behind ``prefers-color-scheme`` (not
# an inversion). Binding rules baked in below: hex budget <= 10 in the light
# block; sizes only {11,12,13,14,17,24}px; 4px spacing grid; radius {4,8};
# shadows banned (the sticky header separates with a hairline border, not a
# shadow); no gradients or glows; transitions <=200ms on interactive state only.
_PAGE_CHROME = """
<style>
body { margin: 0; overflow-x: hidden; }
.ats {
  --plane: #fafaf8;
  --surface: #ffffff;
  --ink: #111110;
  --ink-2: #4b4b47;
  --muted: #8a8a84;
  --grid: rgba(0,0,0,0.08);
  --border: rgba(0,0,0,0.08);
  --baseline: rgba(0,0,0,0.16);
  --series-model: #2a78d6;
  --series-market: #4b4b47;
  --series-third: #8a8a84;
  --good-text: #1a7f37;
  --good: #1a7f37;
  --critical: #c0392b;
  --serious: #b35900;
}
@media (prefers-color-scheme: dark) {
  .ats:not([data-theme="light"]) {
    color-scheme: dark;
    --plane: #0b0c0e;
    --surface: #141518;
    --ink: #f7f8f8;
    --ink-2: #b4b8bf;
    --muted: #7d828b;
    --grid: #23252b;
    --border: #23252b;
    --baseline: #33363d;
    --series-model: #6ea8dc;
    --series-market: #b4b8bf;
    --series-third: #7d828b;
    --good-text: #45a86b;
    --good: #45a86b;
    --critical: #e0705c;
    --serious: #d99a3d;
  }
}
.ats {
  background: var(--plane); min-height: 100vh; color: var(--ink);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.ats .wrap { max-width: 72rem; margin: 0 auto; padding: 24px 18px 52px; }
.ats a { color: var(--series-model); text-decoration: none; }
.ats a:hover { text-decoration: underline; }
.ats nav.site { display: flex; gap: 16px; flex-wrap: wrap; margin: 0 0 16px; font-size: 13px; }
.ats nav.site a { color: var(--ink-2); }
.ats nav.site a[aria-current="page"] { color: var(--ink); font-weight: 600; }
.ats a, .ats summary { transition: color 150ms ease, border-color 150ms ease; }

/* Type scale: only {11,12,13,14,17,24}px; 400 body, 600 emphasis. */
.ats .kicker { font-size: 11px; letter-spacing: 0.08em; }
.ats .title { font-size: 17px; font-weight: 600; }
.ats h2.title { letter-spacing: -0.01em; }
.ats .page-title { font-size: 24px; }
.ats .sub { font-size: 13px; }
.ats .prose { font-size: 14px; }
.ats .fine { font-size: 12px; }
.ats .hero { font-size: 24px; font-weight: 600; line-height: 1.15; }

/* Cards become flat hairline sections: no boxes, no fills, no shadows. */
.ats .card {
  background: none; border: none; border-top: 1px solid var(--grid);
  border-radius: 0; padding: 12px 0 0; margin-top: 12px; box-shadow: none;
}
.ats .tip { box-shadow: none; border-radius: 4px; }
.ats .status, .ats .chip { border-radius: 4px; }

/* Four-panel terminal grid (desktop >=1100px): summary | board spanning tall
   right; ledger mini and challenger watch stacked left beneath the summary.
   Below 1100px everything stacks in DOM order. */
.ats .ledger-grid {
  display: grid; grid-template-columns: minmax(300px, 2fr) 3fr;
  grid-template-areas: "summary board" "ledger board" "watch board";
  column-gap: 32px; align-items: start; margin-top: 8px;
}
.ats .panel { border-top: 1px solid var(--grid); padding-top: 8px; }
.ats .panel-summary { grid-area: summary; }
.ats .panel-board { grid-area: board; }
.ats .panel-ledger { grid-area: ledger; margin-top: 16px; }
.ats .panel-watch { grid-area: watch; margin-top: 16px; }
@media (max-width: 1099px) {
  .ats .ledger-grid { display: block; }
  .ats .panel-ledger, .ats .panel-watch { margin-top: 16px; }
}

/* Week board: one continuous table, 40px game rows, expandable sub-rows at the
   compact 32px scale; the sticky header separates with a hairline border. */
.ats table.week-board th {
  position: sticky; top: 0; background: var(--surface); z-index: 1;
  border-bottom: 1px solid var(--baseline);
}
.ats table.week-board tr.board-game td { padding: 12px 8px 12px 0; font-size: 13px; }
.ats table.week-board tr.board-sub > td { padding: 0 0 8px; }
.ats table.week-board tr.board-sub table.data th,
.ats table.week-board tr.board-sub table.data td { padding: 6px 8px 6px 0; }
.ats details.why-pick > summary {
  cursor: pointer; font-size: 12px; color: var(--series-model); list-style: revert;
}
.ats .flip-flag { color: var(--ink-2); cursor: help; }
.ats .best-flag { color: var(--good-text); cursor: help; }
@media (max-width: 640px) {
  .ats .wrap { padding: 16px 12px 40px; }
  .ats table.week-board thead { display: none; }
  .ats table.week-board, .ats table.week-board tbody,
  .ats table.week-board tr, .ats table.week-board td { display: block; width: 100%; }
  .ats table.week-board tr.board-game td {
    border: none; padding: 2px 0; display: flex; justify-content: space-between;
    gap: 10px; align-items: baseline;
  }
  .ats table.week-board td::before {
    content: attr(data-label); color: var(--muted); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em; flex: none;
  }
}

/* Deep-dive blocks: flat prose sections separated by hairlines, never boxes. */
.ats .deep-game { padding: 16px 0; max-width: 70ch; scroll-margin-top: 48px; }
.ats .deep-game + .deep-game { border-top: 1px solid var(--grid); }

/* Spread explorer: sizing/color only; accent reuses the model-series token. */
.ats .spread-explorer input.se-slider {
  width: 100%; height: 28px; margin: 8px 0 6px;
  accent-color: var(--series-model); touch-action: manipulation;
}
.ats .spread-explorer .se-line-words { color: var(--series-model); }
</style>
"""


def _nav(current: str) -> str:
    items = []
    for filename, label, _title in SITE_PAGES:
        if filename == current:
            items.append(f'<span aria-current="page">{escape(label)}</span>')
        else:
            items.append(f'<a href="{filename}">{escape(label)}</a>')
    return f'<nav class="site">{"".join(items)}</nav>'


def _disclaimer_banner() -> str:
    return f'<p class="sub" style="font-weight:600;margin:0 0 16px;">{DISCLAIMER_SHORT}</p>'


def _footer(generated: datetime, note: str = "") -> str:
    stamp = generated.strftime("%Y-%m-%d %H:%M UTC")
    lead = f"{note} &middot; " if note else ""
    return (
        '<div style="margin-top:36px;padding-top:14px;border-top:1px solid var(--grid);">'
        f'<p class="fine">{lead}page generated {stamp}.</p>'
        f'<p class="fine" style="margin-top:10px;max-width:82ch;">{DISCLAIMER_FULL}</p></div>'
    )


def _page(
    *,
    current: str,
    body: str,
    generated: datetime,
    footer_note: str = "",
    scripts: str = "",
) -> str:
    """Wrap composed fragments in a fully self-contained HTML document."""

    title = next(title for filename, _label, title in SITE_PAGES if filename == current)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
{theme.stylesheet().strip()}
{_PAGE_CHROME.strip()}
</head>
<body>
<div class="ats"><div class="wrap">
{_nav(current)}
{_disclaimer_banner()}
{body}
{_footer(generated, footer_note)}
</div></div>
{scripts}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Sign conventions (shared with the retired internal pages; the composition
# lives here now. Keep them in sync by hand with card_view.py.)
# ---------------------------------------------------------------------------

# A "strong lean" is a fair-line disagreement with the market of at least this
# many points. It gates the explanation, not the pick: every pool pick is
# forced, and our confidence ordering has NOT proven predictive (see the
# findings page), so the lean is a narrative marker, never a weighting.
STRONG_LEAN_POINTS = 1.5
SWEEP_HALF_WIDTH = 4.0

_WEEK_LABELS = {
    "WC": "Wild Card round",
    "DIV": "Divisional round",
    "CON": "Conference championships",
    "SB": "Super Bowl",
}


def spread_words(home: str, away: str, home_spread: float) -> str:
    """``'DEN -3.5'`` style, from the home-oriented nflverse spread."""

    if pd.isna(home_spread) or home_spread == 0:
        return "pick 'em"
    favorite, points = (home, home_spread) if home_spread > 0 else (away, -home_spread)
    return f"{favorite} -{points:g}"


def pick_side(row: pd.Series) -> tuple[str, float]:
    """Forced pool pick: (team, that side's calibrated cover probability)."""

    probability = float(row["home_cover_probability"])
    if probability >= 0.5:
        return str(row["home_team"]), probability
    return str(row["away_team"]), 1.0 - probability


def _kickoff_words(row: pd.Series) -> str:
    weekday = str(row.get("weekday") or "").strip()
    gametime = str(row.get("gametime") or "").strip()
    return f"{weekday} {gametime} ET".strip()


def _number(value: Any) -> float | None:
    """Coerce an artifact field to a float, or ``None`` if absent/unusable."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _default_data_root() -> Path:
    """Same env var and default as ``cli._data_root`` -- duplicated rather than
    imported because that function lives in the CLI module, which this one
    deliberately does not import."""

    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


def confidence_word(probability: float) -> str:
    """Plain-English decision-strength label for the week board (D1).

    Three bands on the final side-oriented score. For an unflipped row it is
    the calibrated model probability; for a production-policy flip it is the
    mirrored raw-model score and must not be read as newly calibrated.
    """

    if probability > 0.56:
        return "strong"
    if probability >= 0.53:
        return "lean"
    return "slight"


# ---------------------------------------------------------------------------
# Spread explorer (owner request, 2026-08-20): "pick a spread for a game and
# see the odds of covering." A per-game slider plus a JS-evaluated Gaussian
# read of the SAME residual sample the published pick's own
# ``home_cover_probability`` came from -- see ``nfl_ats.spread_explorer`` for
# the refit-and-verify discipline that produces each game's (center, mean,
# sd) and the module docstring there for why push probability is
# deliberately not modeled by this widget.
#
# Design choice, declared here per the task spec: the slider spans the full
# [-20, +20] range in 0.5-point steps rather than being restricted to
# half-point-only lines. Roughly half of any real week's card sits on a
# WHOLE-number line (3, 7, ...), so excluding integers would make the
# slider unable to even reproduce several of this very card's own published
# lines -- failing the required consistency check by construction for those
# games. The trade-off is that this widget never shows a push probability
# (mathematically undefined for a continuous Gaussian fit at a single point,
# and the mean/sd-only embedding this task specifies has no discrete sample
# to compute one from honestly) -- a plain-English note says so instead of
# inventing a number, both in the page-level intro and on every widget.
# ---------------------------------------------------------------------------

_SPREAD_EXPLORER_TOLERANCE = 1e-4  # see _assert_spread_explorer_matches_card


def _assert_spread_explorer_matches_card(
    params: Mapping[str, SpreadExplorerGameParams], predictions: pd.DataFrame
) -> None:
    """Build-time consistency check (REQUIRED by the spread-explorer spec):
    at each game's OWN quoted line, the EXACT formula shipped to the browser
    (the Abramowitz-Stegun erf approximation in ``_spread_explorer_script``,
    mirrored in Python by ``nfl_ats.spread_explorer.widget_home_cover_probability``
    and evaluated on the SAME rounded values ``spread_explorer_payload``
    embeds) must reproduce the published card's own ``home_cover_probability``
    well within display rounding. Measured error on a real card: ~7.5e-8;
    the tolerance below is two orders of magnitude looser than that, still
    three orders tighter than the page's own displayed 0.1%. A mismatch
    means the widget would show a reader a DIFFERENT number than the one
    already published for the same game at the same line -- fail the build
    rather than silently ship that.
    """

    if not params:
        return
    lookup = predictions.set_index(predictions["game_id"].astype(str))
    for game_id, values in spread_explorer_payload(params).items():
        widget_probability = widget_home_cover_probability(
            values["line"], values["center"], values["mean"], values["std"]
        )
        published = _number(lookup.loc[game_id, "home_cover_probability"])
        if published is None:
            raise DataContractError(
                f"Spread explorer widget has no usable published home_cover_probability for "
                f"{game_id} to check against"
            )
        if abs(widget_probability - published) > _SPREAD_EXPLORER_TOLERANCE:
            raise DataContractError(
                "Spread explorer widget formula disagrees with the published card for "
                f"{game_id}: widget={widget_probability:.6f} card={published:.6f} "
                f"(tolerance {_SPREAD_EXPLORER_TOLERANCE})"
            )


def _spread_explorer_intro(generated: datetime) -> str:
    """One plain-English paragraph explaining the "as of" caveat -- required
    by the spec, rendered once per page rather than repeated on every widget.
    Only rendered when at least one game actually has a widget (see
    ``render_picks_page``)."""

    stamp = generated.strftime("%Y-%m-%d %H:%M UTC")
    inner = (
        '<div class="prose">'
        "<p>Each game below has a <b>Spread explorer</b> slider: drag it to a hypothetical "
        "home spread and see each side's cover chance at that line, read off the same model "
        "that made the actual pick.</p>"
        f"<p>Odds reflect what the model knew <b>as of this build, {escape(stamp)}</b> -- "
        "frozen at build time; only your hypothetical line changes. A small push chance at "
        "whole-number lines is left out rather than invented.</p>"
        "</div>"
    )
    return f'<div style="margin-top:16px;">{inner}</div>'


def _spread_explorer_widget_html(game_id: str, initial_line: float) -> str:
    """One game's interactive slider. ``initial_line`` is the card's own
    quoted ``spread_line`` -- the same value ``_assert_spread_explorer_matches_card``
    already proved reproduces the published ``home_cover_probability`` before
    this function is ever called."""

    gid = escape(game_id)
    return (
        f'<div class="spread-explorer" data-game-id="{gid}" '
        'style="margin-top:14px;padding-top:12px;border-top:1px solid var(--grid);">'
        '<p class="kicker" style="color:var(--series-model);">Spread explorer</p>'
        f'<input type="range" class="se-slider" min="{SPREAD_EXPLORER_MIN_LINE:g}" '
        f'max="{SPREAD_EXPLORER_MAX_LINE:g}" step="{SPREAD_EXPLORER_STEP:g}" '
        f'value="{initial_line:g}" aria-label="Hypothetical home spread for this game">'
        '<p class="sub" style="margin-top:2px;">If the line were '
        '<b class="se-line-words num"></b>: <span class="se-home-pct num"></span> '
        '&#183; <span class="se-away-pct num"></span></p>'
        "</div>"
    )


def _spread_explorer_script(payload: Mapping[str, Mapping[str, Any]]) -> str:
    """One inline JSON blob (the per-game Gaussian params, build-time-verified
    against the published card -- see ``_assert_spread_explorer_matches_card``)
    plus one small vanilla-JS function that evaluates the Gaussian survival
    function at whatever line the reader drags to. NO external resources
    (self-contained static GitHub Pages site): the erf approximation
    (Abramowitz & Stegun 7.1.26) is the standard closed-form way to evaluate
    a normal CDF without a math library, and is re-implemented byte-for-byte
    in Python as ``nfl_ats.spread_explorer.widget_home_cover_probability`` so
    the two are checked against each other (``tests/test_spread_explorer.py``)
    rather than trusted to stay in sync by hand.
    """

    if not payload:
        return ""
    data_json = json.dumps(payload, separators=(",", ":"))
    return (
        f'<script type="application/json" id="ats-se-data">{data_json}</script>\n'
        "<script>\n"
        "(function () {\n"
        "  var dataEl = document.getElementById('ats-se-data');\n"
        "  if (!dataEl) { return; }\n"
        "  var data;\n"
        "  try { data = JSON.parse(dataEl.textContent); } catch (err) { return; }\n"
        "  function erf(x) {\n"
        "    var sign = x < 0 ? -1 : 1; x = Math.abs(x);\n"
        "    var a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,\n"
        "        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;\n"
        "    var t = 1 / (1 + p * x);\n"
        "    var y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);\n"
        "    return sign * y;\n"
        "  }\n"
        "  function normalCdf(x, mean, std) {\n"
        "    return 0.5 * (1 + erf((x - mean) / (std * Math.SQRT2)));\n"
        "  }\n"
        "  function homeCoverProbability(line, center, mean, std) {\n"
        "    return 1 - normalCdf(line - center, mean, std);\n"
        "  }\n"
        "  function spreadWords(home, away, value) {\n"
        '    if (Math.abs(value) < 0.001) { return "pick \'em"; }\n'
        "    var favorite = value > 0 ? home : away;\n"
        "    var points = Math.abs(value);\n"
        "    var text = (points % 1 === 0) ? points.toFixed(0) : points.toFixed(1);\n"
        "    return favorite + ' -' + text;\n"
        "  }\n"
        "  function fmtPct(p) {\n"
        "    return (Math.max(0, Math.min(1, p)) * 100).toFixed(1) + '%';\n"
        "  }\n"
        "  function updateWidget(widget, game) {\n"
        "    var slider = widget.querySelector('.se-slider');\n"
        "    var line = parseFloat(slider.value);\n"
        "    var p = homeCoverProbability(line, game.center, game.mean, game.std);\n"
        "    widget.querySelector('.se-line-words').textContent = "
        "spreadWords(game.home, game.away, line);\n"
        "    widget.querySelector('.se-home-pct').textContent = "
        "game.home + ' covers ' + fmtPct(p);\n"
        "    widget.querySelector('.se-away-pct').textContent = "
        "game.away + ' covers ' + fmtPct(1 - p);\n"
        "  }\n"
        "  var widgets = document.querySelectorAll('.spread-explorer[data-game-id]');\n"
        "  for (var i = 0; i < widgets.length; i++) {\n"
        "    (function (widget) {\n"
        "      var gameId = widget.getAttribute('data-game-id');\n"
        "      var game = data[gameId];\n"
        "      if (!game) { return; }\n"
        "      var slider = widget.querySelector('.se-slider');\n"
        "      if (!slider) { return; }\n"
        "      slider.addEventListener('input', function () { updateWidget(widget, game); });\n"
        "      updateWidget(widget, game);\n"
        "    })(widgets[i]);\n"
        "  }\n"
        "})();\n"
        "</script>\n"
    )


# ---------------------------------------------------------------------------
# Page 1 -- This week
# ---------------------------------------------------------------------------

# Season ops timeline (owner request, 2026-08-20): now that picks stay
# editable to kickoff (docs/late_week_refresh.md, POL-11) and only the
# GRADING line freezes Tuesday, the weekly cadence itself is new information
# a reader needs to make sense of "why did the pick change" -- this strip is
# that explanation, read once at a glance rather than requiring a trip to
# docs/late_week_refresh.md. Every day/step below is [read] from that
# document's own "Cadence" section and "Per-game deadline" section (both
# read in full while building this); the pool's own Wednesday revision is
# [read] from docs/pool_edge_plan.md line 82 / docs/observed_movement_channel.md
# line 14 ("the pool posts lines Tuesday, revises once Wednesday, then
# freezes for the week"). The 2026 Week 1 lock date is [read] from
# docs/late_week_refresh.md's "Season note" ("Week 1 2026 locks Tuesday
# 2026-09-08"). Nothing here is re-derived or guessed; it restates what those
# two tracked documents already say, in the reader's plain English.
_WEEK1_LOCK_LABEL = "Tuesday, September 8, 2026"

_SEASON_OPS_STEPS: tuple[tuple[str, str], ...] = (
    ("Tue", "the pool's line locks at noon and this card's grading line freezes with it"),
    ("Wed", "the pool revises its own line once; our grading line never moves"),
    ("Thu", "refresh pass before Thursday night, on what changed since Tuesday"),
    ("Sat", "second pass for every game that has not kicked off"),
    (
        "Sun AM",
        "final pass before 4:00 PM ET -- Sunday- and Monday-night games lock there too",
    ),
)


def _movement_policy_note(challengers: Sequence[Mapping[str, Any]]) -> str:
    """The observed-movement pick policy, in plain English, with the exact
    registered evidence sentence quoted from ``model_only_refresh_incumbent``
    (``artifacts/prospective/challengers.json``) when that challenger is
    present -- never a number re-typed by hand here. Absent the challenger
    (an older/untracked artifacts tree), this degrades to a generic pointer
    at the findings page rather than inventing a figure.
    """

    entry = next(
        (
            candidate
            for candidate in challengers
            if str(candidate.get("challenger_id")) == "model_only_refresh_incumbent"
        ),
        None,
    )
    evidence = entry.get("evidence") if isinstance(entry, dict) else None
    threshold_text = evidence.get("threshold_frozen") if isinstance(evidence, dict) else None
    body = (
        '<div class="prose"><p><b>If the market moves a full point, we follow it.</b> '
        "At each pass, if the pool's own line has moved at least 1.0 point off Tuesday's "
        "frozen number, the pick follows the market; below that threshold (or with no "
        "fresh line captured), the model's own re-run pick plays as always.</p></div>"
    )
    if isinstance(threshold_text, str) and threshold_text.strip():
        body += (
            f'<p class="fine" style="margin-top:8px;">As registered '
            f"({_challenger_display_name('model_only_refresh_incumbent')}): "
            f"{escape(threshold_text)}</p>"
        )
    else:
        body += (
            '<p class="fine" style="margin-top:8px;">Not yet measured on this build -- see the '
            f"{_challenger_display_name('model_only_refresh_incumbent')} candidate rule on the "
            "findings page once it is tracked.</p>"
        )
    return body


def _season_ops_timeline_section(challengers: Sequence[Mapping[str, Any]]) -> str:
    """D5: the weekly cadence, compressed to a flat one-line-per-step strip --
    picks stay editable to kickoff; only the grading line freezes Tuesday."""

    header = _section_header(
        "Season ops",
        "Picks stay editable to kickoff; the grading line freezes Tuesday",
        "Five checkpoints turn that flexibility into the same routine every week.",
        top=24,
    )
    steps = "".join(
        f"<li><b>{escape(day)}</b> -- {escape(words)}.</li>" for day, words in _SEASON_OPS_STEPS
    )
    lock_line = (
        f'<p class="sub" style="margin-top:8px;">Week 1, 2026 locks {escape(_WEEK1_LOCK_LABEL)}.'
        "</p>"
    )
    return (
        header
        + f'<ul style="margin:0;padding-left:18px;" class="sub">{steps}</ul>'
        + lock_line
        + '<div style="margin-top:16px;">'
        + _movement_policy_note(challengers)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Per-game attribution ("Why this pick") -- waterfall feed, fail-open
#
# The feed is an optional artifact: absent, stale, or short (a game missing
# from ``games``), every panel degrades to a quiet note. A missing attribution
# must never block a publish, exactly like the sweep and spread-explorer
# optional artifacts above.
# ---------------------------------------------------------------------------

_ATTRIBUTION_UNAVAILABLE = (
    '<p class="fine" style="color:var(--muted);">Attribution not published.</p>'
)


def _signed_points(value: Any) -> str:
    number = _number(value)
    return "&mdash;" if number is None else f"{number:+.2f}"


_MEMBER_WORDS = {
    "coach_fade": "the year-one-coach fade",
    "division_revenge_tilt": "the division-revenge tilt",
    "player_arrests_back_side_policy": "the player-arrest policy",
    "spread_gap_zone_fade": "the mid-spread zone fade",
}


def _member_words(member_id: str) -> str:
    return _MEMBER_WORDS.get(member_id, member_id.replace("_", " "))


def _sentence_case(label: str) -> str:
    stripped = label.strip()
    return stripped[:1].upper() + stripped[1:] if stripped else stripped


def _why_this_pick_panel(
    entry: Mapping[str, Any] | None,
    *,
    interval_text: str = "",
) -> str:
    """The expandable per-game attribution panel, built from feed fields only.

    2026-08-22 de-clutter revision: the panel lives on the game's DETAIL card
    (one click from the board), not on the board row; rationale is capped at
    three sentences; margin quantiles fold into the readout line instead of
    sitting as their own paragraph on the card face.
    """

    entry_map = entry if isinstance(entry, Mapping) else None
    candidate_steps = entry_map.get("steps") if entry_map is not None else None
    steps_ok = isinstance(candidate_steps, list) and bool(candidate_steps)
    if not steps_ok or entry_map is None:
        if not interval_text:
            return _ATTRIBUTION_UNAVAILABLE
        return (
            '<details class="why-pick"><summary>Why this pick</summary>'
            f'<div style="margin-top:8px;">'
            f'<p class="fine" style="margin:0 0 8px;">{interval_text}</p>'
            f"{_ATTRIBUTION_UNAVAILABLE}</div></details>"
        )
    steps = candidate_steps

    readouts = []
    edge = _number(entry_map.get("edge_vs_spread"))
    if edge is not None:
        readouts.append(f"model-vs-market edge {abs(edge):.2f} pts")
    distance = _number(entry_map.get("key_number_distance"))
    if distance is not None:
        readouts.append(f"{distance:.2f} pts from the nearest key number")
    if interval_text:
        readouts.append(interval_text)
    readout_html = (
        f'<p class="fine" style="margin:0 0 8px;">{" &middot; ".join(readouts)}</p>'
        if readouts
        else ""
    )

    step_rows = []
    for step in steps or []:
        if not isinstance(step, Mapping):
            continue
        label = escape(_sentence_case(str(step.get("label", ""))))
        delta = _signed_points(step.get("delta_points"))
        cumulative = _signed_points(step.get("cumulative_points"))
        step_rows.append(
            f"<tr><td>{label}</td>"
            f'<td class="num">{delta}</td><td class="num">{cumulative}</td></tr>'
        )
    steps_table = (
        '<table class="data"><thead><tr><th>Step</th><th>Delta (pts)</th>'
        "<th>Cumulative (pts)</th></tr></thead><tbody>" + "".join(step_rows) + "</tbody></table>"
    )

    flip_items = []
    flip_events = entry_map.get("flip_events")
    for event in flip_events if isinstance(flip_events, list) else ():
        if not isinstance(event, Mapping):
            continue
        overlay = escape(str(event.get("overlay", "")))
        note = (
            "flips this pick on its own"
            if bool(event.get("would_flip_alone"))
            else "fires alongside the other members"
        )
        flip_items.append(f"<li>{overlay}: {note}</li>")
    flips_html = (
        '<p class="kicker" style="margin-top:10px;">Overlay events</p><ul>'
        + "".join(flip_items)
        + "</ul>"
        if flip_items
        else ""
    )

    raw_sentences = entry_map.get("rationale_sentences")
    sentences = [
        escape(str(sentence))
        for sentence in (raw_sentences if isinstance(raw_sentences, list) else ())
        if sentence
    ][:3]
    rationale_html = (
        '<div class="marginalia"><p class="kicker" style="margin-top:10px;">Rationale</p><ul>'
        + "".join(f"<li>{sentence}</li>" for sentence in sentences)
        + "</ul></div>"
        if sentences
        else ""
    )

    summary = "Why this pick"
    return (
        f'<details class="why-pick"><summary>{summary}</summary>'
        f'<div style="margin-top:8px;">{readout_html}{steps_table}{flips_html}'
        f"{rationale_html}</div></details>"
    )


def _margin_interval_text(row: pd.Series) -> str:
    """``50% [-5.6, +10.3] &middot; 80% [...]`` from the card's quantile columns.

    Renders only what the prediction artifacts actually carry: older cards
    without margin quantiles render nothing at all rather than a guess.
    """

    def band(low_key: str, high_key: str) -> str | None:
        low, high = _number(row.get(low_key)), _number(row.get(high_key))
        if low is None or high is None:
            return None
        return f"[{low:+.1f}, {high:+.1f}]"

    parts = []
    inner_50 = band("margin_lower_50", "margin_upper_50")
    if inner_50:
        parts.append(f"50% CI {inner_50}")
    inner_80 = band("margin_lower_80", "margin_upper_80")
    if inner_80:
        parts.append(f"80% CI {inner_80}")
    joined = " &middot; ".join(parts)
    return f"cover margin: {joined}" if joined else ""


def _pick_oriented_lines(row: pd.Series, pick_team: str, home: str) -> tuple[str, str | None]:
    """The market line and fair line restated as THE PICK's handicap:
    ``"-3.5"`` for a 3.5-point favorite, ``"+3.5"`` for the dog (the
    ``spread_line``/``fair_spread`` columns are home-oriented values, so the
    home side's handicap is their negation and the away side's their
    negation-flip). Fair is ``None`` when the card carries no fair spread."""

    home_spread = float(row["spread_line"])
    # home -> -value, away -> +value, for both the quoted line and our fair one.
    sign = -1.0 if pick_team == home else 1.0
    market_value = home_spread * sign
    market_text = "pick'em" if market_value == 0 else f"{market_value:+g}"
    fair = _number(row.get("fair_spread"))
    fair_text = None if fair is None else f"{fair * sign:+.1f}"
    return market_text, fair_text


def _game_deep_dive(
    row: pd.Series,
    game_sweep: pd.DataFrame,
    explanation: str,
    *,
    is_best_pick: bool = False,
    best_pick_note: str = "",
    flip: OverlayFlip | None = None,
    arrest_flip: ArrestFlip | None = None,
    production_members: tuple[str, ...] = (),
    spread_explorer_enabled: bool = False,
) -> str:
    """One flat hairline-separated prose block in the deep-dive section below
    the terminal grid.

    2026-08-23 de-firehose revision (owner's rendered-page review): the
    collapsed default is the matchup header plus ONE line -- pick, its cover
    chance and our fair line -- with everything percentage-dense (the line
    journey, the sweep curve and its 17-row table twin, the spread-explorer
    slider) folded into a single ``<details>``. The page's only dominant
    number is Panel 1's crowned stat; this block's one percentage is inline
    at reading size.
    """

    game_id = str(row["game_id"])
    home, away = str(row["home_team"]), str(row["away_team"])
    market_spread = float(row["spread_line"])
    fair = _number(row.get("fair_spread"))
    residual = _number(row.get("predicted_market_residual")) or 0.0
    pick_team, pick_probability = pick_side(row)
    strong = abs(residual) >= STRONG_LEAN_POINTS
    pick_market_text, pick_fair_text = _pick_oriented_lines(row, pick_team, home)

    # B4 fix: a flip disclosure takes priority over the market-lean
    # explanation (two numbers for one concept must never co-render).
    if production_members:
        member_text = ", ".join(_member_words(name) for name in production_members)
        explanation_html = (
            '<p class="sub" style="font-weight:600;">One of four production rules applied: '
            f"this game flipped by {escape(member_text)}.</p>"
            '<p class="fine" style="margin-top:6px;">Members are evaluated against the raw '
            "model pick; overlapping triggers are OR-composed and flip the pick exactly "
            "once. The selected archive score is selection-inflated; fresh paired "
            "tracking uses the former coach-to-arrests policy as its control.</p>"
        )
    elif arrest_flip is not None:
        # Consolidation law (2026-08-23): the policy's archive percentages
        # are accuracy statistics -- they ride inside a collapsed toggle so
        # the default view never shows them next to the picks.
        explanation_html = (
            '<p class="sub" style="font-weight:600;">Arrest rule applied: '
            f"flipped from {escape(arrest_flip.original_pick_team)} to "
            f"{escape(arrest_flip.flipped_to_team)}.</p>"
            '<details class="why-pick" style="margin-top:6px;"><summary>Policy evidence'
            "</summary>"
            '<p class="fine" style="margin-top:6px;">The sole affected team had a broad '
            "incident dated 1-14 days before Tuesday. Historically this exact opener-grade "
            f"policy scored {POLICY_OPENER_ACCURACY:.2%} versus "
            f"{POLICY_BASELINE_OPENER_ACCURACY:.2%} for the model baseline "
            f"(+{POLICY_EFFECT_ACCURACY_POINTS:.3f} points, "
            f"{glossary_abbr('P+')} {POLICY_PROBABILITY_POSITIVE:.2f}). Both arms continue to "
            "be tracked prospectively.</p></details>"
        )
    elif flip is not None:
        # Same consolidation law: the rule's historical cover rate stays
        # collapsed; the default view carries only the flip disclosure.
        explanation_html = (
            '<p class="sub" style="font-weight:600;">Coach-fade overlay applied: flipped from '
            f"{escape(flip.year_one_team)} (the model&#8217;s own pick) to "
            f"{escape(flip.opponent_team)}.</p>"
            '<details class="why-pick" style="margin-top:6px;"><summary>Rule evidence</summary>'
            '<p class="fine" style="margin-top:6px;">'
            f"{escape(flip.year_one_team)}&#8217;s head coach is in year 1 and "
            f"{escape(flip.opponent_team)}&#8217;s is not; that matchup has covered only "
            "about 47% of the time against the market's own price in weeks 1-8 since 2018 "
            "-- a real-looking gap, but not yet confirmed outside the years it was found "
            "in. We publish and track both versions of every pick this rule touches."
            "</p></details>"
        )
    elif strong and not explanation:
        # Fail-quiet (2026-08-23): a promising kicker above an unpublished
        # breakdown reads as a broken promise -- omit the whole block.
        explanation_html = ""
    elif strong:
        lean_text = (
            f"We make this line {abs(residual):.1f} points different from the "
            f"market, on the {pick_team} side."
        )
        story = f'<p class="prose" style="margin-top:6px;">{escape(explanation)}</p>'
        explanation_html = (
            '<p class="kicker">What we think the market is missing</p>'
            f'<p class="sub" style="font-weight:600;">{escape(lean_text)}</p>{story}'
        )
    else:
        explanation_html = ""

    # LICENSING (MKT-09 provider licensing/quota audit, ROADMAP.md): the public
    # site plots ONLY the one consensus market line this card already publishes
    # and our own fair line. The internal dashboard also plots an
    # archive-derived opener consensus and a predicted close; both stay off the
    # public site until that audit clears redistribution.
    #
    # Everything charted below lives inside ONE collapsed details toggle:
    # the sweep curve (with its table-view twin), the market-vs-fair line
    # journey, and the spread-explorer slider. Collapsed, this block shows a
    # single percentage; expanded, every number the old flat layout had.
    tools: list[str] = []
    if not game_sweep.empty:
        pick_is_home = pick_team == home
        points = [
            (float(offset), float(probability) if pick_is_home else 1.0 - float(probability))
            for offset, probability in zip(
                game_sweep["line_offset"],
                game_sweep["home_cover_probability"],
                strict=True,
            )
        ]
        tools.append(
            viz.sweep_curve(
                f"sweep-{game_id}",
                points,
                quoted_line=0.0,
                pick_text=f"{pick_team} to cover",
                quote_label=spread_words(home, away, market_spread),
            )
        )
    tools.append(
        viz.line_journey(
            opener=market_spread, fair=fair, predicted_close=None, opener_label="market"
        )
    )
    if spread_explorer_enabled:
        tools.append(_spread_explorer_widget_html(game_id, market_spread))

    best_note = (
        f'<div style="margin-top:8px;"><p class="fine">{escape(best_pick_note)}</p></div>'
        if is_best_pick and best_pick_note
        else ""
    )

    summary_line = (
        '<p class="sub">Pick <b>'
        f"{escape(pick_team)}</b> ({escape(pick_market_text)}) &middot; covers "
        f'<span class="num">{pick_probability:.0%}</span>'
    )
    if pick_fair_text is not None:
        summary_line += (
            f' &middot; fair {escape(pick_team)} <span class="num">{pick_fair_text}</span>'
        )
    summary_line += "</p>"

    return (
        f'<section class="deep-game" id="{escape(game_id)}">'
        f'<p class="kicker">{escape(_kickoff_words(row))}</p>'
        f'<h3 class="title">{escape(away)} at {escape(home)}</h3>'
        + summary_line
        + best_note
        + (f'<div style="margin-top:10px;">{explanation_html}</div>' if explanation_html else "")
        + '<details class="line-tools" style="margin-top:12px;">'
        "<summary>Line sweep &amp; explorer</summary>"
        '<div style="margin-top:8px;display:grid;gap:14px;">'
        + "".join(tools)
        + "</div></details>"
        + "</section>"
    )


def _week_board(
    ordered: pd.DataFrame,
    flipped_by_game: Mapping[str, object],
    best_pick_id: str | None,
    why_by_game: Mapping[str, str],
) -> str:
    """P2: ONE continuous table -- kickoff/matchup/line/pick/strength at 40px,
    each game followed by an expandable sub-row carrying the why-this-pick
    steps table, the margin-interval readout and the rationale. Level-3 info
    lives HERE and nowhere else; anchors jump to the deep-dive section."""

    rows = []
    for _, row in ordered.iterrows():
        game_id = str(row["game_id"])
        home, away = str(row["home_team"]), str(row["away_team"])
        market_spread = float(row["spread_line"])
        pick_team, pick_probability = pick_side(row)
        pick_cell = f"<b>{escape(pick_team)}</b>"
        if best_pick_id is not None and game_id == best_pick_id:
            pick_cell += ' <span class="best-flag" title="Best Pick of the week">&#9733;</span>'
        if game_id in flipped_by_game:
            pick_cell += (
                ' <span class="flip-flag" title="Flipped by a production overlay -- '
                'see the note in the deep dive below">&#8646;</span>'
            )
        expansion = why_by_game.get(game_id) or ""
        if not expansion:
            expansion = _why_this_pick_panel(None, interval_text=_margin_interval_text(row))
        rows.append(
            '<tr class="board-game">'
            f'<td data-label="Kickoff">{escape(_kickoff_words(row))}</td>'
            f'<td data-label="Matchup"><a href="#{escape(game_id)}">'
            f"{escape(away)} at {escape(home)}</a></td>"
            f'<td data-label="Line" class="num">'
            f"{escape(spread_words(home, away, market_spread))}</td>"
            f'<td data-label="Pick">{pick_cell}</td>'
            f'<td data-label="Strength">{confidence_word(pick_probability)}</td>'
            "</tr>"
            f'<tr class="board-sub"><td colspan="5">{expansion}</td></tr>'
        )
    return (
        '<table class="data week-board"><thead><tr>'
        "<th>Kickoff</th><th>Matchup</th><th>Line</th><th>Pick</th><th>Strength</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def load_waterfall_feed(artifacts_root: Path) -> dict[str, dict[str, Any]]:
    """``{game_id: feed entry}`` from ``artifacts/waterfall_feed/latest.json``.

    Fail-open like every other optional artifact on this page: a missing
    pointer, a dangling run directory, malformed JSON, or a bad ``games``
    list all yield an empty map (every panel then renders its quiet
    "attribution unavailable" note), never an exception.
    """

    pointer_path = artifacts_root / "waterfall_feed" / "latest.json"
    try:
        pointer = read_json(pointer_path)
    except (ValueError, OSError):
        return {}
    latest = pointer.get("latest") if isinstance(pointer, dict) else None
    if (
        not isinstance(latest, str)
        or not latest
        or "/" in latest
        or "\\" in latest
        or ".." in latest
    ):
        return {}
    try:
        feed = read_json(artifacts_root / "waterfall_feed" / latest / "feed.json")
    except (ValueError, OSError):
        return {}
    games = feed.get("games") if isinstance(feed, dict) else None
    if not isinstance(games, list):
        return {}
    return {
        str(entry["game_id"]): entry
        for entry in games
        if isinstance(entry, dict) and "game_id" in entry
    }


_LEDGER_UNAVAILABLE_HTML = (
    '<div class="card" style="border-left:3px solid var(--warning);margin-top:14px;">'
    '<p class="kicker" style="color:var(--warning);font-weight:700;">'
    "&#9888; MODEL LEDGER UNAVAILABLE</p>"
    '<p class="fine">The challenger registry drifted or could not be read '
    "({detail}); the rest of this page is unaffected.</p></div>"
)


def load_model_ledger_html(artifacts_root: Path) -> str:
    """The rendered Model Ledger fragment, FAIL-OPEN on registry drift.

    A missing ``challengers.json`` means the ledger feature simply does not
    exist yet for this tree, so the section omits itself quietly (the same
    contract :func:`load_prospective_challengers` follows). A registry that
    EXISTS but fails validation is drift the owner should see: the section
    renders a visible warning box instead of raising, because site generation
    must never break on ledger problems. Returns "" when there is nothing to
    show; ``render_picks_page`` skips the section then.
    """

    challengers_path = artifacts_root / "prospective" / "challengers.json"
    if not challengers_path.is_file():
        return ""
    try:
        return build_and_render(
            challengers_path,
            _default_weak_signals_registry_path(),
            artifacts_root / "active_ats_model.json",
        )
    except (ValueError, OSError) as error:
        detail = escape(str(error)) or "unknown error"
        return _LEDGER_UNAVAILABLE_HTML.replace("{detail}", detail)


#: Plain-language glossary for research vocabulary that was unexplained at
#: point of use (wave-1 UX finding on the picks page: "P+", "Ledger mini",
#: "Evidence P+", "Challenger watch"). Rendered as ``<abbr title="...">``
#: tooltips -- the same mechanism the week board already uses for its
#: flip/best flags -- so the visible text stays unchanged and every term is
#: glossed in plain language where it appears.
_GLOSSARY: dict[str, str] = {
    "P+": (
        "Our confidence that a measured effect is real rather than luck; "
        "0.50 would be a coin flip. It is not an accuracy rate or a profit claim."
    ),
    "Ledger mini": (
        "A compact slice of the model ledger: candidate picking rules ranked by "
        "their best evidence. The promoted card is what actually plays."
    ),
    "Evidence P+": (
        "The highest P+ recorded across evaluations of this picking rule -- how "
        "confident we are that its effect is real rather than luck."
    ),
    "Challenger watch": (
        "Alternative picking rules tracked alongside this card in prospective "
        "evaluation. None of them change what plays this week."
    ),
}


def glossary_abbr(term: str) -> str:
    """Wrap a glossary term in an explanatory ``<abbr>`` tooltip."""

    try:
        title = _GLOSSARY[term]
    except KeyError as error:
        raise KeyError(f"term {term!r} is not in the site glossary") from error
    return f'<abbr title="{escape(title)}">{escape(term)}</abbr>'


def _ledger_mini_table(model_id: str | None, challengers: Sequence[Mapping[str, Any]]) -> str:
    """P3: top 5 arms by best-evidence P+, promoted first. Built fresh from
    the registered challenger list -- nothing hand-typed; a build with no
    challengers still shows the promoted active model row."""

    def _sort_key(entry: tuple[str, str, float | None]) -> tuple[int, float]:
        status, probability = entry[1], entry[2]
        return (0 if status == "SUPERSEDED_BY_PROMOTION" else 1, -(probability or 0.0))

    arms: list[tuple[str, str, float | None]] = []
    if model_id:
        arms.append((model_id, "promoted", None))
    for entry in challengers:
        evidence = entry.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        probability = _number(evidence.get("probability_positive"))
        label = _challenger_display_name(str(entry.get("challenger_id", "unknown")))
        status = str(entry.get("status", "unknown"))
        words = "promoted" if status == "SUPERSEDED_BY_PROMOTION" else _humanize(status).lower()
        arms.append((label, words, probability))
    arms.sort(key=_sort_key)
    rows = []
    for label, status_words, probability in arms[:5]:
        probability_text = viz.p_plus_text(probability) if probability is not None else "--"
        rows.append(
            f"<tr><td>{escape(label)}</td><td>{escape(status_words)}</td>"
            f'<td class="num">{glossary_abbr("P+")} {probability_text}</td></tr>'
        )
    return (
        '<table class="data"><thead><tr><th>Arm</th><th>Status</th>'
        f"<th>{glossary_abbr('Evidence P+')}</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


#: Challenger-watch rows shown collapsed-open before the "show all" details.
#: Six reads at a glance; the rest stay one click away, never hidden.
_CHALLENGER_WATCH_VISIBLE = 6


def _challenger_evidence_strength(entry: Mapping[str, Any]) -> float:
    """How far a challenger's best P+ sits from a coin flip, either direction
    -- the same ranking the findings page uses for its open leads (a P+ of
    0.05 is exactly as strong a signal as 0.95, just pointed the other way).
    Unmeasured challengers sort last."""

    evidence = entry.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    probability = _number(evidence.get("probability_positive"))
    return -1.0 if probability is None else abs(probability - 0.5)


def _challenger_watch_items(
    challengers: Sequence[Mapping[str, Any]],
    previews: Mapping[str, str],
) -> list[str]:
    """One ``<li>`` per ACTIVE_PROSPECTIVE challenger -- human name, best P+,
    this week's pick diff vs. promoted -- strongest evidence first."""

    active = [entry for entry in challengers if str(entry.get("status")) == "ACTIVE_PROSPECTIVE"]
    ordered = sorted(
        active,
        key=lambda entry: (
            -_challenger_evidence_strength(entry),
            str(entry.get("challenger_id", "")),
        ),
    )
    items = []
    for entry in ordered:
        challenger_id = str(entry.get("challenger_id", "unknown"))
        evidence = entry.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        probability = _number(evidence.get("probability_positive"))
        probability_text = (
            f" &middot; {glossary_abbr('P+')} {viz.p_plus_text(probability)}"
            if probability is not None
            else ""
        )
        preview = previews.get(challenger_id, "")
        preview_text = (
            f" &middot; {escape(_first_sentence(preview, max_len=60))}" if preview else ""
        )
        items.append(
            f"<li>{escape(_challenger_display_name(challenger_id))}"
            f"{probability_text}{preview_text}</li>"
        )
    return items


def _challenger_watch_panel(
    challengers: Sequence[Mapping[str, Any]],
    week_previews: Mapping[str, str] | None,
) -> str:
    """P4: active challengers in plain English -- top six by evidence strength
    visible, the rest inside a "show all" ``<details>``.

    2026-08-23 de-firehose revision: raw registry ids render as their reader
    names (:data:`_CHALLENGER_DISPLAY_NAMES`), names are plain ink (accent
    discipline -- no colored/green links), and a long roster collapses to six
    lines instead of scrolling the whole panel.
    """

    items = _challenger_watch_items(challengers, week_previews or {})
    if not items:
        body = '<p class="fine">No live challengers registered.</p>'
    else:
        body = (
            '<ul style="margin:0;padding-left:18px;" class="sub">'
            + "".join(items[:_CHALLENGER_WATCH_VISIBLE])
            + "</ul>"
        )
        if len(items) > _CHALLENGER_WATCH_VISIBLE:
            body += (
                '<details class="table-view"><summary>show all</summary>'
                '<ul style="margin:8px 0 0;padding-left:18px;" class="sub">'
                + "".join(items[_CHALLENGER_WATCH_VISIBLE:])
                + "</ul></details>"
            )
    return (
        '<section class="panel panel-watch">'
        '<h2 class="title" style="font-size:17px;margin:0 0 2px;">'
        f"{glossary_abbr('Challenger watch')}</h2>"
        '<p class="fine" style="margin:0 0 8px;">Tracked alongside the card; none change '
        f"what plays.</p>{body}</section>"
    )


#: Panel 1's ONE dominant number: the played card's HONEST EXPECTATION vs
#: Tuesday-frozen lines. 2026-08-23 owner question ("what edge am I playing"):
#: the hero was the chain's measured history, but the card actually plays the
#: four-member overlay union + market-follow refresh, whose forward
#: expectation is a de-inflated PLANNING synthesis -- pinned in
#: :mod:`nfl_ats.dashboard.findings_content` with provenance, never computed
#: from an artifact. The measured chain history is the secondary line.
_CROWNED_LABEL = "PLAYED CARD \u2014 HONEST EXPECTATION VS TUESDAY LINES"


def _crowned_stat_block(played_chain_accuracy: float | None) -> str:
    """Panel 1, per the 2026-08-23 consolidation law (owner, binding):
    EXACTLY four elements -- the label kicker, the ``≈55%`` planning hero,
    the planning-estimate dek, and ONE measured line (the played chain's
    history from :func:`load_played_chain_accuracy`; degraded to
    "Raw chain baseline" with the raw-model opener figure when that artifact
    is unreachable). Every other accuracy percentage on this page lives in
    the collapsed ceiling ladder (:func:`_ceiling_explainer_section`) -- the
    old fine print (sequential-chain composition, raw baseline, selection
    caveat) moved there, so nothing else renders in this block.
    """

    if played_chain_accuracy is not None:
        measured_line = (
            '<p class="sub" style="font-size:14px;margin-top:6px;"><strong>'
            "Measured chain history: "
            f'<span class="num">{played_chain_accuracy:.1%}</span></strong></p>'
        )
    else:
        measured_line = (
            '<p class="sub" style="font-size:14px;margin-top:6px;"><strong>'
            "Raw chain baseline: "
            f'<span class="num">{HEADLINE.opener}</span></strong></p>'
        )
    return (
        '<div class="card" style="margin-top:10px;">'
        f'<p class="kicker">{escape(_CROWNED_LABEL)}</p>'
        '<div class="num" style="font-size:24px;font-weight:600;line-height:1.15;">'
        f"{PLAYED_CARD_EXPECTATION_HERO}</div>"
        '<p class="sub" style="max-width:44ch;">Planning estimate for the played card.</p>'
        '<p class="fine" style="margin-top:6px;">'
        '<a href="track_record.html">What this number means &#8594;</a></p>'
        + measured_line
        + "</div>"
    )


def render_picks_page(
    predictions: pd.DataFrame,
    sweep: pd.DataFrame | None = None,
    explanations: Mapping[str, str] | None = None,
    *,
    season: int | None = None,
    week: int | None = None,
    model_id: str | None = None,
    generated_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
    data_root: Path | None = None,
    overlay: OverlayResult | None = None,
    arrest_overlay: ArrestOverlayResult | None = None,
    production_overlay: FourOverlayCompositionResult | None = None,
    nomination: BestPickNomination | None = None,
    spread_explorer: Mapping[str, SpreadExplorerGameParams] | None = None,
    challengers: Sequence[Mapping[str, Any]] = (),
    waterfall_feed: Mapping[str, Mapping[str, Any]] | None = None,
    challenger_week_previews: Mapping[str, str] | None = None,
    recent_form_text: str | None = None,
    played_chain_accuracy: float | None = None,
) -> str:
    """Render ``docs/index.html`` -- this week's forced picks, one card per game.

    ``predictions`` is one row per game (the active model's ``recommendations.csv``
    for the synchronized weekly forecast, UN-overlaid); ``sweep`` is the matching
    ``line_sweep.parquet`` already filtered to the active method; ``explanations``
    maps ``game_id`` to the market-decomposition sentence. ``metadata`` is the
    forecast's own ``metadata.json`` (season/week/feature_profile/...), needed
    only for the v2 Best Pick nomination rule; omit it (or ``data_root``) and
    nomination degrades to the incumbent v1 rule, exactly as
    ``nfl_ats.publishing`` degrades. An empty ``predictions`` frame renders the
    shell plus an empty state.

    B1/B2 fix (2026-08-19): this function used to render ``predictions`` raw
    and pick the Best Pick with the incumbent v1 rule only, so the public site
    could show a DIFFERENT pick (and a different Best Pick) than the one
    already on the published card. It now applies the coach-fade overlay and
    resolves the Best Pick nomination through :func:`nfl_ats.card_view.resolve_card_view`
    -- the same shared implementation ``nfl_ats.publishing`` uses -- so the two
    can never disagree again.

    Only the allowlisted public fields are rendered -- see the module docstring.

    ``spread_explorer`` (2026-08-20, owner request) is an optional
    ``{game_id: SpreadExplorerGameParams}`` map -- see
    :mod:`nfl_ats.spread_explorer`. ``build_public_site`` computes and
    build-time-verifies this via a refit before ever passing it here (see
    :func:`_assert_spread_explorer_matches_card`); a game absent from the map
    simply renders without the widget, the same graceful-degradation
    contract every other optional artifact on this page follows.

    ``challengers`` (2026-08-20, owner request) is the registered-prospective-
    challenger list -- see :func:`load_prospective_challengers` -- passed
    through only so the season-ops timeline's movement-policy note
    (:func:`_movement_policy_note`) can quote ``model_only_refresh_incumbent``'s
    own registered evidence sentence instead of a hand-typed number. Omitting
    it (every direct caller/test that does not pass it) degrades that one
    note to a generic pointer at the findings page; nothing else on this page
    is affected.

    ``challenger_week_previews`` feeds P4's one-line pick diffs; ``recent_form_text``
    (computed by ``build_public_site`` from prospective scoring) feeds P1's
    recent-form line -- both optional and degrading quietly when absent.

    ``played_chain_accuracy`` (2026-08-23 de-firehose revision) is the active
    model's sequential played-chain opener accuracy -- raw model -> coach fade
    -> player-arrests policy, read by :func:`load_played_chain_accuracy` from
    the newest ``overlay_subset_composition`` run. It is Panel 1's MEASURED
    history line beneath the crowned hero; the hero itself is the pinned
    planning constant ``≈55%``
    (:data:`~nfl_ats.dashboard.findings_content.PLAYED_CARD_EXPECTATION_HERO`)
    and never comes from an artifact. ``None`` (an older artifacts tree)
    degrades the measured line to the raw-model opener baseline
    (:data:`~nfl_ats.dashboard.findings_content.HEADLINE`), labeled exactly
    "Raw chain baseline", never inventing a chain figure.

    Consolidation law (2026-08-23, owner, binding): the default view carries
    exactly two accuracy statistics (the hero and the measured chain line)
    plus the per-game cover chances. There is deliberately no
    ``historical_accuracy`` footer byline anymore -- every other aggregate
    lives in the collapsed ceiling ladder or on track_record.html.
    """

    explanations = explanations or {}
    sweep = sweep if sweep is not None else pd.DataFrame()
    metadata = metadata or {}
    spread_explorer = spread_explorer or {}
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)

    model_text = f"model <code>{escape(model_id)}</code>" if model_id else "model unknown"

    if predictions.empty:
        body = viz.page_header("This week", "No pick card yet") + viz.empty_state(
            "No games are scheduled for this week's forecast yet",
            "Once the week's opening line is captured and a forecast card is built, this "
            "page fills in by itself. The track record is open in the meantime.",
        )
        return _page(
            current=PICKS_PAGE,
            body=body,
            generated=generated,
            footer_note=model_text,
        )

    game_type = str(predictions["game_type"].iloc[0]) if "game_type" in predictions else "REG"
    week_label = _WEEK_LABELS.get(game_type, f"Week {week}")
    season_label = f"{season} · " if season is not None else ""

    # What actually gets submitted: the coach-fade overlay + Best Pick
    # nomination, through the one shared implementation every surface uses.
    # Both are injectable so ``build_public_site`` can compute them ONCE and
    # share the result with the track-record page's Best Pick section rather
    # than paying v2's cross-book dispersion-pool scan twice per site build.
    if overlay is None:
        overlay = resolve_overlay(predictions, data_root)
    if arrest_overlay is None:
        arrest_overlay = resolve_player_arrests_overlay(
            overlay.overlaid_predictions, data_root, now=generated
        )
    recommendations = (
        production_overlay.overlaid_predictions
        if production_overlay is not None
        else arrest_overlay.overlaid_predictions
    )
    flipped_by_game = {flip.game_id: flip for flip in overlay.flips}
    arrest_flipped_by_game = {flip.game_id: flip for flip in arrest_overlay.flips}
    production_members_by_game = (
        {game.game_id: game.member_ids for game in production_overlay.games}
        if production_overlay is not None
        else {}
    )

    sort_columns = [column for column in ("kickoff", "game_id") if column in recommendations]
    ordered = (
        recommendations.sort_values(sort_columns, na_position="last")
        if sort_columns
        else recommendations
    )

    # B3 fix (2026-08-23): the summary's strength counts are COMPUTED from the
    # same frame the board renders, using the same confidence_word buckets --
    # never from a different threshold on a different frame.
    strong_count = sum(
        1 for _, row in ordered.iterrows() if confidence_word(pick_side(row)[1]) == "strong"
    )
    header = viz.page_header(
        f"{season_label}{week_label} · {len(recommendations)} games",
        "This week's picks",
        "Picks are graded against Tuesday-frozen lines all season.",
    )

    # POL-09: the week's Best Pick nomination, resolved exactly as
    # `publish-predictions` would choose it -- v2's calibrated-probability
    # signal when it can be computed, the incumbent v1 sweep_robustness
    # signal otherwise. Scored on the model's OWN, un-overlaid picks (the
    # overlay never influences which GAME is nominated), regular season only.
    best_pick_id: str | None = None
    best_pick_note = ""
    if game_type == "REG":
        resolved_nomination = (
            nomination
            if nomination is not None
            else resolve_nomination(predictions, sweep, metadata, data_root)
        )
        best_pick_id = resolved_nomination.active_game_id
        if best_pick_id is not None:
            if resolved_nomination.active_rule == "v2":
                best_pick_note = f"This pick was {resolved_nomination.method_note}"
            else:
                tie = (
                    f" {resolved_nomination.active_tie_note}"
                    if resolved_nomination.active_tie_note
                    else ""
                )
                best_pick_note = (
                    "This is the pick whose edge survives the widest range of line "
                    "movement -- the best-measured lever among forced picks, budgeted at "
                    "roughly +0.9 points, not the +8.7 once recorded before a tie-break "
                    f"audit.{tie}"
                )

    has_sweep = not sweep.empty and {"game_id", "line_offset", "home_cover_probability"}.issubset(
        sweep.columns
    )

    # Level-3 attribution is built per game ONCE and lives ONLY in the board's
    # expandable sub-rows; the deep-dive blocks below carry level-2 detail.
    why_by_game = {
        str(row["game_id"]): _why_this_pick_panel(
            (waterfall_feed or {}).get(str(row["game_id"])),
            interval_text=_margin_interval_text(row),
        )
        for _, row in ordered.iterrows()
    }
    deep_blocks = []
    for _, row in ordered.iterrows():
        game_id = str(row["game_id"])
        game_sweep = pd.DataFrame()
        if has_sweep:
            game_sweep = sweep.loc[
                sweep["game_id"].astype(str).eq(game_id)
                & sweep["line_offset"].abs().le(SWEEP_HALF_WIDTH)
            ].sort_values("line_offset")
        deep_blocks.append(
            _game_deep_dive(
                row,
                game_sweep,
                explanations.get(game_id, ""),
                is_best_pick=best_pick_id is not None and game_id == best_pick_id,
                best_pick_note=best_pick_note,
                flip=flipped_by_game.get(game_id),
                arrest_flip=arrest_flipped_by_game.get(game_id),
                production_members=production_members_by_game.get(game_id, ()),
                spread_explorer_enabled=game_id in spread_explorer,
            )
        )

    flipped_game_ids = (
        set(production_members_by_game)
        if production_overlay is not None
        else set(flipped_by_game) | set(arrest_flipped_by_game)
    )
    week_board = _week_board(ordered, dict.fromkeys(flipped_game_ids), best_pick_id, why_by_game)
    board_legend = (
        '<p class="fine" style="margin-top:8px;">&#9733; best pick &middot; &#8646; flipped '
        "by an overlay rule &middot; strength runs slight &lt; lean &lt; strong, by "
        "model-vs-market gap.</p>"
    )

    # P1 summary: composition line, recent form when computable, Best Pick.
    composition = ["Synchronized with the active model"]
    if strong_count:
        composition.append(f"{strong_count} strong lean{'s' if strong_count != 1 else ''}")
    if production_overlay is not None:
        composition.append(
            f"{production_overlay.flip_count} pick"
            f"{'s' if production_overlay.flip_count != 1 else ''} flipped by the fix-up rules"
        )
    elif overlay.flip_count:
        composition.append(
            f"{overlay.flip_count} pick{'s' if overlay.flip_count != 1 else ''} flipped by "
            "the coach-fade overlay"
        )
    if arrest_overlay.enabled and production_overlay is None:
        composition.append(
            f"player-arrest policy active &middot; {arrest_overlay.flip_count} pick"
            f"{'s' if arrest_overlay.flip_count != 1 else ''} flipped this week"
        )
    best_callout = ""
    if best_pick_id is not None:
        best_row = recommendations.loc[recommendations["game_id"].astype(str).eq(best_pick_id)]
        if not best_row.empty:
            best_team, _ = pick_side(best_row.iloc[0])
            best_callout = (
                '<div class="card"><p class="kicker">The pool scores one Best Pick a week</p>'
                '<p class="sub" style="font-weight:600;color:var(--good-text);">'
                '&#9733; BEST PICK OF THE WEEK: <span class="num">'
                f"{escape(best_team)}</span></p>"
                f'<p class="fine" style="margin-top:6px;">{escape(best_pick_note)}</p></div>'
            )
    summary_panel = (
        '<section class="panel panel-summary">'
        '<h2 class="title" style="font-size:17px;margin:0 0 4px;">At a glance</h2>'
        + _crowned_stat_block(played_chain_accuracy)
        + '<p class="sub" style="margin-top:8px;">'
        + " &middot; ".join(escape(part) for part in composition)
        + "</p>"
        + (
            f'<p class="fine" style="margin-top:8px;">{escape(recent_form_text)}</p>'
            if recent_form_text
            else ""
        )
        + best_callout
        + "</section>"
    )
    ledger_panel = (
        '<section class="panel panel-ledger">'
        '<h2 class="title" style="font-size:17px;margin:0 0 2px;">'
        f"{glossary_abbr('Ledger mini')}</h2>"
        '<p class="fine" style="margin:0 0 8px;">Top arms by best evidence; the promoted '
        f'card plays. <a href="{MODELS_PAGE}">Full ledger</a>.</p>'
        + _ledger_mini_table(model_id, challengers)
        + "</section>"
    )
    watch_panel = _challenger_watch_panel(challengers, challenger_week_previews)
    grid = (
        '<div class="ledger-grid">'
        + summary_panel
        + f'<section class="panel panel-board">{week_board}{board_legend}</section>'
        + ledger_panel
        + watch_panel
        + "</div>"
    )

    ops_timeline = _season_ops_timeline_section(challengers)
    spread_explorer_intro = _spread_explorer_intro(generated) if spread_explorer else ""
    deep_dive = (
        _section_header("Game notes", "One block per game", "Anchored from the board above.")
        + spread_explorer_intro
        + "".join(deep_blocks)
    )

    return _page(
        current=PICKS_PAGE,
        body=(header + grid + deep_dive + ops_timeline),
        generated=generated,
        footer_note=(
            # Consolidation law: no accuracy percentages in the default view.
            f"{model_text} &middot; lines are home-oriented "
            "spreads at card-build time; the pool's exact number can differ by a half point"
        ),
        # No host sanitizer on a static page, so the sweep's delegated
        # crosshair/tooltip wiring (and the spread-explorer widget's own script,
        # below) ship as their own script tags.
        scripts=viz.interaction_script()
        + _spread_explorer_script(spread_explorer_payload(spread_explorer)),
    )


# ---------------------------------------------------------------------------
# Page 2 -- What we've learned
# ---------------------------------------------------------------------------


def _rows(cards: Sequence[str], *, per_row: int = 2) -> str:
    """Lay cards out ``per_row`` across, wrapping on narrow screens.

    Each card gets its own grid cell so a stacking margin never fires between
    side-by-side cards.
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
        f'<h3 class="title" style="margin-bottom:6px;">{escape(title)}</h3>'
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


def _findings_hero() -> str:
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


def _research_funnel_section(
    *, total_signals: int, active_challengers: int, has_active_model: bool
) -> str:
    """A three-number "shape of the pipeline" strip: every idea tested, down
    to what is actually live, down to what is actually published.

    Every count is computed fresh at build time from the same files every
    other section on this page already reads -- ``total_signals`` from
    ``registry/weak_signals.json`` (the same number "What we're watching"
    quotes in its own count line), ``active_challengers`` from
    ``artifacts/prospective/challengers.json`` (the same list the
    challenger cards below are built from), ``has_active_model`` from
    whether a synchronized active model produced this build at all. Nothing
    here is typed in by hand, so it can never drift from the sections it
    summarizes.

    First use of "challenger" on this page (the dedicated section further
    down explains it again at length) -- the tile's own context sentence
    defines it inline rather than assuming the reader already knows the
    word.
    """

    tiles = _rows(
        [
            viz.stat_tile(
                "Signals recorded",
                f"{total_signals:,}",
                "Every effect this project has measured and logged, resolved or not -- "
                "nothing that gets tested is thrown away, including the negatives.",
            ),
            viz.stat_tile(
                "Live 2026 challengers",
                str(active_challengers),
                "Alternative picking rules and pick-flip overlays -- 'challengers' -- riding "
                "along the model's real weekly card this season, scored against it game for "
                "game. None of them change what actually gets played.",
            ),
            viz.stat_tile(
                "Active model",
                "1" if has_active_model else "0",
                "The one configuration whose picks are the ones actually published each "
                "week. Everything else here is either a past measurement or a challenger "
                "riding alongside it, never the pick itself.",
            ),
        ],
        per_row=3,
    )
    header = _section_header(
        "The research pipeline",
        "From every idea tested to what's actually played",
        "Three honest counts, computed fresh from the same files every other section on "
        "this page reads -- nobody updates these by hand.",
        top=8,
    )
    return header + tiles


def _emphasized(text: str) -> str:
    """Escape ``text`` for HTML, then render ``**spans**`` as <strong>.

    Escaping happens FIRST, so the emphasis markers are the only markup a
    content constant can introduce -- the emphasis pass cannot be abused to
    inject tags. (2026-08-24: the owner's formatting law -- prose blocks must
    carry visual hierarchy, not render as undifferentiated walls.)
    """

    escaped = escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def _finding_card(finding: Finding, group: VerdictGroup) -> str:
    inner = (
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        'gap:14px;margin-bottom:10px;">'
        f'<p class="title" style="max-width:38ch;">{escape(finding.question)}</p>'
        f'<span style="flex:none;">{_verdict_chip(group)}</span>'
        "</div>"
        f'<div class="prose"><p>{_emphasized(finding.plain_answer)}</p></div>'
        '<details class="table-view"><summary>'
        f"{escape(DETAIL_SUMMARY_LABEL)}</summary>"
        f'<p class="fine" style="margin:10px 0 0;max-width:68ch;">{_emphasized(finding.detail)}</p>'
        f'<p class="fine" style="margin:8px 0 0;">{escape(SOURCE_LABEL)}: '
        f'<span style="color:var(--ink-2);">{escape(finding.source)}</span></p>'
        "</details>"
    )
    return viz.card(inner, accent=group.verdict == "helps")


def _group_section(group: VerdictGroup) -> str:
    findings = findings_for(group.verdict)
    header = _section_header(f"{group.kicker} · {len(findings)}", group.title, group.blurb)
    return header + _rows([_finding_card(finding, group) for finding in findings])


#: Plain-English words for the units the weak-signal registry stores. Kept
#: here (rendering), not in ``findings_registry`` (content model), matching
#: this file's existing split between the two modules.
_EFFECT_UNIT_WORDS = {
    "accuracy_points": "accuracy points",
    "ats_points": "line points",
    "brier": "Brier-score points",
    "log_loss": "log-loss points",
    "mae": "points of average error",
}


def _lead_direction_sentence(probability_positive: float) -> str:
    """State a P+ near either end as a lead, pointed the right way.

    AGENTS.md, binding: "a P+ of 0.05 is a lead for the OTHER side" -- a raw
    "P+ 0.05" tile reads like noise to a casual reader even though it is
    exactly as strong a signal as "P+ 0.95", just facing the other
    direction. This sentence states the direction and the confidence in
    THAT direction in words; the raw ``probability_positive`` is still
    reported unchanged in its own tile below (never replaced -- AGENTS.md:
    "Report probability_positive, never 'contains zero'").
    """

    if probability_positive >= 0.5:
        return (
            f"Leans FOR the pattern described below -- {probability_positive:.0%} "
            "confidence in that direction (not yet resolved; see the interval)."
        )
    against = 1.0 - probability_positive
    return (
        "Leans AGAINST the pattern described below -- read this as a lead for the "
        f"OTHER side, {against:.0%} confidence in that direction (not yet resolved; "
        "see the interval)."
    )


def _effect_whisker(
    effect: float, interval: tuple[float, float] | None, *, width: int = 220
) -> str:
    """A compact dot-and-whisker: the point estimate plus its interval, zero marked.

    Pure HTML/CSS percent-positioned ``<div>``s, matching every other chart
    in this design system (no SVG -- see ``dashboard.viz``'s module
    docstring). Each lead sets its OWN axis from its own effect/interval,
    like ``viz.sweep_curve``'s per-game axis -- these are independent small
    multiples, not a shared scale across leads with wildly different units
    (accuracy points vs. Brier-score points vs. line points).
    """

    lo, hi = interval if interval is not None else (effect, effect)
    span_lo, span_hi = min(lo, effect, 0.0), max(hi, effect, 0.0)
    span = (span_hi - span_lo) or max(abs(effect), 1.0)
    pad = span * 0.18
    axis_lo, axis_hi = span_lo - pad, span_hi + pad
    axis_span = (axis_hi - axis_lo) or 1.0

    def pct(value: float) -> float:
        return (value - axis_lo) / axis_span * 100.0

    whisker_html = (
        f'<div style="position:absolute;left:{pct(lo):.2f}%;'
        f"width:{max(pct(hi) - pct(lo), 0.6):.2f}%;top:50%;height:2px;"
        'background:var(--series-model);transform:translateY(-50%);"></div>'
        if interval is not None
        else ""
    )
    interval_words = f", interval {lo:+.3f} to {hi:+.3f}" if interval is not None else ""
    return f"""
<div style="position:relative;height:20px;max-width:{width}px;" role="img"
     aria-label="Effect {effect:+.3f}{interval_words}, zero marked">
  <div style="position:absolute;left:{pct(0.0):.2f}%;top:-2px;bottom:-2px;width:0;
              border-left:1px dashed var(--baseline);"></div>
  {whisker_html}
  <div style="position:absolute;left:{pct(effect):.2f}%;top:50%;width:9px;height:9px;
              border-radius:50%;background:var(--series-model);border:2px solid var(--surface);
              transform:translate(-50%,-50%);"></div>
  <span class="fine" style="position:absolute;left:{pct(0.0):.2f}%;top:100%;
        transform:translateX(-50%);font-size:9px;">0</span>
</div>
"""


def _era_magnitude_row(rows: Sequence[EraMagnitude]) -> str:
    """A small per-era dot-and-whisker strip: same construct, three time
    windows, one whisker each -- reuses :func:`_effect_whisker` unchanged, so
    it draws with the same zero-marked axis convention as every other effect
    on this page.

    Per the era-magnitude finding this exists to show (docs/era_magnitude_profile.md):
    a weaker-looking era is a magnitude reading, never an absence -- the
    caption says so explicitly rather than leaving a reader to infer it from
    three bars of different heights.
    """

    if not rows:
        return ""
    items = "".join(
        '<div style="min-width:118px;">'
        f'<p class="fine num" style="margin-bottom:4px;">{escape(row.era_label)}</p>'
        f"{_effect_whisker(row.effect, row.interval, width=140)}"
        "</div>"
        for row in rows
    )
    return (
        '<div style="margin:10px 0 8px;padding-top:8px;border-top:1px solid var(--grid);">'
        '<p class="kicker">Same pattern, three eras</p>'
        '<p class="fine" style="margin-bottom:8px;">Magnitude moving across eras is the '
        "expected shape for a real effect -- a weaker-reading era is not the same thing as "
        "no effect there.</p>"
        f'<div class="row" style="gap:14px;flex-wrap:wrap;">{items}</div></div>'
    )


def _watching_lead_card(
    lead: WatchingLead, blurb: LeadBlurb | None, era_rows: Sequence[EraMagnitude] = ()
) -> str:
    units_words = _EFFECT_UNIT_WORDS.get(lead.effect_units, lead.effect_units)
    interval_text = (
        f"95% [{lead.interval[0]:+.2f}, {lead.interval[1]:+.2f}]"
        if lead.interval is not None
        else "no interval recorded"
    )
    league_words = "NFL" if lead.league == "nfl" else lead.league.upper()
    headline_text = blurb.text if blurb is not None else lead.description
    registry_link = (
        '<p class="fine" style="margin-top:8px;"><a href="../registry/weak_signals.json">'
        "details in the research registry</a></p>"
        if blurb is not None
        else ""
    )
    inner = (
        f'<p class="prose" style="margin-bottom:6px;">{escape(headline_text)}</p>'
        '<p class="fine" style="margin-bottom:8px;">'
        f"{escape(_lead_direction_sentence(lead.probability_positive))}</p>"
        f'<div style="margin-bottom:10px;">{_effect_whisker(lead.effect, lead.interval)}</div>'
        '<div class="row" style="gap:16px;flex-wrap:wrap;">'
        '<div><p class="kicker">Effect</p>'
        f'<p class="sub num">{lead.effect:+.2f} {escape(units_words)}</p></div>'
        '<div><p class="kicker">Interval</p>'
        f'<p class="sub num">{interval_text}</p></div>'
        '<div><p class="kicker">Chance it is real</p>'
        f'<p class="sub num">P+ {viz.p_plus_text(lead.probability_positive)}</p></div>'
        '<div><p class="kicker">Where measured</p>'
        f'<p class="sub">{escape(league_words)}, {lead.seasons[0]}-{lead.seasons[1]}</p></div>'
        "</div>" + _era_magnitude_row(era_rows) + registry_link
    )
    return viz.card(inner)


def _watching_section(
    leads: Sequence[WatchingLead],
    *,
    total_signals: int,
    shown: int,
    blurbs_by_signal: Mapping[str, LeadBlurb] | None = None,
    era_magnitude: Mapping[str, Sequence[EraMagnitude]] | None = None,
) -> str:
    """ "What we're watching": generated 100% from ``registry/weak_signals.json``
    at build time -- no hand-typed prose, no key to wire, no way to go stale.

    A small, hand-curated subset (``blurbs_by_signal``, from
    :data:`nfl_ats.dashboard.findings_content.LEAD_BLURBS`) gets a plainer
    one-liner in place of the registry's own research-toned ``description``;
    every other lead falls back to that description unchanged -- still a
    written English sentence, just a more technical one. Curation is
    optional by design (see :func:`nfl_ats.findings_registry.validate_curation`,
    called on ``LEAD_BLURBS`` in :func:`render_findings_page`), so a brand
    new registry entry renders correctly with zero code change.

    Render-semantics contract (AGENTS.md, binding): every lead here is
    ``unresolved_below_power``. That classification is NOT a negative and is
    never rendered as "failed" or "no effect" -- this section shows the
    effect, the interval, and ``probability_positive`` and calls it an open
    lead below the instrument's resolving power, exactly as the rule
    requires. The phrase "contains zero" never appears; an interval crossing
    zero is stated as the expected shape for a real small signal, not a
    verdict. A P+ below 0.5 is rendered as a lead for the OTHER side (see
    :func:`_lead_direction_sentence`), never as a weaker or failed lead.
    """

    if not leads:
        return ""
    blurbs_by_signal = blurbs_by_signal or {}
    era_magnitude = era_magnitude or {}
    header = _section_header(
        "What we're watching",
        "The open leads, generated fresh every time this page builds",
        "Every card below comes straight from registry/weak_signals.json at build time -- "
        "nobody typed these in, and nobody has to update them when new evidence is recorded. "
        "Each is 'unresolved_below_power': the interval crosses zero, which at this "
        "evaluator's roughly 2-point resolution is the EXPECTED shape for a real small "
        "signal, not a verdict either way. Ranked by how far the lean sits from a coin flip "
        "in EITHER direction -- a lead near 0% is exactly as strong as one near 100%, just "
        "pointed the other way.",
        top=42,
    )
    count_line = (
        f'<p class="fine" style="margin:-8px 0 12px;">{total_signals} recorded signals; '
        f"{shown} leads shown here; the registry is the full record.</p>"
    )
    cards = _rows(
        [
            _watching_lead_card(
                lead,
                blurbs_by_signal.get(lead.name),
                _era_magnitude_for_lead(lead.name, era_magnitude),
            )
            for lead in leads
        ]
    )
    return header + count_line + cards


def _honesty_section() -> str:
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


def render_findings_page(
    *,
    generated_at: datetime | None = None,
    registry_root: Path | None = None,
    weak_signal_registry: WeakSignalRegistry | None = None,
    challengers: Sequence[Mapping[str, Any]] = (),
    challenger_week_previews: Mapping[str, str] | None = None,
    challenger_prospective_records: Mapping[str, str] | None = None,
    artifacts_root: Path | None = None,
    active_model_id: str | None = None,
) -> str:
    """Render ``docs/findings.html``: curated findings, then two sections
    generated straight from the machine-readable evidence stores.

    Three content sources, in the order they appear on the page:

    1. The hand-curated :data:`~nfl_ats.dashboard.findings_content.FINDINGS`,
       grouped by verdict. Every non-``evergreen`` entry is validated against
       the LIVE registries before anything renders --
       :func:`nfl_ats.findings_registry.validate_curation` raises
       :class:`~nfl_ats.findings_registry.CurationError` the instant a cited
       key no longer exists or its recorded content has moved since the
       prose was last verified, so a stale claim fails the build loudly
       instead of shipping quietly. :data:`~nfl_ats.dashboard.findings_content.LEAD_BLURBS`
       (the small, hand-curated subset of "What we're watching" leads below)
       is validated the SAME way, through the SAME function.
    2. "What we're watching" (:func:`_watching_section`): the open,
       ``unresolved_below_power`` leads, ranked and rendered with no prose to
       write -- see :func:`nfl_ats.findings_registry.top_open_leads`.
    3. The tracked prospective challengers (:func:`_challengers_section`,
       already used by the track-record page -- reused here rather than
       duplicated).

    ``registry_root``/``weak_signal_registry``/``challengers`` are
    injectable for tests; production (``build_public_site``) leaves the
    first two at their tracked-registry defaults and passes the same
    already-loaded ``challengers`` list the track-record page uses.
    ``challenger_week_previews``/``challenger_prospective_records`` are
    optional per-challenger-id sentence maps (see
    :func:`_challenger_week_previews`/:func:`_challenger_prospective_records`);
    omitting them (the default for direct callers/tests) simply renders each
    challenger card without a "this week" line and with the generic "not
    scored yet" record text. ``artifacts_root``, if given, additionally
    feature-detects ``artifacts/era_magnitude_profile/`` for the per-era
    magnitude row on the ``era_trend_*`` lead cards (see
    :func:`load_era_magnitude_profile`); omitting it just renders those
    cards without that row. ``active_model_id`` feeds only the research
    funnel strip's "active model" count (0 or 1) -- everything else on the
    page is unaffected by it.
    """

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)

    registry = (
        weak_signal_registry
        if weak_signal_registry is not None
        else load_weak_signal_registry(registry_root)
    )
    # `registry` -- whether injected (tests) or disk-loaded (production) --
    # backs BOTH curation validation and the leads below. Re-reading a
    # second, independent copy from disk for validation would let the two
    # silently disagree whenever a caller injects its own registry.
    entries = load_all_entries(
        registry_root=registry_root, weak_signal_registry=registry, challengers=challengers
    )
    validate_curation(FINDINGS, entries)
    validate_curation(LEAD_BLURBS, entries)

    leads = top_open_leads(registry)
    blurbs_by_signal = {blurb.weak_signal_name: blurb for blurb in LEAD_BLURBS}
    era_magnitude = load_era_magnitude_profile(artifacts_root) if artifacts_root is not None else {}
    active_challengers = sum(
        1 for entry in challengers if str(entry.get("status")) == "ACTIVE_PROSPECTIVE"
    )
    body = (
        _findings_hero()
        + '<p class="sub" style="max-width:70ch;margin:-6px 0 0;">The evidence library '
        "&#8212; what the bare model does, what we have learned, and the leads still "
        'open. The story of the edge itself: <a href="track_record.html">How good is '
        "this, honestly? &#8594;</a></p>"
        + _research_funnel_section(
            total_signals=len(registry.signals),
            active_challengers=active_challengers,
            has_active_model=bool(active_model_id),
        )
        + "".join(_group_section(group) for group in GROUPS)
        + _watching_section(
            leads,
            total_signals=len(registry.signals),
            shown=len(leads),
            blurbs_by_signal=blurbs_by_signal,
            era_magnitude=era_magnitude,
        )
        + _challengers_section(
            challengers,
            week_previews=challenger_week_previews,
            prospective_records=challenger_prospective_records,
        )
        + _honesty_section()
    )
    return _page(
        current=FINDINGS_PAGE,
        body=body,
        generated=generated,
        footer_note="every claim traces to a committed record in this repository",
    )


# ---------------------------------------------------------------------------
# Page 3 -- Track record
# ---------------------------------------------------------------------------


def _mapping(source: Any, key: str) -> dict[str, Any]:
    value = source.get(key) if isinstance(source, dict) else None
    return value if isinstance(value, dict) else {}


def _versus_coin_flip(value: float) -> str:
    return f"{(value - 0.5) * 100:+.1f} points vs. a coin flip"


def _plausible_range(lower: float, upper: float) -> str:
    return f"could plausibly sit anywhere from {lower:.1%} to {upper:.1%}"


def _season_interval(uncertainty: Any, metric: str) -> tuple[float, float] | None:
    """The season-blocked range for one metric, out of the artifact's own list."""

    if not isinstance(uncertainty, list):
        return None
    for entry in uncertainty:
        if not isinstance(entry, dict):
            continue
        if entry.get("metric") != metric or entry.get("block") != "season":
            continue
        lower, upper = _number(entry.get("lower")), _number(entry.get("upper"))
        if lower is not None and upper is not None:
            return lower, upper
    return None


def _spaced(inner: str) -> str:
    """Vertical rhythm between the track-record page's stacked sections."""

    return f'<div style="margin-top:16px;">{inner}</div>'


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{escape(name)}</th>" for name in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        '<div style="overflow-x:auto;">'
        f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


@dataclass(frozen=True)
class _GradingRuleGrades:
    """Both pick rules' opener/close grades, read once from one
    opener-evaluation artifact's ``metrics`` and shared by the track-record
    tiles and the rule explainer below them, so the two sections can never
    quote numbers that disagree with each other.

    2026-08-19, owner decision: the headline grades the rule production
    actually plays (``home_cover_probability >= 0.5``, the "production"
    fields below). The original protocol graded the sign rule
    (``residual > 0``, the "protocol" fields) -- an instrument infidelity,
    since no pick was ever chosen that way (``pool.py``). Both are reported:
    the production rule leads, the protocol figure stays as provenance.
    Older artifacts without the ``*_probability_rule`` keys leave the
    production fields ``None``; callers fall back to the protocol fields
    (see :func:`docs/opener_evaluation.md` addendum).
    """

    protocol_opener: float | None
    protocol_close: float | None
    production_opener: float | None
    production_close: float | None


def _grading_rule_grades(opener_metadata: Mapping[str, Any]) -> _GradingRuleGrades:
    metrics = _mapping(dict(opener_metadata), "metrics")
    return _GradingRuleGrades(
        protocol_opener=_number(metrics.get("opener_accuracy")),
        protocol_close=_number(metrics.get("close_accuracy")),
        production_opener=_number(metrics.get("opener_accuracy_probability_rule")),
        production_close=_number(metrics.get("close_accuracy_probability_rule")),
    )


def _rule_explainer_section(opener_metadata: Mapping[str, Any]) -> str:
    """Explain the raw model rules and the composed played policy -- reads the SAME
    :func:`_grading_rule_grades` the tiles below use, so the numbers here
    can never drift from the tile numbers.
    """

    grades = _grading_rule_grades(opener_metadata)
    production_words = (
        f"scores {grades.production_opener:.1%} at the opener on this archive"
        if grades.production_opener is not None
        else "has not been measured on this archive yet"
    )
    protocol_words = (
        f"scores {grades.protocol_opener:.1%} on the same games"
        if grades.protocol_opener is not None
        else "has not been measured on this archive yet"
    )
    inner = (
        '<p class="kicker">How the picks are graded</p>'
        '<p class="title" style="margin-bottom:10px;">The model baseline and played policy</p>'
        '<div class="prose">'
        "<p><b>The raw model probability rule -- the baseline beneath today's card:</b> "
        "pick whichever team the model gives at least a 50% chance to cover. It "
        f"{production_words} -- this is the opener baseline quoted in the story's "
        "MEASURED section above.</p>"
        "<p><b>The played policy:</b> apply the year-1-coach policy, then the promoted "
        "player-arrest policy. The arrest component's frozen opener evaluation scored "
        f"{POLICY_OPENER_ACCURACY:.2%} versus {POLICY_BASELINE_OPENER_ACCURACY:.2%} on "
        f"{POLICY_GRADED_GAMES:,} graded games (+{POLICY_EFFECT_ACCURACY_POINTS:.3f} "
        f"accuracy points, {glossary_abbr('P+')} {POLICY_PROBABILITY_POSITIVE:.2f}). It "
        "remains unresolved and paired prospective tracking continues.</p>"
        "<p><b>The sign rule -- the original grading protocol:</b> pick whichever team the "
        "model's single point forecast favors, a slightly different question (the "
        "prediction's midpoint rather than its full probability) that was never used to "
        f"choose a real pick. Graded the same way on the same games, it {protocol_words}. "
        "Both are reported on purpose, every time -- see docs/opener_evaluation.md for why "
        "they can differ.</p>"
        "</div>"
    )
    return _spaced(viz.card(inner))


def _story_sections(played_chain_accuracy: float | None) -> str:
    """The one canonical telling of the edge story (2026-08-24 re-architecture).

    ``track_record.html`` is the ONLY page that carries the full ladder of
    numbers; every other page links here instead of re-quoting them. Each
    canonical figure appears in exactly one section, at the moment it is
    explained. Every figure comes from the pinned constants or the same
    fail-open loader the picks page's crowned stat uses -- never a literal.
    """

    chain_text = f"{played_chain_accuracy:.1%}" if played_chain_accuracy is not None else None
    lift_sentence = (
        f"On the same games, adding the fix-up rules &#8212; fade newly-coached teams, "
        f"adjust for arrested players, fade revenge-game and trap-line spots &#8212; "
        f"lifted it to <b>{chain_text}</b>. That is the measured history."
        if chain_text
        else "Adding the fix-up rules &#8212; fade newly-coached teams, adjust for "
        "arrested players, fade revenge-game and trap-line spots &#8212; lifts it "
        "further; the measured chain figure appears here once its evaluation "
        "artifact is present."
    )

    def _section(kicker: str, title: str, prose: str) -> str:
        return _section_header(kicker, title, "") + f'<div class="prose">{prose}</div>'

    return (
        _section(
            "THE PROJECT",
            "What this is",
            "Every week this project picks a side for every NFL game against the point "
            "spread, before kickoff. It is one person's research project: a model makes "
            "a pick, a few hand-built rules overrule it when they fire, and every claim "
            "on this page is graded against what actually happened.",
        )
        + _section(
            "THE MODEL",
            "The model",
            "The model does not predict football from scratch. It starts from the "
            "market's own spread &#8212; the sharpest number in sports &#8212; and "
            "predicts by how much the actual game will differ from that line. Its "
            "inputs are team strength, weighted toward recent weeks, and who is "
            "actually available to play: practices missed, players ruled out. The "
            "research log calls this &#8220;weak_stack / market_residual&#8221;. To a "
            "human: a beats-the-line model that respects injuries.",
        )
        + _section(
            "MEASURED",
            "What it has done",
            f"Asked to pick winners against Tuesday's opening lines for the last six "
            f"seasons, the bare model was right <b>{HEADLINE.opener}</b> of the time. "
            f"A coin flip is 50%. {lift_sentence}",
        )
        + _section(
            "PLANNING ESTIMATE",
            "What to expect going forward",
            f"There is a catch, and it matters: those fix-up rules were chosen by "
            f"looking at the same history they are graded on. Any rule picked as best "
            f"on a dataset flatters that dataset. Discounted for that, the honest "
            f"expectation for the full card is <b>{PLAYED_CARD_EXPECTATION_HERO}</b> "
            f"against Tuesday lines. Not a promise &#8212; a planning number, and the "
            f"2026 season is graded against it in real time.",
        )
        + '<details class="ceiling-ladder" style="margin:12px 0 0;"><summary>The '
        "selection discount, in numbers</summary>"
        + '<div class="prose">'
        + "".join(f"<p>{rung}</p>" for rung in ladder_rungs(played_chain_accuracy))
        + "</div></details>"
        + _section(
            "TWO LINES, ONE RECORD",
            "The two lines",
            f"Every pick is graded against two lines. The Tuesday opener is what we "
            f"actually pick against &#8212; being early is the whole skill. The closing "
            f"line is the market's final word after a week of injury news and money, "
            f"and it is the hardest test in sports: <b>{HEADLINE.close}</b> against "
            f"the close is the same body of work, measured the harsh way. Both are "
            f"published because a record that only beats soft lines is not an edge.",
        )
        + _section(
            "FALSIFIABILITY",
            "How we would know we are wrong",
            "Starting with the Week 1 lock on September 8, 2026, every pick is written "
            "to a tamper-evident ledger before kickoff and settled after, at both "
            "lines. If the season lands at or below 50%, the honest conclusion is that "
            "the edge is not real. No re-tuning, no excuses &#8212; the ledger is the "
            "referee.",
        )
    )


def _long_run_record_section(active: Mapping[str, Any]) -> str:
    """Appendix section for the active model's own long-run evaluation record
    (formerly the fourth hero tile): same figures, story-page presentation."""

    historical = _mapping(dict(active), "historical_evaluation")
    model_accuracy = _number(historical.get("accuracy"))
    if model_accuracy is None:
        return ""
    model_games = _number(historical.get("games")) or 0.0
    model_correct = _number(historical.get("correct")) or 0.0
    season_range = _mapping(historical.get("intervals"), "season")
    range_lower = _number(season_range.get("lower"))
    range_upper = _number(season_range.get("upper"))
    range_sentence = (
        f" Its plausible range runs from {range_lower:.1%} to {range_upper:.1%} -- a "
        "single sample of seasons, not a promise about the next one."
        if range_lower is not None and range_upper is not None
        else " A single sample of seasons, not a promise about the next one."
    )
    return _section_header(
        "THE MODEL'S OWN LONG-RUN RECORD",
        "The same model, its own evaluation sample",
        "",
        top=40,
    ) + (
        '<div class="prose">'
        f"<p>Graded the same harsh way across its full evaluation sample, the "
        f"model's record is <b>{model_accuracy:.1%}</b> &#8212; "
        f"{int(model_correct):,} correct out of {int(model_games):,} games it was "
        f"tested on but never trained on.{range_sentence}</p>"
        "</div>"
    )


def _season_section(seasons: pd.DataFrame) -> str:
    if seasons.empty or not {"season", "opener_accuracy"}.issubset(seasons.columns):
        return ""

    # Same rule preference as the headline tiles (2026-08-19): grade by the
    # rule production actually plays when the artifact carries it.
    use_production = "opener_accuracy_probability_rule" in seasons.columns
    opener_column = "opener_accuracy_probability_rule" if use_production else "opener_accuracy"
    close_column = "close_accuracy_probability_rule" if use_production else "close_accuracy"

    season_rows: list[tuple[str, float]] = []
    season_cells: list[list[str]] = []
    games_by_season: dict[str, float] = {}
    for _, row in seasons.iterrows():
        label = str(row["season"]).split(".")[0]
        opener_value = _number(row.get(opener_column))
        if opener_value is None:
            continue
        close_value = _number(row.get(close_column))
        games_value = _number(row.get("games"))
        season_rows.append((label, opener_value))
        if games_value is not None:
            games_by_season[label] = games_value
        season_cells.append(
            [
                escape(label),
                f"{int(games_value):,}" if games_value is not None else "--",
                f"{opener_value:.1%}",
                f"{close_value:.1%}" if close_value is not None else "--",
            ]
        )
    if not season_rows:
        return ""

    # B5 fix (2026-08-19): ">=" previously counted a season EXACTLY at the
    # coin flip as "above" it -- the live page claimed "6 of the 6 seasons
    # finished above the coin flip" while its own 2020 bar read exactly
    # 50.0%. A dead-even season is its own category, distinct from both
    # "above" and "did not."
    above = sum(1 for _, value in season_rows if value > 0.5)
    even = [(label, value) for label, value in season_rows if value == 0.5]
    losing = [(label, value) for label, value in season_rows if value < 0.5]
    honesty = f"{above} of the {len(season_rows)} seasons finished above the coin flip"
    if even:
        even_listed = ", ".join(label for label, _ in even)
        honesty += f", {len(even)} landed exactly at it ({even_listed})"
    honesty += ". "
    if losing:
        listed = ", ".join(f"{label} at {value:.1%}" for label, value in losing)
        honesty += (
            f"{'One did not' if len(losing) == 1 else 'Some did not'}: {listed}. "
            "That is on the chart on purpose -- a page that hides its losing seasons is a "
            "sales pitch, not a track record. "
        )
    if any(label == "2020" for label, value in season_rows if value <= 0.5):
        thin = games_by_season.get("2020")
        thin_text = (
            f", and it is also the thinnest slice in the archive at {int(thin):,} games"
            if thin
            else ""
        )
        honesty += (
            "2020 was the COVID season: empty stadiums collapsed home-field advantage across "
            "the league, which a model built on earlier seasons systematically mispriced"
            f"{thin_text}."
        )

    inner = (
        '<p class="kicker">Season by season, against the frozen line</p>'
        '<p class="title" style="margin-bottom:12px;">No season is left off</p>'
        + viz.season_bars(season_rows)
        + f'<p class="sub" style="margin-top:12px;">{escape(honesty)}</p>'
        + '<details class="table-view"><summary>View as table</summary>'
        + _table(["Season", "Games", "Against the opener", "Against the close"], season_cells)
        + "</details>"
    )
    return _spaced(viz.card(inner))


def _honest_reading(opener_metadata: Mapping[str, Any]) -> str:
    pool_range = _season_interval(
        opener_metadata.get("uncertainty"), "opener_accuracy_probability_rule"
    ) or _season_interval(opener_metadata.get("uncertainty"), "opener_accuracy")
    range_sentence = (
        f"The pool grade's honest range runs from about {pool_range[0]:.1%} to "
        f"{pool_range[1]:.1%} across seasons."
        if pool_range
        else "Every headline here is the middle of a range, not a fixed value."
    )
    inner = (
        '<p class="kicker">How to read this honestly</p>'
        '<p class="title" style="margin-bottom:10px;">Four things that keep these numbers '
        "honest</p>"
        '<div class="prose">'
        "<p><b>Nothing here was picked after seeing the answer.</b> Every number is measured "
        "on games the model never trained on, with the recipe frozen before the games were "
        "scored.</p>"
        f"<p><b>A single number is the middle of a range.</b> {escape(range_sentence)} The "
        "headline is the best single guess; the range is where the true skill could plausibly "
        "sit.</p>"
        "<p><b>One look, once.</b> Each headline came from a single planned measurement of a "
        "frozen model. Re-running a test until it finally looks good is the fastest way to "
        "fool yourself, so we do not do it.</p>"
        "<p><b>52-53% against a frozen line is genuinely good.</b> Realistically excellent "
        f"is {HEADLINE.ceiling}. Someone who knew everything knowable before kickoff would "
        f"top out near {PREMEASUREMENT_GUESS_BAND}% against a frozen Tuesday line, because "
        "football itself is noisy -- final "
        "margins scatter about 13.5 points around even a perfect expectation. If this page "
        f"ever shows {CEILING_BUG_MARK_PCT}%, that is a bug to hunt, not a breakthrough.</p>"
        "</div>"
        '<p class="fine" style="margin-top:10px;">The ceiling arithmetic behind that last '
        "paragraph is written up in docs/pool_edge_plan.md.</p>"
    )
    return _spaced(viz.card(inner))


def _humanize(token: str) -> str:
    return token.replace("_", " ").replace("|", " -- ").replace("=", " ")


#: Reader-facing names for every registered challenger id, one per line of the
#: picks page's challenger-watch panel and the ledger mini table. Raw registry
#: ids are operator detail (snake_case config spellings like
#: ``nflcom_friday_refresh_out2_starters_v1``); a reader needs the rule, not
#: the slug. The canonical map lives in findings_content (shared with the
#: model ledger so both surfaces agree); a challenger id missing from it
#: still renders correctly through the :func:`_humanize` fallback, so a
#: brand-new registration never blocks a build -- the map-covers-registry
#: test in ``tests/test_public_board`` is what reminds us to add the name.
_CHALLENGER_DISPLAY_NAMES = CHALLENGER_DISPLAY_NAMES


def _challenger_display_name(challenger_id: str) -> str:
    return _CHALLENGER_DISPLAY_NAMES.get(challenger_id, _humanize(challenger_id))


#: One hand-written sentence per known challenger id: "what it does", never
#: "how it's configured" -- the config/fingerprint/command fields on each
#: registry entry stay off the public page (they are for the CLI operator,
#: not a reader). A challenger not in this dict (a brand-new registration)
#: still renders correctly with a generic fallback -- see
#: :func:`_challenger_blurb` -- so registering a new challenger never
#: requires touching this file.
_CHALLENGER_BLURBS: dict[str, str] = {
    "mod07_weak_signal_stack": (
        "Tracks the active model's own weak-signal stack as its own separate "
        "prospective arm, so the 2026 season scores it cleanly outside the "
        "already-spent historical research windows."
    ),
    "hc_year_one_fade_overlay": (
        "Fades first-year head coaches on the road, weeks 1-8: when the model's own pick "
        "sides with a rookie coach's team against an opponent that kept its coach, this "
        "flips the pick to the other side. It is both a separately tracked attribution "
        "arm and one member of the published four-overlay policy."
    ),
    "best_pick_nomination_v2": (
        "Chooses which single game gets the week's bonus Best Pick using calibrated win "
        "probability among the games the model and market agree on most, instead of the "
        "old rule (how much the edge survives a moving line)."
    ),
    "injury_value_lost_tilt_overlay": (
        "Nudges the pick toward whichever team lost less value to injury, using a "
        "parameter-free read of the injury report."
    ),
    "division_revenge_tilt_overlay": (
        "Nudges the pick toward a team that lost to this same opponent the last time "
        "they played -- a 'revenge game' tilt. It is also one member of the published "
        "four-overlay policy."
    ),
    "backup_qb_fade_overlay": (
        "Fades a team starting a backup quarterback against an opponent starting its usual starter."
    ),
    "surface_switch_tilt_overlay": (
        "Nudges the pick toward the home team when a visiting team that normally plays "
        "on grass switches onto turf."
    ),
    "spread_gap_zone_fade_overlay": (
        "Flips every pick where the market's spread sits between 7.5 and 10 points, "
        "regardless of which side the model liked -- a zone where the favorite has "
        "historically been overbought. It is also one member of the published "
        "four-overlay policy."
    ),
    "overlay_production_chain_coach_arrest_incumbent": (
        "Tracks the exact former production policy -- coach fade followed by the arrest "
        "policy -- against the newly played four-member card on the same fresh games."
    ),
    "interim_hc_first_game_tilt_overlay": (
        "Nudges the pick toward a team playing its first game under a newly appointed "
        "interim head coach -- teams have historically covered that specific first game, "
        "even though the effect fades away for every game after it."
    ),
    "forecast_weather_kn_warm_team_cold_late_tilt": (
        "Nudges the pick toward the home team when a warm-winter-metro visitor plays "
        "outdoors, late in the season, in a live forecast at or below 35F -- the "
        "strongest, best-powered read in this project's whole forecast-weather family."
    ),
    "forecast_weather_kn_precip_high_total_tilt": (
        "Nudges the pick toward the home team in an outdoor game with a high live "
        "forecast rain/snow probability and a high market total -- a newer, less-tested "
        "read that shares its live weather fetch with the warm-team-cold-late tilt above."
    ),
    "player_qb_continuity|ridge_alpha=1|calibration=none": (
        "A different regularization strength for the QB-continuity feature. Its "
        "measured improvement did not survive a predeclared replication on held-out "
        "seasons, so it stays off the card."
    ),
    "injury_signal_refresh_tilt": (
        "At each late-week refresh pass, flips the model's own pick when a fresh "
        "asymmetric injury report or news signal turns against it -- testing whether "
        "acting on injury news itself beats waiting for the market's line to confirm it."
    ),
}


def _challenger_blurb(challenger_id: str) -> str:
    return _CHALLENGER_BLURBS.get(
        challenger_id,
        "A prospective challenger tracked alongside the active model; see its record below.",
    )


_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _first_sentence(text: str, *, max_len: int = 260) -> str:
    """The first sentence of ``text``, or a hard truncation if none is found
    within ``max_len`` -- used to give a plain-English lead line for the
    (often paragraph-length) registry ``status_reason``/``status_reason_update``
    prose, with the full text always still available underneath in a
    ``<details>`` (see :func:`_challenger_card`). Never invents or drops
    words mid-sentence: a truncation always ends in an ellipsis so the reader
    knows more was cut."""

    collapsed = " ".join(text.split())
    match = _SENTENCE_END.search(collapsed)
    if match and match.end() <= max_len:
        return collapsed[: match.end()]
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[:max_len].rstrip() + "..."


#: Evidence keys this file knows are caveats/disclosures worth flagging as
#: their own chip, rather than only readable inside the (long, essay-length)
#: ``status_reason`` prose. Generic by suffix, not a per-challenger hardcoded
#: list, so a new challenger's own caveat fields surface automatically the
#: day it is registered -- see :func:`_evidence_caveat_chips`.
_CAVEAT_KEY_SUFFIXES = ("_caveat", "_disclosure")
_CAVEAT_KEY_EXACT = ("caveats",)


def _evidence_caveat_chips(evidence: Mapping[str, Any]) -> list[str]:
    """Short, honest chip labels for every caveat/disclosure field the
    registry entry's own ``evidence`` block carries (e.g.
    ``tuesday_visibility_caveat``, ``era_caveat``, ``double_counting_caveat``).
    The chip is only the field's OWN key, humanized -- never a summary or
    paraphrase of its (often long) text -- so this can never misstate a
    caveat the way a hand-written summary could; the full text stays
    reachable from ``write_up`` / the challenger's own doc, exactly as
    before this function existed."""

    chips = []
    for key, value in evidence.items():
        if not isinstance(value, str) or not value.strip():
            continue
        lowered = key.lower()
        if lowered in _CAVEAT_KEY_EXACT or any(
            lowered.endswith(suffix) for suffix in _CAVEAT_KEY_SUFFIXES
        ):
            chips.append(_humanize(key))
    return chips


def _opener_close_divergence_chip(evidence: Mapping[str, Any]) -> str | None:
    """Detects, from the entry's OWN evidence block, whether it was graded at
    both the opener and the close (several overlay challengers carry both --
    e.g. ``opener_graded``/``close_graded``, ``mined_opener``/``mined_close``,
    ``nfl_opener_grade_week_blocked``/``nfl_close_grade_week_blocked``), and
    if both sub-blocks carry their own ``probability_positive``, whether the
    two readings land on opposite sides of a coin flip. Purely a computation
    over data already read from the challenger's own JSON -- never a new
    number, never an invented divergence."""

    opener_blocks = [
        value
        for key, value in evidence.items()
        if isinstance(value, dict) and "opener" in key.lower()
    ]
    close_blocks = [
        value
        for key, value in evidence.items()
        if isinstance(value, dict) and "close" in key.lower()
    ]
    if not opener_blocks or not close_blocks:
        return None

    def _first_probability(blocks: list[dict[str, Any]]) -> float | None:
        for block in blocks:
            probability = _number(block.get("probability_positive"))
            if probability is not None:
                return probability
        return None

    opener_p, close_p = _first_probability(opener_blocks), _first_probability(close_blocks)
    if opener_p is not None and close_p is not None and (opener_p - 0.5) * (close_p - 0.5) < 0:
        return "opener/close disagree in sign"
    return "graded at both opener and close"


def _challenger_card(
    entry: Mapping[str, Any],
    *,
    week_preview: str,
    prospective_record_text: str,
) -> str:
    """One challenger, one card: what it does, what it did to this week's
    card (if anything), and its 2026 prospective record.

    Only reader-facing fields ever reach this card: ``challenger_id``,
    ``status``, the pre-registration ``evidence`` block's
    ``classification``/``probability_positive`` (already public elsewhere on
    this page as a weak-signal lead), its own caveat/disclosure field NAMES
    (never their full prose), and -- for a non-active status -- the
    deactivation reason. Config fingerprints, CLI recording commands, and
    feature-table paths -- all present on the raw registry entry -- are
    operator detail and never rendered here.

    A challenger whose status is anything other than ``ACTIVE_PROSPECTIVE``
    (closed, deactivated, or superseded) renders visually dimmed and carries
    a "why it is not live" block sourced from the registry's own
    ``status_reason_update`` (a later correction, when one was recorded) or
    ``status_reason`` (the original rationale) -- never a paraphrase.
    """

    challenger_id = str(entry.get("challenger_id", "unknown"))
    label = _challenger_display_name(challenger_id)
    status = str(entry.get("status", "unknown"))
    status_words = _humanize(status).lower()
    is_active = status == "ACTIVE_PROSPECTIVE"

    evidence = entry.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    classification = evidence.get("classification") or evidence.get("registry_verdict")
    probability = evidence.get("probability_positive")

    status_chip = (
        viz.status_line("good", status_words)
        if is_active
        else f'<span class="chip">{escape(status_words)}</span>'
    )
    evidence_chips = []
    if classification:
        evidence_chips.append(f'<span class="chip">{escape(_humanize(str(classification)))}</span>')
    if isinstance(probability, int | float):
        evidence_chips.append(f'<span class="chip">P+ {viz.p_plus_text(float(probability))}</span>')
    divergence_chip = _opener_close_divergence_chip(evidence)
    if divergence_chip:
        evidence_chips.append(f'<span class="chip">{escape(divergence_chip)}</span>')
    # Caveats live on the ``evidence`` block for most challengers, but a few
    # (e.g. ``forecast_cold_visitor_tilt``'s ``climatology_deviation_disclosure``/
    # ``station_mapping_deviation_disclosure``) are registered as SIBLINGS of
    # ``evidence`` on the entry itself -- scan both, deduplicating by label,
    # so neither location is silently missed.
    caveat_labels = list(
        dict.fromkeys(_evidence_caveat_chips(entry) + _evidence_caveat_chips(evidence))
    )
    for caveat_label in caveat_labels:
        evidence_chips.append(f'<span class="chip">{escape(caveat_label)}</span>')

    blurb_text = escape(_challenger_blurb(challenger_id))
    parts = [
        '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;'
        'align-items:baseline;margin-bottom:6px;">'
        f'<p class="title" style="font-size:17px;">{escape(label)}</p>{status_chip}</div>',
        f'<p class="prose" style="margin-bottom:8px;">{blurb_text}</p>',
    ]
    if evidence_chips:
        parts.append(
            '<div class="row" style="gap:6px;flex-wrap:wrap;margin-bottom:6px;">'
            f"{''.join(evidence_chips)}</div>"
        )
    if is_active and week_preview:
        parts.append(
            '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--grid);">'
            '<p class="kicker">This week</p>'
            f'<p class="sub">{escape(week_preview)}</p></div>'
        )
    if not is_active:
        reason_text = str(entry.get("status_reason_update") or entry.get("status_reason") or "")
        reason_lead = _first_sentence(reason_text) if reason_text else "No reason recorded."
        details = (
            '<details class="table-view" style="margin-top:6px;">'
            "<summary>Full reason</summary>"
            f'<p class="fine" style="margin-top:6px;max-width:68ch;">{escape(reason_text)}</p>'
            "</details>"
            if reason_text and reason_text != reason_lead
            else ""
        )
        parts.append(
            '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--grid);">'
            '<p class="kicker">Why it is not live</p>'
            f'<p class="sub">{escape(reason_lead)}</p>{details}</div>'
        )
    parts.append(
        '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--grid);">'
        '<p class="kicker">2026 prospective record</p>'
        f'<p class="sub">{escape(prospective_record_text)}</p></div>'
    )
    card_html = viz.card("".join(parts))
    if not is_active:
        # Greyed, not hidden (icon/label/text above already carry the status --
        # this is a supplemental visual cue, never the only signal).
        card_html = f'<div style="opacity:0.6;">{card_html}</div>'
    return card_html


#: Shown for every challenger until the ledger has at least one settled
#: game -- true for the whole roster before the season starts (see
#: :func:`_challenger_prospective_records`).
_PENDING_PROSPECTIVE_RECORD = (
    "Not scored yet this season -- this fills in automatically once games are played "
    "and picks are recorded and settled each week."
)


def _challengers_section(
    challengers: Sequence[Mapping[str, Any]],
    *,
    week_previews: Mapping[str, str] | None = None,
    prospective_records: Mapping[str, str] | None = None,
) -> str:
    """D3(a): the registered 2026 prospective challengers, read fresh from
    ``artifacts/prospective/challengers.json`` at generation time -- never
    hardcoded, since another agent registers new ones concurrently.

    ``week_previews``/``prospective_records`` are optional
    ``{challenger_id: sentence}`` maps computed once in
    :func:`build_public_site` (see :func:`_challenger_week_previews` and
    :func:`_challenger_prospective_records`) and shared between this page
    and ``track_record.html``'s own D3(a) section. Omitting either (every
    direct caller/test that does not pass them) simply renders each card
    without a "this week" line and with the generic "not scored yet" record
    text -- the same graceful-degradation contract every other optional
    artifact on this site already follows.
    """

    if not challengers:
        return ""
    week_previews = week_previews or {}
    prospective_records = prospective_records or {}
    cards = _rows(
        [
            _challenger_card(
                entry,
                week_preview=week_previews.get(str(entry.get("challenger_id")), ""),
                prospective_record_text=prospective_records.get(
                    str(entry.get("challenger_id")), _PENDING_PROSPECTIVE_RECORD
                ),
            )
            for entry in challengers
        ]
    )
    intro = viz.card(
        '<p class="kicker">The live test starts Sep 8, 2026</p>'
        '<p class="title" style="margin-bottom:8px;">What else is being tracked '
        "alongside the active model</p>"
        '<div class="prose"><p>Every challenger below rides on the SAME published card '
        "-- none of them spends a research window or changes what gets played. Each "
        "one's forced-pick accuracy is scored against the recorded line (the opener, "
        "primary) and again against the close (secondary), paired game-for-game with "
        "the active model's own paper ledger, the same way the main track record "
        "above is graded.</p></div>"
    )
    return _spaced(intro + cards)


# ---------------------------------------------------------------------------
# "This week" previews for challenger cards -- what each ACTIVE_PROSPECTIVE
# challenger actually did (the one applied overlay) or WOULD have done (every
# other pick-level tilt, all dual-tracked only) to this week's un-overlaid
# card. Every path below degrades to omitting the challenger from the
# returned mapping rather than raising -- a missing local snapshot or
# feature table just means that card renders with no "this week" line,
# exactly like ``card_view.resolve_overlay`` degrades to a no-op.
# ---------------------------------------------------------------------------

_NOT_APPLIED_NOTE = "Prospective evidence only -- not applied to the published card."


def _tilt_preview_sentence(result: Any, detail_fn: Any, *, applied_to_real_card: bool) -> str:
    """A "what happened to this week's card" sentence from any of the
    tilt/fade overlay modules' result objects -- ``coach_fade_overlay``,
    ``backup_qb_fade_overlay``, ``division_revenge_tilt_overlay``,
    ``injury_value_tilt_overlay``, ``surface_switch_tilt_overlay``,
    ``spread_gap_zone_fade_overlay``, and ``interim_hc_first_game_tilt_overlay``
    all share the same
    ``enabled``/``flip_count``/``flips`` shape by design (each module's own
    docstring says so), so one function renders all of them; ``detail_fn``
    adapts each module's differently-named flip fields to a common
    ``(matchup, from_team, to_team)`` tuple.
    """

    if not result.enabled:
        return "Not eligible this week under its own rule."
    if result.flip_count == 0:
        base = "No games matched its rule this week, so nothing would change."
        return base if applied_to_real_card else f"{base} {_NOT_APPLIED_NOTE}"
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{matchup} ({frm} to {to})"
        for matchup, frm, to in (detail_fn(flip) for flip in result.flips)
    )
    if applied_to_real_card:
        return (
            f"Flipped {result.flip_count} pick{plural} on the published card this week: {detail}."
        )
    return (
        f"Would flip {result.flip_count} pick{plural} on this week's card if it were live: "
        f"{detail}. {_NOT_APPLIED_NOTE}"
    )


def _flip_backup_qb(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.backup_team, flip.opponent_team


def _flip_division_revenge(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.opponent_team, flip.revenge_team


def _flip_injury_value(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.hurt_team, flip.healthier_team


def _flip_surface_switch(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.grass_modal_visitor, flip.turf_venue_home


def _flip_spread_gap_zone(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.original_pick_team, flip.flipped_to_team


def _flip_coach_fade(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.year_one_team, flip.opponent_team


def _flip_interim_hc_first_game(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.opponent_team, flip.interim_team


def _real_overlay_preview_sentence(overlay: OverlayResult) -> str:
    """``hc_year_one_fade_overlay`` is the one challenger actually applied to
    the published card -- reuse the SAME ``OverlayResult`` ``build_public_site``
    already computed for the picks page rather than recomputing it, so the
    two pages can never disagree about what the real card did this week."""

    return _tilt_preview_sentence(overlay, _flip_coach_fade, applied_to_real_card=True)


def _team_for_game(predictions: pd.DataFrame, game_id: str | None) -> str | None:
    if game_id is None or predictions.empty or "game_id" not in predictions.columns:
        return None
    row = predictions.loc[predictions["game_id"].astype(str).eq(str(game_id))]
    if row.empty:
        return None
    team, _ = pick_side(row.iloc[0])
    return team


def _best_pick_preview_sentence(
    nomination: BestPickNomination | None, predictions: pd.DataFrame
) -> str:
    if nomination is None or nomination.v2_result is None:
        return (
            "Could not be computed this week (not enough walk-forward training history "
            "yet, or no market snapshot available)."
        )
    v2_team = _team_for_game(predictions, nomination.v2_result.game_id)
    if nomination.active_rule == "v2":
        team_text = v2_team if v2_team else "this week's nominated game"
        return f"This IS the rule actually used this week: it nominates {team_text} for Best Pick."
    v1_team = _team_for_game(predictions, nomination.v1_game_id)
    if v2_team and v1_team and v2_team == v1_team:
        return f"This week it agrees with the rule now in use: both nominate {v2_team}."
    if v2_team:
        return (
            f"This week it would nominate {v2_team}, but the rule actually played "
            f"nominates {v1_team or 'a different game'} instead."
        )
    return "No nomination this week (playoff week, or no line-sweep artifact yet)."


def _best_pick_v3_preview_sentence(
    nomination: BestPickNomination | None,
    predictions: pd.DataFrame,
    metadata: Mapping[str, Any],
    data_root: Path | None,
) -> str:
    """v3's own weekly nominee, computed locally with the SAME inputs v2
    already resolves this week (:func:`nfl_ats.card_view.v2_nomination_inputs`
    -- no live fetch, just the local market-snapshot store and feature
    table both v2 and v3 already read), compared against whichever
    nomination is ACTUALLY played (``nomination.active_game_id``). v3 is a
    side-ledger-only challenger (never wired into the played card -- see its
    registration in ``artifacts/prospective/challengers.json``), so this
    never affects, and is never affected by, which nomination is published.
    """

    inputs = v2_nomination_inputs(metadata, data_root)
    if inputs is None:
        return (
            "Could not be computed this week (not enough walk-forward training history "
            "yet, or no market snapshot available)."
        )
    try:
        features = pd.read_parquet(inputs.feature_table)
    except (OSError, ValueError):
        return "Could not be computed this week (its feature table is not available locally)."
    try:
        result = nominate_v3(
            predictions,
            features,
            market_root=Path(inputs.market_root),
            season=inputs.season,
            week=inputs.week,
            regressor=inputs.regressor,
            feature_profile=inputs.feature_profile,
            min_train_games=inputs.min_train_games,
        )
    except (ValueError, DataContractError):
        return (
            "Could not be computed this week (not enough walk-forward training history "
            "yet, or no market snapshot available)."
        )
    if result is None:
        return "No nomination this week (playoff week, or no line-sweep artifact yet)."
    v3_team = _team_for_game(predictions, result.game_id)
    active_team = (
        _team_for_game(predictions, nomination.active_game_id)
        if nomination is not None and nomination.active_game_id is not None
        else None
    )
    if v3_team and active_team and v3_team == active_team:
        return f"This week it agrees with the nomination actually played: both nominate {v3_team}."
    if v3_team:
        return (
            f"This week it would nominate {v3_team}, differing from the nomination actually "
            f"played, {active_team or 'a different game'}. {_NOT_APPLIED_NOTE}"
        )
    return "No nomination this week (playoff week, or no line-sweep artifact yet)."


#: Challengers this file deliberately does NOT attempt to preview locally --
#: each for a distinct, honest, stated reason (never a bare omission): a
#: weekly model REFIT is too heavy to pay twice per site build
#: (``ecdf_mapping_incumbent``, ``era_weighted_half_life_8``), a genuine LIVE
#: network fetch this static build must not make (``forecast_cold_visitor_tilt``
#: -- the first challenger in the registry to need one -- and its
#: kickoff-nearest siblings ``forecast_weather_kn_warm_team_cold_late_tilt`` /
#: ``forecast_weather_kn_precip_high_total_tilt``, see each one's own
#: ``challengers.json`` registration), or a mechanism that structurally has
#: nothing to preview before a later refresh pass runs
#: (``model_only_refresh_incumbent``). Per the task spec: when a challenger's
#: preview needs more than this page can honestly compute at build time, say
#: so plainly ("evaluated at lock time") rather than guess or omit silently.
_LOCK_TIME_EVALUATED_NOTES: dict[str, str] = {
    "ecdf_mapping_incumbent": (
        "Evaluated at lock time -- this challenger remaps every game's probability from a "
        "fresh weekly model refit, too heavy to recompute for this page's preview. Its 2026 "
        "prospective record (below) fills in once games are recorded and settled."
    ),
    "era_weighted_half_life_8": (
        "Evaluated at lock time -- this challenger refits the model with different "
        "season-weighting every week, too heavy to recompute for this page's preview. Its "
        "2026 prospective record (below) fills in once games are recorded and settled."
    ),
    "forecast_cold_visitor_tilt": (
        "Evaluated at lock time -- this tilt reads a LIVE weather forecast fetched at "
        "recording time, which this page cannot fetch during a static-site build. Its 2026 "
        "prospective record (below) fills in once games are recorded and settled."
    ),
    "forecast_weather_kn_warm_team_cold_late_tilt": (
        "Evaluated at lock time -- this tilt reads a LIVE kickoff-nearest weather forecast "
        "fetched at recording time, which this page cannot fetch during a static-site "
        "build. Its 2026 prospective record (below) fills in once games are recorded and "
        "settled."
    ),
    "forecast_weather_kn_precip_high_total_tilt": (
        "Evaluated at lock time -- this tilt reads the SAME live kickoff-nearest weather "
        "forecast fetched at recording time as the warm-team-cold-late tilt above, which "
        "this page cannot fetch during a static-site build. Its 2026 prospective record "
        "(below) fills in once games are recorded and settled."
    ),
    "model_only_refresh_incumbent": (
        "Evaluated at lock time -- this arm only diverges from the model's own pick when a "
        "later Thursday/Saturday/Sunday refresh pass sees the market move at least 1 point "
        "off the frozen Tuesday line, so there is nothing to preview on the Tuesday build."
    ),
    "injury_signal_refresh_tilt": (
        "Evaluated at refresh passes -- this challenger reads post-Tuesday injury filings "
        "(official Wednesday-Friday reports, or a news-headline fallback), which are by "
        "construction empty at Tuesday noon, so it has nothing to preview on the Tuesday "
        "build. Its first real reading, and its 2026 prospective record (below), fill in "
        "once a Thursday/Saturday/Sunday `nfl-ats refresh-picks` pass runs during the week."
    ),
    "overlay_production_chain_coach_arrest_incumbent": (
        "Recorded at lock time from the same immutable paper-decision row as the played "
        "four-member card, so the former-policy comparison cannot drift between source reads."
    ),
}


def _load_schedules_for_challenger_preview(data_root: Path) -> pd.DataFrame | None:
    """Mirrors ``card_view.resolve_overlay``'s own schedule load exactly, so
    a missing local snapshot degrades a challenger preview the same way it
    degrades the real overlay -- to nothing, never an error."""

    try:
        schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    except FileNotFoundError:
        return None
    return schedules


def _load_injury_features_for_challenger_preview(data_root: Path) -> pd.DataFrame | None:
    path = data_root / "processed" / PLAYER_FEATURE_TABLE_NAME
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path, columns=["game_id", *VALUE_LOST_DIFF_COLUMNS])
    except (OSError, ValueError, KeyError):
        return None


#: (challenger_id, apply function, flip-detail adapter) for the three
#: hypothetical tilts that need only the newest local schedule snapshot.
_SCHEDULE_BASED_TILT_PREVIEWS: tuple[tuple[str, Any, Any], ...] = (
    ("backup_qb_fade_overlay", apply_backup_qb_fade_overlay, _flip_backup_qb),
    ("division_revenge_tilt_overlay", apply_division_revenge_tilt_overlay, _flip_division_revenge),
    ("surface_switch_tilt_overlay", apply_surface_switch_tilt_overlay, _flip_surface_switch),
)


def _challenger_week_previews(
    challengers: Sequence[Mapping[str, Any]],
    predictions: pd.DataFrame,
    data_root: Path,
    *,
    overlay: OverlayResult,
    nomination: BestPickNomination | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """This week's plain-English "what happened to the card" sentence, keyed
    by challenger id, for every ACTIVE_PROSPECTIVE challenger this module
    knows how to preview. A challenger id this dispatcher does not recognize
    (a brand-new registration) is simply absent from the returned mapping --
    its card just renders with no "this week" line until this file is
    updated, matching every other optional-artifact degradation on this
    site.

    Every tilt is evaluated against the active model's own
    UN-overlaid ``predictions`` -- the exact same base card each tilt's own
    ``record_*_challenger_decisions`` function reads from
    ``recommendations.csv`` -- never against ``overlay.overlaid_predictions``,
    so one challenger's hypothetical never sees another's flip.
    """

    active_ids = {
        str(entry.get("challenger_id"))
        for entry in challengers
        if str(entry.get("status")) == "ACTIVE_PROSPECTIVE"
    }
    previews: dict[str, str] = {}
    if predictions.empty:
        return previews

    if "hc_year_one_fade_overlay" in active_ids:
        previews["hc_year_one_fade_overlay"] = _real_overlay_preview_sentence(overlay)
    if "best_pick_nomination_v2" in active_ids:
        previews["best_pick_nomination_v2"] = _best_pick_preview_sentence(nomination, predictions)
    if "best_pick_nomination_v3" in active_ids:
        previews["best_pick_nomination_v3"] = _best_pick_v3_preview_sentence(
            nomination, predictions, metadata or {}, data_root
        )
    if "mod07_weak_signal_stack" in active_ids:
        previews["mod07_weak_signal_stack"] = (
            "This challenger IS the active model's own configuration, so it makes the "
            "exact same picks -- there is nothing separate for it to flip."
        )

    schedules = _load_schedules_for_challenger_preview(data_root)
    if schedules is not None:
        for challenger_id, apply_fn, detail_fn in _SCHEDULE_BASED_TILT_PREVIEWS:
            if challenger_id not in active_ids:
                continue
            try:
                result = apply_fn(predictions, schedules)
            except DataContractError:
                continue
            previews[challenger_id] = _tilt_preview_sentence(
                result,
                detail_fn,
                applied_to_real_card=challenger_id == "division_revenge_tilt_overlay",
            )

    if "spread_gap_zone_fade_overlay" in active_ids:
        try:
            result = apply_spread_gap_zone_fade_overlay(predictions)
        except DataContractError:
            pass
        else:
            previews["spread_gap_zone_fade_overlay"] = _tilt_preview_sentence(
                result, _flip_spread_gap_zone, applied_to_real_card=True
            )

    if "injury_value_lost_tilt_overlay" in active_ids:
        features = _load_injury_features_for_challenger_preview(data_root)
        if features is not None:
            try:
                result = apply_injury_value_tilt_overlay(predictions, features)
            except DataContractError:
                pass
            else:
                previews["injury_value_lost_tilt_overlay"] = _tilt_preview_sentence(
                    result, _flip_injury_value, applied_to_real_card=False
                )

    if "interim_hc_first_game_tilt_overlay" in active_ids:
        # Fail-open by design (nfl_ats.interim_hc_first_game_tilt_overlay's
        # own contract): a missing/unavailable interim-coach snapshot never
        # raises here, it just yields zero flags, which _tilt_preview_sentence
        # already renders as an honest "no games matched its rule this week"
        # sentence -- exactly the "no interim coaches this week" preview
        # Week 1 2026 needs, since mid-season firings cannot exist yet.
        try:
            result = apply_interim_hc_first_game_tilt_overlay(predictions, data_root.parent)
        except DataContractError:
            pass
        else:
            previews["interim_hc_first_game_tilt_overlay"] = _tilt_preview_sentence(
                result, _flip_interim_hc_first_game, applied_to_real_card=False
            )

    for challenger_id, note in _LOCK_TIME_EVALUATED_NOTES.items():
        if challenger_id in active_ids and challenger_id not in previews:
            previews[challenger_id] = note

    return previews


# ---------------------------------------------------------------------------
# "2026 prospective record" for challenger cards -- read fresh from the
# newest ``nfl-ats prospective-score`` run, exactly like the track-record
# tiles read the newest ``opener_evaluation`` run. The ledger starts EMPTY
# (the 2026 season has not kicked off yet), so the common case is the
# generic pending sentence -- this must never fail the build over that.
# ---------------------------------------------------------------------------


def _load_latest_prospective_scoring(artifacts_root: Path) -> dict[str, Mapping[str, Any]]:
    """The newest ``prospective-score`` run's per-entrant report, keyed by
    entrant name (``"active_model"`` or a ``challenger_id``)."""

    directories = artifact_directories(artifacts_root / "prospective_scoring", "metadata.json")
    for directory in directories:
        try:
            metadata = read_json(directory / "metadata.json")
        except (ValueError, OSError):
            continue
        entrants = metadata.get("entrants")
        if not isinstance(entrants, list):
            continue
        return {
            str(item["entrant"]): item
            for item in entrants
            if isinstance(item, dict) and item.get("entrant")
        }
    return {}


def _prospective_record_text(report: Mapping[str, Any] | None) -> str:
    if report is None:
        return _PENDING_PROSPECTIVE_RECORD
    forced = report.get("forced_picks")
    decision = forced.get("decision_line") if isinstance(forced, dict) else None
    if not isinstance(decision, dict):
        return _PENDING_PROSPECTIVE_RECORD
    games = _number(decision.get("games"))
    if not games:
        return _PENDING_PROSPECTIVE_RECORD
    accuracy = _number(decision.get("accuracy"))
    vs_coin_flip = _number(decision.get("vs_coin_flip"))
    accuracy_text = f"{accuracy:.1%}" if accuracy is not None else "--"
    delta_text = f" ({vs_coin_flip:+.1%} vs. a coin flip)" if vs_coin_flip is not None else ""
    return (
        f"{int(games)} games settled this season, {accuracy_text} against the recorded "
        f"line{delta_text}."
    )


def _challenger_prospective_records(
    artifacts_root: Path, challengers: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    reports = _load_latest_prospective_scoring(artifacts_root)
    return {
        str(entry.get("challenger_id")): _prospective_record_text(
            reports.get(str(entry.get("challenger_id")))
        )
        for entry in challengers
    }


def _best_pick_section(
    active_rule: str | None, best_pick_team: str | None, method_note: str
) -> str:
    """D3(b): the honest historical budget for the Best Pick lever, plus this
    week's actual nomination and which rule chose it.
    """

    budget = (
        "<p>The pool pays one Best Pick per regular-season week: the game where the "
        "model is most confident. The honest budget for that call is "
        "<b>about +0.9 points</b> &#8212; real, small, and stated small on purpose. "
        "Every alternative ranking rule measured against it did worse.</p>"
    )
    if best_pick_team and active_rule == "v2" and method_note:
        # v2's own method_note already names the rule in full sentence form
        # (NOMINATION_V2_METHOD_SENTENCE) -- restating "chosen by the v2 rule
        # (calibrated probability...)" alongside it would say the same thing
        # twice in one paragraph.
        this_week = (
            f"<p>This week's nomination: <b>{escape(best_pick_team)}</b>. {escape(method_note)}</p>"
        )
    elif best_pick_team:
        rule_words = (
            "the v2 rule (calibrated probability among low-disagreement games)"
            if active_rule == "v2"
            else "the standard rule (most robust line sweep)"
        )
        tie = f" {escape(method_note)}" if method_note else ""
        this_week = (
            f"<p>This week's nomination: <b>{escape(best_pick_team)}</b>, chosen by "
            f"{rule_words}.{tie}</p>"
        )
    else:
        this_week = (
            "<p>No Best Pick is nominated this week (playoff week, or no line-sweep "
            "artifact yet).</p>"
        )
    inner = (
        '<p class="kicker">Best Pick</p>'
        '<p class="title" style="margin-bottom:8px;">The honest budget, and this '
        "week's nomination</p>"
        f'<div class="prose">{budget}{this_week}</div>'
    )
    return _spaced(viz.card(inner))


def render_track_record_page(
    opener_metadata: Mapping[str, Any] | None = None,
    seasons: pd.DataFrame | None = None,
    active: Mapping[str, Any] | None = None,
    *,
    generated_at: datetime | None = None,
    challengers: Sequence[Mapping[str, Any]] | None = None,
    best_pick_rule: str | None = None,
    best_pick_team: str | None = None,
    best_pick_method_note: str = "",
    challenger_week_previews: Mapping[str, str] | None = None,
    challenger_prospective_records: Mapping[str, str] | None = None,
    played_chain_accuracy: float | None = None,
) -> str:
    """Render ``docs/track_record.html`` -- the ONE page that tells the edge
    story start to finish, then shows the tables behind it.

    2026-08-24 re-architecture: the former three-tile hero is replaced by
    :func:`_story_sections` -- six short sections that explain what the model
    is, what it has measured, what to expect forward, why two lines are
    graded, and how the claim could fail -- each canonical figure appearing
    exactly once, at the moment it is explained. The earlier sections remain
    as the appendix behind the story.

    Everything here is an AGGREGATE statistic (accuracy rates, per-season rates,
    their ranges), which is publishable; no raw market quote reaches this page.
    """

    opener_metadata = opener_metadata or {}
    seasons = seasons if seasons is not None else pd.DataFrame()
    active = active or {}
    challengers = challengers or ()
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)

    body = (
        viz.page_header(
            "Track record",
            "How good is this, honestly?",
            "One record, one story, told start to finish.",
        )
        + _story_sections(played_chain_accuracy)
        + _section_header(
            "APPENDIX",
            "The tables behind every number above",
            "Same data, graded in detail -- for checking the story, not replacing it.",
            top=44,
        )
        + _rule_explainer_section(opener_metadata)
        + _long_run_record_section(active)
        + _season_section(seasons)
        + _best_pick_section(best_pick_rule, best_pick_team, best_pick_method_note)
        + _challengers_section(
            challengers,
            week_previews=challenger_week_previews,
            prospective_records=challenger_prospective_records,
        )
        + _honest_reading(opener_metadata)
    )
    model_id = active.get("model_id")
    return _page(
        current=TRACK_RECORD_PAGE,
        body=body,
        generated=generated,
        footer_note=(
            f"model <code>{escape(str(model_id))}</code>" if model_id else "no active model linked"
        ),
    )


# ---------------------------------------------------------------------------
# Artifact loading: active model manifest -> weekly forecast -> predictions +
# line sweep, the latest market-decomposition attribution for explanations, and
# the latest opener evaluation for the track record. Optional artifacts are
# feature-detected: an older checkout without one still renders a site.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicBoardArtifacts:
    predictions: pd.DataFrame
    sweep: pd.DataFrame
    explanations: dict[str, str]
    metadata: dict[str, Any]
    active: dict[str, Any]


@dataclass(frozen=True)
class OpenerEvaluationArtifacts:
    """The latest opener-evaluation run: its metadata and per-season summary."""

    metadata: dict[str, Any]
    seasons: pd.DataFrame


def load_public_board_artifacts(artifacts_root: Path) -> PublicBoardArtifacts:
    """Load the synchronized weekly forecast and explanations for the picks page.

    Mirrors :func:`nfl_ats.publishing._publication_context`'s validation of the
    active-model manifest chain (active model must be synchronized, the linked
    weekly forecast must match its model id and carry ``SYNCHRONIZED`` status), so
    the public site can never render a forecast the model card itself would
    refuse to publish.
    """

    active = load_active_ats_model(artifacts_root)
    if active is None:
        raise ValueError("No synchronized active ATS model is available to publish")
    forecast_directory = active_artifact_path(artifacts_root, active, "weekly_forecast")
    if forecast_directory is None:
        raise ValueError("Active ATS model has no linked weekly forecast")
    metadata_path = forecast_directory / "metadata.json"
    recommendations_path = forecast_directory / "recommendations.csv"
    if not metadata_path.is_file() or not recommendations_path.is_file():
        raise ValueError("Linked weekly forecast is missing metadata or recommendations")
    metadata = read_json(metadata_path)
    if metadata.get("active_model_id") != active.get("model_id"):
        raise ValueError("Weekly forecast model ID does not match the active model")
    if metadata.get("synchronization_status") != "SYNCHRONIZED":
        raise ValueError("Weekly forecast is not synchronized with an evaluation")

    predictions = pd.read_csv(recommendations_path)
    method = str(active.get("method"))
    if "method" in predictions.columns and not predictions["method"].eq(method).all():
        raise ValueError("Weekly recommendations contain a method other than the active method")

    sweep = pd.DataFrame()
    sweep_path = forecast_directory / "line_sweep.parquet"
    if sweep_path.is_file():
        sweep = pd.read_parquet(sweep_path)
        if "method" in sweep.columns:
            sweep = sweep.loc[sweep["method"].eq(method)]

    explanations = _reconciled_explanations(artifacts_root, predictions)

    return PublicBoardArtifacts(predictions, sweep, explanations, metadata, active)


# B4 fix (2026-08-19): verified live on ATL at PIT -- the header said "We make
# this line 2.0 points different from the market" while the market-decomposition
# explanation paragraph below it said "The model essentially agrees with the
# market on this game (a 0.1-point gap)." Both sentences describe the SAME
# quantity (the model-vs-market residual), but from two different sources: the
# header always reads the CURRENT recommendations.csv, while the explanation
# text is baked into a market_decomposition run that can predate it (the
# dashboard's own state.py already flags this exact staleness for OTHER
# purposes via a feature-table sha256 comparison). A page cannot render two
# numbers for one concept and disagree with itself, so rather than trust an
# opaque, possibly-stale sentence, this compares the attribution artifact's
# OWN recorded ``predicted_residual`` against the live card's
# ``predicted_market_residual`` for the same game, and drops the explanation
# entirely when they diverge -- ``_game_card`` already has an honest fallback
# ("not been published for this card yet") for a missing explanation.
_EXPLANATION_RESIDUAL_TOLERANCE = 0.3


def _reconciled_explanations(artifacts_root: Path, predictions: pd.DataFrame) -> dict[str, str]:
    decomposition_directories = artifact_directories(
        artifacts_root / "market_decomposition", "attribution.parquet"
    )
    if not decomposition_directories:
        return {}
    attribution = pd.read_parquet(decomposition_directories[0] / "attribution.parquet")
    if not {"explanation", "game_id"}.issubset(attribution.columns):
        return {}

    live_residual: dict[str, float] = {}
    if "predicted_market_residual" in predictions.columns and "game_id" in predictions.columns:
        for _, row in predictions.iterrows():
            value = _number(row.get("predicted_market_residual"))
            if value is not None:
                live_residual[str(row["game_id"])] = value
    has_attrib_residual = "predicted_residual" in attribution.columns

    explanations: dict[str, str] = {}
    deduped = attribution.dropna(subset=["explanation"]).drop_duplicates("game_id")
    for _, row in deduped.iterrows():
        game_id = str(row["game_id"])
        if has_attrib_residual:
            attrib_value = _number(row.get("predicted_residual"))
            live_value = live_residual.get(game_id)
            if (
                attrib_value is not None
                and live_value is not None
                and abs(attrib_value - live_value) > _EXPLANATION_RESIDUAL_TOLERANCE
            ):
                continue  # stale/inconsistent attribution -- drop rather than contradict
        explanations[game_id] = str(row["explanation"])
    return explanations


def load_opener_evaluation_artifacts(
    artifacts_root: Path, active_feature_profile: str | None = None
) -> OpenerEvaluationArtifacts:
    """Load the newest opener-evaluation run FOR THE ACTIVE MODEL.

    ``active_feature_profile`` filters runs by their recorded
    ``active_model_config.feature_profile``. Passing ``None`` keeps the old
    newest-wins behaviour and is only for callers with no active model.

    Why the filter exists (2026-08-18): this function used to take
    ``directories[0]`` unconditionally. ``artifact_directories`` sorts by
    directory name descending, so ANY later comparison run silently overrode
    the tile. A ``player_value`` research run written eight minutes after the
    real ``weak_stack`` run put 52.4%/51.8% on the published track-record page
    while the active model's true figures were 52.83%/51.56% -- and the page
    still credited the active model by id. Publishing another model's grade as
    your own is the failure this guard exists to make impossible.
    """

    directories = artifact_directories(artifacts_root / "opener_evaluation", "metadata.json")
    for directory in directories:
        metadata = read_json(directory / "metadata.json")
        if active_feature_profile is not None:
            config = metadata.get("active_model_config") or {}
            if config.get("feature_profile") != active_feature_profile:
                continue
        seasons = pd.DataFrame()
        season_path = directory / "season_summary.csv"
        if season_path.is_file():
            seasons = pd.read_csv(season_path)
        return OpenerEvaluationArtifacts(metadata, seasons)
    return OpenerEvaluationArtifacts({}, pd.DataFrame())


def load_played_chain_accuracy(artifacts_root: Path) -> float | None:
    """The sequential played-chain accuracy -- raw model -> coach fade ->
    player-arrests policy -- from the newest
    ``overlay_subset_composition`` run's ``production_chain_reference``
    block (the same figure ``docs/movement_composition_eval.md`` quotes as
    arm (a), 54.16% on 1,503 paired opener-graded games).

    Feature-detected and fail-open like every other optional loader here: a
    missing directory or malformed payload returns ``None`` and the picks
    page degrades its crowned stat to the raw-model baseline rather than
    inventing a chain figure.
    """

    directories = artifact_directories(artifacts_root / "overlay_subset_composition", "result.json")
    for directory in directories:
        try:
            payload = read_json(directory / "result.json")
        except (ValueError, OSError):
            continue
        reference = payload.get("production_chain_reference")
        if not isinstance(reference, dict):
            continue
        sequential = reference.get("coach_then_arrest_sequential")
        if not isinstance(sequential, dict):
            continue
        return _number(sequential.get("candidate_accuracy"))
    return None


@dataclass(frozen=True)
class EraMagnitude:
    """One era slice of a ``era_trend_*`` signal's magnitude, from
    ``artifacts/era_magnitude_profile/<run>/results.json`` -- the structured
    artifact ``scripts/era_magnitude_profile.py`` writes, distinct from the
    unstructured prose the SAME finding also stuffs into the registry
    signal's own ``notes`` field (era-trend slope, changepoint, modulator
    regression -- not machine-parseable, and not what this reads).
    """

    era_label: str
    effect: float
    interval: tuple[float, float] | None
    probability_positive: float | None


def load_era_magnitude_profile(artifacts_root: Path) -> dict[str, list[EraMagnitude]]:
    """Per-era magnitude slices for every signal the profile covers, keyed by
    the profile's own short signal name (e.g. ``"hc_year_one_fade"`` -- NOT
    the registry's ``era_trend_hc_year_one_fade`` name; callers strip that
    prefix, see :func:`_era_magnitude_for_lead`).

    Feature-detected like every other optional artifact loader in this
    module: a missing directory, an unreadable/malformed file, or a signal
    with no usable era rows simply omits itself rather than raising -- an
    older checkout (or one that has never run the profile script) still
    renders "What we're watching" correctly, just without the extra row.
    Eras the profile itself marked ``insufficient_data`` are dropped rather
    than plotted as a zero -- absent evidence is not the same shape as a
    measured null.
    """

    directories = artifact_directories(artifacts_root / "era_magnitude_profile", "results.json")
    if not directories:
        return {}
    try:
        payload = read_json(directories[0] / "results.json")
    except (ValueError, OSError):
        return {}

    fixed_eras = payload.get("fixed_eras")
    signals = payload.get("signals")
    if not isinstance(fixed_eras, list) or not isinstance(signals, dict):
        return {}

    result: dict[str, list[EraMagnitude]] = {}
    for name, signal in signals.items():
        if not isinstance(signal, dict):
            continue
        era_results = signal.get("era_results")
        if not isinstance(era_results, dict):
            continue
        rows: list[EraMagnitude] = []
        for era in fixed_eras:
            if not isinstance(era, dict):
                continue
            key = era.get("key")
            era_row = era_results.get(key) if isinstance(key, str) else None
            if not isinstance(era_row, dict) or era_row.get("insufficient_data"):
                continue
            # Two shapes coexist in this artifact: most signals nest their
            # bootstrap under "week_blocked" (estimate/lower/upper/
            # probability_positive) alongside a separate top-level "effect".
            # At least one signal (production_model_opener_proxy_edge, a
            # re-sliced-not-rerun variant) instead stores the point estimate
            # as "estimate" with lower/upper/probability_positive directly
            # on the era row, no "effect" key and no nesting. Support both
            # rather than silently dropping every era of the second shape.
            effect = _number(era_row.get("effect"))
            interval: tuple[float, float] | None = None
            probability_positive: float | None = None
            week_blocked = era_row.get("week_blocked")
            if isinstance(week_blocked, dict):
                lower, upper = (
                    _number(week_blocked.get("lower")),
                    _number(week_blocked.get("upper")),
                )
                if lower is not None and upper is not None:
                    interval = (lower, upper)
                probability_positive = _number(week_blocked.get("probability_positive"))
            else:
                if effect is None:
                    effect = _number(era_row.get("estimate"))
                lower, upper = _number(era_row.get("lower")), _number(era_row.get("upper"))
                if lower is not None and upper is not None:
                    interval = (lower, upper)
                probability_positive = _number(era_row.get("probability_positive"))
            if effect is None:
                continue
            season_lo, season_hi = era.get("season_lo"), era.get("season_hi")
            label = (
                f"{season_lo}-{season_hi}"
                if season_lo is not None and season_hi is not None
                else str(key)
            )
            rows.append(
                EraMagnitude(
                    era_label=label,
                    effect=effect,
                    interval=interval,
                    probability_positive=probability_positive,
                )
            )
        if rows:
            result[str(name)] = rows
    return result


_ERA_TREND_PREFIX = "era_trend_"


def _era_magnitude_for_lead(
    lead_name: str, era_magnitude: Mapping[str, Sequence[EraMagnitude]]
) -> Sequence[EraMagnitude]:
    """The per-era rows for a ``WatchingLead``, if it IS an ``era_trend_*``
    signal and the profile covers it -- every other lead gets none, so this
    row only ever appears on the card the data was actually built to
    describe, never guessed onto an unrelated construct by name-matching."""

    if not lead_name.startswith(_ERA_TREND_PREFIX):
        return ()
    return era_magnitude.get(lead_name[len(_ERA_TREND_PREFIX) :], ())


def load_prospective_challengers(artifacts_root: Path) -> list[dict[str, Any]]:
    """The registered 2026 prospective challengers, read fresh every call.

    Feature-detected like every other optional artifact here: an absent or
    malformed ``challengers.json`` (or an untracked artifacts tree that has
    never had one) renders an empty list rather than raising -- the track
    record page's D3(a) section simply omits itself (see
    :func:`_challengers_section`).
    """

    path = artifacts_root / "prospective" / "challengers.json"
    if not path.is_file():
        return []
    try:
        payload = read_json(path)
    except (ValueError, OSError):
        return []
    challengers = payload.get("challengers")
    return (
        [entry for entry in challengers if isinstance(entry, dict)]
        if isinstance(challengers, list)
        else []
    )


def render_models_page(
    ledger_section: str | None,
    *,
    generated_at: datetime | None = None,
) -> str:
    """Render ``docs/models.html`` -- the Model Ledger on its own page.

    2026-08-22 de-clutter revision: the ledger moved OFF the picks page so
    index.html stays a clean week board. Same fragment, same fail-open
    discipline: an unavailable ledger renders a quiet note, never an error.
    """

    body = _section_header(
        "Model Ledger",
        "Every arm the card could come from",
        "The promoted production card first, then each candidate rule by best-evidence confidence.",
    )
    if not ledger_section:
        body += (
            '<p class="sub">Ledger unavailable right now -- '
            "the challenger registry or active-model manifest could not be read. "
            "This page rebuilds with every publish; nothing is hidden.</p>"
        )
    else:
        body += ledger_section
    body += _section_header(
        "WHAT THIS TABLE IS",
        "Reading the ledger",
        "",
        top=40,
    ) + (
        '<div class="prose">'
        "<p>Each row is a version of the picking system. &#8220;Model only&#8221; is "
        "the bare model. The promoted row is what actually makes this week's picks; "
        "candidate rules are measured but not played. P+ is our confidence an effect "
        "is real rather than luck.</p>"
        "</div>"
    )
    return _page(
        current=MODELS_PAGE,
        body=body,
        generated=(generated_at or datetime.now(UTC)),
    )


# ---------------------------------------------------------------------------
# Team explorer (UI-07) -- per-team pregame-state trends, canonical schema only
# ---------------------------------------------------------------------------


def _diverging_bar(z: float, max_abs: float) -> str:
    """A centered diverging bar: league average at 50%, team state left/right.

    Direction is encoded by placement AND a signed numeric label elsewhere, so
    the chart never relies on colour alone.
    """

    if not math.isfinite(z) or max_abs <= 0:
        return (
            '<div style="width:100px;height:8px;background:var(--grid);border-radius:4px;"></div>'
        )
    frac = max(-1.0, min(1.0, z / max_abs))
    center = 50.0
    pos = center + 50.0 * frac
    if frac >= 0:
        style = f"left:{center:g}%;width:{pos - center:g}%;"
    else:
        style = f"right:{100 - center:g}%;width:{center - pos:g}%;"
    return (
        '<div style="position:relative;width:100px;height:8px;background:var(--grid);'
        f'border-radius:4px;flex:none;">'
        f'<div style="position:absolute;top:0;height:8px;background:var(--ink-2);'
        f'border-radius:4px;{style}"></div></div>'
    )


def _signed(value: float, digits: int = 2) -> str:
    """Signed decimal, or an em dash for a missing/non-finite value."""

    if not math.isfinite(value):
        return "\u2014"
    text = f"{abs(value):.{digits}f}"
    return f"+{text}" if value > 0 else (f"-{text}" if value < 0 else text)


def _team_explorer_overview(trends: TeamTrends, metrics: Sequence[str]) -> str:
    """At-a-glance table: one row per team, one column per headline metric."""

    latest = trends.latest
    max_abs: dict[str, float] = {}
    for metric in metrics:
        column = latest.loc[latest["metric"] == metric, "z"]
        max_abs[metric] = float(column.abs().max()) if not column.empty else 0.0

    sort_metric = "point_diff" if "point_diff" in metrics else metrics[0]
    ordered = (
        latest.loc[latest["metric"] == sort_metric]
        .sort_values("z", ascending=False)["team"]
        .tolist()
    )
    teams = [t for t in ordered if t in trends.teams] + [
        t for t in trends.teams if t not in ordered
    ]

    head = "<th>Team</th>" + "".join(f"<th>{escape(metric_label(m))}</th>" for m in metrics)
    rows_html = []
    for team in teams:
        cells = [f"<td><b>{escape(str(team))}</b></td>"]
        for metric in metrics:
            row = latest.loc[(latest["team"] == team) & (latest["metric"] == metric)]
            if row.empty:
                cells.append("<td>\u2014</td>")
                continue
            value = float(row["value"].iloc[0])
            z = float(row["z"].iloc[0])
            bar = _diverging_bar(z, max_abs[metric])
            cells.append(
                "<td style='white-space:nowrap;'>"
                f"{bar}<span class='fine' style='margin-left:8px;'>{_signed(value)}</span></td>"
            )
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table = (
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )
    caption = (
        f'<p class="fine" style="margin:6px 0 14px;max-width:80ch;">'
        f"Latest season shown: {trends.latest_season}. Each bar places the team "
        f"versus the league average for that metric; the number is the raw "
        f"pregame-state value.</p>"
    )
    return table + caption


def _team_explorer_trend_details(trends: TeamTrends, metrics: Sequence[str]) -> str:
    """Collapsible per-team season-trend tables (metric x season)."""

    seasons = sorted(int(s) for s in trends.trend["season"].dropna().unique().tolist())
    blocks = []
    for team in trends.teams:
        head = "<th>Metric</th>" + "".join(f"<th>{escape(str(season))}</th>" for season in seasons)
        rows_html = []
        for metric in metrics:
            cells = [f"<td><b>{escape(metric_label(metric))}</b></td>"]
            for season in seasons:
                mask = (
                    (trends.trend["team"] == team)
                    & (trends.trend["metric"] == metric)
                    & (trends.trend["season"] == season)
                )
                value = trends.trend.loc[mask, "value"]
                cells.append(
                    f"<td>{_signed(float(value.iloc[0])) if not value.empty else '\u2014'}</td>"
                )
            rows_html.append(f"<tr>{''.join(cells)}</tr>")
        table = (
            '<div style="overflow-x:auto;">'
            '<table style="border-collapse:collapse;width:100%;font-size:13px;">'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
        )
        blocks.append(
            '<details style="margin-bottom:8px;border:1px solid var(--grid);'
            'border-radius:8px;padding:10px 14px;">'
            f'<summary style="cursor:pointer;font-weight:600;">{escape(str(team))}</summary>'
            f'<div style="margin-top:10px;">{table}</div></details>'
        )
    return "".join(blocks)


def _team_explorer_matchup(trends: TeamTrends, metrics: Sequence[str]) -> tuple[str, str]:
    """Two-team comparison: server-rendered default pair + a JS re-render hook.

    Returns ``(html, script)``. The comparison shows only ``z`` (team minus
    league mean) so no outcome or market field can leak onto the page.
    """

    teams = trends.teams
    if len(teams) >= 2:
        team_a, team_b = teams[0], teams[1]
    elif teams:
        team_a = team_b = teams[0]
    else:
        team_a = team_b = ""

    options = "".join('<option value="' + escape(t) + '">' + escape(t) + "</option>" for t in teams)
    payload = team_state_payload(trends)

    def _compare_rows(team_a: str, team_b: str) -> str:
        rows = []
        for metric in metrics:
            za = payload.get(team_a, {}).get(metric)
            zb = payload.get(team_b, {}).get(metric)
            if za is None or zb is None:
                rows.append(
                    f"<tr><td><b>{escape(metric_label(metric))}</b></td>"
                    "<td>\u2014</td><td>\u2014</td><td>\u2014</td></tr>"
                )
                continue
            diff = za - zb
            arrow = "\u25b2" if diff > 0 else ("\u25bc" if diff < 0 else "\u25ac")
            rows.append(
                f"<tr><td><b>{escape(metric_label(metric))}</b></td>"
                f"<td>{_signed(za)}</td><td>{_signed(zb)}</td>"
                f"<td>{arrow} {_signed(diff)}</td></tr>"
            )
        return "".join(rows)

    compare_rows = _compare_rows(team_a, team_b)
    head = (
        "<th>Metric</th>"
        f"<th id='ats-te-ha'>{escape(team_a)}</th>"
        f"<th id='ats-te-hb'>{escape(team_b)}</th>"
        "<th>A \u2212 B</th>"
    )
    html = (
        '<div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:12px;">'
        f'<label class="fine">Team A<select id="ats-te-a" style="margin-left:6px;">{options}'
        "</select></label>"
        f'<label class="fine">Team B<select id="ats-te-b" style="margin-left:6px;">{options}'
        "</select></label>"
        "</div>"
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;" '
        'id="ats-te-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody id='ats-te-body'>{compare_rows}</tbody></table></div>"
    )

    data_json = json.dumps(payload, separators=(",", ":"))
    labels_json = json.dumps([metric_label(m) for m in metrics], separators=(",", ":"))
    metrics_json = json.dumps(list(metrics), separators=(",", ":"))
    script = (
        '<script type="application/json" id="ats-te-data">' + data_json + "</script>\n"
        "<script>\n"
        "(function () {\n"
        "  var dataEl = document.getElementById('ats-te-data');\n"
        "  if (!dataEl) { return; }\n"
        "  var data; try { data = JSON.parse(dataEl.textContent); } catch (e) { return; }\n"
        "  var selA = document.getElementById('ats-te-a');\n"
        "  var selB = document.getElementById('ats-te-b');\n"
        "  var body = document.getElementById('ats-te-body');\n"
        "  var ha = document.getElementById('ats-te-ha');\n"
        "  var hb = document.getElementById('ats-te-hb');\n"
        "  var labels = " + labels_json + ";\n"
        "  var metricsArr = " + metrics_json + ";\n"
        "  function signed(v) {\n"
        "    if (v === null || v === undefined || isNaN(v)) { return '\u2014'; }\n"
        "    var t = Math.abs(v).toFixed(2);\n"
        "    return (v > 0 ? '+' : (v < 0 ? '-' : '')) + t;\n"
        "  }\n"
        "  function arrow(d) { return d > 0 ? '\u25b2' : (d < 0 ? '\u25bc' : '\u25ac'); }\n"
        "  function render() {\n"
        "    var a = selA.value, b = selB.value;\n"
        "    if (ha) { ha.textContent = a; } if (hb) { hb.textContent = b; }\n"
        "    var rows = '';\n"
        "    for (var i = 0; i < metricsArr.length; i++) {\n"
        "      var m = metricsArr[i];\n"
        "      var za = (data[a] && data[a][m] != null) ? data[a][m] : null;\n"
        "      var zb = (data[b] && data[b][m] != null) ? data[b][m] : null;\n"
        "      if (za === null || zb === null) {\n"
        "        rows += '<tr><td><b>' + labels[i] + '</b></td><td>\u2014</td><td>\u2014</td><td>\u2014</td></tr>';\n"  # noqa: E501
        "        continue;\n"
        "      }\n"
        "      var d = za - zb;\n"
        "      rows += '<tr><td><b>' + labels[i] + '</b></td>';\n"
        "      rows += '<td>' + signed(za) + '</td><td>' + signed(zb) + '</td>';\n"
        "      rows += '<td>' + arrow(d) + ' ' + signed(d) + '</td></tr>';\n"
        "    }\n"
        "    body.innerHTML = rows;\n"
        "  }\n"
        "  if (selA && selB && body) {\n"
        "    selA.addEventListener('change', render);\n"
        "    selB.addEventListener('change', render);\n"
        "    render();\n"
        "  }\n"
        "})();\n"
        "</script>\n"
    )
    return html, script


def render_team_explorer_page(
    state_table: pd.DataFrame | None,
    *,
    generated_at: datetime | None = None,
    metrics: Sequence[str] | None = None,
) -> str:
    """Render ``docs/team_explorer.html`` -- per-team pregame state trends.
    Consumes only the canonical team-state schema (see
    :mod:`nfl_ats.team_explorer`). With no local feature table available the
    page renders a quiet empty state -- the same fail-open contract every
    optional artifact on the site follows."""

    wanted = list(metrics) if metrics is not None else list(DEFAULT_TREND_METRICS)
    trends = aggregate_team_trends(state_table, metrics=wanted)

    sub = (
        "Each team's pregame state -- the exponentially-weighted offense/defense "
        "signal the model reads at kickoff -- averaged by season. Built only from "
        "the canonical team-state feature schema; no picks, lines, or outcomes."
    )
    body = viz.page_header("Team trends", "Per-team pregame state, by season", sub=sub)

    if trends.latest_season is None:
        body += viz.empty_state(
            "No team-state data yet",
            "The team-state feature table has not been built for this forecast. "
            "The page rebuilds with every publish once that artifact is present; "
            "nothing is hidden.",
        )
        return _page(
            current=TEAM_EXPLORER_PAGE,
            body=body,
            generated=(generated_at or datetime.now(UTC)),
        )

    body += _team_explorer_overview(trends, wanted)
    body += _section_header(
        "Per-team season trend",
        "One team at a time",
        "Expand a team to see its pregame state for every metric, season by season, "
        "against the league average.",
        top=40,
    )
    body += _team_explorer_trend_details(trends, wanted)
    matchup_html, matchup_script = _team_explorer_matchup(trends, wanted)
    body += _section_header(
        "Matchup comparison",
        "Two teams, side by side",
        "Pick any two teams to compare their latest pregame state. Bars and arrows "
        "show each team relative to the league average that season.",
        top=40,
    )
    body += matchup_html
    body += (
        '<p class="fine" style="margin-top:10px;max-width:80ch;">'
        "Bars and arrows show each team's pregame state relative to the league "
        "average for that season and metric. For rate stats (turnover rate, sack "
        "rate) a higher number is not necessarily better, so read the arrows as "
        "direction, not goodness.</p>"
    )
    return _page(
        current=TEAM_EXPLORER_PAGE,
        body=body,
        generated=(generated_at or datetime.now(UTC)),
        scripts=matchup_script,
    )


def render_pool_workbench_page(
    predictions: pd.DataFrame,
    pool_rules: PoolRules | None = None,
    *,
    season: int | None = None,
    week: int | None = None,
    model_id: str | None = None,
    generated_at: datetime | None = None,
    best_pick_game_id: str | None = None,
) -> str:
    """Render ``docs/pool.html`` -- the minimal pool workbench (UI-09).

    Composes the pool-rules input, the forced-pick entry list, the
    confidence ranks derived from the active model forecast, and the
    ownership-scenario placeholder, then wraps them in the shared page
    shell so the licensing/disclaimer guardrails apply unchanged.
    """

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    model_text = f"model <code>{escape(model_id)}</code>" if model_id else "model unknown"
    body = build_pool_workbench_body(
        predictions,
        pool_rules,
        best_pick_game_id=best_pick_game_id,
        season=season,
        week=week,
    )
    return _page(
        current=POOL_PAGE,
        body=body,
        generated=generated,
        footer_note=model_text,
    )


def build_public_site(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    generated_at: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
) -> dict[str, str]:
    """Build all three public pages: ``{file name: complete HTML document}``.

    Raises :class:`ValueError` when no synchronized active model + weekly
    forecast chain exists, exactly as the single-page builder did: publishing a
    board the model card itself would refuse to publish is never right.

    ``data_root`` locates the local schedule snapshot (coach-fade overlay) and
    market snapshot store (v2 Best Pick nomination) -- see
    :func:`nfl_ats.card_view.resolve_card_view`. Defaults to the same
    ``NFL_ATS_DATA_DIR``-driven path ``nfl-ats`` uses everywhere else, so the
    existing ``nfl-ats publish-board`` invocation (which passes only
    ``artifacts_root``) picks up both levers with no CLI change required.
    """

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    resolved_data_root = data_root if data_root is not None else _default_data_root()
    artifacts = load_public_board_artifacts(artifacts_root)
    # Pin the opener tiles to the ACTIVE model's own run. Without this the
    # newest directory wins and an unrelated research profile's grade gets
    # published under the active model's id (2026-08-18 incident).
    opener = load_opener_evaluation_artifacts(
        artifacts_root, active_feature_profile=artifacts.active.get("feature_profile")
    )

    # Consolidation law (2026-08-23): the active manifest's aggregate
    # historical accuracy is deliberately NOT rendered on the picks page
    # anymore -- Panel 1 carries exactly two accuracy stats (the pinned
    # planning hero and the measured chain history), and every other figure
    # lives in the collapsed ladder or on track_record.html.
    model_id = artifacts.active.get("model_id")

    # Computed ONCE and shared with the track-record page's Best Pick
    # section (B1/B2, D3(b)) rather than paying v2's cross-book
    # dispersion-pool scan twice per site build.
    game_type = (
        str(artifacts.predictions["game_type"].iloc[0])
        if "game_type" in artifacts.predictions and not artifacts.predictions.empty
        else "REG"
    )
    view = (
        resolve_card_view(
            artifacts.predictions,
            artifacts.sweep,
            artifacts.metadata,
            data_root=resolved_data_root,
            now=generated,
            require_fresh_arrest_overlay=require_fresh_arrest_overlay,
        )
        if game_type == "REG" and not artifacts.predictions.empty
        else None
    )
    overlay = (
        view.overlay
        if view is not None
        else resolve_overlay(artifacts.predictions, resolved_data_root)
    )
    arrest_overlay = (
        view.arrest_overlay
        if view is not None
        else resolve_player_arrests_overlay(
            overlay.overlaid_predictions,
            resolved_data_root,
            now=generated,
            require_fresh=require_fresh_arrest_overlay,
        )
    )
    nomination = view.nomination if view is not None else None
    best_pick_team: str | None = None
    best_pick_method_note = ""
    if nomination is not None and nomination.active_game_id is not None:
        final_card = view.predictions if view is not None else arrest_overlay.overlaid_predictions
        best_row = final_card.loc[final_card["game_id"].astype(str).eq(nomination.active_game_id)]
        if not best_row.empty:
            best_pick_team, _ = pick_side(best_row.iloc[0])
        best_pick_method_note = (
            nomination.method_note if nomination.active_rule == "v2" else nomination.active_tie_note
        )

    # Loaded once and shared with the findings page's "currently tracked"
    # section, mirroring how ``overlay``/``nomination`` above are computed
    # once and shared rather than paying the IO/scan twice per site build.
    challengers = load_prospective_challengers(artifacts_root)
    # Same reuse discipline: each challenger's "this week" preview and 2026
    # prospective record are computed ONCE here and shared by both pages, so
    # findings.html and track_record.html can never disagree about what a
    # challenger did this week.
    challenger_week_previews = _challenger_week_previews(
        challengers,
        artifacts.predictions,
        resolved_data_root,
        overlay=overlay,
        nomination=nomination,
        metadata=artifacts.metadata,
    )
    challenger_prospective_records = _challenger_prospective_records(artifacts_root, challengers)

    # P1's "recent form" line, read fresh from the newest prospective scoring
    # run -- omitted entirely (never guessed) when nothing has been settled.
    active_report = _load_latest_prospective_scoring(artifacts_root).get("active_model")
    active_record_text = _prospective_record_text(active_report) if active_report else ""
    # P1's "recent form" line, read fresh from the newest prospective scoring
    # run -- suppressed entirely while nothing has been settled yet, rather
    # than rendering the generic pending sentence as if it were a result.
    recent_form_text = (
        f"This season so far: {active_record_text}"
        if active_record_text and not active_record_text.startswith("Not scored yet")
        else None
    )

    # Week-board attribution feed + Model Ledger fragment (2026-08-21
    # integration wave): both optional, both fail-open -- see
    # :func:`load_waterfall_feed` and :func:`load_model_ledger_html`.
    waterfall_feed = load_waterfall_feed(artifacts_root)
    ledger_section = load_model_ledger_html(artifacts_root)

    # Panel 1's crowned stat (2026-08-23): the played chain's own historical
    # opener accuracy, read once here so the picks page never re-types it.
    played_chain_accuracy = load_played_chain_accuracy(artifacts_root)

    # Spread explorer (2026-08-20, owner request): per-game Gaussian params
    # for the picks-page slider widget. Only the "gaussian" probability
    # method (MOD-08, promoted 2026-08-19) has a closed-form mean/sd the
    # widget's erf formula can read -- an older/rolled-back "ecdf" active
    # model has no such closed form (its probability is a raw discretized
    # count, not a fitted density), so the widget is simply omitted for that
    # configuration, the SAME graceful-degradation contract every other
    # optional artifact on this page already follows. Once a gaussian card
    # exists to explain, though, a genuinely MISSING feature table is a real
    # data-integrity problem (the same table `margin-predict` itself needed)
    # and is treated as a hard failure -- matching
    # ``nfl_ats.smooth_cdf_mapping_overlay``'s identical judgment call, not a
    # silent degrade.
    spread_explorer_params: dict[str, SpreadExplorerGameParams] = {}
    if (
        str(artifacts.metadata.get("probability_method")) == "gaussian"
        and not artifacts.predictions.empty
    ):
        explorer_features = load_feature_table_for_forecast(artifacts.metadata, resolved_data_root)
        spread_explorer_params = compute_spread_explorer_params(
            artifacts.predictions,
            explorer_features,
            regressor=str(artifacts.metadata.get("regressor")),
            ridge_alpha=float(artifacts.metadata.get("ridge_alpha", 10.0)),
            feature_profile=str(artifacts.metadata.get("feature_profile")),
            min_train_games=int(artifacts.metadata.get("min_train_games", 500)),
        )
        # REQUIRED consistency check: the widget's own formula must reproduce
        # the published card at each game's own line before it ships.
        _assert_spread_explorer_matches_card(spread_explorer_params, artifacts.predictions)

    # Team explorer (UI-07): per-team pregame-state trends from the canonical
    # team-state feature schema. The exact feature table the active forecast's
    # own provenance points to carries the per-side pregame states; we melt it
    # into the team-long form team_explorer consumes. A missing/unreadable
    # table is the SAME fail-open contract every optional artifact follows --
    # the page renders a quiet empty state, never an error.
    team_states: pd.DataFrame = pd.DataFrame()
    if not artifacts.predictions.empty:
        try:
            explorer_features = load_feature_table_for_forecast(
                artifacts.metadata, resolved_data_root
            )
            converted = feature_table_to_team_states(explorer_features)
            if converted is not None:
                team_states = converted
        except Exception:
            team_states = pd.DataFrame()

    return {
        PICKS_PAGE: render_picks_page(
            artifacts.predictions,
            artifacts.sweep,
            artifacts.explanations,
            season=artifacts.metadata.get("season"),
            week=artifacts.metadata.get("week"),
            model_id=str(model_id) if model_id else None,
            generated_at=generated,
            metadata=artifacts.metadata,
            data_root=resolved_data_root,
            overlay=overlay,
            arrest_overlay=arrest_overlay,
            production_overlay=(view.production_overlay if view is not None else None),
            nomination=nomination,
            spread_explorer=spread_explorer_params,
            challengers=challengers,
            waterfall_feed=waterfall_feed,
            challenger_week_previews=challenger_week_previews,
            recent_form_text=recent_form_text,
            played_chain_accuracy=played_chain_accuracy,
        ),
        MODELS_PAGE: render_models_page(
            ledger_section,
            generated_at=generated,
        ),
        FINDINGS_PAGE: render_findings_page(
            generated_at=generated,
            challengers=challengers,
            challenger_week_previews=challenger_week_previews,
            challenger_prospective_records=challenger_prospective_records,
            artifacts_root=artifacts_root,
            active_model_id=str(model_id) if model_id else None,
        ),
        TRACK_RECORD_PAGE: render_track_record_page(
            opener.metadata,
            opener.seasons,
            artifacts.active,
            generated_at=generated,
            challengers=challengers,
            best_pick_rule=nomination.active_rule if nomination is not None else None,
            best_pick_team=best_pick_team,
            best_pick_method_note=best_pick_method_note,
            challenger_week_previews=challenger_week_previews,
            challenger_prospective_records=challenger_prospective_records,
            played_chain_accuracy=played_chain_accuracy,
        ),
        TEAM_EXPLORER_PAGE: render_team_explorer_page(
            team_states,
            generated_at=generated,
        ),
        POOL_PAGE: render_pool_workbench_page(
            artifacts.predictions,
            PoolRules.from_defaults(),
            season=artifacts.metadata.get("season"),
            week=artifacts.metadata.get("week"),
            model_id=str(model_id) if model_id else None,
            generated_at=generated,
            best_pick_game_id=(nomination.active_game_id if nomination is not None else None),
        ),
    }


__all__ = [
    "DISCLAIMER_FULL",
    "DISCLAIMER_SHORT",
    "FINDINGS_PAGE",
    "MODELS_PAGE",
    "PICKS_PAGE",
    "POOL_PAGE",
    "SITE_PAGES",
    "TEAM_EXPLORER_PAGE",
    "TRACK_RECORD_PAGE",
    "EraMagnitude",
    "OpenerEvaluationArtifacts",
    "PublicBoardArtifacts",
    "build_public_site",
    "confidence_word",
    "load_era_magnitude_profile",
    "load_model_ledger_html",
    "load_opener_evaluation_artifacts",
    "load_played_chain_accuracy",
    "load_prospective_challengers",
    "load_public_board_artifacts",
    "load_waterfall_feed",
    "pick_side",
    "render_findings_page",
    "render_models_page",
    "render_picks_page",
    "render_pool_workbench_page",
    "render_team_explorer_page",
    "render_track_record_page",
    "spread_words",
]
