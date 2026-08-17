"""Render the public GitHub Pages site: three static pages in the dashboard's design.

The internal Streamlit dashboard and this public site are the SAME design system.
Rather than maintain a second visual language, this module imports the dashboard's
pure presentation modules directly and composes the same components into
self-contained static HTML:

* :mod:`nfl_ats.dashboard.theme` -- ``stylesheet()`` (role tokens, light + dark).
* :mod:`nfl_ats.dashboard.viz` -- ``probability_meter``, ``line_journey``,
  ``sweep_curve``, ``season_bars``, ``stat_tile``, ``status_line``, ``card``,
  ``page_header``, ``empty_state``, ``interaction_script``.
* :mod:`nfl_ats.dashboard.findings_content` -- the findings text model.

None of those three imports ``streamlit``, so the CLI publish path
(``nfl-ats publish-board``, and ``nfl-ats publish-predictions --with-board``)
does not gain a Streamlit runtime dependency by reaching them. (This module's
older docstring claimed the opposite; that was true of the pre-rebuild
``nfl_ats.dashboard.ui``/``board`` modules and is obsolete for the modules above.)
What is still off-limits here is :mod:`nfl_ats.dashboard.state`,
:mod:`nfl_ats.dashboard.data`, and everything under ``dashboard.app_pages`` --
all of those import ``streamlit`` at module scope, so this module keeps its own
artifact loading (below) and PORTS the pages' composition instead of importing it.

Two deliberate differences from the Streamlit surface, both because a static page
has no Streamlit host:

* ``theme.theme_sync_script()`` is NOT shipped. It exists to poll Streamlit's
  live theme; with no host there is nothing to poll, and the stylesheet's bare
  ``prefers-color-scheme`` media query handles light/dark on its own. Its
  ``:not([data-theme="light"])`` guard is simply inert without a stamp.
* ``viz.interaction_script()`` ships as its own ``<script>`` tag (picks page
  only). There is no sanitizer here, so the one-script-per-block rule that
  forces the Streamlit pages to merge tags does not apply.

The components' no-SVG / no-tag-inside-JS discipline is preserved regardless:
one implementation serves both surfaces, so a "static pages could use SVG"
divergence would immediately rot the dashboard.

Public-audience guardrail (licensing/ethics constraint, not a style choice)
--------------------------------------------------------------------------
These pages render only fields already published in the repo's tracked public
markdown card (see :func:`nfl_ats.publishing._published_card`):

* the pick and its calibrated confidence,
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

``DISCLAIMER_SHORT`` and ``DISCLAIMER_FULL`` appear on every page: the short
form near the top and again in the footer, the full form in the footer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
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
from nfl_ats.reporting import artifact_directories, read_json

# ---------------------------------------------------------------------------
# Mandatory public-audience disclaimer text
#
# Embedded verbatim (not escaped) wherever it appears in a page: both strings
# are static, hardcoded, developer-authored constants -- never artifact or user
# data -- so there is no injection risk, and escaping would only mangle the
# apostrophe into "&#x27;" for no benefit.
# ---------------------------------------------------------------------------

DISCLAIMER_SHORT = (
    "Research project — simulated, paper picks only. Not betting advice. Past "
    "accuracy ≈ 52% is not proof of a profitable edge."
)

DISCLAIMER_FULL = (
    "This page is the output of a personal research project. Every pick shown is a "
    "simulated, paper pick made to evaluate a forecasting model — it is not betting "
    "advice, not a recommendation to wager, and no real money is risked on these picks "
    "by the author. The model's historical accuracy of roughly 52% is close to a coin flip "
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

# (file name, nav label, browser title) in nav order.
SITE_PAGES: tuple[tuple[str, str, str], ...] = (
    (PICKS_PAGE, "This week", "This week's picks"),
    (FINDINGS_PAGE, "What we've learned", "What we've learned"),
    (TRACK_RECORD_PAGE, "Track record", "Track record"),
)

# Page chrome only. Every color is a role token from theme.stylesheet(), so the
# static site inherits the dashboard's light/dark swap with no second palette.
_PAGE_CHROME = """
<style>
body { margin: 0; }
.ats { background: var(--plane); min-height: 100vh; }
.ats .wrap { max-width: 62rem; margin: 0 auto; padding: 24px 18px 52px; }
.ats a { color: var(--series-model); }
.ats nav.site { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 16px; }
.ats nav.site a.chip { text-decoration: none; }
.ats nav.site .chip.here { color: var(--series-model); border-color: currentColor; }
</style>
"""


def _nav(current: str) -> str:
    items = []
    for filename, label, _title in SITE_PAGES:
        if filename == current:
            items.append(f'<span class="chip here" aria-current="page">{escape(label)}</span>')
        else:
            items.append(f'<a class="chip" href="{filename}">{escape(label)}</a>')
    return f'<nav class="site">{"".join(items)}</nav>'


def _disclaimer_banner() -> str:
    return (
        '<div class="card" style="border-left:3px solid var(--critical);margin-bottom:18px;">'
        f'<p class="sub" style="font-weight:600;">{DISCLAIMER_SHORT}</p></div>'
    )


def _footer(generated: datetime, note: str = "") -> str:
    stamp = generated.strftime("%Y-%m-%d %H:%M UTC")
    lead = f"{note} &middot; " if note else ""
    return (
        '<div style="margin-top:36px;padding-top:14px;border-top:1px solid var(--grid);">'
        f'<p class="fine">{lead}page generated {stamp}.</p>'
        '<p class="fine" style="margin-top:10px;font-weight:600;color:var(--serious);">'
        f"{DISCLAIMER_SHORT}</p>"
        f'<p class="fine" style="margin-top:8px;max-width:82ch;">{DISCLAIMER_FULL}</p></div>'
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
# Sign conventions (ported from dashboard.app_pages.picks, which cannot be
# imported here -- it imports streamlit. Keep the two in sync by hand.)
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


# ---------------------------------------------------------------------------
# Page 1 -- This week (mirrors dashboard.app_pages.picks)
# ---------------------------------------------------------------------------


def _game_card(
    row: pd.Series,
    game_sweep: pd.DataFrame,
    explanation: str,
) -> str:
    game_id = str(row["game_id"])
    home, away = str(row["home_team"]), str(row["away_team"])
    market_spread = float(row["spread_line"])
    fair = _number(row.get("fair_spread"))
    residual = _number(row.get("predicted_market_residual")) or 0.0
    pick_team, pick_probability = pick_side(row)
    strong = abs(residual) >= STRONG_LEAN_POINTS

    # LICENSING (MKT-09 provider licensing/quota audit, ROADMAP.md): the public
    # site plots ONLY the one consensus market line this card already publishes
    # and our own fair line. The internal dashboard also plots an
    # archive-derived opener consensus and the predicted close; both are
    # derived from the purchased point-in-time market feed and stay off the
    # public site until that audit clears redistribution.
    #
    # fair_spread shares spread_line's home-oriented sign convention (margin.py
    # builds it as the predicted home margin), so both numbers plot on one
    # scale without any sign flip.
    # "market", not "opened": the public value is the card's consensus line,
    # never a captured opener (those stay internal pending MKT-09).
    journey = viz.line_journey(
        opener=market_spread, fair=fair, predicted_close=None, opener_label="market"
    )

    # The sweep, +/-4 points around the quote -- reoriented to OUR pick's side.
    # The artifact's own pick_probability re-picks the favored side at every
    # alternative line (a V around the crossing point); what the card promises
    # is "the chance OUR pick covers if the line were X", which is the
    # home-cover probability flipped to the picked side.
    curve_html = ""
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
        curve_html = viz.sweep_curve(
            f"sweep-{game_id}",
            points,
            quoted_line=0.0,
            pick_text=f"{pick_team} to cover",
            quote_label=spread_words(home, away, market_spread),
        )

    if strong:
        lean_text = (
            f"We make this line {abs(residual):.1f} points different from the "
            f"market, on the {pick_team} side."
        )
        story = (
            f'<p class="prose" style="margin-top:6px;">{escape(explanation)}</p>'
            if explanation
            else '<p class="fine" style="margin-top:6px;">The per-game breakdown behind this '
            "lean has not been published for this card yet.</p>"
        )
        explanation_html = (
            '<div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--grid);">'
            '<p class="kicker" style="color:var(--series-model);">What we think the '
            "market is missing</p>"
            f'<p class="sub" style="font-weight:600;">{escape(lean_text)}</p>{story}</div>'
        )
    else:
        explanation_html = (
            '<div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--grid);">'
            '<p class="fine">We land close to the market\'s number here -- no strong '
            "opinion, just the forced pick the probability favors.</p></div>"
        )

    accent = "border-left:3px solid var(--series-model);" if strong else ""
    return (
        f'<div class="card" style="{accent}margin-top:14px;">'
        '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;'
        'gap:8px;align-items:baseline;">'
        f'<div><p class="kicker">{escape(_kickoff_words(row))}</p>'
        f'<h3 class="title" style="font-size:19px;">{escape(away)} at {escape(home)}</h3>'
        f'<p class="sub">Market: <span class="num">'
        f"{escape(spread_words(home, away, market_spread))}</span></p></div>"
        f'<div style="text-align:right;"><p class="kicker">Our pick</p>'
        f'<div class="hero num" style="font-size:26px;color:var(--series-model);">'
        f"{escape(pick_team)}</div></div></div>"
        '<div class="row" style="margin-top:14px;">'
        f"<div>{viz.probability_meter(pick_probability, label='Chance the pick covers')}</div>"
        f'<div><p class="fine" style="margin-bottom:4px;">Where the line sits</p>{journey}</div>'
        "</div>"
        + (
            f'<div style="margin-top:12px;"><p class="fine" style="margin-bottom:4px;">'
            f"Confidence if the line moves (four points either side)</p>{curve_html}</div>"
            if curve_html
            else ""
        )
        + explanation_html
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
    historical_accuracy: float | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Render ``docs/index.html`` -- this week's forced picks, one card per game.

    ``predictions`` is one row per game (the active model's ``recommendations.csv``
    for the synchronized weekly forecast); ``sweep`` is the matching
    ``line_sweep.parquet`` already filtered to the active method; ``explanations``
    maps ``game_id`` to the market-decomposition sentence. An empty ``predictions``
    frame renders the shell plus an empty state.

    Only the allowlisted public fields are rendered -- see the module docstring.
    """

    explanations = explanations or {}
    sweep = sweep if sweep is not None else pd.DataFrame()
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)

    accuracy_text = (
        f"long-run accuracy &asymp;{historical_accuracy:.0%}"
        if historical_accuracy is not None
        else "long-run accuracy &asymp;52%"
    )
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
            footer_note=f"{model_text} &middot; {accuracy_text}",
        )

    game_type = str(predictions["game_type"].iloc[0]) if "game_type" in predictions else "REG"
    week_label = _WEEK_LABELS.get(game_type, f"Week {week}")
    season_label = f"{season} · " if season is not None else ""

    lean_count = int(
        (predictions["predicted_market_residual"].abs() >= STRONG_LEAN_POINTS).sum()
        if "predicted_market_residual" in predictions
        else 0
    )
    header = viz.page_header(
        f"{season_label}{week_label} · {len(predictions)} games",
        "This week's picks",
        "Every game gets a forced pick against the pool's line. Confidence is a "
        "calibrated probability, not bravado -- and our strongest leans have not "
        "proven more likely to win than the rest, so no pick gets extra weight.",
    )
    chips = (
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin:-6px 0 14px;">'
        + viz.status_line("good", "Synchronized with the active model")
        + '<span class="chip model"><span class="dot" style="background:var(--series-model);">'
        + f"</span>{lean_count} strong lean{'s' if lean_count != 1 else ''}</span></div>"
    )

    sort_columns = [column for column in ("kickoff", "game_id") if column in predictions]
    ordered = (
        predictions.sort_values(sort_columns, na_position="last") if sort_columns else predictions
    )

    has_sweep = not sweep.empty and {"game_id", "line_offset", "home_cover_probability"}.issubset(
        sweep.columns
    )
    cards = []
    for _, row in ordered.iterrows():
        game_id = str(row["game_id"])
        game_sweep = pd.DataFrame()
        if has_sweep:
            game_sweep = sweep.loc[
                sweep["game_id"].astype(str).eq(game_id)
                & sweep["line_offset"].abs().le(SWEEP_HALF_WIDTH)
            ].sort_values("line_offset")
        cards.append(_game_card(row, game_sweep, explanations.get(game_id, "")))

    return _page(
        current=PICKS_PAGE,
        body=header + chips + "".join(cards),
        generated=generated,
        footer_note=(
            f"{model_text} &middot; {accuracy_text} &middot; lines are home-oriented "
            "spreads at card-build time; the pool's exact number can differ by a half point"
        ),
        # No sanitizer on a static page, so the sweep's delegated crosshair/tooltip
        # wiring ships as its own script tag rather than riding the theme sync
        # (which is Streamlit-only and deliberately omitted -- see the docstring).
        scripts=viz.interaction_script(),
    )


# ---------------------------------------------------------------------------
# Page 2 -- What we've learned (mirrors dashboard.app_pages.findings)
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
    header = _section_header(f"{group.kicker} · {len(findings)}", group.title, group.blurb)
    return header + _rows([_finding_card(finding, group) for finding in findings])


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


def render_findings_page(*, generated_at: datetime | None = None) -> str:
    """Render ``docs/findings.html`` -- every finding, grouped by verdict.

    All content comes from :mod:`nfl_ats.dashboard.findings_content`; this is
    layout only, and it publishes no market data of any kind.
    """

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    body = (
        _findings_hero() + "".join(_group_section(group) for group in GROUPS) + _honesty_section()
    )
    return _page(
        current=FINDINGS_PAGE,
        body=body,
        generated=generated,
        footer_note="every claim traces to a committed record in this repository",
    )


# ---------------------------------------------------------------------------
# Page 3 -- Track record (mirrors dashboard.app_pages.track_record)
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


def _track_record_tiles(opener_metadata: Mapping[str, Any], active: Mapping[str, Any]) -> str:
    metrics = _mapping(dict(opener_metadata), "metrics")
    opener_accuracy = _number(metrics.get("opener_accuracy"))
    close_accuracy = _number(metrics.get("close_accuracy"))
    opener_games = _number(opener_metadata.get("games"))

    tiles: list[str] = []
    if opener_accuracy is None:
        tiles.append(
            viz.empty_state(
                "The pool grade has not been measured yet",
                "The pool freezes its spread early in the week, so the opening-line grade is "
                "the number that decides it. It is not published here until it is measured.",
            )
        )
    else:
        games_text = f"{int(opener_games):,} games" if opener_games else "every paired game"
        tiles.append(
            viz.stat_tile(
                "Against the pool's line",
                f"{opener_accuracy:.1%}",
                "How often the forced picks landed against the spread frozen early in the week "
                f"-- the line the pool actually grades. Measured on {games_text} from 2020-2025 "
                "that the model never trained on.",
                delta_text=_versus_coin_flip(opener_accuracy),
                delta_good=opener_accuracy >= 0.5,
            )
        )
        if close_accuracy is not None:
            tiles.append(
                viz.stat_tile(
                    "Against the closing line",
                    f"{close_accuracy:.1%}",
                    "The same picks and the same games, graded against the sharper end-of-week "
                    "line. Lower on purpose: the market spends the week drifting toward our "
                    "number, and a frozen line hands that drift back to us.",
                    delta_text=_versus_coin_flip(close_accuracy),
                    delta_good=close_accuracy >= 0.5,
                )
            )

    historical = _mapping(dict(active), "historical_evaluation")
    model_accuracy = _number(historical.get("accuracy"))
    if model_accuracy is None:
        tiles.append(
            viz.card(
                '<p class="kicker">The model\'s own long-run record</p>'
                '<div class="hero num">--</div>'
                '<p class="fine" style="margin-top:6px;">No active model is linked yet, so '
                "there is no long-run record to quote.</p>"
            )
        )
    else:
        model_games = _number(historical.get("games")) or 0.0
        model_correct = _number(historical.get("correct")) or 0.0
        season_range = _mapping(historical.get("intervals"), "season")
        range_lower = _number(season_range.get("lower"))
        range_upper = _number(season_range.get("upper"))
        delta_text = (
            _plausible_range(range_lower, range_upper)
            if range_lower is not None and range_upper is not None
            else None
        )
        tiles.append(
            viz.stat_tile(
                "The model's own long-run record",
                f"{model_accuracy:.1%}",
                f"{int(model_correct):,} correct out of {int(model_games):,} games it was "
                "tested on but never trained on. The range beside it is the honest one: this "
                "is a single sample of seasons, not a promise about the next one.",
                delta_text=delta_text,
                delta_good=None,
            )
        )
    return f'<div class="row">{"".join(tiles)}</div>'


def _season_section(seasons: pd.DataFrame) -> str:
    if seasons.empty or not {"season", "opener_accuracy"}.issubset(seasons.columns):
        return ""

    season_rows: list[tuple[str, float]] = []
    season_cells: list[list[str]] = []
    games_by_season: dict[str, float] = {}
    for _, row in seasons.iterrows():
        label = str(row["season"]).split(".")[0]
        opener_value = _number(row.get("opener_accuracy"))
        if opener_value is None:
            continue
        close_value = _number(row.get("close_accuracy"))
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

    above = sum(1 for _, value in season_rows if value >= 0.5)
    losing = [(label, value) for label, value in season_rows if value < 0.5]
    honesty = f"{above} of the {len(season_rows)} seasons finished above the coin flip. "
    if losing:
        listed = ", ".join(f"{label} at {value:.1%}" for label, value in losing)
        honesty += (
            f"{'One did not' if len(losing) == 1 else 'Some did not'}: {listed}. "
            "That is on the chart on purpose -- a page that hides its losing seasons is a "
            "sales pitch, not a track record. "
        )
    if any(label == "2020" for label, _ in losing):
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
    pool_range = _season_interval(opener_metadata.get("uncertainty"), "opener_accuracy")
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
        "<p><b>52-53% against a frozen line is genuinely good.</b> Realistically excellent is "
        "54-55%. Someone who knew everything knowable before kickoff would top out near "
        "57-58% against a frozen Tuesday line, because football itself is noisy -- final "
        "margins scatter about 13.5 points around even a perfect expectation. If this page "
        "ever shows 60%, that is a bug to hunt, not a breakthrough.</p>"
        "</div>"
        '<p class="fine" style="margin-top:10px;">The ceiling arithmetic behind that last '
        "paragraph is written up in docs/pool_edge_plan.md.</p>"
    )
    return _spaced(viz.card(inner))


def render_track_record_page(
    opener_metadata: Mapping[str, Any] | None = None,
    seasons: pd.DataFrame | None = None,
    active: Mapping[str, Any] | None = None,
    *,
    generated_at: datetime | None = None,
) -> str:
    """Render ``docs/track_record.html`` -- the graded record, against both lines.

    Everything here is an AGGREGATE statistic (accuracy rates, per-season rates,
    their ranges), which is publishable; no raw market quote reaches this page.
    """

    opener_metadata = opener_metadata or {}
    seasons = seasons if seasons is not None else pd.DataFrame()
    active = active or {}
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)

    body = (
        viz.page_header(
            "Track record",
            "How often the picks actually landed",
            "Graded against two different lines. The one the pool uses comes first.",
        )
        + _track_record_tiles(opener_metadata, active)
        + _season_section(seasons)
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

    explanations: dict[str, str] = {}
    decomposition_directories = artifact_directories(
        artifacts_root / "market_decomposition", "attribution.parquet"
    )
    if decomposition_directories:
        attribution = pd.read_parquet(decomposition_directories[0] / "attribution.parquet")
        if "explanation" in attribution.columns and "game_id" in attribution.columns:
            explanations = {
                str(game_id): str(explanation)
                for game_id, explanation in (
                    attribution.dropna(subset=["explanation"])
                    .drop_duplicates("game_id")
                    .set_index("game_id")["explanation"]
                    .to_dict()
                    .items()
                )
            }

    return PublicBoardArtifacts(predictions, sweep, explanations, metadata, active)


def load_opener_evaluation_artifacts(artifacts_root: Path) -> OpenerEvaluationArtifacts:
    """Load the most recent opener-evaluation run, or empties when none exists."""

    directories = artifact_directories(artifacts_root / "opener_evaluation", "metadata.json")
    if not directories:
        return OpenerEvaluationArtifacts({}, pd.DataFrame())
    directory = directories[0]
    metadata = read_json(directory / "metadata.json")
    seasons = pd.DataFrame()
    season_path = directory / "season_summary.csv"
    if season_path.is_file():
        seasons = pd.read_csv(season_path)
    return OpenerEvaluationArtifacts(metadata, seasons)


def build_public_site(
    artifacts_root: Path, *, generated_at: datetime | None = None
) -> dict[str, str]:
    """Build all three public pages: ``{file name: complete HTML document}``.

    Raises :class:`ValueError` when no synchronized active model + weekly
    forecast chain exists, exactly as the single-page builder did: publishing a
    board the model card itself would refuse to publish is never right.
    """

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    artifacts = load_public_board_artifacts(artifacts_root)
    opener = load_opener_evaluation_artifacts(artifacts_root)

    historical_evaluation = artifacts.active.get("historical_evaluation")
    accuracy = (
        _number(historical_evaluation.get("accuracy"))
        if isinstance(historical_evaluation, dict)
        else None
    )
    model_id = artifacts.active.get("model_id")

    return {
        PICKS_PAGE: render_picks_page(
            artifacts.predictions,
            artifacts.sweep,
            artifacts.explanations,
            season=artifacts.metadata.get("season"),
            week=artifacts.metadata.get("week"),
            model_id=str(model_id) if model_id else None,
            historical_accuracy=accuracy,
            generated_at=generated,
        ),
        FINDINGS_PAGE: render_findings_page(generated_at=generated),
        TRACK_RECORD_PAGE: render_track_record_page(
            opener.metadata,
            opener.seasons,
            artifacts.active,
            generated_at=generated,
        ),
    }


__all__ = [
    "DISCLAIMER_FULL",
    "DISCLAIMER_SHORT",
    "FINDINGS_PAGE",
    "PICKS_PAGE",
    "SITE_PAGES",
    "TRACK_RECORD_PAGE",
    "OpenerEvaluationArtifacts",
    "PublicBoardArtifacts",
    "build_public_site",
    "load_opener_evaluation_artifacts",
    "load_public_board_artifacts",
    "pick_side",
    "render_findings_page",
    "render_picks_page",
    "render_track_record_page",
    "spread_words",
]
