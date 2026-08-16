"""Render the public GitHub Pages weekly picks board as one static HTML page.

This is a from-scratch, dependency-light port of the owner-approved design mock
(``build_picks_board.py``, built against the real synchronized Week 1 artifact) into a
pure render function over dataframes, so it can be unit-tested and wired into the CLI
publish path without pulling in extra runtime dependencies. What is ported verbatim
from the mock: the CSS design tokens and light/dark palettes, the per-row grid layout
and its mobile media query, the numberless 17-cell gradient strip with the amber-boxed
market column, the confidence-descending sort, the 57%/52% tier thresholds, and the
client-side "expand all / collapse all" toggle wired via ``addEventListener``.

Why this module does not import :mod:`nfl_ats.dashboard.board` or
:mod:`nfl_ats.dashboard.ui`, even though both already implement equivalent pick-side,
tier, and ribbon logic for the internal Streamlit dashboard: ``nfl_ats.dashboard.ui``
imports ``streamlit`` at module scope, so importing anything from ``nfl_ats.dashboard``
here would make ``streamlit`` a hard runtime dependency of the CLI publish path (`nfl-ats
publish-board`, wired into `nfl-ats publish-predictions --with-board`) purely to reach a
handful of pure functions. The relevant sign conventions, tier thresholds, and label
formatting are instead reproduced directly below; keep them in sync by hand if either
side changes.

Public-audience guardrail (licensing/ethics constraint, not a style choice): this page
renders only the fields already published in the repo's tracked public markdown card
(see :func:`nfl_ats.publishing._published_card`) -- pick, confidence, model fair line,
ONE consensus market line per game (``spread_line`` from the synchronized weekly
forecast, never a per-book quote), kickoff, and a plain-English explanation. It must
never render book names, per-book prices, or any other raw market-feed field. See
``tests/test_public_board.py`` for the enforced blocklist.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.reporting import artifact_directories, read_json

# ---------------------------------------------------------------------------
# Mandatory public-audience disclaimer text
#
# Embedded verbatim (not html.escape()'d) wherever it appears in the page: both
# strings are static, hardcoded, developer-authored constants -- never artifact
# or user data -- so there is no injection risk, and escaping would only mangle
# the apostrophe into "&#x27;" for no benefit.
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
# Sign conventions and label formatting (see module docstring for why these
# duplicate, rather than import, the equivalent dashboard helpers)
# ---------------------------------------------------------------------------

# Bands are defined on the *picked side's* probability (always >= 0.50), matching
# nfl_ats.dashboard.ui.CONFIDENCE_TIERS -- kept in sync by hand.
_TIER_THRESHOLDS: tuple[tuple[float, str, str], ...] = (
    (0.57, "strong", "Strong lean"),
    (0.52, "lean", "Lean"),
    (0.0, "flip", "Coin flip"),
)


def confidence_tier(pick_probability: float) -> tuple[str, str]:
    """(CSS class key, display label) for a picked-side probability."""

    for lower, key, label in _TIER_THRESHOLDS:
        if pick_probability >= lower:
            return key, label
    return "flip", "Coin flip"


def spread_label(value: float) -> str:
    """Bettor-conventional spread text: ``PK``, ``-7½``, ``+3``, ``+3.2``."""

    if value == 0:
        return "PK"
    fraction = abs(value) % 1.0
    if abs(fraction - 0.5) < 1e-9:
        whole = int(value - 0.5) if value > 0 else int(value + 0.5)
        if whole == 0:
            return "+½" if value > 0 else "-½"
        return f"{whole:+d}½"
    if value == int(value):
        return f"{int(value):+d}"
    return f"{value:+.1f}"


def pick_side_and_line(row: pd.Series) -> tuple[str, float, float]:
    """Return (picked team, picked side's spread, picked side's probability)."""

    home_probability = float(row["home_cover_probability"])
    home_pick = home_probability >= 0.5
    team = str(row["home_team"] if home_pick else row["away_team"])
    line = -float(row["spread_line"]) if home_pick else float(row["spread_line"])
    probability = home_probability if home_pick else 1.0 - home_probability
    return team, line, probability


def _pick_side_cells(row: pd.Series, game_sweep: pd.DataFrame) -> list[tuple[float, float, bool]]:
    """(alt_line, pick-side probability, is_market) descending by pick-side line."""

    if game_sweep.empty:
        return []
    team, _pick_line, _pick_probability = pick_side_and_line(row)
    home_pick = team == row["home_team"]
    cells: list[tuple[float, float, bool]] = []
    for _, sweep_row in game_sweep.sort_values("line_offset").iterrows():
        home_probability = float(sweep_row["home_cover_probability"])
        probability = home_probability if home_pick else 1.0 - home_probability
        alternative_home_line = float(sweep_row["alternative_line"])
        alt = -alternative_home_line if home_pick else alternative_home_line
        is_market = abs(float(sweep_row["line_offset"])) < 1e-9
        cells.append((alt, probability, is_market))
    if not home_pick:
        cells.reverse()
    return cells


def _cell_background(probability: float) -> str:
    percent = round(probability * 100)
    if percent >= 50:
        alpha_step = min(1.0, (percent - 50) / 18)
        return f"rgba(46,107,79,{0.12 + 0.55 * alpha_step:.3f})"
    alpha_step = min(1.0, (50 - percent) / 18)
    return f"rgba(140,90,60,{0.06 + 0.30 * alpha_step:.3f})"


def _ribbon_mini_html(cells: list[tuple[float, float, bool]]) -> str:
    parts = []
    for alt, probability, is_market in cells:
        percent = round(probability * 100)
        border = "border:2px solid var(--amber);" if is_market else "border:2px solid transparent;"
        background = _cell_background(probability)
        label = html.escape(spread_label(alt))
        parts.append(
            f'<td style="{border}background:{background};" class="rm" '
            f'title="{label}: {percent}%"></td>'
        )
    return f'<table class="ribbon-mini"><tr>{"".join(parts)}</tr></table>'


def _ribbon_full_html(cells: list[tuple[float, float, bool]]) -> str:
    parts = []
    for alt, probability, is_market in cells:
        percent = round(probability * 100)
        border = "border:2px solid var(--amber);" if is_market else "border:2px solid transparent;"
        background = _cell_background(probability)
        label = html.escape(spread_label(alt))
        parts.append(
            f'<td style="{border}background:{background};" class="rc">'
            f'<span class="rl">{label}</span><br>{percent}%</td>'
        )
    return f'<table class="ribbon-full"><tr>{"".join(parts)}</tr></table>'


@dataclass(frozen=True)
class _BoardRow:
    game_id: str
    matchup: str
    kickoff: str
    pick_text: str
    pick_probability: float
    edge: float
    market_line: float
    fair_line: float
    cells: list[tuple[float, float, bool]]
    explanation: str


def _build_row(row: pd.Series, game_sweep: pd.DataFrame, explanation: str) -> _BoardRow:
    team, pick_line, pick_probability = pick_side_and_line(row)
    home_pick = team == row["home_team"]
    market_line = float(row["spread_line"])
    residual = float(row["predicted_market_residual"])
    fair_line = market_line + residual
    edge = residual if home_pick else -residual
    kickoff = html.escape(
        f"{str(row.get('weekday') or '')[:3]} "
        f"{str(row.get('gameday') or '')[5:].replace('-', '/')} "
        f"{str(row.get('gametime') or '')[:5]}".strip()
    )
    return _BoardRow(
        game_id=str(row["game_id"]),
        matchup=html.escape(f"{row['away_team']} @ {row['home_team']}"),
        kickoff=kickoff,
        pick_text=html.escape(f"{team} {spread_label(pick_line)}"),
        pick_probability=pick_probability,
        edge=edge,
        market_line=market_line,
        fair_line=fair_line,
        cells=_pick_side_cells(row, game_sweep),
        explanation=explanation,
    )


def _row_html(row: _BoardRow) -> str:
    tier_key, _tier_label = confidence_tier(row.pick_probability)
    edge_text = f"{row.edge:+.1f}"
    edge_class = "good" if row.edge > 0.05 else ("bad" if row.edge < -0.05 else "flat")
    explanation_text = (
        html.escape(row.explanation)
        if row.explanation
        else ("Model and market are close on this one.")
    )
    detail = (
        f"<div class='detail'>{_ribbon_full_html(row.cells)}"
        f"<p class='expl'>{explanation_text}</p>"
        f"<p class='mini'>Market line {html.escape(spread_label(row.market_line))} (home) "
        f"&middot; model fair line {html.escape(spread_label(round(row.fair_line * 10) / 10))} "
        "(home)</p></div>"
    )
    return (
        f"<details class='row'><summary class='summary'><span class='kick'>{row.kickoff}</span>"
        f"<span class='match'>{row.matchup}</span>"
        f"<span class='pick {tier_key}'>{row.pick_text}</span>"
        f"<span class='conf'>{round(row.pick_probability * 100)}%</span>"
        f"<span class='edge {edge_class}'>{edge_text}</span>"
        f"<span class='rib'>{_ribbon_mini_html(row.cells)}</span>"
        f"<span class='chev'>&#9662;</span></summary>{detail}</details>"
    )


# ---------------------------------------------------------------------------
# Page shell (CSS ported verbatim from the approved mock; disclaimer, "last
# updated" stamp, and model id are new -- required for a public audience)
# ---------------------------------------------------------------------------

_CSS = """
  :root {
    --ground:#FAFAF7; --surface:#F1F2ED; --ink:#1C2420; --muted:#5C6862;
    --line:#D9DDD6; --accent:#2E6B4F; --amber:#B98A2E; --brick:#A8503C;
    --strong:#2E6B4F; --lean:#4A7A96; --flip:#8A8F8A;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --ground:#141816; --surface:#1C2220; --ink:#E8ECE9; --muted:#9AA69F;
    --line:#2E3733; --accent:#5FA983; --amber:#C9A75A; --brick:#C97B66;
    --strong:#5FA983; --lean:#7FA8C4; --flip:#7A807A;
  } }
  :root[data-theme="dark"] {
    --ground:#141816; --surface:#1C2220; --ink:#E8ECE9; --muted:#9AA69F;
    --line:#2E3733; --accent:#5FA983; --amber:#C9A75A; --brick:#C97B66;
    --strong:#5FA983; --lean:#7FA8C4; --flip:#7A807A;
  }
  * { box-sizing: border-box; }
  body { background:var(--ground); color:var(--ink); margin:0;
    font-family:"Segoe UI",system-ui,sans-serif; line-height:1.45; }
  .wrap { max-width:56rem; margin:0 auto; padding:1.5rem 1rem 3rem; }
  h1 { font-family:"Palatino Linotype",Georgia,serif; font-size:1.7rem; margin:0; }
  .sub { color:var(--muted); font-size:.85rem; margin:.2rem 0 .5rem; }
  .disclaimer-top { font-size:.78rem; font-weight:600; color:var(--brick); margin:0 0 1rem;
    padding:.5rem .7rem; border:1px solid var(--brick); border-radius:6px;
    background:var(--surface); }
  .chips { display:flex; gap:.5rem; margin:0 0 .8rem; flex-wrap:wrap; }
  .chip { font-size:.75rem; font-weight:700; padding:.15rem .6rem; border-radius:999px;
    border:1px solid var(--line); background:var(--surface); }
  .chip b { font-family:ui-monospace,Consolas,monospace; }

  .legend { color:var(--muted); font-size:.78rem; margin:0 0 .6rem; }
  .hdr, .row .summary { display:grid; align-items:center; gap:.6rem;
    grid-template-columns: 6.2rem 6.5rem 6.5rem 3.2rem 3.2rem 1fr 1rem; }
  .row summary { list-style:none; }
  .row summary::-webkit-details-marker { display:none; }
  .row[open] .chev { transform:rotate(180deg); }
  button.toggle { cursor:pointer; color:var(--accent); font:inherit; font-size:.75rem;
    font-weight:700; }
  button.toggle:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .hdr { font-size:.68rem; text-transform:uppercase; letter-spacing:.08em;
    color:var(--muted); padding:.25rem .5rem; }
  .row { border:1px solid var(--line); border-radius:6px; background:var(--surface);
    margin-bottom:.35rem; }
  .row .summary { padding:.45rem .5rem; cursor:pointer; }
  .kick { color:var(--muted); font-size:.78rem; }
  .match { font-weight:600; font-size:.9rem; }
  .pick { font-weight:700; font-family:ui-monospace,Consolas,monospace; font-size:.9rem;
    padding:.1rem .45rem; border-radius:4px; justify-self:start; }
  .pick.strong { color:#fff; background:var(--strong); }
  .pick.lean { color:#fff; background:var(--lean); }
  .pick.flip { color:var(--ink); background:transparent; border:1px solid var(--line); }
  .conf { font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums;
    font-weight:700; font-size:1rem; text-align:right; }
  .edge { font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums;
    font-size:.85rem; text-align:right; }
  .edge.good { color:var(--accent); } .edge.bad { color:var(--brick); }
  .edge.flat { color:var(--muted); }
  .ribbon-mini { width:100%; border-collapse:collapse; table-layout:fixed; height:14px; }
  .ribbon-mini td.rm { padding:0; height:14px; }
  .chev { color:var(--muted); font-size:.8rem; }
  .detail { padding:.3rem .6rem .7rem; border-top:1px solid var(--line); }
  .ribbon-full { width:100%; border-collapse:collapse; table-layout:fixed;
    font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums;
    font-size:.68rem; margin:.4rem 0; }
  .ribbon-full td.rc { text-align:center; padding:2px 0; }
  .ribbon-full .rl { color:var(--muted); font-size:.62rem; }
  .expl { font-size:.85rem; margin:.4rem 0 .2rem; }
  .mini { color:var(--muted); font-size:.75rem; margin:.2rem 0 0; }
  .empty { color:var(--muted); font-size:.9rem; padding:1rem 0; }
  .foot { color:var(--muted); font-size:.75rem; margin-top:1.2rem;
    border-top:1px solid var(--line); padding-top:.6rem; }
  .disclaimer-bottom { color:var(--muted); font-size:.72rem; margin-top:.6rem;
    padding:.6rem .7rem; border:1px solid var(--line); border-radius:6px;
    background:var(--surface); }
  @media (max-width:640px) {
    .hdr { display:none; }
    .row .summary { grid-template-columns: 1fr auto auto 1rem; grid-template-areas:
      "match pick conf chev" "kick kick kick kick" "rib rib rib rib"; }
    .match { grid-area:match; } .pick { grid-area:pick; } .conf { grid-area:conf; }
    .chev { grid-area:chev; } .kick { grid-area:kick; } .rib { grid-area:rib; }
    .edge { display:none; }
  }
"""

_TOGGLE_SCRIPT = """
<script>
  (function () {
    var button = document.getElementById("toggle-all");
    if (!button) return;
    button.style.display = "";
    button.addEventListener("click", function () {
      var rows = document.querySelectorAll("details.row");
      var shouldOpen = Array.prototype.some.call(rows, function (r) { return !r.open; });
      rows.forEach(function (r) { r.open = shouldOpen; });
      button.textContent = shouldOpen ? "collapse all" : "expand all";
    });
  })();
</script>
"""

_HEADER_ROW = (
    '<div class="hdr"><span>Kickoff</span><span>Game</span><span>Pick</span><span>Conf</span>'
    "<span>Edge</span><span>Confidence by line</span><span></span></div>"
)


def render_public_board(
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
    """Render the public picks board as one self-contained static HTML page.

    ``predictions`` is one row per game -- the active model's market_residual
    recommendations for the synchronized weekly forecast (``recommendations.csv``).
    ``sweep`` is the matching line-sweep table (``game_id`` + the columns
    :func:`_pick_side_cells` needs); an empty or ``None`` frame renders every row
    without a confidence strip. ``explanations`` maps ``game_id`` to a plain-English
    sentence from the market-decomposition attribution artifact; a game without one
    falls back to a neutral default sentence.

    Only fields already published on the tracked public markdown card are rendered:
    picks, confidence, the model's fair line, one consensus market line per game,
    kickoff, and explanations. No book names, per-book prices, or other raw
    market-feed fields are ever included -- see the module docstring.
    """

    explanations = explanations or {}
    sweep = sweep if sweep is not None else pd.DataFrame()
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    stamp = generated.strftime("%Y-%m-%d %H:%M UTC")

    rows: list[_BoardRow] = []
    if not predictions.empty:
        sweep_game_key = "game_id" if "game_id" in sweep.columns else None
        for _, game in predictions.iterrows():
            game_sweep = (
                sweep.loc[sweep[sweep_game_key].eq(game["game_id"])]
                if sweep_game_key is not None
                else pd.DataFrame()
            )
            rows.append(_build_row(game, game_sweep, explanations.get(str(game["game_id"]), "")))
        rows.sort(key=lambda r: -r.pick_probability)

    n_strong = sum(1 for r in rows if confidence_tier(r.pick_probability)[0] == "strong")
    n_lean = sum(1 for r in rows if confidence_tier(r.pick_probability)[0] == "lean")
    n_flip = len(rows) - n_strong - n_lean

    season_week = (
        f"{season} &middot; Week {week} &middot; "
        if season is not None and week is not None
        else ""
    )
    accuracy_text = (
        f"long-run accuracy &asymp;{historical_accuracy:.0%}"
        if historical_accuracy is not None
        else "long-run accuracy &asymp;52%"
    )

    body: str
    if not rows:
        body = '<p class="empty">No games are scheduled for this week&rsquo;s forecast yet.</p>'
    else:
        chips = (
            f'<div class="chips"><span class="chip" style="border-color:var(--strong)">'
            f"<b>{n_strong}</b> strong</span>"
            f'<span class="chip" style="border-color:var(--lean)"><b>{n_lean}</b> leans</span>'
            f'<span class="chip"><b>{n_flip}</b> coin flips</span>'
            '<span class="chip">sorted by confidence</span>'
            '<button class="chip toggle" id="toggle-all" type="button" '
            'style="display:none">expand all</button></div>'
        )
        legend = (
            '<p class="legend">Conf = model&rsquo;s estimated chance the pick covers at the line '
            "shown. Edge = points better (+) or worse (&minus;) than the model&rsquo;s fair line. "
            "The strip shows that confidence at every half-point within &plusmn;4 of the market "
            "line (amber box = the market line); tap a row for numbers and the model's "
            "explanation.</p>"
        )
        body = chips + legend + _HEADER_ROW + "".join(_row_html(row) for row in rows)

    model_text = html.escape(model_id) if model_id else "unknown"
    footer = (
        f'<p class="foot">Model <code>{model_text}</code> &middot; last updated {stamp} '
        f"&middot; {len(rows)} game{'s' if len(rows) != 1 else ''}.</p>"
        f'<div class="disclaimer-bottom">{DISCLAIMER_FULL}</div>'
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Week Board</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>The Week Board</h1>
  <p class="sub">{season_week}{len(rows)} games &middot; synchronized with the active model
    &middot; {accuracy_text} (not proof of a profitable edge)</p>
  <p class="disclaimer-top">{DISCLAIMER_SHORT}</p>
  {body}
  {footer}
</div>
{_TOGGLE_SCRIPT}
</body>
</html>
"""
    return page


# ---------------------------------------------------------------------------
# Artifact loading: active model manifest -> weekly forecast -> predictions +
# line sweep, and the latest market-decomposition attribution for explanations
# (feature-detected: an older repo checkout without that artifact still
# renders a board, just without per-game explanations).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicBoardArtifacts:
    predictions: pd.DataFrame
    sweep: pd.DataFrame
    explanations: dict[str, str]
    metadata: dict[str, Any]
    active: dict[str, Any]


def load_public_board_artifacts(artifacts_root: Path) -> PublicBoardArtifacts:
    """Load the synchronized weekly forecast and explanations for the public board.

    Mirrors :func:`nfl_ats.publishing._publication_context`'s validation of the
    active-model manifest chain (active model must be synchronized, the linked
    weekly forecast must match its model id and carry ``SYNCHRONIZED`` status), so
    the public board can never render a forecast the model card itself would
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


def build_public_board_html(artifacts_root: Path, *, generated_at: datetime | None = None) -> str:
    """Load the synchronized weekly forecast and render the public board page."""

    artifacts = load_public_board_artifacts(artifacts_root)
    historical_evaluation = artifacts.active.get("historical_evaluation")
    accuracy = (
        float(historical_evaluation["accuracy"])
        if isinstance(historical_evaluation, dict) and "accuracy" in historical_evaluation
        else None
    )
    return render_public_board(
        artifacts.predictions,
        artifacts.sweep,
        artifacts.explanations,
        season=artifacts.metadata.get("season"),
        week=artifacts.metadata.get("week"),
        model_id=str(artifacts.active.get("model_id"))
        if artifacts.active.get("model_id")
        else None,
        historical_accuracy=accuracy,
        generated_at=generated_at,
    )


__all__ = [
    "DISCLAIMER_FULL",
    "DISCLAIMER_SHORT",
    "PublicBoardArtifacts",
    "build_public_board_html",
    "confidence_tier",
    "load_public_board_artifacts",
    "pick_side_and_line",
    "render_public_board",
    "spread_label",
]
