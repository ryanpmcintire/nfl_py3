"""This week -- the flagship page.

Per game, the three prediction views the owner asked for: our calibrated
confidence at the line we picked against (the pool-relevant number), our
guess at where the line will close, and how our confidence holds up across
alternative lines four points either side of the quote. A plain-English
"what we think the market is missing" appears ONLY when we deviate from the
market enough to call it a strong lean -- everywhere else we say plainly
that we're with the market.
"""

from __future__ import annotations

import math
from html import escape

import pandas as pd
import streamlit as st

from nfl_ats.best_pick import best_pick_tie_note, select_best_pick
from nfl_ats.dashboard import theme, viz
from nfl_ats.dashboard.data import (
    artifact_time,
    artifacts_root,
    data_root,
    explanations_by_game,
    find_latest_attribution_file,
    game_opening_lines,
    list_recent_market_snapshots,
    load_live_quotes,
)
from nfl_ats.dashboard.state import load_parquet, project_state

# A "strong lean" is a fair-line disagreement with the market of at least
# this many points. It gates the explanation, not the pick: every pool pick
# is forced, and our confidence ordering has NOT proven predictive (see the
# findings page), so the lean is a narrative marker, never a weighting.
STRONG_LEAN_POINTS = 1.5
SWEEP_HALF_WIDTH = 4.0

state = project_state()

if state.forecast is None:
    st.html(
        theme.stylesheet()
        + '<div class="ats">'
        + viz.page_header("This week", "No pick card yet")
        + viz.empty_state(
            "Nothing has been generated since the last kickoff",
            "Once the Tuesday opening lines are captured and a forecast card is "
            "built (`nfl-ats margin-predict`), this page fills in by itself. "
            "The track record is open in the meantime.",
        )
        + "</div>"
        + theme.theme_sync_script()
    )
    st.stop()

forecast = state.forecast
recommendations = forecast.recommendations.copy()
metadata = forecast.metadata
season = metadata.get("season", "?")
week = metadata.get("week", "?")
game_type = (
    str(recommendations["game_type"].iloc[0])
    if "game_type" in recommendations and not recommendations.empty
    else "REG"
)
week_label = {
    "WC": "Wild Card round",
    "DIV": "Divisional round",
    "CON": "Conference championships",
    "SB": "Super Bowl",
}.get(game_type, f"Week {week}")

# --- Optional companions, all feature-detected ------------------------------
sweep = load_parquet(forecast.directory / "line_sweep.parquet")
ats_method = str(metadata.get("ats_method", "market_residual"))
if not sweep.empty and "method" in sweep:
    sweep = sweep.loc[sweep["method"].eq(ats_method)]

close_predictions = pd.DataFrame()
if state.close_predictions.directory is not None:
    close_predictions = load_parquet(state.close_predictions.directory / "predictions.parquet")

attribution_path = find_latest_attribution_file(artifacts_root())
explanations: dict[str, str] = {}
if attribution_path is not None:
    explanations = explanations_by_game(load_parquet(attribution_path))

openers: dict[str, float] = {}
# The live capture archive lives at data/market/raw (see the market ingest
# commands in cli.py) -- not data/odds, which does not exist.
snapshots = list_recent_market_snapshots(data_root() / "market" / "raw")
if snapshots:
    quotes = load_live_quotes(tuple(info.directory for info in snapshots))
    if not quotes.empty:
        opener_frame = game_opening_lines(quotes)
        openers = dict(
            zip(
                opener_frame["nflverse_game_id"].astype(str),
                opener_frame["opener_home_spread"].astype(float),
                strict=True,
            )
        )


def _spread_words(home: str, away: str, home_spread: float) -> str:
    """'DEN -3.5' style, from the home-oriented nflverse spread."""

    if math.isnan(home_spread) or home_spread == 0:
        return "pick 'em"
    favorite, points = (home, home_spread) if home_spread > 0 else (away, -home_spread)
    return f"{favorite} -{points:g}"


def _kickoff_words(row: pd.Series) -> str:
    weekday = str(row.get("weekday") or "").strip()
    gametime = str(row.get("gametime") or "").strip()
    return f"{weekday} {gametime} ET".strip()


def _pick_for(row: pd.Series) -> tuple[str, float]:
    """Forced pool pick: the side our calibrated probability favors."""

    probability = float(row["home_cover_probability"])
    if probability >= 0.5:
        return str(row["home_team"]), probability
    return str(row["away_team"]), 1.0 - probability


sections: list[str] = [theme.stylesheet()]

lean_count = int(
    (recommendations["predicted_market_residual"].abs() >= STRONG_LEAN_POINTS).sum()
    if "predicted_market_residual" in recommendations
    else 0
)
sync_chip = (
    viz.status_line("good", "Synchronized with the active model")
    if state.synchronized
    else viz.status_line("warning", "Not the active model's card -- see the engine room")
)
header = viz.page_header(
    f"{season} · {week_label} · {len(recommendations)} games",
    "This week's picks",
    "Every game gets a forced pick against the pool's line. Confidence is a "
    "calibrated probability, not bravado -- and our strongest leans have not "
    "proven more likely to win than the rest, so no pick gets extra weight.",
)
issue_cards = "".join(
    f'<div class="card" style="border-left:3px solid var(--serious);">'
    f"{viz.status_line('warning', issue)}</div>"
    for issue in state.issues
)
sections.append(
    '<div class="ats">'
    + header
    + f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:-6px 0 14px;">{sync_chip}'
    + '<span class="chip model"><span class="dot" style="background:var(--series-model);">'
    + f"</span>{lean_count} strong lean{'s' if lean_count != 1 else ''}</span></div>"
    + issue_cards
    + "</div>"
)

# POL-09: the week's Best Pick from the confirmed sweep_robustness signal, computed
# on the FULL sweep before the +/-SWEEP_HALF_WIDTH narrowing the plot uses -- ranking
# on the narrowed frame would score a different, unconfirmed signal. Regular season
# only: the pool awards a Best Pick per regular-season week.
best_pick_id = (
    select_best_pick(recommendations, sweep)
    if not sweep.empty and str(game_type) == "REG"
    else None
)
if best_pick_id is not None:
    best_row = recommendations.loc[recommendations["game_id"].astype(str).eq(best_pick_id)]
    if not best_row.empty:
        best_team, _best_probability = _pick_for(best_row.iloc[0])
        # POL-05: the signal is censored at 8.0, so weeks routinely tie at the top and
        # the alphabetical game_id tie-break decides among the tied games. Removing that
        # luck cuts the recorded edge from +8.68 to +0.92 (docs/pool_format_levers.md),
        # so a tied week must be shown as arbitrary rather than as a lean. Shared with the
        # published card (nfl_ats.publishing) so the wording cannot drift between them.
        _note = best_pick_tie_note(recommendations, sweep)
        _tie_note = f" {_note}" if _note else ""
        sections.append(
            '<div class="ats"><div class="card" style="border-left:3px solid '
            'var(--good);margin-bottom:14px;">'
            '<p class="kicker" style="color:var(--good-text);font-weight:700;">'
            "&#9733; BEST PICK OF THE WEEK</p>"
            f'<div class="hero num" style="font-size:26px;color:var(--good-text);">'
            f"{escape(best_team)}</div>"
            '<p class="sub">The pool scores one Best Pick a week. This is the pick whose '
            "edge holds up across the widest range of line movement -- the only confidence "
            "signal that has beaten picking among our own picks at random "
            f"(docs/best_pick_ranker.md).{_tie_note}</p></div></div>"
        )

ordered = recommendations.sort_values(["kickoff", "game_id"], na_position="last")
cards: list[str] = []
for _, row in ordered.iterrows():
    game_id = str(row["game_id"])
    home, away = str(row["home_team"]), str(row["away_team"])
    market_spread = float(row["spread_line"])
    fair = float(row["fair_spread"]) if pd.notna(row.get("fair_spread")) else None
    residual = (
        float(row["predicted_market_residual"])
        if pd.notna(row.get("predicted_market_residual"))
        else 0.0
    )
    pick_team, pick_probability = _pick_for(row)
    strong = abs(residual) >= STRONG_LEAN_POINTS

    # View 2 inputs -- all optional, labeled honestly.
    opener = openers.get(game_id)
    predicted_close = None
    if not close_predictions.empty and "game_id" in close_predictions:
        match = close_predictions.loc[close_predictions["game_id"].astype(str).eq(game_id)]
        if not match.empty and "predicted_close_home_spread" in match:
            predicted_close = float(match.iloc[0]["predicted_close_home_spread"])

    # fair_spread shares spread_line's home-oriented sign convention
    # (margin.py builds it as the predicted home margin), so all three
    # numbers plot on one scale without any sign flip. The first slot is
    # labeled honestly: "opened" only when a captured archive opener exists,
    # "market" when it falls back to the card-build spread.
    journey = viz.line_journey(
        opener=opener if opener is not None else market_spread,
        fair=fair,
        predicted_close=predicted_close,
        opener_label="opened" if opener is not None else "market",
    )

    # View 3: the sweep, +/-4 points around the quote -- reoriented to OUR
    # pick's side. The artifact's pick_probability re-picks the favored side
    # at every alternative line (a V around the crossing point); what the
    # card promises is "the chance OUR pick covers if the line were X", which
    # is the home-cover probability flipped to the picked side.
    curve_html = ""
    if not sweep.empty:
        game_sweep = sweep.loc[
            sweep["game_id"].astype(str).eq(game_id)
            & sweep["line_offset"].abs().le(SWEEP_HALF_WIDTH)
        ].sort_values("line_offset")
        if not game_sweep.empty:
            pick_is_home = pick_team == home
            points = [
                (
                    float(offset),
                    float(probability) if pick_is_home else 1.0 - float(probability),
                )
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
                quote_label=_spread_words(home, away, market_spread),
            )

    # The explanation: only for strong leans, and honest when we have none.
    if strong:
        explanation = explanations.get(game_id)
        lean_side = pick_team
        lean_text = (
            f"We make this line {abs(residual):.1f} points different from the "
            f"market, on the {lean_side} side."
        )
        story = (
            f'<p class="prose" style="margin-top:6px;">{escape(explanation)}</p>'
            if explanation
            else '<p class="fine" style="margin-top:6px;">No per-game attribution '
            "artifact is saved for this card yet (`nfl-ats market-decomposition`).</p>"
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

    is_best_pick = best_pick_id is not None and game_id == best_pick_id
    if is_best_pick:
        accent = "border-left:3px solid var(--good);"
        best_badge = (
            '<p class="kicker" style="color:var(--good-text);font-weight:700;">'
            "&#9733; BEST PICK OF THE WEEK</p>"
        )
    else:
        accent = "border-left:3px solid var(--series-model);" if strong else ""
        best_badge = ""
    cards.append(
        f'<div class="card" style="{accent}margin-top:14px;">'
        + best_badge
        # -- header row: matchup + kickoff + market line
        + '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;'
        'gap:8px;align-items:baseline;">'
        f'<div><p class="kicker">{escape(_kickoff_words(row))}</p>'
        f'<h3 class="title" style="font-size:19px;">{escape(away)} at {escape(home)}</h3>'
        f'<p class="sub">Market: <span class="num">'
        f"{escape(_spread_words(home, away, market_spread))}</span></p></div>"
        f'<div style="text-align:right;"><p class="kicker">Our pick</p>'
        f'<div class="hero num" style="font-size:26px;color:var(--series-model);">'
        f"{escape(pick_team)}</div></div></div>"
        # -- the three views
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

sections.append('<div class="ats">' + "".join(cards) + "</div>")
sections.append(
    '<div class="ats"><p class="fine" style="margin:16px 0 4px;">'
    f"Card generated {escape(artifact_time(forecast.directory))} · "
    f"method {escape(ats_method)} · lines are home-oriented nflverse spreads at "
    "card-build time; the pool's exact number can differ by a half point.</p></div>"
)
# One merged script tag: the sanitizer keeps only one <script> per st.html
# block, so the sweep interaction rides inside the theme-sync tag.
sections.append(theme.theme_sync_script(extra_js=viz.interaction_js()))

st.html("".join(sections), unsafe_allow_javascript=True)
