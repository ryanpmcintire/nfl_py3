"""Pool workbench -- everything to check before Tuesday's lock, in one place.

Five things the owner has to look at each week before the pool freezes its
picks, none of which get a page of their own elsewhere:

1. **Submission status** -- is this week still tracking the live forecast, or
   has it already been written to the paper-decision ledger? If it has, does
   the locked record still match what the live forecast would pick today?
   Reads :func:`nfl_ats.clv.load_paper_decisions` live, every rerun -- this
   page never assumes a fixed answer about what the ledger currently holds.
2. **The rules**, as the owner has confirmed them (``docs/pool_edge_plan.md``):
   forced picks on every game, one Best Pick a week, a Tuesday-noon lock. The
   pool's own Best Pick bonus size is not recorded anywhere in this repo
   (``nfl_ats.pool.PoolFormat.best_pick_bonus``'s own docstring says so), so
   this page says that plainly rather than showing a number nobody gave it.
3. **Entries ranked by confidence** -- the same forced-pick pool card the
   public site's picks page (``docs/index.html``, built by
   ``nfl_ats.public_board``) shows as per-game narrative cards, here as a
   compact table for fast scanning before submission.
4. **The Best Pick nomination and its tie disclosure** -- via
   :func:`nfl_ats.dashboard.data.resolve_best_pick`, a thin wrapper over the
   same :mod:`nfl_ats.card_view` resolution ``nfl_ats.public_board`` calls
   directly, so the two surfaces can never show contradictory nominations.
5. **What the pool's own format is worth** -- a plain-English read of the
   field-size/prize-structure simulation in
   ``artifacts/pool_levers/levers.json`` (``scripts/pool_levers.py``,
   POL-05). This is a simulation of how entrants are assumed to behave,
   never real per-game ownership data -- true ownership is structurally
   unavailable before the Tuesday lock (POL-04, closed) -- and the page says
   so every time it renders the chart.

The midweek line-movement section ("what changed since Tuesday open") is
context only, never a decision input: the pool's picks are due before that
information exists (``docs/pool_edge_plan.md``), so late-picking cannot use
it. The banner above that table says this in plain language, not just in a
caption.

Everything here is read fresh, the same way every other page reads through
:mod:`nfl_ats.dashboard.state`.

VOID BOUNDARY: this page never calls ``nfl_ats.decision_rule``'s
``evaluate_candidate``/``fit_empirical_prior``/``model_average`` or quotes any
of their historical output -- that module's arithmetic is owner-voided
(``docs/decision_rule.md``). The "declining to use a signal is itself a bet
that it is worth zero" framing below is stated independently, the same way
``AGENTS.md`` states it, not sourced from that module.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from nfl_ats.clv import load_paper_decisions
from nfl_ats.dashboard import theme, viz
from nfl_ats.dashboard.data import (
    artifact_time,
    artifacts_root,
    data_root,
    display_overlay,
    list_recent_market_snapshots,
    load_live_quotes,
    load_pool_levers,
    resolve_best_pick,
)
from nfl_ats.dashboard.state import load_parquet, project_state
from nfl_ats.market_data import spread_consensus, tuesday_opener_quotes
from nfl_ats.pool import build_ats_pool_card

# A "strong lean" is a fair-line disagreement with the market of at least this
# many points. It gates only the strong-lean badge on the entries table below,
# never the pick itself: every pool pick is forced, and our confidence
# ordering has NOT proven predictive, so the lean is a narrative marker, never
# a weighting.
STRONG_LEAN_POINTS = 1.5

# The pool's own bonus for a correct Best Pick is not recorded anywhere in
# this repo (nfl_ats.pool.PoolFormat.best_pick_bonus's own docstring says so
# explicitly) -- every place this page would otherwise show a bonus number,
# it shows this sentence instead. No invented number.
BONUS_UNKNOWN_NOTE = "bonus size not configured — the pool's rules aren't recorded here."

WEEK_LABELS: dict[str, str] = {
    "WC": "Wild Card round",
    "DIV": "Divisional round",
    "CON": "Conference championships",
    "SB": "Super Bowl",
}


def _num(value: Any) -> float | None:
    """Coerce an artifact cell to a float, treating NaN the same as missing."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # NaN != NaN


def _close(value: Any, target: float, tolerance: float = 1e-6) -> bool:
    number = _num(value)
    return number is not None and abs(number - target) < tolerance


def _market_favorite_words(home: str, away: str, home_spread_line: float) -> str:
    """'DEN -3.5' style, from a raw market quote's home-oriented spread.

    This reads ``nfl_ats.market_data``'s ``home_spread_line`` (the odds API's
    raw home-team point, industry-standard -- negative means the home team is
    favored). The feature table's ``spread_line``/``fair_spread`` is the
    opposite convention on purpose (README.md: "a positive `spread_line`
    means the home team is favored" -- margin.py's predicted home margin), so
    this helper's sign is deliberately NOT interchangeable with a value read
    from that table.
    """

    if math.isnan(home_spread_line) or home_spread_line == 0:
        return "pick 'em"
    favorite, points = (
        (home, -home_spread_line) if home_spread_line < 0 else (away, home_spread_line)
    )
    return f"{favorite} -{points:g}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(name)}</th>" for name in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        '<div style="overflow-x:auto;margin-top:10px;">'
        f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


# ---------------------------------------------------------------------------
# The anchor
# ---------------------------------------------------------------------------

state = project_state()

if state.forecast is None:
    st.html(
        theme.stylesheet()
        + '<div class="ats">'
        + viz.page_header("Pool workbench", "No pick card yet")
        + viz.empty_state(
            "Nothing has been generated since the last kickoff",
            "Once the Tuesday opening lines are captured and a forecast card is built "
            "(`nfl-ats margin-predict`), this page fills in by itself. This week's picks "
            "page is open in the meantime.",
        )
        + "</div>"
        + theme.theme_sync_script()
    )
    st.stop()

forecast = state.forecast
raw_recommendations = forecast.recommendations.copy()
metadata = forecast.metadata
season_raw = metadata.get("season", "?")
week_raw = metadata.get("week", "?")
game_type = (
    str(raw_recommendations["game_type"].iloc[0])
    if "game_type" in raw_recommendations and not raw_recommendations.empty
    else "REG"
)
is_regular_season = game_type == "REG"
week_label = WEEK_LABELS.get(game_type, f"Week {week_raw}")

try:
    season_number: int | None = int(season_raw)
except (TypeError, ValueError):
    season_number = None
try:
    week_number: int | None = int(week_raw)
except (TypeError, ValueError):
    week_number = None

sweep = load_parquet(forecast.directory / "line_sweep.parquet")
ats_method = str(metadata.get("ats_method", "market_residual"))
if not sweep.empty and "method" in sweep:
    sweep = sweep.loc[sweep["method"].eq(ats_method)]

# What actually gets submitted: the coach-fade overlay flips a handful of
# picks post-prediction (docs/coach_fade_overlay.md). The entries table, Best
# Pick and market-movement sections below read this overlaid frame so they
# never disagree with the card already published. Section 2 (submission
# status) deliberately compares against raw_recommendations instead --
# nfl_ats.clv.record_paper_decisions records the model's OWN, un-overlaid
# pick_side (the overlay is tracked separately, in its own challenger
# ledger), so overlaying that comparison would misreport an ordinary
# overlay-flipped week as "the model changed after lock."
overlay = display_overlay(raw_recommendations, data_root())
recommendations = overlay.overlaid_predictions

sections: list[str] = [theme.stylesheet()]

# ---------------------------------------------------------------------------
# 1. Header + sync/issue chips
# ---------------------------------------------------------------------------

sync_chip = (
    viz.status_line("good", "Synchronized with the active model")
    if state.synchronized
    else viz.status_line("warning", "Not the active model's card -- see the engine room")
)
header = viz.page_header(
    f"{season_raw} · {week_label} · {len(recommendations)} games",
    "Pool workbench",
    "One page to check before Tuesday's lock: the submission status, the pool's own rules, "
    "every game ranked by confidence, the Best Pick nomination, and what the pool's format "
    "itself is worth.",
)
issue_cards = "".join(
    f'<div class="card" style="border-left:3px solid var(--serious);">'
    f"{viz.status_line('warning', issue)}</div>"
    for issue in state.issues
)
sections.append(
    '<div class="ats">'
    + header
    + f'<div style="margin:-6px 0 14px;">{sync_chip}</div>'
    + issue_cards
    + "</div>"
)

# ---------------------------------------------------------------------------
# 2. Submission status -- reads the paper-decision ledger LIVE, every rerun.
# ---------------------------------------------------------------------------

ledger = load_paper_decisions(artifacts_root())
week_rows = pd.DataFrame(columns=ledger.columns)
if not ledger.empty and season_number is not None and week_number is not None:
    week_rows = ledger.loc[ledger["season"].eq(season_number) & ledger["week"].eq(week_number)]

mismatches: list[str] = []
if ledger.empty:
    status_kind, status_word = "good", "Not yet recorded"
    status_detail = (
        "No paper picks have been recorded yet, for any week. The first genuine write "
        "happens at the Tuesday lock (`nfl-ats publish-predictions`) -- this is expected "
        "before then, not a problem."
    )
elif week_rows.empty:
    status_kind, status_word = "good", "Live"
    status_detail = (
        "Nothing has been recorded for this week specifically yet. This card is still "
        "tracking the active forecast and will change if the forecast changes, right up "
        "until the Tuesday lock writes it down."
    )
else:
    forced_side = {
        str(row["game_id"]): ("HOME" if float(row["home_cover_probability"]) >= 0.5 else "AWAY")
        for _, row in raw_recommendations.iterrows()
        if "home_cover_probability" in row and pd.notna(row.get("home_cover_probability"))
    }
    team_names = {
        str(row["game_id"]): (str(row["away_team"]), str(row["home_team"]))
        for _, row in raw_recommendations.iterrows()
    }
    for _, row in week_rows.iterrows():
        game_id = str(row["game_id"])
        recorded_side = str(row.get("pick_side", ""))
        live_side = forced_side.get(game_id)
        if live_side is not None and recorded_side != live_side:
            away, home = team_names.get(game_id, ("?", "?"))
            mismatches.append(f"{away} at {home}")
    if mismatches:
        status_kind, status_word = "warning", "Locked -- differs from the live forecast"
        status_detail = (
            f"{len(mismatches)} of {len(week_rows)} locked picks no longer match what the "
            "live forecast would pick today: " + ", ".join(mismatches) + ". That means the "
            "model changed after the lock -- the locked record is what was actually "
            "submitted and is not rewritten."
        )
    else:
        status_kind, status_word = "good", "Locked -- matches the live forecast"
        status_detail = (
            f"All {len(week_rows)} picks recorded for this week match what the live "
            "forecast would pick today. This is the record of what was actually submitted."
        )

locked_count = f"{len(week_rows)} of {len(recommendations)}" if not week_rows.empty else "0"
tiles = [
    viz.stat_tile(
        "This week",
        f"{season_raw} · {week_label}",
        f"{len(recommendations)} games on the card"
        + (", one Best Pick" if is_regular_season else " -- no Best Pick in the postseason"),
    ),
    viz.stat_tile("Submission status", status_word, status_detail),
    viz.stat_tile(
        "Locked picks",
        locked_count,
        "How many of this week's games already have a recorded paper decision.",
    ),
]
sections.append(
    '<div class="ats">'
    + viz.card(viz.status_line(status_kind, status_word))
    + f'<div class="row" style="margin-top:14px;">{"".join(tiles)}</div>'
    + "</div>"
)

# ---------------------------------------------------------------------------
# 3. The rules card -- static prose, sourced from docs/pool_edge_plan.md.
# ---------------------------------------------------------------------------

bonus_sentence = escape(BONUS_UNKNOWN_NOTE)
rules_card = viz.card(
    '<p class="kicker">The rules</p>'
    '<p class="title" style="margin-bottom:10px;">What this pool actually pays for</p>'
    '<div class="prose">'
    "<p>Every game gets a forced pick against the spread -- <b>272 regular-season games "
    "plus 13 playoff games</b>, no passes.</p>"
    "<p>The pool awards <b>one Best Pick each regular-season week</b> -- a "
    "confidence-weighted selection. There is no Best Pick in the postseason.</p>"
    "<p><b>Picks lock Tuesday at noon</b> -- essentially at the opener. The pool posts its "
    "own line Tuesday, revises once Wednesday, then freezes for the week: half-point "
    "numbers, identical for every entrant. A commissioner can override a line by hand, so "
    "the pool's exact number can differ from ours by half a point.</p>"
    "<p>A correct Best Pick earns extra points on top of the point every correct pick "
    f"earns -- but the {bonus_sentence}</p>"
    "</div>"
)
sections.append('<div class="ats">' + rules_card + "</div>")

# ---------------------------------------------------------------------------
# 4. Entries, ranked by confidence
# ---------------------------------------------------------------------------

if not is_regular_season:
    entries_html = viz.empty_state(
        "No playoff model coverage yet",
        "The feature pipeline does not produce postseason rows yet, so there is no "
        "forced-pick card to rank for the playoffs. The pool still needs 13 playoff picks; "
        "extending the model to serve them is tracked in docs/postseason_support.md.",
    )
else:
    try:
        card = build_ats_pool_card(recommendations)
    except ValueError:
        card = pd.DataFrame()
    if card.empty:
        entries_html = viz.empty_state(
            "No pool card could be built",
            "This week's forecast is missing a column the pool card needs.",
        )
    else:
        residual_by_game = (
            {
                str(game_id): _num(residual)
                for game_id, residual in zip(
                    recommendations["game_id"],
                    recommendations["predicted_market_residual"],
                    strict=True,
                )
            }
            if "predicted_market_residual" in recommendations
            else {}
        )
        entry_rows: list[list[str]] = []
        for _, row in card.iterrows():
            game_id = str(row["game_id"])
            residual = residual_by_game.get(game_id)
            strong = residual is not None and abs(residual) >= STRONG_LEAN_POINTS
            badge = '<span class="chip model">strong lean</span>' if strong else ""
            entry_rows.append(
                [
                    str(int(row["confidence_rank"])),
                    escape(str(pd.to_datetime(row["gameday"]).date())),
                    escape(f"{row['away_team']} at {row['home_team']}"),
                    escape(str(row["pool_pick"])),
                    f"{float(row['pick_line']):+.1f}",
                    f"{float(row['pick_probability']):.0%}",
                    f"{float(row['confidence']):.0%}",
                    badge,
                ]
            )
        entries_html = viz.card(
            '<p class="kicker">Entries, ranked by confidence</p>'
            '<p class="title" style="margin-bottom:6px;">Every forced pick, fastest to scan</p>'
            '<p class="sub">Confidence rank is a sort key, not a validated predictor -- the '
            "weekly top-|residual| pick scored 48.6% over 107 weeks. No pick here earns "
            "extra weight for being ranked higher.</p>"
            + _table(
                ["#", "Date", "Matchup", "Pick", "Line", "Chance", "Confidence", ""], entry_rows
            )
        )
sections.append('<div class="ats">' + entries_html + "</div>")

# ---------------------------------------------------------------------------
# 5. Best Pick nomination + tie disclosure
# ---------------------------------------------------------------------------

best_pick = (
    resolve_best_pick(raw_recommendations, sweep, metadata, data_root())
    if is_regular_season
    else None
)
best_pick_id = best_pick.game_id if best_pick is not None else None

if not is_regular_season:
    best_pick_html = viz.empty_state(
        "No Best Pick in the postseason",
        "The pool only awards a Best Pick during the regular season, and the model has no "
        "playoff coverage yet either way.",
    )
elif best_pick is None or best_pick_id is None:
    best_pick_html = viz.empty_state(
        "No Best Pick nomination yet",
        "This card predates the line-sweep artifact the nomination needs "
        "(`nfl-ats margin-predict`).",
    )
else:
    best_row = recommendations.loc[recommendations["game_id"].astype(str).eq(best_pick_id)]
    best_probability = float(best_row.iloc[0]["home_cover_probability"])
    best_team = (
        str(best_row.iloc[0]["home_team"])
        if best_probability >= 0.5
        else str(best_row.iloc[0]["away_team"])
    )
    if best_pick.active_rule == "v2":
        sub_text = f"This pick was {best_pick.method_note}"
    else:
        tie_note = f" {best_pick.tie_note}" if best_pick.tie_note else ""
        sub_text = (
            "The pick whose edge holds up across the widest range of line movement -- the "
            "best-measured lever among forced choices, budgeted at roughly +0.9 points, not "
            f"the +8.7 once recorded before a tie-break audit.{tie_note}"
        )
    best_pick_html = viz.card(
        '<p class="kicker" style="color:var(--good-text);font-weight:700;">'
        "&#9733; BEST PICK OF THE WEEK</p>"
        '<div class="hero num" style="font-size:26px;color:var(--good-text);">'
        f"{escape(best_team)}</div>"
        f'<p class="sub">{escape(sub_text)}</p>',
        accent=True,
    )
sections.append('<div class="ats">' + best_pick_html + "</div>")

# ---------------------------------------------------------------------------
# 6. What changed since Tuesday open -- context only, never a decision input.
# ---------------------------------------------------------------------------

snapshots = list_recent_market_snapshots(data_root() / "market" / "raw")
quotes = (
    load_live_quotes(tuple(info.directory for info in snapshots)) if snapshots else pd.DataFrame()
)

tuesday_frame = pd.DataFrame()
latest_frame = pd.DataFrame()
if not quotes.empty:
    try:
        tuesday_frame = tuesday_opener_quotes(quotes)
    except ValueError:
        tuesday_frame = pd.DataFrame()
    try:
        # spread_consensus (unlike tuesday_opener_quotes) does not validate its
        # required columns up front -- a quotes frame that never carries
        # commence_time_utc/provider_event_id (an older-shaped archive) raises
        # a raw KeyError deep inside it, not a clean ValueError.
        latest_frame = spread_consensus(quotes)
    except (ValueError, KeyError):
        latest_frame = pd.DataFrame()

movement_rows: list[list[str]] = []
if not tuesday_frame.empty and not latest_frame.empty:
    current_game_ids = set(recommendations["game_id"].astype(str))
    merged = tuesday_frame[["nflverse_game_id", "opener_home_spread"]].merge(
        latest_frame[["nflverse_game_id", "consensus_home_spread"]],
        on="nflverse_game_id",
        how="inner",
    )
    merged = merged.loc[merged["nflverse_game_id"].astype(str).isin(current_game_ids)].copy()
    team_names = {
        str(row["game_id"]): (str(row["away_team"]), str(row["home_team"]))
        for _, row in recommendations.iterrows()
    }
    merged["delta"] = merged["consensus_home_spread"] - merged["opener_home_spread"]
    merged = merged.sort_values("delta", key=lambda column: column.abs(), ascending=False)
    for _, row in merged.iterrows():
        game_id = str(row["nflverse_game_id"])
        away, home = team_names.get(game_id, ("?", "?"))
        opener_value = float(row["opener_home_spread"])
        latest_value = float(row["consensus_home_spread"])
        movement_rows.append(
            [
                escape(f"{away} at {home}"),
                escape(_market_favorite_words(home, away, opener_value)),
                escape(_market_favorite_words(home, away, latest_value)),
                f"{latest_value - opener_value:+.1f}",
            ]
        )

if not movement_rows:
    movement_html = viz.empty_state(
        "Nothing has moved to compare yet",
        "This fills in once a Tuesday-captured line and a later quote both exist for at "
        "least one of this week's games.",
    )
else:
    movement_html = viz.card(
        '<p class="kicker">What changed since Tuesday open</p>'
        '<p class="title" style="margin-bottom:6px;">Context only -- not a decision input</p>'
        + viz.status_line(
            "good",
            "No action needed: this pool locks at the opener, so later movement doesn't "
            "change what you submit.",
        )
        + '<p class="fine" style="margin-top:8px;">Picks are due before Wednesday-Friday '
        "injury news even exists, so nothing below is harvestable for pool points -- it is "
        "shown to see whether the market agrees with where we already are.</p>"
        + _table(["Matchup", "Tuesday open", "Latest", "Change (pts)"], movement_rows)
        + '<p class="fine" style="margin-top:8px;">Positive change means the line moved '
        "toward the home team; lines are home-oriented nflverse spreads.</p>"
    )
sections.append('<div class="ats">' + movement_html + "</div>")

# ---------------------------------------------------------------------------
# 7. Decision levers -- POL-05's field-size/prize-structure simulation.
# ---------------------------------------------------------------------------

levers = load_pool_levers(artifacts_root())
levers_path = artifacts_root() / "pool_levers" / "levers.json"

if levers is None:
    levers_html = viz.empty_state(
        "No pool-format simulation saved yet",
        "Run `.\\.tools\\uv.exe run --no-sync python scripts/pool_levers.py` to build one. "
        "It is pure arithmetic conditional on a measured accuracy -- no rotation window, "
        "no new data.",
    )
else:
    baseline_accuracy = _num(levers.get("baseline_accuracy")) or 0.525

    tier_pairs: list[tuple[float, float]] = []
    accuracy_rows = levers.get("accuracy_scaling")
    if isinstance(accuracy_rows, list):
        for row in accuracy_rows:
            if not isinstance(row, dict) or not _close(row.get("entrants"), 100.0):
                continue
            accuracy = _num(row.get("accuracy"))
            probability = _num(row.get("probability_first"))
            if accuracy is not None and probability is not None:
                tier_pairs.append((accuracy, probability))
    tier_pairs.sort(key=lambda item: item[0])

    bars = [(f"{accuracy:.1%}", probability) for accuracy, probability in tier_pairs]
    reference_tier = min(tier_pairs, key=lambda item: abs(item[0] - 0.5)) if tier_pairs else None
    reference = reference_tier[1] if reference_tier is not None else 0.0
    chart_html = (
        viz.season_bars(bars, reference=reference, reference_label="at a coin flip")
        if bars
        else viz.empty_state("No accuracy-scaling table saved", "This artifact predates it.")
    )

    # Field-size multiplier: model vs. coin-flip P(first) at each simulated field size.
    by_entrants: dict[float, dict[str, float]] = {}
    edge_rows = levers.get("edge_value")
    if isinstance(edge_rows, list):
        for row in edge_rows:
            if not isinstance(row, dict):
                continue
            entrants = _num(row.get("entrants"))
            probability = _num(row.get("probability_first"))
            entry = row.get("entry")
            if entrants is None or probability is None or entry not in ("model", "coin_flip"):
                continue
            by_entrants.setdefault(entrants, {})[str(entry)] = probability
    multipliers = [
        (entrants, values["model"] / values["coin_flip"])
        for entrants, values in by_entrants.items()
        if "model" in values and "coin_flip" in values and values["coin_flip"] > 0
    ]

    # Two accuracy points, same field size (100 rivals) -- the honest comparison
    # to the Best Pick ranker's own value, computed the same way just below.
    two_point_delta_pp: float | None = None
    if tier_pairs:
        baseline_tier = min(tier_pairs, key=lambda item: abs(item[0] - baseline_accuracy))
        target_tier = min(tier_pairs, key=lambda item: abs(item[0] - (baseline_accuracy + 0.02)))
        if abs(target_tier[0] - (baseline_accuracy + 0.02)) < 0.006:
            two_point_delta_pp = (target_tier[1] - baseline_tier[1]) * 100

    # The Best Pick ranker's own honest value, same field size and bonus scale.
    honest_probability: float | None = None
    arbitrary_probability: float | None = None
    best_pick_rows = levers.get("best_pick")
    if isinstance(best_pick_rows, list):
        for row in best_pick_rows:
            if not isinstance(row, dict):
                continue
            if not (_close(row.get("best_pick_bonus"), 1.0) and _close(row.get("entrants"), 100.0)):
                continue
            if (
                row.get("ranker_effect") == "tie_break_agnostic_+0.9pts"
                and row.get("strategy") == "ranker_nominates_the_high_game"
            ):
                honest_probability = _num(row.get("probability_first"))
            if row.get("ranker_effect") == "none" and row.get("strategy") == "arbitrary_nomination":
                arbitrary_probability = _num(row.get("probability_first"))
    ranker_delta_pp = (
        (honest_probability - arbitrary_probability) * 100
        if honest_probability is not None and arbitrary_probability is not None
        else None
    )

    multiplier_sentence = ""
    if multipliers:
        low_entrants, low_multiple = min(multipliers, key=lambda item: item[1])
        high_entrants, high_multiple = max(multipliers, key=lambda item: item[1])
        multiplier_sentence = (
            "<p>The format alone already turns our measured edge into a "
            f"<b>{low_multiple:.1f}&times; to {high_multiple:.1f}&times;</b> better chance "
            "of finishing first than a coin-flip entry would get, just from the field size "
            f"-- {int(low_entrants)} rivals sees the smaller multiple, "
            f"{int(high_entrants)} rivals the larger one. Every lever below is small next "
            "to that.</p>"
        )

    accuracy_sentence = ""
    if two_point_delta_pp is not None:
        ranker_clause = (
            f", versus the honest Best Pick ranker's own <b>+{ranker_delta_pp:.2f} "
            "points</b> at the same field size and a placeholder 1-point bonus"
            if ranker_delta_pp is not None
            else ""
        )
        accuracy_sentence = (
            "<p>Accuracy is the real lever, not the contest rules: two more points of "
            "accuracy at a 100-rival pool is worth about "
            f"<b>+{two_point_delta_pp:.1f} percentage points</b> of first-place chance"
            f"{ranker_clause}. The format is a multiplier to protect, not a lever to "
            "pull.</p>"
        )

    differentiation_sentence = (
        "<p>Deliberately taking a worse side to differentiate from the field loses "
        "monotonically at every field size once it costs real accuracy, and is neutral at "
        f"best under a typical top-15%-payout structure. The Best Pick bonus scenarios "
        f"above assume a placeholder value, not a real one -- {bonus_sentence}</p>"
    )

    levers_html = viz.card(
        '<p class="kicker">Decision levers</p>'
        '<p class="title" style="margin-bottom:6px;">What the pool\'s own format is '
        "worth</p>"
        '<p class="sub" style="margin-bottom:12px;">A simulation of how a field of '
        "entrants who resemble each other is assumed to behave -- never real per-game "
        "ownership data, which is structurally unavailable before the Tuesday lock. "
        "Labeled as simulation throughout, on purpose.</p>"
        + chart_html
        + '<div class="prose" style="margin-top:14px;">'
        + multiplier_sentence
        + accuracy_sentence
        + differentiation_sentence
        + "</div>"
    )
sections.append('<div class="ats">' + levers_html + "</div>")

# ---------------------------------------------------------------------------
# 8. Footer / provenance
# ---------------------------------------------------------------------------

footer_bits = [
    f"Card generated {escape(artifact_time(forecast.directory))} · method {escape(ats_method)}"
]
if levers is not None and levers_path.is_file():
    stamp = datetime.fromtimestamp(levers_path.stat().st_mtime, tz=UTC)
    footer_bits.append(
        f"pool-format simulation built {escape(stamp.strftime('%Y-%m-%d %H:%M UTC'))}"
    )
sections.append(
    '<div class="ats"><p class="fine" style="margin:16px 0 4px;">'
    + " · ".join(footer_bits)
    + "</p></div>"
)

sections.append(theme.theme_sync_script())

st.html("".join(sections), unsafe_allow_javascript=True)
