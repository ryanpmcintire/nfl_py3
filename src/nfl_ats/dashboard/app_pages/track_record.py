"""Track record: how often the picks actually landed, against two different lines.

The pool this project targets grades against a spread frozen early in the
week, so the opening-line grade leads and the closing-line grade follows as
context. Every number is anchored to
:func:`nfl_ats.dashboard.state.project_state` -- the single definition of
"current" -- and when that anchor reports staleness the page says so out loud
next to the affected number instead of quietly swapping in a different one.

Plain English is a standing rule here: no "Brier", no "bootstrap", no
"confidence interval". The page says "how often the picks landed" and "the
range the true skill could plausibly sit in", because those are the sentences
the owner actually reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

import streamlit as st

from nfl_ats.dashboard import theme, viz
from nfl_ats.dashboard.data import artifacts_root, describe_artifact_source
from nfl_ats.dashboard.state import load_csv, load_parquet, project_state

# ---------------------------------------------------------------------------
# Small local helpers (formatting only -- all data comes from ProjectState)
# ---------------------------------------------------------------------------


def _html(inner: str) -> None:
    """Render one composed fragment inside the scoped ``.ats`` root."""

    st.html(f'<div class="ats">{inner}</div>', unsafe_allow_javascript=True)


def _number(value: Any) -> float | None:
    """Coerce an artifact field to a float, or ``None`` if it is absent/unusable."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(source: Any, key: str) -> dict[str, Any]:
    value = source.get(key) if isinstance(source, dict) else None
    return value if isinstance(value, dict) else {}


def _commands_as_code(text: str) -> str:
    """Escape a plain-English status string, rendering its `backticked` bits as code."""

    parts = text.split("`")
    return "".join(
        f'<code style="font-family:monospace;">{escape(part)}</code>' if index % 2 else escape(part)
        for index, part in enumerate(parts)
    )


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


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{escape(name)}</th>" for name in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        '<div style="overflow-x:auto;">'
        f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


# ---------------------------------------------------------------------------
# The anchor
# ---------------------------------------------------------------------------

state = project_state()
root = artifacts_root()
opener = state.opener_evaluation
opener_metrics = _mapping(opener.metadata, "metrics")
opener_accuracy = _number(opener_metrics.get("opener_accuracy"))
close_accuracy = _number(opener_metrics.get("close_accuracy"))
opener_games = _number(opener.metadata.get("games"))

st.html(theme.stylesheet())

_html(
    viz.page_header(
        "Track record",
        "How often the picks actually landed",
        "Graded against two different lines. The one the pool uses comes first.",
    )
)

# ---------------------------------------------------------------------------
# 1. Hero row -- the pool number, the sharper number, the model's own record
# ---------------------------------------------------------------------------

tiles: list[str] = []

if opener_accuracy is None:
    tiles.append(
        viz.empty_state(
            "The pool grade has not been measured yet",
            "The pool freezes its spread early in the week, so the opening-line grade is the "
            "number that decides it. Run `nfl-ats opener-evaluation` to measure it.",
        )
    )
else:
    games_text = f"{int(opener_games):,} games" if opener_games else "every paired game"
    tiles.append(
        viz.stat_tile(
            "Against the pool's line",
            f"{opener_accuracy:.1%}",
            "How often the forced picks landed against the spread frozen early in the week -- "
            f"the line the pool actually grades. Measured on {games_text} from 2020-2025 that "
            "the model never trained on.",
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

historical = _mapping(state.active, "historical_evaluation")
model_accuracy = _number(historical.get("accuracy"))
if state.active is None or model_accuracy is None:
    tiles.append(
        viz.card(
            '<p class="kicker">The model\'s own long-run record</p>'
            '<div class="hero num">--</div>'
            '<p class="fine" style="margin-top:6px;">No active model is linked yet, so there is '
            "no long-run record to quote. The engine room lists what to run.</p>"
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
            f"{int(model_correct):,} correct out of {int(model_games):,} games it was tested "
            "on but never trained on. The range beside it is the honest one: this is a single "
            "sample of seasons, not a promise about the next one.",
            delta_text=delta_text,
            delta_good=None,
        )
    )

_html(f'<div class="row">{"".join(tiles)}</div>')

if opener.directory is not None:
    _html(
        '<p class="fine">'
        + _commands_as_code(describe_artifact_source(opener.directory, root))
        + "</p>"
    )

# ---------------------------------------------------------------------------
# 2. Staleness, said out loud next to the number -- never hidden
# ---------------------------------------------------------------------------

if opener.consistent_with_active is False:
    stale = next(
        (issue for issue in state.issues if "opener-evaluation" in issue),
        "The opening-line grade was measured on an earlier build of the model's inputs "
        "than the one behind this week's picks. Re-run `nfl-ats opener-evaluation` to "
        "refresh it.",
    )
    _html(
        viz.card(
            viz.status_line(
                "warning", "This grade and this week's picks come from different builds"
            )
            + f'<p class="sub" style="margin-top:8px;">{_commands_as_code(stale)}</p>'
            + '<p class="fine" style="margin-top:8px;">The number stays on the page because it '
            "is a real measurement -- it simply describes a slightly earlier build of the "
            "inputs than the card on the picks page does. Re-running the grade is what makes "
            "the two agree again.</p>"
        )
    )

# ---------------------------------------------------------------------------
# 3. Season by season -- including the seasons that lost
# ---------------------------------------------------------------------------

season_rows: list[tuple[str, float]] = []
season_cells: list[list[str]] = []
games_by_season: dict[str, float] = {}

if opener.directory is not None:
    seasons = load_csv(opener.directory / "season_summary.csv")
    if not seasons.empty and {"season", "opener_accuracy"}.issubset(seasons.columns):
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

if season_rows:
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
    _html(
        viz.card(
            '<p class="kicker">Season by season, against the frozen line</p>'
            '<p class="title" style="margin-bottom:12px;">No season is left off</p>'
            + viz.season_bars(season_rows)
            + f'<p class="sub" style="margin-top:12px;">{escape(honesty)}</p>'
            + '<details class="table-view"><summary>View as table</summary>'
            + _table(["Season", "Games", "Against the opener", "Against the close"], season_cells)
            + "</details>"
        )
    )

# ---------------------------------------------------------------------------
# 4. Paper picks -- did the line move our way after we picked?
# ---------------------------------------------------------------------------

ledger_directory = state.clv_ledger.directory

if ledger_directory is None:
    _html(
        viz.empty_state(
            "No paper picks recorded yet",
            "Every published card's picks get written down at the exact line they were "
            "published at. Once a week's games close, this fills in with how the line moved "
            "afterwards. Nothing has been published yet, which is normal in the preseason.",
        )
    )
else:
    ledger = load_parquet(ledger_directory / "scored_decisions.parquet")
    recorded = len(ledger)
    scored = (
        ledger.loc[ledger["clv_status"].eq("scored")]
        if "clv_status" in ledger.columns
        else ledger.iloc[0:0]
    )
    pending = recorded - len(scored)
    intro = (
        '<p class="kicker">Paper picks</p>'
        '<p class="title">Did the line move our way after we picked?</p>'
        '<p class="sub" style="margin-top:6px;">This is a check that does not need results. '
        "Every published pick is written down at the line it was published at; if the market "
        "later drifts toward the side we already took, we were early to something real. It is "
        "measured against the closing line, so treat it as a health check on the model rather "
        "than a pool score.</p>"
    )
    if recorded == 0:
        _html(
            viz.card(
                intro
                + '<p class="fine" style="margin-top:10px;">The ledger exists but has no picks '
                "in it yet.</p>"
            )
        )
    elif scored.empty or "clv_points" not in scored.columns:
        _html(
            viz.card(
                intro + f'<p class="sub" style="margin-top:10px;"><b>{recorded:,}</b> '
                f"pick{'s are' if recorded != 1 else ' is'} recorded and waiting for the games "
                "to close. The first numbers land here once a recorded week finishes.</p>"
            )
        )
    else:
        moved_our_way = float(scored["clv_points"].gt(0.0).mean())
        average_move = float(scored["clv_points"].mean())
        direction = "toward" if average_move >= 0 else "away from"
        _html(
            viz.card(
                intro
                + '<div class="row" style="margin-top:14px;">'
                + viz.stat_tile(
                    "The market moved our way",
                    f"{moved_our_way:.0%}",
                    "Share of scored picks where the closing line ended up better than the "
                    "line we picked at.",
                    delta_text=f"{len(scored):,} picks scored",
                    delta_good=None,
                )
                + viz.stat_tile(
                    "Average move after we picked",
                    f"{abs(average_move):.2f} pts",
                    f"On average the line drifted {abs(average_move):.2f} points "
                    f"{direction} the side we took.",
                    delta_text=("in our favour" if average_move >= 0 else "against us"),
                    delta_good=average_move >= 0,
                )
                + viz.stat_tile(
                    "Still waiting",
                    f"{pending:,}",
                    "Recorded picks whose games have not closed yet.",
                )
                + "</div>"
                + '<p class="fine" style="margin-top:12px;">'
                + _commands_as_code(describe_artifact_source(ledger_directory, root))
                + "</p>"
            )
        )

# ---------------------------------------------------------------------------
# 5. How to read this honestly (the ceiling: docs/pool_edge_plan.md)
# ---------------------------------------------------------------------------

pool_range = _season_interval(opener.metadata.get("uncertainty"), "opener_accuracy")
range_sentence = (
    f"The pool grade's honest range runs from about {pool_range[0]:.1%} to "
    f"{pool_range[1]:.1%} across seasons."
    if pool_range
    else "Every headline here is the middle of a range, not a fixed value."
)

_html(
    viz.card(
        '<p class="kicker">How to read this honestly</p>'
        '<p class="title" style="margin-bottom:10px;">Four things that keep these numbers '
        "honest</p>"
        '<div class="prose">'
        "<p><b>Nothing here was picked after seeing the answer.</b> Every number is measured on "
        "games the model never trained on, with the recipe frozen before the games were "
        "scored.</p>"
        f"<p><b>A single number is the middle of a range.</b> {escape(range_sentence)} The "
        "headline is the best single guess; the range is where the true skill could plausibly "
        "sit.</p>"
        "<p><b>One look, once.</b> Each headline came from a single planned measurement of a "
        "frozen model. Re-running a test until it finally looks good is the fastest way to "
        "fool yourself, so we do not do it.</p>"
        "<p><b>52-53% against a frozen line is genuinely good.</b> Realistically excellent is "
        "54-55%. Someone who knew everything knowable before kickoff would top out near 57-58% "
        "against a frozen Tuesday line, because football itself is noisy -- final margins "
        "scatter about 13.5 points around even a perfect expectation. If this page ever shows "
        "60%, that is a bug to hunt, not a breakthrough.</p>"
        "</div>"
        '<p class="fine" style="margin-top:10px;">The ceiling arithmetic behind that last '
        "paragraph is written up in docs/pool_edge_plan.md.</p>"
    )
)

st.html(theme.theme_sync_script(), unsafe_allow_javascript=True)
