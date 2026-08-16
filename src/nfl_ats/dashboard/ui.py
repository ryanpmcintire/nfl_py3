"""Small, reusable rendering helpers shared across pages.

Kept deliberately native where native suffices: progress bars, badges, and
metrics rather than custom HTML/CSS, per the project's Streamlit design
guidance. The one deliberate exception is the line-sweep confidence ribbon
(:func:`render_confidence_ribbon`): it needs every swept line to fit a card's
width with no inner scrolling regardless of the card's actual pixel size, a
layout ``st.dataframe``/``column_config`` cannot express, so it renders a
small self-contained HTML table via ``st.html`` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import streamlit as st

from nfl_ats.dashboard.data import ArtifactSelection, artifact_time

# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------

# Bands are defined on the *picked side's* probability (always >= 0.50).
# They are deliberately conservative: the model's long-run historical edge is
# small (about 52% since 2018), so "strong lean" should be reserved for the
# minority of games where the estimate separates the most from a coin flip.
CONFIDENCE_TIERS: tuple[tuple[float, str, str, str], ...] = (
    # (lower bound, label, band text, semantic color)
    (0.57, "Strong lean", "57%+", "green"),
    (0.52, "Lean", "52-57%", "blue"),
    (0.0, "Coin flip", "50-52%", "gray"),
)


def confidence_tier(pick_probability: float) -> tuple[str, str, str]:
    """Return (label, band text, color) for a picked-side probability."""

    for lower, label, band, color in CONFIDENCE_TIERS:
        if pick_probability >= lower:
            return label, band, color
    return "Coin flip", "50-52%", "gray"


def tier_badge(pick_probability: float) -> None:
    label, band, color = confidence_tier(pick_probability)
    st.badge(f"{label} ({band})", color=color)  # type: ignore[arg-type]


def confidence_meter(pick_probability: float) -> None:
    """A visual meter, not just a decimal: a progress bar plus the tier badge."""

    label, band, _color = confidence_tier(pick_probability)
    st.progress(
        min(max(pick_probability, 0.0), 1.0),
        text=f"{pick_probability:.0%} confidence · {label} band ({band})",
    )


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def render_value(value: Any, *, percent: bool = False, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if percent:
        return f"{float(value):.1%}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def metric_with_context(
    label: str,
    value: Any,
    *,
    percent: bool = False,
    digits: int = 4,
    delta: str | None = None,
    help_text: str | None = None,
    border: bool = False,
) -> None:
    """A metric that always carries a delta, range, or comparison alongside the number."""

    st.metric(
        label,
        render_value(value, percent=percent, digits=digits),
        delta=delta,
        delta_color="off",
        help=help_text,
        border=border,
    )


def favorite_label(row: pd.Series) -> str:
    spread = float(row["spread_line"])
    if spread > 0:
        return f"{row['home_team']} -{spread:g}"
    if spread < 0:
        return f"{row['away_team']} -{abs(spread):g}"
    return "Pick'em"


def pick_side_and_line(row: pd.Series) -> tuple[str, float, float]:
    """Return (picked team, picked side's spread, picked side's probability)."""

    home_probability = float(row["home_cover_probability"])
    home_pick = home_probability >= 0.5
    team = str(row["home_team"] if home_pick else row["away_team"])
    line = -float(row["spread_line"]) if home_pick else float(row["spread_line"])
    probability = home_probability if home_pick else 1.0 - home_probability
    return team, line, probability


def spread_label(line: float) -> str:
    # Exact half points render as the bettor-conventional vulgar fraction
    # ("-7½") so every ribbon header stays at most four glyphs wide and
    # survives the fixed 17-column layout without clipping. Anything off the
    # half-point grid (fair lines like +3.2) keeps decimal formatting.
    if line == 0:
        return "PK"
    if abs(abs(line) % 1.0 - 0.5) < 1e-9:
        whole = int(line - 0.5) if line > 0 else int(line + 0.5)
        if whole == 0:
            return "+½" if line > 0 else "-½"
        return f"{whole:+d}½"
    return f"{line:+g}"


# ---------------------------------------------------------------------------
# Line-sweep confidence ribbon
# ---------------------------------------------------------------------------

# Only integer lines can mathematically push (a real final margin is always
# an integer), so this only ever fires on integer-line cells. It also
# filters out numerically negligible push mass so the marker stays a
# meaningful signal rather than firing on every integer line by default.
_RIBBON_PUSH_MARKER_THRESHOLD = 0.005

# Semantic color scale, centered at 50%: a near-white midpoint fades toward
# a washed-out neutral below 50% and saturates toward green above it. The
# green matches the app's existing "Strong lean" tier color family.
_RIBBON_MID_RGB = (250, 250, 248)
_RIBBON_LOW_RGB = (222, 221, 214)
_RIBBON_HIGH_RGB = (12, 163, 12)

# The quoted market line's column gets this border color instead of a tier
# color -- amber reads as "marker", not as another confidence signal, and
# stays legible against both the washed-neutral and saturated-green cell
# fills the gradient above can produce.
_RIBBON_MARKET_BORDER = "#c9860a"
_RIBBON_HEADER_BORDER = "1px solid rgba(128, 128, 128, 0.45)"


def line_sweep_ribbon(
    game: pd.Series, game_sweep: pd.DataFrame, *, perspective: Literal["pick", "home"] = "pick"
) -> pd.DataFrame:
    """Reorient one game's line-sweep rows into a per-line confidence table.

    ``game_sweep`` is the slice of ``MarginModel.line_sweep()`` output (see
    ``nfl_ats.margin``) for a single ``game_id`` -- one row per swept line,
    home-oriented (``alternative_line`` is a home spread, ``home_cover_probability``
    is the home side's cover probability at that line).

    ``perspective="pick"`` (the default) reorients every line to the
    *picked* side, using exactly :func:`pick_side_and_line`'s sign
    convention: a home pick negates ``alternative_line`` (so the picked
    side's spread reads the same way ``pick_side_and_line`` reads the
    quoted line) and keeps ``home_cover_probability`` as-is; an away pick
    keeps the line unchanged and reads ``1 - home_cover_probability``. This
    is what the confidence ribbon and fair-line callout show: one fixed
    side's confidence, scanned across every swept line.

    ``perspective="home"`` instead reports every line and probability in
    the sweep's own home-spread convention, unconverted. The line journey
    row uses this: it mixes in home-spread market quotes (opener/latest
    consensus, predicted close) that have no "pick" side of their own, so
    everything in that row needs to share one convention.

    Returns a tidy per-line frame sorted ascending by ``line``, with
    columns: ``line`` (float), ``label`` (str, ``spread_label(line)``),
    ``probability`` (float in [0, 1], the oriented side's cover
    probability), ``push_probability`` (float in [0, 1]), ``is_integer``
    (bool), and ``is_market`` (bool, True for the row at the quoted line).
    """

    if perspective not in ("pick", "home"):
        raise ValueError(f"Unknown ribbon perspective: {perspective!r}")

    # Defensive: a legitimate sweep has exactly one row per line_offset for a
    # given game and method. Duplicates would otherwise collide into
    # non-unique ribbon lines/labels, which pandas Styler refuses outright.
    working = (
        game_sweep.drop_duplicates(subset="line_offset")
        .sort_values("line_offset")
        .reset_index(drop=True)
    )
    alternative_line = working["alternative_line"].astype(float)
    home_cover_probability = working["home_cover_probability"].astype(float)

    if perspective == "home":
        line = alternative_line
        probability = home_cover_probability
    else:
        team, _market_line, _market_probability = pick_side_and_line(game)
        home_pick = team == game["home_team"]
        if home_pick:
            line = -alternative_line
            probability = home_cover_probability
        else:
            line = alternative_line
            probability = 1.0 - home_cover_probability

    ribbon = pd.DataFrame(
        {
            "line": line.astype(float),
            "probability": probability.astype(float),
            "push_probability": working["push_probability"].astype(float),
            "is_market": working["line_offset"].round(4).eq(0.0),
        }
    )
    ribbon["is_integer"] = np.isclose(ribbon["line"] % 1.0, 0.0, atol=1e-9)
    ribbon["label"] = ribbon["line"].map(spread_label)
    return ribbon.sort_values("line").reset_index(drop=True)


@dataclass(frozen=True)
class FairLine:
    """The model's implied fair line: where a ribbon's probability crosses 50%.

    ``value`` is ``None`` when the crossing lies outside the swept window --
    ``label`` then reads as a clamp (``"< -6.5"`` / ``"> +7.5"``) against the
    swept range's actual boundary rather than extrapolating past data the
    model was never asked about.
    """

    value: float | None
    label: str


def implied_fair_line(ribbon: pd.DataFrame) -> FairLine:
    """Interpolate a ribbon for the line where its probability crosses 50%.

    Works on either :func:`line_sweep_ribbon` perspective: it only assumes
    probability is roughly monotone in ``line`` (true by construction -- a
    fixed predictive distribution's cover probability only moves one
    direction as the line it is compared against moves further from the
    distribution's center) and reports the first adjacent pair of swept
    lines that straddle 50%, linearly interpolated between them.

    The interpolated crossing is rounded to one decimal before it becomes a
    label -- linear interpolation between arbitrary swept lines otherwise
    produces long, uninterpretable decimals (e.g. ``-3.80952``) instead of
    a readable line.
    """

    if ribbon.empty:
        return FairLine(None, "—")

    ordered = ribbon.sort_values("line").reset_index(drop=True)
    lines = ordered["line"].to_numpy(dtype=float)
    probabilities = ordered["probability"].to_numpy(dtype=float)

    for index in range(len(lines) - 1):
        lower_probability = float(probabilities[index])
        upper_probability = float(probabilities[index + 1])
        if lower_probability == 0.5:
            crossing = float(lines[index])
            return FairLine(crossing, spread_label(crossing))
        crosses_up = lower_probability < 0.5 < upper_probability
        crosses_down = lower_probability > 0.5 > upper_probability
        if crosses_up or crosses_down:
            fraction = (0.5 - lower_probability) / (upper_probability - lower_probability)
            crossing = round(float(lines[index] + fraction * (lines[index + 1] - lines[index])), 1)
            return FairLine(crossing, spread_label(crossing))

    if float(probabilities[-1]) == 0.5:
        crossing = float(lines[-1])
        return FairLine(crossing, spread_label(crossing))
    if float(probabilities[0]) >= 0.5:
        # The pick is favored even at the toughest swept line -- the true
        # breakeven point is tougher than anything the sweep covers.
        return FairLine(None, f"< {spread_label(float(lines[0]))}")
    # The pick is an underdog even at the most favorable swept line -- the
    # true breakeven point is more generous than anything the sweep covers.
    return FairLine(None, f"> {spread_label(float(lines[-1]))}")


def canonical_fair_line(game: pd.Series, home_ribbon: pd.DataFrame) -> FairLine:
    """The card's single source of truth for "the model's fair line" (home-oriented).

    Prefers ``predicted_market_residual`` -- the margin model's actual,
    exact output -- over the line-sweep's 50%-crossing interpolation
    (:func:`implied_fair_line`). Both estimate the same quantity: the
    model's fair home spread is ``spread_line + predicted_market_residual``
    by construction (see ``MarginModel._predicted_margin`` in
    ``nfl_ats.margin``), which is exactly the home spread where the
    sweep's cover probability should cross 50%. But the sweep only
    interpolates between discretely swept lines while the residual is
    exact, so the two used to disagree by more than rounding and show as
    two near-duplicate numbers on one card (e.g. a sweep-implied fair line
    of 0.3 pt next to a residual-based lean of 0.1 pt for the same game).
    Preferring the residual keeps the card down to one number. The sweep
    interpolation is used only as a fallback when no residual is available
    at all (e.g. a legacy card that predates the column).
    """

    residual = game.get("predicted_market_residual")
    spread_line = game.get("spread_line")
    if (
        residual is not None
        and pd.notna(residual)
        and spread_line is not None
        and pd.notna(spread_line)
    ):
        value = round(float(spread_line) + float(residual), 1)
        return FairLine(value, spread_label(value))
    return implied_fair_line(home_ribbon)


def fair_line_gap(line: float, home_fair: FairLine, *, home_pick: bool) -> float | None:
    """The gap between the offered pick-side line and the model's fair line.

    ``line`` is the picked side's quoted spread, in :func:`pick_side_and_line`'s
    convention. ``home_fair`` is home-oriented (see :func:`canonical_fair_line`);
    this projects it into the same picked-side convention before comparing,
    using exactly the sign flip :func:`line_sweep_ribbon`'s
    ``perspective="pick"`` uses (negate for a home pick, keep as-is for an
    away pick).

    Positive means the offered line is *more generous* than fair for this
    side -- the bettor is getting a better number than the model thinks is
    warranted, a good sign. Negative means the offered line is *tougher*
    than fair -- a caution sign. ``None`` when the fair line has no concrete
    value (the sweep-fallback clamp case, i.e. the true fair line lies
    outside the swept window).
    """

    if home_fair.value is None:
        return None
    pick_fair_value = -home_fair.value if home_pick else home_fair.value
    return round(line - pick_fair_value, 1)


def format_value_framing(gap: float | None) -> str:
    """Bettor-facing wording for :func:`fair_line_gap`, with a subtle color cue.

    A positive gap (the offered line is better than fair) gets a soft green
    cue; a negative gap (tougher than fair) gets an orange caution cue --
    matching :func:`format_line_journey`'s existing ``:green:``/``:orange:``
    convention for market movement, via Streamlit's native colored-markdown
    syntax.
    """

    if gap is None:
        return "fair line beyond the swept window"
    if gap > 0:
        return f":green[getting {gap:.1f} pt better than fair]"
    if gap < 0:
        return f":orange[{abs(gap):.1f} pt worse than fair]"
    return "right at fair"


def _ribbon_cell_style(probability: float) -> str:
    """Background + text color for one ribbon cell, graded around 50%."""

    clipped = min(max(float(probability), 0.0), 1.0)
    if clipped >= 0.5:
        fraction = min((clipped - 0.5) / 0.30, 1.0)
        start, end = _RIBBON_MID_RGB, _RIBBON_HIGH_RGB
    else:
        fraction = min((0.5 - clipped) / 0.50, 1.0)
        start, end = _RIBBON_MID_RGB, _RIBBON_LOW_RGB
    rgb = tuple(round(s + (e - s) * fraction) for s, e in zip(start, end, strict=True))
    luma = 0.30 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
    text_color = "#0b0b0b" if luma > 140 else "#ffffff"
    return f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); color: {text_color}"


def render_confidence_ribbon(ribbon: pd.DataFrame, team: str) -> None:
    """A compact, color-graded strip of the picked side's cover probability.

    One column per swept line: cell shading runs on a semantic scale
    centered at 50% (washed out below, increasingly saturated green above),
    and integer-line columns carrying non-trivial push mass get a trailing
    subtle dot. The quoted market line's column is unmistakable: it carries
    both a bracketed label (``[+3.5]``) and a solid amber border running
    through both rows.

    Renders as one self-contained HTML table (``st.html``) rather than a
    styled dataframe. A dataframe scrolls inside its own container when a
    card is too narrow for every column, which used to leave the initial
    view scrolled to one end of the swept range -- often *not* including
    the market line, the one column that matters most. This table instead
    uses percentage-based fixed column widths (``table-layout: fixed``), so
    every swept line always fits the card at once, with no inner
    scrollbar, regardless of the card's actual pixel width.
    """

    if ribbon.empty:
        return

    header_cells: list[str] = []
    body_cells: list[str] = []
    for _, row in ribbon.iterrows():
        label = str(row["label"])
        is_market = bool(row["is_market"])
        if is_market:
            label = f"[{label}]"
        if (
            bool(row["is_integer"])
            and float(row["push_probability"]) > _RIBBON_PUSH_MARKER_THRESHOLD
        ):
            label = f"{label}·"
        border = (
            f"border-left:2px solid {_RIBBON_MARKET_BORDER};"
            f"border-right:2px solid {_RIBBON_MARKET_BORDER};"
        )
        header_cells.append(
            '<td style="padding:2px 1px;text-align:center;font-size:9px;font-weight:600;'
            "white-space:nowrap;overflow:hidden;"
            f"border-bottom:{_RIBBON_HEADER_BORDER};"
            + (f"border-top:2px solid {_RIBBON_MARKET_BORDER};{border}" if is_market else "")
            + f'">{label}</td>'
        )
        probability = float(row["probability"])
        pct = round(probability * 100)
        body_cells.append(
            '<td style="padding:2px 1px;text-align:center;font-size:10px;font-weight:600;'
            "white-space:nowrap;overflow:hidden;"
            f"{_ribbon_cell_style(probability)};"
            + (f"border-bottom:2px solid {_RIBBON_MARKET_BORDER};{border}" if is_market else "")
            + f'">{pct}%</td>'
        )

    table = (
        '<table style="width:100%;table-layout:fixed;border-collapse:collapse;'
        'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin:2px 0 0 0;">'
        f"<tr>{''.join(header_cells)}</tr>"
        f"<tr>{''.join(body_cells)}</tr>"
        "</table>"
    )
    st.html(table)
    st.caption(
        f"Chance {team} covers if the line were the number shown; boxed column "
        "[in brackets] is today's actual market line."
    )


def line_movement_signal(
    opener: float | None, latest: float | None, fair_value: float | None
) -> Literal["toward", "away", "flat", "unknown"]:
    """Whether the market moved toward or away from the model's fair line.

    All three values must share one sign convention (the caller decides
    which -- home-oriented, matching :func:`line_sweep_ribbon`'s
    ``perspective="home"``). Returns ``"unknown"`` whenever any input is
    missing, since there is nothing to compare.
    """

    if opener is None or latest is None or fair_value is None:
        return "unknown"
    if np.isclose(latest, opener, atol=1e-9):
        return "flat"
    opener_gap = abs(opener - fair_value)
    latest_gap = abs(latest - fair_value)
    return "toward" if latest_gap < opener_gap else "away"


def format_line_journey(
    opener: float | None,
    latest: float | None,
    predicted_close: float | None,
    fair: FairLine,
    framing: str | None = None,
) -> str:
    """The compact "line journey" row: opener -> latest -> predicted close -> fair line.

    All of ``opener``, ``latest``, and ``predicted_close`` are home-spread
    values (matching :func:`line_sweep_ribbon`'s ``perspective="home"``, the
    same convention ``fair`` should have been computed in); any of them may
    be ``None`` when that source has no data yet, which renders as an em
    dash. Market movement between opener and latest is subtly color-cued
    using Streamlit's native ``:color[...]`` markdown syntax: green when the
    market moved toward the model's fair line, orange when it moved away.

    ``framing`` is the picked side's value read on the same fair line (see
    :func:`format_value_framing`), appended to the end of this one line
    rather than shown as a separate caption -- this is the card's single
    consolidated fair/journey line, not two.
    """

    open_text = spread_label(opener) if opener is not None else "—"
    close_text = spread_label(predicted_close) if predicted_close is not None else "—"

    if latest is None:
        latest_text = "—"
    else:
        raw_latest = spread_label(latest)
        signal = line_movement_signal(opener, latest, fair.value)
        if signal == "toward":
            latest_text = f":green[{raw_latest}]"
        elif signal == "away":
            latest_text = f":orange[{raw_latest}]"
        else:
            latest_text = raw_latest

    journey = (
        f"Open {open_text} → Now {latest_text} → Close (pred) {close_text} → Fair {fair.label}"
    )
    if framing:
        journey = f"{journey} · {framing}"
    return journey


def format_kickoff(row: pd.Series) -> str:
    """Human kickoff time, preferring the timezone-aware kickoff timestamp.

    Uses only portable strftime codes (no ``%-I``/``%#I``, which differ or
    fail across platforms) and strips a leading zero from the hour by hand.
    """

    raw_kickoff: Any = row.get("kickoff")
    kickoff = pd.to_datetime(raw_kickoff, errors="coerce", utc=True)
    if pd.notna(kickoff):
        local = kickoff.tz_convert("America/New_York")
        hour = local.strftime("%I").lstrip("0") or "0"
        return str(local.strftime(f"%a, %b %d · {hour}:%M %p ET"))
    raw_gameday: Any = row.get("gameday")
    gameday = pd.to_datetime(raw_gameday, errors="coerce")
    if pd.notna(gameday):
        gametime = row.get("gametime")
        suffix = f" · {gametime}" if isinstance(gametime, str) and gametime else ""
        return f"{gameday.strftime('%a, %b %d')}{suffix}"
    return "Kickoff time unknown"


def next_tuesday_capture_label(now_utc: pd.Timestamp | None = None) -> str:
    """Human label for the next Tuesday-morning opening-line capture.

    The live capture cadence anchors its weekly opener on Tuesday 9am Eastern
    (see ``odds_backfill.DECISION_TIMES``'s ``tue_open`` entry).
    """

    now = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    eastern_now = now.tz_convert("America/New_York")
    days_ahead = (1 - eastern_now.weekday()) % 7  # Monday=0 ... Tuesday=1
    if days_ahead == 0 and eastern_now.hour >= 9:
        days_ahead = 7
    target = (eastern_now + pd.Timedelta(days=days_ahead)).normalize() + pd.Timedelta(hours=9)
    hour = target.strftime("%I").lstrip("0") or "0"
    return target.strftime(f"%A, %B %d, around {hour}:%M %p ET")


# ---------------------------------------------------------------------------
# Research-page artifact selection: "The active model" vs. "Older research runs"
# ---------------------------------------------------------------------------


def render_research_run_picker(
    selection: ArtifactSelection,
    render: Callable[[Path], None],
    *,
    key: str,
    latest_label: str = "Latest run",
    active_label: str = "The active model",
    older_label: str = "Older research runs",
) -> None:
    """Render a featured saved run, plus the rest collapsed under an explicit heading.

    Fixes the dashboard's research-tab staleness complaint: a page used to just
    show "the latest artifact of its own type" (newest directory name), with no
    regard for whether that is the run actually backing "This week's picks."
    ``selection`` (see :func:`nfl_ats.dashboard.data.select_research_artifact`)
    already decides which run to feature -- the active model's linked run when
    one exists, else the newest -- so this only renders that decision: a
    subheader naming which case it is, the active-linkage badge when it applies,
    the featured run's own content via ``render``, and every other saved run
    (with its own date) collapsed under an "Older research runs" expander rather
    than silently unreachable. Renders nothing when ``selection.featured`` is
    ``None`` (no saved runs at all, or the active model's declared artifact is
    missing -- the caller handles that error message itself, explicitly, before
    calling this).
    """

    if selection.featured is None:
        return
    st.subheader(active_label if selection.featured_is_active else latest_label)
    if selection.featured_is_active:
        st.badge("Linked to the active model", color="green", icon=":material/verified:")
    render(selection.featured)
    if selection.older:
        with st.expander(f"{older_label} ({len(selection.older)})"):
            older_choice = st.selectbox(
                "Saved run", list(selection.older), format_func=artifact_time, key=f"{key}_older"
            )
            render(older_choice)


def honesty_note() -> None:
    """The mandated honesty caveat: historical accuracy is not proof of an edge."""

    st.warning(
        "**52% historical accuracy is not proof of a profitable edge.** It is close to a "
        "coin flip, sportsbook vig alone would erase it, and the uncertainty range below "
        "still touches 50% in some views. Treat every pick here as a lean, not a promise.",
        icon=":material/warning:",
    )
