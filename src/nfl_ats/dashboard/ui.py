"""Small, reusable rendering helpers shared across pages.

Kept deliberately native: progress bars, badges, and metrics rather than
custom HTML/CSS, per the project's Streamlit design guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import streamlit as st

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
    return "PK" if line == 0 else f"{line:+g}"


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
            crossing = float(lines[index] + fraction * (lines[index + 1] - lines[index]))
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


def render_confidence_ribbon(ribbon: pd.DataFrame) -> None:
    """A compact, color-graded strip of the picked side's cover probability.

    One column per swept line: cell shading runs on a semantic scale
    centered at 50% (washed out below, increasingly saturated green above),
    the quoted market line's column is bracketed, and integer-line columns
    carrying non-trivial push mass get a trailing subtle dot. Renders as a
    single-row styled dataframe, which scrolls inside its own container
    when a card is too narrow to show every line at once rather than
    pushing the page wide.
    """

    if ribbon.empty:
        return

    headers: list[str] = []
    cell_values: list[int] = []
    cell_styles: list[str] = []
    for _, row in ribbon.iterrows():
        label = str(row["label"])
        if bool(row["is_market"]):
            label = f"[{label}]"
        if (
            bool(row["is_integer"])
            and float(row["push_probability"]) > _RIBBON_PUSH_MARKER_THRESHOLD
        ):
            label = f"{label}·"
        headers.append(label)
        cell_values.append(round(float(row["probability"]) * 100))
        cell_styles.append(_ribbon_cell_style(float(row["probability"])))

    wide = pd.DataFrame([cell_values], columns=headers)
    styled = wide.style.apply(lambda _row: cell_styles, axis=1)
    st.dataframe(styled, hide_index=True, width="stretch", height=70)


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
) -> str:
    """The compact "line journey" row: opener -> latest -> predicted close -> fair line.

    All of ``opener``, ``latest``, and ``predicted_close`` are home-spread
    values (matching :func:`line_sweep_ribbon`'s ``perspective="home"``, the
    same convention ``fair`` should have been computed in); any of them may
    be ``None`` when that source has no data yet, which renders as an em
    dash. Market movement between opener and latest is subtly color-cued
    using Streamlit's native ``:color[...]`` markdown syntax: green when the
    market moved toward the model's fair line, orange when it moved away.
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

    return f"Open {open_text} → Now {latest_text} → Close (pred) {close_text} → Fair {fair.label}"


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


def honesty_note() -> None:
    """The mandated honesty caveat: historical accuracy is not proof of an edge."""

    st.warning(
        "**52% historical accuracy is not proof of a profitable edge.** It is close to a "
        "coin flip, sportsbook vig alone would erase it, and the uncertainty range below "
        "still touches 50% in some views. Treat every pick here as a lean, not a promise.",
        icon=":material/warning:",
    )
