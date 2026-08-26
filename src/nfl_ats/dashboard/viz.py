"""Self-contained HTML/CSS chart components for the site.

Every function returns an HTML fragment written against the role tokens in
:mod:`nfl_ats.dashboard.theme`. Pages compose fragments and prepend
``theme.stylesheet()`` once.

IMPORTANT platform constraint (verified live): the sanitizer of the original
embedded-HTML host stripped ``<svg>`` elements entirely (and inline ``on*``
handlers), so every chart here is pure HTML/CSS -- area and line shapes are
drawn with ``clip-path: polygon(...)``, markers and axes are positioned
``<div>``s, and interactivity is wired from a delegated ``<script>`` (see
:func:`cover_curve_script`). That host is retired -- the current, and only,
consumer of this module is the static public site (no sanitizer at all) --
but the discipline is kept anyway so this module stays safe to embed
anywhere again: do not reintroduce SVG.

Mark specs follow the dataviz method: 2px lines, >=8px markers, soft area
fills, recessive grid, selective direct labels, a table-view twin for every
chart, and identity never carried by color alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from html import escape
from itertools import pairwise
from typing import Any

# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------


def page_header(kicker: str, title: str, sub: str = "") -> str:
    """Page title block. The 24px title size ships as the ``page-title``
    class (sized by the host chrome) rather than an inline style, so a page
    can hold exactly ONE inline ``font-size:24px`` -- the picks page's
    crowned stat (see ``public_board._PAGE_CHROME``)."""

    sub_html = f'<p class="sub">{escape(sub)}</p>' if sub else ""
    return (
        f'<div style="margin: 4px 0 18px;"><p class="kicker">{escape(kicker)}</p>'
        # <h1>: every page opens at the top of the document outline (WCAG
        # 1.3.1 Info and Relationships); sizes still come from the classes.
        f'<h1 class="title page-title">{escape(title)}</h1>{sub_html}</div>'
    )


def p_plus_text(value: float) -> str:
    """Honest P+ display: never floor 0.995+ down to "1.00" (or <=0.005 up to
    "0.00") -- state the bound instead. Always shown adjacent to n."""

    if value >= 0.995:
        return ">0.99"
    if value <= 0.005:
        return "<0.01"
    return f"{value:.2f}"


def sweep_offset_label(offset: float) -> str:
    """One-decimal line-offset label; zero never wears a "+" sign."""

    text = f"{abs(offset):.1f}"
    if offset == 0:
        return text
    return f"{'+' if offset > 0 else '-'}{text}"


def card(inner: str, *, accent: bool = False) -> str:
    style = "border-left: 3px solid var(--series-model);" if accent else ""
    return f'<div class="card" style="{style}">{inner}</div>'


def stat_tile(
    kicker: str,
    value: str,
    context: str = "",
    *,
    delta_text: str | None = None,
    delta_good: bool | None = None,
) -> str:
    """Hero number with context; delta ships with a triangle glyph, not color alone."""

    delta_html = ""
    if delta_text is not None:
        if delta_good is None:
            color, arrow = "var(--ink-2)", ""
        elif delta_good:
            color, arrow = "var(--good-text)", "&#9650; "
        else:
            color, arrow = "var(--critical)", "&#9660; "
        delta_html = (
            f'<div style="font-size:13px;font-weight:600;color:{color};">'
            f"{arrow}{escape(delta_text)}</div>"
        )
    context_html = (
        f'<p class="fine" style="margin-top:6px;">{escape(context)}</p>' if context else ""
    )
    return (
        f'<div class="card"><p class="kicker">{escape(kicker)}</p>'
        f'<div class="hero num">{escape(value)}</div>{delta_html}{context_html}</div>'
    )


# Text glyphs inside bordered circles: SVG icons do not survive the sanitizer.
_STATUS_GLYPHS = {"good": "&#10003;", "warning": "!", "critical": "&#215;"}


def status_line(kind: str, text: str) -> str:
    """Status with a glyph badge + label; never color alone."""

    glyph = _STATUS_GLYPHS.get(kind, _STATUS_GLYPHS["warning"])
    badge = (
        '<span style="display:inline-flex;align-items:center;justify-content:center;'
        "width:15px;height:15px;border-radius:50%;border:1.5px solid currentColor;"
        'font-size:10px;font-weight:700;flex:none;line-height:1;">'
        f"{glyph}</span>"
    )
    return f'<span class="status {escape(kind)}">{badge}<span>{escape(text)}</span></span>'


def empty_state(title: str, body: str) -> str:
    return (
        '<div class="card" style="text-align:center;padding:34px 24px;">'
        f'<p class="title" style="margin-bottom:6px;">{escape(title)}</p>'
        f'<p class="sub" style="max-width:52ch;margin:0 auto;">{escape(body)}</p></div>'
    )


# ---------------------------------------------------------------------------
# Probability meter — calibrated cover probability vs. the coin flip
# ---------------------------------------------------------------------------


def probability_meter(probability: float, *, label: str, width: int = 240) -> str:
    """A same-ramp track from 50% outward; the coin-flip line is the anchor."""

    clamped = min(max(probability, 0.0), 1.0)
    fill_left = min(clamped, 0.5) * 100
    fill_width = abs(clamped - 0.5) * 100
    marker_left = clamped * 100
    return f"""
<div style="display:flex;flex-direction:column;gap:4px;max-width:{width}px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">
    <span class="fine">{escape(label)}</span>
    <span class="num" style="font-size:15px;font-weight:650;">{clamped:.0%}</span>
  </div>
  <div style="position:relative;height:10px;border-radius:5px;background:var(--grid);"
       role="img" aria-label="{escape(label)}: {clamped:.0%}">
    <div style="position:absolute;left:{fill_left:.1f}%;width:{fill_width:.1f}%;top:0;
                bottom:0;background:var(--seq-400);border-radius:5px;"></div>
    <div style="position:absolute;left:50%;top:-3px;bottom:-3px;width:1.5px;
                background:var(--baseline);"></div>
    <div style="position:absolute;left:{marker_left:.1f}%;top:50%;width:9px;height:9px;
                border-radius:50%;background:var(--seq-550);
                border:2px solid var(--surface);
                transform:translate(-50%,-50%);"></div>
  </div>
  <div style="position:relative;height:12px;">
    <span class="fine" style="position:absolute;left:50%;transform:translateX(-50%);
                 font-size:10px;">coin flip</span>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# The cover curve — confidence across alternative spreads, with an on-chart
# slider (the flagship chart)
#
# 2026-08-26 merge (owner): this single component REPLACES two things that
# used to answer the same question on their own -- the old ``sweep_curve``
# (a static S-curve with a hover-only tooltip) and the standalone
# spread-explorer slider (a separate ``<input type=range>`` with its own
# separate text readout, wired by ``public_board._spread_explorer_script``).
# The owner's complaint was exactly that duplication ("the Line sweep &
# explorer also shows basically the same thing as the spread explorer"), so
# there is now ONE curve, coloured, with the slider's handle riding on it.
# ---------------------------------------------------------------------------


def _polygon(points: Sequence[tuple[float, float]]) -> str:
    return ", ".join(f"{x:.2f}% {y:.2f}%" for x, y in points)


def _handle_tone(probability: float) -> tuple[str, str]:
    """(class name, fill token) for the draggable handle at ``probability``.

    The fill is set INLINE (by this function, and by the matching JS in
    :func:`cover_curve_script`) rather than through a CSS class rule --
    ``public_board._PAGE_CHROME`` requires every ``var(...)`` it references
    to be declared inside that same stylesheet, and ``--div-pos``/``--div-neg``/
    ``--div-mid`` are theme.py's own tokens, not page-chrome's, so a class
    rule referencing them there would use an undeclared variable. The class
    name still ships (for tests and any future CSS hook); it just is not the
    only place the colour is set.
    """

    if probability > 0.5:
        return "is-pos", "var(--div-pos)"
    if probability < 0.5:
        return "is-neg", "var(--div-neg)"
    return "is-mid", "var(--div-mid)"


def cover_curve(
    element_id: str,
    points: Sequence[tuple[float, float]],
    *,
    quoted_line: float,
    quote_label: str,
    pick_text: str,
    pick_team: str,
    anchor_probability: float,
    game_id: str,
    width: int = 540,
    plot_height: int = 140,
) -> str:
    """Pick-cover probability across alternative lines, with a draggable
    handle that rides the curve instead of a separate slider + readout.

    ``points`` is ``[(line_offset, pick_probability), ...]`` sorted by
    offset -- REAL swept model output wherever the caller has it (this
    project's research-integrity rule: never approximate what is already
    measured). ``anchor_probability`` is the pick's own PUBLISHED
    probability at ``quoted_line`` -- passed in directly rather than read
    back out of ``points``, so the fixed market marker can never drift from
    the number the rest of the page already shows, even if ``points`` came
    from a Gaussian-synthesized fallback (see :func:`cover_curve_script`).

    Colour carries the probability: a two-tone diverging area fill split
    exactly at the 50% coin-flip baseline (``--div-pos`` above it, the pick
    favoured; ``--div-neg`` below it, the pick against the odds --
    theme.py's validated diverging pair, centred on the meaningful
    midpoint). Colour is never the only cue -- the market's own line is a
    square marker with its own text label, the draggable handle is a circle
    with its own live percentage, and the collapsed table below states every
    plotted value in text. Drawn without SVG (see the module docstring):
    area and line shapes are ``clip-path`` polygons over the plot box, exactly
    like the retired ``sweep_curve`` this replaces. The drag handler itself
    ships once per page as :func:`cover_curve_script`.
    """

    if not points:
        return empty_state("No line data saved", "This card predates the line-sweep artifact.")

    ordered_points = sorted(points, key=lambda item: item[0])
    # Reconcile two estimators of the same quantity WITHOUT bending the curve.
    #
    # Measured on a real build (2026-08-26): a card's own
    # ``home_cover_probability`` differs from its OWN ``line_sweep`` row at the
    # same offset by ~1.8 points (0.5169 vs. 0.4989, 2026_01_NE_SEA). That is
    # not a defect -- ``margin.py`` documents the asymmetry deliberately: the
    # published probability comes from ``_smoothed_probability``'s continuous
    # test, while the sweep rows are raw empirical counts over the simulated
    # margin distribution. Two estimators, one quantity.
    #
    # Pinning ONLY the quoted-line point to ``anchor_probability`` (the first
    # attempt at this) put the marker back on its own value but tore that one
    # vertex ~1.8 points away from its neighbours, drawing a visible pinch in
    # the middle of an otherwise smooth curve (owner-reported, 2026-08-26).
    #
    # A rigid translation fixes both at once: shift the WHOLE series by the
    # gap measured at the quoted line. Shape, spacing and monotonicity are
    # preserved exactly, the marker lands on the curve by construction, and
    # every point stays the real swept value plus one constant. The shift is
    # zero when the two estimators already agree.
    quoted_probability = next(
        (p for offset, p in ordered_points if abs(offset - quoted_line) < 1e-9), None
    )
    if quoted_probability is not None:
        shift = anchor_probability - quoted_probability
        if shift:
            ordered_points = [
                (offset, min(1.0, max(0.0, p + shift))) for offset, p in ordered_points
            ]
    y_min, y_max = 0.25, 0.85
    xs = [offset for offset, _ in ordered_points]
    x_min, x_max = min(xs), max(xs)
    span = (x_max - x_min) or 1.0

    def x_pct(offset: float) -> float:
        return (offset - x_min) / span * 100.0

    def y_pct(probability: float) -> float:
        clamped = min(max(probability, y_min), y_max)
        return (y_max - clamped) / (y_max - y_min) * 100.0

    curve = [(x_pct(offset), y_pct(p)) for offset, p in ordered_points]
    baseline_pct = y_pct(0.5)

    # Two-tone diverging fill: clamp each point to the baseline from either
    # side, so each polygon collapses to a zero-height (invisible) sliver
    # wherever the curve is actually on the OTHER side of 50% -- the standard
    # bichromatic-area technique. No curve-crossing solver needed, and it
    # degrades safely even for a non-monotonic curve.
    above = [(x, min(y, baseline_pct)) for x, y in curve]
    below = [(x, max(y, baseline_pct)) for x, y in curve]
    area_pos_polygon = _polygon([*above, (100.0, baseline_pct), (0.0, baseline_pct)])
    area_neg_polygon = _polygon([*below, (100.0, baseline_pct), (0.0, baseline_pct)])
    # The curve's own edge stays one neutral tone (our model's reading) --
    # the two-tone fill beneath it is what carries the diverging encoding.
    line_half = 1.5 / plot_height * 100.0
    line_polygon = _polygon(
        [*[(x, y - line_half) for x, y in curve], *[(x, y + line_half) for x, y in reversed(curve)]]
    )

    # Gridlines are symmetric around the 50% baseline (a diverging chart's
    # meaningful centre), unlike the old asymmetric set.
    grid = "".join(
        f'<div style="position:absolute;left:0;right:0;top:{y_pct(level):.2f}%;height:1px;'
        'background:var(--grid);"></div>'
        for level in (0.3, 0.4, 0.6, 0.7)
        if y_min <= level <= y_max
    )
    # Walk OUTWARD from the quoted line (never from x_min) so the market's
    # own tick is always included, however narrow the domain -- a fixed
    # x_min-anchored stride could step clean over it (verified live: a
    # narrow +/-0.5 domain with the old x_min-anchored walk skipped offset 0
    # entirely). The domain's own edges are always added too, so the chart
    # never implies more range than it actually plots.
    tick_step = max(1.0, round(span / 4))
    tick_offsets = {round(quoted_line, 1), round(x_min, 1), round(x_max, 1)}
    outward = quoted_line + tick_step
    while outward < x_max - 1e-9:
        tick_offsets.add(round(outward, 1))
        outward += tick_step
    outward = quoted_line - tick_step
    while outward > x_min + 1e-9:
        tick_offsets.add(round(outward, 1))
        outward -= tick_step
    ticks = []
    for tick in sorted(tick_offsets):
        label = f"{tick:+g}" if abs(tick - quoted_line) > 1e-9 else escape(quote_label)
        ticks.append(
            f'<span class="fine num" style="position:absolute;left:{x_pct(tick):.2f}%;'
            f'transform:translateX(-50%);font-size:10px;">{label}</span>'
        )

    anchor_x, anchor_y = x_pct(quoted_line), y_pct(anchor_probability)
    # The slider can only land where this chart can actually answer: the
    # finest gap really present in the (real or synthesized) grid, not an
    # invented finer resolution.
    step = min((b - a for a, b in pairwise(xs)), default=0.5) or 0.5
    handle_tone, handle_fill = _handle_tone(anchor_probability)

    payload = escape(json.dumps([[round(offset, 1), round(p, 4)] for offset, p in ordered_points]))
    # The cover probability diverges around 50%: above it the pick is favoured
    # at that alternate line, below it the pick is against the odds. The
    # numeral is always legible, so colour is the second channel, and an exact
    # 50% reads neutral rather than being pushed to a side.
    table_rows = "".join(
        f"<tr><td>{sweep_offset_label(o)}</td>"
        f'<td><span class="delta {"pos" if p > 0.5 else "neg" if p < 0.5 else "zero"}">'
        f"{p:.1%}</span></td></tr>"
        for o, p in ordered_points
    )
    # Built as plain text first, escaped whole, so the 100-col limit does not
    # force a line break INSIDE the rendered attribute value.
    plot_aria_label = escape(
        f"{pick_text} across hypothetical lines near {quote_label}, "
        "with a slider below to explore others"
    )
    sentence = (
        f'At <b class="cover-line-words num">{escape(quote_label)}</b>, '
        f"<b>{escape(pick_team)}</b> covers "
        f'<span class="num cover-pct">{anchor_probability:.0%}</span>.'
    )

    return f"""
<div class="ats-cover" id="{escape(element_id)}" data-game-id="{escape(game_id)}"
     data-points="{payload}" data-xmin="{x_min}" data-xmax="{x_max}"
     data-ymin="{y_min}" data-ymax="{y_max}" style="max-width:{width}px;">
  <div class="plot" style="position:relative;height:{plot_height}px;"
       role="img" aria-label="{plot_aria_label}">
    {grid}
    <div style="position:absolute;left:0;right:0;top:{baseline_pct:.2f}%;height:0;
                border-top:1px dashed var(--baseline);"></div>
    <span class="fine" style="position:absolute;left:2px;top:{baseline_pct:.2f}%;
          transform:translateY(-115%);font-size:10px;">50% &middot; coin flip</span>
    <div style="position:absolute;inset:0;background:var(--div-pos);opacity:0.18;
                clip-path:polygon({area_pos_polygon});"></div>
    <div style="position:absolute;inset:0;background:var(--div-neg);opacity:0.18;
                clip-path:polygon({area_neg_polygon});"></div>
    <div style="position:absolute;inset:0;background:var(--series-model);
                clip-path:polygon({line_polygon});"></div>
    <div class="cover-market" title="The market's own line" style="position:absolute;
                left:{anchor_x:.2f}%;top:{anchor_y:.2f}%;width:10px;height:10px;
                border-radius:2px;background:var(--series-market);
                border:2px solid var(--surface);transform:translate(-50%,-50%);"></div>
    <div class="cover-handle {handle_tone}" style="position:absolute;left:{anchor_x:.2f}%;
                top:{anchor_y:.2f}%;width:13px;height:13px;border-radius:50%;
                background:{handle_fill};
                border:2px solid var(--surface);transform:translate(-50%,-50%);"></div>
  </div>
  <input type="range" class="cover-slider" min="{x_min:g}" max="{x_max:g}" step="{step:g}"
         value="{quoted_line:g}" aria-label="Hypothetical line for {escape(pick_text)}">
  <div style="position:relative;height:16px;margin-top:2px;">{"".join(ticks)}</div>
  <p class="sub cover-sentence" style="margin-top:6px;">{sentence}</p>
  <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:2px;">
    <span class="fine"><span style="color:var(--series-market);">&#9632;</span>
      market &middot; {escape(quote_label)}</span>
    <span class="fine"><span style="color:var(--div-pos);">&#9679;</span>
      hypothetical line (drag the slider)</span>
  </div>
  <details class="table-view"><summary>View as table</summary>
    <table class="data"><thead><tr><th>Line vs. quote</th><th>Confidence</th></tr></thead>
    <tbody>{table_rows}</tbody></table>
  </details>
</div>
"""


# ---------------------------------------------------------------------------
# Line journey — the market's number vs. ours, on one scale
# ---------------------------------------------------------------------------


def line_journey(
    *,
    opener: float | None,
    fair: float | None,
    predicted_close: float | None,
    opener_label: str = "opened",
    width: int = 250,
) -> str:
    """Opener, our fair line, and the predicted close on one number line.

    Identity is carried by color + shape + a labeled legend row (never color
    alone): market opener = orange square, our fair line = blue circle,
    predicted close = blue hollow circle.
    """

    values = [value for value in (opener, fair, predicted_close) if value is not None]
    if len(values) < 2:
        return '<p class="fine">Line comparison appears once two of the three numbers exist.</p>'
    low, high = min(values), max(values)
    center = (low + high) / 2
    half_span = max((high - low) / 2 + 0.75, 1.5)
    axis_low, axis_high = center - half_span, center + half_span

    def left_pct(value: float) -> float:
        return (value - axis_low) / (axis_high - axis_low) * 100.0

    marks, legend = [], []
    if opener is not None:
        marks.append(
            f'<div style="position:absolute;left:{left_pct(opener):.2f}%;top:50%;'
            "width:9px;height:9px;border-radius:2px;background:var(--series-market);"
            'transform:translate(-50%,-50%);"></div>'
        )
        legend.append(
            '<span class="fine"><span style="color:var(--series-market);">&#9632;</span> '
            f'{escape(opener_label)} <span class="num">{opener:+.1f}</span></span>'
        )
    if fair is not None:
        marks.append(
            f'<div style="position:absolute;left:{left_pct(fair):.2f}%;top:50%;'
            "width:10px;height:10px;border-radius:50%;background:var(--series-model);"
            'transform:translate(-50%,-50%);"></div>'
        )
        legend.append(
            '<span class="fine"><span style="color:var(--series-model);">&#9679;</span> '
            f'our number <span class="num">{fair:+.1f}</span></span>'
        )
    if predicted_close is not None:
        marks.append(
            f'<div style="position:absolute;left:{left_pct(predicted_close):.2f}%;top:50%;'
            "width:9px;height:9px;border-radius:50%;background:var(--surface);"
            "border:2px solid var(--series-model);"
            'transform:translate(-50%,-50%);"></div>'
        )
        legend.append(
            '<span class="fine"><span style="color:var(--series-model);">&#9675;</span> '
            f'close guess <span class="num">{predicted_close:+.1f}</span></span>'
        )
    return f"""
<div style="max-width:{width}px;">
  <div style="position:relative;height:22px;" role="img" aria-label="Line comparison">
    <div style="position:absolute;left:2px;right:2px;top:50%;height:1px;
                background:var(--baseline);"></div>
    {"".join(marks)}
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;">{"".join(legend)}</div>
</div>
"""


# ---------------------------------------------------------------------------
# Season bars — one thin bar per season, direct-labeled
# ---------------------------------------------------------------------------


def season_bars(
    rows: Sequence[tuple[str, float]],
    *,
    reference: float = 0.5,
    reference_label: str = "coin flip",
    width: int = 520,
    format_value: str = "{:.1%}",
) -> str:
    """Horizontal per-season bars anchored at a reference line."""

    if not rows:
        return empty_state("Nothing to chart yet", "This fills in once seasons are scored.")
    low = min(min(v for _, v in rows), reference) - 0.02
    high = max(max(v for _, v in rows), reference) + 0.02

    def pct(value: float) -> float:
        return (value - low) / (high - low) * 100.0

    reference_pct = pct(reference)
    bar_rows = []
    for label, value in rows:
        left, right = sorted((reference_pct, pct(value)))
        bar_rows.append(
            '<div style="display:flex;align-items:center;gap:10px;">'
            f'<span class="axis-label num" style="width:52px;text-align:right;'
            f'color:var(--muted);font-size:11px;flex:none;">{escape(label)}</span>'
            '<div style="position:relative;height:18px;flex:1;">'
            f'<div style="position:absolute;left:{left:.2f}%;width:{max(right - left, 0.4):.2f}%;'
            "top:0;bottom:0;border-radius:4px;background:var(--series-model);"
            f'opacity:{"1.0" if value >= reference else "0.55"};"></div>'
            f'<div style="position:absolute;left:{reference_pct:.2f}%;top:-2px;bottom:-2px;'
            'width:0;border-left:1.5px dashed var(--baseline);"></div>'
            f'<span class="num" style="position:absolute;left:{right + 1.2:.2f}%;top:50%;'
            "transform:translateY(-50%);font-size:11px;font-weight:600;"
            f'color:var(--ink-2);">{format_value.format(value)}</span>'
            "</div></div>"
        )
    return f"""
<div style="max-width:{width}px;display:flex;flex-direction:column;gap:8px;"
     role="img" aria-label="By season vs. {escape(reference_label)}">
  {"".join(bar_rows)}
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="width:52px;flex:none;"></span>
    <div style="position:relative;height:14px;flex:1;">
      <span class="fine" style="position:absolute;left:{reference_pct:.2f}%;
            transform:translateX(-50%);font-size:10px;">{escape(reference_label)}</span>
    </div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Delegated interaction layer (ship once per page): one drag handler for
# every ``.ats-cover`` widget, replacing the old hover-only crosshair
# (``interaction_script``/``interaction_js``, retired with ``sweep_curve``)
# and the standalone spread-explorer script it was paired with.
# ---------------------------------------------------------------------------


def cover_curve_script(payload: Mapping[str, Mapping[str, Any]]) -> str:
    """Wire every ``.ats-cover`` widget's slider to its handle, live sentence
    and percentage. ``payload`` is ``{game_id: {home, away, pick, pickIsHome,
    line, [center, mean, std]}}`` -- one shared JSON blob (built by the page,
    not this module, since it is the caller who knows each game's teams and
    Gaussian fit).

    Two probability sources at drag time, and this is which is which:

    * Games whose payload carries a Gaussian (``center``/``mean``/``std`` --
      present only when the active model's probability method has a
      closed-form read): the SAME erf approximation
      ``nfl_ats.spread_explorer.widget_home_cover_probability`` mirrors, and
      the build-time check in ``public_board._assert_spread_explorer_matches_card``
      proves reproduces the published card at the market line. Continuous,
      so the handle reads smoothly between the real swept points.
    * Every other game: linear interpolation across the widget's own
      ``data-points`` (the real swept sample :func:`cover_curve` already
      embedded to draw the curve) -- no formula, bounded by real
      neighbouring measurements.

    The FIXED market marker never uses either path -- it is drawn once, in
    Python, from the published card's own number, and this script never
    touches it.
    """

    if not payload:
        return ""
    data_json = json.dumps(payload, separators=(",", ":"))
    return (
        f'<script type="application/json" id="ats-cover-data">{data_json}</script>\n'
        "<script>\n" + _cover_curve_js() + "</script>\n"
    )


def _cover_curve_js() -> str:
    """The raw drag-handler JavaScript, for embedding into a shared script tag.

    Plain text only: any tag-like sequence in this script's SOURCE (even
    inside a string or a comment) made a previous embedded-HTML host's
    sanitizer drop the whole script element, verified live on that host.
    That host is retired, but the discipline is kept anyway so this module
    stays safe to embed anywhere again -- never write a less-than sign
    followed by a letter anywhere in this file's JS.
    """

    return """
(function () {
  if (window.__atsCoverWired) return;
  window.__atsCoverWired = true;
  var dataEl = document.getElementById("ats-cover-data");
  if (!dataEl) return;
  var data;
  try { data = JSON.parse(dataEl.textContent); } catch (err) { return; }

  function erf(x) {
    var sign = x < 0 ? -1 : 1; x = Math.abs(x);
    var a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    var t = 1 / (1 + p * x);
    var y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }
  function normalCdf(x, mean, std) {
    return 0.5 * (1 + erf((x - mean) / (std * Math.SQRT2)));
  }
  function homeCoverProbability(line, center, mean, std) {
    return 1 - normalCdf(line - center, mean, std);
  }
  function interpolate(points, offset) {
    if (offset <= points[0][0]) return points[0][1];
    var last = points[points.length - 1];
    if (offset >= last[0]) return last[1];
    for (var i = 1; i < points.length; i++) {
      if (offset <= points[i][0]) {
        var lo = points[i - 1], hi = points[i];
        var span = hi[0] - lo[0];
        var t = span === 0 ? 0 : (offset - lo[0]) / span;
        return lo[1] + t * (hi[1] - lo[1]);
      }
    }
    return last[1];
  }
  function spreadWords(home, away, value) {
    if (Math.abs(value) < 0.001) return "pick 'em";
    var favorite = value > 0 ? home : away;
    var points = Math.abs(value);
    var text = (points % 1 === 0) ? points.toFixed(0) : points.toFixed(1);
    return favorite + " -" + text;
  }
  function fmtPct(p) {
    return Math.round(Math.max(0, Math.min(1, p)) * 100) + "%";
  }
  function update(widget, game, points, geo, offset) {
    var pickProbability;
    if (typeof game.center === "number") {
      var homeLine = game.line + offset;
      var homeP = homeCoverProbability(homeLine, game.center, game.mean, game.std);
      pickProbability = game.pickIsHome ? homeP : 1 - homeP;
    } else {
      pickProbability = interpolate(points, offset);
    }
    var xPct = (offset - geo.xmin) / (geo.xmax - geo.xmin) * 100;
    var clamped = Math.max(geo.ymin, Math.min(geo.ymax, pickProbability));
    var yPct = (geo.ymax - clamped) / (geo.ymax - geo.ymin) * 100;
    var handle = widget.querySelector(".cover-handle");
    handle.style.left = xPct + "%";
    handle.style.top = yPct + "%";
    // The fill is set INLINE (mirroring cover_curve's own initial render),
    // not by the is-pos/is-neg/is-mid class alone -- those classes still
    // toggle, for anything that hooks off them, but the colour itself does
    // not depend on a stylesheet rule existing for them.
    var tone = pickProbability > 0.5 ? "is-pos" : pickProbability < 0.5 ? "is-neg" : "is-mid";
    var fills = {
      "is-pos": "var(--div-pos)", "is-neg": "var(--div-neg)", "is-mid": "var(--div-mid)"
    };
    handle.classList.remove("is-pos", "is-neg", "is-mid");
    handle.classList.add(tone);
    handle.style.background = fills[tone];
    var words = widget.querySelector(".cover-line-words");
    if (words) words.textContent = spreadWords(game.home, game.away, game.line + offset);
    var pct = widget.querySelector(".cover-pct");
    if (pct) pct.textContent = fmtPct(pickProbability);
  }
  var widgets = document.querySelectorAll(".ats-cover[data-game-id]");
  for (var i = 0; i < widgets.length; i++) {
    (function (widget) {
      var gameId = widget.getAttribute("data-game-id");
      var game = data[gameId];
      if (!game) return;
      var points = JSON.parse(widget.dataset.points);
      var geo = {
        xmin: +widget.dataset.xmin, xmax: +widget.dataset.xmax,
        ymin: +widget.dataset.ymin, ymax: +widget.dataset.ymax
      };
      var slider = widget.querySelector(".cover-slider");
      if (!slider) return;
      slider.addEventListener("input", function () {
        update(widget, game, points, geo, parseFloat(slider.value));
      });
    })(widgets[i]);
  }
})();
"""
