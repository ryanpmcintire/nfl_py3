"""Self-contained HTML/CSS chart components for the dashboard.

Every function returns an HTML fragment written against the role tokens in
:mod:`nfl_ats.dashboard.theme`. Pages compose fragments, prepend
``theme.stylesheet()`` once, and append ``theme.theme_sync_script()`` plus
:func:`interaction_script` once, then render through a single ``st.html``
call.

IMPORTANT platform constraint (verified live): Streamlit's ``st.html``
sanitizer strips ``<svg>`` elements entirely (and inline ``on*`` handlers),
so every chart here is pure HTML/CSS -- area and line shapes are drawn with
``clip-path: polygon(...)``, markers and axes are positioned ``<div>``s, and
hover behavior is wired from the delegated ``<script>`` in
:func:`interaction_script`. Do not reintroduce SVG.

Mark specs follow the dataviz method: 2px lines, >=8px markers, soft area
fills, recessive grid, selective direct labels, a table-view twin for every
chart, and identity never carried by color alone.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from html import escape

# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------


def page_header(kicker: str, title: str, sub: str = "") -> str:
    sub_html = f'<p class="sub">{escape(sub)}</p>' if sub else ""
    return (
        f'<div style="margin: 4px 0 18px;"><p class="kicker">{escape(kicker)}</p>'
        f'<h2 class="title" style="font-size:24px;">{escape(title)}</h2>{sub_html}</div>'
    )


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
# The line sweep — confidence across alternative spreads (the flagship chart)
# ---------------------------------------------------------------------------


def _polygon(points: Sequence[tuple[float, float]]) -> str:
    return ", ".join(f"{x:.2f}% {y:.2f}%" for x, y in points)


def sweep_curve(
    element_id: str,
    points: Sequence[tuple[float, float]],
    *,
    quoted_line: float,
    pick_text: str,
    quote_label: str = "at the quote",
    width: int = 540,
    plot_height: int = 130,
) -> str:
    """Pick-cover probability across alternative lines, +/-4 around the quote.

    ``points`` is ``[(line_offset, pick_probability), ...]`` sorted by offset.
    Drawn without SVG: the soft area fill and the 2px line are ``clip-path``
    polygons over the plot box (percent coordinates, so it scales
    responsively). A crosshair tooltip is wired by :func:`interaction_script`;
    the quoted line carries the only direct label. The table-view twin ships
    underneath, collapsed.
    """

    if not points:
        return empty_state("No line sweep saved", "This card predates the line-sweep artifact.")

    y_min, y_max = 0.25, 0.85
    xs = [offset for offset, _ in points]
    x_min, x_max = min(xs), max(xs)
    span = (x_max - x_min) or 1.0

    def x_pct(offset: float) -> float:
        return (offset - x_min) / span * 100.0

    def y_pct(probability: float) -> float:
        clamped = min(max(probability, y_min), y_max)
        return (y_max - clamped) / (y_max - y_min) * 100.0

    curve = [(x_pct(offset), y_pct(p)) for offset, p in points]
    baseline_pct = y_pct(0.5)
    # Area: curve down to the 50% baseline. Line: a ~2px band along the curve.
    area_polygon = _polygon([*curve, (100.0, baseline_pct), (0.0, baseline_pct)])
    line_half = 1.5 / plot_height * 100.0
    line_polygon = _polygon(
        [*[(x, y - line_half) for x, y in curve], *[(x, y + line_half) for x, y in reversed(curve)]]
    )

    quoted_probability = min((abs(offset - quoted_line), p) for offset, p in points)[1]
    quoted_x, quoted_y = x_pct(quoted_line), y_pct(quoted_probability)

    grid = "".join(
        f'<div style="position:absolute;left:0;right:0;top:{y_pct(level):.2f}%;height:1px;'
        'background:var(--grid);"></div>'
        for level in (0.4, 0.6, 0.7, 0.8)
    )
    tick_step = max(1.0, round(span / 4))
    ticks = []
    tick = x_min
    while tick <= x_max + 0.01:
        label = f"{tick:+g}" if tick != 0 else escape(quote_label)
        ticks.append(
            f'<span class="fine num" style="position:absolute;left:{x_pct(tick):.2f}%;'
            f'transform:translateX(-50%);font-size:10px;">{label}</span>'
        )
        tick += tick_step
    payload = escape(json.dumps([[round(offset, 1), round(p, 4)] for offset, p in points]))
    table_rows = "".join(f"<tr><td>{o:+g}</td><td>{p:.0%}</td></tr>" for o, p in points)

    return f"""
<div class="ats-sweep" id="{escape(element_id)}" data-points="{payload}"
     data-xmin="{x_min}" data-xmax="{x_max}" data-ymin="{y_min}" data-ymax="{y_max}"
     style="max-width:{width}px;">
  <div class="plot" style="position:relative;height:{plot_height}px;"
       role="img" aria-label="Confidence in {escape(pick_text)} across alternative lines">
    {grid}
    <div style="position:absolute;left:0;right:0;top:{baseline_pct:.2f}%;height:0;
                border-top:1px dashed var(--baseline);"></div>
    <span class="fine" style="position:absolute;left:2px;top:{baseline_pct:.2f}%;
          transform:translateY(-115%);font-size:10px;">50%</span>
    <div style="position:absolute;inset:0;background:var(--series-model);opacity:0.10;
                clip-path:polygon({area_polygon});"></div>
    <div style="position:absolute;inset:0;background:var(--series-model);
                clip-path:polygon({line_polygon});"></div>
    <div class="ats-crosshair" style="position:absolute;top:0;bottom:0;width:1px;
                background:var(--baseline);display:none;"></div>
    <div class="ats-hoverdot" style="position:absolute;width:8px;height:8px;
                border-radius:50%;background:var(--series-model);
                border:2px solid var(--surface);transform:translate(-50%,-50%);
                display:none;"></div>
    <div style="position:absolute;left:{quoted_x:.2f}%;top:{quoted_y:.2f}%;width:10px;
                height:10px;border-radius:50%;background:var(--series-model);
                border:2px solid var(--surface);transform:translate(-50%,-50%);"></div>
    <span class="num" style="position:absolute;left:{quoted_x:.2f}%;
          top:{max(quoted_y - 16.0, 2.0):.2f}%;transform:translate(-50%,-100%);
          font-size:11px;font-weight:650;white-space:nowrap;">
      {quoted_probability:.0%} at the line</span>
    <div class="tip"></div>
  </div>
  <div style="position:relative;height:16px;margin-top:4px;">{"".join(ticks)}</div>
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
# Family comparison bars — reality's weight vs. the market's weight
# ---------------------------------------------------------------------------


def family_comparison_bars(
    *,
    reality_share: float,
    market_share: float,
    scale_max: float,
    width: int = 280,
) -> str:
    """One feature family's paired bars: how much reality rewards it (blue,
    labeled "reality") vs. how much the market already prices it (orange,
    labeled "market"), never color alone.

    Both shares are fractions of a target's total weight (see
    ``nfl_ats.market_decomposition.family_weights_table``'s ``share`` column,
    already scale-free and comparable between the two targets). ``scale_max``
    is caller-supplied and shared across every family's card so the bars stay
    comparable to each other, not just within one pair -- pass the largest
    share across every family being shown, not just this one.
    """

    scale = scale_max if scale_max > 0 else 1.0

    def pct(value: float) -> float:
        return min(max(value, 0.0), scale) / scale * 100.0

    return f"""
<div style="max-width:{width}px;display:flex;flex-direction:column;gap:5px;"
     role="img" aria-label="Reality weighs this {reality_share:.0%}, the market {market_share:.0%}">
  <div style="display:flex;align-items:center;gap:8px;">
    <span class="fine" style="width:46px;flex:none;">reality</span>
    <div style="position:relative;height:12px;flex:1;background:var(--grid);border-radius:3px;">
      <div style="position:absolute;left:0;top:0;bottom:0;width:{pct(reality_share):.2f}%;
                  background:var(--series-model);border-radius:3px;"></div>
    </div>
    <span class="num fine" style="width:38px;text-align:right;flex:none;">{reality_share:.0%}</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <span class="fine" style="width:46px;flex:none;">market</span>
    <div style="position:relative;height:12px;flex:1;background:var(--grid);border-radius:3px;">
      <div style="position:absolute;left:0;top:0;bottom:0;width:{pct(market_share):.2f}%;
                  background:var(--series-market);border-radius:3px;"></div>
    </div>
    <span class="num fine" style="width:38px;text-align:right;flex:none;">{market_share:.0%}</span>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Contribution bars — zero-centered, diverging, per-family, for one game
# ---------------------------------------------------------------------------


def contribution_bars(
    rows: Sequence[tuple[str, float]],
    *,
    pick_text: str,
    width: int = 520,
) -> str:
    """Zero-centered diverging bars: how many points each family contributes
    toward -- or against -- one game's pick.

    ``rows`` is ``[(phrase, points), ...]``, already pick-side-oriented (a
    positive value always means "supports the pick" -- see
    ``nfl_ats.market_decomposition.explain_game_structured``, which produces
    exactly this shape via its ``drivers``/``offsets``). Direction is carried
    by position (left/right of the zero line) *and* color *and* a labeled
    legend, never color alone; the closest existing relative is
    :func:`season_bars`, which is reference-line relative but not signed.
    """

    if not rows:
        return empty_state(
            "No single family stands out",
            "The gap is small, or spread across many minor factors -- see the sentence above.",
        )
    magnitude = max(abs(points) for _, points in rows) or 1.0
    scale = magnitude * 1.15

    def half_pct(points: float) -> float:
        return min(abs(points), scale) / scale * 50.0

    bar_rows = []
    for phrase, points in rows:
        supports = points >= 0
        half = half_pct(points)
        color = "var(--div-pos)" if supports else "var(--div-neg)"
        left = 50.0 if supports else 50.0 - half
        bar_rows.append(
            '<div style="display:flex;align-items:center;gap:10px;">'
            '<span class="fine" style="width:180px;text-align:right;flex:none;">'
            f"{escape(phrase)}</span>"
            '<div style="position:relative;height:16px;flex:1;">'
            '<div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:0;'
            'border-left:1px dashed var(--baseline);"></div>'
            f'<div style="position:absolute;left:{left:.2f}%;width:{half:.2f}%;top:0;bottom:0;'
            f'border-radius:3px;background:{color};"></div></div>'
            f'<span class="num fine" style="width:44px;flex:none;">{points:+.1f}</span>'
            "</div>"
        )
    return f"""
<div style="max-width:{width}px;display:flex;flex-direction:column;gap:6px;"
     role="img" aria-label="Points toward or against {escape(pick_text)}, by feature family">
  <div style="display:flex;gap:16px;padding-left:190px;flex-wrap:wrap;">
    <span class="fine"><span style="color:var(--div-neg);">&#9632;</span> opposes the pick</span>
    <span class="fine"><span style="color:var(--div-pos);">&#9632;</span> supports the pick</span>
  </div>
  {"".join(bar_rows)}
</div>
"""


# ---------------------------------------------------------------------------
# Delegated interaction layer (ship once per page)
# ---------------------------------------------------------------------------


def interaction_script() -> str:
    """Crosshair + tooltip for every ``.ats-sweep`` on the page.

    Inline handlers are stripped by Streamlit's sanitizer, and per-element
    wiring is unreliable because Streamlit executes a block's scripts while
    its content may still be detached (verified live: querySelectorAll at
    execution time found nothing). One delegated document-level listener,
    installed once, resolves the sweep under the cursor at event time -- it
    survives rerenders and attachment order. Geometry is percent-based over
    the ``.plot`` box, matching the clip-path rendering in
    :func:`sweep_curve`.
    """

    return "<script>" + interaction_js() + "</script>"


def interaction_js() -> str:
    """The raw interaction JavaScript, for embedding into a shared script tag.

    Streamlit's sanitizer keeps only ONE ``<script>`` element per ``st.html``
    block (verified live: with two sibling scripts the second is removed
    entirely), so pages that need both the theme sync and this wiring must
    merge them into a single tag via ``theme.theme_sync_script(extra_js=...)``.
    """

    return """
(function () {
  if (window.__atsSweepWired) return;
  window.__atsSweepWired = true;
  var active = null;
  function hide(root) {
    if (!root) return;
    var tip = root.querySelector(".tip");
    var crosshair = root.querySelector(".ats-crosshair");
    var dot = root.querySelector(".ats-hoverdot");
    if (tip) tip.style.display = "none";
    if (crosshair) crosshair.style.display = "none";
    if (dot) dot.style.display = "none";
  }
  document.addEventListener("mousemove", function (event) {
    var target = event.target instanceof Element ? event.target : null;
    var plot = target && target.closest ? target.closest(".ats-sweep .plot") : null;
    var root = plot ? plot.closest(".ats-sweep") : null;
    if (active && active !== root) hide(active);
    active = root;
    if (!plot || !root) return;
    var tip = root.querySelector(".tip");
    var crosshair = root.querySelector(".ats-crosshair");
    var dot = root.querySelector(".ats-hoverdot");
    if (!tip) return;
    var points = JSON.parse(root.dataset.points);
    var xMin = +root.dataset.xmin, xMax = +root.dataset.xmax;
    var yMin = +root.dataset.ymin, yMax = +root.dataset.ymax;
    var rect = plot.getBoundingClientRect();
    var line = xMin + (event.clientX - rect.left) / rect.width * (xMax - xMin);
    var nearest = points[0];
    points.forEach(function (pt) {
      if (Math.abs(pt[0] - line) < Math.abs(nearest[0] - line)) nearest = pt;
    });
    var nx = (nearest[0] - xMin) / (xMax - xMin) * 100;
    var clamped = Math.min(Math.max(nearest[1], yMin), yMax);
    var ny = (yMax - clamped) / (yMax - yMin) * 100;
    if (crosshair) { crosshair.style.left = nx + "%"; crosshair.style.display = ""; }
    if (dot) { dot.style.left = nx + "%"; dot.style.top = ny + "%"; dot.style.display = ""; }
    var sign = nearest[0] > 0 ? "+" : "";
    // Plain text only: any tag-like sequence in this script's SOURCE (even
    // inside a string or this comment) makes the sanitizer drop the whole
    // script element, verified live. Never write a less-than sign followed
    // by a letter anywhere in this file's JS.
    tip.textContent = "line " + sign + nearest[0] + " \\u00b7 " +
      Math.round(nearest[1] * 100) + "% to cover";
    tip.style.display = "block";
    var pixelX = nx / 100 * rect.width;
    var pixelY = ny / 100 * rect.height;
    tip.style.left = Math.min(Math.max(pixelX - 40, 0), rect.width - 120) + "px";
    tip.style.top = Math.max(pixelY - 38, 0) + "px";
  });
})();
"""
