"""Design tokens and the shared stylesheet for the dashboard's visual system.

The palette is the dataviz reference instance (validated: adjacent-pair CVD
DeltaE >= 8 and normal-vision >= 15 in both modes; see the skill's palette
notes). Components are written against ROLE names, never raw hex, so light
and dark swap in one place. Streamlit renders our components inside
``st.html`` blocks; every block gets ``class="ats"`` so this stylesheet
scopes cleanly and never fights Streamlit's own chrome.

Series-role conventions (fixed, never cycled):
  --series-model  (slot 1, blue)    = our model / our number
  --series-market (slot 2, orange)  = the market / the pool's number
  --series-third  (slot 3, aqua)    = a third comparator when one exists
Status colors are reserved for state (sync, safety) and always ship with an
icon + label, never color alone.
"""

from __future__ import annotations

TOKENS_LIGHT: dict[str, str] = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink-2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)",
    "series-model": "#2a78d6",
    "series-market": "#eb6834",
    "series-third": "#1baf7a",
    "seq-100": "#cde2fb",
    "seq-250": "#86b6ef",
    "seq-400": "#3987e5",
    "seq-550": "#1c5cab",
    "seq-700": "#0d366b",
    "div-neg": "#e34948",
    "div-mid": "#f0efec",
    "div-pos": "#2a78d6",
    "good": "#0ca30c",
    "good-text": "#006300",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

TOKENS_DARK: dict[str, str] = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "ink": "#ffffff",
    "ink-2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "baseline": "#383835",
    "border": "rgba(255,255,255,0.10)",
    "series-model": "#3987e5",
    "series-market": "#d95926",
    "series-third": "#199e70",
    "seq-100": "#cde2fb",
    "seq-250": "#86b6ef",
    "seq-400": "#3987e5",
    "seq-550": "#1c5cab",
    "seq-700": "#0d366b",
    "div-neg": "#e66767",
    "div-mid": "#383835",
    "div-pos": "#3987e5",
    "good": "#0ca30c",
    "good-text": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _variables(tokens: dict[str, str]) -> str:
    return "\n".join(f"  --{name}: {value};" for name, value in tokens.items())


def stylesheet() -> str:
    """The one stylesheet every dashboard HTML block starts with."""

    return f"""
<style>
.ats {{
  color-scheme: light;
{_variables(TOKENS_LIGHT)}
  font-family: {FONT_STACK};
  color: var(--ink);
  margin: 0;
}}
@media (prefers-color-scheme: dark) {{
  .ats:not([data-theme="light"]) {{
    color-scheme: dark;
{_variables(TOKENS_DARK)}
  }}
}}
.ats[data-theme="dark"] {{
  color-scheme: dark;
{_variables(TOKENS_DARK)}
}}
.ats * {{ box-sizing: border-box; }}

/* --- Card & layout primitives ------------------------------------------ */
.ats .card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 22px;
}}
/* Vertical rhythm between stacked cards is opt-in (.stack), never global:
   a global sibling rule would also indent side-by-side cards inside .row. */
.ats .stack > .card + .card {{ margin-top: 14px; }}
.ats .row {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: stretch; }}
.ats .row > * {{ flex: 1 1 240px; min-width: 0; }}

/* --- Type scale --------------------------------------------------------- */
.ats .kicker {{
  font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 6px;
}}
.ats .title {{ font-size: 17px; font-weight: 650; margin: 0 0 2px; }}
.ats .sub {{ font-size: 13px; color: var(--ink-2); margin: 0; line-height: 1.45; }}
.ats .hero {{ font-size: 34px; font-weight: 700; line-height: 1.1; margin: 2px 0; }}
.ats .prose {{ font-size: 14px; line-height: 1.55; color: var(--ink); max-width: 68ch; }}
.ats .prose p {{ margin: 0 0 10px; }}
.ats .fine {{ font-size: 12px; color: var(--muted); line-height: 1.4; }}
.ats .num {{ font-variant-numeric: tabular-nums; }}

/* --- Chips -------------------------------------------------------------- */
.ats .chip {{
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12px; font-weight: 600; border-radius: 999px;
  padding: 3px 10px; border: 1px solid var(--border); color: var(--ink-2);
  background: transparent; white-space: nowrap;
}}
.ats .chip .dot {{ width: 8px; height: 8px; border-radius: 50%; flex: none; }}
.ats .chip.model {{ color: var(--series-model); border-color: currentColor; }}
.ats .chip.market {{ color: var(--series-market); border-color: currentColor; }}

/* --- Status (icon + label always; color never alone) -------------------- */
.ats .status {{
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12px; font-weight: 600;
}}
.ats .status svg {{ width: 14px; height: 14px; flex: none; }}
.ats .status.good {{ color: var(--good-text); }}
.ats .status.warning {{ color: var(--serious); }}
.ats .status.critical {{ color: var(--critical); }}

/* --- Chart furniture (pure HTML/CSS -- st.html strips SVG) --------------- */
.ats .axis-label {{ font-size: 11px; color: var(--muted); }}

/* --- Tooltip layer (positioned by inline JS in each component) ---------- */
.ats .tip {{
  position: absolute; pointer-events: none; z-index: 10; display: none;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 7px 10px; font-size: 12px; color: var(--ink);
  box-shadow: 0 4px 14px rgba(0,0,0,0.12); white-space: nowrap;
}}

/* --- Tables (the accessibility twin of every chart) --------------------- */
.ats table.data {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
.ats table.data th {{
  text-align: left; font-weight: 600; color: var(--ink-2);
  border-bottom: 1px solid var(--baseline); padding: 6px 10px 6px 0;
}}
.ats table.data td {{
  padding: 6px 10px 6px 0; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums;
}}
.ats details.table-view > summary {{
  cursor: pointer; font-size: 12px; color: var(--muted); margin-top: 8px;
}}
</style>
"""


def theme_sync_script(extra_js: str = "") -> str:
    """Keep every ``.ats`` root's data-theme synced to Streamlit's LIVE theme.

    Streamlit's theme menu is a client-only change that does not rerun scripts,
    so server-side theme detection goes stale; this polls a Streamlit-owned
    element's actual rendered background (the approach verified live in
    ``board.py``) and stamps data-theme on every component root. Ship this ONCE
    per page, after the last ``.ats`` block. Inline ``on*`` attributes are
    stripped by Streamlit's sanitizer, so all listeners are wired here.

    ``extra_js`` is appended inside the SAME ``<script>`` tag: the sanitizer
    keeps only one script element per ``st.html`` block (verified live), so a
    page needing additional wiring -- e.g. ``viz.interaction_js()`` -- must
    ride along here rather than shipping a second tag.
    """

    return (
        "<script>"
        + extra_js
        + """
(function () {
  function isDarkBackground(el) {
    while (el) {
      var bg = getComputedStyle(el).backgroundColor;
      var match = bg.match(/rgba?\\((\\d+), *(\\d+), *(\\d+)(?:, *([\\d.]+))?\\)/);
      if (match) {
        var alpha = match[4] === undefined ? 1 : parseFloat(match[4]);
        if (alpha > 0) {
          var luma = 0.299 * match[1] + 0.587 * match[2] + 0.114 * match[3];
          return luma < 128;
        }
      }
      el = el.parentElement;
    }
    return false;
  }
  function syncTheme() {
    var appEl = document.querySelector('[data-testid="stApp"]') || document.body;
    var mode = isDarkBackground(appEl) ? "dark" : "light";
    document.querySelectorAll(".ats").forEach(function (root) {
      // Only write on change: unconditional attribute writes invalidate
      // style/layout for the whole subtree every tick and can pin the
      // renderer on content-heavy pages.
      if (root.getAttribute("data-theme") !== mode) {
        root.setAttribute("data-theme", mode);
      }
    });
  }
  syncTheme();
  // Page navigations re-execute this script in the same document; keep ONE
  // interval alive rather than accumulating them across visits.
  if (window.__atsThemeInterval) clearInterval(window.__atsThemeInterval);
  window.__atsThemeInterval = setInterval(syncTheme, 500);
  var media = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
  if (media && media.addEventListener && !window.__atsThemeMediaWired) {
    window.__atsThemeMediaWired = true;
    media.addEventListener("change", syncTheme);
  }
})();
</script>"""
    )
