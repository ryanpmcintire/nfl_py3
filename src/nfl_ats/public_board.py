from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.active_model import active_artifact_path, load_active_ats_model
from nfl_ats.backup_qb_fade_overlay import apply_backup_qb_fade_overlay
from nfl_ats.best_pick_nomination import nominate_v3
from nfl_ats.card_view import (
    BestPickNomination,
    resolve_card_view,
    resolve_nomination,
    resolve_overlay,
    resolve_player_arrests_overlay,
    v2_nomination_inputs,
)
from nfl_ats.coach_fade_overlay import OverlayFlip, OverlayResult
from nfl_ats.dashboard import theme, viz
from nfl_ats.dashboard.findings_content import (
    CHALLENGER_DISPLAY_NAMES,
    CLOSING_NOTE,
    DETAIL_SUMMARY_LABEL,
    FINDINGS,
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
    LEAD_BLURBS,
    LEGEND_KICKER,
    PLAYED_CARD_EXPECTATION_HERO,
    SOURCE_LABEL,
    Finding,
    LeadBlurb,
    VerdictGroup,
    baseline_hero_tiles,
    findings_for,
)
from nfl_ats.data import DataContractError
from nfl_ats.displayed_confidence import (
    STRENGTH_ROUNDING_PLACES,
    StrengthBands,
    displayed_pick_probability,
    displayed_strength_word,
    served_strength_bands,
)
from nfl_ats.division_revenge_tilt_overlay import apply_division_revenge_tilt_overlay
from nfl_ats.findings_registry import (
    WatchingLead,
    load_all_entries,
    load_weak_signal_registry,
    top_open_leads,
    validate_curation,
)
from nfl_ats.four_overlay_composition import (
    POLICY_ID as SERVED_POLICY_ID,
)
from nfl_ats.four_overlay_composition import (
    FourOverlayCompositionResult,
)
from nfl_ats.home_side_location import center_offsets_from_metadata
from nfl_ats.injury_value_tilt_overlay import (
    PLAYER_FEATURE_TABLE_NAME,
    apply_injury_value_tilt_overlay,
)
from nfl_ats.interim_hc_first_game_tilt_overlay import (
    apply_interim_hc_first_game_tilt_overlay,
)
from nfl_ats.key_line_pick_read import pick_overrides_from_metadata
from nfl_ats.model_explanation import load_model_explanation_html
from nfl_ats.model_ledger import build_and_render
from nfl_ats.player_arrests_back_side_overlay import (
    POLICY_BASELINE_OPENER_ACCURACY,
    POLICY_EFFECT_ACCURACY_POINTS,
    POLICY_OPENER_ACCURACY,
    POLICY_PROBABILITY_POSITIVE,
    ArrestFlip,
    ArrestOverlayResult,
)
from nfl_ats.pool_workbench import PoolRules, build_pool_workbench_body
from nfl_ats.reporting import artifact_directories, read_json
from nfl_ats.signal_ledger import build_signal_ledger_body
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.spread_explorer import (
    SPREAD_EXPLORER_STEP,
    SpreadExplorerGameParams,
    compute_spread_explorer_params,
    load_feature_table_for_forecast,
    spread_explorer_payload,
    widget_home_cover_probability,
)
from nfl_ats.spread_gap_zone_fade_overlay import apply_spread_gap_zone_fade_overlay
from nfl_ats.surface_switch_tilt_overlay import apply_surface_switch_tilt_overlay
from nfl_ats.surgical_gating import VALUE_LOST_DIFF_COLUMNS
from nfl_ats.team_explorer import (
    DEFAULT_TREND_METRICS,
    TeamTrends,
    aggregate_team_trends,
    feature_table_to_team_states,
    metric_good_direction,
    metric_help,
    metric_label,
    team_state_payload,
)
from nfl_ats.weak_signals import Registry as WeakSignalRegistry
from nfl_ats.weak_signals import default_registry_path as _default_weak_signals_registry_path

DISCLAIMER_SHORT = "Research project -- simulated, paper picks only. Not betting advice."

DISCLAIMER_FULL = (
    "This page is the output of a personal research project. Every pick shown is a "
    "simulated, paper pick made to evaluate a forecasting model -- it is not betting "
    "advice, and no real money is risked on these picks by the author. The model's "
    "historical accuracy sits close to a coin flip."
)


PICKS_PAGE = "index.html"
FINDINGS_PAGE = "findings.html"
MODELS_PAGE = "models.html"
TEAM_EXPLORER_PAGE = "team_explorer.html"
POOL_PAGE = "pool.html"
LEDGER_PAGE = "ledger.html"

SITE_PAGES: tuple[tuple[str, str, str], ...] = (
    (PICKS_PAGE, "This week", "This week's picks"),
    (MODELS_PAGE, "Models", "Model ledger"),
    (TEAM_EXPLORER_PAGE, "Team trends", "Team pregame-state trends"),
    (FINDINGS_PAGE, "What we've learned", "What we've learned"),
    (POOL_PAGE, "Pool workbench", "Pool workbench"),
    (LEDGER_PAGE, "Signal ledger", "Signal ledger"),
)

_PAGE_CHROME = """
<style>
body { margin: 0; overflow-x: hidden; }
.ats {
  --plane: #fafaf8;
  --surface: #ffffff;
  --ink: #111110;
  --ink-2: #4b4b47;
  /* AA normal-text minimum (4.5:1): measures 5.36:1 on the surface token and
     5.13:1 on the plane token. The former value measured 3.47:1 / 3.32:1 --
     large-text-only -- while --muted drives 11-12px labels (.fine,
     td::before), which need the full bar. Dark mode's muted already passes
     (measured 5.07:1 / 4.73:1) and is untouched. */
  --muted: #6b6b65;
  --grid: rgba(0,0,0,0.08);
  --border: rgba(0,0,0,0.08);
  --baseline: rgba(0,0,0,0.16);
  --series-model: #2a78d6;
  --series-market: #4b4b47;
  --series-third: #8a8a84;
  --good-text: #1a7f37;
  /* Status palette, VALIDATED not eyeballed (dataviz validator,
     --mode light, against the light surface): lightness band PASS, chroma
     floor PASS, CVD separation PASS (worst adjacent dE 12.6 protan),
     normal-vision floor PASS (worst 25.0). The PREVIOUS critical/serious
     pair FAILED both separation checks -- dE 7.5 in NORMAL vision and 2.7
     under deuteranopia, i.e. the two states were effectively one colour. They
     are re-stepped here for lightness separation, which is what survives CVD;
     hue alone cannot fix red-vs-amber because both sit on the red-green
     confusion axis. --serious carries a 2.58:1 contrast WARN, relieved
     (per the validator's own rule) because every status ships with a visible
     text label and every surface using it is already a table. */
  --good: #1a7f37;
  --serious: #d59200;
  --critical: #9b2418;
  /* Everything below is DERIVED from the tokens above, so the 10-hex chrome
     budget is unchanged: meaning-bearing colour is added without spending any
     new chrome, and both themes step themselves. */
  /* Was referenced twice and never defined, so the "MODEL LEDGER
     UNAVAILABLE" card rendered with no colour at all. */
  --warning: var(--serious);
  /* ORDINAL ramp for the three COARSE confidence bands: weak -> middling ->
     strong reads warm -> amber -> green, which is what a reader already
     expects from a quality scale.
     Revised 2026-08-25 (owner): this was one hue at three steps, which made
     every bar the same colour and defeated the point -- a strong pick was no
     easier to spot than a weak one. The literal ask was orange / yellow /
     green; measured, that fails. Orange-vs-yellow lands at dE 13.4 in NORMAL
     vision (validator, light), and in dark mode the L 0.48-0.67 band squeezes
     the two together to dE 10.6-12.6, because orange and yellow are neighbours
     in hue AND the band forbids separating them by lightness.
     So the ramp reuses the STATUS tokens, which are already validated as
     mutually distinguishable in both themes (light: worst adjacent dE 25.0
     normal / 12.6 CVD; dark: 15.2 / 7.6). That is not a compromise on
     legibility -- it is a warmer red at the weak end than the ask, and it
     costs ZERO new hexes. It is also semantically right: a weak pick and a
     warning state ARE the same judgement, so sharing a colour is coherent
     rather than a collision.
     Still deliberately discrete, never a continuous gradient over the
     underlying probability -- three states, three colours, no invented
     precision. */
  --band-1: var(--critical);
  --band-2: var(--serious);
  --band-3: var(--good);
  /* Diverging pair for signed deltas: two hues + a NEUTRAL midpoint, never a
     hue at zero. The sign character is always rendered too, so colour is the
     secondary channel. */
  --pos: var(--good);
  --neg: var(--critical);
  --zero: var(--muted);
}
@media (prefers-color-scheme: dark) {
  .ats:not([data-theme="light"]) {
    color-scheme: dark;
    --plane: #0b0c0e;
    --surface: #141518;
    --ink: #f7f8f8;
    --ink-2: #b4b8bf;
    --muted: #7d828b;
    --grid: #23252b;
    --border: #23252b;
    --baseline: #33363d;
    --series-model: #6ea8dc;
    --series-market: #b4b8bf;
    --series-third: #7d828b;
    --good-text: #45a86b;
    /* Dark is SELECTED, not an automatic flip of light. The previous dark
       steps failed FOUR validator checks against the dark surface: three
       colours outside the L 0.48-0.67 band, --series-model reading gray
       (chroma 0.098 < 0.1 floor), CVD separation dE 5.4, and a normal-vision
       floor of 11.5 between --critical and --serious. Re-stepped and
       re-validated: all checks PASS, worst normal-vision adjacent dE 15.2.
       The remaining CVD WARN (7.6 protan, in the 6-8 floor band) is legal
       here because every status ships with a visible text label. */
    --good: #3fa06a;
    --serious: #b8891f;
    --critical: #c9483c;
  }
}
.ats {
  background: var(--plane); min-height: 100vh; color: var(--ink);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.ats .wrap { max-width: 72rem; margin: 0 auto; padding: 24px 18px 52px; }
.ats a { color: var(--series-model); text-decoration: none; }
.ats a:hover { text-decoration: underline; }

/* Skip link (WCAG 2.4.1 Bypass Blocks): hidden until keyboard-focused. */
.ats .skip-link {
  position: absolute; left: -9999px; top: 0; z-index: 100;
  background: var(--surface); color: var(--series-model);
  padding: 8px 12px; font-size: 13px; border: 1px solid var(--border);
}
.ats .skip-link:focus { left: 0; }

/* Visible keyboard focus everywhere (WCAG 2.4.7 Focus Visible). */
.ats a:focus-visible,
.ats summary:focus-visible,
.ats input:focus-visible {
  outline: 2px solid var(--series-model); outline-offset: 2px;
}

/* --- Emphasis needs somewhere to go -------------------------------------
   Running prose sat at full --ink, so <strong>/<b> could only add WEIGHT, and
   at 14px on a near-black plane a 400->600 step in the identical colour is
   close to invisible. Drop body copy to --ink-2 and let emphasis take --ink:
   bold now carries a luminance step as well as a weight one, and the body is
   easier to read for it (pure white copy on near-black is harsher than it
   needs to be). No new colours -- this is headroom, not paint. Both tokens
   already clear AA on both planes; --ink-2 measures ~11:1 dark and ~8.5:1
   light against their backgrounds.
   Colour is NOT used for emphasis here on purpose: accent blue means
   "interactive" everywhere on this site, and green/amber/red mean
   "direction of a result". Spending either on ordinary bold would blunt the
   signals that carry meaning. */
.ats .prose { color: var(--ink-2); }
.ats .prose strong, .ats .prose b { color: var(--ink); font-weight: 600; }

/* Same headroom trick where the reader's actual question lives: on the week
   board the picked side is the answer, so the supporting columns step back to
   --ink-2 and the pick keeps --ink. */
.ats table.week-board tr.board-game td { color: var(--ink-2); }
.ats table.week-board tr.board-game td b { color: var(--ink); }

/* The one arm that actually plays. Green already means "live/good" here (the
   Best Pick star, the better-than-coin-flip delta), so this reuses that sense
   rather than introducing a colour. Deliberately NOT applied to the P+ column:
   colouring a probability by band would render a threshold on screen, and
   AGENTS.md is explicit that thresholds govern what the docs may claim, never
   which card is played. */
.ats td.status-live { color: var(--good-text); font-weight: 600; }

/* Glossary terms. The UA default for abbr[title] is `text-decoration:
   underline dotted`, which draws the dots tight under the glyph -- and against
   the token "P+" the dotted rule fuses with the plus sign so it reads as "P±".
   Those are different statistics (probability_positive, versus a plus/minus
   interval), so the default styling was actively misinforming. A bottom border
   sits a pixel lower and stays clear of the glyph. */
.ats abbr[title] {
  text-decoration: none;
  border-bottom: 1px dotted var(--muted);
  cursor: help;
}

.ats nav.site { display: flex; gap: 16px; flex-wrap: wrap; margin: 0 0 16px; font-size: 13px; }
.ats nav.site a { color: var(--ink-2); }
.ats nav.site a[aria-current="page"] { color: var(--ink); font-weight: 600; }
.ats a, .ats summary { transition: color 150ms ease, border-color 150ms ease; }

/* Type scale: only {11,12,13,14,17,24}px; 400 body, 600 emphasis. */
.ats .kicker { font-size: 11px; letter-spacing: 0.08em; }
.ats .title { font-size: 17px; font-weight: 600; }
.ats h2.title { letter-spacing: -0.01em; }
.ats .page-title { font-size: 24px; }
.ats .sub { font-size: 13px; }
.ats .prose { font-size: 14px; }
.ats .fine { font-size: 12px; }
.ats .hero { font-size: 24px; font-weight: 600; line-height: 1.15; }

/* Cards become flat hairline sections: no boxes, no fills, no shadows. */
.ats .card {
  background: none; border: none; border-top: 1px solid var(--grid);
  border-radius: 0; padding: 12px 0 0; margin-top: 12px; box-shadow: none;
}
.ats .tip { box-shadow: none; border-radius: 4px; }
.ats .status, .ats .chip { border-radius: 4px; }

/* Four-panel terminal grid (desktop >=1100px): summary | board spanning tall
   right; ledger mini and challenger watch stacked left beneath the summary.
   Below 1100px everything stacks in DOM order. */
.ats .ledger-grid {
  display: grid; grid-template-columns: minmax(300px, 2fr) 3fr;
  grid-template-areas: "summary board" "ledger board" "watch board";
  /* The board spans all three rows and is far taller than the left panels, so
     with default auto rows its surplus height is shared out across them --
     opening a ragged gap between the summary and the ledger that looks like a
     missing element. Pin the first two rows to their content and let the slack
     fall into the last row, where it reads as the end of the column. */
  grid-template-rows: min-content min-content 1fr;
  column-gap: 32px; align-items: start; margin-top: 8px;
}
.ats .panel { border-top: 1px solid var(--grid); padding-top: 8px; }
.ats .panel-summary { grid-area: summary; }
.ats .panel-board { grid-area: board; }
.ats .panel-ledger { grid-area: ledger; margin-top: 16px; }
.ats .panel-watch { grid-area: watch; margin-top: 16px; }
@media (max-width: 1099px) {
  .ats .ledger-grid { display: block; }
  .ats .panel-ledger, .ats .panel-watch { margin-top: 16px; }
}

/* Week board: one continuous table, 40px game rows, expandable sub-rows at the
   compact 32px scale; the sticky header separates with a hairline border. */
.ats table.week-board th {
  position: sticky; top: 0; background: var(--surface); z-index: 1;
  border-bottom: 1px solid var(--baseline);
}
.ats table.week-board tr.board-game td { padding: 12px 8px 12px 0; font-size: 13px; }
.ats table.week-board tr.board-sub > td { padding: 0 0 8px; }
.ats table.week-board tr.board-sub table.data th,
.ats table.week-board tr.board-sub table.data td { padding: 6px 8px 6px 0; }
/* One "Why this pick" disclosure sits under each of the week's ~16 games. In
   accent blue that is sixteen bright links stacked down the page, all saying
   the same thing, competing with the picks themselves for attention -- the
   affordance shouted louder than the content. Muted by default, accent only on
   hover/focus, so it reads as a control you can find rather than a row you
   must read. */
.ats details.why-pick > summary {
  cursor: pointer; font-size: 12px; color: var(--muted); list-style: revert;
}
.ats details.why-pick > summary:hover,
.ats details.why-pick > summary:focus-visible,
.ats details.why-pick[open] > summary {
  color: var(--series-model);
}
/* Lead narrative beside its legend (findings page). Prose keeps the theme's
   68ch reading measure; the second column absorbs the plane that measure would
   otherwise leave empty. Stacks in DOM order below the terminal-grid
   breakpoint, same as every other two-column block here. */
.ats .lede-grid {
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  column-gap: 32px; align-items: start; margin-bottom: 14px;
}
@media (max-width: 1099px) { .ats .lede-grid { display: block; } }

/* Strength meter (see confidence_meter): 4px segments on the 4px spacing grid,
   square corners, no fill colours beyond the existing ink/baseline tokens. */
.ats .strength { white-space: nowrap; }
/* The PICK is the single thing a reader came for, and it was rendering as
   plain bold at the same weight as the matchup and the line beside it
   (owner, 2026-08-25). It gets its own type treatment -- size, weight,
   letter-spacing, colour -- so the pick column reads as the answer column
   rather than as one more field. A left accent rule shipped alongside that
   treatment the same day "as reinforcement, not the only cue"; it read back
   as a stray blue vertical line beside every pick (owner, 2026-08-26) and
   was removed. The type treatment was always the load-bearing cue, so the
   column keeps its emphasis -- and no longer needs the padding the rule
   required to keep the text off it. */
.ats .pick-team {
  font-size: 15px;
  font-weight: 750;
  letter-spacing: -0.01em;
  color: var(--ink);
}
/* The starred Best Pick is one game a week and should be findable instantly:
   its pick text takes the "good" colour, and the pick cell itself carries a
   &#9733; flag (see _week_board) -- colour plus a distinct glyph, no border
   needed. */
.ats table.week-board tr.is-best-pick .pick-team { color: var(--good); }
.ats .meter { display: inline-flex; gap: 4px; vertical-align: middle; margin-right: 8px; }
.ats .meter i { display: block; width: 4px; height: 12px; background: var(--baseline); }
.ats .meter i.on { background: var(--ink-2); }
/* The three coarse confidence bands, coloured on the SAME three states the
   shape already encodes -- one hue, three steps, no continuous gradient. The
   word beside the meter stays the accessible label; the meter is aria-hidden,
   so nothing is conveyed by colour or shape alone. */
.ats .meter.band-1 i.on { background: var(--band-1); }
.ats .meter.band-2 i.on { background: var(--band-2); }
.ats .meter.band-3 i.on { background: var(--band-3); }
/* Signed delta. The +/- character is always rendered, so colour is the
   secondary channel and a zero reads neutral rather than picking a side. */
.ats .delta { font-variant-numeric: tabular-nums; font-weight: 600; }
.ats .delta.pos { color: var(--pos); }
.ats .delta.neg { color: var(--neg); }
.ats .delta.zero { color: var(--zero); font-weight: 500; }
/* Status pill: a colour-carrying dot plus its own text label, never colour
   alone. Reserved for state (promoted/active/retired/win/loss), never reused
   as a series hue. */
.ats .pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase;
  font-weight: 700; color: var(--ink-2); white-space: nowrap;
}
.ats .pill::before {
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: var(--zero); flex: none;
}
.ats .pill.is-good::before { background: var(--good); }
.ats .pill.is-warn::before { background: var(--serious); }
.ats .pill.is-bad::before { background: var(--critical); }
.ats .pill.is-live::before { background: var(--series-model); }
.ats .pill.is-idle::before { background: var(--zero); }
/* Signal-ledger "control arm" status: neither good nor bad -- a deliberately
   unplayable instrument check -- so it takes the third series slot rather
   than borrowing a state hue that would misstate it as a warning. */
.ats .pill.is-control::before { background: var(--series-third); }
/* Model-ledger badges. nfl_ats.model_ledger emits badge/badge-promoted/
   badge-challenger/badge-muted and its own docstring says the fragment
   "reuses the design-system classes (table.data, badge-*, ...)" -- but
   badge-* was never actually defined here, so all 28 badges on the models
   page rendered as plain undifferentiated text. Each already ships a glyph
   AND its word, so adding the status hue gives a third channel rather than
   the only one. */
.ats .badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; letter-spacing: 0.04em; font-weight: 700;
  color: var(--ink-2); white-space: nowrap;
}
.ats .badge-glyph { font-size: 10px; line-height: 1; }
.ats .badge-promoted { color: var(--good); }
.ats .badge-challenger { color: var(--series-model); }
.ats .badge-muted { color: var(--muted); font-weight: 600; }
/* Forced-colors / print: colour is stripped, so the dot becomes a shape and
   the delta keeps its sign. Nothing above is load-bearing on its own. */
@media (forced-colors: active) {
  .ats .pill::before { forced-color-adjust: none; background: CanvasText; }
  .ats .pill.is-bad::before { border-radius: 0; }
  .ats .pill.is-warn::before { border-radius: 0; transform: rotate(45deg); }
}

.ats .flip-flag { color: var(--ink-2); cursor: help; }
.ats .best-flag { color: var(--good-text); cursor: help; }
@media (max-width: 640px) {
  .ats .wrap { padding: 16px 12px 40px; }
  .ats table.week-board thead { display: none; }
  .ats table.week-board, .ats table.week-board tbody,
  .ats table.week-board tr, .ats table.week-board td { display: block; width: 100%; }
  .ats table.week-board tr.board-game td {
    border: none; padding: 2px 0; display: flex; justify-content: space-between;
    gap: 10px; align-items: baseline;
  }
  .ats table.week-board td::before {
    content: attr(data-label); color: var(--muted); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em; flex: none;
  }
}

/* Deep-dive blocks: flat prose sections separated by hairlines, never boxes. */
.ats .deep-game { padding: 16px 0; max-width: 70ch; scroll-margin-top: 48px; }
.ats .deep-game + .deep-game { border-top: 1px solid var(--grid); }

/* Cover curve (2026-08-26 merge): the slider spans the plot's own width and
   sits flush beneath it, so dragging reads as moving a handle ALONG the
   chart rather than operating a separate control -- the owner's complaint
   about the two old charts ("shows basically the same thing") plus the
   part worth keeping ("a slider to shift odds on demand"). Sizing/colour
   only here; accent reuses the model-series token, same as the retired
   spread-explorer slider it replaces. */
.ats .ats-cover input.cover-slider {
  display: block; width: 100%; height: 26px; margin: 4px 0 2px;
  accent-color: var(--series-model); touch-action: manipulation;
}
/* The draggable handle's fill is the diverging pair itself (never the
   model/market series colours, which are reserved for "whose number is
   this" -- circle vs. square already carries that): "--div-pos" when the
   hypothetical line favours the pick, "--div-neg" against it, and the
   neutral midpoint at an exact 50%. Never the only cue -- position (on the
   curve) and the live percentage text carry the same reading. Set INLINE
   (by ``viz.cover_curve`` and its drag handler), not as a class rule here:
   ``--div-pos``/``--div-neg``/``--div-mid`` are theme.py's tokens, and every
   ``var(...)`` referenced in THIS stylesheet must be declared in it too
   (see ``test_every_css_variable_used_is_actually_defined``) -- the
   ``is-pos``/``is-neg``/``is-mid`` classes still ship, for tests and any
   future hook, they just are not where the colour comes from. */

/* Signal ledger (docs/ledger.html): every recorded weak-signal experiment,
   sortable/filterable/searchable. New shapes only where the existing
   vocabulary (.pill, .delta, .chip, table.data) has no equivalent -- the
   status tag, the effect number and the P+ chip all reuse it unchanged. */
.ats .ledger-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.ats .ledger-controls .lbl {
  font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); margin-right: 2px;
}
.ats .ledger-controls .chip[aria-pressed="true"] {
  background: var(--ink); border-color: var(--ink); color: var(--surface);
}
.ats .ledger-search {
  font: inherit; font-size: 13px; padding: 6px 10px; min-width: 220px;
  border: 1px solid var(--border); border-radius: 4px;
  background: var(--surface); color: var(--ink);
}
.ats table.ledger th.sortable { cursor: pointer; user-select: none; }
.ats table.ledger th.sortable:hover { color: var(--ink); }
.ats table.ledger th .arrow { opacity: 0; margin-left: 4px; }
.ats table.ledger th[aria-sort] .arrow { opacity: 1; }
.ats table.ledger th[aria-sort="ascending"] .arrow::after { content: "\2191"; }
.ats table.ledger th[aria-sort="descending"] .arrow::after { content: "\2193"; }
.ats table.ledger td { vertical-align: top; }
.ats .ledger-idea { max-width: 46ch; min-width: 260px; }
.ats .ledger-idea .sub { display: block; margin-top: 4px; font-size: 11px; color: var(--muted); }
.ats .ledger-idea .fallback { font-style: italic; }
.ats .ledger-flags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.ats .ledger-flag {
  font-size: 11px; line-height: 1.3; padding: 2px 6px;
  border: 1px dashed var(--baseline); border-radius: 2px;
  color: var(--ink-2); background: var(--surface);
}
.ats .ledger-effect { min-width: 150px; }
/* The interval under an effect, and the "favours it" cue under a percentage,
   are second lines -- not continuations. As bare inline <span>s they rendered
   flush against the number above with no separator at all ("98%favours it",
   "+17.07 accuracy pts+0.79 to +31.67"; owner-reported 2026-08-26, and no
   amount of whitespace in the markup fixes an inline run). Block display is
   the actual fix: it restores the stacked reading the column was designed for. */
.ats .ledger-sub {
  display: block;
  margin-top: 3px;
  font-size: 12px;
  color: var(--muted);
  line-height: 1.35;
}
.ats .ledger-gauge { position: relative; height: 14px; margin: 5px 0 3px; min-width: 130px; }
.ats .ledger-gauge .rail {
  position: absolute; left: 0; right: 0; top: 6px; height: 1px; background: var(--grid);
}
.ats .ledger-gauge .zero {
  position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: var(--baseline);
}
.ats .ledger-gauge .whisk {
  position: absolute; top: 6px; height: 1px; background: var(--baseline);
}
.ats .ledger-gauge .whisk::before, .ats .ledger-gauge .whisk::after {
  content: ""; position: absolute; top: -3px; width: 1px; height: 7px; background: var(--baseline);
}
.ats .ledger-gauge .whisk::before { left: 0; }
.ats .ledger-gauge .whisk::after { right: 0; }
.ats .ledger-gauge .bar { position: absolute; top: 3px; height: 7px; border-radius: 1px; }
.ats .ledger-gauge .bar.up { background: var(--pos); }
.ats .ledger-gauge .bar.dn { background: var(--neg); }
.ats .ledger-gauge .over {
  position: absolute; top: -1px; font-size: 11px; line-height: 1; color: var(--muted);
}
.ats .ledger-rel { min-width: 100px; }
.ats .ledger-rel .track {
  position: relative; height: 5px; background: var(--grid); border-radius: 1px;
  margin: 6px 0 4px; overflow: hidden;
}
.ats .ledger-rel .fill { position: absolute; top: 0; bottom: 0; left: 0; background: var(--ink-2); }
.ats .ledger-rel .fill.weak { background: var(--serious); }
.ats .ledger-rel .fill.bad { background: var(--critical); }
</style>
"""


def _nav(current: str) -> str:
    items = []
    for filename, label, _title in SITE_PAGES:
        if filename == current:
            items.append(f'<span aria-current="page">{escape(label)}</span>')
        else:
            items.append(f'<a href="{filename}">{escape(label)}</a>')
    return f'<nav class="site">{"".join(items)}</nav>'


def _disclaimer_banner() -> str:
    return f'<p class="sub" style="font-weight:600;margin:0 0 16px;">{DISCLAIMER_SHORT}</p>'


def _footer(generated: datetime, note: str = "") -> str:
    stamp = generated.strftime("%Y-%m-%d %H:%M UTC")
    lead = f"{note} &middot; " if note else ""
    return (
        '<footer style="margin-top:36px;padding-top:14px;border-top:1px solid var(--grid);">'
        f'<p class="fine">{lead}page generated {stamp}.</p>'
        f'<p class="fine" style="margin-top:10px;max-width:82ch;">{DISCLAIMER_FULL}</p></footer>'
    )


def _page(
    *,
    current: str,
    body: str,
    generated: datetime,
    footer_note: str = "",
    scripts: str = "",
) -> str:

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
<a class="skip-link" href="#main-content">Skip to content</a>
{_nav(current)}
{_disclaimer_banner()}
<main id="main-content">
{body}
</main>
{_footer(generated, footer_note)}
</div></div>
{scripts}
</body>
</html>
"""


STRONG_LEAN_POINTS = 1.5
SWEEP_HALF_WIDTH = 4.0

_COVER_CURVE_FALLBACK_OFFSETS: tuple[float, ...] = tuple(
    round(-SWEEP_HALF_WIDTH + step_index * SPREAD_EXPLORER_STEP, 1)
    for step_index in range(round(2 * SWEEP_HALF_WIDTH / SPREAD_EXPLORER_STEP) + 1)
)

_WEEK_LABELS = {
    "WC": "Wild Card round",
    "DIV": "Divisional round",
    "CON": "Conference championships",
    "SB": "Super Bowl",
}


def spread_words(home: str, away: str, home_spread: float) -> str:

    if pd.isna(home_spread) or home_spread == 0:
        return "pick 'em"
    favorite, points = (home, home_spread) if home_spread > 0 else (away, -home_spread)
    return f"{favorite} -{points:g}"


def pick_side(row: pd.Series) -> tuple[str, float]:

    probability = float(row["home_cover_probability"])
    displayed = displayed_pick_probability(row)
    if probability >= 0.5:
        return str(row["home_team"]), displayed if displayed is not None else probability
    return str(row["away_team"]), displayed if displayed is not None else 1.0 - probability


def _kickoff_words(row: pd.Series) -> str:
    weekday = str(row.get("weekday") or "").strip()
    gametime = str(row.get("gametime") or "").strip()
    return f"{weekday} {gametime} ET".strip()


def _number(value: Any) -> float | None:

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _default_data_root() -> Path:

    return Path(os.environ.get("NFL_ATS_DATA_DIR", "data"))


CONFIDENCE_ROUNDING_PLACES = STRENGTH_ROUNDING_PLACES


def confidence_word(probability: float, bands: StrengthBands | None) -> str:

    return "" if bands is None else bands.word(probability)


def row_confidence_word(row: pd.Series, bands: StrengthBands | None) -> str:

    attached = displayed_strength_word(row)
    return attached if attached is not None else confidence_word(pick_side(row)[1], bands)


_CONFIDENCE_FILL = {"slight": 1, "lean": 2, "strong": 3}


def confidence_meter(word: str) -> str:

    filled = _CONFIDENCE_FILL.get(word, 0)
    segments = "".join(f'<i class="{"on" if index < filled else ""}"></i>' for index in range(3))
    band = f" band-{filled}" if filled else ""
    return f'<span class="meter{band}" aria-hidden="true">{segments}</span>'


def delta_html(points: float | None, *, digits: int = 2, suffix: str = "") -> str:

    if points is None or (isinstance(points, float) and points != points):
        return '<span class="delta zero">&#8212;</span>'
    tone = "pos" if points > 0 else "neg" if points < 0 else "zero"
    return f'<span class="delta {tone}">{points:+.{digits}f}{escape(suffix)}</span>'


_PILL_TONES: dict[str, str] = {
    "good": "is-good",
    "warn": "is-warn",
    "bad": "is-bad",
    "live": "is-live",
    "idle": "is-idle",
    "control": "is-control",
}


def p_plus_html(probability: float | None, text: str) -> str:

    if probability is None:
        return f'<span class="delta zero">{escape(text)}</span>'
    tone = "pos" if probability > 0.5 else "neg" if probability < 0.5 else "zero"
    return f'<span class="delta {tone}">{escape(text)}</span>'


def accuracy_vs_coin_flip_html(accuracy: float | None) -> str:

    if accuracy is None:
        return '<span class="delta zero">--</span>'
    tone = "pos" if accuracy > 0.5 else "neg" if accuracy < 0.5 else "zero"
    return f'<span class="delta {tone}">{accuracy:.1%}</span>'


def pill_html(tone: str, label: str, *, title: str = "") -> str:

    modifier = _PILL_TONES.get(tone, "is-idle")
    attrs = f' title="{escape(title)}"' if title else ""
    return f'<span class="pill {modifier}"{attrs}>{escape(label)}</span>'


_SPREAD_EXPLORER_TOLERANCE = 1e-4


def _assert_spread_explorer_matches_card(
    params: Mapping[str, SpreadExplorerGameParams], predictions: pd.DataFrame
) -> None:

    if not params:
        return
    lookup = predictions.set_index(predictions["game_id"].astype(str))
    for game_id, values in spread_explorer_payload(params).items():
        widget_probability = (
            float(values["pinned"])
            if "pinned" in values
            else widget_home_cover_probability(
                values["line"], values["center"], values["mean"], values["std"]
            )
        )
        published = _number(lookup.loc[game_id, "home_cover_probability"])
        if published is None:
            raise DataContractError(
                f"Spread explorer widget has no usable published home_cover_probability for "
                f"{game_id} to check against"
            )
        if abs(widget_probability - published) > _SPREAD_EXPLORER_TOLERANCE:
            raise DataContractError(
                "Spread explorer widget formula disagrees with the published card for "
                f"{game_id}: widget={widget_probability:.6f} card={published:.6f} "
                f"(tolerance {_SPREAD_EXPLORER_TOLERANCE})"
            )


def assert_spread_explorer_matches_card(
    params: Mapping[str, SpreadExplorerGameParams], predictions: pd.DataFrame
) -> None:

    _assert_spread_explorer_matches_card(params, predictions)


def _spread_explorer_intro(generated: datetime) -> str:

    stamp = generated.strftime("%Y-%m-%d %H:%M UTC")
    inner = (
        '<div class="prose">'
        "<p>Each game below has a cover-probability chart: drag the slider under it to a "
        "hypothetical line and the highlighted point on the curve, and the sentence beneath "
        "it, update to that line's cover chance -- read off the same model that made the "
        "actual pick.</p>"
        f"<p>Odds reflect what the model knew <b>as of this build, {escape(stamp)}</b> -- "
        "frozen at build time; only your hypothetical line changes. A small push chance at "
        "whole-number lines is left out rather than invented.</p>"
        "</div>"
    )
    return f'<div style="margin-top:16px;">{inner}</div>'


_WEEK1_LOCK_LABEL = "Tuesday, September 8, 2026"

_SEASON_OPS_STEPS: tuple[tuple[str, str], ...] = (
    ("Tue", "the pool's line locks at noon and this card's grading line freezes with it"),
    ("Wed", "the pool revises its own line once; our grading line never moves"),
    ("Thu", "refresh pass before Thursday night, on what changed since Tuesday"),
    ("Sat", "second pass for every game that has not kicked off"),
    (
        "Sun AM",
        "final pass before 4:00 PM ET -- Sunday- and Monday-night games lock there too",
    ),
)


def _movement_policy_note(challengers: Sequence[Mapping[str, Any]]) -> str:

    entry = next(
        (
            candidate
            for candidate in challengers
            if str(candidate.get("challenger_id")) == "model_only_refresh_incumbent"
        ),
        None,
    )
    evidence = entry.get("evidence") if isinstance(entry, dict) else None
    threshold_text = evidence.get("threshold_frozen") if isinstance(evidence, dict) else None
    body = (
        '<div class="prose"><p><b>If lines move late in the week, we follow them.</b> '
        "At each pass we look at the three books that usually move first: if their "
        "middle move is a full point or more off Tuesday's frozen number, the pick "
        "follows them -- and half a point is enough on the biggest spreads, where a "
        "move by those books is the strongest read on the board -- unless an injury "
        "filed since Tuesday points the other way, "
        "which keeps Tuesday's pick. Below that move (or with no fresh lines "
        "captured), the model's own re-run pick plays as always.</p></div>"
    )
    if isinstance(threshold_text, str) and threshold_text.strip():
        body += (
            f'<p class="fine" style="margin-top:8px;">As registered '
            f"({_challenger_display_name('model_only_refresh_incumbent')}): "
            f"{escape(threshold_text)}</p>"
        )
    else:
        body += (
            '<p class="fine" style="margin-top:8px;">Not yet measured on this build -- see the '
            f"{_challenger_display_name('model_only_refresh_incumbent')} candidate rule on the "
            "findings page once it is tracked.</p>"
        )
    return body


def _season_ops_timeline_section(challengers: Sequence[Mapping[str, Any]]) -> str:

    header = _section_header(
        "Season ops",
        "Picks stay editable to kickoff; the grading line freezes Tuesday",
        "Five checkpoints turn that flexibility into the same routine every week.",
        top=24,
    )
    steps = "".join(
        f"<li><b>{escape(day)}</b> -- {escape(words)}.</li>" for day, words in _SEASON_OPS_STEPS
    )
    lock_line = (
        f'<p class="sub" style="margin-top:8px;">Week 1, 2026 locks {escape(_WEEK1_LOCK_LABEL)}.'
        "</p>"
    )
    return (
        header
        + f'<ul style="margin:0;padding-left:18px;" class="sub">{steps}</ul>'
        + lock_line
        + '<div style="margin-top:16px;">'
        + _movement_policy_note(challengers)
        + "</div>"
    )


_ATTRIBUTION_UNAVAILABLE = (
    '<p class="fine" style="color:var(--muted);">Attribution not published.</p>'
)


def _signed_points(value: Any) -> str:

    return delta_html(_number(value))


_MEMBER_WORDS = {
    "coach_fade": "the year-one-coach fade",
    "division_revenge_tilt": "the division-revenge tilt",
    "player_arrests_back_side_policy": "the player-arrest policy",
    "spread_gap_zone_fade": "the mid-spread zone fade",
}


def _member_words(member_id: str) -> str:
    return _MEMBER_WORDS.get(member_id, member_id.replace("_", " "))


def _sentence_case(label: str) -> str:
    stripped = label.strip()
    return stripped[:1].upper() + stripped[1:] if stripped else stripped


def _why_this_pick_panel(
    entry: Mapping[str, Any] | None,
    *,
    interval_text: str = "",
) -> str:

    entry_map = entry if isinstance(entry, Mapping) else None
    candidate_steps = entry_map.get("steps") if entry_map is not None else None
    steps_ok = isinstance(candidate_steps, list) and bool(candidate_steps)
    if not steps_ok or entry_map is None:
        if not interval_text:
            return _ATTRIBUTION_UNAVAILABLE
        return (
            '<details class="why-pick"><summary>Why this pick</summary>'
            f'<div style="margin-top:8px;">'
            f'<p class="fine" style="margin:0 0 8px;">{interval_text}</p>'
            f"{_ATTRIBUTION_UNAVAILABLE}</div></details>"
        )
    steps = candidate_steps

    readouts = []
    edge = _number(entry_map.get("edge_vs_spread"))
    if edge is not None:
        readouts.append(f"model-vs-market edge {abs(edge):.2f} pts")
    distance = _number(entry_map.get("key_number_distance"))
    if distance is not None:
        readouts.append(f"{distance:.2f} pts from the nearest key number")
    if interval_text:
        readouts.append(interval_text)
    readout_html = (
        f'<p class="fine" style="margin:0 0 8px;">{" &middot; ".join(readouts)}</p>'
        if readouts
        else ""
    )

    step_rows = []
    for step in steps or []:
        if not isinstance(step, Mapping):
            continue
        label = escape(_sentence_case(str(step.get("label", ""))))
        delta = _signed_points(step.get("delta_points"))
        cumulative = _signed_points(step.get("cumulative_points"))
        step_rows.append(
            f"<tr><td>{label}</td>"
            f'<td class="num">{delta}</td><td class="num">{cumulative}</td></tr>'
        )
    steps_table = (
        '<table class="data"><thead><tr><th>Step</th><th>Delta (pts)</th>'
        "<th>Cumulative (pts)</th></tr></thead><tbody>" + "".join(step_rows) + "</tbody></table>"
    )

    flip_items = []
    flip_events = entry_map.get("flip_events")
    for event in flip_events if isinstance(flip_events, list) else ():
        if not isinstance(event, Mapping):
            continue
        overlay = escape(str(event.get("overlay", "")))
        note = (
            "flips this pick on its own"
            if bool(event.get("would_flip_alone"))
            else "fires alongside the other members"
        )
        flip_items.append(f"<li>{overlay}: {note}</li>")
    flips_html = (
        '<p class="kicker" style="margin-top:10px;">Overlay events</p><ul>'
        + "".join(flip_items)
        + "</ul>"
        if flip_items
        else ""
    )

    raw_sentences = entry_map.get("rationale_sentences")
    sentences = [
        escape(str(sentence))
        for sentence in (raw_sentences if isinstance(raw_sentences, list) else ())
        if sentence
    ][:3]
    rationale_html = (
        '<div class="marginalia"><p class="kicker" style="margin-top:10px;">Rationale</p><ul>'
        + "".join(f"<li>{sentence}</li>" for sentence in sentences)
        + "</ul></div>"
        if sentences
        else ""
    )

    summary = "Why this pick"
    return (
        f'<details class="why-pick"><summary>{summary}</summary>'
        f'<div style="margin-top:8px;">{readout_html}{steps_table}{flips_html}'
        f"{rationale_html}</div></details>"
    )


def _margin_interval_text(row: pd.Series) -> str:

    def band(low_key: str, high_key: str) -> str | None:
        low, high = _number(row.get(low_key)), _number(row.get(high_key))
        if low is None or high is None:
            return None
        return f"[{low:+.1f}, {high:+.1f}]"

    parts = []
    inner_50 = band("margin_lower_50", "margin_upper_50")
    if inner_50:
        parts.append(f"50% CI {inner_50}")
    inner_80 = band("margin_lower_80", "margin_upper_80")
    if inner_80:
        parts.append(f"80% CI {inner_80}")
    joined = " &middot; ".join(parts)
    return f"cover margin: {joined}" if joined else ""


def _pick_oriented_lines(row: pd.Series, pick_team: str, home: str) -> tuple[str, str | None]:

    home_spread = float(row["spread_line"])
    sign = -1.0 if pick_team == home else 1.0
    market_value = home_spread * sign
    market_text = "pick'em" if market_value == 0 else f"{market_value:+g}"
    fair = _number(row.get("fair_spread"))
    fair_text = None if fair is None else f"{fair * sign:+.1f}"
    return market_text, fair_text


def _game_deep_dive(
    row: pd.Series,
    game_sweep: pd.DataFrame,
    explanation: str,
    *,
    is_best_pick: bool = False,
    best_pick_note: str = "",
    flip: OverlayFlip | None = None,
    arrest_flip: ArrestFlip | None = None,
    production_members: tuple[str, ...] = (),
    spread_explorer_params: SpreadExplorerGameParams | None = None,
) -> tuple[str, dict[str, Any] | None]:

    game_id = str(row["game_id"])
    home, away = str(row["home_team"]), str(row["away_team"])
    market_spread = float(row["spread_line"])
    fair = _number(row.get("fair_spread"))
    residual = _number(row.get("predicted_market_residual")) or 0.0
    pick_team, pick_probability = pick_side(row)
    strong = abs(residual) >= STRONG_LEAN_POINTS
    pick_market_text, pick_fair_text = _pick_oriented_lines(row, pick_team, home)

    if production_members:
        member_text = ", ".join(_member_words(name) for name in production_members)
        explanation_html = (
            '<p class="sub" style="font-weight:600;">One of three production rules applied: '
            f"this game flipped by {escape(member_text)}.</p>"
            '<p class="fine" style="margin-top:6px;">Members are evaluated against the raw '
            "model pick; overlapping triggers are OR-composed and flip the pick exactly "
            "once. The selected archive score is selection-inflated; fresh paired "
            "tracking uses the former coach-to-arrests policy as its control.</p>"
        )
    elif arrest_flip is not None:
        explanation_html = (
            '<p class="sub" style="font-weight:600;">Arrest rule applied: '
            f"flipped from {escape(arrest_flip.original_pick_team)} to "
            f"{escape(arrest_flip.flipped_to_team)}.</p>"
            '<details class="why-pick" style="margin-top:6px;"><summary>Policy evidence'
            "</summary>"
            '<p class="fine" style="margin-top:6px;">The sole affected team had a broad '
            "incident dated 1-14 days before Tuesday. Historically this exact opener-grade "
            f"policy scored {POLICY_OPENER_ACCURACY:.2%} versus "
            f"{POLICY_BASELINE_OPENER_ACCURACY:.2%} for the model baseline "
            f"(+{POLICY_EFFECT_ACCURACY_POINTS:.3f} points, "
            f"{glossary_abbr('P+')} {POLICY_PROBABILITY_POSITIVE:.2f}). Both arms continue to "
            "be tracked prospectively.</p></details>"
        )
    elif flip is not None:
        explanation_html = (
            '<p class="sub" style="font-weight:600;">Coach-fade overlay applied: flipped from '
            f"{escape(flip.year_one_team)} (the model&#8217;s own pick) to "
            f"{escape(flip.opponent_team)}.</p>"
            '<details class="why-pick" style="margin-top:6px;"><summary>Rule evidence</summary>'
            '<p class="fine" style="margin-top:6px;">'
            f"{escape(flip.year_one_team)}&#8217;s head coach is in year 1 and "
            f"{escape(flip.opponent_team)}&#8217;s is not; that matchup has covered only "
            "about 47% of the time against the market's own price in weeks 1-8 since 2018 "
            "-- a real-looking gap, but not yet confirmed outside the years it was found "
            "in. We publish and track both versions of every pick this rule touches."
            "</p></details>"
        )
    elif strong and not explanation:
        explanation_html = ""
    elif strong:
        lean_text = (
            f"We make this line {abs(residual):.1f} points different from the "
            f"market, on the {pick_team} side."
        )
        story = f'<p class="prose" style="margin-top:6px;">{escape(explanation)}</p>'
        explanation_html = (
            '<p class="kicker">What we think the market is missing</p>'
            f'<p class="sub" style="font-weight:600;">{escape(lean_text)}</p>{story}'
        )
    else:
        explanation_html = ""

    pick_is_home = pick_team == home
    quote_label = spread_words(home, away, market_spread)
    points: list[tuple[float, float]]
    if not game_sweep.empty:
        points = [
            (float(offset), float(probability) if pick_is_home else 1.0 - float(probability))
            for offset, probability in zip(
                game_sweep["line_offset"],
                game_sweep["home_cover_probability"],
                strict=True,
            )
        ]
    elif spread_explorer_params is not None:
        points = [
            (
                offset,
                (
                    widget_home_cover_probability(
                        spread_explorer_params.card_line + offset,
                        spread_explorer_params.center,
                        spread_explorer_params.residual_mean,
                        spread_explorer_params.residual_std,
                    )
                    if pick_is_home
                    else 1.0
                    - widget_home_cover_probability(
                        spread_explorer_params.card_line + offset,
                        spread_explorer_params.center,
                        spread_explorer_params.residual_mean,
                        spread_explorer_params.residual_std,
                    )
                ),
            )
            for offset in _COVER_CURVE_FALLBACK_OFFSETS
        ]
    else:
        points = []

    tools: list[str] = []
    chart_payload: dict[str, Any] | None = None
    if points:
        tools.append(
            viz.cover_curve(
                f"cover-{game_id}",
                points,
                quoted_line=0.0,
                quote_label=quote_label,
                pick_text=f"{pick_team} to cover",
                pick_team=pick_team,
                anchor_probability=pick_probability,
                game_id=game_id,
            )
        )
        chart_payload = {
            "home": home,
            "away": away,
            "pickIsHome": pick_is_home,
            "line": market_spread,
        }
        if spread_explorer_params is not None:
            chart_payload.update(
                center=round(spread_explorer_params.center, 6),
                mean=round(spread_explorer_params.residual_mean, 6),
                std=round(spread_explorer_params.residual_std, 6),
            )
    tools.append(
        viz.line_journey(
            opener=market_spread, fair=fair, predicted_close=None, opener_label="market"
        )
    )

    best_note = (
        f'<div style="margin-top:8px;"><p class="fine">{escape(best_pick_note)}</p></div>'
        if is_best_pick and best_pick_note
        else ""
    )

    summary_line = (
        '<p class="sub">Pick <b>'
        f"{escape(pick_team)}</b> ({escape(pick_market_text)}) &middot; covers "
        f'<span class="num">{pick_probability:.0%}</span>'
    )
    if pick_fair_text is not None:
        summary_line += (
            f' &middot; fair {escape(pick_team)} <span class="num">{pick_fair_text}</span>'
        )
    summary_line += "</p>"

    block = (
        f'<section class="deep-game" id="{escape(game_id)}">'
        f'<p class="kicker">{escape(_kickoff_words(row))}</p>'
        f'<h3 class="title">{escape(away)} at {escape(home)}</h3>'
        + summary_line
        + best_note
        + (f'<div style="margin-top:10px;">{explanation_html}</div>' if explanation_html else "")
        + '<details class="line-tools" style="margin-top:12px;">'
        "<summary>Cover odds across hypothetical lines</summary>"
        '<div style="margin-top:8px;display:grid;gap:14px;">'
        + "".join(tools)
        + "</div></details>"
        + "</section>"
    )
    return block, chart_payload


def _week_board(
    ordered: pd.DataFrame,
    flipped_by_game: Mapping[str, object],
    best_pick_id: str | None,
    why_by_game: Mapping[str, str],
    bands: StrengthBands | None = None,
) -> str:

    rows = []
    for _, row in ordered.iterrows():
        game_id = str(row["game_id"])
        home, away = str(row["home_team"]), str(row["away_team"])
        market_spread = float(row["spread_line"])
        pick_team = pick_side(row)[0]
        pick_cell = f'<span class="pick-team">{escape(pick_team)}</span>'
        if best_pick_id is not None and game_id == best_pick_id:
            pick_cell += (
                ' <span class="best-flag" role="img" '
                'aria-label="Best Pick of the week" '
                'title="Best Pick of the week">&#9733;</span>'
            )
        if game_id in flipped_by_game:
            flip_note = "Flipped by a production overlay -- see the note in the deep dive below"
            pick_cell += (
                ' <span class="flip-flag" role="img" '
                f'aria-label="{flip_note}" title="{flip_note}">'
                "&#8646;</span>"
            )
        expansion = why_by_game.get(game_id) or ""
        if not expansion:
            expansion = _why_this_pick_panel(None, interval_text=_margin_interval_text(row))
        is_best = best_pick_id is not None and game_id == best_pick_id
        row_class = "board-game is-best-pick" if is_best else "board-game"
        rows.append(
            f'<tr class="{row_class}">'
            f'<td data-label="Kickoff">{escape(_kickoff_words(row))}</td>'
            f'<td data-label="Matchup"><a href="#{escape(game_id)}">'
            f"{escape(away)} at {escape(home)}</a></td>"
            f'<td data-label="Line" class="num">'
            f"{escape(spread_words(home, away, market_spread))}</td>"
            f'<td data-label="Pick">{pick_cell}</td>'
            f'<td data-label="Strength" class="strength">'
            f"{confidence_meter(row_confidence_word(row, bands))}"
            f"{row_confidence_word(row, bands)}</td>"
            "</tr>"
            f'<tr class="board-sub"><td colspan="5">{expansion}</td></tr>'
        )
    return (
        '<table class="data week-board"><thead><tr>'
        "<th>Kickoff</th><th>Matchup</th><th>Line</th><th>Pick</th><th>Strength</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def load_waterfall_feed_document(artifacts_root: Path) -> dict[str, Any]:

    pointer_path = artifacts_root / "waterfall_feed" / "latest.json"
    try:
        pointer = read_json(pointer_path)
    except (ValueError, OSError):
        return {}
    latest = pointer.get("latest") if isinstance(pointer, dict) else None
    if (
        not isinstance(latest, str)
        or not latest
        or "/" in latest
        or "\\" in latest
        or ".." in latest
    ):
        return {}
    try:
        feed = read_json(artifacts_root / "waterfall_feed" / latest / "feed.json")
    except (ValueError, OSError):
        return {}
    return feed if isinstance(feed, dict) else {}


def waterfall_games_by_id(feed: Mapping[str, Any]) -> dict[str, dict[str, Any]]:

    games = feed.get("games")
    if not isinstance(games, list):
        return {}
    return {
        str(entry["game_id"]): entry
        for entry in games
        if isinstance(entry, dict) and "game_id" in entry
    }


def load_waterfall_feed(artifacts_root: Path) -> dict[str, dict[str, Any]]:

    return waterfall_games_by_id(load_waterfall_feed_document(artifacts_root))


_LEDGER_UNAVAILABLE_HTML = (
    '<div class="card" style="border-left:3px solid var(--warning);margin-top:14px;">'
    '<p class="kicker" style="color:var(--warning);font-weight:700;">'
    "&#9888; MODEL LEDGER UNAVAILABLE</p>"
    '<p class="fine">The challenger registry drifted or could not be read '
    "({detail}); the rest of this page is unaffected.</p></div>"
)


def load_model_ledger_html(artifacts_root: Path) -> str:

    challengers_path = artifacts_root / "prospective" / "challengers.json"
    if not challengers_path.is_file():
        return ""
    try:
        return build_and_render(
            challengers_path,
            _default_weak_signals_registry_path(),
            artifacts_root / "active_ats_model.json",
        )
    except (ValueError, OSError) as error:
        detail = escape(str(error)) or "unknown error"
        return _LEDGER_UNAVAILABLE_HTML.replace("{detail}", detail)


_GLOSSARY: dict[str, str] = {
    "P+": (
        "Our confidence that a measured effect is real rather than luck; "
        "0.50 would be a coin flip. It is not an accuracy rate or a profit claim."
    ),
    "Ledger mini": (
        "A compact slice of the model ledger: candidate picking rules ranked by "
        "their best evidence. The promoted card is what actually plays."
    ),
    "Evidence P+": (
        "The highest P+ recorded across evaluations of this picking rule -- how "
        "confident we are that its effect is real rather than luck."
    ),
    "Challenger watch": (
        "Alternative picking rules tracked alongside this card in prospective "
        "evaluation. None of them change what plays this week."
    ),
}


def glossary_abbr(term: str) -> str:

    try:
        title = _GLOSSARY[term]
    except KeyError as error:
        raise KeyError(f"term {term!r} is not in the site glossary") from error
    return f'<abbr title="{escape(title)}">{escape(term)}</abbr>'


def _ledger_mini_table(model_id: str | None, challengers: Sequence[Mapping[str, Any]]) -> str:

    def _sort_key(entry: tuple[str, str, float | None]) -> tuple[int, float]:
        status, probability = entry[1], entry[2]
        return (0 if status == "SUPERSEDED_BY_PROMOTION" else 1, -(probability or 0.0))

    arms: list[tuple[str, str, float | None]] = []
    if model_id:
        arms.append((model_id, "promoted", None))
    for entry in challengers:
        evidence = entry.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        probability = _number(evidence.get("probability_positive"))
        label = _challenger_display_name(str(entry.get("challenger_id", "unknown")))
        status = str(entry.get("status", "unknown"))
        words = "promoted" if status == "SUPERSEDED_BY_PROMOTION" else _humanize(status).lower()
        arms.append((label, words, probability))
    arms.sort(key=_sort_key)
    rows = []
    for label, status_words, probability in arms[:5]:
        probability_text = viz.p_plus_text(probability) if probability is not None else "--"
        tone = "good" if status_words == "promoted" else "idle"
        status_cell = f"<td>{pill_html(tone, status_words)}</td>"
        rows.append(
            f"<tr><td>{escape(label)}</td>{status_cell}"
            f'<td class="num">{glossary_abbr("P+")} '
            f"{p_plus_html(probability, probability_text)}</td></tr>"
        )
    return (
        '<table class="data"><thead><tr><th>Arm</th><th>Status</th>'
        f"<th>{glossary_abbr('Evidence P+')}</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


_CHALLENGER_WATCH_VISIBLE = 6


def _challenger_evidence_strength(entry: Mapping[str, Any]) -> float:

    evidence = entry.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    probability = _number(evidence.get("probability_positive"))
    return -1.0 if probability is None else abs(probability - 0.5)


def _challenger_watch_items(
    challengers: Sequence[Mapping[str, Any]],
    previews: Mapping[str, str],
) -> list[str]:

    active = [entry for entry in challengers if str(entry.get("status")) == "ACTIVE_PROSPECTIVE"]
    ordered = sorted(
        active,
        key=lambda entry: (
            -_challenger_evidence_strength(entry),
            str(entry.get("challenger_id", "")),
        ),
    )
    items = []
    for entry in ordered:
        challenger_id = str(entry.get("challenger_id", "unknown"))
        evidence = entry.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        probability = _number(evidence.get("probability_positive"))
        probability_text = (
            f" &middot; {glossary_abbr('P+')} "
            f"{p_plus_html(probability, viz.p_plus_text(probability))}"
            if probability is not None
            else ""
        )
        preview = previews.get(challenger_id, "")
        preview_text = (
            f" &middot; {escape(_first_sentence(preview, max_len=60))}" if preview else ""
        )
        items.append(
            f"<li>{escape(_challenger_display_name(challenger_id))}"
            f"{probability_text}{preview_text}</li>"
        )
    return items


def _challenger_watch_panel(
    challengers: Sequence[Mapping[str, Any]],
    week_previews: Mapping[str, str] | None,
) -> str:

    items = _challenger_watch_items(challengers, week_previews or {})
    if not items:
        body = '<p class="fine">No live challengers registered.</p>'
    else:
        body = (
            '<ul style="margin:0;padding-left:18px;" class="sub">'
            + "".join(items[:_CHALLENGER_WATCH_VISIBLE])
            + "</ul>"
        )
        if len(items) > _CHALLENGER_WATCH_VISIBLE:
            body += (
                '<details class="table-view"><summary>show all</summary>'
                '<ul style="margin:8px 0 0;padding-left:18px;" class="sub">'
                + "".join(items[_CHALLENGER_WATCH_VISIBLE:])
                + "</ul></details>"
            )
    return (
        '<section class="panel panel-watch">'
        '<h2 class="title" style="font-size:17px;margin:0 0 2px;">'
        f"{glossary_abbr('Challenger watch')}</h2>"
        '<p class="fine" style="margin:0 0 8px;">Tracked alongside the card; none change '
        f"what plays.</p>{body}</section>"
    )


_CROWNED_LABEL = "PLAYED CARD \u2014 HONEST EXPECTATION VS TUESDAY LINES"


def _crowned_stat_block(played_chain_accuracy: float | None) -> str:

    if played_chain_accuracy is not None:
        measured_line = (
            '<p class="sub" style="font-size:14px;margin-top:6px;"><strong>'
            "Measured chain history: "
            f'<span class="num">{played_chain_accuracy:.1%}</span></strong></p>'
        )
    else:
        measured_line = (
            '<p class="sub" style="font-size:14px;margin-top:6px;"><strong>'
            "Raw chain baseline: "
            '<span class="num">Unavailable</span></strong></p>'
        )
    return (
        '<div class="card" style="margin-top:10px;">'
        f'<p class="kicker">{escape(_CROWNED_LABEL)}</p>'
        '<div class="num" style="font-size:24px;font-weight:600;line-height:1.15;">'
        f"{PLAYED_CARD_EXPECTATION_HERO}</div>"
        '<p class="sub" style="max-width:44ch;">Planning estimate for the played card.</p>'
        '<p class="fine" style="margin-top:6px;">'
        '<a href="models.html">What this number means &#8594;</a></p>' + measured_line + "</div>"
    )


def render_picks_page(
    predictions: pd.DataFrame,
    sweep: pd.DataFrame | None = None,
    explanations: Mapping[str, str] | None = None,
    *,
    season: int | None = None,
    week: int | None = None,
    model_id: str | None = None,
    active_model: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
    data_root: Path | None = None,
    artifacts_root: Path | None = None,
    overlay: OverlayResult | None = None,
    arrest_overlay: ArrestOverlayResult | None = None,
    production_overlay: FourOverlayCompositionResult | None = None,
    nomination: BestPickNomination | None = None,
    spread_explorer: Mapping[str, SpreadExplorerGameParams] | None = None,
    challengers: Sequence[Mapping[str, Any]] = (),
    waterfall_feed: Mapping[str, Mapping[str, Any]] | None = None,
    challenger_week_previews: Mapping[str, str] | None = None,
    recent_form_text: str | None = None,
    played_chain_accuracy: float | None = None,
    strength_bands: StrengthBands | None = None,
) -> str:

    explanations = explanations or {}
    sweep = sweep if sweep is not None else pd.DataFrame()
    metadata = metadata or {}
    spread_explorer = spread_explorer or {}
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)

    model_text = f"model <code>{escape(model_id)}</code>" if model_id else "model unknown"
    if predictions.empty:
        body = viz.page_header("This week", "No pick card yet") + viz.empty_state(
            "No games are scheduled for this week's forecast yet",
            "Once the week's opening line is captured and a forecast card is built, this "
            "page fills in by itself. History is empty in the meantime.",
        )
        return _page(
            current=PICKS_PAGE,
            body=body,
            generated=generated,
            footer_note=model_text,
        )

    game_type = str(predictions["game_type"].iloc[0]) if "game_type" in predictions else "REG"
    week_label = _WEEK_LABELS.get(game_type, f"Week {week}")
    season_label = f"{season} · " if season is not None else ""

    if overlay is None:
        overlay = resolve_overlay(predictions, data_root)
    if arrest_overlay is None:
        arrest_overlay = resolve_player_arrests_overlay(
            overlay.overlaid_predictions, data_root, now=generated
        )
    recommendations = (
        production_overlay.overlaid_predictions
        if production_overlay is not None
        else arrest_overlay.overlaid_predictions
    )
    flipped_by_game = {flip.game_id: flip for flip in overlay.flips}
    arrest_flipped_by_game = {flip.game_id: flip for flip in arrest_overlay.flips}
    production_members_by_game = (
        {game.game_id: game.member_ids for game in production_overlay.games}
        if production_overlay is not None
        else {}
    )

    sort_columns = [column for column in ("kickoff", "game_id") if column in recommendations]
    ordered = (
        recommendations.sort_values(sort_columns, na_position="last")
        if sort_columns
        else recommendations
    )

    if strength_bands is None and artifacts_root is not None:
        strength_bands = served_strength_bands(artifacts_root, active_model)
    strong_count = sum(
        1 for _, row in ordered.iterrows() if row_confidence_word(row, strength_bands) == "strong"
    )
    header = viz.page_header(
        f"{season_label}{week_label} · {len(recommendations)} games",
        "This week's picks",
        "Picks are graded against Tuesday-frozen lines all season.",
    )

    best_pick_id: str | None = None
    best_pick_note = ""
    if game_type == "REG":
        resolved_nomination = (
            nomination
            if nomination is not None
            else resolve_nomination(predictions, sweep, metadata, data_root)
        )
        best_pick_id = resolved_nomination.active_game_id
        if best_pick_id is not None:
            if resolved_nomination.active_rule == "v2":
                best_pick_note = f"This pick was {resolved_nomination.method_note}"
            else:
                tie = (
                    f" {resolved_nomination.active_tie_note}"
                    if resolved_nomination.active_tie_note
                    else ""
                )
                best_pick_note = (
                    "This is the pick whose edge survives the widest range of line "
                    "movement -- the best-measured lever among forced picks, budgeted at "
                    "roughly +0.9 points, not the +8.7 once recorded before a tie-break "
                    f"audit.{tie}"
                )

    has_sweep = not sweep.empty and {"game_id", "line_offset", "home_cover_probability"}.issubset(
        sweep.columns
    )

    why_by_game = {
        str(row["game_id"]): _why_this_pick_panel(
            (waterfall_feed or {}).get(str(row["game_id"])),
            interval_text=_margin_interval_text(row),
        )
        for _, row in ordered.iterrows()
    }
    deep_blocks = []
    chart_payloads: dict[str, dict[str, Any]] = {}
    for _, row in ordered.iterrows():
        game_id = str(row["game_id"])
        game_sweep = pd.DataFrame()
        if has_sweep:
            game_sweep = sweep.loc[
                sweep["game_id"].astype(str).eq(game_id)
                & sweep["line_offset"].abs().le(SWEEP_HALF_WIDTH)
            ].sort_values("line_offset")
        block, chart_payload = _game_deep_dive(
            row,
            game_sweep,
            explanations.get(game_id, ""),
            is_best_pick=best_pick_id is not None and game_id == best_pick_id,
            best_pick_note=best_pick_note,
            flip=flipped_by_game.get(game_id),
            arrest_flip=arrest_flipped_by_game.get(game_id),
            production_members=production_members_by_game.get(game_id, ()),
            spread_explorer_params=spread_explorer.get(game_id),
        )
        deep_blocks.append(block)
        if chart_payload is not None:
            chart_payloads[game_id] = chart_payload

    flipped_game_ids = (
        set(production_members_by_game)
        if production_overlay is not None
        else set(flipped_by_game) | set(arrest_flipped_by_game)
    )
    week_board = _week_board(
        ordered, dict.fromkeys(flipped_game_ids), best_pick_id, why_by_game, strength_bands
    )
    board_legend = (
        '<p class="fine" style="margin-top:8px;">&#9733; best pick &middot; &#8646; flipped '
        "by an overlay rule &middot; strength runs slight &lt; lean &lt; strong, by where "
        "this pick's cover chance sits among the model's own picks over the past six "
        "seasons.</p>"
    )

    composition = ["Synchronized with the active model"]
    if strong_count:
        composition.append(f"{strong_count} strong lean{'s' if strong_count != 1 else ''}")
    if production_overlay is not None:
        composition.append(
            f"{production_overlay.flip_count} pick"
            f"{'s' if production_overlay.flip_count != 1 else ''} flipped by the fix-up rules"
        )
    elif overlay.flip_count:
        composition.append(
            f"{overlay.flip_count} pick{'s' if overlay.flip_count != 1 else ''} flipped by "
            "the coach-fade overlay"
        )
    if arrest_overlay.enabled and production_overlay is None:
        composition.append(
            f"player-arrest policy active &middot; {arrest_overlay.flip_count} pick"
            f"{'s' if arrest_overlay.flip_count != 1 else ''} flipped this week"
        )
    best_callout = ""
    if best_pick_id is not None:
        best_row = recommendations.loc[recommendations["game_id"].astype(str).eq(best_pick_id)]
        if not best_row.empty:
            best_team, _ = pick_side(best_row.iloc[0])
            best_callout = (
                '<div class="card"><p class="kicker">The pool scores one Best Pick a week</p>'
                '<p class="sub" style="font-weight:600;color:var(--good-text);">'
                '&#9733; BEST PICK OF THE WEEK: <span class="num">'
                f"{escape(best_team)}</span></p>"
                f'<p class="fine" style="margin-top:6px;">{escape(best_pick_note)}</p></div>'
            )
    summary_panel = (
        '<section class="panel panel-summary">'
        '<h2 class="title" style="font-size:17px;margin:0 0 4px;">At a glance</h2>'
        + _crowned_stat_block(played_chain_accuracy)
        + '<p class="sub" style="margin-top:8px;">'
        + " &middot; ".join(escape(part) for part in composition)
        + "</p>"
        + (
            f'<p class="fine" style="margin-top:8px;">{escape(recent_form_text)}</p>'
            if recent_form_text
            else ""
        )
        + best_callout
        + "</section>"
    )
    ledger_panel = (
        '<section class="panel panel-ledger">'
        '<h2 class="title" style="font-size:17px;margin:0 0 2px;">'
        f"{glossary_abbr('Ledger mini')}</h2>"
        '<p class="fine" style="margin:0 0 8px;">Top arms by best evidence; the promoted '
        f'card plays. <a href="{MODELS_PAGE}">Full ledger</a>.</p>'
        + _ledger_mini_table(model_id, challengers)
        + "</section>"
    )
    watch_panel = _challenger_watch_panel(challengers, challenger_week_previews)
    grid = (
        '<div class="ledger-grid">'
        + summary_panel
        + f'<section class="panel panel-board">{week_board}{board_legend}</section>'
        + ledger_panel
        + watch_panel
        + "</div>"
    )

    ops_timeline = _season_ops_timeline_section(challengers)
    spread_explorer_intro = _spread_explorer_intro(generated) if chart_payloads else ""
    deep_dive = (
        _section_header("Game notes", "One block per game", "Anchored from the board above.")
        + spread_explorer_intro
        + "".join(deep_blocks)
    )

    return _page(
        current=PICKS_PAGE,
        body=(header + grid + deep_dive + ops_timeline),
        generated=generated,
        footer_note=(
            f"{model_text} &middot; lines are home-oriented "
            "spreads at card-build time; the pool's exact number can differ by a half point"
        ),
        scripts=viz.cover_curve_script(chart_payloads),
    )


def _rows(cards: Sequence[str], *, per_row: int = 2) -> str:

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
        f'<h2 class="title" style="margin-bottom:6px;">{escape(title)}</h2>'
        f'<p class="sub">{escape(sub)}</p></div>'
    )


def _verdict_chip(group: VerdictGroup) -> str:

    if group.chip_kind in {"good", "warning"}:
        return viz.status_line(group.chip_kind, group.chip_label)
    dot = (
        '<span class="dot" style="background:var(--muted);"></span>'
        if group.chip_kind == "muted"
        else ""
    )
    return f'<span class="chip">{dot}{escape(group.chip_label)}</span>'


def _findings_hero(artifacts_root: Path | None = None) -> str:
    hero_tiles = HERO_TILES
    if artifacts_root is not None:
        active = load_active_ats_model(artifacts_root)
        if active and active.get("feature_table_sha256"):
            baseline = load_baseline_measurement(artifacts_root, active)
            interval = baseline.season_interval
            context = (
                f"95% range [{interval[0]:.2%}, {interval[1]:.2%}]."
                if interval is not None
                else f"{baseline.games:,} paired games."
            )
            hero_tiles = baseline_hero_tiles(f"{baseline.accuracy:.1%}", context)
    tiles = _rows(
        [
            viz.stat_tile(
                tile.kicker,
                tile.value,
                tile.context,
                delta_text=tile.delta_text,
                delta_good=tile.delta_good,
            )
            for tile in hero_tiles
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
        + f'<div class="lede-grid">{story}{legend}</div>'
    )


def _research_funnel_section(
    *, total_signals: int, active_challengers: int, has_active_model: bool
) -> str:

    tiles = _rows(
        [
            viz.stat_tile(
                "Signals recorded",
                f"{total_signals:,}",
                "Every effect this project has measured and logged, resolved or not -- "
                "nothing that gets tested is thrown away, including the negatives.",
            ),
            viz.stat_tile(
                "Live 2026 challengers",
                str(active_challengers),
                "Alternative picking rules and pick-flip overlays -- 'challengers' -- riding "
                "along the model's real weekly card this season, scored against it game for "
                "game. None of them change what actually gets played.",
            ),
            viz.stat_tile(
                "Active model",
                "1" if has_active_model else "0",
                "The one configuration whose picks are the ones actually published each "
                "week. Everything else here is either a past measurement or a challenger "
                "riding alongside it, never the pick itself.",
            ),
        ],
        per_row=3,
    )
    header = _section_header(
        "The research pipeline",
        "From every idea tested to what's actually played",
        "Three honest counts, computed fresh from the same files every other section on "
        "this page reads -- nobody updates these by hand.",
        top=8,
    )
    return header + tiles


def _emphasized(text: str) -> str:

    escaped = escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def _finding_card(finding: Finding, group: VerdictGroup) -> str:
    inner = (
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        'gap:14px;margin-bottom:10px;">'
        f'<p class="title" style="max-width:38ch;">{escape(finding.question)}</p>'
        f'<span style="flex:none;">{_verdict_chip(group)}</span>'
        "</div>"
        f'<div class="prose"><p>{_emphasized(finding.plain_answer)}</p></div>'
        '<details class="table-view"><summary>'
        f"{escape(DETAIL_SUMMARY_LABEL)}</summary>"
        f'<p class="fine" style="margin:10px 0 0;max-width:68ch;">{_emphasized(finding.detail)}</p>'
        f'<p class="fine" style="margin:8px 0 0;">{escape(SOURCE_LABEL)}: '
        f'<span style="color:var(--ink-2);">{escape(finding.source)}</span></p>'
        "</details>"
    )
    return viz.card(inner, accent=group.verdict == "helps")


def _group_section(group: VerdictGroup) -> str:
    findings = findings_for(group.verdict)
    header = _section_header(f"{group.kicker} · {len(findings)}", group.title, group.blurb)
    return header + _rows([_finding_card(finding, group) for finding in findings])


_EFFECT_UNIT_WORDS = {
    "accuracy_points": "accuracy points",
    "ats_points": "line points",
    "brier": "Brier-score points",
    "log_loss": "log-loss points",
    "mae": "points of average error",
}


def _lead_direction_sentence(probability_positive: float) -> str:

    if probability_positive >= 0.5:
        return (
            f"Leans FOR the pattern described below -- {probability_positive:.0%} "
            "confidence in that direction (not yet resolved; see the interval)."
        )
    against = 1.0 - probability_positive
    return (
        "Leans AGAINST the pattern described below -- read this as a lead for the "
        f"OTHER side, {against:.0%} confidence in that direction (not yet resolved; "
        "see the interval)."
    )


def _effect_whisker(
    effect: float, interval: tuple[float, float] | None, *, width: int = 220
) -> str:

    lo, hi = interval if interval is not None else (effect, effect)
    span_lo, span_hi = min(lo, effect, 0.0), max(hi, effect, 0.0)
    span = (span_hi - span_lo) or max(abs(effect), 1.0)
    pad = span * 0.18
    axis_lo, axis_hi = span_lo - pad, span_hi + pad
    axis_span = (axis_hi - axis_lo) or 1.0

    def pct(value: float) -> float:
        return (value - axis_lo) / axis_span * 100.0

    whisker_html = (
        f'<div style="position:absolute;left:{pct(lo):.2f}%;'
        f"width:{max(pct(hi) - pct(lo), 0.6):.2f}%;top:50%;height:2px;"
        'background:var(--series-model);transform:translateY(-50%);"></div>'
        if interval is not None
        else ""
    )
    interval_words = f", interval {lo:+.3f} to {hi:+.3f}" if interval is not None else ""
    return f"""
<div style="position:relative;height:20px;max-width:{width}px;" role="img"
     aria-label="Effect {effect:+.3f}{interval_words}, zero marked">
  <div style="position:absolute;left:{pct(0.0):.2f}%;top:-2px;bottom:-2px;width:0;
              border-left:1px dashed var(--baseline);"></div>
  {whisker_html}
  <div style="position:absolute;left:{pct(effect):.2f}%;top:50%;width:9px;height:9px;
              border-radius:50%;background:var(--series-model);border:2px solid var(--surface);
              transform:translate(-50%,-50%);"></div>
  <span class="fine" style="position:absolute;left:{pct(0.0):.2f}%;top:100%;
        transform:translateX(-50%);font-size:9px;">0</span>
</div>
"""


def _era_magnitude_row(rows: Sequence[EraMagnitude]) -> str:

    if not rows:
        return ""
    items = "".join(
        '<div style="min-width:118px;">'
        f'<p class="fine num" style="margin-bottom:4px;">{escape(row.era_label)}</p>'
        f"{_effect_whisker(row.effect, row.interval, width=140)}"
        "</div>"
        for row in rows
    )
    return (
        '<div style="margin:10px 0 8px;padding-top:8px;border-top:1px solid var(--grid);">'
        '<p class="kicker">Same pattern, three eras</p>'
        '<p class="fine" style="margin-bottom:8px;">Magnitude moving across eras is the '
        "expected shape for a real effect -- a weaker-reading era is not the same thing as "
        "no effect there.</p>"
        f'<div class="row" style="gap:14px;flex-wrap:wrap;">{items}</div></div>'
    )


def _watching_lead_card(
    lead: WatchingLead, blurb: LeadBlurb | None, era_rows: Sequence[EraMagnitude] = ()
) -> str:
    units_words = _EFFECT_UNIT_WORDS.get(lead.effect_units, lead.effect_units)
    interval_text = (
        f"95% [{lead.interval[0]:+.2f}, {lead.interval[1]:+.2f}]"
        if lead.interval is not None
        else "no interval recorded"
    )
    league_words = "NFL" if lead.league == "nfl" else lead.league.upper()
    headline_text = blurb.text if blurb is not None else lead.description
    registry_link = (
        '<p class="fine" style="margin-top:8px;"><a href="../registry/weak_signals.json">'
        "details in the research registry</a></p>"
    )
    inner = (
        f'<p class="prose" style="margin-bottom:6px;">{escape(headline_text)}</p>'
        '<p class="fine" style="margin-bottom:8px;">'
        f"{escape(_lead_direction_sentence(lead.probability_positive))}</p>"
        f'<div style="margin-bottom:10px;">{_effect_whisker(lead.effect, lead.interval)}</div>'
        '<div class="row" style="gap:16px;flex-wrap:wrap;">'
        '<div><p class="kicker">Effect</p>'
        f'<p class="sub num">{delta_html(lead.effect)} {escape(units_words)}</p></div>'
        '<div><p class="kicker">Interval</p>'
        f'<p class="sub num">{interval_text}</p></div>'
        '<div><p class="kicker">Chance it is real</p>'
        f'<p class="sub num">P+ '
        f"{p_plus_html(lead.probability_positive, viz.p_plus_text(lead.probability_positive))}"
        "</p></div>"
        '<div><p class="kicker">Where measured</p>'
        f'<p class="sub">{escape(league_words)}, {lead.seasons[0]}-{lead.seasons[1]}</p></div>'
        "</div>" + _era_magnitude_row(era_rows) + registry_link
    )
    return viz.card(inner)


def _watching_section(
    leads: Sequence[WatchingLead],
    *,
    total_signals: int,
    shown: int,
    blurbs_by_signal: Mapping[str, LeadBlurb] | None = None,
    era_magnitude: Mapping[str, Sequence[EraMagnitude]] | None = None,
) -> str:

    if not leads:
        return ""
    blurbs_by_signal = blurbs_by_signal or {}
    era_magnitude = era_magnitude or {}
    header = _section_header(
        "What we're watching",
        "The open leads, generated fresh every time this page builds",
        "Every card below comes straight from registry/weak_signals.json at build time -- "
        "nobody typed these in, and nobody has to update them when new evidence is recorded. "
        "Each is 'unresolved_below_power': the interval crosses zero, which at this "
        "evaluator's roughly 2-point resolution is the EXPECTED shape for a real small "
        "signal, not a verdict either way. Ranked by how far the lean sits from a coin flip "
        "in EITHER direction -- a lead near 0% is exactly as strong as one near 100%, just "
        "pointed the other way.",
        top=42,
    )
    count_line = (
        f'<p class="fine" style="margin:-8px 0 12px;">{total_signals} recorded signals; '
        f"{shown} leads shown here; the registry is the full record.</p>"
    )
    cards = _rows(
        [
            _watching_lead_card(
                lead,
                blurbs_by_signal.get(lead.name),
                _era_magnitude_for_lead(lead.name, era_magnitude),
            )
            for lead in leads
        ]
    )
    return header + count_line + cards


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


def render_findings_page(
    *,
    generated_at: datetime | None = None,
    registry_root: Path | None = None,
    weak_signal_registry: WeakSignalRegistry | None = None,
    challengers: Sequence[Mapping[str, Any]] = (),
    challenger_week_previews: Mapping[str, str] | None = None,
    challenger_prospective_records: Mapping[str, str] | None = None,
    artifacts_root: Path | None = None,
    active_model_id: str | None = None,
) -> str:

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)

    registry = (
        weak_signal_registry
        if weak_signal_registry is not None
        else load_weak_signal_registry(registry_root)
    )
    entries = load_all_entries(
        registry_root=registry_root, weak_signal_registry=registry, challengers=challengers
    )
    validate_curation(FINDINGS, entries)
    validate_curation(LEAD_BLURBS, entries)

    leads = top_open_leads(registry)
    blurbs_by_signal = {blurb.weak_signal_name: blurb for blurb in LEAD_BLURBS}
    era_magnitude = load_era_magnitude_profile(artifacts_root) if artifacts_root is not None else {}
    active_challengers = sum(
        1 for entry in challengers if str(entry.get("status")) == "ACTIVE_PROSPECTIVE"
    )
    body = (
        _findings_hero(artifacts_root)
        + '<p class="sub" style="max-width:70ch;margin:-6px 0 0;">The evidence library '
        "&#8212; what the bare model does, what we have learned, and the leads still "
        'open. The story of the edge itself: <a href="models.html">How good is '
        "this, honestly? &#8594;</a></p>"
        + _research_funnel_section(
            total_signals=len(registry.signals),
            active_challengers=active_challengers,
            has_active_model=bool(active_model_id),
        )
        + "".join(_group_section(group) for group in GROUPS)
        + _watching_section(
            leads,
            total_signals=len(registry.signals),
            shown=len(leads),
            blurbs_by_signal=blurbs_by_signal,
            era_magnitude=era_magnitude,
        )
        + _challengers_section(
            challengers,
            week_previews=challenger_week_previews,
            prospective_records=challenger_prospective_records,
        )
        + _honesty_section()
    )
    return _page(
        current=FINDINGS_PAGE,
        body=body,
        generated=generated,
        footer_note="every claim traces to a committed record in this repository",
    )


def _spaced(inner: str) -> str:

    return f'<div style="margin-top:16px;">{inner}</div>'


def _humanize(token: str) -> str:
    return token.replace("_", " ").replace("|", " -- ").replace("=", " ")


def humanize_identifier(token: str) -> str:

    return _humanize(token)


_CHALLENGER_DISPLAY_NAMES = CHALLENGER_DISPLAY_NAMES


def _challenger_display_name(challenger_id: str) -> str:
    return _CHALLENGER_DISPLAY_NAMES.get(challenger_id, _humanize(challenger_id))


_CHALLENGER_BLURBS: dict[str, str] = {
    "mod07_weak_signal_stack": (
        "Tracks the active model's own weak-signal stack as its own separate "
        "prospective arm, so the 2026 season scores it cleanly outside the "
        "already-spent historical research windows."
    ),
    "hc_year_one_fade_overlay": (
        "Fades first-year head coaches on the road, weeks 1-8: when the model's own pick "
        "sides with a rookie coach's team against an opponent that kept its coach, this "
        "flips the pick to the other side. It is both a separately tracked attribution "
        "arm and one member of the published three-adjustment policy."
    ),
    "best_pick_nomination_v2": (
        "Chooses the week's bonus Best Pick by calibrated win probability among the games "
        "the model and market agree on most, with no limit on how big the spread is. The "
        "played card now adds that limit, so this tracks the star it would have given "
        "otherwise, on the same weeks."
    ),
    "injury_value_lost_tilt_overlay": (
        "Nudges the pick toward whichever team lost less value to injury, using a "
        "parameter-free read of the injury report."
    ),
    "division_revenge_tilt_overlay": (
        "Nudges the pick toward a team that lost to this same opponent the last time "
        "they played -- a 'revenge game' tilt. It is also one member of the published "
        "three-adjustment policy."
    ),
    "backup_qb_fade_overlay": (
        "Fades a team starting a backup quarterback against an opponent starting its usual starter."
    ),
    "surface_switch_tilt_overlay": (
        "Nudges the pick toward the home team when a visiting team that normally plays "
        "on grass switches onto turf."
    ),
    "overlay_four_member_union_retired_20260907": (
        "The former card keeps its mid-spread fade alongside coach, division revenge "
        "and player-arrest adjustments, measured against the current card on the same games."
    ),
    "overlay_three_member_union_retired_20260909": (
        "The former card used only the coach, division revenge and player-arrest "
        "adjustments, measured against the current card on the same games."
    ),
    "spread_gap_zone_fade_overlay": (
        "Flips every pick where the market's spread sits between 7.5 and 10 points, "
        "regardless of which side the model liked. This unexplained threshold flip "
        "is retired from the played card and remains a prospective challenger."
    ),
    "overlay_production_chain_coach_arrest_incumbent": (
        "Tracks the exact former production policy -- coach fade followed by the arrest "
        "policy -- against the newly played three-member card on the same fresh games."
    ),
    "interim_hc_first_game_tilt_overlay": (
        "Nudges the pick toward a team playing its first game under a newly appointed "
        "interim head coach -- teams have historically covered that specific first game, "
        "even though the effect fades away for every game after it."
    ),
    "forecast_weather_kn_warm_team_cold_late_tilt": (
        "Nudges the pick toward the home team when a warm-winter-metro visitor plays "
        "outdoors, late in the season, in a live forecast at or below 35F -- the "
        "strongest, best-powered read in this project's whole forecast-weather family."
    ),
    "forecast_weather_kn_precip_high_total_tilt": (
        "Nudges the pick toward the home team in an outdoor game with a high live "
        "forecast rain/snow probability and a high market total -- a newer, less-tested "
        "read that shares its live weather fetch with the warm-team-cold-late tilt above."
    ),
    "player_qb_continuity|ridge_alpha=1|calibration=none": (
        "A different regularization strength for the QB-continuity feature. Its "
        "measured improvement did not survive a predeclared replication on held-out "
        "seasons, so it stays off the card."
    ),
    "injury_signal_refresh_tilt": (
        "At each late-week refresh pass, flips the model's own pick when a fresh "
        "asymmetric injury report or news signal turns against it -- testing whether "
        "acting on injury news itself beats waiting for the market's line to confirm it."
    ),
}


def _challenger_blurb(challenger_id: str) -> str:
    return _CHALLENGER_BLURBS.get(
        challenger_id,
        "A prospective challenger tracked alongside the active model; see its record below.",
    )


def challenger_blurb(challenger_id: str) -> str:

    return _challenger_blurb(challenger_id)


_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


_SENTENCE_OVERSHOOT = 24


def _first_sentence(text: str, *, max_len: int = 260) -> str:

    collapsed = " ".join(text.split())
    match = _SENTENCE_END.search(collapsed)
    if match and match.end() <= max_len + _SENTENCE_OVERSHOOT:
        return collapsed[: match.end()]
    if len(collapsed) <= max_len:
        return collapsed
    head = collapsed[:max_len]
    cut = head.rfind(" ")
    if cut > max_len // 2:
        head = head[:cut]
    return head.rstrip(" ,;:([-") + "..."


_CAVEAT_KEY_SUFFIXES = ("_caveat", "_disclosure")
_CAVEAT_KEY_EXACT = ("caveats",)


def _evidence_caveat_chips(evidence: Mapping[str, Any]) -> list[str]:

    chips = []
    for key, value in evidence.items():
        if not isinstance(value, str) or not value.strip():
            continue
        lowered = key.lower()
        if lowered in _CAVEAT_KEY_EXACT or any(
            lowered.endswith(suffix) for suffix in _CAVEAT_KEY_SUFFIXES
        ):
            chips.append(_humanize(key))
    return chips


def _opener_close_divergence_chip(evidence: Mapping[str, Any]) -> str | None:

    opener_blocks = [
        value
        for key, value in evidence.items()
        if isinstance(value, dict) and "opener" in key.lower()
    ]
    close_blocks = [
        value
        for key, value in evidence.items()
        if isinstance(value, dict) and "close" in key.lower()
    ]
    if not opener_blocks or not close_blocks:
        return None

    def _first_probability(blocks: list[dict[str, Any]]) -> float | None:
        for block in blocks:
            probability = _number(block.get("probability_positive"))
            if probability is not None:
                return probability
        return None

    opener_p, close_p = _first_probability(opener_blocks), _first_probability(close_blocks)
    if opener_p is not None and close_p is not None and (opener_p - 0.5) * (close_p - 0.5) < 0:
        return "opener/close disagree in sign"
    return "graded at both opener and close"


def _challenger_card(
    entry: Mapping[str, Any],
    *,
    week_preview: str,
    prospective_record_text: str,
) -> str:

    challenger_id = str(entry.get("challenger_id", "unknown"))
    label = _challenger_display_name(challenger_id)
    status = str(entry.get("status", "unknown"))
    status_words = _humanize(status).lower()
    is_active = status == "ACTIVE_PROSPECTIVE"

    evidence = entry.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    classification = evidence.get("classification") or evidence.get("registry_verdict")
    probability = evidence.get("probability_positive")

    status_chip = (
        viz.status_line("good", status_words)
        if is_active
        else f'<span class="chip">{escape(status_words)}</span>'
    )
    evidence_chips = []
    if classification:
        evidence_chips.append(f'<span class="chip">{escape(_humanize(str(classification)))}</span>')
    if isinstance(probability, int | float):
        evidence_chips.append(
            '<span class="chip">P+ '
            f"{p_plus_html(float(probability), viz.p_plus_text(float(probability)))}</span>"
        )
    divergence_chip = _opener_close_divergence_chip(evidence)
    if divergence_chip:
        evidence_chips.append(f'<span class="chip">{escape(divergence_chip)}</span>')
    caveat_labels = list(
        dict.fromkeys(_evidence_caveat_chips(entry) + _evidence_caveat_chips(evidence))
    )
    for caveat_label in caveat_labels:
        evidence_chips.append(f'<span class="chip">{escape(caveat_label)}</span>')

    blurb_text = escape(_challenger_blurb(challenger_id))
    parts = [
        '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;'
        'align-items:baseline;margin-bottom:6px;">'
        f'<p class="title" style="font-size:17px;">{escape(label)}</p>{status_chip}</div>',
        f'<p class="prose" style="margin-bottom:8px;">{blurb_text}</p>',
    ]
    if evidence_chips:
        parts.append(
            '<div class="row" style="gap:6px;flex-wrap:wrap;margin-bottom:6px;">'
            f"{''.join(evidence_chips)}</div>"
        )
    if is_active and week_preview:
        parts.append(
            '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--grid);">'
            '<p class="kicker">This week</p>'
            f'<p class="sub">{escape(week_preview)}</p></div>'
        )
    if not is_active:
        reason_text = str(entry.get("status_reason_update") or entry.get("status_reason") or "")
        reason_lead = _first_sentence(reason_text) if reason_text else "No reason recorded."
        details = (
            '<details class="table-view" style="margin-top:6px;">'
            "<summary>Full reason</summary>"
            f'<p class="fine" style="margin-top:6px;max-width:68ch;">{escape(reason_text)}</p>'
            "</details>"
            if reason_text and reason_text != reason_lead
            else ""
        )
        parts.append(
            '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--grid);">'
            '<p class="kicker">Why it is not live</p>'
            f'<p class="sub">{escape(reason_lead)}</p>{details}</div>'
        )
    parts.append(
        '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--grid);">'
        '<p class="kicker">2026 prospective record</p>'
        f'<p class="sub">{escape(prospective_record_text)}</p></div>'
    )
    card_html = viz.card("".join(parts))
    if not is_active:
        card_html = f'<div style="opacity:0.6;">{card_html}</div>'
    return card_html


_PENDING_PROSPECTIVE_RECORD = (
    "Not scored yet this season -- this fills in automatically once games are played "
    "and picks are recorded and settled each week."
)


def _challengers_section(
    challengers: Sequence[Mapping[str, Any]],
    *,
    week_previews: Mapping[str, str] | None = None,
    prospective_records: Mapping[str, str] | None = None,
) -> str:

    if not challengers:
        return ""
    week_previews = week_previews or {}
    prospective_records = prospective_records or {}
    cards = _rows(
        [
            _challenger_card(
                entry,
                week_preview=week_previews.get(str(entry.get("challenger_id")), ""),
                prospective_record_text=prospective_records.get(
                    str(entry.get("challenger_id")), _PENDING_PROSPECTIVE_RECORD
                ),
            )
            for entry in challengers
        ]
    )
    intro = viz.card(
        '<p class="kicker">The live test starts Sep 8, 2026</p>'
        '<p class="title" style="margin-bottom:8px;">What else is being tracked '
        "alongside the active model</p>"
        '<div class="prose"><p>Every challenger below rides on the SAME published card '
        "-- none of them spends a research window or changes what gets played. Each "
        "one's forced-pick accuracy is scored against the recorded line (the opener, "
        "primary) and again against the close (secondary), paired game-for-game with "
        "the active model's own paper ledger, the same way the main History "
        "above is graded.</p></div>"
    )
    return _spaced(intro + cards)


_NOT_APPLIED_NOTE = "Prospective evidence only -- not applied to the published card."


def _tilt_preview_sentence(result: Any, detail_fn: Any, *, applied_to_real_card: bool) -> str:

    if not result.enabled:
        return "Not eligible this week under its own rule."
    if result.flip_count == 0:
        base = "No games matched its rule this week, so nothing would change."
        return base if applied_to_real_card else f"{base} {_NOT_APPLIED_NOTE}"
    plural = "" if result.flip_count == 1 else "s"
    detail = "; ".join(
        f"{matchup} ({frm} to {to})"
        for matchup, frm, to in (detail_fn(flip) for flip in result.flips)
    )
    if applied_to_real_card:
        return (
            f"Flipped {result.flip_count} pick{plural} on the published card this week: {detail}."
        )
    return (
        f"Would flip {result.flip_count} pick{plural} on this week's card if it were live: "
        f"{detail}. {_NOT_APPLIED_NOTE}"
    )


def _flip_backup_qb(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.backup_team, flip.opponent_team


def _flip_division_revenge(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.opponent_team, flip.revenge_team


def _flip_injury_value(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.hurt_team, flip.healthier_team


def _flip_surface_switch(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.grass_modal_visitor, flip.turf_venue_home


def _flip_spread_gap_zone(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.original_pick_team, flip.flipped_to_team


def _flip_coach_fade(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.year_one_team, flip.opponent_team


def _flip_interim_hc_first_game(flip: Any) -> tuple[str, str, str]:
    return flip.matchup, flip.opponent_team, flip.interim_team


def _real_overlay_preview_sentence(overlay: OverlayResult) -> str:

    return _tilt_preview_sentence(overlay, _flip_coach_fade, applied_to_real_card=True)


def _team_for_game(predictions: pd.DataFrame, game_id: str | None) -> str | None:
    if game_id is None or predictions.empty or "game_id" not in predictions.columns:
        return None
    row = predictions.loc[predictions["game_id"].astype(str).eq(str(game_id))]
    if row.empty:
        return None
    team, _ = pick_side(row.iloc[0])
    return team


def _best_pick_preview_sentence(
    nomination: BestPickNomination | None, predictions: pd.DataFrame
) -> str:
    if nomination is None or nomination.v2_result is None:
        return (
            "Could not be computed this week (not enough walk-forward training history "
            "yet, or no market snapshot available)."
        )
    result = nomination.v2_result
    unrestricted_id = result.base_game_id if result.base_game_id is not None else result.game_id
    v2_team = _team_for_game(predictions, unrestricted_id)
    if nomination.active_rule == "v2" and result.spread_threshold is None:
        team_text = v2_team if v2_team else "this week's nominated game"
        return f"This IS the rule actually used this week: it nominates {team_text} for Best Pick."
    played_team = _team_for_game(predictions, nomination.active_game_id)
    if v2_team and played_team and v2_team == played_team:
        return f"This week it agrees with the rule now in use: both nominate {v2_team}."
    if v2_team:
        return (
            f"This week it would nominate {v2_team}, but the rule actually played "
            f"nominates {played_team or 'a different game'} instead."
        )
    return "No nomination this week (playoff week, or no line-sweep artifact yet)."


def _best_pick_v3_preview_sentence(
    nomination: BestPickNomination | None,
    predictions: pd.DataFrame,
    metadata: Mapping[str, Any],
    data_root: Path | None,
) -> str:

    inputs = v2_nomination_inputs(metadata, data_root)
    if inputs is None:
        return (
            "Could not be computed this week (not enough walk-forward training history "
            "yet, or no market snapshot available)."
        )
    try:
        features = pd.read_parquet(inputs.feature_table)
    except (OSError, ValueError):
        return "Could not be computed this week (its feature table is not available locally)."
    try:
        result = nominate_v3(
            predictions,
            features,
            market_root=Path(inputs.market_root),
            season=inputs.season,
            week=inputs.week,
            regressor=inputs.regressor,
            feature_profile=inputs.feature_profile,
            min_train_games=inputs.min_train_games,
        )
    except (ValueError, DataContractError):
        return (
            "Could not be computed this week (not enough walk-forward training history "
            "yet, or no market snapshot available)."
        )
    if result is None:
        return "No nomination this week (playoff week, or no line-sweep artifact yet)."
    v3_team = _team_for_game(predictions, result.game_id)
    active_team = (
        _team_for_game(predictions, nomination.active_game_id)
        if nomination is not None and nomination.active_game_id is not None
        else None
    )
    if v3_team and active_team and v3_team == active_team:
        return f"This week it agrees with the nomination actually played: both nominate {v3_team}."
    if v3_team:
        return (
            f"This week it would nominate {v3_team}, differing from the nomination actually "
            f"played, {active_team or 'a different game'}. {_NOT_APPLIED_NOTE}"
        )
    return "No nomination this week (playoff week, or no line-sweep artifact yet)."


_LOCK_TIME_EVALUATED_NOTES: dict[str, str] = {
    "ecdf_mapping_incumbent": (
        "Evaluated at lock time -- this challenger remaps every game's probability from a "
        "fresh weekly model refit, too heavy to recompute for this page's preview. Its 2026 "
        "prospective record (below) fills in once games are recorded and settled."
    ),
    "era_weighted_half_life_8": (
        "Evaluated at lock time -- this challenger refits the model with different "
        "season-weighting every week, too heavy to recompute for this page's preview. Its "
        "2026 prospective record (below) fills in once games are recorded and settled."
    ),
    "forecast_cold_visitor_tilt": (
        "Evaluated at lock time -- this tilt reads a LIVE weather forecast fetched at "
        "recording time, which this page cannot fetch during a static-site build. Its 2026 "
        "prospective record (below) fills in once games are recorded and settled."
    ),
    "forecast_weather_kn_warm_team_cold_late_tilt": (
        "Evaluated at lock time -- this tilt reads a LIVE kickoff-nearest weather forecast "
        "fetched at recording time, which this page cannot fetch during a static-site "
        "build. Its 2026 prospective record (below) fills in once games are recorded and "
        "settled."
    ),
    "forecast_weather_kn_precip_high_total_tilt": (
        "Evaluated at lock time -- this tilt reads the SAME live kickoff-nearest weather "
        "forecast fetched at recording time as the warm-team-cold-late tilt above, which "
        "this page cannot fetch during a static-site build. Its 2026 prospective record "
        "(below) fills in once games are recorded and settled."
    ),
    "model_only_refresh_incumbent": (
        "Evaluated at lock time -- this arm only diverges from the model's own pick when a "
        "later Thursday/Saturday/Sunday refresh pass sees the market move at least 1 point "
        "off the frozen Tuesday line, so there is nothing to preview on the Tuesday build."
    ),
    "injury_signal_refresh_tilt": (
        "Evaluated at refresh passes -- this challenger reads post-Tuesday injury filings "
        "(official Wednesday-Friday reports, or a news-headline fallback), which are by "
        "construction empty at Tuesday noon, so it has nothing to preview on the Tuesday "
        "build. Its first real reading, and its 2026 prospective record (below), fill in "
        "once a Thursday/Saturday/Sunday `nfl-ats refresh-picks` pass runs during the week."
    ),
    "overlay_production_chain_coach_arrest_incumbent": (
        "Recorded at lock time from the same immutable paper-decision row as the played "
        "three-member card, so the former-policy comparison cannot drift between source reads."
    ),
}


def _load_schedules_for_challenger_preview(data_root: Path) -> pd.DataFrame | None:

    try:
        schedules, _team_stats = load_snapshot(latest_snapshot(data_root / "raw"))
    except FileNotFoundError:
        return None
    return schedules


def _load_injury_features_for_challenger_preview(data_root: Path) -> pd.DataFrame | None:
    path = data_root / "processed" / PLAYER_FEATURE_TABLE_NAME
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path, columns=["game_id", *VALUE_LOST_DIFF_COLUMNS])
    except (OSError, ValueError, KeyError):
        return None


_SCHEDULE_BASED_TILT_PREVIEWS: tuple[tuple[str, Any, Any], ...] = (
    ("backup_qb_fade_overlay", apply_backup_qb_fade_overlay, _flip_backup_qb),
    ("division_revenge_tilt_overlay", apply_division_revenge_tilt_overlay, _flip_division_revenge),
    ("surface_switch_tilt_overlay", apply_surface_switch_tilt_overlay, _flip_surface_switch),
)


def _challenger_week_previews(
    challengers: Sequence[Mapping[str, Any]],
    predictions: pd.DataFrame,
    data_root: Path,
    *,
    overlay: OverlayResult,
    nomination: BestPickNomination | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:

    active_ids = {
        str(entry.get("challenger_id"))
        for entry in challengers
        if str(entry.get("status")) == "ACTIVE_PROSPECTIVE"
    }
    previews: dict[str, str] = {}
    if predictions.empty:
        return previews

    if "hc_year_one_fade_overlay" in active_ids:
        previews["hc_year_one_fade_overlay"] = _real_overlay_preview_sentence(overlay)
    if "best_pick_nomination_v2" in active_ids:
        previews["best_pick_nomination_v2"] = _best_pick_preview_sentence(nomination, predictions)
    if "best_pick_nomination_v3" in active_ids:
        previews["best_pick_nomination_v3"] = _best_pick_v3_preview_sentence(
            nomination, predictions, metadata or {}, data_root
        )
    if "mod07_weak_signal_stack" in active_ids:
        previews["mod07_weak_signal_stack"] = (
            "This challenger IS the active model's own configuration, so it makes the "
            "exact same picks -- there is nothing separate for it to flip."
        )

    schedules = _load_schedules_for_challenger_preview(data_root)
    if schedules is not None:
        for challenger_id, apply_fn, detail_fn in _SCHEDULE_BASED_TILT_PREVIEWS:
            if challenger_id not in active_ids:
                continue
            try:
                result = apply_fn(predictions, schedules)
            except DataContractError:
                continue
            previews[challenger_id] = _tilt_preview_sentence(
                result,
                detail_fn,
                applied_to_real_card=challenger_id == "division_revenge_tilt_overlay",
            )

    if "spread_gap_zone_fade_overlay" in active_ids:
        try:
            result = apply_spread_gap_zone_fade_overlay(predictions)
        except DataContractError:
            pass
        else:
            previews["spread_gap_zone_fade_overlay"] = _tilt_preview_sentence(
                result, _flip_spread_gap_zone, applied_to_real_card=False
            )

    if "injury_value_lost_tilt_overlay" in active_ids:
        features = _load_injury_features_for_challenger_preview(data_root)
        if features is not None:
            try:
                result = apply_injury_value_tilt_overlay(predictions, features)
            except DataContractError:
                pass
            else:
                previews["injury_value_lost_tilt_overlay"] = _tilt_preview_sentence(
                    result, _flip_injury_value, applied_to_real_card=False
                )

    if "interim_hc_first_game_tilt_overlay" in active_ids:
        try:
            result = apply_interim_hc_first_game_tilt_overlay(predictions, data_root.parent)
        except DataContractError:
            pass
        else:
            previews["interim_hc_first_game_tilt_overlay"] = _tilt_preview_sentence(
                result, _flip_interim_hc_first_game, applied_to_real_card=False
            )

    for challenger_id, note in _LOCK_TIME_EVALUATED_NOTES.items():
        if challenger_id in active_ids and challenger_id not in previews:
            previews[challenger_id] = note

    return previews


def _load_latest_prospective_scoring(artifacts_root: Path) -> dict[str, Mapping[str, Any]]:

    directories = artifact_directories(artifacts_root / "prospective_scoring", "metadata.json")
    for directory in directories:
        try:
            metadata = read_json(directory / "metadata.json")
        except (ValueError, OSError):
            continue
        entrants = metadata.get("entrants")
        if not isinstance(entrants, list):
            continue
        return {
            str(item["entrant"]): item
            for item in entrants
            if isinstance(item, dict) and item.get("entrant")
        }
    return {}


def _prospective_record_text(report: Mapping[str, Any] | None) -> str:
    if report is None:
        return _PENDING_PROSPECTIVE_RECORD
    forced = report.get("forced_picks")
    decision = forced.get("decision_line") if isinstance(forced, dict) else None
    if not isinstance(decision, dict):
        return _PENDING_PROSPECTIVE_RECORD
    games = _number(decision.get("games"))
    if not games:
        return _PENDING_PROSPECTIVE_RECORD
    accuracy = _number(decision.get("accuracy"))
    vs_coin_flip = _number(decision.get("vs_coin_flip"))
    accuracy_text = f"{accuracy:.1%}" if accuracy is not None else "--"
    delta_text = f" ({vs_coin_flip:+.1%} vs. a coin flip)" if vs_coin_flip is not None else ""
    return (
        f"{int(games)} games settled this season, {accuracy_text} against the recorded "
        f"line{delta_text}."
    )


def _challenger_prospective_records(
    artifacts_root: Path, challengers: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    reports = _load_latest_prospective_scoring(artifacts_root)
    return {
        str(entry.get("challenger_id")): _prospective_record_text(
            reports.get(str(entry.get("challenger_id")))
        )
        for entry in challengers
    }


@dataclass(frozen=True)
class PublicBoardArtifacts:
    predictions: pd.DataFrame
    sweep: pd.DataFrame
    explanations: dict[str, str]
    metadata: dict[str, Any]
    active: dict[str, Any]


@dataclass(frozen=True)
class OpenerEvaluationArtifacts:
    metadata: dict[str, Any]
    seasons: pd.DataFrame


def load_public_board_artifacts(artifacts_root: Path) -> PublicBoardArtifacts:

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

    explanations = _reconciled_explanations(artifacts_root, predictions)

    return PublicBoardArtifacts(predictions, sweep, explanations, metadata, active)


_EXPLANATION_RESIDUAL_TOLERANCE = 0.3


def _reconciled_explanations(artifacts_root: Path, predictions: pd.DataFrame) -> dict[str, str]:
    decomposition_directories = artifact_directories(
        artifacts_root / "market_decomposition", "attribution.parquet"
    )
    if not decomposition_directories:
        return {}
    attribution = pd.read_parquet(decomposition_directories[0] / "attribution.parquet")
    if not {"explanation", "game_id"}.issubset(attribution.columns):
        return {}

    live_residual: dict[str, float] = {}
    if "predicted_market_residual" in predictions.columns and "game_id" in predictions.columns:
        for _, row in predictions.iterrows():
            value = _number(row.get("predicted_market_residual"))
            if value is not None:
                live_residual[str(row["game_id"])] = value
    has_attrib_residual = "predicted_residual" in attribution.columns

    explanations: dict[str, str] = {}
    deduped = attribution.dropna(subset=["explanation"]).drop_duplicates("game_id")
    for _, row in deduped.iterrows():
        game_id = str(row["game_id"])
        if has_attrib_residual:
            attrib_value = _number(row.get("predicted_residual"))
            live_value = live_residual.get(game_id)
            if (
                attrib_value is not None
                and live_value is not None
                and abs(attrib_value - live_value) > _EXPLANATION_RESIDUAL_TOLERANCE
            ):
                continue
        explanations[game_id] = str(row["explanation"])
    return explanations


def load_opener_evaluation_artifacts(
    artifacts_root: Path, active_feature_profile: str | None = None
) -> OpenerEvaluationArtifacts:

    directories = artifact_directories(artifacts_root / "opener_evaluation", "metadata.json")
    active = load_active_ats_model(artifacts_root)
    if active and active.get("feature_table_sha256"):
        match = find_matching_opener_evaluation(artifacts_root, active)
        directories = [match[1]] if match is not None else []
    for directory in directories:
        metadata = read_json(directory / "metadata.json")
        if active_feature_profile is not None:
            config = metadata.get("active_model_config") or {}
            if config.get("feature_profile") != active_feature_profile:
                continue
        seasons = pd.DataFrame()
        season_path = directory / "season_summary.csv"
        if season_path.is_file():
            seasons = pd.read_csv(season_path)
        return OpenerEvaluationArtifacts(metadata, seasons)
    return OpenerEvaluationArtifacts({}, pd.DataFrame())


def _feature_table_sha256_of_opener_evaluation(metadata: Mapping[str, Any]) -> str | None:
    provenance = metadata.get("provenance")
    feature_table = provenance.get("feature_table") if isinstance(provenance, Mapping) else None
    sha = feature_table.get("sha256") if isinstance(feature_table, Mapping) else None
    return str(sha) if sha else None


def find_matching_opener_evaluation(
    artifacts_root: Path, active: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], Path] | None:

    return next(iter(matching_opener_evaluations(artifacts_root, active)), None)


def matching_opener_evaluations(
    artifacts_root: Path, active: Mapping[str, Any] | None = None
) -> Iterator[tuple[dict[str, Any], Path]]:

    if active is None:
        active = load_active_ats_model(artifacts_root)
    if not active:
        return
    target_sha = active.get("feature_table_sha256")
    if not target_sha:
        return
    for directory in artifact_directories(artifacts_root / "opener_evaluation", "metadata.json"):
        try:
            metadata = read_json(directory / "metadata.json")
        except (ValueError, OSError):
            continue
        if _feature_table_sha256_of_opener_evaluation(
            metadata
        ) == target_sha and _opener_model_matches(metadata, active):
            yield metadata, directory


def _opener_model_matches(metadata: Mapping[str, Any], active: Mapping[str, Any]) -> bool:
    config = metadata.get("active_model_config")
    recorded_id = metadata.get("active_model_id", metadata.get("model_id"))
    if recorded_id is not None and recorded_id != active.get("model_id"):
        return False
    if not isinstance(config, Mapping):
        return recorded_id is not None and recorded_id == active.get("model_id")
    if config.get("model_id", active.get("model_id")) != active.get("model_id"):
        return False
    expected = {
        "feature_profile": active.get("feature_profile"),
        "regressor": active.get("regressor"),
        "ridge_alpha": active.get("ridge_alpha", 10.0),
        "target": active.get("method"),
    }
    if any(value is None or config.get(key) != value for key, value in expected.items()):
        return False
    for key, default in (("probability_method", "ecdf"), ("calibration_method", "none")):
        if config.get(key, metadata.get(key, default)) != active.get(key, default):
            return False
    return True


@dataclass(frozen=True)
class RefreshChainMeasurement:
    tuesday_card_accuracy: float
    refresh_chain_accuracy: float
    scored_games: int
    week_blocks: int
    seasons: tuple[int, ...]
    late_week_input_seasons: tuple[int, ...]
    scored_games_with_late_week_inputs: int
    picks_changed: int
    directory: Path
    variant: str


def served_refresh_policy_ids() -> dict[str, str]:

    from nfl_ats.pick_refresh import (
        HANDLE_FOLLOW_POLICY,
        LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY,
        PRODUCTION_COMPOSITION_POLICY_IDS,
        ROOKIE_CREW_POLICY,
    )

    return {
        "composition": PRODUCTION_COMPOSITION_POLICY_IDS[-1],
        "late_week_follow": LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY,
        "rookie_crew": ROOKIE_CREW_POLICY,
        "handle_follow": HANDLE_FOLLOW_POLICY,
    }


def load_refresh_chain_measurement(
    artifacts_root: Path, active: Mapping[str, Any] | None = None
) -> RefreshChainMeasurement | None:

    if active is None:
        active = load_active_ats_model(artifacts_root)
    if not active:
        return None
    directories = list(
        artifact_directories(artifacts_root / "served_refresh_card", "headline.json")
    )
    if not directories:
        return None
    directory = directories[0]
    payload = read_json(directory / "headline.json")
    recorded_model = str(payload.get("model_id") or "")
    active_model = str(active.get("model_id") or "")
    if recorded_model != active_model and not _refresh_chain_archive_matches_active(
        artifacts_root, payload, active
    ):
        raise ValueError(
            f"The served refresh chain in {directory} was measured on model "
            f"{recorded_model!r}, not the active {active_model!r}, and its archive's "
            "predictions differ from every evaluation of the active model; rerun that lane."
        )
    served = served_refresh_policy_ids()
    variants = payload.get("variants") or []
    for variant in variants:
        if not isinstance(variant, Mapping) or variant.get("policy_ids") != served:
            continue
        seasons = tuple(int(value) for value in variant.get("seasons") or ())
        return RefreshChainMeasurement(
            tuesday_card_accuracy=float(variant["tuesday_card_accuracy"]),
            refresh_chain_accuracy=float(variant["refresh_chain_accuracy"]),
            scored_games=int(variant["scored_games"]),
            week_blocks=int(variant["week_blocks"]),
            seasons=seasons,
            late_week_input_seasons=tuple(
                int(value) for value in variant.get("late_week_input_seasons") or ()
            ),
            scored_games_with_late_week_inputs=int(
                variant.get("scored_games_with_late_week_inputs") or 0
            ),
            picks_changed=int(variant.get("picks_changed") or 0),
            directory=directory,
            variant=str(variant.get("variant") or ""),
        )
    raise ValueError(
        f"The served refresh chain in {directory} names none of the rules being served "
        f"({sorted(served.values())}); rerun that lane against the served chain."
    )


def _refresh_chain_archive_matches_active(
    artifacts_root: Path, payload: Mapping[str, Any], active: Mapping[str, Any]
) -> bool:
    used = payload.get("opener_evaluation_used")
    if not used:
        return False
    used_directory = artifacts_root.parent / str(used)
    if not (used_directory / "per_game.parquet").exists():
        used_directory = Path(str(used))
    match = find_matching_opener_evaluation(artifacts_root, active)
    if match is None or not (used_directory / "per_game.parquet").exists():
        return False
    return _opener_per_game_identical(used_directory, match[1])


def _opener_per_game_identical(left: Path, right: Path) -> bool:
    try:
        a = pd.read_parquet(left / "per_game.parquet")
        b = pd.read_parquet(right / "per_game.parquet")
    except (OSError, ValueError):
        return False
    if a.shape != b.shape or set(a.columns) != set(b.columns):
        return False
    keys = [c for c in ("season", "week", "game_id") if c in a.columns]
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    for column in a.columns:
        if a[column].equals(b[column]):
            continue
        numeric = pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(
            b[column]
        )
        if numeric and a[column].astype(float).round(9).equals(b[column].astype(float).round(9)):
            continue
        return False
    return True


@dataclass(frozen=True)
class BaselineMeasurement:
    accuracy: float
    games: int
    week_interval: tuple[float, float] | None
    season_interval: tuple[float, float] | None
    metadata: Mapping[str, Any]
    directory: Path
    played_accuracy: float | None
    composition_directory: Path | None


def load_baseline_measurement(
    artifacts_root: Path, active: Mapping[str, Any] | None = None
) -> BaselineMeasurement:
    match = find_matching_opener_evaluation(artifacts_root, active)
    if match is None:
        raise ValueError("No opener-evaluation matches the active model; rerun opener-evaluation.")
    metadata, directory = match
    metrics = metadata.get("metrics") or {}
    accuracy = _number(metrics.get("opener_accuracy_probability_rule"))
    games = metadata.get("games")
    if accuracy is None or not 0 <= accuracy <= 1 or not isinstance(games, int) or games <= 0:
        raise ValueError(f"Invalid opener measurement in {directory / 'metadata.json'}")

    def interval(block: str) -> tuple[float, float] | None:
        for row in metadata.get("uncertainty") or []:
            if not isinstance(row, Mapping):
                continue
            if row.get("metric") != "opener_accuracy_probability_rule" or row.get("block") != block:
                continue
            lower, upper = _number(row.get("lower")), _number(row.get("upper"))
            if lower is None or upper is None or not 0 <= lower <= upper <= 1:
                raise ValueError(f"Invalid opener interval in {directory / 'metadata.json'}")
            return lower, upper
        return None

    composition = find_matching_overlay_composition(artifacts_root, active)
    return BaselineMeasurement(
        accuracy,
        games,
        interval("week"),
        interval("season"),
        metadata,
        directory,
        played_union_subset_accuracy(composition[0]) if composition else None,
        composition[1] if composition else None,
    )


def find_matching_overlay_composition(
    artifacts_root: Path, active: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], Path] | None:

    if active is None:
        active = load_active_ats_model(artifacts_root)
    expected_paths = {
        (directory / "per_game.parquet").resolve()
        for _metadata, directory in matching_opener_evaluations(artifacts_root, active)
    }
    if not expected_paths:
        return None
    for directory in artifact_directories(
        artifacts_root / "overlay_subset_composition", "result.json"
    ):
        try:
            payload = read_json(directory / "result.json")
        except (ValueError, OSError):
            continue
        source = str(payload.get("source_artifact") or "").replace("\\", "/")
        source_path = Path(source)
        candidates = (
            (source_path,)
            if source_path.is_absolute()
            else (artifacts_root.parent / source_path, artifacts_root / source_path)
        )
        if any(candidate.resolve() in expected_paths for candidate in candidates):
            return payload, directory
    return None


PLAYED_UNION_MEMBER_IDS: frozenset[str] = frozenset(
    {
        "coach_fade_overlay",
        "division_revenge_tilt_overlay",
        "player_arrests_back_side_policy",
    }
)


def played_union_subset_accuracy(payload: Mapping[str, Any]) -> float | None:

    for subset in payload.get("subsets") or []:
        if not isinstance(subset, Mapping):
            continue
        members = subset.get("members")
        if isinstance(members, list) and frozenset(members) == PLAYED_UNION_MEMBER_IDS:
            return _number(subset.get("candidate_accuracy"))
    return None


@dataclass(frozen=True)
class ServedUnionMeasurement:
    accuracy: float
    scored_games: int
    member_count: int
    seasons: tuple[int, int] | None


def _active_archive_sha256(artifacts_root: Path, active: Mapping[str, Any] | None) -> str:
    match = find_matching_opener_evaluation(artifacts_root, active)
    if match is None:
        return ""
    path = match[1] / "per_game.parquet"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def load_served_union_measurement(
    artifacts_root: Path, active: Mapping[str, Any] | None = None
) -> ServedUnionMeasurement | None:

    if active is None:
        active = load_active_ats_model(artifacts_root)
    active_model_id = str((active or {}).get("model_id") or "")
    if not active_model_id:
        return None
    archive_sha = _active_archive_sha256(artifacts_root, active)
    for directory in artifact_directories(
        artifacts_root / "unserved_tilt_marginals", "result.json"
    ):
        try:
            payload = read_json(directory / "result.json")
        except (OSError, ValueError):
            continue
        served = payload.get("served_policy")
        if not isinstance(served, Mapping) or served.get("policy_id") != SERVED_POLICY_ID:
            continue
        same_model = str(payload.get("active_model_id") or "") == active_model_id
        same_archive = bool(archive_sha) and payload.get("source_artifact_sha256") == archive_sha
        if not (same_model or same_archive):
            continue
        accuracy = _number(payload.get("served_card_accuracy"))
        scored = payload.get("n_scored_games")
        members = served.get("members")
        if accuracy is None or not 0 <= accuracy <= 1:
            continue
        if not isinstance(scored, int) or scored <= 0 or not isinstance(members, list):
            continue
        span = payload.get("seasons")
        seasons = (
            (int(span[0]), int(span[1])) if isinstance(span, list) and len(span) == 2 else None
        )
        return ServedUnionMeasurement(accuracy, scored, len(members), seasons)
    return None


def load_played_chain_accuracy(artifacts_root: Path) -> float | None:

    match = find_matching_overlay_composition(artifacts_root)
    if match is None:
        return None
    payload, _directory = match
    return played_union_subset_accuracy(payload)


@dataclass(frozen=True)
class EraMagnitude:
    era_label: str
    effect: float
    interval: tuple[float, float] | None
    probability_positive: float | None


def load_era_magnitude_profile(artifacts_root: Path) -> dict[str, list[EraMagnitude]]:

    directories = artifact_directories(artifacts_root / "era_magnitude_profile", "results.json")
    if not directories:
        return {}
    try:
        payload = read_json(directories[0] / "results.json")
    except (ValueError, OSError):
        return {}

    fixed_eras = payload.get("fixed_eras")
    signals = payload.get("signals")
    if not isinstance(fixed_eras, list) or not isinstance(signals, dict):
        return {}

    result: dict[str, list[EraMagnitude]] = {}
    for name, signal in signals.items():
        if not isinstance(signal, dict):
            continue
        era_results = signal.get("era_results")
        if not isinstance(era_results, dict):
            continue
        rows: list[EraMagnitude] = []
        for era in fixed_eras:
            if not isinstance(era, dict):
                continue
            key = era.get("key")
            era_row = era_results.get(key) if isinstance(key, str) else None
            if not isinstance(era_row, dict) or era_row.get("insufficient_data"):
                continue
            effect = _number(era_row.get("effect"))
            interval: tuple[float, float] | None = None
            probability_positive: float | None = None
            week_blocked = era_row.get("week_blocked")
            if isinstance(week_blocked, dict):
                lower, upper = (
                    _number(week_blocked.get("lower")),
                    _number(week_blocked.get("upper")),
                )
                if lower is not None and upper is not None:
                    interval = (lower, upper)
                probability_positive = _number(week_blocked.get("probability_positive"))
            else:
                if effect is None:
                    effect = _number(era_row.get("estimate"))
                lower, upper = _number(era_row.get("lower")), _number(era_row.get("upper"))
                if lower is not None and upper is not None:
                    interval = (lower, upper)
                probability_positive = _number(era_row.get("probability_positive"))
            if effect is None:
                continue
            season_lo, season_hi = era.get("season_lo"), era.get("season_hi")
            label = (
                f"{season_lo}-{season_hi}"
                if season_lo is not None and season_hi is not None
                else str(key)
            )
            rows.append(
                EraMagnitude(
                    era_label=label,
                    effect=effect,
                    interval=interval,
                    probability_positive=probability_positive,
                )
            )
        if rows:
            result[str(name)] = rows
    return result


_ERA_TREND_PREFIX = "era_trend_"


def _era_magnitude_for_lead(
    lead_name: str, era_magnitude: Mapping[str, Sequence[EraMagnitude]]
) -> Sequence[EraMagnitude]:

    if not lead_name.startswith(_ERA_TREND_PREFIX):
        return ()
    return era_magnitude.get(lead_name[len(_ERA_TREND_PREFIX) :], ())


def load_prospective_challengers(artifacts_root: Path) -> list[dict[str, Any]]:

    path = artifacts_root / "prospective" / "challengers.json"
    if not path.is_file():
        return []
    try:
        payload = read_json(path)
    except (ValueError, OSError):
        return []
    challengers = payload.get("challengers")
    return (
        [entry for entry in challengers if isinstance(entry, dict)]
        if isinstance(challengers, list)
        else []
    )


def render_models_page(
    ledger_section: str | None,
    *,
    explanation_section: str | None = None,
    generated_at: datetime | None = None,
) -> str:

    body = viz.page_header(
        "Model Ledger",
        "Every arm the card could come from",
        "The promoted production card first, then each candidate rule by best-evidence confidence.",
    )
    if not ledger_section:
        body += (
            '<p class="sub">Ledger unavailable right now -- '
            "the challenger registry or active-model manifest could not be read. "
            "This page rebuilds with every publish; nothing is hidden.</p>"
        )
    else:
        body += ledger_section
    body += _section_header(
        "WHAT THIS TABLE IS",
        "Reading the ledger",
        "",
        top=40,
    ) + (
        '<div class="prose">'
        "<p>Each row is a version of the picking system. &#8220;Model only&#8221; is "
        "the bare model. The promoted row is what actually makes this week's picks; "
        "candidate rules are measured but not played. P+ is our confidence an effect "
        "is real rather than luck.</p>"
        "</div>"
    )
    if explanation_section:
        body += explanation_section
    return _page(
        current=MODELS_PAGE,
        body=body,
        generated=(generated_at or datetime.now(UTC)),
    )


def _diverging_bar(z: float, max_abs: float, *, good_direction: int = 0) -> str:

    if not math.isfinite(z) or max_abs <= 0:
        return (
            '<div style="width:100px;height:8px;background:var(--grid);border-radius:4px;"></div>'
        )
    frac = max(-1.0, min(1.0, z / max_abs))
    center = 50.0
    pos = center + 50.0 * frac
    if frac >= 0:
        style = f"left:{center:g}%;width:{pos - center:g}%;"
    else:
        style = f"right:{100 - center:g}%;width:{center - pos:g}%;"
    merit = z * good_direction
    fill = "var(--pos)" if merit > 0 else "var(--neg)" if merit < 0 else "var(--ink-2)"
    return (
        '<div style="position:relative;width:100px;height:8px;background:var(--grid);'
        f'border-radius:4px;flex:none;">'
        f'<div style="position:absolute;top:0;height:8px;background:{fill};'
        f'border-radius:4px;{style}"></div></div>'
    )


def _signed(value: float, digits: int = 2, *, good_direction: int = 1) -> str:

    if not math.isfinite(value):
        return '<span class="delta zero">\u2014</span>'
    text = f"{abs(value):.{digits}f}"
    sign = "+" if value > 0 else ("-" if value < 0 else "")
    merit = value * good_direction
    tone = "pos" if merit > 0 else "neg" if merit < 0 else "zero"
    return f'<span class="delta {tone}">{sign}{text}</span>'


def _team_explorer_overview(trends: TeamTrends, metrics: Sequence[str]) -> str:

    latest = trends.latest
    max_abs: dict[str, float] = {}
    for metric in metrics:
        column = latest.loc[latest["metric"] == metric, "z"]
        max_abs[metric] = float(column.abs().max()) if not column.empty else 0.0

    sort_metric = "point_diff" if "point_diff" in metrics else metrics[0]
    ordered = (
        latest.loc[latest["metric"] == sort_metric]
        .sort_values("z", ascending=False)["team"]
        .tolist()
    )
    teams = [t for t in ordered if t in trends.teams] + [
        t for t in trends.teams if t not in ordered
    ]

    head = "<th>Team</th>" + "".join(f"<th>{escape(metric_label(m))}</th>" for m in metrics)
    rows_html = []
    for team in teams:
        cells = [f"<td><b>{escape(str(team))}</b></td>"]
        for metric in metrics:
            row = latest.loc[(latest["team"] == team) & (latest["metric"] == metric)]
            if row.empty:
                cells.append("<td>\u2014</td>")
                continue
            value = float(row["value"].iloc[0])
            z = float(row["z"].iloc[0])
            direction = metric_good_direction(metric)
            bar = _diverging_bar(z, max_abs[metric], good_direction=direction)
            cells.append(
                "<td style='white-space:nowrap;'>"
                f"{bar}<span class='fine' style='margin-left:8px;'>"
                f"{_signed(value, good_direction=direction)}</span></td>"
            )
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table = (
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )
    caption = (
        f'<p class="fine" style="margin:6px 0 14px;max-width:80ch;">'
        f"Latest season shown: {trends.latest_season}. Each bar places the team "
        f"versus the league average for that metric; the number is the raw "
        f"pregame-state value.</p>"
    )
    return table + caption


def _team_explorer_trend_details(trends: TeamTrends, metrics: Sequence[str]) -> str:

    seasons = sorted(int(s) for s in trends.trend["season"].dropna().unique().tolist())
    blocks = []
    for team in trends.teams:
        head = "<th>Metric</th>" + "".join(f"<th>{escape(str(season))}</th>" for season in seasons)
        rows_html = []
        for metric in metrics:
            cells = [f"<td><b>{escape(metric_label(metric))}</b></td>"]
            for season in seasons:
                mask = (
                    (trends.trend["team"] == team)
                    & (trends.trend["metric"] == metric)
                    & (trends.trend["season"] == season)
                )
                value = trends.trend.loc[mask, "value"]
                direction = metric_good_direction(metric)
                cells.append(
                    "<td>"
                    + (
                        _signed(float(value.iloc[0]), good_direction=direction)
                        if not value.empty
                        else "\u2014"
                    )
                    + "</td>"
                )
            rows_html.append(f"<tr>{''.join(cells)}</tr>")
        table = (
            '<div style="overflow-x:auto;">'
            '<table style="border-collapse:collapse;width:100%;font-size:13px;">'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
        )
        blocks.append(
            '<details style="margin-bottom:8px;border:1px solid var(--grid);'
            'border-radius:8px;padding:10px 14px;">'
            f'<summary style="cursor:pointer;font-weight:600;">{escape(str(team))}</summary>'
            f'<div style="margin-top:10px;">{table}</div></details>'
        )
    return "".join(blocks)


def _team_explorer_matchup(trends: TeamTrends, metrics: Sequence[str]) -> tuple[str, str]:

    teams = trends.teams
    if len(teams) >= 2:
        team_a, team_b = teams[0], teams[1]
    elif teams:
        team_a = team_b = teams[0]
    else:
        team_a = team_b = ""

    options = "".join('<option value="' + escape(t) + '">' + escape(t) + "</option>" for t in teams)
    payload = team_state_payload(trends)

    def _compare_rows(team_a: str, team_b: str) -> str:
        rows = []
        for metric in metrics:
            za = payload.get(team_a, {}).get(metric)
            zb = payload.get(team_b, {}).get(metric)
            if za is None or zb is None:
                rows.append(
                    f"<tr><td><b>{escape(metric_label(metric))}</b></td>"
                    "<td>\u2014</td><td>\u2014</td><td>\u2014</td></tr>"
                )
                continue
            diff = za - zb
            arrow = "\u25b2" if diff > 0 else ("\u25bc" if diff < 0 else "\u25ac")
            direction = metric_good_direction(metric)
            rows.append(
                f"<tr><td><b>{escape(metric_label(metric))}</b></td>"
                f"<td>{_signed(za, good_direction=direction)}</td>"
                f"<td>{_signed(zb, good_direction=direction)}</td>"
                f"<td>{arrow} {_signed(diff, good_direction=direction)}</td></tr>"
            )
        return "".join(rows)

    compare_rows = _compare_rows(team_a, team_b)
    head = (
        "<th>Metric</th>"
        f"<th id='ats-te-ha'>{escape(team_a)}</th>"
        f"<th id='ats-te-hb'>{escape(team_b)}</th>"
        "<th title=\"Team A's number minus Team B's; "
        'positive means Team A is ahead">Advantage</th>'
    )
    html = (
        '<div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:12px;">'
        f'<label class="fine">Team A<select id="ats-te-a" style="margin-left:6px;">{options}'
        "</select></label>"
        f'<label class="fine">Team B<select id="ats-te-b" style="margin-left:6px;">{options}'
        "</select></label>"
        "</div>"
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;" '
        'id="ats-te-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody id='ats-te-body'>{compare_rows}</tbody></table></div>"
    )

    data_json = json.dumps(payload, separators=(",", ":"))
    labels_json = json.dumps([metric_label(m) for m in metrics], separators=(",", ":"))
    metrics_json = json.dumps(list(metrics), separators=(",", ":"))
    directions_json = json.dumps([metric_good_direction(m) for m in metrics], separators=(",", ":"))
    script = (
        '<script type="application/json" id="ats-te-data">' + data_json + "</script>\n"
        "<script>\n"
        "(function () {\n"
        "  var dataEl = document.getElementById('ats-te-data');\n"
        "  if (!dataEl) { return; }\n"
        "  var data; try { data = JSON.parse(dataEl.textContent); } catch (e) { return; }\n"
        "  var selA = document.getElementById('ats-te-a');\n"
        "  var selB = document.getElementById('ats-te-b');\n"
        "  var body = document.getElementById('ats-te-body');\n"
        "  var ha = document.getElementById('ats-te-ha');\n"
        "  var hb = document.getElementById('ats-te-hb');\n"
        "  var labels = " + labels_json + ";\n"
        "  var metricsArr = " + metrics_json + ";\n"
        "  var goodDir = " + directions_json + ";\n"
        "  function signed(v, dir) {\n"
        "    if (v === null || v === undefined || isNaN(v)) "
        "{ return '<span class=\"delta zero\">\u2014</span>'; }\n"
        "    var t = Math.abs(v).toFixed(2);\n"
        "    var merit = v * (dir === undefined ? 1 : dir);\n"
        "    var tone = merit > 0 ? 'pos' : (merit < 0 ? 'neg' : 'zero');\n"
        "    var s = v > 0 ? '+' : (v < 0 ? '-' : '');\n"
        "    return '<span class=\"delta ' + tone + '\">' + s + t + '</span>';\n"
        "  }\n"
        "  function arrow(d) { return d > 0 ? '\u25b2' : (d < 0 ? '\u25bc' : '\u25ac'); }\n"
        "  function render() {\n"
        "    var a = selA.value, b = selB.value;\n"
        "    if (ha) { ha.textContent = a; } if (hb) { hb.textContent = b; }\n"
        "    var rows = '';\n"
        "    for (var i = 0; i < metricsArr.length; i++) {\n"
        "      var m = metricsArr[i];\n"
        "      var za = (data[a] && data[a][m] != null) ? data[a][m] : null;\n"
        "      var zb = (data[b] && data[b][m] != null) ? data[b][m] : null;\n"
        "      if (za === null || zb === null) {\n"
        "        rows += '<tr><td><b>' + labels[i] + '</b></td><td>\u2014</td><td>\u2014</td><td>\u2014</td></tr>';\n"  # noqa: E501
        "        continue;\n"
        "      }\n"
        "      var d = za - zb;\n"
        "      rows += '<tr><td><b>' + labels[i] + '</b></td>';\n"
        "      var gd = goodDir[i];\n"
        "      rows += '<td>' + signed(za, gd) + '</td><td>' + signed(zb, gd) + '</td>';\n"
        "      rows += '<td>' + arrow(d) + ' ' + signed(d, gd) + '</td></tr>';\n"
        "    }\n"
        "    body.innerHTML = rows;\n"
        "  }\n"
        "  if (selA && selB && body) {\n"
        "    selA.addEventListener('change', render);\n"
        "    selB.addEventListener('change', render);\n"
        "    render();\n"
        "  }\n"
        "})();\n"
        "</script>\n"
    )
    return html, script


def _team_explorer_primer(metrics: list[str]) -> str:
    legend = "".join(
        f"<dt><b>{escape(metric_label(m))}</b></dt><dd>{escape(metric_help(m))}</dd>"
        for m in metrics
    )
    return (
        '<div class="card" style="margin-top:8px;">'
        '<h2 class="title" style="font-size:17px;margin:0 0 6px;">'
        "How to read this page</h2>"
        '<p class="prose" style="margin:0 0 6px;">Every NFL team gets a '
        "<b>strength number</b> for each skill -- passing, running, defense, "
        "and a few others. The numbers come from play-by-play data of games "
        "already played. For any given game, only <b>earlier</b> games count: "
        "nothing on this page uses what happened in the game being described."
        "</p>"
        '<p class="prose" style="margin:0 0 6px;">Higher strength usually means '
        "a better team, except for the defense and turnover numbers, where "
        "lower is better (each stat below says which). A team well above "
        "average is playing well; a team trending up is improving.</p>"
        '<p class="prose" style="margin:0 0 8px;">This page describes the past. '
        "It does not predict games and it is not betting advice.</p>"
        '<details style="margin-bottom:4px;"><summary class="fine">'
        "What each stat means (the full list)</summary>"
        f'<dl style="margin:8px 0 0;">{legend}</dl></details>'
        "</div>"
    )


def render_team_explorer_page(
    state_table: pd.DataFrame | None,
    *,
    generated_at: datetime | None = None,
    metrics: Sequence[str] | None = None,
) -> str:

    wanted = list(metrics) if metrics is not None else list(DEFAULT_TREND_METRICS)
    trends = aggregate_team_trends(state_table, metrics=wanted)

    sub = (
        "How strong every NFL team looked going into its games, season by "
        "season -- using only what was knowable before kickoff. Nothing here "
        "uses final scores, betting results, or anything after kickoff."
    )
    body = viz.page_header("Team trends", "Team strength, game by game", sub=sub)

    body += _team_explorer_primer(wanted)

    if trends.latest_season is None:
        body += viz.empty_state(
            "No team-state data yet",
            "The team-state feature table has not been built for this forecast. "
            "The page rebuilds with every publish once that artifact is present; "
            "nothing is hidden.",
        )
        return _page(
            current=TEAM_EXPLORER_PAGE,
            body=body,
            generated=(generated_at or datetime.now(UTC)),
        )

    body += _team_explorer_overview(trends, wanted)
    body += _section_header(
        "Season-by-season trends",
        "One team at a time",
        "Pick a team. Each line shows how strong that team was going into its "
        "games that season, for one skill at a time, compared with the rest of "
        "the league.",
        top=40,
    )
    body += _team_explorer_trend_details(trends, wanted)
    matchup_html, matchup_script = _team_explorer_matchup(trends, wanted)
    body += _section_header(
        "Head-to-head comparison",
        "Two teams, side by side",
        "Choose two teams to see how their most recent strength numbers stack "
        "up, stat by stat. The last column is simply Team A's number minus "
        "Team B's -- positive means the first team is ahead.",
        top=40,
    )
    body += matchup_html
    body += (
        '<p class="fine" style="margin-top:10px;max-width:80ch;">'
        "Bars and arrows compare each team with the league average for that "
        "season and stat. For the defense stats, remember lower numbers are "
        "the good ones -- each stat's plain-language explanation is in the "
        "legend above.</p>"
    )
    return _page(
        current=TEAM_EXPLORER_PAGE,
        body=body,
        generated=(generated_at or datetime.now(UTC)),
        scripts=matchup_script,
    )


def render_pool_workbench_page(
    predictions: pd.DataFrame,
    pool_rules: PoolRules | None = None,
    *,
    season: int | None = None,
    week: int | None = None,
    model_id: str | None = None,
    generated_at: datetime | None = None,
    best_pick_game_id: str | None = None,
    strength_bands: StrengthBands | None = None,
) -> str:

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    model_text = f"model <code>{escape(model_id)}</code>" if model_id else "model unknown"
    body = build_pool_workbench_body(
        predictions,
        pool_rules,
        best_pick_game_id=best_pick_game_id,
        season=season,
        week=week,
        strength_bands=strength_bands,
    )
    return _page(
        current=POOL_PAGE,
        body=body,
        generated=generated,
        footer_note=model_text,
    )


def render_signal_ledger_page(
    *,
    registry_root: Path | None = None,
    weak_signal_registry: WeakSignalRegistry | None = None,
    generated_at: datetime | None = None,
) -> str:

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    registry = (
        weak_signal_registry
        if weak_signal_registry is not None
        else load_weak_signal_registry(registry_root)
    )
    body, script = build_signal_ledger_body(registry)
    return _page(
        current=LEDGER_PAGE,
        body=body,
        generated=generated,
        scripts=script,
    )


def build_public_site(
    artifacts_root: Path,
    *,
    data_root: Path | None = None,
    generated_at: datetime | None = None,
    require_fresh_arrest_overlay: bool = True,
) -> dict[str, str]:

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    resolved_data_root = data_root if data_root is not None else _default_data_root()
    artifacts = load_public_board_artifacts(artifacts_root)
    model_id = artifacts.active.get("model_id")

    game_type = (
        str(artifacts.predictions["game_type"].iloc[0])
        if "game_type" in artifacts.predictions and not artifacts.predictions.empty
        else "REG"
    )
    view = (
        resolve_card_view(
            artifacts.predictions,
            artifacts.sweep,
            artifacts.metadata,
            data_root=resolved_data_root,
            now=generated,
            require_fresh_arrest_overlay=require_fresh_arrest_overlay,
        )
        if game_type == "REG" and not artifacts.predictions.empty
        else None
    )
    overlay = (
        view.overlay
        if view is not None
        else resolve_overlay(artifacts.predictions, resolved_data_root)
    )
    arrest_overlay = (
        view.arrest_overlay
        if view is not None
        else resolve_player_arrests_overlay(
            overlay.overlaid_predictions,
            resolved_data_root,
            now=generated,
            require_fresh=require_fresh_arrest_overlay,
        )
    )
    nomination = view.nomination if view is not None else None
    challengers = load_prospective_challengers(artifacts_root)
    challenger_week_previews = _challenger_week_previews(
        challengers,
        artifacts.predictions,
        resolved_data_root,
        overlay=overlay,
        nomination=nomination,
        metadata=artifacts.metadata,
    )
    challenger_prospective_records = _challenger_prospective_records(artifacts_root, challengers)

    active_report = _load_latest_prospective_scoring(artifacts_root).get("active_model")
    active_record_text = _prospective_record_text(active_report) if active_report else ""
    recent_form_text = (
        f"This season so far: {active_record_text}"
        if active_record_text and not active_record_text.startswith("Not scored yet")
        else None
    )

    waterfall_feed = load_waterfall_feed(artifacts_root)
    ledger_section = load_model_ledger_html(artifacts_root)
    explanation_section = load_model_explanation_html(artifacts_root)

    played_chain_accuracy = load_played_chain_accuracy(artifacts_root)

    spread_explorer_params: dict[str, SpreadExplorerGameParams] = {}
    if (
        str(artifacts.metadata.get("probability_method")) in ("gaussian", "gaussian_median")
        and not artifacts.predictions.empty
    ):
        explorer_features = load_feature_table_for_forecast(artifacts.metadata, resolved_data_root)
        spread_explorer_params = compute_spread_explorer_params(
            artifacts.predictions,
            explorer_features,
            regressor=str(artifacts.metadata.get("regressor")),
            ridge_alpha=float(artifacts.metadata.get("ridge_alpha", 10.0)),
            feature_profile=str(artifacts.metadata.get("feature_profile")),
            min_train_games=int(artifacts.metadata.get("min_train_games", 500)),
            probability_method=str(artifacts.metadata["probability_method"]),
            center_offsets=center_offsets_from_metadata(artifacts.metadata, artifacts.predictions),
            pick_overrides=pick_overrides_from_metadata(artifacts.metadata),
        )
        _assert_spread_explorer_matches_card(spread_explorer_params, artifacts.predictions)

    team_states: pd.DataFrame = pd.DataFrame()
    if not artifacts.predictions.empty:
        try:
            explorer_features = load_feature_table_for_forecast(
                artifacts.metadata, resolved_data_root
            )
            converted = feature_table_to_team_states(explorer_features)
            if converted is not None:
                team_states = converted
        except Exception:
            team_states = pd.DataFrame()

    return {
        PICKS_PAGE: render_picks_page(
            artifacts.predictions,
            artifacts.sweep,
            artifacts.explanations,
            season=artifacts.metadata.get("season"),
            week=artifacts.metadata.get("week"),
            model_id=str(model_id) if model_id else None,
            active_model=artifacts.active,
            generated_at=generated,
            metadata=artifacts.metadata,
            data_root=resolved_data_root,
            artifacts_root=artifacts_root,
            overlay=overlay,
            arrest_overlay=arrest_overlay,
            production_overlay=(view.production_overlay if view is not None else None),
            nomination=nomination,
            spread_explorer=spread_explorer_params,
            challengers=challengers,
            waterfall_feed=waterfall_feed,
            challenger_week_previews=challenger_week_previews,
            recent_form_text=recent_form_text,
            played_chain_accuracy=played_chain_accuracy,
        ),
        MODELS_PAGE: render_models_page(
            ledger_section,
            explanation_section=explanation_section,
            generated_at=generated,
        ),
        FINDINGS_PAGE: render_findings_page(
            generated_at=generated,
            challengers=challengers,
            challenger_week_previews=challenger_week_previews,
            challenger_prospective_records=challenger_prospective_records,
            artifacts_root=artifacts_root,
            active_model_id=str(model_id) if model_id else None,
        ),
        TEAM_EXPLORER_PAGE: render_team_explorer_page(
            team_states,
            generated_at=generated,
        ),
        POOL_PAGE: render_pool_workbench_page(
            artifacts.predictions,
            PoolRules.from_defaults(),
            season=artifacts.metadata.get("season"),
            week=artifacts.metadata.get("week"),
            model_id=str(model_id) if model_id else None,
            generated_at=generated,
            best_pick_game_id=(nomination.active_game_id if nomination is not None else None),
            strength_bands=served_strength_bands(artifacts_root, artifacts.active),
        ),
        LEDGER_PAGE: render_signal_ledger_page(generated_at=generated),
    }


__all__ = [
    "DISCLAIMER_FULL",
    "DISCLAIMER_SHORT",
    "FINDINGS_PAGE",
    "LEDGER_PAGE",
    "MODELS_PAGE",
    "PICKS_PAGE",
    "POOL_PAGE",
    "SITE_PAGES",
    "TEAM_EXPLORER_PAGE",
    "EraMagnitude",
    "OpenerEvaluationArtifacts",
    "PublicBoardArtifacts",
    "ServedUnionMeasurement",
    "assert_spread_explorer_matches_card",
    "build_public_site",
    "challenger_blurb",
    "confidence_word",
    "load_era_magnitude_profile",
    "load_model_explanation_html",
    "load_model_ledger_html",
    "load_opener_evaluation_artifacts",
    "load_played_chain_accuracy",
    "load_prospective_challengers",
    "load_public_board_artifacts",
    "load_served_union_measurement",
    "load_waterfall_feed",
    "load_waterfall_feed_document",
    "pick_side",
    "render_findings_page",
    "render_models_page",
    "render_picks_page",
    "render_pool_workbench_page",
    "render_signal_ledger_page",
    "render_team_explorer_page",
    "row_confidence_word",
    "spread_words",
    "waterfall_games_by_id",
]
