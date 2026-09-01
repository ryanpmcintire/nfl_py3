# The Gridiron Observatory — design spec

> **RETIRED — rejected by the project owner (2026-08). Do not use as a design
> reference.** The live design is the Ledger-Terminal lineage in
> `src/nfl_ats/dashboard/theme.py` (and `src/nfl_ats/public_board.py`'s
> `_PAGE_CHROME`); `test_no_observatory_references_remain_in_generated_pages`
> enforces its absence from the generated site. This document and its mockups
> are kept for history only.

> **Revision 2 (owner feedback, craft-over-costume pivot).** The first pass let
> thematic devices carry information; two failed review and are gone. The
> dot-matrix "scoreboard bulb" numerals are replaced by solid tabular-numeral
> readouts — confidence tiers now live in the small caption dot and its label,
> never in the digits' legibility. The "yard-line grammar" field strip is
> replaced by a precision interval track: a single rail, solid rounded
> confidence bands, distinct model/market markers, and key numbers 3/7 demoted
> to quiet dashed reference ticks. Standing rule going forward: a metaphor may
> decorate an instrument, never be one. Typography, spacing, and contrast do
> the work; the football soul lives in palette, cards, marginalia, and naming.

Status: design mockups only. Nothing here repaints `public_board.py` yet; this
document is the reviewable contract for doing so later.

Scope of this batch (owned files):

- `docs/design/tokens.css` — full custom-property sheet, both themes
- `docs/design/style_guide.html` — every token and primitive rendered live
- `docs/design/mockups/game_card.html` — one ticket-stub card, real Week 1 data
- `docs/design/mockups/waterfall.html` — drawn-play attribution, real Week 1 data
- `docs/design/mockups/model_ledger.html` — promoted/challenger ledger, real numbers

## Concept rationale

A night-game stadium seen through a chalkboard: deep slate-navy surfaces with
faint turf-texture noise, chart linework drawn as slightly textured chalk
strokes, probabilities displayed as glowing scoreboard bulbs, confidence
expressed in yard-line grammar, game cards torn like ticket stubs, rationales
circled like a coordinator's greaseboard notes. The metaphor is deliberate:
this project's output is film-room judgment under uncertainty, not a
spreadsheet of certainties. Every decorative choice is paired with a plain-text
repetition of its meaning so nothing is lost on the color-blind reader, in
print, or in the parchment away kit.

Six signature elements:

1. **Chalk-on-slate world** — slate-navy surface + turf-noise gradients;
   linework roughened by an inline SVG filter (`feTurbulence` +
   `feDisplacementMap`). No image assets anywhere.
2. **Yard-line grammar** — confidence intervals as hash-mark ranges on a field
   strip; key numbers 3/7 flagged like first-down markers; edge-vs-spread
   stated in "yards past the sticks."
3. **Scoreboard-bulb numerals** — dotted-segment display built with pure CSS
   (`background-clip: text` over a radial-gradient dot grid); glow intensity
   maps to the value via a per-instance `--glow-a` custom property. Static by
   construction (no animation), so reduced-motion needs are met trivially.
4. **Ticket-stub game cards** — perforated edges via radial-gradient masks,
   team-color stub stripe, torn-paper footer strip.
5. **Coach's marginalia** — rationales typeset in italic serif inside an
   irregular hand-drawn ellipse (`border-radius` blob), slightly rotated.
6. **Parchment away kit** — full light theme via the same custom properties
   (`--chalk` becomes ink, glows become ink-density ramps); print-friendly.

## Token table

`tokens.css` carries no comments by repo style; this table is its
documentation. Roles marked *shared* keep the names and semantics of
`src/nfl_ats/dashboard/theme.py` so a future port does not fork the vocabulary.

| Token | Night value | Parchment value | Usage |
| --- | --- | --- | --- |
| `--surface` | `#121826` | `#f5eeda` | Page background (slate / cream) |
| `--plane` | `#0d1220` | `#eee5cb` | Deeper recessed band behind cards |
| `--surface-raised` | `#1a2233` | `#fbf6e9` | Raised panels (play-card, ledger) |
| `--turf-a`, `--turf-b` | green/white alphas | olive/brown alphas | Turf-noise gradient stops (`.turf`) |
| `--chalk` | `#edf0e4` | `#33302a` | Primary linework; primary text at night |
| `--chalk-dim`, `--chalk-faint` | chalk alphas | ink alphas | Secondary/faint strokes |
| `--ink`, `--ink-2`, `--muted` | shared roles | shared roles | Text hierarchy (muted ≥ 4.5:1 both themes) |
| `--grid`, `--baseline`, `--border` | chalk alphas | ink alphas | Table rules, hairlines (shared roles) |
| `--series-model` | `#82b8fa` | `#1f5fae` | Our model / our number (fixed slot 1) |
| `--series-market` | `#ffa06c` | `#9c4712` | The market / pool's number (fixed slot 2) |
| `--series-third` | `#4fd0a1` | `#0e7a55` | Third comparator when one exists |
| `--seq-100/400/700` | blues | blues | Ordered ramps |
| `--div-neg`, `--div-pos` | red/blue | red/blue | Diverging fills (always labeled) |
| `--good`, `--good-text` | greens | greens | Synchronized/promoted state (with ✓ glyph) |
| `--warning`, `--serious`, `--critical` | amber/orange/red | darkened for AA on cream | Status states (always glyph + label) |
| `--bulb-core` | `#ffe9b0` | `#4a3a10` | Bulb dot-grid fill; overridden to ink-amber on paper tickets |
| `--bulb-glow-rgb` | `255,196,90` | `176,132,30` | Glow shadow RGB, alpha driven by `--glow-a` |
| `--field-grass`, `--field-line`, `--field-hash` | dark green / chalk | tan / ink | Yard-strip background, midfield line, hash marks |
| `--accent-flag` | `#ffd44d` | `#7d6000` | Key-number pennants 3/7 (always carry the digit) |
| `--stub-home`, `--stub-away` | silver / teal | muted variants | Ticket stub stripes (paired with team abbreviations) |
| `--paper`, `--paper-ink`, `--paper-muted` | warm paper set | brighter paper set | Ticket-stub face and its private text colors |
| `--shadow-card` | heavy night shadow | soft day shadow | Card elevation |
| `--font-ui` / `--font-display` / `--font-hand` | system stacks | same | UI text / tabular numerals / marginalia serif |
| `--radius-card`, `--radius-chip`, `--perf-r` | geometry | same | Card radius, chip pill, perforation size |
| `--space-1..5` | 4–24 px | same | Spacing scale |

Glow mapping convention: `--glow-a = clamp((p − 0.50) × 2.2, 0.02, 0.65)` for a
decision score p. 50.1% → ≈0.05, 54.4% → ≈0.31, 63.8% → ≈0.62. The numeral,
its confidence-band word (`strong` >56%, `lean` 53–56%, `slight` <53% — same
bands as `confidence_word`), and a captioned bulb-dot repeat the value, so the
glow never carries meaning alone.

## Component inventory → existing public_board sections

| Observatory component | Maps to today | Notes |
| --- | --- | --- |
| Ticket-stub game card | `_game_card` (public_board.py:806) | Same content order: kickoff kicker, matchup title, market line, pick, meter, journey, sweep details, explanation block, best-pick star + note |
| Scoreboard bulb | `viz.probability_meter` (dashboard/viz.py:106) | Replaces or supplements the meter; keeps coin-flip anchor wording |
| Yard-line CI strip (hash range) | `viz.line_journey` (viz.py:256) and `_effect_whisker` (public_board.py:1520) | One grammar for per-game lines and effect intervals |
| Key-number flags 3/7 | new, static furniture | Decorative-but-labeled scale ticks at the classic ATS key numbers |
| Chalk stroke SVG charts | `viz.sweep_curve` polygons (viz.py:147) | Same polygon math; add `filter="url(#chalk-filter)"`; st.html strips SVG, so public-site use must ship as HTML/CSS or move off st.html — flagged in implementation plan |
| Coach's marginalia | explanation blocks in `_game_card` (lines 874–943) and `_challenger_blurb` (2252) | Same prose, new typography |
| Ledger table | `render_track_record_page` challenger section (2998), `_challenger_card` (2347) | PROMOTED row first, badge glyphs ✓/▲/—, sort-glyph decorations |
| Stat tiles | `viz.stat_tile` (viz.py:45) | Hero numerals become bulbs; delta keeps triangle glyph + label |
| Status lines | `viz.status_line` (viz.py:80) | Unchanged pattern: circled text glyph + label, never color alone |
| Chips | `.chip` family | model/market role chips keep currentColor borders |
| Week board anchors | `_week_board` (999) | Cards keep stable `id={game_id}` anchors |

## Accessibility notes

- **Color-never-alone (binding):** every color signal ships with a glyph,
  label, position cue, or all three. Badges carry ✓/▲/— glyphs; statuses carry
  circled glyphs; bulbs show their own numeral plus a band word; field markers
  get legend chips naming them; key-number flags carry the digits.
- **Contrast:** body text ≥ 7:1 both themes; `--muted` ≥ ~4.6:1; status colors
  darkened on parchment (e.g. warning `#7d6000`) to clear AA; bulb-on-paper
  uses ink-amber `#8a5a00` (~4.9:1, large text). Series-role hues stay within
  the validated CVD palette from `theme.py` (adjacent-pair DeltaE ≥ 8).
- **Reduced motion:** no animations or transitions exist by default;
  `@media (prefers-reduced-motion: reduce)` hard-disables any that ever land.
- **Charts have table twins:** waterfall.html ships an exact-value table twin;
  every `role="img"` element carries a full-sentence `aria-label` describing
  each marker and interval endpoint.
- **No JS:** zero scripts in any deliverable; theme switching is two
  side-by-side static sections, not a toggle.
- **Mobile-first:** single-column flow below 420 px; tables scroll horizontally
  in a wrapper; SVGs scale via viewBox.

## Implementation plan (when this graduates from mockup)

All changes land in `src/nfl_ats/public_board.py` + `dashboard/viz.py` +
`dashboard/theme.py`; no data-layer changes.

1. `theme.py`: extend `TOKENS_LIGHT/DARK` with the observatory tokens above
   (same names, both dicts), keeping existing roles untouched so the CVD
   validation holds.
2. `viz.probability_meter`: add `bulb=True` variant emitting `.bulb` markup
   with precomputed `--glow-a` (server-side float, no client JS).
3. New `viz.yard_strip(lower80, upper80, lower50, upper50, market, fair)`:
   emits the hash-range fieldstrip + legend chips; reuse for
   `_effect_whisker` intervals.
4. `public_board._game_card`: swap the card shell for `.ticket` markup, stub
   stripe colored by home/away team accent, marginalia wrapper around the
   existing explanation HTML (content unchanged, including flip disclosures).
5. `public_board._week_board`/`render_picks_page`: best-pick star stays;
   bulbs replace hero numerals.
6. `render_track_record_page`: challenger rows gain badges; promoted row first
   when an active model exists.
7. SVG chalk filter: define once per page in the page shell (`_page`,
   public_board.py). The shared components keep pure-CSS linework (the
   original embedded-HTML host stripped `<svg>`); the filter ships only where
   the public site controls the whole document.
8. Tests: extend `tests/test_public_board.py` assertions for the new markup —
   charset-before-non-ASCII still first, `prefers-color-scheme: dark` block
   present exactly once, allowlisted fields unchanged, status lines still
   carry glyphs.

## Existing invariants preserved

- Color never alone: icon/text/position pairs every signal (tested pattern in
  `tests/test_public_board.py` / `tests/test_dashboard.py`).
- `@media (prefers-color-scheme: dark)` block present; themes swap via custom
  properties in one place; components reference role tokens, never raw hex.
- UTF-8 charset declared before any non-ASCII character (em dashes, stars).
- Picks page sorts by kickoff, not confidence; week board anchors per game;
  exactly one ★ Best Pick, regular season only.
- Historical accuracy (52.10% close-graded, week-blocked 50.12–54.24%) stays
  visually and verbally distinct from each game's decision score; the flat-
  confidence note ("no pick gets extra weight") remains on the board header.
- Overlay flips always disclosed at the card they changed, with the policy
  provenance sentence; strong-lean gate (`STRONG_LEAN_POINTS`) unchanged.
- MKT-09 licensing: only the one consensus market line and our fair line plot
  publicly; opener-archive/predicted-close stay internal.
- Research framing: research output, not wagering advice; no automated betting.

## Data embedded in the mockups (provenance)

- Teams/kickoff/line/total/pick/star/score: `CURRENT_PREDICTIONS.md` +
  `artifacts/margin_predictions/2026-week-01-20260820T005017Z/`
  (`pool_card.csv`, `recommendations.csv`) — MIA at LV, Sun Sep 13 2026,
  LV −3.5 (total 40.5), ★ MIA +3.5 at 54.4%, model fair LV −1.2,
  50% margin interval [−5.55, +10.30], 80% [−13.88, +18.75].
- Waterfall route contributions: `artifacts/market_decomposition/20260816T203751Z/attribution.parquet`,
  game `2026_01_MIA_LV` (defense −1.37, player_qb −0.60, results −0.56,
  player_continuity +0.41, elo +0.40, offense +0.31, context +0.26,
  weekly_context −0.25, intercept +0.04; Σ −1.36 → fair LV −2.1). That run
  postdates the active weak_stack card by feature profile; both footnoted in
  the mockup rather than silently mixed.
- Ledger numbers: `artifacts/prospective/challengers.json` evidence blocks
  (mod07 P+ 0.8745 [+1.97, −1.1…+5.0]; hc_year_one_fade P+ 0.9320 +0.75 pts;
  injury_value_lost P+ 0.8875 [+1.32, −0.46…+3.25]; arrests back side
  P+ 0.8562 [+0.40, −0.27…+1.08], opener grade 53.76 vs 53.36%;
  best_pick_v2 P+ 0.8130 [+3.92, −3.92…+11.76]; division_revenge opener
  P+ 0.8642 [+0.29, −0.22…+0.79]; interim_hc P+ 0.8452 [+0.04, −0.03…+0.11])
  plus the live pool read: `nfl-ats weak-signals pool --league nfl --effect-units accuracy_points`
  → pooled +0.009356 accuracy points, 95% [−0.00746, +0.02617],
  excludes_zero false, sign test 177-of-329 candidate (p = 0.186).

Placeholders remaining: none numeric. The "Agreement with promoted" column and
decorative sort glyphs are editorial, and are footnoted as such in
model_ledger.html.

---

## Appendix: theme-pack integration contract (`src/nfl_ats/site_theme/`)

Added as a later batch. This appendix is the handoff contract for the
integrator lane that wires the approved look into the generated site as an
ALTERNATE, TOGGLEABLE skin. Until that lane lands, default rendering stays
byte-identical: nothing in `public_board.py`, `dashboard/theme.py`, or
`dashboard/viz.py` references this package, and the package itself performs no
I/O.

Owned by this batch:

- `src/nfl_ats/site_theme/__init__.py` — asset paths + `render_theme_toggle_head()`
- `src/nfl_ats/site_theme/observatory.css` — the full token sheet re-scoped
- `src/nfl_ats/site_theme/toggle.js` — 59-line vanilla cycle button
- `tests/test_site_theme_pack.py` — the static contract below

### Scoping model

The toggle applies class `theme-obs` (plus `data-mode="day"` for parchment) to
`<body>` at runtime. The live site's tokens live on `.ats` inside `<body>`, and
custom properties do not inherit upward, so every override targets
`.theme-obs .ats` descendants of the themed body:

- `body.theme-obs .ats { ... }` — night tokens + turf background (default when
  the class is present)
- `body.theme-obs[data-mode="day"] .ats { ... }` — parchment palette

Specificity note (measured against `theme.stylesheet()`): `body.theme-obs .ats`
(0,2,1) beats both `.ats { }` (0,1,0) and the dark media scope
`.ats:not([data-theme="light"]) { }` (0,2,0), so injection order does not
matter; the skin wins either way.

### Injection points (from `_page`, public_board.py:284)

1. Head assets: in `_page`, immediately after `{_PAGE_CHROME.strip()}`
   (line 302) and before `</head>`, insert the link + script tags from
   `render_theme_toggle_head(prefix)`.
2. Mount div: immediately after `<body>` (line 304), before
   `<div class="ats">` (line 305), insert the mount div from the same helper.
   The JS falls back to `document.body` if it is missing, so partial injection
   degrades gracefully.
3. Alternative: the existing `scripts=` parameter of `_page` (emitted at line
   311, before `</body>`) can carry the script tag instead; the deferred head
   placement above is preferred so first paint already has the saved theme.

Because the toggle only mutates `<body>` at runtime, pages generated without
injection are byte-identical to today; there is no server-side theme branch.

### Asset delivery recommendation

GitHub Pages serves this project out of `docs/`, so the integrator lane should
copy the two files to a stable path at generation time, e.g.
`docs/site_theme/observatory.css` and `docs/site_theme/toggle.js`, then call
`render_theme_toggle_head(asset_prefix="site_theme")`. Linked beats inline:
Pages caches linked assets across page views, four pages share them, and the
inline fallback would add ~11 KB to every HTML file and force I/O
(`Path.read_text`) into the generator. If a single-file constraint ever
appears, inlining the CSS via `<style>` and dropping the `<script defer>` for
an end-of-body inline script is the documented fallback; the toggle logic does
not depend on being an external file.

The toggle button itself ships unthemed defaults in `observatory.css`
(`.theme-toggle-button`) so it is visible and legible in system-default mode
before the skin activates; only its colors are literal, everything else in the
sheet is scoped under `.theme-obs`.

### Chalk SVG filter defs

CSS cannot reference an SVG filter through a data URI, so the chalk linework
filter must ship as an inline `<svg><defs>` block once per page. The
integrator should emit exactly this snippet right after the mount div (it is
invisible, `width=0 height=0`):

```html
<svg width="0" height="0" aria-hidden="true" focusable="false" style="position:absolute">
  <defs>
    <filter id="chalk-filter" x="-5%" y="-5%" width="110%" height="110%">
      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="1" seed="7" result="grain"/>
      <feDisplacementMap in="SourceGraphic" in2="grain" scale="0.55" xChannelSelector="R" yChannelSelector="G"/>
    </filter>
    <filter id="chalk-filter-soft" x="-5%" y="-5%" width="110%" height="110%">
      <feTurbulence type="fractalNoise" baseFrequency="0.04 0.09" numOctaves="2" seed="11" result="wobble"/>
      <feDisplacementMap in="SourceGraphic" in2="wobble" scale="1.1" xChannelSelector="R" yChannelSelector="G"/>
    </filter>
  </defs>
</svg>
```

Until those defs exist, the shipped rule
`.theme-obs .ats .chalkable { filter: url(#chalk-filter); }` is inert on any
element (no element carries the class yet). Applying `class="chalkable"` to
future chart furniture is opt-in per component.

### Token mapping: mockup var -> live-site counterpart

Shared names keep their `theme.py` semantics exactly (the CVD-validated series
roles are never re-hued beyond the mockup's night/parchment values):

| Mockup (`tokens.css`) | Live site (`theme.py` / chrome) | Skin behavior |
| --- | --- | --- |
| `--surface`, `--plane`, `--ink`, `--ink-2`, `--muted`, `--grid`, `--baseline`, `--border` | same role names in `TOKENS_LIGHT/DARK` | overridden in both scopes |
| `--series-model/market/third`, `--seq-*`, `--div-neg/pos`, `--good`, `--good-text`, `--warning`, `--serious`, `--critical` | same names in `TOKENS_LIGHT/DARK` | overridden in both scopes |
| `--seq-250`, `--seq-550`, `--div-mid` | in `TOKENS_*`, absent from mockup | carried forward (night approximates `--seq-400/--seq-700`; day uses light values) so live charts keep working |
| `--font-ui` | `FONT_STACK` | identical stack, formalized as a variable |
| `--font-display`, `--font-hand` | none | new (numerals / marginalia) |
| `--radius-card`, `--radius-chip`, `--perf-r`, `--space-1..5` | hardcoded values in `theme.stylesheet()` / components | promoted to variables |
| `--turf-a/b`, `--chalk`, `--chalk-dim/faint` | none | new (background texture, stroke alphas) |
| `--bulb-core`, `--bulb-glow-rgb`, `--glow-a` | none | new (caption-dot glow; `--glow-a` has a 0.31 default, per-instance inline override expected) |
| `--field-grass/line/hash`, `--accent-flag`, `--stub-home/away/accent`, `--marker-color` | none | new (interval track, flags, ticket stubs) |
| `--paper`, `--paper-ink`, `--paper-muted`, `--shadow-card` | none | new (ticket face, elevation) |

### What the skin restyles today vs. ships for later

Restyles existing markup (no HTML changes needed): root turf background,
`.card` elevation/radius, `table.data` / `table.week-board` rules,
`.chip` pill radius, `nav.site .chip.here` flag accent, `.hero`/`.num`
monospace solid readouts, `.tip` surface, `.status.*` tints.

Ships styles for markup the integrator lane will add (currently inert):
`.ticket`, `.ticket-stub`, `.ticket-torn`, `.marginalia` (+ `.on-paper`),
`.bulb`, `.bulb-caption`, `.bulb-dot`, the post-revision `.fieldstrip*`
precision interval track, and `.chalkable`.

### Toggle contract (`toggle.js`, 59 lines)

Cycles system-default -> observatory night -> observatory day on click;
persists the mode in `localStorage` key `site-theme-pref` (`default` /
`obs-night` / `obs-day`); applies/removes `theme-obs` and `data-mode` on
`<body>`; attaches its listener with `addEventListener` only (never an inline
attribute, satisfying `assert_public_safe`'s spirit); adds no transitions
(prefers-reduced-motion needs are additionally hard-disabled in CSS); is
idempotent under a double include via the `window.__atsThemeToggleLoaded`
guard. Storage failures (private browsing) fall back to session-only mode.

### Static contract enforced by tests (`tests/test_site_theme_pack.py`)

- Both mode scopes exist and each carries the full mockup palette; night is
  `color-scheme: dark`, day `color-scheme: light`.
- Every `var(--*)` referenced under `.theme-obs` scopes is defined there.
- Nothing outside the two allowlisted toggle-button selectors escapes the
  `.theme-obs` scoping (default rendering untouched).
- Post-revision signature primitives present (solid interval bands, dashed
  quiet key-number ticks, bulb caption dot).
- `toggle.js`: <=60 lines, no `onclick`/`onload`/inline `<script>`,
  contains the storage key and the idempotency guard, touches only the
  documented DOM hooks, no network calls.
- `render_theme_toggle_head()`: link + deferred script + mount div, honors a
  custom asset prefix, no inline handlers.

### Out of scope for the integrator lane

- Any edit to `public_board.py`, `dashboard/theme.py`, `dashboard/viz.py`, or
  registry JSON while wiring the toggle (those belong to later, separately
  reviewed lanes that adopt ticket/marginalia markup per component).
- Changing any default-rendered byte: if a diff shows changes outside the
  injected block, the lane has a bug.
- Server-side theme persistence, automated wagering affordances, or new data
  exposure of any kind.
