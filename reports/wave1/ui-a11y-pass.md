# Task: ui-a11y-pass — mechanical accessibility fixes (dimension 5)

**Branch:** `swarm/ui-a11y-pass` · **Scope source:** the wave-1 cross-cutting
audit (`reports/wave1/uix-crosscutting.md`, commit `eda0913` — not present in
this worktree; read via `git show eda0913`). Dimension 5 scored 4/10 in that
audit. This task executes its **mechanical** items only.

All contrast numbers below were **measured this session** with a WCAG
relative-luminance script (now pinned as a test, see below). All rendered-page
checks were **measured this session** by calling the generators directly and
inspecting the HTML string.

---

## Fixes shipped (each cited to its WCAG criterion)

### 1. Light-mode `--muted` fails AA normal-text contrast → darkened
**WCAG 1.4.3 Contrast (Minimum).**

- Measured before: chrome light `--muted #8a8a84` = **3.47:1** on `#ffffff`,
  **3.32:1** on `#fafaf8`; theme light `muted #898781` = **3.50:1** on
  `#fcfcfb`, **3.41:1** on `#f9f9f7`. All fail the 4.5:1 normal-text bar while
  driving 11–12px text (`.fine`, `.kicker`, `.axis-label`, `td::before`
  data-labels) — exactly the sizes that need it.
- Fix: `src/nfl_ats/public_board.py` `_PAGE_CHROME` light block `--muted:
  #8a8a84` → **`#6b6b65`** (measured **5.36:1 / 5.13:1**); `src/nfl_ats/dashboard/theme.py`
  `TOKENS_LIGHT["muted"]` → **`#6b6963`** (measured **5.35:1 / 5.21:1**).
- Dark modes measured passing already and were left untouched: chrome dark
  `#7d828b` = 5.07:1 / 4.73:1; theme dark `#898781` = 4.85:1 on `#1a1a19`,
  5.41:1 on `#0d0d0d`.
- Also measured AA-passing in both modes (no change needed): good-text,
  critical, serious, ink-2 tokens.
- The light-block hex budget test (`test_light_palette_hex_budget`, ≤10 hexes)
  still passes: `#8a8a84` remains in the set via `--series-third`; no net new
  hex was added to the light block.

### 2. Semantic landmarks: `<main>` + `<footer>`
**WCAG 1.3.1 Info and Relationships.**

- Before (audit, verified): shell was `<body><div class="ats"><div class="wrap">…`
  with zero `<main>` and a plain-`<div>` footer on every page.
- Fix in `_page()` (`public_board.py`): body content wrapped in
  `<main id="main-content">`; `_footer()` now emits a `<footer>` element with
  identical inline styling. `<nav>` already existed.
- Verified rendered picks page this session: exactly one `<main>`, one
  `<footer>`. All six pages share `_page()`.

### 3. Page title promoted to `<h1>`; section headers to `<h2>`
**WCAG 1.3.1 Info and Relationships.**

- `viz.page_header()` emitted `<h2 class="title page-title">`; every page
  opened at h2 or h3, so no document had a top outline level. Now emits
  **`<h1>`** (classes unchanged, so rendering is pixel-identical — both size
  and weight come from `.title`/`.page-title`).
- The models page built its title via `_section_header` (an `<h3>`) instead of
  `page_header`; switched `render_models_page` to `viz.page_header` so every
  page has exactly one `<h1>`.
- `_section_header()` (`public_board.py`) and `_section()`
  (`pool_workbench.py`) promoted `<h3>` → `<h2>` so section titles nest
  directly under the page `<h1>`; deep-dive game blocks stay `<h3>` under the
  "Game notes" `<h2>`. Outline per page is now h1 → h2 → h3.

### 4. Skip link + visible keyboard focus
**WCAG 2.4.1 Bypass Blocks; WCAG 2.4.7 Focus Visible.**

- Added `<a class="skip-link" href="#main-content">Skip to content</a>` as the
  first element inside `.wrap` (hidden off-canvas until focused). Verified
  rendered present this session.
- Added `:focus-visible` outline rules (`2px solid var(--series-model)`,
  `outline-offset: 2px`) for `a`, `summary`, and `input` in `_PAGE_CHROME`;
  previously zero focus-visible styling existed anywhere (audit-measured,
  re-verified).

### 5. Informative glyph flags: aria-labels, not title-only
**WCAG 1.1.1 Non-text Content (and 4.1.2 Name/Role/Value).**

- The Best Pick star (`★`) and overlay-flip arrows (`↔`) on the week board
  carried meaning only through `title=`, which screen readers do not reliably
  announce and touch devices never show (audit finding, re-verified at
  `_week_board`). Each span now ships `role="img"` + `aria-label` alongside
  the unchanged `title`.
- Already compliant, verified this session, no change needed: all chart
  primitives carry `role="img"` + `aria-label` (probability meter, sweep,
  line journey, season bars, family comparison, contribution bars); the
  spread-explorer range input has an `aria-label`.

### 6. Keyboard operability audit — no code change required
**WCAG 2.1.1 Keyboard.**

- Measured this session: all interactive controls are natively keyboard
  operable — nav links are real links, disclosure uses native
  `details/summary`, the spread explorer uses native `<input type="range">`,
  and the only page script (`interaction_script`) wires mousemove-only hover
  enhancement with no click handlers and no keyboard traps. Every sweep chart
  also ships its "View as table" twin, so pointer-only interactions gate no
  content.

### Regression tests added (`tests/test_public_board.py`)
- `test_muted_text_tokens_meet_aa_normal_text_contrast` — computes WCAG ratios
  in-test and pins ≥4.5:1 for chrome + theme muted tokens on all their light
  and dark backgrounds (guards 1.4.3 permanently).
- `test_page_shell_ships_landmarks_skip_link_and_visible_focus` — pins skip
  link, single `<main>`, `<footer>`, focus-visible rule (2.4.1/2.4.7).
- `test_informative_glyph_flags_carry_aria_labels_not_title_only` — pins the
  role/aria-label markup on both flags (1.1.1).
- `test_every_page_opens_at_h1_and_sections_nest_below_it` — pins one
  `<h1 class="title page-title">` per page and `<h2>` section headers (1.3.1).
- One existing assertion updated from `</h2>` to `</h1>` because it pinned the
  old heading defect (`test_index_has_exactly_one_24px_number_the_crowned_stat`);
  no safety or public-audience contract was touched.

---

## Skipped as requiring redesign (not mechanical) — for the merge agent's backlog

1. **Keyboard access to sweep-chart tooltips** (best-practice 2.1.1
   enhancement): the crosshair/tooltip is mousemove-driven; giving keyboard
   users an equivalent requires redesigning the interaction layer, not adding
   attributes. Content parity already exists via the table twins, so this is
   an enhancement, not a conformance failure.
2. **Heading `id`s + working deep links** (audit backlog item 3): touches
   every section builder and anchor wiring; belongs to the dimension-4 IA
   pass, not a mechanical a11y fix.
3. **Fifth nav destination** (backlog item 5): information architecture, needs
   a new page.
4. **Deliberate mobile treatment for `.data`/`.ledger` tables** (backlog item
   2): layout redesign (dimension 10).
5. **`--series-third` shares the failing `#8a8a84` hex in light mode**: it is
   part of the palette whose CVD/contrast properties are documented as
   validated (`theme.py` header), so changing it is a design decision, not a
   mechanical fix. Note it renders chart marks, not text, so 1.4.3 does not
   apply to it directly; non-text contrast (1.4.11) was not flagged by the
   audit.

## Gates

All four run this session, all green:

```
ruff format --check .   → passed
ruff check .            → passed
mypy src                → passed (105 files)
pytest                  → 1859 passed, 5 skipped (all environment-conditional:
                          local parquet/artifacts absent), 0 failed
```

Rendered-output verification run after formatting changes: skip link ✓,
single `<main>` ✓, `<footer>` ✓, exactly one `<h1>` ✓, focus-visible rule ✓,
chrome muted `#6b6b65` present ✓ (measured via generator invocation above).

Files changed: `src/nfl_ats/public_board.py`, `src/nfl_ats/dashboard/viz.py`,
`src/nfl_ats/dashboard/theme.py`, `src/nfl_ats/pool_workbench.py`,
`tests/test_public_board.py`. No data/, artifacts/, or model files touched.
