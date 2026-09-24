# Conditional signal atlas (MOD-19)

## State — 2026-09-24 (visual-check unit DONE, uncommitted; found+fixed a real mobile CSS bug)
Resumed from "rail shows 3 signals correctly, two-column-vs-stacked at wide desktop and mobile
width both unconfirmed" (prior unit hit its tool-call cap before checking). This unit:
- Cleaned up first: killed the two leftover `python -m http.server 8934` processes (PIDs 25840
  parent/37372 child, from the previous unit) and reused/then closed the previous unit's
  claude-in-chrome tab (`1132369014`) pointed at `http://127.0.0.1:8934/findings.html` (that
  server+render dir, `<scratch>/atlas_render/`, was still valid/unchanged, so reused rather than
  re-rendering first).
- Viewed `docs/design/mockups/conditional-signal-atlas.png` for real (prior unit's gap: "never
  re-viewed the PNG"): confirms rail + chart-card + context-card side-by-side is the target
  desktop layout.
- **Wide-desktop check.** `resize_window` to 1800px then 2900px had NO effect —
  `window.innerWidth` stayed pinned at 1172 regardless of requested width; root cause found via
  `screen.width`/`screen.availWidth` = **1200** (this automation environment's virtual display is
  physically only 1200px wide; `outerWidth` did grow to 2902 but the viewport itself cannot
  exceed the physical screen). 1172 < the CSS's `max-width:1200px` breakpoint, so a real resize
  can never reach two-column in this sandbox. Worked around it: injected an `<iframe>` (via
  `javascript_tool`, not a real resize) sized to `width:1900px` pointed at the same
  `findings.html` URL — CSS media queries evaluate against the iframe's own viewport, not the
  parent window/screen, so this genuinely exercises >1200px layout. Confirmed both by
  `getComputedStyle` (`.atlas-workspace-body` `grid-template-columns` = `"648.737px 381.628px"`,
  i.e. real two columns, not `1fr`) and visually by screenshot (scrolled the outer page to see
  the right column): chart card and "Selected context" card render side by side, matching the
  mockup. Also read the CSS directly (`board_terminal_style.css` lines 1120-1125 base rule =
  `grid-template-columns:minmax(340px,1.7fr) minmax(280px,1fr)`; only overridden to `1fr` inside
  `@media (max-width:1200px)` at line 1456-1458; grepped for all `atlas-workspace-body` rules —
  only those two touch `grid-template-columns`, nothing else) to confirm this isn't
  viewport-size-dependent luck. **Wide-desktop gap from the prior unit is now closed: confirmed
  two-column, both structurally and visually.**
- **Mobile check (≤680px, not attempted at all by the prior unit).** Same iframe technique at
  `width:390px` (`window.innerWidth` inside iframe = 387). Found a **real bug**: the two split
  tabs ("Time of season" / "Spread size") rendered on the same line with **no gap**, reading as
  literally `"Time of seasonSpread size"` — not just visually cramped, actually unreadable/
  unusable. Root cause: `@media (max-width:680px)` sets `.atlas-tabs{ display:block }` (dropping
  the desktop flex rule's `gap:20px`, since `gap` only applies under flex/grid), but nothing at
  that breakpoint sets `display:block` on `.atlas-tabs button` itself, so the buttons keep the
  UA-default `inline-block` and sit flush against each other. This is a regression from this
  lane's rendering unit: tabs used to be static `<span>`s (per this lane's own earlier note) and
  are now `<button>` elements from the new `.atlas-tabs button` base-reset rule added this
  session; the existing `@media (max-width:680px)` block predates that change and was never
  updated for button semantics.
  - **Fix applied** (`src/nfl_ats/board_terminal_style.css`, inside the existing
    `@media (max-width:680px)` block, right before the existing `.atlas-tabs .is-active` rule):
    added `body[data-interactive-page="findings"] .atlas-tabs button{ display:block; }`. Minimal,
    CSS-only, no HTML/JS touched. (`.is-active`'s existing higher-specificity `display:inline-block`
    override still wins for the active tab, unchanged — only the inactive tab's display was
    missing before.)
  - Re-ran `nfl-ats publish-board --site-destination <scratch>/atlas_render` to pick up the CSS
    (it's inlined into the HTML `<style>` block at render time, confirmed via
    `grep -o "<style"` — no external stylesheet link, so a stale render would not have shown the
    fix). Reloaded the iframe with a cache-busting query param and re-screenshotted: tabs now
    stack on separate lines ("Time of season" / "Spread size" clearly separated, active one
    gold-underlined). Clicked the `[data-atlas-split]` "Spread size" button via JS at mobile
    width: split correctly switched (chart rows became "Spread 7 or less" +2.5 / "Spread 7.5 or
    more" +2.8, context card updated to "Spread 7 or less" / 95.8%), confirming click-through
    still works post-fix. `document.documentElement.scrollWidth` (372) stayed `<`
    `window.innerWidth` (387) throughout — **no horizontal scroll** at mobile width, before or
    after the fix (the bug was a readability/usability defect, not an overflow defect).
- **Verification re-run after the CSS edit**: `.tools/uv.exe run --no-sync pytest -q -k "board or
  findings"` → **338 passed**, 0 failed (unchanged from the prior unit's baseline — a CSS-only
  change, no Python touched, so `ruff`/`mypy` were not re-run; nothing in the moratorium'd test
  suite asserts on CSS). Did not re-run desktop interactive click-through (rail switch/tab
  switch/detail update) since the prior unit already verified that at the default ~1172px width
  and this unit's iframe checks only needed to confirm the two breakpoints' layout math, not
  redo full interaction coverage.
- Cleanup done: killed the `http.server 8934` processes (both the venv-python parent PID 25840
  and its uv-python child PID 37372 — verified gone via a follow-up process list showing only
  unrelated shell processes), closed the claude-in-chrome tab (`1132369014`, group auto-removed
  since it was the last tab). No stray tabs or background servers left by this unit.
- **Files changed this unit**: `src/nfl_ats/board_terminal_style.css` (one new 3-line rule, see
  above). No other files touched. Everything else in the working tree (`board_terminal.py`,
  `board_interactive_layout.js`, `signal_atlas.py`, `registry/conditional_signal_atlas.json`,
  `scripts/conditional_signal_atlas.py`) is unchanged from the prior (rendering) unit. No commit
  made — still all local working-tree edits, per lane scope (this subagent stops at "rendered to
  scratch, verified").

## State — 2026-09-24 (rendering unit DONE, uncommitted; hit 50-tool-call cap mid-verification)
Rendering unit (b) is implemented and passing tests. Summary for the orchestrator:
- `src/nfl_ats/board_terminal.py`: rewrote `_atlas_explorer_html` to be generic over any
  non-"overall" cells (dropped the hardcoded `weeks_1_4/5_12/13_18` phase_order/labels/titles
  dicts; now uses `cell["label"]` directly, so it works for both `week_in_season` (3 phase
  rows) and `spread_band` (2 band rows) splits unchanged). Added new
  `_signal_atlas_family_panel_html(family, *, tabs_html, visible, bootstrap_draws)` — renders
  one family's full panel (header h2=signal_label, subtitle, tabs, `_atlas_explorer_html`
  chart+context+season-explorer, then per-family "Evaluation views" (3 `_atlas_evaluation_html`
  calls, unchanged reused) + gap + footnote + "Scoring and calibration" `_atlas_metrics_html`).
  Rewrote `_signal_atlas_section_html` to group `report["families"]` (6 entries) by signal (3),
  build one rail `<button data-atlas-signal>` per signal (was a static single div before) and
  one `.atlas-family-panel[data-atlas-family="{signal}__{split}"]` per family (6 panels, only
  the first `hidden`-toggled visible), each with its own `<div class="atlas-tabs">` of
  `<button data-atlas-split>` (2 per signal: `week_in_season`→"Time of season",
  `spread_band`→"Spread size", via `SPLIT_LABELS`/`split_label` already in the report). Shared
  "How to read this comparison" + limitations `<details>` now renders once at the section level
  (methodology text generalized off "combined game-situation signal" wording). Deleted the
  false "Individual situations have not been measured separately here" sentence. All numeric
  rendering still goes through unchanged `_atlas_signed`/`_atlas_percent`/`_atlas_interval`
  helpers (no "contains zero" anywhere — confirmed by grep on rendered output, see below).
  `_atlas_phase_seasons_html`: changed the `<details>` from a document-unique `id="atlas-seasons"`
  (would collide with 6 panels in the DOM) to `data-atlas-season-explorer` attribute, scoped by
  JS via `panel.querySelector(...)` instead of `getElementById`.
- `src/nfl_ats/board_interactive_layout.js` (`page==='findings'` block, ~line 143): replaced the
  single global `data-atlas-context` handler with a per-panel-scoped version. New state model:
  `showFamily(signal, split)` toggles `hidden` on the one matching `[data-atlas-family]` panel
  and sets `aria-current` on the matching rail button. Rail buttons preserve the current split
  when switching signal (reads `activePanel().dataset.atlasSplit`); tab buttons preserve the
  current signal when switching split (reads `panel.dataset.atlasSignalOwner`). Row
  click/keyboard (arrow/home/end) selection and the "Explore seasons" button are now
  `$$(selector, panel)`-scoped per panel so hidden panels' identically-named cell ids
  (`weeks_1_4` etc. repeat across all 3 week_in_season families) never cross-fire.
- `src/nfl_ats/board_terminal_style.css`: `.atlas-signal-choice` was previously a single
  always-gold `<div>`; changed to base style = plain button (transparent bg, grey border/text,
  `width:100%`, `cursor:pointer`, hover state) + `[aria-current="true"]` override = the original
  gold-highlighted look (moved `.atlas-signal-mark` gold fill behind the same
  `[aria-current="true"]` guard, default transparent). Added `.atlas-tabs button` base reset
  (was static `<span>`s before; `.is-active` rule kept, now just overrides border-color/color/
  weight on top of the shared base layout rule instead of owning the layout alone). Rail
  `<small>` subtitle removed (mockup shows signal name only, no rail subtitle) — the CSS rule
  for `.atlas-signal-choice small` is now unused but harmless, not deleted (budget).
- Verification run for real:
  - `.tools/uv.exe run --no-sync ruff format --check src/nfl_ats/board_terminal.py` → "1 file
    already formatted"; `ruff check` same file → "All checks passed!".
  - `.tools/uv.exe run --no-sync mypy src` → "Success: no issues found in 240 source files".
  - `.tools/uv.exe run --no-sync pytest -q -k "atlas or findings or board"` → **338 passed**,
    0 failed (was 333 passed/5 failed before this unit; the 5 `KeyError: 'evaluations'`
    failures are gone, no regressions). No test files/functions added or expanded, per the
    moratorium; none needed editing (the 5 failing tests only called
    `render_findings_page`/`render_model_page` end-to-end, asserted no exception + generic
    content, nothing atlas-shape-specific).
  - Rendered for real: `.tools/uv.exe run --no-sync nfl-ats publish-board --site-destination
    <scratchpad>/atlas_render` (NOT docs/) → wrote `index.html, model.html, history.html,
    findings.html` successfully, no crash.
  - `grep -o 'data-atlas-family="[^"]*"' findings.html | sort -u` → all 6 expected families
    present: `composition_flag_sum__{week_in_season,spread_band}`,
    `market_move_toward_home__{week_in_season,spread_band}`,
    `market_move_available__{week_in_season,spread_band}`.
  - `grep -o 'atlas-signal-choice[^>]*data-atlas-signal="[^"]*"'` → exactly 3 rail buttons, one
    per signal.
  - `grep -c "contains zero"` → 0. `grep -c "Individual situations have not been measured"` → 0.
  - Visual check via claude-in-chrome at a 1536-wide window, served from a background
    `python -m http.server 8934` in the scratch render dir (file:// URLs are blocked by the
    browser tool): **rail shows 3 real signal buttons** (Combined game situations / Market move
    toward the home side / Whether a market move was available), matches mockup layout
    (left rail + right workspace, amber-highlighted current selection, thin gold left-bar
    marker). Clicked "Market move toward the home side" → header, chart rows, and all
    percentage-point numbers updated correctly (confirmed different numbers: e.g. Weeks 1-4
    went from `combined` `+3.3` to `market_move_toward_home` `+1.1`). Clicked the "Spread size"
    tab while that signal was still selected → correctly switched to the `spread_band` family
    for the SAME signal (rows became "Spread 7 or less" `+0.7` / "Spread 7.5 or more" `+3.4`,
    axis auto-rescaled to −4..+12), proving the two-axis (signal × split) state combination
    works, not just a flat 6-button fallback. Scrolled down: "Selected context" card (probability
    68.7% for "Spread 7 or less", "Why it could matter"/"What the history suggests" sections)
    renders and updates per selection, matching the mockup's right-hand detail panel content
    and wording shape.
  - **Not fully confirmed before the tool-call cap cut in**: at the 1536px window I was
    resizing to, the chart and "Selected context" card were rendering STACKED (one column)
    rather than side-by-side two-column like the mockup. This may just be the pre-existing
    `@media (max-width:1200px){ .atlas-workspace-body{grid-template-columns:1fr} }` rule in
    `board_terminal_style.css` firing because the browser's actual inner content viewport was
    narrower than the 1536 outer window size requested (window chrome/DPI), not a regression
    from this change — that media-query breakpoint and the two-column grid rule are both
    pre-existing CSS this unit did not touch. **Not yet re-verified at a wider effective
    viewport, and mobile width (≤680px) was not checked at all this unit** — the cap hit right
    after requesting a resize to 1800px, before the follow-up screenshot.
  - Left running in background, needs cleanup by whoever resumes: a `python -m http.server
    8934` process serving the scratch render dir (started via Bash `run_in_background`-style
    `(python -m http.server 8934 &)`; kill it, e.g. `taskkill` the python process, or it'll just
    die with the shell/session), and one open claude-in-chrome tab (tabId `1132369014` in this
    session, pointed at `http://127.0.0.1:8934/findings.html`) — close it if a fresh session
    can't reuse the tab group. The rendered output itself is at `<scratchpad>/atlas_render/`
    under this session's scratch dir (session-specific temp, will be cleaned up automatically;
    not part of the repo).
- No commit made. Changed tracked files this unit: `src/nfl_ats/board_terminal.py`,
  `src/nfl_ats/board_interactive_layout.js`, `src/nfl_ats/board_terminal_style.css`. Backend
  unit's changes from the prior session (`src/nfl_ats/signal_atlas.py`,
  `registry/conditional_signal_atlas.json`, `scripts/conditional_signal_atlas.py`) are also
  still uncommitted, unchanged since that session, still needed for this to work.

## Goal
Match the approved Findings mockup with an interactive explorer backed by measured, paired signal comparisons. Every pick stays inside one calibrated probability. Current unit: fill the signal rail with every family that has measured paired full/reduced split results (mockup: `docs/design/mockups/conditional-signal-atlas.png` + `.md`).

## State — 2026-09-24 (backend unit DONE, uncommitted). Rendering unit next: board_terminal.py/JS KeyErrors on the new report shape (see below).
- Fixed the broken tail in `build_signal_atlas` (`src/nfl_ats/signal_atlas.py`): deleted the stale top-level `"evaluations": evaluations` / `"in_sample_gap"` pair after `limitations`, changed `pairs.to_parquet(...)` to `combined.to_parquet(...)`. Grepped `declaration\["family"\]|declaration\["signal"\]|declaration\["cells"\]` — zero hits, no stale references remain.
- Rewrote `registry/conditional_signal_atlas.json` to schema_version 2: shared fields kept verbatim from `git show HEAD:...` (`selection_scope`/`availability_scope` copied as-is), added `"families"`: 6 entries crossing signals `composition_flag_sum, market_move_toward_home, market_move_available` x splits `week_in_season` (cells overall/weeks_1_4/weeks_5_12/weeks_13_18), `spread_band` (cells overall/short/long). `composition_flag_sum__week_in_season` is the first entry, reproducing the prior committed family's signal+split.
- Fixed `scripts/conditional_signal_atlas.py` print loop: now `for family in report["families"]: for view in (...): for cell in family["evaluations"][view]:`, tags each printed line with `"family": family["family"]`, also prints `report["look_count"]`.
- **Ran for real**: `.tools\uv.exe run --no-sync python scripts/conditional_signal_atlas.py` -> built `artifacts\signal_atlas\20260924T194540880742Z`, `load_signal_atlas` loaded it back successfully (script would have raised `ValueError("Signal atlas activation failed")` otherwise — it did not). Total `look_count = 252` (3 signals x (week_in_season 4·4·3=48 + spread_band 4·3·3=36) = 252). All 6 families x 2 views (out_of_season, chronological) x their cells printed sane finite numbers, no crash.
  - `composition_flag_sum__week_in_season` overall: out_of_season +2.528 pts [-0.265,+5.388] probability_positive 0.963; chronological +4.202 pts [+1.036,+7.567] probability_positive 0.9952 — matches the previously recorded chronological number exactly and is close on out_of_season (prior +2.462/[-0.347,+5.326]/0.9588; population has grown since that record, per lane caveat that a fresh refit need not match exactly).
  - `market_move_toward_home` overall: out_of_season +1.264 pts [-0.690,+3.440] probability_positive 0.882; chronological +1.337 pts [-0.571,+4.131] probability_positive 0.870.
  - `market_move_available` overall: out_of_season +0.000 pts [-0.818,+0.924] probability_positive 0.484 (decisive 20-20); chronological -0.382 pts [-1.337,+0.374] probability_positive 0.187 (decisive 9-13, wrong-sign direction but interval crosses zero — `unresolved_below_power`, not `wrong_sign_resolved`, since the whole interval is not on the wrong side).
  - Split-cell (`spread_band`, `week_in_season` sub-cells) numbers for all 6 families are in the script's stdout; not re-pasted here for length — rerun the script to regenerate (deterministic given the same source parquet/seed).
- Verification run: `ruff format --check` (2 files already formatted), `ruff check` (all checks passed) on `signal_atlas.py`+`scripts/conditional_signal_atlas.py`; `mypy src` — 240 files, no issues; `pytest -q -k "atlas or findings or board"` — **333 passed, 5 failed**, all 5 failures in `tests/test_board_terminal.py` (`test_findings_page_renders_real_findings`, `test_ledger_rows_appear_on_model_page_not_on_findings_page`, `test_findings_page_signal_registry_summary_renders`, `test_findings_page_real_content_carries_no_banned_boilerplate`, `test_home_side_push_finding_reaches_page_and_assistant`), every one `KeyError: 'evaluations'` at `src/nfl_ats/board_terminal.py:2796` inside `_signal_atlas_section_html` (`report["evaluations"]` — the new report has no top-level `evaluations`, only `report["families"]`). This is the predicted, expected failure for the still-unstarted rendering unit; not fixed in this unit per the assignment (board_terminal.py/JS are out of scope here).
- Not touched this unit, confirmed unchanged: `src/nfl_ats/board_terminal.py`, `src/nfl_ats/board_interactive_layout.js`, `src/nfl_ats/board_content.py`.
- No commit made. `git status`-relevant changed files this unit: `src/nfl_ats/signal_atlas.py`, `registry/conditional_signal_atlas.json`, `scripts/conditional_signal_atlas.py`, plus new artifact `artifacts/signal_atlas/20260924T194540880742Z/` and updated `artifacts/active_signal_atlas.json` pointer (both untracked/gitignored artifact output, not source).

## State (superseded) — original broken-tail diagnosis, kept for history:
- Diagnosed the extension: `registry/conditional_signal_atlas.json` held one family (`composition_flag_sum` x `week_in_season`). The served 4-term model (`FIT_FEATURES` in `src/nfl_ats/pick_probability_fit.py`) has 3 non-model terms: `composition_flag_sum`, `market_move_toward_home`, `market_move_available`. `registry/split_library.json` v2 already predeclares 2 usable splits with source columns present in the pick-probability `per_game.parquet`: `week_in_season` (already scored) and `spread_band` (`tue_open_home_spread`, cut 7.0/7.5, predeclared 2026-09-17, unscored). Plan: 3 signals x 2 splits = 6 families, all reusing the existing paired full/reduced bootstrap machinery in `src/nfl_ats/signal_atlas.py` untouched (`_fit_predict`, `_paired_predictions`, `_scores`, `_metrics`, `_bootstrap`, `_cell` all kept as-is; only the family-loop plumbing around them changed). No new feature engineering, no touch to `board_content.py` GameRow/tiebreaker/source-freshness/inactives.
- **`src/nfl_ats/signal_atlas.py` is mid-edit and currently BROKEN (has a leftover stale tail from the old single-family code). Applied so far, all correct and should stay:**
  - `LABELS` extended with `short`/`long`, `overall` reworded to "All games"; added `SIGNAL_LABELS` and `SPLIT_LABELS` dicts (plain-English names for the 3 signals / 2 splits).
  - `_paired_predictions`: `pairs` now also carries `tue_open_home_spread` (needed for the spread_band mask).
  - Added `_split_masks(frame, split)` (`week_in_season` -> weeks 1-4/5-12/13-18; `spread_band` -> abs(tue_open_home_spread) <=7.0 short / >=7.5 long; both include `overall`).
  - Replaced `_registry_batch` with `_family_registry_batch(family, directory)` (per-family cell rows, each cell dict now carries its own `"family"` key — required because `src/nfl_ats/cli_commands/registry.py` `_BATCH_SHARED_FIELDS`/`_cmd_weak_signals_record_batch` lets a per-cell `family` override the batch-level one; verified by reading that file) + a thin `_registry_batch(report, directory)` that loops `report["families"]`.
  - `build_signal_atlas` top validation rewritten for the new declaration shape: `declaration["families"]` is now a list of `{family, signal, split, cells}`; checks `full_features == FIT_FEATURES`, each family's `signal in FIT_FEATURES`, `cells[0]=="overall"`, `cells[1:] == split_library["splits"][family["split"]]["cells"]`. `required` columns list now also includes `tue_open_home_spread`.
  - Added the fit loop: for each distinct signal (3), build a synthetic per-signal declaration (`full_features`=shared, `reduced_features`=FIT_FEATURES minus that signal, shared `evaluations`/`chronological_minimum_training_seasons`) and call the **unchanged** `_paired_predictions` once — this fits `full` once per signal (redundant 3x refit of the identical 4-term model, acceptable/cheap) and `reduced` once per signal; results kept in `pairs_by_signal[signal]`; each coefficient dict tagged with `entry["signal"] = signal`.
  - Added the `families_report` loop: for each of the 6 declared families, slice `pairs_by_signal[family["signal"]]` through `_split_masks(..., family["split"])`, call the unchanged `_cell(...)` per cell/evaluation, and assemble a per-family dict with `family, signal, split, signal_label, split_label, cells, reduced_features, evaluations, look_count, look_inventory, in_sample_gap`.
  - Added `combined` parquet assembly: base columns (`game_id, season, week, home_covered, model_probability, tue_open_home_spread`) + one shared `{evaluation}_full` (from `pairs_by_signal[signals[0]]`, identical across signals) + `{evaluation}_reduced__{signal}` per signal — keeps the artifact manifest at the same 5 files (`report.json, per_game.parquet, coefficients.json, declaration.json, weak_signals_batch.json`) so `load_signal_atlas`'s fixed file-set check in the same module did **not** need editing (confirmed by re-reading it: hash-based, shape-agnostic).
  - New top-level `report` dict started: `schema_version:2, active_model_id, source_artifact, source_hashes, declaration_sha256, split_library_sha256, bootstrap_draws, seed, interval_level, look_count (sum over families), families: families_report, excluded, availability_certified, serving, limitations: [` — **the `limitations` list items themselves were NOT re-typed this session; they should already be present verbatim from the original file (this edit only changed the dict keys around them, not the list body) — verify on resume.**
- **BROKEN TAIL — fix this first.** Run `grep -n "pairs.to_parquet\|\"evaluations\": evaluations\|\"in_sample_gap\"\|saved_declaration_sha256\|weak_signals_batch.json\|ATLAS_POINTER" F:\Repos\nfl_py3\src\nfl_ats\signal_atlas.py` and read from roughly the `"limitations": [` close-bracket to the end of `build_signal_atlas` (~40 lines). There is a **stale leftover block** immediately after the limitations list closes that still has (from the old single-family version): a stray `"evaluations": evaluations,` and `"in_sample_gap": {...}` at the *top level* of `report` (these keys must be deleted — they belong per-family now, inside `families_report`, already added there; at top level `evaluations` is a leftover loop variable referring to the **last** family processed only, which is wrong/redundant and must not ship), and `pairs.to_parquet(directory / "per_game.parquet", index=False)` which must become `combined.to_parquet(directory / "per_game.parquet", index=False)` (`pairs` at that point in scope is also just the last family's signal frame, not the merged one). Everything after that (`_write(directory / "coefficients.json", coefficients)`, `_write(directory / "declaration.json", declaration)`, `report["saved_declaration_sha256"] = ...`, `_write(directory / "report.json", report)`, `_write(directory / "weak_signals_batch.json", _registry_batch(report, directory))`, the `ATLAS_POINTER` write, `return directory`) should already be correct as-is (unchanged from the original, file names/manifest keys unchanged) — just confirm no other stray references to the old singular `declaration["family"]`/`declaration["signal"]`/`declaration["cells"]` remain (grep `declaration\["family"\]\|declaration\["signal"\]\|declaration\["cells"\]` — should be zero hits after the top-of-function rewrite; if any remain in the leftover tail, remove them too).
- **NOT YET STARTED (do after the fix above, in this order):**
  1. Rewrite `registry/conditional_signal_atlas.json` from the old single-dict schema to the new list-of-families schema. Keep shared fields (`schema_version, declared_before_scoring, full_features:["model_logit","composition_flag_sum","market_move_toward_home","market_move_available"], evaluations:["out_of_season","chronological","in_sample"], arms:["full","reduced","model","market"], metrics:["accuracy_points","brier_improvement","log_loss_improvement"], bootstrap_draws:5000, seed:19092026, interval_level:0.95, bootstrap:"paired seasons, then weeks within each sampled season; fixed fitted predictions", reliability_edges:[0.0..1.0 step 0.1], chronological_minimum_training_seasons:2, serving:false, selection_scope, availability_scope` — copy `selection_scope`/`availability_scope` text verbatim from the current committed file via `git show HEAD:registry/conditional_signal_atlas.json`). Add `"families": [` 6 entries, each `{"family": "<signal>__<split>", "signal": "<signal>", "split": "<split>", "cells": [...]}` for the cross of signals `composition_flag_sum, market_move_toward_home, market_move_available` and splits `week_in_season` (cells `["overall","weeks_1_4","weeks_5_12","weeks_13_18"]`) and `spread_band` (cells `["overall","short","long"]`). `composition_flag_sum__week_in_season` reproduces the existing measured family exactly (same signal+split), so this is a superset, not a replacement.
  2. Fix `scripts/conditional_signal_atlas.py` (currently still assumes the old single-family `report["evaluations"]` shape at its print loop, lines ~22-24) to loop `for family in report["families"]:` then the existing `for view in (...): for cell in family["evaluations"][view]:` printing `family["family"]` alongside.
  3. Run once: `.tools\uv.exe run --no-sync python scripts/conditional_signal_atlas.py` (repo root, Windows path). Expect it to build a new `artifacts/signal_atlas/<timestamp>/` and update `artifacts/active_signal_atlas.json`. Read the printed per-cell JSON to sanity-check all 6 families produced sane `probability_positive`/intervals, and that `composition_flag_sum__week_in_season`'s numbers are close to (not necessarily identical to, since it's a fresh refit) the previously recorded ones in ROADMAP MOD-19 (`+2.462 pts / -0.347,+5.326 / support 0.9588` out-of-season; `+4.202 / +1.036,+7.567 / support 0.9952` chronological).
  4. `src/nfl_ats/board_site_content.py` `_load_findings_content` / `load_signal_atlas` usage is unchanged structurally (still just assigns `content.atlas = load_signal_atlas(...)`), but `src/nfl_ats/board_terminal.py` `_signal_atlas_section_html` (line ~2792) and `_atlas_explorer_html` (line ~2676) **still assume the OLD single-family `report["evaluations"]` shape and WILL CRASH (`KeyError: 'evaluations'`)** once the new report only has top-level `report["families"]`. This is the main remaining implementation: rewrite these two functions (plus maybe `_atlas_evaluation_html`/`_atlas_metrics_html`/`_atlas_phase_seasons_html`, which currently take a single family's `evaluations["out_of_season"]` list — keep taking that per-family, just call per selected family) to render **one rail entry per family** (6 entries) instead of the current single static `<div class="atlas-signal-choice" aria-current="true">` placeholder that says "Individual situations have not been measured separately here." (delete that sentence — it's now false). Reference hint already in the shipped HTML: the `<div class="atlas-tabs">` block (line ~2779, currently static text "Time of season" / "Select a row to explore") looks designed for a second-level split-dimension tab under a selected signal — i.e. rail = signal (`SIGNAL_LABELS`), tabs = split (`SPLIT_LABELS`), chart rows = cells within the selected split, matching the existing `data-atlas-context` row-click pattern in `src/nfl_ats/board_interactive_layout.js` (~line 144). Simplest correct approach if time is short: treat each of the 6 `(signal, split)` pairs as its own rail entry (skip the nested tabs, flatten to 6 buttons with a `<small>` subtitle naming the split), reusing the existing per-cell chart/detail/season markup unchanged per selected family, toggled via a new `data-atlas-family` attribute + a small JS handler mirroring the existing `data-atlas-context` handler in `board_interactive_layout.js`. Whichever approach, **every cell must show its interval and `probability_positive` in plain words (`_atlas_percent`), never the phrase "contains zero"** (existing `_atlas_signed`/`_atlas_percent` helpers already do this correctly for the one shipped family — reuse them, don't rephrase).
  5. `ruff format --check src/nfl_ats/signal_atlas.py scripts/conditional_signal_atlas.py src/nfl_ats/board_terminal.py src/nfl_ats/board_interactive_layout.js`, `ruff check` same files, `mypy src` (or at least the touched files), `pytest -q -k "findings or atlas or board"`. Note: grepped `tests/` for `signal_atlas`/`atlas` (any case) before starting — **zero hits**, so there is no existing test pinned to the old report/declaration shape; the `-k "findings or atlas or board"` selection is board/Findings-page tests in general, not atlas-specific, so don't let that name filter give false confidence — check that FindingsPageContent/board site-content tests still construct/pass without needing an `atlas=` field change (they likely pass `atlas=None`, which both old and new `_signal_atlas_section_html` treat as `return ""`, so should be unaffected).
  6. Render: `.tools\uv.exe run --no-sync nfl-ats publish-board --site-destination <scratch dir under the session scratchpad, NOT docs/>`. Open the rendered Findings page HTML and eyeball against `docs/design/mockups/conditional-signal-atlas.png` at desktop and mobile widths (this session never re-viewed the PNG — do that with Read before final comparison). Check: rail shows multiple real signals (not one static entry), no illustrative/unmeasured rows, no "contains zero" phrasing anywhere, numbers match what the script printed in step 3.
  7. Update ROADMAP.md MOD-19 row with the new measured 6-family run (numbers, look counts) once verified — per AGENTS.md only the primary orchestrator does ROADMAP/publish/commit/push; this subagent lane should stop at "rendered to scratch, verified" and hand back.

## Tried
Earlier sessions: legacy independent-flip atlas card/filter UI was the wrong interpretation of the reference (not reused). This session: chose "extend the existing paired full/reduced harness to more of the served model's own terms across the two already-predeclared, already-sourced splits" over inventing new feature engineering (individual `flag_coach`/`flag_bye`/etc. columns exist in the pick-probability `per_game.parquet` but adding them as their own paired families would mean fitting a 5-term extended model, a materially bigger methodological change deferred — noted under Open).

## Next
Backend unit (a), rendering unit (b), and the visual-check unit (c, this session) are all DONE.
Verified: `ruff format/check` clean, `mypy src` clean, `pytest -q -k "atlas or findings or
board"` → 338 passed / 0 failed (twice — once pre-CSS-fix, once after), real `publish-board
--site-destination <scratch>` render succeeded (re-run after the CSS fix too), grep-confirmed 6
families + 3 rail signals + zero banned phrasing, interactive desktop behavior (rail switch, tab
switch, detail card update) confirmed correct, wide-desktop (>1200px) two-column layout confirmed
both structurally (CSS read) and visually (1897px-effective-viewport iframe screenshot), and
mobile (390px) layout confirmed stacked/no-horizontal-scroll/usable — after fixing one real bug
found there (overlapping split-tab text, see State above). No open visual/layout gaps remain
against `docs/design/mockups/conditional-signal-atlas.png`.

**All subagent-scoped work on this lane is complete.** Nothing left for a fresh subagent to do
on the atlas rendering/visual side. Remaining steps are explicitly primary-orchestrator-only per
AGENTS.md and this lane's own instructions:
1. Update `ROADMAP.md` MOD-19 row with the measured 6-family run (numbers/look counts — printed
   by `scripts/conditional_signal_atlas.py` stdout; headline `composition_flag_sum` numbers are
   already transcribed in this file's earlier "backend unit" State section above).
2. Review the working tree diff (`src/nfl_ats/signal_atlas.py`, `registry/
   conditional_signal_atlas.json`, `scripts/conditional_signal_atlas.py`,
   `src/nfl_ats/board_terminal.py`, `src/nfl_ats/board_interactive_layout.js`,
   `src/nfl_ats/board_terminal_style.css` — six files total, all still uncommitted from prior
   units plus this unit's one CSS rule) and commit.
3. Real `publish-board` into `docs/` (not scratch), review the rendered-page diff, push so
   GitHub Pages rebuilds — per the Dashboard-and-publication rule in AGENTS.md.

## Open
- Individual situational flags (`flag_coach`, `flag_bye`, `flag_arrests`, `flag_division`, `flag_cold_visitor`, `flag_protection`, `flag_tank_zone`, `flag_interim_hc`, `flag_precip` — all present in the pick-probability `per_game.parquet`) are not part of `FIT_FEATURES` and are not covered by this unit; adding them as their own paired families would require fitting an extended (5-term) model and a fresh predeclaration, out of scope here.
- `ol_rush_continuity` split (`registry/split_library.json`) needs a snap-continuity table not present in the atlas source frame; not attempted.
- No commit made this session; all changes are local working-tree edits only (`src/nfl_ats/signal_atlas.py` mid-edit/broken, everything else in the plan not yet started). `registry/conditional_signal_atlas.json`, `scripts/conditional_signal_atlas.py`, `src/nfl_ats/board_terminal.py`, `src/nfl_ats/board_interactive_layout.js` are still exactly as committed (`git diff` will show only `signal_atlas.py` touched).
- Fitted-source and model/declaration freshness guards remain in place (`load_signal_atlas` unedited/untouched this session). Active master and deployment state are recorded by Git and HANDOFF.md.
