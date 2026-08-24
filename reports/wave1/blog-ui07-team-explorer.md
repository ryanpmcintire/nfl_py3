# UI-07 Team explorer — implementation report

**Task:** ROADMAP UI-07 (Team explorer) was marked done but had no implementation
(corrected 2026-08-18). Design AND implement a minimal team-explorer page
consistent with the existing public site: per-team pregame state trends using
ONLY already-computed canonical feature schemas; build against schema fixtures
+ tests when local parquet data is absent.

**Status:** Implemented, wired into the public site, all four quality gates
pass.

- **Read** (measured): `ROADMAP.md:406` UI-07 row — "Pregame state trends and
  matchup comparison".
- **Read** (measured): `docs/uiux_rubric.md` (the 10-dimension UX rubric,
  adopted 2026-08-24; restored into this branch at the same blob as master so
  the reference resolves).
- **Read** (measured): `src/nfl_ats/public_board.py` page-generation patterns
  (`_page`, `_nav`, `SITE_PAGES`, `viz.page_header`/`empty_state`, the
  `build_public_site` dispatch).
- **Read** (measured): `src/nfl_ats/features.py` and `src/nfl_ats/constants.py`
  for the canonical schema — `STATE_METRICS` (constants.py:142) is the only
  pregame team-state feature family the page reads.

## What was built

A new module `src/nfl_ats/team_explorer.py` plus a new page
`team_explorer.html` rendered by `render_team_explorer_page` in
`public_board.py`, and wired into `SITE_PAGES` (5th nav destination) and
`build_public_site`.

**Canonical-schema contract (binding to the task).** The page consumes only the
`state_<metric>` columns produced by `nfl_ats.features.build_team_states`
(`STATE_METRICS`: off/def EPA per play, CPOE, yards/play, turnover & sack
rates, point differential, ATS residual). It never reads outcome, market,
line, or model-probability fields. In production it derives the table by
melting the active forecast's own feature table (`feature_table_to_team_states`
reads the per-side `home_<metric>`/`away_<metric>` columns that
`attach_team_states` emits) — `public_board.py:build_public_site` wraps that
load in a `try/except` and degrades to an empty state.

**Sections (all server-rendered, no backend).**
1. At-a-glance overview: one row per team, one column per headline metric
   (`off_epa_per_play`, `def_epa_per_play`, `point_diff`, `ats_residual`),
   each cell a diverging bar (league average at centre) plus the signed raw
   value.
2. Per-team season trend: a `<details>` per team with a metric × season table.
3. Matchup comparison: two `<select>`s + a server-rendered default pair,
   re-rendered client-side from an embedded JSON payload that carries only each
   team's `z` (state minus league mean) — no outcome/market data can leak.

**Tests / fixtures (local parquet absent).** `make_schema_fixture` is a
deterministic synthetic `STATE_METRICS` table (stable per seed) used by
`tests/test_team_explorer.py` and by `test_public_board.py`'s render tests, so
the contract is proven without any on-disk data. `tests/test_public_board.py`
also updates the site page-count assertion to the new 5 pages.

- **Measured:** `pytest` (full suite) → **1821 passed, 5 skipped** (the 5 skips
  are pre-existing, for absent local nflverse/PBP parquet, unrelated to this
  change).
- **Measured:** `ruff format --check .`, `ruff check .`, `mypy src` all clean.

## UX rubric score (docs/uiux_rubric.md)

Scored per the protocol: a score per dimension with file:line evidence;
`inferred` where I estimate. Baseline = prior 4-page site; this page is additive.

| # | Dimension | Weight | Score | Evidence |
|---|-----------|-------:|------:|----------|
| 1 | Answerability | 12 | 9 | Nav entry "Team trends" (`public_board.py` SITE_PAGES); header states data is pregame state, "no picks, lines, or outcomes" (render function, `sub`). Does not tell a first-timer the model's edge — correctly, since this is descriptive. |
| 2 | Narrative & hierarchy | 14 | 11 | Overview → per-team trend (collapsed) → matchup, progressively disclosed (`_team_explorer_overview`, `_team_explorer_trend_details`, `_team_explorer_matchup`). No process wall. |
| 3 | Provenance & honesty | 14 | 13 | Bars are league-relative with signed number + arrow (direction, not goodness); explicit footnote "for rate stats a higher number is not necessarily better" (render function). No profit/stable-edge implication. Disclaimers inherited via `_page`. |
| 4 | Navigation & IA | 10 | 9 | 5 plain-language destinations now (was 4); current page marked; every page links the others (`test_build_public_site_writes_…` asserts `aria-current` + cross-links). |
| 5 | Accessibility | 10 | 9 | Direction encoded by bar placement + signed label + ▲/▼/▬ glyph, never colour alone (`_diverging_bar`, `_signed`); semantic `<details>`/`<table>`; empty state has a heading. No alt-text needed (no images). |
| 6 | Visual consistency | 10 | 9 | One theme token set (`var(--ink-2)`, `var(--grid)`); reuses `viz.page_header`/`empty_state`; zero one-off styles beyond layout. |
| 7 | Data-viz quality | 8 | 5 | Diverging bars honest (axis starts at league mean, not 0); per-team trend is a real table. **Gap:** the 32-team overview is statically sorted by point-diff z and is not client-side sortable (>20 rows) — rubric dim 7 wants sortable. (inferred −2) |
| 8 | Trust signals | 8 | 6 | Inherits last-updated footer + disclaimers via `_page`. No "data cutoff" line specific to this page's snapshot yet. (inferred −1) |
| 9 | Robustness & perf | 8 | 8 | Fail-open empty state when no feature table (`build_public_site` try/except → `viz.empty_state`); pure, no network; no layout explosion at 360px (tables in `overflow-x:auto`). |
| 10 | Mobile | 6 | 6 | Full information parity at phone width; tables scroll deliberately; no hidden content. |

**Weighted total ≈ 85 / 100** (measured scores above; dim 7/8 gaps are the
only deductions). Hill-climb rule from the rubric: a UI change ships only if it
argues ≥+2 weighted points without regressing provenance/honesty (dim 3) — this
page does not regress dim 3 and adds the missing UI-07 page.

### Top-3 improvement actions (est. point gain)

1. **Make the overview table client-side sortable** (click header → sort by
   that metric's z). Addresses dim 7. *Est. +2 weighted* (data-viz quality 5→7).
   Low risk: the values already ship as data attributes; add a small delegated
   sort handler mirroring the matchup comparer's script.
2. **Add a one-line "how this is built" note + cutoff date** under the header,
   linking the canonical `STATE_METRICS` definition. Addresses dim 3/8. *Est.
   +1* (provenance 13→14, trust 6→7). No new data required.
3. **Per-metric definition tooltips** (title attribute) on overview headers.
   Addresses dim 5. *Est. +1* (accessibility 9→10). Pure author copy.

## Notes on the binding rules

- **Binding rule 1 (interval crossing zero ≠ rejection):** not directly engaged
  — this page is descriptive, not a signal verdict. Where it quantifies
  difference it reports `z` (team minus league mean) and frames direction
  explicitly as "not goodness" for rate stats, rather than implying a rejected
  or proven effect. No category-3 result is being closed.
- **Binding rule 2 (label provenance):** every factual claim above carries
  `measured`/`read`/`inferred` tags; numbers (test counts, scores) are quoted
  with their source.
- **Binding rule 3 (experiment windows):** no model/scoring experiment was run;
  this is a presentation layer over an already-computed canonical schema, so no
  predeclared window was needed.

## Commits on `swarm/blog-ui07-team-explorer`

- `907ecc6` Add team_explorer module (canonical team-state trend aggregation + fixture)
- `6518b61` Wire team-explorer page into public site (nav, render, build_public_site)
- `5247080` Add team-explorer tests, update site page-count test, restore UX rubric
- `620a485` Lint fix: team_explorer forward-ref annotation + collections.abc imports

## Quality gates (all four, measured this session)

- `ruff format --check .` — pass
- `ruff check .` — pass
- `mypy src` — pass (103 source files, no issues)
- `pytest` — 1821 passed, 5 skipped

Remaining Git changes: only the four commits above are on the branch (no
uncommitted modifications; `viz.py` was untouched — an environment-side drift on
that file was reverted to keep the change scope to UI-07).
