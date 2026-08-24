# UI-09 — Pool workbench (minimal)

**Task:** ROADMAP item UI-09 (Pool workbench, 🚧). Read its row, `docs/pool_edge_plan.md`,
and the RWB-04 bankroll engine. Implement the minimal workbench: pool rules input,
entry list, confidence ranks derived from the active model forecast format,
ownership-scenario placeholder. Reuse existing pick/probability outputs; invent
nothing about future games.

**Status:** implemented, all four quality gates green. No new experiments were
run — this is a UI/reuse task only (Binding rule 3: no scoring window was
predeclared for me, and none was needed).

## What I read (measured / read)

- **(measured this session)** `ROADMAP.md` line 408: `UI-09 | 🚧 | Pool workbench | Rules, entries, confidence ranks, ownership scenarios`.
- **(read)** `docs/pool_edge_plan.md`: the pool format is forced ATS sides for **all 272 regular-season games and all 13 playoff games** (285 cards, exactly the forced-pick metric this project evaluates); one "Best Pick" per regular-season week; the line locks Tuesday (revised once Wednesday, then frozen) while picks stay editable up to each game's own per-game deadline (SNF/MNF lock early at Sunday 16:00 ET). The model's confidence ordering has **not** proven to rank pick quality — accuracy is non-monotone across confidence quartiles (53.2%, 47.3%, 55.7%, 53.7%) (read, `docs/pool_edge_plan.md`).
- **(measured this session)** `src/nfl_ats/pool.py` already contains `build_ats_pool_card()` — forces one side per game and ranks by `|pick_probability − 0.5|` from `home_cover_probability`, plus a `PoolFormat` scoring model. This is the existing pick/probability output the workbench reuses.
- **(measured this session)** RWB-04 is `src/nfl_ats/portfolio.py` (`simulate_paper_bankroll`, `simulate_bankroll_paths`, `kelly_fraction`) — the paper bankroll engine. The workbench does not need it directly; it consumes the same `predictions` shape the bankroll engine consumes and reuses `pool.py`'s card builder, so the workbench's entry list is consistent with the rest of the pipeline.
- **(read)** `src/nfl_ats/public_board.py`: the "dashboard" is the public GitHub Pages site generator (UI-15). Pages are `index.html`, `models.html`, `findings.html`, `track_record.html`, built by `build_public_site()` and rendered through the shared `_page()` shell (which carries the licensing `DISCLAIMER_SHORT`/`DISCLAIMER_FULL` guardrails and the `<div class="ats">` chrome). The active model forecast is `recommendations.csv` with columns `game_id, gameday, away_team, home_team, spread_line, home_cover_probability`.

## What I built

New module `src/nfl_ats/pool_workbench.py` (pure logic + a body builder) and one
new page wired into the public site:

1. **Pool rules input** — `PoolRules` dataclass (`from_defaults()` encodes the
   confirmed Splash-style format from `docs/pool_edge_plan.md`; `from_dict()`
   accepts a partial override map so a future operator can tune rules without
   code edits). Exposes `total_games` = **285** (`REGULAR_SEASON_GAMES(272) +
   PLAYOFF_GAMES(13)`, measured from the doc).
2. **Entry list** — `build_entry_list(predictions)` reuses `pool.build_ats_pool_card`
   (the existing forced-pick output). One row per game, forced side + calibrated
   cover probability, ranked. Degrades to an empty frame when the forecast is
   missing the columns it needs (no active forecast yet).
3. **Confidence ranks** — `derive_confidence_ranks(predictions)` returns the
   ranking view derived from the active forecast's `home_cover_probability`
   (confidence = `|pick_probability − 0.5|`), with a `probability_meter` per pick.
4. **Ownership-scenario placeholder** — `OwnershipScenario` dataclass +
   `placeholder_ownership_scenarios()` returning `available=False` with an
   explicit "no feed integrated" note. This is intentionally a placeholder:
   no entry-popularity source exists in this repo, so no ownership numbers are
   invented (Binding rule 2: nothing about future games / no fabricated data).

The page is rendered by `public_board.render_pool_workbench_page()`, which calls
`pool_workbench.build_pool_workbench_body()` and wraps it in the shared `_page()`
shell so the licensing/disclaimer guardrails apply unchanged. It is added as
`pool.html` to `SITE_PAGES` and emitted by `build_public_site()` alongside the
other four pages. It reuses the Best Pick nomination already computed in
`build_public_site()` to badge the week's Best Pick row.

## Reuse, not invention

- Forced picks, sides, and probabilities come entirely from `pool.build_ats_pool_card`
  over the active `recommendations.csv` — no model retraining, no new features,
  no future-game assumptions.
- The confidence ranking is the existing forecast's calibrated cover probability;
  the page states in-line that the ordering is a model aid only and has **not**
  proven to rank pick quality (read, `docs/pool_edge_plan.md`). I report the
  confidence **ordering**, never a "contains zero rejects the signal" claim.

## Gates (all measured this session)

| Gate | Command | Result |
|---|---|---|
| Format | `ruff format --check .` | clean (578 files) |
| Lint | `ruff check .` | All checks passed |
| Types | `mypy src` | no issues found in 103 source files |
| Tests | `pytest` (basetemp outside repo) | **1815 passed, 5 skipped** (the 5 skips are pre-existing data-absent skips) |

New/changed tests: `tests/test_pool_workbench.py` (13 tests: rules defaults/overrides,
entry-list ranking + empty-state, confidence ranks, ownership placeholder, body
sections, and a public-safe render check) and an update to
`tests/test_public_board.py::test_build_public_site_writes_four_pages` to expect the
fifth page and include it in the nav link assertion.

## Files

- Added `src/nfl_ats/pool_workbench.py`
- Added `tests/test_pool_workbench.py`
- Modified `src/nfl_ats/public_board.py` (POOL_PAGE constant, SITE_PAGES entry, import, `render_pool_workbench_page`, wired into `build_public_site`, `__all__`)
- Modified `tests/test_public_board.py` (nav tuple + page count for the 5th page)

## Out of scope (placeholder, not built)

- Live ownership / entry-popularity feed and contrarian Best-Pick leverage
  (no data source; left as `OwnershipScenario` placeholder).
- Any pool scoring simulation (e.g. probability-of-first-place) — that machinery
  already exists in `pool.py`'s contest simulator and is a separate lever from
  this workbench's display role.
