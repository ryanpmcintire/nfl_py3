# UI-08 — Model explanation view: completion report

Task: `blog-ui08-model-explanation` · Branch: `swarm/blog-ui08-model-explanation`
Date: 2026-08-25 · Rubric: `docs/uiux_rubric.md` (dimensions 3 and 8 in scope)

## What the partial work was

The 2026-08-18 start (ROADMAP UI-08 note) built a Streamlit "How the model
decides" page reading the market-decomposition artifact. The Streamlit strip
(commit `238e685`, measured: `git log --follow`) deleted that surface; ROADMAP
UI-15 had already declared the GitHub Pages site THE dashboard, and the site's
generator is `src/nfl_ats/public_board.py`. So the completion target is
**`docs/models.html`** (`render_models_page`), which until now carried only the
Model Ledger and no explanation of what the model weighs.

The RWB-06 artifact lineage (read, `src/nfl_ats/market_decomposition.py`
lines 444–590): `nfl-ats market-decomposition` writes
`artifacts/market_decomposition/<run_id>/classification.csv` -- one row per
feature family with `weight_in_<target>`, `<target>_share`,
`refit_std_in_<target>`, `season_std_in_<target>`, and a four-bucket
`classification` -- plus raw named per-refit `coefficients.csv`.

## What shipped

New module **`src/nfl_ats/model_explanation.py`** (pure reader/builder, no I/O
writes), wired into `render_models_page` via a new fail-open
`load_model_explanation_html(artifacts_root) -> str` consumed by
`build_public_site`. The section renders:

1. **Family-weight table** ("What the model leans on"): reality weight share
   vs market weight share per family (human phrases from
   `FAMILY_PHRASES`), each row carrying:
   - a **stability label**: "steady across refits" / "jumps around between
     refits", computed from `refit_std_in_spread / weight_in_spread` against
     the declared constant `STABILITY_JUMPY_RATIO = 0.15` (page-wording
     threshold, not a modeling threshold); exact std/ratio numbers stay in the
     small print under the label;
   - a **caveat caption** per classification bucket; `unpriced_predictive`
     reads "...the market does not seem to price it -- **unconfirmed**",
     deliberately never "edge".
2. **Staleness tracking**: the run's `provenance.feature_table.sha256` is
   compared with `active_ats_model.json`'s `feature_table_sha256`; mismatch →
   visible warning ("measured on an earlier build of the model's inputs"),
   missing manifest → no claim either way.
3. **Provenance/trust line** (rubric dim 8): fit window seasons, weekly refit
   count, ridge alpha, feature profile, artifact written-timestamp in UTC,
   printed next to the numbers it describes; pointer to the artifact's own
   `coefficients.csv` for the raw named coefficients (RWB-06).
4. **Four honesty notes** travel verbatim at the foot of the section
   (diagnostic-not-discovery, family-level-only, explains-a-pattern-scores-no-
   new-picks, blind-to-the-market's-line-on-purpose).
5. **Fail-open both directions** (mirrors `load_model_ledger_html`): no run
   ever saved → honest empty-state note naming the manual command; run exists
   but unparseable → visible warning box; torn newest run → falls back to the
   previous complete run.

Deliberate scope choice, stated on the page itself: individual feature-level
coefficients are NOT rendered. Ridge smears weight across correlated features,
so a single feature's number would not mean what it looks like it means; the
page is family-only and says so.

Tests: **`tests/test_model_explanation.py`**, 11 tests covering empty state,
stability labels + captions, staleness warning matrix (stale/current/absent
manifest), unreadable-run warning box, torn-newest fallback, provenance line,
metadata-free run, wiring into `render_models_page` (with
`assert_public_safe`), and prior page shape without the section.

Not regenerated here: `docs/models.html` is a generated tracked file;
`build_public_site` requires the synchronized active-model chain, which this
worktree's `artifacts/` does not contain (measured: only
`artifacts/prospective/` present). The section ships on the next
`publish-predictions --with-board`.

## Gates (all run this session)

| Gate | Result |
|---|---|
| `ruff format --check .` | PASS (578 files formatted) |
| `ruff check .` | PASS |
| `mypy src` | PASS (103 source files) |
| `pytest` | **1815 passed, 5 skipped** (deterministic order via `-p no:randomly`) |

Note on flakiness (measured this session): under the repo's test-order
randomization, full-suite runs show varying single-test failures that are not
caused by this change -- the clean tree (stash-verified) failed a *different*
pair (`test_publishing.py::test_v2_nomination_and_the_coach_fade_overlay_do_
not_interfere`, `test_active_model.py` error) while my tree failed one tilt-
overlay test that passes in isolation both with and without my changes. With
order randomization disabled, the full suite passes cleanly with my changes.

## Rubric scoring — models.html, dimensions in scope

Scores cite evidence; anything not re-scored here is left to the page's
existing baseline.

### Dimension 3 — Provenance & honesty (14 pts): **9/10**

Evidence:

- Every number in the new table carries its meaning in prose: shares as
  percentages, stability word + exact std/ratio small print, caveat caption
  per row (`model_explanation.py`, `render_model_explanation_section`).
- Stale-inputs state is disclosed, never silently mixed with the live card
  (feature-table sha comparison; tested in
  `test_staleness_warning_tracks_the_active_manifest`).
- The blind-to-market asymmetry is printed every render, matching the module
  docstring contract inherited from the partial work.
- No profit/stable-edge implication anywhere; `unpriced_predictive` says
  "unconfirmed"; honesty notes block screenshot-separation by riding inside
  the same section.
- Not a 10 because: the table shows point shares without any interval on the
  shares themselves (the artifact carries refit std only for spread weights in
  the rendered columns; margin-side stds exist in the CSV but are not shown).
  A future pass could add a reality-share stability column for symmetry.

### Dimension 8 — Trust signals (8 pts): **8/10**

Evidence:

- Last-updated timestamp with timezone: page footer already renders "page
  generated ... UTC" (`_footer`), and the new section adds its OWN artifact
  timestamp ("artifact written YYYY-MM-DD HH:MM UTC") so readers can tell the
  weights' age from the page build's age.
- Data cutoff near headline numbers: "fit on 2013–2025 completed games · N
  weekly refits" line sits directly under the table.
- Method/provenance: ridge alpha + feature profile named; link-out to the raw
  `coefficients.csv` documented in prose.
- Not scored lower because nothing is missing; held at 8 rather than claimed
  above max since the model-card link element of the rubric is satisfied by
  the ledger page context rather than a dedicated card document.

Weighted effect estimate (inferred, per hill-climb rule): dims 3+8 span 22
points; before this change the models page carried NO explanation view at all,
so the affected-dimension gain is large relative to a 0-content baseline and
cannot regress dim 3 elsewhere (no other page was touched except the shared
import list).

## Files changed

- `src/nfl_ats/model_explanation.py` (new)
- `src/nfl_ats/public_board.py` (import, `render_models_page` param,
  `build_public_site` wiring, `__all__`)
- `tests/test_model_explanation.py` (new)
- `ROADMAP.md` (UI-08 → done)
