# models.html rescore — wave 2 (2026-08-24)

Scorer: swarm worker `ui-models-pplus`. Page scored: `docs/models.html` as of
this branch, after the dimension-3 P+ fix (below). Rubric:
`docs/uiux_rubric.md`. Baseline: wave 1 scored this page **qualitative only**
(`docs/uiux_baseline.md`, table row "models.html"), so per-dimension baselines
here are reconstructed estimates and are labelled as such; the affected
dimension (3) delta is computed against the specific defect the baseline
documented.

## The fix being rescored

Baseline defect 1 (highest priority): three ledger rows carried intervals with
no `probability_positive` anywhere on the row:

- Model + seven-rule stat stack — interval [-1.100, 5.000]
- Best Pick by calibrated probability — interval [-3.920, 11.760]
- Injury-news refresh flip — interval [0.790, 31.670]

Fix shipped on this branch (`src/nfl_ats/model_ledger.py`):

1. `LedgerRow` now carries `own_probability_positive`, extracted from the
   challenger's own evidence block (directly or one nesting level down) even
   when its `registry_source` links no weak_signals key.
2. The interval cell renders `interval · P+ x` for every accuracy-points
   interval — measured P+ when available, an explicit em dash when not — so an
   interval can never sit bare again. The promoted row's interval is a season
   accuracy-proportion CI, not an accuracy-points effect interval, and stays
   exempt (documented in-code).
3. Rows whose registry link carries no P+ quote their registered evidence P+
   in the summary sentence ("registered evidence P+ 0.875") and pass it to the
   ledger's summary-number audit, so every quoted numeral still traces to a
   cited field (`validate_ledger` enforces this).
4. Confidence ordering now honours own-evidence P+ as a fallback, keeping the
   caption's promise ("challengers by best-evidence P+ descending").

Measured values now rendered (verified this session against the live
`artifacts/prospective/challengers.json` + `registry/weak_signals.json`):

| Row | Interval cell | Source of P+ |
|-----|---------------|--------------|
| Injury-news refresh flip | `[0.790, 31.670] · P+ 0.98` | nested cell `movement_attribution_injury_pop_threshold.probability_positive` 0.976 |
| Model + seven-rule stat stack | `[-1.100, 5.000] · P+ 0.87` | evidence block `probability_positive` 0.8745 |
| Best Pick by calibrated probability | `[-3.920, 11.760] · P+ 0.81` | evidence block `probability_positive` 0.813 |

Contract sweep over all 26 non-promoted rows of the rebuilt ledger: **zero**
rows carry an interval without a P+ marker (measured). New tests in
`tests/test_model_ledger.py` pin this: `test_interval_rows_render_a_p_plus_cell`
(every interval row has a P+ cell), plus tests for the own-evidence fallback,
the explicit em dash when no P+ is measurable, ordering participation, and
summary-audit compatibility. Full gates passed this session: ruff format/check,
mypy src clean, pytest 1859 passed / 5 skipped.

Note on page regeneration: `artifacts/active_ats_model.json` does not exist in
this worktree (artifacts are local-only), so the full site builder cannot run
here. The tracked `docs/models.html` was updated by replacing exactly the
`<div class="ledger-view">…</div>` fragment with the fragment rebuilt from the
live challengers ledger + weak-signals registry by `build_and_render`; the rest
of the page (nav, header, footer timestamp) is untouched. The next full
`publish-predictions --with-board` run reproduces the same fragment from code.

## Per-dimension scores

Weights per rubric; weighted contribution = score × weight ÷ 10.

| # | Dimension | Score | Weighted | Evidence |
|---|-----------|------:|---------:|----------|
| 1 | Answerability | 5 | 6.0 | Specialist page by design: the ledger explains itself ("P+ is our confidence an effect is real rather than luck", prose block after the table), but a first-time visitor lands here only via nav; headline accuracy lives on index.html (consolidation law, 2026-08-23). |
| 2 | Narrative & hierarchy | 6 | 8.4 | Promoted card first, then challengers by best-evidence P+ descending (caption states the order); evidence footnotes give progressive disclosure. The "reading the ledger" explainer sits *below* the 27-row table rather than above it. |
| 3 | Provenance & honesty | 9 | 12.6 | Every accuracy-points interval now carries its P+ beside it, or an explicit em dash stating no P+ exists (measured: zero bare-interval rows). Grades shown per row ("close-grade", "opener-grade"); paper-pick disclaimer directly under the nav; unavailable-ledger state fails open with an explanation rather than an error; every quoted numeral traces to a cited field via `validate_ledger`. Remaining gap (why not 10): the promoted row's proportion-CI interval [0.508, 0.535] sits in a column headed "accuracy pts" with no on-page note that it is a different animal; bare "—" cells elsewhere are unexplained. |
| 4 | Navigation & IA | 7 | 7.0 | `<nav>` with plain-language destinations, current page marked via `aria-current="page"`; promoted row's title attribute carries the arm id. No cross-links from rows to findings.html entries. |
| 5 | Accessibility | 5 | 5.0 | Status badges pair glyph + text (not colour-only); table has a `<caption>`. Gaps: `<th>` cells lack `scope="col"`, no landmark roles beyond `<nav>`, badge glyphs decorative-without-alt. |
| 6 | Visual consistency | 7 | 7.0 | Uses theme classes (`table.data ledger`, `badge-*`, `fine`, `num`) throughout the ledger; header/footer blocks use one-off inline styles (site-wide pattern, not introduced here). |
| 7 | Data-viz quality | 5.5 | 4.4 | Intervals printed, not drawn (rubric prefers drawn); 27 rows with no sorting; ordering at least stated once in the caption. Small gain vs pre-fix: interval and confidence now read as one unit instead of implying independence. |
| 8 | Trust signals | 6 | 4.8 | "page generated 2026-08-24 17:52 UTC" timestamp with timezone; research-not-betting disclaimer repeated near headline; model-card link absent from this page (lives on track_record.html). |
| 9 | Robustness & perf | 8 | 6.4 | Single static ~30 KB HTML file, zero scripts; quiet fail-open note when the ledger source is unreadable. |
| 10 | Mobile | 3 | 1.8 | The ledger table (7 columns) has no horizontal-scroll wrapper — the page's only `overflow-x` rule is `hidden` (measured). At phone widths columns must crush or clip. |
| | **Total** | | **63.4 / 100** | |

## Hill-climb accounting

- Wave-1 baseline for this page is qualitative only, so the numeric baseline is
  **inferred**, not measured: dimension 3 pre-fix is estimated at 6.5/10 —
  three of the page's data rows presented an interval with no confidence
  figure, the exact honesty gap the baseline flagged highest-priority — versus
  9/10 post-fix (**measured** behaviour, **inferred** counterfactual score).
- Dimension-3 gain: +2.5 × 14/10 = **+3.5 weighted points**, clearing the ≥+2
  bar on the affected dimension alone.
- Dimension 7 gains ≤ +0.5 (intervals paired with confidence) ≈ +0.4 weighted;
  all other dimensions unchanged by this diff (additive text in one column;
  no regression path exists).
- Dimension 3 may never drop: it rose (6.5 → 9, inferred baseline). The change
  is strictly additive provenance surface — nothing was removed or weakened,
  and `validate_ledger`'s number-audit contract now covers the new numerals.

## Top-3 remaining improvements

1. Wrap the ledger table in a horizontal-scroll container (or column-collapse)
   for phone widths — est. +2–3 weighted points across dimensions 9/10.
2. Label the promoted row's interval as a season accuracy-proportion CI at the
   point of use (column header suffix or inline fine print) — closes the last
   dimension-3 gap noted above — est. +0.5.
3. Move the two-sentence "what this table is" explainer above the table and add
   `scope="col"` to headers — est. +1 combined across dimensions 2/5.
