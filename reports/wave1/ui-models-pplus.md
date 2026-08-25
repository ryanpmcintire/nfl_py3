# Task report — ui-models-pplus (wave 2 worker)

Branch `swarm/ui-models-pplus`, commit d945808. No push; merge left to the
merge agent.

## What was done

Fixed `docs/uiux_baseline.md` item 1 (dimension-3 defect): three models.html
ledger rows carried intervals with no probability_positive:

| Row | Interval | P+ now rendered |
|-----|----------|-----------------|
| Model + seven-rule stat stack | [-1.100, 5.000] | 0.87 (own evidence 0.8745) |
| Best Pick by calibrated probability | [-3.920, 11.760] | 0.81 (own evidence 0.813) |
| Injury-news refresh flip | [0.790, 31.670] | 0.98 (nested cell 0.976) |

Root cause (read, this session): those challengers' `registry_source` fields
name no weak_signals key (scratchpad/docs citations, or a marker fragment
without a colon), so `_link_evidence` produced zero refs and the row had no
P+ anywhere — while their own evidence blocks carry a measured
probability_positive.

Changes (`src/nfl_ats/model_ledger.py`, measured by full gates):

- `LedgerRow.own_probability_positive` extracted from the challenger's own
  evidence block (`_PROBABILITY_KEYS`, direct or one nesting level down).
- `_interval_cell` renders `interval · P+ x` for every accuracy-points
  interval; explicit em dash when no P+ is measurable; promoted row exempt
  (its interval is a season accuracy-proportion CI).
- Summary sentence quotes "registered evidence P+ x.xxx" for rows whose
  registry link carries no P+, passing the `validate_ledger` number audit.
- Confidence ordering falls back to own-evidence P+ (sort key keeps
  no-probability arms last, matching prior semantics).

Tests (`tests/test_model_ledger.py`): contract test that every interval row
renders a P+ cell, plus own-evidence fallback values for the three real-row
shapes, explicit-dash case, ordering participation, summary-audit compatibility,
markdown-table parity.

Page update: `docs/models.html` ledger-view fragment replaced with
`build_and_render` output over the live `artifacts/prospective/challengers.json`
+ `registry/weak_signals.json` (manifest reconstructed to the tracked page's
model_id because artifacts are absent from this worktree). Post-splice sweep:
26/26 non-promoted rows with intervals carry a P+ cell; zero bare.

## Verification (measured this session)

- ruff format --check . : clean (637 files)
- ruff check . : clean
- mypy src : clean (105 files)
- pytest : 1859 passed, 5 skipped (basetemp outside repo)
- Contract sweep over rebuilt ledger and over tracked docs/models.html: no
  interval row without a P+ cell

Commit note: pre-commit hook's handoff refresh cannot run here (no project
venv in this isolated worktree); committed with NFL_ATS_SKIP_HANDOFF=1, the
hook's sanctioned skip. HANDOFF refresh belongs to the master-side flow.

## Rescore

reports/wave2/ui-models-rescore.md: 63.4/100 against docs/uiux_rubric.md.
Dimension 3 (provenance & honesty) estimated 6.5 → 9 (inferred baseline —
wave 1 scored this page qualitative only), +3.5 weighted points on the
affected dimension; no regression path (strictly additive provenance surface).

TASK_COMPLETE ui-models-pplus
