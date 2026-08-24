# Wave-2 execution plan (from wave-1 audits)

Built entirely from swarm audit reports (`reports/wave1/hyg-*.md`). Line
estimates are the auditors'; treat as targets, not promises.

## Scale of the opportunity

| Surface | Finding | Recoverable |
|---|---|---|
| `scripts/` (88k lines) | ≥15-file shared boilerplate blocks | **~11,928 lines** |
| `cli.py` | parser/registration duplication | **~1,800–2,200** (32–40%) |
| CFB cluster | mechanical duplication | ~865 mechanical (+190 design-care) |
| tests/ overlay+tilt scaffolds | 18 files, 9,588 lines | ~1,010 via `tests/_overlay_test_kit.py` |
| best_pick_nomination.py | two near-complete function clones | ~370 lines |
| public_board + findings_content | inline styles vs theme tokens | ~150–190 |
| pyproject deps | unused dependencies identified | removal |
| experiment_runner | loop parameterisation | modest (~70) |

## Correctness debt (do before/with refactors)

1. **models.html P+ gap**: three ledger rows lack `probability_positive`
   (dimension-3 violation). Data exists in registry. Then rescore page.
2. **`docs/modeling.md`**: presents PageRank/HITS numbers (-0.000186 Brier,
   P+~0.028) as live findings with **no surviving artifact** — matches the
   open defect in `docs/revisit_list.md`. Flag unrecoverable, cross-link
   `docs/closure_audit.md`.
3. **CLV-by-season re-task**: real API is
   `build_pairing_table(root: Path, ...)` over the market-capture archive;
   thread the data root through or degrade explicitly. Never guess APIs.

## Wave-2 task assignment

Refactors are gated (full suite must pass; safety/contract tests may not be
weakened). UI work must argue >= +2 weighted rubric points without regressing
dimension 3.
