# docs-modeling-fix — flag unrecoverable PageRank/HITS numbers in docs/modeling.md

Task: execute the CORRECT rows assigned to this worker from
`reports/wave1/hyg-docs-rot.md`. **Provenance note:** that report file does not
exist in this worktree (*measured*: `find reports -iname "*hyg*"` returned no
matches; `reports/wave1/` holds only the blog-rwb07/rwb09/rwb12 and ui07/ui09
reports). The task prompt itself carried the substance, which I verified
independently before editing.

## Verification performed (read/measured, this session)

- **No surviving artifact** (*measured*): searched the worktree for
  `*graph*` / `*pagerank*` artifacts — only source (`src/nfl_ats/graph_ratings.py`)
  and its test exist; nothing under `artifacts/` (which contains only
  `prospective/`).
- **Registry entry** (*read*: `registry/weak_signals.json` lines ~4297-4326):
  `graph_schedule_rating_brier`, classification `unresolved_below_power`,
  effect −0.000186 brier, interval [−0.000376, +0.000005],
  `probability_positive` 0.028, notes state "No stored artifact exists for
  this screen ... only ROADMAP/docs/modeling.md prose survives."
- **Open-defect confirmation** (*read*: `docs/revisit_list.md` lines 59-67):
  "The PageRank/HITS screen's artifact is absent from disk entirely" and the
  lean is "a lean, not a refutation".
- **Audit detail** (*read*: `docs/closure_audit.md` §3, lines 263-322):
  artifact missing, sign-test p≈0.070 / p≈0.289 on the season splits,
  recommendation to treat as a consistent lean, not resolved evidence.
- **Note**: the task prompt's figure "P+~0.028" matches the registry's
  recorded `probability_positive` (*measured*, above).

## Changes made

### `docs/modeling.md`

1. Inserted a provenance-warning blockquote after the market_context graph
   comparison paragraph: flags the PageRank/HITS figures as prose-only and
   unrecoverable, points to the `graph_schedule_rating_brier` registry entry
   (`unresolved_below_power`, P+ ≈ 0.028), and cross-links
   `docs/closure_audit.md` §3 and `docs/revisit_list.md`.
2. Softened the closing sentence of the outcome-model paragraph ("This rules
   out default promotion...") to attribute the claim to the surviving prose
   record and to state, per `docs/closure_audit.md` §3, that it rests on a
   consistent lean under selection/season sign counts — not a resolved
   interval — with an unrecoverable artifact. Per binding rule 1, nothing was
   framed as a zero-crossing rejection; the lean is reported via
   `probability_positive`.

### `ROADMAP.md`

1. Line ~451 narrative paragraph ("The temporal PageRank/HITS comparison is
   also complete..."): appended two sentences noting the artifact does not
   survive, citing `docs/closure_audit.md` §3 and the registry entry, so the
   paragraph no longer reads as re-runnable evidence.
2. MOD-15 status row (line 308): appended "(underlying artifact not retained —
   prose-only record, see `docs/closure_audit.md` §3)".
3. Line 574 (the closure-audit reclassification row) already carries the
   correct language, including "No underlying artifact survives on disk, only
   prose" — left unchanged (*read*, this session).

## Gates

Documentation-only change, so per task rules only format and lint were run:

- `ruff format --check .` — 636 files already formatted (*measured*).
- `ruff check .` — all checks passed (*measured*).

No code changed; `mypy src` / `pytest` not required by the task contract and
not run. No experiment windows were opened (binding rule 3: documentation
only). No files under `data/`, `artifacts/`, or any fitted model were touched.

## HANDOFF.md note

The tracked pre-commit hook refreshes `HANDOFF.md` before every commit. In
this isolated worktree the local `data/`/`artifacts/` trees are absent, so a
refresh overwrites the live model-evidence section with fresh-clone
placeholders (*measured*: ran `python -m nfl_ats handoff`; diff removed the
active-model lines and marked all inventory entries missing). That degraded
file was reverted rather than committed; the commit used the hook's own
sanctioned escape hatch (`NFL_ATS_SKIP_HANDOFF=1`) so the merge agent
integrates an untouched handoff.

## What was NOT done

- No changes to `registry/weak_signals.json` — the entry already exists and is
  correctly classified; recording again would duplicate it.
- `docs/closure_audit.md` untouched — it is the audit of record being
  cross-linked.
