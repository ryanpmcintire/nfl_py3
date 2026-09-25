# Backlog batch 2026-09-25

## Goal

Owner asked (2026-09-25) for a standing third lane that works through the
remaining backlog in bounded units alongside SIM-04 planning and XLG-09.
Done when each item below is shipped, recorded, or blocked with its reason.

## State

- Batch 1 dispatched 2026-09-25:
  - (a) DONE (subagent, see Tried). Reconcile every drafted-but-unrun
    `weak-signals record` command across active lanes against the registry;
    run the missing ones.
  - (b) UI-20 reader-facing improvement: history page opener-vs-close grading
    side by side (queue item h), or the next unshipped queue item. Not
    started.
  - (c) Week 3 challenger records (lane `week3-research-recorders`) dispatched.
  - (d) MKT-08 per-player news value decomposition (lane `news-trigger-refresh`) dispatched.

## Tried

- (a) 2026-09-25 subagent: reconciled all 7 named lanes against
  `registry/weak_signals.json` (7,253 entries before, 7,254 after). Six lanes'
  drafted cells were already recorded: mod18-spread-regime (2),
  total-conditioned-lattice (2), lead59-archive-battery type-trait bins (2),
  line-move-regrade-legacy b2/b3/b4 (14, two of them correctly upgraded to
  `refuted_mechanism`/`wrong_sign_resolved` by whoever ran them, not the stale
  draft's `unresolved_below_power`), market-move-decomposition (6).
  opener-population-backfill has no drafted commands (by design). Only one
  cell was actually missing: pooled-signal-model's unit 6 vs-model-only
  companion -- filled its numbers from the unit's own results.json and ran it
  (`pooled_signal_sixth_fit_vs_model_only`, `unresolved_below_power`). No
  errors, no withheld terminal-classification commands found. Each touched
  lane's Next section got a dated confirmation line. opener-error-transfer
  skipped per instruction (owned by another agent).

## Next

- On return: verify, commit, push; refill from the lane index (`docs/lanes/README.md`).

## Open

- None.
