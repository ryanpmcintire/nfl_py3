# Backlog batch 2026-09-26

## Goal
Work remaining backlog 2026-09-26: triage the 16 research lanes named by the
owner, close every one whose Next/Open showed nothing left to run, and keep
open only lanes with a real remaining action.

## State
Closed 16 lanes to `docs/lanes/done/` (each got a "Closed 2026-09-26" note at
the top of its State section, verified against `registry/weak_signals.json`
and git history, not just the lane's own text):
lead59-archive-battery, line-move-regrade-legacy, market-move-decomposition,
mod18-spread-regime, total-conditioned-lattice, opener-population-backfill,
pooled-signal-model, conditional-signal-atlas, scheduler-once-timeout,
lane-replay-dream-rsi, prospective-leads-2026-09-23, opener-error-transfer,
surface-switch-fitted-term, lead65-protection-window-split,
clv-metric-everywhere, every-metric-every-experiment.

Verification done, not just trusted: registry grep confirmed all named cells
present (e.g. `market_move_decomposition_e_all_books_median_active_window`,
`total_conditioned_key_number_lattice_{log_loss,accuracy}`,
`mod18_spread_regime_{served_2020_2025,extended_2011_2025}`,
`pooled_signal_sixth_fit_vs_{four_term,model_only}`, `opener_error_transfer_v5`
family, `surface_switch_fitted_term` family, `special_teams_return_top_quartile`,
`hc_year_one_fade`); git log/ROADMAP confirmed conditional-signal-atlas's
commit+publish and scheduler-once-timeout's fix (`786a569`) already landed;
lead59's two type-trait cells confirmed recorded and committed (`93ba72d`)
and ROADMAP LEAD-59 already carries the finding (only
`docs/officials_archive_battery.md`'s tail note is still unwritten — optional
hygiene, not blocking).

`docs/lanes/README.md` Active section and Done section both updated to match.

## Tried
Read each lane's Next/Open; where a lane's own reconciliation note said
"nothing further to run" the underlying registry/git state was spot-checked
rather than taken on faith, since several lanes carry stale pre-reconciliation
"Next" text (draft record commands, "script not yet run") below a newer
dated note that supersedes it.

## Next
Nothing queued by this batch. The one open policy question surfaced but not
decided: whether every-metric-every-experiment's backfilled cells count as
new "looks" for AGENTS.md's look-counting rule (proposal on record: same
family, flagged `backfilled` in notes, not counted) — an owner call, not a
command to run.

## Open
- every-metric-every-experiment's backfilled-look accounting question above.
- lead59's `docs/officials_archive_battery.md` tail note (cosmetic, listed
  above, never blocking).
- No lanes outside the named 16 were touched; `sim08-simulator-rebuild`,
  `backlog-batch-2026-09-25`, and other Active lanes retain their own
  real remaining actions untouched by this batch.

## Decided 2026-09-26 (root)

- Backfilled cells (a new metric scored on predictions already preserved from
  a counted fit) stay in the fit's family, flagged backfilled, and do not add
  a look: no new fit, band, or cut was chosen. Each metric is still reported
  beside the others, so no best metric is picked. This closes the question.
- Also closed: inactives-capture-empty (live Thursday capture returned 11 real
  rows), backlog-batch-2026-09-25. Fixed: Friday odds window (`b94e59d`),
  Bovada error evidence and card note plural (`ebf7e02`).
