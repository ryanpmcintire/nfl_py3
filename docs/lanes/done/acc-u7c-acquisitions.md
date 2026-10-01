# acc-u7c-acquisitions

## Goal
Flag recent mid-season starter acquisitions whose effect prior-game stats cannot contain; grade as base-model change at the opener. Parent: docs/lanes/acc-u7-new-information.md.

## State
2026-10-01 absence check done; STOPPED per instruction (an acquisition input was already graded). No script, no scoring, no registry record written.

## Tried
- transaction_wire_battery (docs/transaction_wire_battery.md, 7 cells T1-T7): post-freeze PFR roster-churn counts (all moves, IR, PS elevation) per team-week; binary presence, close-graded cover rate. Not trades-by-starter, not seller side.
- deadline_integration_drag (src/nfl_ats/transaction_flag_features.py, LEAD-23, docs/schedule_flag_battery.md ~1890-2040): PFR trade slugs Sep-Dec, high snap-share acquirer, flag +-1 for acquirer's first 3 games after; sign predeclared FADE; 25 resolved events 2014-2025. Registry: deadline_integration_drag_on_production P+ 0.986 (2020-21, 456 games), deadline_drag_promotion_eval_20260905 effect -0.20 pts [-0.87,+0.47] P+ 0.25 (2020-2025 opener, 1503 games), unresolved_below_power.

## Differs (inferred)
Prior: acquirer only, one sign-fixed flag, PFR slug parse with month-end dates, 3 games, snap share from prior team, no seller, no value weight, no ridge term, no SBR 2011-2019 era. Proposed C1-C3: buyer+seller, nflverse trades, 1-4 weeks, prior-season value weights, QB/OL/edge/CB, ridge terms, both eras.

## Next
Orchestrator decides: C2/C3 (value-weighted, seller side, ridge term) could proceed as a new arm, but C1 buyer flag overlaps the graded one. Event count likely ~25-80 total, far below power. Needs nflverse trades file (not on disk: no data/raw trades dir found).

## Open
Is the seller side and value weighting enough novelty to run? Source: data/raw has no nflverse trades; would need download.

## Orchestrator decision 2026-10-01
Dropped. The buyer side is LEAD-23 deadline_integration_drag (2020-2025
opener -0.20 pts, P+ 0.25, unresolved). The seller side and value weighting
change an input with about 25-80 events in 15 seasons (inferred), which can
add little to the model alone. No registry cell: nothing was measured.
