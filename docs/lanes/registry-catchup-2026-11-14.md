# Registry catch-up 2026-11-14

## Goal
Put measured results that never reached the weak-signal registry into it, and replace one stale entry. Transcription only: no new verdict, no new number.

## State
Done 2026-11-14. Batch files: tests/scratch/registry_catchup_2026_11_14/ (ignored). Every cell is unresolved_below_power with no closing ground. Shards are new untracked files under registry/weak_signals/ (xlg13 is modified).

| Row | Family | Cells | Unit |
|---|---|---|---|
| MOD-52 | mod52_mod41_stability_u1 | 17 | log_loss_improvement |
| LEAD-100 | lead100_repeat_read_ensemble | 1 | correlation |
| LEAD-102 | lead102_name_blinded_control_read | 1 | correlation |
| LEAD-103 | none | 0 | no admissible unit |
| PER-14 unit 1 | per14_oc_mover_portability | 20 | correlation |
| POOL-06 | none | 0 | counts only |
| POOL-07 | pool07_standings_policy | 20 | payout_first_pp |
| XLG-13 | xlg13_screen_transfer | 2 replaced | correlation |

59 new cells, 2 replaced. Final pool (measured, `weak-signals pool --league nfl --effect-units accuracy_points`): exit 0 after each of the six batches; 4543 signals, 4080 informative, sign-test p 0.1002. Before the first batch: 4541 signals, p 0.1001. No catch-up cell is in accuracy_points, so the two extra signals came from another recorder (inferred).

## Tried
- First run stopped at "batch cell 0 is missing required fields: effect_units" with nothing written. Batch shared defaults do not carry effect_units (read: src/nfl_ats/cli_commands/registry.py:319-328). Fixed by writing effect_units, notes, plain_summary and sample_blocks into every cell.
- One dated line appended to each of the eight source lanes.
- No process left running (measured: process list empty for this run).

## Next
- Root: commit the six shards and the lane edits, then move this lane to done/.
- Root: confirm the PER-14 unit 1 season span (see Open) and correct it with a batch re-record if wrong.

## Open
Left unrecorded, with the reason:
- MOD-52: 5 homogeneity statistics and 2 slopes; no admissible effect unit.
- LEAD-100: between-read SD 0.052, between-game SD 0.093, across-view SD 0.078 logit; dispersion statistics. Spearman-Brown 0.38 is in the cell description only.
- LEAD-102: named-minus-blinded mean gap -0.009 logit [-0.034, +0.015], P+ 0.224; a mean logit gap has no admissible unit. The recorded correlation has no probability_positive because the analysis stored none.
- LEAD-103: all six per-block mean absolute logit shifts and noise ratios; non-negative magnitudes, no admissible unit.
- PER-14: 5 OC-to-HC correlations; permutation p only, no interval or probability_positive on disk.
- POOL-06: tie counts only; no effect or interval.
- POOL-07: top-10 chance, expected wins and expected rank changes; no admissible unit. Recorded cells carry the effect and the simulation standard error only (no interval or probability_positive in the artifact); field-model uncertainty is larger than that error.
Flags for root:
- PER-14 cells carry seasons 2009-2025, read from the lane's staff-history note; the artifact stores no span.
- XLG-13 served gains were not re-verified after the harness fix; the cell notes say so.
