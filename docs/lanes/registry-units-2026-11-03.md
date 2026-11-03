# Registry units 2026-11-03

## Goal
Honest effect units for weak-signals cells; no quantity stored under a wrong unit.

## State
- Added logit_coefficient, slope_difference, epa_per_play to EFFECT_UNITS (src/nfl_ats/weak_signals.py), labels in signal_ledger._UNIT_META and board_site_content._RECENT_ACTIVITY_EFFECT_UNIT_WORDS. Validators, classification and closing-ground rules untouched. Pool groups by exact unit, so new units never pool with accuracy_points. None of the three files is pinned or in the independent_validation source list.
- Retagged via `weak-signals retag-units` (audit note appended, all else unchanged): sim07_personnel_interaction_omnibus_fulldata mae -> epa_per_play; sky03_def_cov_coef_sign_season_block correlation -> logit_coefficient; sim08_ma_t_total_slope correlation -> slope_difference.
- Pool accuracy_points eligible_count 4210 before and after.
- Only recorded 2026-11-03 cell with "nearest"/"forced" in notes: sim08_ma_t_total_slope (fixed).
- mod10 reliabilities left as correlation.

## Tried
Re-record with --replace considered; retag-units is the audited path that changes only the unit.

## Next
None.

## Open
None.
