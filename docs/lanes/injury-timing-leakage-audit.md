# Injury timing leakage audit (release block lifted; Tuesday-card skew open)

## Goal
Confirm or rule out look-ahead in the served model's historical injury inputs.

## State
- **Measured (docs/injury_timing_audit.md):** 1,534 of 1,537 opener games use
  injury observations after Tuesday noon ET (median lag 75.8 h). 11 served
  columns: 9 `diff_injury_*` plus `diff_qb_expected_epa_per_dropback`,
  `diff_qb_start_probability`.
- **Read:** the builder clock is kickoff minus 24 h (`players.py:1824`); every
  row passed it. The pick deadline is min(kickoff, Sunday 4 PM ET), so
  kickoff-24 h is before the deadline: **inferred** not a leak against the
  pick deadline, only against a Tuesday-timestamped card. Release block lifted.
- **Measured:** repair replay (11 columns zeroed, nested LOSO 2020-2025):
  862-641 vs original 865-638; repaired minus original -0.20 accuracy pts
  [-1.27, +0.81], probability_positive 0.341; log loss unresolved. Recorded as
  `injury_timing_availability_repair` (unresolved_below_power).
- **Measured:** live Week 4 Tuesday lock used the 2026-09-27 snapshot; its
  injury block is all zero while 95-100% of training rows are nonzero:
  train/serve skew on the Tuesday card. Refresh paths (`pick_refresh.py`,
  `inactives_refresh_overlay.py`, `refresh_triggers.py`) feed later news.

## Tried
- Proposed guard option `decision_clock="tuesday_noon"` for
  `enrich_with_player_features`: tests/scratch/codex/injury_timing_audit/injury_clock_guard.diff
  (compiles, not applied; default behaviour unchanged).

## Next
- Verify the Saturday/Sunday refresh rebuilds the 11 injury columns for the
  served card (not only overlays); if it does, the Tuesday card is a
  preliminary and the skew only affects early picks.
- LEAD-89's larger question: the market-move term is a Sunday quantity and
  carries most of the 57.6% vs model-only 52.2% gap; same deadline logic.

## Open
- Whether to add a runtime guard comparing each feature clock to the card's
  own timestamp rather than kickoff minus 24 h.
