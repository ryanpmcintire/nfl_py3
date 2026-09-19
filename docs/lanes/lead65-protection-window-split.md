# lead65-protection-window-split

## Goal

Answer the owner's 2026-09-13 question about the served pass-protection
mismatch tilt: its four-game pressure window uses the previous four
regular-season games (playoffs excluded) and rolls across the offseason, so
weeks 1-4 flag on last season's line and pass rush. Done when the tilt's
marginal on the played card is measured for window-spans-offseason (weeks
1-4) versus later, both cells are recorded with `nfl-ats weak-signals record`
under family `pbp08_protection_mismatch_window_split`, and the ROADMAP row
LEAD-65 states what the read implies for the served rule.

## State

- Measured 2026-09-13 (subagent, then re-run by the coordinator with
  `uv run --no-sync python tests/scratch/lanes/pbp08_early_season_split_20260913.py`;
  report `tests/scratch/lanes/pbp08_early_season_split_20260913.md`).
  Leave-one-out marginal of the protection tilt inside the played nine-member
  card, 2020-2025, paired week-blocked bootstrap, 2000 samples, seed 20260821:
  - window spans the offseason (weeks 1-4): 360 scored games, 21 flips,
    +2.50 accuracy points of the whole card, 90% interval [+0.55, +4.41],
    probability_positive 0.977; every early flag used a prior-season game.
  - weeks 5-18: 1,143 scored games, 58 flips, -0.70 points, 90% interval
    [-1.66, +0.26], probability_positive 0.099 (unresolved_below_power).
  Both cells are in `registry/weak_signals.json`, family
  `pbp08_protection_mismatch_window_split`.
- Measured 2026-09-17 (subagent, parent-verified by rerun with
  `uv run --no-sync python tests/scratch/lanes/pbp08_week_gated_variant_20260917.py`;
  report `tests/scratch/lanes/pbp08_week_gated_variant_20260917.json`).
  Week-gated variant (tilt fires weeks 1-4 only) versus the served
  unconditional tilt, same archive and card, opener grade, paired
  week-blocked bootstrap, 2000 draws, seed 20260821: +0.53 points,
  90% [-0.20, +1.27], probability_positive 0.874, decisive 33-25 on 58
  (exact two-sided p 0.358, consistent with a coin flip), per-season
  +1.82 / +0.85 / +0.40 / -0.75 / -0.75 / +1.87. Fidelity check recomputed
  the lane's own cells exactly (+2.50 P+ 0.977 early, -0.70 P+ 0.099 late).
  Recorded `pbp08_protection_mismatch_early_window_v1_2020_2025`,
  unresolved_below_power (registry now 6,661 signals).
- Reading: the opposite of the hypothesis. The tilt's whole-card edge is
  concentrated where the window reaches into last season; from week 5 on it
  is a probable drag. Candidate mechanism: in September the market has not
  yet priced a line-versus-rush matchup that only last season's tail shows,
  while by mid-season pressure rates are public and priced. The late cell is
  unresolved, so nothing is closed.

## Tried

- The standing `scripts/unserved_tilt_marginals.py --card served` reports a
  zero marginal for this tilt because it is already a served member; the
  leave-one-out convention from the registry's
  `edge_reproducibility_piece_member_pbp08_protection_mismatch_tilt_leave_one_out_2020_2025`
  was used instead.

## Next

- Decision implication first: on the forced-pick card, firing the tilt in
  weeks 5-18 is more likely to cost than to help (probability_positive
  0.099), so a week-gated variant (fire only when the window spans the
  offseason) is the positive-expected play. Register it as a paired
  challenger `pbp08_protection_mismatch_early_window_v1` against the served
  unconditional tilt, graded at the opener, with the mechanism sentence in
  the challenger doc so the gate is not an unexplained threshold.
- DONE 2026-09-17 as a graded measurement (not yet prospective tracking):
  the variant grades +0.53 [-0.20, +1.27], P+ 0.874, decisive 33-25 on 58
  with exact p 0.358, recorded unresolved_below_power. The positive-expected
  direction holds but the decisive split is coin-flip consistent, so this
  supports paired prospective tracking, not a served change.
- DONE 2026-09-18: prospective tracking is live.
  `src/nfl_ats/pbp08_protection_mismatch_early_window_overlay.py` (subagent-built,
  reviewed in-session; imports the parent's flag build and flip logic, gates
  flips to week ≤ 4 via `PREDECLARED_SPLIT_BOUNDARY_WEEK`, records the
  raw-model side in week 5+; no fitted constants) wired into
  `publish-predictions --record-decisions` beside the parent block and
  registered ACTIVE_PROSPECTIVE paired vs the served unconditional tilt.
  Recorded 15 Week 2 Sunday/Monday games (Thursday post-kickoff skipped);
  variant flips LV_LAC and NO_BAL among them. Served picks untouched.
- DONE 2026-09-19: publish surfacing completed.
  `PUBLISH_CHALLENGER_RESULT_KEYS` gains the challenger id to ledger-key
  mapping and `CHALLENGER_DISPLAY_NAMES` gains "Protection mismatch, early
  weeks only"; both are required by contract tests
  (`test_publish_challenger_result_map_covers_live_active_registry`,
  `test_model_ledger_every_live_challenger_arm_has_a_human_display_name`),
  which pass with the wiring in place. Same session: the no-record
  (skipped) payload gains the matching ledger entry, required by
  `test_publish_predictions_does_not_record_by_default`; `test_cli.py`
  fully green after.
- MOD-19 stage 2 then tests the owner's continuity hypothesis directly:
  split the early cell by offensive-line and pass-rush snap continuity.
- DONE 2026-09-17 (subagent, parent-verified by rerun; split predeclared in
  `registry/split_library.json` v1 first): low-continuity early games +2.64
  [+0.58, +4.65], P+ 0.977, decisive 15-6 on 21 (exact p 0.078);
  high-continuity early games degenerate (19 games, 0 fires). The tilt's
  September flags all fall amid unit turnover, which fits the priced-late
  mechanism without identifying a high-continuity marginal. Recorded under
  family `pbp08_protection_mismatch__continuity`, unresolved_below_power.

## Open

- Whether the week-gated variant should replace the served tilt before a
  season of paired tracking (owner; a served-rule change mid-week needs
  `publish-predictions --record-decisions --replace-week`).
