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

- 2026-09-23 root: this unit's cells are recorded (registry 7,003); the commands below are history.

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

- Predeclared 2026-09-23 (this unit, before running): two looks at whether the
  served four-term fitted pick probability
  (`src/nfl_ats/pick_probability_fit.py`, FIT_FEATURES = model_logit,
  flag_sum, move_toward_home, move_available) should have its
  `composition_flag_sum` term recomputed so the protection-mismatch member
  only counts when its window spans the offseason (weeks<=4), vs the served
  unmodified flag_sum, both fit LOSO by season 2020-2025 on the played card.
  Mechanism: the card-level marginal (recorded above) shows the tilt is a
  net positive contributor in weeks 1-4 and a probable drag weeks 5-18; the
  hypothesis is that gating or reweighting that one member inside the fitted
  composite would recover some of that split at the probability level, not
  just the card-flip level already tracked by the prospective overlay.
  Look 1: gate the member inside flag_sum (same equal weight as the other
  six counted flags, just zeroed outside weeks<=4). Look 2: pull the member
  out of flag_sum entirely and add it as its own fitted term interacted with
  an early-window indicator (5-term fit).
- DONE 2026-09-23 (this unit): `scripts/lead65_gated_flag_in_fit.py`, run
  once, `artifacts/lead65_gated_flag_in_fit/20260923T202315Z/` (summary.json,
  per_game.parquet). 1,503 graded games, 107 season-week blocks, paired
  week-blocked bootstrap, 2000 draws, seed 20260923, 95% interval, comparing
  each variant's LOSO out-of-season probability against the served LOSO
  out-of-season probability (both from the same four/five-term ridge-logit
  refit machinery as `pick_probability_fit.py`).
  - Look 1 `gated_flag_sum_v1`: -0.86 accuracy points, 95% [-2.19, +0.53],
    P+ 0.111, decisive 34-47 on 81 (exact p 0.182). Brier improvement
    -0.0002 [-0.0017, +0.0012] P+ 0.43; log loss -0.0003 P+ 0.44 (both cross
    zero). IS/OOS accuracy gap tiny (variant -0.13 pts, served +0.20 pts;
    neither shows overfitting). Per-fold `gated_flag_sum` coefficient stable,
    0.25-0.31 across all 6 seasons (served `composition_flag_sum` was
    0.24-0.30). Crosses zero -> unresolved_below_power.
  - Look 2 `protection_early_interaction_v1`: -1.46 accuracy points, 95%
    [-2.87, -0.06], P+ 0.022, decisive 35-57 on 92 (exact p 0.028) -> whole
    interval on the wrong side of zero, a RESOLVED wrong sign for this
    specific fit-level hypothesis. Brier +0.0004 P+ 0.68 and log loss +0.001
    P+ 0.69 both cross zero (accuracy and calibration disagree in direction,
    neither calibration read is significant). IS/OOS gap: variant +0.80 pts
    (mild overfit vs served's +0.20 pts). The `protection_early_interaction`
    coefficient itself is large, stable, and positive every fold (0.76-0.95,
    all 6 seasons), i.e. the fit does want to weight the early-window signal
    heavily on its own -- but the added degree of freedom against thinner
    per-fold early-window coverage still makes the whole composite's LOSO
    calibration worse, not better. -> refuted_mechanism, closing ground
    `wrong_sign_resolved` (closes only this reweighting-in-the-fit variant,
    not the card-level early-window finding recorded earlier in this lane,
    and not the live prospective overlay challenger).
  - Implication for the served flag_sum: do not change it. Both ways of
    folding the offseason-window gate into the fitted composite make the
    LOSO fit worse on the played card, one of them by a resolved margin.
    The place the early-window gate keeps earning its value is the
    card-level pick rule already live as
    `pbp08_protection_mismatch_early_window_overlay.py` (prospective,
    paired, not fitted) -- this unit is evidence to keep that separation,
    not merge the gate into `pick_probability_fit.py`. A src change is the
    root's call, but this measurement argues against one.

## Open

None. Decided 2026-09-23 (root): the served fitted probability keeps the unmodified flag sum; the week-gated variant stays a prospective paired challenger and cannot flip a side on its own under the one-probability rule. Revisit only when a season of paired tracking resolves it.
