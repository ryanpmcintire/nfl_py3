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
- MOD-19 stage 2 then tests the owner's continuity hypothesis directly:
  split the early cell by offensive-line and pass-rush snap continuity.

## Open

- Whether the week-gated variant should replace the served tilt before a
  season of paired tracking (owner; a served-rule change mid-week needs
  `publish-predictions --record-decisions --replace-week`).
