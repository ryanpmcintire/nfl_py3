# conditional-tilt-effects

## Goal

MOD-19. Move the served tilts from rules measured alone to rules decided
with the other relevant information in hand. Worked case: the
pass-protection mismatch tilt, which the owner expects to carry no signal
when the offensive line or pass rush turned over in the offseason. Done when
(1) the protection tilt is measured within conditioning cells (line and
pass-rush continuity, weeks since the window's first game, spread size),
(2) every cell is recorded under one predeclared weak-signals family, and
(3) a gated version of the tilt runs as a paired challenger against the
served unconditional tilt.

## State

- Queued 2026-09-13 with ROADMAP row MOD-19. Stage 1 (LEAD-65,
  `docs/lanes/lead65-protection-window-split.md`) is measured: the
  protection tilt's edge sits in weeks 1-4 (probability_positive 0.977),
  and weeks 5-18 read as a probable drag (0.099, unresolved). The first
  conditioning variable that matters is therefore time since the window's
  first game, not continuity; continuity is stage 2's split of the early cell.

## Tried

- Nothing beyond LEAD-65 yet.

## Next

- Build the conditioning table: per team-game, share of offensive-line and
  pass-rush snaps in the current window taken by players who also took them
  in the window's games (source: the lagged player snaps archived by
  `player-ingest`; check `src/nfl_ats/players.py` for the snap table and
  position groups). Start from `scripts/unserved_tilt_marginals.py`'s
  per-game rows so the marginal is the served card's.
- Predeclare the family `served_tilt_conditional_cells` (units
  accuracy_points) before looking at any cell sign; record every cell.
- Then the gated challenger: fire the tilt only in cells whose
  probability_positive exceeds 0.5, registered as a paired challenger and
  graded at the opener.

## Open

- Which continuity source is cleanest for the offensive line: snap counts
  (available, lagged) or depth-chart identity (archived weekly). Decide when
  the table is built, not before.
