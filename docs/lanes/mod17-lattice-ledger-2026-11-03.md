# MOD-17 lattice-centre ledger triage (2026-11-03)

## Goal
Decide retire or repoint the lattice_centre_decisions ledger, then close.

## State
Decision: RETIRE enrollment (already implemented); ledger final at 3 settled rows (Weeks 1-3: tie, tie, challenger).
**Read:** `SERVED_CENTRE_ADOPTION_AT = 2026-09-24` in `src/nfl_ats/lattice_centre_challenger.py:24`; recorder skips enrollment after it by design, still settles old rows. `tiebreaker.py:~406` serves `challenger_centre`, so the challenger arm equals the served arm and a new row would compare a policy with itself.
**Measured:** `independent-validation status --study independent_validation_v7` -> source_unchanged true, 0 of 151 captured.
v7 fence (`independent_validation._sources`): any src .py named in PINNED_NAMES (card_view, clv, constants, four_overlay_composition, independent_validation, mass_preserving_lattice, pick_probability, pick_refresh, pool_hook, prediction_safety, snapshots, source_policy), or ending features.py / _overlay.py, or containing "margin" or "model", plus cli_commands/prediction.py. The pinned-source guard uses the same rule.
`lattice_centre_challenger.py` matches none of these; it is NOT fenced. No fenced edit is needed, so the 2026-11-02 deferral premise was wrong, but no edit is wanted.

## Tried
No recorder run: missing Weeks 4+ cannot be backfilled; the old-policy arm was never published and post-lock data is forbidden.

## Next
Close MOD-17 ledger sub-item (root edits ROADMAP). Joint-residual research half is separate.

## Open
Optional: a truthful "enrollment ended" status (needs no fenced file).
