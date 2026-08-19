# Proposal: era-stratified confirmation windows (not yet implemented)

Status: **proposal only**, written 2026-08-19 during the owner's question about
small, back-to-back evaluation windows. No registry code or validator changes
have been made. Owner sign-off is required before implementation because this
changes the shape of the evaluation substrate, and the registry's validators
are owner-mandated to never be weakened.

## The problem it addresses

Rotation-registry confirmation windows are contiguous 2–3-season ranges
(e.g. `[2013, 2015]`, `[2020, 2021]`). Two costs, both measured elsewhere in
this repo:

1. **No valid season-blocked interval.** Coverage of the season-blocked
   bootstrap at k=2 blocks is 0.466 against a nominal 0.95
   (`docs/estimation_variance.md`, the D4 audit; hence
   `MIN_BLOCKS_FOR_INTERVAL = 10`). Every 2-season window therefore leans
   entirely on week blocking.
2. **Regime confounding.** Adjacent seasons share a rules era, carryover
   rosters (`DEFAULT_OFFSEASON_RETENTION = 0.67`), and a scoring environment.
   Era-localized effects are real here: the year-1 head-coach fade is null in
   2009–2017 and lives entirely in 2018–2025 (PER-07). A contiguous window
   cannot distinguish "real edge" from "era quirk."

## The proposal

Allow a family's one confirmation window to be composed of **non-adjacent
single-season legs**, each evaluated walk-forward with training strictly
prior to that leg (train ≤2012 → score 2013; train ≤2021 → score 2022),
pooled under week blocking across both legs.

- Same data budget per look (2 seasons), more regime diversity per look.
- Forward-chaining is preserved per leg; no leg ever trains on data at or
  after its own scoring season, so FND-05 semantics are untouched.
- Declare-before-look, per-family retirement, earliest-eligible assignment,
  and the closing-ground validators all carry over unchanged. The only new
  registry concept is a window whose `seasons` field is a list of legs
  rather than a contiguous range.

## Scope limits

- **Close-graded families only.** The eligible pool there is 2011–2025
  (warm-up floor 2011, rotation rule 9). Opener-graded families draw from
  2020–2025 — six adjacent seasons total (the purchased snapshot archive's
  coverage, MKT-02) — so stratification buys little and is not proposed there.
- Existing spent windows are untouched; this only affects future assignments.
- The earliest-eligible rule needs a deterministic extension for leg pairs
  (e.g. earliest untouched season + the untouched season maximally distant
  from it). That rule must be fixed in code before any family draws under it,
  so no window can be cherry-picked.

## What this does NOT change

The decision frame is unchanged: windows govern claims; plays are decided on
expected value over the full paired evidence. A stratified window still
resolves nothing smaller than its week-blocked power; its benefit is that the
one bit it does buy (direction, `probability_positive`) spans two regimes
instead of one.
