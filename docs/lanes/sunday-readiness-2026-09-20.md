# Sunday readiness and Best Pick consistency

## Goal

Serve one fitted probability for each pick, confidence display and Best Pick;
refresh Sunday inputs and publish a card without private quote prices.

## State

- Measured 2026-09-20: `lineups_sun@2026-09-20` completed at 12:18 ET.
  Active model `3412097369e4e4ca` uses Week 2 forecast
  `margin_predictions/2026-week-02-20260920T160851Z`, fitted probability
  `pick_probability/20260920T161202Z`, and waterfall feed
  `waterfall_feed/20260920T161344Z`. The fit uses discrete non-push base
  probability and the Sunday-through-pregame move feature.
- The local card and four board pages were regenerated from those active
  artifacts at 16:35 UTC. One calibrated side drives the card, confidence and
  eligible Best Pick. Against the last committed public card at `f2ebbca`, no
  played side changed; the provisional star moved from LA -7.5 (then 65.7%)
  to NO +8.5 (63.5%). LA is now 63.2% and SF remains the pick at MIA at SF,
  with its displayed chance changing from 62.4% to 59.8%.
- The card and board now say this star is provisional and its estimated lead
  over other picks is uncertain. This is a truthful label, not a validation
  of the ranking rule. The current Best Pick order panel reads the current
  15-game ranking artifact; it does not present old held-out ranking accuracy.
- `card-ledger-check` passed after publication: 16 paper rows, six revision
  rows, zero disagreements. Thursday's completed decision remains frozen.
- Today's Bovada and Odds Gap quote snapshots are private research sources.
  Public Books-now has no eligible current public lines; exact private quote
  prices and move deltas stay off the published pages.
- Release `f2ebbca` is the last verified deployment. This newer local card,
  wording and board await the root agent's authorized commit, push and live
  deployment verification.

## Tried

- `nfl-ats publish-predictions --with-board` exited zero; final
  `nfl-ats card-ledger-check` exited zero. The local rendered visible-text
  comparison to `HEAD` is in ignored
  `data/environment_recovery/sunday_rendered_noon.diff` (27 added, 24 removed
  text segments). No refit or research replay was run for this publication.
- Existing focused refresh/publishing tests: 72 passed. Root reported final
  whole-suite verification: 4,529 passed, nine skipped, mypy 235 source files
  passed. No new test files/functions or code comments were added.
- Post-review residue proposal, retained for owner review: the publishing
  contract has a redundant generic phrase assertion beside an exact wording
  assertion and an obsolete `v2` test name. No test was removed.

## Next

- Root commits and pushes this local release, verifies the live Pages build,
  and refreshes `HANDOFF.md`. Continue separate confidence-ranking research
  in `docs/lanes/confidence-best-pick-unification.md` without treating the
  provisional Sunday star as validated.

## Open

- Public Books-now remains unavailable today because the current quote
  snapshots cannot be redistributed.
