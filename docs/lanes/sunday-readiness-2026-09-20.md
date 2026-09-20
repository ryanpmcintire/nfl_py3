# Sunday readiness and Best Pick consistency

## Goal

Serve one fitted probability for each pick, confidence display and Best Pick;
refresh Sunday inputs and publish a card without private quote prices.

## State

- Measured 2026-09-20: active model `3412097369e4e4ca` names the discrete
  Week 2 forecast `margin_predictions/2026-week-02-20260920T150905Z`.
  The active fitted probability is `pick_probability/20260920T152908Z`, with
  the Sunday-through-pregame leader-move feature and the discrete non-push base.
- `scripts/waterfall_feed.py` now loads that fit and distinguishes the smooth
  margin attribution, discrete base probability and fitted served adjustment.
  Its direct regeneration passed for 16 games at `waterfall_feed/20260920T153034Z`.
- One calibrated side drives the card, dashboard, confidence and eligible Best
  Pick. The final local card and four dashboard pages were regenerated. The
  guarded paper recorder replaced 15 pre-kickoff Week 2 decisions and preserved
  the completed Thursday decision. `card-ledger-check` passed: 16 paper rows,
  six revision rows, zero disagreements.
- Against `origin/master`'s last publicly played decisions, three picks change:
  GB to NYJ +3.5 (51.5%), KC to IND +6.5 (55.0%), CHI to MIN +5.5 (54.5%).
  The Best Pick moves from DEN to LA -7.5 (65.7%). The other 13 played sides,
  including Thursday BUF, remain. The final main card has no conflicting
  refresh appendix.
- Today's Bovada and Odds Gap quote snapshots are private research sources.
  Public Books-now uses `current_spread_quotes(public_only=True)` and measured
  zero eligible current public lines. Exact private move deltas were removed
  from the public card appendix, board reason and explanations. The public
  model-page paragraph now describes fitted movement rather than a hard flip.

## Tried

- The Sunday lineup recovery built the forecast, then its waterfall step
  failed because the resolver omitted the active fit. Direct waterfall repair
  and rerun passed; the weekly model was not rebuilt a second time.
- `.\.tools\uv.exe run --no-sync nfl-ats publish-predictions --with-board`,
  `nfl-ats refresh-picks --record-decisions --publish-card --note
  lineups_refresh`, `nfl-ats publish-predictions --record-decisions
  --replace-week`, `nfl-ats card-ledger-check`, and `nfl-ats publish-board`
  completed locally. The supported replace-week guard left one post-kickoff
  row untouched. Rendered visible-text diffs are in ignored
  `data/environment_recovery/sunday_rendered_*.diff`.
- Existing attribution/board contracts: 204 passed. Root's final full
  verification: Ruff format/check, mypy, comment guard and 4,529 tests passed;
  nine tests skipped. No new test files/functions or code comments.

## Next

- Root agent refreshes `HANDOFF.md`, reviews remaining Git changes, then makes
  the user-authorized commit/push. Confirm the public site rebuild after push.

## Open

- Public Books-now remains unavailable today because the current quote
  snapshots cannot be redistributed. The local release is not yet pushed.