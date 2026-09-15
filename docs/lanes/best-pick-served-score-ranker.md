# best-pick-served-score-ranker

## Goal

The owner is unconvinced by the Best Pick methodology after the Week 1 star
(MIA +3.5) lost. Make the star mean what the board says it means and serve the
ranker with the higher expected accuracy. Done when the star is the card's own
most confident eligible pick, the archive read is recorded in the registry, and
the card, board and assistant all say the same thing.

## State

Shipped, uncommitted. **Owner verdict 2026-09-14: the measurement below is
overfit and is not evidence.** Audit (measured): the +5.94 is an 11-5 record
on the 16 weeks where the two rankers graded differently, best of three arms,
sixth reuse of the population, no held-out season; 2024 and 2025 are exact
ties. The star stays on the card's own most confident eligible pick only
because it has no fitted parameter and the board can state it truthfully;
the alpha=2000 ranker records as the paired challenger. Neither ranker has
out-of-sample support. Doc corrected the same day.

- Predeclaration and results: `docs/best_pick_served_score_ranker.md`.
  Script `scripts/best_pick_served_score_ranker_eval.py`; artifact
  `artifacts/best_pick_served_score_ranker/20260914T163425Z/`.
- Measured, 2020-2025, 101 paired weeks: ranking by the card's own decision
  score scores **62.14% (64/103)** against the incumbent alpha=2000 ranker's
  **55.88% (57/102)**: **+5.94 accuracy points [-1.98, +13.86], P+ 0.934**.
  Dropping the dispersion screen as well is a dead heat with it (+5.88,
  P+ 0.879). Foresight control +41.18 [+31.37, +50.98].
- Coherence defect measured: in 37 of 101 weeks the incumbent starred a game
  that was not the card's most confident eligible pick; on those weeks it hit
  48.65% against the card's own top pick at 64.86%.
- Registry, family `bp_rank_served_score_v1`, all `unresolved_below_power`:
  `bp_rank_served_score_vs_alpha2000`,
  `bp_rank_served_score_no_dispersion_vs_alpha2000`,
  `bp_rank_foresight_control_2020_2025`.
- Served: `best_pick_nomination._nominate` ranks by `ranking_score` (the
  displayed pick probability's distance from 0.5, falling back to
  `candidate_dist`); the nomination moved **after** the production overlays
  and onto the served frame (`card_view.resolve_card_view`,
  `publishing._resolve`, `board_content`); the alpha=2000 rule keeps recording
  through the existing v2/v3 challenger ledgers.
- The star is now pinned once its game is past the pick deadline
  (`published_picks.locked_best_pick`, `card_view.apply_locked_best_pick`), so
  Week 1 still reads MIA +3.5 and says why in plain words.
- Card and board republished; `card-ledger-check` reports no disagreements.
  Gates: ruff format/check clean, mypy clean, `pytest -q` 4529 passed,
  9 skipped.

## Tried

- Re-running the deleted `best_pick_bucket_confidence` / `composed_rule` eval
  scripts: both were removed in the repo cut. The stored artifacts and the
  production `dispersion_pool_from_frame` were enough to rebuild the pool
  exactly, so nothing was re-implemented from scratch.
- Ranking on the raw `home_cover_probability` rather than the displayed score
  was not measured; the displayed score is what the card prints and what the
  archive's `home_cover_probability_at_open` carries, so it is the faithful
  arm.

## Next

- Week 2 is the first live week the new ranker decides; report which game it
  stars and how it differs from the alpha=2000 nominee.
- Consider retiring the dispersion screen: A2 is a dead heat with A1 and one
  fewer moving part. Not done here because a dead heat is not a reason to
  change a served rule.

## Open

- The repo grades the Week 1 played card **10-5** with DEN at KC pending; the
  owner reports 9-6. One game is scored differently somewhere. Needs the owner
  to name the game (see the session report's per-game table).
