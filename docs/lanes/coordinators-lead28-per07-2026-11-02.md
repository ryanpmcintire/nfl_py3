# coordinators-lead28-per07-2026-11-02

## Goal
Unit A: confirm the LEAD-28 post-bye new-playcaller overlay reads current 2026 OC rows and list the games it flags. Unit B: PER-07 first-game-under-new-playcaller as a fitted term, leave-one-season-out, opener grade.

## State
Unit A done (read-only, nothing edited). Overlay `_latest_under(.../coordinators, "*/coordinator_history.parquet")` (post_bye_new_playcaller_back_overlay.py:44-53) resolves to 20261031T112746206891Z, which holds 32 preseason + 32 inseason 2026 OC rows. 2026 OC tenure: 21 year-1, 5 year-2, 5 year-3+, 1 unknown. Flagged post-bye new-OC games through Wk9: Wk6 LAC@KC (home), Wk7 MIA@NYJ (away), GB@DET (home), Wk8 BAL@BUF (home), IND@JAX (home), LAC@LA (away), PHI@WAS (home), Wk9 CLE@NO (home), NYG@PHI (away), HOU@LAC (away). On the Wk8 card the overlay returns 4 flips (BUF, JAX, LAC, WAS); no both-flagged games. The Wk1 zero was no byes in Wk1 (inferred), not a stale path.
No 2026 in-season OC/DC changes exist in the snapshot (no team/role with 2 persons), so PER-07/LEAD-29 event set stays 12 (2022-2025).
Unit B: script scripts/per07_first_game_fitted_term.py, output artifacts/per07/first_game_fitted_term.json (results in final report / json).

## Tried
Fitted logit y~logit(p_open)+s, s = home_signal - away_signal, on production_paired_lead29_full.parquet (2020-2025), LOSO by season.

## Next
Record cells with weak-signals record (family per07_first_game_new_playcaller_fitted_term); orchestrator verifies.

## Open
The overlay is a pick flip (inadmissible under AGENTS.md one-probability rule); pinned, not edited. Only 12 events, power very low.
