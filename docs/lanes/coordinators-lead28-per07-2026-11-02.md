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

## LEAD-28 unit (2026-11-03)
Row text (ROADMAP LEAD-28) already holds the unconditional close-grade interaction (2009-2025, P+ 0.203), the production back+fade tilt screen, split-half reliability, and says the predeclared interaction stays unscored until forward accrual through Wk18 (Jan 2027). Missing: fitted-term (LOSO, one probability) historical read. Script scripts/lead28_post_bye_fitted_interaction.py -> artifacts/lead28/post_bye_fitted_interaction.json (flag term + interaction with main effects, within-week permutation null, 2 looks). Started 2026-11-03; if json absent, rerun (slow, sklearn warning spam: pass -W ignore).
Forward: Wk8 fires scorable now via POL-10; full accrual 2027-01 (Wk18).
RESULT: json written (4000 draws), 2 cells recorded under family lead28_post_bye_x_new_oc_fitted_interaction (unresolved_below_power). flag term OOS acc -0.13 [-0.73,+0.47] P+ 0.298, 98 flagged, decisive 13 v 15; interaction coef +0.484 [-0.27,+1.31] P+ 0.893, perm p 0.0185, OOS acc -1.66 P+ 0.047 (205 changed picks, intercept noise). Next: Wk8-18 forward scoring via POL-10, complete 2027-01.

## LEAD-29 overlay-input freshness check (2026-11-21T12:33-0500, measured from data/scheduler_log.txt and file mtimes)
- Week 9 (Tue 11-03): scheduled coordinators_tue OK 07:34 (data/raw/coordinators/20261103T123300124659Z); per07_staff_state_tue MISSED (log 5674, build_staff_state_tue prerequisite). Fresh input exists anyway: data/raw/coordinators/20261103T160624919493Z/change_validation.json, mtime 11:06 ET, before noon. OK (manual write).
- Week 10 (11-10): coordinators_tue OK 07:35, per07_staff_state_tue OK 07:55 -> 20261110T125518324908Z/change_validation.json (16 0). OK.
- Week 11 (11-17): GAP. coordinators_tue FAIL(1) 07:33 (log 7386-7389: SourcePolicyError, invalid risk "amber" for source madden_ratings_github_archive in source_policy.py:57); build/per07 BLOCKED then MISSED. Manual rerun 16:36, change_validation.json written 17:56 ET (20261117T225611089672Z), about 6 h after noon lock. The overlay read the 11-10 file (stale by one week) at the Week 11 lock.
- Open: the amber-risk policy entry (cause) needs a fix by the owner of source_policy; check Week 12 Tue 11-24 07:33 coordinators_tue.
ROADMAP LEAD-29 update: 2026-11-21 (measured) overlay input fresh before noon for Week 9 (manual, 11:06 ET) and Week 10 (scheduled); Week 11 gap (coordinators_tue failed on source-policy risk "amber", manual input 17:56 ET after lock, overlay used the 11-10 table). Verify the policy fix before Tue 11-24.

- Root check (2026-11-21 12:33 EST, measured): the source-policy error is already fixed: load_source_policies() loads 37 policies, madden_ratings_github_archive risk is "yellow", and the last SourcePolicyError in data/scheduler_log.txt is line 7932 (11-17, before the fix); coordinators_tue reran OK by hand at 16:39 on 11-17. No fix is owed before Tuesday 11-24. Week 11's overlay used the 11-10 table (input arrived after the lock); that week is lost for LEAD-29.
