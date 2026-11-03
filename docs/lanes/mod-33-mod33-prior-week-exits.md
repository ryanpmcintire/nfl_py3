
## Unit 1 (measured 2026-11-03)
Script scripts/mod33_prior_week_exits.py (out tests/scratch/mod33), ruff clean. Family mod33_prior_week_exits, 5 term looks (OL, QB, WR1, DB1, all4) + control; 1503 non-push games 2020-25, opener-based served probability as input, intercept-only control. Recorded 5 cells (log_loss_improvement vs control), unresolved.
Result: no term beats control OOS (control LL 0.6922; terms 0.6928-0.6967; all4 acc 50.2% vs 52.9%); decisive record term loses each (OL 24-46, QB 38-47, WR1 13-25, DB1 10-19, all4 72-113). QB coef -0.27 (se 0.17, P>0 0.05), stable negative every fold.
Timing: exited starters next-week Out/Doubtful 28-38% vs 1.5-2.4% non-exit; absent-and-undesignated 28% QB, 15% DB1, 9% WR1 vs 4%/7%/4%. OL absence join broken (pfr id), unmeasured.
Availability: OL term corr with served OL unavailability 0.21, offense continuity -0.26; residual corr with base+injury logit ~0.
Next: fix OL absence join; QB-only fitted term with injury-feature interaction.

## Unit 2 (measured 2026-11-03)
OL join fixed (scripts/mod33_prior_week_exits.py): weekly_rosters pfr_id is 0.4% populated for OL (gsis_id v snap_counts pfr_player_id cannot join); now gsis_id -> roster full_name -> snap_counts.player on season/team/week. OL next-week absent: non-exit 9.7%, exit 65%; absent-undesignated 6.6% v 19.9%.
QB x designation (3 looks, same family, 1503 games, LOSO, v intercept-only; exits designated next week 62 games, undesignated 89): decisive (n, term wins, control wins): split 88, 41-47; undes-only 49, 23-26; des-only 38, 17-21. LL gain v control: split -0.0014 (P>0 0.10), undes -0.0017 (0.03), des +0.0003 (0.62). Coefs -0.26 to -0.28 negative all 6 folds in both. Acc split 52.5% v 52.9% control. Hypothesis (only undesignated carry info) not supported; both negative equally. Caveat: designation is next-week report, may postdate the opener pick. Recorded 3 cells unresolved_below_power. Next: none unless owner wants QB term served-time check.
