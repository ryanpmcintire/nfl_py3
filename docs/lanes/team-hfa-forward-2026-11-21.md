# Team home edge forward log, 2026-11-21

## Goal
Freeze the team-and-stadium home edge (done/team-hfa-2026-11-16.md: +1.32 accuracy pts/SD vs the close 2011-25 [+0.08,+2.82], null 93.0th pct, gate spent) before the Week 12 2026 openers, and log it each week of Weeks 12-18 beside the served card. The log is post-hoc and serves nothing. Owned: scripts/team_hfa_forward.py (freeze | log | batch) and artifacts/forward_terms/team_hfa/.

## State (2026-11-21T15:07Z, measured, freeze exit 0)
- frozen.json sha256 f64a0df4845dc3f27efa88a7a06651b19dfb14692ba1dfe0a352fd39e8160cba. Inputs: schedule snapshot 20261116T204645Z (the screen's), 2026 outcomes blanked; the file sha and canonical-input sha are in `inputs`.
- Identity vs the screen: 576 team-seasons and 3,818 games, max abs diff 0.0. 2026 shrinkage: k 0.8425 (lambda 174.3 games), chosen over home games 2011-25 (3,788).
- 2026 values (pts): NYJ +0.77, PIT +0.73, GB +0.68 at the top; TB -0.81, WAS -0.68, KC -0.56 at the bottom.
- Coefficient, frozen: +0.0563 logit/SD (term mean 0.0165, SD 0.3826 pts). Fit: QW offset at the opener 2020-25 with nested LOSO, 1,537 games (34 pushes at 0.5). Folds 6/6 positive (0.034-0.078); halves +0.089 (2020-22) and +0.036 (2023-25).
- Held-out (post-hoc, not gating): 841-662 v QW 836-667; changed picks 76-71 v 71-76.
  - Accuracy +0.33 [-1.33,+1.95], P+ 0.67.
  - Log loss +0.00003 [-0.00138,+0.00148], P+ 0.52.
  - In-sample minus held-out log loss +0.00036; intercept-only -0.0000003.
- Recorded `unresolved_below_power` as family team_hfa_served_2026_11_21, 2 cells (registry/weak_signals/team_hfa_served_2026_11_21.json); record exit 0. `pool --league nfl --effect-units accuracy_points` exit 0 (pool.log).
- No declaration entry was recorded: the 2026-11-16 forward lanes (nfelo, lead108, lead112) record only at scoring. The rules for the procedure are in procedure.md.
- Week 11 smoke (`log --week 11 --smoke`), 2026_week11_20261121T150628Z.json: 13 games, 13 on the served card, MIN_SF neutral (x 0). The informational with-term side differs on ARI_KC, JAX_NYG and CIN_WAS; served sides are untouched.
- Week 12 (`log --week 12`), 2026_week12_20261121T150632Z.json: 16 games, all at primary venues. No Week 12 card yet, so served is null.

- 2026-11-21 ~17:40Z (dry-run stamp 20261121T174025Z): `score` added to scripts/team_hfa_forward.py (insertions only; ruff clean). It follows procedure.md. Population: card rows Weeks 12-18. Base: the played card (last publication at or before the deadline). x comes from the latest log with `smoke_test` false, frozen sha matching, and logged_at and mtime before min(kickoff, deadline); the logged shift must equal the frozen recompute. y: home cover at the card line, push 0.5. Report order: changed-pick record, log-loss gain (week-block bootstrap, P+), Brier, reliability, then the intercept-only control (one QW-offset intercept fitted on history_games.parquet 2020-25, logit +0.00335). Accuracy is reported, not recorded. Batch: 1 cell, `team_hfa_served_forward_wk12_18_log_loss` → artifacts/forward_terms/team_hfa/weak_signals_batch_forward.json (not recorded). The script refuses while any week 12-18 lacks a result in game_features (measured: real `score` exit 1 "weeks [12, ...] are not final").
- Dry-run (measured, exit 0). Input: a relabelled copy of the Week 11 smoke in tests/scratch/score_modes/team_hfa_input/. The untouched smoke copy was excluded as `smoke`. 12 graded; IND_HOU skipped (`no_log_before_cutoff`). Outcomes were synthetic Bernoulli(card); no real outcome was graded. Outputs: tests/scratch/score_modes/team_hfa_out/.

## Tried
- Mirrored expert_rankings_forward.py (fit rule, card rule, batch) instead of new helpers.

## Next
- Root adds the jobs below to scripts/capture_scheduler.py (diff, not applied). Insert them after `lead108_forward_score_tue`:
```
+    *(
+        Job(
+            f"team_hfa_forward_log_{day}_{at.replace(':', '')}",
+            day, at, grace,
+            [str(UV), "run", "--no-sync", "python",
+             str(REPO / "scripts" / "team_hfa_forward.py"),
+             "log", "--season", "2026", "--week", "auto"],
+            True,
+            "Team home edge forward log (post-hoc): frozen term beside the served card, "
+            "written before kickoff; never changes a side.",
+            requires=("weekly_lock",) if day == "tue" else (),
+            added_on="2026-11-21",
+        )
+        for day, at, grace in (("tue", "14:00", 360), ("sun", "12:15", 30))
+    ),
```
- Next subtask: once Week 18 is final, run `.\.tools\uv.exe run --no-sync python scripts\team_hfa_forward.py score` (no flags). Record weak_signals_batch_forward.json in family team_hfa_served_2026_11_21, run `pool` once, then move this lane to done/.
- ROADMAP MOD-57 update: the team home edge forward scorer exists (`team_hfa_forward.py score`, 1 verdict look, log loss). It scores once after Week 18 2026 and serves nothing.
- ROADMAP update: no row names this term (commit e93fc8f1 touched no ROADMAP row). Its history sits in goal-65-2026-11-16.md, which supersedes MOD-57's lane, so note it on MOD-57 (ROADMAP.md:940).

## Open
- Cards were published at Tue 12:36 ET (Wk9) and Tue 20:21 ET (Wk10). A Tue 14:00 log can miss a late card, and a Thursday game can then lack a carded log. If so, add a Thu 18:00 slot with the same argv.
- After Week 18, `--week auto` exits non-zero. Remove the jobs then.
