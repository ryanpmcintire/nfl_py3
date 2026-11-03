# LEAD-106

## Goal
Injury-report language read: an LLM maps designation text and practice notes to expected snap fraction per starter (ROADMAP LEAD-106).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Designations like questionable carry different meanings by player and note; free text may encode more than the ordinal ladder. Fitted term: expected-availability difference replacing the ordinal injury input, frozen prompt and forward capture.

## Tried
Novelty check only. Nearest: PER-03, PER-10, LEAD-99 (verbal read), LEAD-64 (capture integrity); grep 'injury text' found none. Differs: Language to availability, not language to cover.

## Next
First unit: Run a frozen prompt over the stored captures, compare Brier on realised snaps with the ordinal ladder, then pass to the four-term probability only if it adds out of week.

## Open
Data: Official injury report text captured (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 progress (2026-11-03 ~16:50 local)
Built scripts/lead106_injury_language.py (freeze/run/score) and scripts/lead106_prompt.md (sha256 08170f6d8428c4b31db2b67062172140f3b45a999d05bafa70eb6008061b3d2c). Population 2459 player-weeks (nflverse injuries 2026 wk1-8, outcome from forward_snaps; unmatched names count as not played, matched rate .66; Out/Doubtful never appear in snaps). Sample 331 rows weeks 2-8: all 228 headline-bearing rows plus up to 100 per designation (budget declared, seed from prompt hash). Read model claude-opus-5-5, no tools, 40 rows per batch, outputs in artifacts/lead106_injury_language/raw. Ladder baselines: designation, designation x practice, expanding prior weeks, add-one smoothing; bootstrap over team-week clusters, 2000 reps, stratum-weighted.
State: run launched in background (task bd4w5zxgi, log artifacts/lead106_injury_language/run.log), not yet confirmed finished. Do not rerun finished batches (script skips them).
Next: when raw/ has 9 batch files, run `scripts/lead106_injury_language.py score`, record log_loss_improvement via nfl-ats weak-signals record (unresolved_below_power unless refuted), fill results here.
Mechanism if positive: expected-availability term in player availability features only, never a side rule.
Open: wording signal limited (only practice phrase, body part, headlines since late Sept).
- 2026-11-03 16:43 ET (root review): 33.5% of player-weeks did not match the snap counts and were labelled "not played"; a name-match miss is not a missed game. The scoring unit must score matched rows only (report the unmatched rate by designation) or fix the join before reading Brier or log loss.

## Unit 1 result (2026-11-03)
Read vs designation ladder, log_loss_improvement on P(plays) 0.075 [0.019, 0.137] P+ .9995; Brier +0.026 [0.009, 0.047]; snap-share MAE +0.061 [0.039, 0.082]. Headline rows (228) 0.110 [0.032, 0.206]. Vs designation x practice ladder: 0.048 [0.003, 0.093] P+ .98. Halves even/odd weeks both positive (.027/.134). Caveats: Out/Doubtful trivial (nobody played); gain comes from Questionable/none; unmatched names counted as not played. Recorded as lead106_injury_language_p_plays. Next: forward capture of the same read for Week 9+, then test as availability input.
- 2026-11-03 16:44 ET (root): the 0.075 gain is NOT trusted until rescored: outcomes label unmatched names as not played (33.5% of rows). Next unit: join injuries to snap counts by player id (nflverse players crosswalk gsis_id <-> pfr_id), rescore matched rows, report by designation.

## Unit 2 result (2026-11-03 16:49 ET)
Outcome join rebuilt by id (script cmd `ids`: injuries gsis_id -> weekly_rosters pfr_id -> snap counts pfr_player_id; stored reads reused, no LLM rerun). Verified rate 87.4% (was 66.5% name-matched). By designation: Out 100%, Doubtful 86.5%, Questionable 85.5%, none 83.5%. Unverified: no pfr id (Q 75, none 229, Doubtful 5; mostly practice-squad/new signings absent from the rosters crosswalk). Absent-and-rostered counted not played (INA/RES/ACT/CUT). Sample 294 of 331 verified.
Vs designation ladder (log loss, stratum weighted): all 0.065 [0.014, 0.130] P+ .9985; Brier 0.020 [0.005, 0.041]; MAE snap share 0.035 [0.014, 0.055]. By designation: Out 0.000 (P+ .43), Doubtful -0.026 [-0.051, -0.002] (27 rows), Questionable 0.068 [0.002, 0.135] P+ .979, none 0.095 [0.004, 0.219] P+ .981. Halves even/odd .030/.108.
Vs designation x practice ladder: all 0.036 [-0.002, 0.083] P+ .959; Brier 0.012 [-0.001, 0.027]; MAE 0.036 [0.017, 0.056]; Questionable 0.009 [-0.053, 0.074] P+ .62; none 0.066 [-0.003, 0.150] P+ .967; halves .032/.041.
Verdict: gain survives vs the designation ladder (slightly smaller); vs the practice ladder it is weaker and the Questionable gain is mostly practice information. Registry cells replaced/recorded (unresolved_below_power).
Next: forward capture Week 9+ of the same read, then test as player-availability input; consider the bigger sample for Questionable.
