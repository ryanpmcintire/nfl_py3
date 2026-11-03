# LEAD-111

## Goal
Phrase ablation of the injury read: which part of the wording carries the gain (practice phrase, body part, headline text) (ROADMAP LEAD-111).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: The Questionable gain over the practice ladder was weak (P+ .62) while none rows gained (P+ .97); naming the carrying block tells which text to capture forward. Fitted term: none new; picks the inputs for LEAD-108. Data: stored LEAD-106 packets (local), frozen prompt reruns on 294 verified rows.

## Tried
Novelty check only. Nearest: LEAD-103 (LEAD-99 ablation, different packet), LEAD-106. Differs: ablates the injury packet.

## Next
First unit for one agent: Rerun the frozen prompt on the verified rows with each block removed, score matched rows only against the by-id join, report each block's loss in log-loss gain with intervals.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 declaration (2026-11-03 17:49 local, before any ablated read seen)
Script scripts/lead111_ablation.py (freeze/run/score); artifacts/lead106_injury_language/ablation/. Prompt unchanged, sha256 08170f6d... verified at freeze and every run. Ablations of per-player text: practice (drop "practice:" part), bodypart (drop "injury:" and "also:" parts), headlines (drop "headlines:" part). Plus repeat = identical text, second read (noise).
Subsample 207 of the 294 verified rows: all Questionable (87) and none (80), which carry the gain, plus 40 seeded (prompt-hash seed) of the 127 Out/Doubtful (31 Out, 9 Doubtful). Weights = population stratum rows / subsample rows by designation. Headlines ablation runs only the 147 rows that have headlines (others have identical text and reuse the original read).
Scoring: log loss on P(plays) (verified id outcomes), paired bootstrap over team-week clusters, 2000 reps. Looks: 4 vs-full cells + 3 vs-repeat + 10 gain-vs-ladder cells = 17 (family lead111_phrase_ablation); plus logit shift RMS vs LEAD-100 .052 noise.
Runs launched in background (4 python processes, one per condition; logs ablation/run_<cond>.log).
In flight 17:52: 4 runs (PIDs 14332,32964 practice; 19900,9104 bodypart; 23484,31236 headlines; 20688,32896 repeat) writing ablation/raw/<cond>/batch_*.json. When all exist (practice/bodypart/repeat 6 batches, headlines 4), run `scripts/lead111_ablation.py score`, then record cells (family lead111_phrase_ablation) with weak-signals record, fill results here. Rerun of `run <cond>` skips finished batches.

## Unit 1 result (2026-11-03 ~18:05)
Runs done, scored, 15 cells recorded (family lead111_phrase_ablation, unresolved_below_power; artifacts/.../ablation/score.json, ws_batch.json). Log loss, removal vs full read: practice -.016 [-.043,.010] P+ .13; bodypart -.008 [-.025,.006] P+ .12; headlines -.030 [-.071,.001] P+ .04; repeat (noise) -.004 [-.016,.010]. Read-to-read logit SD here .284 (not .052; headline rms shift .50, practice .49, bodypart .38). Gain vs designation x practice ladder: full .040, repeat .036, practice .024, bodypart .032, headlines .010 [-.014,.043]. Headlines carry the most; practice second; bodypart ~noise.
