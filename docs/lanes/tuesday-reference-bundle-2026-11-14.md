# Tuesday reference bundle (2026-11-14)

## Goal
Both served terms that read a Tuesday line run on the quote age they were fitted on: the pool-line offset anchor at the pool's noon freeze, the move baseline late on Tuesday. This unit prepared the anchor patch and its evidence; the orchestrator applies the bundle before the Week 11 lock, Tue 2026-11-17 noon ET. No src file was edited here.

## State (times are the artifacts' own generated_at_utc stamps)
- Derivation (read pool_hook.py:78, :91; artifacts/eng62/baseline_age.json fit_offset_term): the fitted pool line is the anchor plus or minus a half at one moment, and that anchor is the last snapshot the history held before the noon freeze (Tue 08:55 ET, 3.07 h before). The reference time is therefore the freeze itself, `opened`, which the function already computes; no new constant. Rule: median across books of each book's Tuesday (ET day) quote nearest the freeze, a tie going to the earlier quote; first-snapshot fallback unchanged; still never one snapshot.
- Patch: tests/scratch/tuesday_reference_bundle/pool_hook_anchor.patch (`git apply --check` exit 0 against the working tree), scratch copy pool_hook_patched.py beside it; ruff check and format clean on the patched file and both scripts.
- 2023-25, out of season (measured; scripts/eng62_tuesday_reference_oos.py -> artifacts/eng62/tuesday_reference_oos.json, .out). MOD-57 harness reused: pushes kept, DraftKings-built pool line, 806 games with 5+ books in every arm, stage 1 equals the artifact (gap 0.0), week-block bootstrap. Arms declared before looking, median hours from noon: fit -3.07, before -0.07, after +0.93, nearest (the patch) -0.07, live +1.93, late +11.93. Coefficient held at the served .805 (fitted on all seasons), stage 1 and push table out of season. Levels: stage 1 55.40 pct; fit 57.45, nearest 57.52, live 57.20, late 57.02. First figure uses the pool line from the first quote after noon, second from the last quote before:

| contrast | decisive | accuracy pts [interval] P+ | log loss gain, P+ | by season 23/24/25 |
|---|---|---|---|---|
| nearest - live | 8-5.5, 8-4.5 | +0.31 [-0.32,+0.93] .84; +0.43 [-0.25,+1.13] .89 | -.00008 .45; -.00017 .37 | +0.37 +0.75 -0.18 |
| nearest - late | 14-10, 14.5-9 | +0.50 [-0.69,+1.74] .79; +0.68 [-0.56,+1.86] .86 | -.00039 .33; -.00045 .31 | +1.31 +0.38 -0.18 |
| nearest - fit | 7.5-7, 7.5-8 | +0.06 [-0.68,+0.81] .56; -0.06 [-0.75,+0.68] .43 | +.00005 .52; -.00004 .47 | +0.19 -0.57 +0.55 |
| before - after | 5-4, 5-3 | +0.12 [-0.38,+0.68] .68; +0.25 [-0.25,+0.87] .81 | -.00019 .32; -.00030 .24 | +0.19 +0.57 -0.37 |
| late - live | 7-8.5, 7-9 | -0.19 [-1.17,+0.75] .36; -0.25 [-1.28,+0.69] .31 | +.00031 .69; +.00027 .67 | -0.93 +0.38 0.00 |

  Coefficient fitted leave-one-season-out on the fit arm (folds .70 to 1.61, unstable on two training seasons): nearest - live +0.50 P+ .93 and +0.74 P+ .96; nearest - late +0.56 .80 and +0.99 .90; nearest - fit +0.31 .79 and +0.12 .62. In-sample minus out-of-season accuracy gap: 0.00 to +0.37 pts. 40 looks (5 contrasts x 2 pool builds x 2 coefficient modes x 2 units), family eng62_tuesday_reference_anchor.
- 2026 replay, information only (measured; tests/scratch/tuesday_reference_bundle/replay_2026.py -> artifacts/eng62/tuesday_reference_replay_2026.json). Current structure and coefficients, flags not counted; Weeks 1-9 reproduce the fix-picks replay, 85-51.

| week | anchors changed | sides changed | record current, patched | anchor hours after noon |
|---|---|---|---|---|
| 3, 4, 5, 7 | 0 | 0 | 9-7, 10-6, 11-4, 10-4 (same) | 5.04, 4.62, fallback, 2.44 |
| 6 | 1 (BUF at LV) | 0 | 6-8, 6-8 | 6.59 |
| 8 | 1 (PHI at WAS) | 0 | 7-7, 7-7 | 2.43 |
| 9 | 3 (CLE at NO, GB at NE, JAX at BAL) | 0 | 6-9, 6-9 | 2.46 -> 0.18 |
| 3-9 | 5 | 0 | 59-45, 59-45 | |
| 10 | 3 | 2 | none settled in the file | 2.43 -> 0.19 |

  Book counts per anchor are unchanged every week. No Tuesday quote before noon exists in Weeks 2-10, so live the rule reads the 12:10 capture.
- Week 10, unplayed at run time (published card probability with the offset term swapped; inferred from the additive form, pick_probability.py:232-233): MIN at GB, anchor GB -5.0 -> -5.5, MIN +4.5 51.5 pct -> GB -4.5 50.2 pct (flips); SEA at LV, anchor SEA -4.5 -> -3.5, SEA -3.5 50.2 pct -> LV +3.5 51.6 pct (flips); NE at DET, anchor DET -2.75 -> -2.5, NE +2.5 59.0 -> 60.9 pct. Eleven games unchanged.
- Implication (inferred): the patched anchor is level with the fit-time anchor, so it restores the fitted contract. Against today's live anchor and the late anchor the capture alone would give, accuracy leans to the patch and log loss leans against it; nothing separates. The bundle is a consistency repair, not a measured gain, and the capture alone is not shown harmful. Before or after the freeze does not matter measurably, so the symmetric rule stands. Applied before Week 10 finishes it flips two near-even picks; apply after Monday night.
- Registry (measured): 40 cells recorded `unresolved_below_power` in registry/weak_signals/eng62_tuesday_reference_anchor.json from artifacts/eng62/tuesday_reference_batch.json (record exit 0, "recorded": 40); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0. Uncommitted; the orchestrator commits.

## Tried
One declared run of each script (the OOS script rerun once, seeded, after a wording edit to the record text). No src, scheduler or capture-store edit.

## Next (orchestrator, Tue 11-17 before the noon lock, after Week 10 settles)
1. `git apply tests/scratch/tuesday_reference_bundle/pool_hook_anchor.patch`; mirror the rule in scripts/check_served_terms.py:91 and :108-112 (it rebuilds the old anchor's books; not pinned).
2. Ride-alongs recorded in lanes: ENG-51 unit 2b (eng-51-opener-input-provenance-guard.md:46; roadmap-reconcile-2026-11-14-g3.md:28). backlog-all-2026-11-02.md:27, 38, 126, 128, 173 also match "pinned bundle" but were too long for the search and are unread. fix-picks Next 2 (coefficient refit) is a refit and needs its own evidence.
3. Re-enrol the validation cohort (v9 stops at the pinned edit): new protocol registry/studies/combined_vs_raw_2026_v10.json, `nfl-ats independent-validation enroll` with that protocol and a new `--study` (read cli_commands/prospective.py:483-517; confirm flags with --help), bump DEFAULT_STUDY in independent_validation.py, first capture; template commit f456fc7f (v9).
4. Add the Job in artifacts/eng62/late_tuesday_job.txt after capture_scheduler.py:1610, `--run-job odds_private_tue_2330`, restart the daemon once (check the process list first).
5. After the lock: `scripts/check_served_terms.py` exit 0; Wednesday, confirm the 23:30 quotes are stamped before 00:00 ET.

## Open
- With the bundle, movement between Tuesday noon and 23:30 is read by neither term, as in the fit; a fitted term for it is a separate unit.
- Accuracy and log loss disagree in sign on every patch contrast.
- How many leader books quote at 23:30 (never captured).
- Week 10 Thursday result was not read; its side does not change.
