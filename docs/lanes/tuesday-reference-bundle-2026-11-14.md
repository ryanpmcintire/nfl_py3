# Tuesday reference bundle (2026-11-14)

## Goal
Decide, before the Week 11 lock (Tue 2026-11-17 noon ET), which Tuesday quotes the pool-line offset term and the line-move term read: keep live, apply the prepared bundle, or wait for a refit. Two units prepared the evidence; no src file was edited.

## State (times are the artifacts' own generated_at_utc stamps)
- Anchor unit (measured; scripts/eng62_tuesday_reference_oos.py -> artifacts/eng62/tuesday_reference_oos.json, .out; 40 cells, registry/weak_signals/eng62_tuesday_reference_anchor.json). Read pool_hook.py:78, :91: the fitted pool line is the anchor plus or minus a half at one moment, so the reference is the freeze itself. Rule: median across books of each book's Tuesday quote nearest the freeze, tie to the earlier; fallback unchanged. Patch tests/scratch/tuesday_reference_bundle/pool_hook_anchor.patch (`git apply --check` exit 0). With the move baseline fixed late: nearest - live +0.31 pts [-0.32,+0.93] P+ .84 (8-5.5); nearest - fit +0.06 [-0.68,+0.81] P+ .56.
- 2026 replay, Weeks 3-9 (measured; tests/scratch/tuesday_reference_bundle/replay_2026.py -> artifacts/eng62/tuesday_reference_replay_2026.json): 5 anchors change, 0 sides, 59-45 either way. No Tuesday quote before noon exists in Weeks 2-10, so live the rule reads the 12:10 capture.
- Week 10, corrected. The replay's Week 10 rows started from pool_card.csv `pick_probability`, the forecast step's raw model cover chance (read pool.py:26, cli_commands/prediction.py:850-851; written 18:42 UTC), and subtracted an offset term that number never held. The card's cover chance is the fitted probability computed at publication, 18:50 UTC (read explanations.json `model_probability`, `computed_now`; equals CURRENT_PREDICTIONS.md), which already holds the model weight, flags, move term and offset term: SEA 62.4 pct served against 50.2 raw. Redone on the served probabilities with the replay's own offset change (inferred from the additive form, pick_probability.py:232-233): MIN at GB, MIN +4.5 51.1 pct -> GB -4.5 50.7 pct (changes side); SEA at LV, SEA -3.5 62.4 -> 60.8 pct (holds); NE at DET, NE +2.5 53.5 -> about 55.4 pct. One Week 10 side would change, not two; the two-sides statement is withdrawn. Weeks 1-9 rows rebuild the probability from its terms (read replay_2026.py:129-142) and stand.

## System arms (declared before outcomes were joined; scripts/eng62_reference_system_arms.py)
Same harness and 806 games as the anchor unit (2023-25, pushes kept, stage 1 and push table out of season, week-block bootstrap; pool line from DraftKings' first quote after noon, the last-before build in the artifact). Family eng62_reference_system_arms.
- A fit definitions: anchor Tue 08:55; move = the fit's own column (leader books' last quote before Wednesday, Tue 23:55, to the deadline); offset coefficient served .805.
- B live: anchor and move baseline at the Tue 13:55 snapshot (the 14:25 slot). C bundle: anchor nearest noon (Tue 11:55), move as A. D: anchor as C, move baseline at the same noon quote, coefficients as A. E: D's inputs, stage 1 and offset coefficient fitted leave-one-season-out. A_fold: A with the offset coefficient fitted leave-one-season-out.
- B and D moves = fit column + (replicated move at the arm's baseline - replicated move at the fit baseline). Blind-spot games: the all-book median, or the leader books' baseline median, differs between noon and the last Tuesday quote.
- Looks: 10 contrasts x 3 populations x 2 pool builds x 3 units = 180, plus 4 matched-null looks.

Run (measured; artifacts/eng62/reference_system_arms.json, .out). Stage 1 equals the artifact (gap 0.0). 10 of 816 scored games have under five books at the fit snapshot; no game lacks a leader quote at any reference. Replicated fit move equals the fit column on 795 of 799 games (max gap 0.5). The move input differs from the fit's on 182 games in B, 213 in D. Blind-spot games 254 of 806. Levels, pct / log loss: A 57.45 / .68237, B 57.14 / .68230, C 57.52 / .68231, D 57.02 / .68188, E 57.08 / .68173.

| contrast | decisive | accuracy pts [interval] P+ | log loss gain [interval] P+ | seasons 23/24/25 | blind-spot (253): decisive, pts P+, log loss P+ | rest (552): pts P+, log loss P+ |
|---|---|---|---|---|---|---|
| B - A | 24.5-27 | -0.31 [-2.07,+1.41] .37 | +.00007 [-.00324,+.00329] .52 | +0.37 -1.70 +0.37 | 16-19, -1.19 .32, +.00121 .60 | +0.09 .53, -.00045 .33 |
| C - A | 7.5-7 | +0.06 [-0.68,+0.81] .56 | +.00005 [-.00134,+.00150] .52 | +0.19 -0.57 +0.55 | 4-2, +0.79 .82, +.00118 .82 | -0.27 .26, -.00046 .31 |
| D - A | 26.5-30 | -0.43 [-2.32,+1.49] .32 | +.00049 [-.00287,+.00373] .60 | -1.49 +0.19 0.00 | 20.5-23, -0.99 .36, +.00230 .68 | -0.18 .37, -.00034 .37 |
| E - A | 32-35 | -0.37 [-2.30,+1.66] .37 | +.00064 [-.00267,+.00389] .66 | -2.43 +1.13 +0.18 | 19-22, -1.19 .34, +.00211 .67 | 0.00 .51, -.00004 .49 |
| C - B | 23.5-20.5 | +0.37 [-1.01,+1.76] .70 | -.00001 [-.00274,+.00289] .51 | -0.19 +1.13 +0.18 | 20.5-15.5, +1.98 [-2.52,+6.63] .80, -.00003 .50 | -0.36 .26, -.00001 .50 |
| D - B | 9.5-10.5 | -0.12 [-1.18,+0.93] .40 | +.00042 [-.00075,+.00162] .75 | -1.87 +1.89 -0.37 | 7.5-7, +0.20 .55, +.00109 .70 | -0.27 .28, +.00011 .66 |
| E - B | 18.5-19 | -0.06 [-1.27,+1.22] .47 | +.00057 [-.00098,+.00209] .76 | -2.80 +2.83 -0.18 | 9-9, 0.00 .51, +.00090 .66 | -0.09 .45, +.00042 .76 |
| D - C | 21.5-25.5 | -0.50 [-2.19,+1.11] .28 | +.00043 [-.00264,+.00334] .61 | -1.68 +0.75 -0.55 | 19-23.5, -1.78 [-7.17,+3.45] .25, +.00112 .59 | +0.09 .60, +.00012 .60 |
| E - C | 28.5-32 | -0.43 [-2.23,+1.43] .32 | +.00058 [-.00235,+.00341] .66 | -2.61 +1.70 -0.37 | 17.5-22.5, -1.98 [-7.28,+3.29] .23, +.00093 .59 | +0.27 .65, +.00042 .72 |

Brier follows log loss in sign on every row (P+ .46 to .74). E - A_fold: 34.5-36, -0.19 [-2.26,+1.93] P+ .43, log loss +.00097 P+ .71. In-sample minus out-of-season accuracy gap: -0.43 to +0.50 pts.
- Coefficients (measured). Move, logit per point: served .220; fit-definition folds .229, .244, .188; E folds .233, .208, .177 (all seasons .205). Offset: served .805; A_fold folds 1.25, .70, .87; E folds 1.04, 1.26, .98, each on two training seasons.
- Noise floor (read mod-36-mod36-add-one-noise-null.md:46-48). Power: +1.5 pts at 1,503 games, +3.6 at 576, so about +2.5 at 806 (inferred, interpolated); the largest difference is 0.50 pts on all games and 1.98 on 253. One parameter costs about .0008 log loss at 806 (inferred, 1/n); the largest gain is .00064, .00097 against A_fold. A losing changed-pick record appears in 67-78 pct of uninformative terms. Matched null (noon-to-late change shuffled within season): D - C accuracy 46th percentile, log loss 84th; E - C 47th and 85th. No difference is distinguishable from nothing on 806 games. The one consistent lean: log loss favours the arms that read Tuesday afternoon movement (D, E; P+ .60 to .76) while accuracy leans against them.
- What each choice means (the orchestrator chooses).
  - Keep live (B): no pinned edit, re-enrolment or new capture. Both terms share one reference, so every move after noon is read once. Level with the fit definitions as measured. Right if the coefficients carry to inputs a few hours off the fit (not rejected) and the 14:25 capture keeps its leader books; the anchor then depends on capture time.
  - Apply the bundle (C): the terms run on exactly what they were fitted on. Against live +0.37 pts P+ .70, log loss level. Movement between noon and midnight (254 of 806 games) is read by neither term. Needs the pinned edit, cohort re-enrolment and a 23:30 capture whose book coverage is unobserved; applied before Week 10 settles it changes one near-even side. Right if matching the fitted inputs matters more than reading Tuesday afternoon movement, and the leader books quote at 23:30.
  - Wait for a refit (E): one noon reference, refitted. Accuracy leans against it (2023 -2.61), log loss for it; the move coefficient is stable, the offset coefficient is not. D, the same inputs unrefitted, is level with E, so the refit adds little to the inputs. Right if the log loss lean is real, which needs more seasons or the 2026 weeks; B or C runs meanwhile.
- Registry (measured): 90 cells `unresolved_below_power` from artifacts/eng62/reference_system_arms_batch.json (record exit 0); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0. Uncommitted.

## Tried
One declared run of each script (the anchor script rerun once, seeded, after a wording edit); ruff check and format clean on all three. No src, scheduler or capture-store edit.

## Next (orchestrator, Tue 11-17 before the noon lock)
1. Choose among B, C and E from the block above. B needs nothing. E needs a declared refit unit first.
2. If C, after Week 10 settles: `git apply tests/scratch/tuesday_reference_bundle/pool_hook_anchor.patch`; mirror the rule in scripts/check_served_terms.py:91 and :108-112; re-enrol the cohort (new protocol registry/studies/combined_vs_raw_2026_v10.json, `nfl-ats independent-validation enroll`, bump DEFAULT_STUDY in independent_validation.py; template commit f456fc7f); add the Job in artifacts/eng62/late_tuesday_job.txt after capture_scheduler.py:1610 and restart the daemon once; after the lock run scripts/check_served_terms.py; Wednesday, confirm the 23:30 quotes are stamped before 00:00 ET.
3. Ride-alongs for a pinned edit: ENG-51 unit 2b (eng-51-opener-input-provenance-guard.md:46).

## Open
- Whether the log loss lean toward reading Tuesday afternoon movement is real; 806 games cannot say.
- How many leader books quote at 23:30 (never captured).
- The offset coefficient refit is unstable on two training seasons.
- Week 10 Thursday result was not read; its side does not change.

## Root decision (2026-11-14 15:54 ET)
Keep the live configuration (arm B). No pinned edit, no late-Tuesday capture, validation cohort v9 untouched. Grounds (measured, artifacts/eng62/reference_system_arms.json): on 806 games no arm is distinguishable from another (bundle minus live +0.37 pts [-1.01, +1.76], P+ .70, log loss level; live minus fit definitions -0.31, P+ .37); the bundle would restore the fit's blind spot on Tuesday afternoon movement and costs a cohort re-enrolment for a difference the sample cannot see (AGENTS: a promotion bar is not a decision bar, and a probability just above 0.5 is not a reason to serve). The patch stays on file unapplied (tests/scratch/tuesday_reference_bundle/pool_hook_anchor.patch). Reopen only as arm E, a declared refit of both coefficients on a noon reference, at the next cohort boundary; its offset coefficient was unstable across folds (1.04, 1.26, .98 against the served .805), so it needs the harness-level reconcile first (lane harness-level-reconcile-2026-11-14.md). The earlier Week 10 statement is withdrawn: on served probabilities the bundle would change one side (MIN at GB), not two.
