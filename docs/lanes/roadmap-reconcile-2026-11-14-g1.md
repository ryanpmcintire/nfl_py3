# Roadmap reconcile 2026-11-14, group 1 (MOD-28 to MOD-54)

## Goal
Transcribe the recorded state of the 27 rows MOD-28 to MOD-54 from their 2026-11-03 lanes into a verdict file so the orchestrator can replace the not-started marks in ROADMAP.md. No analysis, no new research verdict.

## State
Done 2026-11-14 14:40 ET. Verdicts: `tests/scratch/reconcile_2026_11_14/g1.json` (27 objects, parses, ASCII, notes at most 45 words).
- done 15: MOD-28, 29, 31, 32, 33, 34, 37, 38, 44, 45, 46, 48, 50, 51, 53.
- in_progress 7: MOD-36, 39, 40, 42, 43, 47, 52.
- forward_only 4: MOD-35, 41, 49, 54 (score date 2026-11-10: Week 9 kicked off Thu 2026-11-05 per the MOD-41 lane, so it settled Mon 2026-11-09; inferred).
- blocked 1: MOD-30. not_started 0.
- MOD-41 serving decision (read, MOD-41 lane root note 2026-11-03 17:44 ET): not served; raw-scale, served-scale and averaged-window terms log forward only. Week 9 and Week 10 records exist for all three, for mod35_cold_exposure and for the MOD-54 mirrored base (measured, directory listing).
- Registry: all 30 families the lanes cite exist under registry/weak_signals (measured, glob and grep). Recorded closing grounds match the lanes: MOD-31 and MOD-38 refuted_mechanism on zero split-half reliability, three MOD-53 2020-25 cells wrong_sign_resolved. Everything else is unresolved_below_power.

## Tried
Read ROADMAP rows (first 800-900 characters each), all 28 lanes, registry file names and closing-ground counts, forward record folders, scripts/mod36_noise_floor.py lines 18-63, scripts/forward_ledger.py arguments. Ran no script. ROADMAP.md, registry, src, scripts and tests untouched.

## Next
Orchestrator updates the 27 ROADMAP rows from g1.json, then the in_progress units, most useful for the weekly card first:
1. MOD-39: pool leverage, elimination and clinch differences as one declared family (scripts/mod39_leverage.py, artifacts/sim09/stakes_teamweek.csv). Its domain is Weeks 10-16, which is now. Current cells mix three effect units, so re-record in one unit before `nfl-ats weak-signals pool`.
2. MOD-52: record the 17 per-cell gains from artifacts/mod52/unit1_stability.json, then the declared term times absolute spread LOSO unit. Bears on the MOD-41 serving read.
3. MOD-47: home-short minus away-short as one LOSO term beside the four served terms (scripts/mod47_unit2_global_calibration.py).
4. MOD-43: weeks 4-12 restriction in scripts/mod43_qb_learning_curve.py.
5. MOD-36: bring the noise floor to the declared 200 matched-marginal draws (`python scripts/mod36_noise_floor.py 100 1,12` resumes draws.jsonl; line 49 still draws standard normal).
6. MOD-40: separation and time to throw as a declared family. 7. MOD-42: snap-share starter definition.
Forward rows can be scored today: `scripts/forward_ledger.py --weeks 9` (not run here).

## Open
- MOD-36 ran 100 standard-normal draws where the row declares 200 draws matched to candidate marginals; marked in_progress on that gap.
- MOD-52 has no registry cells; its lane says no admissible unit for homogeneity statistics, but the per-cell gains are log-loss improvements.
- MOD-48 declared a serving bar and root set it aside the same hour; no operative pre-declared serving rule for MOD-41 stands.
- MOD-35 heat-index half was never built (no humidity or dew point locally); MOD-49 swapped the kneel-and-spike variant for offence-only plays; MOD-40 lane names mod40_fit.py but the file is scripts/mod40_ngs_fit.py.
- MOD-33 has a stub lane (last-game-exits) beside the results lane (prior-week-exits, which has no Goal or State header).
- MOD-30 stays blocked until the owner approves the github_nfelo_public_ratings source entry; it is absent from config/source_policies.json today (measured, grep).
- MOD-54 lane says its scheduler job was proposed, not applied, yet a Week 10 run file exists; confirm what wrote it.
