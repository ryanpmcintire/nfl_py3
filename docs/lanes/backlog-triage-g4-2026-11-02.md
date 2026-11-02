# Backlog triage G4 (simulator and moonshots), 2026-11-02

## Goal
Classify SIM-06/07/08/09/11 and SKY-01/02/03/04/05/07. Read-only; master lane backlog-all-2026-11-02.md.

## State (read: ROADMAP rows + lanes; nothing run)
- SIM-06 NEXT-UNIT (re-scope): no differentiable env exists; SIM-04/SIM-10 are non-parametric resamplers. First unit: gradient-fit a small transition/outcome head on sim10 drive features, compare held-out log loss vs the kernel resampler on the same 2018-25 drives, record in weak-signals. Writes scripts/sim06_diff.py, artifacts/sim06/. CPU 1 core, no sim.
- SIM-07 NEXT-UNIT: overlaps SIM-09 personnel catalogue (35 mechanisms, F1 unit latents). First unit: take docs/sim09_personnel_catalogue.md rows needing below-play data; measure on nflverse pbp (offense_personnel/defense_personnel cols) personnel x defence-personnel EPA interaction, split-half; record. Writes scripts/sim07_personnel.py. CPU 1 core. Participation-level tracking needs NGS/paid data.
- SIM-08 NEXT-UNIT is blocked by its own lane rule: any new mechanism needs a declaration first. Open defect margin-3 mass (.103 v .152). Unit: declare unit (3) market-anchored sim, no scoring until declaration filed.
- SIM-09 NEXT-UNIT: integration run in lane Next (crzhk + U4e/U4a/U4c/U4f variants, 3 e5 seeds). Heavy CPU: 3 workers x 3 seeds, one job at a time; lane files sim09 only.
- SIM-11 NEXT-UNIT: lane Next: special-teams FG-make latent (tests/scratch/sim11_st/proposal.patch, derived N(0,.0945), not a tuning knob). Verify: python scripts/sim10_drive.py sim --seed 11 --workers 11 --realsched 1 --st ... vs rshc_s11. CPU: 6 workers, 1 seed (compare R2 .113 v .131, strength 51.34). Rejected: --lscale (compensating constant), --form 1/2.
- SKY-01 NEXT-UNIT: LEAD-78/87/88/97 already price news moves, but no report-to-status table exists. Unit: build timestamped injury-report -> status table with source column from the existing injury feed; measure lead time vs line move; record.
- SKY-02 OWNER-PHYSICAL: needs lawful video source (licensed broadcast or tracking feed), e.g. NFL Next Gen Stats / Sportradar tracking license; unreachable otherwise.
- SKY-03 NEXT-UNIT: matchup effect separated from opponent selection and game script: WR-CB / pass-rush pair EPA with team-week random effects; split-half; overlaps SIM-09 catalogue F1. Writes scripts/sky03_matchup.py. CPU 1 core.
- SKY-04 NEXT-UNIT: descriptive done (docs/book_leadership.md, artifacts/book_leadership/20260903T201202Z/). Next: lead-lag per book pair on the 7,424 snapshots (latency, diffusion), record. Full model needs sub-minute quote tape (owner cancelled Odds API; no source).
- SKY-05 CLOSE-NOW candidate: nfl-ats rotation / weak-signals registry already propose and record under budgets; LEAD-99 delivers the LLM part. Evidence to cite: rotation CLI; owner approval gate not verified on disk.
- SKY-07 CLOSE-NOW: row records retirement 2026-09-26 (open_benchmark.py removed b7ed31d; docs/open_benchmark_suite.md).

## Tried
Row text and lane State/Next only; no commands run, no processes started.

## Next
Orchestrator picks SIM-11 ST unit and SIM-09 integration (do not overlap CPU); fill claims above.

## Open
SKY-05 approval-gate evidence; SIM-06/07 scope confirmation by owner.
