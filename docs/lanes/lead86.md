# LEAD-86 unit 2: filtered market regularization

## Goal
One declared LOSO replay against the pinned served base; no served changes.

## State
**Measured:** complete; 1,309 games / 107 week blocks; candidate 721-588,
served 757-552. Decisive 34-70, 32.69% [24.14%, 41.84%].
Brier improvement -0.000901 [-0.002573, +0.000616], probability_positive 0.1343;
accuracy -2.7502 points [-4.2113, -1.2432], probability_positive 0.0003.
24 price panels dropped; maximum projection error 0.000972 points.
Penalties (2020-25): 0.1/1/0/0/0/0.1. Candidate Brier IS/OOS
0.244502/0.245404; gap +0.000903 [-0.001088, +0.003055].
**Inferred:** primary Brier/regularization mechanism unresolved; proposed
accuracy-cell wrong_sign_resolved applies only to this training workflow.
Frozen pre-score protocol: docs/lead86_unit2.md (Frozen protocol), also
tests/scratch/codex/lead86_unit2/protocol.md. Two decision comparisons;
851 diagnostic looks disclosed; no outcome-driven protocol changes.

## Tried
**Measured:** real replay exited 0; cached-report refresh and independent
base/row audit passed. Ruff check and format --check passed for the script.
Commands: .tools/uv.exe run --no-sync python scripts/lead86_unit2.py;
same uv prefix with ruff check / ruff format --check scripts/lead86_unit2.py.
Use scratch UV_CACHE_DIR as in report; run/audit logs under tests/scratch/codex/.

## Record commands
Prepared only; orchestrator executes serially. Both compare candidate to served.
~~~bash
export UV_CACHE_DIR=tests/scratch/codex/lead86_unit2_uv_cache
.tools/uv.exe run --no-sync nfl-ats weak-signals record \
  --name lead86-unit2-candidate-vs-served-brier --description 'Price-filtered training workflow versus served LOSO: opener brier.' --source docs/lead86_unit2.md --effect=-0.000901410060267 --effect-units brier_improvement --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2025 --interval-low=-0.00257271593288 --interval-high=0.000615710049488 --probability-positive 0.1343 --sample-games 1309 --sample-blocks 107 --family lead86-unit2-filtered-regularization --category market --plain-summary 'After dropping conflicting sportsbook prices, using their prices in training gave no clear forecast improvement. Keep the current pool picks.' --classification-evidence 'No admissible Brier closing ground established.'
.tools/uv.exe run --no-sync nfl-ats weak-signals record \
  --name lead86-unit2-candidate-vs-served-accuracy --description 'Price-filtered training workflow versus served LOSO: opener accuracy.' --source docs/lead86_unit2.md --effect=-2.75019098549 --effect-units accuracy_points --classification refuted_mechanism --league nfl --season-start 2020 --season-end 2025 --interval-low=-4.2113323124 --interval-high=-1.24315301972 --probability-positive 0.0003 --sample-games 1309 --sample-blocks 107 --family lead86-unit2-filtered-regularization --category market --plain-summary 'After dropping conflicting sportsbook prices, this trial made fewer correct pool picks. Keep the current picks; the broader training idea remains open.' --closing-ground wrong_sign_resolved --classification-evidence 'The full accuracy interval is negative for this specified workflow, not the broader regularization mechanism.'
~~~

## Next
Orchestrator records the two cells, then reviews the training-design mismatch
in a fresh bounded task; do not rerun this unit.

## Open
**Read:** docs/loso_base_artifact.md:26-41 retains upstream holdout limitations.
**Inferred:** served fits five seasons; candidate fits three with separate
selection/calibration. Matched-zero Brier +0.000074 [-0.000045, +0.000178],
probability_positive 0.8984, decisive 3-1: broader mechanism remains unresolved.
Full losses, intervals, coefficients, gaps and reliability: docs/lead86_unit2.md.
