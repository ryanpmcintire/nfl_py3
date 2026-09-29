# LEAD-88 - observed total news

## Goal
Replay one fitted total-move response against the served score-lattice guess; research only.

## State
**Measured:** one response replay complete; 1333/1,343 games and 101/101 last games scored.
All 33 missing base rows retained; 10 production-declined guesses remain in scratch evidence, outside paired MAE.
Diagnostic: 26 zero-move guesses changed through lattice reprojection; this mixes mechanisms.
Last-game MAE gain 0.069307 [-0.089109, 0.237624]; probability_positive=0.8006.
All-game gain 0.008252 [-0.045902, 0.061013]; probability_positive=0.6293.
Proposed unresolved status; report `docs/lead88_unit2.md`; no registry writes.

## Protocol (frozen before outcomes)
Owner amendment: baseline is served lattice guess; half-up market total is a comparator only.
One LAD response to Tuesday-to-deadline total move, six LOSO seasons 2020-2025, pushes retained.
One fixed fitted probability selects sides; no standalone flips. 82 bounded looks within parent 713 looks.
Full declaration saved before scoring in `tests/scratch/codex/lead88_unit2/protocol.md`, copied in report.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead88_unit2.py`; local UV_CACHE_DIR used.
Frozen quote clocks, base probability reproduction and retained-row checks executed in the real command.

## Record commands
Orchestrator only; candidate versus served, execute serially.
```bash
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record \
  --name lead88_unit2_last_mae --family lead88_total_response \
  --description 'Fitted total response plus lattice reprojection versus served guess on the final game of the week; 82 looks' \
  --source docs/lead88_unit2.md --league nfl --season-start 2020 --season-end 2025 \
  --effect 0.0693069306931 --effect-units mae_improvement \
  --interval-low -0.0891089108911 --interval-high 0.237623762376 \
  --standard-error 0.0820544998779 --probability-positive 0.80055 \
  --sample-games 101 --sample-blocks 101 --classification unresolved_below_power \
  --classification-evidence 'Retrospective LOSO; 82 looks; zero-move guesses can change; no power control or mechanism refutation' \
  --plain-summary 'For the final game of the week, this study nudges the score guess using changing sportsbook totals, then rounds the scores again. Keep the current guess while the effects of the nudge and rounding remain unresolved.'

.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record \
  --name lead88_unit2_all_mae --family lead88_total_response \
  --description 'Fitted total response plus lattice reprojection versus served guess on games with a served score guess; 82 looks' \
  --source docs/lead88_unit2.md --league nfl --season-start 2020 --season-end 2025 \
  --effect 0.00825206301575 --effect-units mae_improvement \
  --interval-low -0.045901768417 --interval-high 0.0610128822588 \
  --standard-error 0.0266996876959 --probability-positive 0.6293 \
  --sample-games 1333 --sample-blocks 107 --classification unresolved_below_power \
  --classification-evidence 'Retrospective LOSO; 82 looks; zero-move guesses can change; no power control or mechanism refutation' \
  --plain-summary 'For games with a served score guess, this study nudges the score guess using changing sportsbook totals, then rounds the scores again. Keep the current guess while the effects of the nudge and rounding remain unresolved.'
```

## Next
Orchestrator reviews/records; next unit must predeclare a zero-adjustment identity before further scoring.

## Open
Retrospective upstream/feature-vintage limits remain. No prospective or pool-rank claim; zero crossing closes nothing.
No serving change, registry write, publication, new tests, commit or push by this worker.
