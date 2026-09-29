# LEAD-72

## Goal
Execute the declared LEAD-72 unit within the assigned files.

## State
**Measured:** unit exited 0; 1,537 games / 107 weeks, 2 ranking looks, w=0.5.
Decisive 13 weeks: both 7-6-0, paired better/worse/equal 3/3/7;
gain 0.00 pp [95% -38.4615,+38.4615], probability_positive=0.50175.
All weeks: both 56-49-2, grade 53.2710%; gain 0.00 pp [-4.6729,+4.6729],
probability_positive=0.5016. **Inferred:** unresolved_below_power; registry pending.

## Predeclared protocol
Source: ROADMAP.md:855, copied verbatim with whitespace wrapping.

| LEAD-72 | ⬜ | Push-adjusted Best Pick statistic (rank 7 of 8) | **Added 2026-09-29 (unmeasured).** Mechanism: a push has identical probability on both sides, so it cannot change
the side chosen on any game, but it does change the top-1 Best Pick ranking: a game frozen on 3 or 7 carries about 8-10% push mass (row 693), so ranking on raw cover probability
over-nominates key-number games whose realised outcome is a push; the pool's own push rule (read from the pool rules, never fit) fixes what a push scores. Predeclared: replace the
LEAD-53 served statistic P(served side covers at the frozen line) by P(cover) + w*P(push) with w set by the pool rule, discrete PMF from the served lattice, replay 107 weeks
2020-2025 with the LEAD-53 tie rule, top-1 accuracy under the pool's Best Pick grade, paired week-blocked bootstrap against the current statistic, decisive weeks (where the nominee
changes) reported first, 2 looks, `probability_positive`. Deterministic arithmetic, no fitted parameter, so no fold choice is needed. Checked, not duplicated: LEAD-53 (which
instant, not which statistic), `confidence-best-pick-unification` lane (calibration of the ranking probability), POL-11 (dispersion pool). About 10 tool calls; the smallest
expected gain of the eight. |

Implementation fixed before scoring: use the 20260920T135435Z opener evaluation,
intersect exactly with the LEAD-53 20260818T221459Z archive, 2020-2025 only.
Use the Tuesday dispersion pool and LEAD-53 `select_with_tie_rule`; unresolved
ties retain equal weights. Keep each archived probability-selected side fixed.
**Read:** `PoolRules.push_points=0.5` (`pool_workbench.py:37-45`), hence w=0.5;
grade win/push/loss as 1/0.5/0. Normalized Best Pick grade excludes bonus scaling.
Compare raw unconditional served-side cover with cover + 0.5*push; never use
conditional non-push probability as raw cover. Bootstrap 20,000 paired weekly
draws, seed 20260929, percentile 95% intervals, half credit for zero-effect draws.
Two declared ranking looks; changed nominees are the required decisive subset.
Report each season descriptively with fixed coefficients (1, 0) and (1, 0.5).
There is no fitting: IS score, IS/OOS gap, and fitted fold coefficients are N/A;
OOS refers to archived chronological forecasts, not a fresh untouched outer test.

## Tried
Protocol saved before outcomes; no protocol revised. Ran once (exit 0):
`.tools/uv.exe run --no-sync python -B scripts/lead72_unit1.py`.
`UV_NO_CACHE=1` avoids the inaccessible default uv cache; numerical threads capped at 2.
Scoped Ruff format/check passed. Log: `.tmp/lead72_unit1.log`.
Report: `docs/lead72_results.md`; paired nominations: `docs/lead72_weekly.md`.

## Next
Orchestrator reviews and executes the command below serially, then owns integration.

## Open
Historical reused archive cannot establish prospective benefit. IS/fitting gap N/A;
fixed per-season coefficients and all season results are in the report. No serving change.

## Record commands
Pending; never executed by this worker. Correlated decisive subset retained in notes.
```powershell
.tools/uv.exe run --no-sync nfl-ats weak-signals record `
  --name lead72_push_adjusted_best_pick_2020_2025 `
  --description "Two fixed Best Pick rankings; pool-grade accuracy; pushes score 0.5" `
  --source docs/lead72_results.md --effect-units accuracy_points `
  --effect 0 --standard-error 2.29016759302 `
  --interval-low -4.67289719626 --interval-high 4.67289719626 `
  --probability-positive 0.5016 --sample-blocks 107 `
  --classification unresolved_below_power --league nfl `
  --season-start 2020 --season-end 2025 --family lead72_push_adjusted_best_pick `
  --classification-evidence "No closing ground established; reliability and power control not measured" `
  --notes "2 ranking looks; decisive n=13: 0 pp, CI [-38.4615,38.4615], probability_positive=0.50175; fixed w=0.5; no fitting; do not pool with ordinary ATS accuracy"
```
