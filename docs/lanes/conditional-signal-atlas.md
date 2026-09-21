# Conditional signal atlas — MOD-19

## Goal

Show when a situational input helps the fitted probability, using paired,
out-of-season evidence. The mockup is illustrative; none of its numbers are data.

## State

- 2026-09-21 correctness audit: **read** `scripts/signal_atlas.py:174-181`
  computes unique independent-flip contributions. It cannot measure the current
  combined probability. Historical work is preserved in
  [the legacy lane](done/conditional-signal-atlas-legacy.md), not approved for this chart.
- **Read** `src/nfl_ats/pick_probability_fit.py:390-419,474-485`: the current
  fitter saves held-out-season and chronological predictions separately and
  acknowledges that feature selection used these seasons. Neither is an untouched
  outer test. `artifacts/active_pick_probability.json` identifies the fitted source.
- First bounded increment: Findings explains that distinction and the planned
  paired comparison. No contribution chart, experimental result or serving change.

## Tried

Audited the legacy scorer and current fitter. Reusing old flip deltas would answer
the wrong question. Replaced the page's blanket claim that every result learns
only from earlier games. No new experiment, test function or test file.

**Measured verification (2026-09-21):** all commands below used
`.\.tools\uv.exe run --no-sync` with a writable temporary uv cache:
`ruff format --check .`, `ruff check .`, `mypy src` and `pytest -q` passed
(4,529 passed, 9 skipped). Pytest needed a fresh temporary root after a permissions
error. `nfl-ats publish-board` passed. Rendered text diff: Findings adds the two
explanations above; home injury-feed age updates from 30 to 33 hours; model/history
have no visible text changes. Review found no added tests or research-only runtime
assertions. `git diff --check` passed.
**Measured startup:** scheduler `--once` and `--status --brief` passed; daemon
running, 216 jobs, 206 OK, 10 acknowledged old misses, zero open misses/never-run.

## Next

1. Predeclare the first family before scoring: pooled situational inputs
   (`flag_sum`), overall and every existing `week_in_season` cell. Inventory all
   arms, metrics and looks; do not select the best cell for the headline.
2. Freeze the active fitted population, model identity, feature version, split
   library and hashes. Fail closed on mismatched sources, duplicate games or
   missing paired predictions. Verify each feature's pregame availability in its
   builder; using the newest snapshot alone does not establish chronology.
3. Refit full and reduced models in identical held-out-season folds, removing
   `flag_sum` from the reduced design. Learn all parameters from the training
   fold only. Retain intercept, model and market inputs; no independent pick flips.
   Keep chronological evaluation separate and report in-sample/out-of-season gaps.
4. Save paired game probabilities, fold coefficients and source lineage. Use the
   same games and opener grade for all arms, including market and model-only
   baselines. Report excluded/ungraded/push counts; conditional non-push
   probabilities must not be relabelled as push or unconditional probabilities.
5. Report decisive-game records before accuracy effects; also report Brier/log
   loss, reliability, coefficient stability and season results. Show uncertainty
   and `probability_positive`, use an exact/permutation null for small splits,
   and avoid causal claims. Document resampling and dependence assumptions.
6. Record unresolved cells through `weak-signals record` before publishing a
   verdict. An interval spanning zero never closes work. Serving is a separate
   decision through the one fitted probability; no cell overrides a pick.
7. Only then render the measured chart in Findings, with every declared cell,
   sample size, uncertainty and season context. Read values from artifacts keyed
   to the active model; keep technical lineage outside reader-facing copy.

## Open

Paired evaluator and measured chart remain to build in the next bounded session.
Full-game source availability must be audited before treating the fitted
population as ready for this experiment. The old scorer remains legacy-only
research code; no claim of a new signal result or closed research line was made.
