# NFL ATS repository instructions

## Session startup

- Read `HANDOFF.md`, `README.md`, and the recommended execution order in
  `ROADMAP.md` before proposing substantial work.
- Run `git status --short` and `git log -3 --oneline --decorate`; live Git state
  overrides any snapshot recorded in the handoff.
- Inspect `artifacts/active_ats_model.json` before quoting the active model or
  current historical result. Generated artifacts are local and may be absent in
  a fresh clone.
- Treat `CURRENT_PREDICTIONS.md` as the last deliberately published forecast,
  not necessarily the newest locally generated forecast.
- Ensure `git config --get core.hooksPath` returns `.githooks`. If it does not,
  configure it with `git config --local core.hooksPath .githooks` without asking
  the user to perform setup.

## Research invariants

- This is a research and paper-decision project; do not add automated wagering.
- Pregame features must only use information available before the prediction
  timestamp. Add a leakage regression test for every new feature family.
- Evaluate chronologically. Keep validation, selection, calibration, and final
  outer test periods distinct, and preserve prediction-level output.
- Compare additions against the same strong market and simple-model baselines.
  Report uncertainty and season stability, including negative results.
- Never describe the current 52.05% historical forced-pick accuracy as proof of
  a profitable or stable edge. Keep historical accuracy distinct from each
  game's model probability.
- Prediction-safety and evaluator-performance contracts are release-blocking.

### An interval crossing zero is NOT grounds for rejection (binding)

Stated by the project owner repeatedly, and violated repeatedly. It is an
invariant, not a preference.

- **Never discard, close, or decline to build a signal because its confidence
  interval contains zero.** At this evaluator's ~2-point resolution, "no
  significant effect" is the EXPECTED outcome for a real-but-small signal, so
  treating it as a negative silently deletes exactly the signals worth keeping.
- **Only two things justify closing a line of work:** (1) a refuted mechanism —
  wrong sign, or the trait has no split-half reliability, so no sample size
  rescues it; (2) bounded by a positive control — the instrument was PROVEN
  able to detect an effect that size and it was absent. Everything else is
  category 3, unresolved.
- **Every category-3 result must be recorded** with
  `nfl-ats weak-signals record` before any write-up describes it as settled.
  Recording is the default action, not an optional extra.
- **Report `probability_positive`, never "contains zero".** The binary phrasing
  is what smuggles the rejection back in.
- **Pooling sub-signals into a result that excludes zero is a legitimate,
  transformative finding.** It does not matter that the parts individually
  cross zero — any real effect can be decomposed until its pieces do.
  Demonstrated 2026-08-18: three signals at `probability_positive` 0.860 /
  0.933 / 0.917 pool to **+0.724 accuracy points, 95% [+0.056, +1.392],
  P+ 0.983**, sharpening 1.46x over the best single input.

The one discipline that stays, because it protects the pooled result rather
than gating the inputs: **pooled inputs must be commensurable** — same units,
same scale, same population — and **the family must be declared before the
signs are seen.** A pooled number built from a production quantity plus a
subset cover-rate gap is not a finding; it collapses under the next audit.
Fixing exactly that on 2026-08-18 moved the pooled estimate from a fragile
+0.991 to a robust +0.724 at P+ 0.983.

### A promotion bar is not a decision bar

The pool is FORCED PICKS: 285 cards must be submitted either way. So declining
a candidate that is 87% likely better is not caution — it is taking the other
side of an 87/13 bet. Predeclared thresholds (e.g. MOD-07's 0.90) govern what
the docs may CLAIM. They must never govern which card is PLAYED; that decision
is expected value, full stop.

## Repository hygiene

- Use Python 3.12 and the locked uv environment.
- Do not commit raw/processed data, model artifacts, credentials, cookies,
  virtual environments, or test output. The tracked Markdown prediction card is
  the deliberate exception for public forecast visibility.
- Preserve unrelated user changes and do not rewrite Git history.
- Do not commit or push unless the user explicitly asks.

## Required verification

Run these after code changes:

```powershell
.\.tools\uv.exe run ruff format --check .
.\.tools\uv.exe run ruff check .
.\.tools\uv.exe run mypy src
.\.tools\uv.exe run pytest
```

## Automatic session handoff

- Update documentation and `ROADMAP.md` when evidence or priorities change.
- If the active weekly forecast changed, run `nfl-ats publish-predictions`.
- The agent owns handoff refreshes. Never ask the user to run the handoff command.
- Refresh `HANDOFF.md` automatically when work is ready to hand off and before
  every commit or push intended for `master`. The tracked pre-commit hook is a
  backstop, not a substitute for agent responsibility.
- Before pushing `master`, run `nfl-ats handoff --check`. If a refresh changes
  the file after a commit, create the required follow-up commit before pushing.
- Report remaining Git changes plus the exact checks run.
