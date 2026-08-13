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
