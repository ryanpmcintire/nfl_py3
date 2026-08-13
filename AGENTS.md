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

## Session handoff

- Update documentation and `ROADMAP.md` when evidence or priorities change.
- If the active weekly forecast changed, run `nfl-ats publish-predictions`.
- Run `nfl-ats handoff` after substantive work, review `HANDOFF.md`, and report
  remaining Git changes plus the exact checks run.
