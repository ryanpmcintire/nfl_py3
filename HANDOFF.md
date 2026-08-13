# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-08-13T11:37:02.464464+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `6bb2ecdc47aa` — Refresh session handoff
- Pending change set: 13 paths
  - `A  .githooks/pre-commit`
  - `A  .githooks/pre-push`
  - `M  .github/workflows/ci.yml`
  - `M  AGENTS.md`
  - `M  CONTRIBUTING.md`
  - `M  HANDOFF.md`
  - `M  README.md`
  - `M  ROADMAP.md`
  - `M  src/nfl_ats/cli.py`
  - `M  src/nfl_ats/handoff.py`
  - `M  tests/test_cli.py`
  - `M  tests/test_handoff.py`
  - `A  tests/test_repository_policy.py`

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `be9326573294de5a`
- Method/profile/regressor: `market_residual` / `player` / `ridge`
- Historical ATS classification: **1,080 / 2,075 (52.05%)**
- Linked forecast: **2026 Week 1**, created `2026-08-12T21:15:33.385895+00:00`

The 52.05% figure is historical forced-pick ATS classification accuracy, not a
game-specific probability and not proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 1** from model `be9326573294de5a`, published `2026-08-12T22:23:56.868653+00:00`. It is an early, mutable research preview.

## Local reproducibility inventory

- canonical team features: **present** (`data/processed/game_features.parquet`)
- play-by-play features: **present** (`data/processed/game_features_pbp.parquet`)
- player features: **present** (`data/processed/game_features_player.parquet`)
- active model manifest: **present** (`artifacts/active_ats_model.json`)

Raw data, processed features, fitted models, and evaluation artifacts are intentionally
ignored by Git. A fresh clone therefore starts with documentation, source, tests, and
the last published Markdown forecast but must rebuild or transfer local artifacts.

## Highest-priority work

1. Maintain the prediction-safety contract and add a regression canary for every production error or newly supported output type.
2. Build the historically feasible player layer first: timestamped 2009–2024 injuries, lagged 2013–2025 snap shares, and 2002–2025 roster continuity.
3. Add joint score/total distributions and compare calibration methods inside the nested protocol.
4. Use 2016–2025 participation/NGS for position-unit and formation effects; individual receiver-corner pairs remain too sparse for an initial model.
5. Continue collecting timestamped, book-specific opening/current/closing quotes. The one-season free sample validates plumbing but cannot validate a historical line-movement edge.
6. Attempt drive simulation only after simpler distributional baselines exist.

The roadmap is authoritative. Negative results remain part of the evidence base and
must not be silently removed or retuned away.

## Commands that matter

```powershell
# Manual diagnostic/recovery only; the agent and Git hooks own normal refreshes
.\.tools\uv.exe run nfl-ats handoff --check

# Launch the local dashboard
.\.tools\uv.exe run nfl-ats dashboard

# Quality gates
.\.tools\uv.exe run ruff format --check .
.\.tools\uv.exe run ruff check .
.\.tools\uv.exe run mypy src
.\.tools\uv.exe run pytest
```

## Automatic end-of-session contract

1. Reconcile completed work and new evidence with `ROADMAP.md` and relevant docs.
2. If the synchronized weekly forecast changed, run `nfl-ats publish-predictions`.
3. Run all quality gates and record the result in the final response.
4. The agent refreshes the handoff automatically before a handoff, commit, or push
   to `master`; it must never delegate this command to the user.
5. Check `git status`; never commit ignored data, credentials, or fitted models.
6. Commit or push only when the user explicitly asks, and report the exact branch/hash.
