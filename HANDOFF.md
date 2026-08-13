# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-08-13T13:06:24.286536+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `d8a7ee450b05` — Add player availability research pipeline
- Pending change set: 20 paths
  - `M  HANDOFF.md`
  - `M  README.md`
  - `M  ROADMAP.md`
  - `M  docs/architecture.md`
  - `M  docs/data_feasibility.md`
  - `M  docs/modeling.md`
  - `M  src/nfl_ats/active_model.py`
  - `A  src/nfl_ats/calibration.py`
  - `M  src/nfl_ats/cli.py`
  - `M  src/nfl_ats/dashboard.py`
  - `M  src/nfl_ats/experiments.py`
  - `M  src/nfl_ats/handoff.py`
  - `M  src/nfl_ats/margin.py`
  - `M  src/nfl_ats/outcomes.py`
  - `M  src/nfl_ats/prediction_safety.py`
  - `M  tests/test_active_model.py`
  - `A  tests/test_calibration.py`
  - `M  tests/test_dashboard.py`
  - `M  tests/test_experiments.py`
  - `M  tests/test_margin.py`

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `be9326573294de5a`
- Method/profile/regressor/alpha/calibration: `market_residual` / `player` / `ridge` / `10.0` / `none`
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
- player-value research features: **present** (`data/processed/game_features_player_value.parquet`)
- frozen player-model selection: **present** (`artifacts/player_model_selection/20260813T124809Z/metadata.json`)
- active model manifest: **present** (`artifacts/active_ats_model.json`)

Raw data, processed features, fitted models, and evaluation artifacts are intentionally
ignored by Git. A fresh clone therefore starts with documentation, source, tests, and
the last published Markdown forecast but must rebuild or transfer local artifacts.

## Highest-priority work

1. Maintain the prediction-safety contract and add a regression canary for every production error or newly supported output type.
2. Use 2016–2025 participation to estimate aggressively shrunk player/unit effects and test whether they improve the injury-value layer.
3. Predeclare a low-variance follow-up using the completed gate's fixed leads; do not describe another score on 2018–2025 as independent confirmation.
4. Add joint score/total distributions and compare calibration methods inside the nested protocol.
5. Use 2016–2025 participation/NGS for position-unit and formation effects; individual receiver-corner pairs remain too sparse for an initial model.
6. Continue collecting timestamped, book-specific opening/current/closing quotes. The one-season free sample validates plumbing but cannot validate a historical line-movement edge.

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
