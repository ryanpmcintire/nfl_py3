# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-09-05T13:32:21.547646+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `f589e0b4af3a` — Regenerate the public site; make dashboard improvement a binding per-session rule [skip ci]
- Pending change set: 48 paths
  - `M  CURRENT_PREDICTIONS.md`
  - `M  README.md`
  - `M  ROADMAP.md`
  - `M  docs/findings.html`
  - `M  docs/history.html`
  - `M  docs/index.html`
  - `M  docs/model.html`
  - `M  docs/rotation_registry.md`
  - `A  docs/weather_venue_leads.md`
  - `A  registry/experiments/margin-backtest/20260905T124253Z.json`
  - `A  registry/experiments/margin-backtest/20260905T131238Z.json`
  - `A  registry/experiments/opener-evaluation/20260905T125016Z.json`
  - `A  registry/experiments/opener-evaluation/20260905T131344Z.json`
  - `A  registry/experiments/opener-evaluation/20260905T131519Z.json`
  - `A  registry/experiments/weather-venue-flags-on-production/20260905T130709Z.json`
  - `A  registry/experiments/weather-venue-flags-on-production/20260905T131023Z.json`
  - `A  registry/experiments/weather-venue-flags-on-production/20260905T131348Z.json`
  - `A  registry/experiments/weather-venue-flags-on-production/20260905T131431Z.json`
  - `A  registry/experiments/weather-venue-flags-on-production/20260905T131748Z.json`
  - `A  registry/experiments/weather-venue-flags-on-production/20260905T132107Z.json`
  - ...and 28 more

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `123d60be8c80a35d`
- Method/profile/regressor/alpha/calibration: `market_residual` / `weak_stack` / `ridge` / `10.0` / `none`
- Raw-model baseline (opener-graded probability rule): **53.69%** on **1,537 games** (`opener_evaluation/20260905T131519Z`)
- Promoted player-arrest policy component (opener-graded): **53.76%** versus **53.36%** on **1,503 games** (+0.399 accuracy points; `probability_positive=0.8562`); the live card applies this after the coach policy, while paired prospective tracking continues
- Secondary close-grade historical classification: **1,084 / 2,075 (52.24%)**
- Linked forecast: **2026 Week 1**, created `2026-09-03T14:32:53.143515+00:00`

The 52.24% figure is the distinct secondary close-grade historical classification, not the raw-model opener baseline, the promoted player-arrest policy evaluation, a game-specific probability, or proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 1** from model `123d60be8c80a35d`, published `2026-09-05T13:29:33.446356+00:00`. It is an early, mutable research preview.

## Local reproducibility inventory

- canonical team features: **present** (`data/processed/game_features.parquet`)
- play-by-play features: **present** (`data/processed/game_features_pbp.parquet`)
- player features: **present** (`data/processed/game_features_player.parquet`)
- player-value research features: **present** (`data/processed/game_features_player_value.parquet`)
- participation source snapshot: **present** (`data/players/participation/raw/20260813T131635Z/manifest.json`)
- participation-rating research features: **present** (`data/processed/game_features_player_participation.parquet`)
- learned-availability research features: **present** (`data/processed/game_features_player_learned_availability.parquet`)
- frozen player-model selection: **present** (`artifacts/player_model_selection/20260813T124809Z/metadata.json`)
- participation-rating experiment: **present** (`artifacts/participation_experiments/20260813T132030Z/metadata.json`)
- learned-availability experiment: **present** (`artifacts/availability_experiments/20260813T133345Z/metadata.json`)
- active model manifest: **present** (`artifacts/active_ats_model.json`)

Raw data, processed features, fitted models, and evaluation artifacts are intentionally
ignored by Git. A fresh clone therefore starts with documentation, source, tests, and
the last published Markdown forecast but must rebuild or transfer local artifacts.

## Highest-priority work

1. **Completed 2026-09-02:** the six Tuesday recorders are automatic, `crew_tilt_refresh_v1` is on the late-refresh path, and the verifier covers all 29 active challengers with zero pending wiring. The default `scripts/lockday_rehearsal.py` is now a static-only wiring audit: measured over ten consecutive runs at 2.259-4.079 ms, with 23 publish paths, five refresh paths, one weekly-run path, and zero errors. It imports no model stack, executes no recorder, and touches no ledger. The old production-sized replay is explicit `--full-replay` diagnostics only.
2. On 2026-09-08, run the real lock as `weekly-run --record-decisions`; do not create the genuine Week 1 rows early. Read the command's per-recorder result JSON and immediately run `scripts/lockday_verify.py` against the real rows.

The roadmap is authoritative. Negative results remain part of the evidence base and
must not be silently removed or retuned away.

## Commands that matter

```powershell
# Manual diagnostic/recovery only; the agent and Git hooks own normal refreshes
.\.tools\uv.exe run nfl-ats handoff --check

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
