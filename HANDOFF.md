# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-09-20T16:46:11.645298+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [the lane index](docs/lanes/README.md), and the task's lane.
   Read only the recommended execution order in [ROADMAP.md](ROADMAP.md)
   when choosing new work; do not read the full roadmap or README.
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `f11aa360aa8d` — Qualify Sunday Best Pick and correct ranking replay
- Pending change set: 3 paths
  - `M  HANDOFF.md`
  - `M  docs/lanes/README.md`
  - `M  docs/lanes/sunday-readiness-2026-09-20.md`

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `3412097369e4e4ca`
- Method/profile/regressor/alpha/calibration: `market_residual` / `weak_stack` / `ridge` / `10.0` / `none`
- Served-policy baseline (opener-graded probability rule, home-side push applied): **53.36%** on **1,537 games** (`opener_evaluation/20260920T135435Z`)
- Promoted player-arrest policy component (opener-graded): **53.76%** versus **53.36%** on **1,503 games** (+0.399 accuracy points; `probability_positive=0.8562`); the live card applies this after the coach policy, while paired prospective tracking continues
- Secondary close-grade historical classification: **1,093 / 2,091 (52.27%)**
- Linked forecast: **2026 Week 2**, created `2026-09-20T16:08:51.430516+00:00`

The 52.27% figure is the distinct secondary close-grade historical classification, not the raw-model opener baseline, the promoted player-arrest policy evaluation, a game-specific probability, or proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 2** from model `3412097369e4e4ca`, published `2026-09-20T16:35:39.449781+00:00`. It is an early, mutable research preview.

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

1. **Current priority, 2026-09-20: validate confidence and Best Pick together.** One served probability now decides both side and nominee. This integration does not establish that the largest estimate wins more often or is calibrated. Complete the fixed Sunday ranking replay with every pregame candidate, including eventual pushes; then evaluate any probability repair out of season. Preserve the original outcome-conditioned diagnostic and its limitations. State and reproduction: `docs/lanes/confidence-best-pick-unification.md` and `docs/confidence_best_pick_sunday_matched.md`. Do not restore the separate ranker or independent side-flip rules from the historical instructions below.
2. **Keep Sunday's card current before kickoff.** Fresh private odds and the scheduled lineup update are captured. Publish the latest forecast with the provisional nominee explanation, reconcile the paper ledger while preserving locked games, and verify the deployed page. State: `docs/lanes/sunday-readiness-2026-09-20.md` and `docs/lanes/free-odds-sources.md`.
3. **Continue from lane files with bounded reads and non-Astra delegation.** Follow `AGENTS.md`, run the scheduler each session, and keep uncertainty distinct from research closure and the forced-pick decision. The historical priorities below are preserved context, superseded by these current items.

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
