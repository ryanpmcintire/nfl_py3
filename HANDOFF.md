# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-09-28T10:49:47.155872+00:00`

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
- Baseline commit: `dd6c140354ca` — Save completed backlog checkpoint and pregame monitoring handoff
- Pending change set: 24 paths
  - ` M AGENTS.md`
  - `M  HANDOFF.md`
  - `M  ROADMAP.md`
  - ` M docs/agent_workflow.md`
  - ` M docs/findings.html`
  - ` M docs/history.html`
  - `A  docs/independent_combination_validation.md`
  - ` M docs/index.html`
  - `M  docs/lanes/README.md`
  - `A  docs/lanes/done/week03-early-results-review.md`
  - `A  docs/lanes/done/week03-pick-change-trace.md`
  - `A  docs/lanes/done/week03-variance-assessment.md`
  - `A  docs/lanes/independent-combination-validation.md`
  - ` M docs/model.html`
  - `A  registry/studies/combined_vs_raw_2026.json`
  - `M  scripts/capture_scheduler.py`
  - `M  src/nfl_ats/cli_commands/prospective.py`
  - `A  src/nfl_ats/independent_validation.py`
  - `M  tests/fixtures/cli_contract.json`
  - `?? docs/lanes/windows-shell-popup.md`
  - ...and 4 more

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `397c9cec7b097082`
- Method/profile/regressor/alpha/calibration: `market_residual` / `weak_stack` / `ridge` / `10.0` / `none`
- Served-policy baseline (opener-graded probability rule, home-side push applied): **53.36%** on **1,537 games** (`opener_evaluation/20260927T134738Z`)
- Promoted player-arrest policy component (opener-graded): **53.76%** versus **53.36%** on **1,503 games** (+0.399 accuracy points; `probability_positive=0.8562`); the live card combines this with the coach component in one fitted calibrated probability, while paired prospective tracking continues
- Secondary close-grade historical classification: **1,103 / 2,107 (52.35%)**
- Linked forecast: **2026 Week 3**, created `2026-09-27T16:10:13.762139+00:00`

The 52.35% figure is the distinct secondary close-grade historical classification, not the raw-model opener baseline, the promoted player-arrest policy evaluation, a game-specific probability, or proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 3** from model `397c9cec7b097082`, published `2026-09-27T16:16:50.502320+00:00`. It is an early, mutable research preview.

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

1. **Independent combination validation is the research priority.** Freeze the comparison before 2026 Week 4 and collect every declared Week 4–18 game. Primary paired Brier, secondary ATS/log loss/reliability; no interim performance selection or retroactive backfill. The historical feature-choice overlap means prior scores are development evidence. Follow `docs/lanes/independent-combination-validation.md` and the declared protocol in `docs/independent_combination_validation.md` before further tuning.
2. **Keep Sunday's card current, preserving locked games.** Continue from `docs/lanes/done/sunday-readiness-2026-09-20.md` and `docs/lanes/free-odds-sources.md`; reconcile the paper ledger and inspect the latest scheduled refresh before changing the forecast. Commit and push completed work at verified clear stopping points under the owner's standing authorization.
3. **Confidence calibration: bounded repair measured 2026-09-20.** The complete candidate replay and two predeclared out-of-season temperature repairs are saved in `docs/confidence_best_pick_sunday_matched.md` and `docs/confidence_top_calibration.md`. Chronological nominee Brier improvement +0.001040 [-0.027412,+0.030943], probability_positive 0.5227, accompanies unstable fitted temperatures and worse all-game Brier. The served probability is unchanged; 28 new cells remain unresolved, not closed. Acquire more timestamped nominees before another declared calibration comparison. State: `docs/lanes/confidence-best-pick-unification.md`. Do not restore a separate ranker or independent side-flip rules.
4. **Continue from lane files with bounded reads and scoped delegation.** Follow `AGENTS.md` and the applicable procedures in `docs/agent_workflow.md`. Run the scheduler in operational sessions; harness maintenance and read-only work do not trigger operational jobs. Keep uncertainty distinct from research closure and the forced-pick decision. The historical priorities below are preserved context, superseded by these current items.

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
