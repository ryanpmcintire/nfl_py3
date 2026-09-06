# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-09-06T12:36:25.272243+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `bcf339d402cd` — Handoff: turn on the late-week line-move rule before Thursday's refresh (owner order) [skip ci]
- Pending change set: 27 paths
  - `M  HANDOFF.md`
  - `M  ROADMAP.md`
  - `A  docs/design/mockups/README.md`
  - `A  docs/design/mockups/ball-card-data.js`
  - `A  docs/design/mockups/ball-experience.css`
  - `A  docs/design/mockups/ball-experience.js`
  - `A  docs/design/mockups/ball-findings.html`
  - `A  docs/design/mockups/ball-history.html`
  - `A  docs/design/mockups/ball-model.html`
  - `A  docs/design/mockups/ball-motion.css`
  - `A  docs/design/mockups/ball-motion.js`
  - `A  docs/design/mockups/ball-week.html`
  - `A  docs/design/mockups/command-center-matchup.html`
  - `A  docs/design/mockups/command-center-merged.html`
  - `A  docs/design/mockups/command-center-refined.html`
  - `A  docs/design/mockups/command-center.html`
  - `M  docs/findings.html`
  - `M  docs/history.html`
  - `M  docs/index.html`
  - `M  docs/model.html`
  - ...and 7 more

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `ab29832a4e099766`
- Method/profile/regressor/alpha/calibration: `market_residual` / `weak_stack` / `ridge` / `10.0` / `none`
- Raw-model baseline (opener-graded probability rule): **53.36%** on **1,537 games** (`opener_evaluation/20260905T194919Z`)
- Promoted player-arrest policy component (opener-graded): **53.76%** versus **53.36%** on **1,503 games** (+0.399 accuracy points; `probability_positive=0.8562`); the live card applies this after the coach policy, while paired prospective tracking continues
- Secondary close-grade historical classification: **1,087 / 2,075 (52.39%)**
- Linked forecast: **2026 Week 1**, created `2026-09-05T14:14:53.613676+00:00`

The 52.39% figure is the distinct secondary close-grade historical classification, not the raw-model opener baseline, the promoted player-arrest policy evaluation, a game-specific probability, or proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 1** from model `ab29832a4e099766`, published `2026-09-05T21:21:28.750323+00:00`. It is an early, mutable research preview.

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

1. **DO THIS FIRST (owner order, 2026-09-05): turn on the late-week line-move rule before the Thursday 2026-09-10 refresh so it is in the picks the owner submits for Week 1.** The rule: if the spread moves at least half a point against our Tuesday pick between Wednesday and the pick deadline, switch to the other side. Measured on top of the picks we actually play, 2023-2025 (799 games): +1.752 accuracy points, week-blocked 95% [-0.868, +4.375], `probability_positive` 0.899, positive in every season (+1.13 / +1.13 / +3.00), 143 of 799 picks switched. It is wired as the paired challenger `late_week_move_follow_refresh_v1` (src/nfl_ats/late_week_move_follow_refresh_overlay.py, commit 64b39fc); promote that module's decision to the served refresh pick in `refresh-picks`, keep recording both sides, republish the card, and report the Week 1 games it switches. Do not re-open the decision: the owner's rule is that a 0.90 marginal on the played card is played. The other two EV-positive constructs stay as paired challengers for now: trade-deadline drag (+0.877, P+ 0.986) cannot fire before November, and expected lineup loss (+0.658, P+ 0.665) is smaller.
2. On Monday 2026-09-08 run the real lock as `weekly-run --record-decisions`; do not create the genuine Week 1 rows early. The chain now refits and activates a new model id, then runs `opener-evaluation` and `overlay-composition` (about 17 minutes together; skipped only when the model id is unchanged) and ends with `publish-board` (85d2e79). Read the per-recorder result JSON, run `scripts/lockday_verify.py` against the real rows, confirm the card's injury sentence now says injuries informed the picks, screenshot-check the board, push.
3. Standing per-session contract (AGENTS.md): one visible dashboard improvement, `publish-board`, push; Codex lanes (gpt-6-astra) instead of Claude agents until the owner says otherwise; never trigger GitHub Actions.
4. **Completed 2026-09-02:** the six Tuesday recorders are automatic, `crew_tilt_refresh_v1` is on the late-refresh path, and the static lock-day rehearsal (`scripts/lockday_rehearsal.py`, now 39 active paths, 0 errors) imports no model stack and touches no ledger.

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
