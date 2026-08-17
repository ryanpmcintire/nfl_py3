# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-08-17T20:58:03.413781+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `b8b9770ec1ff` — Enforce confirmation windows in code, then spend three of them
- Pending change set: 1 paths
  - `M  docs/rotation_registry.md`

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `f92d446d0ccb50dd`
- Method/profile/regressor/alpha/calibration: `market_residual` / `player` / `ridge` / `10.0` / `none`
- Historical ATS classification: **1,080 / 2,075 (52.05%)**
- Linked forecast: **2026 Week 1**, created `2026-08-17T20:06:04.851782+00:00`

The 52.05% figure is historical forced-pick ATS classification accuracy, not a
game-specific probability and not proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 1** from model `f92d446d0ccb50dd`, published `2026-08-17T20:37:11.344734+00:00`. It is an early, mutable research preview.

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

1. Maintain the prediction-safety contract and add a regression canary for every production error or newly supported output type.
2. The point-in-time market stack is code-complete: the purchased 2020–2025 snapshot archive is verified and backed up, weekly scheduled captures continue on the free tier, the frozen MKT-06 pilot has taken its one look (direction replicated, no magnitude edge) with `predict-close` wired to the Week Board, and the MKT-04 paper-decision ledger records every published card's picks at publication (`publish-predictions`) and scores them against the close (`clv-ledger`, surfaced on the track-record page). Remaining market items are research questions (MKT-03 diagnostics, MKT-08 timing policy) and the MKT-09 licensing audit.
3. The XLG-04 chain is complete end-to-end: role delivery replicated cross-league for dropbacks and carries (`docs/cfb_role_replication.md`), the departure-vs-temporary-absence prerequisite was measured (only 15.6%/18.7% of qualified holders return the next season; same-season return odds fall to ~10%/7% after four straight missed games), and the ONE predeclared role-continuity family was scored against the XLG-03 benchmark — it did **not** clear: paired accuracy −0.67 points on 8,933 clean-core games (week-blocked [−1.33, +0.01]) with Brier and log-loss resolved worse under both blockings (`docs/cfb_role_features.md`). The market already prices participation disruption. No NFL transfer claim is predeclared from this family and no retuning of it is admitted.
4. XLG-05 therefore has no cleared mechanism to transfer yet; it waits for a family that first clears the CFB benchmark. The remaining CFB-side paths are XLG-06 (rookie/young-player priors) and XLG-07 (availability semantics), plus CFB screens of the distribution work in item 6.
5. Score the active model and any frozen challengers on prospective 2026 outcomes only — now at BOTH grades (opener via the live Tuesday captures, and close), with the opener grade primary per the pool goal; the 2013–2017 and 2014–2017 replication windows are spent, and no new variant of an existing family may be scored on 2018–2025 without a frozen predeclaration that acknowledges the ~130–150-look ledger. New pool-targeted leads from the 2026-08-17 literature sweep (peer-reviewed opener biases: Week-1 playoff-holdover fade, Week-2 anchoring, prior-week recency, low-visibility games moving most) are candidate features for the rotation-registry/stacked-signals pipeline.
6. Model the distribution, not just the mean — with MOD-16's simple scale model now closed at the CFB screen (the pooled residual distribution is already near-correctly calibrated; `docs/margin_variance.md`), the open distribution paths are the joint score/total model (MOD-05) and distributional boosting (MOD-08), each accepted only on held-out distribution calibration; then reliability trait priors (PER-13) and pre-snap penalty discipline (PBP-07) as the first low-dimensional "intangible proxy" screens, run through the CFB benchmark first where the data allows.

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
