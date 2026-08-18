# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-08-18T01:28:07.364531+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `8c325d467e0f` — Keep below-power signals instead of deleting them; sweep finds half the negatives unearned
- Pending change set: 43 paths
  - `M  .gitignore`
  - `M  AGENTS.md`
  - `M  CURRENT_PREDICTIONS.md`
  - `M  HANDOFF.md`
  - `M  README.md`
  - `M  ROADMAP.md`
  - `A  artifacts/prospective/challengers.json`
  - `A  docs/availability_confirmation.md`
  - `A  docs/ecdf_smoothing.md`
  - `M  docs/findings.html`
  - `A  docs/groupwise_ridge.md`
  - `A  docs/hc_year_one_fade.md`
  - `M  docs/index.html`
  - `A  docs/offseason_retention.md`
  - `M  docs/pool_edge_plan.md`
  - `A  docs/pool_format_levers.md`
  - `M  docs/track_record.html`
  - `A  docs/week1_readiness.md`
  - `M  registry/weak_signals.json`
  - `A  scripts/availability_ablation.py`
  - ...and 23 more

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `4b01f055b684e27e`
- Method/profile/regressor/alpha/calibration: `market_residual` / `player` / `ridge` / `10.0` / `none`
- Historical ATS classification: **1,080 / 2,075 (52.05%)**
- Linked forecast: **2026 Week 1**, created `2026-08-18T00:01:32.841713+00:00`

The 52.05% figure is historical forced-pick ATS classification accuracy, not a
game-specific probability and not proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 1** from model `4b01f055b684e27e`, published `2026-08-18T01:24:56.215555+00:00`. It is an early, mutable research preview.

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
5. Score the active model and any frozen challengers on prospective 2026 outcomes only — now at BOTH grades (opener via the live Tuesday captures, and close), with the opener grade primary per the pool goal. **The machinery for this now exists and is the single most time-critical item in the file** (POL-10, `docs/prospective_evidence.md`): win/loss settles at both grades, the weekly Best Pick persists pre-kickoff, MOD-07 is registered as a challenger, and anti-backdating is enforced at write and again at scoring. Week 1 locks Tuesday 2026-09-08 and an unrecorded season is gone. One decision is open before then: the Week 1 ledger rows anchor on the 2026-08-17 rehearsal publish rather than the Tuesday lock the pool actually grades. The 2013–2017 and 2014–2017 replication windows are spent, and no new variant of an existing family may be scored on 2018–2025 without a frozen predeclaration that acknowledges the ~130–150-look ledger. **The peer-reviewed opener biases are no longer a lead**: three were built and, ablated inside MOD-07 on the already-spent window, contributed +0.22 points at `probability_positive` 0.505, while the published Week-1 holdover figure (35.6%) fails to replicate here (52.5% on 120 games). Do not add more of them.
6. Stop trying to measure team quality better; it is bounded near zero. A deliberate-leak positive control (opponent adjustment fit over all of 2006–2025, so the columns see the future) moved margin MAE by only **+0.0129 points** — a measured ceiling on the whole family, and the common explanation for the PBP/drive bundle, PBP-05, MOD-16 and CFB role continuity all failing separately. Our target is the residual from the market line, and the market already prices team quality (`docs/play_level_audit.md`, `docs/cfb_opponent_adjustment.md`). Prefer work that prices what the market prices BADLY — availability is the only candidate carrying a measured lean (`probability_positive` 0.899 in the MOD-07 ablation) — or that exploits the pool's format rather than the line (POL-04/05, largely unexplored). On distributions specifically: the margin lattice is real and large but the ATS *residual* is already near-Gaussian once the varying spread smears it, so MOD-05 is worth building for pushes and half-point questions rather than ATS accuracy, and MOD-08 has no shape signal left to condition on. The one measured distribution win is **smoothing** rather than conditioning: replacing the 518-draw ECDF costs nothing and buys Brier −0.0015 (P=0.998), but it moves picks and so needs its own predeclared window.

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
