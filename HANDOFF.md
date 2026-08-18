# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-08-18T15:25:21.165994+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `a4e6737305c7` — Derive the findings page's headline numbers instead of typing them in
- Pending change set: 71 paths
  - ` M AGENTS.md`
  - ` M HANDOFF.md`
  - ` M ROADMAP.md`
  - ` M docs/ops_runbook.md`
  - ` M docs/pool_edge_plan.md`
  - ` M docs/prospective_evidence.md`
  - ` M docs/week1_readiness.md`
  - ` M registry/weak_signals.json`
  - ` M src/nfl_ats/best_pick.py`
  - ` M src/nfl_ats/cli.py`
  - ` M src/nfl_ats/clv.py`
  - ` M src/nfl_ats/dashboard/app_pages/picks.py`
  - ` M src/nfl_ats/experiments.py`
  - ` M src/nfl_ats/outcomes.py`
  - ` M src/nfl_ats/prospective_scoring.py`
  - ` M src/nfl_ats/publishing.py`
  - ` M src/nfl_ats/reporting.py`
  - ` M src/nfl_ats/weak_signals.py`
  - ` M src/nfl_ats/weekly.py`
  - ` M tests/test_cli.py`
  - ...and 51 more

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `118f31d9a98c815b`
- Method/profile/regressor/alpha/calibration: `market_residual` / `weak_stack` / `ridge` / `10.0` / `none`
- Historical ATS classification: **1,070 / 2,075 (51.57%)**
- Linked forecast: **2026 Week 1**, created `2026-08-18T01:31:39.144065+00:00`

The 52.05% figure is historical forced-pick ATS classification accuracy, not a
game-specific probability and not proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 1** from model `118f31d9a98c815b`, published `2026-08-18T01:37:48.570903+00:00`. It is an early, mutable research preview.

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

1. **Read `docs/revisit_list.md` first.** A 2026-08-18 audit found four defects in the measurement instrument and one in the decision frame, and several terminal verdicts now rest on measurements that may be wrong. **The gating experiment is stated there and must run before any Tier 1 re-run:** does the probability-calibration step attenuate or invert small effects on REAL data, as it demonstrably does on planted ones (`docs/purged_cv.md`)? If yes, every terminal negative in this project is suspect. If no, the list shrinks to the degenerate-bootstrap case. Also binding from that audit: reported intervals are 17-58% too narrow because the block bootstrap never refits (`docs/estimation_variance.md`), the project is **model-limited, not data-limited** (accuracy is flat across a 100x range of training-set size, `docs/scaling_and_transfer.md`), and the empirical-Bayes shrinkage work is **void** (`docs/decision_rule.md`).
2. Maintain the prediction-safety contract and add a regression canary for every production error or newly supported output type.
3. The point-in-time market stack is code-complete: the purchased 2020–2025 snapshot archive is verified and backed up, weekly scheduled captures continue on the free tier, the frozen MKT-06 pilot has taken its one look (direction replicated, no magnitude edge) with `predict-close` wired to the Week Board, and the MKT-04 paper-decision ledger records every published card's picks at publication (`publish-predictions`) and scores them against the close (`clv-ledger`, surfaced on the track-record page). Remaining market items are research questions (MKT-03 diagnostics, MKT-08 timing policy) and the MKT-09 licensing audit.
4. The XLG-04 chain is complete end-to-end: role delivery replicated cross-league for dropbacks and carries (`docs/cfb_role_replication.md`), the departure-vs-temporary-absence prerequisite was measured (only 15.6%/18.7% of qualified holders return the next season; same-season return odds fall to ~10%/7% after four straight missed games), and the ONE predeclared role-continuity family was scored against the XLG-03 benchmark — it did **not** clear: paired accuracy −0.67 points on 8,933 clean-core games (week-blocked [−1.33, +0.01]) with Brier and log-loss resolved worse under both blockings (`docs/cfb_role_features.md`). The market already prices participation disruption. No NFL transfer claim is predeclared from this family and no retuning of it is admitted.
5. XLG-05 therefore has no cleared mechanism to transfer yet; it waits for a family that first clears the CFB benchmark. The remaining CFB-side paths are XLG-06 (rookie/young-player priors) and XLG-07 (availability semantics), plus CFB screens of the distribution work in item 6.
6. Score the active model and any frozen challengers on prospective 2026 outcomes only — now at BOTH grades (opener via the live Tuesday captures, and close), with the opener grade primary per the pool goal. **The machinery for this now exists and is the single most time-critical item in the file** (POL-10, `docs/prospective_evidence.md`): win/loss settles at both grades, the weekly Best Pick persists pre-kickoff, MOD-07 is registered as a challenger, and anti-backdating is enforced at write and again at scoring. Week 1 locks Tuesday 2026-09-08 and an unrecorded season is gone. **The Week 1 ledger-anchoring decision was resolved 2026-08-17:** both prospective rehearsal ledgers were deleted so the first write to the live ledger is the real Tuesday-lock card on 2026-09-08; no manual row insertion is possible, and `is_best_pick` persists in `PAPER_DECISION_COLUMNS` written only when every game of the week is still ahead (`docs/prospective_evidence.md`). The 2013–2017 and 2014–2017 replication windows are spent, and no new variant of an existing family may be scored on 2018–2025 without a frozen predeclaration that acknowledges the ~130–150-look ledger. **The peer-reviewed opener biases are no longer a lead**: three were built and, ablated inside MOD-07 on the already-spent window, contributed +0.22 points at `probability_positive` 0.505, while the published Week-1 holdover figure (35.6%) fails to replicate here (52.5% on 120 games). Do not add more of them.

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
