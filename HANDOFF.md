# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-09-13T01:13:18.484010+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `d57ac7eac6b9` — Session context costs a fifth of what it did: pool summary by default, brief scheduler status, rules-only AGENTS.md, ROADMAP and README history archived
- Pending change set: 28 paths
  - `M  AGENTS.md`
  - `M  HANDOFF.md`
  - `M  README.md`
  - `M  ROADMAP.md`
  - `M  docs/findings.html`
  - `M  docs/history.html`
  - `M  docs/index.html`
  - `A  docs/injury_headlines.md`
  - `A  docs/lanes/README.md`
  - `A  docs/lanes/done/dashboard-2026-09-12-evening.md`
  - `A  docs/lanes/done/halves-ledger-lockday.md`
  - `A  docs/lanes/lead59-archive-battery.md`
  - `A  docs/lanes/lead64-headline-parser.md`
  - `A  docs/lanes/token-diet.md`
  - `M  docs/model.html`
  - `M  docs/officials_archive_battery.md`
  - `M  registry/weak_signals.json`
  - `M  scripts/lockday_verify.py`
  - `A  scripts/officials_archive_battery_eval.py`
  - `M  src/nfl_ats/board_content.py`
  - ...and 8 more

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `1b9bfbddc10a39ef`
- Method/profile/regressor/alpha/calibration: `market_residual` / `weak_stack` / `ridge` / `10.0` / `none`
- Served-policy baseline (opener-graded probability rule, home-side push applied): **54.49%** on **1,537 games** (`opener_evaluation/20260912T140227Z`)
- Promoted player-arrest policy component (opener-graded): **53.76%** versus **53.36%** on **1,503 games** (+0.399 accuracy points; `probability_positive=0.8562`); the live card applies this after the coach policy, while paired prospective tracking continues
- Secondary close-grade historical classification: **1,085 / 2,075 (52.29%)**
- Linked forecast: **2026 Week 1**, created `2026-09-12T17:39:30.848478+00:00`

The 52.29% figure is the distinct secondary close-grade historical classification, not the raw-model opener baseline, the promoted player-arrest policy evaluation, a game-specific probability, or proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 1** from model `1b9bfbddc10a39ef`, published `2026-09-12T17:48:35.057548+00:00`. It is an early, mutable research preview.

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

1. **DO THIS FIRST (owner order, 2026-09-05): turn on the late-week line-move rule before the Thursday 2026-09-10 refresh so it is in the picks the owner submits for Week 1.** The rule: if the spread moves at least half a point against our Tuesday pick between Wednesday and the pick deadline, switch to the other side. Measured on top of the picks we actually play, 2023-2025 (799 games): +1.752 accuracy points, week-blocked 95% [-0.868, +4.375], `probability_positive` 0.899, positive in every season (+1.13 / +1.13 / +3.00), 143 of 799 picks switched. It is wired as the paired challenger `late_week_move_follow_refresh_v1` (src/nfl_ats/late_week_move_follow_refresh_overlay.py, commit 64b39fc); promote that module's decision to the served refresh pick in `refresh-picks`, keep recording both sides, republish the card, and report the Week 1 games it switches. Do not re-open the decision: the owner's rule is that a 0.90 marginal on the played card is played. The other two EV-positive constructs stay as paired challengers for now: trade-deadline drag (+0.877, P+ 0.986) cannot fire before November, and expected lineup loss (+0.658, P+ 0.665) is smaller. **2026-09-06: promoted in code (measured).** The served `refresh-picks` pick now follows the module's 0.5-point equal-book Wednesday-to-deadline decision with precedence over the 1.0-point consensus rule (`LATE_WEEK_MOVE_FOLLOW_POLICY`, shared `late_week_follow_frame` so served and challenger agree by construction); every revision row keeps both arms' evidence plus the model-only counterfactual (pick-revision ledger 33 -> 38 columns), the paired challenger recorder keeps running unchanged, and the board's Season-ops note now reads the half-point rule. Pinned by 6 new `test_pick_refresh.py` cases (served override, sub-threshold stand, precedence, served/challenger parity, summary+ledger+card, fail-open) with the full pre-existing refresh/tilt/contract suites still green. No Week 1 game to report yet: the Tuesday ledger the rule switches from is first written at the Tuesday 2026-09-08 lock (2026-09-08 is a Tuesday; earlier text said Monday), so the first live fire is the NEW Wednesday 2026-09-09 6:15 PM ET pass (`refresh_wed`, below), then Thursday's. **2026-09-07 (measured): the rule would not have fired at all as scheduled.** `data/scheduler_log.txt` shows every in-season refresh job (`refresh_sun`, `refresh_sun_inactives_early/late`, 2026-09-06) dying on `usage: nfl-ats refresh-picks [-h] --season SEASON --week WEEK` -- the parser required a pair the schedule never passed and nothing had ever parsed the jobs' argv. Fixed: `refresh-picks` now defaults `--season`/ `--week` to the active model's linked forecast (`cli_common._add_active_forecast_season_week_args`, `active_model.active_forecast_season_week`), and `tests/test_capture_scheduler.py` parses every scheduled `nfl-ats` argv against the real parser. Second gap, same day: 2026 Week 1 opens on a WEDNESDAY (`2026_01_NE_SEA`, 2026-09-09 20:20 ET, schedules snapshot 20260905T211016Z) and the schedule had no post-Tuesday odds capture and no refresh pass before that kickoff; added `odds_wed_opener` (Wed 18:00 ET) and `refresh_wed` (Wed 18:15 ET, closes 19:45), both backfill-guarded. The scheduler daemon was restarted on the new code (44 enabled jobs). **2026-09-09 (measured): the served arm is now the LEADING BOOKS' MEDIAN, not the equal-book mean.** Predeclared in `docs/sharp_weighted_follow.md` before any arm was computed and measured in `artifacts/sharp_weighted_follow/20260909T233606Z/`: on the frozen 2026-09-05 baseline, 2023-2025, 799 opener-graded games, the leader-median arm (S1) scores +3.004 accuracy points [-0.993, +6.953] `probability_positive` 0.930 against the equal-book arm's (S4) +1.752 [-0.870, +4.326] P+ 0.907; head to head S1 - S4 = +1.252 [-0.990, +3.522] P+ 0.864 week-blocked, [0.000, +2.256] P+ 0.982 season-blocked. Mechanism: S1's fire set almost contains S4's (325 both, same side on all 325; 4 equal-only) plus 198 games where the leaders cleared 0.5 and the twelve-book mean was diluted -- the Tuesday card is 47.94% right on those and following the leaders makes it 52.06%; a fire-count-matched loose equal gate does not recover it (S1 +1.377, P+ 0.961). Served as `LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY` / `late_week_leader_median_follow_0_5`; `late_week_follow_frame` now returns BOTH arms so the equal-book rule keeps recording as the paired OFF challenger under its existing `late_week_move_follow_refresh_v1` id, with the served arm registered as `late_week_leader_median_follow_v1`. The three `late_week_*` pick-revision columns now carry the leader-median move, side and leading-book count. Proven against the real data root (16 games with exposure, 1 followed: `2026_01_MIA_LV`, leader median -0.5 on 3 leading books) and, on a scratch artifacts copy, `--record-decisions` wrote the revision row with `movement_policy=late_week_leader_median_follow_0_5`.
2. On Tuesday 2026-09-08 run the real lock as `weekly-run --record-decisions`; do not create the genuine Week 1 rows early. The chain now refits and activates a new model id, then runs `opener-evaluation` and `overlay-composition` (about 17 minutes together; skipped only when the model id is unchanged) and ends with `publish-board` (85d2e79). Read the per-recorder result JSON, run `scripts/lockday_verify.py` against the real rows, screenshot-check the board, push. **Injury sentence expectation, corrected 2026-09-07 (measured):** the newest player snapshot (`20260905T123614Z`) has zero 2026 injury rows because the league's first Week 1 report is published Wednesday, so the lock's card will read "No injury reports had been published yet when these picks were made; they lean on lineups and recent play." -- that is the true state, not a defect. The ENG-39 `injury_feature_presence` check aborted both the Sunday 2026-09-06 and Monday 2026-09-07 daily forecast refreshes at `weekly-run` step 5 on exactly this; it now passes only when `players.injury_reports_absent_reason` proves the week's rows do not exist in the newest snapshot (recorded verbatim as a warning), and still fails closed otherwise (`docs/injury_timestamp_fallback.md`, 2026-09-07 section).
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
