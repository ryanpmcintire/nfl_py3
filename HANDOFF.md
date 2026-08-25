# Session handoff

This is the durable starting point for a new development session. Git, local files,
and generated artifact manifests remain authoritative; this document is a concise
index, not a substitute for inspecting them.

Handoff schema: `1`

Refreshed at: `2026-08-25T21:05:40.279794+00:00`

## Start here

1. Run `git status --short` and `git log -3 --oneline --decorate`.
2. Read this file, [README.md](README.md), the recommended execution order in
   [ROADMAP.md](ROADMAP.md), and the relevant file under [`docs/`](docs/).
3. Run `.\.tools\uv.exe run nfl-ats doctor` when the local environment exists.
4. Inspect `artifacts/active_ats_model.json` before quoting current model results.
5. Before changing code, state the verified current condition and intended next work.

## Commit context before this refresh

- Branch: `master`
- Baseline commit: `628dcff6c7cb` — Merge session/lockday-scheduler-dashboard: NFL.com gate fix, capture scheduler, dashboard readability
- Pending change set: 2 paths
  - `M  HANDOFF.md`
  - `?? scripts/lockday_rehearsal.py`

The baseline commit and pending paths were observed before the automatic refresh.
They normally describe the parent and contents of the handoff-bearing commit. Always
trust live Git output after checkout.

## Current model evidence

- Status: **SYNCHRONIZED**; linked artifacts present: **true**
- Model ID: `d1f07d773475dc58`
- Method/profile/regressor/alpha/calibration: `market_residual` / `weak_stack` / `ridge` / `10.0` / `none`
- Raw-model baseline (opener-graded probability rule): **53.36%** on **1,537 games** (`opener_evaluation/20260819T174244Z`)
- Promoted player-arrest policy component (opener-graded): **53.76%** versus **53.36%** on **1,503 games** (+0.399 accuracy points; `probability_positive=0.8562`); the live card applies this after the coach policy, while paired prospective tracking continues
- Secondary close-grade historical classification: **1,081 / 2,075 (52.10%)**
- Linked forecast: **2026 Week 1**, created `2026-08-24T12:07:25.489500+00:00`

The 52.10% figure is the distinct secondary close-grade historical classification, not the raw-model opener baseline, the promoted player-arrest policy evaluation, a game-specific probability, or proof of a profitable or stable market edge.

## Last tracked weekly publication

[CURRENT_PREDICTIONS.md](CURRENT_PREDICTIONS.md) contains **2026 Week 1** from model `d1f07d773475dc58`, published `2026-08-24T12:03:27.814589+00:00`. It is an early, mutable research preview.

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

1. **Read `docs/revisit_list.md` first.** A 2026-08-18 audit found four defects in the measurement instrument and one in the decision frame, and several terminal verdicts now rest on measurements that may be wrong. ~~**The gating experiment is stated there and must run before any Tier 1 re-run:** does the probability-calibration step attenuate or invert small effects on REAL data, as it demonstrably does on planted ones (`docs/purged_cv.md`)? If yes, every terminal negative in this project is suspect. If no, the list shrinks to the degenerate-bootstrap case.~~ — **RESOLVED 2026-08-18** (`docs/calibration_distortion.md` §8, `docs/revisit_list.md`): the gating experiment ran. **D1 is a planting artifact, not a real defect** — the recorded 51.3%/53.0% readout was one random seed reported as two independent findings; replicated across 21 seeds, the calibration step is worth at most **~0.35 accuracy points**, not the claimed 2.0-point swing, and does not invert any effect. Tier 1 shrank to the D4 degenerate-bootstrap case (`player_qb_continuity_matched_alpha`) plus the two bare-verdict entries (`pbp_drive_bundle`, `player_qb_continuity`) — D1 does not put every terminal negative in the project under suspicion. ~~Also binding from that audit: reported intervals are 17-58% too narrow because the block bootstrap never refits (`docs/estimation_variance.md`)~~ — **RETRACTED 2026-08-18** (`docs/estimation_variance.md` Part II): that figure double-counted a training-by-game interaction the game bootstrap already carries. The honest refit factor is **1.003x, one-sided 95% upper bound 1.099x**. The real defect was **D4 (too few blocks)**, not D2: measured coverage by block count is 0.000 at k=1, 0.466 at k=2, 0.760 at k=4, 0.896 at k=10, 0.944 at k=50 (nominal 0.95), so `MIN_BLOCKS_FOR_INTERVAL = 10`. Still binding from the original audit: the project is **model-limited, not data-limited** (accuracy is flat across a 100x range of training-set size, `docs/scaling_and_transfer.md`), and the empirical-Bayes shrinkage work is **void** (`docs/decision_rule.md`).
2. Maintain the prediction-safety contract and add a regression canary for every production error or newly supported output type.
3. The point-in-time market stack is code-complete: the purchased 2020–2025 snapshot archive is verified and backed up, weekly scheduled captures continue on the free tier, the frozen MKT-06 pilot has taken its one look (direction replicated, no magnitude edge) with `predict-close` wired to the Week Board, and the MKT-04 paper-decision ledger records every published card's picks at publication (`publish-predictions`) and scores them against the close (`clv-ledger`, surfaced on the track-record page). Remaining market items are research questions (MKT-03 diagnostics, MKT-08 timing policy) and the MKT-09 licensing audit. **MKT-03 update, 2026-08-18** (`docs/novig_diagnostics.md`): the diagnostic itself has now run, read-only against the existing archive, no rotation-registry window spent and nothing fed into any model. The dropped spread price is informative (55.50% of 438,424 quotes are not exactly -110), but the resulting no-vig probability is calibrated within noise at the Tuesday opener for the ATS arm (both buckets cross zero at both blockings) and for four of five moneyline buckets; one moneyline bucket's season-blocked interval excludes zero, reported as continuous evidence, not a finding (secondary-goal-only, no multiplicity correction across the seven buckets read). Still a diagnostic, not a candidate feature; consuming any of it inside a model requires its own predeclared look.
4. The XLG-04 chain is complete end-to-end: role delivery replicated cross-league for dropbacks and carries (`docs/cfb_role_replication.md`), the departure-vs-temporary-absence prerequisite was measured (only 15.6%/18.7% of qualified holders return the next season; same-season return odds fall to ~10%/7% after four straight missed games), and the ONE predeclared role-continuity family was scored against the XLG-03 benchmark: paired accuracy −0.67 points on 8,933 clean-core games (week-blocked [−1.33, +0.01]) with Brier and log-loss worse under both blockings (`docs/cfb_role_features.md`). ~~It did **not** clear... The market already prices participation disruption. No NFL transfer claim is predeclared from this family and no retuning of it is admitted.~~ **REOPENED 2026-08-18**: −0.67 points sits below the instrument's own MDE80 of 0.927 points (f=9.96%, n=9,093), and trait split-half reliability (0.719 dropback / 0.680 carry) rules out a refuted mechanism, so neither admissible closing ground was ever met. Reclassified `unresolved_below_power` (`registry/weak_signals.json`). The market is not shown to price participation disruption; that was never established. No NFL transfer claim is predeclared from this family yet, and no retuning of the spent CFB window is admitted, but the family itself is open again.
5. **XLG-05 therefore has a mechanism to transfer again.** ~~XLG-05 therefore has no cleared mechanism to transfer yet; it waits for a family that first clears the CFB benchmark.~~ The role-continuity family's closure — the reason XLG-05 was waiting — is retracted (item 4 above), so the XLG-04 → XLG-05 transfer path reopens. This is not itself a green light to spend an NFL window: the CFB-side evidence is `unresolved_below_power`, not confirmed, so any XLG-05 predeclaration must say so. The remaining CFB-side paths are XLG-06 (rookie/young-player priors) and XLG-07 (availability semantics), plus CFB screens of the distribution work in item 6.
6. Score the active model and any frozen challengers on prospective 2026 outcomes only — now at BOTH grades (opener via the live Tuesday captures, and close), with the opener grade primary per the pool goal. **The machinery for this now exists and is the single most time-critical item in the file** (POL-10, `docs/prospective_evidence.md`): win/loss settles at both grades, the weekly Best Pick persists pre-kickoff, MOD-07 is registered as a challenger, and anti-backdating is enforced at write and again at scoring. Week 1 locks Tuesday 2026-09-08 and an unrecorded season is gone. ~~**The Week 1 ledger-anchoring decision was resolved 2026-08-17:** both prospective rehearsal ledgers were deleted so the first write to the live ledger is the real Tuesday-lock card on 2026-09-08; no manual row insertion is possible.~~ **That resolution did not hold.** The ordinary, documented `publish-predictions` command repopulated the live ledger within hours, because recording was opt-out at the time: 16 real 2026-Week-1 rows landed at `recorded_at_utc` 2026-08-18T01:24:56Z (`model_id` `4b01f055b684e27e`, `is_best_pick=True` on `2026_01_ARI_LAC`). **Fixed 2026-08-18:** recording is now opt-in (`--record-decisions`, default `False`) and separately refused whenever a week's earliest kickoff is more than `RECORDING_LOCK_WINDOW` (7 days) from the recording instant, so the same command cannot silently repopulate the ledger a third time (`docs/prospective_evidence.md`, "Known divergence"). Disposition of those 16 rows — reset again vs. accept as a rehearsal artifact — was left to the owner (`docs/week1_readiness.md` item 2), and a live check on 2026-08-18 found `artifacts/clv_ledger/decisions.parquet` **absent from the repo entirely** — matching neither documented option and not matching the 16-row state this section and `docs/week1_readiness.md` still describe. A backup of the 16 rows was located (outside the repo, from a prior session) and verified against this description exactly; nothing was deleted, since the file the deletion was supposed to act on was already gone by the time of the check. ~~Whether a reset was already executed somewhere or the local artifact was simply lost needs confirming before Week 1's ledger status can be called resolved either way — see `docs/week1_readiness.md` item 2 for the live finding.~~ **RESOLVED 2026-08-18** (`docs/week1_readiness.md` item 2): a follow-up read-only check re-confirmed `artifacts/clv_ledger/decisions.parquet` is still absent — zero old-model rows. The *cause* of the absence (an executed reset vs. a lost local artifact) was never determined and is not claimed to be; the item is resolved because the end-state matches the owner's 2026-08-18 reset decision regardless of which cause produced it: zero contaminating rows, the promoted `weak_stack` model free to write Week 1 fresh, the 16-row backup preserved outside the repo, and refill guarded by opt-in recording plus the 7-day `RECORDING_LOCK_WINDOW`. **Live consequence: the first genuine write to the primary ledger is now the Sep 8 lock-day `weekly-run`/`publish-predictions` run, and only if it is invoked with `--record-decisions`** — the flag is opt-in, so omitting it publishes the card but records nothing to either ledger. `is_best_pick` persists in `PAPER_DECISION_COLUMNS` written only when every game of the week is still ahead (`docs/prospective_evidence.md`). The 2013–2017 and 2014–2017 replication windows are spent, and no new variant of an existing family may be scored on 2018–2025 without a frozen predeclaration that acknowledges the ~130–150-look ledger. **The peer-reviewed opener biases are no longer a lead**: three were built and, ablated inside MOD-07 on the already-spent window, contributed +0.22 points at `probability_positive` 0.505, while the published Week-1 holdover figure (35.6%) fails to replicate here (52.5% on 120 games). Do not add more of them. **Evening update, 2026-08-18:** two more plays are now LIVE on the real, published card (`CURRENT_PREDICTIONS.md`), both dual-tracked via the challenger ledger so neither spends a rotation-registry window. (a) The clean-case year-1-head-coach fade, weeks 1-8 only (`docs/coach_fade_overlay.md`, `OVERLAY_ENABLED = True`, challenger `hc_year_one_fade_overlay`): the real Week 1 2026 card shows exactly one flip, `2026_01_BAL_IND` (BAL, year-1, at IND, kept coach) from BAL -3.5 to IND +3.5, with `2026_01_MIA_LV` correctly flagged but not flipped (both coaches year-1). (b) The Best-Pick nomination v2 rule (`docs/best_pick_ranker.md` § "2026-08-18: the weekly NOMINATION rule switches", `NOMINATION_V2_ENABLED = True`, challenger `best_pick_nomination_v2`): nominates by calibrated probability among low-disagreement games rather than `sweep_robustness`'s alphabetical tie-break. Re-verified live this session (`scripts/best_pick_nomination_dry_run.py` against the active model's real Week 1 forecast): v2 nominates `2026_01_MIA_LV`, no tie, while the incumbent (`sweep_robustness`, itself a two-way tie) nominates `2026_01_ARI_LAC` — the two rules disagree on which game gets the ★ this week, and the published card now shows v2's pick, matching the `d991c65` republish. `registry/weak_signals.json` now holds **107** recorded signals (verified count, `nfl-ats weak-signals status`), including ranked open leads worth a future look: division revenge (+0.19 accuracy points, `probability_positive` 0.88, `bias_battery_division_revenge_game`); a CFB rivalry-finale proxy whose interval sits entirely negative (`probability_positive` 0.0) — resolved-*shaped* but still recorded `unresolved_below_power` because it is one of 19 mined, uncorrected battery cells (`cfb_bias_battery_rivalry_finale_proxy`); and a penalty-only variant of the weak-signal stack, tracked but not actionable (+0.13 points, `probability_positive` 0.69, `weak_stack_v2_penalty_only`). New screens no longer need hand-transcription into the registry: `nfl-ats experiment run <spec.json>` (`docs/experiment_pipeline.md`) runs the whole reliability-check/screen/bootstrap/classification/record/provenance loop from a declarative spec, built after this session's own recorders caught a 100x scaling bug, a sign bug, and a corrupted source path, all hand-copied from console output. Two more comparisons were run head-to-head against their incumbents and settled this session — neither should be re-derived: the ridge-alpha swap (10.0 to 2,000.0 on the active `weak_stack`/`market_residual` config) was **refused** on EV at the opener grade (`ridge_alpha_2000_nfl_opener_confirmation`, -1.397 points, `probability_positive` 0.0504, ~95% against; the resolved calibration gain routes to Best-Pick/calibration consumers instead, production `ridge_alpha` stays 10.0), and the CFB per-metric offseason-retention feature vector is **closed** `refuted_mechanism`/`wrong_sign_resolved` (`offseason_retention_per_metric_cfb`, -0.739 points, `probability_positive` 0.0037, loses to the uniform 0.67 scalar RWB-01 already ships). **With the ledger fix verified and both overlays live, the single most time-critical action left in this file is unchanged from the top of this item: the Sep 8 lock-day `weekly-run`/`publish-predictions` run must pass `--record-decisions`, or the season's first genuine ledger write — and every challenger's first prospective evidence — silently never happens.** **2026-08-20, later the same session: the challenger count above is stale.** 18 ACTIVE_PROSPECTIVE challengers (21 entries total; see POL-10) must be adjudicated at that same Sep 8 run, not the smaller count this paragraph originally described. One owner action remains, none blocking Sep 8: register the two weekly public-betting capture tasks, Saturday and Sunday noon ET (`docs/public_betting_sourcing.md` §9, exact `schtasks` commands at MKT-12). The GDELT **volume** path is now complete and processed for 32/32 teams and all 37 relocation-era aliases (`docs/gdelt_backfill.md`): the frozen close-grade `attention_battery_both_cold` replication was rerun on 2,038 eligible games and now leans against its sign, -0.2742 accuracy points, `probability_positive=0.23225`, still `unresolved_below_power` with no admissible closing ground. Tone remains rate-limit-blocked at 2/32 teams after BAL/BUF each exhausted eight HTTP 429 retries, but tone is not needed for that completed volume replication; resume it only for a separately predeclared sentiment question. The PFR per-article date fetch is now complete at 4,361/4,361 targeted rows (PER-03). The Sagarin Era B snapshot is also consolidated at 585 parser-valid pages, 18,473 rating rows, and 9,848 Tuesday as-of rows; seven Wayback fetch gaps remain documented, and the prior ATS screen was not rerun (MKT-11).

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
