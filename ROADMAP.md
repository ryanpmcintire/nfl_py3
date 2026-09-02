# NFL ATS research roadmap

This is the living backlog for the revived project. It deliberately includes
boring infrastructure, serious research ideas, and moonshots. A feature is not
"done" because it can be computed: it is done when its availability time is
documented, leakage tests pass, and its incremental value is measured in
walk-forward evaluation.

## North star

Build an auditable system that estimates NFL game and cover probabilities,
measures uncertainty honestly, and supports paper decisions for ATS pools and
simulated bankrolls. Success means better calibrated out-of-time forecasts than
strong baselines—not an isolated profitable backtest.

**Primary goal (clarified 2026-08-17): beat the OPENING line the user's
football pool (Splash Sports) grades against.** Splash's pool engine posts
lines Tuesday, revises once Wednesday, then freezes them for the week — so
the Tuesday-opener grade, not the closing-line grade, is the headline
metric. First predeclared measurement (`docs/opener_evaluation.md`), scored
against the then-frozen incumbent `player` model only (`player` was not
promoted yet): **52.50% against openers** on 1,537 paired 2020–2025 games,
vs **51.09% against closes on the same games** — paired delta **+1.35
points**, ~99.9% probability positive, season-blocked interval vs the coin
flip excludes 50%. **These three numbers (52.50 / 51.09 / +1.35) belong to
`player`, not to the model now active.** The active model is `weak_stack`
(promoted MOD-07, commit `68b4dc0`), separately re-measured on the same
1,537-game protocol
(`artifacts/opener_evaluation/20260818T013115Z/metadata.json`): **52.83%
against openers**, season-blocked **[50.98%, 54.83%]** — a further **+0.33
points over `player`'s 52.50%** opener accuracy. `weak_stack`'s own close
accuracy on these games is **51.56%**, not `player`'s 51.09%; the two
models' numbers are not interchangeable. **Instrument note, 2026-08-19:**
the opener evaluation historically graded with the sign rule
(`residual > 0`), but production plays the probability rule
(`home_cover_probability >= 0.5`, `pool.py`); re-graded under the rule
production actually plays, the same model on the same 1,537 games scores
**53.36% against openers** — measured twice independently this session
(error-analysis reproduction and a fresh `opener-evaluation` run,
`docs/opener_evaluation.md` 2026-08-19 addendum). The 52.83% remains the
predeclared protocol figure; the evaluator now reports both rules, and by
owner decision (2026-08-19) the public site's headline leads with the
production-rule grade (53.4%, protocol figure retained as provenance;
all six seasons above the coin flip under that rule, season-blocked
[51.97%, 54.56%]). Closing lines, CLV, and vig are secondary.
The ceiling analysis, gap accounting (52.5% → ~57% theoretical; 54–55%
practical), and the prioritized build queue live in
`docs/pool_edge_plan.md` — start there when resuming this work.

## Evidence gates

Every research addition must clear these gates:

1. **Prediction integrity:** independently recompute and certify every card's
   schema, probability bounds, market math, decision policy, and training cutoff.
2. **Point-in-time validity:** prove when the value was available.
3. **Data sufficiency:** count independent games/weeks and source regimes, not
   merely player rows or plays; high-dimensional work needs enough repeated
   player/opponent observations to survive an outer-season test.
4. **Baseline improvement:** compare against market-only, Elo, and current full
   models using identical test weeks.
5. **Stability:** report season-by-season effects and uncertainty, not one
   pooled number.
6. **Prospective confirmation:** freeze the method before evaluating new games.

## Status legend

- ✅ Done and verified
- 🚧 In progress in the current development slice
- ⬜ Ready or planned
- 🔬 Research question; build only after prerequisites
- 🌙 Moonshot

### Backlog accounting

Run `.\.tools\uv.exe run --no-sync python scripts\roadmap_inventory.py` from
PowerShell for the live count. The inventory deliberately reports two numbers:
`remaining` includes every non-done row, while `active/planned` includes only
🚧 and ⬜ rows. Research questions, moonshots, and declined/blocked work remain
visible but are not misrepresented as the immediately executable queue. Use
`--json` for machine-readable totals and per-phase counts.

### 2026-09-02 inactives Section 5 result

**Measured** by `scripts/inactives_channel_historical_screen.py` and recorded
through both registries: the opener-graded `[2020, 2021]` historical proxy
screen covered 429 paired games across 33 weeks and measured −1.3986 accuracy
points versus the Tuesday card, week-blocked 95% [−3.1963, +0.2427],
`probability_positive=0.0418`. The realised-margin oracle control measured
+45.9207 points, week-blocked 95% [+40.0911, +51.9139], P+ 1.000. The result
is `unresolved_below_power` (weak-signal registry) / `unresolved` (rotation),
not a closure or promotion; the primary interval and control do not establish
an admissible terminal verdict. Artifact:
`artifacts/inactives_channel_historical_proxy/20260902T155950Z/`. The live
2026 prospective inactives challenger remains active and the played card is
unchanged.

### 2026-09-02 four on-production opener confirmations

**Measured** in the four named `artifacts/*_opener_confirmation/20260902T*/results.json` files: every family was predeclared, assigned by the rotation CLI to 2020-2021, evaluated on 456 paired non-push games across 35 weeks, and retained prediction-level pairs. The primary probability-rule opener deltas versus production were Reddit home comment ratio **+0.439 accuracy points, `probability_positive=0.665`**; pace mismatch **-0.219, P+=0.286**; away active illness **-0.439, P+=0.312**; and red-zone third-down fade **-0.658, P+=0.111**. Each frozen-pick null used 200 within-week permutations and each realized-margin positive control measured **+43.860 points, P+ 1.000**. All four are recorded in both registries as `unresolved_below_power` / `unresolved`; no terminal conclusion, production-card change, or publication follows from this one assigned block.

## Phase 0 — trustworthy foundation

| ID | Status | Item | Definition of done |
|---|---|---|---|
| FND-01 | ✅ | Replace HTML scrapers with nflverse | Maintained loader, schema contracts, real-data integration test |
| FND-02 | ✅ | Immutable data snapshots | UTC ID, source metadata, row counts, SHA-256 manifests |
| FND-03 | ✅ | Modern Python project | Python 3.12, uv lock, src layout, CI |
| FND-04 | ✅ | Explicit ATS semantics | Home-oriented result/spread/cover convention and push handling |
| FND-05 | ✅ | Leakage-safe weekly evaluation | Every prediction records a strictly earlier training cutoff |
| FND-06 | ✅ | Generated artifact governance | Data, credentials, models, and output ignored by Git |
| FND-07 | ✅ | Legacy recovery point | Annotated `legacy-2023-w6` Git tag |
| FND-08 | ✅ | Reproducible CLI | Doctor, ingest, features, backtest, predict |
| FND-09 | ✅ | Quality gates | Formatting, lint, strict typing, tests, ≥85% coverage |
| FND-10 | ✅ | Data-contract monitoring | Scheduled source smoke test and actionable upstream-change alert |
| FND-11 | ✅ | Fail-closed prediction safety | Direct ATS and outcome cards pass an independent runtime contract; frozen cards are revalidated on read |
| FND-12 | ✅ | Adversarial prediction canaries | Corrupted probabilities, prices, decisions, cutoffs, method identities, and target outcomes fail tests |
| FND-13 | ✅ | Evaluator performance contract | Phase timings, reference budgets, numerical-equivalence test, and a structural bootstrap regression canary |
| FND-14 | ✅ | Durable session handoff | Auto-loaded guidance, tracked Git/model/forecast status, automatic commit refresh, master-push guard, and CI semantic check |
| FND-15 | 🚧 | Postseason coverage | The pool requires 13 playoff picks: WC/DIV/CON/SB rows in the canonical and enriched feature tables (two-pass build, REG rows bit-identical), REG-only guards on every training/evaluation path, playoff-week serving through `margin-predict` and the safety contract, postseason-inclusive snapshot contracts, and a January ops rehearsal on 2025 playoff games (`docs/postseason_support.md`) |

## Phase 1 — research workbench

| ID | Status | Item | Definition of done |
|---|---|---|---|
| RWB-01 | ✅ | Season-aware team state | Current-season game count and explicit offseason regression. **Closed 2026-08-18: the regression constant was audited twice and `constants.DEFAULT_OFFSEASON_RETENTION = 0.67` survives both audits.** The season-transition persistence slopes genuinely do measure low, and that stands as a measurement, not an error: three independent NFL-side routes (486 season-to-season transitions) put the slope itself at 0.337-0.400 (median **0.337** at the first-4-games horizon the constant actually governs; the shipped table's own `result ~ diff_point_diff` ratio implying **0.379**; a within-season-centred regression of next season on prior season giving **0.400**, season-blocked [0.347, 0.460]), and the CFB Route-1 per-metric fit (8 metrics, horizon 4 games, frozen XLG-03 benchmark) independently runs **0.166-0.350** per metric. That is what "roughly twice what the data supports" meant, and it is still true of the slope. But the slope answers a different question than which constant the pick pipeline should use, and every test that asked the pipeline question resolves the other way. The predeclared CFB scalar grid (`docs/offseason_retention.md`, grid {0.00, 0.20, 0.337, 0.40, 0.50, 0.67, 0.75} against 8,933 `clean_core` games) found forced-pick accuracy prefers 0.67 or ABOVE: every value from 0.00 to 0.50 loses to 0.67 at `probability_positive` 0.005-0.04 (96-99.5% confidence), and 0.75 leans further ahead (P+ 0.69) -- margin MAE and Brier stay flat across the whole grid. The 2026-08-18 follow-up then built the measured per-metric slopes into an actual feature vector (not just a slope estimate) and scored it head-to-head against the uniform 0.67 scalar on the same benchmark: it loses, **-0.739 accuracy points, 95% [-1.296, -0.199], P+ 0.0037** (`offseason_retention_per_metric_cfb`, `registry/weak_signals.json`, `refuted_mechanism`/`wrong_sign_resolved`). Read together, the state features do better carrying MORE prior-season signal than the persistence slopes suggest -- so `DEFAULT_OFFSEASON_RETENTION = 0.67` survives its audit, and the "too high" framing must not send a future session back to re-derive it from the slopes alone. What remains genuinely open: this closes the CFB Route-1 per-metric construction only -- an NFL-side per-metric test was never run and would be a new predeclaration, not implied by this row. Conditioning retention on coaching change was tested and does NOT pay out of sample (leave-one-season-out, no better than chance) |
| RWB-02 | ✅ | Feature-family registry | Named, documented groups and configurable model allowlists |
| RWB-03 | ✅ | Walk-forward ablation runner | Comparable market/context/Elo/form/full experiment artifacts |
| RWB-04 | ✅ | Paper bankroll engine | Flat, full/fractional Kelly, caps, weekly exposure, drawdown |
| RWB-05 | ✅ | Local research dashboard | Read-only views for data, features, backtests, bankroll, picks |
| RWB-06 | ✅ | Model explanation artifact | Named logistic coefficients/missing indicators per fitted model |
| RWB-07 | ✅ | Season-level scorecards | Accuracy, Brier, log loss, ECE, ROI, CLV, and intervals by season. **Complete 2026-08-25:** season scorecard carries CLV via the real market-capture pipeline (`build_pairing_table` over the capture archive) with an explicit `clv_status` column -- a missing archive is DATA ("capture_unavailable"), never a silent NaN. First cut was rejected in review for inventing the clv.py API and exception-swallowing; re-tasked with the real contract and merged clean (`swarm/ref-scorecards-fix`, gates 1892 passed). |
| RWB-08 | ✅ | Block bootstrap uncertainty | Confidence intervals resampled by NFL week and season |
| RWB-09 | ✅ | Experiment registry | Config hash, code revision, source snapshot, metrics, notes. **Complete 2026-08-25:** registry plus read-only link verification (`nfl-ats experiment verify`, `verify_experiment_links`) -- every committed row's artifact_directory resolves against candidate roots; missing local artifacts are reported as data, not defects, since reproducibility lives in the row hashes (+76 test lines). |
| RWB-10 | ✅ | Prospective prediction ledger | Append-only predictions frozen before kickoff |
| RWB-11 | ✅ | Model cards | Intended use, training period, limitations, calibration history |
| RWB-12 | ✅ | Drift monitoring | Complete August 2026 (`docs/drift_monitoring.md`, `nfl_ats.drift`): weekly step 13 `drift-report` (optional, after the publish, read-only) plus a standalone CLI, monitoring four signals against the six prior completed weeks — per-column standardized mean shift + PSI organized by the feature-family registry, missingness delta in percentage points (non-numeric garbage counts as missing; vanished columns reported rather than dropped), published-probability distribution drift versus earlier same-configuration cards (matched by fingerprint, first-write-wins dedupe), and calibration drift (recent-vs-prior Brier/ECE on settled published probabilities). Artifacts land in `artifacts/drift/<season>-week-NN-<run_id>/` as `drift_report.json` + `feature_drift.csv`. PSI is unscored below 50 current-window games (measured ~0.2 null-level at n=16), calibration below 32/200 settled games. The report is telemetry, not evidence: it adjudicates no candidate, spends no registry window, and carries that disclaimer in every artifact |
| RWB-13 | ✅ | Dependence audit | Team error autocorrelation and season-preserving permutation null |
| RWB-14 | ✅ | Data-feasibility registry | Verified releases, nonempty seasons, row counts, timestamp semantics, source regimes, and effective sample-size tier |
| RWB-15 | ✅ | Evaluator sensitivity audit | Exact active-model reproduction plus null/permuted and known 0.5/1/2-point positive controls across repeated synthetic signals |
| RWB-16 | ✅ | Sensitivity-aware experiment review | Completed August 2026: artifact-verified inventory, ~130–150-look multiplicity ledger, and three predeclared replications on untouched 2013–2017 windows; all three reopened leads resolved (see below) **Swept again 2026-08-17 with a power-aware lens, and half of it does not hold.** Of 27 independent discarded mechanism families: **13 were unresolved below detection power** (recorded as negatives without the evidence), 7 were genuinely refuted mechanisms, and 6 were bounded by a positive control or a tight interval. **The most expensive error is `player_qb_continuity`:** the PRIMARY comparison that closed it and spent [2014, 2017] ran the candidate at ridge alpha 1.0 against a baseline at 10.0, bundling a feature change with an alpha change, and returned exactly 0.000 accuracy. The same artifact's alpha-matched arm returns **+1.1033 points** (season 95% [0.0000, +2.2177]) while the alpha change alone costs **-1.1033** (season 95% [-1.7034, -0.7992], resolvably negative) -- they cancel, and the matched figure nearly reproduces the predeclared 1.25. The deployable question was answered honestly; the FAMILY question was never answered and must not be described as closed. It implies group-wise ridge penalties, since these features want less regularization than the rest of the design. **But the wider hypothesis is NOT supported:** across all independent discards carrying a direction the sign test is 4 candidate / 17 baseline, and restricted to the genuinely-underpowered set it is 4/4, p = 1.000 -- a coin flip at every level of collapsing. Instrument validity was checked rather than assumed: RWB-15's null replicas split 8/16 at an injected effect of exactly zero, and at a true 0.5-point effect the sign is right 13/16 while the interval clears only 4/16, so direction really does accumulate faster than precision here. The one family where it accumulates is **player availability and value**, but the accumulation is weaker than first recorded (`docs/availability_confirmation.md`) -- five measurements, all positive, ~~p = 0.0625 within the family~~. The five were never named; the set that reproduces p = 0.0625 is {M1, M2, M3, M4, M7}, and it excludes a same-kind negative (participation RAPM, -0.43 points) that any principled boundary admits -- 5/6 gives p = 0.219, and the broadest injury-or-value boundary gives 7/9, p = 0.180. All five sit on the SAME 2,075 games (the opener window is 453 of them again), so the sign test's independence assumption is measured rather than assumed: a dependence-preserving centred bootstrap puts the honest figure at **p ~ 0.098**, not 0.0625. Category 3, not a finding. Also found: the drive layer's direction is reported BACKWARDS in the docs (they kept the Brier metric that worsened and omitted paired accuracy of +0.72 and +1.16 favouring the candidate). Below-power results now go to `registry/weak_signals.json` (RWB-18) rather than being deleted |
| RWB-17 | ✅ | Rotation registry | Complete August 2026 and in production use: git-tracked `registry/rotation_registry.json` plus `nfl-ats rotation declare/assign/status/record`, each family drawing one earliest-eligible confirmation window, forward-chained splits enforced in code, and recording a look spending the window permanently (`docs/rotation_registry.md`). Rule 9 (warm-up eligibility, `MIN_ELIGIBLE_START_SEASON`, now 2011 and derived from feasibility rather than an undefended default) was added before any window was spent under it, after the registry correctly offered `best_pick_ranker` a first block the evaluator could not score. Two looks have since been recorded through it — `best_pick_ranker` on [2013, 2015] and `mod07_weak_signal_stack` on [2020, 2021], both `unresolved` — and both windows are permanently spent |
| RWB-18 | 🚧 | Weak-signal registry | **Stop discarding effects that are merely too small to see.** The evaluator resolves ~2 ATS points (RWB-15) and almost every single feature is worth a fraction of that, so "no significant effect" has been the EXPECTED outcome for real-but-small signals -- and recording each as a negative quietly threw away the ones that were genuinely there. Two such errors were caught on 2026-08-17 (4th-down aggressiveness and penalty discipline, both filed as "priced" on intervals far too wide to say so). Built `registry/weak_signals.json` (git-tracked, schema-validated) plus `nfl_ats.weak_signals` and `nfl-ats weak-signals status|pool`. Every category-3 result in the taxonomy in `docs/pool_edge_plan.md` is now recorded with its effect, uncertainty and above all its **direction**. Two things then accumulate that no single experiment can buy: **signs**, since under a true null each estimate is a fair coin, so ten of twelve leaning one way is p ~ 0.039 assembled entirely from individually worthless results; and **precision**, since inverse-variance pooling sharpens by sqrt(K). The honest price is pinned in tests -- a 0.5-sigma signal needs about **sixteen** independent companions to clear 1.96 sigma, and four reaches only 1.0 sigma. Three guards: only category 3 is poolable (folding in a refuted mechanism would launder a known failure); overlapping seasons are reported as shared noise rather than hidden; and a pooled estimate justifies ONE predeclared combined look on a window none of the inputs touched, never being a finding itself. Seeded with the two signals caught today; the historical sweep for further below-power discards is the next step. **2026-08-24:** `weak-signals pool`/`status` now report `overlap_warnings` **per family** (`family_overlap_warnings`: one row per correlated decomposition group — opener/close grades, era and pre/post window splits, screening-battery cells — plus pairwise totals), replacing the 55k+-string pairwise list flagged by the 2026-08-22 correlation audit (risk #3); family inference mirrors `findings_registry`'s duplication passes and an explicit `--family` field can now be declared at record time. Validator gaps closed as code, without touching any recorded entry: a point estimate outside its own interval is refused at record time and surfaced at report time (`coherence_problems`; one pre-existing entry, `recurrence_flags_player_brier_validation`, has this shape and is reported, not rewritten); `bounded_by_control` must cite quantitative evidence; `no_split_half_reliability` cannot cite reliability above 0.10. All 447 recorded entries hold admissible classification values (**measured**: registry loads clean under the full validator).. **Update 2026-08-25:** per-family overlap warnings now surface in `weak-signals pool`/`status` output (registry_source links that share a family are disclosed as correlated, not independent), and a validator pass confirmed all entries carry admissible classification values (+136 test lines). Registry itself remains live and growing. |

## Phase 2 — point-in-time market data

| ID | Status | Item | Definition of done |
|---|---|---|---|
| MKT-01 | ✅ | Live odds provider adapter | Book, market, line, price, observed-at timestamp, raw response hash |
| MKT-02 | ✅ | Opening/current/closing line store | August 2026: six weekly scheduled live captures running (11 books), plus a purchased point-in-time snapshot archive — decision labels for 2020–2025 (paired tue_open+close for 227–272 games every season) plus hourly 2023–2025, playoffs, true openers, and moneylines; 8,746 snapshots, verified read-only backups on two drives |
| MKT-03 | 🚧 | No-vig market probabilities | Documented two-way normalization and favourite-longshot diagnostics |
| MKT-04 | ✅ | Closing-line-value tracking | Complete August 2026: the `clv-score` harness (per-pick points vs close, week-blocked intervals) plus the routine paper-decision ledger — `publish-predictions` appends every final played card's pre-kickoff picks at its published line (the first-recorded anchor is never rewritten by a republish), preserves raw/pre-arrest policy arms and frozen arrest provenance, and `clv-ledger` scores the whole ledger against live-store closes with a schedule-close fallback and surfaces the result on the History page |
| MKT-05 | ✅ | Cross-book consensus | Median line, dispersion, stale-book and outlier detection |
| MKT-06 | ✅ | Line-movement forecasting | Frozen pilot ran 2026-08-16 after the archive re-fetch (train 2020–2023, validate 2024, one look at 2025): direction-of-movement accuracy 59.5% on 2024's 200 movers and 57.2% on 2025's 194 movers, consistent with the pre-registered sign test (54.9% of 778 games, CI [51.3, 58.4], p=0.007); but magnitude never beat the no-movement MAE baseline (−0.001/−0.019 points) and the frozen ≥0.5-point threshold policy made exactly 1 bet (−0.5 CLV). Direction replicated, no exploitable magnitude edge; both retained. Production wiring shipped: `predict-close` writes the Week Board's close_predictions artifact from each week's live Tuesday capture and fails closed without one |
| MKT-07 | ✅ | Market residual model | Estimate only the correction to a market prior |
| MKT-08 | 🔬 | Timing policy | Compare fixed weekly timestamps and news-triggered updates |
| MKT-09 | 🚧 | Provider licensing/quota audit | Terms, redistribution limits, cost, retention, failure policy |
| MKT-10 | ✅ | Free historical close audit | Versioned public close archive plus 2025 opener/nine-book close sample, licenses, normalization, source comparison. **2026-08-19: era-stratified opener evaluation on this archive** (`docs/sbr_opener_evaluation.md`, `sbr_opener_era_*`): grading the frozen production model at SBR's Open across 2011-2021 (2,832 games), the edge concentrates in the recent era — 2011-2014 49.75% (P+ 0.414), 2015-2019 50.89% (P+ 0.734), 2020-2021 53.29% (P+ 0.918, production rule) — all `unresolved_below_power`, but a clean monotonic magnitude gradient consistent with three independent prior readings of the same 2018-2019-ish inflection |
| MKT-11 | 🚧 | Alternative power-rating divergence (Sagarin) | **2026-08-20** (`docs/sagarin_backfill.md`): Jeff Sagarin's NFL ratings backfilled via the Wayback Machine, Era A (`sagarin.com`, 2010-2025) **complete at 113/113 fetched and parsed**, as-of-Tuesday alignment against the project's own schedule (2013-2020/2023-2025 clear 80%+ weekly coverage; 2010-2012/2021-2022 thin at 41-62%). Era B (USA Today, 1998-2011) only 4 of 14 seasons fetched before the session's fetch budget ran out; resumable, does not touch Era A: `.\.tools\uv.exe run --no-sync python scripts/ingest_sagarin_ratings.py --out data/raw/sagarin --snapshot 20260820T112501Z --start-season 1998 --end-season 2011`. The predeclared divergence screen (Sagarin-implied spread minus market line) then ran the same day, and the honest read is mixed, not a one-word verdict: the large-divergence cell leans flat-to-negative for the candidate at BOTH grades (close-grade P+ 0.371, n=1,072; opener-grade P+ 0.254, n=406), and the model-agreement cell leans the same way (P+ 0.360, n=1,492) -- all three `unresolved_below_power`, consistent with, not proof of, the existing team-quality-is-already-priced ceiling. The top-decile-divergence cell leans the other way at the close (+3.53 pts, P+ 0.879, n=269) but that lean does NOT carry to the opener grade on the smaller 2020-2025 paired subset (P+ 0.393, n=106) -- thin, not resolved. A 2010-2016-vs-2017-2025 era split on the same top-decile cell flips direction (P+ 0.845 → 0.115), reported as a magnitude/era disagreement, not a verdict **2026-09-01 (WP19, `docs/sagarin_backfill.md` §9):** the parser gap that zeroed out 2012 (and most of 2013) is fixed -- a transitional 3-bracket `HOME ADVANTAGE=` line and a one-off comma `HOME EDGE=` line (both Nov 2011-Sep 2013) are now handled additively, recovering `home_edge_rating` for all 17 previously-null captures with zero new Wayback fetches (`--reparse-cache-only`); close-grade coverage rose 2,966 -> 3,229 games (2012: 0.0% -> 47.3%, 2013: 30.1% -> 85.5%), while the pool-relevant opener-grade population (1,053 games, 2020-2025) is unchanged. The existing `sagarin_battery_*` registry entries were **not** re-scored on the corrected data -- that is a new look at the same outcomes and needs its own predeclaration and rotation-registry window first. (WP19, 2026-09-01) |
| MKT-12 | 🚧 | Public betting percentage archive (Action Network) | **2026-08-20** (`docs/public_betting_sourcing.md`): 153 Wayback-archived bet%/money% captures ingested (2018-2026, 96.1% parse rate), joined to 800 REG-season games (≤72h kickoff match; coverage ceiling ~34% of REG games in the best-covered season). `covers.com/picks/nfl` -- a prior scout doc's claimed alternative source -- **measured this session to be a dead end** (community handicapper win-rate badges, not a bet% consensus; the actual consensus data is client-side-AJAX-only and was never captured by Wayback). A same-day predeclared 5-cell screen (fade-the-heavy-public, sharp bet%/money% divergence, model-vs-public interaction) found **every point estimate leans NEGATIVE** (P+ 0.10-0.26 at the week block, n=47-91) -- `unresolved_below_power` throughout, but fading the public and following "smart money" both underperform 50% on this sample, opposite the textbook mechanism. A live, prospective weekly capture path was built and verified end-to-end (`scripts/public_betting_live_capture.py`, two real runs against the live site, both HTTP 200). **Owner action needed: register the two weekly capture tasks** (Saturday and Sunday, noon ET; this machine's local time is already ET, no conversion needed) — ~~`schtasks /Create /TN "PublicBetting_Sat" /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"F:\Repos\nfl_py3\scripts\public_betting_capture.ps1\"" /SC WEEKLY /D SAT /ST 12:00 /RL LIMITED` and the same with `/TN "PublicBetting_Sun" /D SUN`~~ **Done 2026-08-21**: both tasks registered (`PublicBetting_Sat`, `PublicBetting_Sun`; weekly Sat/Sun 12:00 local = ET, `/RL LIMITED`) and verified via `schtasks /Query` (Status Ready, next runs 2026-08-22 / 2026-08-23 12:00 PM; first creation attempt failed only on PowerShell 5.1 quote escaping of the `/TR` value, succeeded with PowerShell-native quoting producing the identical command line) **2026-09-01 (WP20):** scouted for a pre-2018 public-betting source. Correction to the gap's framing: `movement_attribution_pop_*_public_*` cannot be extended by ANY pre-2018 source because its population floor (2020) comes from the Tuesday-opener quote archive (`data/market/raw`, starts 2020-08-25), not from public-betting coverage; the extensible target is the close-graded `public_betting_battery_*_close` family (schedule close line back to 2009). A sustained `web.archive.org` outage (measured: full TCP connect-timeout ~17 minutes across two poll windows; `archive.org` itself unaffected) blocked live CDX scouting of vegasinsider/oddsshark/sportsbookreview/pregame/teamrankings; `docs/public_betting_history_scout.md` records the measured local inventory (actionnetwork archive 153 captures, 800 matched games 2018-2026; live captures 7/7 since 08-20), the per-source table with ready-to-run CDX commands (the vegasinsider pre-2012 archive is the highest-value re-check), a predeclared ingest design (`scripts/ingest_public_betting_history.py`, mirroring the Sagarin ingest) and a predeclared follow-up cell (`public_betting_history_fade_heavy_public_close`, 2009-2017). Nothing built beyond the doc. (WP20, 2026-09-01) |
| MKT-13 | ⬜ | Player-prop line archive (Odds API) | **2026-08-20** (`docs/player_props_sourcing.md`): a budgeted pilot pull (2024 `player_pass_yds`, two tranches -- Tuesday-noon and Saturday-noon snapshots) found a real provider-cost quirk (an unposted market costs 0 requests, not the nominal 10) and a market-timing finding: a normal week's Tuesday board is sparse (only that week's earliest-kickoff game is priced; the full Sunday/Monday slate isn't posted until later), while Saturday is nearly fully priced but has already dropped that week's early game entirely -- **the two snapshots are measuring almost entirely disjoint games, not the same game at two points in time** (30 pairable player-game observations across 5 weeks, 28 from the opener week alone). The predeclared QB-prop-disappearance availability-signal experiment was **not run**; its design is now split into two sub-designs (early-game: Wed/Thu-vs-just-before-kickoff; Sunday/Monday slate: Thu/Sat-vs-Sunday-morning) rather than one Tuesday+Saturday pull. **Later 2026-08-20:** the recommended Wednesday tranche ran under a hard 1,500-request floor: 313 credits bought weeks 1-2 complete plus 14/16 week-3 events (1,210 rows), ending at 1,508 remaining. Tuesday-to-Wednesday pairability is 38 player-games (32 from opener week; 2/4 in ordinary weeks), with 24 new Wednesday appearances and zero Tuesday disappearances; this is instrument evidence, not an ATS verdict. `scripts/compare_player_prop_snapshots.py` makes the read reproducible and fails closed on post-kickoff rows; `ingest_player_props.py --earliest-kickoff-only` now prevents the next early-game tranche from spending on the whole Wednesday slate. No further quota spend is admissible until headroom returns |
| MKT-14 | 🚧 | Lagged TV-attention source (Sports Media Watch) | **2026-08-20** (`docs/sports_media_watch_ingest.md`): a free primary-source ingestion now caches and hashes the 2014-2023 seasonal pages plus their ratings-table images. The completed snapshot has 1,065 structured rows from 2014-2021 and all 43 indexed 2022-2023 images; 648 structured regular-week rows identify both teams. Current seasonal pages are living revisions with no row-level first-publication timestamp, so the leakage contract marks every row/asset unusable rather than admitting it historically. A directly fetched 2023 Week 9 article does expose distinct primary publication and modification timestamps and the same published audience figure. Next step: match each prior-game row/image to its dated weekly article (or a predecision archived capture), then predeclare only lagged prior-game audience and season-to-date attention constructs; same-game viewership is forbidden. No ATS screen has run. |

> **2026-08-20 later update to MKT-11:** the resumable Sagarin Era B
> cache advanced from 37 to 242 nonempty pages across eight season keys
> (1998-2005). All 242 cached pages parse, with zero parser exceptions,
> unmapped teams, or zero-byte files. The bounded run stopped before final
> consolidation, so the manifest and Parquet views remain at their prior Era A
> checkpoint and no network-failure count is inferred; rerunning the same
> documented command resumes from these cached pages.

> **2026-08-20 final Era B checkpoint:** the same snapshot is now consolidated
> across all 1998-2011 target keys: 585 durable pages parse cleanly into 18,473
> rating rows and 9,848 Tuesday as-of rows, with zero parser exceptions or
> unmapped teams. A final bounded retry recovered two transient gaps; seven
> Wayback fetches remain documented failures. This completes the usable
> historical source checkpoint. The earlier ATS screen was deliberately not
> rerun, so these additional rows change source coverage, not the current
> 53.4% opener-grade production decision.

> **2026-08-21 complete-source MKT-11 replication:** the frozen seven-cell
> Sagarin battery was rerun against all 585 parsed pages, with hashes for every
> consolidated input added to result provenance and a physical-column-projection
> leakage test. Usable close-grade coverage rose 2,684 -> 2,966 games (2010 and
> 2011 each now 240/256); the 2020-2025 opener population remained 1,053. The
> strongest close-only cell remains the top divergence decile (+3.535 points,
> `probability_positive=0.8908`, n=297), but its opener counterpart remains
> -0.943 points, P+=0.3928 (n=106), and the broad opener divergence cell remains
> -1.478 points, P+=0.2542 (n=406). All seven identities were replaced in place
> as `unresolved_below_power`. No production/prospective change: the complete
> source adds historical close evidence, not a better pool-grade decision.

## Cross-league evidence and transfer — highest research priority

College football supplies many more games and player transitions, but its rows
must not be appended to NFL rows as though both leagues were exchangeable. Its
value is to estimate shared football mechanisms, independently replicate
hypotheses, and construct priors that are subsequently judged on NFL-only outer
weeks.

| ID | Status | Item | Definition of done |
|---|---|---|---|
| XLG-01 | ✅ | CFB source feasibility audit | Completed August 2026: ~12,700 FBS games with PBP and a spread (2006–2025), verified ESPN-id→gsis crosswalk, no historical injury source (fails closed), non-participation unidentifiable in rosters, licensing green for private use; see `docs/data_feasibility.md` and `docs/cfb_data.md` |
| XLG-02 | ✅ | Immutable CFB ingestion | Complete August 2026: full backfills of schedules (2001–2025), lines (2006–2025), PBP (2004–2025, 3.13M plays after the upstream-defect contract), rosters, participants, ESPN betting, plus CFBD gap-fillers (draft picks with verified college→NFL id chain, returning production, recruiting, usage 2023+, portal); optional extras (box scores, usage 2013–2022, older recruiting classes) noted in `docs/cfb_data.md` |
| XLG-03 | ✅ | CFB market-residual benchmark | Completed August 2026: 12,500-game canonical FBS-vs-FBS table (2006–2025, oriented median close-proxy spreads, logged exclusions), frozen Ridge/alpha-10 market-residual evaluator scoring 51.60% forced-pick ATS on the 8,933-game clean core (2012–2019, 2021–2025) vs the 49.55% no-vig market control — week-blocked delta [+0.51, +3.49] points, but margin MAE and Brier unresolved vs market, so it is recorded as an instrument, not an edge; thin 2006–2011 and 2020 regimes reported as separate splits; the ported positive-control audit reproduced all 11,989 benchmark predictions to 1.5e-13 and detects 0.5/1/2-point synthetic effects in 1/8, 5/8, and 8/8 week-blocked replicas (NFL: 3/8, 2/8, 7/8) with zero permuted false positives — the larger sample resolves ~1-accuracy-point effects the NFL evaluator cannot (see `docs/cfb_data.md`) |
| XLG-04 | ✅ | Cross-league role-loss replication | Completed 2026-08-17 with a frozen predeclaration (`docs/cfb_role_replication.md`): matched credited-action shares (dropback/carry/reception), span-8 appearance-only EWM priors mirroring the NFL role state, no ATS outcomes touched. **Dropback and carry delivery replicated** (CFB medians 1.043/0.995 vs NFL matched 1.009/0.970, all gates passed; NFL matched dropback median independently agrees with PER-12's snap-share 1.011). **Reception not replicated** — it failed only the frozen 15% severe-under-delivery ceiling (CFB 19.1%; the matched NFL 17.2% would also have failed) while the league medians agree to 0.005; recorded as-is, no gate retuning. QB absences hand the top replacement a median 100% of dropbacks in both leagues; absence events conflate departures with injuries (documented caveat). Consequence: role-delivery is league-general for dropbacks/carries — a CFB role-loss feature family may now be predeclared against the XLG-03 benchmark |
| XLG-05 | ⬜ | Hierarchical CFB→NFL transfer | Compare matched NFL-only, naïvely pooled control, CFB-pretrained, and partially pooled models on NFL-only outer weeks. ~~The first candidate vehicle — the role-continuity family — failed the XLG-03 screen (August 2026, `docs/cfb_role_features.md`), so this now waits for a family that first clears the CFB benchmark~~ **REOPENED 2026-08-18**: the role-continuity family's −0.672-point closure sat below the instrument's own MDE80 of 0.927 points (f=9.96%, n=9,093), and trait split-half reliability (0.719 dropback / 0.680 carry) rules out a refuted mechanism, so it never cleared an admissible closing ground (`registry/weak_signals.json`, `docs/cfb_role_features.md`). It is `unresolved_below_power`, not failed, and the XLG-04 → XLG-05 transfer path is open again |
| XLG-06 | 🚧 | Rookie/young-player priors | Link college usage/value, recruiting, transfers, and draft identity to NFL players with explicit uncertainty and decay as NFL evidence accumulates **2026-09-01 (WP11): Stage 1 written up and recorded.** `scripts/xlg06_rookie_prior_cfb_screen.py` (committed 2026-08-18) tests whether recruiting rating predicts a true-freshman QB's realised CFB usage. Two runs: 2-cohort (2024-2025, n=125, r=+0.1121, P+ 0.877) superseded the same day by the 13-cohort backfill (2013-2025, n=557, cohort-blocked, r=-0.0018, 95% CI [-0.0920, +0.0882], P+ 0.484) -- more data moved the estimate toward zero. The construct's reliability is high and stable (Spearman-Brown 0.9607 -> 0.9650), so neither closing ground applies: recorded `xlg06_rookie_prior_stage1_qb` (`unresolved_below_power`, units `correlation`; `docs/xlg06_rookie_prior_screen.md`). A secondary RB read (P+ 0.9971, r=+0.0644, CI entirely positive) is flagged for its own look. Stages 2-3 unbuilt. (WP11, 2026-09-01) |
| XLG-07 | ⬜ | Cross-league availability semantics | Determine whether historical CFB injury reports are genuinely pregame and complete enough to learn availability; fail closed if timestamps or missingness are ambiguous |
| XLG-08 | ✅ | FluView illness, CFB replication | **Done 2026-09-01 (WP7).** The NFL FluView home-market illness construct replicated on college football at zero NFL-window cost, on top of the frozen XLG-03 benchmark arm (`docs/fluview_cfb_replication.md`). Two prerequisites built: venue state from cfbfastR-data `team_info` at a pinned commit joined on CFBD/ESPN `team_id` (0 unresolved rows, no CFBD credit); the 19 CFB-only FluView states ingested into a separate directory. On 5,671 clean-core games (2017-2019, 2021-2025, close-proxy grade): home-market **-0.388 accuracy points, week-blocked 95% [-1.272, +0.460], P+ 0.200**; away-market -0.423, [-1.553, +0.694], P+ 0.213; positive control +48.6 pts P+ 1.000. The NFL close-graded on-production direction (+0.969, P+ 0.792) did **not** replicate, in either CFB era; the CFB state-week trait's own split-half reliability replicates (Spearman-Brown 0.9856), so the mechanism is not refuted as noise. Both cells `unresolved_below_power` (`fluview_cfb_replication`, `--league cfb`); no rotation window spent. Method note: the benchmark's `fit_cfb_residual_model(feature_columns=...)` extension point makes "benchmark arm plus one column" a ~3-minute run. |

## Phase 3 — better football state

| ID | Status | Item | Definition of done |
|---|---|---|---|
| PBP-01 | ✅ | Versioned play-by-play ingestion | Partitioned Parquet snapshots and required-field contracts |
| PBP-02 | ✅ | Situation filters | Remove kneels and garbage time with versioned definitions |
| PBP-03 | ✅ | Drive table | Possessions, starting field position, points, success, and duration completed; ATS screen did not support promotion |
| PBP-04 | ✅ | Stable team efficiency | Early-down EPA, success, explosive rate, pressure, PROE |
| PBP-05 | ✅ | Opponent adjustment | Weekly time-decayed ridge offense/defense decomposition completed; no stable improvement, retained research-only. **One untested framing remains (2026-08-17):** every prior test *added* the adjusted columns on top of the raw ones, taking the design from 58 to 106–142 columns on ~4,700 rows at alpha 10, so signal was confounded with dimension inflation. The dimension-neutral **substitution** — same column count, adjusted defensive rates swapped in for raw ones — has never been run. Motivation is measured: split-half reliability of team per-play rates over 2009–2012 is **0.80 offense vs 0.46 defense** (full-season, Spearman-Brown; independently reproduced), so the active model's defensive per-play columns are roughly half noise, which is exactly what opponent adjustment exists to fix. **Screened on CFB 2026-08-17 and CLOSED** (`docs/cfb_opponent_adjustment.md`): the dimension-neutral swap moved clean-core margin MAE by -0.0003 points, `probability_positive` 0.463 on 9,093 games; isolating the opponent block from a time-weighting change bundled with it recovers +0.0005 MAE. A deliberate-leak positive control (adjustment fit on all of 2006-2025, so the columns see the future) moved MAE only **+0.0129** (P+ 0.984) -- so the null is measured rather than underpowered, and that figure is a **ceiling on the whole family**. The reliability argument is sound but does not translate: against a market-residual target the spread already prices team quality, so denoising those columns has almost nothing left to improve. A revisit needs a different target, not a better adjustment. See `docs/play_level_audit.md` |
| PBP-06 | ✅ | Special teams | Kicking, punting, returns, field position above expectation. **Built and screened 2026-08-19** (`docs/special_teams_battery.md`): fresh nflverse PBP fetched season-by-season and aggregated immediately to team-game/team-season tables (never persisted raw, referee-battery precedent). Reliability audit first, per the PBP-08 precedent: year-over-year Pearson r `punt_net_yards` +0.313 [+0.233,+0.391] (the strongest, comparable to PER-07's own +0.320 bar), `kickoff_return_yards` +0.158 [+0.073,+0.243], `punt_return_yards` +0.109 [+0.019,+0.196], `fg_oe` +0.065 [-0.022,+0.153] (weakest kept), `block_rate` -0.024 [-0.105,+0.060] (measured, scoped out of cells -- indistinguishable from noise even though it technically clears the admissible exclusion bar). 8 predeclared cells (4 traits x top/bottom quartile, team-perspective `team_covered`, week-blocked primary / season-blocked secondary bootstrap, 20,000 samples seed 20260819) all recorded `unresolved_below_power` -- none wrong-sign-resolved, none excludes zero, per AGENTS.md's binding rule. Two flagged leads (`probability_positive` outside [0.15, 0.85]): `special_teams_return_top_quartile` +0.499pts, week-blocked 95% [-0.074,+1.080], P+=0.955; `special_teams_punt_net_top_quartile` -0.389pts, week-blocked 95% [-0.967,+0.204], P+=0.095 (opposite the predeclared sign -- still unresolved, not refuted, since the interval does not sit entirely below zero). All 8 recorded to `registry/weak_signals.json` (`special_teams_*`, registry now 193 signals). Research-only; nothing wired into production. **CFB replication ran 2026-08-19** (`docs/cfb_special_teams_replication.md`, results recorded to `registry/weak_signals.json`): neither NFL lead replicates cross-league on the XLG-03 clean core (8,933 games, 17,866 team-rows). `cfb_special_teams_return_top_quartile` reads -0.337 pts, week-blocked 95% [-0.691, +0.021], P+ 0.032 — the opposite sign from the NFL cell it mirrors (+0.499 pts, P+ 0.955). `cfb_special_teams_punt_net_top_quartile` reads +0.307 pts, week-blocked 95% [-0.164, +0.791], P+ 0.897 — also a sign flip, but the other direction: CFB leans toward the originally predeclared positive sign that the NFL read itself inverted (NFL -0.389 pts, P+ 0.095). Both CFB entries `unresolved_below_power` (neither interval sits entirely on one side of zero); no challenger wiring from either league's special-teams read |
| PBP-07 | 🚧 | Penalty state | **Not built, and NOT closed -- the market check is below detection power.** Team penalty rate is 0.0750 +/- 0.0101 with year-over-year reliability +0.261, so there is a real if modest trait to forecast. The market check (most-penalized quartile covers 49.85% vs least-penalized 50.52%) is a 0.67-point gap on quartile-sized samples and cannot separate "priced" from "too small to see" -- category 3 in `docs/pool_edge_plan.md`, not a negative. Treat as a weak-signal stacker input at most, never a standalone candidate and never worth a window. Separately, the construct this row actually names is not measurable from local data -- neither play-by-play snapshot carries `penalty_type` or a description column (45 columns; only `penalty` and `penalty_yards`), so "pre-snap discipline" would need a wider nflverse column pull. ~~Since total penalty discipline is already null against the market, that pull is not worth making~~ **Correction, 2026-08-20: the pull was made after all and it was cheap** (`docs/penalty_crew_tendencies.md`, build-next rank 1 in `docs/archive/data_source_scout_v4.md`) -- nflverse's upstream PBP schema does carry `penalty_type`/`penalty_team`, this project's own trimmed local snapshot was just dropping them. Widening extended `docs/referee_battery.md`'s already-reliable crew penalty-RATE battery (`mean_total` +0.370, 158 referee-season pairs) to penalty-TYPE-specific rates: Offensive Holding (+0.3226) and Defensive Holding (+0.2702) show real, moderate year-over-year persistence; Defensive Pass Interference does not (-0.0663, near zero). Four predeclared cells, all `unresolved_below_power`: the standout is a run-heavy home team facing a top-quartile Offensive-Holding-calling crew covering LESS, +0.1390 pts, P+ 0.9015, consistent direction across both eras (2016-2020 P+ 0.895, 2021-2025 P+ 0.662, real magnitude change not a sign flip); a heavy home underdog facing a top-quartile flag-heavy crew at the OPENER grade leans +0.1056 pts, P+ 0.9204, stable across both opener-era halves despite n_flag=9/era. A DPI-tilt/pass-heavy-favorite cell and a flag-rate/high-total cell both stay near coin-flip or era-inconsistent. Not a standalone candidate; a future pooled read alongside the existing referee-battery cells (same `accuracy_points` unit) is the natural next step, not a promotion decision **2026-09-01 (WP22):** a live, point-in-time capture of the UPCOMING week's officiating assignment now exists (`src/nfl_ats/referee_assignments_capture.py`, scheduler job `referee_assignments_wed` Wed 15:00 ET, `docs/referee_assignments_capture.md`), fetching Football Zebras' weekly assignments post (the only public source; `operations.nfl.com` has no weekly page, verified) and joining the referee's name to `officials.parquet`'s crew traits (17/17 after one alias, "Ron Torbert" -> "Ronald Torbert"; validated live on 2025 Week 18, 16/16 games). Publish timing measured across 10 sampled 2025 weeks is never before Tuesday afternoon (latest normal-week sample Wed 12:42 ET), so this capture structurally cannot feed the Tuesday-lock/opener card -- it feeds a late-week refresh, which is exactly what `crew_tilt_refresh_v1` (WP47, in flight) wires for the two P+ ~0.90 cells. This satisfies the point-in-time precondition PBP-10 names; neither cell has been re-scored prospectively yet. (WP22, 2026-09-01) |
| PBP-08 | 🔬 | Scheme/matchup interactions | Protection-pressure, explosive pass, personnel, coverage proxies |
| PBP-09 | ✅ | Pace and possession forecast | Closed on measurement, not built (2026-08-17; `docs/play_level_audit.md`). Game play volume has almost nothing to forecast and no link to the outcome we care about: on 2009–2012 (1,024 games, no window spent) plays/game is 125.5 with sd 8.7, a coefficient of variation of **0.070**; forecasting it from both teams' prior pace reaches only **R² 0.041**, shaving ~2% off the residual sd. Worse for the premise, `corr(plays, |margin|) = −0.20` — blowouts have *fewer* plays, which is clock-killing and kneel-downs reading backwards from the result — and `corr(drives, |margin|) = +0.004`, i.e. zero. Independently reproduced by a second measurement before this row was changed. A pregame pace forecast has no path to ATS value; reopen only if some mechanism other than volume is proposed |
| PBP-10 | 🔬 | Referee effects | Only if assignments are point-in-time and effects survive shrinkage |

## Phase 4 — players, coaches, injuries, and offseason priors

| ID | Status | Item | Definition of done |
|---|---|---|---|
| PER-01 | ✅ | Weekly roster/participation snapshots | Immutable weekly roster plus season-partitioned 2016–2025 participation snapshots, hashes, source regimes, and outcome-time contracts |
| PER-02 | 🚧 | Quarterback state | Starter probability, player EPA/CPOE, backup adjustment |
| PER-03 | 🚧 | Injury report history | 76,784 canonical 2009–2024 rows ingested with a 24-hour cutoff; weekly files behave as final observations, not revision streams; replacement live source remains. **2026-08-20: a candidate replacement source was ingested and screened for additivity; written up 2026-08-20/21 (see below).** Pro Football Rumors' transaction wire (`docs/pfr_transactions_sourcing.md`, ingestion-only scope as written: 72,368 posts 2014–2026, 29,414 `transaction_relevant`, free URL-path year/month proxy 100% reliable, sitemap `<lastmod>` measurably unreliable at 73–84%) was then screened against the already-ingested PFT injury-news source by a same-day follow-up (`scripts/pfr_pft_additivity_experiment.py`, `registry/experiments/pfr-pft-additivity-experiment/pfr_pft_additivity.json`; the doc/registry mismatch flagged here was resolved — `docs/pfr_transactions_sourcing.md` section 6 now carries the complete-cache rerun (verified against the artifact 2026-08-21, correction note dated there)). At the Saturday-refresh cutoff (the pool's real per-game decision point, not the Tuesday line freeze), PFR adds almost nothing beyond PFT: only 5 additional official-injury-row-visible credits from PFR vs. 162 from PFT, pooled visible share barely moves (93.65% → 93.67%). At the Tuesday-noon cutoff PFR looks relatively more additive (30 additional vs. PFT's 1,977) but off a much smaller base (0.31% → 0.49% visible pre-Tuesday). A bulk per-article date-fetch (`scripts/pfr_bulk_date_fetch.py`) reached 831 of a ~4,361-row targeted 2022–2025 scope (715 with a precise date extracted) before stopping; resumable: `.\.tools\uv.exe run --no-sync python scripts/pfr_bulk_date_fetch.py --snapshot data/raw/pfr_transactions/20260820T011126Z` |
| PER-04 | 🚧 | Depth chart history | Starter/backup roles without using later revisions |
| PER-05 | 🚧 | Snap-weighted player value | Box-score value reached 52.14%; a fixed participation extension fell to 51.71% and was rejected; learned availability and stronger value targets remain |
| PER-06 | 🚧 | Roster continuity | Lagged lineup/roster continuity is implemented and isolated as the strongest current player-family component; returning-snap offseason priors remain |
| PER-07 | 🚧 | Coaching/coordinator changes | **Investigated 2026-08-17; the row's original framing was the wrong shape and is superseded.** Coach identity is priced -- split-half reliability of a coach's mean ATS residual is +0.063 (Spearman-Brown +0.119, 85 coaches), and coaches with an above-50% prior cover rate subsequently cover 49.5-51.0% at every sample threshold. Tenure buys nothing beyond year one (years 2-3 50.74%, 4-6 49.63%, 7+ 51.96% -- no monotonicity) and there is no variance story either (year-1 early-season residual sd 13.53 vs 13.22 for veterans). 4th-down aggressiveness is genuinely reliable and is **NOT closed** -- an earlier draft of this row called it "fully priced" and that was wrong, an underpowered interval read as a null (the RWB-16 error this project exists to avoid). Re-measured independently: expressed relative to each season's league norm (which strips the well-documented league-wide drift toward going for it, and halves the naive figure) year-over-year reliability is **+0.320** on 512 team-season pairs, go-rate-over-expected sd 0.0381. The market test regresses prior-season aggressiveness against the ATS residual on 4,175 games and returns, for a one-sd matchup, **-0.038 points, season-blocked 95% [-0.423, +0.417]** -- an interval 0.84 points wide. The hypothesis it was meant to reject, that the value is *completely* unpriced, is worth about **+0.174 points** and sits comfortably INSIDE that interval. The test therefore cannot distinguish priced from unpriced at this sample size. ~~Resolving 0.174 points at 95% would need roughly 24,000 games -- about 90 NFL seasons (CFB's 12,500 does not close that either).~~ **That framing is retracted**: this project is model/feature-limited, not data-limited (`docs/scaling_and_transfer.md`), and "needs N more games/seasons" is never the conclusion to draw from an underpowered interval. The decision this implies is the one AGENTS.md already sanctions for a category-3 result: pool it. So this can never be confirmed as a standalone candidate and must not be spent on a window; it is a **stacker/pooling input** (RWB-18, `registry/weak_signals.json`), not a dead end. Two honest discounts: the sd 0.123 EPA-points-per-game magnitude is a naive bin-EPA figure that selection almost certainly inflates, since coaches go for it when they expect to convert; and MOD-07 has already returned unresolved once. Interim changes are too thin (29 events; the 65.5% first-game figure is n=29 noise) and are **not provably pregame** -- historical snapshots are backfilled with whoever worked the game, so an interim feature needs an announcement-date source before it could pass a leakage test. **What survives is one binary flag:** a year-1 head coach fades in weeks 1-8, covering 46.72% on 762 games / 104 team-season clusters (cluster 95% [43.46, 50.06]), -1.30 points raw and -1.45 with a cubic quality control, both blocked intervals excluding zero. It is not a quality confound (equal-quality teams that KEPT their coach cover 50.70%) and not a QB proxy (new-QB coefficient is null in the 2x2). The active model does not capture it: it sides with the year-1 team 51.6% of the time and those teams covered 47.6%. Independently replicated on free CFB data at 3.5x the sample (year-1 all weeks 48.73% on 483 clusters; weeks 1-4 47.14%), with the same sign, magnitude and era boundary. **The catch that governs the decision:** the effect is null in 2009-2017 (49.00%, P=0.321) and lives in 2018-2025 (44.79%, P=0.011), i.e. inside the mined era. Worth ~+0.57 pp of full-slate pool accuracy. Do NOT ship on this measurement. **Correction 2026-08-18: the row previously recommended declaring `hc_year_one_fade` with `acknowledges_mined_2018_2025` and buying the answer with one opener window -- that is wrong and has been struck.** The registry entry's own notes say the opposite (`registry/weak_signals.json`): "Do NOT spend an opener window confirming this inside 2018-2025 -- that is its own discovery data. The only non-circular tests are prospective 2026+ and the CFB replication." The circularity is structural, not a style preference: the effect is null in 2009-2017 and lives entirely inside 2018-2025 (the mined era), and the opener grade's entire season pool is `GRADE_POOLS["opener"] = (2020, 2025)` (`src/nfl_ats/rotation.py`) -- every opener-graded block available to spend already sits inside 2018-2025, so no opener window can test this without re-confirming the effect on its own discovery data. **Owner decision 2026-08-18: hold the opener blocks; the honest test is prospective 2026 weeks 1-8, which begins 2026-09-08 at no window cost.** **2026-08-18: overlay implemented and ON** -- the clean-case fade is now a pick-level overlay wired into the published card path (`src/nfl_ats/coach_fade_overlay.py`, `OVERLAY_ENABLED = True`, weeks 1-8), disclosed on the card the same way Best-Pick ties are, and registered as challenger `hc_year_one_fade_overlay` so weeks 1-8 of 2026 record and score both arms prospectively at no window cost; see `docs/coach_fade_overlay.md`. Verified against the real Week 1 2026 card: exactly one flip (BAL -3.5 at IND becomes IND +3.5), MIA at LV correctly left untouched (both coaches year-1). Not yet published -- the card publishes for real at the 2026-09-08 Tuesday lock. **2026-08-20: the "interim changes are too thin / not provably pregame" line above is superseded** (`docs/interim_coach_screen.md`, `docs/archive/data_source_scout_v3.md` section 5). The "not provably pregame" objection is resolved by construction: this build's authoritative source is Pro Football Rumors' own interim-coach list, which carries real public FIRING-ANNOUNCEMENT dates (e.g. "Dec. 6, 2010"), not the backfilled `schedules.parquet` coach field the old note was worried about -- that field is used only as a join/cross-check, with a takeover-date fallback for the 10 of 39 team-seasons where it doesn't reflect the change at all. Sample is still small, honestly reported as such: 39 joinable interim stints (2000-2008 excluded, pre-dates this project's own 2009 data floor), 250 REG-season team-games, 2009-2025. The headline "any game under an interim" flag (`interim_hc_active`) is flat-to-slightly-negative -- 49.20% vs. 50.02% league baseline, effect -0.024 accuracy points, week-blocked 95% [-0.204, +0.165], P+ 0.386 -- a DIFFERENT and larger sample than the old n=29/65.5% figure, and it does not repeat that lean. But splitting by `interim_game_number` finds the old figure's shape in a sharper, cleaner cell: FIRST game only (n=39) covers **58.97%** vs. 49.96% for the rest of the league (`interim_hc_first_game`, effect +0.041 pts, P+ **0.845**), while games 2+ (n=211) actually cover BELOW the league baseline (47.39%) -- the folklore ("teams often cover their first game under an interim") has real support, it just doesn't extend to the rest of the stint the way a single aggregate flag would suggest. Both readings, plus a home/road split within interim games (P+ 0.81-0.95, no predeclared mechanism) and a fired-coach-was-year-1 split (P+ 0.56, essentially a coin flip), plus two era re-slices, are recorded `unresolved_below_power` in `registry/weak_signals.json` (7 entries) per AGENTS.md -- none crosses the refuted-mechanism bar, none is claimed as resolved. ~~Not wired into the published card; `docs/interim_coach_screen.md` section 7 sketches what a live `interim_hc_first_game` overlay would need (a weekly re-fetch of the same PFR list, effort S) without building it.~~ **Built later the same day (2026-08-20).** `interim_hc_first_game_tilt_overlay` is now a dual-tracked, no-window-cost prospective challenger (`src/nfl_ats/interim_hc_first_game_tilt_overlay.py`, fail-open on a stale/missing PFR snapshot; `artifacts/prospective/challengers.json`, `ACTIVE_PROSPECTIVE`) -- never applied to the published card, spends no rotation-registry window. Overlap with the live `hc_year_one_fade_overlay` is real but narrow (both overlays transform the base card independently; only 32 of 250 under-interim games fall in that overlay's weeks 1-8 window). |
| PER-08 | ⬜ | Transaction-aware preseason prior | QB, roster, coaching, draft/free-agency adjustments |
| PER-09 | 🚧 | Latent player ratings | First season-lagged offense/defense adjusted-plus/minus is reproducible but failed its matched ATS screen; hierarchy, units, and special teams remain |
| PER-10 | 🔬 | Injury scenario mixture | Forecast weighted across active/inactive player scenarios |
| PER-11 | ✅ | Learned play probability | Prior-season report/practice/position rates improved player Brier 0.09500 → 0.09056 and ATS 52.14% → 52.24%; blocked ATS intervals remain unresolved |
| PER-12 | ✅ | Expected role delivery | Closed at the intermediate target August 2026: the clipped two-part delivery model lost to both parents on delivered-share MAE in all 11 seasons (0.1888 vs 0.1606/0.1621) because injury-listed players who play at all deliver ~their full prior role (median ratio 1.01); no ATS rows were generated |
| PER-13 | 🚧 | Reliability trait priors | Per-player durability from 16-season injury/participation history and roster-status volatility (including league suspensions from weekly roster status codes) as priors inside the learned-availability model; validate on the player-level availability target before any ATS screen **Stage 1 answered on the intermediate target (WP13, 2026-09-01; `docs/per13_durability_prior.md`, frozen before any Brier).** Six per-player durability columns (EB-shrunken residual-vs-designation rate, played-despite-listed rate, raw unavailability offset, history depth, on-roster-no-snap rate, reserve/suspension rate back to 2009; no hand-picked shrinkage -- beta-binomial and DerSimonian-Laird moment estimators per fold) added to a re-fit learned-availability baseline on PER-11's out-of-season protocol. On 52,382 player-games 2015-2024: Brier 0.09087 -> 0.08332, **+0.0075514**, week-blocked 95% [+0.0067549, +0.0083081], P+ 1.000; 10/10 seasons and 5/5 position groups favour it; positive control passed first; a post-hoc placebo (each player handed another's history) destroys the effect (-0.0000871), so this is row-level player information. Prior-seasons-only retains 51% of the gain. Recorded `per13_durability_prior_availability_brier` (units `brier`, reliability 0.793, `unresolved_below_power`). **The frozen EV gate is met, so Stage 2 -- an ATS test on top of PRODUCTION on a rotation-assigned window -- is warranted (queued as WP26).** Sober framing: PER-11's 0.00444 Brier gain bought only +0.10 ATS points, so this is not an ATS edge and is never pooled with `accuracy_points`. (WP13, 2026-09-01) |

> **2026-09-02 PER-13 follow-through (supersedes the queued-Stage-2 text in
> the row above):** the close-graded on-production screen measured +0.536
> accuracy points, `probability_positive` 0.872 on its assigned [2011, 2013]
> block (`docs/per13_durability_stage2_on_production.md`). The required opener
> confirmation then ran on the rotation-assigned [2020, 2021] block
> (`docs/per13_durability_opener_confirmation.md`): candidate 52.85% versus
> production 53.73%, paired -0.877 points, week-blocked 95% [-2.386, +0.446],
> `probability_positive` 0.084 over 456 games; positive control +43.86 points,
> P+ 1.000. The opener decision read favours leaving production unchanged.
> Both record commands classify the result `unresolved_below_power`; no
> admissible closing ground applies, and no further PER-13 window should be
> spent merely to chase the sign.

> **2026-08-20 later update to PER-03:** the targeted PFR precise-date
> backfill is complete (`docs/pfr_transactions_sourcing.md`): 4,361/4,361
> target rows have valid dates and URL/year-month matches, with zero malformed,
> failed, mismatched, or duplicate-slug records. The earlier PFR/PFT additivity
> artifact used only 715 relevant precise dates and was heavily 2022-weighted;
> its coverage figures are retained as partial-cache evidence, not treated as
> the final cross-season estimate. No ATS screen was rerun.

> **2026-08-20 later update to PER-03:** USA Today's public NFL player-arrests
> archive is now ingested end to end (`docs/player_arrests_sourcing.md`):
> 56/56 pages, 1,116 unique incidents dated 2000-01-24 through 2026-06-23.
> The resumable source contract hashes immutable raw pages and emits a
> mechanically restricted point-in-time view that excludes outcome,
> description, and link fields. This is ingestion evidence only: player/team
> matching and proof that an incident was public before the relevant prediction
> timestamp are still required before any ATS feature or screen.

> **2026-08-20 complete-cache follow-up to PER-03:** the frozen PFR/PFT
> additivity questions were rerun after all 4,361 predeclared PFR pages had
> precise dates (`docs/pfr_transactions_sourcing.md`, artifact
> `artifacts/pfr_pft_additivity/20260820T155757Z/result.json`). PFR contributes
> 1,839 Tuesday rows that PFT misses (25.86% of their matched union) and 1,708
> Saturday rows (22.53% of the union). On the official-injury population,
> pooling raises Tuesday visibility from 12.05% with PFT alone to 13.07%, adding
> 171 rows; Saturday is already saturated and gains 10. This is source-coverage
> evidence, not an ATS effect: retain PFR alongside PFT in the future frozen
> availability feature, but do not infer an improvement over the 53.4%
> opener-grade production rule from coverage alone.

> **2026-08-20 player-arrest screen follow-up to PER-03:** a point-in-time,
> leakage-tested 14-day incident flag is now declarative
> (`docs/player_arrests_screen.md`). The direction fixed before scoring was to
> fade the affected team. That fade leans wrong at both grades but remains
> category 3: close -0.1371 full-slate accuracy points,
> `probability_positive=0.0572` (247 flagged team-games); opener -0.1015,
> `probability_positive=0.1653` (50 flagged). Both are recorded
> `unresolved_below_power`; no admissible closing ground was established. The
> raw split therefore creates a clearly post-result lead in the opposite
> direction -- back the affected team -- which needs a direct 53.4%-policy
> comparison and live Tuesday refresh contract before prospective activation.

> **2026-08-20 severity follow-up:** a second predeclared player-arrest family
> isolated category labels involving violence against a person before reading
> ATS outcomes (`docs/player_arrests_severity_screen.md`). The 14-day fade is
> flat at the close (-0.0000 full-slate accuracy points,
> `probability_positive=0.4732`, 76 flagged team-games) and leans against the
> fade at the opener (-0.0335, `probability_positive=0.2893`, 22 flagged).
> Both cells are recorded `unresolved_below_power` with no admissible closing
> ground. Severity does not sharpen the broad incident lead; it remains useful
> retained evidence rather than a reason to activate a narrower policy.

> **2026-08-20 direct policy follow-up:** the clearly post-result broad-side
> lead was evaluated against the actual 53.36% production probability rule at
> the frozen opener (`docs/player_arrests_policy_eval.md`). Backing the sole
> recently affected team only when production opposed it changed 25 of 1,503
> graded games and scored 53.76% versus 53.36%: **+0.3992 accuracy points**,
> `probability_positive=0.8562`, with every season's delta nonnegative. It is
> recorded `unresolved_below_power`, not presented as fresh confirmation: the
> direction came from overlapping historical outcomes. Under the forced-pick
> EV rule this supports a freshness-gated, no-window-cost 2026 prospective
> challenger; it does not change production.

## Phase 5 — weather, venue, rest, and travel

| ID | Status | Item | Definition of done |
|---|---|---|---|
| ENV-01 | ⬜ | Forecast-time weather | Archive the forecast that existed at the decision timestamp. **2026-08-19: actual-weather mechanism screens ran first** — 8 predeclared cells from schedules' game-time actuals (`scripts/nfl_weather_battery_screen.py`, seasons 2009-2025, all recorded `unresolved_below_power`): wind/cold cells lean positive (high_wind_outdoor +0.16 pts P+ 0.757; warm_team_cold_late +0.16 P+ 0.972; dome_team_outdoors_cold +0.11 P+ 0.825) but are game-time actuals, hence UPPER BOUNDS on any pregame forecast feature. **Sourcing question answered 2026-08-19 evening (`docs/weather_forecast_sourcing.md`), by measurement not just documentation-reading**: Open-Meteo's Historical Forecast API is confirmed paid-only (free tier excludes it outright); NOAA's NDFD archive IS free and genuinely point-in-time — verified by downloading one real archived bulletin (`wmo/maxt/2022/12/11/YGRZ98_KWBN_202212112153`, WMO-header-stamped issuance 21:53 UTC, valid GRIB2 edition-2 payload, 4 embedded messages). Two free tiers: (1) AWS Open Data (`s3://noaa-ndfd-pds`), anonymous HTTPS, instant, but **measured** to cover 2020-present only (year prefixes checked live); (2) NCEI HAS/AIRS "NDFD - By WMO Header," **read** as free and covering 2004-present (the full project window) but request/order-gated, turnaround unverified. Needed elements (`maxt`, `temp`, `wspd`, `wgust`, `wdir`) all confirmed present. Real remaining cost is downstream, not access: no GRIB2 decoder (`pygrib`/`cfgrib`/`eccodes`) is installed in this repo's venv (measured), and no stadium lat/lon table exists yet (measured) — both are prerequisites before any point value can be extracted; HAS/AIRS turnaround for 2009-2019 is the next thing to actually test. Data-quality note: schedules' outdoor wind/temp is missing for 49% of 2022 and 22% of 2023 games. **2026-08-19: the archive was built, and the payoff screen ran** (`docs/forecast_archive_build.md`, `docs/forecast_weather_screen.md`). No GRIB2 decoder was needed after all — the build pivoted to the Iowa Environmental Mesonet's GFS-MOS JSON API (point-in-time station bulletins, no gridded decode). The pool-relevant `tuesday_noon` cutoff archive covers the full 2020-2025 REG population (1,615 games, 100% of 1,598 domestic games fetched OK): temp r=0.897 MAE=7.63°F bias=-5.80°F, wind r=0.387 MAE=4.89mph bias=+2.79mph vs. actuals. A new `kickoff_nearest` cutoff (added this session; ~~not pool-playable since the pool locks Tuesday noon~~ **owner-corrected 2026-08-20: this IS pool-playable, and is in fact the primary target — only the pool's LINE locks Tuesday noon, our picks are editable up to each game's real deadline (refined 2026-08-20: min(kickoff, Sunday 16:00 ET) — SNF/MNF lock early at Sunday 4pm, so kickoff-nearest forecasts are later than the real decision time for those games specifically), so kickoff-nearest is still the actual decision-time information set for the bulk of the slate, not `tuesday_noon`**) validates far tighter on the full 2024 season (272 games): temp r=0.972 MAE=3.05°F bias=+0.49°F, wind r=0.718 MAE=3.11mph bias=+2.24mph, confirmed on a 2020-2023 60-game spot check (temp r=0.924 MAE=2.35°F). GFS's own IEM archive reaches back to at least 2005 (measured), so a `kickoff_nearest` archive for 2009-2019 needs no NCEI order — only a `tuesday_noon`-style archive for those seasons remains blocked on the gated HAS/AIRS order. The payoff screen then re-scored the four strongest actual-weather cells above with the Tuesday-noon forecast substituted for the game-time actual, on the 2020-2025 population (see `docs/forecast_weather_screen.md`'s owner-corrected note: this screen used `tuesday_noon` only and was not re-run against the more pool-relevant `kickoff_nearest`): `forecast_weather_warm_team_cold_late` +0.332 pts P+ 0.9711, `forecast_weather_temp_gap_cold_visitor` +0.430 pts P+ 0.9029, `forecast_weather_high_wind_outdoor` -0.059 pts P+ 0.4585, `forecast_weather_dome_team_outdoors_cold` -0.113 pts P+ 0.3165 — all `unresolved_below_power`, recorded in `registry/weak_signals.json`. **2026-08-20: `kickoff_nearest` archive extended to the FULL 2009-2025 window** (`docs/forecast_weather_screen.md` "2026-08-20 extension"; `tuesday_noon` stays blocked pre-2020 -- measured via a live IEM MOS API probe, `model=MEX` returns zero rows for 2009/2015 Tuesday-noon runtimes while `model=GFS` returns 21 rows for the same 2015 runtime, confirmed impossible at the source). 4,431 REG games, 4,379 fetched `ok` (99.98% domestic coverage), instrument validation holds across the full 17-season window (temp r=0.964 MAE=3.25°F; wind r=0.650 MAE=3.15mph) -- no degradation vs. the 2020-2025-only numbers above. A 6-cell family (2 reruns + 4 new: wind x pass-heavy away favorite, precip probability x high total, temp swing since the visitor's last game, dome-team cold+windy) ran on the full window and a 2009-2019-only split, 12 specs, all `unresolved_below_power`. **`warm_team_cold_late` is the strongest read in this project's weather work and the first of these families to clear the crossing-zero bar on both windows**: full +0.1697 pts, 95% [+0.0091, +0.3169], P+ 0.980; pre2020-only +0.2285 pts, [+0.0208, +0.4167], P+ 0.985 (larger pre-2020, era magnitude not absence). `temp_gap_cold_visitor` swung from a coin flip on an early checkpoint to the largest point estimate in the family on final data (+0.27/+0.30 pts, P+ 0.922/0.871) -- three independent constructions of the same mechanism now converge in the 0.87-0.98 P+ range. `wind_passing_away_favorite` inverted on full data (checkpoint P+ 0.996 on n=4 fell to P+ 0.367 on n=22) -- a caution against a 4-game read, not a refutation. `dome_cold_windy` reads weaker than its cold-only parent, a reportable negative lean. Two of six wired same-day as dual-tracked, no-window-cost prospective challengers sharing one live fetch: `forecast_weather_kn_warm_team_cold_late_tilt` and `forecast_weather_kn_precip_high_total_tilt` (`artifacts/prospective/challengers.json`, both `ACTIVE_PROSPECTIVE`); the other four left unwired by explicit EV reasoning, notably that `temp_gap_cold_visitor`'s kickoff_nearest form is likely redundant with the already-live `forecast_cold_visitor_tilt` (tuesday_noon cutoff) rather than a reason to add a second challenger for the same mechanism |
| ENV-02 | ⬜ | Stadium/roof/surface history | Venue state and roof decision where available. **2026-08-19: the surface half of this row produced the session's strongest new lead.** Grass-modal visitors playing on artificial turf under-cover: home side covers 52.25% vs 47.74% complement, +1.16 full-slate pts, week-blocked [+0.29, +2.04], P+ 0.995 (NFL REG 2009-2025, n_flag=1,112); survives venue control (within turf venues, grass-modal vs turf-modal visitors +1.46 pts, P+ 0.933; franchise-cluster interval similar, not a one-team artifact); **replicates on CFB** (+1.56 pts, P+ 0.916 on the XLG-03 clean core, `cfb_surface_familiarity_turf_venue_visitor_split`); the grass-venue mirror is ~null in BOTH leagues (P+ 0.320 NFL / 0.429 CFB), so the mechanism is asymmetric — turf-specific, not bilateral familiarity. NFL magnitude concentrates in 2018-2025 (mined era, disclosed). All entries `unresolved_below_power` in `registry/weak_signals.json` (`weather_battery_surface_switch_grass_to_turf`, `surface_familiarity_r1..r3`, CFB C1-C3). Exploitation is live at no window cost: `surface_switch_tilt_overlay` 2026 prospective challenger. **Deepened 2026-08-19 evening** (`docs/weather_followup.md`, `scripts/nfl_weather_followup_screen.py`): compounding the surface-switch flag with outdoor/temp<=45F (`weather_followup_surface_switch_x_outdoor_cold`) scores a raw per-game gap of +4.87pts (vs. the parent cell's ~+4.51pt raw gap), i.e. the mechanism does not weaken under the cold interaction, but the narrower subset (153 games vs. 1,112) full-slate-scales to only +0.17pts, week-blocked [-0.12, +0.45], P+ 0.877 — `unresolved_below_power`, registry. **2026-08-20: the roof-decision half of this row (open vs. closed at the five retractable venues, ARI/ATL/DAL/HOU/IND) was screened** (`docs/roof_decision_screen.md`, 6 cells, all `unresolved_below_power`). Pre-2020 completeness of nflverse's own `roof` field was resolved as a non-issue (measured populated back to 2009 despite a "NEW Feb 2020" changelog note that only marks when the column was EXPOSED, not when the value existed). Home cover at these five venues when open vs. closed: close grade (17 seasons) leans toward the home team covering WORSE when open, +0.0526 pts, P+ 0.8293; but the **opener grade on the 2020-2025 window that matters most for a live decision flips the sign**, -0.0504 pts, P+ 0.2736 -- the two honest gradings of the same mechanism disagree on direction, which per AGENTS.md's "grade at the opener" rule is the stronger reason not to wire this yet (not any crossing-zero objection). A visiting fixed-dome team's cover rate when a retractable venue plays open reads sharply the WRONG way at n=10 (80.0% cover vs. 49.97% complement, P+ 0.0147) -- reported plainly as opposing the hypothesized mechanism at this sample size, not spun as a null (too thin a cell, 1 of 20,000 bootstrap draws dropped for an empty arm, to invoke `wrong_sign_resolved`). A TOTALS-market cell (open roofs under-cover slightly, +0.0351 pts, P+ 0.6363 -- different market, not poolable with the ATS entries) and an exploratory "closed despite a benign forecast" cell (-0.2484 pts, P+ 0.0870, no predeclared direction) round out the family. No live T-90-to-kickoff roof-status source exists in this repo; not wired |
| ENV-03 | ⬜ | Travel geometry | Distance, time-zone change, international games, return travel. **2026-08-19: 8-cell predeclared screen ran** (`docs/travel_rest_battery.md`, `scripts/nfl_travel_rest_battery_screen.py`), built on a new reference table `registry/stadium_coordinates.json` (82 stadium names geocoded, 0 unresolved in the 2009-2025 population, haversine formula sanity-checked against 4 known city-pair distances within 10mi). Unlike the weather batteries, these are pregame-known schedule facts (no forecast-vs-actual leakage caveat). Travel-geometry cells came back near-null: `travel_rest_long_distance_road` (away travel >=1500mi) -0.07pts P+ 0.4231; `travel_rest_eastbound_multizone` (>=2 timezones east) -0.14pts P+ 0.3239; `travel_rest_international_game` (neutral-site, n_flag=61, thin) +0.004pts P+ 0.5129 (CFB's analogous `cfb_bias_battery_neutral_site_designated_home` is resolved-negative at P+ 0.0009, offered as a prior only — NFL does not replicate that shape at this n). Strongest lean: `travel_rest_return_trip_hangover` (home team's own prior-game travel >=1500mi, home_rest<=8) +0.21pts P+ 0.7528 — opposite sign from the predeclared "fatigue hangover hurts the home team" direction, interesting but not resolved (interval crosses zero). All 8 `unresolved_below_power` in `registry/weak_signals.json`, no rotation-registry window spent |
| ENV-04 | ⬜ | Rest context | Bye, short week, mini-bye, consecutive road games. **2026-08-19: 4 of the 8-cell travel/rest battery above are rest-context cells**, deliberately built to avoid re-measuring `bias_battery_extra_rest_edge`/`short_week`/`three_plus_road_games` (already recorded pooled/team-perspective constructs) by using side-specific absolute thresholds instead: `travel_rest_home_off_bye` (home_rest>=13) -0.15pts P+ 0.2079; `travel_rest_away_off_bye` (away_rest>=13) -0.06pts P+ 0.3839; `travel_rest_short_week_road` (away_rest<=5, game-level not pooled) +0.04pts P+ 0.5933; `travel_rest_thursday_pure` (any Thursday game, no weather compounding, distinct from `weather_battery_thursday_outdoor_cold`) +0.13pts P+ 0.7592, the strongest lean in this half. None crosses the 0.85/0.15 lead bar; all `unresolved_below_power` |
| ENV-05 | 🔬 | Weather interactions | Wind × passing/kicking style; heat × pace; surface × unit traits. **2026-08-19 evening**: a predeclared 5-cell second-generation battery (`scripts/nfl_weather_followup_screen.py`, `docs/weather_followup.md`) generalized the raw-temperature-threshold cells above into a continuous away-team climatological-gap mechanism and tested wind × pass-heavy visitor directly. Strongest: `weather_followup_temp_gap_cold_visitor` (away team's own-climate outdoor temp minus this game's temp >= 25F) +0.38 full-slate pts, week-blocked [+0.002, +0.754], P+ 0.976, n_flag=240 — narrowly resolved-shaped but from a mined 5-cell family, so recorded `unresolved_below_power` per AGENTS.md, not promoted. `weather_followup_wind_gap_visitor` (wind-sheltered-climate visitor into wind>=15mph) +0.14pts P+ 0.854. `weather_followup_high_wind_pass_heavy_visitor` (prior-season pass rate above median) came back near-null/slightly negative, -0.04pts P+ 0.369 — interval does not sit entirely on the wrong side, so `unresolved_below_power`, not refuted. `weather_followup_rest_disadvantage_cold` +0.02pts P+ 0.606, near-null. All 5 in `registry/weak_signals.json` |
| ENV-06 | 🔬 | Circadian effects | Test local body-clock hypotheses with aggressive shrinkage |
| ENV-07 | 🚧 | Environmental exposure (air quality, drought) | **2026-08-20** (`docs/environmental_exposures.md`): EPA AQS daily-AQI-by-county and US Drought Monitor weekly county statistics ingested end to end, no-auth, **100% coverage 2009-2025** at each home stadium's county (34 counties via a new FCC-geocoded reference table), joined as-of-Tuesday to 4,842 in-scope domestic games (60 international games flagged, not dropped; 2026 explicitly flagged as stale carry-forward, not real coverage). Two predeclared, non-directional screens ran the same day: high-AQI (Unhealthy-for-Sensitive-Groups-or-worse) outdoor games lean toward the home team covering MORE, **+0.1055 accuracy points, 95% [-0.0378, +0.2541], P+ 0.9264, n_flag=42** (both close-grade eras agree in direction, P+ 0.89/0.79); severe-drought grass-stadium games lean the other way at the full period (-0.2155 pts, P+ 0.1035, n_flag=212) but the two eras themselves DISAGREE in direction (2009-2016 entirely negative, 2017-2025 flips positive) -- reported as a within-family era disagreement, not resolved either way. Both families `unresolved_below_power`; a live 2026 AQI feed needs a free AirNow API key (drought needs no new registration). **Later audit, 2026-08-20** (`docs/drought_monitor_screen.md`): the drought join now enforces USDM's official Thursday 08:30 ET release timestamp with DST-aware leakage tests, and the same opener identity was rerun through the declarative pipeline with `--replace`: **+0.1618 accuracy points, 95% [-0.3903, +0.6789], P+ 0.7193, 64/1,486**, still `unresolved_below_power`, one registry row, no production/prospective promotion |

## Phase 6 — modeling and probability distributions

| ID | Status | Item | Definition of done |
|---|---|---|---|
| MOD-01 | ✅ | Regularized logistic baseline | Time-safe preprocessing and chronological calibration |
| MOD-02 | ✅ | Histogram boosting comparator | Bounded complexity and same evaluation contract |
| MOD-03 | ✅ | Margin regression | Predict conditional mean and compare residual vs market spread |
| MOD-04 | ✅ | Margin distribution | Quantiles or parametric distribution; validate coverage and tails. Coverage re-verified 2026-08-17 on the frozen 2018-2025 artifact: the 50% interval covers 50.21% and the 80% covers 78.94%, both near nominal. **One correctness defect found and fixed in the same pass:** `_three_way_probabilities` tested a CONTINUOUS predictive sample for exact equality with the line, which never fires in floating point, so every published card carried `push_probability = 0.0000` while ~4.8% of integer-line games really push (**9.0% at a line of 3**) and `home_loss_probability` silently absorbed it. The sample is now rounded to integer margins. `home_cover_probability` is deliberately untouched, so no pick moved. The bug survived because the existing test used the `market` target, whose residuals are exact half-integers and made the equality fire by luck; a continuous-residual regression test now pins the realistic path |
| MOD-05 | 🚧 | Joint score/total model | Coherent home/away score and total probabilities. **Premise confirmed, ATS payoff undercut (2026-08-17).** The key-number lattice is real, large and stable: P(|margin| = 3) is 14.58% against 5.29% under a fitted normal (2.75x), |margin| = 7 is 8.76% vs 4.83%, ties are 0.29% vs 2.70% (overtime nearly eliminates them), and unimodality is rejected outright (dip 0.0209, p = 0.000 against both uniform and discretised-normal nulls) with the two dominant modes at exactly +3 and -3. A joint score model would reproduce all of it for free. But for *cover* probability it buys ~0.0004 Brier at the settlement line and **zero** over a plain Gaussian at the actual line, because the ATS residual is one line-varying convolution away from smooth (dip p = 1.000; roughness chi2/df 21.2 for raw margin vs 1.6-1.9 for the residual). Build it for push probability, alternative-line/half-point questions and correct-score products -- **not** for ATS accuracy **Correct-score half BUILT and screened (WP23, 2026-09-01; `docs/score_lattice.md`, `nfl_ats.score_lattice`, `artifacts/score_lattice/20260901T192552Z`).** A joint lattice over integer finals conditioned on the market -- the empirical (margin, total) residual cloud recentred on the guess and interpolated onto the score lattice by a mass-preserving triangular kernel of bandwidth 1 (derived: sum_n max(0, 1-|n-x|) = 1), feasible score set enumerated from the data -- walk-forward 2012-2025, 3,829 games, 299 week blocks. **It does NOT replace the shipped exact-score mode list:** top-1 exact score -0.209 accuracy points, week-blocked [-0.495, +0.077], P+ 0.0615 (`score_lattice_top1_exact`, `unresolved_below_power`); any-of-top-3 -1.175, [-1.636, -0.689], P+ 0.0000 -- whole interval below zero, so `score_lattice_top3_exact` is `refuted_mechanism` / `wrong_sign_resolved`. The frozen rule fired NO and `tiebreaker.py` was not changed. **Two products are resolved positive:** the lattice's MEDIAN total beats the shipped weighted-median closest-total answer by +0.263 total points MAE, [+0.162, +0.366], P+ 1.0000 (`score_lattice_closest_total`) -- a free upgrade to the closest-total half of the tiebreak, unwired only because the pool's tiebreak metric is still unrecorded; and P(push), P(margin = m) and alternative-line answers now exist, best built from the UN-recentred lattice (analytically identical to the mode list on exact scores; calibration gap at |line| = 3 of -0.81pp vs the recentred -3.10pp against a realised 8.93% push rate). Week 1 DEN@KC: P(push at KC -2.5) = 0.00% by construction; 6.83% at -3. No card moved. (WP23, 2026-09-01) |
| MOD-06 | 🚧 | Bayesian dynamic team model | Partial pooling, uncertainty, explicit offseason evolution. **The coefficient-level arm is closed on measurement (2026-08-17), before being built into the pipeline.** Prototyped on 12,206 free CFB games (rule 8, no window spent): sweeping ridge shrinkage across five orders of magnitude moves forced-pick accuracy under a point, and *increasing* it — the direction the partial-pooling argument predicts — makes the thin-training buckets resolvably **worse** (500–999 rows: .5292 at α=300 → .4646 under evidence-maximised shrinkage). Empirical-Bayes/BayesianRidge/ARD as an accuracy play are dead, as is sample-size-scaled α. ~~**The structural reason, which generalises:** the pool metric reads only `sign(predicted residual)`, and rescaling by any positive scalar cannot change a sign — so *any* scheme whose whole effect is to rescale the prediction is a no-op for the primary goal, however well-motivated. It can move calibration and confidence ordering, never a pick.~~ **RETRACTED 2026-08-17: that reasoning is wrong and was being used to reject work.** The production pick is `home_cover_probability >= 0.5` (`pool.py:41`, `backtest.py:56`) — the median of the out-of-time residual sample shifted by the prediction, not the sign of the prediction. The two rules disagree on **11.8% of the 2,075 scored games** (244), and because that sample's median is non-zero, rescaling the centre *can* flip picks. **And the production rule is resolvably the better of the two: 52.05% vs 49.93%, +2.12 points, season-blocked 95% [+0.24, +4.17], `probability_positive` 0.990** — on the 244 disagreements it wins 59.0%. That entire margin is the residual sample's location offset, currently the unweighted empirical median of a ~500-900-draw trailing holdout that no one has ever modelled. See "Where to look next" in `docs/pool_edge_plan.md`. Independently, ridge penalty changes were never in the "rescale" class at all: generalized ridge gives `b_j = d_j·b_j^OLS/(d_j + λ_j)`, so differing `λ_j` rotate the coefficient vector rather than scaling it (measured: block penalties flip up to 18.6% of CFB picks, a global α change 10→1e4 flips 20.1%, a positive rescale flips exactly 0; `docs/groupwise_ridge.md`). **MOD-06's conclusion still stands on its own measurement above — do not reopen it — but never again reject penalty-structure, calibration, or shrinkage work by citing this corollary.** Type-II ML also optimises squared error, so it correctly shrinks away the small directional signal the forced pick lives on: the Bayesian objective and the pool objective disagree. Two things survive: (a) the warm-up cliff was removed by **evidence rather than a model** — see `docs/rotation_registry.md` rule 9, floor now 2011; (b) the one live arm is **unit-level** shrinkage toward a *position prior* rather than toward zero (`players.py` currently multiplies a thin player's value by `career/(career+200)`, i.e. treats a barely-seen player as worth nothing). That changes *relative* feature values and can therefore flip a pick. Needs no new dependency — closed-form James-Stein in numpy; a full sampler is disqualified anyway because it would inject nondeterminism into a pipeline whose methodology rests on exact reproducibility. Screen on CFB at `probability_positive >= 0.75` before any NFL window. **NFL screen ran 2026-08-19** (`docs/mod06_position_prior_shrinkage.md`, `mod06_position_prior_shrinkage`): the single-variable isolation (production `weak_stack` player-value shrinkage swapped from a zero target to a data-derived position/channel prior, everything else held fixed) scored **-0.048 accuracy points, week-blocked 95% [-0.771, +0.672], `probability_positive` 0.4148**, on 2,075 paired games (2018-2025, close grade) — `unresolved_below_power`, no `closing_ground`. `probability_positive` below 0.5 means no EV case today for playing `weak_stack_js_prior` over production; the implementation exists opt-in (`value_shrinkage_target="position_prior"`, `MarginFeatureProfile` `weak_stack_js_prior`) but nothing was wired into `artifacts/active_ats_model.json` |
| MOD-07 | 🚧 | Ensemble/stacking | Out-of-fold predictions only; weight stability constraints. **Promoted 2026-08-18.** The first predeclared vehicle — the weak-signal stack (player value composite + learned availability + three documented opener-bias families, profile `weak_stack`) — took its one registry look on [2020, 2021] at the Tuesday-opener grade (August 2026, `docs/mod07_stack.md`): on 456 paired games the candidate scored **53.29% vs the baseline's 51.32%, +1.97 points, week-blocked [−1.10, +5.00], `probability_positive` 0.8745**, short of the predeclared 0.90 threshold, so the registry verdict is `unresolved` and the window is spent. The entire delta comes from the 49 picks the arms split (candidate 29–20). **The registry verdict did not change and was not the basis for promotion.** Per AGENTS.md a promotion bar is not a decision bar: the pool is forced picks, so declining a candidate that is 87% likely better is taking the other side of an 87/13 bet, and what gets PLAYED is decided on expected value at the grade the pool settles on. Broadened to 1,537 paired games at the same opener grade with the same method and alpha, the candidate scores **52.83% vs the baseline's 52.50%, +0.33 points**, and it was promoted on that basis (commit 68b4dc0). At the **close** on 2,075 games 2018-2025 it scores 51.57% vs 52.05% — a promotion was first refused on that comparison and the refusal was wrong instrument, since the close is the market at its sharpest and systematically understates pool-relevant edge. The predeclared 0.90 threshold still governs what these docs may CLAIM: this is a play decision, not a resolved finding. Prospective 2026 evidence is the next real test. Retuning the stack and re-scoring [2020, 2021] is inadmissible. **2026-08-20: a `weak_stack_v3` candidate (15 new columns across 3 sub-families -- division revenge/sandwich-spot/post-blowout bias, penalty-rate discipline, and two travel/rest flags, all already-recorded `probability_positive >= 0.60` registry leans not yet in production) was built and scored opener-graded against the active `weak_stack` on the same 1,537-game archive** (`docs/weak_stack_v3.md`): **53.03% vs. 53.36%, -0.333 accuracy points, week-blocked 95% [-2.107, +1.467], `probability_positive` 0.3415 -- a real lean against, `unresolved_below_power`, not played.** Close grade and the historical sign rule agree on direction against; Brier/log-loss also lean against. The two strongest deferred candidates by reliability (FluView away-market illness, reliability 0.981; the forecast-weather warm-team/temp-gap cells) were NOT built into this pass -- flagged as the leaner v4 stack worth trying next, per the doc's own "the EV read" section |
| MOD-08 | 🔬 | Distributional boosting | Quantile/NGBoost-style margin and total forecasts. **Undercut on measurement (2026-08-17); any predeclaration must carry this negative.** MOD-08 targets a richer *conditional* density, but the conditional shape is already near-Gaussian once you condition on the line (residual dip test p = 1.000, roughness chi2/df 1.6-1.9 against 21.2 for the raw margin), and ATS-residual sd is flat at 12.6-13.5 across every spread bucket. A fixed unconditional key-number lattice adds nothing over a plain Gaussian on Brier or log loss (head-to-head probability_positive 0.478). There is no shape signal left to condition on. **What IS worth doing is smoothing, not conditioning:** the production mapping inverts an ECDF built from only ~518 residual draws, which quantises every probability and injects ~2.2pp of Monte-Carlo noise per game; replacing it with any smooth CDF is worth **Brier -0.0015 and log loss -0.0032, week-blocked 95% interval excluding zero (P = 0.998)** on 1,802 non-reserved games -- roughly 9x the effect size the MOD-16 CFB screen could resolve. It also removes an accidental -0.77-point median tilt in the ECDF that currently flips 8.8% of forced picks, so it MOVES PICKS and needs its own predeclared window. **PROMOTED 2026-08-19** (`docs/smooth_cdf_mapping.md`). The Gaussian read was first measured close-graded (weak_stack production recipe, 10 non-reserved seasons, week-blocked accuracy +0.684 pts [-0.444, +1.841], `probability_positive` 0.8666, recorded `mod08_smooth_cdf_mapping`, `unresolved_below_power`), then per AGENTS.md ("grade the decision at the opener; a close-graded number may never veto") graded a SECOND time (multiplicity disclosed) on the predeclared 1,537-paired-game opener archive, production pick rule, week-blocked: accuracy +0.133 pts [-1.397, +1.715], `probability_positive` **0.5536** (recorded `mod08_smooth_cdf_mapping_opener`, also `unresolved_below_power` -- the interval crosses zero, never grounds to reject). The frozen decision rule fires PROMOTE at any `probability_positive` above 0.5 (EV rule for forced picks, not a threshold-clearing rule), so it fired despite the weak margin -- disclosed plainly, not smoothed over: this candidate's edge over the raw ECDF concentrates more at the close (P+ 0.85-0.87) than at the opener (P+ 0.55) on every measurement so far, the opposite pattern from MOD-07's own opener-vs-close story. Brier/log-loss stay resolvably positive at both grades throughout. **What changed:** `nfl_ats.outcomes.score_outcome_week` (the sole production weekly-forecast entry point) now defaults `probability_method="gaussian"`; every other caller (`walk_forward_outcomes`, historical backtests, research scripts) keeps the `"ecdf"` default unchanged. `nfl_ats.active_model`'s SYNCHRONIZED-matching identity now carries `probability_method` (mirroring the existing `calibration_method` field, defaulting to `"ecdf"` for legacy artifacts), so a mismatched-mapping forecast can never silently re-activate the wrong evaluation -- both `weekly.py`'s `margin-backtest`/`margin-predict` steps now pass `--probability-method gaussian` explicitly. Pinned by `tests/test_probability_method_promotion.py` (9 tests). The `smooth_cdf_mapping` prospective challenger is retired (status `SUPERSEDED_BY_PROMOTION`, kept not deleted) and superseded by `ecdf_mapping_incumbent` (`nfl_ats.ecdf_mapping_incumbent_overlay`, 14 tests), which now tracks the FORMER production ECDF read as the challenger against its own successor. Read-only check against the currently published Week 1 card (16 games): exactly **1 pick would flip** under the new default -- `NE at SEA`, from `NE +3.5` (49.89% ECDF) to `SEA -3.5` (51.69% Gaussian); the other 15 picks and the Best Pick nomination (a separate alpha=2000 ranking model, out of scope) are unchanged. Not republished -- CURRENT_PREDICTIONS.md and the live `artifacts/active_ats_model.json` are untouched by this session; republishing is the orchestrator's call |
| MOD-09 | 🔬 | Sequence model over drives | Small temporal model, benchmarked against summary features. **Evidence against, 2026-08-17** (`docs/play_level_audit.md`): the premise that play-level rows multiply the training set does not hold — plays are near-independent (ICC 0.013), so the per-game mean is already a sufficient statistic for what they say about a team, and the model still fits on ~4,700 game-level labels however many plays are exposed. The drive layer's summary form already worsened Brier, and a sequence model must beat those summaries while spending far more parameters on the same labels. Do not build ahead of the cheaper noise-reduction work in PBP-05 |
| MOD-10 | 🔬 | Graph model | Player/team matchup graph only after player state is reliable |
| MOD-11 | 🚧 | Calibration suite | Platt, isotonic, and beta are leak-safe and evaluated; calibration-by-regime remains |
| MOD-12 | 🚧 | Hyperparameter protocol | Frozen profile/Ridge/calibration budget selected on prior seasons and scored on next-season folds. **Reopened and then answered on 2026-08-18 (`docs/ridge_alpha.md`): `ridge_alpha = 10.0` is undefended but inert for the metric the pool grades, so it is not worth changing.** Both halves of that are true and neither cancels the other. Measured on the 4,431-game active design: the median principal direction is shrunk **0.274%** (mean 5.50%, weakest decile 15.0%), so the model is unregularised least squares on the signal-bearing bulk. A free 19-point CFB sweep (1e-3 to 1e5, 12,500 games, no window spent) finds **forced-pick accuracy flat across seven orders of magnitude** — nothing beats 10.0 resolvably, and only extreme over-shrinkage at 1e5 is clearly worse. Brier/log-loss IS resolvable and U-shaped, minimised near **α ≈ 2,000-2,500** with the 300-10,000 plateau beating 10.0 at `probability_positive` 0.75-0.97 — but worth only **+0.0003 Brier (~0.12% relative)**, i.e. calibration, not picks. Recommendation: leave the active model alone and route the calibration gain to the Best Pick ranker, which needs no confirmation window. **Two stale figures corrected here:** the widely-quoted "142 columns, rank 71" was the retired `player` profile; the active `weak_stack` design is **90 declared → 159 transformed → rank 82**, on 4,431 games not 4,630. And the null space is **not** mainly the `diff = home − away` identity — **59 of the 77 lost dimensions come from `SimpleImputer(add_indicator=True)`**, because the ≥3-game warm-up rule gates a whole team-state vector atomically, so 45 of 69 indicator columns are bit-for-bit identical (one bit of information, copied 45 times). Exact duplicates change no prediction at any α; they only split coefficient credit. At α=10 the whole group-penalty axis is a no-op (every accuracy delta < 0.09 pts). ~~It deflates the `player_qb_continuity` re-read: α=1 vs α=10 differ by 0.03% vs 0.27% at the median direction, so the ±1.1033-point swing is a calibration of evaluator noise (coin-flip sd on 997 games is ~1.58 points), not a signal~~ — **RETRACTED 2026-08-18: this is the exact reasoning the reclassification refutes.** 1.58 points is an UNPAIRED coin-flip sd; the ±1.1033-point verdict is a PAIRED delta, and the two are not comparable. The 0.03%/0.27% shrinkage argument belongs to a genuinely near-null-contrast SEPARATE alpha-only comparison (`base_alpha1` vs `base_alpha10`, both feature sets held fixed), which flips only 25/997 picks (2.51%) and is resolvably negative on its own. The feature-set contrast that actually carries the ±1.1033 swing has BOTH arms fixed at alpha=1 and differs only in features: it flips **177/997 picks (17.75%, split 94-83)**, paired SE 1.33 pts, MDE80 3.74 pts, `probability_positive` **0.796** (`registry/weak_signals.json`, `player_qb_continuity_matched_alpha`) — a real, if unresolved, lean, not noise |
| MOD-13 | 🚧 | Missingness audit | **Diagnostic and Stage 2 recorded (WP15, 2026-09-01/02):** `scripts/missingness_audit.py` + `docs/missingness_audit.md` measured 7 of 90 production `weak_stack` columns as one source-era lineup-continuity family (62 sporadic, 21 complete); the 2026 Week 1 extrapolation-risk list was empty, and `tests/test_missingness_guard.py` re-checks the live lock. The predeclared candidate retained those seven values, replaced their seven implicit imputer indicators with one explicit source-availability flag, and was evaluated marginal to production on rotation-assigned opener [2020, 2021]. **Opener decision:** 456 games / 35 weeks, candidate and production both 53.7281%, delta +0.0000 accuracy points, week/season intervals [0.0000, +0.0000], `probability_positive=0.000`; no production/card change. Close secondary: 524 games, same +0.0000. The implemented target-leak control detected +43.8596 points (P+ 1.000) but was selected at implementation time rather than the literal predeclared source-availability oracle and did not establish candidate-sized sensitivity. Accordingly the initial terminal entries were corrected through both recorder CLIs: weak key `mod13_source_availability_on_production_opener` and rotation family `missingness_availability_flags` are **unresolved/open**, no closing ground; the window remains spent. Prediction-level outputs and the full correction note are in `docs/missingness_audit.md`. |
| MOD-14 | 🔬 | Era weighting | Compare rolling training windows and time-decayed sample weights. **Screened, not resolved (2026-08-19, `docs/era_weighting_screen.md`).** A predeclared six-arm grid (exponential season-decay half-lives {2, 4, 8, 16} vs. rolling windows {6, 10} seasons, all vs. a uniform-weight baseline, `ridge_alpha=10.0` unchanged throughout) ran on two independent instruments: the free 12,500-game CFB XLG-03 benchmark (primary screen) and the real production `weak_stack` recipe on 2,047 non-reserved-season NFL close-graded games (secondary, gated on the CFB lean). Both instruments' self-checks reproduce their frozen baselines bit-for-bit before any candidate arm was trusted. **`half_life_8` (8-season exponential decay) is the strongest arm on every one of six cuts across both instruments** — CFB clean-core week-blocked **+0.347 pts [-0.180, +0.863], P+ 0.8987**; NFL week-blocked **+0.684 pts [-0.542, +1.938], P+ 0.8505**, NFL season-blocked **P+ 0.9533** (lower bound -0.096, one hair from excluding zero) — and never resolves negative on any secondary metric. Two disagreements worth carrying forward rather than smoothing over: `half_life_2` is resolved WORSE on NFL Brier/log-loss (week-blocked Brier -0.0044 [-0.0074, -0.0015], P+0.001) despite a positive accuracy lean, and `rolling_6` is resolved WORSE on CFB Brier/log-loss/margin MAE/RMSE (all four P+ <= 0.0033) — accuracy's coarse ~2-point resolution and the continuous metrics disagree, the same pattern `docs/ridge_alpha.md` found. No arm resolves under the binding crossing-zero taxonomy (every accuracy interval contains zero); all twelve arm x league combinations record `unresolved_below_power` in `registry/weak_signals.json` (`era_weighting_cfb_*`, `era_weighting_nfl_*`). This clears the project's 0.75 CFB screen bar for `half_life_8` but not the 0.90 promotion-claim bar, so no production or rotation-registry change was made here — a future session wanting to promote it needs its own predeclared, opener-graded NFL confirmation window |
| MOD-15 | ✅ | Temporal schedule-graph ratings | Leak-safe PageRank/HITS and ridge/SRS comparator completed; graph selected in 0/8 outer seasons and was not promoted (underlying artifact not retained — prose-only record, see `docs/closure_audit.md` §3) |
| MOD-16 | ✅ | Conditional margin variance | Replace the pooled residual distribution with game-level heteroskedasticity; accepted only if held-out cover/push/loss calibration beats the pooled baseline. The predeclared CFB screen (August 2026, `docs/margin_variance.md`) **failed**: a Ridge scale model on mismatch/total/pace/experience made clean-core cover log-loss resolvably worse (week-blocked [−0.00056, −0.00012]) — the pooled distribution is already near-correctly calibrated. Only a genuinely NFL-specific variant (QB/backup status, weather) with a new ledger-aware predeclaration remains admissible, and it is deprioritized by this result |

## Phase 7 — simulations

| ID | Status | Item | Definition of done |
|---|---|---|---|
| SIM-01 | ✅ | Margin Monte Carlo | **Implemented 2026-09-02** (`src/nfl_ats/margin_simulation.py`, `docs/margin_monte_carlo.md`): deterministic residual resampling from a fitted margin model, with latent and integer-settled draws retained for audit; repository-compatible two-way and three-way ATS probabilities; integer-line push handling; and a fail-closed guard requiring every target game to occur strictly after the model training cutoff. This is simulation plumbing only—no experiment was scored and no wagering action exists. |
| SIM-02 | ⬜ | Drive-level simulator | Possessions, field position, outcomes, pace, game-state behavior |
| SIM-03 | 🔬 | Player availability scenarios | Mixture of lineups propagated into margin distributions |
| SIM-04 | 🔬 | Full play-by-play simulator | Play call, outcome, clock, penalty, turnover, fourth-down policy |
| SIM-05 | 🔬 | Counterfactual simulator | Compare coaching decisions, injuries, weather, and matchup changes |
| SIM-06 | 🌙 | Differentiable football environment | Learn policy/state transitions jointly without sacrificing auditability |
| SIM-07 | 🌙 | Multi-agent tactical model | Personnel and scheme interaction below the play level |

Simulation is accepted only if it improves held-out distribution calibration
over simpler margin models. More realism is not automatically more accuracy.

## Phase 8 — paper portfolios and decision science

| ID | Status | Item | Definition of done |
|---|---|---|---|
| BET-01 | ✅ | Fractional Kelly | Correct American-odds formula and bankroll compounding |
| BET-02 | ✅ | Exposure constraints | Per-bet and per-week caps; simultaneous weekly sizing |
| BET-03 | ✅ | Drawdown analytics | Peak, trough, maximum drawdown, turnover, risk of ruin proxy |
| BET-04 | 🚧 | Probability uncertainty haircut | Size using conservative posterior/lower confidence probabilities |
| BET-05 | ⬜ | Correlated portfolio sizing | Account for shared teams, totals, weather, and market factors |
| BET-06 | ✅ | Monte Carlo bankroll paths | Distribution of terminal wealth/drawdown, not one realized path |
| BET-07 | ⬜ | Policy comparison | Flat unit, confidence tier, quarter-Kelly, risk-constrained Kelly |
| BET-08 | ⬜ | Transaction realism | Limits, stale lines, unavailable books, pushes, price changes |
| BET-09 | ✅ | Responsible-use controls | Paper mode default; prominent limitations and no auto-wager path **Done 2026-09-01 (WP12):** guarded in code -- `tests/test_no_wager_path.py` asserts no `pyproject.toml` dependency matches a sportsbook/exchange wagering-client denylist, no wager-placement verb exists in `src/`/`scripts/` outside the three read-only quote modules (`nfl_ats.odds`, `odds_backfill`, `market_data`), and `docs/responsible_use.md` states the paper-only architecture. Measured: zero wager-placement code and zero wagering-client dependencies today; the guard keeps it that way. (WP12, 2026-09-01) |

## Phase 9 — football pools

| ID | Status | Item | Definition of done |
|---|---|---|---|
| POL-01 | 🚧 | Pool rule configuration | ATS, straight-up, confidence, survivor, scoring, entry count **2026-09-01 (WP12):** `PoolRules` (`src/nfl_ats/pool_workbench.py`) now composes the forced-pick ATS format with the per-game deadline rule (`deadline_for`, importing `pick_refresh.pick_deadline`), the tiebreak rule (final score of the week's last game), `cards_per_season` and `grading_line`, all source-cited and tested (`docs/pool_rules.md`). Remains open only because the straight-up/confidence/survivor/entry-count variants do not apply to this project's actual pool (ATS forced picks only). (WP12, 2026-09-01) |
| POL-02 | ✅ | ATS weekly card | Force a pick for every game and rank confidence |
| POL-03 | ✅ | Straight-up probability model | Separate calibrated target, never reuse cover probabilities |
| POL-04 | ❌ | Pick-popularity input | **Closed for lack of data, not effort** (`docs/pool_format_levers.md`). Splash does show a pick distribution, but it unlocks game-by-game as each game kicks off — a deliberate integrity measure, so it is structurally unavailable before ~~the Tuesday lock~~ **each game's real deadline (owner-corrected 2026-08-20: there is no Tuesday pick lock, only a Tuesday LINE lock; refined 2026-08-20: the real per-game deadline is min(kickoff, Sunday 16:00 ET), SNF/MNF locking early — this row's premise should be re-examined, since our own pick for a later game in the week could in principle be informed by an earlier game's already-unlocked pick distribution; not re-assessed here, flagged only)**, and there is no API. The Odds API sells no betting percentages; no free ticket/handle feed offers an API plus history. The only obtainable proxy is the favourite flag, already on the card — and our picks run 54.8% favourites / 46.7% home, so we sit near-neutral against a favourite-loving field anyway |
| POL-05 | 🔬 | Contest utility optimizer | Simulator built and validated against five closed forms (`nfl_ats.pool`, `tests/test_pool.py`, `scripts/pool_levers.py`); independence measured rather than assumed (weekly variance of the correct share is 1.036× binomial over 107 weeks). Results: the format already multiplies a fair share of first place by 2.3×–12.7× at 52.5%; **deliberate contrarianism loses monotonically at every field size** when it costs the full edge (break-even ~2.5 accuracy points per flip), and under a top-15% prize — Splash's usual structure — the lever is neutral at best. A Best Pick ranker is worth +2.4 pp of P(first) at 100 rivals if the recorded +8.7 were real, and **+0.07 pp at the honest +0.9**. **Two accuracy points are worth +11.8 pp.** The format is a multiplier to protect, not a lever to pull. Open only on two observables — the pool's real field size and prize structure — both recordable in Week 1 |
| POL-06 | ⬜ | Multi-entry diversification | Correlated entries with controlled overlap |
| POL-07 | ⬜ | Survivor planner | Current survival probability plus future team opportunity cost |
| POL-08 | 🔬 | Opponent-field simulation | Simulate standings and strategic picks for winner-take-all pools |
| POL-09 | 🚧 | Best Pick ranker | The pool pays one Best Pick per week and our confidence ordering is flat (top-\|residual\| scored 48.6% over 107 weeks). Three signals were predeclared and screened once on the registry window [2013, 2015] (August 2026, `docs/best_pick_ranker.md`): `calibrated_probability` (−8.16 points vs all-pick, `probability_positive` 0.0925) and `key_number_distance` (−6.19, 0.170) both made the weekly top-1 pick **worse** and are closed; `sweep_robustness` scored +5.57 points at 0.7955, clearing the predeclared 0.75 screen gate. Its earned opener confirmation then ran once on [2020, 2021] with the deployed `player` profile: **top-1 60.0% (21/35) vs 51.32% all-pick, +8.68 points, week-blocked [−7.00, +22.88], `probability_positive` 0.865** — clears the predeclared 0.75 gate; ~~verdict `confirmed`~~ **verdict `unresolved`** (downgraded 2026-08-18, see the re-read below), both windows now spent. Consequence per the predeclaration: **use `sweep_robustness` to choose the Best Pick in 2026** — a pool-play decision, no activation and no model change. Two disjoint windows, two grades, same direction; but 86 top-1 picks total and a rank correlation of +0.067 (p=0.099), so expect regression toward the all-pick rate. **Re-read 2026-08-17 (`docs/pool_format_levers.md`): every recorded figure reproduces exactly, but the confirmation is mostly the TIE-BREAK, not the signal.** `sweep_robustness` is a half-point width censored at 8.0, so weeks routinely tie at the top and `select_best_pick` breaks ties alphabetically by `game_id`. **24 of the 35 confirmation weeks were ties** (39 of 51 on the screen). Scoring the same signal under a random tie-break gives **52.24%, delta +0.92 points — not 60.0% / +8.68**; the recorded value sits at the 96th percentile of the tie-break distribution. On the screen window the tie-break pushed the other way (+9.05 rather than the recorded +5.57), so *direction* survives but the "two windows, same direction, both clearing" argument is partly alphabetical coincidence. Also unmeasurable by construction: top-1 standard error is **8.45 points on 35 weeks**, so anything in [43%, 62%] is indistinguishable from an arbitrary nomination, and resolving a 5-point effect needs ~384 weeks ≈ 21 seasons. **This re-read re-scores nothing, but it DOES change the registry verdict**: `best_pick_ranker_opener`'s `[2020, 2021]` window is downgraded `confirmed` → `unresolved` (2026-08-18; tie-agnostic delta +0.92 points, honest `probability_positive` **0.536-0.554** under D2 widening — never near the predeclared 0.75 gate). Keep using the signal to choose the weekly Best Pick regardless — it is still the right card to play, since both alternatives (`calibrated_probability`, `key_number_distance`) are measured negatives and the pool is forced picks — but budget the edge at +0.9 points, not +8.68. The published card and dashboard now disclose a tie whenever one occurs. **Promotion moved the tie onto the card that is actually played.** 2026 Week 1 on the active `weak_stack` card IS tied: `2026_01_ARI_LAC` and `2026_01_WAS_PHI` both score 8.0, next 5.5, so the published nomination of ARI is the alphabetical `game_id` tie-break rather than a lean (verified 2026-08-18 on `artifacts/margin_predictions/2026-week-01-20260818T013139Z`, filtered to `market_residual`). The retired `player` card was the untied one, at 7.0 with 6.5 next; the earlier note here had the two cards the wrong way round. ~~**The disclosure claim above is true only of the dashboard** — `dashboard/app_pages/picks.py` computes the tie note, `publishing.py` has no tie logic, so the tracked `CURRENT_PREDICTIONS.md` presents an arbitrary nomination as if it were a lean. Being fixed~~ **Fix shipped and tested 2026-08-18** (`docs/week1_readiness.md` §2): the shared computation was extracted into `nfl_ats.best_pick.best_pick_tie_count`/`best_pick_tie_note`, `publishing.py` now threads the tie note through `_forecast_best_pick` → `_publication_context` → `_publication_header`/`_best_pick_note`, and `publish_active_predictions`'s return payload carries `best_pick_tied: bool`. Proven by `test_published_card_discloses_a_tied_best_pick` and `test_published_card_does_not_disclose_an_unambiguous_best_pick` (`tests/test_publishing.py`), and verified against the real, live, tied Week 1 card via a redirected `--destination`/`--readme` run that touched no tracked file. **The tracked card itself, `CURRENT_PREDICTIONS.md`, is deliberately left stale** — republishing it competes with the ledger-contamination decision, so regeneration is deferred pending the `docs/week1_readiness.md` §owner-decision item on the 16 contaminated CLV ledger rows, not done as part of this fix. **2026-08-18 (owner decision): the weekly NOMINATION rule switches to a measured winner** — a same-day screen (`scratchpad/bestpick_opener/`, `scripts/best_pick_opener_ranker_eval.py`) found the alpha=2000 candidate probability's distance from 0.5, restricted to that week's below-median cross-book opener dispersion (fallback to the full week on missing/degenerate data), +3.92 points vs its unfiltered parent, `probability_positive` 0.813, interval [−3.92, +11.76], 102 paired weeks — third reuse of the same 107-week population, still unresolved, but the strongest lean measured in either screen and EV-positive against the now-signal-free `sweep_robustness` incumbent on a forced pick. Sides never change, only which game is nominated. New module `src/nfl_ats/best_pick_nomination.py` (`best_pick.py` stays frozen); the production tie-break composes chooser 6's filter with a SEPARATE chooser's dispersion tie-break, a combination never itself scored (flagged in both the module and `docs/best_pick_ranker.md`). 2026 scores both arms: v1 via the existing `is_best_pick` ledger flag (unchanged), v2 via a new `best_pick_nomination_v2` prospective challenger (`artifacts/prospective/challengers.json`). See `docs/best_pick_ranker.md` § "2026-08-18: the weekly NOMINATION rule switches" |
| POL-10 | 🚧 | Prospective 2026 evidence | Grade the active model and every frozen challenger on games nobody has looked at, at BOTH grades (the recorded/opener line primary, close secondary). Built 2026-08-17, three weeks before Week 1 locks (`docs/prospective_evidence.md`), because prospective scoring is the only way to settle `mod07_weak_signal_stack` (unresolved, P+ 0.8745) and firm up `best_pick_ranker` (60.0% on 35 top-1 picks) **without spending either of the two remaining opener windows** -- and it only produces evidence if picks are recorded before kickoff every week from Week 1, so a season that goes unrecorded is simply gone. Three gaps closed: nothing computed whether a 2026 pick WON (the MKT-04 ledger scored line movement only, never joining `result`); the weekly Best Pick was recomputed at render time and overwritten every publish, so Week 1's nomination would have ceased to exist once Week 2 published; and the MOD-07 stack was not registered, so no challenger card was being produced at all. Now: `nfl-ats prospective-score` settles at both grades reusing `clv.pick_correct` so FND-04 push semantics are literally the same function; `is_best_pick` persists in `PAPER_DECISION_COLUMNS`, written only while EVERY game of the week is still ahead, first-write-wins, exactly one per week enforced on read; challengers resolve by configuration fingerprint rather than directory recency (the baseline and challenger share an artifacts directory, so newest-wins would have silently recorded the baseline's picks as the challenger's); and `weekly-run` gained four optional trailing steps whose failure never aborts the fail-closed publish. **Anti-backdating is enforced twice** -- refused at write, and re-asserted at scoring, so a hand-written row cannot be laundered into evidence by running the scorer. A Week 1 challenger card was generated as a rehearsal and the two arms disagreed on 3 of 16 games. **The rehearsal rows were then reset (2026-08-17)** so that Week 1's first ledger write is the real Tuesday-lock card on 2026-09-08 — the MKT-04 ledger anchors each game at its FIRST publication, which would otherwise have scored August's picks at August's lines instead of what was actually entered. **2026-08-19: the challenger ledger grew to 8 ACTIVE_PROSPECTIVE entries** — `mod07_weak_signal_stack`, `hc_year_one_fade_overlay`, `best_pick_nomination_v2`, plus five new frozen pick-level overlays built this session at zero window cost: `injury_value_lost_tilt_overlay` (parameter-free sign tilt on the registry's strongest lean, +1.32 pts P+ 0.8875), `division_revenge_tilt_overlay` (both grades lean the revenge side, P+ 0.88/0.86), `backup_qb_fade_overlay` (backup side under-covers at both grades, P+ 0.17/0.10), `surface_switch_tilt_overlay` (cross-league-replicated surface lead, see ENV-02), and `spread_gap_zone_fade_overlay` (model picks at |line| 7.5-10 hit 46-48% on three separate windows incl. a never-mined 2011-2017 replication at P+ 0.914; bounds frozen before the replication). Each is tracked independently against the un-flipped active card in `_cmd_publish_predictions` (8 additive fail-open blocks); none touches the production pick path; all eight ride the same **`--record-decisions` requirement on the Sep 8 lock-day run**. **2026-08-20: challenger ledger grew from 11 active to 16 ACTIVE_PROSPECTIVE (19 entries total; measured, `artifacts/prospective/challengers.json`).** Four new this session, all dual-tracked, no-window-cost: `interim_hc_first_game_tilt_overlay` (see PER-07), `forecast_weather_kn_warm_team_cold_late_tilt` and `forecast_weather_kn_precip_high_total_tilt` (see ENV-01), `injury_signal_refresh_tilt` (below). The Sep 8 lock-day run must adjudicate all 16, not the 8 the previous session's note describes. **Two research docs sharpened WHERE the already-wired observed-movement policy's prospective edge concentrates** (`docs/movement_attribution.md`, `docs/opener_error_analysis.md`), building on `docs/observed_movement_channel.md`'s production `refresh-picks`/Sunday-realism rule (committed `9d70817`, itself not yet described in this file's phase tables -- flagged, not backfilled here). Attribution: of 494 games where Tuesday-to-close movement disagreed with the model's opener pick, injury news explains the largest share -- following the market on the 123 games with a pregame skill-position injury asymmetry AND `\|open_move\| >= 1.0` is worth **+17.07 accuracy points, week-blocked 95% [+0.79, +31.67], P+ 0.976** (a correlated decomposition of the already-recorded `observed_movement_threshold_1_0` entry, not independently poolable with it). Weather-attributed moves lean the OPPOSITE way (P+ 0.11-0.19, thin n=24-39). About 42-48% of the disagreement GAMES show no visible cause in these three archives, but those games carry a smaller share of the total flip VALUE, not half of it -- value-weighted, the unattributed share is ~19% unfiltered and ~36% at the threshold cut, where it still scores positive rather than zero (+8.20 pts, P+ 0.797, n=122) -- the market knows things outside these three archives too, just less of it than a naive game-count split would suggest. `injury_signal_refresh_tilt` operationalizes the injury-attribution finding as a live challenger reading Wed/Thu/Fri practice-status filings. Separately, a 12-family mined battery of the active model's own opener-grade errors (`docs/opener_error_analysis.md`, 24 cells recorded) found and CORRECTED a label-swap bug: an apparent contrarian "fade the market when it confirms our pick" lead was backwards -- read correctly, the model does BETTER when observed movement agrees with its pick (+3.60 pts) and WORSE when it disagrees (-5.99 pts, whole week-blocked CI below zero), which is the *same* mechanism the production movement policy already exploits, not a new contrarian overlay; the corrected finding sharpens where that policy's edge concentrates rather than adding a new one. The battery's strongest surviving lead is a big-spread dampener: the model loses accuracy at opener spreads of 10+ points (-7.91 pts, whole week-blocked CI below zero, P+ 0.020) and 7-9.5 (-4.38 pts), while the two smallest spread buckets are both positive -- a candidate Best-Pick-eligibility discount, not yet built |
| POL-11 | 🚧 | Observed-movement late-week pick refresh | **Built and wired live 2026-08-20** (`docs/late_week_refresh.md`, `src/nfl_ats/pick_refresh.py`, CLI `nfl-ats refresh-picks`) -- a production feature with no prior row in this file, flagged rather than silently added by two independent passes this session. Owner correction 2026-08-20: pool picks are editable up to each game's own kickoff; only the grading LINE freezes at the Tuesday-noon lock -- a previously-unspent structural edge, since a Friday injury designation, a kickoff-nearest weather forecast, or simply a fresher model run can inform a pick the frozen Tuesday line never had a chance to price. Four named refresh passes (Thursday pre-TNF, Saturday, Sunday morning before 4:00pm ET) recompute the active model's probability with current features but always GRADE against the original Tuesday card (read from the paper-decision ledger, never rewritten in place); the real per-game deadline is `min(kickoff, that week's Sunday 16:00 ET)`, so SNF/MNF picks lock early with the rest of the week. One market-based decision rule IS applied to the actual played pick, distinct from every other pick-level overlay (those stay challenger-tracked only, never touching the real pick): if the currently-captured line has moved `>=1.0` point from the frozen Tuesday line, the refreshed pick follows the market's side; otherwise the model's own recomputed pick stands. **The evidence base** (`docs/observed_movement_channel.md`, 6 predeclared cells, week-blocked bootstrap seed 20260819, paired against the production pick on the same 1,503-game opener archive): `observed_movement_threshold_1_0` (Tuesday-to-close, full slate) +1.863 pts, 95% [-0.469, +4.267], P+ 0.935; `observed_movement_oracle_full_slate` +1.730 pts, [-1.110, +4.564], P+ 0.883; the deadline-respecting Sunday-morning-realism variant (2023-2025, n=799 -- a measured LOWER bound, since the `intraday_hourly` archive's real coverage ceiling is ~10:55 ET Sunday, well before the true 16:00 ET deadline) reads stronger: `observed_movement_threshold_1_0_sunday_am_realism` **+3.254 pts, 95% [+0.251, +6.267], P+ 0.981**. All six entries `unresolved_below_power` -- no positive control has been run for this channel (so none is `bounded_by_control`), and no whole interval sits below zero (so none is `wrong_sign_resolved`). Per AGENTS.md's "a promotion bar is not a decision bar," this is an EV decision, not a claim the channel is resolved: `probability_positive` 0.935-0.981 clears the standing forced-pick EV bar, so the policy is wired and executing, not merely proposed. The 1.0-point threshold is the stronger of the predeclared {0.5, 1.0} grid at both readings, fixed before either number was seen, not tuned after the fact. **2026-08-20 same-day follow-up sharpens WHERE this edge concentrates; it does not re-decide whether to play it** (full figures under POL-10): `docs/movement_attribution.md` attributes most of the Tuesday-to-close disagreement games to injury news -- the ≥1.0-threshold, injury-flagged subset reads +17.07 pts, P+ 0.976, a correlated decomposition of `observed_movement_threshold_1_0`, not independently poolable with it -- and wires a dedicated `injury_signal_refresh_tilt` challenger that front-runs the Wed/Thu/Fri practice-report signal instead of waiting for the close itself to move. `docs/opener_error_analysis.md` independently confirms the same mechanism after fixing a label-swap bug: the model covers 56.96% when observed movement already agrees with its pick vs. 47.37% when it disagrees (whole week-blocked CI below zero on the disagreement side, but this describes where the active model underperforms, not a beat-the-baseline candidate, so still `unresolved_below_power` by design). Deliberately deferred, not built: wiring any research overlay (injury value-lost, division revenge, etc.) into the refresh's own decision path, and settling `final_pick_per_game` against real outcomes inside `prospective-score` (needs a real week of refresh data first) **Inactives (T-90) channel, 2026-09-01 (WP5 design + WP17 build):** feasibility study `docs/inactives_channel.md` -- 238/272 (87.5%) of 2026 REG games are playable under `pick_refresh.pick_deadline` (SNF/MNF structurally excluded; the Sunday 16:05-17:00 ET slot IS playable, computed not assumed); `nfl.com/inactives/` chosen as primary source (robots clear; mirrors the `/injuries/` scraper), RotoWire fallback; ESPN's JSON API and nflverse confirmed NOT to carry a T-90 inactives feed; `snap_counts.parquet` zero-snap rows are a leak-free grading label for a backtest but never a pregame feature. Built: `src/nfl_ats/inactives_capture.py` + `scripts/capture_inactives.py` (point-in-time snapshots under `data/players/inactives/<stamp>/`), seven `inactives_*` scheduler rows at each slot's T-90 window, 18 tests + 7 scheduler pins. Caveat stated plainly: nfl.com served only its preseason placeholder all session and Wayback was unreachable, so the populated-page parser is inferred by analogy to `/injuries/` and exits non-zero with `empty_reason="unrecognized_page_structure"` the first time a real page does not match. Not done: wiring the capture into `refresh-picks` (the existing `refresh_sun` 10:00 ET pass fires before both Sunday inactives windows) and the predeclared Section 5 experiment; no window spent. (WP17, 2026-09-01) |
| POL-12 | 🚧 | Tiebreaker score guess | The pool breaks ties on the final score of the week's LAST game (owner, 2026-09-01; Week 1: DEN @ KC, Monday). Shipped same day: `nfl-ats tiebreaker` (`src/nfl_ats/tiebreaker.py`, `tests/test_tiebreaker.py`) — market-implied score from the freshest local odds snapshot (median across books, totals captured every run since `odds-ingest --markets spreads,h2h,totals`), calibrated against all 4,630 lined finals 2009–2025 via a widening (margin, total) neighborhood; reports the closest-total-optimal median guess AND the neighborhood's most common exact finals, because the pool's tiebreak metric is not recorded anywhere — capture it in Week 1 alongside POL-05's field-size/prize observables. Honest error bars printed with every guess (measured 2026-09-01): market total MAE 10.5 (median 9.0, bias +0.5 — actuals run half a point over), implied per-team score MAE ~7.4. **Open: the owner-anticipated over/under training regime — design FROZEN 2026-09-01 in `docs/totals_model.md`** (predeclared before any totals fit: market-residual target, explicit 40-column pregame allowlist from the canonical table, production's exact impute→scale→Ridge(10) pipeline, expanding walk-forward min-500, MAE blend sweep with `clv.week_blocked_bootstrap`, registry recording, `TOTALS_RESIDUAL_WEIGHT` wiring into the tiebreaker). Execution is one session of work; owner confirmed queued-not-now (2026-09-01). The model's margin blend already ships: the guess margin = market + 0.2 × the active model's disagreement, the 0.2 measured on 1,537 opener-graded games (`tiebreaker.MODEL_RESIDUAL_WEIGHT` docstring carries the sweep) — the raw model is WORSE than the market as a point estimate (MAE 10.00 vs 9.91) despite beating it on sides, which calibrates expectations for the totals regime too **Regime RUN (WP1, 2026-09-01).** `docs/totals_model.md` executed as frozen: `src/nfl_ats/totals.py`, `nfl-ats totals-backtest`, `tests/test_totals.py`, one backtest (`artifacts/totals_backtest/20260901T184010Z`), one registry entry, the tiebreaker wiring. Measured on 3,935 walk-forward REG games 2010-2025: the raw model total is WORSE than the market total (MAE 10.5495 vs 10.4249), replicating the margin axis; the MAE-minimising blend `total_line + 0.1 * predicted_residual` is worth +0.0008 total points, week-blocked 95% [-0.0062, +0.0077], `probability_positive` 0.583 over 261 week blocks -- `unresolved_below_power`, recorded `totals_market_residual_blend` (family `totals_market_residual`, units `mae_improvement`). `tiebreaker.TOTALS_RESIDUAL_WEIGHT = 0.1` is derived from that sweep; playoffs reported separately (188 games, +0.0237), never pooled. **Wave 2 (WP18, same day):** wave 1's 41 columns plus 24 drive-pace columns from `game_features_pbp.parquet` (`docs/totals_model_wave2.md`, positive control P+ 1.000): wave 2's blend beats wave 1's on a paired per-game basis by +0.0020 MAE points, week-blocked 95% [-0.0024, +0.0063], **P+ 0.8235** -- the EV favourite; recorded `totals_market_residual_wave2_vs_wave1`; wiring the tiebreaker's totals view at `totals_wave2` is the proposed next edit (not applied yet). **Tiebreaker neighborhood fixed (WP14, same day):** the totals blend surfaced a half-point-quantization edge -- a +0.042 nudge to the centre dropped the whole 38-game `total_line == 41.5` bucket and moved the published Week 1 guess DOWN two points while the model argued higher. The calibration neighborhood is now kernel-weighted (`max(0, 1 - d)`, `d = sqrt((dmargin/1.0)^2 + (dtotal/1.5)^2)`, bandwidths inherited from the old first window, widened continuously until the Kish effective sample size reaches the same 150 floor; weighted medians and weighted exact-final counts). Measured: the guess now moves monotonically with the line (41 -> 42 -> 43 as the blended line rises 43.00 -> 43.50, versus the old 43 -> 41 -> 43). Consequence stated plainly: the Week 1 DEN@KC card reads **KC 22 - DEN 19** (was KC 23 - DEN 20) at the unblended market total as well, a 2-point difference far inside the 10.5-point market-total MAE. (WP1, 2026-09-01) |

> **2026-09-02 POL-11/POL-12 follow-through (supersedes the stale "not done"
> and "proposed next edit" phrases in the rows above):** `refresh-picks` now
> records T-90 inactives only to its separate challenger ledger, with seven
> post-capture scheduler passes and rejection of stale, future, wrong-week,
> malformed, duplicate, partial, or game/team-misaligned inputs; the Section 5
> ATS experiment remains unrun and no window was spent
> (`docs/inactives_channel.md`). The tiebreaker now serves the validated
> 65-column totals-wave-2 view at its already-recorded 0.1 blend weight, falls
> back to wave 1 only when the whole PBP table is absent, and uses market-only
> when a present table fails identity checks (`docs/totals_model_wave2.md`). A
> live 2026 Week 1 command served wave 2; no card or active-model artifact
> changed.

> **2026-08-20 later update to POL-10:** live registration is now **17
> ACTIVE_PROSPECTIVE challengers / 20 entries total**, measured from
> `artifacts/prospective/challengers.json`. The added
> `best_pick_big_spread_eligibility` rule (`docs/best_pick_big_spread_challenger.md`)
> prospectively excludes 10+ spreads from v2's eligible nomination pool,
> falls back to v2 if that would empty the mandatory weekly choice, and never
> changes the published Best Pick. Week 1 remains MIA +3.5 under both rules;
> the first informative evidence awaits a week where their nominees differ.

> **2026-08-20 lock-day canary follow-up to POL-10:** the publish command's
> 15 live publish-time challengers are now represented by one explicit
> registry-ID-to-result-key map, and a test compares that map with the live
> `ACTIVE_PROSPECTIVE` registry. The audit found and fixed one real response
> gap: the default no-record path omitted the v3 nomination challenger's
> explicit skipped result. Registration drift or an unwired new publish-time
> challenger now fails the CLI test before Sep 8 rather than silently losing
> its first prospective week. The three non-publish-time active challengers
> remain on their separate refresh/prospective-record paths.

> **2026-08-21 production follow-up:** the forced-pick decision now uses the
> exact four-member OR-union found in the overlay-composition audit: coach fade,
> division revenge, player arrests, and spread-gap zone are each evaluated
> against the raw model card; a game is flipped exactly once if any member
> fires. On the 1,503 opener-graded archive games this exact policy scores
> 833/1,503 = **55.4225%**, versus 814/1,503 = **54.1583%** for the former
> production coach-to-arrests chain, a paired +1.2641 accuracy points with
> week-blocked `probability_positive=0.85715`. The archive score is the maximum
> of 127 correlated subsets and therefore selection-inflated; it is not the
> operating expectation or a resolved-effect claim. Publication is fail-closed
> on the production inputs, and all pages consume the composed final card. The
> primary paper ledger freezes all four member flags, the union decision,
> schedule/arrest provenance, and the former-policy side; refresh reuses the
> frozen union rather than reopening Tuesday sources. Active challenger
> `overlay_production_chain_coach_arrest_incumbent` records that exact former
> policy from the same primary row for prospective paired scoring.
> The 2026 Week 1 August preview was regenerated across Markdown and all three
> static pages: the policy changes BAL-at-IND via coach fade and CLE-at-JAX via
> spread-gap, with no overlap; MIA-at-LV remains Best Pick. No ledger row was
> written before the September 8 lock.

## Phase 10 — dashboard and operations

| ID | Status | Item | Definition of done |
|---|---|---|---|
| UI-01 | ✅ | Local Streamlit dashboard | One command, no external service, graceful empty states |
| UI-02 | ✅ | Data health view | Snapshot provenance, coverage, missingness, season counts |
| UI-03 | ✅ | Backtest view | Scorecards, season metrics, calibration, cumulative returns |
| UI-04 | ✅ | Paper bankroll view | Equity curve, drawdown, stakes, settings, ledger |
| UI-05 | ✅ | Prediction view | Weekly probabilities, passes/picks, lines, model cutoff |
| UI-06 | ✅ | Experiment view | Feature-set comparisons and sortable metrics |
| UI-07 | ✅ | Team explorer | Pregame state trends and matchup comparison. **Live 2026-08-25** (`docs/team_explorer.html`, `src/nfl_ats/team_explorer.py`): canonical-schema team-state trends, one-team season views, two-team comparison. Owner review found the first cut rendered raw schema labels with no reader context ("the reader has no idea who He is"); rewritten same day -- visible plain-language primer, per-stat explanations with direction, jargon purged. Reader-comprehension is now part of the acceptance bar for every page. |
| UI-08 | ✅ | Model explanation view | Coefficients/SHAP with stability and caveat labels. **Done 2026-08-25** (`src/nfl_ats/model_explanation.py`, section on ``docs/models.html``): family-level walk-forward coefficient weights from the market-decomposition artifact, each row carrying a refit-to-refit **stability label** ("steady across refits" / "jumps around between refits", declared ratio threshold 0.15) and a four-bucket caveat caption (the ``unpriced_predictive`` bucket reads "unconfirmed", never "edge"), plus staleness tracking against the active manifest's feature-table hash, provenance line (fit window, refits, ridge alpha, artifact UTC timestamp), and four honesty notes. Fail-open both ways: no run → honest empty state; unreadable run → visible warning box. Feature-level coefficients stay in the artifact's own ``coefficients.csv`` by design -- ridge smears weight across correlated features, so the page is family-only and says so |
| UI-09 | ✅ | Pool workbench | **Completed 2026-09-02** (`docs/pool.html`, `src/nfl_ats/pool_workbench.py`, `docs/pool_workbench.md`): the forced ATS entry is editable by side and Best Pick, saves/restores/resets browser-local state under a versioned season/week key, rejects stale/invalid stored values, and never mutates the published forecast. The former placeholder is now a live 50%/65%/85% favorite-share sensitivity table that recomputes with entry edits and is labeled hypothetical throughout because no ownership feed exists. Rules and model confidence ranks remain visible. |
| UI-10 | ✅ | Guided interpretation | Plain-language verdicts, score references, glossary, and historical/live separation |
| UI-11 | ✅ | Active-model synchronization | One atomic manifest links the exact evaluation and weekly forecast used by every headline page |
| UI-12 | ✅ | Question-based navigation | Five plain-language destinations replace internal lab/tool names; researcher diagnostics live under Advanced research |
| UI-13 | ✅ | GitHub weekly card | Synchronized publisher places the full current card at the top of README and in a tracked standalone page |
| UI-15 | ✅ | One dashboard + registry-driven findings | **2026-08-19, owner decision ("why are you maintaining two separate dashboards?").** The GitHub Pages site is THE dashboard. (a) The Streamlit app's duplicated pages (picks, findings, historical record) are deleted; it is now a 3-page internal research console (engine room, model explanation, pool workbench) whose sidebar states the public site is the dashboard. (b) The findings page is generated from the evidence stores at render time: new `src/nfl_ats/findings_registry.py` loads weak_signals + rotation + challengers, every curated finding names its `registry_keys` with content fingerprints and `validate_curation()` FAILS the render when evidence moves under a stale sentence (evergreen methodology entries exempt); a "What we're watching" section renders the top open `unresolved_below_power` leads by \|P+−0.5\| straight from the registry (12 of 143 shown), and the challenger section renders live from `challengers.json` on both findings and History pages. Six stale curated facts corrected in the wiring pass (see `docs/site_content_pipeline.md` for the pipeline contract). (c) `publish-predictions` regenerates the site BY DEFAULT (`--no-board` is the rehearsal opt-out; failure is fail-open and surfaced in the result) — the stale-card episode cannot recur through a forgotten flag; `weekly-run` already passed `--with-board` explicitly. Recording new evidence now updates the page with NO human step; only curated framing needs a human, and only when the story changes |
| UI-14 | ✅ | Public site card fidelity + week board | **2026-08-19.** Two live correctness bugs found by browser inspection: the GitHub Pages generator (`public_board.py`) rendered raw pre-overlay picks — the served site showed BAL and a v1 ARI Best Pick where the published card plays IND (coach-fade flip) and MIA (v2 nomination). Overlay+nomination resolution extracted into shared `src/nfl_ats/card_view.py`, now consumed by `publishing.py` (pure refactor, byte-identical `CURRENT_PREDICTIONS.md` proven), `public_board.py`, and the Streamlit dashboard (whose duplicated copy — and an app-breaking ImportError — were removed). Design pass on the same date: `index.html` gained an at-a-glance week board (kickoff/line/pick/plain-words confidence, ★ Best Pick, ⇄ flip markers, anchors to cards), per-game sweep charts demoted behind `<details>`, a stale-attribution guard (an explanation whose own residual disagrees with the live card by >0.3 pts is dropped rather than shown — fixed a live contradiction on the ATL-PIT card), mobile CSS for all three pages, and an exact season caption (5 of 6 seasons above the coin flip, 2020 exactly at it). The historical row-level ledger now lives on `history.html`; it renders prospective challenger assessments from the settled ledger and keeps pending outcomes private. Deliberately skipped: per-season weekly cumulative strip (data present, scope cut) |
| OPS-01 | 🚧 | Scheduled refresh | Data → features → frozen predictions with idempotent jobs **2026-09-01 (WP10):** `scripts/capture_scheduler.py` gained catch-up semantics: a `Job.catch_up` field (default `False`, behaviour unchanged) lets an idempotent job whose window closed unrun run at the next tick instead of being written off as `MISSED`, recorded as a distinct `CAUGHT_UP` state so `--status` stays honest about the original miss. Set on `backup_data` (whose 2026-08-30 miss motivated it) and on a new `player_arrests_tue` row (Tue 07:00 ET, before `odds_tue_open`) giving `nfl-ats ingest-player-arrests` -- previously unscheduled, feeding the PROMOTED player-arrest policy component -- its own point-in-time capture; verified live (first occurrence self-healed to `CAUGHT_UP`, 56-page / 1,116-row snapshot). Still open: the Tuesday `weekly-run --record-decisions` lock remains a manual command. (WP10, 2026-09-01) |
| OPS-02 | 🚧 | Artifact retention policy | Keep manifests/ledgers, compact or prune large derived files **2026-09-01 (WP6):** measurement + dry-run planner shipped (`scripts/artifact_retention.py` `--report`/`--plan [--older-than-days N]`, `docs/artifact_retention.md`, 34 tests). Measured footprint: `artifacts/` 2.2 GB (435 runs / 145 families), `data/` ~3.5 GB. Protected-set discovery is programmatic (doc/registry citations, the active-model manifest, the prospective/clv_ledger ledgers); the default 30-day plan finds 0 candidates today, 14-day finds 14 (~9.9 MB). No delete mode exists and none should be added without explicit approval. **Gap found:** `artifacts/` -- including the never-prune `prospective/` and `clv_ledger/` -- had zero off-device backup coverage (`backup_data.py --include-artifacts` never run). (WP6, 2026-09-01) |
| OPS-03 | ✅ | Container image | **Implemented 2026-09-02** (`Dockerfile`, `compose.yaml`, `.dockerignore`, `deploy/nginx.conf`, `docs/container_deployment.md`): digest-pinned unprivileged NGINX serving only the four generated public pages, deny-by-default build context, loopback binding by default, read-only root, all capabilities dropped, no privilege escalation, constrained temporary storage, security headers, exact-route allowlisting, and `/healthz`. Static contract tests and `docker compose config` pass; an actual image build awaits a host with a running Docker daemon. |
| OPS-04 | ⬜ | Read-only hosted dashboard | Authentication and no credential/data leakage |

## Phase 11 — ambitious research

| ID | Status | Item | Definition of done |
|---|---|---|---|
| SKY-01 | 🌙 | News-to-availability model | Time-stamped reports mapped to player status with source provenance |
| SKY-02 | 🌙 | Video-derived tracking proxies | Only with lawful data and reproducible computer-vision evaluation |
| SKY-03 | 🌙 | Causal matchup effects | Separate persistent skill from opponent selection and game script |
| SKY-04 | 🌙 | Market microstructure model | Quote arrivals, book leadership, latency, and information diffusion |
| SKY-05 | 🌙 | Automated research agent | Propose experiments but require fixed budgets and human approval |
| SKY-06 | 🌙 | Public forecast archive | Signed pre-kickoff probabilities and long-horizon calibration record |
| SKY-07 | 🌙 | Open benchmark suite | Reproducible point-in-time NFL forecasting tasks and leaderboards |

## Recommended execution order

**Next action, refreshed 2026-09-02 after the lock-path and opener-confirmation
waves.** Protect the 2026-09-08 Week 1 lock and prospective evidence write.

1. **Completed 2026-09-02:** the six Tuesday recorders are automatic,
   `crew_tilt_refresh_v1` is on the late-refresh path, and the verifier covers
   all 29 active challengers with zero pending wiring. The default
   `scripts/lockday_rehearsal.py` is now a static-only wiring audit: measured
   over ten consecutive runs at 2.259-4.079 ms, with 23 publish paths, five
   refresh paths, one weekly-run path, and zero errors. It imports no model
   stack, executes no recorder, and touches no ledger. The old production-sized
   replay is explicit `--full-replay` diagnostics only.
2. On 2026-09-08, run the real lock as `weekly-run --record-decisions`; do not
   create the genuine Week 1 rows early. Read the command's per-recorder result
   JSON and immediately run `scripts/lockday_verify.py` against the real rows.

The historical inactives screen and MOD-13 Stage 2 are now recorded, and neither
changes the played card. Further MOD-13 work requires a new predeclaration with
a candidate-sized positive control; the injury-value follow-up remains gated on
the Week 1 prospective observation. The four formerly queued opener
confirmations are now complete on the rotation-assigned 2020-2021 block (456
paired non-push games / 35 weeks each): Reddit home-comment ratio +0.439
accuracy points, `probability_positive=0.665`; pace mismatch -0.219, P+=0.286;
away-team illness -0.439, P+=0.312; and third-down red-zone reversion -0.658,
P+=0.111. All four are recorded through both registries as
`unresolved_below_power` / `unresolved`; none changes the played card or
supplies an admissible terminal ground.

**Historical graph pointer, carried from 2026-08-26 (Wave 11) and resolved
2026-08-31.** Test schedule-adjusted
sack rate on top of the model we actually play, not on top of a bare market
baseline. In the graph screen it scored **+2.949 accuracy points** against the
bare baseline (`graph_team_stat_off_sack_rate`, week-blocked P+ 0.987,
[+0.401, +5.707]) -- the best single cell of the 38 -- but two things stand
between that number and the card, and neither is a threshold:

1. It was measured as a marginal over a **zero-feature market baseline**. The
   project's own recorded lesson is that a component positive on its own can go
   negative once stacked on the played chain, so the marginal that decides is
   the one measured on top of production (`weak_stack` / `market_residual`,
   ridge alpha 10).
2. Roughly 40% of the +2.949 is the home-tilt measurement artifact documented
   in `docs/graph_ratings_v2_screen.md` section 6: that cell's own within-week
   permutation null centres at **+1.227**, not zero, and the observed value
   sits at the 92.5th percentile of it.

**Measured 2026-08-31 (this item is now resolved as a next-action):** the
on-production marginal was scored under new rotation family
`graph_off_sack_rate_on_production` (window [2014, 2016], 749 games, 51 weeks,
predeclared in `docs/graph_team_stat_on_production.md` before scoring).
Candidate = production `weak_stack` + the frozen graph sack-rate column;
baseline = production alone; full production chain fit on both arms. Paired
delta **-0.935 accuracy points**, week-blocked 95% [-2.625, +0.809],
`probability_positive` **0.122**; the cell's own within-week permutation null
centres at +0.416 and the observed value sits at its 6.0th percentile; a
positive control (column leaked to the realized margin) scores +49.0 pts,
P+ 1.000, so the harness is not blind. On EV grounds the number does not
favour adding the feature to the played chain; recorded
`unresolved_below_power` (registry 605 → 606), NOT closed — the interval's
upper bound is positive and no positive control bounded an effect of the
screen's +2.949 size. The bare-baseline screen reading did not survive
measurement on top of production — the recorded composition lesson, again.
Parent family `graph_ratings_v2_team_stat` was not touched and keeps its
**two eligible close windows** for its own bare-baseline question; the
harness (`scripts/graph_team_stat_screen.py`,
`scripts/graph_team_stat_record.py`) remains available. The hard deadline in
this file remains Week 1 locking **2026-09-08**, which is a separate item (see
item 6 below and `docs/prospective_evidence.md`).

**Historical pointer refreshed 2026-08-31** (read-only registry survey; full
detail in `docs/pool_edge_plan.md`'s "2026-08-31 registry state and next
shots" addendum). With `off_sack_rate` now resolved-as-next-action (above),
the family's own doc names two remaining predeclared on-production
candidates: `graph_team_stat_def_yards_per_play` (95.5th percentile of its
own permutation null, the least artifact-contaminated of the three named
cells) first, then `graph_team_stat_off_rush_epa_per_play` (highest
reliability in the family, 0.987, but only the 53.5th percentile of its own
null). Three more candidates are queued behind those in the addendum's
ranked agenda: a fresh opener-graded `fluview_home_market_elevated`
confirmation on an unspent window, the `injury_value_lost_gradient` /
`_narrowed` on-production test (blocked on the 2026 prospective look landing
first), and one clean confirmation of the Best Pick dispersion-filtered
ranker on a window that is not a fourth reuse of the same 107 opener weeks.

**Historical pointer refreshed 2026-09-01 (afternoon fleet session; every number below is measured that day from the named artifact or registry entry).** Every item the 2026-08-31 pointer named as pending has now run or been resolved:

1. **The graph-on-production line is three-for-three negative, and the reliability premise is what failed.** `graph_team_stat_off_rush_epa_per_play` measured on top of PRODUCTION `weak_stack` (close-graded, rotation-assigned [2014, 2016], 749 games, 51 weeks; `docs/graph_team_stat_off_rush_epa_on_production.md`, artifact `artifacts/graph_team_stat_off_rush_epa_per_play_on_production/20260901T184239Z/results.json`): paired delta **-0.935 accuracy points**, week-blocked 95% **[-1.998, +0.135]**, `probability_positive` **0.037**; candidate 50.07% vs production 51.00%; observed at the 0.5th percentile of its own within-week null. Recorded `unresolved_below_power` under `graph_off_rush_epa_on_production` and the window spent `unresolved`; no card change. With `off_sack_rate` (-0.935, P+ 0.122) and `def_yards_per_play` (-0.668, P+ 0.189) all three cells the 38-family screen carried forward now read negative on production. The family's highest split-half reliability (0.987), replicated in sign on a disjoint opener-graded window (+1.996 pts, P+ 0.828), did **not** predict survival on top of production: reliability measures consistency, not whether the played chain already prices the trait. All three graph on-production families report `remaining_eligible_windows: 0`.
2. **The CFB replication explains the shape of that result.** On the XLG-03 clean core (8,933 graded games, 199 weeks; `docs/graph_team_stat_cfb_replication.md`), the graph column BEATS the raw differential as a single feature on all three available cells (+0.369 / +0.291 / +0.694 accuracy points, week-blocked P+ 0.798 / 0.765 / 0.897, moving 17-26% of picks) -- the first cross-league corroboration of the `graph_input_screen` family -- but ADDED on top of a benchmark that already carries the raw statistic it is worth about nothing (-0.011 / +0.022 / -0.179 points, P+ 0.467 / 0.535 / 0.266, intervals of +/-0.29 to +/-0.69 points, roughly four times tighter than the NFL on-production reading's). Interpretive consequence: the NFL -0.935 / -0.668 readings sit inside the noise of a true effect near zero and should be described as consistent with a null, not as the graph damaging the chain. The graph lane, if pursued, is a REPLACEMENT for a raw team-state input, not another column (queued as WP24 on CFB first). `off_sack_rate` has no CFB counterpart (no sack/pressure column in the CFB table).
3. **`fluview_home_market_elevated` now has three independent negative-leaning reads against one close-graded positive.** The family's second opener window, (2022, 2023), read **-1.751 pts, 95% [-4.501, +0.986], P+ 0.094** (`docs/fluview_opener_look.md` sections 8-9; first window -0.439, P+ 0.341; a doc-only post-hoc union read 2020-2023 -1.134, P+ 0.113), and the CFB replication reads -0.388, P+ 0.200 on 5,671 games (XLG-08). Nothing closes (no interval wholly on one side of zero; trait reliability 0.98 in both leagues), but spending further NFL window on this cell is now a worse bet than it looked; the family has no default-size opener block left.
4. **Best Pick dispersion-filter confirmation remains window-blocked** (`docs/best_pick_followup.md`, "Determination: STOP"); **`injury_value_lost_gradient` remains gated** on the 2026 prospective look landing (Week 1 locks 2026-09-08).
5. **New leads opened by the fleet:** PER-13 Stage 1 met its EV gate (availability Brier +0.0075514, P+ 1.000, placebo-clean) so its Stage 2 on production is warranted; the totals tiebreaker regime ran (POL-12) and wave 2 (drive pace) is the EV favourite at P+ 0.8235; the inactives T-90 channel is designed and its capture is being built (`docs/inactives_channel.md`); the era-magnitude report (`docs/era_magnitude_report.md`) found `bye_overval_home_edge`'s split is already covered by `docs/bye_overvaluation_screen.md` and that the Sagarin family's era heterogeneity is most plausibly its own 2012 coverage hole (0 of 256 games usable), now being fixed.

**2026-09-02 completed-wave status.** The three 2026-09-01 fleet leads are now
handled: PER-13 has its opener confirmation (P+ 0.084; production is the EV
favourite, family unresolved), totals wave 2 is served behind strict identity
guards, and T-90 inactives feed a separate late-refresh challenger. The
inactives Section 5 historical proxy subsequently measured -1.3986 accuracy
points with `probability_positive=0.0418` and remains unresolved because its
large oracle control did not bound an effect of the candidate's size. MOD-13
Stage 2 was pick-identical to production on its assigned opener window (0.0000
points, `probability_positive=0.000`) and also remains unresolved: its
implementation-time target-leak control deviated from the literal predeclaration
and demonstrated only gross sensitivity. The hard operational priority is the
2026-09-08 Week 1 lock/prospective recording; it is a deadline, not a research
window to spend early.

**Graph lane, replacement question (WP24, 2026-09-01; `docs/graph_team_stat_cfb_replacement.md`).** On WP8's CFB window (8,933 games, 199 weeks), swapping a statistic's whole raw `home_`/`away_`/`diff_` triple for its graph katz differential reads **-0.257 / +0.056 / +0.022 accuracy points** (week-blocked P+ **0.145 / 0.573 / 0.528**) at the close and **-0.022 / +0.034 / -0.325** (P+ **0.462 / 0.543 / 0.221**) at the opener, moving 5-11% of picks; substitution beats addition on two of three cells (`off_success_rate` from P+ 0.266 to 0.528 on the identical window). The predeclared bar for opening an NFL replacement predeclaration (a majority above P+ 0.5 at the opener) is **not met (one of three)**, so the NFL spec is written in that doc's §10 and not run. Two by-products: an ablation arm shows each raw team-state triple is worth at most 0.21 points, so a one-metric swap moves into an almost-free slot; and with the raw columns gone the graph column earns its slot on both OFFENCE cells (`off_epa_per_play` **+0.213 pts at the opener, P+ 0.827**; `off_success_rate` +0.123 close, P+ 0.710) while losing it on DEFENCE (-0.190, P+ 0.064) -- the offence-only version is being predeclared and run as a sequential confirmation (WP35). Split-half reliability of the three CFB graph traits is now measured at Spearman-Brown **+0.9929 / +0.9925 / +0.9938** (P+ 1.000, above the raw statistic in every cell), so `no_split_half_reliability` is inadmissible for this family. All three cells `unresolved_below_power` (`graph_team_stat_cfb_replacement`); no card moves on CFB evidence.

The August 2026 build slice completed the fair-margin/market-residual outcome
workbench, empirical margin distributions, current-odds archive, cross-book
consensus, versioned 2009–2025 PBP ingestion, PBP efficiency ablation,
timestamped QB/depth foundation, straight-up pool cards, conservative Kelly
haircuts, conditional bankroll paths, a free versioned historical closing-line
cross-check, and nested rolling-origin model/feature selection. The first
full-history PBP result was negative and is retained. A later nested comparison
found only a small logistic probability improvement and no reliable ATS edge;
this is also retained. A free 2025 opener/nine-book closing sample validated
the line-movement contract, and the purchased 2020–2025 point-in-time archive
has since closed the multi-season quote-timing gap: the frozen close-prediction
pilot has taken its one look (direction of movement replicated out-of-sample;
magnitude no better than a no-movement baseline).
QB promotion is blocked by historical point-in-time coverage rather than by
model code. MOD-12 is complete for the current residual-player workbench; new
candidate families still require a newly frozen budget.

The temporal PageRank/HITS comparison is also complete. Graph candidates were
selected in zero of eight outer seasons; the simpler ridge/SRS schedule rating
was selected three times and the existing market/context model five times.
Season-blocked paired intervals found no reliable improvement, so both remain
research features rather than defaults. The PageRank/HITS screen's underlying
artifact does not survive on disk or in any commit; per `docs/closure_audit.md`
§3 its numbers are an unrecoverable prose record, registered as
`graph_schedule_rating_brier` (`registry/weak_signals.json`,
`unresolved_below_power`) rather than re-runnable evidence.

The first player-family ablation is also complete. The existing 52.05% player
profile was driven mainly by lineup continuity and QB state: injury-only reached
51.28%, continuity 51.95%, and QB plus continuity 52.34% on the same 2,075
games. A value-weighted injury extension using lagged offensive EPA and
defensive disruption per snap raised the full fixed profile to 52.14%, but its
increment over the prior full player profile was only 0.10 points and its
blocked interval crossed zero. Two-season nested profile selection reached
52.47% over 2020–2025 versus 50.88% for fixed base. Its paired accuracy interval
excluded zero under both week and season blocking, while Brier worsened and
2025 scored only 49.82%. Retain it as a refinement lead, not a promoted model;
the completed regularization/calibration gate below is the stronger decision
record, and the completed participation-based player-rating screen is documented below.

The frozen regularization/calibration gate is now complete. Four profiles ×
three Ridge strengths × four calibration policies produced 48 declared
candidates from 12 reused raw prediction streams. The prior-two-season selector
scored 50.70% over 1,582 outer games versus 50.88% for fixed base; its -0.19
point paired change had a week-blocked interval of [-2.48, +2.21] points. It is
not promoted. QB+continuity/alpha-1 led the pooled table at 52.63%, while
full-player/alpha-1/beta reached 52.34% with 0.24965 Brier; both are explicitly
post-grid leads. A separately fixed participation-rating signal was tried next
instead of expanding this grid.

The participation source is now fully ingested: 478,989 rows from 2016–2025.
The declared adjusted-plus/minus candidate used competitive valid 11-on-11
plays, a three-season lag, team effects, Ridge alpha 1,000, and a 500-play
reliability prior. Adding its two injury-value contrasts reduced matched ATS
classification from 52.14% to 51.71% over 2,075 games. The -0.43-point change
had a week-blocked interval of [-1.53, +0.63] points, and probability error
worsened. It is retained as a negative result and will not be retuned on these
seasons. The next player-availability experiment targeted the other half of the
original hypothesis: learn actual play probability from historical report,
practice, position, and next-game snap outcomes instead of assigning fixed
questionable/doubtful weights by hand. That replacement improved the direct
availability Brier score from 0.09500 to 0.09056 over 57,294 out-of-season
player-games. It moved matched ATS classification from 52.14% to 52.24%, only
two additional correct games; the week-blocked change interval was [-0.63,
+0.78] points and 2025 remained 49.08%. Retain it as a promising refinement,
not a promotion. Expected role delivery is next because any-snap participation
treats a one-play appearance as equivalent to a full workload.

Prediction integrity is permanently release-blocking: no modeling or feature
work is allowed to bypass FND-11/FND-12, even when the research code itself
appears to run successfully.

Evaluator performance is also guarded. The outcome bootstrap was rewritten
from per-draw pandas reconstruction to block-level sufficient statistics: the
standard 2,000-draw report fell from roughly 290 seconds to 0.38 seconds while
matching the old saved intervals to floating-point precision. See
`docs/performance.md` for budgets and regression rules.

The deterministic positive-control audit now establishes what those intervals
can and cannot tell us. It reproduced all 2,127 active residual predictions to
`6e-14`, reproduced every cover probability exactly, and recovered the exact
1,080/2,075 classification result. Across eight synthetic pregame signals, a
known 0.5-point-per-standard-deviation effect improved ATS accuracy by 0.78
points on average but cleared the week-blocked interval only three times; a
known 1-point effect improved accuracy by 1.42 points but cleared it only twice.
A 2-point effect improved accuracy by 3.96 points and cleared seven of eight
week and season intervals. Permuted controls produced no false interval clears.
Therefore an unresolved interval is evidence of uncertainty, not proof that a
small effect is absent. Conversely, this audit does not rescue candidates that
moved in the wrong direction or justify searching the same outcomes again.

The August 2026 replication program then spent the untouched pre-2018 windows
on the three strongest reopened leads, with every specification and decision
rule frozen before any run. ~~All three closed.~~ **Two of the three were
reclassified `unresolved` on 2026-08-18** (`registry/rotation_registry.json`,
`scripts/audit_terminal_verdicts.py`) — both negatives sat below the
instrument's own resolving power, not evidence of a null. (1) The raw-PBP
market-residual bundle, whose post-hoc 2018–2025 comparison showed +1.69
points, scored −0.08 points against base on 1,247 never-selected-on 2013–2017
games with margin error resolved worse; the +1.69 is recorded as within-window
noise plus build selection. **Re-read 2026-08-18:** f=229/1247=18.36%, split
114-115, paired SE 1.21 pts, **MDE80 3.40 pts** — the −0.08 is noise against a
3.40-point resolving power. The closure cited margin MAE, which the same
predeclaration listed as a secondary endpoint (direction only, no
gate-shopping); the one declared override was Brier, whose interval
[−0.00553, +0.00072] crosses zero and never fired. (2) QB-plus-continuity at
the declared alpha-1 scored exactly +0.00 points on 997 games in 2014–2017
(arms disagreed on 176 picks and split 88–88) with all probability diagnostics
worse; the 52.34/52.63 figures are recorded as selection artifacts. **Re-read
2026-08-18:** f=176/997=17.65%, paired SE 1.33 pts, **MDE80 3.73 pts** — the
family's own predeclaration called a null at this sample size "the modal
expected outcome, declared acceptable in advance," so a predicted null cannot
also be the evidence that closes the family. (3) Expected role delivery failed its intermediate-target
gate — the clipped delivery model lost to both parents in all 11 seasons
because injury-listed players who play at all deliver essentially their full
prior role — so no ATS rows were generated and the 2018–2025 outcome set was
not viewed again. A conservative ledger now counts roughly 130–150 candidate
streams scored against the 2018–2025 outcomes; the best pooled numbers there
(52.47–52.77%) are what selection on noise plus a possibly small real effect
would produce. The strategic conclusion is that further mining of 2018–2025
with variants of existing families is unlikely to yield trustworthy gains; new
information sources — point-in-time market quotes, CFB-replicated mechanisms,
and prospective 2026 outcomes — are the admissible paths forward. Both
replication windows (2013–2017 and 2014–2017) are declared spent for their
respective families.

## Sensitivity-aware review of completed experiments

Reopening means a newly specified representation, external/CFB replication, or
one frozen low-variance follow-up. It never means rerunning the identical
candidate on 2018–2025 until it wins.

| Priority | Prior experiment | Evidence-aware decision |
|---|---|---|
| Resolved (kept as shipped refinement) | Learned any-snap availability | Remains the availability parent: it improved a 57,294-row out-of-season player target and its diagnostics moved coherently. Its role-delivery successor closed at the target level (below), which strengthens rather than weakens this model — any-snap already captures most of the availability signal. |
| Closed at target (August 2026) | Expected role delivery | The frozen two-part clipped-delivery model lost to both parents on delivered-share MAE in all 11 seasons. Mechanistic finding: injury-listed players who log any snap deliver ~their full prior role (median delivered/prior 1.01; 55.8% of played rows clip at 1). Any successor (e.g., unclipped or asymmetric delivery target) is a new predeclarable candidate, not a rerun. |
| ~~Closed by replication (August 2026)~~ **Unresolved (reclassified 2026-08-18)** | QB plus lineup continuity | The predeclared alpha-1 candidate scored exactly +0.00 points vs base on 997 untouched 2014–2017 games with all probability diagnostics worse; 52.34/52.63 are recorded as within-window selection artifacts. **The +0.00 sits below the instrument's own MDE80 of 3.73 points (f=17.65%, paired SE 1.33 pts) and the family's own predeclaration called a null "the modal expected outcome, declared acceptable in advance" at this sample size — a predicted null cannot also close the family.** The 2014–2017 window is spent for the player family; the family status is `unresolved`, not `closed_negative`. |
| ~~Closed at the CFB benchmark (August 2026)~~ **Unresolved_below_power (reclassified 2026-08-18)** | CFB role-continuity feature family | The predeclared dropback/carry participation-continuity family (season-scoped, streak-capped per the absence-separation study) scored −0.67 accuracy points vs the frozen XLG-03 arm on 8,933 clean-core games (`docs/cfb_role_features.md`). **The closure does not hold**: the effect sits below the instrument's own MDE80 of 0.927 points (f=9.96%, n=9,093), and trait split-half reliability is 0.719 (dropback) / 0.680 (carry) — comparable to `injury_value_lost`'s kept 0.87-0.93 and far above the reliabilities that correctly closed other lines — which rules out "refuted: no split-half reliability". No family-specific positive control bounds it either. Participation disruption is not shown to be market-priced; it is unresolved. This closure was gating the XLG-04 → XLG-05 transfer path, which is open again. Any successor (roster-aware departures, replacement quality, XLG-07 availability semantics) is a new predeclaration, not a rerun. |
| Screened, unresolved (August 2026) | QB-dependence interaction feature (pool_edge_plan.md queue item 6) | New CFB-only feature (`docs/qb_dependence.md`, `src/nfl_ats/cfb_qb_dependence.py`): `qb_starter_epa_per_dropback x off_pass_rate`, built per side then differenced, added alongside (not instead of) the frozen XLG-03 baseline. Split-half reliability is exceptionally high — 0.9588 (interaction), 0.9581/0.9923 (constituents) on 2,325-2,366 team-seasons — well above every prior "kept" reliability in this project (`injury_value_lost` 0.87-0.93, `cfb_role_continuity` 0.68-0.72), ruling out a refuted mechanism. But the measured clean-core effect is tiny: +0.0224 accuracy points, week-blocked 95% [-0.8206, +0.8681], `probability_positive` 0.5140 on 8,933 games — about 2% of this screen's own MDE80 floor (1.093 pts, f=13.85%, n=9,093). `unresolved_below_power`, not a negative; recorded via `nfl-ats weak-signals record --league cfb` (proposed command in `docs/qb_dependence.md`, not yet executed). No NFL rotation window touched or family declared — an NFL Step 3 (raw vs injury-blended QB state, rotation-family declaration) is a separate, later, separately predeclared spec. |
| Closed at the CFB benchmark (August 2026) | Conditional margin variance (MOD-16 screen) | The predeclared Ridge residual-scale model (mismatch/total/pace/experience, clipped [2/3, 3/2]) produced real per-game variation (p10 0.905, p90 1.115) yet made clean-core cover log-loss and Brier resolvably worse under week and season blocking (`docs/margin_variance.md`). The pooled out-of-time residual distribution is already near-correctly calibrated. Only an NFL-only-feature variant with a new ledger-aware predeclaration remains admissible; distributional successors (MOD-05, MOD-08) are separate predeclarations. |
| Redesign, then reopen | Snap-weighted player value | The +0.10-point box-score extension was too small to resolve and its value proxy is coarse. Revisit with replacement quality, position hierarchy, and CFB/NFL partial pooling—not the identical two-field rerun. |
| Revisit only through transfer | Opponent-adjusted PBP and matchup effects | Small probability movement could be below NFL power, but ATS remained below 50%. Use CFB to choose low-dimensional mechanisms, then freeze an NFL transfer test; do not reopen the broad NFL bundle. |
| Revisit only after a stronger signal | Beta/other calibration | Calibration can improve probability magnitudes but does not create side information. Re-evaluate inside a newly fixed player model, not as an alpha search by itself. |
| ~~Keep closed in current form~~ **Unresolved_below_power (reclassified 2026-08-18, `docs/closure_audit.md`)** | Participation offense/defense RAPM | Accuracy fell 0.43 points, week-blocked 95% [-1.53, +0.63] and season-blocked [-1.01, +0.10] — both cross zero, and MDE80 at this sample (f=5.5%, n=2,075) is ~1.4 points, well above the observed effect, so accuracy alone cannot resolve this either way. Brier and log-loss resolve worse under the naive season-blocked bootstrap ([-0.00171, -0.00007] and [-0.00371, -0.00018]), but both sit within 8-10% of re-crossing zero — inside the 3.7-33.0% width understatement `docs/estimation_variance.md` measures for comparisons this size — so this is a lean, not a settled negative. Recorded as `participation_offense_defense_rapm` (`registry/weak_signals.json`), category 3, rather than closed. Position-unit or matchup hierarchy would be a new model, preferably learned with CFB; alpha retuning is not admitted. |
| ~~Keep closed in current form~~ **Unresolved_below_power (reclassified 2026-08-18, `docs/closure_audit.md`)** | PageRank/HITS schedule graph | Graph candidates were selected in 0/8 outer seasons; cover Brier was worse in 7/8 and 6/8 seasons across two model comparisons (sign-test p≈0.07 and p≈0.29), but the season-blocked Brier interval itself ([-0.000376, +0.000005]) does not exclude zero. A consistent negative lean across replications, not a resolved refutation — recorded as `graph_schedule_rating_brier` (`registry/weak_signals.json`), category 3. No underlying artifact survives on disk, only prose. CFB may support player/unit graphs, but the lean gives no reason to rerun this specific team graph without a new mechanism. |
| ~~Keep closed — confirmed by replication (August 2026)~~ **Unresolved (reclassified 2026-08-18)** | Drive aggregates and broad raw PBP bundle | The re-review found the matched market-residual comparison had never been computed and showed +1.69 points post hoc; the predeclared 2013–2017 replication scored −0.08 points with margin error resolved worse. **This does not confirm closure**: MDE80 is 3.40 points (f=18.36%, paired SE 1.21 pts), so −0.08 is noise against that resolving power. The closure cited margin MAE, which the same predeclaration listed as a secondary endpoint (direction only, no gate-shopping); the one declared Brier override, interval [−0.00553, +0.00072], crosses zero and never fired. The 2013–2017 window is spent for this family; the family status is `unresolved`, not `closed_negative`. Preserve the layers for a future joint score/pace distribution, not another ATS screen. |
| Keep closed | Broad 48-row player selection grid | The nested selector failed and pooled winners are multiplicity-exposed. Its rows may nominate one mechanistic hypothesis, but the grid itself is not independent evidence. |

1. **Read `docs/revisit_list.md` first.** A 2026-08-18 audit found four
   defects in the measurement instrument and one in the decision frame, and
   several terminal verdicts now rest on measurements that may be wrong.
   ~~**The gating experiment is stated there and must run before any Tier 1
   re-run:** does the probability-calibration step attenuate or invert small
   effects on REAL data, as it demonstrably does on planted ones
   (`docs/purged_cv.md`)? If yes, every terminal negative in this project is
   suspect. If no, the list shrinks to the degenerate-bootstrap case.~~ —
   **RESOLVED 2026-08-18** (`docs/calibration_distortion.md` §8,
   `docs/revisit_list.md`): the gating experiment ran. **D1 is a planting
   artifact, not a real defect** — the recorded 51.3%/53.0% readout was one
   random seed reported as two independent findings; replicated across 21
   seeds, the calibration step is worth at most **~0.35 accuracy points**,
   not the claimed 2.0-point swing, and does not invert any effect. Tier 1
   shrank to the D4 degenerate-bootstrap case
   (`player_qb_continuity_matched_alpha`) plus the two bare-verdict entries
   (`pbp_drive_bundle`, `player_qb_continuity`) — D1 does not put every
   terminal negative in the project under suspicion. ~~Also binding from
   that audit: reported intervals are 17-58% too narrow because
   the block bootstrap never refits (`docs/estimation_variance.md`)~~ —
   **RETRACTED 2026-08-18** (`docs/estimation_variance.md` Part II): that
   figure double-counted a training-by-game interaction the game bootstrap
   already carries. The honest refit factor is **1.003x, one-sided 95% upper
   bound 1.099x**. The real defect was **D4 (too few blocks)**, not D2:
   measured coverage by block count is 0.000 at k=1, 0.466 at k=2, 0.760 at
   k=4, 0.896 at k=10, 0.944 at k=50 (nominal 0.95), so
   `MIN_BLOCKS_FOR_INTERVAL = 10`. Still binding from the original audit: the
   project is **model-limited, not data-limited** (accuracy is flat across a
   100x range of training-set size, `docs/scaling_and_transfer.md`), and the
   empirical-Bayes shrinkage work is **void** (`docs/decision_rule.md`).
2. Maintain the prediction-safety contract and add a regression canary for
   every production error or newly supported output type.
3. The point-in-time market stack is code-complete: the purchased 2020–2025
   snapshot archive is verified and backed up, weekly scheduled captures
   continue on the free tier, the frozen MKT-06 pilot has taken its one look
   (direction replicated, no magnitude edge) with `predict-close` wired to
   the Week Board, and the MKT-04 paper-decision ledger records every
   published card's picks at publication (`publish-predictions`) and scores
   them against the close (`clv-ledger`, surfaced on the History page).
   Remaining market items are research questions (MKT-03 diagnostics, MKT-08
   timing policy) and the MKT-09 licensing audit. **MKT-03 update,
   2026-08-18** (`docs/novig_diagnostics.md`): the diagnostic itself has now
   run, read-only against the existing archive, no rotation-registry window
   spent and nothing fed into any model. The dropped spread price is
   informative (55.50% of 438,424 quotes are not exactly -110), but the
   resulting no-vig probability is calibrated within noise at the Tuesday
   opener for the ATS arm (both buckets cross zero at both blockings) and
   for four of five moneyline buckets; one moneyline bucket's season-blocked
   interval excludes zero, reported as continuous evidence, not a finding
   (secondary-goal-only, no multiplicity correction across the seven buckets
   read). Still a diagnostic, not a candidate feature; consuming any of it
   inside a model requires its own predeclared look.
4. The XLG-04 chain is complete end-to-end: role delivery replicated
   cross-league for dropbacks and carries (`docs/cfb_role_replication.md`),
   the departure-vs-temporary-absence prerequisite was measured
   (only 15.6%/18.7% of qualified holders return the next season;
   same-season return odds fall to ~10%/7% after four straight missed
   games), and the ONE predeclared role-continuity family was scored
   against the XLG-03 benchmark: paired accuracy −0.67 points on 8,933
   clean-core games (week-blocked [−1.33, +0.01]) with Brier and log-loss
   worse under both blockings (`docs/cfb_role_features.md`). ~~It did
   **not** clear... The market already prices participation disruption. No
   NFL transfer claim is predeclared from this family and no retuning of it
   is admitted.~~ **REOPENED 2026-08-18**: −0.67 points sits below the
   instrument's own MDE80 of 0.927 points (f=9.96%, n=9,093), and trait
   split-half reliability (0.719 dropback / 0.680 carry) rules out a
   refuted mechanism, so neither admissible closing ground was ever met.
   Reclassified `unresolved_below_power` (`registry/weak_signals.json`).
   The market is not shown to price participation disruption; that was
   never established. No NFL transfer claim is predeclared from this
   family yet, and no retuning of the spent CFB window is admitted, but
   the family itself is open again.
5. **XLG-05 therefore has a mechanism to transfer again.** ~~XLG-05
   therefore has no cleared mechanism to transfer yet; it waits for a
   family that first clears the CFB benchmark.~~ The role-continuity
   family's closure — the reason XLG-05 was waiting — is retracted (item 4
   above), so the XLG-04 → XLG-05 transfer path reopens. This is not itself
   a green light to spend an NFL window: the CFB-side evidence is
   `unresolved_below_power`, not confirmed, so any XLG-05 predeclaration
   must say so. The remaining CFB-side
   paths are XLG-06 (rookie/young-player priors) and XLG-07 (availability
   semantics), plus CFB screens of the distribution work in item 6.
6. Score the active model and any frozen challengers on prospective 2026
   outcomes only — now at BOTH grades (opener via the live Tuesday captures,
   and close), with the opener grade primary per the pool goal. **The
   machinery for this now exists and is the single most time-critical item
   in the file** (POL-10, `docs/prospective_evidence.md`): win/loss settles
   at both grades, the weekly Best Pick persists pre-kickoff, MOD-07 is
   registered as a challenger, and anti-backdating is enforced at write and
   again at scoring. Week 1 locks Tuesday 2026-09-08 and an unrecorded
   season is gone. ~~**The Week 1 ledger-anchoring decision was resolved
   2026-08-17:** both prospective rehearsal ledgers were deleted so the
   first write to the live ledger is the real Tuesday-lock card on
   2026-09-08; no manual row insertion is possible.~~ **That resolution did
   not hold.** The ordinary, documented `publish-predictions` command
   repopulated the live ledger within hours, because recording was opt-out
   at the time: 16 real 2026-Week-1 rows landed at `recorded_at_utc`
   2026-08-18T01:24:56Z (`model_id` `4b01f055b684e27e`, `is_best_pick=True`
   on `2026_01_ARI_LAC`). **Fixed 2026-08-18:** recording is now opt-in
   (`--record-decisions`, default `False`) and separately refused whenever
   a week's earliest kickoff is more than `RECORDING_LOCK_WINDOW` (7 days)
   from the recording instant, so the same command cannot silently
   repopulate the ledger a third time (`docs/prospective_evidence.md`,
   "Known divergence"). Disposition of those 16 rows — reset again vs.
   accept as a rehearsal artifact — was left to the owner
   (`docs/week1_readiness.md` item 2), and a live check on 2026-08-18 found
   `artifacts/clv_ledger/decisions.parquet` **absent from the repo
   entirely** — matching neither documented option and not matching the
   16-row state this section and `docs/week1_readiness.md` still describe.
   A backup of the 16 rows was located (outside the repo, from a prior
   session) and verified against this description exactly; nothing was
   deleted, since the file the deletion was supposed to act on was already
   gone by the time of the check. ~~Whether a reset was already executed
   somewhere or the local artifact was simply lost needs confirming before
   Week 1's ledger status can be called resolved either way — see
   `docs/week1_readiness.md` item 2 for the live finding.~~ **RESOLVED
   2026-08-18** (`docs/week1_readiness.md` item 2): a follow-up read-only
   check re-confirmed `artifacts/clv_ledger/decisions.parquet` is still
   absent — zero old-model rows. The *cause* of the absence (an executed
   reset vs. a lost local artifact) was never determined and is not claimed
   to be; the item is resolved because the end-state matches the owner's
   2026-08-18 reset decision regardless of which cause produced it: zero
   contaminating rows, the promoted `weak_stack` model free to write Week 1
   fresh, the 16-row backup preserved outside the repo, and refill guarded
   by opt-in recording plus the 7-day `RECORDING_LOCK_WINDOW`. **Live
   consequence: the first genuine write to the primary ledger is now the
   Sep 8 lock-day `weekly-run`/`publish-predictions` run, and only if it is
   invoked with `--record-decisions`** — the flag is opt-in, so omitting it
   publishes the card but records nothing to either ledger. `is_best_pick`
   persists in `PAPER_DECISION_COLUMNS` written only when every game of
   the week is still ahead (`docs/prospective_evidence.md`). The 2013–2017
   and 2014–2017 replication windows are spent, and no new variant of an
   existing family may be scored on 2018–2025 without a frozen predeclaration
   that acknowledges the ~130–150-look ledger. **The peer-reviewed opener
   biases are no longer a lead**: three were built and, ablated inside
   MOD-07 on the already-spent window, contributed +0.22 points at
   `probability_positive` 0.505, while the published Week-1 holdover
   figure (35.6%) fails to replicate here (52.5% on 120 games). Do not add
   more of them. **Evening update, 2026-08-18:** two more plays are now
   LIVE on the real, published card (`CURRENT_PREDICTIONS.md`), both
   dual-tracked via the challenger ledger so neither spends a
   rotation-registry window. (a) The clean-case year-1-head-coach fade,
   weeks 1-8 only (`docs/coach_fade_overlay.md`, `OVERLAY_ENABLED = True`,
   challenger `hc_year_one_fade_overlay`): the real Week 1 2026 card shows
   exactly one flip, `2026_01_BAL_IND` (BAL, year-1, at IND, kept coach) from
   BAL -3.5 to IND +3.5, with `2026_01_MIA_LV` correctly flagged but not
   flipped (both coaches year-1). (b) The Best-Pick nomination v2 rule
   (`docs/best_pick_ranker.md` § "2026-08-18: the weekly NOMINATION rule
   switches", `NOMINATION_V2_ENABLED = True`, challenger
   `best_pick_nomination_v2`): nominates by calibrated probability among
   low-disagreement games rather than `sweep_robustness`'s alphabetical
   tie-break. Re-verified live this session (`scripts/best_pick_nomination_dry_run.py`
   against the active model's real Week 1 forecast): v2 nominates
   `2026_01_MIA_LV`, no tie, while the incumbent (`sweep_robustness`, itself
   a two-way tie) nominates `2026_01_ARI_LAC` — the two rules disagree on
   which game gets the ★ this week, and the published card now shows v2's
   pick, matching the `d991c65` republish. `registry/weak_signals.json`
   now holds **107** recorded signals (verified count, `nfl-ats weak-signals
   status`), including ranked open leads worth a future look: division
   revenge (+0.19 accuracy points, `probability_positive` 0.88,
   `bias_battery_division_revenge_game`); a CFB rivalry-finale proxy whose
   interval sits entirely negative (`probability_positive` 0.0) —
   resolved-*shaped* but still recorded `unresolved_below_power` because it
   is one of 19 mined, uncorrected battery cells
   (`cfb_bias_battery_rivalry_finale_proxy`); and a penalty-only variant of
   the weak-signal stack, tracked but not actionable (+0.13 points,
   `probability_positive` 0.69, `weak_stack_v2_penalty_only`). New screens no
   longer need hand-transcription into the registry: `nfl-ats experiment run
   <spec.json>` (`docs/experiment_pipeline.md`) runs the whole
   reliability-check/screen/bootstrap/classification/record/provenance loop
   from a declarative spec, built after this session's own recorders caught a
   100x scaling bug, a sign bug, and a corrupted source path, all
   hand-copied from console output. Two more comparisons were run
   head-to-head against their incumbents and settled this session — neither
   should be re-derived: the ridge-alpha swap (10.0 to 2,000.0 on the active
   `weak_stack`/`market_residual` config) was **refused** on EV at the
   opener grade (`ridge_alpha_2000_nfl_opener_confirmation`, -1.397 points,
   `probability_positive` 0.0504, ~95% against; the resolved calibration
   gain routes to Best-Pick/calibration consumers instead, production
   `ridge_alpha` stays 10.0), and the CFB per-metric offseason-retention
   feature vector is **closed** `refuted_mechanism`/`wrong_sign_resolved`
   (`offseason_retention_per_metric_cfb`, -0.739 points, `probability_positive`
   0.0037, loses to the uniform 0.67 scalar RWB-01 already ships). **With the
   ledger fix verified and both overlays live, the single most
   time-critical action left in this file is unchanged from the top of this
   item: the Sep 8 lock-day `weekly-run`/`publish-predictions` run must pass
   `--record-decisions`, or the season's first genuine ledger write — and
   every challenger's first prospective evidence — silently never happens.**
   **2026-08-20, later the same session: the challenger count above is
   stale.** 18 ACTIVE_PROSPECTIVE challengers (21 entries total; see
   POL-10) must be adjudicated at that same Sep 8 run, not the smaller
   count this paragraph originally described. One owner action remains,
   none blocking Sep 8: register
   the two weekly public-betting capture tasks, Saturday and Sunday noon
   ET (`docs/public_betting_sourcing.md` §9, exact `schtasks` commands at
   MKT-12). The GDELT **volume** path is now complete and processed for 32/32
   teams and all 37 relocation-era aliases (`docs/gdelt_backfill.md`): the
   frozen close-grade `attention_battery_both_cold` replication was rerun on
   2,038 eligible games and now leans against its sign, -0.2742 accuracy
   points, `probability_positive=0.23225`, still
   `unresolved_below_power` with no admissible closing ground. Tone remains
   rate-limit-blocked at 2/32 teams after BAL/BUF each exhausted eight HTTP
   429 retries, but tone is not needed for that completed volume replication;
   resume it only for a separately predeclared sentiment question. The PFR
   per-article date fetch is now complete at 4,361/4,361 targeted rows
   (PER-03). The Sagarin Era B snapshot is also consolidated at 585
   parser-valid pages, 18,473 rating rows, and 9,848 Tuesday as-of rows; seven
   Wayback fetch gaps remain documented, and the prior ATS screen was not
   rerun (MKT-11).
7. Stop trying to measure team quality better; it is bounded near zero.
   A deliberate-leak positive control (opponent adjustment fit over all of
   2006–2025, so the columns see the future) moved margin MAE by only
   **+0.0129 points** — a measured ceiling on the whole family, and the
   common explanation for PBP-05 and MOD-16, both genuinely closed on
   measurement. **The PBP/drive bundle and CFB role continuity are not part
   of that closed set** — both were reclassified `unresolved`/
   `unresolved_below_power` on 2026-08-18 (their negatives sat below the
   instrument's own resolving power); the +0.0129 ceiling still bounds what
   either could plausibly be worth even if resolved, so the strategic
   advice below is unchanged, but neither should be described as failed.
   Our target is the residual from the
   market line, and the market already prices team quality
   (`docs/play_level_audit.md`, `docs/cfb_opponent_adjustment.md`). Prefer
   work that prices what the market prices BADLY — availability is the only
   candidate carrying a measured lean (`probability_positive` 0.899 in the
   MOD-07 ablation) — or that exploits the pool's format rather than the
   line (POL-04/05, largely unexplored). On distributions specifically: the
   margin lattice is real and large but the ATS *residual* is already
   near-Gaussian once the varying spread smears it, so MOD-05 is worth
   building for pushes and half-point questions rather than ATS accuracy,
   and MOD-08 has no shape signal left to condition on. The one measured
   distribution win is **smoothing** rather than conditioning: replacing the
   518-draw ECDF costs nothing and buys Brier −0.0015 (P=0.998), but it
   moves picks and so needs its own predeclared window.
8. Use 2016–2025 participation/NGS for position-unit and formation effects;
   individual receiver-corner pairs remain too sparse for an initial model.
9. Attempt drive simulation only after simpler distributional baselines exist.

The dashboard and experiment registry should make failed hypotheses easy to
retain. Negative results are project assets; quietly deleting them invites the
same experiment to be rediscovered and overfit later.

## 2026-08-21 mass-screening wave (30-lane orchestration)

A single-session wave of parallel screens, builds, and scouts. All experiment
verdicts flowed through `nfl-ats weak-signals record` (56 cells recorded
centrally this session from artifacts; earlier lanes recorded their own).
Registry now pools 326 NFL accuracy-point signals (random-effects +0.009 pts;
sign test 175/326 favouring candidate). Gates green at wave close: ruff format
492 files, ruff check clean, mypy clean, pytest 1,642 passed.

**Direct edge result of the day — overlay subset composition**
(`docs/overlay_subset_composition.md`,
`artifacts/overlay_subset_composition/20260821T174356Z`): all 127 non-empty
subsets of the six pick-flipping overlays plus the arrest policy were scored on
the frozen 1,537-game opener archive. Best subset coach_fade + division_revenge
+ arrest + spread_gap_zone scores **55.42%** (+2.06 pts over the raw model's
53.36%, season-blocked P+ 0.915); the four predeclared identities are recorded.
The naive all-seven stack remains resolvably WORSE (-2.86 pts) — composition,
not accumulation, is the lever. The top figure is a selection-inflated UPPER
BOUND (max of 127 correlated candidates on already-looked-at data), not a
claim. **Promoted 2026-08-21 as a forced-pick EV decision:** the exact runtime
policy is `overlay_union_coach_division_revenge_player_arrests_spread_gap_v1`
with fingerprint `bbdd60a1712386541546c8e757615fb5ff216f49eb81397502cb360809bc5ded`.
Against the actual former coach-to-arrests production chain it is +1.2641
accuracy points, week-blocked P+ 0.85715; planning should use roughly +1 point,
not assume the selected 55.42% reproduces. The former chain is now the active
prospective control.

**Era weighting (MOD-14): confirmation look SPENT, stays open.** Family
`era_weighting_half_life_8` took its one predeclared opener look on [2020,2021]
(`docs/era_weighting_promotion.md`, rotation window spent forever):
-0.2193 pts, week-blocked [-3.39,+2.68], P+ 0.425 — unresolved_below_power,
recorded; no production change (EV case not established either way).

**Best-Pick ranker follow-ups (POL-09)** (`docs/best_pick_followup.md`, CFB free
benchmark, 280 weeks): smooth_cdf_distance +0.71 pts P+ 0.584;
alpha2000_distance -1.79 P+ 0.239; dispersion-gated -1.43 P+ 0.217; ensemble
+0.36 P+ 0.527 — none cleared the 0.75 gate, none earned an NFL window; all
four recorded `unresolved_below_power`.

**Combined weak-signal stacker predeclared** (`docs/combined_stacker_predeclaration.md`):
mechanical input rule selects 4 columns (injury-value-lost narrowed,
temp-gap-cold-visitor, warm-team-cold-late, spread-gap-zone), claims window
[2022,2023], decision = sign of paired opener delta at EV, claim gate P+ >=
0.90. NOT RUN; run is one command away for a future session.

**Ten new screened families (all cells `unresolved_below_power`, docs under
matching names, week-blocked primary seed 20260821):**
red-zone/third-down reversion (strongest cell third_down_over_fade +0.37 pts
P+ 0.872, trait reliability +0.407); close-game/turnover luck regression
(turnover_under_rebound +0.41 pts P+ 0.920, season-secondary P+ 0.981);
ENV-06 body-clock early windows (all lean OPPOSITE the classic mechanism, dose-
response fails, control null — family unresolved, night-game version now the
priority per `docs/archive/literature_leads_20260821.md` §2 lead 1); altitude
adaptation; late-season motivation ladder (fighter_vs_nothing primary sits
wholly below prediction but declared secondary crosses zero — owner decision
on primary-only grading FLAGGED not decided; tank zone leans opposite at P+
0.986 season); QB age/experience curve (second_year_jump +0.24 pts P+ 0.855);
weather x total tercile (**precip x high-total resolves MIRROR-opposite its
prediction, both blockings entirely above zero, P+ 0.998/0.9995 — recorded
unresolved because the validator cannot express a mirror wrong-sign closure;
n_flag=50, one look**); OL continuity via snap-share overlap (within-season
split-half +0.479 but YoY only +0.075); venue milestones (former-stadium swing
set measured EMPTY — zero qualifying games exist); divisional rematch dynamics
(deployed revenge overlay confirmed as the unsplit parent; home-loser split +
0.109 P+ 0.822 exceeds road split).

**New sources scouted** (`docs/data_source_scout_v5.md`, six sections) — top:
NFL.com official weekly injury-report archive (free, PIT-A, verified to 2011;
replaces the dead-after-2024 nflverse feed AND feeds the Friday-designation
late-week channel); Big Ten 2023+/SEC 2024+ availability reports (XLG-07's
fail-closed premise outdated for 2023+); VegasInsider Wayback boards
2005-2016; FantasyFootballCalculator ADP API. **Literature mined**
(`docs/archive/literature_leads_20260821.md`, four sections) — top: Smith et al. Sleep
2013 west-coast NIGHT-game ATS effect (+5.26 pts, n=106) untested here;
bye-advantage market-overvaluation reversal post-2011 CBA; Management Science
2024 line-move negative autocorrelation; hamstring-recurrence RR 2.7-4.8 as
availability-feature designs.

**Ops:** the two weekly public-betting capture tasks are REGISTERED and
verified (Sat/Sun noon ET, next runs 8/22-8/23/2026); PER-03 doc/registry
mismatch resolved; POL-04 re-examined under the corrected deadline model
(pick-distribution unlocks feed the simulator's never-measured public_lean —
in-season measurement task, row stays closed as a data question).

## 2026-08-21/22 Wave 1 (transparency fleet)

**Fabric**: `scripts/fabricate_worktrees.ps1` (junctioned git-worktree fabricator, smoke-tested non-destructive) + `scripts/batch_record.py` (locked queue so parallel agents can never corrupt the registries) — `docs/fleet_orchestration.md`, `docs/batch_record.md`.

**Edge**: combined-stacker look SPENT [2022,2023] (`docs/combined_stacker_predeclaration.md` §8): the four pooled columns scored **−0.97 pts vs incumbent, P+ 0.133** at the opener — EV rule keeps the production card; family unresolved, not closed. NFL.com official injury archive ingested (54/54 pages, 17,483 rows 2022-24, **99.63% agreement** with nflverse — confirmed replacement for the dead-after-2024 feed, `docs/nflcom_injuries_sourcing.md`); its Friday-designation screen found the wave's strongest new lead: **teams with >=2 OUT designations cover less**, P(direction) ~0.976, season-blocked entirely below zero (`docs/nflcom_friday_designation_screen.md`). Night-game body clock leans the published mechanism at P(direction) up to 0.93 with monotone dose-response (`docs/body_clock_night_screen.md`). Bye-week screen reproduces the overvaluation shape (era flip exactly as the mechanism requires; fade arm P+ 0.870) — `docs/bye_overvaluation_screen.md`.

**Transparency**: `src/nfl_ats/attribution_waterfall.py` — reconciling per-pick breakdown (market line -> feature-family contributions -> probability-rule offset -> overlay flips -> final pick) with hard sum-to-final asserts and artifact IO; `src/nfl_ats/model_ledger.py` — the Model Ledger contract (25 validated rows: PROMOTED/SUPERSEDED/RETIRED/CHALLENGER badges, evidence linked to registry keys with fingerprints); "Gridiron Observatory" design system mocked with real Week 1 data under `docs/design/` (style guide + game-card/waterfall/model-ledger mockups) pending owner review before any site repaint. **Streamlit removed entirely** (shell/pages/tests/deps; shared viz/theme/findings_content preserved for the public site; uv.lock -487 lines; zero streamlit references outside historical ROADMAP prose).

## 2026-08-22 Wave 2 (transparency ships, alternate theme)

The Week Board now explains itself, and the Observatory look ships as a TOGGLEABLE ALTERNATE per owner decision (default rendering unchanged; `site_theme/toggle.js` cycles default -> obs-night -> obs-day with localStorage persistence). Both themes carry identical information.

**Components**: waterfall feed pipeline (`scripts/waterfall_feed.py`; 16/16 published Week-1 games, reproduces the deployed card to 4.4e-16, mechanical rationale sentences regex-audited against field values) rendered as fail-open "Why this pick" panels on every board game row (steps table, overlay flips, edge-vs-market, key-number distance); Model Ledger embedded on index below the board (`src/nfl_ats/model_ledger.py` HTML renderer: 25 validated rows, PROMOTED/CHALLENGER/SUPERSEDED/RETIRED badges glyph+text, evidence footnoted to registry keys); margin-interval text rows on every game card where quantiles exist. Theme pack (`src/nfl_ats/site_theme/`) scoped under body.theme-obs so defaults are untouched; asset sync wired into `_write_public_site`.

**New sources**: VegasInsider Wayback pilot = GO (`docs/vegasinsider_pilot.md`: 14/14 snapshots parsed, 7 books consistent across pages, spread coverage 99.3%, totals 96.8% — full 2005-2016 backfill is effort M); FFC ADP archive ingested end-to-end (`docs/ffc_adp_sourcing.md`: 2010-2025 x ppr/standard, 50,607 drafts, exact mock-window stamps, team-aggregate table built; Weeks-1-4 divergence screen designed but not run).

Gates at wave close: ruff clean, mypy clean (101 files), pytest **1,707 passed**.

## 2026-08-22 Wave 3 (red team, de-overfit, sources, availability)

**Adversarial audit** (`docs/edge_audit_redteam.md`): the overlay-composition claim was DOWNGRADED by independent attacks (LOSO-CV of subset choice pools to 0.0000 pts; rank-stability rho 0.72 does not separate from a within-week flip-shuffle null, p=0.24) — the honest expectation is unresolved, not ~+1pt. Three claims SURVIVED everything thrown at them: NFL.com Friday out>=2 (stronger after bad-team controls; starters-only -12.4 raw pts, P(neg)=0.9995), night-game body clock (west-road coefficient holds with distance controlled), bye fade (real assignment beats 99% of sham-bye placebos). The audit also caught a real instrument bug in `bye_overvaluation_screen.build_bye_maps` (season openers misflagged off-bye); fixed with a failing-first regression test and corrected cells re-recorded (--replace).

**De-overfit**: `docs/shrunk_overlay_weights.md` — continuous ridge-logistic weights over the seven flip indicators (alpha=100 chosen by LOSO log-loss CV before accuracy was seen). Nested walk-forward deployable estimate: **-0.08 pts, flips 3/1283 games** vs the discrete max-hunt's +2.06 attribution — selection inflation measured directly; incumbent card confirmed as the right play.

**Composition**: chain + observed-movement rule scores **55.69%** on the paired opener archive (+1.53 over incumbent, week P+ 0.894 / season 0.930) — attribution upper bound, movement data covers all six archive seasons (`docs/movement_composition_eval.md`). Registered; prospective activation is the obvious next EV decision.

**Sources**: VegasInsider Wayback boards FULLY backfilled 2005-2016 (all 12 seasons, 3 layout generations parsed, dispersion computable everywhere; 2006 flagged reduced-confidence) — multi-book dispersion + totals features are now buildable. FFC ADP divergence screen: high-ADP underdog back cell +4.18 pts, season-blocked P+ 0.983 (n=275, mined family). NFL.com combine dataset ingested (8,968 player-seasons, 91% join for 2016+ classes). B1G availability PDFs snapshotted 2023 W1-6 (parse blocked on missing PDF dependency — deliberate stop); SEC reports found ingestible via public Google Sheet (137 rows, 3 point-in-time states recoverable). Arctic Shift attention gate FAILED its predeclared shared-variance bar (r=0.732 >= 0.70 vs Wikipedia) — no ATS work, recorded honestly.

**Availability science**: recurrence-hazard features built player-level (body-part classes from NFL.com-scrape text, 99.97% id match, unmapped 2.6%); same-history hazard ratios echo published RRs directionally (hamstring 2.01, shoulder 1.95); but added flags do NOT beat a refit designation baseline on next-game availability Brier (P(helps)=0.000) — trait reliable (+0.742 split-half), mechanism open, family unresolved.

**Registry integrity** (`docs/registry_correlation_audit_20260822.md`): 6 duplicate entries found across the week's additions (3 body_clock dose buckets superseded as pointer-only via --replace); overlap warnings +82%; sign test now 182/346 candidate. Idea ledger consolidated: `docs/archive/idea_ledger.md`, 84 ranked rows + next-ten-actions.

## 2026-08-22/23 Wave 4 (variance decomposition program + ceiling attack)

The owner question "why does the 13.1 exist, and could 60 percent ever happen" now has measured answers.

**Variance decomposition** (all REG 2009-2025, docs/vardec_*): outcome variance around the line is dominated by PLAY-LEVEL EXECUTION NOISE — a calibrated resampling simulator (`vardec_noisefloor`) puts ~80 percent of margin variance on within-play execution; perfect team-strength knowledge still leaves sd ~6.4 pts. Component shares: turnovers 33-46 percent gross but forecastable fraction ~0.5 percent (recovery luck ~10 percent of fumble slice); penalties 3-9 percent share, forecastable ~0; special teams and QB-replacement/decision lanes in flight. Variance accumulates near-uniformly across quarters (Q4 alone 21 percent); a perfect halftime model removes only ~49 percent of residual. The sigma is nearly CONSTANT across conditions — one mined exception: |rest differential| >= 4 days games run sd ratio 0.915 [0.852,0.977], a Best-Pick-amplification lead (+0.25 pts/pick conversion), category 3.

**Ceiling attack**: (1) MSE split (`docs/ceiling_error_split.md`): execution-noise floor is 95.9 percent of market MSE; better-team-model headroom <= +0.80 pts; late information +1.72 pts measured (movement oracle). (2) Deliberate-leak positive control (`docs/leak_ceiling_control.md`): fitting ON the same outcomes, pregame features reach only **55.6-56.1 percent** — BELOW the assumed 57-58 wall's top; same-game PBP leak hits 84.05 percent proving instrument soundness. The practical pregame ceiling on this grading structure revises DOWNWARD to ~56 percent; 60 percent is empirically excluded for any model in this class, answering the owner question with data instead of assumption.

**Live-market check** (`docs/sbr_halftime_mining.md`, in-house SBR 2H columns first explored): halftime lines absorb only ~11 percent of remaining-half variance; live increment beyond realized score +0.215 pts — even books in-game barely beat the scoreboard.

**Production leads**: NFL.com Friday out>=2 starters fade composed ON the chain scores **57.31 percent (+2.18 pts, week P+ 0.9954)** on 2022-2024 — the strongest composed figure ever measured here, sitting ABOVE the leak ceiling and therefore flagged small-n/selection-inflated pending prospective confirmation; paste-ready challenger rule `nflcom_friday_refresh_out2_starters_v1` drafted with refresh-path integration contract. VI multi-book dispersion screen: mechanism dead at this instrument (spread-SD split-half -0.042), recorded honestly across three cells.

## 2026-08-23 Wave 5 (harvest: challengers live, PBP-08 finds real residue)

**PBP-08 matchup interactions — first mean-edge cell since NFL.com to resolve positive-shaped**: protection mismatch (top-quartile pressure-allowed offense vs top-quartile pressure-generating defense, back the defense) scores **+0.336 pts, [+0.014,+0.658], excluding zero on BOTH blockings**, era-consistent (+0.45/+0.23), mirror nulls clean (`docs/pbp08_matchup_screen.md`). Exactly the coarse-pricing residue the leak-ceiling analysis predicted could exist; mined family -> earns one predeclared confirmation look, not a claim. Pass-mismatch leans positive (P+ 0.81), unresolved.

**Two challengers registered and wired into the publish flow before Week 1 locks Sep 8** (`artifacts/prospective/challengers.json`, now 27 entries):
1. `movement_rule_composed_v1` — follow the market when the captured line moves >=1.0 pt off the frozen Tuesday line, else keep chain pick. Evidence +1.53 pts P+ 0.894/0.930 attribution; composition caveat disclosed (live composes onto the four-member OR union vs the measured three-member chain).
2. `nflcom_friday_refresh_out2_starters_v1` — flip iff picked team carries >=2 starter-caliber Outs on the week's FINAL NFL.com page under the Friday-16:00-ET freshness gate. Evidence +2.18 pts P+ 0.9954 on three seasons only; selection-inflation caveat carried verbatim.
Both fail-open end to end (stale capture / missing snapshot / no snap counts -> skipped or chain pick, never a broken publish); publish-time result-key map extended; 1721 tests passing.
   **2026-09-01:** a 5-cell predeclared expansion battery (`docs/movement_expansion_battery.md`, results appended 2026-09-01) tested untested magnitude (2.0 pt) and timing (Thursday pre-TNF, Saturday midday) variants of this rule on the rotation-registry-governed `[2020, 2021]` window. All five read negative-leaning on point estimate but stay `unresolved_below_power` (P+ 0.07-0.37; no interval fully below zero on both blockings) -- none is folded into the live rule; its evidence chain (+1.53 pts, P+ 0.894/0.930, full 2020-2025 archive) is unchanged.

## 2026-08-23 Dashboard redo (Ledger-Terminal hybrid)

After three failed themed attempts, the dashboard was rebuilt from researched first principles (docs research brief: Stripe/Linear/Vercel/TradingView/FotMob/Bloomberg/FiveThirtyEight teardowns distilled into 15 binding parameters — one accent voltage, tabular figures everywhere, <=6 type sizes {11,12,13,14,17,24}, 4px grid with two row densities, borders-over-shadows, earn-every-surface, two info levels per screen with the third behind interaction, hed/dek per section, designed-not-inverted dark mode). Main view = four-panel terminal grid (summary / continuous week-board table with expandable why-this-pick sub-rows / ledger mini / challenger watch); cards flattened to hairline sections site-wide; Observatory theme fully removed. Cold-read QA found 6 blockers + 12 should-fixes — ALL fixed (leaked audit prose replaced with plain-English blurbs at render source, false no-jargon dek corrected, computed-not-hardcoded summary counts, placeholder blocks fail-quiet, decorative sort glyphs removed, self-narrating badges deleted, P+ floored >0.99 with n adjacent, disclaimer deduplicated, dark-token contrast fixes, best-pick legend added). Index visible words -13.3%; chrome color census 8 hexes (budget 10); NE@SEA Wednesday slot verified real in schedules source.

## 2026-08-25 Wave 7 (lock-day silent-no-op sweep)

Three weeks before Week 1 locks, an audit of what the Sep 8 run would actually
record found the strongest wired challenger could never have recorded anything.

**The NFL.com Friday out>=2 arm was structurally dead** (`docs/nflcom_friday_refresh.md`
"2026-08-25 correction"). Its freshness gate demanded a page fetched at or after
Friday 16:00 ET AND before the week's EARLIEST kickoff — a Thursday night in
every 2026 week but week 18. Measured unsatisfiable on 7 of 7 real weeks (gate
opens ~19.8h after the Thursday kickoff; 43.7h after Week 1 2026's Wednesday
opener). The arm failed open into permanent silence with no error to notice.
Corrected in both implementations (`prospective.py`, `nflcom_refresh_overlay.py`)
to the per-game boundary the codebase already encodes,
`pick_refresh.pick_deadline` = min(own kickoff, Sunday 16:00 ET lock); a Friday
page now scores the Sunday/Monday slate and drops only the Wed/Thu games it
genuinely post-dates. Pinned by `test_a_thursday_game_no_longer_silences_the_whole_week`
in both test files.

**The published +2.1795 was measured on a population the corrected gate excludes.**
`scripts/nflcom_friday_refresh_feature.py` joins Out counts on (season, week, team)
with no kickoff filter, so Wed/Thu games consumed a Friday page published after
their own kickoff. Re-scored from the frozen artifact with the study's own
machinery (reproduction gate matched the published figure to 1.3e-5): the
production-reachable estimate is **+1.9471 accuracy points, week-blocked
[+0.1416, +3.7635], P+ 0.9827** (n=719; season-blocked [+0.4367, +4.0984],
P+ 1.0000). 61 of 799 games excluded (57 Thu, 2 Wed, 2 Fri); 7 of the 67 changed
picks sat on excluded games and ran 5/7 for the arm vs 2/7 for the chain.
Recorded as `nflcom_refresh_out2_starters_on_chain_gate_admitted`
(`unresolved_below_power`; registry now 448 signals). The signal survives the
correction about 0.23 points smaller — quote +1.95, not +2.18.

**Live injury capture built** (PER-03, `docs/nflcom_injuries_sourcing.md`): the
only local NFL.com snapshot covered 2022-2024 and nothing in the weekly pipeline
refreshed it, so the arm had no 2026 data at all regardless of the gate.
`ingest_nflcom_injuries.py --current` now resolves the live REG week from
schedules and fetches only that page into a fresh UTC-stamped snapshot
(verified across dates; each run preserves a revision). Snapshot selection was
made week-aware in the same change — it had read the lexicographically newest
directory, so the first weekly capture would have hidden the 2022-2024 backfill
from every historical read. **First live 2026 capture taken 2026-08-25.**

**Scheduling moved out of Windows Task Scheduler** (`docs/capture_scheduling.md`).
The eight opaque task entries are replaced by `scripts/capture_scheduler.py`,
which holds all 15 jobs in version control. GitHub Actions was evaluated and
ruled out on two measured facts: the repo is PUBLIC while the odds feed is
purchased (artifacts would be publicly downloadable, and MKT-09's licensing
audit is still open), and `odds-ingest` requires a local feature table a fresh
runner lacks. The design schedules WINDOWS rather than instants — each job has
a grace period, so a late run still captures instead of losing the week, and a
window that closes unrun is recorded as MISSED rather than vanishing.
`ALREADY-CAPTURED` absorbs the benign duplicate cases (the Windows tasks still
exist and could not be removed by the agent session; whichever runner fires
first satisfies the other), which is what keeps MISSED meaningful as an alarm.
Persistence is a Startup-folder shortcut — an ordinary file, no service, no
admin. This also covers `model_only_refresh_incumbent` and
`injury_signal_refresh_tilt`, which record only via `refresh-picks`.

**Two further silent-no-op risks confirmed, both operational not code:**
`model_only_refresh_incumbent` and `injury_signal_refresh_tilt` record ONLY via
`nfl-ats refresh-picks --record-decisions`, which `weekly-run` never calls; and
`mod07_weak_signal_stack` records via `weekly-run` step 11, not via a bare
`publish-predictions`. Consequence for Sep 8: the lock-day command must be
`weekly-run --record-decisions`, and the refresh passes are handled by the
catch-up runner above rather than by anyone remembering them.

Gates at wave close: ruff format 659 files, ruff check clean, mypy clean
(107 files), pytest **1,894 passed**.

## 2026-08-25 Wave 8 (lock-day chain proven, PBP-08 played, forecast arm settled)

Fourteen days before Week 1 locks, three things closed.

**The lock-day recording chain rehearsed clean end to end for the first time.**
The 2026-08-24 rehearsal crashed twice after the card write and never obtained
a clean summary; its own fix-list asked for a re-run. Two new tracked scripts do
it: `scripts/lockday_rehearsal.py` drives every real recorder at a simulated
lock instant against an isolated artifacts root, and `scripts/lockday_verify.py`
is the aggregate check that did not exist. Result (measured): **18 recorded,
3 gated, 0 MISSING of 21 active challengers**, paper ledger 16 rows, Best Pick
`2026_01_MIA_LV`.

Two guards make this chain unrehearsable at wall-clock time and pull in opposite
directions — `refuse_if_outside_recording_lock_window` needs a simulated `now`
inside the lock week, while the 36-hour arrests-snapshot guard needs a fetch
close to that instant. The rehearsal shifts the CLOCK, not the data: hard-linked
data mirror, only the 3.7 MB arrests tree real-copied so one snapshot can be
restamped. Nothing fabricated reaches the production data root.

**The finding that made silent no-ops invisible: prospective evidence lives in
FOUR ledgers, not one** (`prospective/challenger_decisions.parquet`,
`injury_signal_refresh_decisions.parquet`, `pick_revisions.parquet`,
`nflcom_friday_refresh_decisions.parquet`). Any audit reading only the shared
ledger reports four of the active challengers as missing when they are fine —
the first version of this rehearsal's own check made exactly that mistake.
Compounding it, `cli._cmd_publish_predictions` wraps seventeen recorders in
fail-open `try/except`, so zero rows and a successful run look identical.
`lockday_verify.py` reads all four, cross-references the run's JSON summary, and
classifies every challenger recorded / skipped-with-a-named-gate / MISSING.

Operational facts worth knowing before Tuesday: `nflcom_friday_refresh_out2_starters_v1`
**cannot** record at the Tuesday lock (its gate needs a Friday-16:00-ET page); it
records only on a weekend refresh pass. `model_only_refresh_incumbent` records
only games whose pick actually changed. And `mod07_weak_signal_stack`'s
fingerprint resolves to the active model's OWN card — it is comparing
`weak_stack` to itself, the structural no-op flagged open on 2026-08-18. Its
rows will carry no information; registry disposition is an owner call.

**PBP-08 protection-mismatch is now played as a challenger** (`docs/pbp08_matchup_screen.md`,
`src/nfl_ats/pbp08_matchup_flags.py`, `pbp08_protection_mismatch_tilt_overlay.py`).
P+ 0.9785 is far above the 0.5 that makes backing it the favoured side, so it
accrues 2026 evidence from Week 1. The flag builder reproduces the screen's
published counts EXACTLY (733 flagged team-games, 114 pushes dropped). Live
Week 1 reading: 4 of 16 games carry a lean, 3 picks flip.
The rotation-registry confirmation look was deliberately NOT run: the opener
pool has exactly one unspent window left, `[2024, 2025]`, and sizing it from the
screen's own numbers puts the confirmation half-width near +/-0.97 points around
a +0.23 effect. That is a statement about a scarce asset, not a power-based
rejection — nothing is closed.

**weak_stack_v4 (forecast weather as model features) is NOT promoted.**
v3's post-mortem said forecast weather was deferred "for lack of merge surface";
that surface now exists (the completed kickoff-nearest archive, 4,431/4,431).
Six continuous columns, predeclared in `docs/weak_stack_v4.md` before scoring.
Measured on 1,537 paired opener games: baseline **53.36%** (matches production
exactly), candidate **52.30%**, delta **-1.065 points**, week-blocked
[-2.688, +0.595], **P+ 0.0956**. Keeping the incumbent is the ~90/10 favoured
side. 142 picks flipped and the baseline was right on 55.8% of them vs the
candidate's 44.2%.
Recorded as two entries (registry now 450): the primary probability-rule
endpoint `unresolved_below_power` (interval contains zero, no ground claimed),
and the sign-rule endpoint `refuted_mechanism`/`wrong_sign_resolved` (whole
interval below zero on both blockings).
**This closes nothing about forecast weather as a phenomenon.** The live
pick-level forecast challengers — including `forecast_weather_kn_warm_team_cold_late`
at P+ 0.9800 — are a different construction and remain active. The useful,
narrower finding: **the hand-coded cells beat the raw variables.**

Gates at wave close: ruff format 670 files, ruff check clean, mypy clean
(110 files), pytest **1,914 passed**.

## 2026-08-25 Wave 9 (composition re-selection; a correction, and no change)

Owner question: why are we not playing the highest-EV combination of everything
that reads positive? Answered with measurement, and it produced a correction.

**Correction first.** Mid-analysis this session the agent asserted that the
four-member subset `coach + division_revenge + arrests + spread_gap_zone` was
measured best but NOT played, and recommended promoting `spread_gap_zone` into
production. That was wrong. `nfl_ats.clv._FOUR_OVERLAY_POLICY_ID`
(`overlay_union_coach_division_revenge_player_arrests_spread_gap_v1`) is what
`record_paper_decisions` resolves whenever `require_fresh_arrest_overlay=True`,
which `cli._cmd_publish_predictions` always passes — verified by running the
real recorder against the real active model, which reported
`spread_gap_zone_flip_count = 1` on the live Week 1 card. The error came from
reading `a_incumbent_chain` in `max_ev_composition`'s arms table as "what
ships"; it is the FORMER chain, which is precisely why
`overlay_production_chain_coach_arrest_incumbent` exists as its paired control.
Lesson, already in AGENTS.md: verify what production does by running
production's own code path, not by reading a study's label for its baseline.

**What is played (measured, 1,503 opener games):** raw model 53.3599%, played
four-member union **55.4225%, +2.063 accuracy points.** That is the in-sample
argmax of the 2026-08-21 study's 127 subsets — production already is the best
composition this project has evidence for.

**Re-selection over TWELVE members / 4,095 subsets**
(`scripts/overlay_subset_holdout_v2.py`, `docs/overlay_subset_holdout_v2.md`;
predeclared decider = choose on 2020-2022, apply unchanged to 2023-2025).
Widening the search made selection WORSE, not better: shrinkage factor fell
from 0.636 (7 members) to **0.593**, rank stability from Spearman 0.721 to
**0.617**, and the selected subset scored **+0.626** on the holdout against the
former chain's +0.876 on the same 799 games. The "play everything" control is
decisive against itself: all twelve members score **−2.128 points** on the
holdout, flipping 501 of 799 games.

**Single addition to the PLAYED union** — ten candidates, a far smaller
selection space. Ranked on 2020-2022 only, the sole candidate with a positive
marginal is `interim_hc_first_game_tilt_overlay` (+0.142), which fires on five
games across the whole archive and **zero** in the holdout: marginal +0.000.
**No addition to the current policy survives an honest forward test.**

Two holdout numbers were deliberately NOT acted on, because both ranked
negative on the selection half and using them would be selecting on the
holdout: `pbp08_protection_mismatch` +1.001 (P+ 0.8978) and
`forecast_cold_visitor` +0.501 (P+ 0.9039). Both stay challenger-tracked; 2026
supplies the independent read.

Three ACTIVE challengers are negative *in composition* on both halves —
`injury_value_lost` (−3.835 / −1.252), `surface_switch` (−0.284 / −1.627,
holdout P+ 0.0220), `backup_qb_fade` (−0.994 / −1.377). That closes nothing
about any of them standing alone (no resolved wrong sign on both blockings, no
reliability measured, no positive control); it measures only that they subtract
as additions to THIS policy. They are recording-only and cost nothing today.

**Decision: no production change.** Not caution, not a threshold refusal — the
best available addition changes zero holdout games and wholesale re-selection
underperforms what already ships. Registry now 452 signals
(`overlay_subset_reselection_twelve_member_forward_holdout`,
`overlay_single_addition_to_played_union_forward_holdout`, both
`unresolved_below_power`).

The standing rule is unchanged and was never the obstacle: a signal with
`probability_positive` above 0.5 is worth playing. What this wave adds is that
**composition is a separate decision from the signal** — an overlay can be
positive alone and negative on top of four others that flip overlapping games,
and choosing among compositions is itself an act with its own error, measured
here at a 0.59 shrinkage factor.

Gates at wave close: ruff format 672 files, ruff check clean, mypy clean
(110 files), pytest **1,914 passed**.

## 2026-08-25 Wave 10 (colour, a lock-day abort, and two ceilings)

**A lock-day abort, found by accident.** `nfl-ats ingest-player-arrests` raised
`ModuleNotFoundError: No module named 'scripts'`. `scripts` is not part of the
installed package and resolves only when the repo root is on `sys.path`, which
`python -m nfl_ats` provides and the console script does not.
`weekly._cli_runner` dispatches every step IN-PROCESS and step 7 is fail-closed,
so **the documented Tuesday command would have aborted before publishing
anything on 2026-09-08.** Fixed in `cli._repo_root_on_path()`, pinned by a
static test proven to fail without the fix.

The rehearsal had reported "0 MISSING" while this was live, because it drove
recorder FUNCTIONS directly and never touched the entry point. It now runs a
**stage 0 command-surface probe** through the real console script in a
subprocess (including a lazy-import probe executed from outside the repo so a
cwd `sys.path` entry cannot mask it), and a broken surface fails the run's exit
code even when every challenger records.

**Semantic colour, site-wide.** From ~3 meaning-bearing colour elements to
**3,024**, on every page. The palette is validated rather than eyeballed, which
caught three defects: `--critical` vs `--serious` measured **dE 7.5 in normal
vision and 2.7 under deuteranopia** (two "distinct" states that were one
colour); dark failed four checks including `--series-model` below the chroma
floor; `--warning` was used twice and **defined nowhere**; and `model_ledger`
emitted **28 badges no stylesheet ever styled** despite a comment claiming they
reused design-system classes. The 10-hex chrome budget is unchanged -- every new
token is derived via `var()`/`color-mix()`. Colour is never the only channel:
signs, numerals, labels and glyphs all persist, midpoints read neutral, and a
forced-colors block turns status dots into distinct shapes.

**Weather is bounded, and the wind lead is closed as not-worth-building.** The
owner's hypothesis was that forecasting is too inaccurate to help. Measured, the
archive is `kickoff_nearest` -- median lead **6.0 hours**, temperature
**r = 0.9642 / MAE 3.25F**. Wind is genuinely noisy (r = 0.6486). So a control
arm handed the model the weather that ACTUALLY happened:
baseline 53.3599%, forecast arm 52.2954%, **oracle 52.2289%**.
**Oracle minus forecast = -0.066 points** -- the entire headroom available to
better weather forecasting, including wind, is approximately zero. Recorded
`unresolved_below_power` (registry 453), deliberately NOT `bounded_by_control`:
the strict ground requires a control PROVEN able to detect, and this shows an
absence. Scope is the feature-vector construction only; the pick-level weather
cells keep their evidence and stay ACTIVE_PROSPECTIVE.

**The ceiling at the grade we actually play is HIGHER than believed.**
`docs/leak_ceiling_control.md` measured 55.57% and drew the project's headline
"~3.5 points of room" from it -- but that is **close-graded**, and the pool
settles at the opener. Re-run at the opener on the same estimator class
(`scripts/leak_ceiling_opener.py`, 1,503 games): **59.28%** (59.41% at
alpha=1), about **3.7 points above the close-graded ceiling**. Headroom from
the played card's de-inflated ~54.6% expectation is therefore **~4.7 points**,
not ~3.5. That is the project's own thesis -- the opener is the softer line --
now carrying a ceiling rather than only a per-model comparison. It remains a
leak arm bounding ridge on linear designs, and this session's two feature arms
both came back negative, so room existing is not evidence any family reaches
it. Nothing recorded to the registry; no rotation window spent.

Gates at wave close: ruff format 674 files, ruff check clean, mypy clean
(110 files), pytest **1,925 passed**.

## 2026-08-26 Wave 11 (the two graph lanes meet, and a measurement artifact)

Two lanes finished earlier the same day and did not meet. The engine
(`src/nfl_ats/graph_ratings_v2.py`, predeclared in `docs/graph_ratings_v2.md`)
accepted exactly two edge signals, `residual` and `raw_margin`, and raised on
anything else. The input screen (`docs/graph_input_screen.md`) had ranked 83
statistic families down to **38 cluster representatives** for the express
purpose of feeding them to that engine. There was no way to feed one.

**The `team_stat` arm is that mechanism.** `edge_signal="team_stat"` with
`signal_column=` builds one graph per screened statistic, edges weighted by
`home_<family> - away_<family>`. Five of the 38 representatives use the feature
table's *suffix* naming (`gap_division_revenge_home`) rather than the prefix
form, so `signal_column_pair=` names the columns explicitly -- without it those
five were silently unreachable, not merely inconvenient. `cfb_structural_
coherence` now refuses the arm by design: its correlation would be the rating
diff against the very quantity its edges were built from. 11 new tests
(2,088 -> 2,099), including a known-answer identity against the raw-margin
control and a leakage regression for the new input family.

**The mechanism check that had to come first, and cost no window.** Is a
graph-propagated statistic actually different from the raw differential the
input screen already tested? Measured over all 38, at the frozen config, no
outcome column touched: median |r| = **0.368**, range **0.004 to 0.779**. At the
median the propagated rating shares about 14% of its variance with the statistic
it came from. The lane is not a no-op, so the window was worth spending.

**Structural hyperparameters frozen from CFB, on the honest reading rather than
the convenient one.** The CFB grid returned residual-arm coherence of -0.000287,
-0.001013, -0.005681 and +0.004119 -- every reading within +-0.006 of zero, so
that grid **did not resolve** `half_life_weeks`, and taking its +0.004119
"winner" (16 weeks) would have been ranking noise. The one setting CFB actually
resolved is the raw-margin control at alpha 0.85 / half-life 8 weeks (+0.531),
which are also the module defaults. Frozen there.

**The instrument check found a real artifact in the measurement, not in the
plumbing.** The positive control passed emphatically -- planting the realized
`ats_margin` as the treatment feature scores **+51.9 to +53.1** accuracy points,
P+ 1.000, so the instrument can see an effect. The null did not, and the first
version of that check was itself wrong twice over. It used a **single**
permutation and read its one draw of -2.53 points as a broken harness; one draw
is not a test, and every family in a run shares it. Rebuilt at 200 permutations
(model fitting never sees the grading outcome, so the fitted models are reused
and only the grade changes), the null still does not centre on zero -- **and
that is the design, not a defect**. Within-week permutation preserves each
week's realized home-cover rate `c_w`, so an arm picking home at rate `h_w` has
expected null accuracy `1 - h_w - c_w + 2*h_w*c_w`, making the expected null
delta `2*mean_w[(h_treat - h_control)(c_w - 0.5)]`. That closed form reproduced
the Monte-Carlo null means to within ~0.3 points (-1.856 vs -1.892, -1.400 vs
-1.266, -1.111 vs -0.842, -0.189 vs -0.216). The cause is that these arms carry
**55-67% home-pick rates against a 49.67% cover rate**.

**The screen: one close-graded look on the rotation-assigned window
[2011, 2013]**, 746 games, 51 weeks, 38 families, declared in
`docs/graph_ratings_v2_screen.md` before any outcome was scored. Recorded as 38
weak-signal entries (registry **567 -> 605**) plus one `rotation record` look
that spent the window; family `graph_ratings_v2_team_stat` stays **open** with
two eligible close windows left.

**Against zero** the arm leans slightly negative: mean paired delta **-0.123**
points, positive in 12 of 38, random-effects pool **-0.107, 95% [-0.424,
+0.209]**. **Against each family's own permutation null** it is a **coin flip**:
median percentile **50.2**, 19 of 38 above their null centre, sign test
**p = 1.000**, null-adjusted mean **+0.007** points. The mean null offset is
-0.131 points -- essentially the entire apparent negative lean. So the finding
is not that schedule-adjusting a statistic hurts; it is that at this window's
~2-point resolution **the transform is indistinguishable from using the
statistic raw**, and a naive zero-reference would have reported a negative that
belongs to the home-tilt artifact.

The single row that justifies having built the null reference:
`off_rush_epa_per_play` reads **P+ 0.911 against zero** and sits at the
**53.5th percentile of its own null** (+1.609 observed against a +1.450 null
centre). By the conservative reference the leaders are `def_yards_per_play`
(95.5th percentile) and `injury_skill_unavailability` (93.5th);
`off_sack_rate` is the only family whose interval excludes zero on the positive
side (+2.949, P+ 0.987) but roughly 40% of that headline is its null offset.
Four families sit at or below the 2.5th percentile of their own null and one at
or above the 95th, uncorrected across 38 cells -- roughly what chance produces,
so no individual tail is a finding. Disclosed: the 38 cells share the same 746
games so the pooled interval overstates precision; two arms
(`special_teams_lineup_continuity`, `injury_skill_epa_value_lost`) hit training
folds with no observed values and partly collapse to the baseline; a one-family
plumbing smoke test (`def_takeaway_rate`, 2012) preceded the window assignment
and is carried in the family's declaration; and the window shares 2013 with the
input screen's own close-graded selection window.

Also this day, before this wave: the dashboard's plain-English ledger rows and
self-generating pages shipped (`826dbea`), ledger summaries went from a median
of 71 words to 9 with zero repeats (`0082074`), and the cover-curve pinch and
run-together ledger labels were fixed (`ef03b1d`).

Gates at wave close: ruff format clean, ruff check clean, mypy clean (114
files), pytest **2,099 passed**.

