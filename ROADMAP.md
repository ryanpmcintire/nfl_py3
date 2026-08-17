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
metric. First predeclared measurement (`docs/opener_evaluation.md`): the
frozen active model scores **52.50% against openers** on 1,537 paired
2020–2025 games (vs 51.09% against closes on the same games; paired delta
+1.35 points with ~99.9% probability positive; season-blocked interval vs
the coin flip excludes 50%). Closing lines, CLV, and vig are secondary.
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
| RWB-01 | ✅ | Season-aware team state | Current-season game count and explicit offseason regression |
| RWB-02 | ✅ | Feature-family registry | Named, documented groups and configurable model allowlists |
| RWB-03 | ✅ | Walk-forward ablation runner | Comparable market/context/Elo/form/full experiment artifacts |
| RWB-04 | ✅ | Paper bankroll engine | Flat, full/fractional Kelly, caps, weekly exposure, drawdown |
| RWB-05 | ✅ | Local research dashboard | Read-only views for data, features, backtests, bankroll, picks |
| RWB-06 | ✅ | Model explanation artifact | Named logistic coefficients/missing indicators per fitted model |
| RWB-07 | 🚧 | Season-level scorecards | Accuracy, Brier, log loss, ECE, ROI, CLV, and intervals by season |
| RWB-08 | ✅ | Block bootstrap uncertainty | Confidence intervals resampled by NFL week and season |
| RWB-09 | 🚧 | Experiment registry | Config hash, code revision, source snapshot, metrics, notes |
| RWB-10 | ✅ | Prospective prediction ledger | Append-only predictions frozen before kickoff |
| RWB-11 | ✅ | Model cards | Intended use, training period, limitations, calibration history |
| RWB-12 | ⬜ | Drift monitoring | Feature, missingness, probability, and calibration drift |
| RWB-13 | ✅ | Dependence audit | Team error autocorrelation and season-preserving permutation null |
| RWB-14 | ✅ | Data-feasibility registry | Verified releases, nonempty seasons, row counts, timestamp semantics, source regimes, and effective sample-size tier |
| RWB-15 | ✅ | Evaluator sensitivity audit | Exact active-model reproduction plus null/permuted and known 0.5/1/2-point positive controls across repeated synthetic signals |
| RWB-16 | ✅ | Sensitivity-aware experiment review | Completed August 2026: artifact-verified inventory, ~130–150-look multiplicity ledger, and three predeclared replications on untouched 2013–2017 windows; all three reopened leads resolved (see below) |
| RWB-17 | ✅ | Rotation registry | Complete August 2026 and in production use: git-tracked `registry/rotation_registry.json` plus `nfl-ats rotation declare/assign/status/record`, each family drawing one earliest-eligible confirmation window, forward-chained splits enforced in code, and recording a look spending the window permanently (`docs/rotation_registry.md`). Rule 9 (warm-up eligibility, `MIN_ELIGIBLE_START_SEASON = 2013`) was added before any window was spent under it, after the registry correctly offered `best_pick_ranker` a first block the evaluator could not score. Two looks have since been recorded through it — `best_pick_ranker` on [2013, 2015] and `mod07_weak_signal_stack` on [2020, 2021], both `unresolved` — and both windows are permanently spent |

## Phase 2 — point-in-time market data

| ID | Status | Item | Definition of done |
|---|---|---|---|
| MKT-01 | ✅ | Live odds provider adapter | Book, market, line, price, observed-at timestamp, raw response hash |
| MKT-02 | ✅ | Opening/current/closing line store | August 2026: six weekly scheduled live captures running (11 books), plus a purchased point-in-time snapshot archive — decision labels for 2020–2025 (paired tue_open+close for 227–272 games every season) plus hourly 2023–2025, playoffs, true openers, and moneylines; 8,746 snapshots, verified read-only backups on two drives |
| MKT-03 | 🚧 | No-vig market probabilities | Documented two-way normalization and favourite-longshot diagnostics |
| MKT-04 | ✅ | Closing-line-value tracking | Complete August 2026: the `clv-score` harness (per-pick points vs close, week-blocked intervals) plus the routine paper-decision ledger — `publish-predictions` appends every published card's pre-kickoff picks at their published line (the first-recorded anchor is never rewritten by a republish), and `clv-ledger` scores the whole ledger against live-store closes with a schedule-close fallback and surfaces the result on the track-record page |
| MKT-05 | ✅ | Cross-book consensus | Median line, dispersion, stale-book and outlier detection |
| MKT-06 | ✅ | Line-movement forecasting | Frozen pilot ran 2026-08-16 after the archive re-fetch (train 2020–2023, validate 2024, one look at 2025): direction-of-movement accuracy 59.5% on 2024's 200 movers and 57.2% on 2025's 194 movers, consistent with the pre-registered sign test (54.9% of 778 games, CI [51.3, 58.4], p=0.007); but magnitude never beat the no-movement MAE baseline (−0.001/−0.019 points) and the frozen ≥0.5-point threshold policy made exactly 1 bet (−0.5 CLV). Direction replicated, no exploitable magnitude edge; both retained. Production wiring shipped: `predict-close` writes the Week Board's close_predictions artifact from each week's live Tuesday capture and fails closed without one |
| MKT-07 | ✅ | Market residual model | Estimate only the correction to a market prior |
| MKT-08 | 🔬 | Timing policy | Compare fixed weekly timestamps and news-triggered updates |
| MKT-09 | 🚧 | Provider licensing/quota audit | Terms, redistribution limits, cost, retention, failure policy |
| MKT-10 | ✅ | Free historical close audit | Versioned public close archive plus 2025 opener/nine-book close sample, licenses, normalization, source comparison |

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
| XLG-05 | ⬜ | Hierarchical CFB→NFL transfer | Compare matched NFL-only, naïvely pooled control, CFB-pretrained, and partially pooled models on NFL-only outer weeks. The first candidate vehicle — the role-continuity family — failed the XLG-03 screen (August 2026, `docs/cfb_role_features.md`), so this now waits for a family that first clears the CFB benchmark |
| XLG-06 | ⬜ | Rookie/young-player priors | Link college usage/value, recruiting, transfers, and draft identity to NFL players with explicit uncertainty and decay as NFL evidence accumulates |
| XLG-07 | ⬜ | Cross-league availability semantics | Determine whether historical CFB injury reports are genuinely pregame and complete enough to learn availability; fail closed if timestamps or missingness are ambiguous |

## Phase 3 — better football state

| ID | Status | Item | Definition of done |
|---|---|---|---|
| PBP-01 | ✅ | Versioned play-by-play ingestion | Partitioned Parquet snapshots and required-field contracts |
| PBP-02 | ✅ | Situation filters | Remove kneels and garbage time with versioned definitions |
| PBP-03 | ✅ | Drive table | Possessions, starting field position, points, success, and duration completed; ATS screen did not support promotion |
| PBP-04 | ✅ | Stable team efficiency | Early-down EPA, success, explosive rate, pressure, PROE |
| PBP-05 | ✅ | Opponent adjustment | Weekly time-decayed ridge offense/defense decomposition completed; no stable improvement, retained research-only |
| PBP-06 | ⬜ | Special teams | Kicking, punting, returns, field position above expectation |
| PBP-07 | ⬜ | Penalty state | Rate, leverage, accepted/declined distinction, regression to mean |
| PBP-08 | 🔬 | Scheme/matchup interactions | Protection-pressure, explosive pass, personnel, coverage proxies |
| PBP-09 | ⬜ | Pace and possession forecast | Expected drives and play volume distribution |
| PBP-10 | 🔬 | Referee effects | Only if assignments are point-in-time and effects survive shrinkage |

## Phase 4 — players, coaches, injuries, and offseason priors

| ID | Status | Item | Definition of done |
|---|---|---|---|
| PER-01 | ✅ | Weekly roster/participation snapshots | Immutable weekly roster plus season-partitioned 2016–2025 participation snapshots, hashes, source regimes, and outcome-time contracts |
| PER-02 | 🚧 | Quarterback state | Starter probability, player EPA/CPOE, backup adjustment |
| PER-03 | 🚧 | Injury report history | 76,784 canonical 2009–2024 rows ingested with a 24-hour cutoff; weekly files behave as final observations, not revision streams; replacement live source remains |
| PER-04 | 🚧 | Depth chart history | Starter/backup roles without using later revisions |
| PER-05 | 🚧 | Snap-weighted player value | Box-score value reached 52.14%; a fixed participation extension fell to 51.71% and was rejected; learned availability and stronger value targets remain |
| PER-06 | 🚧 | Roster continuity | Lagged lineup/roster continuity is implemented and isolated as the strongest current player-family component; returning-snap offseason priors remain |
| PER-07 | ⬜ | Coaching/coordinator changes | Coach IDs, tenure, scheme tendencies, change flags |
| PER-08 | ⬜ | Transaction-aware preseason prior | QB, roster, coaching, draft/free-agency adjustments |
| PER-09 | 🚧 | Latent player ratings | First season-lagged offense/defense adjusted-plus/minus is reproducible but failed its matched ATS screen; hierarchy, units, and special teams remain |
| PER-10 | 🔬 | Injury scenario mixture | Forecast weighted across active/inactive player scenarios |
| PER-11 | ✅ | Learned play probability | Prior-season report/practice/position rates improved player Brier 0.09500 → 0.09056 and ATS 52.14% → 52.24%; blocked ATS intervals remain unresolved |
| PER-12 | ✅ | Expected role delivery | Closed at the intermediate target August 2026: the clipped two-part delivery model lost to both parents on delivered-share MAE in all 11 seasons (0.1888 vs 0.1606/0.1621) because injury-listed players who play at all deliver ~their full prior role (median ratio 1.01); no ATS rows were generated |
| PER-13 | ⬜ | Reliability trait priors | Per-player durability from 16-season injury/participation history and roster-status volatility (including league suspensions from weekly roster status codes) as priors inside the learned-availability model; validate on the player-level availability target before any ATS screen |

## Phase 5 — weather, venue, rest, and travel

| ID | Status | Item | Definition of done |
|---|---|---|---|
| ENV-01 | ⬜ | Forecast-time weather | Archive the forecast that existed at the decision timestamp |
| ENV-02 | ⬜ | Stadium/roof/surface history | Venue state and roof decision where available |
| ENV-03 | ⬜ | Travel geometry | Distance, time-zone change, international games, return travel |
| ENV-04 | ⬜ | Rest context | Bye, short week, mini-bye, consecutive road games |
| ENV-05 | 🔬 | Weather interactions | Wind × passing/kicking style; heat × pace; surface × unit traits |
| ENV-06 | 🔬 | Circadian effects | Test local body-clock hypotheses with aggressive shrinkage |

## Phase 6 — modeling and probability distributions

| ID | Status | Item | Definition of done |
|---|---|---|---|
| MOD-01 | ✅ | Regularized logistic baseline | Time-safe preprocessing and chronological calibration |
| MOD-02 | ✅ | Histogram boosting comparator | Bounded complexity and same evaluation contract |
| MOD-03 | ✅ | Margin regression | Predict conditional mean and compare residual vs market spread |
| MOD-04 | ✅ | Margin distribution | Quantiles or parametric distribution; validate coverage and tails |
| MOD-05 | ⬜ | Joint score/total model | Coherent home/away score and total probabilities |
| MOD-06 | ⬜ | Bayesian dynamic team model | Partial pooling, uncertainty, explicit offseason evolution |
| MOD-07 | 🚧 | Ensemble/stacking | Out-of-fold predictions only; weight stability constraints. The first predeclared vehicle — the weak-signal stack (player value composite + learned availability + three documented opener-bias families, profile `weak_stack`) — took its one registry look on [2020, 2021] at the Tuesday-opener grade (August 2026, `docs/mod07_stack.md`): **53.29% vs the frozen model's 51.32% on 456 paired games, +1.97 points, week-blocked [−1.10, +5.00], `probability_positive` 0.8745** — short of the predeclared 0.90 threshold, so `unresolved`, not promoted, and the window is spent. The entire delta comes from the 49 picks the arms split (candidate 29–20). Prospective 2026 scoring as a frozen challenger is the recommended next evidence; retuning the stack and re-scoring [2020, 2021] is inadmissible |
| MOD-08 | 🔬 | Distributional boosting | Quantile/NGBoost-style margin and total forecasts |
| MOD-09 | 🔬 | Sequence model over drives | Small temporal model, benchmarked against summary features |
| MOD-10 | 🔬 | Graph model | Player/team matchup graph only after player state is reliable |
| MOD-11 | 🚧 | Calibration suite | Platt, isotonic, and beta are leak-safe and evaluated; calibration-by-regime remains |
| MOD-12 | ✅ | Hyperparameter protocol | Frozen profile/Ridge/calibration budget selected on prior seasons and scored on next-season folds |
| MOD-13 | ⬜ | Missingness audit | Drop source-era indicators and test explicit availability flags |
| MOD-14 | ⬜ | Era weighting | Compare rolling training windows and time-decayed sample weights |
| MOD-15 | ✅ | Temporal schedule-graph ratings | Leak-safe PageRank/HITS and ridge/SRS comparator completed; graph selected in 0/8 outer seasons and was not promoted |
| MOD-16 | ⬜ | Conditional margin variance | Replace the pooled residual distribution with game-level heteroskedasticity; accepted only if held-out cover/push/loss calibration beats the pooled baseline. The predeclared CFB screen (August 2026, `docs/margin_variance.md`) **failed**: a Ridge scale model on mismatch/total/pace/experience made clean-core cover log-loss resolvably worse (week-blocked [−0.00056, −0.00012]) — the pooled distribution is already near-correctly calibrated. Only a genuinely NFL-specific variant (QB/backup status, weather) with a new ledger-aware predeclaration remains admissible, and it is deprioritized by this result |

## Phase 7 — simulations

| ID | Status | Item | Definition of done |
|---|---|---|---|
| SIM-01 | ⬜ | Margin Monte Carlo | Sample calibrated margin distribution and derive ATS probabilities |
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
| BET-09 | ⬜ | Responsible-use controls | Paper mode default; prominent limitations and no auto-wager path |

## Phase 9 — football pools

| ID | Status | Item | Definition of done |
|---|---|---|---|
| POL-01 | ⬜ | Pool rule configuration | ATS, straight-up, confidence, survivor, scoring, entry count |
| POL-02 | ✅ | ATS weekly card | Force a pick for every game and rank confidence |
| POL-03 | ✅ | Straight-up probability model | Separate calibrated target, never reuse cover probabilities |
| POL-04 | ⬜ | Pick-popularity input | Manual/imported ownership estimates and uncertainty |
| POL-05 | ⬜ | Contest utility optimizer | Max expected points or probability of finishing first |
| POL-06 | ⬜ | Multi-entry diversification | Correlated entries with controlled overlap |
| POL-07 | ⬜ | Survivor planner | Current survival probability plus future team opportunity cost |
| POL-08 | 🔬 | Opponent-field simulation | Simulate standings and strategic picks for winner-take-all pools |
| POL-09 | 🚧 | Best Pick ranker | The pool pays one Best Pick per week and our confidence ordering is flat (top-\|residual\| scored 48.6% over 107 weeks). Three signals were predeclared and screened once on the registry window [2013, 2015] (August 2026, `docs/best_pick_ranker.md`): `calibrated_probability` (−8.16 points vs all-pick, `probability_positive` 0.0925) and `key_number_distance` (−6.19, 0.170) both made the weekly top-1 pick **worse** and are closed; `sweep_robustness` scored +5.57 points at 0.7955, clearing the predeclared 0.75 screen gate. Its earned opener confirmation then ran once on [2020, 2021] with the deployed `player` profile: **top-1 60.0% (21/35) vs 51.32% all-pick, +8.68 points, week-blocked [−7.00, +22.88], `probability_positive` 0.865** — clears the predeclared 0.75 gate, verdict `confirmed`, both windows now spent. Consequence per the predeclaration: **use `sweep_robustness` to choose the Best Pick in 2026** — a pool-play decision, no activation and no model change. Two disjoint windows, two grades, same direction; but 86 top-1 picks total and a rank correlation of +0.067 (p=0.099), so expect regression toward the all-pick rate |

## Phase 10 — dashboard and operations

| ID | Status | Item | Definition of done |
|---|---|---|---|
| UI-01 | ✅ | Local Streamlit dashboard | One command, no external service, graceful empty states |
| UI-02 | ✅ | Data health view | Snapshot provenance, coverage, missingness, season counts |
| UI-03 | ✅ | Backtest view | Scorecards, season metrics, calibration, cumulative returns |
| UI-04 | ✅ | Paper bankroll view | Equity curve, drawdown, stakes, settings, ledger |
| UI-05 | ✅ | Prediction view | Weekly probabilities, passes/picks, lines, model cutoff |
| UI-06 | ✅ | Experiment view | Feature-set comparisons and sortable metrics |
| UI-07 | ✅ | Team explorer | Pregame state trends and matchup comparison |
| UI-08 | 🚧 | Model explanation view | Coefficients/SHAP with stability and caveat labels |
| UI-09 | 🚧 | Pool workbench | Rules, entries, confidence ranks, ownership scenarios |
| UI-10 | ✅ | Guided interpretation | Plain-language verdicts, score references, glossary, and historical/live separation |
| UI-11 | ✅ | Active-model synchronization | One atomic manifest links the exact evaluation and weekly forecast used by every headline page |
| UI-12 | ✅ | Question-based navigation | Five plain-language destinations replace internal lab/tool names; researcher diagnostics live under Advanced research |
| UI-13 | ✅ | GitHub weekly card | Synchronized publisher places the full current card at the top of README and in a tracked standalone page |
| OPS-01 | ⬜ | Scheduled refresh | Data → features → frozen predictions with idempotent jobs |
| OPS-02 | ⬜ | Artifact retention policy | Keep manifests/ledgers, compact or prune large derived files |
| OPS-03 | ⬜ | Container image | Reproducible local/server dashboard deployment |
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
research features rather than defaults.

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
rule frozen before any run. All three closed. (1) The raw-PBP market-residual
bundle, whose post-hoc 2018–2025 comparison showed +1.69 points, scored −0.08
points against base on 1,247 never-selected-on 2013–2017 games with margin
error resolved worse; the +1.69 is recorded as within-window noise plus build
selection. (2) QB-plus-continuity at the declared alpha-1 scored exactly +0.00
points on 997 games in 2014–2017 (arms disagreed on 176 picks and split 88–88)
with all probability diagnostics worse; the 52.34/52.63 figures are recorded as
selection artifacts. (3) Expected role delivery failed its intermediate-target
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
| Closed by replication (August 2026) | QB plus lineup continuity | The predeclared alpha-1 candidate scored exactly +0.00 points vs base on 997 untouched 2014–2017 games with all probability diagnostics worse; 52.34/52.63 are recorded as within-window selection artifacts. The 2014–2017 window is spent for the player family. |
| Closed at the CFB benchmark (August 2026) | CFB role-continuity feature family | The predeclared dropback/carry participation-continuity family (season-scoped, streak-capped per the absence-separation study) scored −0.67 accuracy points vs the frozen XLG-03 arm on 8,933 clean-core games, with week- and season-blocked Brier/log-loss intervals excluding zero in the wrong direction (`docs/cfb_role_features.md`). Participation disruption is market-priced. Any successor (roster-aware departures, replacement quality, XLG-07 availability semantics) is a new predeclaration, not a rerun. |
| Closed at the CFB benchmark (August 2026) | Conditional margin variance (MOD-16 screen) | The predeclared Ridge residual-scale model (mismatch/total/pace/experience, clipped [2/3, 3/2]) produced real per-game variation (p10 0.905, p90 1.115) yet made clean-core cover log-loss and Brier resolvably worse under week and season blocking (`docs/margin_variance.md`). The pooled out-of-time residual distribution is already near-correctly calibrated. Only an NFL-only-feature variant with a new ledger-aware predeclaration remains admissible; distributional successors (MOD-05, MOD-08) are separate predeclarations. |
| Redesign, then reopen | Snap-weighted player value | The +0.10-point box-score extension was too small to resolve and its value proxy is coarse. Revisit with replacement quality, position hierarchy, and CFB/NFL partial pooling—not the identical two-field rerun. |
| Revisit only through transfer | Opponent-adjusted PBP and matchup effects | Small probability movement could be below NFL power, but ATS remained below 50%. Use CFB to choose low-dimensional mechanisms, then freeze an NFL transfer test; do not reopen the broad NFL bundle. |
| Revisit only after a stronger signal | Beta/other calibration | Calibration can improve probability magnitudes but does not create side information. Re-evaluate inside a newly fixed player model, not as an alpha search by itself. |
| Keep closed in current form | Participation offense/defense RAPM | Accuracy fell 0.43 points and Brier worsened with a season-blocked interval excluding zero in the wrong direction. Position-unit or matchup hierarchy would be a new model, preferably learned with CFB; alpha retuning is not admitted. |
| Keep closed in current form | PageRank/HITS schedule graph | Graph candidates were selected in 0/8 outer seasons and worsened probability/margin diagnostics. CFB may support player/unit graphs, but it does not warrant rerunning this team graph. |
| Keep closed — confirmed by replication (August 2026) | Drive aggregates and broad raw PBP bundle | The re-review found the matched market-residual comparison had never been computed and showed +1.69 points post hoc; the predeclared 2013–2017 replication scored −0.08 points with margin error resolved worse, confirming closure. The 2013–2017 window is spent for this family. Preserve the layers for a future joint score/pace distribution, not another ATS screen. |
| Keep closed | Broad 48-row player selection grid | The nested selector failed and pooled winners are multiplicity-exposed. Its rows may nominate one mechanistic hypothesis, but the grid itself is not independent evidence. |

1. Maintain the prediction-safety contract and add a regression canary for
   every production error or newly supported output type.
2. The point-in-time market stack is code-complete: the purchased 2020–2025
   snapshot archive is verified and backed up, weekly scheduled captures
   continue on the free tier, the frozen MKT-06 pilot has taken its one look
   (direction replicated, no magnitude edge) with `predict-close` wired to
   the Week Board, and the MKT-04 paper-decision ledger records every
   published card's picks at publication (`publish-predictions`) and scores
   them against the close (`clv-ledger`, surfaced on the track-record page).
   Remaining market items are research questions (MKT-03 diagnostics, MKT-08
   timing policy) and the MKT-09 licensing audit.
3. The XLG-04 chain is complete end-to-end: role delivery replicated
   cross-league for dropbacks and carries (`docs/cfb_role_replication.md`),
   the departure-vs-temporary-absence prerequisite was measured
   (only 15.6%/18.7% of qualified holders return the next season;
   same-season return odds fall to ~10%/7% after four straight missed
   games), and the ONE predeclared role-continuity family was scored
   against the XLG-03 benchmark — it did **not** clear: paired accuracy
   −0.67 points on 8,933 clean-core games (week-blocked [−1.33, +0.01])
   with Brier and log-loss resolved worse under both blockings
   (`docs/cfb_role_features.md`). The market already prices participation
   disruption. No NFL transfer claim is predeclared from this family and
   no retuning of it is admitted.
4. XLG-05 therefore has no cleared mechanism to transfer yet; it waits for
   a family that first clears the CFB benchmark. The remaining CFB-side
   paths are XLG-06 (rookie/young-player priors) and XLG-07 (availability
   semantics), plus CFB screens of the distribution work in item 6.
5. Score the active model and any frozen challengers on prospective 2026
   outcomes only — now at BOTH grades (opener via the live Tuesday captures,
   and close), with the opener grade primary per the pool goal; the
   2013–2017 and 2014–2017 replication windows are spent, and no new
   variant of an existing family may be scored on 2018–2025 without a
   frozen predeclaration that acknowledges the ~130–150-look ledger. New
   pool-targeted leads from the 2026-08-17 literature sweep (peer-reviewed
   opener biases: Week-1 playoff-holdover fade, Week-2 anchoring,
   prior-week recency, low-visibility games moving most) are candidate
   features for the rotation-registry/stacked-signals pipeline.
6. Model the distribution, not just the mean — with MOD-16's simple scale
   model now closed at the CFB screen (the pooled residual distribution is
   already near-correctly calibrated; `docs/margin_variance.md`), the open
   distribution paths are the joint score/total model (MOD-05) and
   distributional boosting (MOD-08), each accepted only on held-out
   distribution calibration; then reliability trait priors (PER-13) and
   pre-snap penalty discipline (PBP-07) as the first low-dimensional
   "intangible proxy" screens, run through the CFB benchmark first where
   the data allows.
7. Use 2016–2025 participation/NGS for position-unit and formation effects;
   individual receiver-corner pairs remain too sparse for an initial model.
8. Attempt drive simulation only after simpler distributional baselines exist.

The dashboard and experiment registry should make failed hypotheses easy to
retain. Negative results are project assets; quietly deleting them invites the
same experiment to be rediscovered and overfit later.
