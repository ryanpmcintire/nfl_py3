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
| RWB-01 | 🚧 | Season-aware team state | Current-season game count and explicit offseason regression. **The regression constant is measurably wrong (2026-08-17) and is the next derived-constant fix.** `offseason_retention = 0.67` is roughly twice what the data supports. Three independent routes agree and none of their intervals contains 0.67: fitting the retention slope per metric and horizon over 486 season-to-season transitions puts all 24 cells' 95% upper bounds below it (median **0.337** at the first-4-games horizon the constant actually governs); the shipped table's own `result ~ diff_point_diff` slope is 0.333 in week 1 against 0.588 in weeks 9-18, implying **0.379**; and a plain within-season-centred regression of next season on prior season gives **0.400**, season-blocked [0.347, 0.460]. At 0.67 the carried state is inflated enough that carrying NOTHING forward beats it on full-season point differential (RMSE 6.16 at retention 0 vs 6.38 at 0.67; 5.75 at ~0.30). A shared ridge coefficient cannot absorb this: with EWM span 8 the offseason initial condition still carries weight 0.78/0.60/0.47/0.37 after 1-4 games, so it is load-bearing for roughly weeks 1-6 while the coefficient is fit mostly on late-season rows -- the early-season state feature is over-weighted by ~1.8x, and no single coefficient fixes a week-varying scale error. The literal has been centralised as `constants.DEFAULT_OFFSEASON_RETENTION` (value unchanged, so no prediction moved) because it shapes the feature table and correcting it is a scored change: screen on CFB first (free under rule 8). Consider per-metric values rather than one global number -- fitted retention runs 0.195 (`off_turnover_rate`) to 0.382 (`off_sack_rate`), and forcing them to share a constant is its own undefended assumption. Conditioning retention on coaching change was tested and does NOT pay out of sample (leave-one-season-out, no better than chance) |
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
| RWB-16 | ✅ | Sensitivity-aware experiment review | Completed August 2026: artifact-verified inventory, ~130–150-look multiplicity ledger, and three predeclared replications on untouched 2013–2017 windows; all three reopened leads resolved (see below) **Swept again 2026-08-17 with a power-aware lens, and half of it does not hold.** Of 27 independent discarded mechanism families: **13 were unresolved below detection power** (recorded as negatives without the evidence), 7 were genuinely refuted mechanisms, and 6 were bounded by a positive control or a tight interval. **The most expensive error is `player_qb_continuity`:** the PRIMARY comparison that closed it and spent [2014, 2017] ran the candidate at ridge alpha 1.0 against a baseline at 10.0, bundling a feature change with an alpha change, and returned exactly 0.000 accuracy. The same artifact's alpha-matched arm returns **+1.1033 points** (season 95% [0.0000, +2.2177]) while the alpha change alone costs **-1.1033** (season 95% [-1.7034, -0.7992], resolvably negative) -- they cancel, and the matched figure nearly reproduces the predeclared 1.25. The deployable question was answered honestly; the FAMILY question was never answered and must not be described as closed. It implies group-wise ridge penalties, since these features want less regularization than the rest of the design. **But the wider hypothesis is NOT supported:** across all independent discards carrying a direction the sign test is 4 candidate / 17 baseline, and restricted to the genuinely-underpowered set it is 4/4, p = 1.000 -- a coin flip at every level of collapsing. Instrument validity was checked rather than assumed: RWB-15's null replicas split 8/16 at an injected effect of exactly zero, and at a true 0.5-point effect the sign is right 13/16 while the interval clears only 4/16, so direction really does accumulate faster than precision here. The one family where it accumulates is **player availability and value** -- five measurements, all positive, p = 0.0625 within the family. Also found: the drive layer's direction is reported BACKWARDS in the docs (they kept the Brier metric that worsened and omitted paired accuracy of +0.72 and +1.16 favouring the candidate). Below-power results now go to `registry/weak_signals.json` (RWB-18) rather than being deleted |
| RWB-17 | ✅ | Rotation registry | Complete August 2026 and in production use: git-tracked `registry/rotation_registry.json` plus `nfl-ats rotation declare/assign/status/record`, each family drawing one earliest-eligible confirmation window, forward-chained splits enforced in code, and recording a look spending the window permanently (`docs/rotation_registry.md`). Rule 9 (warm-up eligibility, `MIN_ELIGIBLE_START_SEASON`, now 2011 and derived from feasibility rather than an undefended default) was added before any window was spent under it, after the registry correctly offered `best_pick_ranker` a first block the evaluator could not score. Two looks have since been recorded through it — `best_pick_ranker` on [2013, 2015] and `mod07_weak_signal_stack` on [2020, 2021], both `unresolved` — and both windows are permanently spent |
| RWB-18 | 🚧 | Weak-signal registry | **Stop discarding effects that are merely too small to see.** The evaluator resolves ~2 ATS points (RWB-15) and almost every single feature is worth a fraction of that, so "no significant effect" has been the EXPECTED outcome for real-but-small signals -- and recording each as a negative quietly threw away the ones that were genuinely there. Two such errors were caught on 2026-08-17 (4th-down aggressiveness and penalty discipline, both filed as "priced" on intervals far too wide to say so). Built `registry/weak_signals.json` (git-tracked, schema-validated) plus `nfl_ats.weak_signals` and `nfl-ats weak-signals status|pool`. Every category-3 result in the taxonomy in `docs/pool_edge_plan.md` is now recorded with its effect, uncertainty and above all its **direction**. Two things then accumulate that no single experiment can buy: **signs**, since under a true null each estimate is a fair coin, so ten of twelve leaning one way is p ~ 0.039 assembled entirely from individually worthless results; and **precision**, since inverse-variance pooling sharpens by sqrt(K). The honest price is pinned in tests -- a 0.5-sigma signal needs about **sixteen** independent companions to clear 1.96 sigma, and four reaches only 1.0 sigma. Three guards: only category 3 is poolable (folding in a refuted mechanism would launder a known failure); overlapping seasons are reported as shared noise rather than hidden; and a pooled estimate justifies ONE predeclared combined look on a window none of the inputs touched, never being a finding itself. Seeded with the two signals caught today; the historical sweep for further below-power discards is the next step |

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
| PBP-05 | ✅ | Opponent adjustment | Weekly time-decayed ridge offense/defense decomposition completed; no stable improvement, retained research-only. **One untested framing remains (2026-08-17):** every prior test *added* the adjusted columns on top of the raw ones, taking the design from 58 to 106–142 columns on ~4,700 rows at alpha 10, so signal was confounded with dimension inflation. The dimension-neutral **substitution** — same column count, adjusted defensive rates swapped in for raw ones — has never been run. Motivation is measured: split-half reliability of team per-play rates over 2009–2012 is **0.80 offense vs 0.46 defense** (full-season, Spearman-Brown; independently reproduced), so the active model's defensive per-play columns are roughly half noise, which is exactly what opponent adjustment exists to fix. **Screened on CFB 2026-08-17 and CLOSED** (`docs/cfb_opponent_adjustment.md`): the dimension-neutral swap moved clean-core margin MAE by -0.0003 points, `probability_positive` 0.463 on 9,093 games; isolating the opponent block from a time-weighting change bundled with it recovers +0.0005 MAE. A deliberate-leak positive control (adjustment fit on all of 2006-2025, so the columns see the future) moved MAE only **+0.0129** (P+ 0.984) -- so the null is measured rather than underpowered, and that figure is a **ceiling on the whole family**. The reliability argument is sound but does not translate: against a market-residual target the spread already prices team quality, so denoising those columns has almost nothing left to improve. A revisit needs a different target, not a better adjustment. See `docs/play_level_audit.md` |
| PBP-06 | ⬜ | Special teams | Kicking, punting, returns, field position above expectation |
| PBP-07 | ⬜ | Penalty state | **Not built, and NOT closed -- the market check is below detection power.** Team penalty rate is 0.0750 +/- 0.0101 with year-over-year reliability +0.261, so there is a real if modest trait to forecast. The market check (most-penalized quartile covers 49.85% vs least-penalized 50.52%) is a 0.67-point gap on quartile-sized samples and cannot separate "priced" from "too small to see" -- category 3 in `docs/pool_edge_plan.md`, not a negative. Treat as a weak-signal stacker input at most, never a standalone candidate and never worth a window. Separately, the construct this row actually names is not measurable from local data -- neither play-by-play snapshot carries `penalty_type` or a description column (45 columns; only `penalty` and `penalty_yards`), so "pre-snap discipline" would need a wider nflverse column pull. Since total penalty discipline is already null against the market, that pull is not worth making |
| PBP-08 | 🔬 | Scheme/matchup interactions | Protection-pressure, explosive pass, personnel, coverage proxies |
| PBP-09 | ✅ | Pace and possession forecast | Closed on measurement, not built (2026-08-17; `docs/play_level_audit.md`). Game play volume has almost nothing to forecast and no link to the outcome we care about: on 2009–2012 (1,024 games, no window spent) plays/game is 125.5 with sd 8.7, a coefficient of variation of **0.070**; forecasting it from both teams' prior pace reaches only **R² 0.041**, shaving ~2% off the residual sd. Worse for the premise, `corr(plays, |margin|) = −0.20` — blowouts have *fewer* plays, which is clock-killing and kneel-downs reading backwards from the result — and `corr(drives, |margin|) = +0.004`, i.e. zero. Independently reproduced by a second measurement before this row was changed. A pregame pace forecast has no path to ATS value; reopen only if some mechanism other than volume is proposed |
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
| PER-07 | 🚧 | Coaching/coordinator changes | **Investigated 2026-08-17; the row's original framing was the wrong shape and is superseded.** Coach identity is priced -- split-half reliability of a coach's mean ATS residual is +0.063 (Spearman-Brown +0.119, 85 coaches), and coaches with an above-50% prior cover rate subsequently cover 49.5-51.0% at every sample threshold. Tenure buys nothing beyond year one (years 2-3 50.74%, 4-6 49.63%, 7+ 51.96% -- no monotonicity) and there is no variance story either (year-1 early-season residual sd 13.53 vs 13.22 for veterans). 4th-down aggressiveness is genuinely reliable and is **NOT closed** -- an earlier draft of this row called it "fully priced" and that was wrong, an underpowered interval read as a null (the RWB-16 error this project exists to avoid). Re-measured independently: expressed relative to each season's league norm (which strips the well-documented league-wide drift toward going for it, and halves the naive figure) year-over-year reliability is **+0.320** on 512 team-season pairs, go-rate-over-expected sd 0.0381. The market test regresses prior-season aggressiveness against the ATS residual on 4,175 games and returns, for a one-sd matchup, **-0.038 points, season-blocked 95% [-0.423, +0.417]** -- an interval 0.84 points wide. The hypothesis it was meant to reject, that the value is *completely* unpriced, is worth about **+0.174 points** and sits comfortably INSIDE that interval. The test therefore cannot distinguish priced from unpriced. Resolving 0.174 points at 95% would need roughly **24,000 games -- about 90 NFL seasons** (CFB's 12,500 does not close that either). So this can never be confirmed as a standalone candidate and must not be spent on a window; it belongs in the weak-signal stacker or nowhere. Two honest discounts: the sd 0.123 EPA-points-per-game magnitude is a naive bin-EPA figure that selection almost certainly inflates, since coaches go for it when they expect to convert; and MOD-07 has already returned unresolved once. Interim changes are too thin (29 events; the 65.5% first-game figure is n=29 noise) and are **not provably pregame** -- historical snapshots are backfilled with whoever worked the game, so an interim feature needs an announcement-date source before it could pass a leakage test. **What survives is one binary flag:** a year-1 head coach fades in weeks 1-8, covering 46.72% on 762 games / 104 team-season clusters (cluster 95% [43.46, 50.06]), -1.30 points raw and -1.45 with a cubic quality control, both blocked intervals excluding zero. It is not a quality confound (equal-quality teams that KEPT their coach cover 50.70%) and not a QB proxy (new-QB coefficient is null in the 2x2). The active model does not capture it: it sides with the year-1 team 51.6% of the time and those teams covered 47.6%. Independently replicated on free CFB data at 3.5x the sample (year-1 all weeks 48.73% on 483 clusters; weeks 1-4 47.14%), with the same sign, magnitude and era boundary. **The catch that governs the decision:** the effect is null in 2009-2017 (49.00%, P=0.321) and lives in 2018-2025 (44.79%, P=0.011), i.e. inside the mined era. Worth ~+0.57 pp of full-slate pool accuracy. Do NOT ship on this measurement -- declare `hc_year_one_fade` with `acknowledges_mined_2018_2025` and buy the answer with one opener window |
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
| MOD-04 | ✅ | Margin distribution | Quantiles or parametric distribution; validate coverage and tails. Coverage re-verified 2026-08-17 on the frozen 2018-2025 artifact: the 50% interval covers 50.21% and the 80% covers 78.94%, both near nominal. **One correctness defect found and fixed in the same pass:** `_three_way_probabilities` tested a CONTINUOUS predictive sample for exact equality with the line, which never fires in floating point, so every published card carried `push_probability = 0.0000` while ~4.8% of integer-line games really push (**9.0% at a line of 3**) and `home_loss_probability` silently absorbed it. The sample is now rounded to integer margins. `home_cover_probability` is deliberately untouched, so no pick moved. The bug survived because the existing test used the `market` target, whose residuals are exact half-integers and made the equality fire by luck; a continuous-residual regression test now pins the realistic path |
| MOD-05 | ⬜ | Joint score/total model | Coherent home/away score and total probabilities. **Premise confirmed, ATS payoff undercut (2026-08-17).** The key-number lattice is real, large and stable: P(|margin| = 3) is 14.58% against 5.29% under a fitted normal (2.75x), |margin| = 7 is 8.76% vs 4.83%, ties are 0.29% vs 2.70% (overtime nearly eliminates them), and unimodality is rejected outright (dip 0.0209, p = 0.000 against both uniform and discretised-normal nulls) with the two dominant modes at exactly +3 and -3. A joint score model would reproduce all of it for free. But for *cover* probability it buys ~0.0004 Brier at the settlement line and **zero** over a plain Gaussian at the actual line, because the ATS residual is one line-varying convolution away from smooth (dip p = 1.000; roughness chi2/df 21.2 for raw margin vs 1.6-1.9 for the residual). Build it for push probability, alternative-line/half-point questions and correct-score products -- **not** for ATS accuracy |
| MOD-06 | 🚧 | Bayesian dynamic team model | Partial pooling, uncertainty, explicit offseason evolution. **The coefficient-level arm is closed on measurement (2026-08-17), before being built into the pipeline.** Prototyped on 12,206 free CFB games (rule 8, no window spent): sweeping ridge shrinkage across five orders of magnitude moves forced-pick accuracy under a point, and *increasing* it — the direction the partial-pooling argument predicts — makes the thin-training buckets resolvably **worse** (500–999 rows: .5292 at α=300 → .4646 under evidence-maximised shrinkage). Empirical-Bayes/BayesianRidge/ARD as an accuracy play are dead, as is sample-size-scaled α. ~~**The structural reason, which generalises:** the pool metric reads only `sign(predicted residual)`, and rescaling by any positive scalar cannot change a sign — so *any* scheme whose whole effect is to rescale the prediction is a no-op for the primary goal, however well-motivated. It can move calibration and confidence ordering, never a pick.~~ **RETRACTED 2026-08-17: that reasoning is wrong and was being used to reject work.** The production pick is `home_cover_probability >= 0.5` (`pool.py:41`, `backtest.py:56`) — the median of the out-of-time residual sample shifted by the prediction, not the sign of the prediction. The two rules disagree on **11.8% of the 2,075 scored games** (244), and because that sample's median is non-zero, rescaling the centre *can* flip picks. **And the production rule is resolvably the better of the two: 52.05% vs 49.93%, +2.12 points, season-blocked 95% [+0.24, +4.17], `probability_positive` 0.990** — on the 244 disagreements it wins 59.0%. That entire margin is the residual sample's location offset, currently the unweighted empirical median of a ~500-900-draw trailing holdout that no one has ever modelled. See "Where to look next" in `docs/pool_edge_plan.md`. Independently, ridge penalty changes were never in the "rescale" class at all: generalized ridge gives `b_j = d_j·b_j^OLS/(d_j + λ_j)`, so differing `λ_j` rotate the coefficient vector rather than scaling it (measured: block penalties flip up to 18.6% of CFB picks, a global α change 10→1e4 flips 20.1%, a positive rescale flips exactly 0; `docs/groupwise_ridge.md`). **MOD-06's conclusion still stands on its own measurement above — do not reopen it — but never again reject penalty-structure, calibration, or shrinkage work by citing this corollary.** Type-II ML also optimises squared error, so it correctly shrinks away the small directional signal the forced pick lives on: the Bayesian objective and the pool objective disagree. Two things survive: (a) the warm-up cliff was removed by **evidence rather than a model** — see `docs/rotation_registry.md` rule 9, floor now 2011; (b) the one live arm is **unit-level** shrinkage toward a *position prior* rather than toward zero (`players.py` currently multiplies a thin player's value by `career/(career+200)`, i.e. treats a barely-seen player as worth nothing). That changes *relative* feature values and can therefore flip a pick. Needs no new dependency — closed-form James-Stein in numpy; a full sampler is disqualified anyway because it would inject nondeterminism into a pipeline whose methodology rests on exact reproducibility. Screen on CFB at `probability_positive >= 0.75` before any NFL window |
| MOD-07 | 🚧 | Ensemble/stacking | Out-of-fold predictions only; weight stability constraints. The first predeclared vehicle — the weak-signal stack (player value composite + learned availability + three documented opener-bias families, profile `weak_stack`) — took its one registry look on [2020, 2021] at the Tuesday-opener grade (August 2026, `docs/mod07_stack.md`): **53.29% vs the frozen model's 51.32% on 456 paired games, +1.97 points, week-blocked [−1.10, +5.00], `probability_positive` 0.8745** — short of the predeclared 0.90 threshold, so `unresolved`, not promoted, and the window is spent. The entire delta comes from the 49 picks the arms split (candidate 29–20). Prospective 2026 scoring as a frozen challenger is the recommended next evidence; retuning the stack and re-scoring [2020, 2021] is inadmissible |
| MOD-08 | 🔬 | Distributional boosting | Quantile/NGBoost-style margin and total forecasts. **Undercut on measurement (2026-08-17); any predeclaration must carry this negative.** MOD-08 targets a richer *conditional* density, but the conditional shape is already near-Gaussian once you condition on the line (residual dip test p = 1.000, roughness chi2/df 1.6-1.9 against 21.2 for the raw margin), and ATS-residual sd is flat at 12.6-13.5 across every spread bucket. A fixed unconditional key-number lattice adds nothing over a plain Gaussian on Brier or log loss (head-to-head probability_positive 0.478). There is no shape signal left to condition on. **What IS worth doing is smoothing, not conditioning:** the production mapping inverts an ECDF built from only ~518 residual draws, which quantises every probability and injects ~2.2pp of Monte-Carlo noise per game; replacing it with any smooth CDF is worth **Brier -0.0015 and log loss -0.0032, week-blocked 95% interval excluding zero (P = 0.998)** on 1,802 non-reserved games -- roughly 9x the effect size the MOD-16 CFB screen could resolve. It also removes an accidental -0.77-point median tilt in the ECDF that currently flips 8.8% of forced picks, so it MOVES PICKS and needs its own predeclared window |
| MOD-09 | 🔬 | Sequence model over drives | Small temporal model, benchmarked against summary features. **Evidence against, 2026-08-17** (`docs/play_level_audit.md`): the premise that play-level rows multiply the training set does not hold — plays are near-independent (ICC 0.013), so the per-game mean is already a sufficient statistic for what they say about a team, and the model still fits on ~4,700 game-level labels however many plays are exposed. The drive layer's summary form already worsened Brier, and a sequence model must beat those summaries while spending far more parameters on the same labels. Do not build ahead of the cheaper noise-reduction work in PBP-05 |
| MOD-10 | 🔬 | Graph model | Player/team matchup graph only after player state is reliable |
| MOD-11 | 🚧 | Calibration suite | Platt, isotonic, and beta are leak-safe and evaluated; calibration-by-regime remains |
| MOD-12 | 🚧 | Hyperparameter protocol | Frozen profile/Ridge/calibration budget selected on prior seasons and scored on next-season folds. **Reopened 2026-08-17: `ridge_alpha = 10.0` is very nearly inert and was never derived** (`docs/groupwise_ridge.md`). Measured on the real 4,630-game active-model design: the median principal direction is shrunk by **0.27%** (mean 6.4%, weakest decile 16.0%). The frozen model is unregularised least squares in all but name on every direction that carries signal. The nuance matters — the design is **rank-71-of-142** (`diff = home − away` holds identically, so half of it is degenerate), and there α is what makes the fit well-posed at all, so "inert" is right for the bulk and wrong as a blanket claim. Consequence: at α=10 the whole group-penalty axis is a no-op (every accuracy delta < 0.09 pts), so **deriving this constant precedes any further penalty-structure work**, and it is free on CFB. It also deflates the `player_qb_continuity` re-read: α=1 vs α=10 differ by 0.03% vs 0.27% at the median direction, so the ±1.1033-point swing recorded there is a calibration of evaluator noise (coin-flip sd on 997 games is ~1.58 points), not a signal |
| MOD-13 | ⬜ | Missingness audit | Drop source-era indicators and test explicit availability flags |
| MOD-14 | ⬜ | Era weighting | Compare rolling training windows and time-decayed sample weights |
| MOD-15 | ✅ | Temporal schedule-graph ratings | Leak-safe PageRank/HITS and ridge/SRS comparator completed; graph selected in 0/8 outer seasons and was not promoted |
| MOD-16 | ✅ | Conditional margin variance | Replace the pooled residual distribution with game-level heteroskedasticity; accepted only if held-out cover/push/loss calibration beats the pooled baseline. The predeclared CFB screen (August 2026, `docs/margin_variance.md`) **failed**: a Ridge scale model on mismatch/total/pace/experience made clean-core cover log-loss resolvably worse (week-blocked [−0.00056, −0.00012]) — the pooled distribution is already near-correctly calibrated. Only a genuinely NFL-specific variant (QB/backup status, weather) with a new ledger-aware predeclaration remains admissible, and it is deprioritized by this result |

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
| POL-04 | ❌ | Pick-popularity input | **Closed for lack of data, not effort** (`docs/pool_format_levers.md`). Splash does show a pick distribution, but it unlocks game-by-game as each game kicks off — a deliberate integrity measure, so it is structurally unavailable before the Tuesday lock, and there is no API. The Odds API sells no betting percentages; no free ticket/handle feed offers an API plus history. The only obtainable proxy is the favourite flag, already on the card — and our picks run 54.8% favourites / 46.7% home, so we sit near-neutral against a favourite-loving field anyway |
| POL-05 | 🔬 | Contest utility optimizer | Simulator built and validated against five closed forms (`nfl_ats.pool`, `tests/test_pool.py`, `scripts/pool_levers.py`); independence measured rather than assumed (weekly variance of the correct share is 1.036× binomial over 107 weeks). Results: the format already multiplies a fair share of first place by 2.3×–12.7× at 52.5%; **deliberate contrarianism loses monotonically at every field size** when it costs the full edge (break-even ~2.5 accuracy points per flip), and under a top-15% prize — Splash's usual structure — the lever is neutral at best. A Best Pick ranker is worth +2.4 pp of P(first) at 100 rivals if the recorded +8.7 were real, and **+0.07 pp at the honest +0.9**. **Two accuracy points are worth +11.8 pp.** The format is a multiplier to protect, not a lever to pull. Open only on two observables — the pool's real field size and prize structure — both recordable in Week 1 |
| POL-06 | ⬜ | Multi-entry diversification | Correlated entries with controlled overlap |
| POL-07 | ⬜ | Survivor planner | Current survival probability plus future team opportunity cost |
| POL-08 | 🔬 | Opponent-field simulation | Simulate standings and strategic picks for winner-take-all pools |
| POL-09 | 🚧 | Best Pick ranker | The pool pays one Best Pick per week and our confidence ordering is flat (top-\|residual\| scored 48.6% over 107 weeks). Three signals were predeclared and screened once on the registry window [2013, 2015] (August 2026, `docs/best_pick_ranker.md`): `calibrated_probability` (−8.16 points vs all-pick, `probability_positive` 0.0925) and `key_number_distance` (−6.19, 0.170) both made the weekly top-1 pick **worse** and are closed; `sweep_robustness` scored +5.57 points at 0.7955, clearing the predeclared 0.75 screen gate. Its earned opener confirmation then ran once on [2020, 2021] with the deployed `player` profile: **top-1 60.0% (21/35) vs 51.32% all-pick, +8.68 points, week-blocked [−7.00, +22.88], `probability_positive` 0.865** — clears the predeclared 0.75 gate, verdict `confirmed`, both windows now spent. Consequence per the predeclaration: **use `sweep_robustness` to choose the Best Pick in 2026** — a pool-play decision, no activation and no model change. Two disjoint windows, two grades, same direction; but 86 top-1 picks total and a rank correlation of +0.067 (p=0.099), so expect regression toward the all-pick rate. **Re-read 2026-08-17 (`docs/pool_format_levers.md`): every recorded figure reproduces exactly, but the confirmation is mostly the TIE-BREAK, not the signal.** `sweep_robustness` is a half-point width censored at 8.0, so weeks routinely tie at the top and `select_best_pick` breaks ties alphabetically by `game_id`. **24 of the 35 confirmation weeks were ties** (39 of 51 on the screen). Scoring the same signal under a random tie-break gives **52.24%, delta +0.92 points — not 60.0% / +8.68**; the recorded value sits at the 96th percentile of the tie-break distribution. On the screen window the tie-break pushed the other way (+9.05 rather than the recorded +5.57), so *direction* survives but the "two windows, same direction, both clearing" argument is partly alphabetical coincidence. Also unmeasurable by construction: top-1 standard error is **8.45 points on 35 weeks**, so anything in [43%, 62%] is indistinguishable from an arbitrary nomination, and resolving a 5-point effect needs ~384 weeks ≈ 21 seasons. **This re-scores nothing and changes no registry verdict** — keep using the signal, since the alternative has no evidence at all, but budget it at +0.9 points. The published card and dashboard now disclose a tie whenever one occurs. 2026 Week 1 on the LIVE `player` card is NOT tied (ARI@LAC 7.0, next 6.5), so no disclosure fires; the two-way tie at 8.0 is in the `weak_stack` challenger card, which is not what is played |
| POL-10 | 🚧 | Prospective 2026 evidence | Grade the active model and every frozen challenger on games nobody has looked at, at BOTH grades (the recorded/opener line primary, close secondary). Built 2026-08-17, three weeks before Week 1 locks (`docs/prospective_evidence.md`), because prospective scoring is the only way to settle `mod07_weak_signal_stack` (unresolved, P+ 0.8745) and firm up `best_pick_ranker` (60.0% on 35 top-1 picks) **without spending either of the two remaining opener windows** -- and it only produces evidence if picks are recorded before kickoff every week from Week 1, so a season that goes unrecorded is simply gone. Three gaps closed: nothing computed whether a 2026 pick WON (the MKT-04 ledger scored line movement only, never joining `result`); the weekly Best Pick was recomputed at render time and overwritten every publish, so Week 1's nomination would have ceased to exist once Week 2 published; and the MOD-07 stack was not registered, so no challenger card was being produced at all. Now: `nfl-ats prospective-score` settles at both grades reusing `clv.pick_correct` so FND-04 push semantics are literally the same function; `is_best_pick` persists in `PAPER_DECISION_COLUMNS`, written only while EVERY game of the week is still ahead, first-write-wins, exactly one per week enforced on read; challengers resolve by configuration fingerprint rather than directory recency (the baseline and challenger share an artifacts directory, so newest-wins would have silently recorded the baseline's picks as the challenger's); and `weekly-run` gained four optional trailing steps whose failure never aborts the fail-closed publish. **Anti-backdating is enforced twice** -- refused at write, and re-asserted at scoring, so a hand-written row cannot be laundered into evidence by running the scorer. A Week 1 challenger card was generated as a rehearsal and the two arms disagreed on 3 of 16 games. **The rehearsal rows were then reset (2026-08-17)** so that Week 1's first ledger write is the real Tuesday-lock card on 2026-09-08 — the MKT-04 ledger anchors each game at its FIRST publication, which would otherwise have scored August's picks at August's lines instead of what was actually entered |

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
   and close), with the opener grade primary per the pool goal. **The
   machinery for this now exists and is the single most time-critical item
   in the file** (POL-10, `docs/prospective_evidence.md`): win/loss settles
   at both grades, the weekly Best Pick persists pre-kickoff, MOD-07 is
   registered as a challenger, and anti-backdating is enforced at write and
   again at scoring. Week 1 locks Tuesday 2026-09-08 and an unrecorded
   season is gone. One decision is open before then: the Week 1 ledger rows
   anchor on the 2026-08-17 rehearsal publish rather than the Tuesday lock
   the pool actually grades. The 2013–2017 and 2014–2017 replication
   windows are spent, and no new variant of an existing family may be
   scored on 2018–2025 without a frozen predeclaration that acknowledges
   the ~130–150-look ledger. **The peer-reviewed opener biases are no
   longer a lead**: three were built and, ablated inside MOD-07 on the
   already-spent window, contributed +0.22 points at `probability_positive`
   0.505, while the published Week-1 holdover figure (35.6%) fails to
   replicate here (52.5% on 120 games). Do not add more of them.
6. Stop trying to measure team quality better; it is bounded near zero.
   A deliberate-leak positive control (opponent adjustment fit over all of
   2006–2025, so the columns see the future) moved margin MAE by only
   **+0.0129 points** — a measured ceiling on the whole family, and the
   common explanation for the PBP/drive bundle, PBP-05, MOD-16 and CFB role
   continuity all failing separately. Our target is the residual from the
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
7. Use 2016–2025 participation/NGS for position-unit and formation effects;
   individual receiver-corner pairs remain too sparse for an initial model.
8. Attempt drive simulation only after simpler distributional baselines exist.

The dashboard and experiment registry should make failed hypotheses easy to
retain. Negative results are project assets; quietly deleting them invites the
same experiment to be rediscovered and overfit later.
