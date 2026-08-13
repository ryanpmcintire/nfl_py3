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

## Phase 2 — point-in-time market data

| ID | Status | Item | Definition of done |
|---|---|---|---|
| MKT-01 | ✅ | Live odds provider adapter | Book, market, line, price, observed-at timestamp, raw response hash |
| MKT-02 | 🚧 | Opening/current/closing line store | Append-only live observations plus a preserved 2025 opener/reported-close sample; never overwrite a historical quote |
| MKT-03 | 🚧 | No-vig market probabilities | Documented two-way normalization and favourite-longshot diagnostics |
| MKT-04 | 🚧 | Closing-line-value tracking | Each paper decision compared with same-book close |
| MKT-05 | ✅ | Cross-book consensus | Median line, dispersion, stale-book and outlier detection |
| MKT-06 | 🔬 | Line-movement forecasting | Predict close from an earlier decision timestamp |
| MKT-07 | ✅ | Market residual model | Estimate only the correction to a market prior |
| MKT-08 | 🔬 | Timing policy | Compare fixed weekly timestamps and news-triggered updates |
| MKT-09 | 🚧 | Provider licensing/quota audit | Terms, redistribution limits, cost, retention, failure policy |
| MKT-10 | ✅ | Free historical close audit | Versioned public close archive plus 2025 opener/nine-book close sample, licenses, normalization, source comparison |

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
| PER-01 | 🚧 | Weekly roster/participation snapshots | Source audit passed: weekly rosters 2002–2025 and participation 2016–2025; ingestion and as-of contracts remain |
| PER-02 | 🚧 | Quarterback state | Starter probability, player EPA/CPOE, backup adjustment |
| PER-03 | 🚧 | Injury report history | 84,684 timestamped rows across 2009–2024 verified; ingest, cutoff audit, and replacement live source remain |
| PER-04 | 🚧 | Depth chart history | Starter/backup roles without using later revisions |
| PER-05 | ⬜ | Snap-weighted player value | 324,611 player-game rows across 13 nonempty seasons support a lagged historical test |
| PER-06 | ⬜ | Roster continuity | Weekly rosters provide 24 seasons; build returning-snap priors by position group |
| PER-07 | ⬜ | Coaching/coordinator changes | Coach IDs, tenure, scheme tendencies, change flags |
| PER-08 | ⬜ | Transaction-aware preseason prior | QB, roster, coaching, draft/free-agency adjustments |
| PER-09 | 🔬 | Latent player ratings | Hierarchical offense/defense/special-teams contribution model |
| PER-10 | 🔬 | Injury scenario mixture | Forecast weighted across active/inactive player scenarios |

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
| MOD-07 | ⬜ | Ensemble/stacking | Out-of-fold predictions only; weight stability constraints |
| MOD-08 | 🔬 | Distributional boosting | Quantile/NGBoost-style margin and total forecasts |
| MOD-09 | 🔬 | Sequence model over drives | Small temporal model, benchmarked against summary features |
| MOD-10 | 🔬 | Graph model | Player/team matchup graph only after player state is reliable |
| MOD-11 | ⬜ | Calibration suite | Platt, isotonic, beta calibration, calibration-by-regime |
| MOD-12 | 🚧 | Hyperparameter protocol | Nested walk-forward tuning with frozen search budgets |
| MOD-13 | ⬜ | Missingness audit | Drop source-era indicators and test explicit availability flags |
| MOD-14 | ⬜ | Era weighting | Compare rolling training windows and time-decayed sample weights |
| MOD-15 | ✅ | Temporal schedule-graph ratings | Leak-safe PageRank/HITS and ridge/SRS comparator completed; graph selected in 0/8 outer seasons and was not promoted |

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
this is also retained. A free 2025 opener/nine-book closing sample now validates
the line-movement contract, while full multi-season quote timing remains open.
QB promotion is blocked by historical point-in-time coverage rather than by
model code. MOD-12 remains in progress until estimator hyperparameters also use
the frozen nested budget.

The temporal PageRank/HITS comparison is also complete. Graph candidates were
selected in zero of eight outer seasons; the simpler ridge/SRS schedule rating
was selected three times and the existing market/context model five times.
Season-blocked paired intervals found no reliable improvement, so both remain
research features rather than defaults.

Prediction integrity is permanently release-blocking: no modeling or feature
work is allowed to bypass FND-11/FND-12, even when the research code itself
appears to run successfully.

Evaluator performance is also guarded. The outcome bootstrap was rewritten
from per-draw pandas reconstruction to block-level sufficient statistics: the
standard 2,000-draw report fell from roughly 290 seconds to 0.38 seconds while
matching the old saved intervals to floating-point precision. See
`docs/performance.md` for budgets and regression rules.

1. Maintain the prediction-safety contract and add a regression canary for
   every production error or newly supported output type.
2. Build the historically feasible player layer first: timestamped 2009–2024
   injuries, lagged 2013–2025 snap shares, and 2002–2025 roster continuity.
3. Add joint score/total distributions and compare calibration methods inside
   the nested protocol.
4. Use 2016–2025 participation/NGS for position-unit and formation effects;
   individual receiver-corner pairs remain too sparse for an initial model.
5. Continue collecting timestamped, book-specific opening/current/closing
   quotes. The one-season free sample validates plumbing but cannot validate a
   historical line-movement edge.
6. Attempt drive simulation only after simpler distributional baselines exist.

The dashboard and experiment registry should make failed hypotheses easy to
retain. Negative results are project assets; quietly deleting them invites the
same experiment to be rediscovered and overfit later.
