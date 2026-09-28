# Lanes

A lane file is the state a fresh session needs to continue a task after the
previous session was cleared. One file per task, named by a short slug. The
session that works a lane updates it whenever a unit of work completes, and
every final response ends with a one-line clear verdict (AGENTS.md, Lanes and
clearing). The status line shows the most recently touched lane.

A lane file has five short sections and stays under one page:

- **Goal**: one or two sentences, including what done looks like.
- **State**: what is shipped, committed or pushed; exact file paths.
- **Tried**: what was attempted and the measured result, so it is not redone.
- **Next**: the exact next step, with the command or file to open.
- **Open**: questions only the owner can answer, and decisions deferred.

Lanes that are finished move to `done/` with a final State section. A fresh
session reads this index, then the lane the prompt names, or the most recently
modified lane when the prompt just says to continue.

## Active

- [backlog research checkpoint](done/backlog-research-2026-09-28.md) - 2026-09-28: corrected frozen scorecard and pool chronology; completed four-cell precision replay and simulator terminal-rule repair/regrade.
- [independent-combination-validation](independent-combination-validation.md) - 2026-09-28: historical selection replay completed on 533 games; calibration-stage diagnostic and 30 registry records saved; untouched Week 4–18 collection remains active.
- [backlog-batch-2026-09-26](backlog-batch-2026-09-26.md) - 2026-09-26: 16 finished research lanes closed to done/; see State for what each closed on.
- [sim08-simulator-rebuild](sim08-simulator-rebuild.md) - 2026-09-26: SIM-04 was graded while failing its own key-number gate; rebuild endgame, undamp conditioning, anchor on the opener, re-grade.
- [lead53-sunday-renomination](lead53-sunday-renomination.md) - 2026-09-24: ranks on the served four-term probability; after Sun 10:00 ET grep BEST-PICK-LEDGER in data/scheduler_log.txt for the Week 3 pairing.
- [lead64-friday-designations](lead64-friday-designations.md) - 2026-09-24: headline designations lead the official file but only n=3 checkable; not wired; rerun the join after more weeks now that inactives capture works.
- [pol10-prospective-2026](pol10-prospective-2026.md) - 2026-09-28: frozen card 28-19, one pending; Best Pick 1-2; 23 decisive games have valid matched baselines. Rerun scripts/prospective_scorecard_2026.py after grading.
- [positive-control-power](positive-control-power.md) - 2026-09-23: minimum detectable effect harness; decides which unresolved cells are bounded by a control.
- [free-odds-sources](free-odds-sources.md) - 2026-09-24: Books now shows posted lines (one number or a low-to-high range); no-publication rule removed; mid-week captures back on; direct Bovada capture fixed (v2 endpoint).
- [news-trigger-refresh](news-trigger-refresh.md) - MKT-08 dispatch implemented and verified; prospective comparison awaits new events.
- [gh-window-incident](gh-window-incident.md) - prior runaway CLI windows remain a separate investigation; no gh commands used for this deployment.

- [confidence-best-pick-unification](confidence-best-pick-unification.md) - shared calibrated selector published in `5fb88a2`; confidence reliability and within-week ranking research remains open


- [pool-rank-card](pool-rank-card.md) — POOL-01, unit 1 measured 2026-09-16 (unresolved); unit 2 only with a fitted field
- [token-diet](token-diet.md) — session-startup token cost cut about 80%; remaining: trim the three 15 KB+ open ROADMAP rows (owner text) and decide whether `.claude/` hooks should be tracked
- [sunday-market-probability](sunday-market-probability.md) — activated 2026-09-20 as `leader_median_through_sunday_prekick_v1`; six unresolved metric/protocol cells recorded under `sunday_market_probability_fixed_v1`.

## Done

- [current-week-pressure-inputs](done/current-week-pressure-inputs.md) - 2026-09-28: Week 3 updated; current-season PBP ingestion and publication coverage repaired; locked decisions and enrolled model preserved.
- [backlog-batch-2026-09-25](done/backlog-batch-2026-09-25.md) - 2026-09-25: standing backlog lane; batch 1 = queued registry records + one UI-20 improvement.
- [inactives-capture-empty](done/inactives-capture-empty.md) - 2026-09-24: RotoWire fallback now parses (150 rows on an archived page); closed 2026-09-26: live Thursday T-90 capture returned 11 real rows.
- [lead59-archive-battery](done/lead59-archive-battery.md) - 2026-09-26: referee type-trait bins recorded and committed (93ba72d); ROADMAP LEAD-59 reflects the finding.
- [line-move-regrade-legacy](done/line-move-regrade-legacy.md) - 2026-09-26: batches 2-4 all recorded; division-revenge and special-teams-return resolved wrong-sign, rest unresolved_below_power.
- [market-move-decomposition](done/market-move-decomposition.md) - 2026-09-26: all 6 market_move_decomposition_v1 cells recorded, all unresolved_below_power.
- [mod18-spread-regime](done/mod18-spread-regime.md) - 2026-09-26: served and extended-window spread-regime cells recorded unresolved_below_power; served card unchanged.
- [total-conditioned-lattice](done/total-conditioned-lattice.md) - 2026-09-26: total-banded key-number lattice log-loss and accuracy cells recorded unresolved_below_power.
- [opener-population-backfill](done/opener-population-backfill.md) - 2026-09-26: 2011-2025 discrete-read population built as a challenger/sanity artifact; no record commands were ever drafted (looks only).
- [pooled-signal-model](done/pooled-signal-model.md) - 2026-09-26: MOD-20 units 1-6 all recorded; served four-term base beats every structural variant tried.
- [conditional-signal-atlas](done/conditional-signal-atlas.md) - 2026-09-26: MOD-19 atlas committed, ROADMAP updated, published to the live site in a prior session.
- [scheduler-once-timeout](done/scheduler-once-timeout.md) - 2026-09-26: fixed and committed `786a569`; scheduler suites green.
- [lane-replay-dream-rsi](done/lane-replay-dream-rsi.md) - 2026-09-26: ENG-45 units 1-2 measured (unresolved); unit 3 not recommended.
- [prospective-leads-2026-09-23](done/prospective-leads-2026-09-23.md) - 2026-09-26: total-conditioned-lattice challenger fully wired and verified; nothing blocks it.
- [opener-error-transfer](done/opener-error-transfer.md) - 2026-09-26: XLG-09 unit 5 recorded; pooling mechanism refuted on line movement, CFB-only term stays unresolved.
- [surface-switch-fitted-term](done/surface-switch-fitted-term.md) - 2026-09-26: six ENV-02 cells recorded unresolved_below_power; not served.
- [lead65-protection-window-split](done/lead65-protection-window-split.md) - 2026-09-26: served flag sum kept unchanged; week-gated variant stays a tracked prospective challenger.
- [clv-metric-everywhere](done/clv-metric-everywhere.md) - 2026-09-26: ENG-47 line-move-toward-pick yardstick live everywhere; paired eval (5 cells) unresolved_below_power.
- [every-metric-every-experiment](done/every-metric-every-experiment.md) - 2026-09-26: ENG-46 implementation and backfill complete; backfilled-look accounting policy question carried to backlog-batch-2026-09-26.
- [week3-research-recorders](done/week3-research-recorders.md) - 2026-09-25: all Week 3 challengers recorded (59 recorded, 4 skipped, 0 missing); the tiebreaker shade skips by design.
- [card-freshness-and-log-labels](done/card-freshness-and-log-labels.md) - 2026-09-24: refresh passes rewrite the card's freshness line; no follow-rule log label.
- [four-term-nonlinear-check](done/four-term-nonlinear-check.md) - 2026-09-24: trees on the four served terms resolved worse (-1.26 pts, wrong sign); interactions unresolved.
- [four-term-extended-training](done/four-term-extended-training.md) - 2026-09-24: 2011-2025 training does not beat served 2020-2025 training (-0.27 pts, P+ 0.25); recorded.
- [publish-predictions-preservation](done/publish-predictions-preservation.md) - 2026-09-24: same-week republish keeps the late-week refresh block; inactive challengers skip by name.
- [ui20-2026-09-24](done/ui20-2026-09-24.md) - 2026-09-24: Cover chance tooltip states distance from a coin flip.
- [lead61-second-half-channel](done/lead61-second-half-channel.md) - 2026-09-24: no free second-half source; challenger skips by name.
- [mod17-unified-served-numbers](done/mod17-unified-served-numbers.md) - 2026-09-24: tiebreaker lattice centres on the served side.
- [coordinator-adjudication-2026](done/coordinator-adjudication-2026.md) - 2026-09-24: in-season scan covers 2026; weekly coordinators_tue re-run adjudicates any 2026 edit.
- [statistical-audit-2026-09-15](done/statistical-audit-2026-09-15.md) - closed 2026-09-24: flip-chain and lookup findings superseded by the four-term probability and served-probability display; decisive-record rule lives in AGENTS.md.
- [injury-scenario-producer](done/injury-scenario-producer.md) - PER-10 closed 2026-09-24: margin mapping fit and scenario mixture graded; both cells recorded unresolved_below_power; kernel stays out of src/.
- [odds-api-key-deactivated](done/odds-api-key-deactivated.md) - closed: owner cancelled The Odds API on purpose (2026-09-19); paid jobs disabled; current odds come from free sources (free-odds-sources lane). Never raise it as an owner action.
- [base-model-recency](done/base-model-recency.md) - 2026-09-23: recent-season weighting flat in the pick model; standalone margins resolved worse.
- [extended-population-regrade](done/extended-population-regrade.md) - 2026-09-23: three Tuesday terms on 2011-2025, all unresolved.
- [fit-ridge-derivation](done/fit-ridge-derivation.md) - 2026-09-23: fit constants traced; nested ridge picks the same games; Strong beats Slight on 2011-2025.
- [tuesday-terms-line-move](done/tuesday-terms-line-move.md) - 2026-09-23: five Tuesday terms on line movement; fan-forum variant a resolved wrong sign.
- [players-on-field-rating](done/players-on-field-rating.md) - MOD-22 units 1-3, 2026-09-23: four fitted terms unresolved, Tuesday availability change not reconstructible; family open, cached construct exhausted.
- [ui20-confidence-column-guide](done/ui20-confidence-column-guide.md) - 2026-09-23: Confidence column hover help and Column guide entry published.
- Superseded 2026-09-23 when the four-term fitted probability became the served side selector (2026-09-20); their open questions were about flip rules that no longer serve: [four-term-probability](done/four-term-probability.md), [joint-probability-model](done/joint-probability-model.md), [market-updated-model](done/market-updated-model.md), [served-flip-rules-hold](done/served-flip-rules-hold.md), [leader-median-model-confidence](done/leader-median-model-confidence.md), [handle-follow-override-defect](done/handle-follow-override-defect.md), [best-pick-served-score-ranker](done/best-pick-served-score-ranker.md).
- [ui20-books-now-count](done/ui20-books-now-count.md) - 2026-09-23: Books now shows the book count, never a sportsbook name.
- [displayed-confidence-served-probability](done/displayed-confidence-served-probability.md) - 2026-09-23: in-sample lookup removed; the card shows the served fitted probability.
- [spread-explorer-discrete](done/spread-explorer-discrete.md) - 2026-09-23: spread explorer serves the discrete margin read.
- [small-maintenance-2026-09-23.md](done/small-maintenance-2026-09-23.md) — roadmap scheduler guidance now matches the conditional workflow; lane-index and root README links verified.
- [week3-lines-2026-09-22](done/week3-lines-2026-09-22.md) - genuine pool lines, all 16 Week 3 picks, paper decisions, and generated dashboard completed.
- [week-card-consistency](done/week-card-consistency.md) - Weeks 1-3 share seven columns and interactive game details, verified on desktop and mobile.
- [agent-harness-hygiene](done/agent-harness-hygiene.md) - shared policy and conditional workflow consolidated; repeated local prompt injection removed.
- [dashboard-week-filter-repair](done/dashboard-week-filter-repair.md) - 2026-09-22: filter only the weekly card; preserve the dashboard and interactive game details, verified on desktop and mobile.
- [injury-current-season-capture](done/injury-current-season-capture.md) - 2026-09-22: failed current-season requests preserve existing data and leave no empty capture directory; broader Friday/Sunday acquisition remains open.
- [Backlog triage and deployment](done/backlog-triage-2026-09-22.md) - completed batch deployed; scoped blockers and next-work pointers saved.

- [weekly-card-readiness](done/weekly-card-readiness.md) - 2026-09-22: current-week selector, original archived picks, readiness, and parallel dashboard fixes verified; Week 3 input acquisition remains active.
- [dashboard-lineup-toggle-state](done/dashboard-lineup-toggle-state.md) - 2026-09-22: initial lineup toggle state agrees with visible content.
- [dashboard-formation-selection-state](done/dashboard-formation-selection-state.md) - 2026-09-22: formation controls expose the selected player.
- [dashboard-column-guide](done/dashboard-column-guide.md) - 2026-09-22: accessible plain-language guide to card columns.
- [dashboard-ticker-keyboard](done/dashboard-ticker-keyboard.md) - 2026-09-22: each ticker matchup appears once in keyboard navigation.
- [lead64-card-designations](done/lead64-card-designations.md) - 2026-09-22: current injury designations and names appear on the card; wider acquisition workflow remains open.

- [windows-linux-recovery](done/windows-linux-recovery.md) - 2026-09-20: restored Windows dependencies and scheduler, isolated Linux environment instructions, repaired Shopify skill YAML, verified all gates and regenerated dashboard locally

- [halves-ledger-lockday](done/halves-ledger-lockday.md) — 2026-09-12
- [dashboard-2026-09-12-evening](done/dashboard-2026-09-12-evening.md) — 2026-09-12, Books-now market-move line
