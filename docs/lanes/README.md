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

- [inactives-capture-empty](inactives-capture-empty.md) - 2026-09-24: RotoWire fallback now parses (150 rows on an archived page); check the first live T-90 manifest tonight (ATL at GB) and Sunday.
- [lead53-sunday-renomination](lead53-sunday-renomination.md) - 2026-09-24: ranks on the served four-term probability; after Sun 10:00 ET grep BEST-PICK-LEDGER in data/scheduler_log.txt for the Week 3 pairing.
- [lead64-friday-designations](lead64-friday-designations.md) - 2026-09-24: headline designations lead the official file but only n=3 checkable; not wired; rerun the join after more weeks now that inactives capture works.
- [pol10-prospective-2026](pol10-prospective-2026.md) - 2026-09-24: prospective scorecard (card 24-8, Best Pick 1-1, n=32); rerun scripts/prospective_scorecard_2026.py after each graded week.
- [mod18-spread-regime](mod18-spread-regime.md) - 2026-09-24: MOD-18 Unit 1 measured; served large-spread deficit shrinks and does not replicate on 2011-2025; fitted spread-size candidate unresolved_below_power both windows (P+ 0.34, 0.26); two record commands queued for the root.
- [market-move-decomposition](market-move-decomposition.md) - 2026-09-24: all arms recorded (Unit 3 all-books active-window arm unresolved, P+ 0.22); only an optional train/serve window reconciliation look remains.
- [positive-control-power](positive-control-power.md) - 2026-09-23: minimum detectable effect harness; decides which unresolved cells are bounded by a control.
- [free-odds-sources](free-odds-sources.md) - 2026-09-24: Books now shows posted lines (one number or a low-to-high range); no-publication rule removed; mid-week captures back on; direct Bovada capture fixed (v2 endpoint).
- [opener-error-transfer](opener-error-transfer.md) - XLG-09 units 1-3, 2026-09-23: point-in-time-correct CFB training on the extended 2011-2025 NFL population still reads unresolved_below_power (+0.12 pts all-graded, +0.27 pts on 2020-2025); four record commands (v2 + v3) queued for the root.
- [opener-population-backfill](opener-population-backfill.md) - 2026-09-23: 2011-2025 fit population built (3,734 games, +2,231); discrete rebuild of 2011-2019 in progress.
- [week3-research-recorders](week3-research-recorders.md) - nine challenger records remain missing; all 16 served Week 3 paper decisions are complete.
- [news-trigger-refresh](news-trigger-refresh.md) - MKT-08 dispatch implemented and verified; prospective comparison awaits new events.
- [gh-window-incident](gh-window-incident.md) - prior runaway CLI windows remain a separate investigation; no gh commands used for this deployment.

- [confidence-best-pick-unification](confidence-best-pick-unification.md) - shared calibrated selector published in `5fb88a2`; confidence reliability and within-week ranking research remains open


- [pooled-signal-model](pooled-signal-model.md) — MOD-20, units 1-6 recorded; unit 6 (reddit attention term) -0.07 pts [-0.34,+0.28] P+ 0.27 unresolved, served four-term base still wins after 6 fits
- [pool-rank-card](pool-rank-card.md) — POOL-01, unit 1 measured 2026-09-16 (unresolved); unit 2 only with a fitted field
- [every-metric-every-experiment](every-metric-every-experiment.md) — ENG-46 implementation done 2026-09-16; backfilled-look accounting remains open (runner emits four metrics, backfill recorded, margin cells corrected same day, pool + findings read the margin family).
- [lane-replay-dream-rsi](lane-replay-dream-rsi.md) — ENG-45, units 1-2 done 2026-09-16 (both unresolved); unit 3 not recommended
- [conditional-signal-atlas](conditional-signal-atlas.md) - MOD-19: next unit fills the approved mockup's signal rail with already-measured split families from the atlas registry; no unmeasured families.
- [lead65-protection-window-split](lead65-protection-window-split.md) — decided 2026-09-23: served flag sum unchanged; week-gated variant stays a prospective challenger.
- [lead59-archive-battery](lead59-archive-battery.md) — LEAD-59 src fix and battery re-run done, 23 cells recorded; type-trait binning measured 2026-09-23 (2 cells, both unresolved_below_power, archive unreachable pre-2015), record commands queued for the root
- [token-diet](token-diet.md) — session-startup token cost cut about 80%; remaining: trim the three 15 KB+ open ROADMAP rows (owner text) and decide whether `.claude/` hooks should be tracked
- [total-conditioned-lattice](total-conditioned-lattice.md) — total-banded key-number lattice measured 2026-09-23 vs served discrete read, mixed sign across metrics/seasons, unresolved_below_power; two `weak-signals record` commands queued for the root
- [line-move-regrade-legacy](line-move-regrade-legacy.md) — 2026-09-23: batch 1 (8 legacy terms) measured, roof_state_predicted_open best-of-8 fails OOS replication on 2011-2019; batch 2 partial, 7/8 recorded (rookie_priors, low_total_div_home_dog resolved wrong-sign); division_revenge_tilt builder failing, batch-1/2 record commands queued for the root.
- [prospective-leads-2026-09-23](prospective-leads-2026-09-23.md) — 2026-09-23: total_conditioned_key_number_lattice_v1 registered and live-dry-run verified as a prospective challenger; all-books median market move dropped (does not beat the served move, P+ 0.22).
- [clv-metric-everywhere](clv-metric-everywhere.md) — ENG-47, 2026-09-23: line-move-toward-pick yardstick now emitted everywhere opener-evaluation runs; paired eval (5 cells) all unresolved_below_power.
- [scheduler-once-timeout](scheduler-once-timeout.md) — fixed and committed `786a569`: `--once` defers to a live daemon via a file lock, scheduler suites green.
- [sunday-market-probability](sunday-market-probability.md) — activated 2026-09-20 as `leader_median_through_sunday_prekick_v1`; six unresolved metric/protocol cells recorded under `sunday_market_probability_fixed_v1`.

## Done

- [card-freshness-and-log-labels](done/card-freshness-and-log-labels.md) - 2026-09-24: refresh passes rewrite the card's freshness line; no follow-rule log label.
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
