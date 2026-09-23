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

- [week3-research-recorders](week3-research-recorders.md) - nine challenger records remain missing; all 16 served Week 3 paper decisions are complete.
- [news-trigger-refresh](news-trigger-refresh.md) - MKT-08 dispatch implemented and verified; prospective comparison awaits new events.
- [gh-window-incident](gh-window-incident.md) - prior runaway CLI windows remain a separate investigation; no gh commands used for this deployment.

- [confidence-best-pick-unification](confidence-best-pick-unification.md) - shared calibrated selector published in `5fb88a2`; confidence reliability and within-week ranking research remains open


- [pooled-signal-model](pooled-signal-model.md) — MOD-20, units 1-5 recorded; unit 5 interactions -0.40 pts unresolved; next: a genuinely new family, not a reparameterisation
- [pool-rank-card](pool-rank-card.md) — POOL-01, unit 1 measured 2026-09-16 (unresolved); unit 2 only with a fitted field
- [every-metric-every-experiment](every-metric-every-experiment.md) — ENG-46 implementation done 2026-09-16; backfilled-look accounting remains open (runner emits four metrics, backfill recorded, margin cells corrected same day, pool + findings read the margin family).
- [lane-replay-dream-rsi](lane-replay-dream-rsi.md) — ENG-45, units 1-2 done 2026-09-16 (both unresolved); unit 3 not recommended
- [conditional-signal-atlas](conditional-signal-atlas.md) - MOD-19: owner rejected the deployed Findings page as not useful and below expectations; original mockup is saved, feature remains unresolved, further implementation awaits a grounded direction.
- [lead65-protection-window-split](lead65-protection-window-split.md) — gate folded into the fit loses (2026-09-23); the separate early-window overlay stays prospective; owner question open
- [lead59-archive-battery](lead59-archive-battery.md) — LEAD-59 src fix and battery re-run done, 23 cells recorded; open: type-trait binning with the archive on
- [token-diet](token-diet.md) — session-startup token cost cut about 80%; remaining: trim the three 15 KB+ open ROADMAP rows (owner text) and decide whether `.claude/` hooks should be tracked
- [odds-api-key-deactivated](odds-api-key-deactivated.md) — 2026-09-19: bulk odds captures failing HTTP 401 DEACTIVATED_KEY (billing); owner action needed before the next odds window

## Done

- [players-on-field-rating](done/players-on-field-rating.md) - MOD-22 units 1-3, 2026-09-23: four fitted terms unresolved, Tuesday availability change not reconstructible; family open, cached construct exhausted.
- [opener-error-transfer](done/opener-error-transfer.md) - XLG-09 units 1-2, 2026-09-23: widened CFB transfer term +0.13 pts, P+ 0.76, unresolved; not served.
- [ui20-confidence-column-guide](done/ui20-confidence-column-guide.md) - 2026-09-23: Confidence column hover help and Column guide entry published.
- Superseded 2026-09-23 when the four-term fitted probability became the served side selector (2026-09-20); their open questions were about flip rules that no longer serve: [four-term-probability](done/four-term-probability.md), [joint-probability-model](done/joint-probability-model.md), [market-updated-model](done/market-updated-model.md), [served-flip-rules-hold](done/served-flip-rules-hold.md), [leader-median-model-confidence](done/leader-median-model-confidence.md), [handle-follow-override-defect](done/handle-follow-override-defect.md), [best-pick-served-score-ranker](done/best-pick-served-score-ranker.md).
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
