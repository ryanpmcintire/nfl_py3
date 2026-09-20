# Lanes

A lane file is the state a fresh session needs to continue a task after the
previous session was cleared. One file per task, named by a short slug. The
session that works a lane updates it whenever a unit of work completes, and
every response ends with a one-line clear verdict (AGENTS.md, Lanes and
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

- [confidence-best-pick-unification](confidence-best-pick-unification.md) - shared calibrated selector published in `5fb88a2`; confidence reliability and within-week ranking research remains open

- [sunday-readiness-2026-09-20](sunday-readiness-2026-09-20.md) - noon forecast, provisional NO Best Pick and board republished locally; 16 paper and six revision rows agree, root release pending

- [pooled-signal-model](pooled-signal-model.md) — MOD-20, unit 1 done 2026-09-16 (first pooled fit recorded, unresolved); next: grow the reproducible feature set
- [market-derived-ratings](market-derived-ratings.md) — MOD-21, closed 2026-09-16 (refuted on accuracy; Brier companion unresolved)
- [line-move-target](line-move-target.md) — MKT-20, closed 2026-09-16 (refuted as a target; the move stays a fitted term)
- [pool-rank-card](pool-rank-card.md) — POOL-01, unit 1 measured 2026-09-16 (unresolved); unit 2 only with a fitted field
- [every-metric-every-experiment](every-metric-every-experiment.md) — ENG-46, done 2026-09-16 (runner emits four metrics, backfill recorded, margin cells corrected same day, pool + findings read the margin family)
- [lane-replay-dream-rsi](lane-replay-dream-rsi.md) — ENG-45, units 1-2 done 2026-09-16 (both unresolved); unit 3 not recommended
- [week1-covers-settled](week1-covers-settled.md) — 2026-09-15: Week 1 graded, the card went 9-7; three surfaces were grading the Tuesday lock and showing 10-6, all now read the served card through `served_paper_decisions`; totals loaders join their own build snapshot
- [serve-calibrated-probability](serve-calibrated-probability.md) — MKT-19, 2026-09-14: the per-game cover chance is back on every reader surface, sourced from MKT-18's four-term model instead of the raw model whose number ran backwards against its own record; coefficients live in `artifacts/pick_probability` behind an active pointer and are refit with `nfl-ats fit-pick-probability`, never as literals in `src/`; the nine-member flip chain still records as the paired challenger but no longer decides the card
- [four-term-probability](four-term-probability.md) — MKT-18, owner directive: one shared flag weight (not nine) plus the model logit and the market move, LOSO 2020-2025; headline finding is a calibration ordering (M5's accuracy rises with its own confidence, M0's falls), M5 leans ahead of M1 on every metric (unresolved), the model term `a` is not distinguishable from zero in any fold while the flag weight and move both clear zero in nearly every fold — keeping the model term recommended anyway per rule (a); Best Pick tied exactly against the currently-served ranker
- [joint-probability-model](joint-probability-model.md) — MKT-17, owner directive: base logit + all nine composition flags + market move fit as one joint logistic model, LOSO 2020-2025; raw model overconfident (fitted slope 0.135-0.424 on its own logit), flags jointly fit beat the card on log loss/Brier (intervals entirely positive) but the served flip chain (M1) reads an even cleaner win over the card, so the joint model does not beat what is served; recalibration alone is the supported change
- [market-updated-model](market-updated-model.md) — owner directive 2026-09-14: the line move as a fitted term in the model's probability, out of season +2.00 vs the card [-2.10, +6.05] P+ 0.84, a dead heat with the hard rule's own out-of-season +2.13; flips concentrate in the near-50/50 band as the owner expected; next: fold the composition flags in as terms
- [served-flip-rules-hold](served-flip-rules-hold.md) — 2026-09-14: inventory of the ten served rules that flip a side outright; line-move, handle and rookie-crew off, interim-HC and precip held (n 39 and 29-50), seven larger-sample flips still served pending the combined model
- [handle-follow-override-defect](handle-follow-override-defect.md) - the fake 50.0% is fixed for future locks only (published weeks untouched); the leader-median override-distance measurement was redone from scratch, see leader-median-model-confidence
- [leader-median-model-confidence](leader-median-model-confidence.md) — from-scratch redo, owner-ordered 2026-09-14: identical 61-49 confirmed and explained (a mathematical identity, not a defect); nested leave-one-season-out search finds the served rule's honest out-of-sample effect is +2.13 pts, and a confidence gate makes it worse, not better — no gate recommended
- [best-pick-served-score-ranker](best-pick-served-score-ranker.md) - 2026-09-14: the star sits on the card's own most confident eligible pick only because that ranking has no fitted parameter; the +5.94 was 11-5 on 16 decisive weeks and is recorded as no evidence; the model's per-game confidence is uncalibrated (see joint-probability-model), so no ranker has a basis yet
- [conditional-signal-atlas](conditional-signal-atlas.md) — MOD-19, owner direction 2026-09-13: every signal measured inside predeclared splits across any dimension and decided together; LEAD-65 is the worked case
- [lead65-protection-window-split](lead65-protection-window-split.md) — owner question 2026-09-13: measured: the tilt's edge is IN weeks 1-4 (P+ 0.977), weeks 5-18 a probable drag (P+ 0.099); next a week-gated paired challenger
- [sunday-gameday-2026-09-13](sunday-gameday-2026-09-13.md) — Week 1 Sunday: hand odds capture, new 09:30 lineups_sun_am refit job, daemon restarted; open: Week 1 --replace-week re-record (owner)
- [lead64-headline-parser](lead64-headline-parser.md) — LEAD-64 headline designation parser, three passes done; next is the morning comparison against the official feed
- [lead59-archive-battery](lead59-archive-battery.md) — LEAD-59 src fix and battery re-run done, 23 cells recorded; open: type-trait binning with the archive on
- [token-diet](token-diet.md) — session-startup token cost cut about 80%; remaining: trim the three 15 KB+ open ROADMAP rows (owner text) and decide whether `.claude/` hooks should be tracked
- [odds-api-key-deactivated](odds-api-key-deactivated.md) — 2026-09-19: bulk odds captures failing HTTP 401 DEACTIVATED_KEY (billing); owner action needed before the next odds window

## Done

- [windows-linux-recovery](done/windows-linux-recovery.md) - 2026-09-20: restored Windows dependencies and scheduler, isolated Linux environment instructions, repaired Shopify skill YAML, verified all gates and regenerated dashboard locally

- [halves-ledger-lockday](done/halves-ledger-lockday.md) — 2026-09-12
- [dashboard-2026-09-12-evening](done/dashboard-2026-09-12-evening.md) — 2026-09-12, Books-now market-move line
