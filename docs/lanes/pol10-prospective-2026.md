# POL-10 prospective 2026 evidence

## Goal
Score frozen pregame decisions at the recorded line, retain close grades as secondary evidence, and compare challengers on exactly paired decisive games.

## State
**Measured September 28:** the served scorecard now uses actual frozen published
decisions: 28–19, one pending; Best Pick 1–2. The previous 30–17 reconstruction is
superseded. Exact source proof now supports 24 raw baselines (23 decisive) and three
same-book market pairs. The other rows remain explicit; missing market proof does
not remove valid raw comparisons. Current prediction rows and reliability cells:
`artifacts/prospective_scorecard/20260928T175751Z/`. Root independently verified
80 paired metrics and 26 source hashes. The [seven-category report](../seven_backlog_completion_20260928.md)
supersedes earlier timestamp-only market comparisons. No model selection.

**Measured 2026-10-08 (through Week 4, descriptive only, no selection):** `scripts/prospective_scorecard_2026.py` exited 0 -> `artifacts/prospective_scorecard/20261008T213825Z/summary.json`. Frozen published record 38-26 (59.4%, n=64; weeks 9-7, 14-2, 6-10, 9-7); bootstrap 95% 46.9%-71.9%, probability_positive vs 0.5 about 0.92 (root bootstrap, not a scorer field). Best Pick 1-3 (Wk1 L, Wk2 W, Wk3 L, Wk4 L). Raw-baseline pairs n=40 of 64: served Brier 0.2473 / log loss 0.6876 vs raw 0.2400 / 0.6729 vs coin 0.25 / 0.6931. Exact same-book market pairs n=4: served 0.2268 / 0.6463 vs market 0.2544 / 0.7019 (raw 0.2652 / 0.7245). All-64 served Brier 0.2392, log loss 0.6714. Market n=4 is input coverage only. `nfl-ats prospective-score` (bootstrap 1000) exited 0 -> `artifacts/prospective_scoring/20261008T213837Z`; challengers 57 scored, 1 invalid_arm, 3 missing_ledger, 3 unsupported. Wk1-4 active model 10/14/7/8 of 16; base-no-overlay 10/11/7/7. Look count: 11 record, 120 probability, 20 reliability, 63 paired cells.

**Measured:** `nfl-ats prospective-score --features data/processed/game_features.parquet --start-season 2026 --bootstrap-samples 1000` exited 0 on 2026-09-27. Current artifact: `artifacts/prospective_scoring/20260927T154056Z`. Challenger statuses: 57 scored, one invalid arm, three missing ledgers, three unsupported. A scored status does not imply every game has finished.
**Measured:** the immutable rookie-crew recorder and paired scorer adapter passed root callable verification. Both probabilities and their lineage survive settlement. Enrollment remains off; the real scorer correctly reports `missing_ledger` for this challenger.

**Measured 2026-11-02 (through Week 8, 1 game pending CHI at SEA, descriptive only):** scorer exited 0 -> `artifacts/prospective_scorecard/20261102T220207Z/summary.json`; `nfl-ats prospective-score` (bootstrap 1000) exited 0 -> `artifacts/prospective_scoring/20261102T220227Z`. Frozen published card 64-56, 1 pending (53.3%, n=120 decisive); weeks 9-7,14-2,6-10,9-7,8-7,6-8,7-7,5-8. Best Pick 4-4. Ledger active_model at recorded (opener) line 65/120 (54.2%), probability_positive vs coin 0.749, 95% 43.7%-66.1%; at close 63/116 (54.3%, 4 pushes). Tuesday-line grade is not a scorer field. Served Brier 0.2470 / log loss 0.6872 vs coin 0.25 / 0.6931; raw pairs n=96 raw 0.2478 / 0.6886 vs served 0.2522 / 0.6979; market pairs n=7 only (served 0.2654, market 0.2573, raw 0.2707). 24 games (weeks 1-2) raw unavailable, 113 no market pair; all stay listed. Paired decisive (challenger wins / served wins / ties): 45 scored, 10 grading_line_mismatch, 5 unsupported, 3 missing_scored_rows, all listed in summary.json. Largest: injury_signal_refresh_tilt 32-24 of 120, turnover_luck_rebound 27-22, pbp08_protection_mismatch 21-13; worst: third_down_reversion 21-26, bye_edge_fade 19-26. No paired probability_positive field exists; none recorded to weak-signals (record cells not declared by row; pending root).

**Why scorer stopped:** manual-only. No job in `scripts/capture_scheduler.py` references it and `data/scheduler_log.txt` has no entry; it has no argv. Same for prospective-score.

## Tried
- Immutable enrollment/game keys, exact replay, strict pre-decision/pre-kickoff timing, source identity and payload hashes are validated.
- Root verification retained matched on/off arms, home and selected-side probabilities, policy and calibration provenance. Altered hashes and late records failed closed; missing rows were not fabricated.
- Existing generic/dedicated collisions fail closed. No historical ledger was rewritten.
- Future published rows preserve the exact forecast artifact used to build the card,
  its creation time and original line, and SHA-256 hashes for both recommendations and
  metadata. Complete explicit bindings take precedence; partial, late, wrong-line, or
  mutated bindings never fall back to the legacy decision ledger. Historical nulls are
  not enriched from later artifacts. `scripts/research_input_readiness.py` retains all
  frozen rows and gives each unavailable comparison a source-specific reason and hashes.
- The raw baseline uses `home_cover_probability`, the forecast's conditional non-push
  probability. The misleadingly named `home_cover_probability_excluding_push` column is
  cover mass and is not a binary scoring probability. Pushes remain ungraded, so future
  integer-line rows compare the same event as the frozen served probability.
- A market quote is eligible only when its immutable forecast row records an observation
  no later than forecast creation and an immutable completed raw snapshot contains the
  same game, opposing lines and both prices at one bookmaker. The proof records the
  bookmaker plus quote, manifest and response hashes. A missing timestamp, unmatched
  nominal price or later observation is unavailable market evidence and never removes an
  otherwise valid raw probability. **Measured readiness:** 24/48 raw forecasts and 3/48
  exact market pairs are available; this is input coverage, not a serving verdict.
- Evidence: `.tmp/resume-crew-root-probe.log`, `.tmp/resume-prospective-score.log`, and `.tmp/resume-prospective-score-result.json`.

## Next
**Root, after the Week 9 lock, add to capture_scheduler.py** Job("prospective_scorecard_tue","tue","09:30",20,[UV,"run","--no-sync","python",str(REPO/"scripts"/"prospective_scorecard_2026.py")],True,desc,added_on,catch_up=True) (measured runtime about 20 s) and Job("prospective_score_tue","tue","09:40",30,[UV,"run","--no-sync","nfl-ats","prospective-score","--features","data/processed/game_features.parquet","--start-season","2026","--bootstrap-samples","1000"],True,desc,catch_up=True). Tuesday morning is after Monday night settles and before the 12:20 lock. Rerun after CHI at SEA settles.
**Measured:** final Ruff, mypy and all 1,645 tests passed. Future explicit enrollment requires a valid frozen candidate probability and new pregame observations; the recorder does not create a serving rule.

## Open
The equal-book source still lacks a valid pick side. Three ledgers and three scorer adapters remain unavailable. No research closure or serving promotion is made here. AGENTS.md requires admissible mechanism/power grounds for closure; zero-crossing intervals remain unresolved below power.

**Measured 2026-11-03 Week 8 scored (CHI at SEA settled):** see docs/lanes/week8-forward-scoring-2026-11-03.md.
