# POL-10 prospective 2026 evidence

## Goal
Score frozen pregame decisions at the recorded line, retain close grades as secondary evidence, and compare challengers on exactly paired decisive games.

## State
**Measured September 28:** the served scorecard now uses actual frozen published
decisions: 28–19, one pending; Best Pick 1–2. The previous 30–17 reconstruction is
superseded. Valid pre-publication raw/market bindings exist for 24/48 games, 23
decisive; all four baseline arms compare those same games. Unavailable bindings
and unmatched challenger lines remain explicit. Evidence and reliability cells:
`artifacts/prospective_scorecard/20260928T162048Z/` and
[the checkpoint report](../backlog_research_20260928.md). No model selection.

**Measured:** `nfl-ats prospective-score --features data/processed/game_features.parquet --start-season 2026 --bootstrap-samples 1000` exited 0 on 2026-09-27. Current artifact: `artifacts/prospective_scoring/20260927T154056Z`. Challenger statuses: 57 scored, one invalid arm, three missing ledgers, three unsupported. A scored status does not imply every game has finished.
**Measured:** the immutable rookie-crew recorder and paired scorer adapter passed root callable verification. Both probabilities and their lineage survive settlement. Enrollment remains off; the real scorer correctly reports `missing_ledger` for this challenger.

## Tried
- Immutable enrollment/game keys, exact replay, strict pre-decision/pre-kickoff timing, source identity and payload hashes are validated.
- Root verification retained matched on/off arms, home and selected-side probabilities, policy and calibration provenance. Altered hashes and late records failed closed; missing rows were not fabricated.
- Existing generic/dedicated collisions fail closed. No historical ledger was rewritten.
- Evidence: `.tmp/resume-crew-root-probe.log`, `.tmp/resume-prospective-score.log`, and `.tmp/resume-prospective-score-result.json`.

## Next
**Measured:** final Ruff, mypy and all 1,645 tests passed. Future explicit enrollment requires a valid frozen candidate probability and new pregame observations; the recorder does not create a serving rule.

## Open
The equal-book source still lacks a valid pick side. Three ledgers and three scorer adapters remain unavailable. No research closure or serving promotion is made here. AGENTS.md requires admissible mechanism/power grounds for closure; zero-crossing intervals remain unresolved below power.
