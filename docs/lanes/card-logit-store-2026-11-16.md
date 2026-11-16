# Card logit store (2026-11-16)

## Goal
Every pick revision persists, per game, the served card decomposition: model logit, each fitted term's contribution (intercept, model, flags, move, move-available, hook), served home probability and side, fit artifact id, input row hash. Lets an input fix be scored on card sides (gap named in docs/lanes/serve-parity-audit-2026-11-16.md Open).

## State
- Writer is src/nfl_ats/pick_refresh.py (plan_refresh builds RefreshedGame; record_plan appends to artifacts/prospective/pick_revisions.parquet). It is PINNED (independent_validation PINNED_NAMES), so nothing in src/ was edited.
- Patch: tests/scratch/card_logit_store/card_logit_store.patch (pick_refresh.py + artifact_contracts.py; `git apply --check` OK; ruff check and format --check clean on patched copies in tests/scratch/card_logit_store/nfl_ats_patched/). pick_refresh: 17 served_* columns appended to PICK_REVISION_COLUMNS, NaN/"" legacy defaults in load_pick_revisions, RefreshedGame.served_terms, helpers _served_terms/_input_row_sha256/_finite_or_zero, record_plan writes them. artifact_contracts: same 17 names on _PICK_REVISION_LEDGER_COLUMNS (required by test_pick_revision_ledger_columns_match_pick_refresh_source_of_truth). Nothing rendered changes.
- Measured: verify_patch.py 2026 10 (patched module, live data): 14 games, max |sum(contributions) - logit(served p)| 3.3e-16, side and probability match on all; record_plan frame round-trips through parquet, 84 legacy rows load with blank defaults. Sample 2026_10_BUF_NYJ: model logit 0.4404, contributions intercept -0.0196, model 0.1357, flags 0, move 0.1102, move-available 0.0250, hook -0.0218, home logit 0.2295, p 0.5571 HOME, artifact pick_probability/20261116T171555Z.
- Measured: pytest -k "ledger or pick or board": unpatched 445 passed; pick_refresh-only injection 444 passed 1 failed (the contract test above). Rerun with both modules injected (plugin tests/scratch/card_logit_store/card_logit_plugin.py, PYTHONPATH=that dir, `-p card_logit_plugin`) was blocked by the tool cap: NOT run.

## Tried
- Week 11 at now: no Tuesday card yet. Week 10 with a past now: arrest snapshot future-dated guard refuses.

## Next
1. Run the both-injected pytest command above once; expect 445 passed.
2. Orchestrator applies the patch at a pinned change (re-enroll the independent-validation source digest same session), then `capture_scheduler.py --run-job refresh_thu --dry` and one recorded refresh.

## Open
- Ledger stores changed games only; the Tuesday original card (paper-decision ledger, clv.py, pinned) has no decomposition. Scoring unchanged card sides needs the same columns on the paper-decision record or a per-run all-games snapshot.
