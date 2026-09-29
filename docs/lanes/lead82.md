# LEAD-82 — Tuesday move, unit 2

## Goal
Complete one hourly Tuesday-term replay; own only the unit-2 script, report, and this lane. Research only; no publication, registry writes, tests, or Git mutations.

## State
**Measured:** replay complete: 799 games, 54 week blocks; served probability and late-move parity exact. Changed picks 13-23, 36.11% [20.00, 53.85]. Candidate-versus-served improvement: Brier +0.000242 [-0.001137, 0.001572], probability_positive 0.6467; log loss +0.000557 [-0.002389, 0.003383], 0.6567; accuracy -1.2516 points [-3.1017, 0.3704], 0.0606. **Inferred:** unresolved_below_power; serial recording pending, no serving conclusion.

## Predeclared protocol and amendment
Saved before outcome access; full declaration retained in `tests/scratch/codex/lead82_unit2/protocol_before_run.md`. **Read:** move audit attributes 164 late differences to later scheduled captures. Preserve the correct served hourly late values; build only Tuesday on `intraday_hourly`. Median same-leader-book terminal-minus-anchor spread: last quote at/before Tuesday noon ET to last before Wednesday, requiring a post-noon terminal. Enforce source/week/observation/snapshot/bookmaker/kickoff clocks. No parameter search or threshold changes.

Historical opener is the frozen pool proxy; 2020-2025 source, 2023-2025 complete nonpush scoring subset. Preserve pushes in inventory. Owner-requested LOSO 2023/2024/2025 supersedes the original single outer year. Base four-term coefficients fit all other source seasons, then freeze; one Tuesday coefficient fits other covered seasons with base logit as offset, training SD without centering, ridge 0.001. Optimistic all-data IS; no new intercept/slope/side rule. One fitted probability selects sides; zero crossing closes nothing. Retrospective LOSO is not chronological deployment evidence.

Primary Brier; accuracy/log loss/binary conditional RPS (identical to Brier). Compare served, discrete model-only, fair market; report decisive records first, coefficients, IS/OOS/gaps, five reliability bins. Bootstrap 10,000 season-stratified week draws, seed 20260929; 95% intervals, half-credit zero draws for probability_positive. Fixed accounting: **136 metric panels** + 48 coefficients + 20 reliability cells + 12 decisive records = 216 within **261 looks**, 45 unspent; no protocol revision after outcomes.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead82_unit2.py` exited 0 after one pre-fit integrity failure: corrected result-minus-opener convention. One completed fit/score replay; two threads, writable scratch uv cache, offline. Saved-row arithmetic, combined probabilities, syntax/comment/docstring/whitespace and source hashes verified. Report/batch generated from saved results without refitting. Post-noon gate removed one stale book pair: 2,348 pairs; one Tuesday game median differs from unit 1 by 0.25 points. Fold Tuesday coefficients 0.276968/0.127540/0.217182; full intervals and IS/OOS in `docs/lead82_unit2.md`.

## Record commands
Prepared, NOT run. Batch contains only three correlated candidate-versus-served endpoints; RPS duplicates Brier. Source and plain summary are also stored inside the batch (batch handler reads those fields there).

```bash
UV_CACHE_DIR=tests/scratch/codex/lead82_uv_cache UV_OFFLINE=1 .tools/uv.exe run --no-sync nfl-ats weak-signals record \
  --batch tests/scratch/codex/lead82_unit2/registry_batch.json --source docs/lead82_unit2.md \
  --plain-summary 'Adding Tuesday afternoon line moves slightly improved the confidence estimates, but the changed picks went 13-23. Keep the current picks while this is studied further.'
```

## Next
Orchestrator reviews the report and executes the candidate-versus-served batch serially. Rows, executed-source snapshot, frozen protocol, hashes and diagnostics: `tests/scratch/codex/lead82_unit2/`.

## Open
Upstream weekly cutoffs verified by source inspection, not row-level replay; selected features and retrospective folds limit inference. No closing ground established; no serving change. Owner-confirmed launcher workaround remains a workaround.
