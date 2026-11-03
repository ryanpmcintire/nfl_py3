# ENG-52

## Goal
Opener timestamp audit: capture time versus first public posting, with graded results split by capture lag (ROADMAP ENG-52).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: An opener captured hours late is partly a moved line; the fair opener test assumes it is the first number. Report the lag distribution per game and regrade the fair test with late captures excluded. Capture lag enters as a grade filter, never a model term.

## Tried
Novelty check only. Nearest: ENG-51 (input provenance guard), LEAD-64 (capture integrity), MKT-08; grep 'timestamp drift' found none. Differs: Capture lag is a grade input, not a model input.

## Next
First unit: Compute lag per graded game from capture timestamps and any archived first-post times; regrade with and without late games with a paired bootstrap.

## Open
Data: Local capture logs (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03, 21:25Z)
Script scripts/eng52_opener_lag_audit.py, table artifacts/eng52/opener_lag.csv (1,537 games, artifact 20261103T195331Z). Opener = tue_open historical_backfill snapshot (the-odds-api, requested at Tuesday 09:00 ET per odds_backfill.DECISION_TIMES; API returns the snapshot at or before the request). Target = that Tuesday 09:00 ET per game cycle.
Measured: snapshot minus target is -0.083 h (2020-21) and -0.072 to -0.083 h (2022-25); max +0 h. Flagged (snapshot after target) = 0 of 1,537 in every season. Median bookmaker quote age at snapshot 0.12 h (2020-21), 0.04 (2022), 0.01-0.02 (2023-25). Regrade with flagged removed is identical to the headline (nothing removed); flagged-only is empty, so no interval, no weak-signals record (no effect unit estimable). No cited number moves.
Limit: the audit shows the opener is captured on time, not that the line is the first posting; first-post time is not in the archive, so movement between first post and Tuesday 09:00 ET is unmeasured (inferred only).
Next: unit 2 only if a first-post source exists (Monday/previous Sunday captures); else close ENG-52 as audit-clean.
