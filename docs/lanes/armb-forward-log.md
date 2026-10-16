# armb-forward-log

## Goal
Prospective test of MOD-23 unit 6 arm B (rolling inputs adjusted for the team's own non-established-QB starts) and unit 9 (arm B logit in the four-term card, Tuesday view). Log every 2026 game from Week 7 before its pick deadline, frozen and untuned. Score only after Week 18 is final, paired, week-blocked; no scoring before.

## State
Built `scripts/armb_forward_log.py` (standalone; execs mod23_unit6.py source up to `main()` to reuse flags/coef/wrapped; reuses mod24_forward_log helpers). Per run: rebuilds raw and arm B state from the latest raw snapshot plus snap_counts, applies delta to the feature table, scores via the served pipeline (score_outcome_week, home-side offsets, discrete reader).
Row fields: arm B model-alone prob/side, same-pipeline base (zero delta) prob/side, card (frozen served coefs, move terms 0) with arm B logit, card with base logit, composition flag sum, served model prob/side and served pool side/prob when the week's forecast exists, line, kickoff, deadline, recorded_at_utc.
Frozen at enrollment 2026-10-16 22:31 UTC: `artifacts/armb_forward/frozen.json` (arm B 2026 coefficients, fit seasons 2013-2025, sha256 90ad74f0...; card artifact pick_probability/20261016T221402Z, coefficients sha256 a72aa1e8...). Later runs abort if the arm B hash differs.
Canonical row = latest row per game_id recorded before its deadline (declared now).
Week 7 logged: `artifacts/armb_forward/runs/2026-week-07-20261016T223138Z.json` (14 games; arm B side equals base side on all 14).
Scheduler jobs `armb_forward_log_thu_1910`, `armb_forward_log_sun_1110` added in scripts/capture_scheduler.py; `--run-job armb_forward_log_thu_1910 --dry` printed MANUAL-DRY-RUN OK.

## Tried
- Zero-delta control through this pipeline vs served predictions.csv (Week 6): corr 0.20, max abs diff 0.076, so the feature table here differs from the served forecast build; arm B is therefore compared with the same-pipeline base, served shown beside.
- Week 6 dry check (13 games): rows carry served fields; no Week 6 file written.

## Next
Let the scheduler run Thu/Sun for Weeks 7-18 (default week = next with an upcoming kickoff, never before Tuesday noon line freeze). After Week 18 final: score arm B vs base and card B vs card base on decisive games, paired, season/week-block bootstrap, accuracy, Brier, log loss; record through `weak-signals record`.

## Open
- Composition flags here omit the protection/forecast inputs the served card uses; card columns are internally consistent but may differ from the served card's flag sum.
- Features parquet vs served forecast mismatch (above) unexplained; not tuned away.
- Snap counts for 2026 completed games refresh with the player snapshot; a missing refresh leaves a flag at 0.
