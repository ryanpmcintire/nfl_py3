# draftcap-forward-log

## Goal
Prospective test of MOD-23 unit 10 candidate 2 arm B (home-minus-away snap-weighted draft_ovr of the prior game's top-11 offence and defence starters added to the 88 base inputs; corrected run B 2020-25 +0.53 pp [0.14, 1.11], B 2013-19 +0.29, 4 looks, coefficients negative). Log every 2026 game from Week 7 before its pick deadline, frozen and untuned. Never changes the served card.

## State
`scripts/draftcap_forward_log.py` (execs mod23_unit10b.py source up to `main()` for add_features/build_inputs; reuses mod24_forward_log helpers). Per run: builds draft-capital features (merge_asof, prior game strictly earlier, asserted twice), scores arm (88 + 2 columns) and same-pipeline zero-delta base (88) through score_outcome_week with home-side offsets and discrete reader. Rows: arm prob/side, base prob/side, served model/pool prob and side when the week's forecast exists, line, kickoff, deadline, recorded_at_utc, the two feature values, each team's prior-game order used.
Frozen 2026-10-18 enrollment: `artifacts/draftcap_forward/frozen.json` (arm B ridge coefficients on the two columns fitted on seasons 2013-2025 with the corrected unit10b input rules; sha256 of that table; later runs refit and abort if it differs). The in-season pipeline still refits its ridge on all completed games each week, as the served pipeline does; only the 2013-2025 fit is the frozen reference.
Canonical row = latest row per game_id recorded before its deadline (declared now).
Week 7 logged: `artifacts/draftcap_forward/runs/2026-week-07-20261018T225048Z.json` (14 games; arm side equals base side on 0 of 14 differing, i.e. 0 flips). No served forecast folder for Week 7 yet, so served fields are absent in that row set.
Scheduler jobs `draftcap_forward_log_thu_1915`, `draftcap_forward_log_sun_1115` added in scripts/capture_scheduler.py beside armb; `--run-job draftcap_forward_log_thu_1915 --dry` printed MANUAL-DRY-RUN OK.

## Tried
- Week 7 run: combine match 63.8% plus name-key fallback to 71.8% (as unit10b); feature coverage 2026 = 1.0.

## Next
Let the scheduler run Thu/Sun Weeks 7-18. After Week 18 is final: score arm vs base on decisive games, paired, season/week-block bootstrap, accuracy, Brier, log loss v market-only; record through `weak-signals record` (family mod23_unit10_draft_capital, forward look). No interim performance reads.

## Open
- DATA GAP (measured): the latest snap_counts file (data/players/raw/20261018T131559Z) ends at season 2025 week 22, so every 2026 game currently uses each team's final 2025 game as "prior game" (latest_snap_order 202522; Week 7 rows show prior orders 202518-202522). Pregame-safe but stale; the 2026 snap feed must refresh for the arm to use 2026 starters. Until it does, the log is mostly the 2025 end-of-season starters; flag to the orchestrator.
- Zero Week 7 flips is a consequence of the above plus the small coefficient on two columns; not read as performance.
- Served-vs-same-pipeline base mismatch (see armb lane) applies here too.
