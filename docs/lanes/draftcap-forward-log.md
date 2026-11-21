# draftcap-forward-log

## Goal
Prospective test of MOD-23 unit 10 candidate 2 arm B (home-minus-away snap-weighted draft_ovr of the prior game's top-11 offence and defence starters added to the 88 base inputs; corrected run B 2020-25 +0.53 pp [0.14, 1.11], B 2013-19 +0.29, 4 looks, coefficients negative). Log every 2026 game from Week 7 before its pick deadline, frozen and untuned. Never changes the served card.

## State
`scripts/draftcap_forward_log.py` (execs mod23_unit10b.py source up to `main()` for add_features/build_inputs; reuses mod24_forward_log helpers). Per run: builds draft-capital features (merge_asof, prior game strictly earlier, asserted twice), scores arm (88 + 2 columns) and same-pipeline zero-delta base (88) through score_outcome_week with home-side offsets and discrete reader. Rows: arm prob/side, base prob/side, served model/pool prob and side when the week's forecast exists, line, kickoff, deadline, recorded_at_utc, the two feature values, each team's prior-game order used.
Frozen 2026-10-18 enrollment: `artifacts/draftcap_forward/frozen.json` (arm B ridge coefficients on the two columns fitted on seasons 2013-2025 with the corrected unit10b input rules; sha256 of that table; later runs refit and abort if it differs). The in-season pipeline still refits its ridge on all completed games each week, as the served pipeline does; only the 2013-2025 fit is the frozen reference.
Canonical row = latest row per game_id recorded before its deadline (declared now).
Week 7 logged: `artifacts/draftcap_forward/runs/2026-week-07-20261018T225048Z.json` (14 games; arm side equals base side on 0 of 14 differing, i.e. 0 flips). No served forecast folder for Week 7 yet, so served fields are absent in that row set.
Scheduler jobs `draftcap_forward_log_thu_1915`, `draftcap_forward_log_sun_1115` added in scripts/capture_scheduler.py beside armb; `--run-job draftcap_forward_log_thu_1915 --dry` printed MANUAL-DRY-RUN OK.
2026-11-21 11:20 ET RE-ENROLMENT v2 (measured): the 11-19 19:15 run failed the guard (refit 2013-25 no longer reproduces v1). `artifacts/draftcap_forward/frozen_v2.json` enrolled 2026-11-21T16:19:56Z, same construction (fit_frozen, seasons 2013-2025, alpha 10, 11 starters), current feature table sha 406f8677...; coefficients off 0.028659 / def -0.231327 (v1 0.027339 / -0.230242), digest 633f2257... (v1 37d1a0a8...). Declared in the file: reason (inferred cause, not yet diffed: 11-17 lock commit 62d554ea, QB repair + pregame weather rewrote 2013-25 rows), window (W11 games not kicked off at enrolment, logged before deadline, through W18), v1 kept as history, never pooled without saying so. `frozen.json` (v1) untouched (mtime 10-18). Script now reads `frozen_v2.json` (FROZEN_PATH), keeps the allclose guard against v2, creates v2 only if missing, payload adds `frozen_version` and `frozen_file`; v1 run files lack the field (= v1). Week 11 v2: `runs/2026-week-11-20261121T162000Z.json`, 12 rows (all Sun/Mon games), IND_HOU refused past_deadline (lost, no back-fill); served fields present on 12; arm side differs from base on 2 of 12 (descriptive, not a performance read). Second `--dry` reproduced v2 (exit 0).

## Tried
- Week 7 run: combine match 63.8% plus name-key fallback to 71.8% (as unit10b); feature coverage 2026 = 1.0.

- 2026-11-02 abort cause (measured): refit coefficients equal frozen to float noise (0.02733909161105847 vs 0.027339091611058446; -0.23024201922077514 vs -0.23024201922077533) but the sha256 of their JSON text differs, so the hash gate aborted every run since Week 8. Inputs unchanged (4630 rows, 3546 with features). Fix: freeze() compares coefficients with np.allclose against frozen.json and the payload keeps the frozen digest. In-season scoring still refits on completed games as declared; frozen set is the reference only. frozen.json untouched.
- Week 8 not logged: the repaired job found every Week 8 game past_deadline; Week 8 is recorded as missed (no backfill support). Week 9 --dry ok (latest_snap_order 202608, snap feed now refreshes via forward_snaps).

## Next
Let the scheduler run Thu/Sun through Week 18 (Sun 11-22 11:15 will refit, compare to v2, and append W11 rows; a guard failure there means 2013-25 rows moved again). After Week 18 is final: score arm vs base on decisive games, paired, season/week-block bootstrap, accuracy, Brier, log loss v market-only, separately for v1 (W7, W9-10) and v2 (W11-18), and pooled only as a stated third read; record through `weak-signals record` (family mod23_unit10_draft_capital, forward look). No interim performance reads.

ROADMAP MOD-23 update: unit-10 arm B draft-capital forward log re-enrolled as frozen v2 on 2026-11-21 (feature-table rebuild broke the v1 refit); v1 W7/W9-10 kept as history, W8 and W11 Thursday lost, v2 covers W11 Sun/Mon through W18; scored after Week 18 per version.

## Open
- DATA GAP (measured): the latest snap_counts file (data/players/raw/20261018T131559Z) ends at season 2025 week 22, so every 2026 game currently uses each team's final 2025 game as "prior game" (latest_snap_order 202522; Week 7 rows show prior orders 202518-202522). Pregame-safe but stale; the 2026 snap feed must refresh for the arm to use 2026 starters. Until it does, the log is mostly the 2025 end-of-season starters; flag to the orchestrator.
- Zero Week 7 flips is a consequence of the above plus the small coefficient on two columns; not read as performance.
- Served-vs-same-pipeline base mismatch (see armb lane) applies here too.
- Confirm the v1-to-v2 drift cause by diffing 2013-25 rows against a pre-11-17 copy of game_features_weak_stack.parquet (E: mirror); the protocol change is not yet noted in docs/lanes/mod23-unit10b-draft-capital-v2.md (root's file).
- The snap-file Open above is resolved: runs now use data/players/forward_snaps (latest_snap_order 202610 at W11).
