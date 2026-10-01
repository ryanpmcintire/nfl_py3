# acc-u4-forward-log

## Goal
Log the MOD-23 unit-5a man/zone arm's pregame picks weekly beside the served model, 2026 Weeks 4-18, frozen and untuned; no scores until Week 18 is final. Parent: docs/lanes/accuracy-ceiling-theory.md.

## State
Built `scripts/mod24_forward_log.py` (standalone; `prospective-record` needs a registered margin-predict artifact built from src features, which cannot carry the man/zone terms without src edits). Reuses src read-only: `score_outcome_week` with the arm columns patched in, served home-side offsets, discrete push reader, key-line read (same path as margin-predict), and `pick_deadline`/`sunday_pick_lock`.
Week 4 logged at 2026-10-01 17:35 UTC: all 16 games, one file `artifacts/mod24_forward/runs/2026-week-04-20261001T173557Z.json` (PIT_CLE first, kickoff 00:15 UTC Oct 2). Arm and served sides agree on all 16; arm probabilities within about 0.03 of served except KC_LV (0.545 vs 0.501).
`--status` prints coverage by week only.

## Tried
- Line: features parquet `spread_line` equals the served forecast line (post_lock basis).
- Rows record game_id, kickoff, deadline, recorded_at_utc, line, arm probability and side, served probability, side, pool side, terms, script and feature-table sha256, snapshot ids.
- Refuses outcome-present and past-deadline games (also rechecked at write time). Files opened with mode x (immutable).

## Next
Schedule `.tools\uv.exe run --no-sync python scripts\mod24_forward_log.py` Thursday ~19:00 ET and Sunday ~11:00 ET, Weeks 4-18 (default week = next week with an upcoming kickoff; do not run Tuesday before the noon line freeze). Canonical row per game = latest row before its deadline (declared now, before outcomes). Score only after Week 18 final, paired with served, week-blocked.

## Open
- Participation snapshot ends 2025 (latest 20260813T131635Z). Measured 2026-10-01: nflverse pbp_participation_2026.parquet returns HTTP 404 (pbp_participation_2025.parquet is 200, 45184 rows, 285 games). ftn_charting_2026.parquet exists (8065 rows, 48 games, Weeks 1-3 only) but has no defense_man_zone_type or defense_coverage_type columns (29 columns, plan-level flags only), so it cannot feed the arm. Inferred: until the 2026 participation release appears, the forward arm is base plus 2025 coverage tendencies; no refresh path built. Re-check the 404 URL before each run; if it turns 200, add an opt-in snapshot refresh (fixed recipe, snapshot id plus sha256 per run, as in docs/independent_combination_validation.md).
- Home-side offsets and discrete reader come from the served base-model fit, not the arm's own stream (small inferred difference on big spreads).
- Week 4 rows were logged Thursday, before the Sunday refresh; later runs add rows, never overwrite.
