# Card term recorder (2026-11-16)

## Goal
Unpinned side recorder: every run appends, for every game on the current card, the served logit decomposition (model logit; intercept, model, flags, move, move-available, hook contributions; home logit; served p; side; pick_probability artifact; sha256 of the input row) so card sides can be scored per input fix before the pinned ledger patch (docs/lanes/card-logit-store-2026-11-16.md) lands.

## State
- scripts/card_term_recorder.py (ruff format + check clean). Calls pick_refresh.plan_refresh, spying (runtime wrap, no src edit) on pick_refresh.apply_external_lines (input rows) and card_view.resolve_card_probabilities (served frame + fit). Terms reproduce the patch's _served_terms formula.
- Output: data/card_terms/<season>_week_<WW>/<run UTC>.parquet, one new file per run opened with "xb" (never rewrites); read the week with pd.read_parquet(dir). data/card_terms/.gitignore = `*`.
- Exit 1 (rows still written) when: side or p differs from plan_refresh's served game; contributions miss logit(p) by >1e-9; an open (eligible) game's card side (latest pick_revisions new_pick_side, else published side) differs from the served side.
- Measured 2026-11-16: `--season 2026 --week 10` and default (resolves Week 10) each wrote 14 rows, exit 0, 28 rows total; all 14 ineligible (locked), card agrees on 14/14. 2026_10_BUF_NYJ matches verify_patch exactly (contributions -0.0196/0.1357/0/0.1102/0.0250/-0.0218, home logit 0.2295, p 0.5571 HOME, pick_probability/20261116T171555Z).

## Tried
- Nothing failed.

## Next
1. Root adds scheduler jobs (argv `[UV, run, --no-sync, python, scripts/card_term_recorder.py]`, catch_up True) a few minutes after each card refresh job: refresh_wed, refresh_thu, refresh_sat, refresh_sun, every refresh_last_call_*, and refresh_{wed,thu,sat,sun}_inactives_* (scripts/capture_scheduler.py ~1780-1840, ~2300-2390). Plus one Tuesday run after the lock so the original card is recorded.

## Open
- Locked-game side disagreements are recorded (card_agrees False) but do not fail; only open games gate.
