# Pick-chain audit 2026

## Goal
Owner (2026-10-25): the card's picks are degraded; find every failing layer of the
pick chain, not one decision point, and fix the served side before Week 8 lines
freeze (Tuesday noon ET).

## State
Measured on 93 finished 2026 games (Weeks 1-6), graded at the Splash opener
(`data/splash/2026_week*.json`), card = last publication before each deadline in
`artifacts/clv_ledger/published_picks.parquet`, model = latest pre-deadline
`margin_predictions` market_residual row. Scratch tables:
`C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/ff94262b-dd63-435b-9f95-6908112a318b/scratchpad/`
`games2026_edge.csv` (per game) and `decomp.csv` (stack terms per game, recomputed
with production `signed_composition_flags` and `market_move_toward_home`).

- Card 52-41; W1-2 23-9; W3-6 29-32. Model final read 52-41; W3-6 34-27.
- Card overrode the model on 34 games: 17-17 (W3-6 8-13).
- Market move (leader median, open to Sunday prekick) toward the model's picks:
  mean 0.00 pts, 26 toward / 22 against. Base model shows no line value.
- Following the move alone on the 48 moved games: 29-19. Card on them 25-23;
  card against the move 3-7 (W3-6 0-5).
- Flags pooled 22-25. Coach 11-20 (W3-6 5-14; by season 2023 21-12, 2024 25-18,
  2025 21-20). Protection 18-8. Bye 0-2. Cold visitor 0-1.
- Fit on 2020-2025, scored on 2026: served 52-41 ll 0.6788; model+move 55-38
  ll 0.6725; per-flag 53-40; model only 52-41; served minus coach 62-31 (post-hoc,
  chosen after seeing coach lose). Five specs = five looks.
- Week 7 remaining (lock 20:00 UTC 10-25): GB@DET card GB is against a +1.0 move
  toward DET (bye flag drives it). LV, SEA, DAL, ARI unaffected by any finding.
  First decision (GB stays) superseded once the gate below was forward-tested and
  shipped: the gated fit serves DET 55.4%; ARI, LV, SEA, DAL unchanged.
- SHIPPED (uncommitted): post-discovery flag gate. `pick_probability.py`
  MEMBER_DISCOVERY_DATES (first commit date per overlay module, git
  --diff-filter=A) and model.counted_flag_columns; `pick_probability_fit.py`
  post_discovery_flag_records grades each flag at the Splash opener on games after
  discovery and counts only winning ones. Activated 2026-10-25T18:16Z as
  `pick_probability/20261025T181629Z` (counted: protection; coefs model .309,
  protection .319, move .221; LOSO 833-670). Previous pointer backed up in the
  scratchpad as active_pick_probability.before.json. 294 related tests pass; ruff,
  mypy clean. weak-signals recorded: coach_fade_post_discovery_2026,
  protection_mismatch_post_discovery_2026, post_discovery_flag_gate_forward_2026.
- Model audit (report_model.md): line value 2020-25 about +0.10 to +0.14 pts
  toward the model's pick; 2026 0.00 within noise. Injury and QB groups carry the
  move but not the residual in 2024-26.
- Surface audit (report_surfaces.md): History counts last pre-deadline side;
  paper `played` arm keeps the Tuesday side. W1-W5 flips (18 PICK-CHANGE before
  2026-10-06) were never notified; every flip since 10-11 has NOTIFY-OK.

- Stack audit (verified by rerun, `scratchpad/stack_audit.py`, report_stack.md):
  every flag and the move term were first committed 2026-08-18..09-05, so all
  2020-2025 records are in-sample to their discovery; 2026 is the only clean data.
  Post-discovery: coach 11-20 pp .055; protection 18-8 pp .974; move 29-19 pp .924;
  bye 0-2; cold 0-1; division, arrests, tank 0 fires. Forward-chained W2-6
  (77 games, 5 looks): served 43-34 ll .6776; model+move 49-28 dLL -.0084
  [-.031,.014] P .77; per-flag 43-34; model only 42-35; flags qualified on
  post-discovery data before each week (protection only every week) 54-23 dLL
  -.0216 [-.039,-.004] P .99. Rest of season: division ~43 fires (12 in Week 18)
  with zero clean evidence; protection ~22; cold ~21; bye ~19; coach Week 8 only.

## Tried
- Stale in-season features: not stale (rolling stats update weekly).
- Line mismatch: published lines equal Splash lines on every game.

- LIVE 2026-10-25 14:25 ET: `--run-job refresh_sun_inactives_late` recorded GB@DET
  GB +3.5 -> DET -3.5 (55%), NOTIFY-OK id EoETMtcJBkgn, site republished d7d0b6d9.
  Board verified: locked NYJ, IND, HOU unchanged; open ARI 50.8, DET 55.4, LV 57.2,
  SEA 54.5, DAL 54.3. Model page carries the plain-English gate sentence
  (`pick_probability.rule_gate_sentence`, appended in `board_content.py`).

## Next
- Week 8: the Tuesday refit recomputes the gate from the latest results; check
  the counted list in the new pick_probability metadata.
- Grade Week 7 and later through the History page against the old chain.
- Base model: separate availability terms (injury, QB) stamped at the opener from
  slow team-quality terms; forward-chained 2026 refit (MOD-23 lane).

## Open
- Coach fade is in `FAIL_CLOSED_MEMBERS` (`four_overlay_composition.py:88`); a hold
  needs `OWNER_HELD_MEMBERS` plus a refit (held_members check, `pick_probability.py:507`).
