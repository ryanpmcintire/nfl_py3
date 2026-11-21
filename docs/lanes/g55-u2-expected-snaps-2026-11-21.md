# G55-U2 expected snap share (servable O2)

## Goal
Servable replacement for P2-L oracle O2 (realised snap share, scripts/mod57_p2l_availability_ceiling.py role_rows lines 99-116): each active player's EXPECTED offense+defense snap share from strictly earlier games plus that game's active list. Parent goal-55-backwards-2026-11-21.md; sibling G55-U1 builds the active list.
Not a look: reads no scores, margins or ATS results. Snap-share error is a build check only.

## Declared forecaster (2026-11-21 14:13, before any fit)
- Unit offense_pct + defense_pct (harness unit, line 116); snapshot data/players/raw/20261108T141611Z, REG only, ids via attach_snap_player_ids.
- Baseline b: per-player EWMA over his strictly earlier played games (state = a*share + (1-a)*state). Span n (a = 2/(n+1)) chosen leave-one-season-out over integers 1..34 (two seasons of team games, read from data), minimising training MAE.
- Redistribution: an inactive roster player's freed share moves to active same-role teammates (QB, RB, WR, TE, OL, DL, LB, DB, other; harness `_position_group` puts QB in skill) in proportion to b, capped at 1 with overflow refilled; equal split when no receiver has b > 0. Fraction lambda in [0,1] = LOSO weighted-median MAE minimiser.
- Active list is `--actives` (game_id, team, gsis_id[, active]); uncovered team-games fall back to the stand-in (in snap counts with snaps > 0).

## State
2026-11-21 14:19 DONE (v2). scripts/g55_u2_expected_snaps.py (ruff check + format clean); artifacts/g55_u2/expected_snaps.parquet (442,231 rows, 2013-25 REG, keys game_id/team/player_key unique, gsis_id null on 47 pfr-only rows; inactive rows carry expected 0) + manifest.json (folds, MAE table).
- measured v2: span 2 in all 13 folds (no grid edge), lambda 0.494-0.516. MAE 2020-25 active players with history (forecast / no-redistribution / span-8 / naive last game, n):
  all 0.1189 / 0.1211 / 0.1335 / 0.1253 (147,811); skill 0.1199 / 0.1228 / 0.1288 / 0.1313 (41,181); OL 0.1409 / 0.1464 / 0.1780 / 0.1404 (23,712); front 0.1112 / 0.1107 / 0.1172 / 0.1180 (43,850); secondary 0.1501 / 0.1532 / 0.1716 / 0.1566 (28,908); other 0.0085 / 0.0087 / 0.0107 / 0.0082 (10,160). No-history actives 0.2072 v zero 0.2102 (2,532).
- Active expected sum per team-game median 22.5 (11 + 11), 1st-99th pct 19.2-25.2.

## Swap into the P2-L harness (grading unit)
role_rows (line 70): pass `expected` = {(game_id, team, gsis_id): expected_share} from the parquet (gsis_id rows only). Lines 115-116: replace `r = real.get(pid, ...)` / `1.0 - (r[0] + r[1]) / prior` with `share = expected.get((str(game.game_id), team, pid), 0.0)` / `1.0 - share / prior`; keep prior (line 112). Leave 136-150 (post-game update). Line 117 sev1 is U1's swap; build both from one active list. Line 273 reuses the cached LEAK table if present: write to a new OUT.
- Inferred: redistribution only raises active shares, so sev2 = 1 for inactive (same as O1) and > 0 for an active player only when the span-2 expectation is below the span-8 prior (shrinking role). Using baseline_share as prior collapses servable O2 to O1.

## Tried
- v1 as declared (14:16): span 2, lambda 0.03, MAE all 0.1211 (no gain). Cause: freed mass counted every roster absentee's old share although teammates' baselines had absorbed older absences.
- v2 (declared before refit): freed mass = b x (1-a)^gap, gap = team games he has missed since his last played game.

## Next
- Grading unit: rerun P2-L with U1 actives + this O2 via the swap above (that run is a look; count it).

## Open
- Snapshot has no 2026 snaps; live use needs a newer player snapshot. Postseason not covered (harness REG only).
