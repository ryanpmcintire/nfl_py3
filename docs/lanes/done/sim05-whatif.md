Goal
SIM-05 (docs/sim04_play_simulator_plan.md unit 9): build scripts/sim05_whatif.py
with two what-ifs on 2009-2025 REG tables — (1) 4th-down aggression swap,
(2) starting-QB-out offense rating drop — 20,000 sims/arm, shared seeds,
report.json under artifacts/sim05_whatif/<UTC>/. Research/dashboard artifact
only; never re-picks a card game.

State
Engine scripts/sim04_engine.py extended (additive only, signatures unchanged):
PLAY_TYPE_CODES map added; build_transition_frame now emits play_type_code;
build_tables arrays dict gained down_i, dist_raw, fp_raw, play_type_code;
run_one_game's drawn dict gained a play_type_code key. Verified with a 20-game
smoke sim (mean margin -1.1, ok) plus the real engine's existing validation
path untouched (no existing keys removed).

scripts/sim05_whatif.py written: fourth-down policy replaces a punt/FG draw
with a real go-for-it outcome (pool of 4,764 historical rows: down 4, dist<=3,
yardline_100<=50, play_type run/pass) when outside the final 2 minutes of a
half; QB-out derivation uses data/pbp/raw/20260925T202544Z 2015-2025 REG,
passer_player_id, backup-led vs primary-led games per team-season, weighted
bootstrap (2000 reps) over 167 team-seasons. Both what-ifs simulate an
average-vs-average matchup (league_off_mean/league_def_mean from conditioned
build_tables) using shared-seed baseline/whatif pairs (common random numbers).

Smoke test at --n-games 50 passed end to end
(artifacts/sim05_whatif/20260926T034130Z/report.json): QB drop measured
-0.0532 EPA/play, CI [-0.0688, -0.0373], n=167 team-seasons, 55,332 backup
plays vs 115,527 primary plays. 4th-down whatif at n=50 (noisy, small-n only):
mean margin shift +5.08 (SE 2.93), home win prob shift +0.08.

First background attempt (bash id b7281n0kk) used `nohup ... & ; echo pid` and
was a mistake: the tool's own run_in_background tracked only the wrapper
shell, which returns almost instantly, so the "completed" notification fired
long before the real python process finished. The detached python process
(pid 28383) was still alive and running when checked via `ps aux`, alongside
a stray earlier process (pid 28282, likely the n=50 smoke test's interpreter
not yet exited). Both killed with `kill -9` to avoid concurrent writers.
Relaunched correctly as a single plain foreground command with
`run_in_background: true` on the Bash tool call itself (no nohup, no trailing
`&`) — background id bjun9m5p0 — so the eventual notification corresponds to
the actual script's exit. Command: `F:/Repos/nfl_py3/.venv/Scripts/python
scripts/sim05_whatif.py --n-games 20000`, started ~2026-09-26T03:5x UTC.
Timing basis: 50 conditioned games took 1.24s (~40 games/sec) => 20,000
games/arm x 4 arms (fourth-down base+whatif, QB-out base+whatif) ~= 33 min
total, plus one build_tables call (~5s) reused across arms.

Tried
Confirmed data/pbp/raw/20260925T202544Z has season=2009..2025 dirs and the
needed columns (game_id, season, week, posteam, play_type, epa,
passer_player_id). Confirmed game_features_pbp.parquet covers seasons
2009-2026 (needed for condition_on_team=True ratings). Considered basing the
QB comparison on a simpler per-game EPA average across all offensive plays
(run/pass, epa notna) rather than every play type, to avoid punt/FG-skewed
EPA; documented as an inferred modeling choice in the report's qb_drop note.

DONE. Full run completed: `artifacts/sim05_whatif/20260926T034314Z/report.json`
(n_games_per_arm=20000, exit code 0). Results (measured, this session):

Fourth-down aggression: baseline mean margin +1.66 (SE 0.097), whatif +1.45
(SE 0.099); shift -0.21 points (SE 0.139, ~1.5 SE from zero,
unresolved_below_power, not a rejection per AGENTS.md). Home win prob 0.540 ->
0.533 (shift -0.008). Line shifts (cover prob) all under 1 percentage point at
-3/-7/+3. Key-number mass shifted at 14 (+0.0086, SE 0.0020, largest shift)
and 3 (-0.0034, SE 0.0030); 4,764 historical go-for-it rows fed the
replacement pool.

Starting-QB-out: derivation from 2015-2025 REG pbp
(data/pbp/raw/20260925T202544Z), 167 team-seasons, 55,332 backup-led plays vs
115,527 primary-led plays: mean EPA/play drop -0.0532 (90% CI [-0.0688,
-0.0373], weighted bootstrap n=2000 over team-seasons). Simulated with home
offense lowered by that drop vs a league-average matchup: baseline mean
margin +1.56 (SE 0.096), whatif +0.32 (SE 0.096); shift -1.24 points (SE
0.136, clearly nonzero). Home win prob 0.533 -> 0.497 (shift -0.037). Cover
prob at -3 drops 0.425->0.387, at -7 drops 0.302->0.270, at +3 drops
0.605->0.571 (each shift roughly -3 to -4 points, several SE from zero at
n=20000).

Next (for whoever picks this up)
1. Nothing further authorized in this lane's scope (no commit/push/publish;
   subagent task). If a future task wants this fed into the registry, run it
   through `nfl-ats weak-signals record` per AGENTS.md before any write-up
   calls either result settled — the fourth-down shift is
   unresolved_below_power at this n, not closed.
2. If someone wants tighter fourth-down SEs, increase --n-games (script
   accepts it) and/or narrow the go-for-it pool by yardline sub-band.

Open
None blocking. The two prior artifact dirs
(20260926T034130Z n=50 smoke test, 20260926T034158Z empty/crashed duplicate
from the nohup double-backgrounding mistake) are harmless leftovers, safe to
ignore or delete.
