# acc-u7a-availability-history

## Goal
Rebuild rolling team inputs so past games played by a different QB (or OL depleted, or against a fill-in opponent QB) are excluded or down-weighted; grade as a base-model change. Parent: docs/lanes/acc-u7-new-information.md.

## State
Done 2026-10-01 (script scripts/mod24_u7a.py, outputs artifacts/mod24_u7a/, recorded under family mod24_availability_history, 3 cells).
Absence verified (read): the 90 weak_stack inputs are EWMA(span 8) of build_team_game_metrics (team_stats) via build_team_states/attach_team_states in src/nfl_ats/features.py; no starter, snap or opponent-depletion conditioning. Only the current-game diff_qb_*/injury columns know players.
Measured pooled accuracy diff vs base (season-blocked, 15 blocks): A1 exclude +0.161 [-0.46,+0.68] P+0.711, 1950-1784, 8/15 seasons+, flips 198-192; A2 down-weight (w by inner MSE, 0.25 to 0.75) +0.187 [-0.35,+0.73] P+0.739, 1951-1783, 8/15, flips 90-83; A3 +OL -0.107 [-1.01,+0.87] P+0.390, 1940-1794, 5/15, flips 229-233.
Eras: proxy 2011-19 A1 +0.672 P+0.9986 (6/9), A2 +0.359 P+0.83, A3 -0.045; true 2020-25 A1 -0.599 P+0.095 (2/6), A2 -0.067, A3 -0.200.
Recalibrated 2020-25 (u2b temp, gain = base minus arm): LL A1 -0.00007 P+0.42, A2 +0.00003 P+0.53, A3 +0.00012 P+0.63; Brier same sign order; RPS A1 -0.0013, A2 -0.0011, A3 +0.0006.
Rows affected: any flagged game in last 8 for 81% (2011-19) and 90% (2020-25) of team-sides in A1; A3 94%/100%.
All three recorded unresolved_below_power; no closing ground.

## Tried
A1/A2/A3 built per target game (expected starter = last starter, unless Out/Doubtful report evidenced by kickoff-24h: 156 overrides, effectively 2020+). Fill-in opponent QB = differs from both neighbors (next only if played before target). Weights apply to own-QB group (7 off metrics plus point_diff, ats_residual) and defense group (6). Window 48 games. Rebuild reproduces base columns exactly except ats_residual on 2026 rows.

## Next
Proxy-era signal (A1 +0.67, P+0.999 on 9 blocks) does not replicate in the Tuesday era (-0.60): look-count 3 arms plus 3-weight grid; treat as unresolved, a later unit could test a QB-only variant without the results group. Optional: u7b/u7c lanes.

## Open
A2 grade runs each w across all seasons and picks rows by selected w (home-side offset stream not strictly path-matched). 2011-12 OL flags unavailable (snap counts from 2013). u2b pick_identity tail errors (KeyError u3_compact_net); results.json/csv written before it.

## Orchestrator decision 2026-10-01
Stopped before grading on owner direction: it reuses data already on disk.
Work moves to acquiring NEW information (units 8-9). No registry cell.
