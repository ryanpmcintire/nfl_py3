# SIM-22

## Goal
Plays-per-drive logging in the resampler: draw plays with seconds from matched real drives so the split is observed (ROADMAP SIM-22).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: SIM-21 could only infer plays from seconds (71% of the trailing gap by a log-log slope); the resampler logs no plays. Measured mechanism: carry each matched real drive's play count and seconds as a joint draw. Data: SIM-19 and SIM-21 artifacts, drive pool (local).

## Tried
Novelty check only. Nearest: SIM-21 (inferred split), SIM-10 (resampler), SIM-18. Differs: observes rather than infers.

## Next
First unit for one agent: Add logging of matched play count with flag-off hash equal to HEAD, run a small seed set, and compare plays per drive to real by sign of score difference.

## Open
Rules: Measured mechanisms only; no tuned constant, offset or boost; outputs are checks; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; flag-off hash must equal HEAD before any flag-on run. Look count and family declared before any sign is seen.

## 2026-11-11 11:02 ET first unit (sub-agent)
- measured: the engine keeps per candidate drive (segments.parquet 53769 rows + csr.npz): elapsed seconds el, plays (diff pl_ptr; run/pass only), points (ev), start score diff sd, yards-to-go yl. The run output (plays.parquet, games.parquet) holds no drive id, seconds or score state, so plays per drawn drive are not recoverable from dflt_s11. The Nov 3 SIM10_INSTR log (artifacts/sim10/instrument/instr_s11, per drive seconds, points, side, state) plus instr_s11_s11 plays gives plays only in 1968 of 16384 games (run count equals logged drives; others carry extra logged drives with no plays), a biased subset.
- measured (scripts/sim22_drive_length.py, artifacts/sim22/drive_length.json, real 2009-17 regulation complete drives, game-cluster bootstrap): secs per play trailing real 26.99 [26.88,27.11] vs sim 28.02 [27.88,28.15] (world sd .15, 3 worlds); leading real 30.60 [30.47,30.74] vs sim 29.52 [29.38,29.66]; tied real 30.64 vs sim 30.21. Scoring rate (all 16384 games): trailing real .335 [.329,.342] vs sim .328 [.325,.330]; leading .378 [.370,.385] vs .367 [.364,.369]. Plays per drive (matched subset, biased): trailing 5.78 vs 5.62; leading 5.45 vs 5.64. Single seed 11; look count 12 cells (3 states x 4 metrics).
- Logging patch ready: tests/scratch/sim_finish_log/sim22-drplays.patch (env SIM10_DRPLAYS plus SIM10_INSTR; records hf,pos,j,secs,margin at start,plays,kept plays,el,trunc; no draw or float touched). git apply --check OK on the real file and on top of sim14-sim18-combined.patch. Not applied.
- Verdict: logging patch ready; needs one instr run (seed 11, ~20 min, after the three-seed run). No weak-signals cell recorded (no effect with an interval).

## 2026-11-11 14:21 second unit: instrumented seed 11 read (sub-agent)
- read (scripts/sim10_drive.py 1576-1640, 1760-1770): per game jsonl keys id, season, m300, m120 (margin with 300 s / 120 s left), last, dr, dp. dr row = [hf, pos, yards-to-go, secs left in half at start, points for possession team, points against, elapsed seconds (min of drawn and secs left), trunc]. dp row (same length and order) = [hf, pos, drawn candidate index j, secs left at start, margin at start from possession team view, plays in drawn drive, plays kept, drawn elapsed, trunc]. hf 1,2 halves, 3 overtime.
- measured (scripts/sim22_drive_play_decomposition.py, artifacts/sim22/drive_play_decomposition.json; real 2009-17 segments.parquet, bad=0, he=0; sim non-truncated regulation drives 352,132 v real 48,452; game-cluster bootstrap 1000, seed 22, 95%): secs per play sim minus real, whole game: trail 9+ +1.30 [1.09,1.50]; trail 1-8 +0.12 [-0.04,0.28]; tied -0.55 [-0.70,-0.39]; lead 1-8 -1.22 [-1.39,-1.05]; lead 9+ -1.82 [-2.05,-1.60]; trailing +0.66 [.53,.78] (27.65 v 26.99); leading -1.46 [-1.60,-1.32] (29.14 v 30.60). Plays per drive: trailing 5.52 v 5.78 (-0.26 [-.31,-.21]); leading 5.53 v 5.45 (+0.08 [.03,.13]). Secs per drive: trailing -3.43 [-4.97,-1.96]; leading -5.58 [-7.38,-3.96].
- measured decomposition (log secs per play = log secs per drive minus log plays per drive): trailing +.024 = secs per drive -.022 [-.032,-.013] plus plays per drive +.046 [.038,.055] (too few plays per drive; seconds per drive slightly short); leading -.049 = secs per drive -.034 [-.045,-.024] (70%) plus plays per drive -.015 [-.024,-.006] (30%).
- measured largest cells (secs per play sim minus real): lead 9+ last 5 min -5.65 [-6.71,-4.68] (19.80 v 25.45; plays per drive 4.43 v 3.65); lead 1-8 last 5 -4.07 [-4.70,-3.36]; lead 9+ fourth quarter -3.19 [-3.56,-2.78] (31.52 v 34.71). Trailing: trail 9+ fourth quarter +2.60 [2.27,2.92], plays per drive 5.68 v 6.31, seconds per drive equal (-0.04).
- Looks: 60 descriptive cells (20 state x band cells x 3 quantities, plus 3 aggregate states, whole game and 4 bands per aggregate, not counted as independent). One seed, one world; real is 2,304 games.
- Recorded: family sim22_drive_play_decomposition, 8 elapsed_seconds_bias cells (five whole-game states plus three largest), unresolved_below_power; record exit 0; pool --effect-units accuracy_points exit 0. Plays-per-drive differences have no admissible unit and stay here.
- inferred: simulator drives are drawn whole, so play count and seconds are carried together; the shortfall in late tempo is a joint draw problem: late trailing real drives pack more plays in equal seconds (hurry-up), late leading real drives carry fewer plays and burn clock per play (4th-quarter lead 9+ real 34.7 s a play). The lead 1-8 gap is 83% seconds per drive, the quantity the lead-clock arm scales; the lead 9+ and last-five gaps carry a plays-per-drive part the lead-clock arm (seconds only) cannot reach. Not yet tested against the restricted-pool arm.

## Next (SIM-22)
Check, from existing artifacts of the restricted-pool and lead-clock arms, whether their drives move plays per drive toward real in the same cells (no new run needed if their dp logs exist); a hook is warranted only if a named mechanism (late-game clock management by state) shows in the real joint of plays and seconds by state, not by tuning a constant. No hook built.
