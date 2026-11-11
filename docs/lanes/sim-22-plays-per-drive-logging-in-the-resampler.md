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
