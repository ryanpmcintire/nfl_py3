# mod25-v3-sim11-distill

## Goal
MOD-25 distillation v3: same student/transfer as v2 (scripts/mod25_pipeline.py features, student2, diag, grade) but trained on SIM-11 worlds (scripts/sim10_drive.py locked default engine, artifacts/sim11/strength_terms.json) instead of the October generator.

## State
Adapter built: scripts/mod25_v3_produce.py (imports sim10_drive, mod25_generator, mod25_pipeline read-only; patches pipeline SYN/OUT). Subcommands: produce, truth, features, student2, diag, grade --arm. Global flags --syn/--out go BEFORE the subcommand. Defaults: data/processed/synthetic/sim11_v3, artifacts/mod25_pipeline_v3.
Chunk format = v2 (games, team_stats, plays, meta with engine_sha and settings) plus latents.parquet (lat_d, dxo, dxd per game). True expected margin: truth stage fits margin on (1, dxo, dxd) over all chunks (SIM-11 has no verify step), writes truth.parquet like v2.
Smoke (measured 2026-10-27; two one-world 2-season chunks, workers 1; data/processed/synthetic/sim11_v3_smoke, artifacts/mod25_pipeline_v3_smoke): 544 games per chunk, 1.04-1.22 min each (about 8 games/s/worker). produce, truth, features, student2 run end to end. True expected margin (fit on dxo, dxd): hfa 2.51, SD 8.85 (v2 crj 7.97, hfa 1.70). Synthetic line SD 5.4 (real 6.0), line mean 2.5 vs result mean 2.9; result SD 16.0 vs real 14.8. Student numbers from this smoke (one holdout world, 544 training rows) are MEANINGLESS.

Full v3 run (same size as v2: 8 chunks x 6 worlds x 10 seasons = 130,560 games). Start only after the SIM-11 agent stops editing sim10_drive.py and mod25_generator.py (engine_sha is in each meta). Env OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1, BelowNormal, background, one job at a time, prefix `.tools\uv.exe run --no-sync python scripts\mod25_v3_produce.py`:
1. `produce --workers 6 --chunks 8`
2. `truth`
3. `features --workers 6`
4. `student2`, `diag`, `grade --arm ridge_nomkt`, `grade --arm hgb_nomkt`
Outputs data/processed/synthetic/sim11_v3, artifacts/mod25_pipeline_v3. Runtime (inferred from 8 games/s/worker): produce about 45 min at 6 workers (25-30 at 11 if CPU free); features 5-10 min; truth under 1 min; student2 and each grade a few min. The v2 verify step is replaced by the truth regression; linearity of E[margin] in (dxo, dxd) is unchecked.

## Tried
Smoke only.

## Next
Orchestrator launches full v3 after the SIM-11 agent finishes (see commands below), then student2, diag, grade --arm ridge_nomkt and hgb_nomkt; record via weak-signals record under family mod25_distillation_v3.

## Open
Teams per world, schedule: random (make_schedule), season-to-real-season map is SIM-11 default. Engine files must be frozen before the full run (engine_sha in each meta). 
