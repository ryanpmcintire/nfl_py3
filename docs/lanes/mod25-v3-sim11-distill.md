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

## Result 2026-10-27 (measured, artifacts/mod25_pipeline_v3/, run.log)
Full run: 8 chunks x 16,320 = 130,560 SIM-11 games (hecarry 1, latent 2009-17), about 5 min per chunk at 11 workers. Grades against base (802-701 2020-25; 1142-1089 2011-19 SBR open), season-block bootstrap:
- ridge_nomkt: true 801-702, -0.07 pts [-1.02, 1.11] P+ 0.40, recal log loss flat (P+ 0.52); proxy 1170-1061, +1.26 [-0.27, 3.02] P+ 0.93, recal log loss +0.00133 [0.00019, 0.00242] P+ 0.989 (8/9 seasons).
- hgb_nomkt: true 798-705, -0.27 P+ 0.36; proxy 1171-1060, +1.30 P+ 0.94, log loss +0.00128 [0.00017, 0.0024] P+ 0.990.
- Fitted distillation weight falls with training length: 0.70 (last train 2010) to 0.22 (2025).
v2 for comparison: true +0.20 P+ 0.58, proxy +0.63 P+ 0.70. Recorded family mod25_distillation_v3 (5 cells, unresolved_below_power). Not served (true era flat).
Next: rerun after the era-aligned latent (rsdef) if the sim screens adopt it; the generator was fitted on 2009-17 while 2020-25 is the scored era.

## v4 result and correction 2026-10-28 (measured)
v4 = adopted SIM-11 default (era latent 2009-25, q4dec, hecarry), 130,560 games, artifacts/mod25_pipeline_v4/. True 2020-25: ridge 795-708 (-0.47, P+ .26), hgb 799-704 (-0.20, P+ .32), log loss flat. Proxy 2011-19: ridge 1179-1052 (+1.66 [0.18, 3.35], P+ .987), hgb +1.61 (P+ .990).
CORRECTION (read): the generator's drive library is real 2009-17 drives (scripts/sim10_drive.py SEASONS, line 18), which contains the 2011-17 test seasons. The proxy-era gains of v3 and v4 are chronologically contaminated and do not count; registry cells re-annotated. The clean test is 2020-25 (library before the test era), where neither v3 (-0.07) nor v4 (-0.47) helps. v4's latent variance was fitted on 2009-25 (global parameters overlapping the test era; mild).
Next: walk-forward library. For each test season T in 2020-25, build the drive library and latent fit from seasons < T only (extend the library past 2017), produce a smaller synthetic set per T, train, and score season T. That is the only admissible synthetic-data test of the current era.

## Walk-forward tooling 2026-10-28 (measured)
Env-parameterised windows: sim10_drive.py:17-20 (SIM10_OUT, SIM10_SEASONS), sim11_strength.py:17 (SIM11_OUT), mod25_v3_produce.py:20-27 (MOD25_STRENGTH/BANDWIDTHS/LATENT/Q4DEC). New scripts/mod25_wf.py (run, join, grade). State per T in artifacts/sim10_wf/T<T> (library 2009..T-1: build, outfit, patfit, latfit, strfit, q4dec; hgb_params and bandwidths_week copied from the 2009-17 fit, which is before every T). State build for T=2021 about 160 s (outfit 135 s). Smoke T=2021 (2 worlds x 3 seasons x 1 chunk, workers 4) passed end to end; pipeline_smoke is a smoke only. Reproduction: gates of the unmodified HEAD default differ run to run (max diff 2.5 between HEAD and edited, 15-17 vs old dflt/rsq4fg), so max diff 0.0 is not reproducible even before the edit.
Next: owner launches `mod25_wf.py run` then join, grade.

## Walk-forward result 2026-10-28 (measured, artifacts/sim10_wf/pipeline/, full_run.log; 1:18-4:12 AM)
Per test season T 2020-25: library and all fits from seasons < T, 4 chunks (65k games) per T. Clean 2020-25: ridge 794-709 (-0.53 [-1.84, 0.84], P+ .20; by season -2.27, -1.27, -0.81, -2.26, +0.75, +2.25), hgb 802-701 (0.00, P+ .47; 2025 +3.75, 2023 -3.38); log loss flat. Proxy 2011-19 rows still use the T2020 state (library includes them): inadmissible. Recorded family mod25_distillation_walkforward (2 cells). Not served.
Next (optional): T2026 state (library 2009-2025) scored on 2026 Weeks 1-7 at the pool lines as a clean forward read; the latest-library seasons (2024, 2025) were the positive ones.

## T2026 forward read, in progress 2026-10-28
Running (bg task b7v0k8spm, log artifacts/sim10_wf/t2026_run.log): `mod25_wf.py run --ts 2026 --chunks 4 --workers 6` (state from 2009-2025, produce, truth, features, student2; output artifacts/sim10_wf/pipeline/T2026/real_distilled.parquet). Written, not yet run: scripts/mod25_wf_forward.py (joins per-season distilled frames, freezes w/qbar/bbar from the week-1 fit through 2025, scores 2026 W1-7 at pool and Tuesday-open lines, writes artifacts/sim10_wf/forward_2026/). Next: when the run ends, `uv run --no-sync python scripts/mod25_wf_forward.py`, then weak-signals record (family mod25_distillation_walkforward, names mod25_wf_{ridge,hgb}_forward_2026_w01_w07, --classification unresolved_below_power).

## T2026 forward read result 2026-10-28 (measured; artifacts/sim10_wf/forward_2026/report.json, script scripts/mod25_wf_forward.py)
T2026 state 2009-2025, 4 chunks. Weight frozen from fits through 2025 (ridge w .261, hgb .266). 2026 W1-7, 107 games at pool lines (table equals splash exactly): base 64-43; ridge 63-44 (flips 13, arm 6-7, -0.93 pts, P+ .34, log loss gain -0.0018 P+ .37); hgb 63-44 (flips 15, 7-8, -0.93, P+ .35, log loss +0.00003 P+ .50). Tuesday open (101 games with open quote and no push): ridge 58-43 = base, hgb 57-44 vs 58-43, log loss worse (P+ .13/.22). Both cells recorded unresolved_below_power. Not served.
- Orchestrator check 2026-10-28 (measured): the forward read's base (64-43) used the current refit, which includes 2026 Weeks 1-7 results; the model-alone picks recorded before each kickoff (paper ledger model_pick_side) went 59-48. Both forward cells re-annotated as in-sample. A clean forward read must use the per-week locked predictions (artifacts/margin_predictions/2026-week-NN-* created before each lock) plus the student term built only from seasons <= 2025.
