# SIM-06 learned drive-outcome head vs SIM-10 kernel (2026-11-02)

## Goal
First measured SIM-06 step: gradient-trained drive-outcome head vs the SIM-10 kernel resampler's
conditional outcome distribution, same drive features and outcome classes, trained 2018-2023,
scored on held-out 2024-2025 drives. Drive-level comparison only; no served-card effect, no engine swap.

## State
- scripts/sim06_learned_head.py (imports sim10_drive read-only; SIM10_SEASONS=2018-2025,
  SIM10_OUT=artifacts/sim06/sim10 built with `sim10_drive.py build`).
- Classes: SIM-10 outcome_classes recipe, tercile/decile cut points taken from training seasons only.
- Kernel A: locked bandwidths_week.json (fit on 2009-17). Kernel B: same kernel, bandwidths
  refit by season-block CV on 2018-23 (nested, no held-out data).
- Head: hyperparameters (MLP width/L2, spline-logit C) chosen by 3-block season CV on 2018-23 only.
- Outputs: artifacts/sim06/results.json, run.log. Results in the final report of this lane.

## Tried
Run done (370 s, 1 process). Held-out 2024-25 (11,443 drives, 543 games, K=94, train 34,429):
log loss/drive learned MLP(64,64) -3.468, logit -3.473, kernel B -3.526, kernel A -3.535, marginal
-4.045. Learned minus kernel B +0.0577 [0.0456,0.0701] P+=1.00 (game bootstrap); 8-season LOSO
+0.0560 [0.0495,0.0627] season-block, every season positive. Recorded:
sim06_learned_head_vs_kernel_heldout_2024_25 (log_loss_improvement, unresolved_below_power).
Calibration: head ECE .00063 v kernel A .00033; head group totals closer on type 7-9, kernel under on type 0/1.

## Next
If learned head wins: swap would need the clock-feasibility mask (kernel uses it, head cannot),
strength tilt (DR targets) re-fit on the head, and a gate rerun of the 14 sim10 gates.

## Open
Kernel clock-feasibility mask has no head analogue; held-out has only 2 seasons so the primary
interval is a game-cluster bootstrap, the season-block interval comes from 8-season LOSO.

## Step 2 (build, 2026-11-02)
`scripts/sim10_drive.py sim --head 1` (default 0, env SIM10_HEAD): Head class (head_codes/fit_head/Head.reweight) fits the
step-1 selected MLP (artifacts/sim06/results.json head_selected) on the engine's own segments (2009-17, ok rows, half<3),
cached in OUT/head_mlp.npz. Engine.draw reweights the kernel weights after the clock-feasibility mask (the `early` z=inf
line) so class mass equals the head's masked/renormalised class probabilities; then the unchanged SIM-11 DR/strength/joint
tilts run on those weights. OT drives stay on the kernel.

## Next (step 2 done, smoke only)
Smoke 1x1 seed 5 realsched (artifacts/sim10/h0orig_s5, h0new_s5 identical games.parquet = default unchanged; h1_s5 = head): head
run 41 s v 29 s; v default: margin_sd 15.71 v 14.83, pts_game 47.8 v 44.2, strength 44.6 v 31.9, R2 .069 v .021, q4 -.084 v -.033 (256 games, noise; sanity only).
Full one-seed screen (not run), workers 3:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/sim10_drive.py sim --seed 11 --workers 3 --realsched 1 --head 1 --label head
Compare gates v artifacts/sim10/rshc_s11 / current default same seed (--head 0 --label headctl).
Open: mass_10/mass_14 identical across default and head smoke runs (check gate_stats).

## Step 2 screen (launched 2026-11-02 23:32 local)
Seed 11 --head 1 --label head running in background (log artifacts/sim10_head_s11.log, out artifacts/sim10/head_s11 or similar by label). Compare to artifacts/sim10/rsjt_s11/gates.json (13/14; fails ot_share only; gate pass = sim inside real interval in table). Rule: if >=13/14 and loses none, run seeds 12-16 one at a time at workers 3, 6-seed mean vs rsjt; adopt only if all rsjt-passing gates pass and one improves without worsening beyond seed SD. Kill by PID at 11:30 ET Tuesday. Not yet compared.

## Step 2 result (seed 11, measured, 680 s)
--head 1 (artifacts/sim10/head_s11/gates.json) passes ~6 of the 14 table gates (sim inside real interval) v rsjt_s11 13/14. Loses margin_sd 15.46, q4_slope -0.111, mass_7 .0787, mass_14 .0432, pts_game 46.54, home_edge 2.77, noise_re_var 186.7, r2_w10_18 .110 (floor .1313). Gains ot_share (.0582). Declared rule fails: seeds 12-16 NOT run, default stays --head 0. Likely cause (inferred): head class probs (no tilt-aware calibration, higher scoring drives) shift scoring +1.7 pts/game. Not adopted; no code edited.

## Step 3 diagnosis (2026-11-03, scripts/sim06_head_diagnosis.py, 85 s, 1 core; artifacts/sim06/diag2.log)
LOSO by season 2009-17 (engine's regime), 900 random drives per held-out season (8,100), kernel A weights, head = selected MLP refit per fold.
Measured E[offense pts/drive]: obs 1.7605; kernel 1.7733 (+0.013); head 1.7916 (+0.031); head masked to kernel support as Head.reweight does 1.8005 vs kernel support 1.7698 (+0.031). Head log loss -3.4595 v kernel -3.5167 (head still wins per drive).
Head-minus-kernel over-score concentrates: first half +0.053/drive (head +0.051, kernel -0.002 v obs), yl 50-75 +0.13, leading by 1-14 +0.04..+0.06, last 600 s both over (+0.09 head, +0.08 kernel).
Mechanism (inferred from measured): the head is a pooled smooth surface with no clock-feasibility mask and no season/era or strength input; it over-scores real states by +0.031/drive ~ +0.7 pts/game (22.4 drives/game), about 40% of the +1.7 seen in the engine. Static bias on real states does NOT explain the other ~1.0 pt/game: unexplained as measured. Candidates untested: engine-visited states outside real support (no engine query log exists), interaction with the SIM-11 DR/joint tilts (calibrated on kernel weights, now applied to head weights).
Repair spec (next heavy run, no constants): (1) log engine draw queries (q) for head runs to a parquet and compute head-v-kernel E[pts] on those states (tests covariate shift); (2) fit head with a held-out-season-chosen season/era input and the clock-feasibility mask folded into the loss (train on kernel-masked class support, as Head.reweight uses it); (3) refit the DR tilt targets on head-reweighted weights, then rerun gates. Do not offset points.
Run later: .tools/uv.exe run --no-sync python -I scripts/sim06_head_diagnosis.py (diagnosis); engine: sim10_drive.py sim --seed 11 --workers 3 --realsched 1 --head 1 --label head2 after (1)-(3).

## Step 4 v2 refit (2026-11-03, scripts/sim06_head_v2.py, 540 s, 1 core; artifacts/sim06/v2.log, results_v2.json, head_v2.npz)
Protocol = diagnosis (LOSO 2009-17, same 900 qi/season, kernel A). v2 = numpy masked-softmax MLP 64-64 trained only on classes feasible at the drive's clock (class has a non-hurry-up drive in that half, or a hurry-up drive with secs >= this one); candidates masked / masked+season, chosen per outer fold by inner 3-block season CV (looks: 2 cands x 3 blocks x 9 folds). Chosen: masked 8/9 folds, masked_season only 2017.
Measured out-of-sample (per-drive ll, E[pts] minus obs 1.7605): kernel -3.5049, +0.0129; head v1 -3.4595, +0.0311 (kernel-masked +0.0400); v2 chosen -3.4600, +0.0393 (kmask +0.0370); v2 masked-only (fixed) -3.4616, +0.0219 (kmask +0.0200); masked+season -3.4595, +0.0258 (kmask +0.0234). ll v2 chosen minus v1 -0.0004 [-0.0088,0.0078] P+=.46 (9 season blocks); v2 chosen minus kernel +0.0449 [0.0322,0.0579] P+=1.00, 9/9 seasons.
In-sample (900 fit-season drives, obs 1.7780): v1 +0.0275 ll -3.4447; v2 chosen +0.0230 ll -3.3998; masked-only +0.0191 ll -3.4027. ll gap in-v-out: v1 0.015, v2 masked 0.059 (v2 overfits more).
Bias: season input is not supported by inner CV (8/9 folds reject it); the nested choice leaves the bias unrepaired (+0.037 kmask) because the 2017 fold picks season. Fixed masked-only halves the kmask bias (0.040 -> 0.020, ~0.45 pt/game of the 0.9), but fixing it is a post-hoc look, not the LOSO choice. In-sample bias also exists (+0.019..0.027), so the over-score is not an out-of-season shift. Unexplained as measured: remaining +0.020/drive (~0.45 pt/game) plus the ~1.0 pt/game engine gap (needs the query log).
Over-score bins (v2 chosen kmask v1 kmask minus obs): yl 0-25 .106 v .226, yl 50-75 .065 v .066 (unchanged), lead 1-7 .090 v .105, lead 7-14 .062 v .106, first half .043 v .054.
Engine change to load v2 (NOT applied; sim10_drive.py fenced). v2 npz has no "classes" (full K, masked at inference by the existing mass>0 mask, equal to a masked softmax); no season feature when use_season=0:
```
--- a/scripts/sim10_drive.py
+++ b/scripts/sim10_drive.py
@@ def head_path():
-    return OUT / "head_mlp.npz"
+    return REPO / "artifacts" / "sim06" / "head_v2.npz" if os.environ.get("SIM10_HEAD_V2") == "1" else OUT / "head_mlp.npz"
@@ class Head: __init__
-        if not head_path().exists() or int(np.load(head_path())["n"]) != len(seg):
+        if os.environ.get("SIM10_HEAD_V2") != "1" and (not head_path().exists() or int(np.load(head_path())["n"]) != len(seg)):
             fit_head(seg)
         z = np.load(head_path())
         self.k = int(z["k"])
-        self.classes = z["classes"]
+        self.classes = z["classes"] if "classes" in z.files else np.arange(self.k)
@@ nl = len(...)
-        nl = len([n for n in z.files if n.startswith("w")])
+        nl = len([n for n in z.files if n[0] == "w" and n[1:].isdigit()])
```
(the old nl line also counts none extra; keep the isdigit guard). Regenerate head_v2.npz: `.tools/uv.exe run --no-sync python -I scripts/sim06_head_v2.py final masked` (7 s).
Heavy gate (later, after SIM-08 releases sim10_drive.py): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SIM10_HEAD_V2=1 .tools/uv.exe run --no-sync python -I scripts/sim10_drive.py sim --seed 11 --workers 3 --realsched 1 --head 1 --label headv2`; compare to artifacts/sim10/rsjt_s11 and head_s11. Still owed: engine query log (repair step 1) and DR tilt refit on head weights (step 3).
Next: if v2 masked still over-scores in the engine, run step 1 before any more head work.

## Step 5 engine query log, prepared not applied (2026-11-03)
Patch: tests/scratch/sim06-query-log.patch (48 +/- lines, `git apply --check` passes on current sim10_drive.py). Env SIM10_QLOG=<label> turns it on (default off: Head.last stays None, no extra RNG draw, no state mutation; reweight arithmetic is unchanged, ph = proba(f) then masked). Per worker writes artifacts/sim06/query_log/<label>/<pid>_<n>.npz every 5000 queries and at the end of each sim_world (npz, not parquet, because K-wide float32 prob rows): f (8 head features), half, home, p_head (full), p_head_masked, p_kernel (clock-masked kernel class mass, pre-head), row (drawn segment), drawn_cls.
Byte-identical check (later): run the smoke `sim --seed 5 --workers 1 --realsched 1 --head 1 --label qa` without and with SIM10_QLOG and compare SHA256 of artifacts/sim10/<label>/games.parquet (and with --head 0 for the untouched kernel path).
Analysis: scripts/sim06_query_log_analysis.py (code path proven on an in-memory synthetic log only, nothing saved): E[pts] head v kernel by yl/sd/secs/half bin vs real-drive obs, head-minus-kernel at engine shares v real-drive cell shares, nearest-neighbour covariate-shift distance engine-to-real v real-to-real.
Later, after SIM-08 releases sim10_drive.py:
1. git apply tests/scratch/sim06-query-log.patch
2. $env:OMP_NUM_THREADS=1 (and MKL/OPENBLAS/NUMEXPR); $env:SIM10_QLOG="head_s11"; .tools/uv.exe run --no-sync python -I scripts/sim10_drive.py sim --seed 11 --workers 3 --realsched 1 --head 1 --label head_qlog  (about 12 min, 680 s head run plus npz writes; add SIM10_HEAD_V2=1 only with the v2 diff)
3. .tools/uv.exe run --no-sync python -I scripts/sim06_query_log_analysis.py --label head_s11  -> artifacts/sim06/query_log/head_s11_analysis.json (under 1 min)
Note: the log label dir and --label are independent; keep them equal for traceability. Reading: if head_minus_kernel at engine shares >> at real shares, covariate shift explains it; else tilt interaction (repair step 3).
