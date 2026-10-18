# Hook review 2026-10-18 (read-only)

## Goal
Correctness review of default-off MOD-25e hooks and crH install order.

## State
Reviewed gstop2, toh, v1ad, safeg, safegp, stbk, fd5, rgt, rg, ldr, ldr2, tiltl, servedidx, crH. tdr and ldr2 are NOT installed by crH (LDR2/TDR env ignored, label unchanged).

## Findings (ranked)
1. toh.py:171 dv.task_rng("toh", seed) -> int("toh") ValueError on first eligible late play. Breaks smoke_g2sth.
2. gstop2.py:10,34 global counter _STATE["n"] never reset per task: RNG depends on worker task order, not reproducible; paired boots noisy. Affects smoke_g2s, smoke_g2sth.
3. crH install: LDR (pick wrapper) and RG (Geo.fix repick weights) both correct late-lead class mix; double application in every LDR+RGR run (smoke_ldrg/g2s/g2sth/rq/sg/tk, run_ldrg, run_fe).
4. rg.py/rgt.py ratio applies only inside Geo.fix (violating draws), not all draws it was fit on (inferred).
5. safeg.py:44,53 loss.npy built from build_transition_frame(pbp) with no team_ratings; row alignment with engine arrays unverified (smoke_sg).
6. E162i LDR anomaly: arrays are not stale (neg runs before ldr); suspect downstream f2 tostep/cpick re-picks (sim09_f2.py:385) bypass LDR. Unresolved.
7. tiltl.install() and v1ad use "all" fit only (no fold); fine for post-2017 scoring, but TOHS/season fold never selectable from crH.
8. GSTOP and GSTOP2 not mutually exclusive in crH.

## Next
Fix 1-2 (use dv.task_rng with int salt) before running smoke_g2sth/g2s.

## Orchestrator actions 2026-10-18
- Fixed #1: mod25e_toh.py task_rng salt "toh" (int() crash) -> SALT 9901. Fixed #2: mod25e_gstop2.py process-global counter RNG -> per-task dv.task_rng(SALT 9903, cfg seed). Both compile; unused salts checked against all mod25e SALT constants.
- #4 false alarm: GEO=2 forces every run/pass through Geo.fix (E176d measured 97% exact-distance served), so RG/RGT apply to all Geo-served run/pass draws.
- #3 judged unlikely (LDR reweights play type in the pick; RG reweights outcome class in Geo.fix); keep watching late-lead cells in the LDR+RG 3-seed.
- #5 open: verify SAFEG loss.npy row alignment with engine arrays before running smoke_sg.sh.
- #6 forwarded to E162j (f2 tostep/cpick re-pick after LDR).

## #5 SAFEG alignment
Aligned (measured). build_transition_frame with and without team_ratings both give 346552 rows, identical game_id, play_id, down_i, fp_raw, yards_gained, points_def, next_yardline in order; loss.npy length 346552. Ratings merge is a left merge on game_id and runs before the sort, so no rows are dropped. No edit to mod25e_safeg.py; check not re-run.
