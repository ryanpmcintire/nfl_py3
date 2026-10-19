# MOD-25e rebase queue onto GZIDX base

## Goal
Queued 3-seed hooks were built on FD4/FD5 bases. New base is FD5+GZIDX (label ...gx2faldrggi, seeds s11-13 exist in artifacts/mod25e3). Each kept hook compares candidate v that base.

## State (2026-10-19)
- Runner reads tests/scratch/e178/queue.txt (head -1, sed 1d); smoke_tk2 was running, untouched.
- New queue order: run_p2_tiny_gi, run_tr_tiny_gi, run_ldr3_tiny_gi (rule 7: one real-path run, --worlds 1 --seasons 1 --workers 1 --seed 31, none had one), then run_p2_3seed_gi, run_tr_3seed_gi, run_ldr3_3seed_gi.
- Candidate labels: ...faldrgp2gi, ...faldrgtrgi, ...fal3rggi (LDR=0 LDR3=1; label order from crH lines 377-548).
- Scripts: tests/scratch/e182/run_p2_{tiny,3seed}_gi.sh, e174/run_tr_*_gi.sh, e162/run_ldr3_*_gi.sh; tgt output tests/scratch/<dir>/tgt_3seed_<hook>_gi.txt.

## Tried / dropped
- run_cs_3seed (CTXSCR): dropped. Requires CTX=2 (CTX F), not in the base; CTX F alone widens persistence (SD +0.27/+0.51, xq_cov +4.8/+9.5, e177) and its comparator cx2 has no GZIDX seeds. Re-queue only after a CTX F 3-seed on the GZIDX base.
- run_tl_3seed (TILTL): dropped. Smoke FAILED declared bar (noise, Q4 slope; e159/e165); E175 shows tilt not the leader-excess carrier; its target (strength var toward 55.8) is already met by GZIDX (55.1 v real ~58). Unresolved_below_power, not a refutation; nothing in the 3-seed contradicts re-queueing later.

## Next
Read tiny logs for Traceback (artifacts/mod25e3/e182_p2_gi_tiny.log, e174_tr_gi_tiny.log, e162_ldr3_gi_tiny.log) before the 3-seeds start; if one fails, delete its 3seed line from queue.txt. Then read boots/tgt vs the GZIDX base.

## Open
Hooks not in the existing queue tail (smoke_tk2 onward in sim_queue.log) are unaffected.
