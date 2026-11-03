# Engine patch queue (2026-11-03)

## Goal
One verified combined patch for scripts/sim10_drive.py (SIM-08 spec A, SIM-06 Step 6, SIM-09 situational hooks, each default off) so post-lock heavy runs start without conflicts. Root applies it; this lane never edited scripts/sim10_drive.py.

## State
- Base = on-disk sim10_drive.py (OT rule + QLOG committed, instrumentation applied, uncommitted); copy at tests/scratch/engine_queue/s0.py (byte-equal to disk, LF).
- Patches (tests/scratch/): sim08-spec-a-engine.patch (s0->s1), sim06-step6-engine.patch (s1->s2), sim09-situational-engine.patch (s2->s3), combined engine-queue-2026-11-03.patch (s0->s3). Sequential `git apply --check`+apply of the three in a temp tree: all clean. Combined `git apply --check` on the real file: OK (measured). s3.py compiles; ruff E501 count 4 on s0 and s3 (no new errors).
- Textual conflicts: none in the hunks. Choices (no logic change): spec A tg override and SIM-09 tg_shift both sit before `tg0 = tg`; spec A first, then sit. Spec A hunks 2-3 touch the q4dec/fgdec guard lines and the init next to q4dec; SIM-06 and SIM-09 both insert after `self.hecarry = ...`; SIM-06 first, then sit. SIM-06 pieces: init, DR line, ny branch. SIM-09 pieces 1-7 as listed in its lane (sim_world `book`, cmd_sim env, `--sit`).
- Static read, all flags off (read): 4 lines changed, all others pure additions: fgdec/q4dec guards gain `self.term is None and` (term None => same truth value); DR line `adj * DR_MASK * 1.0` (identical float); `ny` branch same result (ny+d if nsame==1 else ny-d); koseason/stslope/sit/term None => no branch, no RNG draw (new rng.integers only under koseason), no reordering. cmd_sim sets SIM10_SITUATIONAL="" (falsy).
- Identity smoke (all flags off, seed 5, 1 world, 3 seasons, workers 1, realsched 1, label idq_a = disk copy, idq_b = patched): copies run_idq_a.py/run_idq_b.py in tests/scratch/engine_queue/ (REPO pinned to the repo path; otherwise identical to s0/s3). NOT RUN: it was queued as a deferred background job but killed (PID 34448) before 11:35 because the agent cannot wait without polling. Root launches `powershell -NoProfile -NonInteractive -File F:\Repos\nfl_py3\tests\scratch\engine_queue\run_identity.ps1` as a background call at or after 11:35 ET (waits to 11:35 itself, then runs idq_a then idq_b, one core, ~5-10 min each) and reads tests/scratch/engine_queue/identity_result.txt (exit codes plus SHA256 of artifacts/sim10/idq_{a,b}_s5/games.parquet).

## Tried
- First write produced CRLF copies (git apply failed at line 650); rewritten with LF.

## Next
Apply: `git -C F:\Repos\nfl_py3 apply F:\Repos\nfl_py3\tests\scratch\engine-queue-2026-11-03.patch` (check first with `--check`). Heavy order after the lock (one at a time; `$env:OMP_NUM_THREADS=1`; all via `F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim10_drive.py sim ...`):
1. Identity: read identity_result.txt; equal hashes required (else `git apply -R`).
2. instr_s11: `$env:SIM10_INSTR='instr_s11'` `sim --seed 11 --workers 3 --realsched 1 --label instr_s11`, ~20 min; then `scripts\sim08_instrument_analysis.py run instr_s11` (1-2 min).
3. SIM-06 Step 6 gate seed 11: `SIM10_QLOG=head_rep SIM10_DRLAM=0.25 SIM10_STSLOPE=1 SIM10_KOSEASON=1`, `sim --seed 11 --workers 3 --realsched 1 --head 1 --label head_rep`, ~20 min; then tests\scratch\sim06_startmix.py and scripts\sim06_query_log_analysis.py --label head_rep. Bar in the SIM-06 lane. If a gate is lost run each flag alone.
4. SIM-09 Tier B (seeds 11,12,13, workers 11, ~253 s each): dflt for seeds missing gates.json (`--label dflt`), then `--sit nohuddle --label sit_nohuddle` first, then urgency, fatigue, stakes, then all four (`--sit nohuddle,urgency,fatigue,stakes`; flag names from Situational). Refit if segments changed: `scripts\sim10_situational.py fit`. Compare gates.json to dflt per the adoption rule in the SIM-09 lane.
5. SIM-08 Amendment 4 declared run: `$env:SIM10_TERMINAL=<json>` `sim --seed 11 --workers 3 --realsched 1 --label a4_s11`, ~20 min. BLOCKED, see Open.

## Open
- Spec A cannot run yet (read): sim08_terminal_fit.py has no `terminal_cum`, `terminal_probs`, or OT lookup; the lane diff's OT branch (poss_idx, kneel/other mapping) is prose, not a diff, so the patch carries only init, guards and the regulation tg override. Also the fit writes one transitions_<season>.json per season, but SIM10_TERMINAL takes one file. Needs a unit writing those functions and the OT hunk.
- Regulation override sets td and fg only; kneel/turnover/punt mass unmapped (reported in spec A lane).
- Kickoff-season flag is keyed to `season`, valid only with --realsched 1.
