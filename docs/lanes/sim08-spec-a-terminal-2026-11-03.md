# SIM-08 spec A: fitted terminal-state table (2026-11-03)

## Goal
Replace the q4dec/fgdec logit offsets (scripts/sim10_drive.py:1186-1197, last-15-min regime x margin-bin shifts on td/fg) in the last 5 minutes of regulation and OT with a drive-outcome table fitted from real pbp, leave-one-season-out. Fitted only; engine untouched, no simulation run.

## State
- Module `scripts/sim08_terminal_fit.py` (ruff check and format clean). `fit` writes `artifacts/sim08_terminal/transitions_<season>.json` for 2009..2025 (table from all other seasons; REG only). `check` reruns a Monte Carlo over the table only and writes `artifacts/sim08_terminal/reproduction.json` and `check.log`. Drives cached in `artifacts/sim08_terminal/drives.parquet`.
- Table: per possession-start cell (60 s bin of the last 300 s x possessing-team margin clipped at +-17) the probabilities of td, fg, fg_miss, turnover, punt, kneel, def_score, other; conditional net-points distribution per (margin, outcome); elapsed-time pool per (bin, outcome). OT: per (possession index 0/1/2+, margin clipped +-8) the same, OT elapsed per outcome.
- Named constants: window 300 s (spec), 60 s bins, 17 = largest key number, OT length 900 s to 2016 and 600 s from 2017 and OT rule by season (2009-11 sudden death, 2012-24 modified, 2025 both possess; measured in sim08-simulator-rebuild.md OT section). Empty cells take the nearest populated margin in the same bin.
- Reproduction (held-out, 100 draws per game from the real state at the first drive starting <=300 s): see Result below.

## Tried
- Start state is the real first drive beginning at or below 300 s, so earlier scoring is real; elapsed is clamped to the clock left (a scoring drive at the clock edge keeps its points).

## Next
Integration diff below, to apply when sim10_drive.py is released (about 11:50 ET). Then the declared unit with instrumentation (margin at 300 s, last scoring type) and the gates.

## Open
- Drive-level table has no timeout or line-cell dimension (line cell was in spec; sample per cell too thin, report not filled).
- Elapsed clamp and the kickoff time are unmodelled; the OT tie share gap, if any, is reported in Result as measured.

## Integration diff (unapplied)
In `Engine.__init__` next to q4dec (line ~1084), behind env flag defaulting off:

    self.term = None
    if os.environ.get("SIM10_TERMINAL"):
        self.term = json.loads(Path(os.environ["SIM10_TERMINAL"]).read_text())
        self.term_cp = terminal_cum(self.term)

In the drive-outcome step (lines 1186-1197), after the fgdec/q4dec blocks and guarded so they are skipped when `self.term` is set:

    -if self.fgdec is not None and hf == 2 and FG_DEC_LO <= q[1] <= 0.0:
    +if self.term is None and self.fgdec is not None and hf == 2 and FG_DEC_LO <= q[1] <= 0.0:
    -if self.q4dec is not None and hf == 2 and secs <= Q4_SECS:
    +if self.term is None and self.q4dec is not None and hf == 2 and secs <= Q4_SECS:
    +if self.term is not None and hf == 2 and secs <= TERMINAL_SECS:
    +    pt = terminal_probs(self.term_cp, secs, q[1])
    +    tg = np.array(tg, dtype=np.float64)
    +    tg[1] = pt["td"]
    +    tg[2] = pt["fg"] + pt["fg_miss"]

with `TERMINAL_SECS = 300.0`, `terminal_cum` = `cum_probs` and `terminal_probs` = lookup of `reg_cp[bin_of(secs), margin_idx(q[1], 17)]` differenced back to per-outcome probabilities, both imported from scripts/sim08_terminal_fit.py. Kneel and `other` mass maps to the engine's clock-run-out branch (set tg for the kneel outcome where the engine has one; otherwise unmodelled and reported). OT branch: when the engine enters OT (sim10_drive.py:1332-1376) use `ot_cp[min(poss_idx,2), margin_idx(q[1],8)]` the same way, and fix the OT opening-score rule to the season rule (2025 both possess), which is already the lane's change.

## Gate (SIM-08 evaluation, declared unit)
- K3 effect interval includes 0 with sim mass at 3 >= real minus one SE (now sim .130 v real .151, -0.0205).
- OT tie share of OT games: real 6.1%, sim 15% at baseline; must move toward real, reported as measured.
- K7 (+.0096), K17 (-.0094), O (OT diff -0.0025) not worse; L1 not worse than -0.0136.
- SIM-11 14 gates pass; planted control re-run.

## Result (measured, `artifacts/sim08_terminal/reproduction.json`, 4292 reg games 2009-2025, each season from the table fitted without it)
Mass at final margin 3: real .1468, table .1430 (-0.4 pp). Paths (real v table): reg FG from tie 219 v 217.6; OT FG 163 v 152.8; loser TD trims 10-lead 88 v 75.6; winner TD from -4 85 v 95.8; other 75 v 71.8. OT tie share real 5.0% (13 of 260) v table 9.4% (baseline engine 15%): a gap remains, unexplained, reported as measured. Note the table starts from real scores at 300 s, so it tests the terminal 5 minutes only; it does not test the pre-300 s score distribution.

## OT tie-share unit (2026-11-03, in flight)
- Measured (drives.parquet, REG OT drives): OT games 09-11 45 (0 ties), 12-16 83 (5), 17-24 118 (7), 25 14 (1); possessions/game 2.78/2.60/2.59/2.21; start yardline_100 mean ~71 all eras; drive score share by start distance: <=40 .875, 40-60 .50, 60-80 .385, >80 .224; tied OT 4th down in FG range: 19 of 19 kick (no go-for-win gap).
- Named mechanism: game-ending OT scoring drives had elapsed = whole clock left (fg mean 688 s / td 775 s in 09-11; 604/775 in 12-16; fg 328 s 17-24 v 284 s non-ending), pooled into the elapsed pool, so simulated scoring drives burned the period and cut possessions, inflating ties.
- Fix applied in scripts/sim08_terminal_fit.py drives_of: OT last drive end time = its own last play start (lower bound by one play). Old drives.parquet and reproduction.json copies in scratchpad (repro_before.json, drives_before.parquet; before: OT tie sim 9.4% v real 5.0%, margin-3 mass .1430).
- In flight: background job (fit then check) rewriting artifacts/sim08_terminal/ ; log scratchpad fit.log, chk.log. Next: read artifacts/sim08_terminal/reproduction.json for after numbers; if drives.parquet is missing rerun fit then check.
- DONE (measured, reproduction.json): after fix OT tie sim 6.57% v real 5.0% (before 9.4%); OT fg path 156.5 v real 163 (before 152.8); reg FG from tie 218.9 v 219; loser trims 76.2 v 88; winner TD 95.4 v 85; other 74.8 v 75. Ruff clean. Residual 1.6 pp tie gap reported as measured.

## Instrumentation (prepared 2026-11-03, nothing run)
- Patch `tests/scratch/sim10-instrument.patch` (git apply --check passes on current sim10_drive.py; 10 hunks, +37 -0; env `SIM10_INSTR=<label>` default off, no RNG use). Logs per game to `artifacts/sim10/instrument/<label>/w<pid>.jsonl`: m300, m120, last two reg score events, every drive (hf,pos,start yl,secs,pts off/def,elapsed,trunc), OT opening draw, end score.
- Analysis `scripts/sim08_instrument_analysis.py` (ruff clean; `selftest` passes on in-memory synthetic data): OT possessions/game, sec/possession, OT score rate by start distance, margin-3 paths by m300 bucket, drives per team-half, tie share, by OT era; real side from drives.parquet + pbp yardline_100.
- Apply (after the gate-check run releases the file, ~11:50 ET): `git apply F:\Repos\nfl_py3\tests\scratch\sim10-instrument.patch`
- Run: `$env:SIM10_INSTR='instr_s11'; F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim10_drive.py sim --seed 11 --workers 3 --realsched 1 --label instr_s11` (match the flags of the baseline being compared; ~20 min per the one-seed screen).
- Analyse: `F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim08_instrument_analysis.py run instr_s11` (~1-2 min, reads 17 seasons of pbp).
- Note: real m300/m120 are the margin at the first drive starting <=300/120 s (drive-start approximation).

## Instrumentation identity unit (2026-11-03)
Patch tests/scratch/sim10-instrument.patch APPLIED to scripts/sim10_drive.py (uncommitted); ruff 4 errors before and after. Run s8idA (seed 5, 1 world, 3 seasons, workers 1, realsched 1) was still running after about 6 min under CPU contention with SIM09 and was killed (PIDs 32116, 30276). Hashes not yet compared. Next: rerun A (no SIM10_INSTR) then B (SIM10_INSTR=1, label s8idB) after SIM09 finishes, compare SHA256 of games.parquet and plays.parquet, then scripts/sim08_instrument_analysis.py run s8idB. Revert with git apply -R if identity fails.

## Engine patch completed (2026-11-03, nothing run)
- tests/scratch/sim08-spec-a-engine.patch regenerated (s0->s1): `TerminalTables` class in sim10_drive.py (lookup via sim08_terminal_fit cum_probs/bin_of/margin_idx; reg state = (60 s bin, possessing-team margin clipped +-17); OT state = (min(possession idx,2), margin clipped +-8)); tg[1:4] = td, fg (made only, fg_miss not added), def_score; fgdec/q4dec skipped only inside the 300 s window (`term_on`); OT branch tilts the drive pool to the OT table (`self.poss_k` set from n_poss before draw; needs models loaded). Elapsed, kneel, punt, turnover, other mass left to the pool (unmapped, reported).
- `SIM10_TERMINAL` is now a directory ("1" = artifacts/sim08_terminal); table per simulated season: transitions_<season>.json for 2009-2025 (fitted without it), transitions_all.json otherwise; built on first need via `fit-all` (written: all 10488 reg / 678 OT drives). Added to sim08_terminal_fit.py: table_key, table_path, write_table, fit_all_table, load_table, `fit-all` command; `fit` also writes transitions_all.json.
- sim09-situational-engine.patch context lines updated to the new spec A text; s1/s2/s3.py and engine-queue-2026-11-03.patch regenerated (order spec A, SIM-06, SIM-09). Combined `git apply --check` on the real file OK; ruff E501 on s3 5 v s0 4 (extra one is SIM-06 koseason line, not spec A).
- Harness tests/scratch/engine_queue/spec_a_harness.py: reg and OT lookups equal the table probabilities on 8 states, 2026/2027 select transitions_all.json, 2009/2025 their own: PASS.
- Next: root applies combined patch, runs identity, then the A4 declared run: `$env:SIM10_TERMINAL='1'` with the usual flags.

## A4 declared run, launch-ready (2026-11-03 11:45 ET, subagent; nothing run)
- (1) Iterated anchor ALREADY implemented, flag off by default: `iterate_anchor` (sim08_market_anchored.py:197) Newton on the quadratic response Jacobian, CRN pilot seed+17, stop within 1 SE on mean margin and mean total or when z stops improving (best iterate kept), ITER_CAP 4 declared as a budget (Amendment 4 item 3, declaration lines 329-337); final draws use a fresh seed. Jacobian step h 1e-4 is a numerical differencing step, not a fitted constant. No edit made. Read-only caution (inferred): both targets within 1 SE has only about .46 chance at the true optimum, so cap-hit games will be common; they are reported via iter_cap_hit in the helper but `anchored_draws` discards that dict (x_diff, x_env, margins are saved, iter stats are not): report cap-hit share by recomputing |sim mean - line| / SE from the npz.
- (2) SIM10_TERMINAL format matches: a directory (or "1" = artifacts/sim08_terminal), files `transitions_<season>.json` per LOSO season 2009..2025 (all 17 present plus transitions_all.json); TerminalTables.cum(season) keys via table_key(int(season)) so graded season s (2020..2025) uses the table fitted without s; seasons outside 2009-2025 use `all`. Workers spawned by the fold inherit the env.
- (3) Smoke NOT run (11:45 ET, Week 9 lock holds the machine until 13:00). Smoke after 13:00, 1 core, avoids the 9600-game design by reusing the Unit 3 fit (smoke only proves start and write):
  New-Item -ItemType Directory -Force F:\Repos\nfl_py3\artifacts\sim08_market_anchored\smoke_a4; Copy-Item F:\Repos\nfl_py3\artifacts\sim08_market_anchored\main\fit_2025.json F:\Repos\nfl_py3\artifacts\sim08_market_anchored\smoke_a4\
  $env:SIM08_HELD='2025'; $env:SIM10_TERMINAL='1'; F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim08_market_anchored.py fold --held-out 2025 --arm primary --run smoke_a4 --workers 1 --draws 2 --amend4 1
  Expect ~7 min (272 games x <=12 sims x ~0.13 core-s), writes smoke_a4/primary_2025.npz and total_target_2025.json. Read, then delete the smoke_a4 directory. Note `fold` has no game limit.
- (4) Heavy command (after the engine patch identity check passes and the heavy slot frees):
  $env:SIM10_TERMINAL='1'; F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim08_market_anchored.py full --run amend4 --arms primary --workers 4 --draws 372 --amend4 1
  (`full` sets SIM08_HELD per fold and inherits SIM10_TERMINAL; asserts free RAM >= 16 GB; checkpoints per fold; resumable.) Outputs artifacts/sim08_market_anchored/amend4/{fit_,total_target_,design_}<s>.*, primary_<s>.npz for 2020..2025, run.log. Runtime (inferred from Unit 3: design 315 s, 3181 s total per fold at 372 draws, 4 workers): anchor cost 2x to 6x draws per game, so about 2-5 h per fold, 12-30 h total.
  Score: F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim08_evaluate.py --run amend4 --arms primary  -> amend4/evaluation.json. Then re-run the solo sim cover coefficient (scripts/sim08_negcoef_decompose.py, point at run amend4; check its run argument first) and record via weak-signals record. The planted control re-run: control --noisy 1 as in Unit 3.
- Open: sim10_drive.py diff needed? None found. If SIM10_TERMINAL is unset in the shell the run silently reproduces spec-A-off; verify `echo $env:SIM10_TERMINAL` before launch.
