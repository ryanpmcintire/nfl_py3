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
