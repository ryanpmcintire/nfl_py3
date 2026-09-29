# Lane: tiebreaker total study

Goal: does any pregame adjustment or rounding of the market total beat entering the market total, out of season (protocol: docs/tiebreaker_total_study.md).

State: script scripts/tiebreaker_total_study.py run once (json at tests/scratch/codex/tiebreaker_out.json). Measured, 3407 REG games 2013-2025 (226 last games), closing-grade total, integer guesses. MAE market (M0) 10.503; served proxy (0.1*resid, -1) 10.504, 6/13 seasons better, reduction -0.002 (95% [-0.042, 0.038], P+ 0.47). LOSO shade M2 10.493, 7/13, reduction +0.010 ([-0.021, 0.042], P+ 0.73); LOSO weight+shade M3 10.505 (in-sample 10.488, gap +0.018), LOSO picks w 0-0.1, s -0.5/-1.0 every fold. Residual blend alone M1 worse (P+ 0.0). Floor 6/13, ceil identical to M0. Mode-window rounding R3 10.562 and R4 10.575 significantly worse. Actual minus market averages +0.39. Simulated field (30 entrants, market+N(1.5,sd)) win share: M0 0.026 (sd6), M4 0.023, M3 0.019, R3 0.025; shading loses share because the field sits above the line.
Verdict: no arm meets the declared bar (9 of 13 seasons); no served change recommended; served guess is indistinguishable from the market total on MAE and slightly worse on simulated win share. Unresolved_below_power, not closed.

Tried: 9 arms x 2 populations (18 looks). Not tested: opener totals (none in file), weather terms.
Next: orchestrator records the commands below; optional later test with opener totals.
Open: field model constants (mean 1.5, sd 6) are borrowed from the prior shading script, not measured.

Record commands: (orchestrator fills flags from `nfl-ats weak-signals record --help`; classification unresolved_below_power, effect units mae_points, M2 reduction +0.010 P+ 0.73; M4 served proxy -0.002 P+ 0.47)

**Recorded 2026-09-29 (root):** `tiebreaker_total_loso_shade_vs_market` and `tiebreaker_total_served_proxy_vs_market`, both unresolved_below_power. No served change.
