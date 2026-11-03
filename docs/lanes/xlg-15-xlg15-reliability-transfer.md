# XLG-15 Reliability transfer

## Goal
Measure NFL situational trait reliability on college first, then test whether CFB openers price it (family xlg15_cfb_trait_reliability).

## State
Unit 1 done 2026-11-03 (scripts/xlg15_cfb_trait_reliability.py, scripts/xlg15_batch.py, artifacts/xlg15/report.json; 12 cells recorded, unresolved_below_power). Split-half (alternating games, Spearman-Brown) CFB 2014-19 / 2023-25: pass_oe when ahead 8+ in Q4 .60/.57; 4th-down go residual (LOSO-season GBM, team-season) .59/.63; venue away false starts (home team, season parity) .61/.22 (2023-25 CI -0.18..0.48). Pricing, opener ATS margin per SD of as-of trait, game bootstrap (ats pts, P+): pass_oe -0.30 (.12) / +0.23 (.75); 4th-down +0.10 (.66) / +0.29 (.81); venue +0.03 (.54) / -0.14 (.31). All intervals about +-0.5 to 0.9; unresolved. CPOE gap not buildable.

## Tried
12 looks (3 traits x 2 eras x reliability, pricing). Team stands in for coach; no referee crew term for false starts.

## Next
Pool CFB+NFL pricing per trait only if units match; NFL sharper tests for 4th-down and pass_oe remain.

## Open
Signs not pre-declared; no sign consistent across eras. Slopes near zero with wide intervals.
