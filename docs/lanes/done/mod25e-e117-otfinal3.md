# E117 OT final-margin-3 drop (B -> L)

## Goal
Explain why P(final3 | tied after regulation) fell .5726 -> .4584 (real .6822) when TPO/CLQ=2/URG/CKY were added to base B (crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkekudw), though none of them acts on qtr 5.

## State (measured 2026-10-09, read-only, tests/scratch/e117/a1..a12.py on artifacts/mod25e3/e5_{B,L}_s11-13 play parquets)
- Definition is not the cause: tied-after-regulation games == games with qtr 5 plays (B 2251, L 2260), no mismatches.
- Regulation-end state not the cause: OT first snap gsr 900 both, same start-yl dist as Q3, OT drive-2 tied-state start yl 70.5 v 69.7, clock 758 both.
- OT drive outcomes changed only in FG vs TD split: tied drive 2 FG .273 -> .199, TD .098 -> .182, scored total .371 -> .381. Regulation TD/FG per drive unchanged.
- OT FG attempts on downs 1-3 per OT game: B .425 -> L .141 (diff -.284 [-.308,-.259]); dn4 attempts .367 -> .493; OT TDs per OT game .327 -> .453 (+.127 [+.105,+.151]); OT mass3 per game -.0065 [-.0078,-.0051]. Same direction in all 3 seeds and 6 seasons.
- CAUSE (read): scripts/mod25e_wfg.py:58-61 walks frames up to the first one with local "offense", then reads in_ot / ot_possession_index (engine-only locals). scripts/mod25e_tpo.py:221 binds a local named `offense` in its own pol frame, installed outside wfg (scripts/mod25e_crH.py:158 after wfg:~137), so wfg's walk stops at the TPO frame, in_ot is None -> False and the walk-off FG hook never fires in L. B's extra ~640 forced dn1-3 OT FG attempts vanish (956 -> 319). Inferred: ADJ/YLM/KICK walks (adj:290, ylm:64, kick:277) also stop at TPO but only read offense/off_to/def_to/qtr, which TPO relays correctly; their gsr/possessions are unused (F841). wfg is the only reader of in_ot.
- Consequence: every E114 L number that touches OT (final3 | tied, OT mass3, margin SD, pts/game) is confounded by WFG silently off. Regulation cells (mass3 .0917 -> .0987, final-drive, Q3) do not depend on WFG.

## Tried
Rejected: tie-definition/logging change, regulation-end state, OT first-play clock/elapsed (el/play 26.21 v 26.20), kneel/4th-down mix at sd 0 (FG share rose), qtr-5 leak of TPO/CLQ/URG/CKY (all return on qtr gate), per-task rng leak (separate rng per hook).

## Next
1. Fix (not applied; scripts frozen during smoke): in scripts/mod25e_tpo.py:221 stop binding `offense` (use `off_to, def_to = fr.f_locals["off_to"], fr.f_locals["def_to"]`; `offense` is unused); also make scripts/mod25e_wfg.py:59 test "ot_possession_index" instead of "offense" so no wrapper can shadow it. Check urg.py:394 and any new hook for the same shadow.
2. Re-run L with the fix (3 seeds, same label) and E100 PART 2; expectation (inferred): OT FG dn1-3 back near .42/game and P(final3|tied) back near .57. Then re-judge E114 aggregates.
3. Re-run any other earlier layered-hook candidate whose pol assigns `offense`.

## Open
Whether other pre-TPO hooks read non-relayed engine locals through walks (grep f_locals.get in clk/ckc/e49/stb) - none found reading in_ot.

## Fix applied 2026-10-09 (orchestrator)
mod25e_tpo.py:221 no longer binds a local `offense` (reads off_to/def_to only); mod25e_wfg.py:59 walks to the engine-only local `ot_possession_index`. Same hazard (inferred): STF (mod25e_stf.py:218-229) reads gsr/possessions with .get from the first `offense` frame, so with TPO=1 its per-game reset saw None; E114 L aggregates are confounded too. Rule for every hook: never bind a local named `offense`; walk to an engine-only local. Next: GEO smoke rerun, then 3-seed L_fixed (+GEO if smoke OK) v dw.
