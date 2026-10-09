# mod25e E105 Q2 last-drive play choice

## Goal
Name the play-choice mechanism behind clock-ended Q2 last-play kneel share (sim .557 v real .478) and run share (.165 v .281) on base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbgh (gh seeds 11-13, burn 2); hook only if a mechanism is named.

## State (2026-10-08, measured; no hook, no smoke run)
- Policy located: endgame DECIDE (mod25e_endgame.py:276-304, EGT=1 EGH=1 KNW=1) draws the class (run/pass/fg/kneel/spike) from an HGB fit on real down<=3 rows (Q4 gsr<=600, Q2 gsr<=1980; features down,dist,yl,sd,gsr,timeouts,hs,q4,pstop) then swaps in a nearest real row; KN (mod25e_clk.py:424-437) redraws kneel elapsed; FD4 handles 4th down.
- Snap-level choice matches real (tests/scratch/e105/pc.py, std.py; downs 1-3, hs<=120). Real rates re-weighted onto the sim state mix (cells qtr x score sign x def TO x off TO x hs bin x down) v sim actual: Q2 hs(0,30] kneel .147 v .154, run .142 v .152, pass .661 v .654, spike .049 v .039; Q2 hs(30,60] kneel .010 v .011; Q4 hs(0,120] kneel .162 v .157, run .219 v .231, pass .593 v .589, spike .026 v .023. Raw by score/timeouts also within about .02 except Q4 defence 1-2 timeouts kneel .187/.097 v .147/.060 (state-mix, not rate).
- So kneel eligibility, timeouts and spike usage are not the carrier. The last-play share gap is occupancy at the clock edge (lastp.py, spsn.py, runel.py): snaps starting at hs=1 per 1000 halves real 26 v sim 90 (hs=2 41 v 61, hs=3 38 v 53, hs>=7 equal). Predecessors of snaps with hs<=2 per 1000 halves: run 10.6 v 57.0, pass 74.9 v 118.1.
- After a run starting hs(20,30]: ends half real .342 v sim .202, leaves 1-2 s real .008 v sim .109; hs(30,45]: ends .234 v .124, leaves 1-2 s .006 v .068; hs(10,20] ends .43 v .32, leaves 1-2 s .039 v .153. Pass equal within .01 except 1-2 s .017/.005 v .027/.009.
- Spike at hs<=2: sim 941 per 78336 halves v real 28 per 4608 (about 2x), and ends the half 47% v 11%; hs 2-4 equal. Spike elapsed matches by bin (hs 5-20 means .91/1.77 v .94/1.76).

## Tried
- Kneel/run/pass/spike by sdb, hb, dtb, otb, down, yardzone (pc.py); standardised comparison (std.py); is-last-play by kind and hs (lastp.py).

## Next
Clock side, not choice: run elapsed near the edge. The carrier is run plays at hs 10-45 that the sim leaves with 1-2 s (pool replays an ended-half row at a nearby larger hs as el=hs-1..2 after clipping, mod25e_clk.py:70-72 plus real_rows_fixed clip in mod25e_gfl.py:27). Candidate: draw the remaining time after a run from real hs-hs_next incl. ends (ended-half rows censored at hs, KM; CKC tried it for run and lost LOSO ll, but score by P(rem 1-2)/P(end) rather than el ll before closing). Second small item: spike eligibility at hs<=2.

## Open
CKC run LOSO loss is not a closing ground; record via weak-signals if closed. No process started by this unit.
