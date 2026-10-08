# mod25e E101 log recheck

## Goal
Re-score play-log results on synced logs of artifacts/mod25e3/e5_crHpqokgndecsmfwtjo2as2ypw2xfdcd_s1{1,2,3} v earlier numbers and real 2009-17.

## State
Done (measured). Driver scripts/mod25e_e101.py (patched copies of e93, e95, e97, e99 at exec time, plus snap check); outputs artifacts/mod25e3/e101/ (e93, e95, e97, e97b = same script on old ypw2 logs, e99, snap.txt); clock2 at artifacts/mod25e3/clock2/clock2_xfdcd_e101.txt.
- Punt (e93 sim column): slope .698 v real .692; start 74.19 v 74.19; touchback .001 v .097. Unchanged.
- E95 nonscoring: end-spot gap +.026 [-.031,+.084] (was +.022); start gap +.032 [-.005,+.065] (was +.021). Slightly larger, same sign.
- E97 OT FG share downs 2-3 idx 1/2+: new sim .16-.21 v real .15-.19; same script on old ypw2 logs .02-.03. The old gap was the stale log (WFG row edit).
- CDR snap: P(gsr<=5 | Q4 gsr<=300, tied/trail<=2, yl<=40, downs 2-3, my filter) real .095 v ypw2 .027 v xfdcd .032. e99 P(FG kick gsr<=5) real .355 v .187 -> .224.
- Clock2: clock-ended drive share and Q2/Q4 last-play timing flat (Q4 last play <=5s real .128 v sim .306 -> .311).

## Tried
Real snap share (.39 = 58/148) not reproduced by my filter; used e99's own kick share as cross-check.

## Next
Edit generator-fidelity lane: close the E97 OT FG gap as a stale-log artifact; CDR snap composition and clock timing gaps remain open; punt touchback defect remains.

## Open
xfdcd differs from ypw2 by more than log sync (PATY/FGD/CDR), so the CDR rows mix both.
