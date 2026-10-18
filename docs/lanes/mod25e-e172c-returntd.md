# E172c return-TD excess

## Goal
Find why sim returns turnovers for TDs more (def TD/game .458 v .395; share of turnover rows 4.5% v 3.4%).

## State (measured, tests/scratch/e172c/r1.py r2.py; base crH..gx2faldrg s11-13, run/pass turnover or def-TD rows, 314,635 sim v 20,242 real)
- Geometry is NOT the carrier. Sim spot below source spot (q<src fp) 60%, above 0.6%; TD share q-src<-2 4.21% v source base 3.99%; |gap|<=2 4.68% v 4.39%; q>src 5.05% v 5.05%. Lower q shortens the needed return, so no far-field TD is served at a longer return.
- Composition is NOT the carrier: expected TD share at real per-10yd-bin rates on the sim yl mix 3.30% v sim 4.52%. Excess is within bin: 0-10 .006 v .003; 20-30 .026 v .017; 40-50 .041 v .027; 50-60 .073 v .054; 70-80 .141 v .098; 90+ .174 v .147 (ratio 2.0 to 1.2).
- Type: INT TD 13.2% v 11.2% real; fumble 11.7% v 9.2%; downs/other 0.42% v 0.0%.
- Sim state: TD share 5.2% trailing by 8+, 3.4% leading by 8+ (no real comparison made).
- Source row TD share of drawn turnover rows 4.26% v pool-wide 3.40% (draw over-selects TD sources 1.25x). 894 of 14,214 sim def-TD rows (6.3%) have a source with points_def 0 (84% scored po>0, sim yl 14, source fp 18, flip 56%): unexplained label/idx mismatch (GEOX tdp replacement keeps idx j, mod25e_geo.py:395-401), not yet traced.
- Geo.violates mod25e_geo.py:312-313 returns False for flip/points_def rows, and candidates mod25e_geo.py:344 admits free rows at any fp in the band while non-free rows are filtered by y+shift<q; this favours free (turnover) rows in the neighbour set. Untested as carrier.

## Tried
Gap-conditioned TD share, bin decomposition, INT v fumble, crosstab sim v source points_def.

## Next
1. Offline (no sim): replay Geo.candidates on logged states to measure free-row share in the k-set v pool share, per bin.
2. Trace the 894 mismatched rows (GEOX path, idx j vs t).
3. Fix is a draw-weight or pool-label defect, not a TD-iff-yards-reach-goal geometry rule; do not write the geometry module. Order with SAFEG: safety fix is independent (safeties 3,484 all from 2-pt sources, yards rule 0% pass); apply SAFEG first, then any free-row selection fix; crH patch deferred.

## Open
Pool-level real TD 3.40% v drawn 4.26%: which weight (weight_fn, tilt, KNN k) over-selects.

## E172d (measured, tests/scratch/e172c/r3.py r4.py, 40k real run/pass query states, GEOX=2, band 0, k 200, unit weights)
- Geo is NOT the carrier. Free-row share of served k-set 3.36% v symmetric natural 3.32% (1.01x); return-TD source share per play 0.26% v 0.23% (1.13x; 70-90 1.18x, 90+ 1.26x); served rows sit +2.2 yd farther out than natural. Too small and Geo only fires on violators.
- Variant tried and REFUTED: free rows on a symmetric window [q-w,q+w]: free share 5.9%, ret 0.46% (worse); non-free buckets unchanged (yards 5.27 v 5.27). Module deleted. Current one-sided window already matches natural free density.
- 894 mismatched def-TD rows (6.3%): by sim yl 0-2 98%, 2-5 92%, 5-10 73%, 10-20 32%, 20-30 10%, 30-50 4%, 50+ <1%, 90+ 0. Tracks goal-to-go incidence = logging artifact of mod25e_gz.py:127-128 (idx=i0 logged, served j; SERVEDIDX fixes). mod25e_geo.py:395-401 swaps in a points_off>=6 row t, so it cannot create a def TD: not the carrier.
- Residual carrier is upstream of Geo: base KNN draw weights (team kernel, IPW, PASS_W, tilt) over-select TD-return sources 1.25x pool-wide.

## Next (E172e)
Replay weight_fn (needs ns/tables from dv._G) on the base k-set: return-TD source share v unweighted k-set, per weight term; rerun logged states with SERVEDIDX=1 to re-count mismatches.
