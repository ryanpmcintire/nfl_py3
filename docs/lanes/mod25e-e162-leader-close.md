# mod25e-e162-leader-close

## Goal
Find why sim leaders in Q4 (<=120 s) gain more yards and pass more than real; name carrier; default-off hook fitted LOSO; deferred crH install patch. Offline only; no existing scripts/ edited.

## State (measured 2026-10-18; X2 seeds 11-13, game bootstrap 90%; tests/scratch/e162/)
- Q4 leaders <=120 s sim v real (lead 1-3/4-8/9-16): pass share .182/.103/.064 v .124/.082/.058; run yds 3.72/3.85/3.71 v 2.64/2.94/2.65; first-down rate .265/.256/.265 v .153/.160/.169. Kneel share matches.
- Pass/punt/kneel carrier (read): engine neighbour draw sim04_engine.py:477-537 (K_STATE=200 :33). LDR hook fixes it (below).
- Run-yards: Geo GEO=2 always repick (mod25e_geo.py:433,447) pools all phases, nearest on (dist,score,time) only (:339-362). Sim phase-3 leader run sources 18.6%; phase-3 source 3.16/3.25/2.73 v other-phase 3.85/3.98/3.93 (b1/b5).
- Not carriers: yard_shift, YLM, team tilt (b1-b5).
- E162c (geo_phase.py, geo_phase_b.py, geo_phase_an.py, geo_phase_bprep.py, logs gp_an.log gpc_an.log; 290,204 real run/pass queries, pools leave-one-season-out, uniform K=200, no team weights): phase mismatch is widespread: 45.8% of Geo-served rows differ in phase from the query (Q1-3 32.6%, Q4<=120 77.7%, Q4 120-300 82.2%, Q2<=120 87.8%). Yard gap v real from it is small overall (+0.085 [.056,.117] all; +0.38 [.23,.51] Q4<=120).
- Variant A (declared first): exact phase match, widen fp band to K rows, fall back to all phases if <K at band 99. FAIL: held-out LL worse 9/9 folds (yards -.0076 [-.0083,-.0069]); fallback 21% overall and 59-66% in the leader cells; leader run yds only -0.06..-0.10 (real-state Geo expects 4.54/4.69/4.79 v real 2.82/3.26/2.94). Where phase-3 pool exists (no fallback, mismatch 0): ph 3.76/3.85/3.78 v real 2.95/3.23/3.42 v cur 3.94/4.11/3.94, so a residual +.5..+.8 survives exact phase.
- Variant B (second look, declared after A failed): time axis scale lam in {2,4,8,16,32} chosen nested LOSO by held-out total LL. FAIL: held-out gain -.00028 v selection-set +.00017 (gap .00045), chosen lam inconsistent (2,8,16), 3/9 folds positive; yards LL delta -.00011 [-.00057,+.00034] P(>0)=.325; leader yds +.01..+.03; Q2<=120 yards +.0030 [+.0011,+.0050] only. 45+5 looks. 12 E161 buckets: A flips 3 IN->OUT (4th d1 conv, TD yl21-40, pp 21-30 reach), B flips 2 (TD yl21-40, pp 21-30 reach); neither fixes an OUT bucket.
- Conclusion: pool phase composition is not the main carrier of leader run yards; the residual is a state feature Geo lacks (defence/box expectation when the run is known, or timeouts/clock-kill intent). No hook written, no install patch (fail rule). Unused names kept for later: env GEOPH, label suffix "gp".
- Pass share and kneel are not Geo effects (Geo keeps play type).

## Hook (default off, not installed)
scripts/mod25e_ldr.py: LDR=1. LOSO refit 2026-10-18 (9 classes), held-out LL/play adjusted -1.1333 v engine draw -1.3650 (9/9 folds positive), in-sample gap .002. Class ratios run 1.012/1.047, pass .835/.793, kneel 1.155. Deferred crH patch: after `ge.install_geo()` block (mod25e_crH.py:182) add `import mod25e_ldr as ld` / `if ld.enabled():` / `    ld.install_ldr()`; after the GEOX=="2" label block add `if os.environ.get("LDR") == "1": label = label + "ld"`. Install after KN.
Pass bar smoke: leaders <=120 pass share lead 1-3 <=.15, first-down rate <=.20, kneel within .03. Run yds <=3.2 is NOT reachable by LDR alone while GEO=2 repicks.

## Next
1. Orchestrator installs LDR patch after runs end; smoke then 3-seed.
2. E162d: find the missing Geo state feature for leader runs: candidates are timeouts (defence timeouts left, in engine kernel but not Geo f3), run-expected-by-defence (score x time x lead already in f3, so test residual by down/distance and offence rating), clock-kill intent; fit LOSO by held-out LL on Q4<=300 leader rows only (record via weak-signals).

## Open
Team-kernel weights omitted from E162c (uniform K). Selection for variant B uses other seasons' pools that include the query's own season less itself (mild leak, reported gap). Fixed-bin game-end drive share in a1 mis-binned.
