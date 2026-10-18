# mod25e E175 leader Q4 300-900 conversion excess

## Goal
Decompose sim v real leader (live lead sd>=1, offence leading) Q4 gsr 300-900 drive excess by lead bin; name carrier; fit one behavioural mechanism LOSO if one fits. Offline, base XFA+LDR+RG s11-13 v real 2009-17.

## State (measured 2026-10-18; tests/scratch/e175/an.py an.txt, an3.py an3.txt, an4.py an4.txt; game bootstrap 400, 90%; ch2.py/ch3.py rebuild yl/dist aligned to e174 D pickles)
- Caveat: sim play-log `yards` is not served yards (3.2 v 5.5 per play); position and yards here use logged yl differences, which match real (5.12 v 5.12).
- Excess pts/drive beyond start-yl value, sim minus real: lead 1-8 -0.09 [-0.22,+0.02]; 9-16 -0.05 [-0.20,+0.10]; 17+ +0.26 [+0.12,+0.41] P.99 (real -0.35, sim -0.09); all leads +0.01. The E174b +4.5 sits in the 17+ bin.
- 17+ bin (0.4 drives/game): TD share +3.6pp [1.3,5.7] (20.6 v 17.1), downs share -1.7pp, plays/drive +0.43, burn +14 s, 3rd conv 43.1 v 36.1 (+7.0pp [4.8,9.3]), 4th GFI att/drive -0.02, RZ trips +3.1pp [0.5,5.6], RZ TD/trip +5.4pp [0.3,10.5], FG attempts in range -1.3pp [-5.8,+2.4] (not the carrier), start yl +0.6 [-0.6,+2.0], explosive rate parity (+0.05pp).
- 3rd down by distance band (play level, lead>=9, same distance mix, same run share): sim conversion higher in every band, 9-16 +3.8pp all [1.3,6.4], 17+ +5.0pp [2.6,7.4]; neutral +1.5pp [-0.4,3.1]. So within-distance effectiveness, not play calling or mix.
- Trailing defense: yards allowed 17+ sim 5.02 v real 4.79 (+0.23 [+0.04,+0.43]); trail-minus-neutral sim-real +0.34 [+0.09,+0.61]; explosive allowed +0.4pp [-0.2,+1.1]; 1-8 and 9-16 parity. Sim defenses do not tighten as real trailing ones do, but not by selling out (real is the tighter one).
- Real-only LOSO logistic on 9820 Q4 300-900 3rd-down plays: lead>=9 dummy with team strength control -0.046 logit (-1.1pp), negative in all 9 held-out folds; held-out LL gain -0.00006/play [-0.00023,+0.00009] P+ 0.25 (no gain). Leaders are stronger teams (S +4.6 v -1.6), which supplies about +1.4pp; real leader 3rd conv (40.0) equals neutral (38.5) while sim leader 17+ is 45.0.

## E175b (measured 2026-10-18; tests/scratch/e175/b1.py b2.py b3.py; play-level LPM, game bootstrap 400, 90%)
17+ Q4 300-900 3rd conv gap sim-real +5.0pp [+2.3,+7.5] (play level; +7.0 was drive level). Real n=1165, sim n=19491.
- (a) strength: leave-game-out ratings; sim 17+ leaders not stronger vs opponents (sdiff +6.2 v +5.7); distance+strength composition explains +0.5pp [-0.2,+1.2] (~10%); residual +4.5pp [+2.0,+7.1].
- (b) personnel: real backup QB 12.4% [10.4,14.6] of 17+ Q4 300-900 3rd plays; backup-v-starter conv +3.8pp [-2.8,+10.9] (not worse); starter-only sim implies -0.5pp [-1.3,+0.4]. Share ~0, wrong sign; no personnel mechanism.
- (c) pool (offline proxy: real all-row rate at the 17+ distance mix, no engine replay): pool 40.3 v real 17+ 40.0 (-0.3 [-2.4,+2.0]); sim 17+ 45.0 is +4.6pp [3.7,5.6] above the pooled rate; pool share ~6%.
- Carrier: none of a/b/c. Sim leaders convert above real leaders AND the real pooled rate at the same distance mix. Source unlocated, inside the sim.

## Tried
Mechanism "trailing defense prevent / leader conservatism on 3rd down": real effect is -1pp, below the 3-7pp gap, no LL gain. Unresolved_below_power; not installed; no module or crH patch written.

## Next
1. Primary: run `nfl-ats weak-signals record` for the 17+ leader 3rd-down conversion gap (unresolved_below_power, closing ground not met).
2. Done E175b: not strength/personnel/pool. Next: locate the sim-internal source (sim leader 3rd conv with LDR/RG hooks off v on; engine pool replay for 17+ queries).

## Open
Whether the sim's 17+ TD-share excess is strength selection or a pool conditioning artifact.

## E175c (measured 2026-10-18; tests/scratch/e175/c1.py c1.txt c2.py; next-play first-down-or-TD 3rd conv, run/pass, s11-13 burn 2, game bootstrap 150, 90%)
17+ Q4 300-900 minus all-state conv (pp; level 39.8 base here v 45.0 in b1: different TD/drive-end definition and mix, read trend only): B -0.2, D1 +1.9, X +2.7, X2 +2.4, XFA +3.7, base +3.9, GSTOP +3.5. All-state conv flat 34.8-36.2 across labels (not inflated). 17+ excess is outside the red zone (yl>25: B -0.5, D1 +2.5, X +3.9, XFA +4.6, base +5.2; yl<=25 about +1). Steps: B->D1 largest (GEO=2 force repick of every run/pass play, mod25e_geo.py:438,450,471), D1->X +0.8, X2->XFA +1.3 (CIs overlap), LDR/RG ~0, GSTOP -0.4.
Mechanism (inferred, unreplayed): GEO=2 sends every 3rd-down play through geo.fix (mod25e_geo.py:361-400): one-sided window fp>=q (:322-323), exact-distance filter (:325), k-nearest in (dist,sd,time), tilt_factor weight by team ratings (:413-415, :466) plus yard shift bias+gain*(net_sim-net_row) (:378). Leader states get rating-scaled weight and yard shift on top of the base engine rating gain.
## Next (E175c)
Offline pool replay from real rows (no engine run): weighted conv of fix() candidates for 17+ sim query states with tilt on/off and shift on/off; verify tilt-off removes the +4-5pp before any code change.

## E175d (measured 2026-10-18; tests/scratch/e175/d1.py d1.txt d2.py d2.txt; offline Geo.candidates replay, GEO=2 GEOX=2 GZT=3 cols 1,3, additive shift; 1165 real 17+ Q4 300-900 3rd-down run/pass queries own rating, self row excluded, query-bootstrap 400, 90%; 4995 random all-state 3rd-down queries; looks: 4 weight variants x 2 shift x 2 sets + 1 candidate-state split = 17)
- Weighted conv, leaders: uniform candidates 40.8; team-kernel only (tilt off) 42.6; current tilt 43.2 (+0.59 [0.55,0.64]); TILTL fold 43.7 (+1.10 [1.02,1.17]); real 40.1. All-state: uniform 39.2, kernel 39.3, current 39.3, TILTL 39.4, real 40.9. Leader minus all-state replay: +3.9 (current), +3.3 (tilt off).
- Yard shift: identical to 0.00 in every cell (bias 0, gain 0 at this config). Not a carrier.
- Tilt: only +0.6 of the +3.9; TILTL does NOT remove it (adds +0.5). Do not expect the 3-seed TILTL run to fix leaders.
- Carrier: rating kernel x candidate state. Leaders are strong offenses (mean rating 2.61); kernel adds +1.9 over uniform, but only 8% of candidate weight comes from 17+ states; 17+ candidates convert 39.5, non-leader candidates 43.1. So strong-offense non-leader rows stand in for leaders; real leaders do not convert at strong-neutral rate.
- Fix candidate (data definition, no constant): nearest-state ranking should put score state first inside the exact-distance set (candidates of the same lead class), instead of dist-squared sum over (dist, score, time) with k=200. Offline evidence only: 17+ candidate conv 39.5 v real 40.1; the replayed fix itself not yet run.
## Next (E175d)
Offline: re-rank candidates by score-state first (same lead class), recompute weighted conv for 17+ and 1-8/9-16 leaders and all-state; record weak-signals look (unresolved_below_power).

## E175e (in flight 2026-10-18; tests/scratch/e175/e1.py -> e1.pkl, e1.log; analysis e2.py ready)
Declared variant: inside the exact-distance one-sided-window set, rank candidates by lam*(d score feat)^2 + (d time feat)^2 (lam=1 is current; large lam = score state first, time tiebreak, band widens by nearest score), K in {50,100,200,400}; weights unchanged (team kernel x tilt). 24 configs x 14 sets (L17 3rd, neutral 3rd&7+ runs, late-leader runs 1-3/4-8/9-16 gsr<=120, late trailers 3rd, 8 E161-like conv buckets; TD yl buckets skipped, censor-dependent). (lam,K) chosen LOSO by held-out LL of real FD/TD outcome. e1.py was launched as a background job (python e1.py, cap 2500 queries/set); no results yet.
Next: when e1.pkl exists run `python tests/scratch/e175/e2.py` (held-out LL v current per fold, rows, widening share); write scripts/mod25e_geoctx.py only if it passes; compare with E177. Kill the e1.py python process by PID (find via CommandLine containing e175) if still running.
## E175e RESULT (measured; e1.py e1.pkl e2.py; 24 configs x 14 sets = 336 looks; LOSO (lam,K) by held-out LL; query bootstrap 90%): FAILS, no module written
- 17+ leader 3rd: real 40.1, current 43.2, chosen-LOSO 43.4 (all 9 folds chose lam=1,K=400, i.e. no score-first). dLL +.0034 [-.0013,+.0080] from K only. mean |dscore| of weighted rows 23.6 pts (cur 22.2).
- Reason: exact-distance one-sided window holds only ~100-230 rows per query (leaders ~210), so K<=200 is the whole window and re-ranking cannot bring in score-matched rows; widening is 98-100% in every set (rows >3 pts off). Needs a larger pool (other sources), not a different ranking.
- Neutral 3rd&7+ run FD: real 27.1, cur 19.2, no gain (dLL -.0004, 0/9 folds). Late-leader runs gsr<=120 (FD real 13.8/14.3/16.4 v cur 24.5/25.1/26.8): dLL +.0087 [+.0008,+.0174] / -.0069 / -.0136 [-.0268,-.0035]; the 25% FD overstatement is not touched (served-row y>=dist includes tail rows). Late trailers 3rd: real 40.4 v cur 36.8, dLL +.0025 (K400). Other conv buckets dLL -.0005..+.0059 (2nd&7+ +.0059 [+.0020,+.0098], lam 4-256 K400). Pooled dLL/query +.0016, n 20806, nothing replaces RG/RGT.
- Next: E177 class-ratio reweight is the live competitor; pool enlargement (drop exact-dist for leaders, reweight) untested.
