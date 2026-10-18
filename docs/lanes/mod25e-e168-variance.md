# mod25e E168 variance decomposition on base XFA+LDR+RG

## Goal
Redo the E131 Var(margin) decomposition on base label ...g2rtgw13gx2faldrg (s11-13) v real 2009-17; name the component carrying the remaining excess. Offline, no sim run.

## State (2026-10-18; measured; regulation-only points; 300 game bootstraps, 90%; ~45 stats x 2 sims, one family; tests/scratch/e168/an.py, an.txt; sim table tests/scratch/e131/S_ldrg.pkl)
Delta = sim minus real, [90%]; B = old base S.pkl (E131), LDRG = this base.
- Var(M) real 221.3: B +14.3 [+2.8,+27.2] P.98; LDRG +8.2 [-3.6,+20.1] P.88. Basis note: real reg SD here 14.88 v gate 14.63 (different basis, OT/definition); LDRG reg SD 15.15.
- Within-possession (sum of squared possession margins) B +5.3, LDRG -8.4 [-11.7,-4.9]; points/possession var -0.2 [-.35,-.06]; possession-count var 9.88 v 10.75 (narrower, -.87 [-1.40,-.28]). So neither drive count nor points per drive carries the excess.
- Cross-possession part (VarM minus ss): B +9.0, LDRG +16.6 [+4.7,+28.8] P.99. This is the excess.
- By quarter: Var m1..m4 within about +-4 (m2 -3.9, m3 -3.7 sim lower). Q4 vs earlier covariances all too positive: Cov(m3,m4) +2.6 [+.6,+4.5], (m2,m4) +1.4, (m1,m4) +1.6; 2Cov(M3,m4) +11.2 [+5.1,+18.6] P.997 (B +4.9). Q4 beta on M3 -.047 v real -.074. Q1-Q3 serial covs m1m2 +1.0, m2m3 +2.1 (P.95), also positive.
- Lead at Q4 start (VarM|bin, contribution): 17+ +35.7 [+5.9,+72.3] / +3.8 (B +11.0); 8-16 +8.7 / +4.6 [+.7,+9.5] (B +3.0); <=7 -3.5 / -2.2. Leader Q4 change in 17+: -.69 v real -1.27; 8-16 -1.19 v -1.61 (leaders hold lead; intervals span 0).
- Total points var 179.2 v 191.8 (-12.7 [-22.2,-3.5]); -2Cov(H,A) +10.4 [+2.4,+18.1]: H and A more negatively correlated than real (-.12 v -.07 approx); mean pts 43.4 v 44.5.
- OT: tie share 5.5% v 6.2% (-.7pp [-1.6,+.1]); not a carrier (inferred).
- Noise v strength (gate numbers, reported): noise +6.3, strength +0.8; so about 88% noise.

## Tried
Same method as E131 on new base; no catch-up/serial correlation model run.

## Next
ONE mechanism: GLC (E137: TD only when the gain reaches the sim yardline, sim04_engine.py:998 and :1015-1033 copy points_off from the neighbour row at its own yardline; E137 found leader conversion excess 95% TD). Identity, no fit; verify agreement in real pbp 2009-17 by held-out season, smoke seed 31, targets Var(M) 221.3, xq cov, Q4 beta -.074, mean pts 45.21 (expected to fall, report not fill). Not verified that XFA/X2 does not already contain it.

## Open
Positive Q1-Q3 serial covariance and too-negative Cov(H,A) (total-points variance low) unattributed; unresolved_below_power.

## E168b Q4 lead-change decomposition (measured 2026-10-18; tests/scratch/e168/q4.py q4b.py q4c.py q4d.py, q4.txt q4b.txt; possession tables e131 R/S_ldrg; sim 2000-game boots 150-200, real game boots, 90%; ~150 cells one family; turnovers/onside flags not in table, not done)
- Q4 drives per side per game: sim lower in every bin (-.14 to -.20 [CIs exclude 0]); sim drives longer (+4 to +13 s); leader TOP 8-16 +8.6 [-5,+20].
- 17+ leader change -.69 v -1.27 = leader pts/g +.14, trailer pts/g -.44 [-.99,+.11] (n.s. each). Leader pts/drive after trailer non-score (punt/turnover) 1.31 v 1.10 (+.21 [+.02,+.37]) at equal start yl (64.4 v 64.3); trailer after non-score 1.46 v 1.63 (-.17 [-.42,+.07]) and starts 2.1 yd worse (+2.1 [+.8,+3.4]; share<=40 .055 v .072).
- E88 start shift present in Q4 for trailers after leader non-scores (17+), not after scores (+.2).
- Leader starts after trailer score: share<=60 .357 v .212 (17+, [-.20,-.10]), 8-16 .239 v .137; 1-7 .126 v .091; bulk >60 and <=40 match. After TD late (gsr<=600) 17+: .418 v .238, 40-60 band .367 v .183. Nearest-sqrt(N) pool draw expects .36 (q4c): kick tail exists in the pool but only ~half is realised in sim. Kick draw IS score-state conditioned (mod25e_kick.py:264-301, kd,t kernel; KGZ,OKK in base); mod25e_kos.py (KOS, off) flattens it. Punt/turnover returns: kNN on scaled score (sim04_engine.py:284-322, 398-433), score term refuted E81.
- Sign check (inferred): short fields for leaders raise leader scoring, so this gap alone does not explain leader change; leader conversion at equal start (above) is the bigger sign-matching term.

## Next (E168b)
1) Replay mod25e_kpath.py on current base: is the pool draw honoured late 17+ (realised .24 v expected .36)? If not find the overwrite, fix like KGZ (no constant). 2) Fitted score-state term in drive outcome at equal start (leader/trailer role, LOSO), unresolved_below_power until record.

## E168c Short-field overwrite found (measured 2026-10-18; tests/scratch/e168/kp2.py; full base label, s11-13; kpath.py LABEL is the OLD short label, so its stored kpath.txt is not this base)
- Draw honoured for onside: kicker-kept share after TD, kd<-16.5 t<600: sim .054 v draw .058 v real .049; kd -16.5..-8.5: .100 v .100 v .095. Not diluted, not the onside decision.
- Short field (next start <=60) kd<-16.5 t<600: sim .270, nearest-sqrt(N) draw expects .454 (n512 sub), real .420 (n143); kd -16.5..-8.5: sim .313, draw .553, real .539 (n243). Other cells match (t>=600 and kd>=-8.5 within .01).
- Overwrite: mod25e_stb.py:141 pol calls base (kick draw already made) then :158 `if u < prob(cls, y)` replaces next_yardline with a 75/80 spot for every flipped score kick; fit.json kick class is const, P=.504 for all y and state. Real touchback-spot (75/80) share in the late cell is .27-.30 (kd<-8.5 t<600 .267), v .464 all kicks. STB discards ~40% of the drawn short fields. 44% of sim kick starts are non-integer (STF z shift, stf.py:243-247, small).
- Fix (data definition, no constant): STB kick class must not resample what install_kick already drew (the pool row's nfp already contains 75/80 vs short, conditioned on score diff and time by KGZ/OKK). New default-off module mod25e_stbk.py wrapping install_stb: for cls kick, return drawn untouched when the kick pol drew it (po>=3 or pdf>=6, qtr<=4); punts keep STB. Deferred crH patch: call order unchanged, env STBK=1. Offline check: replay kernel draws over TD kicks s11-13 and compare short-field and TB shares in the cells above to real; then LOSO nothing to fit.
Next: build mod25e_stbk.py + patch, offline replay; sim run is orchestrator's.
