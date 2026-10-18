# mod25e E106 clock after runs (censored draw)

## Goal
Why the sim leaves 1-2 s after runs at hs 10-45 (real .008-.04 v sim .07-.15) and ends the half too rarely (real .24-.43 v sim .12-.32). Base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbgh (GFL=2), seeds 11-13. Test the right-censored pool as the carrier.

## State (2026-10-08, measured)
- Task 1 audit: CKC/KM scoring was already censored (mod25e_ckc.py fold_rows: uncensored rows log pm[el], censored rows log P(el >= ceil(hs)) = tail; pool mode puts censored training mass at its own hs bin, which is what the sim replays). So the earlier loss was not a point-density artefact. Rescored on GFL=2 rows (gfh spec), LOSO 9 seasons, hs<=45 (tests/scratch/e106/score.py, score.txt, dec.py): ll pool / KM / CP (CP = censored rows redistributed over the class-level KM pmf beyond hs, a smoothed KM): inc -1.923/-1.931/-1.931; term -2.471/-2.467/-2.468 (+.004/+.003, 5-6/9); run -2.439/-2.530/-2.504 (KM -.091 se .025, 1/9; CP -.065 se .022, 2/9); pass -2.580/-2.572/-2.568 (CP +.0125 se .006, 7/9); spike -1.751/-1.800/-1.791.
- Run ll loss sits in uncensored rows at hs 20-45 (diff*share -.046/-.091) while censored rows gain (+.049): consistent with hs-dependent elapsed breaking KM's independent-censoring assumption inside wide cells (inferred, not tested).
- Behavioural targets (held out by season, observed v pool v KM v CP): P(end) run (20,30] .378/.315/.387/.384; (30,45] .256/.199/.253/.254; (10,20] .477/.509/.473/.470; pass (10,20] .181/.257/.187/.190. P(1-2 s) run (20,30] .012/.110/.017/.019; (30,45] .008/.074/.013/.012; (10,20] .040/.113/.047/.049; pass (0,10] .123/.223/.124/.125. Brier for end and near improves in every run/pass bin (e.g. run (20,30] end .241 -> .229, near .0236 -> .0118). So the pool replay of clipped half-ending rows is the carrier of the 1-2 s surplus (measured).
- Task 2: hook CKH=1 (scripts/mod25e_ckc.py CP tables, classes run+pass only, label kh before gh; fit artifacts/mod25e3/ckc/fit_h.json). Smoke seed 31 1x4 launched, label crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhgh (log artifacts/mod25e3/kh_smoke_s31.log).

## Tried
- KM draw (CKC) on run lost ll; CP reduces but does not remove the loss. ll is a poor yardstick for the half-end target (bulk elapsed dominates).

## Next
Score smoke v artifacts/mod25e3/smoke/e5_..tbgh_s31 (tests/scratch/e106/post.py, sc31_kh.py, clock2 as script with CLK_LABEL); then orchestrator decides 3-seed run and weak-signals record for the run ll result.

## Open
Run ll loss is not a closing ground (no resolved wrong sign on the target). No processes left by this unit once the smoke exits.

## Smoke result (2026-10-09, measured; seed 31, 1x4, base ..tbgh v kh ..tbkhgh; tests/scratch/e106/post.txt, sc31_kh.txt, artifacts/mod25e3/clock2/clock2_*_e106.txt)
After run P(end)/P(1-2s) (real .425/.040, .337/.008, .233/.006 for hs 10-20/20-30/30-45): base .317/.129, .175/.140, .105/.072 -> kh .433/.062, .278/.015, .203/.021. Pass changes small (10-20 end .124 -> .158, real .160). Snaps hs<=1 per 1000 halves real 42 base 136 kh 62; hs(1,2] 75/97/72; spike<=2 6.1/15.6/4.6. Last play <=5s: Q2 real .263 base .374 kh .305; Q4 real .128 base .264 kh .202. Margin SD 16.61 -> 15.68 (real 14.63), noise 189 -> 191, strength 86 -> 54, mass3 .140 -> .131 (.141), P(final3|tied last 5) .48 -> .375 (n 16-24, noise). Smoke 1x4 is noisy (kh dir left at top level; base smoke copy removed).
Next: orchestrator decides 3-seed CKH=1 run (seeds 11-13) and bootstrap; record run ll loss via weak-signals.

## 3-seed result (measured 2026-10-09; artifacts/mod25e3/e106_kh_boot/e100.txt, clock2/clock2_xfdcdtbkhgh_e106_3seed.txt)
Clock-ended drives, last play <=5s gh -> kh (real): Q2 .363 -> .270 (.263; diff +.006 [-.014,+.026], closed), <=15s .673 -> .602 (.605); Q4 .241 -> .212 (.128, still +.084).
Aggregates (season bootstrap 90%): SD +.02, noise +.3, strength +.7, mass3 -.001, Q4 slope flat, pts -.20 [-.49,+.05] toward 45.21, late r2 +.002, P(final3|tied after reg) flat, P(final3|tied last 5) -.014 [-.043,+.012].
KEPT. New base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhgh (base env + ... STB=1 GFL=2 CKH=1). Run held-out ll loss (-.065, 2/9) logged here as a lane result; generator hooks are judged in lanes, not the pick-signal registry (precedent E91-E104).
Next: Q4 remaining gap (.212 v .128): clock-ended Q4 drives start later (hs0 80 v 86) and shorter (hl 21 v 25) with more plays (3.67 v 3.45) from a worse spot (yl0 67.6 v 64.1); then P(final3 | tied last 5) .451 v .625.
