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
