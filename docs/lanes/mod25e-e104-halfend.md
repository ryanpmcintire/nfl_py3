# mod25e E104 half-end clock

## Goal
Q4/Q2 last play starts <=5s: base tb Q4 .309 / Q2 .401 v real .128 / .263. Measure on gf seeds; test whether Q2 half-ending rows carry the same possession-flip labelling defect as game-ending rows; fix as GFL=2 (default off) and smoke.

## State (2026-10-08, measured)
- Scores on gf seeds 11-13 (artifacts/mod25e3/clock2/clock2_xfdcdtbgf_e104.txt, CLK_LABEL=...tbgf, clock-ended drives): Q4 <=5s real .128 v tb .309 -> gf .258 (n 33738); <=15s .289 v .472 -> .439; <=30s .586 v .735 -> .728. Q2 <=5s real .263 v tb .401 -> gf .409 (GFL does not touch Q2).
- Q2 defect confirmed (tests/scratch/e104/q2.py, real_fit.parquet): Q2 run/pass rows with next_qtr==3 (n 896) have flip True .516 (non-ending rows .064; flip without score .487 v .024), term class .537; all have el==hs. Flip is a coin toss (home posteam: flip .53, away .45) set by the second-half kickoff, not behaviour. Q4 game-ending rows: flip 1.000.
- Q2 last-play decomposition (q2dec.py, gf seeds, clock-ended Q2 drives, P(last play starts <=5s) real v sim): kneel .297 v .383 (share .478 v .557); pass .362 v .526; run .118 v .295 (share .281 v .165); gap present in every score bucket (.20-.33 v .35-.46), timeout bucket (.09-.36 v .29-.53), field zone, and start-time bucket; not a single state.
- Hook: GFL=2 in scripts/mod25e_gfl.py (also relabels qtr 2 rows with next_qtr 3), label suffix gh in mod25e_crH.py, fits to artifacts/mod25e3/gfh/. GFL=1 unchanged.

## Tried
- Refit GFL=2 done (artifacts/mod25e3/gfh/fit.json, fit2.json, 4374+108 looks). Rows: term 11669->11233 (-436), run +224, pass +205. Best LOSO ll gfl -> gfh: term -2.8442 -> -2.8214, run -3.2688 -> -3.2681, pass -3.4608 -> -3.4562, inc/kneel/spike equal (row sets differ, so differences are indicative only). Q2-vs-Q4 split chosen (qs=1) for run/pass/term in gfh.
- Smoke s31 GFL=2 1x4 (artifacts/mod25e3/e5_..tbgh_s31, 554 s, scores clock2_gf_s31_e104.txt, clock2_gh_s31_e104.txt): Q2 <=5s gf .401 -> gh .374 (real .263, n 367/382, CI half-width ~.05); Q4 .310 -> .264 (real .128). Smoke noise is the size of the move.

## Next
Orchestrator decides: 3-seed GFL=2 run (seeds 11-13, ~2 h each) and bootstrap, or stop. Q2 kneel (share .557 v .478) and pass/run timing gaps persist in every state bucket; next suspect is play choice/hurry in the last drive of Q2.

## Open
Kneel and pass last-play timing gaps are not label defects; they stay if the relabel does not move them.
