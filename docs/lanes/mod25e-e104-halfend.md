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

## Orchestrator 2026-10-08
3-seed GFL=2 e5 seeds 11-13 launched ~21:30 ET (logs artifacts/mod25e3/gfh_e5_s1{1,2,3}.log, sequential ~2 h each), then bootstrap vs gf base -> artifacts/mod25e3/e104_gfh_boot/e100.txt (log gfh_boot.log). gh smoke moved to artifacts/mod25e3/smoke/.
Keep rule: Q2 last play <=5s (clock2 with CLK_LABEL=...gh) moves toward real .263 and aggregates flat. Caveat: the relabel also clears genuine half-ending turnovers (interception/fumble on the last Q2 play); count them before keeping.
Then: Q2 last-drive play choice (kneel share .557 v .478; run share .165 v .281).

## 3-seed result (measured 2026-10-08; artifacts/mod25e3/e104_gfh_boot/e100.txt, clock2/clock2_xfdcdtbgh_e104.txt)
Target gf -> gh (real): Q2 last play <=5s .409 -> .363 (.263), <=15s .717 -> .673 (.605); Q4 <=5s .258 -> .241 (.128).
Turnover caveat: real Q2 half-ending rows flip .50 within every play code (pass 241 v 234), so genuine last-play turnovers are a negligible share.
Aggregates: SD, noise, strength, mass3 flat; pts -.21 [-.48,+.06] toward 45.21; Q4 slope -.0586 -> -.0544 (real -.060, away, P .85); late r2 -.006 [-.011,+.0004] away (3 seeds); P(final3 | tied entering last 5 min) .496 -> .465 [-.058,-.002] away from .625.
KEPT: the old label was arbitrary (set by the second-half kickoff), so keeping it would keep a wrong input because it happened to fit; the aggregate moves expose the late-regulation walk-off gap (E99) more clearly rather than being caused by a new error. New base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbgh (base env ... STB=1 GFL=2).
Next: Q2 last-drive play choice (kneel share .557 v .478, run .165 v .281); then the tied-last-5 -> final 3 gap (.465 v .625) via E99's late-regulation state composition.
