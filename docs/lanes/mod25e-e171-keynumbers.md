# mod25e-e171-keynumbers

## Goal
Key-number profile sim v real and path decomposition of the mass-at-3 shortfall; name path, sim mechanism file:line, ONE testable mechanism with LOSO plan. Base XFA+LDR+RG label crHpq...faldrg s11-13 (burn 2, 39168 games) v real 2009-17 REG (2304 games).

## State (measured 2026-10-18, tests/scratch/e171/a1.py a1.log, a2.py a2.log; 127 looks, one family: 22 profile + 35 path + 70 4th-down cells)
- Profile s-r (90% game bootstrap): |M|=3 .1272 v .1471 -.0199 [-.032,-.008]; 14 -.0137 [-.022,-.007]; 7 -.0076 [-.017,+.003]; 10 -.0069 [-.014,+.001]; 16 -.0059; 21 -.0062. Sim excess off-lattice: 9 +.0070, 11 +.0068, 12 +.0079, 13 +.0080, 15 +.0118 (all P>0 >=.98). 6, 17 equal.
- mass3 path: OT games -.0089 [-.016,-.002]; regulation-decided -.0110 [-.022,-.0005]. Last score FG by winner: reg -.0105 [-.018,-.004], OT-FG -.0096 [-.017,-.003] (together = whole gap; TD/PAT/2pt-last cells zero). Reg-decided, tied entering last 5 min: -.0083 [-.015,-.003]; enter-120s |m|<=3 -.0112 [-.020,-.001].
- Late 4th down (q4, last 5 min), sim-real: P(FG) trailing 1-3 -.081 [-.126,-.044]; P(punt) trailing 1-3 +.110, last 2 min punt .097 v .004; tied P(go) .062 v .020; made FG/game trailing1-3 -.013, tied -.014, lead1-3 -.009 (sum -.035/game). FG make|att lower in sim in most cells (.82-.84 v .83-.92).

## Tried
Profile, path partition, 4th-down state table. Caveat: made FG in real = posteam_score_post-posteam_score==3 (no field_goal_result column); sim sd sign assumed offence-minus-defence.

## Next
One mechanism: late close-game 4th-down FG/go/punt choice (pol4 gradient-boosted classifier mod25_mechanisms.py:330-349, applied :426-433; mod25e_fourth.py:391 install_fd4 if enabled). Fit P(fg|go|punt) with game clock and one-score state features in last 5 min, LOSO by season, score held-out log loss and mass3 + FG attempts/game in states tied/trail1-3/lead1-3. Check first which of pol4/FD4 is live in this label and whether classes are overridden after the draw (punt .097 trailing 1-3 in last 2 min is a defect candidate). Also FG make rate by distance (separate mechanism).

## Open
Which 4th-down hook is live; mass at 9-15 excess is the mirror of missing 3/14.

## E171b (measured 2026-10-18; tests/scratch/e171b b1-b4 logs; ~12 looks: 5+4+2 model specs, 2 FG tables)
- Live: FD4=1 in run_ldrg_3seed.sh -> mod25e_fourth.py install_fd4 (HGB d6 leaf50, LOSO-selected from 40 specs; sees yl, dist, hs=secs left in half, sd clipped +-21, qtr, off/def timeouts, era). pol4 mod25_mechanisms.py:426 is NOT live for this label.
- Policy is applied faithfully in sim: FD4 model on sim states predicts punt .101 / fg .346 / go .552 (q4 hs<=120, sd -3..-1) v realized .113/.360/.527. So the sim defect is not an override after the draw; the model itself over-smooths: LOSO prediction on REAL rows in that cell: punt .074 v observed .005 (1/211), fg .376 v .441; sd -8..-4 hs<=120 fg .047 v 0.
- FG make rate: sim .8357 v real .8383 overall; by distance within .03 in every band (25-30 .944 v .949, 35-40 .865 v .839, 45-50 .701 v .711); real mix x real rate gives .8336. NOT a defect; cell gaps are small-n.
- Fit (scripts/mod25e_fd5.py, NEW): FD4 features + one-score score bins (lattice edges -8.5,-3.5,-.5,.5,3.5,8.5), trail/tie/lead flags, log hs; base HGB d8 leaf20 plus q4 hs<=300 specialist HGB d4. LOSO LL -0.1780 v FD4 -0.1801, 9/9 folds better (deltas +.0002..+.0040); in-sample gap of the d6 variant .026. Late q4 cell LL -0.2418 v -0.2576 (8/9 folds). Offline hs<=120 trailing 1-3: punt .031 v .005 real (FD4 .074); fg .413 v .441. Not fully closed; residual is 1 real punt in 211.
## Next (E171b)
- Owner/orch: apply crH patch after sim runs finish: mod25e_crH.py after the fd.install_fd4 block add `import mod25e_fd5 as f5; if f5.enabled(): f5.install_fd5()`; label `if FD5=="1": label+="fe"` near line 285. Run with FD5=1 and FD4 unset (never both). Fitted artifacts already at artifacts/mod25e3/fd5/ (model.pkl, fit.json).
- Pass bar v base label: mass3 shortfall -.0199 shrinks; FG made/game q4 last5 tied/trail1-3/lead1-3 (v -.014/-.013/-.009); P(final 3|tied last 5 min) toward real; P(punt) trailing 1-3 last 2 min toward .004 (was .097); guards: P(go)/punt overall shares, scoring mean/sd, 4th-down plays/game unchanged.

## E171c (measured 2026-10-18; tests/scratch/e171c/c1.py c1.log; 5 late-cell specs + 12 yl x dist cells)
- Residual punt .031 v .005 sits entirely in own territory (yl>40, 109 of 211 rows; model .03-.08 per cell, real 1 punt of 109, the lone punt at yl>80 dist>7). yl<=40 cells predict .002-.003 (correct). Wider hs<=300 trailing 1-3: obs punt .152 v FD5 .168, fg .408 v .393, so real teams do punt there; the miss is sparse-cell smoothing inside hs<=120, not a systematic defect. Expected punts in the 109 rows about 4-5 v 1 observed.
- Remedies tried (late specialist replaced, outer LOSO late-row LL): current HGB d4 -0.2418; logit C0.1/1/10 -0.407/-0.372/-0.365; HGB d2 leaf40 -0.299. None beats current in any fold (0/9). Logit C10 cell punt .023 but LL far worse. No change to mod25e_fd5.py or its fit.
- Conclusion: FD5 stays as fitted; remaining gap is unresolved_below_power at 1/211 (not recorded via weak-signals; no mechanism claim).
