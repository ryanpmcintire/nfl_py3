# mod25e E108 half-final drive start (Q2 clock-ended drives)

## Goal
Why Q2 clock-ended drives start at hs0 43.7 s v real 30.1 with 2.92 v 2.06 plays, Q2 last play <=5s .300 v .263, leader kneel-out start, Q4 .183 v .128. Base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghke (seeds 11-13). Name the mechanism; hook default off.

## State (2026-10-09, measured; scratch tests/scratch/e108/*.py,*.txt)
- Start time of the half's LAST possession matches (start.txt, hz.txt): Q2 hs0 quantiles 10/25/50/75/90 real 5/14/31/63/109 v sim 5/15/35/71/115; mean 50.3 v 54.2; prev-possession end mix near equal (punt .331 v .372, td .286 v .262); prev elapsed by end type within 1-10 s (td 152 v 141 in last-drive set). Share of last drives by hs0 bin equal (Q4 also). So start time is NOT the carrier.
- Gap is within the drive: given hs0 bin, sim last drive ends by clock more often (Q2 hs0 (60,90] .619 v real .404; (90,120] .542 v .291) and by FG less (.262 v .461; .331 v .548) at equal plays in the short bins. Play-level FG choice matches (P(FG|down 4,hs) .80 v .78; hs<=10 down1-3 .57 v .57), so it is not the decision. Sim reaches FG range less because it burns more seconds per play in the drive: secs/play 12.43 v 11.23 (burn.txt).
- Mechanism (measured): real elapsed is shortened when a timeout is used on/after the play (Q2 hs30-60 run 21.0 -> 8.4 s, pass 14.2 -> 8.6; defense timeout plays Q2 hs60-120 9.3 v 17.1 none). Sim elapsed is independent of off_to_used/def_to_used (lo_otu/lo_dtu): Q2 hs60-120 def-TO plays 19.4 s v none 15.4 (real 9.3 v 17.1); Q4 def-TO hs60-120 20.0 v real 7.3; lo2.py. CKH/CKS cells use timeouts REMAINING only; CDR fixes only offense TO in Q4 (fit scope q4_all); Q2 and defense timeouts uncovered. CKS Q2 cells are Q2-only (qs=1 splits Q4 from Q2); the "Q4-pooled" hypothesis is false.
- Hook CDT=1 (scripts/mod25e_cdt.py, label suffix dt after ke, wired in mod25e_crH.py after CDR): on code 0/1 plays in Q2/Q4 with hs<=hz and a timeout used, redraw elapsed from real timeout-play pool by (offense v defense timeout, hs bin). Fit artifacts/mod25e3/cdt/fit.json: LOSO 2009-17, 200 looks, best hz 120 edges w5 a 32 qs 0 mode rem: held-out ll per timeout play -2.116 v pooled-by-hs baseline -2.995 (+.880, 9/9 seasons); best el mode +.274 (9/9). Rem-mode ll is not strictly commensurable with el-mode baseline; el-mode is the clean number.
- Smoke seed 31 --worlds 1 --seasons 4 launched, workers 3, log artifacts/mod25e3/dt_smoke_s31.log, out dir artifacts/mod25e3/e5_crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkedt_s31. PIDs 38564 (parent) and 29360 (child) at launch (more worker children may exist; kill by PID/command line `mod25e_crH.py` seed 31 only). Env: NEG EGT EGH F2PR QBC OTY=2009-2017 KICK GZ DKF DK2 CLK SEL KN KNW FD4 CLK2 TDC KGZ OKK ADJ STF =1, YLM=1 YLM_CLASSES=punt WFG PATY FGD CDR STB =1 GFL=2 CKH=1 CKS=1 CDT=1.

## Tried
- Prev-possession end/elapsed as carrier: no. FG decision as carrier: no. CKS Q2 Q4-pooling: no.

## Next
When the smoke log ends with `done ... 31`: move artifacts/mod25e3/smoke/e5_..tbkhghke_s31 would stay (base); move the dt dir stays at top level; run `CLK_LABEL=crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkedt CLK_SEEDS=31 python scripts/mod25e_clock2.py 200 e108_dt_smoke.txt` (base smoke needs temporary move to top level for its own run, then back). Check: Q2 clock-ended hs0/n/hl, last<=5s, FG share by hs0 bin (tests/scratch/e108/hz.py, burn.py, lo2.py), secs/play, timeout-play el by type. If direction right, 3-seed run seeds 11-13 + bootstrap. None-play contamination: CKH pool for non-timeout plays still includes timeout plays (second step if secs/play now too short).

## Open
Leader kneel-out start (l9+ hs0 94 v 114) and Q4 .183 v .128 not yet examined for the timeout mechanism (Q4 gap in defense-TO plays is the same sign). Edited: scripts/mod25e_cdt.py (new), scripts/mod25e_crH.py (CDT wiring).
