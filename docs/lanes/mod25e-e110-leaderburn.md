# mod25e E110 leader clock burn (Q4 2:00-5:00 left, offense up 9+)

## Goal
Find which play-level behaviour of leaders (up 9+, Q4 drives starting 2:00-5:00 left) the sim gets wrong (E109: clock-ended .185 v real .282, punt .511 v .435; lead 17+ extension .191 v .152). Fitted LOSO hook, default off. Base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkeku, seeds 11-13, smoke seed 31.

## State (2026-10-09, measured; tests/scratch/e110/{pl,el,fit.log,choose.log}.txt)
- Not wrong: run share (lead 9+ hs 120-300 .846 v real .839; lead 17+ .859 v .862), kneel/punt/fg shares, first-down rate (.199 v .193), turnover rate, timeout-use rate (P(def to used) l17+ .168 v .145, l9-16 .350 v .396), incomplete share (.047 v .055).
- Wrong: elapsed of non-timeout run plays, Q4 hs (120,300]: mean el run 23.6 v real 35.4 (l17+), P(el<=12) .46 v .135, median 15 v 41; pass 16.4 v 20.7. Same in l9-16 and l1-8 (not leader specific; real non-used run el 33-35 in every band). Starts at hs ~420 (P(el<=12) run l9+ 300-420 .131 v .048; 240-300 .30 v .11; 180-240 .42 v .10). Sim secs/play lead 17+ 24.3 v 30.4.
- Mechanism: pools for CLK/CLK2 elapsed (class x hs bin x score sign x timeouts remaining, not the used flag) mix timeout-stopped plays (short) with snap-to-snap plays (play clock runs, ~40 s); the sim draws timeout use separately, so used plays get long draws and non-used plays get short draws. CKH/CKS/CKU cover hs<=45 only; CDT=2 (E108b) covered hs<=120 without sign, rejected before CKU existed.
- Hook CDW=1 (scripts/mod25e_cdw.py, label suffix dw after ku, wired in mod25e_crH.py after CDT): on Q2/Q4 code 0/1 plays with 45<hs<=420 redraw elapsed from real pool (offense/defense timeout used, class, 20 s hs bin, score sign, Q4 split), shrink a=128 to (used, class, hs bin) marginal. Fit (artifacts/mod25e3/cdw/fit.json, 540 looks): LOSO 2009-17 held-out ll per play -3.0602 v -3.1547 pooled (class x hs bin) (+.0945, 9/9) and v -3.1362 CLK-like (sign, no used flag) (+.0759, 9/9). Held-out by season (lead 9+, Q4 hs 120-300, non-used run) mean el pred 33.7 v act 34.4 (CLK-like 26.2); P(el>=38) .59 v .64.
- Lead-size band (5-level) did not beat sign: no lead-size term in the hook; the lead effect enters through the used-flag rate, which the sim already matches.

## Tried
Play mix, first-down, OOB (not in data), kneel start: no gap found. Lead conditioning: no LOSO gain.

## Next
Smoke s31 1x4 (script tests/scratch/e110/run_dw.sh, log artifacts/mod25e3/dw_smoke_s31.log); score v smoke/e5_..keku_s31 with tests/scratch/e110/{drv2,lead,sc31_dw}.py (CLK_SEEDS=31; move run dirs to top level and back). Targets: Q4 l9+ drives starting 120-300 clock-ended .282, punt .435, score .157; leader-signed change 5:00->2:00 lead 17+ real -.407, P(extend) .152. Then 3-seed run + bootstrap if direction right.

## Open
Q2 same pool defect (hook includes Q2). Aggregate effect on noise/SD unknown at 1x4.

## Smoke s31 1x4 result (measured 2026-10-09; tests/scratch/e110/{drv2,lead,sc31}.txt; ku dir back in smoke/, dw dir at top level)
ku -> dw (real), Q4 offense up 9+, drives starting 2:00-5:00: clock-ended .158 -> .247 (.282); punt .570 -> .500 (.435); score .158 -> .133 (.157); plays 5.09 -> 4.94 (4.89); first-last dur 97.8 -> 108.2 (105.4). Lead 17+ 5:00->2:00: leader-signed change +.153 -> -.267 (real -.407); P(extend) .215 -> .157 (real .152). Lead 9-16 change -1.00 -> -.49 (real -.465). All toward real; n small (one seed, 4 seasons).
Aggregates 1x4 (noisy): SD 16.42 -> 16.30, noise 192 -> 201 (165), strength 77 -> 65, mass3 .108 -> .151 (.141), P(final3|tied) .52 -> .45 (.625, away), q4 slope -.015 -> -.017.
Next: 3-seed run seeds 11-13 CDW=1 + bootstrap (E100_BASE/E100_CAND); decide keep on behavioural targets toward real; noise/final3 are exposed gaps to read on 3 seeds.
