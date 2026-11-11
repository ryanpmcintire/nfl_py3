# fix-picks-u2-line-value

## Goal
Does "latest leader-book consensus line minus pool line", read through the discrete margin lattice (p_mkt), carry value the served stack (move + hook) misses? Parent: fix-picks-2026-11-10.md. Measurement only.

## State
Done 2026-11-10. Scripts tests/scratch/fixpicks/u2_b.py (backfill via scripts/pool_relative_move_forward_log.build, weeks 1-9), u2_c.py (analysis); outputs u2_backfill.csv, u2_games.csv (W3-9 per game).
p_mkt = band_read(history 2009-2025 closing line/result, line=pool, point=leader median, conditioning_line=leader median); cover+0.5*push. n=136 finished non-push games W1-W9.
New vs prior lanes: p_mkt is a lattice cover probability of the whole market-minus-pool gap in ONE term, graded on 2026 forward-chained; MOD-55/56/pool-hook used move/hook as separate linear terms.
Caveat: "card" = last feed final_probability side before deadline (W3-9 47-57), not published_picks (46-52 in earlier audit); z_base = served logit minus move terms (hook mass NOT removable, hook_rows not joined); second arm (2023-25 simulated lines) NOT run.

## Tried (measured)
- p_mkt side: all 75-61; W3-9 55-49; card all 70-66, W3-9 47-57. Differ W3-9: 34 games, p_mkt 21-13, card 13-21.
- By |leader-pool| gap: 0: 15-23 (card 20-18); 0.5: 34-17 (card 24-27); >=1: 26-21 (card 26-21). Gap>0: 60-38.
- Calibration p_mkt (pred/obs/n): <.40 .375/.25/4; .40-.45 .432/.280/25; .45-.50 .473/.468/47; .50-.55 .530/.457/46; .55-.60 .567/.571/14.
- Forward-chained logit (W3-9, 104 games): chain acc .529 LL .719 Brier .2624; icpt .558/.6932/.2500; srv card .490/.6996/.2531; coin .5/.6931/.25; unfitted p_mkt .529/.6842/.2456. Chain coefs per week (n,const,b_mkt,b_base): W3 32,.31,4.29,3.20; W4 .49,3.92,2.25; W5 .25,2.78,1.53; W6 .05,2.65,1.59; W7 -.04,2.49,1.41; W8 -.20,2.67,1.10; W9 -.08,2.32,1.08. Overfits at n<130.
- Paired week bootstrap (7 weeks, 20000): p_mkt minus srv LL better by .0154 [.0004,.0314] P(+)=.978; Brier .0075 [.0001,.0153] .975; acc +.0385 [-.060,.126] .81. Chain vs srv LL P(+)=.196.
- Rule take p_mkt side when |logit p_mkt|>|z_base|: changes 4,3,1,3,0,2,2 games W3-9 (15); changed games rule 9-6 vs card 6-9; rule W3-9 50-54 vs card 47-57.
- Looks: 3 decision families (p_mkt side, chain 7 folds x 4 models, rule) + 3 gap bins + 6 cal bins + 4 bootstrap contrasts x3 metrics.

## Next
Orchestrator: join hook mass to z_base; run arm 2 if pool-hook simulated lines exist; fit p_mkt as a term LOSO on 2020-25 simulated-late-line data before serving. Proposed record (verify syntax): nfl-ats weak-signals record --signal fixpicks_u2_pmkt_line_value --classification unresolved_below_power --effect-units accuracy_points (W3-9 differ 21-13, LL P(+) .978 vs served).

## Open
Gap=0 games (p_mkt side 15-23) show the lattice-only side has no value without a market move; value sits in games where the market moved (60-38).
