# best-pick-chain-audit-2026-11-08

## Goal
Owner called the Best Pick choices terrible after Week 9 (Sunday star NO -2.5 lost 23-22 OT; Tuesday star DEN -2.5 won 26-21). Grade every layer of the star chain on the live 2026 season, fix the defects that do not need fitting, and start the measurements that do. Done = defects fixed and published, measurement units recorded.

## State
Measured 2026-11-08 22:13Z, scratch script `bp_audit.py` (session scratchpad; rebuilds from `artifacts/clv_ledger/published_picks.parquet`, last row before deadline, graded at the pool line, Week 9 from ESPN finals). Card 69-61 W1-9 (board says 66-56 W1-8; reconstruction 65-56, one game differs); W3-W9 46-52.
- Stars: Tuesday nominee 5-4; star the card served at lock 4-5. Sunday re-nomination changed the star in W2, W3, W6, W8, W9; outcome changed in 3: W3 L->W, W8 W->L (CIN -10.5), W9 W->L (NO).
- Served cover chance ranks nothing on 2026: weekly top 4-5, ranks 2-3 11-7, 4-6 14-13, 7+ 40-36; top minus rest -9.3 pts, within-week permutation P(>=obs) 0.80. POL-09 already found the ordering flat over 107 archive weeks.
- Calibration: served 0.50-0.55 50.7% (pred 52.4, n=67); 0.55-0.60 53.1% (pred 57.0, n=49); 0.60+ 9-5 (pred 62.7). Log loss 0.6895 vs coin 0.6931.
- Raw base model W4-W9 (n=82): log loss 0.7018 (worse than coin); its weekly top pick 1-5.
- Pool hook: for pick 19-14, against 6-10, none 44-37. Not the defect.
- Close (nflverse) vs pool line: moved toward pick >=1 7-6; moved against >=1 22-11; flat 40-44.
- Overlays: explanations still list complement "firings" (coach fade 2-11 W4-W8, bye 2-4) but `card_view.py:445-455` feeds the raw model probability to the stack; firings are display-only since the 10-25 coach fix.
- Defect A (read): history page flags the Tuesday nominee from the paper ledger (`board_site_content.py:1612`), not the served star in `pick_revisions.best_pick_after` (`pick_refresh.py:317`). Week 9 history shows DEN, card and index show NO.
- Defect B (read): Sunday re-nomination pool is every playable game (`best_pick_renomination.py:273-275` pool_pass True, no spread cap), while the board says the star comes from spreads of 6.5 or less.
- Defect C (read + measured): move term sums per-book diffs from Wednesday on (`sharp_book_movement_features.py:136-140`), so it skips moves between the pool-line capture (Tue noon) and Wednesday. CLE at NO quotes: 2.5 Tue 12:10, 1.0 Tue 14:27, 3.0 Wed, 3.0-3.5 Sun; the term saw 2.0 toward NO against 0.5-1.0 versus the pool line.
- Defect D (lane week9 item 4): explanation text says "Situational adjustment fired: bye-week rest edge" on a pick no overlay chose.

- FIXED A: `board_site_content._served_stars_by_week` resolves each week's star like `locked_best_pick` (served star overrides the paper flag). Scratch render: history now stars W2 NO +8.5, W3 CHI +3.5, W6 CLE +4.5, W8 CIN -10.5, W9 NO -2.5; season star record 4-4 through W8.
- FIXED D (subagent, lane explanation-counted-terms.md; wording fixed by root): CLE at NO now reads "the betting line has moved toward NO since midweek, worth about 14 points of cover chance". Measured: the move term alone took NO from ~53% to 67.1%.
- FIXED B text: glossary (`board_assistant.py` Best Pick) now says Sunday can move the star to any playable game.
- C measured (MOD-55, lane mod-55-move-anchor-at-pool-line.md): Tuesday-noon anchor is a dead heat on held-out 2023-25 (log loss diff -1.3e-5, P+ 0.497; weekly top pick 32 of 54 vs 33 of 54). Anchor is not the lever; CLE_NO's served move was 2.5 from 2 leader books. Week 5 2026 had 0 leader books at serve time.
- Test `test_played_card_expectation` was failing since 7ec3b8f7 (plain-English rewrite dropped "reused data"); assertion updated to the new wording.

## Tried
- Grading `market_spread` as pick-side was wrong: it is home-favoured-positive; fixed in the script.

## Next
1. Republish (publish-predictions, no recording) running; diff the card table, review rendered diff, commit, push.
2. In flight: MOD-56 leader-book depth in the move term (lane mod-56-leader-book-depth-in-move-term.md); POL-13 Sunday re-nomination rule, four predeclared star rules out of season (lane pol-13-sunday-renomination-rule.md).
3. card-ledger-check compares sides only; extend it to the star (ENG row).

## Open
- Which star the owner actually entered in Splash for Weeks 2, 3, 6, 8, 9 is not recorded anywhere.
