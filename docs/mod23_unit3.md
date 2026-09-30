# MOD-23 unit 3: opponent-adjusted efficiency inputs on the base model alone

Script `scripts/mod23_unit3.py`; outputs `artifacts/mod23_unit3/report.json` and per-arm per_game parquet. Numbers **measured** this session.

## Prior work (read, not redone)
- **read**: served offense (21) and defense (18) columns are raw span-8 EWM of per-game team stats (`features.py` build_team_states/attach_team_states); defense is the opponent's raw offense against the team. Neither is opponent-adjusted.
- **read**: the PBP opponent-adjusted profile (`pbp_adjusted`, PBP-05, `opponent_adjustment.py`) was tested only as an addition on PBP columns and did not help (`docs/modeling.md` ~line 102: Brier 0.25993 vs 0.25944 raw, worse). **read**: `docs/cfb_opponent_adjustment.md` ran the dimension-neutral substitution on CFB (market-residual MAE -0.0003 pts, P 0.463; a deliberate future-leak arm gained only +0.013, so the family ceiling there is tiny). Neither is an NFL base-model accuracy test at the opener; this unit is that test on the served 2020-2025 protocol.

## Construction (leak-free by construction)
For each team-game, raw metric minus (opponent's served pre-game state of the opposing metric minus the league mean of that metric over team-games with a strictly earlier gameday). The opponent state is the served pre-game column (games strictly before the current game); the league mean uses only strictly earlier days; missing states adjust by 0. Adjusted per-game values then pass through the identical `build_team_states`/`attach_team_states` (span 8, same retention). Six offense/defense pairs adjusted (EPA, pass EPA, rush EPA, yards per play, turnover/takeaway, sack); CPOE has no counterpart and stays raw. One-pass additive adjustment (the opponent state itself is raw). Raw-state reproduction of the served parquet columns on REG games: max abs diff 0.0.

## Arms (3 looks, one family `mod23_unit3_opponent_adjusted_inputs`), ridge alpha 10, all other served columns kept
(a) replace_adjusted: 36 of the 39 offense/defense columns swapped for adjusted (90 columns). (b) add_adjusted: served set plus 36 adjusted (126). (c) compact_net: drop all 39 offense/defense, add adjusted net EPA, pass EPA, rush EPA (home/away/diff, 60 columns).

## Results (baseline 802-701, LL 0.6969, Brier 0.2517; even 0.6931/0.25; 107 season-week blocks)
| arm | record | diff (pts) | 95% CI | prob_positive | LL | Brier |
|---|---|---|---|---|---|---|
| replace_adjusted | 790-713 | -0.80 | [-2.65, 1.05] | 0.181 | 0.6975 | 0.2520 |
| add_adjusted | 780-723 | -1.46 | [-3.24, 0.33] | 0.051 | 0.6992 | 0.2528 |
| compact_net | 785-718 | -1.13 | [-3.20, 0.81] | 0.131 | 0.6944 | 0.2506 |

Per-season diff (2020..2025): replace -0.9,-0.4,-3.6,0,-0.8,+0.7; add -5.0,-1.3,-4.0,+1.1,-2.3,+1.9; compact -0.5,+0.8,-4.4,+1.5,-5.3,+1.1.

## Verdict
No arm beats the baseline record. compact_net improves log loss (0.6944) and Brier (0.2506) over baseline, but LL 0.6944 is still above market-even 0.6931 and Brier 0.2506 above 0.25; the gain does not convert to picks. All three recorded `unresolved_below_power`; no closing ground. Nothing served changes.
