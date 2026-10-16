# MOD-23 unit 6: availability-matched history (acc-u7 survivor 1)

## Goal
Test whether each past game's contribution to the rolling team inputs, adjusted for its own and the opponent's starting-QB availability on that game day, raises the base model alone on the fair opener test (baseline 802-701 on 2020-2025; 2011-2019 proxy era graded at the SBR open).

## State
Declared before scoring (2026-10-16). Script scripts/mod23_unit6.py (scripts only, monkeypatches F.build_team_game_metrics like mod24_u7b, grades with the mod24_u1 harness). Outputs artifacts/mod23_unit6/.
- Availability source: nflverse snap_counts (2013+, completed games only). Flag = the game's top offense-snap QB differs from the team's established starter before that game (most cumulative snaps in earlier games of the season; week 1 uses last prior-season game). Flag 0 before 2013.
- Adjustment: m_adj = m - b_own*own_flag - b_opp*opp_flag, b from a within-season-demeaned regression over seasons strictly before the game's season (2013+); b = 0 when no prior flagged seasons. The current game's lineup is unchanged.
- Arms (2): A both flags; B own flag only.
- Looks: 2 arms x 2 eras (2011-2019, 2020-2025) on accuracy = 4 cells; LL/Brier/decisive reported alongside, not extra looks.
- Leak check: coefficient seasons < game season asserted; flags zeroed after a cutoff leave earlier games' features identical.

## Tried
Done 2026-10-16. Leak check passed (zeroing flags after 2022 wk10 leaves earlier features identical; later cells differ). Base reproduced 802-701. Coefficients (A, 2020): QB-change own -0.082 off EPA/play, -5.4 pts. 2020-25: A 794-709 (-0.53, [-2.01,0.95], P+ 0.22, LL 0.6973 vs 0.6969); B 808-695 (+0.40, [-0.25,1.27], P+ 0.85, LL 0.6972). 2011-19: A -0.18 (P+ 0.30), B +0.40 (P+ 0.78). B pooled 1959-1775 vs 1944-1790 (+0.40, [-0.08,1.03], P+ 0.93); changed picks B 75-60 pooled (40-34 in 2020-25). 4 cells recorded unresolved_below_power (family mod23_unit6_availability_history).

## Next
Log arm B picks beside served model weekly; a pre-registered extension (RB/OL availability from snap counts) would be a new family.

## Open
Raw rebuild differs from the features parquet on ~5% of games (deltas applied additively, so inherited). Flags need snap_counts, 2013+ only.
