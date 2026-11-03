# UI-34

## Goal
Half-point games: plain line on where the crowd leaned (ROADMAP UI-34).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: POOL-08 measured the field taking the half-point side more often (+7.7 pp). Readers entering a pool want to know whether the crowd piled on the half-point side this week. Display only the week's share, no edge claim. Data: pool_observables captures (local).

## Tried
Novelty check only. Nearest: UI-31 (hook chip), UI-28 (history week strip). Differs: crowd share on hook games only.

## Next
First unit for one agent: Add one plain sentence to the settled-week view stating how many half-point games the crowd took the half-point side on, only for captured weeks.

## Open
Rules: Display only measured claims, in the current design system and plain words; no ids, hashes or research jargon in reader text; no side rule. Look count and family declared before any sign is seen.

## Unit 1 done 2026-11-03T17:31
History week chip now adds one sentence on half-point games (count, crowd share of picks, covered, ours). Code: board_site_content.py _half_point_week_texts + SeasonSoFar.half_point_week_texts; board_terminal.py _week_strip_html. Source: data/splash/field/2026_field_best_picks_w*.parquet + Tuesday market (pool_hook). 44 games over W1-8 matches POOL-08. Built in memory, not published. Next: root publishes; check 390 width.
