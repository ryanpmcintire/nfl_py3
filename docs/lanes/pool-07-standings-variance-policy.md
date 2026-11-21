# POOL-07

## Goal
Standings-conditional variance policy: how far ahead or behind we sit changes which picks to take (ROADMAP POOL-07). Challenger in the pool simulator only; one fitted cover probability selects every served side.

## State
Unit 2, 2026-11-21 10:05 local (fitted per-game leverage cut, leave-one-week-out), measured and recorded. `scripts/pool07_standings_policy.py --mode lowo --paths 10000 --chunk 2000 --weeks 8 --points 77 --rank 90` -> `artifacts/pool07/pool07_lowo_leverage_w10.json` (the quartile mode and its Unit 1 artifact are unchanged; `--mode quartile` is the default).
- Start (read, `artifacts/pool_rank_replay/weekly_points_field.json`, Week 10 read): our total 77 (z 0.413), 89 entries above, rank set to 90 (13 entries on 77; tie order by Best Picks simulated), 8 weeks left, paid places 1 (rules observation). Entrant skill split-half 0.019, so field spread is nearly pure luck (dsd 0.005).
- Rule: per game, leverage ratio (2p-1)/(2share-1) on the side the model favours when the field also favours it; when trailing the leader before the week, take the less-owned side if the ratio <= lambda. Lambda is chosen on the other nine weeks' games (candidates are every training ratio, so no grid constant) by simulated first-place chance, then scored on the held-out week's games with common random numbers against max-probability picks.
- Held-out change in first-place chance: **+0.54 pp [0.32, 0.77] (10-fold bootstrap), P+ 1.000**, 9 of 10 folds positive (Week 2 fold tilts no held-out game, exactly 0). In-sample +0.76 pp, gap 0.22 pp. Per-fold simulation SE 0.12-0.20 pp. Lambda per fold 0.076-0.286 (median about 0.165); it tilts 15-26 of about 135 training games and 0-4 held-out games a week. Max-probability first-place chance per fold 0.6%-8.2% (depends on the week's game mix).
- Implies: inside this simulator, trailing by 12 with 8 weeks left, taking the less-owned side of near coin flips the field leans on raises the first-place chance by about half a point out of week. Card unchanged: this is a pool-objective challenger, not a fitted term in the served probability, and Unit 1 showed the sign depends on the field model (Weeks 1-4 shares hurt, 5-8 helped). Recorded `pool07_lowo_leverage_tilt_first`, payout_first_pp, `unresolved_below_power` (interval is over week game pools only; field-model and start-state uncertainty are not in it).

## Tried
Unit 1 (2026-11-03): quartile cuts from the arm's own |p-0.5| (in-sample), 24 looks; light tilt +0.51 pp, heavier tilts lower it; 20 cells recorded 2026-11-14. Unit 2 looks: one held-out cell; inner selection over 59-64 candidate cuts per fold x 10 folds. Expected-wins cost not measured in lowo mode.

## Next
1. Add the field-model arms (week 1-4 v 5-8 shares, no-skill) to `--mode lowo` and report each held-out change; that is the uncertainty the interval omits.
2. Report expected wins and top-10 change for the chosen lambda per fold (cost of the tilt).
3. Rerun each week with the new standings (`--points`, `--rank`, `--weeks`).

## Open
- Behind check uses standings before each simulated week (Unit 1 used after); trailing is true in nearly all paths, so conditioning is weak here.
- Zero crossing closes nothing; no side rule enters the card; no entrant names stored.
