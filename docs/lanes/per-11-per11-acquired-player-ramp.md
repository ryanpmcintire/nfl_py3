# PER-11 Acquired-player integration

## Goal
Ramp-up term after a mid-season trade or waiver arrival; 2026 deadline is the live case (ROADMAP PER-11). Family per11_arrival_rampup.

## State
Unit 1 (2026-11-03, scripts/per11_arrival_rampup.py): snap-team-change arrivals 2013-25, no EPA dip (+0.0023 [-0.018,+0.022] P+ .60); 10 cells unresolved.

Unit 2 done 2026-11-21 10:56. scripts/per11_arrival_dates.py (parse 83, events 157 with runtime pregame assertion, term 225); artifacts/per11_arrival_rampup/dated_{candidates,events,drops}.csv, dated_results.json, dated_batch.json. Served base artifacts/pick_probability/20261120T171725Z.
- Dating (measured): Sep-Dec trade/waiver slugs dated by article datePublished, else earliest sitemap lastmod across all snapshots (never before publication, MOD-27). Every flagged game has tx_ts strictly before its Tuesday-noon ET prediction time; later nix slugs (Ebiketie CIN, Henley DEN) cancel only after their own date.
- 2026 Oct-Nov: 125 trade + 5 waiver slugs (index grew from 59), all dated, none undated; 91 rumor wording, 17 no completed pattern (hand-checked), 21 parsed completions, 12 starter arrivals (Jurgens BAL, Harrison Jr. HOU, Nelson SEA, Buckner SF, Simmons CHI, Brissett BAL, J. Davis JAX, Bates JAX, Hooker BUF, K. Allen SF, Downs KC, Cousins MIN).
- History game-1 events 2014-25: 95 (91 published-dated). Graded 2020-25 decisive, served-logit + free slope, LOSO, 200 within-season shuffles, planted ladder. 8 looks, family above, declared before outcomes.
- Game 1, all (45 flagged games): served 20-25, arrival team covered 23-22. Acc -0.40 [-0.93,+0.13] P+ .062, null pct 14.2; changed sides 5-11; LL -0.0012 [-0.0022,-0.0003] (free sparse term costs out of sample), null pct 5.0; fold coefs -0.05 -0.10 +0.07 -0.01 +0.19 -0.38.
- Games 1-2, all (90): served 50-40, covered 44-46; acc -0.20 [-0.80,+0.40] P+ .25, null 51.3.
- Pass rush g1 (11): covered 4-7 (exact one-sided .27), fold coefs all negative (-0.20 to -1.46), acc +0.07 P+ .65, null 74; g1-2 (21): 10-11, acc -0.20 P+ .12. OL g1 4 games, g1-2 8; QB g1 2, g1-2 4 (no power, ladder not reached).
- Detectable (planted, 80% power at P+ .90): all g1 0.21 logit/sd (+0.29 acc true), g1-2 0.26 (+1.17); standing add-one floor about 1.5 points.
- Recorded 8 cells unresolved_below_power (per11_dated_*_served, accuracy_points); pool loaded; shard now 18 signals.
- Week 12 2026 (informational, nothing served): KC at BUF both teams in game 2 (Downs, Hooker), net term 0; SEA at SF, SF game 2 (Keenan Allen), all-g1-2 coef -0.087 leans SEA; ATL at MIN, MIN game 2 (Cousins QB), QB cell coef +0.13 on 4 games leans MIN.

ROADMAP PER-11 update: 2026-11-21 unit 2: day-dated arrivals (95 history, 12 in 2026, all 130 Oct-Nov 2026 slugs dated); game-1 term on the served logit acc -0.40 [-0.93,+0.13] P+ .062, null pct 14, arrival teams 23-22 ATS; pass-rush g1 4-7 with stable negative fold sign on 11 games; 8 cells unresolved_below_power; nothing served. Next: EPA ramp curve on dated events by position group.

## Tried
Looks: unit 1 10, unit 2 8 (all/QB/OL/pass rush x game 1/games 1-2).

## Next
Dated EPA ramp by group (does pass-rush arrival defense dip in games 1-2?) using dated_events.csv; college transfer arrivals as a power source.

## Open
Not closeable: no control at this size, no resolved wrong sign (LL cost is a free-parameter cost, not a wrong-sign mechanism).
