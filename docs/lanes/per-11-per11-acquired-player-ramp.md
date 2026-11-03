# PER-11 Acquired-player integration

## Goal
Ramp-up term after a mid-season trade or waiver arrival; 2026 deadline is the live case (ROADMAP PER-11). Family per11_arrival_rampup.

## State
Unit 1 done 2026-11-03. scripts/per11_arrival_rampup.py; artifacts/per11_arrival_rampup/ (arrivals.csv, events.csv, opener_rows.csv, results.json, batch.json). 10 cells recorded, unresolved_below_power.
- Arrivals (measured): 263 players, 205 team-arrival events 2013-2025, from snap-count team changes with prior-team snap share >= 0.5 (HIGH_SNAP_SHARE_THRESHOLD). Snap data starts 2013, not 2009. Arrival = first new-team game, so game 1 is not pregame-knowable here; pregame cells use games 2-3.
- Team EPA vs as-of baseline, arrival minus 3-game-shifted placebo: own side +0.0023 [-0.018,+0.022] P+ .60 (n 205); games 2-3 +0.0024 P+ .59; offense arrivals -0.002 P+ .47; defense arrivals +0.007 P+ .66. Parity halves disagree in sign. Vs as-of only: +0.011 P+ .91 (no dip, a small lift).
- Opener, 2020-25, 115 events: decisive team-games 311, model 162-149, picks on arrival team 77-73, against 85-76. Cover margin vs zero -0.31 [-1.77,+1.19] P+ .34; minus placebo -2.0 [-4.1,+0.18] P+ .035 (placebo games ran +1.7, noise).
- Step 3 not run: no EPA dip, no stable sign, so nothing to fit LOSO.

## Tried
Looks: 8 EPA cells + 2 opener cells = 10.

## Next
Date-honest game-1 cell and QB/OL/pass-rush splits need transaction dates (MOD-27 index covers deadline-month trades 2014+, 27 dated). 2026 deadline: 2026 index has 59 Oct-Nov trade slugs (e.g. 49ers acquire DT DeForest Buckner, Cardinals-Texans Marvin Harrison Jr. completed); not yet parsed or dated, and no EPA term exists to carry, so none carries a term in Weeks 10-12.

## Open
Position splits not run (small n). Not closeable: no control or resolved wrong sign.
