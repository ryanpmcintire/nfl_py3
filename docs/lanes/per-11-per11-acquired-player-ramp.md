# PER-11 Acquired-player integration

## Goal
Acquired-player integration: games since a trade or waiver arrival as a ramp-up term, with the trade-deadline week as the live case (ROADMAP PER-11).

## State
Added 2026-11-03, unmeasured. Players added by trade or waiver claim learn a new playbook while the market prices their name. If the first games after arrival run below the name value for QB, OL and pass-rush arrivals, the opener overstates the acquiring team; the 2026 deadline is now, so a forward log can start immediately.

## Tried
Nothing yet. Novelty: MOD-27 (day-level transaction dates, a leakage repair of deadline timing, not arrivals as a signal), LEAD-55 (participation reports), registry availability_experiments (injury availability). Grep of ROADMAP, docs/lanes incl. done, docs/*.md and registry/weak_signals for 'trade deadline' returned no hit.

## Next
First unit: Build the arrival table 2009-2025 for starters by prior-team snap share, track new-team snap share by games since arrival, and test the acquiring team's opener margin residual in games 1-3 after arrival against matched non-arrival games, with the trade date shifted four weeks as placebo and a split-half by season parity.

## Open
Data: nflverse trades (2002 on), snap counts (2012 on) and weekly player stats; reported available, schema to be read in the first unit. Zero crossing closes nothing; record unresolved results with a closing ground.
