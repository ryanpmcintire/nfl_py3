# PER-13 Week 17-18 resting hazard

## Goal
Week 17-18 resting hazard: clinched or locked-seed teams and the probability a starter plays few snaps (ROADMAP PER-13).

## State
Added 2026-11-03, unmeasured. When a seed is locked, coaches rest starters; the opener posts on early information and the real lineup is known only Saturday or Sunday. A team-and-coach hazard of reduced starter snaps, fitted from standings, is pregame information that can enter the PER-10 scenario mixture and the pick.

## Tried
Nothing yet. Novelty: SIM-09 (stakes and urgency inside the simulator; its catalogue mentions resting starters only as a state), PER-10 (scenario mixture for ordinary inactives). Grep for 'resting starters' hit docs/sim09_personnel_catalogue.md only.

## Next
First unit: Reconstruct clinch and seed-lock status from schedules before Weeks 17 and 18 for 2012-2025, measure starter snap share versus the team's own prior weeks by lock status, then test the margin residual against the opener in locked games with a leave-one-season-out hazard.

## Open
Data: nflverse schedules and results (reconstruct seeds), snap counts 2012-2025, depth charts. Zero crossing closes nothing; record unresolved results with a closing ground.
