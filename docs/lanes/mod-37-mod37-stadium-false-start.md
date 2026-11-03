# MOD-37 Stadium crowd-noise premium

## Goal
Stadium crowd-noise premium: home false-start and delay penalties forced on visiting offenses as a venue trait ROADMAP row MOD-37.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Loud stadiums force visiting false starts and delay-of-game penalties, which stall drives and cut visiting scoring. The effect belongs to the venue and may persist across seasons, unlike crew penalty tendencies or the single home-side offset. Data: nflverse pbp penalty types by posteam and venue 2009-2025, indoor and outdoor flags.

## Tried
Novelty check only: Nearest: penalty_crew_tendencies.md (referee crews), home_side_location.md and home_side_prior.md (one home offset), sim09_personnel_catalogue.md (false start mentioned only). Differs: venue-level false-start premium as a trait.

## Next
First unit: Per-venue visiting false-start rate over expected, split-half reliability across seasons with a venue-shuffle null, then a leave-one-season-out margin term for the visiting offense only if reliable.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
