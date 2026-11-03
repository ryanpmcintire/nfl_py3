# MKT-28

## Goal
Served-base re-test of the kickoff-slot and opener-revision terms and the return-from-absence read (ROADMAP MKT-28).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: MKT-24, MKT-25 and PER-15 were graded against the raw model probability or an uncorrected base; their opener-error signal may be absorbed by or add to the served four-term fit. Fitted term: each declared signal as a fifth term on the served logit.

## Tried
Novelty check only. Nearest: MKT-24, MKT-25, PER-15, MOD-51. Differs: the market-side terms on the true served base, separate from the MOD terms in MOD-51.

## Next
First unit for one agent: Rerun the MKT-24 slot residual and MKT-25 revision size as added terms on out_of_season_home_probability, leave-one-season-out, with the raw-base arm beside it, and report which of their earlier cells change.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
