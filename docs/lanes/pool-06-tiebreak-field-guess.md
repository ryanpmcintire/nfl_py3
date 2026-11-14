# POOL-06

## Goal
Tiebreaker field distribution: choose the last-game total guess to maximise the chance of winning ties on wins (ROADMAP POOL-06).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Ties on wins are decided by the guess closest to the last game's total; the field's guesses cluster, so a guess just off the cluster wins more ties. Fitted term: tie-win probability from the field guess distribution and the discrete total distribution.

## Tried
Novelty check only. Nearest: POL-12 (tiebreaker score guess), POOL-04, LEAD-105 (language read of the total); grep found no field-guess model. Differs: The field's guess distribution, not the model's centre.

## Next
First unit: Check whether captured field files carry tiebreak guesses; if so, estimate cluster density and score the tie-win chance of the served guess versus alternatives over Weeks 1-8.

## Open
Data: Pool field files (captured) and the local totals distribution (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03T21:27Z, read-only)
- measured: no captured field file holds entrant tiebreak guesses (data/splash/field/*, data/pool_observables/*; only rules text in observations.json).
- rules (read, observations.json): weekly order is Points, Best pick wins, then closest last-game total (no guess = 0); season order is Points, Best pick wins, no total. So the guess only matters in weekly prizes, after two earlier tiebreaks.
- measured (241 of 250 entries, weekly parquet): entries tied at the weekly top: W1 1, W2 1, W3 1, W4 1, W5 8, W6 2, W7 1, W8 1. Only W5 (8 tied) and W6 (2) reach a tiebreak, before best-pick. Season top: 75 pts has 2 entries.
- Capture spec if pursued: view-only read of the entry detail / weekly picks view for the designated last game; rows week, entry_idx (no names), tiebreak_total (or home/away score guesses), designated_game_id, read_ts. Needs top-tied entries per week at minimum; full field if shown.
- Next: root checks whether Splash shows other entrants' guesses after lock; if not, POOL-06 is bounded by ~1 weekly tie in 8 reaching the total; any gain must come from the lattice (docs/tiebreaker.md) or a measured field offset, never a side change.
- 2026-11-14 15:17 local, registry catch-up (docs/lanes/registry-catchup-2026-11-14.md): nothing recorded. Unit 1 measured only counts (entries tied at the weekly top: 8 in Week 5, 2 in Week 6, 1 in each other week; 2 entries on 75 season points) and found no captured tiebreak guesses; a count has no effect, interval or admissible unit.
