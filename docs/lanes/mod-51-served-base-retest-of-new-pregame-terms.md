# MOD-51

## Goal
Served-base re-test of the MOD-40 and MOD-42 to MOD-47 pregame terms graded on the raw model probability (ROADMAP MOD-51).

## State
Unit 1 DONE Tue Nov 3 2026 5:43 PM local (measured). Served base = out_of_season_home_probability (artifacts/pick_probability/20261103T195411Z), LOSO 2020-25, patched copies and logs in tests/scratch/mod51/ (originals untouched; MOD-45/46 via served_delta.py). Bases read: MOD-40 raw (line 13), 42 raw (105), 43 raw (219), 44 raw (49), 45 raw discrete read (389), 46 raw discrete read (p_s), 47 raw (35). All 7 raw; none already served.
dLL vs served (old raw -> new served, [95%], P+), decisive term-served: 40 sum +.00024 P+.59 -> +.00040 [-.0017,.0025] .64, 193 flips +7 net; 42 cont+inter -.00106 .18 -> -.00079 [-.0034,.0018] .28, 86-109; 43 career -.00001 .48 -> -.00003 [-.0011,.0010] .47, 28-31; 44 both states ~-.0004 -> -.00019 [-.0022,.0019] .43, 63-72; 45 scale -.00024 (own base) -> -.00031 [-.0027,.0020] .40, 81-64; 46 run_adj +.0028 (placebo +.0028) -> +.00231 [-.0028,.0075] .82 vs placebo +.00234 .83; 47 interaction -.00149 P+.08 raw -> +.00029 [-.0019,.0024] .60, 57-69, zs coef now + 6/6 folds. Intercept-only: served beats it +.0064 (P+ ~.98) in every rerun; no term beats served beyond noise. Conclusions: none flips; only 47 changes sign/P+ (from mild negative to null).
Recorded 7 unresolved_below_power cells, family mod51_served_base_retest (tests/scratch/mod51/record_batch.json). Looks: 21 arms computed (40:4, 42:5, 43:2, 44:4, 45:1, 46:4, 47:1), 7 recorded.

## Tried
Novelty check only. Nearest: MOD-41 unit 3 (served rerun done), MOD-47 unit 2, ENG-51. Differs: reruns the other terms on the true served base; MOD-41 already was.

## Next
Unit 2 candidate: MOD-47 short-week as side-specific term (home short minus away short) inside the four-term fit; MOD-44 interim cell on served. (Done: first unit for one agent: List which of MOD-40, MOD-42, MOD-43, MOD-44, MOD-45, MOD-46 and MOD-47 used the raw base (read their scripts), rerun only those terms on out_of_season_home_probability with one declared family, and report gain, decisive record and which conclusion changes against the raw-base cell.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
