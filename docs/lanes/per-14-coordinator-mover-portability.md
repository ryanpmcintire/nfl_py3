# PER-14

## Goal
Coordinator mover test: does an OC trait travel with the person to the new team or stay with the roster (ROADMAP PER-14).

## State
Unit 1 done 2026-11-03 21:59Z. Script scripts/per14_mover_portability.py (ruff clean, one process). Artifacts: artifacts/per14/mover_portability.json, events_<trait>.csv, run.log. Traits are OC stint means, shrunk by method-of-moments variance components (tau2 from OC spells >=16 games, sigma2 within) so reliability is derived, not chosen. Min stint 8 games = MIN_SPELL_GAMES/2, the shortest unit whose split-half reliability was measured (SIM-17/18/19 halves). Events = consecutive qualifying stints of one person at different teams; successor = immediate next OC at the old team (>=8 games); cluster-by-person bootstrap (4000), permutation null (5000).
Movers 77 (52 persons) for halftime and trailing pass; 57-59 for lead-protection (35-37 persons). Reliability (median long spell): halftime .32, lead sec .69, lead run .78, trailing pass .74 incl / .65 shift. None zero: no closing ground.
Person portability r (stint1 vs stint2) [95% CI], P+: halftime .03 [-.19,.21] .62; lead sec -.01 [-.28,.27] .51; lead run -.00 [-.22,.19] .48; trail pass incl -.02 [-.27,.23] .43; trail pass shift (style-adjusted) .17 [-.06,.40] .93, perm p .07.
Team persistence r (stint1 vs successor at old team): halftime .09 [-.10,.29] .83; lead sec .09 [-.18,.38] .75; lead run .17 [-.07,.40] .91; trail incl .01 [-.22,.25] .53; trail shift .35 [.15,.53] 1.0, perm p .001.
Paired person minus team diff: all negative (-.03 to -.20), shift -.20 [-.47,.12] P+ .10. Partial r of stint2 on stint1 given new team's prior OC: shift .16 [-.07,.40] .91; others ~0.
OC to HC (descriptive, play-calling unknown, n 29-33): trail incl r .48 perm p .003 (one of 5 looks, not adjusted); others negative.
Looks: 5 traits x 5 correlations = 25 (plus 5 OC-to-HC); no best-cell claim.

## Tried
Reused sim17 (off_res LOSO residual), sim18 (leader cell-mean residual sec/run), sim19 (trailing pass excess, incl and shift) definitions. weak-signals record not used: no accuracy_points effect unit applies to a correlation; recorded here and in artifacts.

## Next
Verdict (inferred): no trait shows person portability; trailing pass-rate shows team persistence (.35, whole interval positive), the stable part stays with the team once team style is removed, i.e. no evidence the OC person carries it. All unresolved_below_power for portability (n ~50-77, CI half-width ~.2-.25). No simulator/pick-model update warranted: a coordinator change should NOT shift a team's trait to the mover's old-team value; if later portability appears, the update would be x_new = lambda*x_mover_shrunk + (1-lambda)*x_team_prior with lambda = partial r from this table (shift trait .16, un-significant), applied as a fitted term, leave-one-season-out. Next unit: pool OC and DC movers (DC halftime/defensive traits), or an event-study on the 2 weeks after in-season OC firings.

## Open
Preseason-only staff snapshots 2009-2021 mean mid-season OC changes before 2022 may be mis-attributed (PER-07 caveat). HC play-caller identity unknown for OC-to-HC. Power: need ~n>=200 movers to detect r=.2.
