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

## Unit 2 (2026-11-03 22:10Z): in-season play-caller change event study
Script scripts/per14_unit2_firings.py (ruff clean), artifacts/per14/firings_event_study.json, firings_events.csv, firings_run.log, firings_record_batch.json. Windows declared before looking: K=4 games each side (MIN_STINT_GAMES//2), >=3 pre and >=3 post games, >=2 observed games per side per trait; controls = same-season teams with no HC/OC change, same game index; 5000 placebo change indexes per event; cluster bootstrap 4000.
Events 26 (32 before window filter): 18 HC change from schedule coach column (2011-2023; no play-caller identity, interim HC may not call plays), 6 OC change from coordinator_history in-season sample (2022-2025 only), 2 both. 16 in 2020-25. Pre-2022 OC-only changes are NOT observable (no pbp play-caller field).
DiD (post-pre minus controls) [95%] perm p: halftime EPA delta -.012 [-.099,.075] p .83 (n25); lead sec -.14 [-3.7,3.4] (n6); lead run +2.5pp [-8.3,10.6] (n8); trail pass incl -2.6pp [-5.4,.2] P+ .035 p .13 (n24); trail shift -3.6pp [-6.8,-.3] P+ .015 p .12 (n21); off EPA/play +.087 [.044,.127] perm p .001 (n25), but selection (firing after bad stretch, regression to mean) not removed by controls: inferred, not an effect of the change.
Market (2020-25, 1503 games, 46 flagged team-games, 16 events; 2011-19 has no served opener probability, SBR-proxy not built): first3 decisive 11 term v 18 base (flagged baseline-pick record 28-46), LOSO log loss gain +.00004 [-.0015,.0015] P+ .52, acc -.47pp [-1.14,.27] P+ .087, coef +.34 [-.24,1.01]; first1 decisive 2 v 7, ll +.00027, acc -.33pp, coef +1.06 [.13,2.64], perm p .14. In-sample gap ~.0004-.001 ll. Looks: 6 DiD + 2 windows x (ll, acc, coef) = 12.
Recorded 10 cells, family per14_inseason_playcaller_change_event_study, unresolved_below_power. Next: pool OC/DC movers (unit 1 Next), or build SBR-proxy opener for 2011-19 to raise market events from 16 to ~26.
- 2026-11-14 15:17 local, registry catch-up (docs/lanes/registry-catchup-2026-11-14.md): unit 1 recorded, 20 cells, family per14_oc_mover_portability, correlation, unresolved_below_power (5 traits x person r, team r, person-minus-team difference, partial r) from artifacts/per14/mover_portability.json with each trait reliability attached; the correlation unit exists, so the Tried note above no longer blocks a record. Not recorded: the 5 OC-to-HC correlations (permutation p only, no interval or probability_positive on disk). The season span 2009-2025 on these cells was read from the staff-history note in Open, not from the artifact: verify.
