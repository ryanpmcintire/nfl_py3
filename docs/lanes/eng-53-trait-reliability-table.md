# ENG-53

## Goal
Trait reliability table: one table of split-half and season-to-season reliability for every keyed trait, with the shrink factor each study used (ROADMAP ENG-53). No model term; infrastructure for the ridge prior variance.

## State
Unit 1 done 2026-11-03 17:04 local. Script scripts/eng53_reliability.py (ruff clean, no comments), output artifacts/eng53/trait_reliability.csv, 51 rows: 39 reported only, 10 recomputed with the shared estimator, 2 not measured (MOD-42 OL continuity, LEAD-100 read ensemble: lanes say unmeasured, nothing on disk). Row set: SIM-17 halftime (HC off/def with pick-term arms from unit3_heldout; unit2 HC/OC/QB/DC own sample), SIM-18 lead tempo and run share (HC/OC; OC rows carry pick arms and the shrink K from unit2), SIM-19 trailing x5 outcomes x HC/OC x incl/shift, XLG-15 CFB 3 traits x 2 eras (priced vs ATS, not a pick term), MOD-40 CPOE and RYOE (from lane text; batch json has only log-loss cells), PER-14 mover rows.
Shared estimator (pair_estimate): Pearson r between a group's value in period 1 and period 2, bootstrap over persons (4000), two-sided permutation of period 2 (5000), seed 53. Applied to the 5 PER-14 traits x 2 pairings (person x1 vs x2; team x1 vs successor) from artifacts/per14/events_*.csv = 10 looks. Result: shared r equals reported r to 6 digits on all 10, shared interval within 0.02 of reported; zero flags. Only difference: permutation p is two-sided and +1-smoothed (shift person .144 vs reported .07 one-sided; shift team .0026 vs .001); the shared two-sided value is the right one to quote.
**measured** coverage: 10 of 49 measured rows recomputed (all PER-14); the other 39 are reported only because SIM-17/18/19, XLG-15 and MOD-40 saved no per-game or per-group output (only summary json), so the odd/even Spearman-Brown with sign-flip null and the season t to t+1 bootstrap could not be rerun on them. Odd/even function not shipped: it would have no input on disk.
Shrink factors on record: SIM-17 split-half SB (.14 off, .33 def); SIM-18 OC mean half reliability .403 at K 59 drives; PER-14 method of moments (.32-.78 by trait); MOD-40 pregame shrink (factor not stated in lane); SIM-19, XLG-15, SIM-17 unit2 none stated.
Pick-term record (decisive W-L, log-loss gain vs served, P+): halftime off 787-716 +.00004 .55; def 793-710 -.00058 .34; lead sec (OC) 770-733 -.0011 .03; lead run (OC) 814-689 +.00009 .53; CPOE 802-701 +.00004 .50; RYOE 796-707 -.00002 .50. None qualifies a closure; all unresolved.
Spell-inflation caveat (SIM-19 odd/even .32-.61 inflated by shared baseline) is in the table as reported; the shared estimator on spell data is not yet run, so it is unresolved here. No weak-signals records (reliability is not an admissible effect unit).

## Tried
Looks: 10 shared-estimator correlations; no cells selected as findings. Rerunning sim17/18/19 scripts to regenerate per-group data was not done (they write their own artifacts and rebuild from raw pbp).

## Next
Unit 2: add a save-per-group-output step (spell id, odd/even half sums, season means per spell) to a copy of the sim17/18/19 residual builders under scripts/ (new file, not editing their artifacts), then add the odd/even Spearman-Brown with sign-flip null and the season t to t+1 bootstrap to eng53_reliability.py and fill the 39 reported-only rows; resolve the shared-baseline inflation in SIM-19 with one estimator. Then XLG-15 per-team-season halves and MOD-40 from game_features_ngs inputs.

## Open
MOD-40 shrink factor and MOD-43 (coach/QB learning curve, mod43_qb_learning_curve.py) not read for this table.
