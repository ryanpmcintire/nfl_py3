# LEAD-106

## Goal
Injury-report language read: a frozen language-model prompt maps each listed player's designation, practice note and headlines to P(plays) and expected snap share (ROADMAP LEAD-106). Availability input only; one fitted probability selects every side, so nothing here changes a pick.

## State (2026-11-21 11:16 local, forward Week 10 scored; numbers measured this session)
Snap source: `data/players/forward_snaps/snap_counts_2026_before_week_11.parquet` (3,051,770 bytes, written 2026-11-19), holds Week 10. Command `scripts/lead106_forward_score.py --week 10` -> `artifacts/lead106_injury_language/forward_score/2026_week10{.json,_scored.parquet,_registry_batch.json}`. 345 reads (20 Thursday, 325 Sunday), all at least 164 min before kickoff, 0 excluded, 298 verified, 28 team-week clusters, 14 games. No model calls. Positive = read better than the ladder; team-week bootstrap 2000.

| log loss gain | vs designation ladder | vs designation x practice |
|---|---|---|
| Week 10 (298, 28) | .020 [-.025, .068] P+ .78 | .008 [-.016, .035] P+ .75 |
| Weeks 9-10 cumulative (607, 58) | .052 [.012, .101] P+ .996 | .023 [-.002, .048] P+ .965 |
| Pooled weeks 2-10 (901, 201) | .062 [.020, .114] P+ 1.0 | .033 [.002, .071] P+ .977 |

- Cumulative 9-10 Brier .014 [.002, .026] P+ .99 / .008 [.001, .018] P+ .99; snap MAE .016 [.004, .027] / .017 [.005, .027] P+ .999.
- Cumulative by designation (des / practice ladder): none 331 rows +.113 [.051, .183] P+ 1.0 / +.056 [.017, .094] P+ .9995; Questionable 135 -.025 [-.101, .050] P+ .27 / -.021 [-.062, .020] P+ .16; Doubtful 12 -.100 / -.078, P+ 0 (0 of 12 played, read .13 vs ladder .03); Out 129 -.010 / -.007, P+ 0 (nobody played, read floor .013 vs .003). Status-filed games 560 rows +.025 [-.007, .060] P+ .935 / +.014 [-.004, .035] P+ .93.
- Concentration repeats: Week 10 net +5.9 nats vs the designation ladder, of which LAC@BAL (Monday, no designations at the Saturday capture) is +5.3; other 13 games +.002 [-.034, .040] P+ .55 (vs practice +.012 P+ .86). Worst SF@DAL -3.4, HOU@CLE -1.8.
- Scorer change this unit: ladders now expand over every prior week, adding scored forward weeks (Week 10 ladder = weeks 1-9, 2461 rows); before the change it stopped at week 8. Sensitivity with the weeks 1-8 ladder (unchanged script, first run): Week 10 .023 [-.026, .076] P+ .79 / .014 [-.006, .039] P+ .89. Also added: cumulative forward section, pooled over all forward weeks, per-game gain, registry batch builder keyed by `--week`. ruff check and format clean. Week 9 artifacts untouched.
- Looks: 102 this run (forward 36, cumulative 36, pooled 30) plus 66 in the sensitivity run, 2 of them quoted; family lead106 now 234 computed. Recorded 34 new cells, unresolved_below_power: `lead106_forward_wk10_*` (16), `lead106_forward_wk9_10_*` (16), `lead106_pooled_wk2_10_ll_{des,desprac}_all`. Cumulative and pooled cells carry the week range in their names, so each week adds rows and nothing is replaced. `weak-signals pool --league nfl --effect-units accuracy_points` exit 0, no coherence problems.
- Implication (inferred): the read's edge stays in untagged players, mostly Monday teams read before filing; it is over-cautious on Out and Doubtful. Enter it as a fitted term beside the ladder (LEAD-108), not as a replacement.
- ROADMAP LEAD-106 update: Week 10 scored (298 verified, 14 games): log loss vs tag ladder +.020 P+ .78, vs tag+practice +.008 P+ .75; Weeks 9-10 +.052 P+ .996 and +.023 P+ .965; gain sits in Monday teams read before filing; 34 cells unresolved_below_power. Next: score Week 11 when its snap counts land.

## Tried
- Unit 1-2 (11-03): `scripts/lead106_injury_language.py`, prompt sha256 08170f6d...3d2c; gsis->pfr id join, Weeks 2-8 sample 294 rows: .065 P+ .9985 / .036 P+ .96.
- Unit 4: gsis-keyed outcome (`artifacts/per03/played_by_gsis.parquet`, ends Week 8): cells `lead106_injury_language_p_plays*`.
- Unit 3: forward scheduler jobs `lead106_forward_*` -> `artifacts/lead106_injury_language/forward/`.
- Unit 5 (11-14): Week 9 forward (309 verified, 15 games) .083 P+ .9975 / .036 P+ .967; BUF@MIN (Monday) carried most of it; 18 cells `lead106_forward_wk9_*`, `lead106_pooled_wk2_9_*`.

## Next
1. Score Week 11 once `snap_counts_2026_before_week_12.parquet` exists: `scripts/lead106_forward_score.py --week 11`, then `weak-signals record --batch artifacts/lead106_injury_language/forward_score/2026_week11_registry_batch.json` (new names, no `--replace`), then pool once.
2. LEAD-108 / LEAD-109 can use `forward_score/2026_week{09,10}_scored.parquet` (607 verified census rows).
3. Root (scheduler-owned): add a Monday view so Monday teams are read after filing.

## Open
- Whole-interval-below-zero sub-cells for the orchestrator, not closed here: forward and cumulative Out and Doubtful (both ladders; read floors above the ladders on players who never play), pooled Doubtful vs the designation ladder (-.049 [-.073, -.024], 39 rows; vs practice +.0004 P+ .52).
- No split-half on forward weeks yet; with two weeks, a week-9 vs week-10 split is possible next unit.
- 47 of 345 Week 10 rows unverified (no pfr id).
- Rules: zero crossing closes nothing; one fitted probability selects the side; parameters out of season; every cell a look.

## Root decisions (2026-11-21 11:18 EST)
- Scorer change accepted: the Week 10 ladder now includes Week 9, as the declared "expanding prior-week ladders" requires; the earlier stop at Week 8 was a defect. Both readings are kept in the lane.
- Out and Doubtful cells with the whole interval below zero stay unresolved_below_power: they are the worst of 102 looks in this run, and choosing the worst cell to close is the mirror of presenting the best cell as the finding (AGENTS.md: count every cell as a look). For Out, the ladder is near certain, so the read has almost no room; that is a ceiling, not a refuted mechanism.
