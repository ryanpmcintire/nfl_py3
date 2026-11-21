# LEAD-104

## Goal
View-to-view revision of the LEAD-99 read: does the change from the first view to the last view before each pick deadline carry information, or only track the line move (ROADMAP LEAD-104). Research only, nothing served. The LEAD-99 term of record stays the declared last view (2026-10-20); the first view is a monitor.

## State (measured 2026-11-21 09:54 ET, unit 3: `scripts/lead104_revision_path.py --weeks 7 8 9 10 11 --first-view earliest tuesday`, exit 0; artifacts/lead104_revision_path/summary.json, per_game_*.csv)
- Weeks 7-10 settled; 11 skipped as unsettled. Week 8 has no Tuesday view, so its first view is Thursday. Earliest construction: 57 games, 56 with two distinct views, 55 with a quote in both; pool line equals the feature table on 57 of 57, no pushes. Strict Tuesday construction: Weeks 7, 9, 10, 43 games. Home covered 20 of 57 (Week 10: 3 of 14, inferred from 17 of 43 before).
- Decisive record first: 15 side changes, last view 6-9 (exact sign test .607); with the move 10 games 4-6, no move 5 games 2-3, against the move 0. Week 10 alone (first forward week): first 6-8, last 8-6.
- Revision against the move between views: r +0.752 [+0.545, +0.888] P+ 1.00; slope 0.139 logit a point [0.112, 0.172] (served move coefficient 0.2204, artifacts/pick_probability/20261111T011859Z/coefficients.json); intercept +0.050 toward home; residual SD 0.115, 1.55 times revision noise.
- First view 34-23: +9.6 points [-2.6, +21.9] P+ .934, within-week null +4.7, p .282; log loss vs 0.5 +0.0169 [-0.0016, +0.0355] P+ .966, p .112.
- Last view 31-26: +4.4 [-7.9, +16.7] P+ .749; +0.0075 [-0.0187, +0.0340] P+ .710, p .265. Last against first -0.0093 [-0.0315, +0.0117] P+ .198 (was -0.0200 P+ .058 on Weeks 7-9); shuffled-outcome null -0.0072, lower p .411. On the 15 changes: -0.0333 [-0.0947, +0.0285] P+ .144, lower p .254.
- Net of the move: residual direction 29-26, +2.7 points [-10.0, +15.5] P+ .661; first plus residual vs first -0.0019 [-0.0168, +0.0118] P+ .410, a dead heat. Pieces added to the first view: move part -0.0004 [-0.0164, +0.0159] P+ .478; drift -0.0084 [-0.0145, -0.0021] P+ .004 (a constant: equals the weeks' home cover rate, the within-week null cannot test it). Move side at the Tuesday number on 28 moved games 13-15, -3.6 [-21.4, +14.3] P+ .352.
- Tuesday construction: last vs first -0.0112 [-0.0368, +0.0146] P+ .199; drift -0.0163 [-0.0234, -0.0084].
- Looks: 58 outcome looks this run (19 cells x 2 constructions, by-week records, move splits), family lead104_revision_path; 18 cells re-recorded with `--replace` (exit 0), all unresolved_below_power.

## Reading (inferred)
- Week 10, the first forward week, went the other way (last view 8-6), halving the deficit. The revision is still the line move plus a home drift with nothing measurable beyond them (residual a dead heat). The only interval wholly below zero is the drift constant, which is the four weeks' 35 pct home cover rate, not a mechanism. Nothing is refuted; the declared last view stays.

## Tried
- Unit 1 (11-03, Week 7) regraded at the pool line: 7-6. Unit 2 (11-14): Weeks 7-9 (last vs first -0.0200). Unit 3 (11-21): Weeks 7-10, no code change.

## Next
- Rerun each Tuesday with the newly settled week, record with `--replace`, run `pool` once. Week 10 onward is the forward sample (1 week so far).
- From five settled weeks (now four), fit revision, slope times move and drift leave-one-week-out instead of in-sample.

## Open
- Slope and intercept are fitted on the scored games without outcomes; residual SD has no registry unit.
- registry/weak_signals_base_tags.csv has no rows for these cells; not edited.
- Zero crossing closes nothing.
