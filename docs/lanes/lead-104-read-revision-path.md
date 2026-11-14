# LEAD-104

## Goal
View-to-view revision of the LEAD-99 read: does the change from the first view to the last view before each pick deadline carry information, or only track the line move (ROADMAP LEAD-104). Research only, nothing served. The LEAD-99 term of record stays the declared last view (2026-10-20); the first view is a monitor.

## State (2026-11-14, unit 2; measured by `scripts/lead104_revision_path.py --weeks 7 8 9 10 --first-view earliest tuesday` unless marked)
- Weeks 7, 8, 9 are settled; 10 is skipped as unsettled. Week 8 still has no Tuesday view (read, lead99-llm-read.md:34: the 13:30 job found no captured pool week on 2026-10-27), so its first view is Thursday. Primary construction: earliest view to the last view before min(kickoff, Sunday 16:00 ET), the ledger's `lead99_first_view` and `lead99`; 43 games, 42 with two distinct views, 41 with a quote in both. Strict Tuesday construction: Weeks 7 and 9, 29 games. Graded at the pool's Tuesday line (margin minus line; matches the feature table on 43 of 43; no pushes).
- Revision against the line move between the views: r +0.734 [+0.452, +0.919], probability_positive 1.00; slope 0.134 logit a point [0.111, 0.158] (read, lead99 lane: the card's fitted move coefficient is 0.2204); intercept +0.059 logit toward home; residual SD 0.115, 1.56 times revision noise. Tuesday only: r +0.702 [+0.294, +0.935]. 28 revisions toward home, 8 away; 13 moves toward home, 6 away; home covered 17 of 43.
- First view 28-15: +15.1 points [+1.2, +29.1] P+ .978, within-week permutation p .150 (null +6.2); log loss against 0.5 +0.0259 [+0.0051, +0.0474] P+ .993, p .037.
- Last view 23-20: +3.5 points [-10.5, +17.4] P+ .676, p .442; +0.0059 [-0.0231, +0.0344] P+ .647, p .342. Last against first -0.0200 [-0.0453, +0.0044] P+ .058; outcomes shuffled within week give -0.0097, lower-tail p .177. By week, first / last: W7 9-5 / 9-5, W8 9-5 / 8-6, W9 10-5 / 6-9.
- The 13 side changes: revision direction (the last view's side) 4-9, -19.2 points [-42.3, +3.8] P+ .071, null -7.8, lower-tail p .368, exact sign test .267. Log loss on the 13: first +0.0143 [-0.0216, +0.0533] P+ .773; last -0.0409 [-0.0730, -0.0101] P+ .003, lower-tail p .012; last against first -0.0552 [-0.1168, +0.0088] P+ .047, p .112. 8 of the 13 had a line move and all 8 went with it (2-6); 5 had none (2-3). Median size 0.20 logit, 2.7 times revision noise, so they are not read noise.
- Net of the move: residual direction 20-21, -1.2 points [-15.9, +13.4] P+ .437, p .683 upper, .619 lower; first view plus residual against first view -0.0073 [-0.0242, +0.0081] P+ .198. A dead heat.
- Where the -0.0200 sits, each piece added alone to the first view: slope times move -0.0073 [-0.0241, +0.0105] P+ .204; home drift -0.0076 [-0.0159, +0.0011] P+ .041 (a constant, so the within-week null cannot test it; it is the weeks' home cover rate); residual -0.0073. The move's own side against the Tuesday number on 19 moved games: 6-13, -18.4 points [-39.5, +2.6] P+ .046, null -9.4, lower-tail p .350.
- Looks: 54 outcome looks this unit (19 cells in each of 2 constructions, 10 by-week records, 6 move splits) plus the Week 7 look; family lead104_revision_path. 19 cells in registry/weak_signals/lead104_revision_path.json, all unresolved_below_power; `pool` ran once, exit 0.

## Reading (inferred)
- The revision is the line move plus a drift toward home, and nothing measurable beyond them. The later view trails in these weeks where it followed the move (the move's side went 6-13 at the frozen number) and where it drifted home while home covered 17 of 43. Half of the deficit survives shuffling outcomes within a week, so it is weekly lean as much as game selection. 13 changed games cannot separate this from chance.
- Following the move is the right direction at a frozen line: the side the market moves toward is the value side at the Tuesday number, and the six-season move fit says so. Three weeks of 6-13 do not contradict that fit.
- The declared row would be the wrong term only if last against first resolves negative on the forward sample from Week 10 (whole interval below zero, within-week null agreeing) through one of: the move between views not predicting cover at the Tuesday number; the home drift being a standing bias rather than three weeks' home cover rate; the residual being negative rather than a dead heat. None is shown. Stacked on the card, the last view also overlaps the card's own move term (r .73); that bears on the stack design, not the declaration.

## Tried
- Unit 1 (2026-11-03, Week 7): r .885 [.699, .988], residual direction 7-6. Its grade was margin plus line, the wrong sign for a line that is positive for a home favourite; regraded it is again 7-6, +3.8 points [-19.2, +26.9] P+ .617, re-recorded with replace.
- Script now takes `--weeks`, `--first-view` (tuesday, earliest), `--cells`, `--tag`, reads deadlines from `game_deadlines`, and writes artifacts/lead104_revision_path/ (summary.json, weak_signals_batch.json, per_game csv). ruff check and format clean.

## Next
- Rerun with `--weeks 7 8 9 10` once Week 10 is settled (then each week), record the batch with `--replace`, run `pool` once. Weeks 7-9 are exploratory for this family; Week 10 onward is the forward sample.
- From five weeks on, fit revision, slope times move and drift leave-one-week-out instead of the in-sample pieces above.

## Open
- Slope and intercept are fitted on the scored games, without outcomes. Residual SD and its noise ratio have no registry unit and stay here.
- registry/weak_signals_base_tags.csv has no rows for the new cells; not edited.
- Zero crossing closes nothing. The one interval wholly below zero (last view on the 13) is one of 54 looks on a subset where one view must lose when the other wins; it closes nothing.
