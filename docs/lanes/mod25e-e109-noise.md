# mod25e E109 where ku raised noise (diagnosis)

## Goal
Find the late-game channel where final-margin variance moved ke -> ku (noise 180.5 -> 185.2, real 165) and name the real mechanism that counters it. Base ...kehghkeku seeds 11-13 v ...kehke; real 2009-17 REG. Read-only on sim logs.

## State (2026-10-09, measured; tests/scratch/e109/{dec,drv,drv2,lead,seed}.py and .txt/.csv)
- Final-margin variance (39168 games each) ke 239.3 -> ku 242.3 (+3.0, 90% game boot [-0.7,+7.0], real 221.7). Checkpoint var (ke->ku): Q1 51.5->51.9, half 127.5->129.0, Q3 200.9->201.8, 5:00 left 238.9->239.9, 2:00 left 249.3->252.6. Variance created inside windows is flat except minutes 5:00 to 0:00: Var(delta) 26.40 -> 25.74, cov(margin at 5:00, delta) -13.59 -> -12.27 (+1.32 [+0.39,+2.11], P .999; real -13.08; 3/3 seeds). Effect on Var: -0.66 + 2.64 = +2.0 of the +3.0. Minutes 2:00 to 0:00 cov flat (-11.96 -> -12.03). So the move is minutes 5:00-2:00 (cov(m300,delta m300->m120) shift).
- Carrier bucket: lead 17+ at 5:00 (28% of games, real 27%). Leader-signed margin change 5:00->2:00: mean real -0.407, ke -0.268, ku -0.129 (seeds -0.08/-0.12/-0.19 v -0.28/-0.24/-0.29); P(leader extends) real .152, ke .179, ku .191 (3/3). Lead 9-16: mean real -0.465, ke -0.327, ku -0.279; P(trailer scores net) real .300 v .235/.230.
- Drive level (Q4 drives starting 2:00-5:00, offense up 9+): clock-ended real .282 v ke .187 ku .185; punt .435 v .512/.511; score .157 v .183/.185; drives/g l9+ .306 v .297; drives starting <=2:00 l9+ .232 v .290/.288. ke and ku drive stats are equal to within .003 in every state: ku did not change leader drives; it removed the spurious half-end after a timeout, so the lead-17+ extension and the lost late regression are the visible shift.
- Final drive of Q4 P(score) by offense state (real/ke/ku): trailing 1-8 .202/.120/.103; tie .322/.136/.121 (walk-off FG .316/.129/.116); trailing 9+ .043/.038/.044; lead 1-8 .002/.011/.012. Real trailing final drives end in a turnover .18 v sim .04-.045. Trailing teams score too RARELY on final drives, not too often; the proposed counter (trailing final drives score too often with preserved clock) is refuted.
- Not moved: OT rate real .0616, ke .0566, ku .0586; drives/g 23.14 v 23.54/23.54; late TD/FG per game in last 5 Q4 flat; Q2 last-drive score .225/.150/.156.

## Tried
Game-level bootstrap (200 reps) on ku-ke; per-seed splits; drive-level frames via clock2 real_plays/sim_plays/frames. No hook written.

## Next
- Leader clock-burn in Q4 (offense up 9+, drive start 2:00-5:00: clock-ended .28 v .19, punt .435 v .51, scores .157 v .183) is the same open carrier as the E107/E108 leader kneel-out gap (hs0 94 v 114). Fit: drive-ending/clock-burn for leaders by lead size, LOSO, then smoke.
- Final-drive desperation (trailing turnover .18 v .04, walk-off FG .316 v .13 in tie) is the E99 walk-off gap.

## Open
Is the lead-17+ extension (.19 v .15) from leader scoring drives (+.026 score rate) or defensive/special scores? Not split.
