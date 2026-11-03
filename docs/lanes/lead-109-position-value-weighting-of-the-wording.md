# LEAD-109

## Goal
Position-value weighting of the wording read: which absences the read prices, by position group and trailing snap share (ROADMAP LEAD-109).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: An availability gain only matters through the player's marginal value; QB and offensive line absences move margin more than depth absences. Fitted term: read availability weighted by position-group coefficients fitted out of season, replacing equal weighting. Data: LEAD-106 reads, snap counts, PER-09 ratings (local).

## Tried
Novelty check only. Nearest: MOD-42 (line continuity), SIM-16 (position-group attrition), LEAD-106. Differs: weights the language read, no new rating build.

## Next
First unit for one agent: Fit group coefficients (QB, OL, skill, front seven, secondary) leave-one-season-out on read-weighted absence, with a permutation null, and report the count of groups as looks.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03, script scripts/lead109_position_weights.py, artifacts/lead109_position_weights/report.json)
Declared before signs: 8 groups x 2 ladders (designation, designation x practice) = 16 group looks; 2 value-weighted gains; 2 weighted-minus-unweighted differences with within-designation label-shuffle null (2000); 2 leave-one-week-out group-mean fits (weeks 2-8 of 2026, the only season of reads). 24 looks, family lead109_position_value. Population: the 294 id-verified LEAD-106 reads (no LLM rerun). Values: |spread points per Out| from LEAD-110 OLS on 2011-2025 (prior seasons only): QB 2.68, ST .81, OL .38, RB .34, RECV .28, LB .26, DL .22, DB .16. Clusters: team-week, 2000 reps.
Gain by group (log loss, vs designation / vs designation x practice; n): QB 13: .127 [-.17,.48] P+ .65 / .279 [-.12,.90] P+ .70; OL 12: ~0 / .010 (degenerate interval, few clusters); RECV 90: .120 [-.01,.29] .945 / .060 [-.02,.16] .885; RB 44: -.005 / -.028 [-.07,.002] P+ .04; DL 40: .080 .94 / .035 .92; LB 32: .047 [.02,.08] 1.0 / .035 1.0; DB 60: .030 .86 / .002 .51; ST 3: n too small.
Value-weighted vs unweighted: vs designation .080 [-.011,.196] P+ .947 vs .065 (diff +.015 [-.05,.12] P+ .57, null p .19 one-sided); vs designation x practice .100 [-.021,.299] P+ .86 vs .036 (diff +.064 [-.03,.24] P+ .66, null p .068 one-sided and two-sided). Halves even/odd flip: value-weighted .123/.039 and .191/.012, so not stable.
LOWO group means predict held-out gain worse than the grand mean (MSE -.012 and -.010, P+ 0): group coefficients are not stable across weeks; Spearman of held-out group gain vs value -.14 / -.01. Reading: the gain is not shown to concentrate in high-value positions; QB sits at the right tail but on 13 rows. Unresolved below power (not refuted, no closing ground). Recorded: lead109_value_weighted_gain_des, lead109_value_weighted_gain_desprac (unresolved_below_power).
Model entry (declared): per team-week, V = sum over listed players of value_g x (1 - P_read(plays)) x expected snap share, with value_g fitted leave-one-season-out on the multi-season archive (not 2026 alone), entering the fitted four-term margin/probability as one availability term; never a side rule. Needs the forward Week 9+ read and a QB/OL-enriched sample.
Next: forward capture Week 9+, sample all QB/OL rows to reach power on the high-value groups.
