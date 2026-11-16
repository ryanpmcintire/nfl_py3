# Two-base stage: QW and OP model logits as separate fitted terms (2026-11-16)

## Goal
One declared look: the served four-term stage with BOTH base logits (QW base, input and target at the close; OP base, input and target at the as-of opener) as separate fitted terms, graded LOSO 2020-25 at the opener against the QW four-term reference 836-667. Parent: opener-parity-2026-11-16.md (OP alone 815-688 v QW 801-702, but OP four-term 831-672 v 836-667). Goal context: goal-65-2026-11-16.md.

## State
DECLARED 2026-11-16 18:00 -05:00, before grading. Script scripts/two_base_stage.py, artifacts artifacts/two_base_stage/.
- Inputs: cached base predictions artifacts/opener_parity/opener_QW.parquet and opener_OP.parquet (home_cover_probability_at_open, written by scripts/opener_parity.py; bases trained with opener pushes kept). Population per_game of artifacts/pick_probability/20261116T171555Z (1503 games 2020-25), same as opener_parity.
- Reference QW: v4.stage_two four terms (model_logit, flag sum, move, move available), LOSO; must reproduce 836-667 first or stop.
- Arm TB (one arm): same Fit (FIT_RIDGE, standardized) with five terms = QW model_logit + OP model_logit + flag sum + move + move available, leave-one-season-out 2020-25, held-out season scored.
- Report: record v QW; paired season-block (primary) and week-block P+ for accuracy_points and log_loss_improvement; changed picks and their records; fold coefficients of both model terms; Pearson correlation of the two model logits on the population; in-sample minus held-out gap (log loss and record).
- Looks: 1 arm x {accuracy_points, log_loss_improvement} = 2 cells, family two_base_stage_2026_11_16, unresolved_below_power unless an admissible closing ground applies. No variants.

## Tried
DONE 2026-11-16 (measured: artifacts/two_base_stage/results.json, log artifacts/two_base_stage_run.log, exit 0). QW reproduced 836-667; missing model 0/0; model alone QW 801-702, OP 815-688.
- TB 828-675 v QW 836-667 (1503 games). Season block (week block): acc -0.53 [-1.63,+0.65] P+ .179 (.122); LL -0.00029 [-0.00096,+0.00021] P+ .161 (.260). Changed picks 70, arm 31-39 (QW 39-31).
- Logit correlation QW v OP 0.949 on the population.
- Fold coefficients 2020..2025: QW term -.146 -.076 +.011 -.121 +.258 +.198; OP term +.544 +.300 +.243 +.476 +.041 +.129. The two sum to about the single QW coefficient (.40 .23 .25 .36 .30 .33), and the split between them is unstable and flips sign: collinear, no separate information identified.
- Gap: TB held-out LL .68702 v in-sample .68560 (gap .00142; QW gap .00103). In-sample record 826-677 (QW 849-654).
- Recorded 2 cells, family two_base_stage_2026_11_16, both unresolved_below_power (artifacts/two_base_stage/cells.json, exit 0). weak-signals pool --league nfl --effect-units accuracy_points exit 0.
- Reading (inferred): at r .95 the second base adds a degree of freedom, not independent error; the stage splits one model signal between two terms and loses 8 picks held out.

## Next
- Nothing to serve; QW stays the reference. One look spent; no variants per the packet. Lane complete; orchestrator may move it to docs/lanes/done/.

## Open
- None.
