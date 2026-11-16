# Opener parity: train feature and target against the opener (2026-11-16)

## Goal
Test full opener parity on the QW table: training spread_line = as-of opener AND target ats_margin = result minus that opener, so the base learns the residual against the number it is graded on. Graded through the served four-term stage LOSO 2020-25 at the opener, reference QW 836-667. Parent: hindsight-audit-team-2026-11-16.md (P/TP refuted only feature-without-target).

## State
Novelty (read): MOD-23 unit 7 arm A (done/mod23-unit7-target-line.md, 2026-10-16) already trained spread_line := opener, ats_margin := result - opener, on 2011+ known-opener rows, and graded the model ALONE on the old weak_stack base: 801-702 v 802-701, P+ .43, unresolved. MOD-57 P2-C Qc restricted rows to 2011+ known-opener but kept feature and target at the close (not parity). This unit differs from unit 7 A by: (1) QW table (QB repair + forecast weather), (2) graded through the served four-term stage 2 (unit 7 never ran stage 2), (3) as-of opener = Tuesday capture, else SBR open 2009-2021, else close (2009-10 included), (4) intercept-only control. The model-alone record is a partial repeat of unit 7 A and is reported as a check, not a new look.

DECLARED 2026-11-16 17:03 -05:00, before grading. Script scripts/opener_parity.py, artifacts artifacts/opener_parity/. Stage 1 clv.opener_pick_evaluation (v4.opener), stage 2 v4.stage_two LOSO 2020-25, population per_game of pick_probability/20261116T171555Z (pushes vs opener kept in training; ats_margin 0 rows stay in fit). History rows < 2026 only are rebased.
- OP: QW table, spread_line and ats_margin rebased to the as-of opener; rows without an opener keep the close (count reported).
- OPr: as OP, but training rows without a real opener (Tuesday or SBR) get ats_margin NaN (dropped from fit).
- OPi: control, OP table with an intercept-only base (feature set = one constant zero column).
Looks: 3 arms x {accuracy_points, log_loss_improvement} = 6 cells v QW, season-block P+ primary, week-block reported. Family opener_parity_2026_11_16, unresolved_below_power unless an admissible closing ground applies.

## Tried
DONE 2026-11-16 17:11 -05:00 (measured, artifacts/opener_parity/results.json, run log artifacts/opener_parity_run.log, exit 0). Identity S 829-674 max gap 0.0; QW 836-667 reproduced. Opener source: 4415 of 4431 completed 2009-25 REG rows rebased (Tuesday or SBR), 16 fall back to the close (all 2022); spread changed on 3497 rows, mean |diff| 1.21; pushes in training 122 v opener (114 v close).
Vs QW, four-term LOSO 2020-25, 1503 games (season block; week block in brackets):
- OP 831-672: acc -0.33 [-1.36,+0.66] P+ .240 (.255); LL -0.00002 P+ .478 (.467); changed 77 (arm 36-41). Model alone 815-688 v QW 801-702 (+14, partial repeat of unit 7 A; not a declared look).
- OPr 827-676: acc -0.60 [-1.80,+0.60] P+ .158 (.123); LL P+ .495; changed 79 (35-44); model alone 808-695.
- OPi control 810-693: acc -1.73 [-3.18,-0.27] P+ .011 (.038); LL P+ .104; model alone 762-741.
Fold model_logit (2020..2025): QW .401 .226 .256 .362 .300 .328; OP .415 .232 .253 .370 .268 .304; OPr .415 .233 .248 .366 .275 .308. Out-of-sample LL OP .68675 v in-sample .68560 (QW .68673 v .68570).
Recorded 6 cells, family opener_parity_2026_11_16: OP/OPr both metrics and OPi LL unresolved_below_power; OPi accuracy refuted_mechanism, wrong_sign_resolved (the control only: the intercept base is worse, so the model term carries information). weak-signals pool exit 0.
Reading (inferred): opener parity lifts the model alone (+14) but the four-term stage absorbs it: the other terms already carry the opener-relative information, so stage 2 gains nothing.

## Next
- Nothing to serve; QW stays the reference. A possible follow-up would be a four-term refit with OP plus the QW model logit both as terms, but that needs its own declared unit.

## Open
- None.
