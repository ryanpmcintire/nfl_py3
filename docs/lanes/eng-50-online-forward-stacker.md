# ENG-50

## Goal
Online forward stacker: one calibrated probability combining the served model with every matured forward term, updated each week from prior weeks only (ROADMAP ENG-50).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: AGENTS.md asks for one probability with every signal as a fitted term, but forward terms are scored separately at Week 18. An online logistic updated after each slate (prior weeks only, ridge toward zero with prior variance taken from each term's own leave-one-season-out coefficient variance) is out of sample by construction and shows what the combination would have done, weekly.

## Tried
Novelty check only. Nearest: ENG-48 (per-log monitor), ENG-49 (the table), MOD-07 (stacking, prospective collection of 48 games), MOD-20 (pooled signals, retrospective). Grep for an online or prospective combiner found no row. Differs: a prospective combiner on matched games.

## Next
First unit: Dry replay on the ENG-49 core match (28 graded games, Weeks 7-8): predict Week 8 from Week 7, report paired log loss against served with the coefficient path; no card effect.

## Open
Changes: Combines terms the way policy says, in real time. Data: ENG-49 ledger (artifacts/forward_ledger_core). Online ridge-logistic; prior variance from LOSO coefficient variance, not a constant. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03T21:06Z)
Built scripts/eng50_online_stacker.py (output artifacts/eng50_online_stacker/). Model: logit(served) as fixed offset plus ridge-weighted deltas logit(p_k)-logit(served) for lead99, mod24, mod25_combined (the only ledger candidates with p on all 28 matched games). Frozen forward terms (sky03.. mod35) only have Week 9 files, ungraded, so none enter. Penalty rule fixed in advance: grid 1e-2..1e2, chosen by prior-weeks-only leave-one-week-out (leave-one-game-out when one prior week). Only one fold exists (fit W7 14 games, predict W8 14 games). Result: log loss stacker 0.7497 vs served 0.7288, improvement -0.0209, bootstrap 95% [-0.231, 0.180], P+ 0.428; Brier 0.2751 vs 0.2677; accuracy 8/14 vs 3/14; decisive 9 games, stacker 7 wins vs served 2. Coefficients (penalty 0.01, the grid floor): lead99 -1.32, mod24 -9.17, mod25 -2.68, i.e. overfit to 14 games. Recorded: eng50_online_stacker_week8_dry_replay, unresolved_below_power. Looks: 1 fold, 5 penalties, 3 terms, 1 cell.
## Next
Rerun each slate as Week 9+ settle and Week 9 frozen terms grade; with 2+ prior weeks the leave-one-week-out penalty applies. Serving would need the ENG-48 monitor to show positive paired log-loss improvement with sequence-valid evidence over a pre-declared horizon, stable same-sign coefficients across folds, and a decisive record, before entering the single calibrated probability.
- 2026-11-03 16:25 ET ROOT CORRECTION (measured): the ENG-49 ledger's served_p (forward_ledger.py:50,67,94) is each candidate run's recorded base probability at record time, not the played card. On Week 8 it takes a different side from the final published pick on 5/14 games (ARI_DAL, IND_JAX, KC_DEN, NE_MIA, PHI_WAS); the played card went 6-8 (scorecard), the ledger "served" 3-11. Against the played card the stacker's 8-6 is +2 games, not +5, and the 7-2 decisive record is against the wrong base. Unit 1 numbers are void as a comparison with the card; the record stays unresolved_below_power. Next: ENG-49 fix (served = played-card probability from artifacts/clv_ledger/published_picks.parquet, last row before the deadline; keep the candidate's record-time base as a separate column), then rerun.
- 2026-11-03 16:12 ET CORRECTION APPLIED (measured): ledger served now = played card (see ENG-49). Corrected Week 8 replay (fit W7, predict W8, 14 games) vs played card: log loss stacker 0.7157 vs card 0.7180 (improvement +0.0023, 95% [-0.0793, 0.1076], P+ 0.4766); Brier 0.2612 vs 0.2613; accuracy 5/14 vs 6/14; decisive 5 games, stacker 2 wins vs card 3 (decisive record first: stacker behind). Coefficients penalty 1.0 (leave-one-game-out): lead99 0.43, mod24 0.38, mod25 0.57 (same sign, not the earlier overfit -9). Registry cell eng50_online_stacker_week8_dry_replay replaced (effect 0.0023, P+ 0.4766, unresolved_below_power). Looks unchanged (1 fold, 5 penalties, 3 terms).
- Next: rerun when Week 9 grades (2 prior weeks -> leave-one-week-out penalty).
