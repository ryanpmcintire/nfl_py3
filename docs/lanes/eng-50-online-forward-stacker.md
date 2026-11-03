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
