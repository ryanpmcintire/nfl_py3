# Week 1 model weight (2026-11-16)

## Goal

Test the stale-input mechanism for Week 1: the base model's Week 1 inputs are last season's carried values, so its disagreement with the opener may be noise the market has already absorbed. One fitted term lets the Week 1 model weight go to zero inside the served probability. 2027 candidate only; nothing served now.

## State

- Prior work read: P2-T (mod57-sixty-percent.md) fitted four phase slopes together, never a Week-1-only weight. Its Week 1 scope: four-term 40-48, four-slope arm 44-44 (changed 10, 7-3); Week 1 fold slope .40 (range -.10 to 1.07). Registry family week1_dog_on_production is a different term (underdog flag). So the single-term test is untested.
- Declaration, stamped 2026-11-16T21:19:52Z, before any fit: term `model_w1` = model_logit x 1[week == 1], added to the four served stage-one terms (model_logit, composition_flag_sum, move toward home, move available). Primary grade: LOSO by season on 2020-25, opener-graded, keep-pushes population (pushes fit-excluded, scored with label 0.5 in log loss, excluded from records). Screen: 2011-25 against the close, model_logit (market_residual walk-forward backtest from 2011) plus the same term versus model_logit alone, LOSO. Forward: frozen 2020-25 fit on 2026 Week 1 frozen replay. Looks, family `week1_model_weight`: opener all-games accuracy, log loss, Brier; opener Week 1 accuracy; close screen accuracy and log loss; 2026 forward Week 1 accuracy (7).
- Script scripts/week1_structure.py, outputs artifacts/week1_structure/.

## Tried

- Run 2026-11-16 (measured; artifacts/week1_structure/results.json). Opener, 1,537 games incl. 34 pushes, 94 Week 1: all games arm 832-671 v four-term 829-674, changed 5 (4-1), +0.20 pts [-0.07, +0.52] P+ .87; log loss .68718 v .68691, P+ .019; Brier P+ .019. Week 1: 42-46 v 40-48, changed 2 (2-0), +2.3 pts [0, +4.8] P+ .92, Week 1 log loss .69285 v .68882 (P+ 0). Weeks 2-18 changed 3 (2-1). In-sample arm 839-664, gap 0.5 pts.
- Fitted Week 1 model weight (model + term) per fold 2020-25: .34 .01 .17 .22 .57 .43 (all positive) v base weight .23-.38. The weight does not go to zero; mechanism (Week 1 model disagreement is noise) not supported.
- Close screen 2011-25 (base-profile market_residual walk-forward, artifacts/margins/20261116T212043Z; served weak_stack lacks player columns before 2018, so this is a proxy model): 3,919 games, arm 1952-1866 v 1970-1848, changed 60 (21-39), -0.47 pts P+ .03; Week 1 119-116 v 127-108, changed 24 (8-16); Week 1 weight positive in 14 of 15 folds.
- 2026 Week 1 frozen replay (pool-line harness): current, frozen base and arm all 12-4, 0 changed picks.
- Recorded 7 cells, family week1_model_weight, unresolved_below_power; `pool` ran once and loads.

## Next

- Nothing to serve. A 2027 Week 1 lane should target the input (offseason-aware ratings), not the weight.

## Open

- 2026 Week 1 harness current record differs from the played card (P2-T forward Week 1 read 12-4 on the pool-line harness; the played card was 9-7).
