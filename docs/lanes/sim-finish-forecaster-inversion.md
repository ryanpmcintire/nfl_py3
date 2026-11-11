# Lane: sim-finish-forecaster-inversion (done 2026-11-11 10:46:50)
## Goal
Mechanism behind inverted sim cover logit (main run, 1,537 games 2020-25). Data-only.
## State
scripts/sim08_inversion_diagnosis.py -> artifacts/sim08_inversion/diagnosis.json (ruff clean, run once). 3 cells recorded in registry/weak_signals/sim08_market_anchored.json (31 -> 34). Looks counted: about 70 (H1 6, H2 12, H3 ~40 incl bands, H4 ~14).
H1 not supported (slope 1.16, 40/40 sign, labels match). H2 supported: solo raw -0.452; leftover alone -0.044/pt (pp .05); corr raw logit v leftover .89; after exact centring of saved draws coef +0.448 (SE .415), corr with raw logit -.06.
H3: centred LL .7631 v served .7638 (served minus centred +.0007 [-.0068,.0085] pp .56), v market hist .7613 (-.0017 pp .15); raw gap -.0136, centred removes +.0143.
H4: sim sd ~12.3 flat (corr with real abs resid .04); raw inversion strongest in lowest total tercile (-.78) and high total-leftover tercile (-.52), centred vanishes; not separable.
## Tried
Centring is a diagnostic shift of saved draws, not served. Main run used the A3 one-shot quadratic inverse, not the A4 iterated loop.
## Next
Final run: A4 iterated anchor as declared, plus exact conditioning by exponential tilt (sim04_loso.tilt_to_mean) of the final draws to the line, declared as an Amendment before the run; then re-measure solo coef.
## Open
Why the leftover is anti-informative (sim favourites about 1 pt too strong; leftover slope .16 on line) is not identified.
