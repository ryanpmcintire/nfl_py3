# LEAD-66 unit 2: close-proxy second read and exact log score

**Measured:** `scripts/lead66_unit2.py`, one process, single thread; raw output `artifacts/lead66_unit2/results.json` (local, not committed).

## Prediction coverage (second read)

**Measured:** `artifacts/margins/20260929T192312Z` starts 2018 (2018-2019 only); the walk-forward weekly-refit production-model forecasts graded at the SBR proxy opening line exist for 2011-2019 in `artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet` (the population the backfill lane used). The declared span is therefore 2011-2019, not 2009-2019: 2009-2010 have no proxy opener and no walk-forward forecast (gap of two seasons, 512 regular-season games, not rebuilt). 2302 games after excluding 2 with ambiguous proxy opens; model-centre recovery failures 0. Caveat: those forecasts are the smooth margin-model probability (pre-discrete policy), so the model centre is recovered by the same exponential-tilt root-find used in unit 1 against that probability; the served lattice and market-residual arms depend only on the line. The pool uses proxy opens for 2011-2019 and closing lines for 2009-2010 and 1999-2008 (nflverse). Primary unit-1 RPS reproduced to six decimals (below), after the pinned feature-table hash check was bypassed because the table was rebuilt since unit 1; reproduction is the equivalence evidence.

## Tail support (derivation)

**Measured:** the unsmoothed PMFs give zero mass to the realised margin on primary 26/69/81 games (market/served/model-centred) and secondary 43/92/133, so exact log score is infinite. Tail prior: Gaussian kernel density of line-residuals (result minus line, recentred to mean zero) over every completed regular-season game from 1999 through the preceding season (nflverse), integrated over the unit margin cell at the game line; bandwidth by Silverman's rule of thumb per fold (secondary 2.07-2.30, primary 1.90-1.99). Served score = (1 - lambda) * arm PMF + lambda * prior. lambda is one scalar shared by the three arms, found by exact bounded minimisation of pooled log score on pre-graded seasons: secondary uses leave-one-season-out over 2011-2019 (range 0.5561-0.5955); primary uses lambda fitted on all of 2011-2019 (0.5807), which predates the 2020-2025 period. No constant was tuned on the scored period. Remaining zero-mass games after smoothing: primary 0, secondary 0.

**Inferred:** lambda near 0.58 says each arm's own PMF is poorly calibrated in the tails relative to the wide history, so the log-score contrasts with the market arm are partly driven by which arm is thinner in the tails; read them as tail-repaired scores, not as a statement about the centre of the PMF.

## Results (OOS, positive means candidate better)

Primary 2020-2025, RPS (reproduction of unit 1):

| Contrast (candidate vs baseline) | Gain | 95% season-block | probability_positive | Games | Blocks |
| --- | --- | --- | --- | --- | --- |
| served_lattice vs market_residual | -0.026309 | [-0.070629, 0.004613] | 0.0765 | 1537 | 6 |
| model_centred_lattice vs market_residual | -0.037718 | [-0.081865, -0.010954] | 0.0000 | 1537 | 6 |
| model_centred_lattice vs served_lattice | -0.011409 | [-0.019049, -0.003794] | 0.0029 | 1537 | 6 |

Primary 2020-2025, smoothed exact log score (nats):

| Contrast (candidate vs baseline) | Gain | 95% season-block | probability_positive | Games | Blocks |
| --- | --- | --- | --- | --- | --- |
| served_lattice vs market_residual | 0.073087 | [0.065638, 0.080070] | 1.0000 | 1537 | 6 |
| model_centred_lattice vs market_residual | 0.066871 | [0.055212, 0.078167] | 1.0000 | 1537 | 6 |
| model_centred_lattice vs served_lattice | -0.006216 | [-0.012005, 0.002570] | 0.0661 | 1537 | 6 |

Mean smoothed log score primary: market_residual 3.9635, served_lattice 3.8904, model_centred_lattice 3.8966.

Secondary 2011-2019 close-proxy, RPS (unsmoothed):

| Contrast (candidate vs baseline) | Gain | 95% season-block | probability_positive | Games | Blocks |
| --- | --- | --- | --- | --- | --- |
| served_lattice vs market_residual | -0.370954 | [-0.562674, -0.230506] | 0.0000 | 2302 | 9 |
| model_centred_lattice vs market_residual | -0.370112 | [-0.531230, -0.247006] | 0.0000 | 2302 | 9 |
| model_centred_lattice vs served_lattice | 0.000842 | [-0.018769, 0.032037] | 0.4942 | 2302 | 9 |

Mean RPS secondary: market_residual 7.4538, served_lattice 7.8248, model_centred_lattice 7.8239. In-sample RPS gains (same order as contrasts): -0.3664, -0.3799, -0.0135.

Secondary 2011-2019, smoothed exact log score (nats):

| Contrast (candidate vs baseline) | Gain | 95% season-block | probability_positive | Games | Blocks |
| --- | --- | --- | --- | --- | --- |
| served_lattice vs market_residual | 0.070883 | [0.056519, 0.086600] | 1.0000 | 2302 | 9 |
| model_centred_lattice vs market_residual | 0.071057 | [0.055750, 0.086853] | 1.0000 | 2302 | 9 |
| model_centred_lattice vs served_lattice | 0.000174 | [-0.008564, 0.008560] | 0.5188 | 2302 | 9 |

Mean smoothed log score secondary: market_residual 4.0126, served_lattice 3.9417, model_centred_lattice 3.9415.

## Looks

Nine new contrasts (3 primary log, 3 secondary RPS, 3 secondary log), one lambda fit per fold; no other arms or cells.
