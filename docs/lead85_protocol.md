# LEAD-85 frozen protocol

Verbatim copy of the lane snapshot saved before the research run; the original is in `tests/scratch/codex/lead85_unit1/protocol_declaration.md`.

# LEAD-85: integrate coefficient uncertainty

## Goal
Implement and run the declared integrated-probability adapter; no served changes.

## State
Protocol copied before loading or computing outcomes. Unit 1 in progress.

### Frozen declaration (verbatim)
| LEAD-85 | ⬜ | Average coefficient uncertainty into the Best Pick probability (batch C rank 4 of 8) | **Inferred mechanism:** a frozen opener cannot contain late news, but the largest resulting fitted edge can also depend on poorly estimated coefficients; selecting the weekly maximum amplifies that estimation noise. Predeclare a Gaussian/Laplace approximation from the earlier-season combined-logit fit and integrate its predictive probability with fixed 20-node quadrature, then apply the same separate-season calibration to every game. Mix the corresponding discrete PMFs so cover/push answers stay coherent; rank by the resulting expected pool reward, using the existing push/tie rules. No lower-bound haircut or uncertainty threshold selects a side. **Measured inputs:** `artifacts/pick_probability/20260929T192747Z/per_game.parquet` (1,503 rows, 107 weeks), `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet` (1,537, retaining pushes), `data/processed/game_features_weak_stack.parquet` (4,902). Protocol C: fixed weekly contender population, nominee Brier primary plus weekly Best Pick reward and ordinary opener metrics, F=3, B=6, K=6, **713 looks**; all fit parameters and covariance come from earlier folds. **Read/checked:** BET-04, MOD-06/20, POL-09, LEAD-53/70/72 and `docs/confidence_top_calibration.md:5-9`. Existing temperature calibration preserves ordering; heterogeneous parameter uncertainty can alter ranking. This is neither a team-model replacement nor a sizing haircut. Units: integrated-probability adapter, 15–20 calls; paired nomination replay, 20–25. Rank rationale: no source joins or new signal family; expected benefit is mainly the Best Pick bonus, with small weekly sample size disclosed. |

Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly noon boundary. No closing inputs.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros. LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs. Reused archives are retrospective; freeze survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`.

Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four contrasts × IS/OOS/gap × K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly reward/nominee Brier; 88 adds all-game/last-game total MAE. No unlisted variants.

## Tried
Read the assigned row, Protocol C, research rules and shell workaround. No fit or score.

### Unit 1 implementation fixed before outcomes
Adapter unit only; paired five-arm nomination replay remains unit 2 as the row specifies. Use the existing standardized four-term ridge (0.001), its inverse penalized Hessian, and fixed 20-node Gauss-Hermite integration. Fit through Y-3; reserve Y-2 without choosing a new hyperparameter; fit one positive temperature per arm on Y-1. Reweight each archived discrete PMF's home/away masses to each quadrature-node probability while retaining its push mass, mix, then calibrate using the same mass-preserving transform. Reconstruct archived PMFs and check parity; reconstruct dated market moves in both arms, retaining only source-complete games (including pushes). Report the two-arm adapter's accuracy/Brier/log-loss/RPS, IS/OOS/gap, fold coefficients, five equal-width reliability bands and decisive record; 10,000 paired hierarchical season/week bootstrap draws, seed 85, half-credit zero draws. Keep all prediction/PMF rows in tests/scratch/codex/lead85_unit1. The study's 713 declared looks remain charged; unit 1 makes no nominee-Brier or weekly-reward claim.

## Record commands
None yet; the orchestrator runs any registry commands serially.

## Next
Inspect sources and upstream cutoffs, implement scripts/lead85_unit1.py, run once.

## Open
Source, clock, PMF and chronology contracts remain to be checked. No new tests, Git mutation or publication.
