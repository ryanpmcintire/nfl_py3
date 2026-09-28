# Week 3 variance assessment

## Goal

Assess whether the bad week establishes an implementation or modeling mistake, using the completed games and saved validation artifacts.

## State

- **Measured:** public nflverse scores retrieved September 28 at 10:02 UTC give the published card **5-10**, raw card **9-6**, with all four disagreements lost. Frozen sides match the graded published sides. The fitted probabilities imply 8.19 wins; five or fewer has probability 8.15%, with a conditional central 95% outcome range of 4-12 wins. All four disagreements losing has conditional probability 4.53%. Both calculations assume correct probabilities and independent outcomes; neither is a posterior probability that the model is sound.
- **Measured:** on historical disagreements, the season-held-out combination went **291-230**. Across the same 1,503 games it went 863-640 versus raw 802-701: **+4.06 accuracy points**, week-bootstrap 95% range **+1.53 to +6.66**, `probability_positive=0.99905`. Brier scores: combination **0.244838**, raw **0.251715**, neutral market **0.250000**. Brier improvement over raw is 0.006877 [0.002823, 0.011167], `probability_positive=0.99975`.
- **Measured:** the chronological subset has 354 disagreements, combined **202-152**. Across its same 1,047 games: combined 603-444 versus raw 553-494, **+4.78 points** [**+1.62, +8.08**], `probability_positive=0.998375`. Brier: 0.244855 versus 0.251391 and neutral 0.250000.
- **Measured:** all six season rows favor the combination on accuracy and Brier. In-sample accuracy 57.55% versus season-held-out 57.42% is a 0.13-point gap. All six fitted model, flag, and movement coefficients retain positive signs; per-fold coefficients are in the active probability metadata.
- **Read:** `artifacts/pick_probability/20260927T161329Z/metadata.json:156` explicitly says features were selected using the evaluated seasons and this is not an untouched outer test. `pick_probability_fit.py:390-418` holds fitting out by season and also computes forward-only fitting. Neither removes prior feature-selection bias. The bootstrap ranges above condition on the selected features and do not account for that search.

**Measured reliability:** rows show games / mean estimated cover chance / observed cover rate. These are descriptive cells, not newly selected cutoffs.

| Estimated band | Combined | Raw model |
| --- | --- | --- |
| 50-55% | 835 / 52.56% / 56.41% | 747 / 52.38% / 52.74% |
| 55-60% | 425 / 57.26% / 55.29% | 468 / 57.34% / 55.13% |
| 60-65% | 163 / 61.94% / 68.10% | 219 / 61.92% / 52.51% |
| 65-100% | 80 / 69.18% / 57.50% | 69 / 67.40% / 50.72% |

Neutral market: 1,503 / 50.00% / 49.63%. Audit family: three models across two aggregate views and six season rows, plus nine reliability cells; 33 descriptive cells, no new fit or threshold search.

## Tried

Fetched public finals, checked frozen sides, and remeasured saved prediction rows with the existing week-blocked bootstrap: 20,000 draws, seed 20260928. Detailed season rows, reliability, intervals, and assumptions are saved in `.tmp/week03-variance-review.json`; replay script is `.tmp/week03_variance_review.py`. The prior mechanism trace found exact arithmetic and lock agreement for its three target games. No production changes or research closure.

## Next

Assessment complete. Any corrective modeling decision needs a predeclared prospective comparison and separation of feature selection from final evaluation; this week alone does not identify a term to disable.

## Open

**Inferred:** an unlucky week is plausible and the historical combination has support, but incomplete independent validation prevents attributing the result confidently to luck. No implementation error was identified in the traced paths. The shared-weight and calibration choices remain empirical questions; no signal is closed by this review.
