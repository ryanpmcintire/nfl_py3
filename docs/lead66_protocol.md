# LEAD-66 frozen protocol

Copied unchanged from the lane declaration saved before outcome scoring.

## Verbatim roadmap row
| LEAD-66 | ⬜ | Grade the whole margin distribution, not one cover bit (rank 1 of 8 by held-out gain per unit of work) | **Added 2026-09-29 (unmeasured).** Mechanism: the pool line
is a single number, but the market's real belief is a distribution with mass at 3, 7, 10, 14; a cover bit spends one Bernoulli per game and cannot see whether the discrete margin
read (MOD-18 K1-K3, `DISCRETE_PUSH_READ_SERVED`) beats the smooth or pooled read, which is why every such comparison sits near 2-point resolution. Score instead the ranked
probability score and log score of the full predicted margin PMF at every integer threshold, on the same games. Predeclared: population every game with a frozen opener 2020-2025
(1,503 non-push pairs; the 2009-2019 close-proxy population as a labeled second read), LOSO by season for any fitted piece, three arms only (market-line-centred empirical residual
PMF conditional on line = baseline, served discrete lattice, served model-centred lattice), 3 looks, per-season and pooled metric with a season-block bootstrap,
`probability_positive` and the IS-OOS gap reported, calibration by predicted-tail band. Not a duplicate of: MOD-18 K1-K3 and `DISCRETE_PUSH_READ_SERVED` (row 693; graded on cover
accuracy at the opener, not on the whole PMF), ENG-46 (Brier and log loss of the cover, one threshold), MKT-17..19 (cover probability). A challenger-vs-served gain here decides
only whether the served cover, push and flip-line answers read the better PMF; it never flips a pick alone. Record through `weak-signals record` (units: rps_improvement); zero
crossing closes nothing. About 20 tool calls: one script over `artifacts/margins/` predictions plus the opener stream. |

## Implementation declaration
Freeze margins `20260929T192312Z`, opener `20260929T192743Z`, and their matching
feature hash. Primary: all completed 2020-2025 opener games, including pushes;
decisive cover records exclude pushes. Secondary requires complete 2009-2019
same-configuration predictions; inventory and report missing coverage otherwise.
Three arms: line-conditioned empirical residual PMF recentered at the market;
served lattice conditioned on market line and tilted to archived model centre;
same lattice conditioned on that model centre and tilted to it. Use production
band constants and tilt, with no parameter search. Recover the archived centre
from the production historical lattice and stored conditional cover probability;
require reconstructed cover/push/loss agreement before any scoring.
Chronological LOSO: each season's OOS PMF uses only earlier seasons in the
production five-season window; no held-out-season outcomes enter its mass fit.
Archived model forecasts stay frozen; this tests PMF mapping, not model refits.
IS diagnostic adds the held-out season to the same mass pool (explicitly optimistic).
RPS sums squared CDF error at every integer threshold; log score is exact negative
log mass (zero mass gives infinity). Three paired arm contrasts; game-weighted
season-block bootstrap, 10,000 replicates, seed 20260929, percentile 95% intervals,
half-weight ties for probability_positive. Report per-season IS/OOS and OOS-minus-IS
gap, per-fold tilt coefficient ranges, RPS decisive records and PMF-implied cover
disagreement records (diagnostics only; no side is served). Tail calibration uses
five fixed equal-width probability bands per arm, game-equal weights over integer
thresholds -70..70; 3 primary looks plus 15 descriptive calibration cells, 18 total.
No fitted combined-pick coefficients: no pick model is fitted or served by this unit.
Preserve prediction-level scores in an assigned Markdown report; do not refit sources.

