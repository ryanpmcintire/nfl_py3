# Variance timing: when does the ATS residual materialize?

Question: decompose final ``|margin - market line|`` by quarter-of-origin.
What share of the ATS residual accumulates Q1/Q2/Q3/Q4? How much could a
PERFECT in-game model starting at halftime recover?

Status: measure-only decomposition, first run 2026-08-22. No selection strategy,
nothing recorded to the registry, no window spent.

## Method

``scripts/vardec_lategame.py``, artifact
`artifacts/vardec_late/20260822T215524Z/results.json` (**measured**):

- Population: REG 2009-2025, snapshot `data/raw/20260817T235649Z/schedules.parquet`
  joined to PBP snapshot `data/pbp/raw/20260817T184927Z` on ``game_id``;
  **4,431 games**, pushes kept (**measured**).
- Per game, every play's ``score_differential`` (posteam perspective) is mapped
  to a home-perspective differential and read at the LAST play of each quarter;
  quarter deltas sum to the final margin by construction. The market line is a
  pregame constant, so all ATS-residual variance lives in those deltas.
- **Data defect found and fixed** (**measured**): the narrowed snapshot stores
  `posteam` as modern franchise codes (`LV`/`LAC`/`LA`) on 75,822 rows while the
  schedules use historical codes (`OAK`/`SD`/`STL`), which silently blanked the
  sign on those plays and left 67 games with stale quarter boundaries (max 7
  points off). A per-game alias resolver restores reconciliation:
  99.75% of games within half a point of the schedule result, 11 residual
  mismatches are upstream nflverse quirks, coverage ratio 1.000 (**measured**).
- Variance share of quarter q = Cov(d_q, R)/Var(R), R = ``ats_margin``.
  Raw shares sum to 101.31%, not 100%, because corr(line, residual) = +0.0285
  (the deltas sum to the margin, and Cov(line, R) > 0); normalized shares close
  to 100% (**measured**).
- Halftime-recoverable bound: OLS of R on [1, true first-half differential,
  line], week-blocked bootstrap (5,000 draws). Upper-bound proxy for a perfect
  in-game model: it conditions on the TRUE halftime state, which any real model
  only estimates (**inferred**, stated in code docstring).
- Lead-change volatility: WP-favourite flips between consecutive plays with
  defined WP; one-score finish = |margin| <= 8.
- MINED FAMILY DISCLOSURE: five Q4-EPA-swing correlates chosen after the data
  landscape was known; roughly one spurious 95% exclusion expected by chance.
  An interval crossing zero is NOT grounds for rejection (AGENTS.md).

## Results

All numbers from the artifact above unless marked otherwise (**measured**).

### Variance by quarter

| Quarter | mean (home persp) | sd | variance share | cumulative | absolute share |
|---|---|---|---|---|---|
| Q1 | +0.689 | 6.856 | 23.59% | 23.59% | 22.68% |
| Q2 | +0.801 | 8.331 | 32.79% | 56.39% | 28.01% |
| Q3 | +0.154 | 7.128 | 23.71% | 80.10% | 23.59% |
| Q4 | +0.369 | 7.428 | **20.82%** | 100.92% | 24.79% |
| OT | +0.019 | 1.013 | 0.39% | 101.31% | 0.93% |

Baseline ATS-residual sd = **13.130 points**. The residual accumulates almost
uniformly (~21-33% per quarter); Q2 is the largest single quarter, Q4 the
smallest full quarter despite end-game drama.

### Q4 alone

**20.82%** of ATS-residual variance is created in Q4 (normalized 20.55%).

### Q4-EPA-swing vs pregame observables (should be ~0 if irreducible)

| Correlate | r | week-blocked 95% CI | P(neg) |
|---|---|---|---|
| wind mph (outdoor known) | -0.0264 | [-0.0597, +0.0087] | 0.927 |
| rest diff (home-away) | +0.0018 | [-0.0244, +0.0289] | 0.443 |
| primetime slot | -0.0002 | [-0.0273, +0.0283] | 0.511 |
| abs spread | **+0.0862** | [+0.0544, +0.1160] | 0.000 |
| dome/closed | +0.0106 | [-0.0183, +0.0400] | 0.237 |

Weather/rest/primetime/dome are indistinguishable from zero, consistent with an
irreducible Q4 (**measured**; probability_positive for wind is 0.073, i.e. weak
evidence toward a small negative wind effect — unresolved, not closed).
abs_spread excludes zero (+0.086): single mined look, no multiplicity
correction, reported as continuous evidence, not a finding (**measured**;
mechanism guess: favourites bank larger positive-EPA leads, so swing magnitude
tracks line size — **inferred**).

### Lead-change volatility and one-score rates

Mean 7.42 WP-favourite flips per game (median 6); 77.1% of games see 3+ flips.
One-score finishes fall monotonically with line size: 56.8% (|line| <= 3),
53.3% (3-7), 46.4% (7-10), 34.8% (>10), overall 51.8%. Mean flips also fall
with line size (7.66 -> 5.65) (**all measured**).

### Halftime-recoverable estimate

A perfect model starting at halftime (knowing the TRUE first-half differential
and the line) removes **48.91% of residual variance** [46.67, 51.19],
residual sd 12.97 -> **9.29 points**. The H1-differential coefficient is 0.877,
so first-half leads partially regress (**measured**). This is an UPPER bound:
any real halftime model estimates the state from score only (same thing here)
but cannot predict the remaining second-half noise, whose sd is ~9.3 points
regardless (**inferred**: the bound equals Var(H2 | S_2) since H2 is nearly
uncorrelated with S_2).

Implication paragraph (ties to the halftime-bound lane): roughly half of the
total ATS-residual variance is in principle recoverable from perfect halftime
information, and that recovery is entirely about pinning down the first-half
state — the post-halftime flow itself adds ~9.3 points of irreducible sd, of
which Q4 contributes only ~21%. Concretely: a hypothetical perfect in-game
model would cut the residual sd from 13.13 to 9.29 points (-29%), which at the
project's usual +1-point edge translates to a Phi(edge/sigma) forced-pick
accuracy ceiling far above anything historical models have shown; the practical
gap between this bound and any implementable halftime model is dominated by how
little a real model can extract from the halftime score beyond what the
pregame line already knows — the regression's own line coefficient (-0.43)
shows most halftime-state information is already priced (**inferred** from the
measured coefficients).

## Gates

From `results.json` "gates" (**measured**):

| Gate | Result |
|---|---|
| coverage_ratio >= 0.95 | PASS (1.000) |
| margin reconciliation >= 99% within 0.5 pt | PASS (99.75%) |
| covariance shares sum to 1 within 0.02 | PASS (1.0131 raw; closure note records the Cov(line,R)>0 explanation) |

## Classification

Category 3 / descriptive measure-only. Nothing here is terminal; nothing was
recorded (`weak-signals record` deliberately not run — measure-only lane,
matching `docs/vardec_noisefloor.md` convention; owner may record if this feeds
a rotation family later). The abs_spread correlate is the only interval
excluding zero and carries the mined-family disclosure.
