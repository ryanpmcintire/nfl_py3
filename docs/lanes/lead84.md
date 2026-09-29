# LEAD-84 — mix book lattices before the frozen line

## Goal
Execute LEAD-84 unit 1 without changing served predictions. **Read:** ROADMAP.md:867 and Protocol C in docs/lanes/ideation-2026-09-29c.md:9.

## State
Protocol copied before outcomes: equal-weight per-book spread/total/moneyline lattices, sharing an earlier-season fitted prior; average the PMFs, then calculate the frozen-line nonpush probability. Add one Jensen-gap term: mixture logit minus the logit from a lattice fitted to the same books' averaged constraints, alongside the current combined fit. Population: source-complete 2020–2025 archived historical openers, preserving pushes for distributions and excluding them only from conditional cover fitting. Enforce target-week, observation, bookmaker and kickoff clocks; no closing inputs. Historical tue_open captures are available anchors, not noon-capture evidence.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros. Verify upstream training cutoffs. Reused archives are retrospective; freeze survivors prospectively. One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and probability_positive. Zero crossing closes nothing; default unresolved_below_power. RPS is primary; RPS alone does not authorize a served side change. F=3, B=8, K=4; L=(27K+B+4)(F+1)+25=505 looks. No unlisted variants. Unit 1 constructs the two fixed distributions; unit 2 is the paired probability-term replay.

## Tried
**Measured:** one successful `.tools/uv.exe run --no-sync python scripts/lead84_unit1.py` after one pre-fit pandas conversion failure. Local UV_CACHE_DIR was required. 131 files/171,190 quotes yielded 1,321 declared nonpush games plus 33 pushes; 18,415 book panels. Saved 3,350 distribution pairs, 51,184 projections, and 714 outer rows. Prior games: 208/424/640 for outer 2023/2024/2025. Scoped Ruff format/check, comment/docstring audit, and saved-artifact hash/index/probability checks pass.

Unit-1 numerical declaration before fitting: reuse LEAD-71's integer margin grid −100..100 and entropic moneyline/spread projection (ridge 1e−6, bounds ±40); total sets variance as in that lattice. Select one common historical line band at the books' mean spread, using only source-complete games through Y−3; share its empirical prior and fitted variance across both constructions. Average de-vigged win/cover probabilities, spread points and total points equally for the comparator. Save all fit/tune/calibration/outer rows and PMFs, with pushes, for unit 2; training rows are explicitly optimistic IS. Unit 1 reads no held-out outcomes for scoring and spends zero performance looks; the full replay retains the declared 505-look family. No combined coefficients, decisive record, effect intervals or probability_positive can be reported before that replay.

## Record commands
None: zero performance looks, no effect estimate or terminal verdict. The orchestrator records the unit-2 results; planned family remains 505 looks.

## Next
Review incompatible source constraints before the unit-2 five-arm calibrated replay. Report: docs/lead84_unit1.md; features, PMFs, fold parameters and provenance: tests/scratch/codex/lead84_unit1/. Reconstruct timestamp-matched moves for every year in that replay.

## Open
**Measured:** PMF normalization error ≤8.88e−16 and solver stationarity ≤3.15e−7, but maximum constraint-price error is 0.134682. Seventeen unique book panels in nine games have incompatible win/cover constraints; 17 fold projections exceed one percentage point (two games). **Inferred:** optimizer convergence does not establish price feasibility. Population and protocol remain unchanged; source repair needs review before replay. IS/OOS effects/gaps, combined coefficients, decisive record, intervals and probability_positive remain unmeasured. No closure, served change, registry write, publication or Git mutation.
