# LEAD-71 — market-implied margin lattice, unit 1

## Goal
Inventory local Tuesday-open Odds API `h2h`, spread and total coverage for 2020–2025.

## State
**Measured:** unit 1 complete, exit 0; 1,602/1,693 games (94.62%) have all three markets;
1,601 at the same book. All six seasons exceed 80%; zero archive errors. Exact census: no CI.
**Read:** ROADMAP.md:854 declaration below was saved before source counts or outcomes.

### Predeclared protocol (verbatim roadmap row)
| LEAD-71 | ⬜ | Market-implied margin lattice from moneyline, spread and total (rank 6 of 8) | **Added 2026-09-29 (unmeasured).** Mechanism: the spread quotes one point of the distribution, but the moneyline prices P(win) and the total prices scoring, together a market-implied lattice for the margin whose mass near 3 and 7 can differ from the pooled empirical residual PMF the served discrete read uses; the pool line is exactly where a difference is a mispricing of push and cover mass. Unit 1 is a source inventory: which seasons of the Odds API archive carry `h2h` at the Tuesday open beside spreads and totals (a grep of this file found no moneyline row). Unit 2, only if 2020-2025 coverage exceeds 80% of games: build the implied lattice (family and fitting choices predeclared), grade it with the LEAD-66 whole-PMF score against the empirical-residual lattice, LOSO by season, 2 looks, then enter the implied cover probability as one fitted term in the four-term model. Never a flip rule. Checked, not duplicated: MOD-18 K1-K3 (empirical residual lattice), SKY-04 (book timing, not products), LEAD-61 (half-game markets). About 25 tool calls across two units. |

Unit 1 reads source metadata and quote fields only; no game outcomes, fitting or scoring.
Population: 2020–2025 games; target: pregame Tuesday-open availability of all three markets.
Terms/folds: none in inventory; unit 2 reserves LOSO season folds and two predictive looks.
Metric: source coverage, with seasons and denominator stated; unit 2 requires strictly >80%.
IS/OOS, gap, coefficients, probability_positive and decisive-game record are not estimable in unit 1.
Zero crossing never closes a signal; one fitted calibrated probability selects the side.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead71_unit1.py` executed once;
initial uv launch failed before execution (shared-cache access denied), then used
`UV_CACHE_DIR="$TMPDIR/lead71-uv-cache"` (Windows: `$env:TEMP`) and thread limits of 2.
**Measured:** 8,834 manifests scanned, 131 valid Tuesday snapshots; report:
`docs/lead71_unit1_inventory.md`. Per-season coverage: 84.76%, 88.42%, 94.37%,
99.65%, 100%, 100%. Predictive looks: 0; reserved for unit 2: 2.
**Measured:** Python 3.12.13; `ruff check --no-cache scripts/lead71_unit1.py` passes after style-only fixes.

## Next
Predeclare unit 2 lattice family/fitting choices and use the LEAD-66 whole-PMF score,
LOSO, then the fitted four-term model. The >80% source gate is met.

## Open
Unit 2 remains unmeasured; no outcome read, fitted coefficient, IS/OOS gap,
probability_positive or decisive-game record exists from this inventory.

## Record commands
None: source inventory estimates no signal; a weak-signals/rotation command would invent an effect.
No registry command, test addition, publication or Git mutation was performed.
