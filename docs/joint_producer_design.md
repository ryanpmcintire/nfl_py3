# Joint lineup-scenario producer: comprehensive design (PER-10)

**Status:** designed 2026-09-03 under owner directive "comprehensive, never
the independence shortcut". This document is the frozen design; implementation
is queued as the next PER-10 build. Nothing here is implemented, scored, or
wired.

**Parents:** `docs/injury_scenario_mixture.md` (the kernel: needs joint
probabilities + per-scenario margin centers), `docs/absence_dependence.md` +
`docs/absence_pairwise_dependence.md` (measured dependence structure, the
calibration targets below).

## Measured inputs (all read, none re-derived here)

- Pairwise co-absence runs 20–27% BELOW pooled independence but ABOVE
  label-shuffling (observed/null ≈ 1.13–1.25): heterogeneous per-player
  marginals first, small positive coupling on top. Pooled independence
  overstates joint absence 25–40% and is out.
- Named-QB depth supplies QB1-active probability plus QB1/QB2 identities
  and values (`src/nfl_ats/quarterbacks.py`); the player pipeline supplies
  expectation-aggregated marginal injury burden, not joint states.
- Full-unit sits never occur (degenerate, reported) — the producer must put
  ~zero mass there, which falls out of per-player marginals naturally.

## Producer structure (frozen design)

Per game, strictly pregame inputs only:

1. **Marginals:** per-player active probability from the learned
   availability rates (existing), QB1 from named depth (existing).
2. **Deterministic cascades (not probabilistic):** QB1-out implies QB2-in
   with probability 1 (depth-chart logic); game-day active/inactive lists,
   when observed pre-decision, collapse their players to 0/1 (prospective
   inactives feed, separate ledger).
3. **Stochastic coupling:** unit-level shared frailty — each unit draws one
   health shock per game; members' absence log-odds shift jointly by a
   unit-specific loading. Illness weeks (FluView home-market elevated flag
   set) raise the skill-unit shock variance by a predeclared multiplier.
   Loadings are calibrated so simulated pair co-absence reproduces the
   measured observed/null ratios (1.13–1.25), NOT fit to ATS outcomes.
4. **Enumeration:** top-K most uncertain players (K=8, predeclared) form at
   most 256 joint states; keep the smallest set covering 99% of the mass
   (cap 64 states); every other player fixed at their modal state. Output
   feeds the kernel's revision contract verbatim (probabilities sum to 1,
   complete active/inactive partitions, observed/effective timestamps).
5. **Out of scope, disclosed:** per-scenario MARGIN CENTERS still need the
   EPA/value-to-points mapping (the kernel's blocker 2, a separate
   decision); game-script coupling stays out (not pregame); the frailty
   loadings are calibrated to descriptive rates, never to ATS outcomes.

## Validation (frozen; no window, no ATS)

- **Calibration on realized 2024 lineups:** the producer must rank realized
  joint states above random states from its own distribution (rank test,
  predeclared threshold: median realized rank in the top half — a
  calibration floor, not an edge claim).
- **Leakage tests:** every input timestamped strictly pre-decision;
  post-decision injury revisions cannot move a scenario (mirror the
  kernel's own regression style).
- **Determinism:** fixed seeds, bit-identical reruns.
- **Coverage:** every game gets ≥2 positive-probability scenarios summing
  to 1, or the game fails closed (kernel contract).

## What this design may therefore claim

At most: a comprehensive, calibrated joint-probability producer whose
dependence structure matches measured co-absence. It may not claim ATS
value, optimal K/loadings (both predeclared, never tuned here), or margin
centers. ATS evaluation needs the mapping decision plus open windows and
is queued separately.
