# Pairwise co-absence excess by unit (predeclaration)

**Status:** predeclared 2026-09-03, BEFORE any pairwise co-absence number
was computed. Sections 0–7 are frozen; section 8 (Results) is appended
after the run and nothing above it is edited afterwards.

**Motivation, disclosed (sequential design, not a first look):**
`docs/absence_dependence.md` §8 found the full-sit estimand degenerates
(exactly 0 whole-unit sits in ~4,964 team-games per unit — neither model
produces them at 6–15 contributors). Coupling, if present, lives at the
pair level (the QB-WR1-LT states the kernel scores), so this document
re-asks the question there. The pairwise numbers below have never been
computed; only the degenerate aggregate motivated the redesign.

**Owning work package:** PER-10 (owner directive 2026-09-03: comprehensive,
never the independence shortcut) and SIM-03. Files: this document,
`scripts/absence_pairwise_screen.py`,
`tests/test_absence_pairwise_screen.py`,
`artifacts/absence_pairwise/`.

---

## 0. Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

---

## 1. What this slice asks (frozen)

For each unit, do established-contributor pairs sit out the same game more
often than their individual rates imply? The pooled log-ratio is the
pairwise coupling parameter a second-order joint producer needs.

## 2. Population (frozen)

Identical contributor frame as `docs/absence_dependence.md` §2 (amended):
established contributors, schedule-joined team-games, ≥100-snaps
data-presence gate, seasons 2016–2025 REG. Pairs are unordered players
sharing ≥10 overlapping established team-games (predeclared overlap floor —
below it a pair's joint rate is noise by construction, excluded and
counted).

## 3. Statistic (frozen)

Per unit, over eligible pairs:

- pooled observed joint rate = (games both absent) / (games both established);
- pooled expected rate = mean over pairs of (q_i · q_j) with q_i, q_j the
  pair members' own absence rates on overlapping games;
- excess = observed / expected, with a team-season block bootstrap CI
  (2,000 samples, seed 20260906; resample team-seasons, recompute).

## 4. Controls (frozen, diagnostic)

Within-team-season permutation of absence labels (each player's total held
fixed): the observed excess must clear the permuted null center.
Diagnostic only.

## 5. Leakage (frozen)

None applicable (completed seasons, no pregame application, no model join).
Tests pin pair construction, the overlap floor, and bootstrap determinism
on synthetic frames.

## 6. Test contract (release-blocking)

`tests/test_absence_pairwise_screen.py` covers, without network access:
pair enumeration on a hand-computed frame (independent pairs ≈ 1.0,
perfectly-coupled pair ≫ 1), the ≥10-game overlap floor, bootstrap
determinism, and empty-population fail-closed.

## 7. Decision rule (frozen)

No registry entry under any outcome of this slice: none of the CLI's
effect units honestly contains a co-absence excess ratio, and forcing one
in would corrupt the pool (the same unit-misuse rule Stage 1's doc states).
Descriptive rates live in §8, the artifact, and the row; they inform the
producer design directly. No card, model, window, or ATS comparison under
any outcome.

## 8. Results (added after the run, 2026-09-03)

Measured by `scripts/absence_pairwise_screen.py` in one run (artifact
`artifacts/absence_pairwise/20260903T215502Z/results.json`,
220,757 contributor player-games):

| Unit | Pairs | Observed joint | Expected (pooled) | Excess | 95% CI | Permuted null | Obs percentile |
|---|---|---|---|---|---|---|---|
| OFF_OL | 4,732 | 0.03985 | 0.05462 | 0.7295 | [0.6539, 0.7363] | 0.6119 | 1.0000 |
| OFF_SKILL | 19,383 | 0.03106 | 0.03873 | 0.8020 | [0.7181, 0.7907] | 0.6491 | 1.0000 |
| DEF_FRONT | 18,192 | 0.02476 | 0.03254 | 0.7609 | [0.6817, 0.7610] | 0.6476 | 1.0000 |
| DEF_SECONDARY | 6,969 | 0.04329 | 0.05462 | 0.7926 | [0.7083, 0.7914] | 0.6366 | 1.0000 |

What this implies for the decision, before what is wrong with it: pairs
sit out together 20–27% LESS often than pooled independence implies (every
CI below 1.0) — yet MORE often than label-shuffling implies (observed at
the 100th percentile of all four nulls, which centre 0.61–0.65). Both
statements hold at once: heterogeneous marginals explain part of the gap
(depth players miss more games than starters, so a pooled q overstates
joint absence), and a residual positive coupling survives on top (shared
causes: illness, short weeks, game-day inactives decided jointly). What is
wrong with it: non-participation conflates rotation (negative dependence
by coaching design) with injury/illness (positive); the net below 1.0
means rotation dominates at the pair level, so an independence-based
producer OVERSTATES joint absence by 25–40% — the comprehensive producer
must use heterogeneous per-player marginals first, then a small positive
coupling term, never pooled independence. No registry entry (frozen §7);
the numbers go directly into the producer design.

## Producer design inputs (measured, for the future builder)

1. Per-player marginal absence rates (not one pooled q).
2. Pairwise coupling residual: observed/null ≈ 1.13–1.25 across units
   (0.73–0.80 observed vs 0.61–0.65 null) — the magnitude a copula or
   shared-frailty term must reproduce.
3. QB layer stays separate (named depth, deterministic backup elevation);
   game-script coupling stays OUT of any pregame producer (not pregame).
