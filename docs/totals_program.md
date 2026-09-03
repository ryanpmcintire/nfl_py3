# Totals program: multi-session build queue (POL-12 expansion)

**Status:** program declared 2026-09-03 by owner directive ("a massive
backlog of work for this that just hasn't been invented yet"). This document
owns the queue; each session below gets its own predeclaration doc before it
runs. Nothing here is a result.

**Shipped baseline (read, `docs/totals_model.md`, `docs/totals_model_wave2.md`):**
market-residual totals regression (k=0.1 blend, MAE +0.0008, P+ 0.583),
65-column wave-2 view served live in the tiebreaker at the same 0.1 weight.
The pool submits a tiebreaker total every week; everything below serves that
card or the research behind it.

## Data feasibility (measured 2026-09-03, local snapshots only)

- Full-game `total_line` + over/under odds: present (4,742/4,902 schedule
  rows). The over/under classifier (session 1) is feasible today.
- Team totals, first/second-half lines: ABSENT from nflverse schedules. Those
  tracks need Odds API alternate markets (quota-gated) or a new archive, so
  they sit behind a feasibility look of their own, not behind modeling.

## Session queue

1. **Over/under classifier** (next executable research session): binary
   over/under pick vs the total, production pipeline mirrored (impute →
   scale → Ridge(10), expanding walk-forward, min-500), opener-graded
   paired evaluation against the market-total baseline implied by over/under
   prices. Predeclared in `docs/totals_over_under_screen.md` (this program's
   first predeclaration, written with it). Family
   `totals_over_under_on_production` declared open; assign/record NOT run
   (rotation pools read zero unspent on 2026-09-03) — scoring waits on fresh
   blocks or the 2026 prospective season.
2. **Totals feature waves**: pace (wave 2 shipped), weather×total
   interactions, officiating pace/flags, rest/travel totals splits — each its
   own predeclaration, each marginal to the shipped blend.
3. **Totals calibration**: probability quality of over/under/total-distribution
   outputs (Brier/log-loss/ECE diagnostics mirroring the ATS calibration work).
4. **Team-totals/H2 feasibility look**: Odds API alternate-market coverage and
   cost audit (quota-gated; do not spend until the MKT-13 headroom rule clears
   it), else archive scout.
5. **Tiebreaker integration**: serve the classifier's over/under lean and the
   calibrated total distribution in the tiebreaker report; prospective
   challenger tracking for every totals read that touches the played card.
6. **Second-half/live totals**: only after 1–3 exist; needs a live line source
   that does not exist in this repo (no design until the source exists).

## What this program may not do

Spend an exhausted rotation window (verified zero unspent in every pool on
2026-09-03), tune the shipped k=0.1 blend against the same seasons, or
present the tiebreaker's served total as an edge — it is a guess with
honestly reported ±10-point error bars.
