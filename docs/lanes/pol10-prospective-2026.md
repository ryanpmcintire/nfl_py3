# POL-10 prospective 2026 evidence

## Goal

Build the 2026 prospective scorecard for Weeks 1-2 (Week 3 as soon as graded)
from what was actually recorded before kickoff: served card record at the
pool line, Best Pick record, the served four-term probability's Brier/log
loss vs 0.5 and vs a market-only baseline, and each ACTIVE_PROSPECTIVE
challenger's paired record vs served on decisive games only.

## State

- Measured 2026-09-24: ran `nfl-ats settle --start-season 2026
  --no-refresh-results` (local recorded results, no live fetch needed;
  `artifacts/settlement/graded_decisions.parquet` was already current for
  weeks 1-2 from the prior session's Thursday refresh) then
  `scripts/prospective_scorecard_2026.py` (new, one run), writing
  `artifacts/prospective_scorecard/20260924T193946Z/summary.json`.
- Served card (`paper_decisions`/`played` arm = final pick after every
  revision, graded at `decision_home_spread`, the pool line): Week 1
  10-6-0 (n=16), Week 2 14-2-0 (n=16), to date 24-8-0 (n=32). Week 3 (16
  games) recorded but ungraded as of this run (0/16 final; a Thursday game
  played tonight had not posted a final `result` at run time).
- Best Pick (`paper_decisions`/`best_pick` arm, one nominee/week): Week 1
  0-1, Week 2 1-0, to date 1-1 (n=2); Week 3 nominee pending.
- Served four-term probability (`recorded_home_probability` = latest
  `pick_revisions.new_home_cover_probability` where the pick was revised,
  else `home_cover_probability_excluding_push` from the forecast
  artifact named in `artifacts/clv_ledger/decisions.parquet.forecast_artifact`;
  13 of 32 games were revised at least once) vs outcome, decisive games only:
  - Week 1 (n=16): Brier 0.2383 served vs 0.2500 coin-flip vs 0.2517 market;
    log loss 0.6692 served vs 0.6931 coin-flip vs 0.6965 market.
  - Week 2 (n=16): Brier 0.1933 served vs 0.2500 vs 0.2494 market; log loss
    0.5773 served vs 0.6931 vs 0.6920 market.
  - To date (n=32): Brier 0.2158 served vs 0.2500 vs 0.2505 market; log loss
    0.6232 served vs 0.6931 vs 0.6942 market.
  - Market-only baseline = de-vigged `home_spread_odds`/`away_spread_odds`
    from the same recorded forecast artifact (American-odds implied
    probability, normalised to remove the hold), not a separate model.
- Challenger paired record vs served, decisive games only, weeks 1-2, all 47
  ACTIVE_PROSPECTIVE challengers present in `challenger_decisions.graded`
  (of 63 registered ACTIVE_PROSPECTIVE ids; 16 record through other ledgers
  -- best-pick/tiebreaker/lattice challengers -- not `challenger_decisions`):
  full per-challenger `n / challenger_beats_served / served_beats_challenger
  / ties` list, sorted only by challenger_id, is in the summary.json
  `challenger_paired_vs_served` array (47 rows, n ranges 3-32 depending on
  each challenger's own recording cadence). No challenger is highlighted;
  this is a status report, not a promotion claim.
- **n=32 decisive games (2 weeks) is far too few to decide anything.** No
  verdict, closure, or promotion follows from these numbers; this is the
  read-only prospective status POL-10 asks for.
- A first run of the script had a merge bug (joined every forecast snapshot
  to every game_id instead of matching `(game_id, forecast_artifact)`
  together), doubling every count to n=64/n=16(paired). That run's artifact
  directory (`20260924T193913Z`) was deleted; only `20260924T193946Z` (the
  corrected run) is kept.

## Tried

- Considered recomputing probabilities via the `opener-evaluation` /
  `sunday_market_probability` walk-forward pipelines instead; rejected
  because POL-10 and the task both require what was actually recorded
  pre-kickoff, not a retrospective recompute, and those pipelines still stop
  at season 2025 (per `docs/lanes/confidence-best-pick-unification.md`).
- `prospective_accuracy`/`prospective_accuracy_metrics` in
  `src/nfl_ats/prospective_scoring.py` give forced-pick ATS accuracy per
  entrant but carry no probability field for challengers (`challenger_decisions`
  has `pick_side`/`decision_home_spread`/`edge` only), so Brier/log loss is
  reported for the served card only, not per challenger -- consistent with
  AGENTS.md ("one calibrated probability... a member returning only pick
  side is not being asked to calibrate").

## Next

- If a follow-up wants challenger probability calibration (not just paired
  ATS record), that needs a probability field added to
  `challenger_decisions.parquet`'s recording path first; out of scope here.
- Re-run `scripts/prospective_scorecard_2026.py` once Week 3 grades (likely
  within days) to add a third graded week; still far below a decision
  threshold.
- No commits/pushes/publication were made (task scope: read-only audit +
  one new script). The script and this lane are uncommitted.

## Open

- None blocking. Script is idempotent and safe to re-run once more results
  are graded.
