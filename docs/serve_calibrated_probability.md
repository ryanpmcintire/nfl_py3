# The served cover chance: where the number on the board comes from

## Why the number changed source

The raw model's per-game probability was inverted against its own record.
**Measured**, `docs/four_term_probability.md` (MKT-18), 1,503 opener-graded
games 2020-2025: picks the raw model rated 50-52% won 56.0%, picks it rated
62%+ won 50.7% -- accuracy fell as its stated confidence rose. That number was
stripped from every reader surface on 2026-09-14. This document describes what
replaced it.

The served number is MKT-18's four-term model, refit as a production artifact:

```
logit P(home covers) = intercept
                     + a * logit(model's own opener probability)
                     + b * (signed sum of the composition flags)
                     + c * (leader-median late-week move toward home)
                     + d * (was a market move available at all)
```

`b` is one shared weight on the signed flag sum (+1 favours home, -1 favours
away), not nine separate weights -- the owner's read that the served flip chain
behaves like one large shared weight, measured in MKT-18. `d` is a nuisance
control so the fit does not confuse "no late-week move recorded" with "the
market did not move"; it carries no finding.

## Where the coefficients live and how they are refitted

`nfl-ats fit-pick-probability` writes
`artifacts/pick_probability/<UTC stamp>/coefficients.json` and `metadata.json`,
then repoints `artifacts/active_pick_probability.json` -- the same
stable-pointer pattern `artifacts/active_ats_model.json` uses. **No coefficient
is a literal in `src/`.** `pick_probability.load_pick_probability_model` fails
closed when the pointer is missing, names no artifact, has the wrong schema
version, is missing a coefficient, or carries a non-finite one. `publish-board`
and `publish-predictions` both call it before doing anything, so a publish from
an artifacts root that has an active model but no fitted cover chance refuses
rather than quietly serving the old number; a pointer that exists but is
malformed raises everywhere, including inside `resolve_card_view`.

Refit it whenever the active model changes or a season completes: the fit reads
the opener evaluation matched to the active model, so a mismatched opener
evaluation raises rather than silently fitting from another model's record.

## What the fit uses

- **Population**: every completed season in the opener-evaluation stream
  matched to the active model, opener pushes and ungraded games dropped.
  **Measured** at the first fit (`artifacts/pick_probability/20260914T232538Z`):
  1,503 graded games, 2020-2025, 34 pushes dropped, 799 carrying a market move.
- **Model term**: `home_cover_probability_at_open`, the offset-adjusted opener
  probability production actually serves. MKT-18's research arm used the raw
  pre-offset column; serving must feed the model the quantity it will be fed at
  prediction time, so the production fit uses the served one and the fitted `a`
  is not expected to equal the research fold values exactly.
- **Flags**: rebuilt from the composition members' own side builders
  (`year_one_by_game`, `division_revenge_side_by_game`, `_broad_side_flags`,
  `bye_edge_flag_by_game`, `forecast_cold_visitor_flag_by_game`,
  `build_flag_table`, `tank_zone_flag_by_game`) -- never re-implemented.
  Members the owner holds off the card (`OWNER_HELD_MEMBERS`: the interim
  head-coach tilt and the precipitation tilt) contribute zero, at fit time and
  at serve time alike, so the fitted `b` is the weight on the flags that are
  actually served.
- **Market move**: `leader_median_net`, the leader-book median late-week move
  signed toward home, from the newest `artifacts/sharp_weighted_follow`
  population at fit time and from the live intraday odds archive at serve time.
  Absent move means `move = 0` and `move_available = 0`, which is the same
  regime the 2020-2022 games sit in.

**Measured**, first fit: `a` 0.217, `b` 0.260, `c` 0.210, `d` 0.046, intercept
-0.066. Every one of `a`, `b`, `c` sits inside the per-fold ranges MKT-18
reported (a 0.09-0.32, b 0.24-0.30, c 0.16-0.25). Leave-one-season-out record
841-662 (55.95%) against the model's own 819-684 (54.49%).

## Strength words

The three words come from the artifact, not from literals. The fit takes the
out-of-season confidence distribution (`max(p, 1-p)` on the held-out season)
and cuts it at its own terciles; those cut points are stored as the `minimum`
of each strength band, with the measured landing rate of each band beside it.
**Measured**, first fit: Lean from 52.8%, Strong from 56.5%; slight won 54.3%
of 484 games, lean 54.0% of 513, strong 59.5% of 506. The board's legend reads
those numbers out of the artifact, so they move when the model is refit.

The five-band reliability table in `coefficients.json` uses the same band edges
MKT-18 reported (50-52, 52-55, 55-58, 58-62, 62+). **Measured**, first fit,
out of season: 48.9 / 56.4 / 58.3 / 59.3 / 60.2%. Accuracy rises with stated
confidence from the second band up, the ordering the raw model did not have.

## What is served, and what still runs behind it

`card_view.resolve_card_view` computes the calibrated probability from the raw
prediction card, then serves it as the card's `home_cover_probability`, its
displayed pick probability and its strength word. The pick side is whichever
side that probability favours. The Best Pick star ranks on the same number.

The nine-member flip composition still runs on every publish: its member
provenance, flip counts, disclosure notes and challenger ledgers are unchanged.
What changed is that the served card no longer takes its side from the
composition's flips -- the flips are recorded and scored as the paired
challenger arm, and the served number is the calibrated one.

Games past their pick deadline stay frozen at the number and side the pool saw
at lock, as before; the calibrated number governs every game still open.

## What a reader sees

One sentence, wherever the number appears: the chance beside each pick is how
often picks like it have actually landed, with the measured strong and slight
landing rates and the game count behind them. That sentence is built from the
artifact, so it can never drift from the model being served.

## Open

- The late-week refresh path (`pick_refresh.plan_refresh`) still recomputes
  sides from the model directly rather than from this calibrated number. Its
  ledger and the served card agree today only because the flip rules it can
  apply are switched off; folding the calibrated probability into the refresh
  decision is the next step.
- No forward read exists yet. MKT-18's look-count caveat carries over in full:
  the flags and the move were promoted and fit on largely these same games
  before this model existed, so `b` and `c` inherit that optimism. Nothing here
  is a claim that the served number beats the flip chain; it is a claim that it
  is correctly ordered against its own record, which the raw number was not.
