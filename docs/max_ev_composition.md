# MAX-EV played-card composition — predeclaration

Written **before** `scripts/max_ev_composition.py` scored any cover rate.
No composed cover rate, delta, interval, or probability_positive below was
computed beforehand. This document freezes the rules, arms, seeds, and
disclosures for measuring the MAX-EV card: the sequential production chain
(raw model -> coach fade -> player-arrest policy) with ALL THREE discovered
edges applied on top, in this fixed order:

1. **Movement rule** (`observed_movement_threshold_1_0`, docs/observed_movement_channel.md):
   if |close - tue_open| >= 1.0 pt, follow the market side; else keep the
   incoming pick. Machinery reused VERBATIM from
   `scripts/movement_composition_eval.py` (`reload_market_lines`,
   `movement_overlay`). Data exists archive-wide (every archived game carries
   a resolved tue_open/close pair).
2. **NFL.com Friday fade** (`redteam_nflcom_out2_starters_only`,
   docs/nflcom_friday_refresh.md): flip to the opponent iff the picked team
   carries >=2 Out designations on starter-caliber players (>=50% prior-week
   snap-share proxy) on the final league injury page AND the opponent does
   not; both flagged keeps. Machinery imported VERBATIM from
   `scripts/nflcom_friday_refresh_feature.py` (`build_out_counts`,
   `attach_counts`, `apply_overlay`). Data exists ONLY for seasons 2022-2024
   (immutable snapshot `data/raw/nflcom_injuries/`); elsewhere the overlay is
   a NO-OP (counts default 0, never a flip).
3. **Protection-mismatch tilt** (`pbp08_protection_mismatch`,
   docs/pbp08_matchup_screen.md): prior-4-game top-quartile pressure-rate-
   allowed offense facing top-quartile pressure-generating defense -> back
   the DEFENSE side. Flag logic reused EXACTLY from
   `scripts/pbp08_matchup_screen.py` (`load_population`, `build_game_trait_tables`,
   `build_long_table`, `build_cells` — expanding strictly-prior quartiles,
   MIN_QUANTILE_POOL 200, window 4/min 3). At game level: flip iff the PICKED
   team's row is flagged (its offense vulnerable AND its opponent generates)
   AND the opponent's row is NOT; both flagged keeps the incoming pick. Rows
   without an assigned flag (early-season windows, pushed games absent from
   the screen's population) are NO-OP. Trait windows reach 2009-2025, so
   every archived game (2020-2025) sits deep inside the expanding-quartile
   history; only early-season window gaps make the tilt unavailable.

Tie-handling symmetry: both later overlays reuse the frozen nflcom rule
(flip iff picked flagged AND opponent unflagged; both flagged keeps). This is
the only admissible reading when both sides qualify simultaneously and was
frozen here before scoring.

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero".

## Arms (opener grade, production probability-rule chain)

All arms graded with `nfl_ats.clv.pick_correct` against `margin_vs_open`
(pushes NaN, excluded identically), paired deltas vs arm (a) in accuracy
points, week-blocked bootstrap PRIMARY and season-blocked SECONDARY,
20,000 samples, seed **20260824** (owner-specified for this task).

- **(a)** incumbent chain reproduction gate: sequential coach-fade-then-
  arrests chain MUST reproduce `0.541583499667332` (artifacts/
  overlay_subset_composition production_chain_reference.coach_then_arrest_sequential)
  to <=1e-9 or the script ABORTS.
- **(b)** chain + movement (slate-wide partial application).
- **(c)** chain + movement + NFL.com (slate-wide; a NO-OP outside 2022-2024,
  so arm (c) differs from arm (b) only there).
- **(d)** FULL MAX-EV: chain + movement + NFL.com + protection tilt,
  SLATE-WIDE PARTIAL APPLICATION — each edge applies where its data exists,
  else no-op. This is what would actually ship.
- **(d-restricted)** the SAME full-stack picks scored ONLY on games where ALL
  FOUR data sources exist (movement pair resolved [archive-wide],
  NFL.com-covered season-week, both teams' protection quartile rows assigned):
  the cleanest comparison.
- **(e)** leave-one-out marginals within the full stack: drop movement
  (chain->nflcom->protection), drop nflcom (chain->movement->protection),
  drop protection (chain->movement->nflcom, identical to arm (c)).

## Effective-n honesty

Movement exists on every archived game (the archive is conditioned on a
resolvable tue_open/close pair). NFL.com exists only for 2022-2024
season-weeks present in the immutable snapshot (~3 seasons of the 6-season
archive). Protection tilt exists wherever the pbp08 machinery assigns both
teams' relevant quartile rows (missing for early-season games whose prior-4
window is incomplete, and for games the screen's push-drop removed). Every
arm reports its own scored n and its per-edge application counts; no arm
pretends an edge fired where its data does not exist.

## Attribution upper bound (mandatory disclosure)

THREE OF THE FOUR stacked components were selected and/or red-teamed on
seasons this archive overlaps: the movement rule (+1.863 pts P+ 0.935 solo,
docs/observed_movement_channel.md) and the arrest policy were registered on
2020-2025 windows, and the NFL.com out>=2 fade was selected and red-teamed on
exactly the 2022-2024 population scored here. The protection cell was mined
on 2009-2025. These composed numbers are therefore attribution on
already-looked-at data — an UPPER BOUND, continuous evidence, never a fresh
confirmation — and the component effects must never be pooled as independent.
No rotation-registry window was spent by this measurement.

## Recording discipline

Measure-only. The script writes `artifacts/max_ev_composition/<run_id>/` and
stamps `registry/experiments/max-ev-composition/`; it NEVER writes either
registry JSON. The proposed `nfl-ats weak-signals record` line for
`maxev_full_stack` (arm (d), slate-wide partial application) is printed and
saved in metadata.json, predeclared classification
`unresolved_below_power` regardless of interval shape.

---

## Results (run 20260823T024809Z, seed 20260824)

Measured by `scripts/max_ev_composition.py`
(`artifacts/max_ev_composition/20260823T024809Z/`, registry stamp
`registry/experiments/max-ev-composition/20260823T024809Z.json`; the run
reproduced deterministically against an earlier same-seed run before this
doc section was written). Arm (a) reproduced `0.541583499667332` exactly
(gate passed). All deltas are paired accuracy points vs arm (a) on identical
scored games; week-blocked primary, season-blocked secondary.

### Arm table — slate-wide partial application (n=1503 scored each)

| arm | acc | changed vs chain | delta pts | wk 95% CI | wk P+ | se P+ |
|---|---|---|---|---|---|---|
| (a) incumbent chain | 54.1583% | 0 | reference | — | — | — |
| (b) + movement | 55.6886% | 293 | +1.5303 | [-0.79, +3.86] | 0.8986 | 0.9268 |
| (c) + movement + nflcom | 55.6221% | 322 | +1.4637 | [-0.94, +3.89] | 0.8745 | 0.8890 |
| **(d) FULL max-EV** | 54.6241% | 371 | **+0.4657** | [-1.95, +2.92] | 0.6379 | 0.6258 |
| (e1) LOO no movement | 55.2229% | 156 | +1.0645 | [-0.26, +2.39] | 0.9385 | 0.8669 |
| (e2) LOO no nflcom | 54.4245% | 342 | +0.2661 | [-2.08, +2.65] | 0.5776 | 0.6052 |
| (e3) LOO no protection (= c) | 55.6221% | 322 | +1.4637 | [-0.94, +3.89] | 0.8745 | 0.8890 |

### FULL stack, both framings

| framing | n scored | acc | delta pts | wk P+ | se P+ |
|---|---|---|---|---|---|
| (d) slate-wide partial application (what ships) | 1503 | 54.6241% | +0.4657 [-1.95, +2.92] | 0.6379 | 0.6258 |
| (d-restricted) all-four-sources games only | 761 | 56.3732% | +1.3141 [-1.87, +4.52] | 0.7788 | 0.7420 |

### Leave-one-out marginals within the full stack

In-stack marginal of each edge = arm (d)'s delta minus the corresponding
leave-one-out arm's delta (measured): movement **-0.60** points (removing
movement RAISES the composed delta from +0.47 to +1.06), nflcom **+0.20**
points, protection **-1.00** points (removing it recovers arm (c)'s +1.46).
The three edges interact heavily — they flip overlapping games and fight each
other: chain+movement alone (+1.53) and chain+nflcom+protection alone (+1.06)
BOTH outscore the full four-edge stack (+0.47). Sub-additivity this size is a
composition finding, not a refutation of any component: each edge's own
interval still crosses zero where applicable and none meets a terminal
closing ground; per AGENTS.md these stay recorded, not rejected.

### Per-season stability of arm (d) (delta points vs chain)

| season | scored | movement flips | nflcom flips | protection flips | delta pts | all-four n |
|---|---|---|---|---|---|---|
| 2020 | 220 | 40 | 0 | 10 | -3.18 | 0 |
| 2021 | 236 | 55 | 0 | 8 | +0.85 | 0 |
| 2022 | 248 | 54 | 14 | 17 | -2.42 | 246 |
| 2023 | 266 | 47 | 16 | 17 | +1.50 | 258 |
| 2024 | 266 | 52 | 37 | 19 | +5.26 | 268 |
| 2025 | 267 | 45 | 0 | 26 | +0.00 | 0 |

Two of six seasons negative; 2024 (+5.26) carries the stack; 2025 nets exactly
zero (verified directly from per_game.parquet: 59 flipped picks won 29 and
lost 29). Edge applications (measured): movement fired on 664 eligible games /
293 net flips; nflcom on 799 covered games / 67 flips (zero coverage gaps
within 2022-2024); protection on 105 picked-flagged games / 97 flips.

### Upper-bound caveat (mandatory)

Three of the four stacked components were selected and/or red-teamed on
seasons this measurement overlaps: the movement rule and the arrest policy
were registered on 2020-2025 windows, and the NFL.com out>=2 fade was selected
AND red-teamed on exactly the 2022-2024 population scored here; the
protection cell was mined on 2009-2025. Every number above is therefore
attribution on already-looked-at data — an UPPER BOUND on what these rules
will do going forward, not a fresh confirmation — and the four component
effects must never be pooled as independent. No rotation-registry window was
spent.

### Shipped-composition statement

Decision first, at the opener grade the pool actually settles on: the
measured-best composition is **chain + movement (arm b, +1.53 pts, wk P+
0.90, season P+ 0.93)**. Adding further edges LOWERED the measured opener
accuracy in this attribution run (c +1.46, full stack +0.47), so the full
MAX-EV four-edge stack is NOT the recommended shipped card despite its
positive sign — its own evidence (+0.47 pts, P+ 0.64, two of six seasons
negative, sub-additive interactions) does not support playing it over arm (b),
and this upper-bound attribution cannot rescue it. The NFL.com fade remains a
defensible rider only inside its 2022-2024 data range (in-stack marginal
+0.20, P+ modest); the protection tilt stays challenger-tracked with its
positive solo reading (+0.336 pts solo, P+ 0.9785, docs/pbp08_matchup_screen.md)
intact — its negative in-stack marginal is an interaction result, not a
refuted mechanism, and no terminal closing ground was met anywhere in this
study.
