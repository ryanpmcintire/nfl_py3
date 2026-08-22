# Movement-rule-on-composed-chain attribution study

Owner task, 2026-08-22. Attribution only: no rotation-registry window was
spent, no model refit, no new data captured. Companion scripts:
`scripts/movement_composition_eval.py`. Run artifact:
`artifacts/movement_composition_eval/20260822T144746Z/` (per-game parquet,
`arms_summary.csv`, `season_summary.csv`, `metadata.json`), stamped into
`registry/experiments/movement-composition-eval/`.

## Question

Production today composes **raw model -> coach fade -> player-arrest policy**
(the live card order in `src/nfl_ats/card_view.py`; published historical
opener grade 53.76%). The observed-movement rule
(`observed_movement_threshold_1_0`: if the current line moved >= 1.0 pt from
the frozen Tuesday line, follow the market side; else keep the pick;
docs/observed_movement_channel.md, recorded solo +1.863 accuracy points,
P+ 0.935) is challenger-tracked only. This study measures what that rule is
worth **on top of the composed chain**, on the same paired opener archive used
everywhere (`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`,
1,537 REG games 2020-2025, production probability rule, graded at the frozen
Tuesday line).

## Method

- Lines reloaded from the market archive through the exact path
  `opener_pick_evaluation` uses (`build_pairing_table` with `tue_open` +
  `CLOSE_LABEL_PRIORITY`, then `close_reference_table`) — the same path
  `scripts/observed_movement_channel.py` consumes. Verified: every reloaded
  `tue_open`/`close` matches the archive's own columns exactly, and the
  recomputed `open_move = close - tue_open` matches the archive column.
- Coach-fade flip set rebuilt via the frozen `apply_coach_fade_overlay` on a
  predictions frame rebuilt from the archive + latest schedule snapshot (107
  flips, matching `artifacts/overlay_subset_composition`).
- Arrest flags rebuilt via `player_arrests_policy_eval`'s frozen machinery
  against the predeclared point-in-time incidents snapshot (25 baseline flips,
  matching).
- Incumbent chain arm composes **sequentially** (raw pick -> coach fade
  complement -> arrest back-side flip evaluated after the coach pass),
  matching `card_view`'s live order and
  `overlay_subset_composition`'s `coach_then_arrest_sequential`.
- Bootstrap: `nfl_ats.clv.week_blocked_bootstrap`, 20,000 samples, seed
  20260822, week-blocked primary / season-blocked secondary, paired deltas in
  accuracy points, full slate (pushes NaN-masked identically in every arm).

## Reproduction check (arm a)

Two published references exist and BOTH reproduce exactly:

| reference | value | measured | match |
|---|---|---|---|
| sequential coach->arrest chain (`overlay_subset_composition.production_chain_reference.coach_then_arrest_sequential`) | 0.5415835 | 0.5415835 | yes |
| published 53.76% headline = arrest policy applied directly to the frozen 53.36% baseline (`docs/player_arrests_policy_eval.md`) | 0.5375915 | 0.5375915 | yes |
| raw baseline 53.36% (`correct_at_open_probability_rule`) | 0.5335995 | 0.5335995 | yes |

Flip counts also match: coach fade 107, arrest on baseline 25, arrest flips
after coach pass 25, n scored 1,503 (34 pushes).

## Arm table (full slate, graded at the frozen Tuesday line)

Paired deltas are vs arm (a) on identical games; P+ is the blocked-bootstrap
probability of the delta being positive.

| arm | n | accuracy | delta vs (a), pts | week CI (pts) | wk P+ | season CI (pts) | se P+ | picks changed |
|---|---|---|---|---|---|---|---|---|
| (a) incumbent chain (raw -> coach -> arrest) | 1503 | 54.1583% | — | [51.46, 56.80] (acc) | — | — | — | 126 vs raw |
| (b) chain + movement >= 1.0 | 1503 | 55.6886% | +1.5303 | [-0.8065, +3.8765] | 0.8942 | [-0.4902, +3.3333] | 0.9297 | 293 vs (a) |
| (c) movement >= 1.0 solo on raw model | 1503 | 55.2229% | +1.0645 | [-1.4855, +3.6424] | 0.7774 | — | 0.7433 | 299 vs raw |

Context reads, all measured this session:

- Arm (c)'s accuracy minus the raw baseline is +1.8630 points (55.2229 -
  53.3599), matching the registry entry `observed_movement_threshold_1_0`
  (+1.862941) exactly in point estimate; its P+ differs from the recorded
  0.935 only because this run uses seed 20260822 rather than the recorded
  20260819.
- Composition beats both parts alone on point estimate: (b) 55.69% >
  (c) 55.22% > (a) 54.16%. Of the 664 flip-eligible games (|move| >= 1.0),
  the overlay changed 293 of the chain's picks.

### Per-season stability (scored games)

| season | n | (a) | (b) | (c) | eligible >= 1.0 |
|---|---|---|---|---|---|
| 2020 | 220 | 53.18% | 50.00% | 49.55% | 117 |
| 2021 | 236 | 56.36% | 57.20% | 56.36% | 118 |
| 2022 | 248 | 53.63% | 55.24% | 54.84% | 109 |
| 2023 | 266 | 59.02% | 59.40% | 56.02% | 116 |
| 2024 | 266 | 52.63% | 56.77% | 57.52% | 101 |
| 2025 | 267 | 50.19% | 54.68% | 56.18% | 103 |

The movement overlay helps the chain in five of six seasons and hurts it
visibly in 2020 (-3.18 pts); the season-blocked interval is wide accordingly.

## Effective-n disclosure

Movement data exists only where the market archive resolved BOTH a Tuesday
opener and a resolvable close; the opener archive itself is conditioned on
that pair, so every game present carries movement data and seasons without
close coverage never entered the population at all (selection upstream of
this archive, not missingness within an arm). Effective n is therefore
identical across arms: 1,503 scored games of 1,537 (34 pushes), distributed
per season as the table above.

## Disclosure: upper bound

This is attribution on already-looked-at data. The movement rule, the coach
fade, and the arrest policy were each selected or registered using windows
this 2020-2025 archive covers, so these composed figures are an upper bound —
continuous evidence, not a fresh confirmation, and not a promotion
measurement.

## Record line (NOT executed; no registry JSON written)

```text
nfl-ats weak-signals record \
  --name movement_rule_composed_chain \
  --description "Observed-movement rule (|close-tue_open|>=1.0, follow market side) applied ON TOP of the composed production chain raw model -> coach fade -> player-arrests policy; paired full-slate accuracy delta vs the un-composed incumbent chain on the same games, graded at the frozen Tuesday line." \
  --source artifacts/movement_composition_eval/20260822T144746Z/metadata.json; docs/movement_composition_eval.md \
  --effect-units accuracy_points \
  --classification unresolved_below_power \
  --league nfl --season-start 2020 --season-end 2025 \
  --effect 1.530273 \
  --interval-low -0.806465 --interval-high 3.876478 \
  --probability-positive 0.894200 \
  --sample-games 1503 \
  --classification-evidence "Week-blocked paired delta vs incumbent chain = +1.5303 accuracy points [-0.8065, +3.8765], P+ 0.8942; season-blocked interval [-0.4902, +3.3333] P+ 0.9297. Attribution on already-looked-at data (upper bound), no window spent, no terminal ground met." \
  --notes "Attribution-only composition study (owner task 2026-08-22); seed 20260822, 20000 samples, week primary / season secondary; close-grade line availability limits which seasons have movement data at all (effective n per arm disclosed in artifact); movement-rule solo reference arm reproduces observed_movement_threshold_1_0 design at this seed; NOT written to registry JSON by the script."
```

Classification note (binding AGENTS.md taxonomy): the week-blocked interval
crosses zero, which is NOT grounds for rejection; `wrong_sign_resolved` would
require the whole interval below zero, which does not hold (+1.53 point
estimate, probability_positive 0.894). No positive control was run and no
split-half claim is made, so the classification is `unresolved_below_power`.

## What this is not

- Not a window-spending evaluation; nothing new was looked at that the
  component experiments had not already looked at.
- Not a claim about prospective value; the archive is 2020-2025 history and
  every input was selected on overlapping windows.
- Not a change to the active card, challenger tracking, or either registry
  JSON.
