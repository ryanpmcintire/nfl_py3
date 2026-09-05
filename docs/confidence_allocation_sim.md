# LEAD-51: confidence-point allocation, simulated

## Closing-grounds taxonomy (binding, verbatim, AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator.

## The decision, stated first

**Do not move the 2026 confidence allocation off flat/random.** Two
candidates were tested against it; neither clears the bar, and one is
measurably worse:

- **`calibrated_edge` (isotonic-calibrated |p-0.5|) is RESOLVED WORSE than
  flat on P(finish first)** — the metric the pool actually pays, per
  `nfl_ats.pool`'s own docstring ("a different objective from maximising
  expected correct picks, and it is the one the pool actually pays") — at
  field sizes 100 and 500, in the overall sample AND in both halves of the
  archive independently: the whole 95% week-blocked interval sits below
  zero (`probability_positive` 0.00 in four of six overall/half cells, 0.01
  and 0.10 in the other two). This is a refuted-mechanism-grade result under
  the taxonomy above (a RESOLVED wrong sign), for this specific
  implementation (isotonic, walk-forward, `min_train=200`) on this metric.
  It is **not** recorded via `nfl-ats weak-signals record` — see
  "Why nothing is recorded" below — but the result is real and the
  recommendation follows from it regardless: do not calibrate the
  confidence order this way.
- **`edge_proportional` (raw |p-0.5|) shows no reliable edge over flat** —
  every interval crosses zero, `probability_positive` ranges 0.17-0.74
  across scopes/field-sizes with no consistent lean. This reproduces, in
  the pool's own payout metric, the finding already on record in
  `docs/pool_edge_plan.md`: *"the model's sign carries the signal; its
  residual magnitude does not rank pick quality."* Raw probability edge
  doesn't rank pick quality any better here.
- **`market_only` (|opener spread|, a known public heuristic) is
  unresolved but leans mildly positive on P(first)** in 5 of 6
  overall/half cells (`probability_positive` up to 0.91 in one cell),
  never resolved. Worth a further, dedicated look some day; not a basis
  for a 2026 change on its own.
- The **oracle positive control dominates every strategy in every one of
  the 107 weeks** (`oracle_dominates_every_week=True`, checked
  programmatically) and every bootstrapped cell resolves fully positive
  (`probability_positive` 1.0, whole interval positive) — the simulator is
  not blind; a genuinely informative confidence order would show up.

Net: FLAT is not beaten by anything tested. The pool's confidence slots
should stay flat/arbitrary for 2026 until a candidate actually resolves
positive (or, at minimum, shows a positive lean nobody has yet measured
against `market_only`'s open thread).

## What could be wrong with this

- `calibrated_edge`'s isotonic step function collapses each week's ~14
  games to **~4.4 distinct calibrated values on average** (measured:
  `walk_forward_calibration` output, 91 of 107 weeks with
  `calibration_applied=True`, mean 4.44 distinct values per week, min 1,
  max 7) given only ~200-1500 rows of walk-forward training history. That
  collapses most of the week into a few tie groups broken by `game_id`
  ascending (arbitrary, unrelated to correctness) — the measured
  point-biserial correlation between `|calibrated_p-0.5|` and correctness
  is **-0.022** (vs. **-0.003** for raw edge), both tiny, both
  indistinguishable from noise on their own, but the week-blocked bootstrap
  on the confidence-WEIGHTED scoring rule amplifies that tiny gap into a
  resolved cost on P(first) at larger field sizes. This is a property of
  the specific parameterization tested (isotonic, `min_train=200`, the
  project's existing `DEFAULT_MIN_CALIBRATION_GAMES`), not necessarily of
  "calibration" in general — a different minimum-training threshold or a
  Platt (logistic) fit instead of isotonic was not tested and could behave
  differently. Not tested here for scope reasons; flagged for anyone who
  wants to reopen this specific implementation choice.
- This is a **paper simulation**, not a live pool result: real Splash
  entrants are not i.i.d. public-side-with-probability-p bettors, ties are
  not literally broken alphabetically by a real opponent, and no field-size
  or prize-structure observable has actually been captured yet (LEAD-52,
  queued for the 9/08 lock week). The field model's `public_lean` here is
  informed by a measured input (`p_favorite`, below) rather than an assumed
  constant, but it is still an assumption, disclosed as one.

## Frozen design

Reused rather than duplicated: `nfl_ats.clv.week_blocked_bootstrap` for
every cross-week interval; the tie-credit / "exact, not an approximation"
philosophy of `nfl_ats.pool.simulate_pool_finish`, extended to a
confidence-weighted scoring rule that a flat 1-point-per-correct-pick
format doesn't need; `nfl_ats.pool_workbench.PoolRules.confidence()`
already names this pool type ("confidence pools require
`confidence_assignment='unique_1_to_game_count'`") — this experiment
supplies the missing scoring simulator for that pool type. Nothing in
`nfl_ats.pool.py` was duplicated; its scoring rule (flat +1 per correct pick,
plus a single Best Pick bonus) is a genuinely different mechanism from
Splash-style confidence points, so a new (small) scorer was written rather
than bent to fit.

**Population.** Every week in the frozen opener-evaluation archive
(`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`,
`nfl_ats.clv.opener_pick_evaluation`'s output) with at least 8 *graded*
(non-push) games. Measured: the archive holds 107 weeks across the 2020-2025
seasons (it does not reach back to 2009-2017, so the required per-era split
falls back to its own stated alternative — the two halves of the archive,
54 and 53 weeks by chronological (season, week) order), 1,537 total games,
1,503 graded after dropping 34 pushes (`margin_vs_open == 0`); every one of
the 107 weeks clears the 8-game floor (measured min 10, max 16 graded games
per week) so none are dropped.

**The forced pick is shared across every strategy.** All five strategies
below pick the SAME side for every game — the production probability-rule
side, `pick_home_at_open_probability_rule` (`home_cover_probability_at_open
>= 0.5`), already the project's forced-pick rule. Only the CONFIDENCE
ORDER — which of the week's games gets which point value — differs. This
matches the "Best Pick lever" framing already on record in
`docs/pool_edge_plan.md`: the side is decided elsewhere; this experiment is
purely about the format lever.

**Scoring.** Splash-style: every game in a week gets a distinct point value
from 1 to n (n = that week's graded-game count), highest value to the most
confident pick; a correct pick earns its assigned point value, an incorrect
one earns zero. Ties in a ranking criterion are broken by `game_id`
ascending (the same deterministic tie-break `nfl_ats.pool.build_ats_pool_card`
already uses).

**Five entrant strategies:**

1. **`flat`** — control. Random distinct point values each week (an
   i.i.d.-uniform random permutation).
2. **`edge_proportional`** — rank by `|home_cover_probability_at_open - 0.5|`.
3. **`calibrated_edge`** — same, using
   `|calibrated_probability_at_open - 0.5|` after a walk-forward isotonic
   calibration (`sklearn.isotonic.IsotonicRegression`) fit each week ONLY on
   graded games from strictly earlier (season, week) pairs (calibration
   honesty: no in-sample or same-week information reaches the calibrator).
   Weeks without at least `DEFAULT_MIN_CALIBRATION_GAMES` (200, the project's
   already-derived floor — `nfl_ats.constants`) prior graded games, or whose
   prior history is single-class, keep the raw probability
   (`calibration_applied=False`, 16 of 107 weeks; measured) — disclosed, not
   silently smoothed over.
4. **`market_only`** — rank by `|tue_open_home_spread|`, a known public
   confidence heuristic ("bigger favorites are safer picks").
5. **`oracle`** — positive control. Rank by realized correctness itself
   (`correct_at_open_probability_rule`), which is the score-MAXIMIZING
   permutation for that week's fixed pick pattern by construction (it
   assigns the top-k point values to the k actually-correct picks). Must
   dominate every other strategy, every week — checked programmatically,
   not asserted.

**Field model.** N i.i.d. opponents (N in {20, 100, 500}), each
independently: (a) picks the game's "public" side — the spread favorite —
with probability `p_favorite`, measured archive-wide (not walk-forward; it
is a field-model assumption exactly like `nfl_ats.pool.FieldModel`'s
`public_lean` default, and never touches which side WE pick) as the share
of graded games where our own forced pick already lands on the favorite:
**`p_favorite = 0.4688`** (measured, `measure_favorite_share` on the full
1,503-game population). Pick'em games (opener spread exactly 0, no
favorite) give every opponent a 50/50 coin flip regardless of `p_favorite`.
(b) ranks its own picks with an independent, uniformly random permutation.

**Per-week computation is EXACT wherever the fixed pick pattern allows it,**
not literal (samples x entrants x games) Monte Carlo: the four non-FLAT
strategies' weekly score is a single deterministic number (points are a
fixed function of the ranking criterion and the already-known outcome).
Only two distributions are Monte-Carlo-estimated per week — the field's
one-entrant score distribution and FLAT's own-score distribution — because
i.i.d. entrants make the field's maximum-of-N a pure function of that ONE
entrant distribution and N (closed-form tie-credit sum, see
`scripts/confidence_allocation_sim.py::probability_first`'s docstring for
the derivation); this is the same "exact, not an approximation" choice
`nfl_ats.pool` already makes for its own field draws, extended one level to
a confidence-weighted score. `MC_DRAWS = 20,000` per week; one fixed seed
(`SIM_SEED = 20260905`) consumed sequentially across weeks in ascending
(season, week) order, so the entire run is bit-reproducible from that one
number (pinned by a determinism test).

**Metrics, per strategy:** expected points per week (exact for the four
non-FLAT strategies; analytic `k*(n+1)/2` for FLAT, k = that week's number
of correct picks) and P(finish first) against the simulated field at each
of the three field sizes (fractional tie credit, matching
`nfl_ats.pool.simulate_pool_finish`'s `probability_first` convention).

**Cross-week comparison.** Paired difference vs. FLAT, per week, resampled
with `nfl_ats.clv.week_blocked_bootstrap` (1,000 draws, fixed seed
`BOOTSTRAP_SEED = 20260905`, 95% interval, `probability_positive` reported
directly — never "contains zero"). Reported for the overall 107-week sample
and for the two archive halves independently (LEAD-51's per-era requirement,
using the archive's own two halves since it does not span 2009-2017).

## Measured results

Command: `.\.tools\uv.exe run --no-sync python scripts/confidence_allocation_sim.py`
Artifact: `artifacts/confidence_allocation_sim/20260905T035216Z/`
(`per_week.csv` — 1,605 rows, one per week x strategy x field size;
`strategy_comparison.csv` — 108 rows; `metadata.json`). Registry row:
`registry/experiments/confidence-allocation-sim/20260905T035216Z.json`
(provenance only — see "Why nothing is recorded"). Re-running the exact
command above reproduces every number below bit-for-bit (fixed seeds
throughout; pinned by `tests/test_confidence_allocation_sim.py`).

`p_favorite = 0.4688` (measured, archive-wide, 1,503 graded games).

### Overall (107 weeks)

| strategy | metric | field | estimate | 95% CI | P+ |
|---|---|---|---|---|---|
| edge_proportional | expected_points | — | -0.164 | [-1.566, +1.383] | 0.430 |
| edge_proportional | P(first) | 20 | -0.0102 | [-0.0337, +0.0132] | 0.210 |
| edge_proportional | P(first) | 100 | -0.0032 | [-0.0175, +0.0152] | 0.352 |
| edge_proportional | P(first) | 500 | +0.0032 | [-0.0055, +0.0207] | 0.635 |
| calibrated_edge | expected_points | — | -1.201 | [-2.650, +0.346] | 0.080 |
| calibrated_edge | P(first) | 20 | -0.0214 | [-0.0407, -0.0026] | 0.010 |
| calibrated_edge | P(first) | 100 | -0.0168 | [-0.0283, -0.0078] | 0.000 |
| calibrated_edge | P(first) | 500 | -0.0040 | [-0.0080, -0.0010] | 0.000 |
| market_only | expected_points | — | -1.276 | [-2.808, +0.309] | 0.048 |
| market_only | P(first) | 20 | +0.0028 | [-0.0221, +0.0289] | 0.620 |
| market_only | P(first) | 100 | +0.0059 | [-0.0098, +0.0240] | 0.745 |
| market_only | P(first) | 500 | +0.0021 | [-0.0038, +0.0107] | 0.623 |
| oracle (positive control) | expected_points | — | +22.799 | [+21.724, +23.869] | 1.000 |
| oracle | P(first) | 20 | +0.3670 | [+0.3095, +0.4226] | 1.000 |
| oracle | P(first) | 100 | +0.2179 | [+0.1641, +0.2754] | 1.000 |
| oracle | P(first) | 500 | +0.0876 | [+0.0528, +0.1276] | 1.000 |

All deltas are `strategy - flat`, positive favours the candidate.

### Per-era magnitudes (two halves of the archive; no 2009-2017 coverage exists)

| strategy | metric | field | first half (54 wks) est / P+ | second half (53 wks) est / P+ |
|---|---|---|---|---|
| edge_proportional | expected_points | — | -0.019 / 0.478 | -0.311 / 0.399 |
| edge_proportional | P(first) | 500 | -0.0055 / 0.000 (resolved neg, isolated cell) | +0.0121 / 0.631 |
| calibrated_edge | expected_points | — | -0.500 / 0.300 | -1.915 / 0.050 |
| calibrated_edge | P(first) | 20 | -0.0257 / 0.023 (resolved neg) | -0.0170 / 0.100 |
| calibrated_edge | P(first) | 100 | -0.0212 / 0.000 (resolved neg) | -0.0124 / 0.000 (resolved neg) |
| calibrated_edge | P(first) | 500 | -0.0057 / 0.000 (resolved neg) | -0.0022 / 0.000 (resolved neg) |
| market_only | P(first) | 20 | -0.0162 / 0.194 | +0.0222 / 0.908 |
| market_only | P(first) | 100 | +0.0037 / 0.567 | +0.0080 / 0.821 |
| oracle | every metric | every field | resolved fully positive | resolved fully positive |

`calibrated_edge`'s negative P(first) result at field sizes 100 and 500 is
the one that replicates identically-signed and resolved in BOTH halves of
the archive independently — the most robust finding in this experiment.
The single resolved-negative `edge_proportional` cell (field=500, first
half) does NOT replicate in the second half (which leans positive,
unresolved) and reads as noise, not a mechanism — reported per AGENTS.md's
"report the number, not a one-word verdict" rule rather than suppressed.
Full 108-row table: `strategy_comparison.csv` in the artifact directory.

## Why nothing is recorded to the weak-signal registry

AGENTS.md's pooling discipline: *"pooled inputs must be commensurable —
same units, same scale, same population."* This experiment's metrics
(expected confidence points per week; P(finish first) against a simulated
field) are not accuracy, not Brier, not log-loss, not a correlation
coefficient, and not on the same scale as forced-pick accuracy —
they are not any entry in `nfl_ats.weak_signals.EFFECT_UNITS`
(`ats_points`, `accuracy_points`, `brier`, `log_loss`, `mae`, `correlation`,
`mae_improvement`, `brier_improvement`, `log_loss_improvement`; read,
`src/nfl_ats/weak_signals.py`). Forcing either metric into
`accuracy_points` "as a numeric container" is exactly the mislabeling
AGENTS.md already flags as a past mistake ("an NFL entry had been forced
into `accuracy_points` as a numeric container only for lack of a real
unit"). So: **no `nfl-ats weak-signals record` call for this experiment.**
The artifact (`registry/experiments/confidence-allocation-sim/`, written
automatically by `write_experiment_artifact` as a provenance log, not a
weak-signal claim) and this document are the record.

## Files

- `scripts/confidence_allocation_sim.py` — the simulator (loads the archive,
  measures `p_favorite`, runs the walk-forward calibration, runs the
  per-week simulation, runs the cross-week bootstrap comparison, writes the
  artifact).
- `tests/test_confidence_allocation_sim.py` — rank validity (distinct
  points 1..n every week), oracle dominance on a synthetic week,
  calibration-uses-only-earlier-weeks (leakage regression, per AGENTS.md's
  "add a leakage regression test for every new feature family"), and
  determinism under a fixed seed.
- `artifacts/confidence_allocation_sim/20260905T035216Z/` — the measured
  run (`per_week.csv`, `strategy_comparison.csv`, `metadata.json`).
