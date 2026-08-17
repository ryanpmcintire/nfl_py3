# MOD-07 weak-signal stack — predeclaration and the opener-graded look

Executing `docs/opus_execution_specs.md` § SPEC-4. Run 2026-08-17.

**Result in one line:** the stack beat the frozen model by **+1.97 accuracy
points** against the Tuesday opener on 456 paired games, with
`probability_positive = 0.8745` — **short of the predeclared 0.90
confirmation threshold**, so the registry verdict is `unresolved` and
nothing is promoted.

## Hypothesis (predeclared before any number below existed)

Stacking the surviving weak signals — learned availability,
value-weighted injuries, and three peer-reviewed opener-bias features —
onto the frozen player profile improves forced-pick accuracy against the
Tuesday opener the pool grades on.

## The two arms

| | baseline | candidate |
|---|---|---|
| profile | `player` (frozen active config) | `weak_stack` |
| table | `game_features_player.parquet` | `game_features_weak_stack.parquet` |
| injury semantics | hand-authored fixed priors | **learned availability rates** |
| extra families | — | `player_values` + `bias` |
| feature columns | 79 | 90 |
| regressor / alpha / calibration | ridge / 10 / none | ridge / 10 / none |

`weak_stack` = `FEATURE_SETS["full_player_value"] + FEATURE_FAMILIES["bias"]`,
which is the player composite plus the nine bias columns and no duplicates.
The candidate table is built through `build-learned-availability-features`,
so its injury columns carry learned semantics under the same *names* the
fixed-prior table uses — the two tables must never be mixed in one run, and
`_MARGIN_PROFILE_FEATURE_SETS` carries that warning in code.

The three bias families (`features.add_bias_features`, computed leak-safely
from schedules alone):

- `bias_playoff_holdover_*` — week 1, team appeared in any postseason game
  last season (literature: week-1 playoff holdovers covered 35.6%);
- `bias_prior_week_ats_*` — the team's single previous completed game's
  `ats_margin` this season, strictly earlier games only;
- `bias_week2_anchor_*` — the same value masked to week 2 (the anchoring
  result is specifically week 2).

Contract-year / friction events were excluded from v1 as specified: no data
source in repo, literature ≈ null.

## Window and grading

`rotation declare --grade opener --acknowledge-mined` then
`rotation assign` ⇒ **[2020, 2021]**, the earliest eligible opener block,
computed by the registry with no override.

Both arms were graded by the existing `clv.opener_pick_evaluation` — one
weekly-refit market-residual model per arm, trained on completed games
strictly before each week's first kickoff, with `spread_line` overridden to
the archived Tuesday opener consensus and each forced pick settled against
that opener. Scoping came from `rotation.confirmation_split`, so the fit
saw exactly the forward-chained training the registry assigned. The
inherited approximation of that machinery applies and is declared, not
fixed: only `spread_line` is swapped to the opener; every other feature
(including `total_line`) is close-era.

Runner: `scripts/mod07_weak_stack.py`. Artifact:
`artifacts/mod07_weak_stack/opener_2020_2021.json` plus the per-game paired
parquet beside it.

## Result

| metric | value |
|---|---|
| paired games / weeks | 456 / 35 |
| baseline accuracy at opener | 51.32% |
| candidate accuracy at opener | **53.29%** |
| paired delta | **+1.97 points** |
| week-blocked 95% interval | [−1.10, +5.00] |
| `probability_positive` | **0.8745** |
| picks where the arms disagreed | 49 |
| — baseline correct on those | 20 / 49 (40.8%) |
| — candidate correct on those | 29 / 49 (59.2%) |

**Verdict: `unresolved`** (predeclared: ≥ 0.90 → `confirmed`; ≤ 0.10 →
`closed_negative`; otherwise `unresolved`). Window spent.

## Reading it honestly

**This is a near-miss, and a near-miss is not a pass.** 0.8745 was measured
against a threshold fixed before the run precisely so that a number landing
just underneath could not be talked upward. It is not promoted, not
activated, and not re-scored.

**The disagreement split is where the whole effect lives.** The arms agreed
on 407 of 456 picks; every point of the delta comes from the 49 they split,
where the candidate went 29–20. Nine correct picks *net* is the entire
result. That is a small enough number to be luck, and the interval
([−1.10, +5.00]) says so.

**The window carries the standing discount.** [2020, 2021] sits inside the
mined 2018–2025 era, acknowledged at declaration per rule 6. The ~130–150
prior candidate streams scored against those outcomes mean a +1.97-point
result here is roughly what selection on noise plus a possibly small real
effect would produce. This does not make the result worthless — the
candidate was predeclared and scored once — but it is not clean evidence.

**Both arms are below the headline.** The baseline scores 51.32% here
against the 52.50% headline over 1,537 games in 2020–2025; this is a
456-game subset of two seasons, and that spread is ordinary sampling.

## What happens next

Nothing automatic. The candidate is not promoted. Three options exist and
the choice is the owner's:

1. **Prospective 2026 scoring as a frozen challenger** — the cleanest
   evidence available, needs no registry window at all (`docs/rotation_registry.md`,
   "what this deliberately does not do"), and costs nothing but patience.
   This is the recommended path.
2. **A second opener window** ([2022, 2023] is next-eligible). Available
   under rule 4, but spending a second scarce opener window on the *same*
   candidate after seeing 0.8745 is iterating-until-it-wins by another
   name. Not recommended without a stated reason that does not reduce to
   "it nearly cleared".
3. **Drop it.**

What is *not* admissible: retuning the stack's contents, the regressor, or
the threshold and re-scoring [2020, 2021]. That window is spent for this
family, permanently.
