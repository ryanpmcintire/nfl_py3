# `offseason_retention` — re-derivation, consumer map, and CFB screen

Written 2026-08-17, following up on `af5238f` ("Close four underived
constants by measuring them"), which found `DEFAULT_OFFSEASON_RETENTION =
0.67` roughly twice too large (three routes: 0.337 / 0.379 / 0.400, none
overlapping 0.67) but left the constant unchanged because moving it touches
every prediction and the standing rule requires a free-data screen first
(`docs/rotation_registry.md` rule 8; the user's rule that an underived
constant gating an irreversible decision is a defect).

> **STATUS: re-derived, screened, and left UNCHANGED at 0.67.** The three
> NFL-side routes reproduce (two almost exactly, one qualitatively). But the
> CFB screen — the free, large-sample test this was gated on — says the
> opposite of what the routes predict for the metric this project is built
> around: forced-pick accuracy prefers retention **at or above** 0.67, not
> below it, with high confidence across 12,500 games *(corrected 2026-08-18;
> this line previously read 12,206, the wrong count for the full 2006-2025
> corpus — the parquet has 12,500 rows, measured directly)*. See **§ Recommendation**
> for why that outranks the routes here, and **§ Consumer map** for two
> related defects found along the way (an Elo wiring gap, and a
> second, independently underived retention constant in the player/QB
> family that this correction would not have reached anyway).

## 1. The three routes, re-derived

No script from the original measurement survived (it was run ad hoc), so
this is a fresh implementation of the same methodology against real local
data, not a replay of the same code — reproduced independently in
`scripts/offseason_retention_routes.py`. All three fit the same model:
`current = league_mean + retention**gap * (current - league_mean)`.

Data: the latest local NFL raw snapshot, `nfl_ats.features.build_team_game_metrics`
→ 8,862 team-game rows, 2009-2025, 32 teams.

### Route 3 — plain point-differential regression (reproduces almost exactly)

Team-season mean `point_diff`, centered within season, regressed against the
same for the following season, across 512 team-season transitions (16 season
boundaries × 32 teams), season-blocked 95% CI (2,000 resamples):

| horizon | slope (this run) | 95% CI (this run) | reported 2026-08-17 |
|---|---|---|---|
| full next season | **0.3998** | [0.347, 0.457] | 0.400, [0.347, 0.460] |
| first 4 games | **0.4748** | [0.391, 0.569] | 0.475, [0.391, 0.573] |

Both intervals match to three significant figures. **Confirmed.**

### Route 2 — shipped feature table's own week-1-vs-late-season slope (reproduces almost exactly)

Using the already-built `data/processed/game_features.parquet` (built at the
current 0.67), regressing `result ~ diff_point_diff` separately for week-1
games (freshly offseason-regressed state) and weeks 9-18 (steady-state,
in-season EWM dominates):

| quantity | this run | reported 2026-08-17 |
|---|---|---|
| week-1 slope (n=255) | 0.3328 | 0.333 |
| weeks 9-18 slope (n=2,393) | 0.5881 | 0.588 |
| ratio | 0.5659 | 0.566 |
| implied retention (0.67 × ratio) | **0.3791** | 0.379 |

**Confirmed**, to three decimal places.

### Route 1 — per-metric/horizon regression slope (reproduces qualitatively, not cell-for-cell)

Same transition design as route 3, extended to all 15 `STATE_METRICS` and
horizons 1-4 games into the next season (60 cells; the original's "24
metric × horizon cells" grid could not be recovered exactly — no record of
which metric subset or which horizons it used survives).

At horizon 4 games — the one the original note singled out as "the horizon
the constant actually governs" — **all 15 metrics** have a slope well below
0.67, median **0.370** (reported: 0.337), range **[0.072, 0.498]** (reported:
[0.195, 0.382] for the cited extremes `off_turnover_rate`/`off_sack_rate`,
which land at 0.245/0.464 here). Every one of the 15 horizon-4 cells' 95%
upper bound stays under 0.67 (max upper bound 0.607).

At horizons 1-3, results are noisier and the "all cells below 0.67" claim
does **not** hold in this reproduction: 9 of the 60 cells (mostly horizon 1,
where "next season" means a single game) have a 95% upper bound reaching or
exceeding 0.67 — expected, since a single game's per-play rate stats (e.g.
`off_turnover_rate`) are dominated by one-game noise, not by how much of last
year survived. This reproduction therefore **confirms the qualitative
conclusion and the magnitude at the relevant horizon**, but not the specific
"all 24 cells" framing, which cannot be checked without the original grid.
Full cell table: `route1_cells.csv` (scratch output).

**Net on step 1:** two of three routes reproduce almost exactly; the third
reproduces in substance (same order of magnitude, same direction, all
metrics comfortably under 0.67 at the horizon that matters) with an honest,
explained gap in the exact multi-cell claim.

## 2. Consumer map

`offseason_retention` means "fraction of last season's state that survives
the offseason, after regressing the rest toward the league/prior mean" —
**but it is not one parameter in the codebase; it is at least three,
sharing a name and a similar formula but not all wired to the same
constant or measuring the same thing.**

| module | function | reads `DEFAULT_OFFSEASON_RETENTION`? | what it regresses | same meaning as the routes above? |
|---|---|---|---|---|
| `features.py` | `build_team_states` / `attach_team_states` | yes | team-level EWM state (offense/defense EPA, point_diff, ats_residual) toward the season league mean | **yes** — this is exactly what routes 1-3 measured |
| `features.py` | `add_elo_features` | **no — wiring gap** | Elo rating toward 1500 | same *intended* meaning, but see defect below |
| `cfb_features.py` | `build_cfb_team_states` / `attach_cfb_team_states` | yes | identical formula, CFB team stats — module docstring says "NFL parameters taken verbatim," a deliberate, disclosed choice not to re-tune for CFB | yes, by construction (mirrors features.py exactly) |
| `pbp.py` | `_build_pbp_states` / `enrich_with_pbp_features` | yes | PBP-derived team state (EPA/success/explosive rate, drive stats) | yes, same formula |
| `graph_ratings.py` | `GraphRatingConfig.offseason_retention` | yes | multiplicative decay of the PageRank/HITS performance and scoring **matrices** (`performance *= retention**gap`), not mean-reversion toward a league average | **related but structurally different** — no mean subtracted, a raw magnitude decay instead. The three routes above measured mean-reversion coefficients; applying the same number here is an analogy, not a direct derivation |
| `players.py` | `enrich_with_player_features(..., offseason_retention: float = 0.75, ...)` | **no — different constant entirely** | forwarded *only* to `quarterbacks.build_qb_states`; not used for injury or lineup-continuity features at all | **no** — this measures an individual QB's own year-over-year skill persistence (the same person, presumably more durable than 53-man roster turnover), not team-roster composition. A materially different, defensibly-larger quantity |
| `quarterbacks.py` | `build_qb_states` / `enrich_with_qb_features(..., offseason_retention: float = 0.75, ...)` | **no** | per-QB EPA/CPOE/sack/INT/explosive-pass state toward the league mean | same as above |

### Defect 1 — Elo silently ignored the override (RESOLVED, fixed in this same commit)

**Corrected 2026-08-18: this section originally described an unpatched bug
and offered a call-site patch "not applied." That was wrong — the patch WAS
applied, in the identical commit (`7455d62`) that added this document.**
`features.py:492` reads, and has read since that commit:

```python
games = add_elo_features(games, offseason_retention=offseason_retention)
```

The bug this section describes below no longer exists in shipped code; the
rest of this section is kept as a record of what the bug was and why it
mattered, not as a description of the current state.

Originally, `features.py:492`, inside `_build_features_pass`, read:

```python
games = add_elo_features(games)
```

`add_elo_features` has its own `offseason_retention: float =
DEFAULT_OFFSEASON_RETENTION` default, but the call site never forwarded the
pass's own `offseason_retention` parameter. Every other consumer in the same
function *did* forward it (`add_schedule_strength_features`,
`build_team_states`, `attach_team_states`, lines 493-509). Consequence: if
`DEFAULT_OFFSEASON_RETENTION` was edited directly in `constants.py`, Elo
correctly picked up the new value, because its default binds to the constant
at import time. But any **override** — the CLI's `--offseason-retention`
flag on `build-features`, or a script sweeping several values, as this
task's own screens do — silently left Elo's offseason regression fixed at
whatever the raw constant equalled, while every other channel moved.
`elo_diff` and `elo_home_win_prob` are part of `MODEL_FEATURE_COLUMNS`, so
this was a real, if minor (2 of ~40 columns), contamination of any sweep
that used `build_game_features`'s `offseason_retention` parameter directly
rather than editing the constant. Fixed by forwarding the parameter at the
call site, shown above.

### Defect 2 (pre-existing, out of scope) — the player/QB family has its own, separately underived 0.75

`players.py:961`, `quarterbacks.py:279`, `quarterbacks.py:328`, and four CLI
flags (`cli.py:3770, 3796, 3836, 3876` — `build-qb-features`,
`build-player-features`, and the participation/availability variants) all
hardcode the literal `0.75`, independent of `DEFAULT_OFFSEASON_RETENTION`.
No comment, docstring, or doc anywhere derives it; it predates this
investigation and this task's ownership boundary excludes `players.py` /
`quarterbacks.py`. **Correcting `DEFAULT_OFFSEASON_RETENTION` has zero effect
on the player/QB family** — worth its own measurement someday, but it is a
second, separate defect, not a special case of this one.

## 3. CFB screen (predeclared, free — rotation rule 8)

**Predeclared before any arm was scored** (this document): grid
`{0.00, 0.20, 0.337, 0.40, 0.50, 0.67, 0.75}` — spans the shipped default and
the three route estimates, plus a no-retention control and a symmetric point
above 0.67 — against the frozen XLG-03 harness
(`nfl_ats.cfb_benchmark`, same Ridge/alpha/training-floor/residual recipe as
production, only `offseason_retention` varies across feature builds), full
2006-2025 window (12,500 canonical games *(corrected 2026-08-18; was
12,206)*; 8,933 in the `clean_core` split,
2,354 `thin_2006_2011`, 493 `regime_2020`). Bootstrap: 2,000 samples,
week- and season-blocked, seed 20260817. Baseline for every paired comparison
is the shipped 0.67. Script: `scripts/offseason_retention_cfb_screen.py`.
Rotation registry: **untouched** — no family declared, no window assigned or
spent.

CFB has far more roster turnover than the NFL (more scholarship players,
shorter careers, transfer-portal attrition), so whatever is CFB-optimal is
**not** the NFL answer, and this screen does not claim otherwise. What it
*can* establish: does nudging retention from 0.67 toward the NFL-measured
~0.35-0.40 help, hurt, or do nothing on a large, independent walk-forward
benchmark using the mechanically identical formula. That makes it a
direction-and-sanity check on the correction, not a transfer estimate.

### Result: accuracy prefers 0.67 or higher, with high confidence

Forced-pick accuracy vs the 0.67 baseline, week-blocked, `clean_core`
(8,933 games):

| retention | accuracy (raw) | Δ accuracy vs 0.67 | 95% CI | `probability_positive` |
|---|---|---|---|---|
| 0.00 | 50.83% | −0.76 pts | [−1.45, −0.10] | **0.0105** |
| 0.20 | 50.88% | −0.72 pts | [−1.28, −0.17] | **0.0045** |
| 0.337 | 50.98% | −0.62 pts | [−1.11, −0.13] | **0.0050** |
| 0.40 | 51.01% | −0.58 pts | [−1.03, −0.16] | **0.0050** |
| 0.50 | 51.28% | −0.31 pts | [−0.69, +0.03] | 0.0380 |
| **0.67 (baseline)** | **51.60%** | — | — | — |
| 0.75 | 51.66% | +0.07 pts | [−0.17, +0.30] | 0.6940 |

Every value between 0.00 and 0.50 is **worse** than 0.67 with
`probability_positive` between 0.005 and 0.04 — i.e., 96-99.5% confidence
that 0.67 beats it — on the primary, highest-power window. The `all`-seasons
window (11,780 paired games) tells the identical story, tighter still
(P+ = 0.0000-0.05 for every value below 0.67). The `thin_2006_2011` split
(2,354 games, sparser-line era) shows the same direction but weaker, as
expected from its lower power (P+ = 0.007-0.44). The small `regime_2020`
split (493 games, COVID) is noisy and directionless (all CIs cross zero) —
too little data to read. Season-blocked intervals match week-blocked to two
decimal places throughout; the finding does not depend on the blocking
choice.

Margin MAE and Brier tell a different, much quieter story: **both are
statistically flat across the entire grid.** Margin MAE differences vs 0.67
are all under 0.0015 points with every CI crossing zero (P+ = 0.36-0.63,
pure noise). Brier improvements are tiny (under 0.0002) but lean the same
direction as accuracy — 0.00 is weakly worse than 0.67 (P+=0.21), 0.75 is
weakly better (P+=0.84) — consistent in sign, just too small to matter
economically on their own.

### Why the direction is plausible, not just a fluke

The production Ridge pipeline (`make_margin_estimator`) runs every feature
through a `StandardScaler` before the Ridge fit. That undercuts the original
note's "the early-season state feature is over-weighted by about 1.8×"
argument for *why* a lower retention should help — that reasoning describes
a raw-scale effect, but standardization removes pure scale differences before
the regression ever sees the column. What retention actually changes, after
standardization, is the **information composition** of the early-season
feature: a lower retention leans more on the noisy, thin, in-season EWM
signal (which has had only 1-4 games to accumulate) relative to the anchor
from a full prior season. Forced-pick accuracy is exactly the metric most
sensitive to that trade — it depends on the *sign* of a small residual, which
noise flips more readily than a shrunk-but-stable anchor does — while a
Ridge model with 34 columns and the market spread already in it barely moves
its aggregate error (MAE, Brier) either way. That is consistent with this
screen: accuracy moves cleanly, margin error does not.

## 4. Secondary check: the two genuinely unspent NFL seasons (not a registry look)

Per `docs/rotation_registry.md`, 2013-2017 and 2014-2017 are spent
(`pbp_drive_bundle`, `player_qb_continuity`, `best_pick_ranker`), 2020-2021
are spent (`mod07_weak_signal_stack`, `best_pick_ranker_opener`), and
2018-2025 carries the separate ~130-150-look prose multiplicity discount the
task brief says to avoid outright. That leaves **2011-2012** as the only NFL
seasons no family has ever scored and that sit outside the mined era —
thin (496 paired games), so this is a sanity check, not a confirmation-grade
look. **No rotation window was declared, assigned, or recorded** — this is
plain iteration on unreserved seasons (rule 8), run directly through
`nfl_ats.outcomes.walk_forward_outcomes`, never through
`nfl_ats.rotation`. Script: `scripts/offseason_retention_nfl_free_seasons.py`.
It carries the same disclosed Elo-wiring confound as any use of
`build_game_features`'s override (§ Defect 1) — a minor one, since Elo is 2
of ~40 columns.

Grid `{0.00, 0.337, 0.40, 0.67, 0.75}`, week-blocked, 2,000 samples:

| retention | Δ accuracy vs 0.67 | P+ | Δ Brier vs 0.67 | P+ | raw margin MAE |
|---|---|---|---|---|---|
| 0.00 | +0.81 pts | 0.71 | +0.0030 (better) | **0.95** | **10.95** |
| 0.337 | −0.20 pts | 0.34 | +0.0016 (better) | 0.92 | 11.01 |
| 0.40 | −0.20 pts | 0.32 | +0.0010 (better) | 0.82 | 11.02 |
| 0.67 (baseline) | — | — | — | — | 11.07 |
| 0.75 | −0.40 pts | 0.19 | −0.00005 (worse) | 0.45 | 11.08 |

Here Brier/log-loss lean toward **lower** retention (P+ 0.82-0.95, though
every CI still crosses zero at n=496) and raw margin MAE is monotonically
better at lower retention — the opposite lean from CFB. Accuracy itself is
flat and noisy in both directions (CIs wide, crossing zero throughout),
which is expected at this sample size. This result is **suggestive, not
decisive** — 496 games is roughly 1/20th of the CFB clean-core sample, and
its direction conflicts with the much higher-power CFB result rather than
confirming it.

## 5. Recommendation: leave `DEFAULT_OFFSEASON_RETENTION = 0.67` unchanged

This is not "inconclusive, so do nothing by default" — it is a screen that
came back with an answer, and the answer argues against the correction on
the metric the whole project is graded on:

- The three NFL-side routes are **confirmed** as measurements of a real,
  specific quantity: how much of a team's season-level deviation from the
  league mean genuinely persists across the season boundary. That number is
  genuinely closer to 0.35-0.45 than to 0.67, and the derivation stands as
  documented fact.
- But "the retention value that best predicts next season's raw
  point-differential in isolation" and "the retention value that maximizes
  forced-pick accuracy inside the shipped, standardized, 34-column
  market-residual Ridge model" are **different questions**, and only the
  second one is the one this project is scored on.
- The one large, direct test of the second question — 12,500 CFB games
  *(corrected 2026-08-18; was 12,206)*,
  the frozen production harness, nothing tuned — says accuracy prefers
  **0.67 or higher**, not lower, with high confidence on the highest-power
  splits (P+ as extreme as 0.0005-0.05 favoring 0.67 over every value
  0.00-0.50). Margin error and Brier are statistically neutral across the
  whole grid, so there is no offsetting reason to move on those either.
- The small, low-power NFL free-season check (2011-2012) leans the opposite
  way on Brier/MAE but is not statistically significant and is outweighed by
  CFB's far larger sample.
- Changing the constant now, on the strength of the routes alone, would move
  every published prediction on the strength of a correctly-measured number
  answering the wrong question — exactly the failure mode the screening
  requirement exists to catch.

**No change to `constants.DEFAULT_OFFSEASON_RETENTION`.** If this is
revisited, the CFB result suggests the more promising next experiment is
sweeping *upward* from 0.67 (e.g. 0.75-0.90) rather than downward, and/or
directly testing per-metric retention values (the routes already show the
"true" persistence rate itself varies 0.195-0.464 by metric) inside the full
Ridge pipeline rather than in isolation — both out of scope here.

## Artifacts

- `scripts/offseason_retention_routes.py` — routes 1-3 (NFL raw snapshot).
- `scripts/offseason_retention_cfb_screen.py` — the predeclared CFB screen.
- `scripts/offseason_retention_nfl_free_seasons.py` — the 2011-2012 secondary
  check.
- Scratch outputs (not committed — regenerate via the scripts above):
  `route1_cells.csv`, CFB `predictions.parquet` / `summary_by_retention.csv`
  / `paired_accuracy_brier.csv` / `paired_margin.csv` / `diagnostics.json`,
  and the equivalent NFL free-season outputs.
