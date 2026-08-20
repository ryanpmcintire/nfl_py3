# Recent player-arrest back-side policy evaluation

## Predeclaration

**Written before this policy was scored.** The broad 14-day incident screen in
`docs/player_arrests_screen.md` fixed a fade direction before looking at ATS
outcomes. Its recorded estimate leaned the other way at both grades. This
follow-up does not rewrite that hypothesis: it declares the opposite direction
as a visibly post-result policy lead and asks the decision-relevant question
directly against the production rule.

The population and baseline are frozen to
`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`: the 2020-2025
paired Tuesday-opener archive and the production probability rule represented
by `pick_home_at_open_probability_rule` / `correct_at_open_probability_rule`.
That baseline is the project's documented 53.36% opener grade, shown publicly
as 53.4%; the distinct 52.10% active-model artifact is close-graded and is not
the decision baseline here.

The candidate is parameter-free after the earlier screen:

1. Reuse the broad USA Today safe-view flag exactly: any mapped incident 1-14
   calendar days strictly before the game's Tuesday decision date. Same-Tuesday
   incidents remain excluded because the source has no intra-day publication
   time.
2. If exactly one team in a game is flagged and the production rule picks the
   other team, flip to the flagged team.
3. If neither or both teams are flagged, or production already backs the sole
   flagged team, retain production's pick.
4. Grade both arms at the same frozen Tuesday opener. Preserve pushes.

The primary endpoint is candidate-minus-production forced-pick accuracy,
week-blocked over the same 107 season-week blocks with 20,000 resamples and
seed 20260820. Season blocking is secondary and carries the runner's existing
small-block warning. Report `probability_positive` continuously.

This is a diagnostic on reused 2020-2025 history and a direction discovered
from an earlier view of overlapping outcomes. It is not a fresh confirmation
and spends no rotation window. Every result must be recorded in
`registry/weak_signals.json` as `unresolved_below_power` unless an admissible
closing ground is genuinely established. A positive expected-value result may
support a no-window-cost 2026 prospective challenger only after a live Tuesday
source-refresh/freshness contract is implemented; it does not by itself change
the published card.

## Results

**Measured** (`artifacts/player_arrests_policy_eval/20260820T162321Z/metadata.json`):
the candidate scores 53.7591% against production's 53.3599% on the same 1,503
graded opener games, a paired +0.3992 accuracy-point effect. The primary
107-week bootstrap estimates +0.3992 points, interval [-0.2688, +1.0774],
standard error 0.3484, and `probability_positive=0.8562` from 20,000 resamples
at seed 20260820.

**Measured** (same artifact): the frozen population has 1,537 rows, including
34 opener pushes retained as pushes. Exactly one side is flagged in 52 games,
both sides are flagged in zero games, and the candidate changes 25 production
picks among the 1,503 graded games.

**Measured** (`artifacts/player_arrests_policy_eval/20260820T162321Z/season_summary.csv`):
candidate-minus-production accuracy points by season are +0.4545 (2020),
0.0000 (2021), +0.4032 (2022), +1.1278 (2023), 0.0000 (2024), and +0.3745
(2025), on 7, 6, 1, 5, 2, and 3 flips respectively.

**Measured** (`artifacts/player_arrests_policy_eval/20260820T162321Z/metadata.json`):
the secondary season-blocked estimate is +0.3992 accuracy points with
`probability_positive=0.9990`. The runner marks the six-block bootstrap
degenerate and instructs readers to use the estimate and
`probability_positive`, not treat its percentile endpoints as a calibrated
95% interval.

## Decision

**Inferred**: the +0.3992-point opener estimate and
`probability_positive=0.8562` favor the candidate in expected-value terms.
Following the frozen decision rule, the appropriate next action is to recommend
a no-window-cost 2026 prospective challenger after implementing a live Tuesday
source-refresh and freshness contract.

**Read** (task scope and the predeclaration above): this lane is not authorized
to build or register that challenger, and this reused-history diagnostic does
not change the published card. No production or prospective activation was
made here.

**Measured** (`registry/weak_signals.json`): exactly one row named
`player_arrests_recent_14d_back_side_policy_opener` records the result as
`unresolved_below_power`, with no closing ground. This is a category-3 result,
not fresh confirmation or a terminal adjudication.

## Reproducibility

**Measured** (2026-08-20 command output): the single authoritative command was:

```powershell
.\.tools\uv.exe run python scripts/player_arrests_policy_eval.py
```

**Measured** (`artifacts/player_arrests_policy_eval/20260820T162321Z/`):
`per_game.parquet` preserves the prediction-level baseline pick, candidate
pick, flags, flip decision, and both opener grades; `season_summary.csv`
preserves the season slices; `metadata.json` records configuration, all input
hashes, the dirty-tree provenance, and both bootstrap summaries.

**Measured** (`registry/experiments/player-arrests-policy-eval/20260820T162321Z.json`):
the versioned provenance identity is
`player-arrests-policy-eval/20260820T162321Z`, linked to the one weak-signal
name above.
