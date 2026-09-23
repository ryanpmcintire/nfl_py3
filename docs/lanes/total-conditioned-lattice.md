# Total-conditioned key-number lattice

## Goal
Test whether the served discrete cover/push read
(`discrete_conditional_non_push_v1`, `src/nfl_ats/mass_preserving_lattice.py`)
should additionally condition its key-number lattice on the game's market
total, since low-total games should carry more mass at 3 and less at 7/10/14
than high-total games. Grade a challenger that adds total-banding against the
served read on the opener grade; classify per AGENTS.md
`weak-signals`/"An interval crossing zero is not grounds for rejection".

## State
Read (confirmed): the served read (`band_read`,
`mass_preserving_lattice.py:112-149`) selects prior games only by
`abs(pool_line - anchor) <= band` (spread-line band, expanding until
`MIN_BAND_GAMES=200` or `MAX_BAND=20`, `mass_preserving_lattice.py:26-28`).
`prior_pool` (`mass_preserving_lattice.py:163-183`) carries only
`game_id, season, week, gameday, spread_line, result, line` -- **no total is
read anywhere in the served path**. `total_line` exists pregame in
`data/processed/game_features_weak_stack.parquet` back to 2009 (checked:
2023 total_line mean 43.1, n=285/season).

Built and ran once for real: `scripts/total_conditioned_lattice.py` (new,
`ruff format` + `ruff check` clean, no comments). Predeclared total bands
(fixed before grading, round numbers near league-average total, not fit to
this test population): low `<42.5`, mid `42.5-47.5`, high `>47.5`. Challenger
= identical `tilted_atoms`/band-expansion mechanism, but the eligible prior
pool is additionally masked to the target game's total band before the line-
band expansion; falls back to the unconditioned served pool if total-banded
`MIN_BAND_GAMES` is unreachable by `MAX_BAND`. `point = spread_line` for BOTH
arms (isolates the lattice-selection mechanism from any margin-model point;
not the served `predicted_margin` point -- labeled in the artifact). Line
source: feature-table `spread_line` for both arms (no opener-evaluation
override matched this run), so the total-band delta is a controlled A/B, not
an apples-to-production comparison.

Artifacts: `artifacts/total_conditioned_lattice/20260923T205550Z/`
(`summary.json`, `primary_2020_2025.parquet`, `extended_2009_2025.parquet`).

**Primary 2020-2025 (measured, n=1,615 games, 1,582 decisive):**
- Log loss (3-way cover/push/loss): served 0.76623 vs challenger 0.76199
  (challenger better by 0.00424 nats).
- Brier (3-way): served 0.51896 vs challenger 0.52019 (challenger worse by
  0.00123).
- Forced-pick accuracy on decisive games: served 51.58% vs challenger 51.14%
  (challenger worse by 0.44 pts).
- Paired disagreement: 285/1,615 games picked opposite sides; served correct
  146/285 (51.2%), challenger correct 139/285 (48.8%) on those disagreements.
- Push calibration at integer lines (n=738): served predicted-minus-actual
  push rate = +0.00856; challenger = +0.00428 (challenger closer to
  calibrated; both over-predict push slightly).
- LOSO by season, mixed sign both metrics: log loss -- challenger better in
  2020/2022/2025, served better in 2021/2023/2024 (3-3 split). Accuracy --
  challenger better only in 2021/2025, served better in 2020/2022/2023/2024
  (4-2 split for served). No consistent sign across seasons.
- Pool size: challenger median band_games 210 (just above `MIN_BAND_GAMES`),
  served median 374; challenger fallback-to-unconditioned rate 0.06% in the
  primary window.

**Extended 2009-2025 (reported only, per task instruction, not the primary
decision population), n=4,415:** log loss served 0.77714 vs challenger
0.77566 (challenger better by 0.00148); accuracy served 50.77% vs challenger
50.95% (challenger better by 0.18 pts -- sign flips vs primary); fallback
rate to unconditioned rises to 13.8% (thinner total-banded pools in the early
seasons).

**Unit 2 (measured): paired bootstrap intervals, exact null, split-half
reliability.** Built `scripts/total_conditioned_lattice_intervals.py` (new,
`ruff format` + `ruff check` clean, no comments), run once
(`/f/Repos/nfl_py3/.tools/uv.exe run python scripts/total_conditioned_lattice_intervals.py`,
exit 0). Reads the saved Unit-1 parquets, no refit. Artifact:
`artifacts/total_conditioned_lattice_intervals/20260923T210335Z/summary.json`.
2,000 draws, seed 20260923, for both season-block (resample the 6 primary
seasons with replacement) and week-block (resample the 107 primary
`(season, week)` blocks with replacement) bootstraps. Sign convention:
positive = favors the total-conditioned challenger throughout (log loss and
Brier as served-minus-challenger; accuracy as challenger-minus-served points;
push calibration as served absolute error minus challenger absolute error).

Primary 2020-2025 (n=1,615; season-block reported as primary interval, more
conservative than week-block; week-block in parens):
- Log loss improvement: observed +0.00424. Season-block SE 0.00558, 95% CI
  [-0.00351, +0.01681], **P+ 0.708** (week-block CI [-0.00425, +0.01822],
  P+ 0.678). Favors challenger but crosses zero.
- Brier improvement: observed -0.00123. Season-block SE 0.00067, 95% CI
  [-0.00249, +0.00005], **P+ 0.034** (week-block CI [-0.00319, +0.00062],
  P+ 0.105). Favors served; interval nearly excludes zero on the served side
  but does not fully exclude it (upper bound +0.00005 season-block) --
  `unresolved_below_power`, not a resolved wrong sign.
- Accuracy points (challenger - served, decisive games): observed -0.44.
  Season-block SE 0.745, 95% CI [-1.78, +1.05], **P+ 0.269** (week-block CI
  [-2.58, +1.64], P+ 0.315).
- Push-calibration improvement (integer lines, n=738): observed +0.00428.
  Season-block SE 0.00413, 95% CI [-0.00584, +0.00547], **P+ 0.681**
  (week-block CI [-0.00489, +0.00530], P+ 0.789).
- Exact null, 285 disagreement games: challenger correct 139/285 (48.77%,
  95% CI [42.8%, 54.7%]), served correct 146/285; two-sided exact binomial
  p = 0.722 against p=0.5. No evidence either side wins the disagreement set
  more than chance.
- Split-half reliability of the per-game log-loss improvement (method: mean
  improvement per season on odd-parity vs even-parity weeks, n=6 season
  units, Pearson r across those units, Spearman-Brown corrected for
  full-length): raw r = 0.026 (p=0.961), Spearman-Brown r = 0.050. Both are
  below the registry's 0.10 `no_split_half_reliability` closing ceiling, but
  a Fisher-z 95% CI on the raw r from n=6 units is **[-0.80, +0.82]**
  (computed by hand from the reported r and n, not re-run in the script) --
  this measurement cannot distinguish zero reliability from moderate
  reliability at this sample size, so it does NOT license closing via
  `no_split_half_reliability`; it is itself `unresolved_below_power`.

Extended 2009-2025 (reported only, not re-bootstrapped in this pass, n=4,415):
log loss improvement +0.00147, Brier improvement +0.00013, accuracy points
+0.186, push-calibration improvement -0.00124 (matches Unit 1's reported sign
flip on accuracy vs primary).

## Tried
Unit 1: ran `scripts/total_conditioned_lattice.py` once, point estimates only.
Unit 2: ran `scripts/total_conditioned_lattice_intervals.py` once (above),
adding season-block and week-block bootstrap intervals, P+, the 285-game
exact disagreement null, and split-half reliability. No refit, no served-path
change.

## Next (for the root orchestrator; this subagent made no registry writes)
1. Record with (season-block interval used as primary; edit if root prefers
   week-block or a pooled interval):

```
nfl-ats weak-signals record \
  --name total_conditioned_key_number_lattice_log_loss \
  --description "Discrete cover/push lattice additionally conditioned on the opening total (3 predeclared bands) vs the served discrete_conditional_non_push_v1 read, point=spread_line both arms, 2020-2025" \
  --source artifacts/total_conditioned_lattice_intervals/20260923T210335Z/summary.json \
  --effect 0.00424 \
  --effect-units log_loss_improvement \
  --standard-error 0.00558 \
  --interval-low -0.00351 --interval-high 0.01681 \
  --probability-positive 0.708 \
  --classification unresolved_below_power \
  --league nfl \
  --season-start 2020 --season-end 2025 \
  --sample-games 1615 \
  --sample-blocks 6 \
  --reliability 0.05 \
  --family total_conditioned_lattice \
  --classification-evidence "Season-block bootstrap (2000 draws, seed 20260923) P+ 0.708 for log loss favors the challenger but the 95% interval [-0.00351, 0.01681] crosses zero; Brier moves the other way (P+ 0.034); forced-pick accuracy P+ 0.269 also favors served. The 285-game disagreement set is an exact coin flip (139/285 challenger, two-sided binomial p=0.722). Split-half reliability of the log-loss improvement is nominally low (Spearman-Brown r=0.050) but its Fisher-z 95% CI from only 6 season units is [-0.80, 0.82], so it cannot support the no_split_half_reliability closing ground. No whole-interval wrong sign, no positive control: unresolved, not closed." \
  --category modeling \
  --plain-summary "We tried giving the model's key-number math a heads-up about how high-scoring a game is expected to be, on top of the point spread it already uses. In resampling checks it nudged the probabilities in the right direction about 7 times in 10, but a stricter accuracy check leaned the other way -- still not a clear win or loss." \
  --notes "Week-block bootstrap (107 season-week blocks) gives a consistent read: CI [-0.00425, 0.01822], P+ 0.678. Extended 2009-2025 window (reported only, not recorded): log loss improvement +0.00147."

nfl-ats weak-signals record \
  --name total_conditioned_key_number_lattice_accuracy \
  --description "Same challenger, paired forced-pick accuracy on decisive (non-push) games, 2020-2025" \
  --source artifacts/total_conditioned_lattice_intervals/20260923T210335Z/summary.json \
  --effect -0.44 \
  --effect-units accuracy_points \
  --standard-error 0.745 \
  --interval-low -1.78 --interval-high 1.05 \
  --probability-positive 0.269 \
  --classification unresolved_below_power \
  --league nfl \
  --season-start 2020 --season-end 2025 \
  --sample-games 1582 \
  --sample-blocks 6 \
  --family total_conditioned_lattice \
  --classification-evidence "Season-block bootstrap (2000 draws, seed 20260923): 95% CI [-1.78, 1.05] points, P+ 0.269 (week-block CI [-2.58, 1.64], P+ 0.315) -- crosses zero both ways, not a resolved wrong sign. On the 285 games where the challenger disagreed with the served pick, the exact two-sided binomial test against a 50/50 null gives p=0.722 (challenger 139/285, 95% CI [42.8%, 54.7%]): no evidence either side wins the disagreement set more than chance. No admissible closing ground reached." \
  --category modeling \
  --plain-summary "Giving the key-number math a heads-up about the game's total didn't reliably change how often the forced pick was right. Our resampling checks give it only about 1-in-4 odds of actually helping, and on the games where it disagreed with the current approach it split almost exactly 50/50 (139-146 of 285) -- no real edge either way yet." \
  --notes "Extended 2009-2025 window (reported only, not recorded): accuracy points +0.19 (sign flips vs primary's -0.44); not re-bootstrapped."
```

2. If pursued further: a smooth total-interaction (continuous, not 3 bands)
   is the natural next challenger per the task's "or a smooth total
   interaction" option -- not attempted here (budget), predeclare its
   functional form before looking if tried. Also open: a season-unit split
   -half reliability check on a wider window (e.g. extended 2009-2025's 17
   seasons instead of primary's 6) would materially narrow the Fisher-z CI
   before any reliability-based closing decision.
3. This subagent made no served-path changes, no commits, no publish, no
   registry writes -- root runs the `weak-signals record` commands above (or
   revises them) and any commit/push.

## Open
- Whether to prefer feature-table `total_line` as an opener-total proxy long
  term, or build a true opening-total archive analogous to the
  opener-evaluation `tue_open_home_spread` matching
  (`find_matching_opener_evaluation`, `src/nfl_ats/public_board.py:2964`) --
  this run used the feature table's `total_line` directly (pregame-available
  per `src/nfl_ats/features.py:679`, same fallback tier the served spread
  pool already accepts), flagged as a labeled limitation, not fixed here.
- Owner has not been asked whether 3 fixed total bands vs a smooth
  interaction is preferred if this line of work continues.
