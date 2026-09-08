# Out-of-sample declaration of the two live MOD-18 candidates (lane V)

Both candidates now sitting in front of the played card were chosen after
somebody looked at the archive they are graded on.

- **Lane T's KL1b** serves lane K's discrete key-number read for the pick, but
  only where the opener line is quoted exactly on 3 or 7. The atom set `{3, 7}`
  was chosen after lane K's bucket table showed the gain sitting at 7, and
  after lane T saw that dropping the 10 atom improved the card. Reported card
  effect +0.200 accuracy points, `probability_positive` 0.7426, on 2020-2025.
- **Lane U's R2b** re-scales the model's raw residual by a walk-forward
  per-bucket slope, but only in the `10.5+` bucket. The bucket was chosen after
  lane R's per-bucket table showed the whole loss living in `7.5-10` and the
  only favourable reading living in `10.5+`. Reported card effect +0.000,
  standalone +0.200, `probability_positive` 0.736, on 2020-2025.

Both numbers therefore carry a selection discount that neither lane could
quantify. This lane takes the discount off the only way it can be taken off
without new football: **declare each rule's structure mechanically on an
earlier block of seasons, then grade it on seasons the declaration never
saw.**

---

## 1. Predeclaration (frozen before any number in section 6 existed)

### 1.1 The two declaration rules, stated exactly

Both rules are mechanical. Neither admits a judgement call, an eyeballed
table, or a tie broken by taste. They were fixed verbatim in this lane's
assignment before any candidate arm existed.

**Rule A -- the atom set (key-line arm).**

> Every key atom in `{3, 7, 10, 14}` whose standalone paired delta on the
> declaration seasons is positive.

Made fully explicit:

1. For each atom `a` in `{3, 7, 10, 14}`, build the single-atom arm: lane K's
   mass-preserving read `p_MP1` on every game whose opener line
   `tue_open_home_spread` has absolute value exactly `a` (tolerance `1e-9`,
   so 6.75, 7.25 and 7.5 are NOT the 7 atom), and the served S3 read on every
   other game.
2. Score that arm against S3 with `spread_regime_opener_eval.comparison` on
   the declaration seasons only: paired accuracy points, week-blocked
   bootstrap, 20,000 draws, seed 20260817, whole (season, week) blocks,
   **within-week correlation ZERO, never estimated and never padded**.
3. The atom is selected if and only if that delta is **strictly greater than
   zero**. An atom that moves no graded pick has a delta of exactly zero and is
   therefore NOT selected.

Note that the point estimate is used, not its interval: a rule that required a
delta to exclude zero would be the banned crossing-zero rejection wearing a
different hat. Note also that scoring the single-atom arm over ALL declaration
games and scoring it over only the games on that atom give deltas differing by
the positive factor `n_atom / n_all`, so the two readings always agree on the
sign and the rule is invariant to that choice.

**Rule B -- the bucket set (residual re-scale arm).**

> Every bucket whose walk-forward slope interval on the declaration seasons
> lies wholly below +0.5.

Made fully explicit:

1. Take the opener archive as the walk-forward prior stream
   (`nfl_ats.home_side_location.archive_prior_stream`) and keep the rows of the
   declaration seasons that have a result.
2. For each of the five declared spread buckets (`0-3`, `3.5-6.5`, `7`,
   `7.5-10`, `10.5+`), take the least-squares slope of the realised margin
   against the line on the model's raw out-of-time residual, with lane R's
   week-blocked bootstrap interval (`residual_slope_opener_eval.slope_intervals`
   -> `week_blocked_slope`; 20,000 draws, seed 20260817, whole weeks as blocks,
   **within-week correlation ZERO**).
3. The bucket is selected if and only if the **97.5th percentile of that
   interval is strictly below +0.5**. A bucket with no estimable slope is not
   selected.

+0.5 is the midpoint between "the residual is worth nothing" (slope 0) and
"the residual is worth its face value" (slope 1, the incumbent S3). It is
declared as the boundary because a bucket whose slope is resolved to be nearer
0 than 1 is the only place the mechanism claims to act; it is not tuned, and no
other threshold is computed.

**The served arm on the held-out seasons.** Only the STRUCTURE is declared: the
atom set for rule A, the bucket set for rule B. The reads themselves stay
walk-forward exactly as lanes K, R and U built them -- lane K's five-season
prior pool, declared bandwidth and exponential tilt for the atoms; lane R's
per-bucket least-squares slope shrunk toward 1.0 with the 100-game prior
(`PRIOR_WEIGHT_GAMES`) for the buckets, refit before every target week. Nothing
is re-tuned, and no held-out outcome ever enters a fit for its own week.

### 1.2 The combination, and its precedence rule

The two arms can name the same game: a line quoted exactly on 14 sits in the
`10.5+` bucket. The precedence is declared here, before the combination was
computed:

> **The key-line read wins.** On a game whose line sits exactly on a declared
> atom, the mass-preserving read is served. Otherwise, if the game's bucket is
> in the declared bucket set, the re-scaled read is served. Otherwise the game
> keeps the served S3 read, bit for bit.

The atom rule is the narrower restriction -- it names an exact number rather
than a range -- so it takes the games it names.

### 1.3 The two cuts

| Cut | Declaration seasons | Held-out seasons |
|---|---|---|
| **W1** | 2020-2023 | 2024-2025 |
| **W2** | 2020-2022 | 2023-2025 |

W2 exists so the answer does not rest on a single split. The two cuts overlap
heavily (2023 is held out in W2 and declared on in W1) and are NOT independent
replications.

### 1.4 Grading, on the held-out seasons only

- Paired accuracy points against the **served S3 read**, on held-out-season
  games with a decided opener grade.
- Week-blocked bootstrap, **20,000 draws, seed 20260817**, whole (season, week)
  blocks, **within-week correlation ZERO -- hard-coded, never estimated, never
  padded** (`within-week-correlation-is-zero`, binding).
- `probability_positive` reported for every cell. The binary "the interval
  contains zero" is never reported and is never grounds for anything.
- Brier and log-loss improvement against S3, positive meaning the candidate's
  stated chances were closer to what happened.
- **Through the played three-member card**: each arm is written as a research
  opener evaluation restricted to the held-out weeks
  (`research_arm` in the metadata, `active_model_id` `research_laneV_oos1`),
  put through `nfl-ats overlay-composition`, and its composed picks taken from
  `spread_regime_opener_eval.composed_picks` (the second, three-member return).

### 1.5 The decision rule, declared before the numbers

`probability_positive` above 0.5 on the held-out seasons favours PLAYING the
arm in place of S3. The pool is forced picks: 285 cards get submitted either
way, so declining an arm that is more likely than not to be better is taking
the other side of that bet (AGENTS.md, "A promotion bar is not a decision
bar"). No arm is closed here: no closing ground is claimed, and every cell is
recorded `unresolved_below_power`.

### 1.6 Disclosed before the fact

- The held-out blocks are **two seasons (~530 graded games) and three seasons
  (~800)**. A two-point effect is not resolvable on that, and is not expected
  to be. The held-out read is a discount check on a post-hoc choice, not a
  power run.
- W1 and W2 share their declaration seasons and overlap in 2023; the two cuts
  are correlated, and so are the three arms within a cut.
- The archive itself was mined by lanes H, K, L, R, S, T and U. Declaring the
  structure out of sample removes the discount on THAT CHOICE ONLY; it does not
  make the archive fresh, and S3 -- the baseline -- is itself a restriction
  fitted on these games.
- A feasibility probe in the session scratchpad ran both rules and the
  composition timing before this document was written. It could not change
  either rule, because both were fixed verbatim in the lane assignment; it is
  disclosed so the reader can discount the predeclaration accordingly. The
  combination's held-out result was not computed before the precedence rule in
  section 1.2 was written.
- No candidate here is served, no forecast is regenerated, no card is
  republished, and `nfl-ats weak-signals record` is not run by this lane -- the
  record commands are written to
  `artifacts/research/laneV/record_commands.ps1` for serial execution.

### 1.7 Closing grounds (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`.

---

## 2. What this lane does NOT do

It does not re-implement the replay gate, the mass-preserving read, the slope
fit, the composition or the bootstrap. All of them are imported from
`scripts/mass_preserving_lattice_opener_eval.py` (lane K),
`scripts/key_line_lattice_opener_eval.py` (lane T),
`scripts/residual_slope_opener_eval.py` (lane R),
`scripts/residual_slope_shrunk_opener_eval.py` (lane U),
`scripts/home_side_location_opener_eval.py` (lane S) and
`scripts/spread_regime_opener_eval.py`.

The two candidate reads are reconstructed **bucket-locally and atom-locally**
from the frozen lane artifacts, because both mechanisms are local by
construction:

- The key-line arm's probability on a game is lane K's `p_MP1` if the game's
  line is on a declared atom and the served `p_S3` otherwise. That is exactly
  lane T's `restrict`, imported.
- The re-scale arm's served point is `line + beta_b * raw residual + offset_b`,
  and `beta_b` depends only on that game's own bucket. So the arm's probability
  on a game is lane R's `p_R1` if the game's bucket is declared and `p_S3`
  otherwise, whatever the rest of the bucket set is.

Both reconstructions are gated against the frozen lanes rather than trusted:
the atom-local build with `{3, 7}` must reproduce lane T's `p_KL1b` exactly,
and the bucket-local build with `{10.5+}` must reproduce lane U's `p_R2b`
exactly.

---

## 3. Gates (all fail closed)

| Gate | Rule |
|---|---|
| Feature digest | equals `active_ats_model.json`'s `feature_table_sha256` |
| Archive identity | lanes T, R and U all name the archive the active model matches |
| S3 replay | lane K's `verify_replay`, stop above 1e-9, on both stored frames |
| Cross-lane S3 | lane T's `p_S3` equals lane R's `p_S3` exactly |
| Reconstruction, atoms | `{3, 7}` build equals lane T's `p_KL1b` exactly |
| Reconstruction, buckets | `{10.5+}` build equals lane U's `p_R2b` exactly |
| Untouched games | every arm's probability equals `p_S3` exactly off its declared set |
| Composition restriction | composed picks on the held-out-restricted artifact equal the full-archive composition restricted to those rows |
| Declaration hygiene | no held-out season contributes a row to any declaration |

---

## 4. Reproduction

```powershell
.\.tools\uv.exe run --no-sync python scripts\out_of_sample_declaration_opener_eval.py --stage declare
.\.tools\uv.exe run --no-sync python scripts\out_of_sample_declaration_opener_eval.py --stage score
.\.tools\uv.exe run --no-sync python scripts\out_of_sample_declaration_opener_eval.py --stage record
.\.tools\uv.exe run --no-sync pytest tests\test_out_of_sample_declaration.py -n 2
```

Artifacts land under `artifacts/research/laneV/`; the live registry is never
written by this lane.

---

## 5. What each declaration rule selected

Filled in by the `declare` stage. See `artifacts/research/laneV/declarations.json`.

---

## 6. Held-out results

Filled in by the `score` stage. See `artifacts/research/laneV/cells.json`.
