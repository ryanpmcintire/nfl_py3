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

Sections 5 to 7 were appended to this file after the measurement, so this
file's own digest no longer certifies the predeclaration. The byte copy taken
by the `declare` stage before any candidate was scored is
`artifacts/research/laneV/predeclaration.md`, sha256
`743a79d4045f92ebef3dd6d3d949641e196c037564c33cd95a7c207effd2655c`, and that
same digest is stamped into `artifacts/research/laneV/declarations.json` as
`predeclaration_sha256`. That copy is the predeclaration of record.

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

Measured, `artifacts/research/laneV/declarations.json`. Archive
`artifacts/opener_evaluation/20260908T115957Z`, active model `a4c757efd2525da6`,
feature digest `457aafb7...`. Both reconstruction gates returned **exactly 0.0**.

### 5.1 Rule A, the atom set

| Cut | Declared on | 3 | 7 | 10 | 14 | **Rule A selects** | Post-hoc choice |
|---|---|---:|---:|---:|---:|---|---|
| W1 | 2020-2023 | -0.206 | **+0.309** | -0.103 | 0.000 | **{7}** | {3, 7} |
| W2 | 2020-2022 | **+0.142** | **+0.426** | 0.000 | 0.000 | **{3, 7}** | {3, 7} |

Per-atom standalone paired deltas in accuracy points on the declaration
seasons. On the four-season declaration the mechanical rule does **not**
reproduce lane T's post-hoc atom set: the 3 atom is negative there (six picks
moved, all of them on balance for the worse) and only the 7 atom survives. On
the three-season declaration the 3 atom is positive by one pick in ninety games
and the rule reproduces `{3, 7}` exactly. The 10 and 14 atoms are never
selected on either cut -- on W2 neither moves a single graded pick, so both
score exactly zero, and zero is not positive.

### 5.2 Rule B, the bucket set

| Cut | Bucket | n | weeks | slope | 95% | Top below +0.5? |
|---|---|---:|---:|---:|---|---|
| W1 (2020-2023) | 0-3 | 367 | 71 | +0.581 | [+0.049, +1.119] | no |
| | 3.5-6.5 | 353 | 71 | +0.560 | [-0.073, +1.160] | no |
| | 7 | 47 | 37 | +0.520 | [-1.255, +2.382] | no |
| | 7.5-10 | 130 | 59 | -0.050 | [-1.045, +0.885] | no |
| | **10.5+** | 96 | 55 | **-0.932** | [-2.015, +0.137] | **yes** |
| W2 (2020-2022) | 0-3 | 245 | 53 | +0.425 | [-0.239, +1.054] | no |
| | 3.5-6.5 | 264 | 53 | +0.410 | [-0.286, +1.035] | no |
| | 7 | 40 | 30 | -0.084 | [-1.975, +1.789] | no |
| | 7.5-10 | 97 | 44 | -0.024 | [-1.112, +1.034] | no |
| | **10.5+** | 75 | 43 | **-0.942** | [-2.268, +0.399] | **yes** |

**Rule B selects `{10.5+}` on both cuts, which is exactly lane U's post-hoc
choice.** Two buckets have a negative point estimate on both cuts -- `7.5-10`
and `10.5+` -- and the rule takes only the one whose whole interval clears the
ceiling. That is the mechanism's own boundary doing the work: `7.5-10`'s slope
is not resolved away from face value, and lane U measured that serving it there
is where R2's loss came from.

---

## 6. Held-out results

Measured, `artifacts/research/laneV/cells.json`. Paired against the served S3
read on held-out-season games with a decided opener grade; week-blocked
bootstrap, 20,000 draws, seed 20260817, within-week correlation ZERO.

### 6.1 W1 -- declared on 2020-2023, graded on 2024-2025 (533 graded games, 36 weeks)

| Arm | Declared set | Touched | Standalone % | Card % | Card delta [95%] | P+ | Standalone delta [95%] | P+ | Brier | P+ | Log loss | P+ |
|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| S3 | -- | -- | 54.409 | 54.784 | reference | -- | reference | -- | reference | -- | reference | -- |
| **KL** | atoms {7} | 27 | 54.972 | **55.347** | **+0.563 [-0.189, +1.331]** | **0.880** | +0.563 [-0.378, +1.498] | 0.835 | -0.00002 | 0.480 | -0.00005 | 0.474 |
| **RS** | buckets {10.5+} | 38 | 54.221 | 54.597 | -0.188 [-0.771, +0.380] | 0.183 | -0.188 [-0.771, +0.380] | 0.183 | -0.00037 | 0.236 | -0.00074 | 0.239 |
| **Both** | {7} + {10.5+} | 65 | 54.784 | **55.160** | **+0.375 [-0.558, +1.304]** | **0.727** | +0.375 [-0.923, +1.679] | 0.665 | -0.00039 | 0.244 | -0.00079 | 0.242 |

### 6.2 W2 -- declared on 2020-2022, graded on 2023-2025 (799 graded games, 54 weeks)

| Arm | Declared set | Touched | Standalone % | Card % | Card delta [95%] | P+ | Standalone delta [95%] | P+ | Brier | P+ | Log loss | P+ |
|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| S3 | -- | -- | 55.069 | 55.820 | reference | -- | reference | -- | reference | -- | reference | -- |
| **KL** | atoms {3, 7} | 142 | 54.944 | **56.195** | **+0.375 [-0.501, +1.253]** | **0.761** | -0.125 [-1.261, +1.003] | 0.372 | -0.00008 | 0.404 | -0.00017 | 0.404 |
| **RS** | buckets {10.5+} | 59 | 55.444 | 55.820 | +0.000 [-0.625, +0.622] | 0.419 | **+0.375 [-0.378, +1.244]** | **0.776** | +0.00033 | 0.731 | +0.00069 | 0.733 |
| **Both** | {3, 7} + {10.5+} | 201 | 55.319 | **56.195** | **+0.375 [-0.628, +1.377]** | **0.731** | +0.250 [-1.256, +1.838] | 0.593 | +0.00025 | 0.659 | +0.00053 | 0.663 |

### 6.3 Per held-out season, accuracy points through the card

| Season | W1 KL | W1 RS | W1 Both | W2 KL | W2 RS | W2 Both |
|---|---|---|---|---|---|---|
| 2023 | -- | -- | -- | -0.376 (0.186) | +0.376 (0.615) | +0.000 (0.421) |
| 2024 | +0.752 (0.778) | +0.376 (0.645) | +1.128 (0.962) | +2.256 (0.987) | +0.376 (0.645) | +2.632 (1.000) |
| 2025 | +0.375 (0.643) | -0.749 (0.000) | -0.375 (0.184) | -0.749 (0.000) | -0.749 (0.000) | -1.498 (0.000) |

`probability_positive` in brackets. 2024 favours every arm on both cuts; 2025
goes against every arm on both cuts. Nothing here is closed: the 2025 cells
whose upper bound is exactly 0.000 are one or two flipped picks in a season,
recorded `unresolved_below_power` with no closing ground.

### 6.4 On the games each arm actually changes

| Cut | Arm | n | Card delta [95%] | P+ |
|---|---|---:|---|---:|
| W1 | KL | 27 | **+11.111 [-4.000, +28.000]** | **0.883** |
| W1 | RS | 37 | -2.703 [-12.903, +5.714] | 0.178 |
| W2 | KL | 131 | +2.290 [-3.077, +7.759] | 0.758 |
| W2 | RS | 58 | +0.000 [-8.621, +8.333] | 0.415 |

### 6.5 Gates, measured

| Gate | Value |
|---|---|
| Feature digest vs active manifest | equal |
| Archive named by lanes T, R, U | all `20260908T115957Z`, the archive the active model matches |
| S3 replay, lane T frame / lane R frame | passed lane K's `verify_replay` (stop above 1e-9) |
| Lane T `p_S3` vs lane R `p_S3` | exactly equal |
| Archive served pick vs served S3 probability | exactly equal |
| Reconstruction, atoms `{3, 7}` vs lane T's `p_KL1b` | **0.0** |
| Reconstruction, buckets `{10.5+}` vs lane U's `p_R2b` | **0.0** |
| Untouched-game probability gap, all six arms | **0.0** |
| Composition on the held-out-restricted artifact vs the full archive composition | **0 disagreements**, all six arms and the incumbent |
| Degenerate cells skipped | none: every cell moved at least one graded pick |

---

## 7. What the held-out reads mean for the decision

**Serve KL1b. The key-line restriction survives the discount; the re-scale does
not, but it is not refuted either.**

- The key-line arm is the better card on BOTH held-out blocks, at
  `probability_positive` **0.880** (W1) and **0.761** (W2). Declining it is
  taking the other side of a 88/12 and a 76/24 bet on a pool that submits a
  card either way.
- The bucket rule reproduced lane U's post-hoc `{10.5+}` choice exactly on both
  cuts, so R2b's structure carries no selection discount at all -- but its
  held-out card reading is -0.188 (P+ 0.183) on W1 and +0.000 (P+ 0.419) on W2.
  On expected value that is a bet against playing it, so **R2b stays off the
  card**. It is not closed: no wrong sign is resolved, no positive control
  bounds it, and its stated chances are the better ones on W2 (Brier P+ 0.731,
  log loss P+ 0.733) -- which is the same coherent split lanes R and U measured,
  better probabilities and worse forced picks.
- The combination is positive through the card on both cuts (+0.375, P+ 0.727
  and 0.731) but it is worse than the key-line arm alone on W1 and identical to
  it on W2. Adding the re-scale buys nothing on top.
- The atom rule did **not** reproduce `{3, 7}` on the four-season declaration --
  it picked `{7}` alone, and that narrower arm is the strongest held-out card
  reading in the lane (+0.563, P+ 0.880, on 27 touched games). On the
  three-season declaration it reproduced `{3, 7}` exactly. So the part of
  KL1b that survives out of sample cleanly is the 7 atom; the 3 atom's evidence
  is one pick in ninety declaration games on the cut that kept it.

**Then the caveats, in order of size.** The held-out blocks are two and three
seasons -- 533 and 799 graded games -- and the two cuts share their declaration
seasons and overlap in 2023, so they are two readings of nearly the same
football, not two replications. Every held-out interval crosses zero, which at
this evaluator's resolution is the expected outcome for an effect this size and
is never grounds to reject anything. The arms move few games: 27 to 201 of 533
to 799. And the declaration removes the discount on the ATOM SET and the BUCKET
SET only -- the archive, the S3 baseline and the mass-preserving construction
were all chosen on these same seasons.

**Nothing is closed.** All 66 recorded cells are `unresolved_below_power` with
no closing ground, including the negative ones.
