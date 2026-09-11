# Calibrating the served probability by spread size, on the discrete margin lattice (MOD-18 C2)

Lane C3. Predeclared 2026-09-10 before any candidate number was computed;
everything above the "Measured" heading was frozen first and nothing above it
was edited afterwards. Family `spread_size_calibration`, category `modeling`.

## The question this lane answers

Read: `ROADMAP.md` row MOD-18. The owner's diagnosis, measured 2026-09-07 on
the then-active model, is that the served probability is **miscalibrated by
spread size**: accuracy 56.2% (0-3) / 56.2% (3.5-6.5) / 52.7% (7) / 48.5%
(7.5-10) / 44.3% (10.5+) while the stated confidence sits at 55-57% in every
bucket. The owner's rule for what to do about it (`AGENTS.md`, "No unexplained
threshold flips on the played card"): do not bolt a flip onto the buckets that
dip -- **calibrate by spread instead**, and name the mechanism.

Read: `AGENTS.md`, "Football margins are multimodal, not Gaussian" -- any
served cover probability is computed against the DISCRETE margin distribution
conditional on the line; a smooth pooled-residual read may run only as a
CHALLENGER and must beat the discrete read on the opener grade.

So the candidate is a **spread-conditional recalibration fitted on the
discrete lattice**, and the same recalibration fitted on the smooth read is
carried as the challenger the rule requires, not as the primary.

## What is already measured, so this lane does not repeat it

| Lane | Arm | Result |
|---|---|---|
| H (2026-09-07) | C3, walk-forward spread-bucket calibration of the **smooth** read, pre-S3 baseline | -2.26 pts, `probability_positive` 0.09 |
| K (`docs/mass_preserving_lattice.md`) | MP1, the discrete lattice read on every game | -0.399 pts through the card, `probability_positive` 0.311 |
| T (`docs/key_line_lattice.md`) | KL1b, the lattice where `abs(line)` is exactly 3 or 7 -- **served** | +0.200 pts through the card, 0.7426 |
| C2 (`artifacts/research/laneC2/predeclaration.md`) | G1-G4, the lattice at every line and at the half points beside 3 and 7 | -0.13 to -0.60 pts through the card, 0.15-0.41 |

None of those is a **calibration**: every one of them swaps the probability
MAPPING and leaves the stated confidence uncorrected. Lane H's C3 was a
calibration but on the smooth read, on a superseded baseline, and it never
carried a chronological fit/score split. Nothing has fitted a spread-conditional
recalibration ON the lattice, on earlier seasons, and scored it on later ones.

## The construction

The lattice is imported, never reimplemented:
`nfl_ats.mass_preserving_lattice.walk_forward_reads` -- the empirical
integer-margin atoms of prior games quoted within a band of the line, atoms at
their ABSOLUTE positions (MOD-05's recentring constraint), the model's point
entering only as an exponential tilt solved so the tilted mean equals the
served point. Band half-width 2.5 widened in 0.5-point steps to a 200-game
floor; five trailing seasons; the target week, every later week and each
target game id excluded before an atom is counted. The lattice probability is
`cover + push / 2` (`MassPreservingRead.home_cover_probability`), the served
convention.

The served point is replayed from the archive, not refitted: the opener
evaluation's `residual_at_open_served` already carries the served home-side
offset (S3, `docs/home_side_offset_promotion.md`), and the per-week
`(median, std)` of the fitted residual sample is recovered by algebraic
inversion of the `gaussian_median` read per (season, week). The replay is
gated at 1e-9 against the archive's own served probability; the run fails
closed above that.

The recalibration, per spread bucket `b` (the repo's own
`nfl_ats.spread_regime.BUCKETS`: 0-3, 3.5-6.5, 7, 7.5-10, 10.5+):

```
logit(p_calibrated) = a_b + s_b * logit(p_read)
```

fitted by maximum likelihood on the FIT seasons only, then shrunk toward the
identity `(0, 1)` with a 100-game prior -- the same prior weight lane S
declared for the served home-side offset, inherited, not tuned here.

The intercept `a_b` is what can move a pick: it is a per-bucket home-side tilt
in probability space. The slope `s_b` is pure confidence scaling and cannot
cross 0.5 on its own. Both are reported per bucket so the mechanism is
readable.

## The arms, frozen

| Arm | Read | Recalibration |
|---|---|---|
| **A0** | lattice, every game | none (reference; lane K's MP1 re-measured on the current active model) |
| **A1** | lattice, every game | `a_b` and `s_b` |
| **A2** | lattice, every game | `a_b` only (`s_b = 1`) |
| **A3** | smooth `gaussian_median` (S3) | `a_b` and `s_b` -- the CHALLENGER `AGENTS.md` allows |
| **A4** | smooth `gaussian_median` (S3) | `a_b` only |

Two baselines, both reported for every arm, neither used to select: **S3**,
the served smooth read the archive holds, and **KL1b**, S3 with lane T's
served exact-atom override, which is what is PLAYED at the model level.

Ranking, fixed before scoring: report all five arms against both baselines.
There is no selection step and no arm is dropped. The decision is expected
value (`AGENTS.md`, "A promotion bar is not a decision bar").

## Metric and protocol

- Forced-pick ATS accuracy at the **OPENER**, paired per game, on the archived
  Tuesday-opener evaluation matching the active model. Non-push games are the
  graded set.
- **Chronological split: fit on 2020-2022, score on 2023-2025.** The
  recalibration sees no game from a season it is scored on.
- Block bootstrap over (season, week) blocks, 20,000 draws, seed 20260817.
  Within-week game correlation is ZERO by owner mandate; no variance inflation
  and no ICC is estimated.
- `probability_positive` reported for every cell. The binary "the interval
  contains zero" is never reported and is never a verdict.
- Reliability table (stated confidence vs realised accuracy, stated home-cover
  vs realised home-cover rate, Brier) by bucket, before and after, on the
  scored seasons.
- Per-bucket paired deltas in accuracy points and Brier.
- Picks changed, counted. A recalibration that changes probabilities and no
  picks is still a finding -- the reader-facing confidence on the Model page
  is the thing it fixes -- and will be reported as exactly that.

## Reuse discount, disclosed before the run

The 1,537-game opener archive is the one lanes K, S, T, V and C2 were selected
on, and S3 itself is a post-hoc restriction fitted on it. The chronological
split removes the recalibration's own in-sample advantage but not the
archive's history: 2023-2025 has been read by earlier MOD-18 lanes. These are
descriptive reused-window measurements, not independent confirmation. No
rotation window is spent (precedent: `artifacts/research/laneC2/predeclaration.md`,
`docs/player_arrests_policy_eval.md`).

## Scope

This lane measures and proposes. It does not change the served model, the
played card, or any file under `artifacts/`. Run outputs are written to the
session scratch directory; every number below is reproduced by the command in
"Verification".

## Measured

Everything below was produced by one command (see "Verification") on the
active model `d49194e04945a5e5`, feature table
`2fb3451b2aa15a6ac1994a35fc7b1e12fb8bcbfd0bc3490d9d51d580f5a6be53`, opener
evaluation `artifacts/opener_evaluation/20260910T211255Z`: 1,537 archived
games, 1,503 non-push, 704 in the fit seasons and 799 in the scored seasons.
The served-probability replay reproduced the archive's own number on every
game to **2.2e-16** (gate 1e-9) before any candidate was built, and the
recovered per-week residual median range [-2.2354, +0.7384] matches the range
lane C2's inversion recorded on the earlier archive to 1e-15. The
walk-forward lattice used 200-641 prior games per read across 107 week
blocks.

Four arms beyond the frozen five appear below. **A0-A4 are the predeclared
set; A5, A6, A7 and A8 were added after A0-A4 had been scored** and are
labelled post-hoc wherever they are quoted (section 9). They exist because
A0-A4 answered the question the predeclaration asked and raised a different
one: every predeclared arm moves picks, and the reliability table says the
defect on the board is the stated CONFIDENCE, which can be fixed without
moving a pick at all.

### 1. The reliability table, at the opener, on the whole archive (2020-2025, 1,503 non-push)

Stated confidence is the average confidence the read puts on the side it
picked; accuracy is how often that side covered.

**Before the served home-side push (the raw model):**

| Spread | Games | Stated | Accuracy | Gap | Brier |
|---|---:|---:|---:|---:|---:|
| 0-3 | 552 | 55.4% | 56.2% | +0.8 | 0.24907 |
| 3.5-6.5 | 552 | 55.8% | 56.2% | +0.3 | 0.24892 |
| 7 | 74 | 56.6% | 52.7% | -3.9 | 0.25002 |
| 7.5-10 | 194 | 56.3% | **48.5%** | **-7.8** | 0.26007 |
| 10.5+ | 131 | 55.5% | **44.3%** | **-11.2** | 0.26350 |
| all | 1,503 | 55.7% | 54.0% | -1.8 | 0.25174 |

This reproduces the owner's 2026-09-07 diagnosis to the decimal on a model
that has been refitted since (56.2 / 56.2 / 52.7 / 48.5 / 44.3, confidence
flat at 55-57% in every bucket). The defect is not an artefact of the model
that was active that day.

**As served today (S3, the home-side push on spreads of seven or more):**

| Spread | Games | Stated | Accuracy | Gap | Brier |
|---|---:|---:|---:|---:|---:|
| 0-3 | 552 | 55.4% | 56.2% | +0.8 | 0.24907 |
| 3.5-6.5 | 552 | 55.8% | 56.2% | +0.3 | 0.24892 |
| 7 | 74 | 56.5% | 51.4% | -5.2 | 0.25015 |
| 7.5-10 | 194 | 56.2% | 51.5% | **-4.6** | 0.26030 |
| 10.5+ | 131 | 55.9% | 47.3% | **-8.6** | 0.26130 |
| all | 1,503 | 55.8% | 54.6% | -1.2 | 0.25158 |

**As played (KL1b, S3 plus the key-line read on lines of exactly 3 or 7):**

| Spread | Games | Stated | Accuracy | Gap | Brier |
|---|---:|---:|---:|---:|---:|
| 0-3 | 552 | 55.4% | 55.6% | +0.2 | 0.24928 |
| 3.5-6.5 | 552 | 55.8% | 56.2% | +0.3 | 0.24892 |
| 7 | 74 | 56.7% | 59.5% | +2.8 | 0.24887 |
| 7.5-10 | 194 | 56.2% | 51.5% | -4.6 | 0.26030 |
| 10.5+ | 131 | 55.9% | 47.3% | -8.6 | 0.26130 |
| all | 1,503 | 55.8% | 54.8% | -1.0 | 0.25160 |

**The finding of this table:** S3 took three points off the 7.5-10 hole and
three off 10.5+ (48.5 -> 51.5, 44.3 -> 47.3) and moved the stated confidence
by a tenth of a point. The location correction is served; the CONFIDENCE has
never been corrected, and that is what is still open. A reader on those games
is being told 56% about a pick that lands 47-51% of the time.

### 2. The correction, learned on 2020-2022

`logit(p) -> a_b + s_b * logit(p)` per bucket, shrunk to the identity with a
100-game prior. `a_b` is the part that can move a pick; `s_b` is confidence
scaling.

| Spread | Fit games | `a_b` raw | `s_b` raw | `a_b` served | `s_b` served |
|---|---:|---:|---:|---:|---:|
| 0-3 | 235 | +0.018 | +0.872 | +0.013 | 0.910 |
| 3.5-6.5 | 262 | -0.019 | +0.456 | -0.014 | 0.606 |
| 7 | 37 | -0.280 | +0.519 | -0.076 | 0.870 |
| 7.5-10 | 97 | +0.006 | -0.306 | +0.003 | 0.357 |
| 10.5+ | 73 | +0.306 | -1.247 | +0.129 | 0.052 |

Two things are readable here, and both are the mechanism rather than a
threshold. The intercept runs POSITIVE in the big buckets (+0.129 at 10.5+,
+0.003 at 7.5-10 after the served home-side push has already taken its share)
-- the same home-side direction lanes L, P, Q and S diagnosed. The slope falls
away with spread size -- 0.91 / 0.61 / 0.87 / 0.36 / 0.05 across the five
buckets, the 7 bucket the one exception and the thinnest cell at 37 games --
and the RAW fit is negative in both buckets above 7.5, which says the model's
own confidence ordering carries little information on big spreads and, on the
fit seasons, pointed the wrong way at 10.5+.

### 3. What each arm scores on 2023-2025 (799 non-push games, 54 week blocks)

Accuracy in points, positive favours the candidate; Brier positive means
better calibrated. Baseline S3 scores 55.069% accuracy / 0.25115 Brier;
baseline KL1b 54.944% / 0.25123.

| Arm | Accuracy | Brier | vs S3 accuracy | `P+` | vs S3 Brier | `P+` | Picks changed |
|---|---:|---:|---|---:|---|---:|---:|
| A0 lattice, no calibration | 52.190% | 0.25173 | -2.879 [-5.861, +0.251] | 0.036 | -0.00058 [-0.00274, +0.00155] | 0.301 | 151 |
| **A1 lattice + tilt and scale** | 53.066% | **0.25060** | -2.003 [-5.025, +1.009] | 0.100 | **+0.00056** [-0.00238, +0.00357] | **0.641** | 154 |
| A2 lattice + tilt only | 52.566% | 0.25188 | -2.503 [-5.429, +0.495] | 0.050 | -0.00073 [-0.00311, +0.00165] | 0.274 | 138 |
| A3 smooth + tilt and scale | 53.442% | 0.25066 | -1.627 [-4.648, +1.389] | 0.145 | +0.00049 [-0.00217, +0.00325] | 0.636 | 135 |
| A4 smooth + tilt only | 53.942% | 0.25144 | -1.126 [-3.125, +0.877] | 0.131 | -0.00028 [-0.00158, +0.00101] | 0.334 | 59 |
| A5 smooth + scale only | 54.819% | 0.25056 | -0.250 [-2.020, +1.529] | 0.394 | +0.00059 [-0.00186, +0.00310] | 0.683 | 58 |
| A6 lattice + scale only | 51.189% | 0.25079 | -3.880 [-7.117, -0.499] | 0.013 | +0.00036 [-0.00262, +0.00339] | 0.596 | 199 |
| **A7 smooth + scale only, never inverting** | **55.069%** | **0.25055** | **0.000** (identity) | 0.500 | **+0.00060** [-0.00178, +0.00303] | **0.691** | **0** |
| **A8 the played read + scale only, never inverting** | 54.944% | 0.25058 | -0.125 [-1.261, +1.003] | 0.415 | +0.00057 [-0.00181, +0.00298] | 0.680 | 23 |

Against the PLAYED baseline (KL1b), which is the decision-relevant one: A1
-1.877 [-4.637, +0.876] `P+` 0.089; **A8 changes 0 of 799 picks, scores the
identical 54.944%, and improves Brier by +0.00065 [-0.00173, +0.00309],
`probability_positive` 0.706.** A7 and A8 are the same correction on two
different reads: A7 is built on S3 and therefore discards the served
key-line override on 23 of the scored games, A8 is built on the read that is
actually played and leaves every pick alone. A8 is the serving form.

### 4. The decomposition that is the actual finding

Comparing each calibrated arm to the SAME read without calibration isolates
what the calibration itself buys:

| Comparison | Accuracy | `P+` | Brier | `P+` | Picks changed |
|---|---|---:|---|---:|---:|
| A1 vs A0 (calibration on the lattice) | **+0.876** [-0.625, +2.387] | **0.871** | **+0.00114** [-0.00066, +0.00298] | **0.895** | 37 |
| A2 vs A0 (tilt only, on the lattice) | +0.375 [-1.261, +1.906] | 0.682 | -0.00015 [-0.00149, +0.00118] | 0.412 | 63 |
| A3 vs S3 (calibration on the smooth read) | -1.627 [-4.648, +1.389] | 0.145 | +0.00049 [-0.00217, +0.00325] | 0.636 | 135 |
| A7 vs S3 (scale only, never inverting) | 0.000 (identity) | 0.500 | +0.00060 [-0.00178, +0.00303] | 0.691 | 0 |
| **A8 vs KL1b (scale only, never inverting, on the played read)** | **0.000** (identity) | 0.500 | **+0.00065** [-0.00173, +0.00309] | **0.706** | **0** |

**The spread-conditional recalibration is an improvement -- on the discrete
lattice.** Given the lattice read, adding the per-bucket tilt and scale is
worth +0.876 accuracy points at `probability_positive` 0.871 and +0.00114
Brier at 0.895 on seasons it never saw. Given the smooth read, the same
correction is worth -1.627 points at 0.145. That is the comparison
`AGENTS.md` requires between a discrete read and a smooth pooled-residual
challenger. On the SIDE the smooth challenger is ahead and is what is served.
On the calibration the discrete read is nominally ahead -- A1's Brier 0.25060
against A3's 0.25066 -- but that is a gap of 6e-5 read off two separate
paired comparisons, not a paired test of A1 against A3, and it decides
nothing on its own.

**What the candidate still loses is the SIDE, and the loss is the lattice,
not the calibration.** A0 -- the lattice with no correction at all -- reads
-2.879 against S3. A1 recovers most of a point of that and is still -2.003.
So a lane cannot get to a better card by calibrating the lattice; the lattice
read itself has to be right first, and on the pool's half-point lines it is
not (lane C2 measured the same thing from the other direction).

### 5. Where it happens, by bucket (A1 against S3, 2023-2025)

| Spread | Games | S3 accuracy | A1 accuracy | Delta | `P+` | Brier delta | `P+` | Picks changed |
|---|---:|---:|---:|---|---:|---|---:|---:|
| 0-3 | 317 | 54.26% | 53.31% | -0.946 [-5.281, +3.268] | 0.336 | +0.00062 | 0.640 | 53 |
| 3.5-6.5 | 290 | 57.59% | 51.72% | **-5.862** [-10.473, -1.119] | 0.007 | -0.00245 | 0.148 | 55 |
| 7 | 37 | 45.95% | 56.76% | **+10.811** [-5.405, +27.778] | 0.902 | -0.00221 | 0.390 | 12 |
| 7.5-10 | 97 | 55.67% | 52.58% | -3.093 [-12.088, +5.455] | 0.245 | +0.00557 | 0.846 | 25 |
| 10.5+ | 58 | 51.72% | 56.90% | **+5.172** [-5.556, +16.129] | 0.817 | +0.00865 | 0.859 | 9 |

The candidate wins exactly the buckets the owner named -- 10.5+ (+5.17,
`P+` 0.817, with the best Brier gain of any cell in the family) and the games
quoted at 7 (+10.81, `P+` 0.902 on 37 games) -- and pays it all back in
3.5-6.5, where the smooth read was already the best-calibrated thing on the
board on those seasons (57.6% right against 55.0% stated). It is the same trade lanes K (K1),
H (M1) and T (T2) reported in the MOD-18 row, reproduced here on a refitted
model with a chronological fit/score split: the key-number structure helps where the line
sits on or beside a mass point and adds noise where it does not.

### 6. A7 and A8: the correction that fixes the page without touching a pick

A7 and A8 are the two pick-preserving arms -- the recalibration scales the
log-odds by a per-bucket factor that is never allowed below 0.05, so it can
never cross 0.5 and can never invert a side. They change the number a reader
sees and nothing else. A7 is fitted and applied to the smooth read (S3), A8
to the read that is played (KL1b). The table below is A7; A8's per-bucket
Brier is the same to four decimals except at 0-3, where it is 1.9e-7 (its
0-3 slope fits at 0.99997, so it leaves that bucket alone), and its 7 bucket
reads +0.00031, `probability_positive` 0.853.

| Spread | Games | Stated before | Stated after | Accuracy | Gap before | Gap after | Brier delta | `P+` |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 0-3 | 317 | 54.6% | 54.7% | 54.3% | -0.4 | -0.5 | -0.00017 | 0.020 |
| 3.5-6.5 | 290 | 55.0% | 52.7% | 57.6% | +2.6 | +4.9 | -0.00072 | 0.354 |
| 7 | 37 | 54.9% | 54.5% | 45.9% | -9.0 | -8.6 | +0.00066 | 0.817 |
| 7.5-10 | 97 | 55.5% | 50.9% | 55.7% | +0.1 | +4.8 | +0.00422 | 0.743 |
| 10.5+ | 58 | 56.8% | 50.3% | 51.7% | -5.1 | +1.4 | +0.00530 | 0.701 |
| all | 799 | 55.0% | 53.2% | 55.1% | +0.0 | +1.9 | +0.00060 | 0.691 |

On the held-out seasons it takes the 10.5+ overstatement from -5.1 points to
+1.4; on the whole archive that bucket is where the served read overstates
itself most (55.9% stated against 47.3% realised, table 1). It costs a little at 3.5-6.5 and
7.5-10, where it now UNDERSTATES by about five points because the fit seasons
put a weak slope on both (0.53 and 0.16), and a sixth of a thousandth of
Brier at 0-3.

### 7. Week 1 2026, read only

The full-archive fit (in-sample, used for this count only) applied to the live
forecast `artifacts/margin_predictions/2026-week-01-20260910T210852Z`
would change **4 of 16 sides** under A1 -- BUF at HOU (0.540 -> 0.484), CHI at
CAR (0.517 -> 0.459), NO at DET (0.531 -> 0.486), TB at CIN (0.506 -> 0.486)
-- every one of them a near coin flip on the served card. Under A7 and A8 it
changes **0 of 16**, by construction. Nothing was regenerated, activated, or written:
the forecast was read, not rebuilt.

### 8. Decision

- **Keep the served read for the PICK.** A1 against the played baseline is
  -1.877, `probability_positive` 0.089: on expected value that is an 8.9/91.1
  bet against the incumbent, so the incumbent is played. Nothing here is
  closed: every one of the 240 cells is recorded `unresolved_below_power`.
- **The spread-conditional recalibration is worth having on the read it was
  built for.** On the lattice it is +0.876 accuracy points at `P+` 0.871 and
  +0.00114 Brier at 0.895 out of sample; the lane that gets the lattice side
  read right on half-point lines should carry this correction with it rather
  than re-deriving it.
- **A8 is the proposal this lane makes.** Fitted on and applied to the read
  that is played, it is free by construction on the side (zero picks changed,
  accuracy identical at 54.944%), it is better calibrated at
  `probability_positive` 0.706 overall, and it closes the largest
  stated-versus-realised gap in table 1 -- 55.9% stated against 47.3%
  realised at 10.5+ on the whole archive, which on the held-out seasons
  becomes 50.3% stated against 51.7% realised. Serving it is a display and
  Brier decision, not a card decision, and it needs the Model page's
  weak-spots table and the displayed-confidence module to agree on one
  policy, which is engineering this lane did not do. A7 is the same
  correction on the smooth read and is reported only so the two are
  comparable.

### 9. Disclosed against it

- The 3.5-6.5 accuracy loss is -5.862 for every lattice arm, because a
  monotone recalibration cannot move a side and KL1b differs from S3 only on
  lines of exactly 3 or 7, which are outside this bucket. It is therefore ONE
  measurement occupying six cells (three lattice arms against two baselines),
  not six votes. Its whole interval sits below zero
  ([-10.473, -1.119] for A1) and it meets the letter of `wrong_sign_resolved`
  for the direction "the lattice read beats the smooth read at 3.5-6.5". It
  is left `unresolved_below_power` here and named for a human look rather
  than closed from a lane report (precedent: lanes G and U in the MOD-18 row).
- 15 of the 240 cells have their whole interval on one side of zero, against
  roughly 12 expected by chance at a 95% band if every cell were null; the
  arms are nested and share games, so the cells are not independent votes.
  Six of the 15 are the one 3.5-6.5 measurement above, and one (A8 at 0-3) is
  an effect of 1.9e-7. The list is reproducible from the run's `cells.json`.
- The slope fits above a spread of seven rest on 37-97 fit games and come out
  negative before shrinkage. The 100-game prior was borrowed from a MEAN
  estimator (lane S's home-side offset) and lane R already measured that it
  is too weak for a SLOPE. A7's floor at 0.05 is the guard that keeps a
  negative fit from inverting a pick, and it was added AFTER A5 was scored
  and found to flip 58 picks on a negative 10.5+ slope -- disclosed as
  post-hoc, as are A5, A6 and A8: only A0-A4 were frozen before scoring, and
  A5-A8 were added after A0-A4 were read. None of them can change a pick
  except A5 and A6, whose accuracy readings are reported and not acted on.
  On Brier the floor is worth 1e-5 against the unfloored A5
  (0.25055 against 0.25056), so the calibration claim does not rest on it;
  what the floor buys is the 58 picks A5 would have flipped with no mechanism
  behind them, which `AGENTS.md` bans outright.
- The archive is the mined one. The chronological split removes the
  correction's own in-sample advantage; it does not make 2023-2025 unseen by
  the project.

## Verification

```powershell
.\.tools\uv.exe run --no-sync python scripts\spread_size_calibration.py --output <run directory>
.\.tools\uv.exe run ruff format --check scripts\spread_size_calibration.py
.\.tools\uv.exe run ruff check scripts\spread_size_calibration.py
```

The script fails closed if the feature table does not match the active model,
if no opener evaluation matches it, or if the served-probability replay misses
the archive by more than 1e-9. Delete `scored.parquet` in the run directory to
force the walk-forward lattice to be refitted from scratch.

## What happened when the serving lane tried to wire A8 (2026-09-10, later the same day)

Measured this session, and it changes the proposal in section 8. **A8 cannot be
served as written, because the displayed confidence is no longer the raw stated
probability A8 was measured against.** Lane AH shipped a displayed-confidence
transform hours earlier (`docs/displayed_confidence.md`): the score beside each
pick is already shrunk, per line-size bucket and stated-probability band, toward
what that cell of the archive really hit. A8 was scored against the raw read; the
question the serving lane actually faces is A8 against that incumbent.

Measured on the active model's own opener archive
`artifacts/opener_evaluation/20260910T211255Z` (1,503 non-push games,
2020-2025, walk-forward for every arm so no game is calibrated with its own
week, week-blocked paired bootstrap, 20,000 draws, seed 20260821):

| candidate | Brier vs the served display | `P+` | log loss vs the served display | `P+` |
|---|---|---:|---|---:|
| A8 alone, replacing it | -0.00049 [-0.00241, +0.00138] | 0.312 | -0.00625 [-0.02008, +0.00236] | 0.150 |
| A8's scale on top of it | -0.00053 [-0.00271, +0.00133] | 0.316 | -0.00921 [-0.03035, +0.00265] | 0.224 |
| the served display on top of A8 | +0.00004 [-0.00163, +0.00160] | 0.531 | -0.00068 [-0.00522, +0.00309] | 0.399 |
| **A8's never-invert guard on it** | **+0.00030** [-0.00013, +0.00074] | **0.912** | **+0.00060** [-0.00027, +0.00148] | **0.912** |

So the decision is not the one section 8 anticipated. Serving A8 in place of the
incumbent is a 31/69 bet on Brier and a 15/85 bet on log loss, and the project's
own rule -- decide on expected value, not on a threshold -- says take the other
side. **What was served instead is the one part of A8 that wins: the guard.**
A8's defining property is that its floored scale can never carry a probability
across 0.5, so it can never advertise the side the card is picking as a loser.
The incumbent has no such property, and on this archive it printed a score below
50% on **179 of 1,503** picks the card would have made. Applying that floor to
the served transform is worth +0.00030 Brier and +0.00060 log loss, both at
`probability_positive` 0.912, and takes the count to 0 of 1,503.

It is not free, and the cost is disclosed rather than buried: the floor fires
exactly in the buckets the owner named, so it raises the mean displayed score
there and widens the bucket-average overstatement -- 7.5-10 from -0.89 to -1.70
points, 10.5+ from -3.73 to -4.77, lines quoted at 7 from -3.64 to -4.28. The
trade is a coarse bucket average against the two proper scoring rules for the
per-game number a reader actually acts on, and both scoring rules favour the
floor at 0.912. A bucket gap can always be driven to zero by under-displaying
everything, which is why it is not the criterion.

Served as `displayed_confidence_reliability_v2_pick_side_floor` in
`src/nfl_ats/displayed_confidence.py`. Two other things went in with it. The
display is now **fitted fail-closed**: if no opener evaluation matches the
active model, or the matched archive names a different model, publishing raises
instead of quietly falling back to the newest archive on disk, so the reader's
number can never come from a model that is not the one being served. And
`attach_displayed_confidence` asserts at runtime that no displayed score lands
below 50% on a side the card picks; the pick itself is still taken from
`home_cover_probability` before any of this runs, so no transform here can move
a side.

On the live card: measured through `nfl-ats publish-predictions`, **0 of 16**
Week 1 scores are floored and all 16 sides are unchanged, because none of this
week's picks lands in a cell that points below 50%. The guard is a live hazard
that did not fire this week, not a change to this week's card.

One caveat on provenance, stated plainly. The archive stores the smooth served
probability, not the key-line read, so the production fit is A7's form (the
smooth read) applied to whatever probability the card carries. Section 3
measures A7 and A8 apart at 6e-5 of Brier, and rebuilding the lattice at publish
time to recover the difference is not worth that; the distinction is recorded
here rather than papered over.

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.
