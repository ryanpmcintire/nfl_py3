# MOD-18 C2: the discrete side read at every line — predeclaration

Frozen 2026-09-08, before any candidate number was computed. Family
`mod18_discrete_side_read_v1`. This document is the predeclaration; the
measured sections at the end were appended after the run and never altered
anything above the "Measured" heading.

## What this lane answers, and why it is not one of the lanes already run

Read: `ROADMAP.md` row MOD-18, last sentence — "**C2 is now the live
candidate and it is OPEN, not tried: price the served cover probability off
the discrete conditional distribution at EVERY half-point line, not only
where the line equals an atom**". Read: `docs/key_line_pick_read.md`, the
section "TODO, predeclared: generalise the side read to every half-point
line (MOD-18 candidate C2)", which states that direction with its numbers
and says it is "Not implemented, not measured, and owned by another lane."

Read, so this lane does not repeat them:

| Lane | Arm | Result through the played card |
|---|---|---|
| K (`docs/mass_preserving_lattice.md`) | MP1, the discrete read on EVERY game | -0.399 pts, `probability_positive` 0.311 |
| T (`docs/key_line_lattice.md`) | KL1b, the discrete read where `abs(line)` is EXACTLY 3 or 7 | +0.200 pts, `probability_positive` 0.7426 — SERVED |
| H (`docs/big_spread_lattice.md`) | M1, lane K's kernel lattice on spreads of 7 or more | -0.13 pts, `probability_positive` 0.401 |

So the two ends of the restriction axis are measured: everywhere loses,
exactly-on-an-atom wins. **Nothing has measured the middle**, and the middle
is the only part of the axis the owner's pool actually quotes: measured
2026-09-08 by lane W on `data/splash/2026_week01_20260908_noon.json`, all
sixteen Week 1 lines are half points and nine of them sit within half a
point of 3. The served exact-match read therefore cannot fire on a single
pool line.

The mechanism this lane tests is the one `docs/key_line_pick_read.md`
already measured on 4,431 completed games and could not act on: **14.58% of
finals land exactly on `abs(3)`, and on a half-point line that entire block
falls on ONE side of the number instead of pushing.** Crossing the 3 atom
from 2.5 to 3.5 the empirical home-cover rate falls 7.81 points where the
smooth read says 2.72, and the two reads err in OPPOSITE directions on the
two sides of the atom. A discrete conditional distribution puts that block
where the football put it; a Gaussian spreads it across a continuum. That is
a named mechanism at a named set of lines, not an accuracy dip located at a
threshold (AGENTS.md, "No unexplained threshold flips on the played card").

## Baselines

Two, both declared here, both reported for every arm:

1. **`S3`** — the served home-side offset plus the smooth `gaussian_median`
   two-way read on every game. This is what the opener archive holds and what
   every MOD-18 lane has been graded against.
2. **`KL1b`** — `S3` with lane T's served exact-atom override on
   `abs(line) in {3, 7}`. This is what is actually PLAYED as of the
   2026-09-08 12:20 lock, so it is the decision-relevant incumbent.

Baseline 1 is the primary for comparability with lanes K/T/H; baseline 2 is
the primary for the DECISION. Both are reported for every arm and every
slice. No arm is selected on one and reported on the other.

## The construction (nothing is re-tuned)

Every arm reads the SAME object lane K built and lane S productionised,
imported, not reimplemented: `nfl_ats.mass_preserving_lattice.band_read` —
the empirical integer-margin atoms of prior games quoted within a band of
the line, atoms at their ABSOLUTE positions, the model's point entering only
as an exponential tilt solved so the tilted mean equals the served point.
Band half-width 2.5 widened in 0.5-point steps to a 200-game floor; the
served point is the offset-corrected centre plus the residual median, the
identical quantity the served push read already tilts to. Walk-forward by
construction: the prior pool for a target week is completed regular-season
games from the trailing five seasons whose gameday plus a one-day completion
allowance precedes the week's earliest gameday, with the target week, every
later week and each target game id excluded before an atom is counted.

The candidate two-way number on a touched game is
`cover + push / 2` (`MassPreservingRead.home_cover_probability`), lane T's
served convention, which at a half-point line is simply `cover` because the
push atom is empty there. Push probability falls out of the same object.

## The arms, frozen

`d` is the distance from `abs(line)` to the nearest declared atom.

| Arm | Touched when | What it adds over the served read |
|---|---|---|
| **G1** | every game | the literal ROADMAP sentence; also a re-measurement of lane K's MP1 under the current active model and feature table |
| **G2** | `d <= 0.5` for atoms `{3, 7}` | the served exact-atom read plus the half points either side of it |
| **G3** | `d <= 0.5` for atoms `{3, 7, 10, 14}` | the same at all four key numbers |
| **G4** | `d == 0.5` for atoms `{3, 7}` (half points only) | DISJOINT from the served read: only lines the pool actually quotes, so it measures the new territory alone |

Atom sets are MOD-05's key numbers, the set AGENTS.md makes binding; the
`{3, 7}` restriction is lane T's served set, inherited, not re-chosen. The
half-width 0.5 is the only half-point step a spread can take, so it is the
smallest non-empty neighbourhood and is not a tuned radius. No other radius
is tried.

Ranking, fixed before scoring: report all four; there is no selection step
and no arm is dropped. The coordinator decides on expected value through the
played card.

## Metric and protocol

- Forced-pick ATS accuracy at the **OPENER**, week-blocked, paired per game,
  on the archived Tuesday-opener evaluation matching the active model
  (1,537 games, 2020-2025; non-push games are the graded set).
- Two reads for every arm and slice: **standalone** (the probability rule on
  the raw arm) and **through the played card** (the frozen three-member
  overlay union — coach fade, division revenge, player arrests — recomputed
  against each incoming card, membership frozen before scoring, no subset
  search). The card read is the decision read.
- Block bootstrap over (season, week) blocks, 20,000 draws, seed 20260817.
  Within-week game correlation is ZERO by owner mandate; no variance
  inflation and no ICC is estimated.
- `probability_positive` is reported for every cell. The binary "the
  interval contains zero" is never reported and is never a verdict.
- Slices, declared here: overall; per season; the games each arm actually
  touches; and — this is grading question 2 from
  `docs/key_line_pick_read.md` — by `abs(line)` band, separately for the
  half points adjacent to 3 (`abs(line) in {2.5, 3.5}`), the atom itself
  (`abs(line) == 3`), the same pair at 7, and everything else. The mechanism
  predicts the gain concentrates in the adjacent half points and at the
  atoms; a gain spread flat across the line range is a different effect
  wearing this one's name and will be reported as such.
- Brier and log loss are reported alongside accuracy, as calibration, never
  as a veto on the side read.

## Positive control

Run on the same harness, same blocks, same seed, to prove it can detect an
effect of the size at stake (AGENTS.md: only a proven-able control bounds a
null).

- **PC-full**: on G2's touched games the candidate probability is replaced by
  the realised outcome (1.0 if the home side covered, 0.0 if not) — unit
  slope, zero noise, the same leak shape `nfl_ats.totals_wave2`'s control and
  MOD-17's control use. Predeclared expected shape: a large positive delta
  with `probability_positive` at or near 1.
- **PC-small**: the same leak applied to a fixed pseudo-random subset of
  G2's touched games, sized so the injected effect is about **+1.0 accuracy
  point overall** — the scale of the arms themselves. Subset drawn with
  `numpy.random.default_rng(20260817)` before scoring. Predeclared purpose:
  if PC-small resolves (interval clear of zero) the instrument has power at
  one accuracy point on this archive; if it does not, then no null this lane
  produces at that scale can be bounded by a control, and every arm stays
  `unresolved_below_power` by definition.

A positive control that behaves as predeclared does NOT close any arm. It
only establishes the resolution at which a null could ever be read as
absence.

## Week 1 2026

Read-only: the arms are applied to the active forecast's own served
probabilities and the resulting card sides compared, both raw and through
the three-member union. No forecast is regenerated, nothing is activated,
`artifacts/active_ats_model.json` is not touched, and the served card is not
changed by this lane.

## Reuse discount, disclosed before the run

This is the same mined 1,537-game archive lanes K, T, H, S and V were
selected on. S3 is itself a post-hoc restriction fitted on it; lane T's atom
set was chosen after seeing lane K's bucket-7 gain on it; the 2.5/3.5
asymmetry that motivates this lane was measured on the same games before
this arm was declared. Four correlated, nested arms (G4 ⊂ G2 ⊂ G3 ⊂ G1) add
selection. These are descriptive reused-era measurements, not independent
confirmation. No numerical discount is invented and **no rotation window is
spent**: precedent `docs/player_arrests_policy_eval.md` and MOD-17, both of
which graded a promotion-style look on this archive without a
`rotation record-look` entry.

## Leakage gates

- The prior pool for a week can contain no game from that week or later, and
  never the target game itself; asserted in tests, not only by construction.
- The archive replay must reproduce the served `S3` probability to 1e-9
  before any candidate is built; the run fails closed above that.
- An untouched game's probability must equal the baseline exactly (0.0 gap),
  asserted per arm.
- Feature-table digest must equal the active model's before anything runs.

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero". The registry
code hard-rejects inadmissible closures; if a record command errors, the
verdict is wrong, not the validator. Never use 95%, 0.90, or any threshold as
a DECISION bar; decide on expected value (`probability_positive` above 0.5
favours playing it), thresholds only govern what docs may CLAIM. Grade at the
OPENER (the pool's grade); a close-graded number may never veto a play.
Within-week game correlation is ZERO by owner mandate: never estimate or pad
it. Never say something "needs N more games": the data is fixed and the
project is model-limited.

## Where the code lives

- `src/nfl_ats/discrete_margin_mapping.py` — the neighbourhood predicate, the
  per-game read, the arm application, and the walk-forward frame helper. It
  imports lane K's construction; it does not reimplement it.
- `src/nfl_ats/calibration.py` — two additive `probability_method` values,
  `discrete_conditional_lattice` and `discrete_conditional_key_neighbourhood`,
  at the same plug-in point `gaussian_median` uses. The incumbent methods are
  untouched and byte-identical.
- `scripts/discrete_side_read_opener_eval.py` — the frozen research run.
- `tests/test_discrete_margin_mapping.py` — leakage, invariants, and a
  reproduction of lane T's served exact-atom read as a special case.

---

# Measured (2026-09-08, after the predeclaration above was frozen)

Everything above this line was written, and its bytes digested into
`artifacts/research/laneC2/reproduction.json` (`predeclaration_sha256`),
before a single candidate number existed.

Run: `scripts/discrete_side_read_opener_eval.py --stage replay|map|score|week1|record`.
Archive: `artifacts/opener_evaluation/20260908T184350Z` (active model
`c526cf6636cef6f8`, feature table `5377ca14...`), 1,537 games, 1,503 non-push,
2020-2025, 107 (season, week) blocks.

## The decision first

**Do not play any of the four arms.** Through the played three-member card
every arm's expected value is negative against both declared baselines, and
the best of them -- G2 -- is 41% likely to beat the smooth-everywhere read and
25% likely to beat the card that is actually played. Declining a candidate at
`probability_positive` 0.25-0.41 is the right side of that bet, which is the
rule (AGENTS.md, "A promotion bar is not a decision bar", read the other way
round). The served read stays exactly as it is: the S3 home-side offset, the
smooth `gaussian_median` two-way probability, and lane T's exact-atom override
on lines of exactly 3 and 7.

| Arm | Through the played card, vs `S3` | vs `KL1b` (played) | Standalone vs `S3` |
|---|---|---|---|
| G1 every line | **-0.399** [-2.088, +1.318] P+ **0.324** | -0.599 [-2.116, +0.927] P+ 0.222 | -1.131 [-3.098, +0.867] P+ 0.133 |
| G2 `{3,7}` plus or minus a half point | **-0.133** [-1.265, +1.002] P+ **0.409** | -0.333 [-1.318, +0.607] P+ 0.254 | -0.732 [-2.079, +0.665] P+ 0.150 |
| G3 four keys plus or minus a half point | -0.333 [-1.584, +0.939] P+ 0.304 | -0.532 [-1.636, +0.595] P+ 0.176 | -0.865 [-2.310, +0.657] P+ 0.126 |
| G4 half points only | -0.333 [-1.296, +0.602] P+ 0.248 | -0.532 [-1.582, +0.471] P+ 0.155 | -0.931 [-2.101, +0.266] P+ 0.060 |

Card accuracies on the 1,503 non-push games: `S3` 55.89%, `KL1b` **56.09%**,
G1 55.49%, G2 55.76%, G3 55.56%, G4 55.56%. Standalone: `S3` 54.56%,
`KL1b` 54.76%, G1 53.43%, G2 53.83%, G3 53.69%, G4 53.63%.

Nothing is closed. Every one of the 212 recorded cells is
`unresolved_below_power` with no closing ground.

## Two cross-checks that make the numbers usable

Measured, not asserted:

- **The S3 baseline replays to 1.1e-16.** The week's fitted residual median
  and standard deviation are recovered algebraically from the archive's own
  served probability and `residual_at_open_served` (two unknowns, one exact
  linear relation, many games per week) instead of by ~110 ridge refits; the
  run refuses to continue above 1e-9 and lands eleven orders of magnitude
  inside that. `reproduction.json`.
- **Lane K and lane T both reproduce under the new active model.** G1 is
  lane K's MP1 construction served on every game and scores **-0.399** points
  through the card at `probability_positive` 0.324 against lane K's recorded
  -0.399 at 0.311 -- the same point estimate on a different active model
  (`c526cf6636cef6f8`, not `a4c757efd2525da6`) and a feature table rebuilt the
  same day. The rebuilt `KL1b` baseline touches **272 of 1,537** games, lane
  T's exact count, and lifts the card from 55.89% to 56.09%, lane T's exact
  pair.

## Grading question 2, answered -- and the answer is the opposite of the prediction

The predeclaration named the prediction: the gain should concentrate in the
2.5/3.5 band, because that is where the 14.58% block of finals on a margin of
exactly 3 falls wholly on one side of the number. **It does not.** The
discrete read's whole gain sits ON the atoms and it loses on the half points
either side of them.

| Slice | n (non-push) | Smooth `S3` right | Lattice right | Standalone delta |
|---|---:|---:|---:|---|
| line exactly 7 | 70 | 52.86% | **61.43%** | **+8.571** [-1.449, +19.403] P+ **0.949** |
| line exactly 3 | 179 | 53.07% | 51.40% | -1.676 [-5.979, +2.488] P+ 0.226 |
| line 2.5 or 3.5 | 342 | 56.14% | 53.80% | **-2.339** [-6.742, +1.983] P+ 0.145 |
| line 6.5 or 7.5 | 156 | 56.41% | 52.56% | **-3.846** [-10.417, +2.454] P+ 0.117 |
| away from 3 and 7 | 745 | 53.96% | 53.15% | -0.805 [-3.668, +2.016] P+ 0.290 |

Through the card: line 3 **+0.559** P+ 0.612, line 7 **+2.857** P+ 0.792,
2.5/3.5 -0.292 P+ 0.436, 6.5/7.5 **-2.564** P+ 0.173, away -0.537 P+ 0.325.

**The mass is where it should be; the side decision is not.** At lines near 3
the lattice puts **15.5%** of its modelled mass on a final margin of exactly
3 either way, against the 14.58% the football actually produces (the smooth
read puts about 3.5% there -- `docs/discrete_push_read.md`), so the owner's
binding rule is honoured by construction. On the 2.5/3.5 games the lattice's
MEAN predicted home-cover probability is 47.20% against a realised 44.74%,
closer to the truth than the smooth read's 48.31%. But the per-game Brier on
that slice is slightly WORSE (0.24835 against 0.24745), and where the
correction crosses 0.5 it moves about two more picks per hundred to the away
side and those picks land wrong.

Read plainly: **the key-number block matters to the side only when the line
sits ON the number.** There the block is the push, and the split either side
of it is decided by mass a Gaussian spreads over a continuum, so the lattice
wins the games outright (the 7 atom, +8.571 standalone). One half point away
the block sits entirely on one side by arithmetic -- both reads already agree
on WHICH side -- so all that is left for the lattice to contribute is a
band-sampled estimate of how big the block is, which is noise at these sample
sizes. That is a mechanism, not a threshold, and its boundary is the same one
lane T's served rule already uses.

## Positive control: what this harness can and cannot resolve

| Control | Injected | Standalone | Through the card |
|---|---|---|---|
| PC-full (truth on all 758 touched non-push games) | -- | **+22.62** [+20.55, +24.77] P+ 1.0 | +14.44 [+12.60, +16.36] P+ 1.0 |
| PC-small (truth on 15 games, sized to +1.0 point) | +1.00 | **+0.998** [+0.535, +1.477] P+ 1.0 | +0.532 [+0.200, +0.925] P+ 0.9999 |

PC-small recovers the injected effect to within 0.002 accuracy points and
resolves it clearly, so the harness has power at one accuracy point **when
the effect arrives as a handful of picks all moving the right way**: its
standard error is 0.24. The arms move 78-217 picks in BOTH directions and
carry standard errors of 0.96-1.40, four to six times larger. So the control
proves the instrument works and **does not bound any arm**: no arm may be
classified `bounded_by_control`, and every cell stays
`unresolved_below_power`. This is exactly the distinction the taxonomy asks
for, and it is why "the interval contains zero" would have been the wrong
thing to say about any of these numbers.

## Calibration

Overall Brier / log-loss improvement against `S3` (positive favours the
candidate): G1 **+0.000080** / +0.000157 (P+ 0.543), G2 -0.000345 /
-0.000719 (0.283), G3 -0.000324 / -0.000680 (0.306), G4 -0.000298 /
-0.000622 (0.299). All four are within 0.0004 Brier of the incumbent; this is
calibration noise, and it never vetoes a side read.

## Per-season, and the thirteen cells whose interval clears zero

Every arm is carried by 2021 and paid for in 2025. G2 against the played
card: 2020 0.000 (P+ 0.521), 2021 **+1.695** [0.000, +3.524] (0.962),
2022 -0.403 (0.332), 2023 -0.376 (0.378), 2024 -0.376 (0.411),
2025 **-2.247** [-4.461, -0.370] (0.014).

Thirty-one of the 212 recorded cells have an interval clear of zero;
eighteen are positive-control cells, as predeclared. The remaining **thirteen
are all season cells, twelve of them 2025**, and they meet the LETTER of
`wrong_sign_resolved` for the direction "this arm beats this baseline in
2025". They are recorded `unresolved_below_power` and named here for a human
look rather than closed from a lane report -- the precedent is lane U
(`docs/residual_slope_shrunk.md`), the primary metric is the overall card
read, and thirteen of two hundred correlated cells is only modestly above
what a nominal 5% two-sided rate produces by chance. The names:
`ds_g1_vs_kl1b_season_2025_card`, `ds_g1_vs_s3_season_2023_standalone`,
`ds_g1_vs_s3_season_2025_card`, `ds_g1_vs_s3_season_2025_standalone`,
`ds_g2_vs_kl1b_season_2025_card`, `ds_g2_vs_kl1b_season_2025_standalone`,
`ds_g2_vs_s3_season_2025_card`, `ds_g2_vs_s3_season_2025_standalone`,
`ds_g3_vs_kl1b_season_2025_card`, `ds_g3_vs_kl1b_season_2025_standalone`,
`ds_g3_vs_s3_season_2025_card`, `ds_g3_vs_s3_season_2025_standalone`,
`ds_g4_vs_s3_season_2025_standalone`.

## Week 1 2026

Read-only, against the served forecast
`artifacts/margin_predictions/2026-week-01-20260908T183944Z` (sidecar replay
gap 1.0e-15). The lines are the pool's own, every one a half point.

- **The served exact-atom read touches 0 of 16 games**, confirming
  `docs/key_line_pick_read.md`'s `inapplicable` status from the other side.
- G2, G3 and G4 touch 10-11 of the 16 and would change **three** card sides,
  all of them from the home side to the away side: **CHI at CAR** (-2.5,
  0.5164 to 0.4358), **NO at DET** (6.5, 0.5225 to 0.4737) and **TB at CIN**
  (3.5, 0.5089 to 0.4839).
- G1 touches all 16 and would change four, adding **BUF at HOU** (-1.5,
  0.5418 to 0.4818).

None of this is applied. `artifacts/active_ats_model.json` is untouched and
the served card is unchanged by this lane.

## Invariants and gates that passed

- Untouched games are bit-identical to the baseline: maximum probability gap
  **0.0** and zero card disagreements for G2 (756 untouched), G3 (630) and
  G4 (1,039). `artifacts/research/laneC2/invariants.json`.
- Leakage: the target week, every later week and the target game itself are
  excluded inside the window helper; `tests/test_discrete_margin_mapping.py`
  poisons the target week and every later week with a 60-point margin and
  asserts the read is unchanged frame for frame.
- The feature-table digest matched the active model before anything ran.
- 24 comparison cells were degenerate (no graded pick moved) and were skipped
  rather than recorded as a bootstrap of zeros; the list is in
  `artifacts/research/laneC2/skipped.json`.

## Recorded

212 rows under family `mod18_discrete_side_read_v1`, every one
`unresolved_below_power`, no closing ground, each with a plain-English
summary, written serially through `nfl-ats weak-signals record` (registry
4,143 to 4,355 rows, zero failures). No rotation window spent -- this is a
promotion-style look on the reused Tuesday-opener archive, the precedent
`docs/player_arrests_policy_eval.md` and MOD-17 both set.

## What this leaves open

The generalisation is answered and it is negative, but the diagnosis it
produced is a live direction: the discrete read's value is concentrated on
the **7 atom** (+8.571 accuracy points standalone on 70 games,
`probability_positive` 0.949, replicating under a new active model), and the
served rule that exploits it **cannot fire on a single line the owner's pool
posts**. The gap is not in the mapping -- it is that the archive's decision
line (nflverse `spread_line`, whole numbers) and the pool's decision line
(always a half point) are different objects. Two directions follow, neither
of them this lane's arms:

1. Grade the whole MOD-18 mapping family on a **pool-shaped** archive -- the
   captured Splash lines as they accumulate -- rather than on whole-number
   history, so an arm's measured domain is the domain it would serve in.
2. Ask whether the 7-atom gain survives on lines the pool would quote near 7
   once the 2026 rows exist; the paired challenger
   `key_line_pick_read_off_incumbent` already accrues them at no window cost.
