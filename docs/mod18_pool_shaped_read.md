# MOD-18: the discrete mapping family graded on a pool-shaped archive

Frozen 2026-09-11, before any candidate number was computed. Family
`mod18_pool_shaped_read_v1`. Everything above the "Measured" heading was
written first and its bytes are digested into
`artifacts/research/laneP1/reproduction.json` (`predeclaration_sha256`).

## The gap this lane closes

Read: `docs/mod18_discrete_margin_mapping.md`, section "What this leaves
open", direction 1 -- grade the whole MOD-18 mapping family on a
**pool-shaped** archive rather than on whole-number history, "so an arm's
measured domain is the domain it would serve in". Read: the same doc's
diagnosis that the served exact-atom rule KL1b (`abs(line)` exactly 3 or 7)
"cannot fire on a single line the owner's pool posts", and
`src/nfl_ats/key_line_pick_read.py`'s `KeyLineApplicability.reason`, which
reports the same thing from the forecast side.

The owner's pool always quotes a half point. So the decision-relevant
question is not "does the discrete read beat the smooth read somewhere in
history" but "does it beat it **on half-point lines**", which is the only
kind of line a card is ever submitted against.

## The archive, and a correction to the premise

Measured 2026-09-11 on
`artifacts/opener_evaluation/20260910T211255Z/per_game.parquet` (active model
`d49194e04945a5e5`, confirmed against `artifacts/active_ats_model.json` via
`nfl_ats.public_board.find_matching_opener_evaluation`): of the 1,537 archive
games, **842 (54.78%) have a `tue_open_home_spread` that is exactly a half
point**, and the column takes quarter-point values (0.75, 1.25, 2.25, 3.25,
...) as well as whole numbers and half points.

That is a **correction to the premise of direction 1**. The archive's
decision line is not the nflverse whole-number `spread_line`: it is the
Tuesday-opener consensus across the captured books (`opener_books`, nine
books on the rows inspected), built by `nfl_ats.clv` from the game-market
snapshots, and more than half of it is already the shape the pool quotes.
The pool-shaped archive direction 1 asked for therefore **already exists**
inside the archive every MOD-18 lane has been graded on; it had never been
sliced out and graded as its own domain. This lane does exactly that, and
does not need a substitute archive.

The one respect in which the consensus is not literally a pool line: a
consensus can land on a quarter point, which no book posts. Those rows are
outside the pool-shaped domain by construction and are reported only as a
contrast slice, never as the lane's answer.

### What was searched, and what was not used

Measured 2026-09-11 by survey of the repository. A per-book, point-in-time
opener archive also exists and is larger:
`data/market/raw/<snapshot>/quotes.parquet`, 8,797 snapshots and 6.15M quote
rows (gitignored, local only), whose manifests carry
`request.decision_label` in `{true_open, tue_open, ...}`. Its book-level
`home_spread_line` sits on a clean 0.5 grid and is a half point on 47-61% of
opener quotes per season (2020-2025, 53,551 HOME spread quote rows across 24
books), which is the same shape as the consensus. That archive is **not**
used here: the served probability, the home-side offset, `margin_vs_open` and
every graded outcome in the opener evaluation are all defined against the
CONSENSUS opener, so regrading at one book's posted line would require
refitting the model at that line and would break comparability with every
other MOD-18 lane. Grading the family at a single book's posted half point is
a separate direction, named in this doc's closing section, not this lane's
arms.

Also searched and not usable for 2020-2025:
`data/processed/sbr_odds.parquet` (a labelled opener with no capture
timestamp, seasons 2007-2021 only, so it stops before 2022) and
`artifacts/vegasinsider_backfill/` (seasons 2005-2016). The files named
"half lines" throughout the repo (`docs/half_game_markets.md`,
`docs/vi_half_lines.md`, LEAD-02 / LEAD-61) are first-half and second-half
PERIOD markets, not half-point spread precision, and are unrelated to this
lane.

Finally, a second correction to the premise: the nflverse `spread_line`
column is **not** whole-number either. Measured on
`data/raw/20260908T162105Z/schedules.parquet`, 877 of 1,615 regular-season
2020-2025 games (54.30%) carry a half-point `spread_line`. Its actual defect
for this purpose is timing, not granularity: its mean absolute gap to the
captured Tuesday opener is 0.998 points and it matches exactly on 24.7% of
the paired games, while against the captured close it is 0.196 points and
68.1%. It behaves as a closing line.

## Domains, frozen

`POOL` is the lane's primary domain. `d` is the distance from `abs(line)` to
the nearest declared atom.

| Domain | Definition |
|---|---|
| **POOL** | every archive game whose opener line is exactly a half point -- the shape the pool always quotes |
| **POOL_NEAR** | `POOL` and `d <= 0.5` for atoms `{3, 7}`; equivalently `abs(line)` in `{2.5, 3.5, 6.5, 7.5}` |
| **POOL_NEAR_3** | `abs(line)` in `{2.5, 3.5}` |
| **POOL_NEAR_7** | `abs(line)` in `{6.5, 7.5}` |
| **POOL_AWAY** | `POOL` minus `POOL_NEAR` |
| **OFF_POOL** | the complement of `POOL` -- whole and quarter points, reported for contrast only |

Per-season cells are cut inside `POOL` only. `POOL` and `POOL_NEAR` are the
two the task names as (a) and (b); the other four are reported so the answer
cannot be quoted outside its evidence domain.

## The arms, imported not reimplemented

Every arm is `nfl_ats.discrete_margin_mapping.ARMS`, applied through
`apply_arm`, whose candidate probability comes from
`nfl_ats.mass_preserving_lattice.band_read` -- lane K's construction, the
same core the served push read and the line sweep already call. Nothing is
re-tuned: band half-width 2.5 widened in 0.5-point steps to a 200-game floor,
five trailing seasons of strictly prior completed games, the target week and
every later week and the target game itself excluded, the model entering only
as an exponential tilt solved so the tilted mean equals the served point, and
the two-way read `cover + push / 2`.

| Arm | Touched when |
|---|---|
| **G1** | every game |
| **G2** | `d <= 0.5` for atoms `{3, 7}` |
| **G3** | `d <= 0.5` for atoms `{3, 7, 10, 14}` |
| **G4** | `d == 0.5` for atoms `{3, 7}` (half points only) |
| **KL1b** | `abs(line)` exactly 3 or 7 -- the SERVED rule, carried as a baseline |

**Predeclared structural prediction, to be verified and not assumed.** On
`POOL` the exact-atom rule KL1b can touch nothing, because no half point is a
whole number; and on `POOL` the arms G2 and G4 must have identical touch sets,
because `d <= 0.5` and a half-point line force `d == 0.5`. Both are asserted
by the run and reported as measured invariants. If either fails the run stops
before scoring.

## Baselines

1. **`S3`** -- the served home-side offset plus the smooth `gaussian_median`
   read on every game; the archive's own `home_cover_probability_at_open`.
2. **`KL1b`** -- `S3` with the served exact-atom override, the card that is
   actually played.

Both are reported for every arm and every domain. On `POOL` they are expected
to be identical; that identity is measured, not assumed, and it is the whole
reason this domain is the honest one for the decision.

## Metric and protocol

- Forced-pick ATS accuracy at the **OPENER**, paired per game, week-blocked
  block bootstrap over (season, week) blocks, 20,000 draws, seed 20260817.
  Within-week game correlation is ZERO by owner mandate: never estimated,
  never padded.
- Two reads per arm and domain: **standalone** (the probability rule on the
  raw arm) and **through the played card** (the frozen three-member overlay
  union -- coach fade, division revenge, player arrests -- recomputed against
  each incoming card, membership frozen before scoring, no subset search).
  The card read is the decision read.
- `probability_positive` is reported for every cell. The binary "the interval
  contains zero" is never reported and is never a verdict.
- Brier and log loss are reported alongside accuracy, as calibration, never
  as a veto on the side read.
- Arms are applied on the full 1,537-game frame and the card is composed on
  the full frame; the **scoring** is then restricted to each domain. This
  keeps the overlay membership identical across arms and makes every cell a
  paired within-game comparison.

## Positive control

Same harness, same blocks, same seed.

- **PC-full**: on the games G4 touches inside `POOL`, the candidate
  probability is replaced by the realised outcome. Predeclared expected
  shape: a large positive delta with `probability_positive` at or near 1.
- **PC-small**: the same leak on a fixed pseudo-random subset sized so the
  injected effect is about **+1.0 accuracy point on `POOL`** -- the scale of
  the arms themselves. Subset drawn with `numpy.random.default_rng(20260817)`
  before scoring.

A control behaving as predeclared does NOT close any arm. It establishes only
the resolution at which a null could ever be read as absence.

## Decision rule, declared before the numbers are seen

Through-card `probability_positive` above 0.5 on `POOL` favours the arm, and
that reading is reported before any limitation. The pool is forced picks, so
declining a candidate more likely than not better is taking the other side of
the bet (AGENTS.md, "A promotion bar is not a decision bar"). No threshold --
0.5 included -- is used as a bar on what may be MEASURED or RECORDED; 0.5 is
the expected-value line for the action.

The action, declared here: an arm whose through-card `probability_positive`
on `POOL` exceeds 0.5 is registered as an **ACTIVE_PROSPECTIVE paired
challenger** in `artifacts/prospective/challengers.json`, following
`key_line_pick_read_off_incumbent`. This lane does **not** change the served
card, the active model, or any published forecast.

## Leakage gates

- The prior pool for a week can contain no game from that week or later, and
  never the target game itself; enforced inside
  `nfl_ats.mass_preserving_lattice.prior_pool_for_week`.
- The archive replay must reproduce the served `S3` probability to 1e-9
  before any candidate is built; the run fails closed above that.
- An untouched game's probability must equal the baseline exactly (0.0 gap),
  asserted per arm.
- The feature-table digest must equal the active model's before anything runs.

## Reuse discount, disclosed before the run

This is the same mined 1,537-game archive lanes K, T, H, S, V and C2 were
selected on, and `POOL` is a subset of it. `S3` is itself a post-hoc
restriction fitted on that archive; the atom set `{3, 7}` is lane T's, chosen
after seeing lane K's bucket-7 gain on these same games; the arms are lane
C2's, already scored once on the full archive. The domain restriction is new
and was named by the predecessor doc rather than chosen from a result, but
the arms and the atoms were not.

These are descriptive reused-era measurements, not independent confirmation.
No numerical discount is invented and **no rotation window is spent**:
precedent `docs/player_arrests_policy_eval.md` and MOD-17, both of which
graded a promotion-style look on this archive without a
`rotation record-look` entry. The 2026 rows accruing through the registered
challenger are the evidence that settles it at no window cost.

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero". The registry
code hard-rejects inadmissible closures; if a record command errors, the
verdict is wrong, not the validator. Verdicts flow through
`nfl-ats weak-signals record` / `nfl-ats rotation record-look`, never through
prose in a doc. Never compute or state that something needs N more games: the
data is fixed and the project is model-limited. Never use 95% or 0.90 as a
DECISION bar; decide on expected value (`probability_positive` above 0.5
favours playing). Within-week game correlation is ZERO by owner mandate;
never estimate or pad it. Grade at the OPENER; a close-graded number may
never veto a play.

## Where the code lives

- `src/nfl_ats/discrete_margin_mapping.py` -- the arms and the neighbourhood
  predicate, imported.
- `src/nfl_ats/mass_preserving_lattice.py` -- the walk-forward discrete read,
  imported.
- `scripts/mod18_pool_shaped_read_eval.py` -- this lane's run.
- `artifacts/research/laneP1/` -- its artifacts.

## What this lane will leave open, named now

Grading the family at a SINGLE BOOK's posted half point, rather than at the
cross-book consensus, using `data/market/raw/*/quotes.parquet` at
`decision_label` `tue_open`. That requires refitting the residual, the
home-side offset and the graded outcome at that book's line, so it is a new
opener evaluation rather than a slice of this one. It is the only way to
measure the family at literally the number a pool posts, and it is not this
lane.

---

# Measured (2026-09-11, after the predeclaration above was frozen)

Everything above this line was written first; its bytes are digested into
`artifacts/research/laneP1/reproduction.json` as `predeclaration_sha256`
`999f72ceb8a0aeb09b4187c32ea33276ee4872aad5e54b44f62b446a75b7273a`.

Run: `scripts/mod18_pool_shaped_read_eval.py --stage replay|map|score|week1|record`.
Archive `artifacts/opener_evaluation/20260910T211255Z`, active model
`d49194e04945a5e5`, feature table `2fb3451b...`, 1,537 games over 107
(season, week) blocks, of which **842 are the pool-shaped `POOL` domain**.

## The decision first

**Do not play any arm on the pool-shaped domain, and register none of them.**
Through the played three-member card, on the 842 half-point-line games, every
arm is negative against the served read and the best of them is 25% likely to
be better:

| Arm | POOL, through the played card | POOL, standalone |
|---|---|---|
| G1 every line | **-0.713** [-3.012, +1.659] P+ **0.273** | -2.019 [-4.718, +0.806] P+ 0.077 |
| G2 3 and 7 plus or minus a half point | **-0.594** [-2.334, +1.066] P+ **0.248** | -1.663 [-3.781, +0.466] P+ 0.060 |
| G3 four keys plus or minus a half point | -0.831 [-2.735, +1.075] P+ 0.199 | -1.781 [-4.069, +0.482] P+ 0.065 |
| G4 half points only | **-0.594** [-2.334, +1.066] P+ **0.248** | -1.663 [-3.781, +0.466] P+ 0.060 |

Declining a candidate at `probability_positive` 0.20-0.27 is taking the 73-80%
side of the bet, which is the expected-value action the forced-pick pool asks
for. The predeclared registration rule (through-card P+ on `POOL` above 0.5)
is not met by any arm, so **no challenger was registered**.

Card accuracy on the 842 pool games: `S3` and `KL1b` both **55.11%**, G2 and
G4 54.51%, G1 54.39%, G3 54.28%. Standalone: `S3` and `KL1b` both 56.53%,
G2/G4 54.87%, G3 54.75%, G1 54.51%.

Nothing is closed. All **234 recorded cells** are `unresolved_below_power`
with no closing ground.

## The structural invariants, measured rather than assumed

All three predeclared invariants hold (`mapping.json`, `invariants.json`):

- **The served exact-atom rule KL1b touches 0 of the 842 pool games.** It
  fires on 272 archive games, every one of them a whole number. The rule that
  is played today cannot reach a single line the pool posts.
- **`KL1b` is bit-identical to `S3` on `POOL`**, in probability and in the
  composed card. That is why the two baseline columns above agree exactly, and
  it is the honest form of the claim `docs/mod18_discrete_margin_mapping.md`
  made from the other side.
- **G2 and G4 have identical touch sets on `POOL`** (498 games), because a
  half-point line at distance at most 0.5 from an atom is at distance exactly
  0.5. Their cells are therefore identical throughout, as they should be.

Untouched games are bit-identical to the baseline: maximum probability gap
**0.0** and zero card disagreements for G2 (756 untouched), G3 (630) and G4
(1,039). The `S3` replay gate landed at **1.11e-16**, sixteen orders inside
its 1e-9 refusal threshold.

## The mechanism does not cross the half point

This is the lane's substantive finding. `docs/key_line_lattice.md` measured
the discrete read's entire gain at the line of exactly 7: **+8.571** accuracy
points standalone on 70 games, `probability_positive` 0.934. One half point
either side of that atom -- the only place the pool ever quotes -- the same
read goes the other way.

| Domain | n | `S3` standalone / card % | arm standalone / card % | Standalone delta (P+) | Card delta (P+) |
|---|---:|---|---|---|---|
| `POOL_NEAR_7` (6.5, 7.5) | 156 | 56.41 / 58.97 | 52.56 / 56.41 | **-3.846** [-10.417, +2.454] 0.117 | **-2.564** [-8.054, +2.532] 0.173 |
| `POOL_NEAR_3` (2.5, 3.5) | 342 | 56.14 / 52.92 | 53.80 / 52.63 | -2.339 [-6.742, +1.983] 0.145 | -0.292 [-3.922, +3.297] 0.436 |
| `POOL_NEAR` (both) | 498 | 56.22 / 54.82 | 53.41 / 53.82 | -2.811 [-6.369, +0.787] 0.060 | -1.004 [-3.922, +1.815] 0.248 |
| `POOL_AWAY` | 344 | 56.98 / 55.52 | G1 56.10 / 55.23 | -0.872 [-4.913, +3.116] 0.336 | -0.291 [-3.715, +3.257] 0.434 |

Read plainly, and it is the same reading lane C2 reached from the other
direction: **the key-number block decides the SIDE only when the line sits ON
the number.** There the block is the push, and the split either side of it is
mass a Gaussian spreads over a continuum, so the discrete read wins the games
outright. One half point away the block falls wholly on one side by
arithmetic, both reads already agree on WHICH side, and all the lattice adds
is a band-sampled estimate of how big the block is -- noise at these sample
sizes.

The consequence for the pool is blunt: the part of the MOD-18 family that
works lives on lines the pool never posts, and on the lines it does post the
family is a small loss.

## Where the family's value actually sits

Measured on `OFF_POOL` (the 661 whole- and quarter-point games), `KL1b` beats
`S3`: 52.50% against 52.04% standalone and 57.34% against 56.88% through the
card. G2's gain there (+0.454 card, P+ 0.758) is **the same picks**:
`G2 vs KL1b` on `OFF_POOL` is +0.000 with 0 flips. So the entire measured
value of the discrete mapping family on this archive is the exact-atom rule
that is already served, and it is confined to the domain the pool never
quotes.

## Positive control: what this harness can and cannot resolve

| Control | Injected | POOL standalone | POOL through the card |
|---|---|---|---|
| PC-full (truth on the 498 games G4 touches in `POOL`) | -- | **+25.89** [+22.80, +28.99] P+ 1.0 | +17.70 [+15.04, +20.45] P+ 1.0 |
| PC-small (truth on 8 games, sized to +1.0 point on `POOL`) | +1.00 | **+0.950** [+0.360, +1.620] P+ 0.9999 | +0.594 [+0.121, +1.142] P+ 0.9971 |

PC-small recovers the injected effect to within 0.05 accuracy points and
resolves it, so the harness has power at one accuracy point **when the effect
arrives as a handful of picks all moving the right way**: its standard error
is 0.319 standalone, 0.256 through the card. The arms move 55-131 picks in
BOTH directions and carry standard errors of 0.87-1.21, three to four times
larger. The control therefore proves the instrument works and **does not bound
any arm**: no arm may be classified `bounded_by_control`, and every cell stays
`unresolved_below_power`.

## Calibration

Brier and log-loss improvement against `S3` on `POOL` (positive favours the
candidate): G1 -0.000613 / -0.001206 (P+ 0.313 / 0.318), G2 and G4 -0.000532 /
-0.001111 (0.299 / 0.294), G3 -0.000404 / -0.000850 (0.352 / 0.347). All four
sit within 0.0007 Brier of the incumbent; this is calibration noise and it
never vetoes a side read.

## Per season, and the eight cells whose interval clears zero

Every arm is carried by 2021 and paid for in 2025, the same shape lane C2
found on the full archive. G2/G4 against the card on `POOL`: 2020 0.000
(P+ 0.521), 2021 **+2.920** [0.000, +6.107] (0.962), 2022 -0.833 (0.332),
2023 -1.418 (0.254), 2024 -0.787 (0.411), 2025 **-3.030** [-6.329, 0.000]
(0.029).

Twenty-four of the 234 recorded cells have an interval clear of zero; sixteen
are positive-control cells, as predeclared. The remaining **eight are all 2025
season cells**: `ps_g1_vs_kl1b_pool_season_2025_card`,
`ps_g1_vs_s3_pool_season_2025_card`,
`ps_g2_vs_kl1b_pool_season_2025_standalone`,
`ps_g2_vs_s3_pool_season_2025_standalone`,
`ps_g3_vs_kl1b_pool_season_2025_standalone`,
`ps_g3_vs_s3_pool_season_2025_standalone`,
`ps_g4_vs_kl1b_pool_season_2025_standalone`,
`ps_g4_vs_s3_pool_season_2025_standalone`. They meet the LETTER of
`wrong_sign_resolved` for the direction "this arm beats this baseline in
2025", and they are recorded `unresolved_below_power` and named here for a
human look rather than closed from a lane report -- the precedent is lane U
(`docs/residual_slope_shrunk.md`), the primary metric is the overall `POOL`
card read, and eight of 234 correlated cells is close to what a nominal 5%
two-sided rate produces by chance.

## Cross-lane reproduction, and its limit

The `POOL_NEAR_3` and `POOL_NEAR_7` standalone cells reproduce lane C2's
`line_adjacent_3` and `line_adjacent_7` cells to every printed digit
(-2.339 [-6.742, +1.983] P+ 0.145 and -3.846 [-10.417, +2.454] P+ 0.117),
under a different active model (`d49194e04945a5e5`, not `c526cf6636cef6f8`)
and a feature table rebuilt two days later.

That reproduction is weaker than it looks and is reported with its limit:
measured directly on the two archives, the served probabilities differ by at
most 0.0035 and **zero of the 1,537 picks disagree**, so any accuracy-based
comparison was arithmetically bound to be identical. It confirms the pipeline
replays, not that the finding survives a materially different model.

## Week 1 2026

Read-only, against the served forecast
`artifacts/margin_predictions/2026-week-01-20260910T210852Z`, whose
`key_line_pick_read.json` sidecar replays the card's served probability to
**5.55e-17**. No forecast was regenerated, `artifacts/active_ats_model.json`
was not touched, and the served card is unchanged by this lane.

All sixteen lines are half points. **The served exact-atom rule touches 0 of
the 16 games**, confirming the structural invariant from the live card side.
G1 touches all 16, G3 eleven, G2 and G4 ten.

Standalone, G2, G3 and G4 would change **three** sides and G1 **six**:

| Game | Line | Served | Lattice | Changed by |
|---|---:|---:|---:|---|
| CHI at CAR | -2.5 | 0.5166 | 0.4360 | G1, G2, G3, G4 |
| NO at DET | 6.5 | 0.5309 | 0.4817 | G1, G2, G3, G4 |
| TB at CIN | 3.5 | 0.5063 | 0.4813 | G1, G2, G3, G4 |
| BUF at HOU | -1.5 | 0.5405 | 0.4805 | G1 only |
| CLE at JAX | 8.5 | 0.5336 | 0.4664 | G1 only |
| NYJ at TEN | 1.5 | 0.4847 | 0.5075 | G1 only |

Through the played three-member card, G2, G3 and G4 change **two** sides
(CHI at CAR and TB at CIN) and G1 three (those two plus BUF at HOU). Every
standalone change but one moves from the home side to the away side, the same
direction lane C2 reported on the previous week's card, and the three games
G2/G3/G4 touch are the identical three lane C2 named.

Correction, disclosed: a first pass of this stage read the smooth baseline
from `discrete_push_read.json`, whose `smooth.cover` is a different quantity
and misses the card's served probability by up to 0.0186. It reported a single
knife-edge flip and was wrong. The stage now reads
`key_line_pick_read.json`, which replays the card to 5.55e-17, and **fails
closed above 1e-9** -- the gate every other MOD-18 lane already had and this
one initially lacked. The table above is from the corrected run.

The operational consequence, measured: the paired challenger
`key_line_pick_read_off_incumbent`, which
`docs/mod18_discrete_margin_mapping.md` named as the evidence that would
settle this at no window cost, records the **identical** read to the incumbent
this week, because the rule it brackets fires on none of the sixteen games.
`nfl-ats prospective-score --start-season 2026` scores it at 2 of 2 settled
and 14 pending, exactly the active model's own row. It is accruing no
information about the key-number rule, and it cannot while every pool line is
a half point.

## An unpredeclared finding this lane fell over, reported with that label

Not an arm, not predeclared, and noticed only while reading the domain table
after the signs were visible. It is reported here because it is larger than
anything the arms found and it bears on the played card.

Splitting the SAME `POOL` / `OFF_POOL` axis, the played three-member overlay
union (coach fade, division revenge, player arrests) against the model's own
raw pick:

| Domain | n | Overlay union vs the raw read | P+ |
|---|---:|---|---:|
| `POOL` (half-point lines) | 842 | **-1.425** [-3.933, +1.160] | **0.136** |
| `OFF_POOL` (whole and quarter points) | 661 | **+4.841** [+1.506, +8.235] | **0.999** |
| `POOL_NEAR_3` (2.5, 3.5) | 342 | -3.216 [-7.243, +0.855] | 0.060 |
| `POOL_NEAR_7` (6.5, 7.5) | 156 | +2.564 [-3.448, +8.861] | 0.791 |

The overlays' whole measured contribution on this archive sits on lines the
pool never posts, and on the lines it always posts they are 86% likely a loss.
That is a bigger number, in the direction that matters, than any arm in this
lane.

It is **not** acted on here and must not be, for two reasons stated plainly:
it was not predeclared, and the `POOL` / `OFF_POOL` split is a property of the
consensus-opener archive rather than of any game, so an overlay membership
rule conditioned on it would be a threshold with no named mechanism -- exactly
what AGENTS.md's "No unexplained threshold flips on the played card" forbids.
What it warrants is a predeclared lane of its own, with a mechanism for why a
line's granularity should predict whether a coach-fade or revenge rule helps.

The four cells are recorded in their own family
`mod18_overlay_union_by_line_shape_v1`, all `unresolved_below_power`, with the
post-hoc status in their `--notes` so they can never be pooled with the
predeclared arms above.

## Recorded

234 rows under family `mod18_pool_shaped_read_v1`, every one
`unresolved_below_power`, no closing ground, each with a plain-English
summary, written serially through `nfl-ats weak-signals record` with
`--replace` (registry 6,065 to 6,299 rows, zero failures). The same 234 argv
were first executed against an isolated registry under the session scratchpad
and all returned 0, so the emitted argv is admissible to the real validator.
Ten degenerate comparisons in which no graded pick moved at all are named in
`skipped.json` and deliberately not recorded, because a bootstrap of zeros
would report a `probability_positive` of 0.0 for what is really "no change".

No rotation window spent -- this is a promotion-style look on the reused
Tuesday-opener archive, the precedent `docs/player_arrests_policy_eval.md` and
MOD-17 both set.
