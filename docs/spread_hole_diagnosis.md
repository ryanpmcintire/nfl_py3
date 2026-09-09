# Why the model is weak on big spreads (MOD-18 lane F)

Written 2026-09-09. Every number below is **measured this session** unless it
carries another tag. The commands are listed at the bottom; the tables live in
`artifacts/spread_hole_diagnosis/20260909T212843Z/`.

Closing-grounds taxonomy, verbatim, because this document reports intervals:
an interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero".

## The defect, reproduced

Measured from `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`
(active model `c657058903f3232b`, weak_stack market-residual ridge, alpha 10,
gaussian_median, 1,537 archived games 2020-2025, 1,503 non-push at the opener):

| line size | n | model right, raw | model right, as served (S3) | model's stated confidence | favourite actually covers | model takes the favourite (raw) |
|---|---|---|---|---|---|---|
| 0-6.5 | 1,104 | 56.16% | 56.16% | 55.6% | 49.63% | 51.9% |
| exactly 7 | 74 | 52.70% | 51.35% | 56.5% | 45.95% | 52.7% |
| 7.5-10 | 194 | 48.45% | 51.55% | 56.2% | 45.88% | 57.2% |
| 10.5+ | 131 | 44.27% | 47.33% | 56.0% | 53.44% | 51.2% |

(The last two columns exclude the 12 pick'em games, which have no favourite.)

The raw numbers reproduce the defect as stated in the brief. Two corrections
worth carrying forward: (1) the **served** model is already better than the raw
model in both big-spread buckets — the S3 home-side offset promoted 2026-09-07
adds +3.1 points at 7.5-10 and +3.1 at 10.5+ — so the hole the played card
inherits is 51.5% / 47.3%, not 48.5% / 44.3%; (2) the favourite/underdog split
at 7.5-10 is 111 favourite picks at 45.05% and 83 underdog picks at 53.01% on
the RAW read, 113 at 47.79% and 81 at 56.79% as served.

## (a) Which feature groups push toward the favourite on big spreads

The active ridge is `imputer -> StandardScaler -> Ridge(alpha=10)` over 90
weak_stack columns (read: `src/nfl_ats/margin.py:626-661`). Its output for one
game is the intercept plus a sum of per-column terms `coef_j * z_j`, so the
prediction decomposes exactly into the project's own declared feature families
(`resolve_feature_groups`, read: `src/nfl_ats/margin.py:455`). The
decomposition was recomputed by refitting every one of the 107 archive weeks
with the identical walk-forward cutoff; the reconstructed residual matches the
archive's `residual_at_open` to 1.3e-14 points, so this is the served model's
own arithmetic, not a lookalike.

Mean contribution **toward the favourite**, in points, by line size:

| group | 0-3 | 3.5-6.5 | 7 | 7.5-10 | 10.5+ |
|---|---|---|---|---|---|
| **results** | **+0.485** | **+1.580** | **+2.484** | **+2.975** | **+4.200** |
| player_continuity | -0.023 | +0.111 | +0.166 | +0.180 | +0.301 |
| bias | -0.062 | -0.082 | -0.034 | +0.010 | +0.090 |
| player_injuries | +0.001 | -0.017 | -0.046 | +0.007 | +0.038 |
| context | +0.014 | +0.007 | -0.014 | -0.010 | +0.001 |
| experience | +0.007 | -0.002 | +0.021 | -0.012 | -0.012 |
| intercept | -0.005 | -0.010 | -0.023 | -0.028 | -0.044 |
| player_values | -0.007 | -0.070 | -0.119 | -0.031 | -0.176 |
| player_qb | -0.060 | -0.203 | -0.215 | -0.355 | -0.495 |
| elo | -0.079 | -0.284 | -0.397 | -0.543 | -0.729 |
| market | -0.159 | -0.310 | -0.425 | -0.546 | -0.785 |
| defense | -0.096 | -0.186 | -0.579 | -0.544 | -0.849 |
| offense | -0.068 | -0.352 | -0.515 | -0.616 | -1.100 |

**One family leans to the favourite and it is `results`** — the six columns
`home_point_diff`, `away_point_diff`, `diff_point_diff`, `home_ats_residual`,
`away_ats_residual`, `diff_ats_residual` (season-to-date scoring margin and
season-to-date ATS residual). It pushes toward the favourite in 63.6% of 0-3
games, 98.5% of 7.5-10 games and **100.0%** of 10.5+ games, and its size grows
almost linearly with the line: +0.49 points at 0-3, +2.98 at 7.5-10, +4.20 at
10.5+. Every other family leans the other way, and the sum of all of them
(-0.96 / -2.48 / -3.78) very nearly cancels it:

On the 1,491 non-push, non-pick'em games (the population the accuracy numbers
are computed on, so the two tables are directly comparable):

| line size | results | everything else | net residual | + residual median | model's net lean to the favourite |
|---|---|---|---|---|---|
| 0-6.5 | +1.021 | -0.956 | +0.065 | -0.422 | +0.054 pts, favourite picked 51.9% |
| 7 | +2.510 | -2.218 | +0.292 | -0.488 | +0.167 pts, favourite picked 52.7% |
| 7.5-10 | +2.978 | -2.476 | +0.502 | -0.517 | +0.356 pts, favourite picked 57.2% |
| 10.5+ | +4.168 | -3.781 | +0.387 | -0.624 | +0.088 pts, favourite picked 51.2% |

(The per-family table above it is computed on all 1,525 lined games including
pushes, so its column sums are -0.052 / +0.183 / +0.305 / +0.486 / +0.442 on
the five fine buckets; the difference is the push rows, not a different
calculation.)

That is the mechanism, in plain terms: **season-to-date form is a restatement
of the spread**. A team favoured by 13 has a big point differential *because*
it is the kind of team the market makes a 13-point favourite. The ridge is
trained on `ats_margin` (margin minus the line), so it learns a positive weight
on form; on a small line that weight adds half a point, on a big line it adds
four. The rest of the model then has to spend four points undoing it, and what
reaches the card is the **difference between two large opposing quantities** —
+0.50 points at 7.5-10, +0.39 at 10.5+. A half-point net formed by cancelling
±4 points is not an edge; it is arithmetic noise with a sign.

Is `results` also the family that carries the edge on small spreads? Yes.
Removing one family's contribution from the served point and re-picking
(leave-one-group-out, week-blocked, 2,000 draws, seed 20260817):

| group dropped | 0-3 | 3.5-6.5 | 7 | 7.5-10 | 10.5+ |
|---|---|---|---|---|---|
| **results** | **-2.72** | **-5.98** | +4.05 | **+3.09** | -1.53 |
| defense | -5.07 | -1.81 | 0.00 | -6.70 | +2.29 |
| offense | -0.72 | 0.00 | +2.70 | -5.67 | +4.58 |
| elo | -0.72 | -0.54 | +1.35 | -6.70 | +5.34 |
| market | -0.72 | 0.00 | 0.00 | -4.64 | -0.76 |
| player_qb | +0.18 | -0.36 | +5.41 | -3.09 | -0.76 |
| player_values | -1.27 | -1.99 | +1.35 | -5.15 | -3.05 |
| player_continuity | -0.54 | -0.72 | +5.41 | -1.55 | -4.58 |
| player_injuries | -1.45 | -0.54 | +4.05 | -1.55 | 0.00 |
| context | -1.45 | -1.81 | +12.16 | -9.79 | 0.00 |
| bias | +0.72 | -0.36 | -4.05 | -1.03 | -1.53 |
| experience | -0.72 | +1.09 | +1.35 | -3.61 | +0.76 |
| intercept | +0.36 | +0.54 | -1.35 | -1.55 | +0.76 |

`results` is the single most consequential family (it moves 129 / 175 / 25 / 86
/ 70 picks per bucket, more than any other), it is the only one whose removal
costs several points on small spreads, and it is one of only two that **gain**
on 7.5-10. These per-bucket deltas are 74-552-game cells and the individual
cells are far below the resolution that would resolve any of them; they are
read here as attribution of an already-known dip, not as five independent
findings. The 10.5+ column is the one place `results` is not the story.

**Read section "Why R1 read flat" before acting on this table.** Arm R1 removed
the line's linear share from all six `results` columns and the favourite lean
barely moved, because `elo_diff` (R² 0.575 with the line) and
`diff_off_epa_per_play` (R² 0.485) restate the line just as well as
`diff_point_diff` (R² 0.646) does. `results` is where a ridge on 90 collinear
columns parks the lean, not its cause. The mechanism is the whole team-quality
block, and the decomposition above locates it rather than isolating it.

## (b) Is the residual's sign informative on big spreads

Forced-pick accuracy of the sign, by bucket, week-blocked bootstrap (2,000
draws, seed 20260817, within-week correlation zero by owner mandate):

| line size | n | raw point > line | served point (S3) | favourite share of picks |
|---|---|---|---|---|
| 0-3 | 552 | 56.16% [51.88, 60.22] | 56.16% [51.88, 60.22] | 49.6% |
| 3.5-6.5 | 552 | 56.16% [51.48, 61.31] | 56.16% [51.48, 61.31] | 54.3% |
| 7 | 74 | 52.70% [42.10, 63.42] | 51.35% [40.00, 62.30] | 51.4% |
| 7.5-10 | 194 | 48.45% [41.26, 55.50] | 51.55% [44.15, 58.79] | 58.2% |
| 10.5+ | 131 | 44.27% [36.51, 51.97] | 47.33% [39.20, 55.32] | 60.3% |

Read this as the level, not as a verdict. On 7.5-10 the served sign sits at
51.55% with the interval reaching 58.8%; on 10.5+ at 47.33% reaching 55.3%.
Neither is resolved in either direction at 194 and 131 games — a 2-point cell
cannot be. What *is* clear from (a) is that the number the sign is taken from
is a half-point remainder of a ±4-point cancellation, so its sign carries very
little of the model's information on big lines even where its accuracy happens
to land above 50%. The recorded registry cells report
`probability_positive`, never a binary.

Split by which side the model took, with the same bootstrap:

| line size | side taken | n | accuracy | stated confidence |
|---|---|---|---|---|
| 0-6.5 | favourite | 567 | 55.56% [50.82, 59.93] | 55.6% |
| 0-6.5 | underdog | 525 | 56.76% [52.52, 61.18] | 55.7% |
| 7 | favourite | 38 | 47.37% [33.31, 63.89] | 56.5% |
| 7 | underdog | 36 | 55.56% [40.81, 71.44] | 56.6% |
| 7.5-10 | favourite | 113 | 47.79% [38.18, 56.90] | 56.6% |
| 7.5-10 | underdog | 81 | 56.79% [46.43, 68.42] | 55.6% |
| 10.5+ | favourite | 79 | 50.63% [40.00, 60.68] | 56.4% |
| 10.5+ | underdog | 52 | 42.31% [29.78, 55.10] | 55.3% |

The stated confidence is 55.3-56.6% in every one of those eight cells. It is
flat because it is read off a residual sample whose spread is ~12.9 points in
every bucket (measured: mean fitted residual scale 12.89 / 12.88 / 12.91 /
12.93 across the four buckets) — the model has one uncertainty and reuses it
everywhere, so the confidence a reader sees carries no information about which
bucket a game is in.

## (c) Does the market line already carry the favourite/underdog tilt

Underdog ATS cover rate on the full nflverse schedule table, 2009-2025 regular
season, non-push, using the table's own line:

| era | 0-6.5 | 7 | 7.5-10 | 10.5+ |
|---|---|---|---|---|
| 2009-2013 | 49.46% (n=837) | 49.44% (n=89) | 51.93% (n=181) | 53.62% (n=138) |
| 2014-2019 | 52.16% (n=1,066) | 47.19% (n=89) | 54.36% (n=195) | 44.85% (n=136) |
| 2020-2025 | 51.16% (n=1,124) | 52.33% (n=86) | 55.24% (n=210) | 48.15% (n=162) |
| **2009-2025 pooled** | **51.04% (n=3,027)** | **49.62% (n=264)** | **53.92% (n=586)** | **48.85% (n=436)** |

At 7.5-10 the underdog tilt is present in all three eras, at 51.9 / 54.4 / 55.2
percent, 11 of 17 individual seasons above 50%, and 53.92% pooled over 586
games. It is **not** a 2020-2025 artefact; magnitudes vary by era, which is the
expected shape (era magnitude, not presence). At 10.5+ there is no underdog
tilt at all over the long run — 48.85% pooled, and the archive's own 2020-2025
window has favourites covering 53.44%. Per-season detail is in
`market_favourite_bias_by_season.csv`.

So the two big-spread buckets fail for different reasons, and they should stop
being described as one hole:

- **7.5-10 is a mechanism error.** The market leans to the underdog by roughly
  4 points of cover rate across 17 seasons and the model leans to the favourite
  57-58% of the time, for the reason in (a). The model is systematically on the
  wrong side of a durable, market-level tilt.
- **10.5+ is not explained by market tilt.** The long-run dog rate there is
  below 50%. What the decomposition shows instead is the largest cancellation
  in the model (+4.17 against -3.78), i.e. the least stable point estimate in
  the book.

## (d) Where the final margin actually lands, and how the served mapping misplaces it

Signed distance from the line to the final margin (positive = favourite-ward)
for the 329 archived games with a line of 7.5 or more, rounded to the integer
the football recorded (`big_spread_margin_mass.csv`): **17.9%** of them (59 of
329) land at a distance of exactly 3, 7, 10 or 14 points from the line, and
**5.5%** (18 of 329) land on it. Empirical share against the share the served
Gaussian assigns to the
same integer, using each week's own fitted scale:

| line size | margin − line | empirical | served Gaussian | gap |
|---|---|---|---|---|
| 7.5-10 | +10 | 5.67% | 1.75% | **+3.92 pts** |
| 7.5-10 | −14 | 5.15% | 2.12% | +3.04 pts |
| 7.5-10 | 0 (push) | 5.15% | 2.50% | +2.65 pts |
| 7.5-10 | −10 | 3.61% | 2.42% | +1.19 pts |
| 7.5-10 | +3 | 2.06% | 2.34% | −0.28 pts |
| 7.5-10 | +7 | 1.55% | 2.03% | −0.48 pts |
| 10.5+ | +10 | 4.58% | 1.25% | **+3.33 pts** |
| 10.5+ | 0 (push) | 3.05% | 1.94% | +1.11 pts |
| 10.5+ | −14 | 0.00% | 2.39% | −2.39 pts |
| 10.5+ | −7 | 0.00% | 2.36% | −2.36 pts |
| 0-6.5 | 0 (push) | 6.25% | 2.92% | +3.33 pts |
| 0-6.5 | −7 | 1.18% | 2.66% | −1.48 pts |
| 0-6.5 | −3 | 1.09% | 2.90% | −1.81 pts |

The Gaussian is wrong about the mass, in the direction the owner's binding
premise says it is: it spreads 12.9 points of standard deviation smoothly and
puts 1.2-2.9% on each key integer where the football puts 0% or 5.7%.

But — and this is the part that explains why three separate discrete-lattice
arms (K1/K2/K3, at -1.4/-0.2/-1.9 points) and a bucket recalibration (mod11,
-0.4) all came back flat — **the served mapping cannot change a forced pick at
all except through its location.** Read `src/nfl_ats/calibration.py:279-305`:
every smooth method returns `P(residual > line − centre)`, so
`P > 0.5` if and only if `centre + location > line`. For gaussian_median the
location is the residual median; for the ECDF it is the empirical median. Any
symmetric-mass read of the same residual sample around the same centre gives
**the same side on every game**. A discrete read can only flip a pick where the
integer mass is asymmetric about the centre — a handful of games per season.
Fixing the mapping was never going to fix a bucket that is wrong by 8 points,
because the mapping is not what decides the side. The point is.

That is the finding to act on: **the hole on big spreads is in the POINT, and
the point's big-spread behaviour is set by team-quality features that restate
the line** (section (a) locates the lean in `results`; the Part 2 R1 result
below shows the whole collinear block carries it). A mass-correct probability is
still worth having for
the push and alternative-line answers the board serves (the +3.9-point
misplacement at exactly +10 is real, and the discrete read is the binding
requirement for those quantities), but it is not the repair for the side.

## Part 2 predeclaration — frozen 2026-09-09 before any candidate was scored

Family `mod18_spread_hole_v1`. Two arms, no tuning after scoring, no subset
search, no threshold rule and no pick flip in either arm; both are model-level
changes applied to every game at every line.

Grading, identical for both arms and for the incumbent: walk forward exactly as
`nfl_ats.clv.opener_pick_evaluation` does — one weekly market-residual ridge
per archive week, trained on completed regular-season games strictly before
that week's first kickoff, minimum 500 training games, alpha 10, mapping
`gaussian_median`, the S3 home-side offset refitted from **each arm's own** raw
out-of-time opener stream exactly as the served policy does. Scored at the
**opener** on the same 1,537-game archive (1,503 non-push). Then scored THROUGH
the played three-member card — the OR union of `coach_fade_overlay`,
`division_revenge_tilt_overlay` and `player_arrests_back_side_policy`
(`overlay_union_coach_division_revenge_player_arrests_v2`) — with every member
trigger recomputed against each incoming card, using
`nfl_ats.overlay_composition`'s own machinery. Paired week-blocked and
season-blocked bootstrap, 20,000 samples, seed 20260821, within-week
correlation zero. Reported overall and by the five declared line buckets.
Primary quantity is the paired accuracy-point difference through the card at
the opener with its `probability_positive`. Thresholds govern only what this
document may CLAIM; the played card is an expected-value decision.

**Arm C1 (already declared in `docs/spread_regime_program.md`, re-run against
the current incumbent).** Feature profile `weak_stack_spread_regime`: weak_stack
plus eight home-oriented row-local spread features — `sign(spread)` times
indicators for |spread| in [3.5, 6.5], [7.5, 10] and [10.5, inf), the signed
absolute spread, and `sign(spread)` times |spread| distance to each of 3, 7, 10
and 14. Reference bucket 0-3, no separate indicator at exactly 7, alpha stays
10. The regime columns are recomputed from the **opener** line at scoring, as
lane H did, because they are a deterministic function of the line and using the
close-era value would put post-Tuesday information into a Tuesday pick. Reason
to run it again: lane H's registry rows
(`mod18_spread_regime_opener_v1_c1_2020_2025`, effect **-1.730** accuracy
points, 95% [-3.446, 0.000], `probability_positive` **0.0221**, 1,503 games,
recorded 2026-09-07) were scored against the retired incumbent
`a4c757efd2525da6` **without** the S3 home-side offset and **standalone**, never
through the played card.

**Arm R1 (new, and the reason it exists is section (a) above).** Mechanism: the
`results` family is a level that co-moves mechanically with the line, so on a
big spread the ridge re-adds the market's own favourite margin on top of a line
that already contains it (+4.20 points of favourite lean at 10.5+, in 100% of
those games). The fix removes that co-movement at the source instead of
patching the output: each of the six `results` columns is replaced by its
residual after an ordinary least-squares regression of that column on the
game's own `spread_line`, with the slope and intercept fitted **only on the
training rows of that week's walk-forward** and applied to the scoring rows at
the opener line. The block then carries only the part of recent form the market
has not already priced. Nothing else changes: same 90 columns, same names, same
alpha 10, same gaussian_median mapping, same S3 offsets refitted from this
arm's own stream. No outcome enters the regression, so it is pregame-only. This
arm is motivated by a descriptive decomposition computed on the same archive it
will be graded on; that is a stated reuse discount, exactly the one
`docs/spread_regime_program.md` declares, not independent confirmation.

Predeclared secondary reads, for both arms: per-bucket paired deltas through
the card, and the standalone (pre-card) paired delta. Every cell is recorded
with `nfl-ats weak-signals record` under
`spread_hole_<arm>_<surface>_<bucket|overall>_<window>`, where surface is
`card` or `standalone`.

## Part 2 results

The incumbent replay is exact: re-fitting all 107 archive weeks inside this
lane's own script reproduces
`artifacts/opener_evaluation/20260909T183120Z/per_game.parquet` with a maximum
residual gap of **0.0** and a maximum probability gap of **0.0** across all
1,537 games, so the pairing below is the served model against itself plus one
change.

Played card = the three-member OR union
`overlay_union_coach_division_revenge_player_arrests_v2`, every member trigger
recomputed against each incoming card. Incumbent card accuracy at the opener:
**55.89%** on 1,503 non-push games; the served model's own picks, before the
card's three overlays, score 54.56% on the same games.

Paired delta through the card, week-blocked, 20,000 samples, seed 20260821:

| arm | scope | n | incumbent card | candidate card | picks changed | delta (pts) | 95% week-blocked | probability_positive | season-blocked P+ |
|---|---|---|---|---|---|---|---|---|---|
| C1 | overall | 1,503 | 55.89% | 54.36% | 109 | **-1.530** | [-2.869, -0.199] | **0.0120** | 0.0032 |
| C1 | 0-6.5 | 1,104 | 57.61% | 55.98% | 96 | -1.630 | [-3.333, +0.092] | 0.0323 | 0.0098 |
| C1 | 7 | 74 | 50.00% | 51.35% | 3 | +1.351 | [-2.941, +5.714] | 0.7103 | 0.8302 |
| C1 | 7.5-10 | 194 | 51.55% | 51.03% | 3 | -0.516 | [-2.286, +1.081] | 0.2909 | 0.2824 |
| C1 | 10.5+ | 131 | 51.15% | 47.33% | 7 | -3.817 | [-7.752, 0.000] | 0.0224 | 0.0010 |
| R1 | overall | 1,503 | 55.89% | 55.69% | 15 | **-0.200** | [-0.669, +0.267] | **0.2056** | 0.2066 |
| R1 | 0-6.5 | 1,104 | 57.61% | 57.61% | 10 | 0.000 | [-0.466, +0.522] | 0.4992 | 0.4778 |
| R1 | 7 | 74 | 50.00% | 50.00% | 0 | 0.000 | [0, 0] dead heat | 0.5000 | 0.5000 |
| R1 | 7.5-10 | 194 | 51.55% | 50.00% | 3 | -1.546 | [-3.483, 0.000] | 0.0240 | 0.0080 |
| R1 | 10.5+ | 131 | 51.15% | 51.15% | 2 | 0.000 | [-2.273, +2.273] | 0.5039 | 0.5000 |

Standalone (before the card), same pairing:

| arm | scope | n | incumbent | candidate | picks changed | delta (pts) | 95% week-blocked | probability_positive |
|---|---|---|---|---|---|---|---|---|
| C1 | overall | 1,503 | 54.56% | 52.69% | 154 | -1.863 | [-3.485, -0.268] | 0.0106 |
| C1 | 0-6.5 | 1,104 | 56.16% | 54.08% | 131 | -2.083 | [-4.159, -0.089] | 0.0222 |
| C1 | 7 | 74 | 51.35% | 54.05% | 8 | +2.703 | [-4.000, +8.974] | 0.7935 |
| C1 | 7.5-10 | 194 | 51.55% | 50.52% | 4 | -1.031 | [-3.141, +0.990] | 0.1595 |
| C1 | 10.5+ | 131 | 47.33% | 43.51% | 11 | -3.817 | [-8.759, +0.807] | 0.0604 |
| R1 | overall | 1,503 | 54.56% | 54.16% | 22 | -0.399 | [-0.986, +0.197] | 0.0893 |
| R1 | 0-6.5 | 1,104 | 56.16% | 55.89% | 15 | -0.272 | [-0.908, +0.366] | 0.2014 |
| R1 | 7 | 74 | 51.35% | 51.35% | 0 | 0.000 | [0, 0] dead heat | 0.5000 |
| R1 | 7.5-10 | 194 | 51.55% | 49.48% | 4 | -2.062 | [-4.167, -0.498] | 0.0083 |
| R1 | 10.5+ | 131 | 47.33% | 48.09% | 3 | +0.763 | [-1.613, +3.478] | 0.7139 |

C1's standalone number corroborates lane H's independent 2026-09-07 run against
the retired incumbent (-1.730, 95% [-3.446, 0.000], `probability_positive`
0.0221) at -1.863, 95% [-3.485, -0.268], `probability_positive` 0.0106. Two
different incumbents, two different scripts, the same answer.

All 20 cells are in the registry under family `mod18_spread_hole_v1`
(`spread_hole_<arm>_<surface>_<scope>_2020_2025`). The two **overall** C1 cells
are the only terminal classifications: both their week-blocked and their
season-blocked intervals sit **entirely below zero**, which is the one
admissible closing ground (`wrong_sign_resolved`) — C1 as a replacement for the
played card is resolved worse, not merely unmeasured. Every other cell,
including all eight per-bucket C1 cells and all ten R1 cells, is recorded
`unresolved_below_power` and reports `probability_positive`.

### Why R1 read flat, and what it teaches

R1 was built to delete the +2.98 / +4.20-point favourite lean that section (a)
attributes to the `results` family. It changed **15 picks out of 1,503** and
moved the favourite lean almost not at all:

| arm | favourite share of picks, 7.5-10 | favourite share, 10.5+ | mean favourite-ward residual, 7.5-10 |
|---|---|---|---|
| incumbent | 58.25% | 60.31% | +0.502 |
| R1 (form orthogonalised to the line) | 59.28% | 59.54% | +0.483 |
| C1 (spread-regime features added) | 60.31% | 56.49% | +0.509 |

The reason is measurable and it corrects section (a)'s emphasis: the line does
not only explain the `results` block. Correlation of `spread_line` with each
column across every completed game in the table —

| column | r | R² |
|---|---|---|
| diff_point_diff | +0.804 | 0.646 |
| elo_diff | +0.758 | 0.575 |
| diff_off_epa_per_play | +0.696 | 0.485 |
| home_point_diff | +0.570 | 0.324 |
| diff_ats_residual | +0.498 | 0.248 |
| home_off_epa_per_play | +0.487 | 0.237 |
| diff_def_epa_per_play | -0.412 | 0.170 |

**Every team-quality column in the model is a restatement of the line.**
`results` is not the cause of the favourite lean; it is merely where a ridge on
90 collinear columns happens to park it. Strip the line out of six of them and
the ridge rebuilds the same lean out of elo and EPA — which is exactly what the
15 changed picks show. The correct statement of the mechanism is therefore:

> On a big spread the model's point is a small remainder left over after ±4
> points of mutually-cancelling team-quality terms, all of which are proxies
> for the line the model is trying to beat. Removing one of them does nothing.

A repair has to act on the whole collinear block at once — orthogonalise all of
it against the line, or penalise the block as a group while leaving
`spread_line` itself unpenalised, so that the market's own estimate is carried
by the market column and the team-quality columns can only add what the market
has not priced. That is the next arm to declare, and it is not this one.

### Decision

**Keep the incumbent card.** At the opener, on the forced-pick pool's own
grade, the three cards score 55.89% (incumbent), 55.69% (R1) and 54.36% (C1) on
the same 1,503 non-push games. Playing C1 instead of the incumbent is a bet at
`probability_positive` 0.0120 and playing R1 is a bet at 0.2056; both are the
wrong side of an expected-value call, which is the only bar that governs which
card is played. Nothing here rests on an interval containing zero: R1 is
unresolved and stays open as a mechanism, it simply does not earn the card
today.

If either arm HAD won through the card, promoting it would require editing
`artifacts/active_ats_model.json` (feature_profile / model_id /
feature_table_sha256 / historical_evaluation), regenerating the opener
evaluation, the overlay composition and the weekly forecast under the new
model, and refreshing `CURRENT_PREDICTIONS.md` via `nfl-ats publish-predictions`
and the board via `nfl-ats publish-board`. Nothing in this lane was promoted and
the active manifest was not touched.

## Commands run

```
uv run --no-sync python scripts/spread_hole_diagnosis.py \
  --features data/processed/game_features_weak_stack.parquet \
  --archive artifacts/opener_evaluation/20260909T183120Z \
  --market-root data/market/raw \
  --out artifacts/spread_hole_diagnosis/20260909T212843Z --draws 2000

uv run --no-sync python scripts/spread_hole_arms.py \
  --features data/processed/game_features_weak_stack.parquet \
  --data-root data --market-root data/market/raw \
  --archive artifacts/opener_evaluation/20260909T183120Z \
  --out artifacts/spread_hole_diagnosis/20260909T212843Z/arms

uv run --no-sync nfl-ats weak-signals record ...   # 20 cells, mod18_spread_hole_v1
```
