# Group-penalising the team-quality block against the line (MOD-18 lane M)

Follow-on to `docs/spread_hole_diagnosis.md`. That document ends by naming this
arm: *"A repair has to act on the whole collinear block at once — orthogonalise
all of it against the line, or penalise the block as a group while leaving
`spread_line` itself unpenalised, so that the market's own estimate is carried
by the market column and the team-quality columns can only add what the market
has not priced. That is the next arm to declare, and it is not this one."*
(read: `docs/spread_hole_diagnosis.md:425-428`).

Closing-grounds taxonomy, verbatim, because this document reports intervals:
an interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (b) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never "contains zero".

## Design, frozen 2026-09-09T22:47:06Z before any candidate was scored

Family `mod18_spread_hole_group_penalty_v1`. Three arms, no tuning after
scoring, no subset search, no threshold rule and no pick flip in any arm; all
three are model-level changes applied to every game at every line.

### The team-quality block, listed explicitly

The block is every column of the served `weak_stack` profile whose declared
family (`nfl_ats.margin.FEATURE_FAMILIES`, surfaced by
`margin_feature_groups("market_residual", "weak_stack")`) is `results`, `elo`,
`offense` or `defense` — **47 of the 90 columns**:

- `results` (6): `home_point_diff`, `away_point_diff`, `diff_point_diff`,
  `home_ats_residual`, `away_ats_residual`, `diff_ats_residual`
- `elo` (2): `elo_diff`, `elo_home_win_prob`
- `offense` (21): `home_off_epa_per_play`, `away_off_epa_per_play`,
  `diff_off_epa_per_play`, `home_off_pass_epa_per_play`,
  `away_off_pass_epa_per_play`, `diff_off_pass_epa_per_play`,
  `home_off_rush_epa_per_play`, `away_off_rush_epa_per_play`,
  `diff_off_rush_epa_per_play`, `home_off_cpoe`, `away_off_cpoe`,
  `diff_off_cpoe`, `home_off_yards_per_play`, `away_off_yards_per_play`,
  `diff_off_yards_per_play`, `home_off_turnover_rate`,
  `away_off_turnover_rate`, `diff_off_turnover_rate`, `home_off_sack_rate`,
  `away_off_sack_rate`, `diff_off_sack_rate`
- `defense` (18): `home_def_epa_per_play`, `away_def_epa_per_play`,
  `diff_def_epa_per_play`, `home_def_pass_epa_per_play`,
  `away_def_pass_epa_per_play`, `diff_def_pass_epa_per_play`,
  `home_def_rush_epa_per_play`, `away_def_rush_epa_per_play`,
  `diff_def_rush_epa_per_play`, `home_def_yards_per_play`,
  `away_def_yards_per_play`, `diff_def_yards_per_play`,
  `home_def_takeaway_rate`, `away_def_takeaway_rate`,
  `diff_def_takeaway_rate`, `home_def_sack_rate`, `away_def_sack_rate`,
  `diff_def_sack_rate`

The remaining 43 columns are `bias` (9), `context` (7), `experience` (2),
`market` (2: `spread_line`, `total_line`), `player_continuity` (9),
`player_injuries` (7), `player_qb` (5) and `player_values` (2).

### Arm G1 — team-quality penalised 3x, `spread_line` unpenalised

`nfl_ats.margin.GroupPenaltyScaler` scales column *j* by `1/sqrt(m_j)` before a
plain `Ridge(alpha=10)`, so the column is effectively penalised by
`alpha * m_j` (read: `src/nfl_ats/margin.py:569-579`). G1 sets `m_j = 3.0` for
the 47 team-quality columns and `m_j = 1.0` for the other 42 penalised
columns, then normalises those 89 multipliers through the project's own
`column_penalty_multipliers(..., normalize=True)` so their count-weighted
geometric mean is exactly 1 and the AVERAGE penalty stays at `ridge_alpha`;
only the split moves. `spread_line` is exempted from that normalisation and
given `m = 1e-6`, i.e. an effective penalty of `1e-5` — **the scaler and
`column_penalty_multipliers` both reject a literal zero as non-finite/
non-positive, so "unpenalised" is implemented as this epsilon and is reported
as such**. Missing-value indicators inherit their source column's multiplier,
which is the scaler's own documented behaviour.

Everything else is the served configuration: the same 90 columns, the same
`weak_stack` profile, `ridge_alpha` 10, target `market_residual`, mapping
`gaussian_median`, and the S3 home-side offset refitted from this arm's own
raw out-of-time opener stream exactly as the served policy does.

### Arm G2 — the same with 10x

Identical to G1 with `m_j = 10.0` on the 47 team-quality columns.

### Arm G3 — the whole team-quality block orthogonalised to the line

Each of the 47 team-quality columns is replaced by its residual after an
ordinary least-squares regression of that column on the game's own
`spread_line`, with slope and intercept fitted **only on the training rows of
that week's walk-forward** and applied to the scoring rows at the opener line
(and, for the secondary close read, at the closing line). No outcome enters the
regression, so it is pregame-only. Nothing else changes: same 90 columns, same
names, same alpha 10, no `column_penalties`, same mapping, same S3 offsets
refitted from this arm's own stream. This is arm R1 of the diagnosis widened
from 6 columns to all 47, which is the correction R1's own post-mortem demands
(read: `docs/spread_hole_diagnosis.md:414-422`).

### Grading, identical for all three arms and for the incumbent

Walk forward exactly as `nfl_ats.clv.opener_pick_evaluation` does, reusing
`scripts/spread_hole_arms.py`'s harness so the incumbent replays exactly: one
weekly market-residual ridge per archive week, trained on completed
regular-season games strictly before that week's first kickoff, minimum 500
training games. Scored at the **opener** on the same 1,537-game archive
(1,503 non-push at the opener). Then scored THROUGH the played three-member
card — the OR union of `coach_fade_overlay`, `division_revenge_tilt_overlay`
and `player_arrests_back_side_policy`
(`overlay_union_coach_division_revenge_player_arrests_v2`) — with every member
trigger recomputed against each incoming card, using
`nfl_ats.overlay_composition`'s own machinery.

Paired week-blocked **and** season-blocked bootstrap, 20,000 samples, seed
20260821, within-week correlation zero by owner mandate. Reported overall and
by the four declared line buckets (0-6.5, exactly 7, 7.5-10, 10.5+).

The replay is only trusted if the incumbent reproduces
`artifacts/opener_evaluation/20260909T183120Z/per_game.parquet` to a maximum
residual gap of 0.0, as it did in the diagnosis lane.

**Primary quantity**: the paired accuracy-point difference through the card at
the opener, overall, with its `probability_positive`. **Predeclared secondary
reads**: per-bucket paired deltas through the card; the standalone (pre-card)
paired delta overall and per bucket; the favourite share of picks per arm per
bucket; the mean `results`-family contribution toward the favourite per arm per
bucket (so the mechanism is visible, not just the score); and Brier score and
log loss of the served home-cover probability at the opener, overall and per
bucket.

Every cell is recorded with `nfl-ats weak-signals record`, units
`accuracy_points`, league `nfl`, named
`spread_hole_<arm>_<card|standalone>_<scope>_2020_2025`. Only a whole-interval
wrong sign on BOTH the week-blocked and the season-blocked interval may be
recorded as `refuted_mechanism` / `wrong_sign_resolved`; everything else is
`unresolved_below_power`.

**Reuse discount, stated up front**: these arms are motivated by a descriptive
decomposition computed on the same 2020-2025 archive they are graded on. That
is the same stated discount `docs/spread_regime_program.md` declares, not
independent confirmation.

Thresholds govern only what this document may CLAIM. Which card is PLAYED is an
expected-value decision on the opener grade, full stop.

## Results

Every number in this section is **measured this session** by the command at the
bottom; the tables live in
`artifacts/spread_hole_group_penalty/20260909T224706Z/`.

The incumbent replay is exact: re-fitting all 107 archive weeks inside this
lane's own script reproduces
`artifacts/opener_evaluation/20260909T183120Z/per_game.parquet` with a maximum
residual gap of **0.0** and a maximum probability gap of **0.0** across all
1,537 games, so the pairing below is the served model against itself plus one
change. The per-family decomposition used in the mechanism table closes against
each arm's own prediction to at most **9.3e-15** points.

Penalty multipliers actually handed to the ridge, after the 89-column
normalisation that holds their count-weighted geometric mean at 1:

| arm | team-quality (47 cols) | other penalised (42 cols) | `spread_line` |
|---|---|---|---|
| G1 | 1.6794 | 0.5598 | 1e-6 |
| G2 | 2.9642 | 0.2964 | 1e-6 |

Incumbent card accuracy at the opener: **55.89%** on 1,503 non-push games; the
served model's own picks, before the card's three overlays, score **54.56%** on
the same games.

### Paired delta through the played card, at the opener

Week-blocked and season-blocked bootstrap, 20,000 samples, seed 20260821.

| arm | scope | n | incumbent card | candidate card | picks changed | delta (pts) | 95% week-blocked | `probability_positive` | 95% season-blocked | season P+ |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | overall | 1,503 | 55.89% | 55.76% | 16 | **-0.133** | [-0.718, +0.403] | **0.3231** | [-0.443, +0.140] | 0.2039 |
| G1 | 0-6.5 | 1,104 | 57.61% | 57.88% | 9 | +0.272 | [-0.266, +0.824] | 0.8427 | [-0.164, +0.761] | 0.8938 |
| G1 | 7 | 74 | 50.00% | 50.00% | 0 | 0.000 | [0, 0] dead heat | 0.5000 | [0, 0] | 0.5000 |
| G1 | 7.5-10 | 194 | 51.55% | 50.00% | 3 | -1.546 | [-3.535, 0.000] | 0.0237 | [-2.778, -0.490] | 0.0076 |
| G1 | 10.5+ | 131 | 51.15% | 49.62% | 4 | -1.527 | [-4.724, +1.471] | 0.1562 | [-3.049, 0.000] | 0.0437 |
| G2 | overall | 1,503 | 55.89% | 55.02% | 31 | **-0.865** | [-1.652, -0.133] | **0.0095** | [-1.677, +0.068] | 0.0348 |
| G2 | 0-6.5 | 1,104 | 57.61% | 57.25% | 20 | -0.362 | [-1.170, +0.385] | 0.1856 | [-1.347, +0.832] | 0.2448 |
| G2 | 7 | 74 | 50.00% | 48.65% | 1 | -1.351 | [-4.348, 0.000] | 0.1809 | [-3.896, 0.000] | 0.1666 |
| G2 | 7.5-10 | 194 | 51.55% | 48.97% | 5 | -2.577 | [-5.000, -0.526] | 0.0025 | [-4.396, -0.990] | 0.0009 |
| G2 | 10.5+ | 131 | 51.15% | 48.85% | 5 | -2.290 | [-5.839, +0.781] | 0.0837 | [-4.000, -0.741] | 0.0083 |
| G3 | overall | 1,503 | 55.89% | 55.82% | 11 | **-0.067** | [-0.466, +0.332] | **0.3654** | [-0.409, +0.476] | 0.3221 |
| G3 | 0-6.5 | 1,104 | 57.61% | 57.61% | 8 | 0.000 | [-0.450, +0.453] | 0.4909 | [-0.355, +0.498] | 0.4723 |
| G3 | 7 | 74 | 50.00% | 50.00% | 0 | 0.000 | [0, 0] dead heat | 0.5000 | [0, 0] | 0.5000 |
| G3 | 7.5-10 | 194 | 51.55% | 50.52% | 2 | -1.031 | [-2.632, 0.000] | 0.0667 | [-2.083, 0.000] | 0.0453 |
| G3 | 10.5+ | 131 | 51.15% | 51.91% | 1 | +0.763 | [0.000, +2.459] | 0.8174 | [0.000, +1.899] | 0.8330 |

### Standalone, before the card

| arm | scope | n | incumbent | candidate | picks changed | delta (pts) | 95% week-blocked | `probability_positive` | 95% season-blocked | season P+ |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | overall | 1,503 | 54.56% | 54.29% | 20 | -0.266 | [-0.870, +0.335] | 0.2013 | [-0.580, +0.070] | 0.0767 |
| G1 | 0-6.5 | 1,104 | 56.16% | 56.16% | 12 | 0.000 | [-0.625, +0.636] | 0.4964 | [-0.348, +0.405] | 0.4989 |
| G1 | 7 | 74 | 51.35% | 52.70% | 1 | +1.351 | [0.000, +4.286] | 0.8194 | [0.000, +3.896] | 0.8334 |
| G1 | 7.5-10 | 194 | 51.55% | 50.00% | 3 | -1.546 | [-3.535, 0.000] | 0.0237 | [-2.778, -0.490] | 0.0076 |
| G1 | 10.5+ | 131 | 47.33% | 45.80% | 4 | -1.527 | [-4.724, +1.471] | 0.1562 | [-3.049, 0.000] | 0.0437 |
| G2 | overall | 1,503 | 54.56% | 53.29% | 45 | -1.264 | [-2.188, -0.392] | 0.0030 | [-1.937, -0.343] | 0.0047 |
| G2 | 0-6.5 | 1,104 | 56.16% | 55.25% | 30 | -0.906 | [-1.882, 0.000] | 0.0302 | [-1.810, +0.196] | 0.0532 |
| G2 | 7 | 74 | 51.35% | 52.70% | 3 | +1.351 | [-2.899, +6.173] | 0.7109 | [-3.448, +8.571] | 0.6434 |
| G2 | 7.5-10 | 194 | 51.55% | 48.45% | 6 | -3.093 | [-5.700, -1.010] | 0.0009 | [-5.208, -1.000] | 0.0009 |
| G2 | 10.5+ | 131 | 47.33% | 44.27% | 6 | -3.053 | [-6.923, 0.000] | 0.0426 | [-6.088, -0.741] | 0.0083 |
| G3 | overall | 1,503 | 54.56% | 54.16% | 18 | -0.399 | [-0.925, +0.133] | 0.0662 | [-0.842, +0.143] | 0.0746 |
| G3 | 0-6.5 | 1,104 | 56.16% | 55.71% | 13 | -0.453 | [-1.049, +0.093] | 0.0641 | [-0.987, +0.096] | 0.0435 |
| G3 | 7 | 74 | 51.35% | 51.35% | 0 | 0.000 | [0, 0] dead heat | 0.5000 | [0, 0] | 0.5000 |
| G3 | 7.5-10 | 194 | 51.55% | 50.00% | 3 | -1.546 | [-3.365, 0.000] | 0.0241 | [-2.577, -0.505] | 0.0078 |
| G3 | 10.5+ | 131 | 47.33% | 48.85% | 2 | +1.527 | [0.000, +3.968] | 0.9321 | [0.000, +3.053] | 0.9559 |

### The mechanism, and the finding that outlives the score

This is the part worth keeping. Mean per-family ridge contribution **toward the
favourite**, in points, on the 1,525 lined archive games
(`family_lean_by_bucket.csv`), incumbent versus G3:

| bucket | arm | results | elo | offense | defense | **market** | net residual toward favourite |
|---|---|---|---|---|---|---|---|
| 0-6.5 | incumbent | +1.031 | -0.181 | -0.209 | -0.141 | **-0.234** | **+0.065** |
| 0-6.5 | G3 | -0.079 | +0.001 | -0.008 | +0.079 | **+0.267** | **+0.045** |
| 7 | incumbent | +2.484 | -0.397 | -0.515 | -0.579 | **-0.425** | **+0.305** |
| 7 | G3 | +0.441 | -0.078 | -0.126 | -0.168 | **+0.529** | **+0.301** |
| 7.5-10 | incumbent | +2.975 | -0.543 | -0.616 | -0.544 | **-0.546** | **+0.486** |
| 7.5-10 | G3 | +0.396 | -0.124 | -0.143 | -0.021 | **+0.645** | **+0.476** |
| 10.5+ | incumbent | +4.200 | -0.729 | -1.100 | -0.849 | **-0.785** | **+0.442** |
| 10.5+ | G3 | +0.440 | -0.133 | -0.400 | -0.093 | **+0.954** | **+0.417** |

G3 does exactly what it was designed to do. At 10.5+ the `results` block's
favourite lean falls from **+4.200 to +0.440** points, `offense` from -1.100 to
-0.400, `defense` from -0.849 to -0.093, and the whole 47-column team-quality
block's net contribution goes from **+1.522 to -0.186**. The +/-4-point
cancellation the diagnosis identified is genuinely dismantled.

And the quantity that decides the pick does not move. The net favourite-ward
residual at 10.5+ is **+0.442 (incumbent), +0.437 (G1), +0.443 (G2), +0.417
(G3)**; at 7.5-10 it is **+0.486 / +0.472 / +0.458 / +0.476**. The favourite
share of picks at 10.5+ is **60.45% / 61.19% / 59.70% / 58.96%**, and at 7.5-10
**57.95% / 58.46% / 57.95% / 58.46%**. The reason is in the `market` column of
the table above: it **flips sign**, from -0.785 to +0.954 at 10.5+. Strip the
line's linear share out of all 47 team-quality columns and the ridge rebuilds
the identical lean out of `spread_line` itself, which is the one column left
that still carries it.

So the correct statement of the mechanism, after acting on the whole block as
the diagnosis asked, is one step stronger than the diagnosis's:

> The favourite-ward lean on big spreads is not attached to any block of
> columns. It is what the ridge learns from the TARGET, and it is rebuilt from
> whichever proxy for the line is left free. Re-parameterising which columns
> carry it changes the size of the cancellation (from +/-4.2 points to
> +/-1.0) without changing the half-point remainder that reaches the card.

The group penalties say the same thing more bluntly: penalising the block
**ten times harder** (G2) moves the `results` lean at 10.5+ only from +4.200 to
+3.832 and the net residual lean from +0.442 to +0.443. Inside a collinear
block a group penalty redistributes; it does not delete.

Two consequences worth carrying to the next arm. First, an intervention that
leaves `spread_line` free cannot remove the lean, because the lean is exactly
what a free line column is able to express — so the next arm should either
constrain the line column's coefficient too, or change the TARGET the ridge is
trained on rather than the parameterisation. Second, G3's per-bucket signs are
the ones a mechanism story would predict if anything survived: standalone,
+1.527 points at 10.5+ (P+ 0.9321 week, 0.9559 season) against -1.546 at
7.5-10 (P+ 0.0241) on 2 and 3 changed picks. Those are 131- and 194-game cells
moved by a handful of games; they are recorded and left open, not read as two
findings.

### Probability quality

Brier score and log loss of the served home-cover probability at the opener
(1,503 non-push games), and of the card's probability once the three overlays
mirror the flipped games:

| arm | Brier (model) | log loss (model) | Brier (card) | log loss (card) |
|---|---|---|---|---|
| incumbent | 0.25158 | 0.69662 | 0.24877 | 0.69080 |
| G1 | 0.25166 | 0.69679 | 0.24873 | 0.69073 |
| G2 | 0.25193 | 0.69737 | 0.24882 | 0.69091 |
| G3 | 0.25146 | 0.69638 | 0.24874 | 0.69074 |

Every arm sits within 0.0005 Brier of the incumbent, so none of them changes
the calibration either. Mean stated confidence is 55.76% (incumbent), 55.82%
(G1), 55.90% (G2) and 55.81% (G3) overall, and 56.2-56.4% at 7.5-10 and
56.0-56.2% at 10.5+ for every arm — the flat, bucket-blind confidence the
diagnosis flagged is untouched by all three.

### Decision

**Keep the incumbent card.** At the opener, on the forced-pick pool's own
grade, the four cards score **55.89% (incumbent), 55.82% (G3), 55.76% (G1) and
55.02% (G2)** on the same 1,503 non-push games. Playing G3 instead of the
incumbent is a bet at `probability_positive` **0.3654** (season-blocked
0.3221), G1 at **0.3231** (0.2039) and G2 at **0.0095** (0.0348). All three are
the wrong side of an expected-value call, which is the only bar that governs
which card is played. Nothing here rests on an interval containing zero: the
point estimates and the `probability_positive` values themselves favour the
incumbent, and G3 stays open as a mechanism.

Nothing in this lane was promoted, no ledger was touched and the active
manifest was not edited. Had a candidate won through the card, promoting it
would require: (1) a named feature profile or penalty configuration wired into
`nfl_ats.margin` and into the weekly refit path (`weekly-run`) so the served
model fits with it; (2) editing `artifacts/active_ats_model.json`
(`feature_profile` / `model_id` / `feature_table_sha256` /
`historical_evaluation`); (3) regenerating the opener evaluation under the new
model; (4) re-running the overlay composition, because the three card members'
triggers are recomputed against the incoming card; (5) regenerating the weekly
forecast and refreshing `CURRENT_PREDICTIONS.md` via
`nfl-ats publish-predictions`; and (6) `nfl-ats publish-board`, which fails
closed when a headline number's source names a different model.

Which Week 1 2026 picks would change: **none, under any of the three arms.**
Fitting each arm on the 4,431 completed regular-season games before the
2026-09-09 cutoff and predicting the 16 Week 1 games gives an identical raw
pick on every game for the incumbent, G1, G2 and G3; the largest residual gap
between the incumbent and G3 is 0.14 points, on DEN at KC. That is consistent
with the archive, where G3 changes 11 of 1,503 picks through the card over six
seasons.

### Registry

All 30 cells are recorded under family
`mod18_spread_hole_group_penalty_v1`, named
`spread_hole_<arm>_<card|standalone>_<scope>_2020_2025`. **Three** are
terminal: `spread_hole_g2_standalone_overall_2020_2025`,
`spread_hole_g2_standalone_7p5_10_2020_2025` and
`spread_hole_g2_card_7p5_10_2020_2025` — for each, both the week-blocked and
the season-blocked interval sit **entirely below zero**, which is the one
admissible closing ground (`wrong_sign_resolved`): a 10x group penalty is
resolved worse than the incumbent there. The other **27** are
`unresolved_below_power` and report `probability_positive`.

Three cells are exact dead heats (0 picks changed at a line of exactly 7:
`spread_hole_g1_card_7`, `spread_hole_g3_card_7`,
`spread_hole_g3_standalone_7`). Their bootstrap standard error is identically
zero, which the recorder rejects as non-positive, so they are recorded without
a `standard_error` and carry `probability_positive` 0.5 — the dead-heat value
under the 2026-09-08 `P(>0) + 0.5*P(==0)` convention. That is a recorder
limitation on a zero-variance cell, not a wrong verdict.

## Commands run

```
python scripts/spread_hole_group_penalty.py \
  --features data/processed/game_features_weak_stack.parquet \
  --data-root data --market-root data/market/raw \
  --archive artifacts/opener_evaluation/20260909T183120Z \
  --out artifacts/spread_hole_group_penalty/20260909T224706Z

nfl-ats weak-signals record ...   (30 cells, mod18_spread_hole_group_penalty_v1;
the exact argv lists are saved as record_commands.json and
record_commands_deadheat.json in the artifact directory)
```
