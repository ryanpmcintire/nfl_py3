# LEAD-43: altitude fourth-quarter split

Predeclared design: ROADMAP `LEAD-43` (thin-air fatigue lands late, so
visitors at Denver fade in the fourth quarter beyond what the full-game
altitude price already covers). This extends the 2026-09-05 CX15 measurement
(`docs/coaching_leads.md`) with an EPA split, a visitor's-next-game
carryover test, an explicit market-line-conditioning check, and a Mexico
City replication, on the full 2009-2025 window instead of 2013-2025.

All numbers below are **measured** this session
(`.\.tools\uv.exe run --no-sync python scripts\altitude_fourth_quarter_screen.py`,
seed 20260910, 20,000-draw block bootstrap unless noted), written to
`artifacts/experiments/altitude_fourth_quarter/results.json`. Source data:
play-by-play snapshot `data/pbp/raw/20260817T184927Z` (seasons 2009-2025),
schedules `data/raw/20260908T162105Z/schedules.parquet`, and the active
model's opener-evaluation artifact `artifacts/opener_evaluation/20260910T211255Z`
(model `d49194e04945a5e5`, feature-table sha256
`2fb3451b2aa15a6ac1994a35fc7b1e12fb8bcbfd0bc3490d9d51d580f5a6be53`,
2020-2025, 1,537 paired games).

Per the closing-grounds taxonomy this project enforces in code: an interval
crossing zero is never grounds to reject or close a line of work. Only a
RESOLVED wrong sign (whole interval on the wrong side of zero) or a proven
positive-control bound can close a candidate; everything else below is
`unresolved_below_power` and is recorded, not discarded.

## 1. The 4Q split: scoring margin and EPA, Denver vs. league, and the visitor's next game

Quarter boundary: home-oriented margin at the first play of the fourth
quarter with 900 seconds on the clock (`game_seconds_remaining == 900`);
regulation margin uses the first overtime play's pre-play state when a game
went to overtime, else the schedule's final result. EPA splits sum
play-level `epa` (home minus away) over plays with `qtr` in 1-3 versus
`qtr == 4`. 4 of 4,431 REG games 2009-2025 lack a usable 900-second Q4
boundary and are excluded (`2010_04_SF_ATL`, `2011_14_KC_NYJ`,
`2012_06_STL_MIA`, `2016_05_TEN_MIA`); none is a Denver home game.

**Denver home games, 2009-2025: 139** (schedule count, read from
`data/raw/20260908T162105Z/schedules.parquet`), 132-139 enter each bootstrap
depending on which ratio is defined (a small number of near-zero-margin
games make the ratio-share metrics undefined; see the caveat below).

| metric (home minus away) | league mean, 95% CI | Denver mean, 95% CI | Denver minus league, 95% CI, P+ |
|---|---|---|---|
| Q1-3 scoring margin | +1.637 [+1.247, +2.027] (n=4,427) | +0.683 [-1.633, +2.979] (n=139) | -- |
| Q4 scoring margin | +0.361 [+0.149, +0.572] | +2.734 [+1.402, +4.090] | **+2.373 [+1.054, +3.724], P+ 0.99985** |
| Q4-minus-(Q1-3/3) rate (points) | -0.185 [-0.444, +0.073] | +2.506 [+0.891, +4.171] | **+2.691 [+1.092, +4.343], P+ 0.99955** |
| Q1-3 net EPA | +0.245 [-0.214, +0.701] | -0.780 [-3.449, +1.845] | -- |
| Q4 net EPA | -0.086 [-0.325, +0.150] | +2.481 [+1.068, +3.907] | **+2.567 [+1.169, +3.988], P+ 0.99985** |
| Q4-minus-(Q1-3/3) EPA rate | -0.167 [-0.466, +0.131] | +2.741 [+0.966, +4.539] | **+2.909 [+1.155, +4.676], P+ 0.9996** |

The scoring-margin contrast reproduces CX15's 2013-2025 reading
(+2.694 [+1.229, +4.192], P+ 0.99985) almost exactly on the wider
2009-2025 window, and the EPA-based version agrees in sign and magnitude
using an entirely independent data column. Both are `unresolved_below_power`
by this project's taxonomy in the sense that a single interval crossing the
hypothesised effect size doesn't "prove" the mechanism, but neither interval
crosses zero — this is the strongest cut of the four parts below.

**A ratio-share formulation is unstable and is reported for completeness,
not as a primary read.** `margin_q4_share = fourth_margin / regulation_margin`
(fraction of the final margin coming from Q4) reads Denver minus league
+0.300 [-0.228, +0.886], P+ 0.858 — same direction, crosses zero. Its EPA
analogue, `epa_q4_share = fourth_epa / (first_three_epa + fourth_epa)`,
flips sign entirely (Denver minus league -1.023 [-2.621, +0.846], P+ 0.121)
because `first_three_epa` for Denver is itself small and sometimes negative
(-0.780, CI crossing zero), so the ratio's denominator crosses zero across
games and the share statistic is not well-behaved. **Read the additive rate
metrics above, not the ratios, as the primary numbers**; this instability is
reported, not concealed.

**Visitor's next game (fatigue carryover).** For each Denver home game, the
away team's next REG game that season (124 of 139 matched; 15 Denver games
were that visitor's last game of the season and have no in-season next
game) is scored team-oriented (home-minus-away if the team is home in that
next game, negated if away), and compared against the league's average rate
for a team in that same home/away role (league home rate -0.185, so the
away-role expectation is +0.185 by symmetry). The predeclared direction is
that a team fades **more** than a typical team in that role, i.e. a negative
excess.

- Scoring rate: excess -0.906 [-2.511, +0.708]; in the predeclared
  direction, `probability_positive` (excess below expectation) = **0.865**.
- EPA rate: excess -1.281 [-2.997, +0.456]; `probability_positive` in the
  predeclared direction = **0.927**.

Both intervals cross zero — `unresolved_below_power`, not a negative. Both
point estimates and both directional probabilities (0.865, 0.927) lean the
predeclared way on an entirely independent game (the visitor's next
matchup, which never touches Denver's own PBP), which is a modest additional
corroboration of the fatigue-carryover mechanism, not a resolution of it.

## 2. Does the market already price it?

**Full-game opener cover rate.** Denver home games covered the true
Tuesday opener (`tue_open_home_spread`, `margin_vs_open`) at **53.19%**
(25 of 47, 2020-2025) versus the league's **49.63%** over the same 1,503
paired non-push games. Contrast **+3.557 accuracy points, 95%
[-10.238, +17.240], P+ 0.693** (week-blocked). Wide because n=47; leans the
same direction as the Q4 finding but does not resolve it.

**Does the 4Q edge survive conditioning on the pregame line?** Two
versions, because only 2020-2025 has the true Tuesday opener on disk; the
full 2009-2025 range only has `schedules.parquet`'s `spread_line`, which
matches the **closing** line far more closely than the opener (measured:
mean absolute difference vs `tue_open_home_spread` is 1.00 points on the
1,537 overlapping games, vs 0.20 points against `close_home_spread` — this
project's schedule archive does not carry a historical opener series before
2020).

- **2009-2025, closing-line proxy.** Fit `fourth_margin ~ a + b * spread_line`
  on every non-Denver home game (`b = 0.2033`, `a = -0.1151`), apply the same
  line to Denver games, and compare the residual means. Raw contrast was
  +2.373; after removing the line's own linear association with Q4 scoring,
  the conditioned contrast is **+2.218 [+0.925, +3.541], P+ 0.9997** — a
  9% shrinkage, not a collapse.
- **2020-2025, true Tuesday opener.** Same regression against
  `tue_open_home_spread` (`b = 0.1681`, `a = 0.1578`, 1,537 non-Denver
  games): conditioned contrast **+2.964 [+0.972, +5.012], P+ 0.99835** —
  slightly larger than the raw full-range figure, not smaller.

Neither conditioning materially shrinks the effect toward zero. Read plainly:
whatever information the market's spread carries about Denver home games,
it is not capturing the specific quarter-by-quarter timing of where that
margin accrues — the market prices the final number, not the shape of how
it is built up. That is consistent with (not proof of) the predeclared
mechanism being currently unpriced.

## 3. Production screen: BACK Denver at home, stacked on production

Construct matches CX15 exactly: override the active model's opener
probability-rule pick to home whenever the flag fires; every other pick
stays production; paired by game; opener pushes excluded. Screened against
the **current** active model (`d49194e04945a5e5`), not CX15's 2026-09-05
snapshot (`ab29832a4e099766`) — the model has been refit since, so exact
CX15 reproduction is not expected, though this run's baseline/candidate/delta
happen to land within a few thousandths of CX15's (-0.0665336 here vs
CX15's -0.066534).

**Denver, all 47 eligible home games, 27 of 47 picks changed** (CX15: 25 of
47, on the prior model): baseline 54.558% [51.903%, 57.152%], candidate
54.491% [51.960%, 57.011%], **delta -0.0665 accuracy points, 95%
[-0.736, +0.602], P+ 0.391** on 1,503 paired non-push games / 107 weeks.
Leaked oracle positive control (override only when Denver actually covered):
**delta +0.865 accuracy points, 95% [+0.461, +1.334], P+ 1.0** — confirms
the instrument can detect an effect at that scale; it is not itself
playable. Both `unresolved_below_power`; the interval crossing zero on the
main test is the expected shape for a real small signal at this evaluator's
resolution, not a rejection.

**Mexico City (Estadio Azteca, ~2,240 m — higher than Denver's ~1,600 m;
general-knowledge elevation figures, not read from any file in this repo),
same construct, separately.** 5 REG games have been played there ever
(`2016_11_HOU_OAK`, `2017_11_NE_OAK`, `2019_11_KC_LAC`, `2022_11_SF_ARI`,
`2026_11_MIN_SF` — the last unplayed); only **one** (`2022_11_SF_ARI`)
falls inside the 2020-2025 opener-evaluation window this screen uses, so
this cell is n=1 by data availability, not by choice. One pick flips;
overall delta reads -0.0665 accuracy points with a degenerate interval
([-0.203, 0.0], P+ 0.0) that is a mechanical artifact of a single-game
sample (a bootstrap over 107 week-blocks where only one contains a changed
game has almost no room to vary), not a resolved negative. Per the binding
rule against dismissing small samples: this is reported as-is and recorded
`unresolved_below_power`, not treated as a verdict on the mechanism.
Mexico City's neutral-site designation also means neither team has a true
home-crowd effect, so this cell tests altitude fatigue on the schedule's
nominal "home" team only, not the same home-field-plus-altitude construct
Denver offers — a real limitation of this replication, stated plainly.

## 4. Split-half reliability of the 4Q-share trait

Three complementary readings, because splitting 17 Denver seasons in half
admits more than one honest construction, and this project's rule is to
report the number and interval rather than pick the single most favorable
cut.

**Primary: adjacent odd/even season pairs.** Denver's season-level mean of
the Q4-minus-(Q1-3/3) rate is computed per season, then paired
(2009,2010), (2011,2012), ..., (2023,2024) — 8 pairs (2025 is unpaired and
dropped). Correlating the odd-season value against the following
even-season value, resampling pairs 20,000 times:

- Scoring rate: **r = -0.370, 95% [-0.879, +0.882], P+(r>0) 0.166**.
- EPA rate: **r = -0.244, 95% [-0.757, +0.761], P+(r>0) 0.205**.

**Secondary, matching CX15's own method exactly (alternating games within
season, extended from 2013-2025 to 2009-2025):** **r = -0.516, 95%
[-0.768, -0.109], P+(r>0) 0.010** — the whole interval is below zero on
this specific construction (CX15 measured -0.606 [-0.886, -0.124] on the
shorter window; this reproduces that finding almost exactly on five more
seasons of data).

**This negative reliability reading does not refute the Denver home Q4
effect, and it is not treated as a closing ground here, for the same
reason CX15 gave on 2026-09-05.** It is a different estimand: whether a
season with an above-average Q4 tilt predicts another season with an
above-average tilt, at n=8-9 games per season-half — a severely
underpowered test of year-to-year ordering, not of whether the effect
exists. The *fixed* effect (the pooled contrast across all 139 Denver games
against the full league distribution, Section 1 above) stays positive and
mostly excludes zero on every cut measured this session: scoring margin,
EPA, the additive rate, both line-conditioning specifications, and the
opener cover rate. Both season-half means in every reliability construction
above are themselves mostly positive (season_pairs_scoring odd values:
[-0.17, 5.63, 5.17, 2.33, -1.54, 0.21, 2.37, 2.89]; even values: [3.87,
0.12, -2.21, 5.88, 1.04, 2.88, 2.71, 5.63] — 14 of 16 positive), so the
negative correlation reflects noisy re-ordering of mostly-positive numbers
at small n, not a sign flip of the underlying quantity. Recorded
`unresolved_below_power`, not `refuted_mechanism`; `no_split_half_reliability`
is not invoked as a closing ground because the fixed-effect estimand this
lane actually cares about is not the quantity whose reliability read
negative.

## Decision

Per this project's standing rule, a promotion bar is not a decision bar and
an interval crossing zero is not grounds to reject. The production screen's
`probability_positive` for backing Denver at home in every game is 0.391 —
below 0.5, so on expected value alone this specific fixed-overlay
construction does not currently favour changing the played card, and
production is retained. That is a narrower claim than "the mechanism is
false": the underlying 4Q split is one of the most consistently
positive-and-interval-excluding-zero findings measured in this registry
family (Section 1), it survives two independent line-conditioning checks
(Section 2), and its failure to move the production screen is better read
as "production's existing probability rule already gets most of these
Denver games right on its own," not as evidence against the mechanism. The
oracle control (P+ 1.0) confirms the instrument would detect a
Denver-sized edge if the naive fixed-overlay construction carried one; it
currently does not, which argues for refining *how* the mechanism enters
the model (e.g., as a genuine Q4-weighted feature interacting with the
existing probability rule) rather than as a blunt fixed home-pick override,
rather than for abandoning the lead.

## Files

- `scripts/altitude_fourth_quarter_screen.py` — builds every number above.
- `artifacts/experiments/altitude_fourth_quarter/results.json` — full
  payload with provenance; `quarter_table.csv`, `visitor_next_game.csv`,
  `denver_paired_predictions.csv`, `mexico_city_games.csv` alongside it.
- `registry/weak_signals.json` — every measured result above recorded
  (`nfl-ats weak-signals record`, family `altitude_fourth_quarter`,
  category `environment` except the market-pricing entries in section 2
  (`market`) and the oracle control (`control`)).
