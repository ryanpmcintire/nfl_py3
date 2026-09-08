# Why the model is weak on spreads of 10.5 or more (MOD-18 lane L, 2026-09-08)

A diagnosis, not a fix. Nothing here changes a pick; the owner's rule
(AGENTS.md, "No unexplained threshold flips on the played card") is that a
dip located at a spread threshold is something to EXPLAIN and publish, never
a flip to bolt on. Every number below is **measured** this session by
`.\.tools\uv.exe run --no-sync python scripts\big_spread_diagnosis.py`
(tables under `artifacts/research/laneL/`: `cells`, `crosses`, `ladder`,
`lean`, `residual_information`, `joined`, `summary.json`) on the active
model `a4c757efd2525da6`'s matching opener evaluation
`artifacts/opener_evaluation/20260908T115957Z` (1,537 games 2020-2025, the
served read after the S3 home-side offset, policy
`home_side_offset_big_spreads_v2`), joined to the pregame columns of
`data/processed/game_features_weak_stack.parquet` and the
`data/raw/20260905T211016Z/schedules.parquet` snapshot. Intervals are 95%
week-blocked bootstraps, 20,000 draws, seed 20260817, whole weeks resampled,
within-week correlation zero (never estimated or padded). The archive is the
same mined 1,537 games every MOD-18 lane has scored, so the intervals
describe the archive; none of the cells is a resolved finding and none is
closed. Where a claim is my reasoning rather than a measurement it is
marked **inferred**.

## In one paragraph, for a pool player

On lines of 10.5 points or more the model has been right 47% of the time
(62 of 131 decided games since 2020) while saying it was about 56% sure.
The reason is not that it backs the wrong kind of team in general -- its
picks of the home team on these lines are right 55% of the time. The whole
hole is its picks of the ROAD team: 37% right in 54 games, and worse the
bigger the line (14% in the 14 games at 13-14 points) and worse still when
the betting line moved against the pick between Tuesday and kickoff (24%).
On these games the home team beats the model's own point forecast by about
4.6 points on average. What the model is doing in those games is reading a
big spread as a mistake: its team ratings say the gap between the two teams
is smaller than the line, so it leans to the road side. On 10.5+ lines that
argument has lost more often than it has won -- the market's big number has
usually known something the ratings do not (a quarterback, an injury, a
situation), and the model's disagreement with the market points the wrong
way there. The small push toward the home team added this week helps (it
turned 4 of these games from wrong to right) but removes only about a third
of the home team's remaining edge on these lines; the rest of the problem is
the model still trusting its own disagreement on the biggest spreads.

## 1. What the model does on 10.5+ (131 decided games, 134 with pushes)

| read | 10.5+ | for comparison, 7.5-10 (194) |
|---|---|---|
| model right (served, after S3) | **47.3%** [39.0, 55.5], P(above 50%) 0.25 | 51.5% [44.1, 58.7], P 0.65 |
| model right, raw (before S3) | 44.3% | 48.5% |
| stated confidence | 56.0% | 56.2% |
| picks the favourite | 60.3% | 58.2% |
| favourites actually cover | 53.4% | 45.9% |
| picks the home team | 59.0% | 42.1% |
| home team actually covers | **58.0%** | 52.6% |
| model's average home probability | 51.7% | 48.8% |
| home margin minus served point | **+1.97** [-0.12, +4.08], P+ 0.97 | +0.66 [-1.45, +2.80], P+ 0.72 |
| same, raw point (before S3) | +2.91 [+0.83, +5.01] | +1.08 |
| served offset actually applied (mean) | +0.94 | +0.42 |

By home side at 10.5+: home favourite (99 decided) right 46.5% [35.9, 57.0],
home covered 57.6% against the model's 52.3%, point error +1.48 [-0.83,
+3.74] P+ 0.90 (raw +2.41); home underdog (32) right 50.0% [33.3, 66.7],
home covered 59.4% against 50.0%, point error +3.53 [-1.66, +8.46] P+ 0.91
(raw +4.50). The home team out-performs the served point on BOTH sides, as
lanes Q and S found on the raw point; S3 has removed roughly a third of it.

## 2. Where the errors sit (10.5+ unless stated)

**By which team the model picked (the decisive cut).**

| pick | n | right | 95% | P(above 50%) | home margin minus served point |
|---|---|---|---|---|---|
| picked the home team | 77 | 54.5% | [44.2, 64.9] | 0.79 | +0.12 [-2.66, +2.86] |
| picked the road team | 54 | **37.0%** | [25.0, 49.1] | 0.02 | **+4.63 [+1.29, +7.89]**, P+ 1.00 |

Road picks split as: home favourite faded (37): 35.1% [21.1, 51.4], error
+4.33 [+0.84, +7.66]; road favourite backed (17): 41.2% [18.8, 64.7], error
+5.29 [-2.57, +12.13]. Home picks: home favourite backed (62): 53.2%; home
underdog backed (15): 60.0%. In the small buckets road picks are fine (0-3:
56.7% on 326; 3.5-6.5: 56.8% on 317); at 7.5-10 they are 49.1% (112) against
54.9% for home picks -- a faint version of the same tilt.

**By line size and key number.** 10.5-12.5 (72): 55.6% [43.8, 66.7];
**13-14 (41): 34.1% [22.0, 46.3]**, P 0.003; 14.5 or more (18): 44.4% [18.8,
66.7]. Crossed with the pick: 13-14 and picked the road team (14): 14.3%
[0.0, 35.7], home beats the served point by +9.45 [+3.19, +15.56]; 13-14 and
picked the home team (27): 44.4% [29.4, 60.0], error -1.10. Nearest key
number: on or beside 10 (38) 50.0%; on or beside 14 (31) 38.7% [22.6, 54.5];
on or beside 17 (7) 14.3%. The 13-14 cell is where two touchdowns sit;
**inferred**: the accuracy collapse there is the road-pick problem
concentrated, not a separate key-number effect -- the home picks at 13-14
are merely poor (44%), the road picks are near-certain losers.

**By open-to-close line movement relative to the pick.** Market moved
against the model's side (33): **27.3%** [12.9, 41.2], P 0.001; moved with it
(76): 59.2% [48.5, 69.6], P 0.95; no move (22): 36.4% [17.4, 57.1]. Crossed:
against and road pick (17): 23.5%, home beats the served point by +8.75
[+3.37, +14.94]; against and home pick (16): 31.2%, error -5.08 [-8.70,
-0.68]; with and home pick (51): 66.7% [54.0, 79.2]. The same against/with
gap exists in every bucket (0-3: 54.8% vs 58.3%; 3.5-6.5: 48.0% vs 61.1%;
7.5-10: 46.5% vs 53.5%) but at 10.5+ it is 32 points wide, two to eight
times the others. Movement is not known at the Tuesday line lock; it is
known before the pick deadline.

**By the model's own residual.** Residual leans to the underdog (41): 43.9%
[30.0, 58.0], home beats the served point by +5.43 [+1.31, +9.59] P+ 0.995;
leans to the favourite (90): 48.9%, error +0.44 [-2.05, +2.85]. Leans
underdog and picked the road team (23): 30.4% [12.5, 50.0], error +6.49
[+2.16, +10.89]. Residual size: under 1 point (47) 48.9%; 1-2 (32) 43.8% with
error +6.12 [+1.41, +11.13]; 2-3 (29) 51.7%; 3 or more (23) 43.5% with error
-2.21 [-7.16, +3.21]. Stated confidence: 50-52% (28) 57.1% [37.0, 75.9]; 52%
and above (103 games pooled from the three higher bands) 44.7%. So the
MARGINAL picks are fine and the confident ones lose: the model's stated
edge on big spreads is anti-informative.

The cleanest statement of this is the slope of the actual margin against
the opener on the model's served residual (1 = the residual is right in size,
0 = it says nothing, negative = it points the wrong way), from
`residual_information.parquet`:

| bucket | decided | slope (served residual) | 95% | P(slope > 0) | model minus always-home | 95% |
|---|---|---|---|---|---|---|
| 0-3 | 552 | +0.19 | [-0.34, +0.71] | 0.77 | +8.0 pts | [+1.7, +14.3] |
| 3.5-6.5 | 552 | +0.53 | [-0.03, +1.06] | 0.97 | +7.8 | [+0.4, +15.0] |
| 7 | 74 | +0.20 | [-1.31, +1.72] | 0.61 | +4.1 | [-13.2, +21.7] |
| 7.5-10 | 194 | +0.47 | [-0.33, +1.23] | 0.87 | -1.0 | [-12.4, +10.2] |
| 10.5+ | 131 | **-0.99** | [-1.98, +0.02] | **0.03** | **-10.7** | [-21.0, -0.7] |

The always-home column is a BASE RATE for reading the model against (home
covers 58.0% of decided 10.5+ games), not a rule; publishing it as a flip is
exactly what the owner's rule forbids. What it shows is that inside the
10.5+ bucket the model's per-game disagreement with the line subtracts
information: the raw residual's slope is -0.85 [-1.87, +0.15] and the
served residual's -0.99 [-1.98, +0.02], against +0.5 in the 3.5-6.5 and
7.5-10 buckets.

**What the residual is made of when it leans to the underdog** (from
`lean.parquet`, 10.5+; home-minus-away features, so on the 76% of these
games where the home team is favoured a small value means the ratings see a
small gap): Elo difference 54 when leaning underdog versus 138 when leaning
favourite; season point-differential gap 1.3 versus 9.1; offensive EPA gap
0.011 versus 0.129; expected QB EPA gap 0.04 versus 0.18. None of those
features predicts the served error itself (correlations between -0.09 and
+0.09). **Inferred**: the model leans to the road side on a big spread when
its ratings say the two teams are closer than the line -- it reads the big
number as market error -- and the market is right about it far more often
than not.

**By rest and travel.** Favourite on normal 7-day rest (76): 44.7% [34.2,
55.4], home beats the served point by +3.22 [+0.15, +6.36] P+ 0.98; favourite
on 8 or more days (35): 54.3% [38.2, 69.7], error -2.21 [-5.75, +1.25] P+
0.10; favourite on 6 or fewer (20): 45.0%, error +4.58 [-1.02, +10.79]. When
the favourite rested MORE than the underdog (30): 46.7%, home error -3.53
[-7.38, +0.27] P+ 0.03; equal rest (79): 48.1%, error +3.97 [+0.97, +6.98] P+
0.995. Travel: the favourite is the home team in 116 of 131 decided games,
so a travelling favourite is rare (one zone 8 games, two or more 7 games) and
no travel cut has enough games to read; there were no neutral-site 10.5+
games in the archive.

**By week of season.** Weeks 1-6 (29): 41.4% [24.1, 58.6], error +3.89
[-0.17, +8.30] P+ 0.97; weeks 7-12 (45): 57.8% [44.7, 70.8], error +3.22;
weeks 13-18 (57): 42.1% [29.1, 54.4], P 0.10, error +0.09 [-3.24, +2.98].
Weeks 15-18 (42, the playoff-race proxy -- no playoff-race column exists in
the feature table or the schedules snapshot, **read** from their column
lists): 45.2% [30.6, 58.8], error +0.37 [-3.59, +3.48]. Crossed with the
pick: late-season home picks (41) 46.3% with the home team falling SHORT of
the served point by -2.15 [-5.90, +1.31] P+ 0.11; late-season road picks
(16) 31.2%, error +5.77. So late in the season the home-location error
disappears and a second, smaller weakness appears in home-favourite picks;
**inferred**: teams with nothing to play for and rested starters are the
obvious candidate, and it cannot be tested without a playoff-race column.

**By season.** 2020 (20 decided): 50.0%, home error -0.86 [-5.69, +3.79] P+
0.36 with the served offset about zero; 2021 (34): 44.1%, error **+4.73
[+1.22, +8.00]** P+ 0.995; 2022 (19): 36.8%, +0.22; 2023 (21): 38.1%,
+1.47; 2024 (13): 38.5%, +2.17; 2025 (24): 70.8% [53.6, 88.0] P 0.99, +2.12.
The 2020 no-crowd season shows no home edge on big spreads; it returned in
2021 at nearly five points, and because the served offset is walk-forward
with a fixed 100-game prior, 2020 sat inside the training window for every
later fit and held the correction down (lane S measured the fitted 10.5+
offset at 0 in 2020, +0.5 in 2021, +1.2 in 2022, +1.5 by 2024).

**Other cuts.** Non-division games (78): 46.2%, home error +3.51 [+0.84,
+6.22] P+ 0.996; division games (53): 49.1%, error -0.25 [-4.13, +3.61] --
but at 7.5-10 the division cut runs the other way (+1.14 division, +0.35
non-division), so it does not replicate across the two buckets and is not
carried forward. Indoors (28): 39.3% [22.2, 56.3], error +4.43 [-0.36, +9.45];
outdoors (103): 49.5%, +1.29 -- reversed at 7.5-10 (-1.90 indoors, +1.90
outdoors), same verdict. Primetime (22): 59.1%, error -0.56; day games (109):
45.0%, +2.46 [-0.03, +4.90]. Favourite's starting QB in doubt (22): 54.5%,
error +3.10; expected (109): 45.9%, +1.73. Underdog's QB in doubt (32): 40.6%,
error -0.97; expected (99): 49.5%, +2.89 [+0.35, +5.37].

## 3. The point error after S3, up the spread ladder (`ladder.parquet`)

Home margin minus the served point, pushes included; the raw (pre-S3) error
beside it; the model's accuracy on decided games in the band.

| opening line | games | raw error | served error | 95% (served) | P+ | served offset | right |
|---|---|---|---|---|---|---|---|
| 0-3 | 573 | -0.10 | -0.10 | [-1.16, +0.96] | 0.42 | 0.00 | 56.2% |
| 3.5-6.5 | 558 | +0.11 | +0.11 | [-0.87, +1.13] | 0.58 | 0.00 | 56.2% |
| 7 | 77 | -0.91 | -0.56 | [-3.78, +2.63] | 0.36 | -0.35 | 51.4% |
| 7.5-9 | 138 | +0.69 | +0.27 | [-2.26, +2.72] | 0.58 | +0.42 | 50.4% |
| 9.5-10 | 57 | +2.03 | +1.59 | [-2.18, +5.15] | 0.80 | +0.44 | 54.4% |
| 10.5-12.5 | 72 | +2.45 | +1.52 | [-1.50, +4.55] | 0.83 | +0.93 | 55.6% |
| 13-14 | 43 | +3.58 | +2.58 | [-1.40, +6.81] | 0.90 | +1.00 | **34.1%** |
| 14.5 or more | 19 | +3.10 | +2.25 | [-4.07, +7.97] | 0.76 | +0.85 | 44.4% |

By home side (raw error, home favourite / home underdog): 7.5-9 +0.47 /
+1.07; 9.5-10 +2.36 / +1.31; 10.5-12.5 +2.18 / +3.51; 13-14 +3.62 / +3.50;
14.5+ +0.87 (14 games) / +11.44 (4 games).

## 4. One phenomenon or two?

Two, and they should be worked separately.

- **The point-location error is ONE phenomenon shared by 7.5-10 and 10.5+.**
  The raw home error rises smoothly with line size -- about zero through 9,
  +2.0 at 9.5-10, +2.5 at 10.5-12.5, +3.6 at 13-14 -- and it is present on
  both home sides at every step. That is what S3 corrects, and S3's bucket
  offsets (+0.42 and +0.94 on average) remove under half of it because they
  are shrunk toward zero with a fixed 100-game prior and fitted on a window
  that includes the no-crowd 2020 season.
- **The accuracy collapse is NOT shared.** 7.5-10 after S3 is 51.5% and
  roughly flat across its cuts (its road picks 49.1%, its 13-14 equivalent
  does not exist, its against-the-move cell 46.5%), and its residual still
  carries positive information (slope +0.47, P+ 0.87). 10.5+ has a specific
  hole -- road picks 37%, 13-14 lines 34%, against-the-move 27% -- and its
  residual points the wrong way (slope -0.99, P+ 0.03). The 10.5+ accuracy
  problem is the model's DISAGREEMENT with big lines being wrong, on top of
  the shared location error.

## 5. Ranked mechanisms for a future lane (predeclarable; none is a flip)

1. **The residual's weight should fall with spread size, because on 10.5+
   lines the ridge's disagreement with the market is anti-informative.**
   Columns: `residual_at_open` (the model's disagreement, in points) and
   `tue_open_home_spread` (its size), from the opener evaluation; on the
   forecast, the same two quantities in `MarginModel.predict`. Direction:
   a walk-forward slope of the actual margin-against-the-line on the raw
   residual, fitted per lane-J bucket on prior games only (C3's calibration
   layer applied to the POINT, not to the probability), will come out near
   +0.5 below 10.5 and at or below zero at 10.5+, so the served point on big
   spreads becomes the line plus the home-side offset with the residual
   scaled toward nothing; picks then follow the offset's home lean rather
   than the ratings' road lean. **Inferred** football reason: a spread of
   two touchdowns is set with information (quarterback, injuries,
   situation) that season-long ratings do not carry, and the ratings' "the
   gap is smaller than that" is the wrong side of that trade. A positive
   control exists for this lane: the same slope in the 3.5-6.5 bucket
   (+0.53, P+ 0.97) must stay positive under the same procedure.
2. **Post-lock line movement on big spreads belongs in the late-week
   refresh, not in the Tuesday model.** Column: `open_move` (close minus
   Tuesday opener, home frame) signed toward the served pick, on lines of
   10.5 or more; on the card, the same quantity from the frozen Tuesday line
   and the latest captured line before the pick deadline. Direction: a move
   AGAINST the served pick predicts a loss (27.3% [12.9, 41.2] on 33 games,
   against 59.2% with the move), with the home team beating the served point
   by +8.75 when the pick was the road team. The refresh channel
   (`pick_refresh`) already exists and the pick deadline is min(kickoff,
   Sunday 4 PM ET), so the information is usable; the movement-attribution
   finding that such moves are mostly injury news is **reported** from the
   session memory and not verified here. Honest caveat: the same direction
   exists in every bucket, so the lane should predeclare 10.5+ as the
   primary cell and the whole archive as the secondary.
3. **The home-side offset is under-sized on big spreads, for two derivable
   reasons.** Columns: `season` (the 2020 no-crowd season inside the
   five-season training window) and the constant `PRIOR_WEIGHT_GAMES = 100`
   in `src/nfl_ats/home_side_location.py` (an underived constant; owner's
   rule that such constants are defects until derived). Direction: the
   remaining served error at 10.5+ is +1.97 [-0.12, +4.08] P+ 0.97 with a
   mean served offset of +0.94 against a raw error of +2.91; the 2020 cell
   reads -0.86 (P+ 0.36) and 2021 +4.73 [+1.22, +8.00]. A prior weight
   derived from the between-bucket variance of the home error (empirical
   Bayes) and a 2020 down-weighting will both move the offset up at 7.5+;
   the predeclared read is the served error at 10.5+ shrinking toward zero
   and the composed-card accuracy, at the opener, week-blocked.

Not carried forward, stated so the next lane does not rediscover them: the
division, roof, rest-edge and QB-availability cuts each show a large cell at
10.5+ that reverses or vanishes at 7.5-10; the travel cuts have under ten
games each; a playoff-race column does not exist and would be needed to
test the late-season home-favourite weakness.
