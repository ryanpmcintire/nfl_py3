# Latent player ratings, expected-lineup aggregated, stacked on PRODUCTION: predeclaration

Lane PER-09. Written **before any ATS number is produced by this comparison**,
following the construction and discipline of
`docs/graph_team_stat_def_ypp_on_production.md` (the precedent for an
"on production" marginal) and `docs/apm_unit_feature_on_production.md` (the
PER-09 sibling that measured the season-lagged unit APM feature the same way).
**Sections 1-7 are the predeclaration** and contain no accuracy, cover-rate,
Brier or `probability_positive` number against NFL outcomes. **Section 8 was
added after the look** and reports what it found; it changes nothing above it.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) **bounded by a
positive control** — the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". A promotion threshold governs only what the docs may
CLAIM; it never governs which card is PLAYED, which is expected value:
`probability_positive` above 0.5 favours playing the candidate.

## 1. What PER-09 already measured, and what is different here

PER-09's first component is the season-lagged offense/defense adjusted
plus-minus in `src/nfl_ats/participation.py`: one ridge per target season over
competitive 11-on-11 participation plays from the previous three completed
seasons, with a reliability shrinkage toward zero by play count. ROADMAP's
PER-09 row records that this rating "is reproducible but failed its matched ATS
screen" — the screen in question is the PER-05 participation feature profile,
which reached **51.71%** against the then-active model's box-score value
profile at **52.14%**, both measured on the whole feature-profile comparison,
not on top of what is played.

Two things about that screen are worth naming, because this document changes
both:

1. **It entered the ratings as an injury-loss channel, not as a lineup.**
   `nfl_ats.players._injury_participation_value_features` sums
   `severity x role share x rating` over the rows on the **injury report only**
   (`injury_offense_participation_value_lost` /
   `injury_defense_participation_value_lost`). That column answers "how much
   rated value is the team missing", a delta. It never says who is expected to
   be on the field, so two teams with identical injury reports and completely
   different rosters are identical to it.
2. **It was graded as a feature-profile swap against a bare-ish baseline.**
   The memory-derived owner rule of 2026-08-26 — restated in
   `docs/graph_team_stat_def_ypp_on_production.md` §1 and in ROADMAP's
   "composition is not the signal" lesson — is that a new feature is tested
   ON TOP OF PRODUCTION, because production already prices team quality.

The PER-09 sibling `docs/apm_unit_feature_on_production.md` fixed (2) for the
unit-level APM and found -0.399 accuracy points, P+ 0.283, on [2020, 2021] /
[2022, 2023] / [2024, 2025]. It did **not** fix (1): its six columns are
team-unit rating aggregates, still roster-shaped rather than lineup-shaped.

This document fixes both at once for the **player-level** ratings. The
candidate is an **expected-lineup rating sum**: for each team-game, the sum
over players of (deadline-visible probability the player plays) x (strictly
prior role share) x (that player's season-lagged rating). It is the sum of
the rated players expected to take the field, not the roster and not the
injury delta.

## 2. Why this could add what a team rating cannot, and where

The owner's standing build filter ("team quality is already priced", carried in
`docs/availability_confirmation.md` and `docs/estimation_variance.md`) says a
feature that only measures
team quality better is bounded near zero, because the spread already contains
it. An expected-lineup rating sum is mostly a team-quality measurement and is
expected to inherit that bound in the average game.

The part that is **not** team quality is the gap between the lineup actually
expected to play and that team's own full-strength lineup: a star ruled out, or
a backup who has taken over a starter's snap share. The market prices the
headline absence; what it prices less reliably is the *rated* size of the gap,
which depends on who the replacement is. So the pre-registered expectation is:

- near-zero on the full card (bounded by the team-quality ceiling), and
- the informative cell is **lineup divergence** — games where the expected
  lineup's rating differs a lot from that team's full-strength rating.

Both are measured. The cell is declared here, before scoring, so it is not a
subgroup found after the fact.

## 3. The ratings, inherited and not refit

`data/processed/player_participation_ratings.parquet`, built by
`nfl-ats build-participation-features` from
`nfl_ats.participation.build_season_lagged_player_ratings` at its frozen
defaults, unchanged here:

```
lookback_seasons=3, ridge_alpha=1000.0, team_feature_scale=11.0,
reliability_prior_plays=500.0, epa_clip=5.0, rating_version="v1"
```

Sources: participation snapshot `data/players/participation/raw/20260813T131635Z`
(seasons 2016-2025) and play-by-play snapshot `data/pbp/raw/20260817T184927Z`
— the same pbp snapshot the production feature table names in its manifest.
Target seasons 2017-2026; a target season's ridge sees only completed earlier
seasons, which `canonicalize_participation_ratings` asserts
(`source_end_season < target_season`).

`offense_rating` is positive-is-better offensive EPA-per-play contribution;
`defense_rating` is positive-is-better because defensive players enter the
design matrix at -1 against an offense-positive EPA target.

**Reproducibility check (run first, no ATS outcome involved):** refit target
seasons 2020 and 2021 from those two snapshots and compare against the stored
artifact; report the maximum absolute difference.

## 4. Split-half reliability of the ratings themselves (gate, run first)

Predeclared exactly as `docs/st_player_ratings.md` and
`docs/unit_apm_ratings.md` did for their own constructs:

- Population: competitive 11-on-11 participation plays, 2016-2025.
- Half A = **odd** source seasons (2017, 2019, 2021, 2023, 2025);
  half B = **even** source seasons (2016, 2018, 2020, 2022, 2024).
- One ridge fit per half at the frozen configuration in section 3.
- Correlate the **raw** coefficients (not the play-count shrunk ratings, which
  would fold a deterministic function of sample size into the correlation)
  across players with at least **500** offensive (respectively defensive)
  plays in BOTH halves — 500 being the construct's own
  `reliability_prior_plays`.
- Report Pearson and the Spearman-Brown full-length adjustment
  `2r / (1 + r)`, offense and defense separately.

Per the binding taxonomy, **zero** split-half reliability is one of the two
admissible closing grounds. A low-but-nonzero reliability is not.

## 5. The candidate feature

Built over REG games, seasons 2017-2025, from the PRODUCTION inputs:

| input | source |
|---|---|
| games, kickoff | `data/processed/game_features_weak_stack.parquet` (the production table, sha256 `2fb3451b…`) |
| snap counts, weekly rosters, injury reports | player snapshot `data/players/raw/20260910T205112Z` (the one the production table names as `source_player_snapshot`) |
| availability model | `data/processed/weak_stack_availability_rates.parquet` (production's season-lagged learned rates) |
| player ratings | section 3 |

Per team-game:

- **Decision time** = kickoff − 24 hours, production's own
  `decision_hours_before_kickoff`. Only injury rows whose
  `effective_observed_at` is at or before that instant are visible; the latest
  visible revision per player wins.
- **Role share** = an exponentially weighted mean of that player's
  `offense_pct` / `defense_pct` over that team's **strictly earlier** games,
  `alpha = 2 / (role_span + 1)` at production's `role_span = 8`, updated only
  after a game is scored. A player with no prior game for the team has no
  share and contributes nothing.
- **Probability the player plays** = `1 − resolve_unavailability(...)` using
  the production learned season-lagged rates, falling back to the fixed status
  prior exactly as `nfl_ats.availability.resolve_unavailability` does. A player
  with no deadline-visible injury row is treated as playing.

Columns, per side:

```
lineup_offense = SUM_p  ewma_offense_pct(p) * P_play(p) * offense_rating(p, season)
lineup_defense = SUM_p  ewma_defense_pct(p) * P_play(p) * defense_rating(p, season)
lineup_total   = lineup_offense + lineup_defense
full_strength_total = the same sum with P_play == 1 for every player
divergence     = lineup_total - full_strength_total          (<= 0 by construction)
```

and per game `diff_lineup_total = home − away`,
`diff_divergence = home − away`.

Leakage: every term is either a completed prior season (ratings), a strictly
earlier game (role shares), or an injury revision timestamped at or before
kickoff − 24h (availability). Nothing from the game itself enters.

## 6. The comparison: production's opener pick, plus a tilt

The baseline is **not** a refit model. It is the served production opener pick
itself, read from the active model's own opener evaluation:
`artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`, active model
`d49194e04945a5e5` (`weak_stack` / `market_residual` / ridge alpha 10,
`gaussian_median`), including the served home-side offset — the columns
`pick_home_at_open_probability_rule`, `correct_at_open_probability_rule` and
`home_cover_probability_at_open`. That archive covers 2020-2025 REG games.

**Predeclared thresholds, fixed from the pre-window seasons 2017-2019** —
disjoint from any evaluation window the rotation registry can assign out of the
opener archive (which starts in 2020), and computed from the feature's own
distribution with no ATS outcome read:

- `M1` = 75th percentile of `|diff_lineup_total|` over 2017-2019 REG games.
- `M2` = 75th percentile of `|diff_divergence|` over 2017-2019 REG games.
- `C`  = 75th percentile of `max(|home divergence|, |away divergence|)` over
  2017-2019 REG games.

**Screen A (primary, the lane's specified construction).** Production's opener
pick stands, except on games where `|diff_lineup_total| >= M1`, where the pick
becomes the side with the higher expected-lineup rating sum.

**Screen B (secondary, predeclared here, not after the fact).** Production's
opener pick stands, except on games where `|diff_divergence| >= M2`, where the
pick becomes the side whose expected lineup is **less** degraded relative to
its own full strength. This is the availability-shaped half of the mechanism,
the part section 2 argues is not already priced as team quality.

**The declared cell (section 2's "where").** Games with
`max(|home divergence|, |away divergence|) >= C` — at least one team materially
short of its own full-strength rated lineup. Both screens are reported inside
and outside that cell.

**Primary quantity.** Paired candidate-minus-baseline forced-pick accuracy on
the assigned window, in `accuracy_points`, opener-graded, pushes excluded
(`margin_vs_open == 0`, which the archive already carries as a null
`correct_at_open_probability_rule`).

**Brier.** The tilt is a pick flip, so its probability mapping is the mirror:
`1 − p` on flipped games, `p` elsewhere, against the realized home-cover
outcome. `brier_improvement = baseline Brier − candidate Brier`, positive
favouring the candidate. This is the maximal-confidence reading of a flip and
is reported as such.

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, 2,000 samples, seed
20260911, week-blocked as the primary reference (within-week game correlation
is zero by owner mandate) and season-blocked as a secondary read, never
averaged with it. `probability_positive` comes from
`nfl_ats.evidence_conventions.probability_positive_from_draws`, so an exactly
zero draw takes half credit.

**Within-week permutation null**, 200 draws: `diff_lineup_total` and
`diff_divergence` are permuted within week and the same tilt re-applied. This
answers "what does a tilt of this shape and this flip rate do when the feature
carries no information about the game", and is reported alongside the
bootstrap-vs-zero interval, never instead of it.

**Positive control**, run before the real screen: the tilt feature is replaced
by the realized opener ATS margin (`margin_vs_open`), thresholded at the 75th
percentile of its own absolute value. The harness must show a large, obvious
effect; a "no effect" reading from a blind instrument would mean nothing.

**No Gaussian margin assumption is introduced anywhere in this document.** The
screen never maps a margin to a cover probability; it flips picks and counts
them, and the only probabilities it touches are production's own served ones.

## 7. Rotation, recording, and the decision rule

**Rotation family.** A new family `latent_ratings_on_production`, grade
`opener`, declared `--inherits participation_offense_defense_rapm` (the
family the underlying ratings belong to) and `--acknowledge-mined`. It does
**not** inherit `apm_unit_on_production`: that family is a SIBLING under the
same parent, not an ancestor, and `nfl_ats/rotation.py::_inherited_names`
walks declared edges upward only — AGENTS.md rule 4 retires windows
per-family, so an independent family may legally draw seasons a sibling has
drawn. The assigned block is whatever `nfl-ats rotation assign` computes; the
lane's preference is [2020, 2021], and the actual block is confirmed in
section 8 rather than asserted here.

**Recording.** One `nfl-ats weak-signals record` entry per measured cell,
`--league nfl --family latent_ratings_on_production --category onfield`, units
`accuracy_points` or `brier_improvement` or `correlation` as appropriate, and
one `nfl-ats rotation record` call spending the assigned window. A resolved
wrong sign at the opener grade is the only result that could carry a terminal
classification, and only with `--closing-ground wrong_sign_resolved`; zero
split-half reliability would be the other. Anything else is
`unresolved_below_power`.

**Decision rule, frozen before scoring.** This is a FORCED-PICK pool: 285 cards
are submitted either way, so declining a candidate that is more likely than not
better is taking the other side of that bet. The decision is expected value:
`probability_positive` above 0.5 favours serving the tilt, below 0.5 favours
keeping the incumbent card. No 0.90 or 95% bar governs the card. Separately,
AGENTS.md forbids an **unexplained** threshold flip on the played card; the
mechanism named here is explicit (rated value of the players expected to take
the field, and the gap to full strength), so a served version would be a
mechanism, not a bolt-on — but nothing is served by this document.

## 8. Results (added after the look, 2026-09-11)

Everything below was produced by `scripts/latent_ratings_on_production.py` into
`artifacts/latent_ratings_on_production/20260911T000000Z/` after sections 1-7
were written.

### 8.1 The ratings reproduce exactly

`--mode reproducibility` (`reproducibility.json`): target seasons 2020 and 2021
refit from participation snapshot `20260813T131635Z` and pbp snapshot
`20260817T184927Z` give **5,508 rows, all 5,508 matched to the stored artifact,
maximum absolute difference 0.0** on both `offense_rating` and
`defense_rating`. ROADMAP's "reproducible" claim for this construct holds
bit-for-bit.

### 8.2 Split-half reliability: real, modest, and not zero

`--mode reliability` (`reliability.json`). Half A = odd seasons 2017/19/21/23/25,
**136,685** competitive 11-on-11 plays; half B = even seasons 2016/18/20/22/24,
**132,966** plays. Raw ridge coefficients, players with at least 500 plays per
channel in BOTH halves:

| channel | players | Pearson | Spearman | Spearman-Brown |
|---|---:|---:|---:|---:|
| offense | 760 | **+0.174** | +0.145 | **+0.296** |
| defense | 808 | **+0.154** | +0.140 | **+0.268** |

Both are positive and clearly nonzero, so the closing ground
`no_split_half_reliability` is **not available** for this construct. They are
also the lowest reliabilities PER-09 has measured — below the unit-level APM
fits in `docs/unit_apm_ratings.md` (Spearman-Brown 0.236-0.394) and below the
special-teams rating in `docs/st_player_ratings.md` (0.436) — which is the
expected ordering: an individual player carries a fraction of the plays a unit
does, so the per-player coefficient is the noisiest of the three.

### 8.3 The feature

`--mode feature` (`feature_summary.json`, `expected_lineup_ratings.parquet`):
**2,383** REG games, seasons 2017-2025, zero coverage gaps on the window.
Mean **40.8** rated players contributing per team-game; the expected-lineup
snap-share sums average **10.53** (offense) and **10.39** (defense) against the
11 on-field units they are reconstructing, so the aggregation is doing what it
claims and is not quietly summing a roster.

Thresholds, fixed on the **768** pre-window games of 2017-2019 before any window
outcome was read (EPA-per-play units):

```
M1 = 0.056158   M2 = 0.012329   C = 0.013312
```

**The team-quality ceiling is visible directly in the feature, before any ATS
number.** On the window, `diff_lineup_total` correlates **+0.608** with the
opening home spread — the market and the expected-lineup rating sum are largely
the same statement — while `diff_divergence` correlates only **-0.206** with it
(the sign is a scale effect: a better team has more rated value to lose, so its
divergence is larger in magnitude). Against the realized opener ATS margin both
correlate positively but weakly: **+0.071** and **+0.019**.

### 8.4 Window, baseline, instrument checks

`nfl-ats rotation assign` handed the family **[2020, 2021]** — the block other
lanes drew tonight, computed by the CLI and not chosen. The paired population is
**456** non-push games in **35** weeks out of the archive's 466. The baseline,
the served production opener pick for active model `d49194e04945a5e5`, is
**53.29%** with a 21.05% home-pick rate.

**Positive control** (`positive_control.json`): the tilt feature replaced by the
realized opener ATS margin, thresholded at the 75th percentile of its own
absolute value (15.5 points). 116 eligible games, **53 picks changed, all 53
wrong-to-right**. Delta **+11.623 accuracy points**, week-blocked 95%
[+8.171, +15.066], `probability_positive` **1.000**; Brier improvement
+0.016846, P+ 1.000. The harness is not blind. It proves detection at ~11.6
points, which is the leaked ceiling at a 53-flip rate — it does **not** prove
detection of a one-point effect, so `bounded_by_control` is not available
either.

**Within-week permutation null** (`null.json`, 200 draws). This null is
emphatically **not** centred on zero, and the reason matters: flipping picks
away from a 53.3% baseline costs accuracy by arithmetic alone, so an
information-free tilt at this flip rate loses about 1.7-2.0 points.

| arm | null mean | null sd | null 95% | mean flips | observed | observed flips | percentile |
|---|---:|---:|---|---:|---:|---:|---:|
| A: expected-lineup rating | -1.726 | 1.758 | [-5.044, +1.535] | 97.4 | -1.096 | 81 | **69.0th** |
| B: lineup divergence | -1.986 | 1.746 | [-5.049, +1.321] | 86.6 | -0.439 | 86 | **83.0th** |

### 8.5 Screen A — expected-lineup rating tilt

203 tilt-eligible games, **81 picks changed**. Baseline 53.29% vs candidate
**52.19%**.

- Delta **-1.096 accuracy points**, week-blocked 95% **[-4.450, +2.298]**,
  `probability_positive` **0.256**.
- Season-blocked [-2.273, 0.000], P+ 0.121 — two season blocks only, so this
  is a low-power secondary read, not a sharper answer.
- Brier improvement **-0.0000609**, week-blocked [-0.004734, +0.004494],
  P+ **0.470** — essentially indistinguishable.
- Per season: 2020 **-2.273** (39 flips), 2021 **0.000** (42 flips).
- On the 81 flipped games the baseline was right 53.09% and the tilt 46.91%.

### 8.6 Screen B — lineup-divergence tilt

165 eligible games, **86 picks changed**. Baseline 53.29% vs candidate
**52.85%**.

- Delta **-0.439 accuracy points**, week-blocked 95% **[-4.626, +3.254]**,
  `probability_positive` **0.435**.
- Season-blocked [-5.085, +4.545], P+ 0.250.
- Brier improvement **+0.001555**, week-blocked [-0.006785, +0.009274],
  P+ **0.667** — the one reading in this document that leans toward the
  candidate.
- Per season: 2020 **+4.545** (36 flips), 2021 **-5.085** (50 flips).
- On the 86 flipped games the baseline was right 51.16% and the tilt 48.84%.

### 8.7 The declared cell: at least one team short of full strength

`max(|home divergence|, |away divergence|) >= C` selects **187** of 456 games
across 34 weeks.

| arm | population | games | delta | week-blocked 95% | P+ | flips |
|---|---|---:|---:|---|---:|---:|
| A | inside cell | 187 | -1.070 | [-7.568, +4.908] | 0.369 | 36 |
| A | outside cell | 269 | -1.115 | [-5.682, +3.292] | 0.315 | 45 |
| B | inside cell | 187 | -0.535 | [-8.187, +7.464] | 0.453 | 59 |
| B | outside cell | 269 | -0.372 | [-4.396, +3.729] | 0.431 | 27 |

Brier improvement inside the cell is +0.002414 (P+ 0.612) for arm B and
-0.000436 (P+ 0.456) for arm A.

The cell was predeclared as the place a lineup-aware rating should beat a team
rating, and at this sample it **did not separate**: inside and outside are
within a quarter of a point of each other on both arms, and every interval is
wide enough to hold both. That is a measurement at 187 games, not a finding
about the mechanism.

### 8.8 What this implies for the decision, before what is wrong with it

The decision rule is expected value. Screen A's `probability_positive` is
**0.256** and Screen B's is **0.435** — both below 0.5, so on this window the
EV call is to **keep the incumbent card** and serve neither tilt. Nothing here
is served, and the card is unchanged.

That is the card decision. The evidence decision is different and should not be
collapsed into it:

- **Both screens beat their own information-free null**, at the 69th and 83rd
  percentile. A tilt that flips ~85 picks away from a 53.3% baseline is
  expected to cost ~1.7-2.0 points; these cost 1.10 and 0.44. The feature is
  therefore doing better than chance at *choosing which picks to flip*, while
  still not doing well enough to pay for the flips.
- **The signs point the right way where it is cheapest to look**: the feature
  correlates +0.071 with the realized opener ATS margin, Screen B's Brier
  improvement is positive at P+ 0.667, and the 2020 season is +4.545 points for
  arm B.
- **The construct is reliable enough to keep**: +0.174 / +0.154 split-half,
  reproducing exactly.

No admissible closing ground is established. `wrong_sign_resolved` fails on
both arms (the week-blocked intervals reach +2.298 and +3.254, so neither sits
entirely below zero). `no_split_half_reliability` fails (8.2). The positive
control proved detection at +11.6 points, not at the ~1-3 points this
comparison is arguing about, so `positive_control_bound` fails too. Everything
is recorded `unresolved_below_power`, family `latent_ratings_on_production`,
and the window is spent `unresolved`.

The honest summary of what changed relative to PER-09's earlier work: the
season-lagged player ratings **do** reproduce and **do** carry reliable
signal, and aggregating them to an expected lineup rather than an injury delta
does produce a feature that outperforms a random tilt. What it does not do is
clear the forced-pick bar on top of production, and §8.3 shows why in one
number — at +0.608 correlation with the opening spread, most of what this
feature knows is what the market already priced. That is the team-quality
ceiling showing up exactly where the owner's build filter says it should, now
measured rather than assumed for this construct.

### 8.9 One design property found after the fact, disclosed rather than fixed

The pre-window threshold sample (2017-2019) is **not** homogeneous, and the
thresholds are lower than intended because of it. 2017 is the first target
season the ratings cover, so its ridge sees one source season instead of three,
its per-player play counts are about a third of a full lookback, and the
`plays / (plays + 500)` reliability shrinkage therefore bites much harder; the
role-share EWMA is also cold for the first weeks of the loop. Measured: mean
`|diff_lineup_total|` is 0.0176 in 2017, 0.0429 in 2018 and 0.0572 in 2019, and
the share of games clearing M1 within the threshold sample itself is 1.6% /
28.5% / 44.9% by season.

The consequence is that the predeclared "top quartile" thresholds select more
than a quarter of the window: **44.3%** of window games clear M1, 36.0% clear
M2, 40.3% are in the C cell. The thresholds were fixed before scoring and are
**not** revised here — moving a threshold after seeing the result is exactly
the move this document's discipline exists to prevent.

What it changes is which reference to read. A higher flip rate mechanically
costs more accuracy against a 53.3% baseline, so the bootstrap-vs-zero interval
is penalised by the design flaw, while the **within-week permutation null uses
the identical thresholds and therefore the identical flip rate** — it is the
reference that is immune to this, and it is the one on which both arms land in
the upper third. A rerun on the family's next window should draw its thresholds
from seasons with a full three-season lookback (2019 onward), and that is a
change to declare in advance, not to apply to this result.

The part of the mechanism the ceiling argument does **not** cover — the
availability gap in §8.6, which correlates only -0.206 with the spread — is the
arm that scored best against its null (83rd percentile), carried the only
positive Brier read, and was strongly positive in one of the two seasons. It is
the piece of PER-09 worth carrying into the family's next window, and it is
carried as `unresolved`, not closed.
