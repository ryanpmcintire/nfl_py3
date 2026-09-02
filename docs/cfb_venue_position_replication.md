# Venue milestones and schedule position, replicated on COLLEGE FOOTBALL: predeclaration

Written **before any ATS outcome, cover rate, accuracy delta or sign is
computed on college-football data by this line of work**. Sections 1-8 are the
predeclaration and were saved before any scoring mode was run. Section 9 is the
Results section; **as of 2026-09-01 no cell has been scored**, and section 9
says exactly that and lists what remains.

This is a **cross-league replication**, not a new NFL look. It spends **no NFL
evaluation window and no rotation window** — CFB is this project's sanctioned
free replication ground, the same posture `docs/fluview_cfb_replication.md`
declares (**read**, its lines 8-18) and the same posture
`scripts/cfb_surface_familiarity_screen.py` established.

Family: **`cfb_venue_position_replication`**.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) **bounded by a
positive control** — the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible closures;
if a record command errors, the verdict is wrong, not the validator. A
promotion threshold governs only what the docs may CLAIM; it never governs
which card is PLAYED, which is expected value.

## 1. What is being replicated, and what it is not

Four frozen NFL constructs, transcribed in section 3 and adapted to CFB. Their
NFL reads as they stand today, **read** this session out of
`registry/weak_signals.json`:

| NFL entry | effect | 95% CI | P+ | n | seasons |
|---|---|---|---|---|---|
| `venue_milestone_home_opener` | −0.1329 | [−0.6671, +0.3900] | 0.3115 | 4,317 | 2009-2025 |
| `venue_milestone_new_stadium_debut` | +0.0263 | [−0.0682, +0.1146] | 0.7449 | 4,317 | 2009-2025 |
| `bias_battery_three_plus_road_games` | −0.0410 | [−0.1514, +0.0714] | 0.2209 | 8,634 | 2009-2025 |
| `bias_battery_division_revenge_game` | +0.1907 | [−0.1146, +0.5053] | 0.8825 | 8,634 | 2009-2025 |

All four are `unresolved_below_power` with `closing_ground: null` (**read**).
The fourth is the highest-P+ construct in the set and is the one this
replication most wants to speak to.

### Three things this document flags loudly, up front

**(a) The NFL entries are NOT paired model accuracy deltas.** They are
subset-vs-complement, full-slate-scaled cover-rate gaps — the subset's cover
rate minus its complement's, scaled by the subset's share of the slate
(**reported**, unverified by me: Worker A of this program states the formula is
`(subset_cover − complement_cover) × 100 × fraction_of_slate`, citing
`docs/body_clock_screen.md:107-109`; I did not open that file). The estimator
declared in section 6 below is a **different quantity** — the paired
candidate-minus-baseline forced-pick accuracy delta of the XLG-03 estimator
with one extra column. The two are **never pooled** and a magnitude comparison
between them is not meaningful; only the **direction** is comparable, and even
that with care. Section 9 must therefore report direction agreement, not
magnitude agreement. Computing the NFL's own subset-vs-complement estimator on
CFB as a secondary direction check is a declared, optional extra; **it has not
been computed.**

**(b) Cell 4 is an ADAPTED construct, not an identical one.** The NFL cell is a
*within-season* division rematch. CFB teams almost never play the same opponent
twice in a regular season: **measured** this session on
`data/processed/cfb_game_features.parquet`, the whole 2006-2025 benchmark table
holds **62** (season, team-pair) combinations with two or more meetings, i.e.
**124 games**, and the clean core alone holds **52 pairs / 104 games**. That is
0.6% of a 9,093-game clean core. The cell scored here therefore looks back
across seasons (section 3). **A null on the adapted CFB cell does not refute
the NFL within-season cell**, and section 9 must repeat that.

**(c) The baseline already sees week-of-season and games-played.** The frozen
XLG-03 contract `nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS` (35 columns)
already includes `neutral_site`, `week_sin`, `week_cos`, `home_team_games` and
`away_team_games` (**read**, `data/processed/cfb_game_features.parquet` column
list, measured this session). Every cell here is therefore a **marginal on top
of a model that already knows where in the season the game sits, how many games
each side has played, and whether the site is neutral.** These are the hardest
possible conditions for a schedule-position column, and that is deliberate: a
marginal measured against a bare baseline would mostly re-measure
week-of-season.

**What this document adds that the NFL reads cannot**: an independent sample,
in a different league, at zero NFL-window cost. It is replication evidence
about a mechanism's sign. It is not, and cannot be, a promotion or play/no-play
decision for the NFL card.

## 2. Population

`data/processed/cfb_game_features.parquet` — the XLG-03 canonical benchmark
table (**read**, `docs/cfb_data.md`): completed regular-season FBS-vs-FBS games
carrying both an orientable spread and play-by-play, with the NFL ATS sign
convention (`ats_margin = result − spread_line`, `home_cover` 1/0/NaN-on-push).
**Measured** this session: 12,500 rows × 60 columns, seasons 2006-2025.

Scored population: `nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` (2012-2019
plus 2021-2025), **reused verbatim, never redeclared** (**read**,
`src/nfl_ats/cfb_benchmark.py:46`). **Measured** this session: **9,093 games
across 199 (season, week) blocks and 13 seasons.**

**No coverage-defined restriction applies.** Unlike the FluView replication,
every column here is a complete schedule fact for every game — there is no
archive floor, no missing predictor and no NaN. Section 5 reports flag counts,
not coverage fractions, and the scored season set is the whole clean core.

**Neutral-site handling.** Neutral-site rows are KEPT in the scored population
(the benchmark scores them and `neutral_site` is already a baseline feature).
What changes per cell is whether a neutral-site game can FLAG — see section 3.

## 3. The four cells: NFL definition transcribed, then the CFB adaptation

Outcome sign convention for every cell: the **candidate-minus-baseline paired
accuracy delta**, so **positive = the extra column helped the model**, matching
how the NFL entries' signs are read even though the estimator differs (1a).

### Cell 1 — `home_opener`

> **NFL, transcribed verbatim** (**read**, `docs/venue_milestone_screen.md:40-43`):
> "`venue_milestone_home_opener` — the HOME team's first `location=='Home'`
> game of its season (per-team chronological order by `gameday`). Mechanism:
> crowd energy / ceremony elevation. Predicted direction: **positive**
> home_cover edge."
>
> Registry description (**read**): "HOME team's first home game of its season
> (crowd-energy mechanism), predicted positive; leans negative at P+ 0.311,
> unresolved. Schedule fact, point-in-time safe."

**CFB adaptation.** The home team is playing its first **true home game** of
its season, where a true home game is a non-neutral game at which the team is
the scheduled home side — the direct mirror of the NFL's `location=='Home'`.
Ordering is by kickoff over the team's **full** season sequence from the
schedules snapshot, not the benchmark subset (section 4). A neutral-site game
is not a true home game, so it neither flags **nor consumes the slot**: a team
that opens at a neutral site still flags on its first genuine home date.
Column `cfb_venue_home_opener`, 1/0, never NaN. Predicted direction: positive.

### Cell 2 — `new_venue_debut`

> **NFL, transcribed verbatim** (**read**, `docs/venue_milestone_screen.md:44-53`):
> "`venue_milestone_new_stadium_debut` — a franchise's FIRST regular-season
> home game in a venue that is new to the franchise that season (brand-new
> stadium opening or relocation destination). One flagged game per
> franchise-venue change. Mechanism: unfamiliarity/no established home-field
> routine outweighs novelty energy. Predicted direction: **negative**
> home_cover edge." Exclusions (**read**, lines 69-73): one-off neutral-site
> internationals are NOT counted; renames of the same physical venue are NOT
> debuts; a one-off forced relocation (MIN 2010, Metrodome roof collapse) is
> NOT a debut.
>
> Registry description (**read**): "Franchise's first REG home game in a venue
> new to it (all 12 declared debuts verified against schedules; internationals
> excluded), predicted negative; tiny n_flag=12, leans opposite."

**CFB adaptation.** The NFL version is driven by a hand-curated 12-row table.
CFB has far more venue churn and no such table exists, so the debut is derived
from the schedules snapshot's own history under three conditions, all frozen
here:

1. the game is at the team's **declared home venue** for that season — the
   venue hosting the **plurality** of that team's true home games that season,
   ties broken by earliest kickoff. This is the CFB stand-in for "new permanent
   home", and it is what excludes a one-off relocated or designated-home game
   (the CFB analogue of the exclusions the NFL doc lists);
2. the team **never hosted a true home game at that venue in any strictly
   earlier season** of the snapshot;
3. **left-edge rule**: the team hosted at least one true home game in a
   strictly earlier snapshot season. **A team's first season in the snapshot is
   never a venue debut**, because the snapshot cannot distinguish "new venue"
   from "no history". Enforced in code and regression-tested.

Exactly one game per qualifying team-season flags. Column
`cfb_venue_new_venue_debut`, 1/0, never NaN. Predicted direction: negative.

### Cell 3 — `three_plus_road`

> **NFL, transcribed verbatim** (**read**, registry description): "3rd+
> consecutive true road game this season — subset cover rate vs. complement."
> Implementation (**read**, `scripts/nfl_bias_battery_screen.py:235-240`):
> `is_true_road = (~is_home) & (neutral_site == 0)`; grouped by
> `(team, season)`, `three_plus_road_flag = is_true_road & shift(1) & shift(2)`
> with missing shifts filled `False`. Direction: negative.

**CFB adaptation.** The **away** team is on its 3rd or later consecutive true
road game of the season. The NFL implementation is transcribed line for line,
including its streak-reset semantics, which are therefore the **declared CFB
rule**:

- **a neutral-site game is NOT a road game and BREAKS the streak** — it
  occupies a slot in the sequence and is not `is_true_road`, so the `shift(1)`
  test fails on the game after it;
- a true home game breaks the streak;
- a bye does not break the streak (a bye is not a row);
- the streak never carries across seasons (grouping is `(team, season)`).

The sequence is the team's full season sequence from the schedules snapshot
(section 4). The home team cannot be on a road streak, so the column is purely
away-oriented. Column `cfb_schedule_three_plus_road`, 1/0, never NaN. Predicted
direction: negative (fading the travelling team helps the home-oriented model).

### Cell 4 — `revenge_prior_meeting_loss` (**ADAPTED — flagged loudly**)

> **NFL, transcribed verbatim** (**read**, registry description): "2nd meeting
> this season vs. same division opponent, team lost the 1st meeting — subset
> cover rate vs. the rest of the slate." Implementation (**read**,
> `scripts/nfl_bias_battery_screen.py:248-253`): grouped by
> `(team, opponent, season)`, `revenge_flag = (meeting_rank >= 1) &
> (first_margin < 0)`, where `first_margin` is the first meeting's own SCORE
> margin (not its ATS margin). Direction: positive.

**CFB adaptation, and why it is material.** As measured in 1(b), the CFB
within-season rematch population is 52 pairs / 104 games in the entire clean
core. The construct scored here is therefore:

> the side **lost its most recent PRIOR meeting** with this same opponent,
> where the prior meeting is the immediately preceding game between the two
> teams anywhere in the schedules snapshot (regular season, conference
> championship or bowl), its kickoff is **strictly earlier** than this game's,
> and it sits within a **lookback of at most 2 seasons** (`0 ≤ season −
> prior_season ≤ 2`).

The within-season case is a strict SUBSET of this definition, so nothing is
lost; section 5 reports the decomposition. Two things follow and are repeated
in section 9 and in every `--notes` field:

- **this is an ADAPTED construct, not an identical one**;
- **a null here does not refute the NFL within-season cell.**

**Home-oriented signed encoding.** The NFL cell is a team-side flag; the CFB
model is home-oriented, so the column is signed: **+1** the HOME team lost the
most recent prior meeting, **−1** the AWAY team did, **0** no qualifying prior
meeting (or, vacuously, both — impossible short of a tie). Column
`cfb_schedule_revenge_prior_meeting_loss`. Predicted direction: positive (the
revenge side outperforms, which for a home-oriented column means the signed
column carries positive weight).

## 4. Venue identity and within-season schedule position: the derivation

**Source of the sequence: `data/cfb/schedules/raw/20260816T162105Z/`.**
**Measured** this session: 25 season partitions covering **2001-2025**, 36,915
rows, **0 duplicate `game_id`**, 0 unparseable `start_date`, season types
`regular` 36,636 / `postseason` 279, `neutral_site` a real boolean (1,118 True),
`venue_id` missing on 1.08% of 2006+ rows.

**Why not the benchmark table.** The benchmark is a filtered subset. A CFB
team's first home game of the season is very often against an FCS opponent and
is absent from it, so "first home game" or "3rd consecutive road game" computed
inside the subset would be systematically wrong. **Measured**: the snapshot
carries every benchmark team's full sequence — 2,473 benchmark team-seasons, **0
with no schedule rows**, median 12 games per team-season (13 from 2023), minimum
11 outside 2020 (2020 is COVID-shortened and is outside the clean core by the
benchmark's own regime rule). **Measured**: all 12,500 benchmark `game_id`s are
present in the snapshot (100%, 0 missing) — asserted as a regression test.

All season types are kept in the sequence: a conference-championship or bowl
meeting is a genuine prior meeting for cell 4, and it can never disturb a
within-season position because it always kicks off after every regular-season
game of the same season.

**The team_info own-venue map: joined, measured, and DEMOTED to a diagnostic.**
`data/cfb/team_info/raw/20260901T185247Z/` — 20 season partitions 2006-2025,
10,200 rows, 0 duplicates on `(season, team_id)`, columns `team_id, school,
venue_id, venue_name, city, state`. Joined on `(season, team_id)` its coverage
on the benchmark table is **1.000 on both sides in every season 2006-2025**
(**measured**; asserted as a regression test). That much of the orchestrating
survey's brief reproduces exactly.

**But it cannot serve as a per-season own-venue map, and this is a correction
to the brief.** **Measured** this session: its `venue_id` is **identical across
all 20 season partitions for all 706 teams** — 0 teams carry more than one
distinct value. It is one current-state snapshot replicated per season, not a
venue history. Its agreement with the schedules snapshot's own per-game
`venue_id` on non-neutral home games rises monotonically from **0.9070 in 2006
to 0.9968 in 2025** (**measured**, `artifacts/cfb_venue_position_replication/coverage.json`,
`team_info_venue_agreement.agreement_by_season`) — the exact signature of a
current-state map applied to historical games. Using it to decide whether a
2012 game was at the team's own venue would inject a systematic, era-graded
error.

**Consequences, declared:**

- The **operative venue identity is the schedules snapshot's own per-game
  `venue_id`**, which is what cell 2 requires anyway ("new to it, relative to
  the schedules snapshot's own history"). **Measured**: that id space is stable
  under renames — 44 of 715 venue ids carry more than one venue NAME over time
  (renames collapsed onto one id, which is the behaviour cell 2 needs), and only
  8 of 746 names map to more than one id (generic names such as "Memorial
  Stadium" at different schools, which is correct).
- Cells 1 and 3 do **not** need venue identity at all: the NFL definitions they
  transcribe key on `location=='Home'` / `neutral_site == 0`, not on venue. A
  venue refinement would be a departure from the NFL construct, not a
  faithfulness improvement, so none is applied.
- `team_info` is still joined and its per-season agreement rate is reported as
  a diagnostic in every artifact, so the discrepancy stays visible rather than
  silently absorbed. A regression test asserts the constancy finding, so if a
  future snapshot ever carries a real per-season history the test fails and
  this section gets revisited.

**Point-in-time safety.** Cells 1-3 are pure schedule facts, known at schedule
release, exactly as `docs/venue_milestone_screen.md`'s "Point-in-time safety"
section argues for the NFL battery. Cell 2's plurality rule uses the season's
published schedule, which is likewise known before week 1. Cell 4 reads the
final score of a **strictly earlier** game, which is pregame-legal by the same
argument the NFL cell uses; the lookup is a `shift(1)` over the kickoff-ordered
per-opponent sequence **and** imposes `prior_kickoff < kickoff` explicitly, so a
game at or after the current kickoff is unreachable twice over. The belt and
braces is not decorative: **measured**, the snapshot carries exactly two
team-side rows (`game_id` 400361387, season 2008, team ids 99 and 2026) whose
immediately preceding same-opponent row shares an identical kickoff timestamp —
a duplicated matchup record. The strict comparison drops it.

## 5. Coverage, measured predictor-only before any scoring

**Measured** 2026-09-01 by a predictor-only driver that passes only
`game_id, season, week, home_id, away_id, neutral_site` into the feature module
and reads back flag counts; **no outcome column of any scored game is touched**.
Artifacts: `artifacts/cfb_venue_position_replication/coverage.json` and
`reliability.json`.

Scored population, frozen: the **whole clean core, 9,093 games, 199 week
blocks, 13 seasons** (2012-2019, 2021-2025).

| cell | column | n flagged (clean core) | rate | era 2012-2019 (n=5,440, 122 wk) | era 2021-2025 (n=3,653, 77 wk) |
|---|---|---|---|---|---|
| 1 home opener | `cfb_venue_home_opener` | **771** | 8.48% | 471 | 300 |
| 2 new venue debut | `cfb_venue_new_venue_debut` | **5** | 0.05% | 4 | 1 |
| 3 3rd+ road | `cfb_schedule_three_plus_road` | **131** | 1.44% | 87 | 44 |
| 4 revenge (adapted) | `cfb_schedule_revenge_prior_meeting_loss` | **6,278** nonzero | 69.04% | 3,883 | 2,395 |

Cell 4's signed split: **3,368** rows +1 (home side is the revenge side),
**2,910** rows −1 (away side is).

**Cell 4's decomposition, which is the whole point of 1(b).** On the full
12,500-row table, **62** flagged rows have a prior meeting in the SAME season —
the NFL-identical within-season construct — against **8,886** whose most recent
prior meeting is in an earlier season. The adaptation supplies essentially the
entire population; the identical construct supplies 0.7% of it.

**Cell 2 is thin and that is expected.** An enumeration of the underlying
schedule events (**measured**) finds 44 declared-home-venue debut team-seasons
across 2002-2025, 25 of them in clean-core seasons, 10 of those for benchmark
teams — but only **5** of those debut games are themselves benchmark games,
because roughly half of new-stadium openers are scheduled against an FCS
opponent and so fall outside the FBS-vs-FBS benchmark. n=5 against the NFL
cell's n_flag=12. This is disclosed, not corrected: inventing a larger
population would break the transcription.

**Split-half reliability of each cell's underlying trait** (**measured**,
`nfl_ats.cfb_qb_dependence.split_half_reliability`, seed 20260901). Panel: the
team-side sequence restricted to clean-core seasons, one row per team per game,
metric = that cell's own flag; the function splits each team-season by odd/even
week, correlates the two halves' team-season means and Spearman-Brown corrects.
n = 4,873 team-seasons for all four.

| cell | Pearson r | Spearman-Brown | P+ | reading |
|---|---|---|---|---|
| 1 home opener | −0.9218 | −23.57 | 0.000 | **instrument inapplicable** |
| 2 new venue debut | −0.0021 | −0.0042 | 0.000 | **instrument inapplicable** |
| 3 3rd+ road | **+0.4209** | **+0.5925** | 1.000 | reliable team-season trait |
| 4 revenge (adapted) | **+0.5536** | **+0.7127** | 1.000 | reliable team-season trait |

**Cells 1 and 2 must NOT be read as "zero reliability".** Both are
once-per-team-season events: a team has exactly one home opener, and at most one
venue debut. That single flagged game falls in either the odd-week half or the
even-week half, never both, so the two half-means are **mechanically
anti-correlated** — the −0.92 is an artefact of applying an odd/even split-half
instrument to a once-per-season event, not evidence about the construct.
Reading it as `no_split_half_reliability` would be exactly the unearned closure
AGENTS.md bans.

**And for all four cells `no_split_half_reliability` is unavailable as a
closing ground on a stronger argument anyway**: these are schedule facts, not
noisy measurements of a latent trait. The flag is observed without error, so
there is no measurement-reliability ceiling to bound the effect. Cells 3 and 4
additionally show a resolved positive team-season propensity reliability
(P+ 1.000), which is a bonus, not the load-bearing argument.

## 6. The comparator, and the overlap disclosure

| arm | feature columns | estimator |
|---|---|---|
| baseline (shared) | `nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS` (35 columns, frozen XLG-03 contract) | `nfl_ats.cfb_benchmark.fit_cfb_residual_model`, `target="market_residual"`, ridge `alpha=10.0` |
| candidate, one per cell | the same 35 **plus** exactly one of the four columns | identical |

Regressor, alpha and target are held at the benchmark's own frozen values; only
the feature contract differs, isolating each column's marginal contribution
against everything the benchmark already explains — including, per 1(c),
`neutral_site`, `week_sin`, `week_cos`, `home_team_games` and
`away_team_games`. The extension point is the benchmark's own declared
`feature_columns` parameter (**read**, `src/nfl_ats/cfb_benchmark.py:100-103`).
Candidate columns are never mixed with each other: one column per cell.

**Walk-forward.** Every scored week's two models are trained on all completed
games in the whole table that kicked off **strictly before that week's own
earliest kickoff**, with the benchmark's own
`CFB_BENCHMARK_MIN_TRAIN_GAMES = 500` floor — the same forward chaining
`cfb_walk_forward_benchmark` performs (**read**,
`src/nfl_ats/cfb_benchmark.py:200-209`). Training draws on the whole table, not
only the scored seasons.

**Grade, named.** The CFB benchmark grades on `spread_line`, a close-proxy
median-book spread, and the CFB line archive records no quote observation times
(**read**, `docs/cfb_data.md`). This replication is therefore **close-graded**
and can never be opener-graded. Per the binding "grade the decision at the
opener" rule, a close-graded number settles no NFL play/no-play or promotion
decision — and a CFB number could not do so in any case.

### Overlap disclosure (repeated in every `--notes` field)

`registry/weak_signals.json` already holds two CFB entries on an **overlapping
population** (**read** this session):

- `cfb_bias_battery_neutral_site_designated_home` — effect −0.2676, 95%
  [−0.4318, −0.1004], P+ 0.0009, n 8,953, seasons 2012-2025;
- `cfb_bias_battery_rivalry_finale_proxy` — effect −0.3856, 95% [−0.5780,
  −0.2025], P+ 0.0000, n 17,914 team-side rows, seasons 2012-2025.

Both come from `scripts/cfb_bias_battery_screen.py` and both are
**subset-vs-complement cover-rate gaps, full-slate scaled** (**read**, that
script's `_delta_metric_fn` and `full_slate_scaled_points`), measured on the
same CFB clean core. That is a **different quantity** from this document's
paired accuracy delta of a fitted 36-column ridge. AGENTS.md's commensurability
rule forbids pooling non-commensurable comparators, so:

> **`cfb_venue_position_replication` is a separate pooling family and is never
> pooled with `cfb_bias_battery`**, nor with the NFL entries in section 1,
> whose estimator is also the subset-vs-complement gap (1a).

The four cells here are also **correlated subsets of one window** and are **not
independent votes**; they are never sign-test-pooled against each other.

## 7. Metric, uncertainty, instrument checks, leakage, era split

**Metric.** The paired **candidate-minus-baseline forced-pick accuracy delta**,
in `accuracy_points`, picks taken at `home_cover_probability >= 0.5` and graded
with `nfl_ats.clv.pick_correct` (pushes NaN, excluded) — the same
`_paired_metric` shape `scripts/fluview_cfb_replication.py` uses.

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, **week-blocked primary**
(within-week correlation is ZERO by owner mandate — no ICC term),
**season-blocked secondary**, never averaged together. **1,000 samples**, seed
**20260901**.

**Within-week permutation null, 200 draws.** Both arms' models are fit ONCE on
the REAL `ats_margin`; only the grading margin is shuffled within week. This
null is **not** centred on zero by design (it preserves each week's realised
home-cover rate and the two arms carry different home-pick rates) and is
reported ALONGSIDE the bootstrap-vs-zero interval, never instead of it.

**Positive control**, run BEFORE the real screen, per cell: the candidate's one
new column is REPLACED by the realised `ats_margin` — a deliberate, large
leak — so the harness must show an obvious, large effect. This proves the full
36-column ridge fit can detect a real effect of meaningful size when present.

**Leakage**, regression-tested in `tests/test_cfb_venue_position_feature.py`
(17 tests, all passing, **measured** this session):

1. every candidate column is invariant to shuffling the CURRENT game's
   `result` / `ats_margin` / `home_points` / `away_points`, and invariant to
   dropping those columns entirely;
2. cell 4's prior-meeting lookup **never reaches a game at or after the current
   kickoff** — asserted directly on the fixture AND on the real 73,830-row
   team-side sequence, together with the ≤2-season lookback bound;
3. join correctness: every benchmark `game_id` is present in the schedules
   snapshot; the `(season, team_id)` team_info join resolves both sides of
   every benchmark row; the team_info venue_id constancy finding of section 4
   is asserted so a future snapshot change breaks the test;
4. known-answer fixtures per cell on a hand-built five-team, two-season
   schedule, including a neutral-site game that must not count as a home opener
   or a road game, a left-edge season that must not flag a venue debut, and a
   cross-season rivalry pair;
5. the road streak resets after a home game **and** after a neutral-site game,
   and never carries across seasons.

**Era split, declared before scoring.** The window spans the benchmark's own
declared 2020 regime gap. Two eras, magnitudes reported separately and **never
averaged across a sign flip** (owner rule "era magnitude, not presence"):
**`2012_2019`** (n=5,440, 122 week blocks) and **`2021_2025`** (n=3,653, 77
week blocks).

## 8. Decision rule and recording, frozen before scoring

**Decision rule.** Expected value, never a threshold: `probability_positive`
above 0.5 favours the candidate over the baseline. Predeclared thresholds
govern only what a document may CLAIM. **A CFB result is replication evidence
about a mechanism; it never by itself changes an NFL card**, and this run is
close-graded besides.

**Recording.** One `nfl-ats weak-signals record` entry per cell, `--league cfb`,
`--effect-units accuracy_points`, `--family cfb_venue_position_replication`,
week-blocked `--interval-low/--interval-high/--probability-positive`,
`--sample-games`, `--sample-blocks`, `--source` naming the artifact, and
`--reliability` carrying the **CFB-measured** figure from section 5, never an
NFL one.

| cell | registry name | `--category` |
|---|---|---|
| 1 | `cfb_venue_home_opener_on_benchmark` | `schedule` |
| 2 | `cfb_venue_new_venue_debut_on_benchmark` | **`environment`** |
| 3 | `cfb_schedule_three_plus_road_on_benchmark` | `schedule` |
| 4 | `cfb_schedule_revenge_prior_meeting_loss_on_benchmark` | `schedule` |

**Category choice, declared.** Cell 2 takes `environment` because it is about
the physical venue itself — an unfamiliar building — rather than about where in
the schedule the game sits; this matches the existing CFB entry
`cfb_bias_battery_neutral_site_designated_home`, which is also venue-flavoured
and is also recorded `environment` (**read**). The other three are `schedule`,
matching both NFL siblings (**read**: `venue_milestone_home_opener` and
`bias_battery_three_plus_road_games` are both `category: schedule`).

Era slices (`<cell>_era_2012_2019` / `<cell>_era_2021_2025`, same family) are
recorded **if and only if** the era magnitudes differ materially — a sign flip
between eras, or a magnitude ratio large enough that the pooled figure would
misdescribe either era.

Every `--notes` field must carry all five of: (i) close-graded CFB, no NFL or
rotation window spent; (ii) the four cells are correlated subsets of one window
and are not independent votes; (iii) the NFL sibling entry name; (iv) the
`cfb_bias_battery` overlap disclosure and the separate-pooling-family
declaration from section 6, **including that the NFL siblings' estimator is a
subset-vs-complement cover-rate gap and is not commensurable with this paired
accuracy delta**; (v) for cell 4, that the construct is ADAPTED across seasons
and is not the NFL within-season cell.

**Classification.** `unresolved_below_power` for every cell unless a cell
literally meets a terminal ground. `wrong_sign_resolved` requires the WHOLE
week-blocked interval on the wrong side of zero. `positive_control_bound`
requires the control to have PROVEN detection of an effect of the size in
question. `no_split_half_reliability` is unavailable here on the section-5
argument. An interval containing zero is not a ground; if a record command
errors, the verdict is wrong, not the validator.

**No rotation window is spent.** `nfl-ats rotation` is not invoked by this
document.

## 9. Results

**Nothing has been scored. As of 2026-09-01 no cell has an outcome number, and
nothing has been recorded in `registry/weak_signals.json`.** The session that
built this predeclaration was wound down on an orchestrator budget stop after
the predictor-only work was complete and before any scoring mode was written or
run. Sections 1-8 above are frozen; section 5's numbers are all predictor-only.

**What exists and is verified (measured this session):**

- `src/nfl_ats/cfb_venue_position_feature.py` — all four columns, complete,
  typed, no NaN anywhere on the real population.
- `tests/test_cfb_venue_position_feature.py` — **17 tests, all passing**,
  covering leakage, joins, known answers per cell, and the streak-reset rules.
- `artifacts/cfb_venue_position_replication/coverage.json` and
  `reliability.json` — the predictor-only coverage and reliability of section 5.

**What a later session needs to run to finish**, in this order:

1. Write `scripts/cfb_venue_position_replication.py` with
   `--mode coverage|null|positive-control|screen` and
   `--cell home_opener|new_venue_debut|three_plus_road|revenge_prior_meeting_loss`,
   mirroring `scripts/fluview_cfb_replication.py`'s walk-forward / grade /
   bootstrap / permutation-null / positive-control shape. The feature module's
   `attach_cfb_venue_position_features(frame) -> (frame, diagnostics)` is the
   only new dependency; everything else is imported from the shared helpers
   that harness already uses.
2. `--mode null` per cell (200 draws). A harness reporting a real effect here
   is broken; stop if it does.
3. `--mode positive-control` per cell. A harness that cannot detect the leak is
   blind; stop if it cannot.
4. `--mode screen` per cell, then record all four per section 8, under the
   cross-process lock, and append the per-cell table, the per-era magnitudes,
   and the "What this implies for the decision, before what is wrong with it"
   subsection here.

### What this implies for the decision, before what is wrong with it

No cell has an effect estimate, so **this document implies nothing about any
decision, in either league, and no NFL card is affected**. What the
predictor-only work already settles for the DECISION about whether to spend
further effort here:

- **Cells 3 and 4 are worth scoring.** Both have a resolved, positive
  team-season propensity reliability (Spearman-Brown +0.59 and +0.71, P+ 1.000)
  and populations of 131 and 6,278 flagged rows in a 9,093-game window. Cell 4
  in particular is the CFB read on the highest-P+ NFL construct in the set
  (P+ 0.8825), and its 69% flag rate means the column is not a rare-event
  column at all.
- **Cell 1 is worth scoring** at 771 flagged rows, even though its NFL sibling
  leans against its own frozen direction (P+ 0.3115).
- **Cell 2 will be very thin — 5 flagged games — and should still be scored and
  recorded.** Its NFL sibling is equally thin (n_flag=12) and is recorded. A
  thin population is not a reason to decline: declining a measurement because
  it will be imprecise is the crossing-zero error wearing a different hat.

**What is NOT concluded, and why.** Nothing is closed, because nothing was
measured. In particular `no_split_half_reliability` is unavailable as a closing
ground for all four cells (section 5), so no cell can be closed on reliability
grounds even after scoring, and the negative split-half numbers for cells 1 and
2 are an artefact of a once-per-season event, not evidence.

**One correction to the brief that this document carries**, stated because it
would otherwise propagate: the cfbfastR-data `team_info` snapshot does **not**
give each team's own home venue per season. Its `venue_id` is constant across
all 20 season partitions for all 706 teams (**measured**), so it is a
current-state map; its agreement with the historical per-game venue falls to
0.9070 in 2006. The join coverage figure of 1.000 in the brief is correct and
reproduces; the interpretation of what the joined column means is what needed
fixing. Section 4 records the consequence: cell 2's venue history comes from
the schedules snapshot's own per-game `venue_id`, and cells 1 and 3 need no
venue map at all.

### Files added

- `docs/cfb_venue_position_replication.md` (this document).
- `src/nfl_ats/cfb_venue_position_feature.py` — the four candidate columns plus
  the coverage diagnostics.
- `tests/test_cfb_venue_position_feature.py` — 17 leakage, join, known-answer
  and streak-reset tests.
- `artifacts/cfb_venue_position_replication/coverage.json`, `reliability.json`.

`registry/weak_signals.json` and `registry/rotation_registry.json` are
**untouched** by this document.
