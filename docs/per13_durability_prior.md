# PER-13 reliability trait priors, Stage 1: predeclaration

Work package WP13, ROADMAP row **PER-13** ("Reliability trait priors:
per-player durability from 16-season injury/participation history and
roster-status volatility"). **Stage 1 only.** No ATS window is spent by this
document and none may be spent from it without a separate rotation
assignment.

**Sections 1–8 are the predeclaration and were written before any candidate
Brier, log-loss, or `probability_positive` existed.** The only outcome numbers
appearing above §9 are (a) reproductions of figures already published in
`ROADMAP.md` / `docs/data_feasibility.md` for the *incumbent* learned-
availability model and (b) the §3 feasibility measurements, which are
properties of the *history depth and trait reliability* and are what decide
whether the family is buildable at all. **§9 was added after the look** and
changes nothing above it.

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
CLAIM; it never governs which card is PLAYED, which is expected value.

## 1. The question, and why it is asked on availability and not on ATS

PER-11 replaced hand-authored questionable/doubtful weights with rates learned
from prior seasons' report status × practice status × position group. **Read**,
`docs/data_feasibility.md`:205–212 and `ROADMAP.md`:211: that model improved the
direct availability Brier from **0.09500 to 0.09056** on **57,294** out-of-season
player-games, and moved matched ATS classification only from **52.14% to
52.24%** with a week-blocked interval of **[−0.63, +0.78]** points.

**Measured this session** (reproduction, `scripts/per13_durability_stage1.py`
runs the same three-call pipeline): the incumbent learned model scores Brier
**0.09055568530006708** on **57,294** rows out of **62,206** visible
player-games, bit-identical to the manifest field
`learned_availability_brier` in
`data/processed/game_features_player_learned_availability.manifest.json`
(**read**, line `"learned_availability_brier": 0.09055568530006708`). The
protocol is therefore reproduced exactly, not approximately. (The *fixed* arm
no longer reproduces to the digit — this session measures 0.095192 against the
manifest's 0.095 — because `fixed_unavailability` was deliberately re-frozen
after that artifact was written; see its docstring in
`src/nfl_ats/availability.py`. The fixed arm is not used anywhere in this
experiment.)

What that model knows about a player is **only this week's designation**. It
does not know that this particular player has been listed Questionable
twenty-six times and played twenty-four of them, nor that another has spent
three of the last four Septembers on reserve. PER-13's hypothesis is that a
**per-player durability prior** — built from that player's own multi-season
injury-report, participation and roster-status history — sharpens P(plays)
beyond the designation cell.

Stage 1 asks **only the intermediate question, on the availability target**.
This mirrors PER-12, which was closed at its intermediate target and generated
**no ATS rows at all** (**read**, `ROADMAP.md`:213 and
`docs/data_feasibility.md`:219–229). The gate in §7 is frozen here, before the
look, for the same reason.

## 2. The disclosed prior, in full, before the look

Three readings bear on this and they do not agree.

**(a) The closest sibling in this repository is a measured negative.**
**Read**, `docs/recurrence_hazard_features.md`:96–107 and the registry entry
`registry/weak_signals.json` → `recurrence_hazard_*` (category `health`,
`effect_units` `brier`, effect **−0.0239**, interval **[−0.0303, −0.0176]**,
`probability_positive` **0.0**, reliability **0.742**, seasons [2022, 2024],
2,815 games, classification `unresolved_below_power`): per-player recurrence
flags added on top of a **re-fit** designation baseline made held-out
availability Brier **worse** on both val-2023 and test-2024, with
P(recurrence helps) ≈ 0.000. That is player-history information failing on
exactly this target. It is the single most relevant prior and it points
against.

**(b) That sibling also supplies the design's most important warning.**
**Read**, same doc:102–104: "The large improvement over the *raw* base
(−0.027 val, −0.014 test) is a recalibration artifact: pushing the designation
probability through the same LR accounts for essentially all of it." Any
comparison of a fitted candidate against the *unfitted* incumbent measures
recalibration, not information. §5 therefore makes the **re-fit** baseline the
primary comparator and demotes the raw incumbent to a provenance reference.

**(c) The trait itself is real and strongly reliable, which (a) is not evidence
against.** See §3. The recurrence family's information was body-part episode
structure over a 3-season NFL.com scrape window; PER-13's is the player's own
missed-game rate over up to eleven outcome-bearing seasons and sixteen seasons
of roster status. Those are different constructs on different windows, and the
project's standing rule is that a negative on one is not a negative on the
other.

**AGENTS.md's "team quality is already priced" filter does not obviously bind
here**, because the object being improved is not an ATS feature but the
availability probability itself, on its own held-out target.

## 3. Feasibility: what per-player history actually exists (measured this session)

All figures from `scripts/per13_durability_stage1.py --mode screen`'s
`feasibility.json`, reproducing the same numbers this session's exploratory
pass produced.

**Sources and their reach** (**read**, `data/players/raw/20260817T184901Z/manifest.json`):

| Source | Seasons | Rows | What it can date |
| --- | --- | --- | --- |
| `injuries.parquet` | 2009–2024 (16) | 79,818 | report/practice status, `date_modified` timestamp |
| `weekly_rosters.parquet` | 2009–2025 (17) | 677,513 | weekly roster status code (ACT/RES/PUP/SUS/…) |
| `snap_counts.parquet` | 2013–2025 (13) | 324,611 | the played/did-not-play **label** |

The binding constraint is that the *label* needs snaps, so outcome-bearing
history starts in **2013**; roster-status history reaches back to **2009** and
is used as such (column 6 in §4). The two local snapshots
(`20260812T200527Z`, `20260817T184901Z`) produce **identical** 62,206/57,294
frames — the newer one's extra rows are postseason, which the regular-season
default drops — so this experiment is snapshot-invariant.

**History depth on the evaluation frame** (prior injury-report appearances by
the same player, ordered by kickoff):

| Statistic | Value |
| --- | --- |
| mean prior appearances | 14.08 |
| median | 9 |
| share with ≥ 1 | 93.15% |
| share with ≥ 3 | 80.95% |
| share with ≥ 5 | 70.36% |
| share with ≥ 10 | 49.54% |
| share with ≥ 20 | 25.02% |
| unique players | 5,181 |

Depth grows with the panel: median prior appearances is 4 in 2014 and 12 in
2024; share with ≥ 5 is 49.9% in 2014 and 75.5% in 2024. **Per the owner's
"era magnitude, not presence" rule, §9 reports per-season magnitudes rather
than declaring early seasons uninformative.**

**Split-half reliability of the trait** (odd/even split of a player's
appearances, Pearson r on per-half means, Spearman–Brown corrected):

| Trait | ≥5/half | ≥10/half | ≥20/half |
| --- | --- | --- | --- |
| raw unavailability rate | r 0.628, SB **0.771** (n=1,963) | r 0.746, SB **0.855** (n=919) | r 0.815, SB **0.898** (n=227) |
| **residual vs the designation cell** | r 0.546, SB **0.707** (n=1,963) | r 0.657, SB **0.793** (n=919) | r 0.669, SB **0.802** (n=227) |

(Both rows are computed on the 57,294-row scored frame so that they are
directly comparable. An earlier exploratory pass computed the raw-rate row on
the wider 62,206-row outcome frame and got SB 0.779 / 0.854 / 0.893 on
n=2,119 / 1,000 / 249; the residual row is identical either way. The table
above is the one `feasibility.json` reproduces.)

The second row is the one that matters: after the incumbent model's own
prediction is subtracted, a player's tendency to miss more or fewer games than
his designation implies still splits half-to-half at **SB 0.793** (≥10/half),
with a cross-player standard deviation of **0.102** in residual units. For
scale, `injury_value_lost` is kept at reliability 0.933 and the CFB
role-continuity family was explicitly ruled *not* closable at 0.719/0.680
(**read**, `ROADMAP.md`:627). **The trait is reliable; the history is
sufficient; the family is buildable.** Whether the information survives the
designation cell is what §5 measures.

The between-player excess variance is real, not binomial: on the raw rate,
weighted variance 0.04948 against a binomial component of 0.01237, excess
0.03711, implying a beta-binomial prior strength of ≈5.1 prior observations.
That number is **not** hard-coded — §4 re-derives it per fold from the fold's
own prior seasons.

## 4. The durability prior: definition, frozen

Let a row be a visible player-game *i* for player *g* with kickoff `k_i` and
decision cutoff `c_i = k_i − 24h` (the same 24-hour cutoff
`build_availability_outcomes` already applies).

**Point-in-time rule.** The outcome history for row *i* is every outcome row
of player *g* whose **game kickoff is strictly earlier than `c_i`**. This is
enforced by construction (per-player kickoff-ordered cumulative sums plus a
`searchsorted` on `c_i`), not by a season filter, so a row can never see its
own game and can never see a game that had not finished by its decision time.
The roster history for row *i* is every weekly-roster row of player *g* with
**strictly earlier `(season, week)`** — the contract the snapshot manifest
itself declares (`"weekly_rosters": "strictly earlier season/week only"`).

**Shrinkage.** Every rate below is empirical-Bayes shrunk toward a group
target, and **every prior strength is derived from the data's own between-
player variance on the fold's strictly-prior seasons. No prior strength is
hand-picked.**

- *Residual columns* use the random-effects / James–Stein weight
  `n / (n + M)` with `M = σ²_within / σ²_between`, both estimated by the method
  of moments on the fold's prior pool: `σ²_within` is the pooled within-player
  residual variance and
  `σ²_between = max(ε, Var_w(r_j) − σ²_within · mean_w(1/n_j))`, the
  DerSimonian–Laird / Efron–Morris (1975) moment estimator.
- *Rate columns* use the beta-binomial method-of-moments prior strength
  `M = m(1−m)/s² − 1` (Kleinman 1973), computed **within position group** on
  the fold's prior pool, floored at 1 and capped at 500 observations.

**The six candidate columns** (all exactly 0 when the player has no prior
history, so a debutant is scored by the designation cell alone):

| # | Column | Definition |
| --- | --- | --- |
| 1 | `durability_residual` | `Σ_H (y − p_cell) / (n_H + M_res)`, over prior scored rows |
| 2 | `durability_listed_active_residual` | column 1 restricted to prior rows whose `report_category != "out"` — the "played despite Questionable" channel |
| 3 | `durability_rate_logit_offset` | `logit(p̂_g) − logit(g_pos)` where `p̂_g` is the beta-binomial shrunken prior unavailability rate over **all** prior outcome rows (2013+) and `g_pos` the fold's position-group rate |
| 4 | `durability_log_observations` | `log1p(n_H)` — how much history the prior rests on |
| 5 | `roster_absence_rate_logit_offset` | shrunken rate of *on the weekly roster and logged zero snaps*, over prior roster weeks in snap-covered seasons, as a logit offset from the position-group rate. Counts weeks with **no injury report at all** — the durability the report never sees |
| 6 | `roster_reserve_rate_logit_offset` | shrunken rate of prior roster weeks with a reserve-list status (`RES`, `PUP`, `SUS`, `NFI`, `EXE`, `RSN`, `RSR`, `E01`, `E14`, `NON`), as a logit offset. Uses roster history from **2009**, the 16-season reach PER-13 names, and is the channel that carries league suspensions |

Logits are clipped to `[0.002, 0.998]` before differencing. Columns 1 and 2
rest on the scored frame (2014+, since `p_cell` must exist); columns 3–6 use
the full outcome/roster history.

## 5. The comparison, frozen

**Population.** The out-of-season scored frame produced by
`build_availability_outcomes` → `build_season_lagged_availability_rates` →
`score_availability_rates` on `data/processed/game_features_pbp.parquet` and
the latest player snapshot, `decision_hours_before_kickoff = 24`. Target:
`unavailable` (player logged zero offense, defense and special-teams snaps).

**Folds.** Expanding and chronological, identical in shape to the parent
model's own "expanding completed prior seasons only": for each target season
`S`, fit on every scored row with `season < S`, evaluate on `season == S`.
Season 2014 has no scored predecessor (2013 has no lookup table), so the fitted
arms are evaluated on **2015–2024**; both arms are evaluated on exactly the
same rows, so the comparison stays paired. A fold requires ≥2,000 training
rows; all ten satisfy it. The 57,294-row unfitted incumbent figure is carried
as a provenance reference only.

**Estimator, identical in every arm.** `StandardScaler` (fit on the fold's
training rows only) → `sklearn.linear_model.LogisticRegression(C=1.0,
solver="lbfgs", max_iter=1000)`. This is the same estimator the recurrence
sibling used, chosen so the two results are commensurable.

| Arm | Design matrix |
| --- | --- |
| **A** raw incumbent (reference) | `p_cell` used directly, not fitted |
| **B** re-fit baseline (**primary comparator**) | `logit(p_cell)` |
| **C** candidate | `logit(p_cell)` + the six columns of §4 |
| **PC** positive control | `logit(p_cell)` + `leaked_played` (the row's own outcome) |

**Primary comparison: C vs B.** This isolates the durability information from
recalibration, which §2(b) shows is otherwise the dominant term. **Secondary:
C vs A**, reported for completeness and explicitly labelled as
information-plus-recalibration.

## 6. Metric and uncertainty, frozen

Per row, `d_i = (p_B,i − y_i)² − (p_C,i − y_i)²`. The effect is `mean(d_i)`,
so **positive means the candidate is better** — the storage convention pinned
at `src/nfl_ats/weak_signals.py`:112 ("Effects are always stored so that
POSITIVE FAVOURS THE CANDIDATE… Brier and MAE improve downward, so a caller
recording those must negate before storing"). Log-loss difference is reported
the same way, as a secondary.

Uncertainty: `nfl_ats.clv.week_blocked_bootstrap` on the paired per-row
differences, `block="week"` (the rows carry `season` and `week`),
`samples=2000`, `seed=20260901`, 95%. `probability_positive` is that
function's own field: the fraction of week-blocked resamples in which the
improvement is positive. Season-blocked is reported alongside as a secondary.
Player-games inside a week are correlated through team and opponent, which is
why the block is the week; the owner's zero-within-week-correlation mandate is
about **games**, and does not apply to a player-game panel.

**Reported as `probability_positive`, never as "the interval contains zero".**

## 7. The gate, frozen

**Stage 2 — an ATS on-production test on a rotation-assigned window — is
warranted if and only if `probability_positive > 0.5` on the primary pooled
Brier improvement (C vs B).**

This is an expected-value statement, not a significance threshold. The pool is
forced picks; a candidate more likely than not to carry information that the
production availability channel lacks has positive expected value against the
cost of one rotation window, and a candidate more likely than not to be noise
has negative expected value against that same cost. `P+ = 0.5` is the EV
break-even for spending the window, and nothing about this gate licenses a
claim in any document — what may be CLAIMED is governed separately, and at
Stage 1 the answer is: nothing about ATS.

Whatever the outcome, the result is recorded with `nfl-ats weak-signals record`
under family `per13_durability_prior_stage1`. **A `P+` below 0.5 is
`unresolved_below_power` unless one of the two admissible closing grounds
applies** — and §3 has already measured split-half reliability at 0.793, so
`no_split_half_reliability` cannot apply here.

## 8. Controls and the leakage test, frozen

**Positive control (run first, before the screen).** Arm PC adds the row's own
outcome as a column. Its pooled Brier must collapse to **< 0.02**. If it does
not, the estimator/fold/scoring harness cannot detect an effect it is being
handed for free, no verdict is issued, and the harness is the finding. This is
also what makes `bounded_by_control` available as a closing ground later: a
control proven able to detect an effect of a given size.

**Leakage regression tests** (`tests/test_durability_prior.py`, the family's
required leakage regression per AGENTS.md):

1. *Future outcomes never move an earlier prior.* Take a player's frame,
   arbitrarily flip every outcome in seasons after `T`, and assert the
   durability columns for every row at or before `T` are **bit-identical**.
2. *A row never sees itself.* A player's first appearance must get
   `n_H = 0` and all six columns exactly 0.
3. *Kickoff, not season, is the boundary.* A prior game whose kickoff falls
   inside the target row's 24-hour decision window must be excluded.
4. *Roster history is strictly earlier `(season, week)`.* Same-week roster
   rows must not enter columns 5 and 6.
5. *Shrinkage math.* `n = 0` shrinks to the group target exactly; `n → ∞`
   approaches the raw rate; the moment estimators recover a planted
   between-player variance.
6. *Join correctness.* Attaching the columns preserves row count, row order
   and the one-to-one key.

## 9. Results

**Added after the look. Only one thing above this line changed, and it is
disclosed here: §3's raw-rate reliability row was restated on the 57,294-row
scored frame so that both rows share a population, with the earlier
exploratory figures kept alongside. No definition, comparison, metric, gate or
control was touched.**

All numbers **measured this session**. Artifacts:

- positive control: `artifacts/per13_durability_stage1/20260901T190549Z/`
- screen: `artifacts/per13_durability_stage1/20260901T190609Z/`
- post-hoc diagnostics: `artifacts/per13_durability_stage1/20260901T191113Z/`

Every fitted arm is evaluated on the same **52,382** player-games, seasons
**2015–2024**, **174** week blocks, **10** season blocks.

### 9.1 Positive control (run first, before the screen)

| Arm | Brier | Log-loss |
| --- | --- | --- |
| B re-fit baseline | 0.09087474 | 0.29624155 |
| PC baseline + the row's own outcome | **0.00000053** | 0.00040928 |

The control drives Brier from 0.0909 to **5.28e-07**, a factor of ~172,000,
against the frozen ceiling of 0.02. Paired improvement **+0.09087421**,
week-blocked 95% [+0.08872671, +0.09301337], `probability_positive` **1.000**;
season-blocked [+0.08683038, +0.09521801], `probability_positive` **1.000**.
**The harness detects an effect it is handed**, so a null from it would have
been informative rather than vacuous, and `bounded_by_control` is a live
closing ground for this instrument in future work.

### 9.2 The primary comparison, C vs B

| Arm | Brier | Log-loss |
| --- | --- | --- |
| A raw incumbent, unfitted (reference) | 0.09037411 | 0.29472996 |
| B re-fit baseline (**primary comparator**) | 0.09087474 | 0.29624155 |
| C baseline + the six durability columns | **0.08332335** | **0.26811727** |

**Primary: the durability prior improves out-of-season availability Brier by
+0.00755140** (B − C; positive favours the candidate), week-blocked 95%
**[+0.00675488, +0.00830811]**, **`probability_positive` 1.000** over 2,000
resamples. Season-blocked: [+0.00630437, +0.00905103],
`probability_positive` **1.000**. Log-loss agrees: **+0.02812428**,
week-blocked 95% [+0.02547961, +0.03062379], `P+` **1.000**.

**Secondary, and it is the conservative one — which is the opposite of what
§2(b) predicted.** Against the *unfitted* incumbent A the improvement is
**+0.00705076**, week-blocked 95% [+0.00629008, +0.00781391], `P+` **1.000**.
The recurrence sibling found that re-fitting the designation probability
through the same logistic accounted for essentially all of its apparent gain;
here re-fitting makes the baseline slightly **worse** (0.09087 vs 0.09037), so
the primary comparison is marginally flattering rather than conservative. The
honest summary is that both readings clear the gate and the gap between them
(0.00050 Brier) is small next to the effect itself.

For scale, PER-11's entire parent gain was 0.09500 → 0.09056, i.e. **0.00444**.
This candidate adds **+0.00755 on top of the parent's own output** — about
**1.7x** the parent's whole improvement, from history the designation cell
never sees.

The derived shrinkage strengths behaved as an estimator should: the pooled
beta-binomial rate strength grew monotonically with the training panel, 3.51
(fold 2015) to 6.38 (fold 2024), and the residual random-effects strength from
2.50 to 5.27 (`metadata.json` → `fold_calibrations`). §3's pre-look pooled
estimate of ≈5.1 sits inside that range. Nothing was hand-picked.

### 9.3 Per-season magnitude (owner rule: magnitude, not presence)

| Season | n | Brier B | Brier C | improvement (B−C) |
| --- | --- | --- | --- | --- |
| 2015 | 5,008 | 0.0836554 | 0.0765881 | +0.0070673 |
| 2016 | 4,928 | 0.1015518 | 0.0892802 | +0.0122717 |
| 2017 | 4,949 | 0.0927700 | 0.0869843 | +0.0057857 |
| 2018 | 4,952 | 0.0843564 | 0.0792810 | +0.0050755 |
| 2019 | 5,195 | 0.0817300 | 0.0762074 | +0.0055226 |
| 2020 | 5,399 | 0.0959347 | 0.0880126 | +0.0079221 |
| 2021 | 5,288 | 0.0986373 | 0.0910606 | +0.0075767 |
| 2022 | 5,386 | 0.0881317 | 0.0826841 | +0.0054476 |
| 2023 | 5,400 | 0.0974544 | 0.0869718 | +0.0104825 |
| 2024 | 5,877 | 0.0848889 | 0.0766450 | +0.0082439 |

**Ten seasons out of ten favour the candidate** (sign test p = 0.00195,
two-sided), magnitudes ranging **+0.0051 to +0.0123**, a 2.4x spread. There is
no era trend: the two largest seasons are 2016 and 2023, the two smallest 2018
and 2022. The magnitude tracks how hard the season was to predict at all — the
four highest baseline Briers (2016, 2021, 2023, 2020) are four of the five
largest improvements — rather than tracking panel depth.

### 9.4 Per-position magnitude

| Position group | n | Brier B | Brier C | improvement |
| --- | --- | --- | --- | --- |
| offensive_line | 9,442 | 0.0969628 | 0.0856338 | **+0.0113290** |
| skill | 16,289 | 0.0934699 | 0.0857851 | +0.0076848 |
| front | 15,348 | 0.0829209 | 0.0764969 | +0.0064240 |
| other | 688 | 0.0801739 | 0.0740291 | +0.0061448 |
| secondary | 10,615 | 0.0936708 | 0.0879632 | +0.0057077 |

**Five groups out of five favour the candidate.** The magnitude spread is 2.0x,
with offensive linemen nearly double the secondary. That ordering is
mechanically sensible and not something the design aimed at: the injury report
is least informative for linemen, whose designations are both frequent and
weakly predictive, so a player-specific history has the most left to add there.

### 9.5 Post-hoc diagnostics (not part of the frozen comparison)

An improvement 1.7x the size of the parent model's entire gain deserves an
audit before it is written up, so three were run
(`--mode diagnostics`, artifact `20260901T191113Z/diagnostics.json`).

**Placebo — the decisive one.** Permuting the aggregate rows so each player is
handed a randomly chosen other player's history destroys the effect entirely:
Brier **0.09096180** against the baseline's 0.09087474, an improvement of
**−0.00008706**. Six columns of correctly-shaped but wrongly-attributed history
buy nothing (slightly less than nothing, as six noise columns should). **The
gain is row-level information about the specific player, not an artifact of
giving the candidate arm more columns.**

**Single-column and leave-one-out.** Redundancy is heavy, which is expected of
five near-substitute measurements of one trait:

| Column | alone (Brier gain) | leave-one-out cost |
| --- | --- | --- |
| `durability_residual` | +0.0061778 | +0.0000672 |
| `durability_listed_active_residual` | +0.0059672 | −0.0000177 |
| `durability_rate_logit_offset` | +0.0040126 | −0.0000685 |
| `roster_absence_rate_logit_offset` | +0.0029094 | **+0.0006275** |
| `durability_log_observations` | +0.0021027 | **+0.0006388** |
| `roster_reserve_rate_logit_offset` | +0.0004401 | +0.0000246 |

The single strongest column is the residual-vs-cell one — the construct §3's
0.793 reliability was measured on. The two columns that are *uniquely*
irreplaceable are the roster-absence rate and the history-depth term, each
worth ~+0.00063 that nothing else supplies; the three residual/rate columns are
mutual substitutes, and two of them are individually removable at a small
*gain*. A leaner four- or five-column version is the obvious refinement and is
a separate predeclaration, not a rerun of this gate.

**Prior seasons only.** Re-deriving every column with all same-season history
removed — so the prior is purely the multi-season trait PER-13 names, and 19.8%
of rows have no history at all instead of 6.8% — still yields
**+0.00381506**, **51% of the full effect**. Roughly half the gain is the
cross-season durability trait and roughly half is within-season history. Both
halves are pregame-legal; the split matters because only the first half is what
the PER-13 row asks about.

### 9.6 What this implies for the decision, before what is wrong with it

**The gate frozen in §7 is met, and Stage 2 is warranted.**
`probability_positive` on the primary pooled Brier improvement is **1.000**
against an EV break-even of 0.5, on 52,382 player-games, 10/10 seasons and 5/5
position groups in the same direction, both blockings agreeing, the
conservative secondary comparison agreeing, a positive control proving the
harness responsive, and a placebo proving the effect is real row-level
information. Under the project's expected-value rule this is not a close call:
declining a rotation window here is taking the far side of a bet the evidence
does not support. **The recommended Stage 2 is an ATS on-production test on a
rotation-assigned window, measured on top of what is actually PLAYED** — the
"composition is not the signal" lesson — not against a bare baseline. Stage 2
was not run and no ATS window was spent by this document.

**Now what is wrong with it. Five things, and none reverses the decision.**

1. **This is an availability result, and the measured conversion rate from
   availability to ATS is poor.** PER-11's parent improvement was 0.00444 on
   this metric and bought **+0.10 ATS points**, week-blocked [−0.63, +0.78]
   (**read**, `docs/data_feasibility.md`:213–216). A naive scaling of this
   candidate's 0.00755 lands around a fifth of an accuracy point — an order of
   magnitude below this evaluator's ~2-point resolution. Stage 2 is worth its
   window because the pool is forced picks and EV is EV, **not** because a
   visible ATS gain should be expected. Anyone quoting this result as an edge
   is misquoting it.
2. **A `P+` of 1.000 on 52,382 player-games is cheap.** The Brier difference is
   a smooth paired quantity on a large panel; this interval is not comparable
   to a `P+` of 1.000 on 2,000 ATS games and must never be pooled with
   accuracy-point entries (AGENTS.md's commensurability rule). It says the
   columns carry information about who plays; it says nothing about whether
   that information survives aggregation to a team-game feature, which is
   exactly where PER-09, PER-12 and the participation-RAPM candidate all died.
3. **Half the effect is within-season recency, not the 16-season trait the
   ROADMAP row names.** §9.5 measures the split at 51/49. The row's own
   hypothesis is supported at about half the headline size.
4. **The six columns were specified in one pass and the ablation says two of
   them are removable at a small gain.** The pooled result belongs to the
   family, not to any column. Nothing here licenses a claim about which
   mechanism carries the signal beyond §9.5's own table.
5. **The closest sibling still points the other way and this does not refute
   it.** The recurrence-hazard family put player history on this exact target
   against a re-fit baseline and lost with P(helps) ≈ 0.000 on 2,815 rows
   (§2(a)). Different construct, different window, 19x fewer rows — but an
   honest reader should record that this is the second attempt at "player
   history beats designation" and the first one failed. This result does not
   reopen or close that entry; it stands beside it.

### 9.7 Registry

Recorded as `per13_durability_prior_availability_brier`, family
`per13_durability_prior_stage1`, league nfl, seasons 2015–2024,
`--effect-units brier`, effect **+0.0075514** — the Brier *improvement*
(B − C), stored so that positive favours the candidate per
`src/nfl_ats/weak_signals.py`:112 ("Brier and MAE improve downward, so a caller
recording those must negate before storing"), which is the same mapping the
`recurrence_hazard` row uses. Interval [+0.0067549, +0.0083081],
`probability_positive` 1.0, reliability **0.793** (the residual trait, §3),
52,382 sample games, 174 blocks, classification **`unresolved_below_power`**.

That classification is not a hedge. The taxonomy has exactly three states and
no "confirmed": `refuted_mechanism` and `bounded_by_control` are both closures
and neither applies to a positive result with a live control, so category 3 is
the only admissible record. It means "keep this open and spend the next
window", which is precisely the decision §9.6 reaches.
