# Graph `team_stat` as an OFFENCE-ONLY replacement input — CFB (predeclaration)

**Status:** predeclared 2026-09-01, BEFORE any outcome number for this
comparator was computed. Sections 1–11 are frozen; section 12 (Results) is
appended after the look and nothing above it is edited afterwards.

**Owning work package:** WP35. Files: this document,
`scripts/graph_team_stat_cfb_offense_replacement.py`,
`tests/test_graph_team_stat_cfb_offense_replacement.py`,
`artifacts/graph_team_stat_cfb_offense_replacement/`.

**Parents:** `docs/graph_team_stat_cfb_replication.md` (WP8, complete) and
`docs/graph_team_stat_cfb_replacement.md` (WP24, complete). This document reuses
WP8's feature builder (`src/nfl_ats/graph_team_stat_cfb_feature.py`), WP8's
evaluator conventions and WP24's contract-substitution helpers by IMPORT and
modifies none of them.

---

## 0. Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

Decisions are expected value. `probability_positive` above 0.5 favours the
candidate; predeclared thresholds govern only what a document may CLAIM, never
which card is played. This experiment is CFB, so it changes no card either way
(section 10). Owner rule "era magnitude, not presence" applies to section 12's
era table: a weaker era is a smaller magnitude, never an absence.

---

## 1. Provenance of this question — a sequential confirmation, NOT a first look

**This predeclaration follows a diagnostic whose sign has already been seen, and
it must be read as a sequential confirmation rather than as an independent first
look.** Stating that plainly is the point of this section.

WP24's design named three comparisons per cell and declared the third
(`replacement` − `ablation`, "secondary B") a *diagnostic*: the arm that exists
to disambiguate its 3-for-1 swap, not a finding in its own right. Its numbers
are now published (**read** `docs/graph_team_stat_cfb_replacement.md` §11.4):

| WP24 secondary B cell | close | close P+ | opener | opener P+ |
|---|---|---|---|---|
| `def_epa_per_play` | −0.190 pts | 0.064 | −0.112 pts | 0.175 |
| `off_epa_per_play` | +0.157 pts | 0.749 | **+0.213 pts** | **0.827** |
| `off_success_rate` | +0.123 pts | 0.710 | −0.112 pts | 0.313 |

WP24's own write-up drew the obvious line — "once the raw columns are gone,
putting the graph column into their slot is favoured on both offence cells and
disfavoured on the defence cell" — and then refused to call it a finding,
because "secondary B was declared as a diagnostic, and an offence-only version
is a different family that would need its own predeclaration and its own
declared cells before the signs were seen" (**read**, same file §11.4 point 3
and §"What this implies", point 3).

This document is that predeclaration. Three consequences follow, and all three
are binding on how section 12 may be written:

1. **The side of the ball was chosen AFTER seeing signs.** "Offence, not
   defence" is not a prior — it is a selection made on a 3-cell diagnostic
   sweep. So a positive result here is a *confirmation of a selected subset*,
   with the multiplicity that implies, and section 12 must say so beside every
   number rather than in a footnote.
2. **The motivating diagnostic reading was grade-mixed.** WP24's summary quoted
   `off_epa_per_play` at the OPENER (+0.213, P+ 0.827) and `off_success_rate` at
   the CLOSE (+0.123, P+ 0.710); at the opener `off_success_rate`'s secondary B
   is **−0.112, P+ 0.313** and at the close `off_epa_per_play`'s is +0.157,
   P+ 0.749. Neither grade shows both offence cells favoured by as much as the
   quoted pair suggests. That is precisely why section 6 fixes ONE primary grade
   for the decision rule before this document is run, and it is why this
   document's primary cell is a *joint two-metric* swap rather than a re-run of
   either single-metric diagnostic.
3. **This is not a re-run of secondary B.** Section 4's primary comparator is
   genuinely new: both offence triples out of the contract *simultaneously* and
   both graph columns in, scored against the UNMODIFIED benchmark — a comparator
   no arm in WP8 or WP24 ever fitted. WP24's secondary B was per-cell and used
   the *ablation* as its reference. The two-metric arm has 31 columns; no WP24
   arm had 31. The joint swap is also the version that would actually be
   proposed for a production contract, where you would not carry one metric's
   graph encoding and another metric's raw triple side by side without a reason.

**What this document may therefore claim.** At most: whether the joint
offence-side swap is favoured on expected value, on a window and instrument
already used once for a related question, with the subset chosen after signs
were seen. It may not claim a fresh discovery, and it may not be pooled with
WP24's rows as an independent vote (section 11).

---

## 2. Population, data, and what is inherited unchanged

Everything in this list is inherited from WP8/WP24 verbatim and is NOT
redeclared:

- Source table `data/processed/cfb_game_features.parquet` — **read**
  `docs/graph_team_stat_cfb_replacement.md` §2: 12,500 rows, seasons 2006–2025.
  No CFBD API credit is spent.
- Scored window = `nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` (2012–2019 +
  2021–2025), 9,093 rows, 13 seasons.
- Graph build corpus = the full 2006–2025 table (WP8 adaptation A2).
- Graph node identity = `home_id`/`away_id` (WP8 adaptation A1).
- Frozen structural configuration
  `nfl_ats.graph_team_stat_cfb_feature.CFB_GRAPH_FROZEN_STRUCTURE` (alpha 0.85,
  half-life 8.0 weeks, `max_row_l1` 1.0, prior weight 1.0, `min_games` 16,
  signed-Katz, `injury_beta` 0.0). **Never retuned** — in particular not retuned
  for the replacement role, which remains a declared limitation (section 12's
  "what is wrong with it").
- Estimator `nfl_ats.cfb_benchmark.fit_cfb_residual_model` (Ridge alpha 10,
  out-of-time residual distribution), weekly refits, 500-game training floor
  (`CFB_BENCHMARK_MIN_TRAIN_GAMES`).
- Benchmark contract `CFB_MODEL_FEATURE_COLUMNS`, 35 columns.

**The two metrics of this work package are declared here and are not reopened:**
`off_epa_per_play` and `off_success_rate` — the two OFFENCE members of WP8's
frozen three-cell list. `def_epa_per_play` is deliberately excluded and its
exclusion is the seen-sign selection disclosed in section 1. No third metric may
be added after these are scored, and neither may be dropped after its sign is
seen.

---

## 3. WHICH raw columns are replaced

The rule is WP24's, applied to two metrics at once instead of one: **all three
of `home_<stat>`, `away_<stat>` and `diff_<stat>` are removed together, for each
of the two offence metrics — six columns out.** The justification is unchanged
and was established before any outcome number in either work package (**read**
`docs/graph_team_stat_cfb_replacement.md` §3):

1. The contract generates the triple as one atomic unit per metric
   (`CFB_TEAM_STATE_FEATURES`, **read** `src/nfl_ats/cfb_features.py:89-93`), so
   "that statistic's raw columns" has exactly one honest reading.
2. `diff_<stat>` equals `home_<stat> − away_<stat>` exactly, so with the levels
   still present a linear ridge would reconstruct the differential and a
   `diff_`-only swap would test nothing.

The graph builder attaches one column per metric, so the joint swap is **6 out,
2 in**. That asymmetry is declared, not hidden, and section 4's `offense_ablation`
arm exists specifically to separate "the graph differentials are worse than the
raw differentials" from "losing four home/away LEVEL columns cost something the
differentials cannot carry". Neither reading may be reported as the other.

---

## 4. Arms and cells

Five arms, all fitted with the benchmark's own estimator and refit weekly:

| arm | feature contract | columns |
|---|---|---|
| `benchmark` | frozen `CFB_MODEL_FEATURE_COLUMNS` | 35 |
| `offense_replacement` | benchmark minus BOTH offence triples, plus BOTH offence graph katz differentials | 31 |
| `offense_ablation` | benchmark minus BOTH offence triples, nothing added | 29 |
| `replacement_off_epa_per_play` | benchmark minus that one triple, plus that one graph column | 33 |
| `replacement_off_success_rate` | benchmark minus that one triple, plus that one graph column | 33 |

The defence triple `home_def_epa_per_play` / `away_def_epa_per_play` /
`diff_def_epa_per_play`, the market features, the context features, the
experience features and the other six team-state metrics' triples are UNTOUCHED
in every arm. The test file asserts that column-set identity exactly (section 9).

### The cells, declared in priority order

| # | cell name | comparison | status |
|---|---|---|---|
| 1 | `offense_two_metric_replacement` | `offense_replacement` − `benchmark` | **PRIMARY.** The headline, and the only cell the decision rule in section 10 reads. |
| 2 | `off_epa_per_play_alone` | `replacement_off_epa_per_play` − `benchmark` | Secondary, **continuity only** — see below. |
| 3 | `off_success_rate_alone` | `replacement_off_success_rate` − `benchmark` | Secondary, **continuity only** — see below. |
| 4 | `offense_two_metric_vs_ablation` | `offense_replacement` − `offense_ablation` | Secondary. What the two graph columns are worth once the six raw columns are gone. |

**Cells 2 and 3 are NOT new evidence and are declared as such before running.**
They are, comparator for comparator and window for window, WP24's PRIMARY
comparison for those two metrics (**read**
`docs/graph_team_stat_cfb_replacement.md` §4, "PRIMARY — `replacement` −
`benchmark`"). They are re-run here only so that the joint swap in cell 1 can be
read against its own two single-metric parts computed by the same code in the
same process, rather than against numbers copied out of another document. If
they reproduce WP24's published figures the shared machinery is consistent; if
they do not, something has drifted and section 12 must say so. Their registry
rows carry an explicit "do not pool with the WP24 row, same comparator and same
window" note (section 11).

A fifth comparison, `offense_ablation` − `benchmark`, is computed and printed as
a **report-only diagnostic** (what both offence triples are worth inside the
contract at all). It is not a cell, gets no registry row, and cannot be promoted
to one after its sign is seen. It is declared here because cell 1 minus cell 4
is arithmetically that quantity, so hiding it would be pointless as well as
dishonest.

---

## 5. Metric and uncertainty

**Metric.** Paired candidate-minus-reference forced-pick accuracy in
`accuracy_points` (percentage points), registry `--effect-units
accuracy_points`. Picks are forced by the production probability rule
`home_cover_probability >= 0.5` and graded with `nfl_ats.clv.pick_correct`;
pushes drop out; a row with no opener quote is unscorable at the opener grade,
not a zero, and is masked out of the opener comparison only.

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, 1,000 samples, seed
20260901 (WP8's and WP24's seed, reused so all three documents' intervals are
drawn the same way). `block="week"` is the **PRIMARY** — within-week correlation
is zero by owner mandate, so the week is the honest block — and `block="season"`
is reported beside it, never averaged in. `probability_positive` is reported for
both. The binary "contains zero" is never the verdict.

**Within-week permutation null.** 200 draws, settle margins shuffled within each
(season, week), reusing WP8's `permuted_margins` / `week_positions` by import.
This null is deliberately NOT centred on zero: within-week permutation preserves
each week's realised cover rate, so two arms with different home-pick rates have
a non-zero expected null delta. It is the conservative reference reported
ALONGSIDE the bootstrap, never instead of it, and one permutation is a draw, not
a test. Home-pick rates per arm are printed next to every grade so the size of
that offset is visible rather than assumed.

**Positive control.** `--mode positive-control` replaces **exactly one** of the
two swapped-in graph columns — `graph_v2_team_stat_off_epa_per_play_katz_diff`,
declared here by name — with the realised `ats_margin`, in the
`offense_replacement` arm and in `replacement_off_epa_per_play` only. The
`benchmark`, `offense_ablation` and `replacement_off_success_rate` arms are
untouched, so cell 3's and the diagnostic's numbers must come back byte-identical
to the screen run's, and that identity is checked in section 12. An instrument
that cannot see a leak embedded among 31 features is blind and its null would
mean nothing.

**Picks moved.** Reported as a count and a fraction next to every delta. A delta
near zero produced by a contract that moves almost nothing is a different fact
from one produced by a contract that moves a tenth of the board.

**Era split (report only, no extra registry rows).** Era 1 = 2012–2019, era 2 =
2021–2025 — WP8's split, at the 2020 season the benchmark itself excludes. Per
the owner rule these are MAGNITUDES; a weaker era is never written up as an
absence.

**Run order, binding.** `--mode null` first, then `--mode positive-control`,
then `--mode screen` exactly once.

---

## 6. Grade — the choice, made before running

Both grades are computed and both are reported for every cell:

- **close** — `settle_margin = result − spread_line`, the frozen XLG-03 CFB
  benchmark's own grade. This is what keeps a number commensurable with that
  published instrument, and it is the grade WP8 and WP24 both declared PRIMARY
  for exactly that reason (**read** `docs/graph_team_stat_cfb_replication.md`
  §6: "`spread_line` remains the PRIMARY here only because it is the frozen
  XLG-03 benchmark's own grade and the whole point is commensurability with that
  instrument").
- **open** — `result − spread_open`, with `spread_line` REPLACED by
  `spread_open` in the scoring frame so the market feature and the grade refer
  to the same number.

**Declared before running: the OPENER is the primary grade for this document's
decision rule.** The close is reported beside it, never averaged with it, and is
the figure to quote when comparing against WP8's or WP24's published tables.

The reasoning, stated now so section 12 cannot be accused of picking the grade
that suited the answer:

1. **AGENTS.md is explicit.** "Grade the decision at the OPENER. A close-graded
   number may never veto a play… The close is the market at its sharpest and
   systematically understates pool-relevant edge; using it to reject a candidate
   inverts the project's stated priority." That rule is about decisions, and
   section 10's rule is a decision — the only thing this document decides.
2. **The CFB caveat WP8 documented does not reach this decision.** WP8 chose the
   close because its output was a *measurement to be placed beside XLG-03's
   published benchmark numbers*, and mixing grades there would have broken
   commensurability with the instrument. That constraint still binds the
   REPORTED close figures (which is why they are all still computed and printed),
   but it does not bind the decision, because what this decision governs — the
   section 11 NFL follow-up — is an NFL production-chain experiment that WP24
   already predeclared as opener-graded (**read**
   `docs/graph_team_stat_cfb_replacement.md` §10: "OPENER primary — the
   project's declared primary goal and the grade the pool settles on"). Deciding
   at the close whether to run an opener-graded experiment is the wrong
   instrument for the question.
3. **It is the harder choice here, not the convenient one.** From section 1's
   table, the seen diagnostic that motivated this work package is *stronger at
   the close* for `off_success_rate` (+0.123 / P+ 0.710) and *negative at the
   opener* (−0.112 / P+ 0.313). Choosing the opener therefore selects the grade
   under which the motivating evidence is weaker, which is the direction a
   pre-registration should err in.

**Consequence for recording (declared now).** The effect written to the registry
for every cell is the **opener-graded** delta, because that is the number the
decision rests on. The close-graded delta, its interval and its P+ go in each
entry's `notes`. WP8's and WP24's CFB rows record the CLOSE, so this family's
rows and theirs are **not commensurable for pooling without first picking one
grade** — section 11 repeats that warning in the rows themselves.

---

## 7. Reliability

Not re-measured. WP24 measured split-half reliability for all three CFB graph
traits on this exact population with `n_boot` 4000, seed 20260901, using
`nfl_ats.cfb_qb_dependence.split_half_reliability` (**read**
`artifacts/graph_team_stat_cfb_replacement/reliability/20260901T191734Z/results.json`):

| trait | Spearman–Brown | Pearson r | 95% CI on r | P+ |
|---|---|---|---|---|
| `graph_off_epa_per_play_katz` | +0.99246 | +0.98504 | [+0.98321, +0.98667] | 1.000 |
| `graph_off_success_rate_katz` | +0.99382 | +0.98771 | [+0.98614, +0.98904] | 1.000 |
| `raw_off_epa_per_play` (reference) | +0.98004 | +0.96086 | [+0.95477, +0.96665] | 1.000 |
| `raw_off_success_rate` (reference) | +0.97993 | +0.96063 | [+0.95626, +0.96444] | 1.000 |

Both graph traits are more stable than the raw statistics they propagate.
`no_split_half_reliability` is therefore **inadmissible** as a closing ground for
every cell in this document, before any outcome is seen.

**How reliability is filled on the registry rows.** Cells 2 and 3 take their own
metric's Spearman–Brown (+0.99246 / +0.99382). Cells 1 and 4 are two-trait cells;
they take the **minimum** of the two, +0.99246, declared here as the conservative
choice rather than a mean, because the weaker input bounds a joint construct.
Every row's `--source` cites the artifact above alongside this document.

WP24's honest limitation carries over verbatim: the reliability numbers are
inflated relative to a per-game reliability, because both the raw statistic (a
strictly-lagged span-8 EWM) and the katz rating (a cumulative half-life-8
propagated state) are heavily autocorrelated within a season, so an odd/even-week
split-half is closer to a stability measure. That does not weaken the
inadmissibility conclusion — which needs reliability resolved at or below zero,
and these are at +0.99 — but these numbers must not be compared with
reliabilities of event-like per-game traits elsewhere in the registry.

---

## 8. Leakage

Leakage safety is INHERITED, not re-argued.
`tests/test_graph_team_stat_cfb_feature.py` (WP8, unmodified) proves on
synthetic CFB-shaped frames that week `w` reads only through week `w−1`, that the
join back is by `game_id` in the caller's row order with no loss or duplication,
that adaptation A1's id mapping keeps a rebranded program a single node, and that
an undeclared cell is refused. This work package imports the same builder — twice,
once per metric — and adds no new feature construction, so those proofs cover it.

The estimator side is unchanged: `fit_cfb_residual_model` is trained per week on
games that kicked off strictly before that week's earliest kickoff, with the same
500-game floor.

The one new mechanical risk is that chaining two `add_cfb_graph_team_stat_feature`
calls could disturb row order or drop rows. Section 9's test 5 asserts that the
chained call returns the caller's index and row order with exactly two columns
added.

---

## 9. Test contract (release-blocking)

`tests/test_graph_team_stat_cfb_offense_replacement.py` must prove:

1. **The joint substitution is exact.** The `offense_replacement` contract
   contains BOTH offence graph katz columns and contains NONE of
   `home_off_epa_per_play`, `away_off_epa_per_play`, `diff_off_epa_per_play`,
   `home_off_success_rate`, `away_off_success_rate`, `diff_off_success_rate`.
2. **The defence triple is intact, and nothing else moves.** All three of
   `home_def_epa_per_play`, `away_def_epa_per_play`, `diff_def_epa_per_play`
   survive in every arm; the symmetric difference between the `benchmark`
   contract and the `offense_replacement` contract is exactly the six removed
   names plus the two added ones; every surviving benchmark column keeps the
   benchmark's own order.
3. **The ablation arm is a strict subset.** `offense_ablation` is the benchmark
   minus the same six columns, adds nothing, and is a strict subset of the
   benchmark contract.
4. **The FITTED design matrix agrees with the contract.** A model fitted on a
   synthetic CFB frame exposes `feature_names_in_` on its estimator equal to the
   declared contract for each of the five arms, so the assertion is about what
   the ridge actually saw rather than about a tuple that could drift from it.
5. **Chaining the two builders is order-preserving.** Two chained
   `add_cfb_graph_team_stat_feature` calls on a synthetic frame return the same
   index and `game_id` order with exactly the two graph columns added.
6. **The positive control touches exactly one column.** Under `leak=True` the
   `offense_replacement` contract differs from the screen contract in exactly one
   name — `graph_v2_team_stat_off_epa_per_play_katz_diff` becomes `ats_margin` —
   the `off_success_rate` graph column is still present, and the `benchmark`,
   `offense_ablation` and `replacement_off_success_rate` contracts are identical
   to their screen versions.
7. **Undeclared metrics are refused**, inheriting WP8's cell gate: asking for a
   metric outside this document's declared pair raises.

---

## 10. Decision rule

Expected value: `probability_positive` above 0.5 favours the candidate.

**CFB is replication evidence and never by itself changes an NFL card.** No
result in section 12 promotes, demotes, or edits any NFL feature profile,
`artifacts/active_ats_model.json`, or `CURRENT_PREDICTIONS.md`.

The frozen rule, in one sentence: **if cell 1 (`offense_two_metric_replacement`,
`offense_replacement` − `benchmark`) lands with week-blocked
`probability_positive` above 0.5 at the OPENER grade (section 6), the NFL
follow-up predeclared in section 11 is warranted on expected value and should be
written as its own predeclaration; if it lands at or below 0.5, the follow-up is
not warranted at this reading's strength.**

Only cell 1 reads on this rule. Cells 2–4 are reported and recorded but do not
vote — cells 2 and 3 because they are continuity re-runs of WP24's primary
(section 4), and cell 4 because it uses the ablation as its reference and would
let a large `offense_ablation` − `benchmark` gap masquerade as a graph effect.

**Neither branch is a closure and neither is a promotion.** A P+ at or below 0.5
refutes nothing: section 0's two admissible grounds are the only ones that close
this line, `no_split_half_reliability` is already inadmissible (section 7), and
`wrong_sign_resolved` requires a whole interval on the wrong side of zero. Every
cell is recorded either way (section 11).

**The NFL test is NOT run by this work package under either branch.**

---

## 11. NFL follow-up (predeclared design, NOT run here)

Written now so that section 12's numbers cannot shape it. This is a SPEC, not a
result, and running it is a separate work package's decision — which section 10's
frozen rule either warrants or does not.

- **Family** `graph_offense_replacement_on_production`, `--league nfl`.
- **Question.** Does swapping the production chain's OFFENCE-side team-statistic
  raw `home_`/`away_`/`diff_` triples for their graph katz differentials —
  jointly, all offence metrics at once — beat the production chain as it stands?
- **What this is NOT.** The three published NFL graph readings (−0.935 / −0.668 /
  −0.935 accuracy points, P+ 0.122 / 0.189 / 0.037 on 749 games — **read**
  `docs/graph_team_stat_cfb_replacement.md` §10) are ADD-ON readings on a
  close-graded `[2014, 2016]` window. They say nothing about a swap and nothing
  about an offence-only joint swap.
- **Cells.** The OFFENCE-side statistics whose raw triples are actually present
  in the NFL production feature contract, read off that contract, capped at
  three, declared in writing before any sign is computed. If the production
  contract carries no offence triple in this form, the experiment is not run and
  that fact is reported — it is not worked around by inventing a substitute.
- **Arms.** `production` (the active chain, unmodified) / `offense_replacement`
  (all declared offence triples swapped for their graph columns simultaneously) /
  `offense_ablation` (those triples removed, nothing added). Primary is
  `offense_replacement` − `production`; the ablation arm is the same
  disambiguator section 4 uses here.
- **Grade.** OPENER primary, close reported beside it. Same choice, same reason
  as section 6, and here without even the XLG-03 commensurability caveat.
- **Window.** Declared by the family's own rotation entry via `nfl-ats rotation`,
  never reused silently. Windows retire per-family, not globally; a reused window
  carries a stated discount, not a ban.
- **Uncertainty, null, positive control, era split.** Identical in kind to
  section 5 — week-blocked bootstrap primary, season-blocked secondary, 200-draw
  within-week permutation null, one-column `ats_margin` leak as the positive
  control run before the screen, era magnitudes reported, picks-moved counts.
- **Prerequisite.** The graph columns must exist for the chosen NFL cells at the
  frozen structure, and their split-half reliability must be measured on the NFL
  population by the same `split_half_reliability` call section 7 cites — never
  assumed from the CFB numbers.
- **Sequential-selection disclosure carries forward.** Any NFL write-up must
  repeat section 1: the offence side was chosen after seeing a CFB diagnostic's
  signs, so the NFL run is the first genuinely out-of-sample test of that
  selection, and it is the one entitled to be called a first look.
- **Decision.** EV. A promotion bar is not a decision bar: the pool is forced
  picks, so a candidate more likely than not to be better is played, and any
  predeclared threshold governs only what the docs may claim.

### Recording (frozen before the run)

Four entries, one per cell, `--league cfb`,
`--family graph_team_stat_cfb_offense_replacement`, `--category onfield`,
`--effect-units accuracy_points`, seasons 2012–2025:

| cell | registry name |
|---|---|
| 1 (primary) | `graph_team_stat_cfb_offense_replacement_two_metric` |
| 2 (continuity) | `graph_team_stat_cfb_offense_replacement_off_epa_per_play_alone` |
| 3 (continuity) | `graph_team_stat_cfb_offense_replacement_off_success_rate_alone` |
| 4 (secondary) | `graph_team_stat_cfb_offense_replacement_two_metric_vs_ablation` |

- The recorded `--effect`, `--interval-low/high` and `--probability-positive` are
  the **opener-graded, week-blocked** figures (section 6). The close-graded
  triple, the season-blocked P+, the era magnitudes and the picks-moved counts go
  in `--notes`.
- `--reliability` is filled from section 7: +0.99246 for cells 1, 2 and 4;
  +0.99382 for cell 3.
- `--classification unresolved_below_power` unless a terminal ground is
  *literally* met — a whole interval on the wrong side of zero
  (`wrong_sign_resolved`) or reliability resolved at or below zero
  (`no_split_half_reliability`, already inadmissible by section 7). A P+ below
  0.5, an interval containing zero, or a disagreement between the two grades is
  **none of those** and closes nothing.
- Every row's notes state that cells 1–4 share ONE window and overlapping
  football, that cells 2 and 3 are the SAME comparator on the SAME window as
  WP24's `graph_team_stat_cfb_replacement_off_epa_per_play` /
  `_off_success_rate` rows and must not be pooled with them as independent votes,
  and that this family records the OPENER while WP8's and WP24's CFB rows record
  the CLOSE.
- The four cells are **not pooled** with each other, for the same reason.

---

## 12. Results (added after the look, 2026-09-01)

_Nothing above this line was edited after an outcome number was seen._

Every number below is **measured** this session. Artifacts:

- null — `artifacts/graph_team_stat_cfb_offense_replacement/null/20260901T194509Z/results.json`
- positive control — `.../positive-control/20260901T194852Z/results.json`
- screen — `.../screen/20260901T195141Z/results.json`

199 weeks fitted, 13 seasons, 9,093 clean-core rows, 8,933 graded at the close
and 8,925 at the opener. Column counts came back exactly as section 4 declared:
benchmark 35, `offense_replacement` 31, `offense_ablation` 29, both
single-metric arms 33.

### 12.1 Instrument checks (run first, in the declared order)

**Null** (`--mode null`, 200 within-week permutations, close grade — the grade
WP8's and WP24's null tables use, so the three are comparable):

| comparison | null mean | sd | null 95% |
|---|---|---|---|
| cell 1 `offense_replacement` − `benchmark` | +0.088 pts | 0.326 | [−0.561, +0.673] |
| cell 2 `replacement_off_epa_per_play` − `benchmark` | −0.011 pts | 0.304 | [−0.616, +0.550] |
| cell 3 `replacement_off_success_rate` − `benchmark` | +0.078 pts | 0.323 | [−0.517, +0.705] |
| cell 4 `offense_replacement` − `offense_ablation` | −0.017 pts | 0.238 | [−0.505, +0.359] |
| diagnostic `offense_ablation` − `benchmark` | +0.105 pts | 0.300 | [−0.437, +0.739] |

Every null mean sits at most 0.35 sd from zero. The offsets are the home-tilt
artifact this null exists to expose and they are small here because all five arms
carry the same market feature: **measured** home-pick rates at the close are
benchmark 0.417, `offense_replacement` 0.393, `offense_ablation` 0.395,
`replacement_off_epa_per_play` 0.407, `replacement_off_success_rate` 0.392 — a
spread of 2.5 points. Cells 2 and 3 reproduce WP24's published null table exactly
(**read** `docs/graph_team_stat_cfb_replacement.md` §11.2: −0.011 / 0.304 and
+0.078 / 0.323).

*Disclosed:* WP8's imported `null_distribution` prints the OBSERVED delta beside
the null, so cell 1's close-graded observed value was visible at the null step
rather than at the screen. That is a property of the inherited harness, not a
choice made here, and it changed nothing — sections 0–11 including the decision
rule and the primary grade were already frozen and are byte-unchanged.

**Positive control** (`--mode positive-control`,
`graph_v2_team_stat_off_epa_per_play_katz_diff` — and only that column — replaced
by the realised `ats_margin`):

| comparison | close delta | close week 95% | opener delta | week P+ (both) |
|---|---|---|---|---|
| cell 1 (carries the leak) | **+48.405 pts** | [+47.374, +49.464] | **+44.885 pts** | 1.000 |
| cell 2 (carries the leak) | +48.405 pts | [+47.374, +49.464] | +44.908 pts | 1.000 |
| cell 4 (carries the leak) | +48.528 pts | [+47.524, +49.502] | +45.020 pts | 1.000 |

Cell 1 sits at the **100.0th percentile** of its own permutation null at both
grades. Cell 2's leaked figure reproduces WP24's published positive control
(+48.405 close / +44.885 opener — **read**
`docs/graph_team_stat_cfb_replacement.md` §11.2) exactly.

**The one-column-only requirement of section 5 is satisfied and was checked
field by field, not asserted.** The two arms that do not carry the leak return
values *identical to the screen run's on every reported field* — delta,
both intervals, both `probability_positive` values, both arms' accuracies, the
game count and the picks-moved counts:

| comparison | grade | screen | positive control |
|---|---|---|---|
| cell 3 (not leaked) | open | −0.324930 | −0.324930 |
| cell 3 (not leaked) | close | +0.022389 | +0.022389 |
| diagnostic (not leaked) | open | −0.134454 | −0.134454 |
| diagnostic (not leaked) | close | −0.123139 | −0.123139 |

The instrument is not blind to a real effect embedded among 31 features. It is
**not** thereby proven able to resolve a 0.3-point effect, so
`positive_control_bound` is inadmissible for every cell below.

**Benchmark reproduction.** The `benchmark` reference arm scores **51.595209%**
at the close and **51.719888%** at the opener on the paired population —
identical to WP8's and WP24's published figures for the same arm (**read**
`docs/graph_team_stat_cfb_replacement.md` §11.2: 51.5952% and 51.7199%). The
shared machinery reproduces across three work packages.

### 12.2 The screen — cell 1, the PRIMARY

`offense_replacement` (31 columns) − `benchmark` (35 columns). The declared
primary grade for the decision rule is the **opener** (section 6).

| grade | delta | week 95% | **week P+** | season P+ | season 95% | picks moved |
|---|---|---|---|---|---|---|
| **open (PRIMARY)** | **−0.235 pts** | [−1.005, +0.516] | **0.269** | 0.245 | [−0.945, +0.373] | 1,080 / 9,076 = 11.90% |
| close (commensurability) | −0.134 pts | [−0.848, +0.641] | 0.344 | 0.342 | [−0.841, +0.465] | 1,087 / 9,093 = 11.95% |

Observed deltas sit at the 12.0th (opener) and 22.0th (close) percentile of their
own permutation nulls.

### 12.3 The screen — cells 2, 3, 4 and the report-only diagnostic

| cell | grade | delta | week 95% | week P+ | season P+ | picks moved |
|---|---|---|---|---|---|---|
| 2 `off_epa_per_play_alone` (continuity) | open | +0.034 pts | [−0.492, +0.566] | 0.543 | 0.546 | 714 / 9,076 = 7.87% |
| 2 | close | +0.056 pts | [−0.487, +0.638] | 0.573 | 0.584 | 709 / 9,093 = 7.80% |
| 3 `off_success_rate_alone` (continuity) | open | −0.325 pts | [−1.072, +0.417] | 0.221 | 0.150 | 1,026 / 9,076 = 11.30% |
| 3 | close | +0.022 pts | [−0.696, +0.807] | 0.528 | 0.512 | 1,043 / 9,093 = 11.47% |
| 4 `two_metric_vs_ablation` | open | −0.101 pts | [−0.527, +0.370] | 0.320 | 0.274 | 393 / 9,076 = 4.33% |
| 4 | close | −0.011 pts | [−0.464, +0.410] | 0.458 | 0.449 | 416 / 9,093 = 4.57% |
| diagnostic `ablation` − `benchmark` (no registry row) | open | −0.134 pts | [−0.806, +0.562] | 0.379 | 0.331 | 969 / 9,076 = 10.68% |
| diagnostic | close | −0.123 pts | [−0.791, +0.612] | 0.370 | 0.379 | 967 / 9,093 = 10.63% |

**The continuity check passes exactly.** Cells 2 and 3 reproduce WP24's published
primary figures to four decimal places at both grades (**read**
`docs/graph_team_stat_cfb_replacement.md` §11.3: `off_epa_per_play` +0.056 close
/ P+ 0.573 and +0.034 opener / P+ 0.543; `off_success_rate` +0.022 close / P+
0.528 and −0.325 opener / P+ 0.221). Nothing has drifted between the two work
packages, so every difference below is a real difference of comparator and not a
code change.

**The decomposition is exact, and it is an identity rather than an
approximation.** On the same paired population, cell 1 = diagnostic + cell 4 to
the last printed digit: at the opener −0.1345 + (−0.1008) = **−0.2353**; at the
close −0.1231 + (−0.0112) = **−0.1343**.

### 12.4 Era magnitudes (report only, cell 1, both grades)

| era | opener delta | opener week P+ | opener 95% | close delta | close week P+ | close 95% |
|---|---|---|---|---|---|---|
| 2012–2019 | −0.188 pts (n=5,333) | 0.292 | [−0.982, +0.621] | +0.019 pts (n=5,349) | 0.525 | [−0.854, +0.859] |
| 2021–2025 | −0.306 pts (n=3,592) | 0.307 | [−1.646, +1.071] | −0.363 pts (n=3,584) | 0.267 | [−1.657, +0.885] |

Magnitudes, per the owner rule — no era is described as showing no effect. The
larger single-era magnitude is 2021–2025, at −0.306 points on the opener and
−0.363 on the close, and both intervals reach past +0.88, so neither is a
resolved wrong sign and neither closes anything. The two eras agree in sign at
the opener (−0.19 and −0.31) and disagree at the close (+0.02 and −0.36); at a
±0.3-point instrument that difference is smaller than the instrument, and it is
reported rather than interpreted.

### 12.5 What this implies for the decision, before what is wrong with it

**1. The decision the predeclaration asked for: the NFL follow-up is NOT
warranted at this reading's strength.** Section 10's frozen rule reads cell 1 at
the OPENER grade, and it lands at **−0.235 accuracy points, week-blocked P+
0.269**, 95% [−1.005, +0.516], on 8,925 paired games. P+ 0.269 is below 0.5, so
on expected value the joint offence-side swap is *disfavoured* — roughly 73/27
against, on this window and this instrument. `graph_offense_replacement_on_production`
stays a written, unrun spec in section 11.

**The verdict does not depend on the grade choice, which is the strongest thing
that can be said for it.** The close reads −0.134 at P+ 0.344 — the same side of
0.5 — and both season-blocked readings agree (0.245 opener, 0.342 close). Had the
close been primary the answer would have been the same. This is stated because
section 6 fixed the opener as primary before the run and the reader is entitled
to check that the choice was not doing the work.

**2. The result of a sequential confirmation is that the selected reading did
NOT carry over, and that is the finding.** WP24's per-metric diagnostic favoured
the graph column on both offence cells (+0.213 opener P+ 0.827 on
`off_epa_per_play`; +0.123 close P+ 0.710 on `off_success_rate`). Asked as a
JOINT question by cell 4 — the same "graph in the vacated slot" comparator, both
metrics at once — the answer is **−0.101 pts at the opener, P+ 0.320** and
−0.011 at the close, P+ 0.458. The two positive per-metric diagnostics do not
add up to a positive joint diagnostic. That is exactly the outcome a
confirmation of a post-hoc subset exists to detect, and it is why section 1
insisted this document could not claim a discovery: the offence-side reading that
motivated the work package is weaker when asked as its own question than it
looked when read off WP24's diagnostic sweep.

**3. Nothing is closed, and the mechanism is not refuted.** All four cells are
`unresolved_below_power`. No interval sits entirely on one side of zero — cell 1's
opener 95% reaches +0.516, cell 3's +0.417, cell 4's +0.370 — so
`wrong_sign_resolved` is inadmissible on every one of them. The positive control
resolves +48 points and not ±0.3, so `positive_control_bound` is inadmissible.
Section 7 made `no_split_half_reliability` inadmissible before the run: both
graph traits score Spearman–Brown ≥ +0.9925. A P+ of 0.269 is a 27% chance the
candidate is better, not a zero, and the pooled family stays open.

**4. `off_epa_per_play` alone remains the one arm above 0.5, and this document's
rule did not adjudicate it.** Cell 2 is +0.034 opener / P+ 0.543 and +0.056 close
/ P+ 0.573 — favoured on EV at both grades. But cell 2 is *not evidence produced
here*: it is WP24's already-recorded primary re-run for continuity, on the same
window, and WP24's own frozen rule already declined the NFL follow-up on it (one
of three cells above 0.5 at the opener). A single-metric NFL follow-up is a
different question that would need its own predeclaration, its own window and its
own justification, and **it is deliberately not proposed here on the strength of
a number this document re-ran rather than earned.** Naming it is the honest thing
to do; promoting it after seeing this screen would repeat exactly the error
section 1 was written to guard against.

**5. The two offence triples are close to free real estate, confirming WP24 at
double the width.** The report-only diagnostic says deleting BOTH metrics' whole
`home_`/`away_`/`diff_` triples — six of the 35 contract columns — costs only
−0.134 points at the opener (P+ 0.379) and −0.123 at the close (P+ 0.370), while
moving about 10.7% of the board. WP24 measured the same thing one triple at a
time; for the two offence metrics its secondary A read −0.101 close / −0.179
opener (`off_epa_per_play`) and −0.101 close / −0.213 opener
(`off_success_rate`), and on the defence cell it was **positive** at the opener,
+0.090 — the ablation was better (**read**
`docs/graph_team_stat_cfb_replacement.md` §11.4). Removing six columns therefore
costs no more than removing three did. That is
decision-relevant on its own: the XLG-03 CFB benchmark's team-state metrics are
largely redundant with each other and with the market column, so any swap into
that slot has very little room to move in either direction — which is also why
none of the numbers here are large.

**6. A 31-column contract still very nearly matches a 35-column one.**
`offense_replacement` scores 51.4846% at the opener against the benchmark's
51.7199% — 0.24 points worse, on four fewer columns. That is a parsimony fact
rather than an accuracy win, and it is quoted as one.

Now what is wrong with it. **The offence side was selected after seeing signs**
(section 1), so this was never an independent look; what it delivers is the
correction that such a look exists to deliver, and the correction happens to be
negative. **The swap is 6-for-2 and cell 1 alone cannot separate its two moving
parts** — declared in section 3; the decomposition in §12.3 is exact, but
attributing −0.101 of the opener's −0.235 to "the graph columns are worse than
nothing in that slot" still leans on the ablation being the right counterfactual.
**Cells 2 and 3 are not evidence**, they are the same comparator on the same
window as WP24's rows, they reproduce those rows to four decimals, and pooling
them with WP24 would double-count one window — their registry rows say so.
**All four cells share ONE window and overlapping football**, so they are
correlated decompositions and are deliberately not pooled with each other either.
**Cell 3 disagrees in sign between grades** (+0.022 close, −0.325 open) and cell
1's own era split disagrees in sign at the close (+0.019 vs −0.363); at ±0.3-point
resolution those are coin-flip-sized disagreements and no single one of those
numbers should be quoted alone. **The graph structure was never retuned for a
replacement role** — alpha 0.85 and half-life 8.0 were derived for NFL as an
ADD-ON, and a joint swap is the first design that would justify re-deriving them;
deliberately not done, because a replication that retunes answers a different
question, and a negative here is therefore a negative *for the frozen structure*
and not for the transform. **The whole instrument resolves about ±0.35 points**
(the null sds above), so a −0.235 reading is well inside the noise floor and this
document has measured a direction, not a magnitude. **And this remains CFB**: it
changes no NFL card, and it never could.

### 12.6 Recorded

Four entries, `--league cfb`,
`--family graph_team_stat_cfb_offense_replacement`, `--category onfield`,
`--effect-units accuracy_points`, seasons 2012–2025, all
**`unresolved_below_power`**, recorded effect = the OPENER-graded week-blocked
delta as section 6 declared:

| name | effect (opener) | interval | week P+ | reliability |
|---|---|---|---|---|
| `graph_team_stat_cfb_offense_replacement_two_metric` | −0.2353 | [−1.0053, +0.5158] | 0.269 | 0.99246 |
| `graph_team_stat_cfb_offense_replacement_off_epa_per_play_alone` | +0.0336 | [−0.4922, +0.5659] | 0.543 | 0.99246 |
| `graph_team_stat_cfb_offense_replacement_off_success_rate_alone` | −0.3249 | [−1.0719, +0.4170] | 0.221 | 0.99382 |
| `graph_team_stat_cfb_offense_replacement_two_metric_vs_ablation` | −0.1008 | [−0.5266, +0.3704] | 0.320 | 0.99246 |

Each row's notes carry the close-graded triple, the season-blocked P+, the era
magnitudes, the picks-moved count, the sequential-selection disclosure from
section 1, and the two pooling warnings from section 11 (do not pool these four
with each other; do not pool cells 2 and 3 with WP24's rows).
