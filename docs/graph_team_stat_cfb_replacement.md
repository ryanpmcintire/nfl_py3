# Graph `team_stat` as a REPLACEMENT input — CFB (predeclaration)

**Status:** predeclared 2026-09-01, BEFORE any outcome number for this
comparator was computed. Sections 1–10 are frozen; section 11 (Results) is
appended after the look and nothing above it is edited afterwards.

**Owning work package:** WP24. Files: this document,
`scripts/graph_team_stat_cfb_replacement.py`, `scripts/cfb_graph_reliability.py`,
`tests/test_graph_team_stat_cfb_replacement.py`,
`artifacts/graph_team_stat_cfb_replacement/`.

**Parent:** `docs/graph_team_stat_cfb_replication.md` (WP8, complete). This
document reuses that work package's feature builder
(`src/nfl_ats/graph_team_stat_cfb_feature.py`) and evaluator conventions by
IMPORT and changes neither. It answers the first of the two follow-ups WP8's own
"unfinished" list named, and measures the second (reliability).

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
(section 9). Owner rule "era magnitude, not presence" applies to section 11's
era table: a weaker era is a smaller magnitude, never an absence.

---

## 1. The question WP8 left open

WP8 asked whether the graph `team_stat` transform ADDS anything to the XLG-03
CFB market-residual benchmark, and got a split answer (**read**
`docs/graph_team_stat_cfb_replication.md` §10.2–10.3, 8,933 graded games, 199
weeks):

| direction | `def_epa_per_play` | `off_epa_per_play` | `off_success_rate` |
|---|---|---|---|
| graph column ALONE vs the raw `diff_` alone | +0.369 pts, P+ 0.798 | +0.291 pts, P+ 0.765 | +0.694 pts, P+ 0.897 |
| graph column ADDED to the full benchmark | −0.011 pts, P+ 0.467 | +0.022 pts, P+ 0.535 | −0.179 pts, P+ 0.266 |
| picks moved, added-on | 1.92% | 4.56% | 5.10% |

WP8's own reading of that split (**read**, same file §"What this implies"):
"The graph is a better single feature than the raw statistic; it is not
additional information once a full team-state contract is present."

That sentence has an untested corollary, and it is the question this document
freezes: **if the graph column is a better single feature than the raw
statistic, is it also a better REPRESENTATION of that statistic inside the
contract?** WP8 measured *addition*. Nobody has measured *substitution*. Those
are different comparators with different reference arms, and a family that is
worth nothing as an addition can still be worth something as a swap — the
add-on arm pays a ridge-penalty and collinearity cost for carrying two encodings
of the same football; the replacement arm does not.

This is a distinct predeclaration with its own arms, its own reference, its own
null and its own positive control. No number from WP8's run is reused as a
result here; WP8's figures appear only as the published prior above.

---

## 2. Population, data, and what is inherited unchanged

Everything in this list is inherited from WP8 verbatim and is NOT redeclared:

- Source table `data/processed/cfb_game_features.parquet` — **measured** this
  session: 12,500 rows, seasons 2006–2025. No CFBD API credit is spent.
- Scored window = `nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` (2012–2019 +
  2021–2025) — **measured**: 9,093 rows, 13 seasons.
- Graph build corpus = the full 2006–2025 table (WP8 adaptation A2; the
  walk-forward is leak-safe so warm-up seasons cost nothing).
- Graph node identity = `home_id`/`away_id` (WP8 adaptation A1).
- Frozen structural configuration
  `nfl_ats.graph_team_stat_cfb_feature.CFB_GRAPH_FROZEN_STRUCTURE` (alpha 0.85,
  half-life 8.0 weeks, `max_row_l1` 1.0, prior weight 1.0, `min_games` 16,
  signed-Katz, `injury_beta` 0.0). Never retuned.
- The three cells, frozen by WP8 before any sign was seen and NOT reopened
  here: `def_epa_per_play`, `off_epa_per_play`, `off_success_rate`. No fourth
  cell may be added after these three are scored.
- Estimator `nfl_ats.cfb_benchmark.fit_cfb_residual_model` (Ridge alpha 10,
  out-of-time residual distribution), weekly refits, 500-game training floor
  (`CFB_BENCHMARK_MIN_TRAIN_GAMES`).
- **Measured** feature-contract sizes: `CFB_MODEL_FEATURE_COLUMNS` holds **35**
  columns. (WP8's prose says "31-feature ridge"; the counted value is 35. The
  qualitative point — a full team-state contract — is unchanged, and nothing in
  either experiment depends on the count.)
- **Measured** clean-core non-null counts for every cell's raw columns:
  `home_<stat>` 9,072 / `away_<stat>` 9,059 / `diff_<stat>` 9,038 of 9,093, and
  `spread_open` 9,076 of 9,093, identically for all three cells.

---

## 3. WHICH raw columns are replaced — one rule, applied to all three cells

**The rule: all three of `home_<stat>`, `away_<stat>` and `diff_<stat>` are
removed together.** Justified by two facts, both established before any outcome
number:

1. **The contract itself treats the metric as an atomic triple.** **Read**
   `src/nfl_ats/cfb_features.py:91-94`:
   `CFB_TEAM_STATE_FEATURES = tuple(column for metric in CFB_STATE_METRICS for
   column in (f"home_{metric}", f"away_{metric}", f"diff_{metric}"))`. The
   contract has no notion of "the diff feature" separable from its metric — the
   three columns are generated as one unit per metric, so "that statistic's raw
   columns" has exactly one honest reading.
2. **Removing only `diff_<stat>` would remove nothing.** **Measured** this
   session on all 12,500 rows: `diff_<stat>` equals `home_<stat> − away_<stat>`
   to within 0.0 (max absolute residual exactly 0.0 across 12,431 rows where
   all three are non-null, for each of the three cells). The estimator is a
   linear ridge on standardised columns, so with `home_` and `away_` still
   present it can reconstruct the differential exactly. A `diff_`-only swap
   would be a cosmetic edit that tests nothing.

So, for cell `<stat>`:

```
replacement_columns = tuple(c for c in CFB_MODEL_FEATURE_COLUMNS
                            if c not in {f"home_{stat}", f"away_{stat}", f"diff_{stat}"}
                           ) + (graph_v2_team_stat_<stat>_katz_diff,)
```

**Measured** sizes: benchmark 35 columns → replacement 33 columns (three out,
one in). Nothing else in the contract changes: the market features, context
features, experience features and the other seven team-state metrics' triples
are untouched, and `tests/test_graph_team_stat_cfb_replacement.py` asserts that
column-set identity exactly (section 8).

**The 3-for-1 asymmetry is declared, not hidden.** The graph builder attaches
one column — a signed-Katz differential — so a swap necessarily drops the
home/away LEVELS as well as the differential. A negative replacement delta could
therefore mean "the graph differential is worse than the raw differential" OR
"the home/away levels were carrying something the differential cannot". Section
4's third arm exists specifically to separate those two readings, and neither
is allowed to be reported as the other.

---

## 4. Arms and comparisons

Three arms, all fitted with the benchmark's own estimator and refit weekly:

| arm | feature contract | columns |
|---|---|---|
| `benchmark` | frozen `CFB_MODEL_FEATURE_COLUMNS` | 35 |
| `replacement` | benchmark minus the cell's `home_`/`away_`/`diff_`, plus the cell's graph katz differential | 33 |
| `ablation` | benchmark minus the cell's `home_`/`away_`/`diff_`, nothing added | 32 |

Comparisons, in declared priority order:

1. **PRIMARY — `replacement` − `benchmark`.** The question of this document: is
   the graph column a better representation of that statistic inside the
   contract than its three raw columns? This is the headline and the recorded
   effect.
2. **Secondary A — `ablation` − `benchmark`.** What the three raw columns are
   worth inside the contract at all. If this is ≈ 0, the primary is measuring a
   swap into an almost-free slot and must be read that way.
3. **Secondary B — `replacement` − `ablation`.** What the graph column is worth
   once the raw columns are gone — the cleanest "graph as the representation"
   read, and the arm that disambiguates the 3-for-1 asymmetry declared in
   section 3.

Both secondaries are reported and carried in the recorded entry's `notes`
field; neither becomes its own registry row, because a row per comparison would
triple-count one window.

---

## 5. Grade, metric, uncertainty

**Grade.** Exactly WP8's convention. `settle_margin = result − spread_line` is
the PRIMARY (the frozen XLG-03 benchmark's own grade, which is what keeps this
number commensurable with that instrument); the opener grade
`result − spread_open`, with `spread_line` REPLACED by `spread_open` in the
scoring frame so the market feature and the grade refer to the same number, is
computed and reported BESIDE it, never averaged with it. The project's "grade
the decision at the opener" rule means the close-graded figure is never quoted
alone.

Picks are forced by the production probability rule
`home_cover_probability >= 0.5` and graded with `nfl_ats.clv.pick_correct`;
pushes drop out; a row with no opener quote is unscorable at the opener grade,
not a zero, and is masked out of the opener comparison only.

**Metric.** Paired candidate-minus-reference forced-pick accuracy in
`accuracy_points` (percentage points), registry `--effect-units
accuracy_points`.

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, 1,000 samples, seed
20260901 (WP8's seed, reused so the two documents' intervals are drawn the same
way). `block="week"` is the PRIMARY — within-week correlation is zero by owner
mandate, so the week is the honest block — and `block="season"` is reported
beside it, never averaged in. `probability_positive` is reported for both. The
binary "contains zero" is never the verdict.

**Within-week permutation null.** 200 draws, settle margins shuffled within each
(season, week), reusing WP8's `permuted_margins` / `week_positions` by import.
This null is deliberately NOT centred on zero: within-week permutation preserves
each week's realised cover rate, so two arms with different home-pick rates have
a non-zero expected null delta. It is the conservative reference reported
ALONGSIDE the bootstrap, never instead of it, and one permutation is a draw, not
a test.

**Positive control.** `--mode positive-control` replaces the graph column in the
`replacement` arm — and only that column — with the realised `ats_margin`. The
`benchmark` and `ablation` arms are untouched, so the primary comparison becomes
"benchmark with a leak swapped in for one statistic's raw columns" versus
"benchmark". An instrument that cannot see that leak is blind and its null would
mean nothing.

**Run order, binding.** `--mode null` first, then `--mode positive-control`,
then `--mode screen` exactly once per cell. Reliability (section 6) runs before
all of them, because it involves no outcome at all.

**Era split (report only, no extra registry rows).** Era 1 = 2012–2019, era 2 =
2021–2025 — WP8's split, at the 2020 season the benchmark itself excludes. Per
the owner rule, these are MAGNITUDES; a weaker era is never written up as an
absence.

---

## 6. The reliability measurement (WP8's second unfinished item)

WP8's three registry rows carry a null `reliability`, and AGENTS.md makes that
field decisive: zero split-half reliability is one of only two admissible
closing grounds, and a signal recorded without it cannot be adjudicated later.
This work package measures it.

**Method — mirrored from `scripts/reliability_map.py`, stated explicitly.**
That script (**read** `scripts/reliability_map.py:200-247`) reshapes a
game-level table with `home_<x>`/`away_<x>` pairs into TEAM-WEEK long form —
each game contributes two rows, one per side, carrying `team_id`, `season`,
`week` and the metric value — and then calls
`nfl_ats.cfb_qb_dependence.split_half_reliability` (imported, never
reimplemented) on each metric. `split_half_reliability` (**read**
`src/nfl_ats/cfb_qb_dependence.py:486-546`) splits each team-season's values by
odd/even week, correlates the two halves' team-season MEANS (Pearson r and
Spearman rho), applies the Spearman–Brown full-length correction `2r/(1+r)`, and
block-bootstraps over team-seasons for a 95% CI and `probability_positive`. A
team-season needs ≥2 observations in EACH half to be included. This work package
uses the identical call with `n_boot=4000`, mirroring the map.

**What is measured, per cell — six numbers, three families:**

- `graph_<cell>_katz` — the graph trait itself. Its home/away pair is
  `home_graph_v2_team_stat_<cell>_katz` / `away_..._katz`, produced by the same
  `add_graph_ratings_v2_features` call WP8's builder makes; WP8's public
  function attaches only the differential, so this script calls the underlying
  builder with WP8's own `cfb_graph_config(cell)` and takes the pair. A
  consistency assertion re-derives WP8's differential from the pair and requires
  an exact match, so the two paths cannot silently diverge.
- `raw_<cell>` — the raw trait, `home_<cell>`/`away_<cell>`, as a REFERENCE.
  Reliability alone is not interpretable without knowing what the untransformed
  statistic scores on the same population; the comparison is the point.

**Population.** The clean core (2012–2019 + 2021–2025), the same window the
screen scores, so the reliability describes the trait as used. The CFB feature
table is already regular-season FBS-vs-FBS only, so `reliability_map.py`'s
`game_type == "REG"` restriction has no CFB analogue to apply.

**Seed** 20260901, `n_boot` 4000.

**How reliability is recorded — and the honest limitation.** `nfl-ats
weak-signals --help` (**measured** this session) exposes `status`, `record`,
`pool` and `retag-units`. `retag-units` changes only `effect_units`. `record`
has a `--reliability` flag but writes a NEW entry, and its `--replace` flag is a
whole-row overwrite of an existing name, not a field-level set — using it on
WP8's three rows would mean this work package rewriting another work package's
recorded entries in full. **No CLI sets `reliability` on an existing entry**, and
hand-editing registry JSON is forbidden. Therefore: the reliabilities go into
THIS document and into THIS work package's three new entries, and WP8's three
rows keep their null `reliability`. That gap is reported, not papered over.

**No closure is bought with this measurement in advance.** Reliability is
measured because AGENTS.md calls it decisive, not to license a negative. The
`no_split_half_reliability` ground requires a reliability RESOLVED at or below
zero on the trait itself — a point estimate near zero whose interval contains
zero is `unresolved_below_power`, exactly as for any other quantity here.

---

## 7. Leakage

Leakage safety is INHERITED, not re-argued.
`tests/test_graph_team_stat_cfb_feature.py` (WP8, unmodified) already proves on
synthetic CFB-shaped frames that week `w` reads only through week `w−1`, that
the join back is by `game_id` in the caller's row order with no loss or
duplication, that adaptation A1's id mapping keeps a rebranded program a single
node, and that an undeclared cell is refused. This work package imports the same
builder and adds no new feature construction, so those proofs cover it. WP24's
own test file imports that module's public surface and asserts the CONTRACT
substitution instead (section 8).

The estimator side is unchanged: `fit_cfb_residual_model` is trained per week on
games that kicked off strictly before that week's earliest kickoff, with the
same 500-game floor.

---

## 8. Test contract (release-blocking)

`tests/test_graph_team_stat_cfb_replacement.py` must prove:

1. **The substitution is exact.** For every cell, the `replacement` contract
   contains the graph katz column and contains NONE of `home_<cell>`,
   `away_<cell>`, `diff_<cell>`.
2. **Nothing else changes.** The symmetric difference between the `benchmark`
   contract and the `replacement` contract is exactly those four column names —
   every other benchmark column survives, in the benchmark's own order, and no
   other cell's team-state triple is disturbed.
3. **The FITTED design matrix agrees with the contract.** A model fitted on a
   synthetic CFB frame exposes `feature_names_in_` on its estimator equal to the
   declared contract, so the assertion is about what the ridge actually saw, not
   about a tuple that could drift from it.
4. **The ablation arm adds nothing.** Its contract is the benchmark minus the
   same three columns and is a strict subset of the benchmark.
5. **Undeclared cells are refused**, inheriting WP8's cell gate.
6. **The reliability long-frame reshape is faithful.** Each game contributes
   exactly two rows, home value to the home id and away value to the away id,
   with no cross-assignment.

---

## 9. Decision rule

Expected value: `probability_positive` above 0.5 favours the candidate.

**CFB is replication evidence and never by itself changes an NFL card.** No
result in section 11 promotes, demotes, or edits any NFL feature profile,
`artifacts/active_ats_model.json`, or `CURRENT_PREDICTIONS.md`.

What a CFB result CAN do is decide whether an NFL predeclaration is worth
writing. The rule, frozen here:

- If the primary (`replacement` − `benchmark`) lands with `probability_positive`
  **above 0.5 on a majority of the three cells** at the opener grade, the NFL
  follow-up in section 10 is warranted on EV and should be written as its own
  predeclaration.
- If it lands at or below 0.5 on a majority, the follow-up is not warranted on
  EV *at the CFB reading's strength*, and the cells are recorded
  `unresolved_below_power` — which is a description of the evidence, not a
  closure. Nothing is refuted by a P+ below 0.5; only the two admissible grounds
  in section 0 can close this line.

Either way every cell is recorded. Neither branch is a promotion decision.

---

## 10. NFL follow-up (predeclared design, NOT run by this work package)

Written now so that section 11's numbers cannot shape it. This is a SPEC, not a
result, and running it is a separate work package's decision.

- **Family** `graph_replacement_on_production`, `--league nfl`.
- **Question.** Does swapping a production-chain team statistic's raw
  `home_`/`away_`/`diff_` columns for its graph katz differential beat the
  production chain as it stands? Note this is NOT what NFL has already measured:
  the three published NFL readings (−0.935 / −0.668 / −0.935 accuracy points,
  P+ 0.122 / 0.189 / 0.037 on 749 games — **read**
  `docs/graph_team_stat_cfb_replication.md` §1 and the WP24 brief) are ADD-ON
  readings, the same comparator WP8 ran on CFB, and they say nothing about a
  swap.
- **Cells.** The NFL statistics whose raw `home_`/`away_`/`diff_` triples are
  actually present in the production feature contract, chosen by reading that
  contract, capped at three, declared before any sign.
- **Arms.** `production` (the active chain, unmodified) / `replacement`
  (production with the cell's raw triple swapped for its graph katz
  differential) / `ablation` (production minus the raw triple). Same three
  comparisons, same priority order, as section 4.
- **Grade.** OPENER primary — the project's declared primary goal and the grade
  the pool settles on — with the close reported beside it. This INVERTS WP8's
  and this document's CFB grade priority, deliberately: the CFB documents lead
  with the close only to stay commensurable with the frozen XLG-03 instrument,
  and no such constraint exists on the NFL production chain.
- **Window.** A window declared by the family's own rotation entry via
  `nfl-ats rotation`, never a window reused silently. Windows retire per-family,
  not globally; a reused window carries a stated discount, not a ban.
- **Uncertainty, null, positive control, era split.** Identical in kind to
  sections 5 — week-blocked bootstrap primary, season-blocked secondary,
  200-draw within-week permutation null, `ats_margin` leak as the positive
  control run before the screen, era magnitudes reported.
- **Decision.** EV. A promotion bar is not a decision bar: the pool is forced
  picks, so a candidate more likely than not to be better is played, and any
  predeclared threshold governs only what the docs may claim.
- **Prerequisite.** The graph columns must exist for the chosen NFL cells at
  the frozen structure, and their split-half reliability must be measured on the
  NFL population by the same `split_half_reliability` call used in section 6 —
  never assumed from the CFB numbers.

---

## 11. Results (added after the look, 2026-09-01)

_Nothing above this line was edited after an outcome number was seen._

Every number below is **measured** this session. Artifacts live under
`artifacts/graph_team_stat_cfb_replacement/`:
`<cell>_<mode>/<timestamp>/results.json` for the screen and its two instrument
checks, `reliability/<timestamp>/results.json` for section 6.

### 11.1 Split-half reliability (run first, no outcome touched)

`artifacts/graph_team_stat_cfb_replacement/reliability/20260901T191734Z/results.json`,
clean core, seed 20260901, `n_boot` 4000, 1,672 team-seasons for every graph
trait and 1,671 for every raw reference.

| trait | Spearman–Brown | Pearson r | 95% CI on r | P+ |
|---|---|---|---|---|
| `graph_def_epa_per_play_katz` | **+0.9929** | +0.9858 | [+0.9842, +0.9872] | 1.000 |
| `raw_def_epa_per_play` (reference) | +0.9775 | +0.9560 | [+0.9497, +0.9617] | 1.000 |
| `graph_off_epa_per_play_katz` | **+0.9925** | +0.9850 | [+0.9832, +0.9867] | 1.000 |
| `raw_off_epa_per_play` (reference) | +0.9800 | +0.9609 | [+0.9548, +0.9667] | 1.000 |
| `graph_off_success_rate_katz` | **+0.9938** | +0.9877 | [+0.9861, +0.9890] | 1.000 |
| `raw_off_success_rate` (reference) | +0.9799 | +0.9606 | [+0.9563, +0.9644] | 1.000 |

The consistency assertion passed on all three cells: re-deriving WP8's
differential from the rating pair reproduces it with a maximum absolute gap of
**0.000e+00** over 12,459 rows, so the two builders have not diverged.

**What this settles.** `no_split_half_reliability` is now **definitively
inadmissible** as a closing ground for any CFB graph `team_stat` cell — WP8's
three cells included. The traits are not noise; they are near-perfectly stable
odd-to-even week within a team-season, and the graph trait is *more* stable than
the raw statistic it propagates in all three cells (+0.0154 / +0.0125 / +0.0139
Spearman–Brown). Nothing about these signals can be closed by pointing at
reliability again.

### 11.2 Instrument checks (run first, both passed)

**Null** (`--mode null`, 200 within-week permutations, close grade), all three
comparisons:

| cell | primary null mean | sd | 95% |
|---|---|---|---|
| `def_epa_per_play` | +0.034 pts | 0.213 | [−0.426, +0.471] |
| `off_epa_per_play` | −0.011 pts | 0.304 | [−0.616, +0.550] |
| `off_success_rate` | +0.078 pts | 0.323 | [−0.517, +0.705] |

Every primary null mean is within 0.25 sd of zero. The two secondaries are
likewise near-centred (means −0.101 to +0.095), because in this design all three
arms carry the same market feature and their home-pick rates differ by at most
1.3 points — the home-tilt artifact this null exists to expose is small here,
unlike WP8's `graph_only`-vs-`market` comparison.

**Positive control** (`--mode positive-control`, the graph column in the
`replacement` arm replaced by the realised `ats_margin`): primary delta
**+48.405** accuracy points at the close, week-blocked P+ **1.000**, 95%
[+47.374, +49.464], at the **100.0th percentile** of its own null; opener
+44.885 / +44.908 / +44.852 across the three cells. The `ablation` arm was
untouched by the leak, as the design requires (its delta is byte-identical to
the screen run's: −0.067 close). The instrument is not blind to a real effect
embedded among 33 features. It is **not** thereby proven able to resolve a
0.3-point effect, so `positive_control_bound` is inadmissible for every cell
below.

**Benchmark reproduction.** The `benchmark` reference arm scores **51.5952%** at
the close and **51.7199%** at the opener on the paired population — identical to
the figures WP8 published for the same arm (**read**
`docs/graph_team_stat_cfb_replication.md` §10.2). The shared machinery
reproduces exactly across the two work packages.

### 11.3 The screen — PRIMARY (`replacement` − `benchmark`)

199 weeks fitted, 13 seasons, 9,093 clean-core rows, 8,933 graded at the close
and 8,925 at the opener.

| cell | delta (close) | week 95% | week P+ | season P+ | delta (opener) | opener week 95% | opener P+ | picks moved |
|---|---|---|---|---|---|---|---|---|
| `def_epa_per_play` | **−0.257 pts** | [−0.684, +0.233] | **0.145** | 0.179 | −0.022 pts | [−0.478, +0.471] | **0.462** | 450 / 9,093 = 4.95% |
| `off_epa_per_play` | **+0.056 pts** | [−0.487, +0.638] | **0.573** | 0.584 | +0.034 pts | [−0.492, +0.566] | **0.543** | 709 / 9,093 = 7.80% |
| `off_success_rate` | **+0.022 pts** | [−0.696, +0.807] | **0.528** | 0.512 | −0.325 pts | [−1.072, +0.417] | **0.221** | 1,043 / 9,093 = 11.47% |

Observed deltas sit at the 8.0th, 60.0th and 40.0th percentiles of their own
permutation nulls.

**Side by side with WP8's add-on read on the same window and grade:**

| cell | WP8 add-on (close) | WP8 add-on P+ | WP8 picks moved | WP24 replacement (close) | WP24 P+ | WP24 picks moved |
|---|---|---|---|---|---|---|
| `def_epa_per_play` | −0.011 pts | 0.467 | 1.92% | −0.257 pts | 0.145 | 4.95% |
| `off_epa_per_play` | +0.022 pts | 0.535 | 4.56% | +0.056 pts | 0.573 | 7.80% |
| `off_success_rate` | −0.179 pts | 0.266 | 5.10% | +0.022 pts | 0.528 | 11.47% |

(WP8 column **read** from `docs/graph_team_stat_cfb_replication.md` §10.2.)

### 11.4 The screen — the two secondaries, which carry the mechanism

**Secondary A — `ablation` − `benchmark`: what the raw triple is worth at all.**
Delta is candidate minus reference, so a NEGATIVE number means dropping the
three raw columns cost that much.

| cell | close | week P+ | opener | opener P+ | picks moved (close) |
|---|---|---|---|---|---|
| `def_epa_per_play` | −0.067 pts | 0.396 | +0.090 pts | 0.662 | 387 / 9,093 = 4.26% |
| `off_epa_per_play` | −0.101 pts | 0.367 | −0.179 pts | 0.261 | 584 / 9,093 = 6.42% |
| `off_success_rate` | −0.101 pts | 0.392 | −0.213 pts | 0.301 | 933 / 9,093 = 10.26% |

**Secondary B — `replacement` − `ablation`: the graph column in the vacated
slot.** This is the cleanest "graph as the representation" read and the arm
section 3 declared to disambiguate the 3-for-1 swap.

| cell | close | week P+ | season P+ | opener | opener P+ | picks moved (close) |
|---|---|---|---|---|---|---|
| `def_epa_per_play` | −0.190 pts | 0.064 | 0.046 | −0.112 pts | 0.175 | 171 / 9,093 = 1.88% |
| `off_epa_per_play` | +0.157 pts | 0.749 | 0.677 | **+0.213 pts** | **0.827** | 445 / 9,093 = 4.89% |
| `off_success_rate` | +0.123 pts | 0.710 | 0.708 | −0.112 pts | 0.313 | 348 / 9,093 = 3.83% |

### 11.5 Era magnitudes (report only, close grade, primary comparison)

| cell | 2012–2019 (n=5,349) | 2021–2025 (n=3,584) |
|---|---|---|
| `def_epa_per_play` | +0.000 pts, P+ 0.485, 95% [−0.563, +0.582] | −0.642 pts, P+ 0.040, 95% [−1.408, +0.112] |
| `off_epa_per_play` | +0.262 pts, P+ 0.770, 95% [−0.375, +0.872] | −0.251 pts, P+ 0.272, 95% [−1.226, +0.618] |
| `off_success_rate` | −0.037 pts, P+ 0.468, 95% [−0.941, +0.815] | +0.112 pts, P+ 0.550, 95% [−1.228, +1.443] |

Magnitudes, per the owner rule — no era is described as showing no effect. The
largest single era magnitude is `def_epa_per_play` in 2021–2025 at −0.642
points, and its 95% interval still reaches +0.112, so it is **not** a resolved
wrong sign and closes nothing.

### What this implies for the decision, before what is wrong with it

**1. The decision the predeclaration asked for: the NFL replacement follow-up is
NOT warranted at this reading's strength.** Section 9's frozen rule is a
majority of the three cells above P+ 0.5 on the primary at the OPENER grade.
Measured: 0.462 / 0.543 / 0.221 — **one of three**. (At the close it is two of
three, 0.145 / 0.573 / 0.528, but the rule named the opener before the numbers
existed and the opener is the project's declared primary grade, so the close
does not get to overturn it.) The section 10 spec stays written and unrun. This
is a statement about the strength of the CFB evidence, **not** a closure: no
interval sits entirely on one side of zero, so `wrong_sign_resolved` is
inadmissible; the positive control resolves +48 points and not ±0.3, so
`positive_control_bound` is inadmissible; and section 11.1 has just made
`no_split_half_reliability` inadmissible outright. All three cells are
`unresolved_below_power`.

**2. Substitution is worth more than addition on two of three cells, and the
gap is the finding.** On `off_success_rate` the swap moves the reading from WP8's
−0.179 pts / P+ 0.266 to +0.022 pts / P+ 0.528 at the same grade on the same
window; on `off_epa_per_play` from +0.022 / P+ 0.535 to +0.056 / P+ 0.573; and
it moves it the other way on `def_epa_per_play`, −0.011 / P+ 0.467 to −0.257 /
P+ 0.145. So WP8's summary sentence — "not additional information once a full
team-state contract is present" — turns out to be answering a narrower question
than it sounds like. Addition and substitution genuinely differ, the difference
is worth up to 0.2 accuracy points, and it is not one-signed.

**3. The mechanism is visible in secondary B, and it splits by side of the
ball.** Once the raw columns are gone, putting the graph column into their slot
is favoured on both offence cells (`off_epa_per_play` +0.157 close / **+0.213
opener at P+ 0.827**; `off_success_rate` +0.123 close at P+ 0.710) and disfavoured
on the defence cell (−0.190 close at P+ 0.064, −0.112 opener at P+ 0.175). That
is a sharper, better-posed question than the 3-for-1 swap the primary measures,
and on EV it favours the graph column on offence. It is *not* claimed as a
finding here: secondary B was declared as a diagnostic, and an offence-only
version is a different family that would need its own predeclaration and its own
declared cells before the signs were seen.

**4. The raw team-state triples are close to free real estate.** Secondary A
says deleting a metric's whole `home_`/`away_`/`diff_` triple from the 35-column
contract costs at most 0.21 accuracy points and on `def_epa_per_play` at the
opener is worth +0.090 — i.e. the ablation was *better*. That is decision-relevant
independently of the graph: the XLG-03 CFB benchmark's individual team-state
metrics are largely redundant with each other and with the market column, which
is why any single-metric swap into that slot has so little room to move. It also
means the primary's near-zero readings are exactly what a swap into an
almost-free slot should produce, and should not be read as "the graph column
failed".

**5. Reliability is now settled for this whole family and it is high.** Every
CFB graph trait scores Spearman–Brown ≥ +0.9925 with P+ 1.000, above its raw
counterpart in all three cells. Whatever is limiting these signals, it is not
that the trait is noise, and no future write-up may say it is.

Now what is wrong with it. **The swap is 3-for-1 and the primary cannot separate
the two things that change** — this was declared in section 3 and secondary B is
the partial answer, but the primary number alone conflates "the graph
differential versus the raw differential" with "losing the home/away levels".
**The three cells share one window and overlapping football**, so they are
correlated decompositions and are deliberately **not pooled**. **The primary
grade is the close**, frozen for commensurability with XLG-03, and the two
grades disagree in sign on `def_epa_per_play` (−0.257 close, −0.022 opener) and
on `off_success_rate` (+0.022 close, −0.325 opener) — at ±0.3-point resolution
that is a coin-flip-sized disagreement, so neither grade's number should be
quoted alone and the opener is the one the decision rule uses. **The
reliability numbers are inflated by construction relative to a per-game
reliability**: both the raw statistic (a strictly-lagged span-8 EWM) and the
katz rating (a cumulative, half-life-8 propagated state) are heavily
autocorrelated within a season, so an odd/even-week split-half is closer to a
stability measure than to an independent-observation reliability. That does not
weaken the conclusion drawn from it — the ground `no_split_half_reliability`
requires reliability resolved at or below zero and these are at +0.99 — but it
does mean the numbers should not be compared with reliabilities of event-like
per-game traits elsewhere in the registry. **And the graph column was never
retuned for a replacement role**: the frozen structure was chosen for NFL as an
ADD-ON, and a swap arm is the first thing that would justify re-deriving
`half_life_weeks` or `alpha` — deliberately not done here, because a
replication that retunes answers a different question.

### 11.6 Recorded

Three entries, `--league cfb`, `--family graph_team_stat_cfb_replacement`,
`--category onfield`, `--effect-units accuracy_points`, seasons 2012–2025, all
**`unresolved_below_power`** with `--reliability` filled from section 11.1. The
recorded effect is the PRIMARY at the CLOSE grade (the predeclared primary
comparison and primary grade); the opener figures, both secondaries, the era
magnitudes and the picks-moved counts are carried in each entry's `notes`.

| name | effect (close) | interval | week P+ | reliability |
|---|---|---|---|---|
| `graph_team_stat_cfb_replacement_def_epa_per_play` | −0.2575 | [−0.6844, +0.2326] | 0.145 | 0.9929 |
| `graph_team_stat_cfb_replacement_off_epa_per_play` | +0.0560 | [−0.4866, +0.6375] | 0.573 | 0.9925 |
| `graph_team_stat_cfb_replacement_off_success_rate` | +0.0224 | [−0.6961, +0.8071] | 0.528 | 0.9938 |

**WP8's three rows still carry a null `reliability`, and this work package did
not fill them.** `nfl-ats weak-signals` exposes no command that sets a field on
an existing entry: `retag-units` changes only `effect_units`, and `record
--replace` is a whole-row overwrite of another work package's entry rather than
a field-level set. Hand-editing registry JSON is forbidden. The measured
reliabilities therefore live in section 11.1 of this document and in the three
entries above; anyone adjudicating
`graph_team_stat_cfb_def_epa_per_play` / `_off_epa_per_play` /
`_off_success_rate` should read them from here.
