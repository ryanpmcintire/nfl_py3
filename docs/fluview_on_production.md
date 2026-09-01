# FluView elevated-illness indicators, stacked on PRODUCTION: predeclaration

Written **before any ATS number is produced by this comparison**, per the
same rule that governs `docs/graph_team_stat_on_production.md` (the sibling
protocol this document mirrors) and `docs/graph_ratings_v2_screen.md`.
**Sections 1-6 are the predeclaration** and contain no accuracy, cover-rate,
or `probability_positive` number against NFL outcomes. **Section 7 was added
after the look** and reports what it found; it changes nothing above it.

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

## 1. What this closes, and why it is not the same question as the screen

`docs/fluview_battery.md` predeclared and froze five cells measuring CDC
Delphi FluView state-level as-of illness (ILI) indicators against a BARE
market baseline (`home_cover` subset-vs-complement, no model). Two of those
cells are the direct ancestors of this document (**read**,
`docs/fluview_battery.md` section 8, recorded 2026-08-20, re-verified
2026-08-31):

- `fluview_away_market_elevated`: +0.368 accuracy points, week-blocked 95%
  CI [-0.257, +1.001], P+ 0.883.
- `fluview_home_market_elevated`: +0.309 accuracy points, week-blocked 95%
  CI [-0.409, +0.949], P+ 0.818.
- Underlying trait split-half reliability (shared by both cells): Pearson r
  0.9636, Spearman-Brown-corrected 0.9814.

Both were measured against a **zero-feature market baseline**, not against
the chain that is actually PLAYED. The project's own recorded lesson,
restated in ROADMAP.md and AGENTS.md — **"composition is not the signal"** —
is that an overlay or feature positive alone can go negative once stacked on
the chain that is actually played, because the played chain already explains
some of the variance a bare-baseline comparison credits to the candidate.
The marginal that decides is the one measured on top of what is played.

This document declares that stacked measurement for both cells — **it
answers a different question than the screen did**: do the FluView
elevated-illness indicators add anything on top of the full PRODUCTION
`weak_stack`/`market_residual`/ridge/alpha-10 chain (**read**,
`artifacts/active_ats_model.json`: `feature_profile: "weak_stack"`,
`method: "market_residual"`, `regressor: "ridge"`, `ridge_alpha: 10.0`),
rather than on top of a zero-feature market baseline?

The other three screened cells (`fluview_differential_home_worse`,
`fluview_peak_home_elevated`, `fluview_peak_away_elevated`) are **not**
carried into this measurement: they were the weakest three of the five by
both the raw screen and by population size (58-170 flagged games each,
against 206-214 for the two carried forward), and mixing five stacked
comparisons into one predeclaration would dilute the two cells the mission
actually asks about. They remain open, recorded `unresolved_below_power`,
untouched by this document.

## 2. The candidate features, reusing the frozen construction unchanged

Two columns, `fluview_home_market_elevated` and `fluview_away_market_elevated`
(`nfl_ats.fluview_production_feature`), at the exact frozen construction from
`docs/fluview_battery.md` (read that document's sections 2-3 in full; not
repeated or re-derived here):

- **Team → state mapping**: the same static 23-state `STATE_BY_TEAM` dict
  (`scripts/fluview_battery_ingest.py`), imported directly, not copied.
- **Point-in-time-safe as-of construction**: the same per-state checkpoint
  table (`build_checkpoint_tables`) and `merge_asof`-against-Tuesday-cutoff
  lookup (`attach_asof_ili`), both imported directly from
  `scripts/fluview_battery_screen.py` — the actual functions the frozen
  battery itself calls, not a reimplementation. `nfl_ats.fluview_production_feature`
  puts the repository root on `sys.path` to reach them, the same guarded
  pattern `nfl_ats.cli._ensure_repo_root_on_path` already uses for the same
  reason (a script-level construction reused from a src module).
- **Per-state top-decile thresholds**: read directly from the frozen
  battery's own already-recorded results artifact
  (`artifacts/fluview_battery/20260831T145604Z/results.json`,
  `state_thresholds` field) — computed ONCE on the battery's own population
  (docs/fluview_battery.md section 3), never re-derived here on a
  potentially different game population. This is the single most important
  reuse discipline in this document: a threshold re-derived on the
  PRODUCTION table's own (slightly different) game population would no
  longer be the frozen quantity the screen itself measured.
- **Location restriction**: inherited unchanged (docs/fluview_battery.md
  section 2) — both columns are NaN (missing, not "not elevated") for any
  game whose `location` is not `"Home"` (neutral/displaced sites), since the
  home-market mechanism does not apply there.
- **No new API calls.** Both the raw FluView snapshot
  (`data/raw/fluview/20260820T003258Z/fluview_raw.parquet`, 809,716 rows,
  24/24 regions, already on disk) and the frozen thresholds are reused
  as-is.

## 3. Candidate profiles: `weak_stack_fluview_home`, `weak_stack_fluview_away`

Two new `MarginFeatureProfile`s, each production `weak_stack`'s exact feature
set (`FEATURE_SETS["football_weak_stack"]` / `FEATURE_SETS["full_weak_stack"]`)
plus **exactly one** new column — `weak_stack_fluview_home` adds
`fluview_home_market_elevated`, `weak_stack_fluview_away` adds
`fluview_away_market_elevated`. Built on
`data/processed/game_features_weak_stack_fluview.parquet` (which carries
BOTH new columns, additively merged onto the PRODUCTION table by
`scripts/build_weak_stack_fluview_table.py`), never on
`weak_stack_v3`/`_surface`/`_v4`/`_graph_sack` — mirroring `weak_stack_v4`'s
and `weak_stack_graph_sack`'s own declared reason verbatim: stacking a
candidate onto a profile already refused or still undecided would confound
the answer to "does this add to what is actually played." Never referenced
by the active model. Never mixed with any other candidate profile, and the
two FluView profiles are never mixed with each other (each tests one
column's marginal in isolation, matching the screen's own F1/F2 cell
separation).

## 4. The comparison

**Two cells, one baseline, one evaluator, one shared rotation window,
close-graded:**

| cell | feature_profile | new column | predicted sign (inherited from the screen) |
|---|---|---|---|
| baseline (shared) | `weak_stack` (production, unmodified) | — | — |
| primary: away-market | `weak_stack_fluview_away` | `fluview_away_market_elevated` | POSITIVE |
| secondary: home-market | `weak_stack_fluview_home` | `fluview_home_market_elevated` | NEGATIVE |

Both arms hold `regressor="ridge"`, `ridge_alpha=10.0`,
`target="market_residual"` fixed at the active model's own values — only
`feature_profile` differs, isolating each FluView column's marginal
contribution against everything the production chain already explains. Both
are fit with `nfl_ats.margin.fit_margin_model`, the same estimator production
itself uses, not a single-feature model — this is the whole point of "on top
of production" rather than "on top of a bare baseline."

The primary quantity is the **paired candidate-minus-baseline** forced-pick
accuracy delta, in `accuracy_points` (percentage points), `pick_correct`
against `home_cover_probability >= 0.5` (the same probability rule production
plays), per `nfl_ats.clv.pick_correct`. **Away-market is the primary cell,
home-market the secondary cell** — both predeclared now, both recorded later
regardless of sign, matching the mission's own framing and the screen's own
relative strength (away P+ 0.883 vs home P+ 0.818 at the bare-baseline
grade).

## 5. Grade, window, and why a new rotation family with a stratified window

**Grade.** Close-graded, mirroring `docs/graph_team_stat_on_production.md`
section 5 and `docs/graph_ratings_v2_screen.md` section 6 in full: this is a
screen, not a play/no-play decision. Per the binding "grade the decision at
the opener" rule, nothing here may settle a play/no-play or promotion call;
every result here is recorded `unresolved_below_power` regardless of sign,
exactly as `scripts/graph_team_stat_record.py`'s `classify()` already
enforces for the sibling family.

**Why a new family.** No existing rotation family covers FluView: the
5-cell screen (`docs/fluview_battery.md`) was recorded directly to
`registry/weak_signals.json` via `nfl-ats weak-signals record`, never through
the rotation registry, so there is no spent rotation window to legitimately
inherit from (`src/nfl_ats/rotation.py`'s `inherits` field discloses genuine
hypothesis lineage — reusing an unrelated family's `inherits` purely to skip
past its already-spent seasons would misrepresent that lineage, so this
family declares `inherits=()`). This document therefore declares a new,
close-graded rotation family, **`fluview_elevated_on_production`**.

**The coverage constraint, disclosed before any window is drawn (measured,
this session, reproducing `docs/fluview_battery.md` section 8's own
measurement against the current production table).** FluView's
point-in-time-recoverable coverage is **0.0% for every NFL season
2009-2016**, 48.7% in 2017 (partial year, first checkpoints appear
2017-10-24), then 85.9-94.4% every season 2018-2025 (`fl`, hosting
JAX/MIA/TB, starts later still, 2021-10-15; `ny`, BUF's state, has zero
resolvable coverage across the whole panel — both gaps disclosed in
`docs/fluview_battery.md` section 1 and unchanged here). A **contiguous**
rotation window is legally unusable for this family: `nfl_ats.rotation`'s
warm-up floor (`MIN_ELIGIBLE_START_SEASON`) is **measured, this session, at
2011** (`nfl_ats.rotation.earliest_eligible_start_season` against the
current feature table), and `assign_window` always returns the **earliest**
eligible block for a fresh family — **measured, this session**: for a brand
-new close-graded family with `acknowledges_mined_2018_2025=True` and no
inherited history, `eligible_blocks(..., size=3)` returns `(2011, 2013)`
first, a block with **0.0% FluView coverage** by the measurement above. Every
contiguous default-size block through the mid-2010s is equally unusable, and
there is no honest way to skip past them without either (a) fabricating an
`inherits` relationship to an unrelated family purely to burn its
already-touched seasons, which this document refuses to do (section 5,
above), or (b) sequentially spending two or three real, mostly-uninformative
windows (2011-2013, then 2014-2016, then 2017-2019) merely to reach usable
seasons, wasting rotation capacity on a foregone-null measurement. Per the
mission's own contingency, this is the point at which a contiguous-window
plan would **stop before scoring and report the blocker** rather than
improvise around it.

**The era-stratified mechanism (`docs/era_stratified_windows_proposal.md`,
owner-approved 2026-08-19) is exactly the legal way through this
constraint**, and was built for close-graded families specifically. A
close-graded family may draw a **two-leg window** instead of a contiguous
block: `assign_stratified_window` deterministically pairs `(min(eligible),
max(eligible))` — no hidden choice, nothing to tune, nothing cherry-picked.
**Measured, this session**, simulating the assignment against the live
registry before declaring for real: for a fresh close-graded family with
`acknowledges_mined_2018_2025=True`, `eligible_stratified_seasons` returns
every season 2011-2025 (15 seasons, nothing yet touched), so the
deterministic pair is **(2011, 2025)**. Leg 2011 sits at the coverage floor
(0.0% FluView coverage, confirmed by direct model-fit probe below); leg 2025
sits deep in the well-covered era (94.0% coverage). This is not cherry-picked
to reach 2025 — it is what the registry's own fixed rule produces for this
family's position in the ledger, and it will be confirmed (not asserted) by
the real `rotation declare` / `rotation assign --stratified` calls in
section 7.

**Per-leg walk-forward, one design choice beyond `confirmation_split_legs`'s
own single-cutoff-per-leg convention, disclosed here.**
`nfl_ats.rotation.confirmation_split_legs` gives each leg ONE forward-chained
cutoff (every completed game strictly before that leg's first gameday) and
scores the whole leg with one fixed model. This document instead refits
**per week within each leg** — every week's model is trained on all
completed games strictly before that week's own earliest kickoff, exactly
matching `docs/graph_team_stat_on_production.md`'s own `run_window` design,
generalized from "weeks inside a contiguous range" to "weeks inside either
leg season." This is strictly **more** conservative than the registry
helper's own single-per-leg cutoff (training strictly prior to every week is
a fortiori training strictly prior to the leg), never less, and it keeps
this measurement's harness identical in shape to the sibling's own
already-reviewed one.

**A predeclared instrument check specific to this family's coverage gap,
run BEFORE the real screen, reported here in advance of the real modes
below (measured on this document's own feature table, not a cover-rate
result — the same "predictor-distribution-only computation" exception
`docs/team_style.md`'s reliability gate and this document's own section 6
use).** Fitting `weak_stack_fluview_away`/`_home` on training data where the
new column is **100% missing throughout** (true for all of leg 2011's own
walk-forward training, since every season before 2017 has zero FluView
coverage) does not raise: `sklearn`'s `SimpleImputer(strategy="median",
add_indicator=True)` silently drops an all-missing column from that week's
fit (a documented, pre-existing sklearn behaviour, not new to this
document), so leg 2011 collapses to a **degenerate, uninformative
comparison** — candidate and baseline predictions come back numerically
identical (max absolute probability difference **0.0**, confirmed by direct
probe against the real production table) for every 2011 game. This is the
**expected, predeclared shape** for the no-coverage leg: it will pool into
the primary week-blocked estimate at its own honest weight (zero
information, zero pull toward either sign) and is reported as its own
`LegResult` per the era-stratified proposal's binding refinement (below),
never silently averaged away.

## 6. Uncertainty and instrument checks, reusing the sibling's design unchanged

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, week-blocked as the
primary reference (within-week game correlation is zero by owner mandate)
and season-blocked as a secondary read (degenerate at only two season
values across two legs — reported anyway, never averaged with the primary,
same discipline the sibling doc applies to its own season-blocked read).
Same tool, same `BOOTSTRAP_SAMPLES=1000`/`SEED=20260826` constants
`scripts/graph_team_stat_off_sack_rate_on_production.py` already uses, for
comparability across the two "on production" families.

**Per-leg magnitudes are mandatory, not optional** — the owner's binding
refinement on the era-stratified proposal (docs/era_stratified_windows_proposal.md,
2026-08-19): "era variation is expected to be a change in effect MAGNITUDE,
not binary presence/absence." Here that refinement meets its most literal
case: leg 2011 is not a weaker-but-still-real era, it is a **zero
-information** era by direct data-coverage absence, and the registry's own
`record_look` validator refuses to load or record a spent stratified window
without a `leg_effects` entry naming both legs. Both this document and the
harness report each leg's own delta/`probability_positive`/`sample_blocks`
alongside the pooled (both-legs, week-blocked) read, never collapsed into
it, exactly as `docs/era_stratified_windows_proposal.md`'s implementation
requires.

**Within-week permutation null**, 200 permutations, identical mechanism to
`docs/graph_ratings_v2_screen.md` section 6 and
`docs/graph_team_stat_on_production.md` section 6: both arms' models are fit
ONCE per week on the REAL `ats_margin`; only the grading margin is shuffled
within week for the null (pooled across both legs), so 200 draws costs no
extra model fits. This null is **not** centred on zero by design (it
preserves each week's realized home-cover rate, and the two arms may carry
different home-pick rates), and is reported ALONGSIDE the bootstrap-vs-zero
interval, never instead of it.

**Positive control**, run BEFORE the real screen, per cell: the candidate
profile's one new column is temporarily REPLACED by the realized
`ats_margin` — a deliberate, large leak — so the harness must show an
obvious, large effect on the leg(s) where the leak is actually visible to
the model (leg 2025, which has real rows to leak into; leg 2011's own
positive-control read is expected to remain near-degenerate for the same
reason the real screen is: the model still cannot see a signal in a leg
whose training data received it in years the actual candidate column would
never have covered — the leak is injected as the SCORED column, so the
leg-2011 control DOES get real leaked information at scoring time even
though the real candidate never could; both legs are reported). This proves
the FULL-PROFILE ridge fit (not just a single-feature model) can detect a
real effect of meaningful size when one is actually present, exactly
mirroring the sibling family's own instrument check.

## Decision rule, frozen before scoring

This is a FORCED-PICK pool: 285 cards must be submitted either way. The
decision is expected value, never a 0.90/95% threshold —
`probability_positive` above 0.5 favours playing the candidate over the
baseline (predeclared thresholds govern only what a doc may CLAIM). This run
is close-graded, so it settles no play/no-play decision by itself; what it
DOES settle is whether the FluView family's screen-stage finding, measured
the honest way (stacked on what is actually played, not a bare baseline),
still looks worth an eventual opener-graded confirmation look — mirroring
`docs/graph_team_stat_on_production.md`'s own "what this implies for the
decision" framing exactly.

## Recording

Two `nfl-ats weak-signals record` entries, `effect_units=accuracy_points`,
family `fluview_elevated_on_production` (a DIFFERENT pooling bucket from the
original `fluview_battery` screen entries, stated explicitly: the two
measure the same construct against two non-commensurable comparators — bare
market baseline vs. the full production chain — and AGENTS.md's
commensurability rule forbids pooling them together). Both cells recorded
`unresolved_below_power` at this close grade regardless of sign, exactly as
`scripts/graph_team_stat_record.py::classify` already enforces for the
sibling family — a resolved wrong sign at this grade is reported as
continuous evidence, never as a `refuted_mechanism` closure, because that
closure is reserved for the opener grade.

One `nfl-ats rotation record --name fluview_elevated_on_production --verdict
unresolved --leg-effects '[...]'` call spends the assigned stratified
window, carrying the PRIMARY (away-market) cell's paired effect, interval,
`probability_positive`, and per-leg magnitudes; the secondary (home-market)
cell's numbers are disclosed in the same call's `--notes`, matching how
`docs/best_pick_ranker.md`'s own recorded window reports one primary number
plus the measured alternatives in notes.

## 7. Results (added after the look, 2026-08-31)

**Window, confirmed not asserted.** `nfl-ats rotation declare --name
fluview_elevated_on_production --grade close --acknowledge-mined` then
`nfl-ats rotation assign --name fluview_elevated_on_production --stratified`
returned legs **(2011, 2025)**, exactly as section 5 predicted from
simulating the assignment against the live registry before declaring.
15 seasons were eligible (2011-2025, nothing yet touched by this fresh
family); the deterministic `(min(eligible), max(eligible))` rule paired the
earliest with the latest.

**Instrument check 1 -- all-missing-column probe (measured, before any mode
was run).** Fitting `weak_stack_fluview_away` on leg 2011's own walk-forward
training data (every completed game before 2011, in which the new column is
100% missing) does not raise: `sklearn`'s `SimpleImputer` silently drops the
all-missing column for that week's fit, and the probe's direct comparison
confirmed baseline and candidate predictions come back numerically identical
(max absolute probability difference **0.0**) on 2011 week 1. Leg 2011 is a
clean, expected degenerate leg, not a crash or a silent miscount.

**Instrument check 2 -- null** (`--mode null`, 200 within-week permutations,
pooled across both legs): away cell mean **+0.103** accuracy points, sd
0.489, 95% [-0.775, +1.163], observed 0.000. Home cell mean **+0.135**
points, sd 0.850, 95% [-1.357, +1.744], observed +0.969. Both sane, finite
distributions.

**Instrument check 3 -- positive control** (`--mode positive-control`, the
candidate's one column replaced by the realized `ats_margin`): **identical
for both cells** (the same leaked column, by construction) -- pooled
**+52.326** accuracy points, week-blocked P+ **1.000**, 95% [+48.361,
+55.884], n=516 games. Both legs individually exceed +50 points (leg 2011:
+53.878, P+ 1.000; leg 2025: +50.923, P+ 1.000) -- the leak is injected
across the WHOLE historical panel used for training, not just the scored
leg, so leg 2011 detects it too despite carrying zero real FluView coverage.
The full-profile ridge fit is not blind to a real effect of meaningful size
on either leg.

**The real screen** (`--mode screen`, per cell):

| cell (role) | pooled delta | week-blocked 95% CI | week-blocked P+ | season-blocked P+ | leg 2011 delta / P+ | leg 2025 delta / P+ |
|---|---|---|---|---|---|---|
| away (primary) | **0.000** pts | [-1.156, +1.161] | **0.403** | 0.000 (degenerate) | 0.000 / 0.000 | 0.000 / 0.456 |
| home (secondary) | **+0.969** pts | [-1.150, +3.119] | **0.792** | 0.756 | 0.000 / 0.000 | +1.845 / 0.789 |

Artifacts: `artifacts/fluview_elevated_on_production/20260831T162041Z/results.json`
(away, screen), `artifacts/fluview_elevated_on_production/20260831T162153Z/results.json`
(home, screen); null/positive-control artifacts alongside them in the same
directory.

**Leg 2011 is exactly degenerate for both cells** (delta 0.000, P+ 0.000,
CI [0, 0]) -- baseline and candidate probabilities are numerically
identical for all 245 games, confirming the predeclared expectation (section
5) rather than revealing a bug.

**Leg 2025's away-cell delta is ALSO exactly 0.000, but not degenerately so**
-- verified by direct per-game inspection: 8 of 271 picks flip side between
the two arms, and by coincidence exactly 4 flip to correct and 4 flip to
incorrect, netting to zero. The underlying probabilities do shift (pooled
home-pick rate rises from 59.8% baseline to 61.4% candidate), so this is a
real but currently net-neutral effect on this forced-pick population, not an
inert feature -- disclosed here rather than read as "no effect."

**Permutation-null percentiles**: away cell's observed 0.000 sits at the
**27.5th** percentile of its own null (a mild negative lean); home cell's
observed +0.969 sits at the **78.0th** percentile (a positive lean, not an
extreme tail value).

### What this implies for the decision, before what is wrong with it

On EV grounds -- `probability_positive` above 0.5 favours playing the
candidate, the only decision rule this project uses -- the **home-market
cell (secondary) favours adding `fluview_home_market_elevated` on top of
what is actually played**: week-blocked P+ 0.792, season-blocked P+ 0.756,
both comfortably above 0.5, and roughly as strong as the same cell's own
bare-baseline screen reading (P+ 0.818) -- unlike the "composition is not
the signal" pattern seen elsewhere in this project, this marginal did **not**
evaporate once stacked on production. The **away-market cell (primary) is
close to a coin flip, leaning slightly against the candidate**: week-blocked
P+ 0.403. This DOES look like the "composition is not the signal" pattern --
the screen's own bare-baseline reading for this cell was P+ 0.883, and
essentially all of that lean is gone once measured on top of production.

This is **not** a closure for either cell: neither week-blocked interval
sits entirely on one side of zero (away: [-1.156, +1.161]; home: [-1.150,
+3.119]), so `wrong_sign_resolved` is unavailable for either, and the
positive control demonstrated the harness CAN see a large effect but did not
bound whether an effect the size of either cell's own screen-stage reading
would be reliably detected at this window's ~516-game sample (both
week-blocked CIs span more than 2 points, wider than either screen-stage
reading), so `bounded_by_control` is not available either. Both cells are
recorded `unresolved_below_power`, exactly as the predeclared close-grade
discipline requires, and the family stays **open**.

The honest reading, stated plainly: the away-market cell's bare-baseline
finding does not survive being measured the way that actually decides
anything; the home-market cell's does, and even reads a bit stronger. Both
are unresolved at this sample size, which per the binding taxonomy is the
expected shape for a real small signal and is never grounds to close the
line of work on its own -- and per the mission's own EV framing, the
home-market reading is exactly the kind of result ("87% likely better is not
caution, it's taking the other side of an 87/13 bet") this project's
promotion-bar rule says should not be waved off just because it has not yet
cleared an opener-graded confirmation.

**Caveats, after the numbers above, not instead of them.** (1) This is
CLOSE-graded; per the binding "grade the decision at the opener" rule, it
settles no play/no-play or promotion decision by itself -- an opener-graded
confirmation is the next legitimate step for the home-market cell
specifically. (2) The stratified window's own power is asymmetric: half the
window (leg 2011) carries zero information by construction, so this
measurement's effective sample is closer to one 271-game season than to a
516-game window, which the wide week-blocked CIs already reflect honestly.
(3) `fl` (JAX/MIA/TB) and `ny` (BUF) carry narrower or zero coverage even
within leg 2025 (docs/fluview_battery.md section 1), disclosed there and
unchanged here.

### Registry, verified by reading it back (not by trusting the CLI's own echo)

`registry/weak_signals.json`: **606 -> 608** signals (measured before and
after: `python -c "json.load(...)['signals']"`... length). Both new entries
read back with the expected `classification=unresolved_below_power`,
`family=fluview_elevated_on_production`: `fluview_away_market_elevated_on_production`
(effect 0.000, P+ 0.403) and `fluview_home_market_elevated_on_production`
(effect +0.969, P+ 0.792).

`registry/rotation_registry.json`: `fluview_elevated_on_production`'s window
`(2011, 2025)`, `window_kind: "stratified"`, is now `state: "spent"`,
`verdict: "unresolved"`, carrying the primary (away) cell's pooled
effect/interval/probability_positive plus both legs' `leg_effects`; the
secondary (home) cell's numbers are disclosed in the same entry's `notes`.

### Files touched

- `docs/fluview_on_production.md` (this document).
- `src/nfl_ats/fluview_production_feature.py` (new: derive/attach the two
  columns, reusing `scripts/fluview_battery_screen.py`'s as-of construction
  and the frozen battery's own recorded thresholds).
- `scripts/build_weak_stack_fluview_table.py` (new: builds
  `data/processed/game_features_weak_stack_fluview.parquet`).
- `scripts/fluview_elevated_on_production.py` (new: the era-stratified
  harness, `--cell {home,away}` x `--mode {null,positive-control,screen}`).
- `src/nfl_ats/constants.py`, `src/nfl_ats/margin.py` (new profiles
  `weak_stack_fluview_home`/`weak_stack_fluview_away`, mirroring
  `weak_stack_graph_sack`'s registration exactly).
- `src/nfl_ats/market_decomposition.py` (`FAMILY_PHRASES` entries for the two
  new families).
- `tests/test_features.py` (the four new `FEATURE_SETS` names added to the
  bias-inheritance "admitting" set expectation).
- `tests/test_fluview_production_feature.py` (new: 8 leakage/wiring tests).
- `registry/weak_signals.json`, `registry/rotation_registry.json` (the three
  record calls above).
