# Rotation registry: per-family confirmation windows

Written 2026-08-17 (queue item 2 in `docs/pool_edge_plan.md`). The registry
is the evaluation substrate for every research item behind it: it hands each
candidate model family a logged confirmation window it has never touched, so
we can iterate freely without silently spending the data our conclusions
depend on. It turns the methodology agreements from the pool-edge plan into
enforced code, not habits.

## The problem it solves

The August 2026 review counted **roughly 130-150 candidate streams scored
against the 2018-2025 outcomes**. At that multiplicity, the best pooled
numbers there (52.47-52.77%) are what selection on noise would produce, so
further mining of 2018-2025 with variants of existing families cannot yield
trustworthy gains. Meanwhile the replication program spent the untouched
pre-2018 windows for two families (2013-2017 for the raw-PBP/drive bundle,
2014-2017 for the QB+continuity player family). Window accounting currently
lives in prose (ROADMAP, docs); nothing stops a future session from
accidentally re-scoring a spent window.

## Binding rules

1. **One family, one declaration.** A family is a named research hypothesis
   (e.g. `mod07_weak_signal_stack`) declared BEFORE any confirmation run,
   with: description, the grade it targets (`opener`, `close`, or
   `nflverse_spread`), the training policy, and its inherited contamination
   (see rule 6). Declarations are append-only; editing a declaration after a
   look is recorded creates a NEW family.
2. **Windows are drawn, logged, then spent — in that order.** A confirmation
   window is a block of consecutive seasons assigned to a family at
   declaration (or later via an explicit `assign`). Scoring the family on
   that window is ONE look; recording the look marks the window spent for
   that family forever, whatever the verdict. Unrecorded looks are a
   methodology violation, not a shortcut.
3. **Forward-chaining only.** Training data must end strictly before the
   window's first game. Never random k-fold, never train-after-test. The
   enforcement helper refuses to build a frame that violates this.
4. **Windows retire per-family, not globally** (per the pool-edge plan).
   Two different families MAY draw overlapping seasons — their hypotheses
   are independent — but the ledger records global usage per season so
   accumulating cross-family multiplicity stays visible instead of silent.
5. **Grade determines the eligible pool.** Opener-graded confirmations
   need the paired Tuesday-opener archive: **2020-2025 only** (six
   seasons — scarce; the tooling reports remaining unspent opener capacity
   loudly). Close- and nflverse-spread-graded confirmations may draw from
   **2009-2025**.
6. **Contamination is inherited, honestly.** A family that is a variant of
   an existing line of work inherits that line's spent windows at
   declaration (declared as `inherits`). Any window intersecting
   **2018-2025** additionally requires the declaration to set an explicit
   `acknowledges_mined_2018_2025` flag — the ~130-150-look ledger means a
   result there carries a discount that the write-up must state.
7. **Continuous evidence, recorded verbatim.** A recorded look carries the
   artifact path, the headline metric, and `probability_positive` (the
   fraction of blocked resamples favoring the candidate) — never a bare
   pass/fail. "Unresolved at this sample size" is a recordable verdict.
8. **CFB and non-reserved seasons stay free.** The registry governs NFL
   confirmation looks only. Iteration on the CFB benchmark and on seasons
   no family has reserved needs no registry entry.
9. **Warm-up eligibility.** (Added 2026-08-17, before any window was spent
   under the rule.) No window may START before the evaluation substrate can
   score its first week. The rule asks a **feasibility** question, so both of
   its terms are derived rather than chosen: **50 training games**, the
   smallest set `margin.fit_margin_model` can actually fit (its own
   preconditions — `n >= 50`, `int(0.20n) >= 10`, and `n - int(0.20n) >= 40`
   — are first satisfied together at exactly 50, and the middle one fails at
   49), then **200 prior out-of-sample prediction rows** before the stream
   can be calibrated (`calibration.calibrate_cover_prediction_stream`,
   derived below). Walked against the real schedule that puts the floor at
   **2011**: no block starts before it. Enforced at assignment
   (`MIN_ELIGIBLE_START_SEASON`); `confirmation_split` additionally refuses
   any window with an empty training frame. Historical ledger entries are
   not re-judged, so lowering the floor never un-spends a window.

   **The two requirements are chained, not additive.** Prediction rows only
   begin accruing once the training floor is first met, and only at week
   granularity, since a week is scorable only if every game before it clears
   the floor. Summing them under-counts by up to one partial week at each
   boundary. That error was invisible while the training term was 500 — the
   slack absorbed it — and became load-bearing at 50, where the naive sum
   claims 2010, a season whose week 1 has too few prediction rows behind it
   to calibrate. `rotation.earliest_eligible_start_season` computes the exact
   answer from a real schedule, and a test pins the closed form against it so
   it can never again drift optimistic.

   Origin: the first `nflverse_spread` block [2009, 2011]
   was offered to `best_pick_ranker` — warm-up would have consumed
   2009-2010 entirely (17 scorable weeks, all in 2011) and calibration
   could not run at all (`docs/archive/opus_session_blockers.md`, Issue 1).

   > **The floor moved 2013 → 2012 on 2026-08-17, because the calibration
   > constant behind it was finally derived instead of inherited.** Both
   > constants were undocumented defaults nobody had tested:
   > `min_train_games=500` is ten times `fit_margin_model`'s own stated
   > minimum of 50 games, and `min_calibration_games=400` demanded 200
   > observations per parameter to fit a two-parameter Platt sigmoid — and
   > raised rather than degrading. An underived constant has no claim to
   > correctness, and these were load-bearing for an irreversible decision.
   >
   > **Calibration floor — measured, then changed to 200.** A floor is real,
   > but 400 was twice what the evidence supports. Measured on the real
   > 2009-2025 walk-forward stream by opening the gate fully and bucketing
   > calibrated-vs-raw Brier by the history each week's calibrator actually
   > had: 100-199 rows makes Brier **worse** (0.206 → 0.284, on only 16
   > games); **200-399 rows already improves it** (0.269 → 0.250 on 204
   > games); 400-799 improves it by about the same (+0.017); 800+ by less as
   > the raw stream sharpens. 200 is the smallest demonstrated-safe value.
   > Consequence at the time: the requirement became 500 + 200 = 700 games,
   > covered by three prior seasons (768), moving the floor to 2012 and
   > giving a fresh `nflverse_spread` family [2012, 2014]. The training term
   > was replaced later the same day (below), which moved it again to 2011.
   >
   > **Training floor — measured 2026-08-17, and no derivable value exists.**
   > The question was put to the data twice, and both answers agree that the
   > quantity 500 was meant to protect **has no threshold**:
   >
   > - **NFL, test rows held fixed.** Holding the evaluated games constant
   >   (2012-2025, 3,573 games, all features warm) and truncating training to
   >   the most recent N isolates sample size as the only moving part.
   >   Forced-pick ATS accuracy — the project's actual bar — is **flat**:
   >   .509 at N=50, .499 at N=500, .508 on full training, with every paired
   >   blocked-bootstrap delta straddling zero. Brier and residual MAE degrade
   >   smoothly and monotonically with **no cliff at 500 or anywhere else**.
   > - **CFB, 12,500 games** *(corrected 2026-08-18; this line previously
   >   read 12,206, the wrong count for the full 2006-2025 corpus — the
   >   parquet has 12,500 rows, measured directly)*. The segment this floor
   >   REFUSES (training rows
   >   < 500) scores 0.4906, week-blocked [0.4313, 0.5511],
   >   `probability_positive` 0.376 on 12 independent blocks — unresolved and
   >   leaning mildly negative, **not demonstrably bad**. This is the
   >   measurement the earlier note called impossible; it is impossible only
   >   for the `clean_core` *split* (hardcoded to 2012+), not for the
   >   benchmark, which populates the thin buckets directly at floor 50.
   >
   > The real failure mode below ~500 is **over-confidence, not error**: mean
   > |predicted residual| is 8.84 at N=50 against 1.68 at full training, and
   > the Brier half of that is fully repaired by the already-derived 200-row
   > calibration floor (raw .313 → calibrated .2504 at N=50). A cliff is the
   > wrong instrument for a regularization problem.
   >
   > **What changed: 500 no longer gates this rule.** Rule 9 asks whether a
   > window's first week can be *scored*, which is feasibility, so it now
   > derives from `fit_margin_model`'s own arithmetic minimum instead of
   > borrowing an undefended reporting default. 500 survives in
   > `constants.DEFAULT_MIN_TRAIN_GAMES` only as a conservative default for
   > reporting runs, where it binds nothing — every NFL evaluation window
   > sits over 1,000 games clear of it. The floor moved **2012 → 2011**, and
   > a fresh `nflverse_spread` family now draws **[2011, 2013]**, which also
   > lets an inheriting family avoid reaching into the mined 2018-2025 era
   > for its first window.
   >
   > **Not a defect, checked:** `fit_margin_model`'s 80/20 split does NOT
   > shorten training. The 20% holdout trains only a throwaway model to
   > produce honest out-of-time residuals; the returned estimator is refitted
   > on all training rows.
   >
   > **Reporting quirk:** moving the floor shifts the fixed capacity
   > partition, so `rotation status`'s "N windows unspent" counter can fall
   > while real capacity rises. The counter is a rough global gauge;
   > per-family eligibility is the binding quantity.
   >
   > **Contamination audit (2026-08-17): no recorded result needs re-running.**
   > All 36 artifacts that record `min_train_games` were checked for whether
   > the floor actually BOUND — whether any week inside the artifact's own
   > evaluation window would have been scored at a lower floor and was
   > skipped at 500. Every NFL experiment: **zero weeks skipped**, because
   > every NFL window starts in 2013 or later and the thinnest of them still
   > had 1,024 completed games in front of it (2018-start experiments had
   > 2,304; 2022-start, 3,344). The floor is a startup gate on an expanding
   > window, never a cap, so it cannot bite once a window clears it. That
   > covers the frozen 52.05% backtest, the 52.50% opener measurement, the
   > market decomposition, every player/participation/availability ablation,
   > and both spent replications (2013-2017 pbp at 1,024; 2014-2017 QB at
   > 1,280).
   >
   > The single place it bound is the CFB benchmark: **14 weeks skipped, all
   > in 2006-2007**, entirely inside `CFB_THIN_REGIME_SEASONS`. They are
   > therefore absent from the `thin_2006_2011` and `all` splits, and absent
   > from `clean_core` — the 8,933-game XLG-03 headline — only because
   > `clean_core` excludes those seasons anyway. The affected splits are
   > already published as separate, explicitly-thin regimes, so no
   > conclusion rests on them.
   >
   > Tightest calibration margin found: `player_model_selection` ran with
   > 512 prior prediction rows against the 400 requirement, and the SPEC-5
   > screen with the same 512. Neither failed, but both clear by ~28%, so a
   > modestly higher constant would have made them impossible. The constants
   > shape what is *reachable* far more than what was *computed*.
   >
   > Net: the damage is capacity, not correctness — one three-season
   > confirmation block permanently removed from the `nflverse_spread` pool.
   >
   > The right fix is not a better threshold. The target is the residual
   > from the market line, so the null model is "residual = 0" — a good
   > prior. Shrink toward it and predict from game one, letting data earn
   > weight, instead of refusing to predict below an arbitrary cliff; same
   > for the calibrator, starting from the identity map. That was the MOD-06
   > proposal on the ROADMAP.
   >
   > **MOD-06 was then built and it falsified this paragraph** (2026-08-17,
   > 12,500 free CFB games *(corrected 2026-08-18; this line previously read
   > 12,206, the wrong count for the full 2006-2025 corpus — the parquet has
   > 12,500 rows, measured directly)*). Sweeping shrinkage across five orders
   > of magnitude moves forced-pick accuracy by less than a point, and moving
   > it UP — the direction the argument predicts — makes the thin-data
   > buckets *worse*, not better.
   >
   > **The "structural" reason given here for why this had to be true is
   > wrong, and was retracted 2026-08-17 in `docs/pool_edge_plan.md` — kept
   > verbatim below, not deleted, per this project's rule for recording
   > superseded claims:**
   >
   > > The reason is structural: the pick is `sign(predicted residual)`, and
   > > rescaling a prediction by any positive scalar cannot change a sign, so
   > > **any pooling scheme that only rescales is a no-op for the pool
   > > metric.**
   >
   > **That premise is false.** The production forced pick is
   > `home_cover_probability >= 0.5` (`pool.py:41`, `backtest.py:56`), which
   > thresholds the *median of the out-of-time residual sample shifted by the
   > prediction* — not the sign of the prediction. The two rules disagree on
   > **11.8% of the 2,075 scored games** (244, measured directly), and the
   > production probability rule beats the sign rule by **+2.12 points**
   > (season-blocked 95% `[+0.24, +4.17]`, `probability_positive` **0.990**).
   > **MOD-06's own empirical conclusion survives** — sweeping shrinkage over
   > five orders of magnitude on the actual production rule still moved
   > accuracy by under a point, in the wrong direction — but the *general
   > mechanism* claimed here, "any rescaling pooling scheme is a no-op for the
   > pool metric," does not survive and must not be cited to close future
   > pooling, penalty-structure, or calibration work. There is no warm-up ramp
   > to remove. What replaced the cliff was evidence, not a model: the floor
   > now derives
   > from feasibility (above), which is what rule 9 was always asking.
   > Windows already spent stay spent regardless.

## The ledger

`registry/rotation_registry.json` — **git-tracked** (committed with the
look that changes it, so the history of spent windows is the git history),
schema-versioned, and validated on load. Structure:

- `families.<name>`: declaration fields, status
  (`open` / `confirmed` / `closed_negative` / `retired` /
  `declared_for_coverage`, the last added 2026-09-04 by ENG-27 — see
  "Coverage" below), and its window list, each window `{seasons, state:
  assigned|spent, assigned_at, spent_at, artifact, verdict,
  probability_positive, notes}`. A `declared_for_coverage` family also
  carries `coverage_weak_signal_family` / `coverage_league` /
  `coverage_effect_units`; every other family omits those three keys
  entirely rather than writing them `null`.
- `season_usage`: derived global count of families that have spent each
  season (reported by the CLI, recomputed on write).
- `no_rotation_needed.<weak_signal_family>` (added 2026-09-04, ENG-27):
  weak-signal families explicitly excused from ever needing a rotation
  family — see "Coverage" below. Omitted entirely while empty.
- `notes`: the standing 2018-2025 multiplicity acknowledgment and pointers
  to the prose record (ROADMAP, RWB-16).

Seeded at creation with the documented history:

- `pbp_drive_bundle` — closed_negative; **2013-2017 spent**
  (the -0.08-point replication on 1,247 games).
- `player_qb_continuity` — closed_negative; **2014-2017 spent**
  (the +0.00-point replication on 997 games).
- `cfb_role_continuity` — closed_negative at the CFB benchmark; no NFL
  window was ever assigned (recorded so the family name is reserved).

## Window mechanics

- Default window size **3 consecutive seasons** (~800 games — the scale the
  2026-08 replications resolved real effects at); declarable at 2-4.
  Opener-graded windows default to **2 seasons** (the 2020-2025 pool only
  holds three such windows; spending them deserves deliberation).
- **Era-stratified windows** (added 2026-08-19,
  `docs/era_stratified_windows_proposal.md`, owner-approved): a **close-graded**
  family may instead draw a two-leg **stratified** window — two non-adjacent
  single-season legs, each scored walk-forward with its own training cutoff,
  rather than one contiguous block. `Window.window_kind` distinguishes the
  two (`"contiguous"` default, `"stratified"`); a stratified window's
  `seasons` holds its two leg seasons, not a range endpoint pair, and
  `Window.covered_seasons` is the general accessor every overlap/usage/
  capacity computation uses. Opener- and nflverse_spread-graded families
  cannot request one (scope limit; see the proposal doc's "Resolution
  decisions" for why nflverse_spread is excluded too). The owner's binding
  refinement: era variation is a change in effect **magnitude**, so a
  stratified window's `record_look` call requires a per-leg magnitude
  (`leg_effects`) alongside the pooled read — enforced, not just documented.
- Assignment is the **earliest eligible block**: the lowest-starting block
  of the requested size, within the grade's pool, that (a) starts at or
  after rule 9's warm-up floor (2011), (b) does not intersect any window
  this family or its `inherits` chain has spent or holds, and (c) satisfies
  rule 6's acknowledgment requirement. Assignment is deterministic given
  the ledger — no hidden choice, nothing to tune.
- A family may hold at most ONE assigned-unspent window at a time.

## Enforcement API (`nfl_ats.rotation`)

- `load_registry()` / round-trip validation (unknown fields, overlapping
  same-family windows, spent-without-artifact all raise).
- `declare_family(...)`, `assign_window(family, size=None)`.
- `confirmation_split(features, family)` → `(training, window)` frames:
  window rows are exactly the assigned seasons (REG rows only, via the
  existing `regular_season_rows` guard); training is every completed game
  strictly before the window's first gameday. Raises if no assigned window
  exists, if it is spent, or if the feature table lacks the window seasons.
  Raises with a redirect message if the assigned window is stratified.
- `record_look(family, artifact, verdict, probability_positive, notes)` →
  marks the window spent and rewrites the ledger. `leg_effects=[...]` is
  required when the window is stratified and rejected when it is not.
- **Era-stratified additions**: `eligible_stratified_seasons(family)` (every
  individual season a close-graded family may still draw as a leg);
  `assign_stratified_window(family)` (assigns the deterministic leg pair —
  earliest untouched season paired with the untouched season maximally
  distant from it, which reduces to `(min(eligible), max(eligible))`; see the
  proposal doc for the derivation); `confirmation_split_legs(features,
  family)` → one `LegSplit(season, training, scoring)` per leg, each with its
  own forward-chained training cutoff.
- CLI: `nfl-ats rotation declare|assign|status|record|validate|declare-coverage`.
  `status` prints every family, its windows, remaining unspent capacity per
  grade pool, and the season-usage table. `assign --stratified` draws a
  two-leg window instead of a contiguous block (close-graded only; not valid
  with `--size`). `record --leg-effects '<json list>'` supplies the per-leg
  magnitudes a stratified look requires. `validate` and `declare-coverage`
  are ENG-27 additions, described below.

## Validator (ENG-27)

Added 2026-09-04 (ROADMAP.md Phase 13, ENG-27) after the tracked ledger was
found to hold a window wider than `assign_window` itself would ever draw
(`pbp_drive_bundle`'s `[2013, 2017]`, five seasons, predating the
`MAX_WINDOW_SIZE=4` limit) with nothing catching it: `_validate` (the hard
load/save gate) never checked window WIDTH, only pool bounds, the
mined-2018-2025 acknowledgment, and no-overlap — because those three were
already reachable through `assign_window`/`assign_stratified_window`'s own
call paths, while width limits were enforced only at the moment
`assign_window` draws a FRESH block, not against a window written any other
way.

`rotation.validate_registry(registry) -> list[Issue]` is a full-audit pass,
separate from `_validate`: it never raises, and returns EVERY issue in one
pass rather than stopping at the first one. Four checks:

1. `window_width_out_of_range` (error) — a contiguous window's span outside
   `[MIN_WINDOW_SIZE, MAX_WINDOW_SIZE]` (2-4 seasons). Stratified windows are
   exempt by design: a leg pair's two seasons are always exactly
   `STRATIFIED_LEG_COUNT` (2) distinct seasons (enforced at load), and the
   GAP between two legs is deliberately unlimited — era-stratified windows
   exist specifically to pair widely-separated eras
   (`docs/era_stratified_windows_proposal.md`), so a wide gap is the intended
   design, not a violation of a contiguous-window rule that was never written
   for leg pairs. **`fluview_elevated_on_production`'s `[2011, 2025]`** —
   this item's own motivating example — is `window_kind: "stratified"`
   (read directly from the tracked JSON), i.e. the two single-season legs
   2011 and 2025, not a 15-season contiguous span; it does NOT trigger this
   check, correctly. The one real, measured violation is
   `pbp_drive_bundle`'s CONTIGUOUS `[2013, 2017]` (5 seasons). Neither window
   is modified by the validator or by this change — changing an
   already-recorded window is a research decision for the project owner.
2. `overlapping_windows_within_family` (error) — pairwise overlap within one
   family's own windows. `_validate` already hard-refuses this at load time,
   so it can never actually fire on a registry that loaded at all; kept so
   `validate_registry` is a complete, standalone audit.
3. `missing_mined_acknowledgment` (error) — same defense-in-depth
   relationship to `_validate` as (2).
4. `status_look_with_no_window` (warning) — a family's `status` is
   `confirmed` or `closed_negative` (only ever set by `record_look` spending
   a window) but it holds no window in the `spent` state — a data-integrity
   flag `_validate` never checked.

CLI: `nfl-ats rotation validate [--json]`. Read-only; exits non-zero (`1`)
if any issue is `error`-severity, `0` otherwise. Wired into `save_registry`
too, but there it only WARNS on stderr and never refuses the write — existing
tracked data (the `pbp_drive_bundle` width violation above) predates several
of these checks and must keep loading and saving; `rotation validate` is the
tool for failing a session deliberately on one of them.

## Coverage (ENG-27)

Measured 2026-09-04: the rotation registry declared ~29 families against
350+ distinct weak-signal families in `registry/weak_signals.json`, so
`registry_explorer.next_shots` reported `unspent_rotation_window: None` for
nearly every top row — not because nothing was available, but because no
rotation family had ever been DECLARED for most weak-signal families at all.

`nfl-ats rotation declare-coverage [--dry-run|--apply]` closes that gap,
additively. For every weak-signal `(league, family)` that
`registry_explorer.matching_rotation_families` finds no rotation-family
match for, and that isn't already excused by an existing
`no_rotation_needed` record, it plans exactly one of:

- **A coverage stub** (`rotation.declare_coverage_stub`) — a rotation family
  named after the weak-signal family, `status: "declared_for_coverage"`,
  `grade: "close"` (a placeholder, not a commitment — the true grade is a
  research decision for whoever first assigns a real window), zero windows,
  and the weak-signal family/league/effect-units recorded as metadata
  (`coverage_weak_signal_family`, `coverage_league`,
  `coverage_effect_units` on the `Family`). It reserves the NAME so
  `next_shots`/`matching_rotation_families` finds it by equality, and a
  future session can run `rotation assign --name <name>` directly instead of
  first declaring it. It makes NO research commitment and spends NO window.
- **An explicit `no_rotation_needed` record** (`rotation.
  record_no_rotation_needed`, a new top-level `no_rotation_needed` section
  in the ledger, keyed by weak-signal family name) — for a family that is
  not itself a candidate research hypothesis: a split-half reliability
  measurement, a positive-control/oracle instrument, or a retired profile.
  `reason` is one of `rotation.NO_ROTATION_FIXED_REASONS`
  (`reliability_measurement`, `positive_control`, `oracle`,
  `retired_profile`) or a `decomposition_of_parent:<family>` tag — **never
  free text, and never guessed**: `rotation.classify_no_rotation_reason`
  determines it deterministically from the entry's `category` and name only
  (name contains "oracle" -> `oracle`; name contains "reliability" ->
  `reliability_measurement`; name contains "retired" -> `retired_profile`;
  `category == "control"` and none of those -> `positive_control`); anything
  it cannot classify gets a stub instead, never a guessed reason.

Both write paths are append-only, like `declare_family`: a name/family
already declared, or already excused, is refused rather than silently
overwritten. Neither assigns a window nor records a look — this command
grows the registry's NAME coverage only; a family still needs `rotation
assign` before it holds anything spendable.

**Additive to the tracked JSON, verified two ways.** The per-family
`coverage_*` fields and the top-level `no_rotation_needed` section are
OMITTED entirely (not written as `null`) for every family/registry that does
not carry them — writing them unconditionally would insert new sibling keys
into every pre-existing family's JSON object and force `json.dumps(...,
sort_keys=True)` to rewrite the comma on whatever used to be that family's
last field, which is a reformat, not a value change, but shows up as noise
in `git diff` and defeats the additions-only contract. Because the ledger is
serialized with `sort_keys=True`, inserting ~365 new alphabetically-sorted
family blocks among the 30 existing ones still moves *where* several
pre-existing families' lines sit in the file (git's default line-based diff
has no move-detection, so a naive `git diff | grep '^-'` shows hundreds of
lines that LOOK like removals) — this is a presentation artifact, not a
mutation: `tests/test_rotation_coverage.py` proves every pre-existing family's
serialized JSON object is byte-for-byte identical before and after
`declare-coverage --apply`, and that every "removed" line's exact text
reappears among the "added" lines. Read `git diff --stat` for total line
churn, but trust the structural/byte comparison, not the raw `-` count, for
whether anything was actually mutated.

## What this deliberately does not do

- It does not retroactively bless past 2018-2025 numbers; the standing
  discount stays.
- It does not schedule looks — humans decide when a family is ready; the
  registry only makes the spend explicit and irreversible.
- It does not replace prospective 2026 scoring, which remains the cleanest
  evidence and needs no window at all.
