# MOD-14 promotion confirmation: `half_life_8` era weighting at the opener

Predeclared 2026-08-21, **before** `nfl-ats rotation declare`, **before**
`rotation assign`, and **before** `scripts/era_weighting_promotion_look.py`
scores anything. Provenance tags: **measured** (run this session, command/path
given), **read** (file opened this session), **reported** (another doc's
claim, unverified here), **inferred** (reasoning, not evidence).

## Binding closing-grounds taxonomy (pasted verbatim, per AGENTS.md)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry code hard-rejects inadmissible
> closures; if a record command errors, the verdict is wrong, not the
> validator.

## 1. Family definition

- **Family name** (`rotation declare`): `era_weighting_half_life_8`.
- **Hypothesis**: training the frozen production recipe
  (`weak_stack` / `ridge` / `ridge_alpha=10.0` / `market_residual`,
  `min_train_games=500`) with exponential season-decay sample weights at an
  8-season half-life improves forced-pick accuracy against the same recipe
  fit with uniform weights on all history. This is a TRAINING-RECIPE family,
  not a feature family -- the feature contract is frozen and identical across
  both arms.
- **Grade**: `opener` (AGENTS.md: grade the decision at the opener).
- **Arms** (`era_weighting_lib.ERA_WEIGHTING_ARMS`, reused verbatim,
  **read** this session): `baseline` (uniform `sample_weight=1`) vs.
  `half_life_8` (weight `0.5 ** ((predict_season - row_season) / 8)`, season
  granularity, clamped at 0). Nothing else varies between arms.
- **Inherits**: none (no registered family is a predecessor of this training
  recipe; MOD-06's shrinkage closure and MOD-12's alpha freeze are different
  levers, per `docs/era_weighting_screen.md` Section 1, **read**).
- **`acknowledges_mined_2018_2025`: true** -- every opener-grade window sits
  inside 2020-2025, so the mined-ledger discount is unavoidable and is
  acknowledged at declaration.

## 2. What was already spent vs. what this look spends (**read**, all this session)

- `docs/era_weighting_screen.md` Sections 3-7: CFB screen (free, rule 8) and
  NFL close-grade screen on non-reserved seasons (rule 8) -- no rotation
  window spent by either.
- `docs/era_weighting_screen.md` Section 8 /
  `scripts/era_weighting_opener_read.py`: an OPENER-GRADE INFORMATION READ on
  the full frozen 1,537-paired-game 2020-2025 archive -- explicitly "not a
  promotion", explicitly "Rotation registry: untouched (rule 8)" (script
  docstring and diagnostics `rotation_registry_touched: false`). It assigned
  and spent NO window.
- `registry/rotation_registry.json` contains **no** `era_weighting*` family
  (**measured** this session: grep for `era_weighting` in
  `registry/rotation_registry.json` returns nothing; `nfl-ats rotation status`
  lists six families, none of them era weighting).
  `registry/weak_signals.json` holds thirteen `era_weighting*` entries -- the
  twelve screen arms plus the Section 8 information read
  `era_weighting_nfl_half_life_8_opener` -- and **no**
  `era_weighting_half_life_8_opener_confirmation` entry (**measured**: grep
  for `"name": "era_weighting` in `registry/weak_signals.json`, this session).

Therefore no opener-grade confirmation look has been SPENT for this family,
and this document spends exactly one: the deterministic earliest eligible
opener block for a fresh family, which per `rotation.eligible_blocks`
(**read** this session) is **[2020, 2021]** -- per-family retirement (rule 4)
makes other families' spent [2020, 2021] windows (`mod07_weak_signal_stack`,
`best_pick_ranker_opener`) non-blocking; cross-family overlap is legal and
visible in `season_usage`.

## 3. Disclosures, stated before any number exists

(a) **Selection inflation.** `half_life_8` was selected best-of-six on two
    screens (CFB clean-core week-blocked P+ 0.8987; NFL close-grade
    week-blocked P+ 0.8505 / season-blocked P+ 0.9533, both **reported** from
    `docs/era_weighting_screen.md`) and then read a third time at the opener
    (P+ 0.2990, **reported**). This confirmation is therefore the arm's
    FOURTH look, not an independent blind draw; its P+ carries selection
    inflation and is recorded with that caveat attached.
(b) **Population overlap with the Section 8 information read.** The opener
    pool is 2020-2025 and the information read scored ALL of it, so whatever
    window this family draws overlaps games whose outcome-relative picks were
    already seen once. The write-up must state this discount; the two reads'
    P+ figures must never be multiplied as independent.
(c) **Mined-ledger discount.** The window intersects 2018-2025
    (~130-150-look ledger, ROADMAP RWB-16); acknowledged at declaration and
    restated in every write-up.
(d) **Cross-family window reuse.** [2020, 2021] was already spent by
    `mod07_weak_signal_stack` and `best_pick_ranker_opener`. Rule 4 permits
    this (independent hypotheses), but global multiplicity on these seasons
    rises to three families and is reported in the notes.

## 4. Endpoints and decision rule (stated BEFORE running)

- **Primary endpoint**: paired forced-pick accuracy improvement
  (`half_life_8` minus `baseline`) on the assigned window's paired Tuesday-
  opener games, under the production probability rule
  (`home_cover_probability >= 0.5`, model default ECDF read -- the identical
  mapping behind `docs/opener_evaluation.md`'s 53.36% production number),
  week-blocked bootstrap, 20,000 samples, seed 20260819.
- **Secondary endpoints (direction only, no gate-shopping)**: Brier and
  log-loss improvements from the same bootstrap call. Season-blocked is
  reported but degenerate at 2 blocks (estimate/P+ only, no valid interval).
- **Decision rule**: this is a PLAY decision under expected value. Per
  AGENTS.md "a promotion bar is not a decision bar": the predeclared
  promotion-claim threshold (P+ >= 0.90, matching MOD-07's bar) governs ONLY
  what docs may CLAIM ("confirmed" language); it never decides which card is
  played. The play decision weighs this look together with all three prior
  looks, graded at the opener, on expected value. A negative or flat lean is
  recorded, never treated as grounds to decline building the signal.
- **Mechanical classification** (decided by the recorder reading the
  artifact, never asserted in prose first): whole week-blocked primary
  interval below zero -> `refuted_mechanism` /
  `--closing-ground wrong_sign_resolved`; otherwise
  `unresolved_below_power`. No positive control is run anywhere here, so
  `bounded_by_control` is unavailable. A wholly-above-zero interval also
  records `unresolved_below_power` (this taxonomy has no "resolved positive"
  terminal state).

## 5. Protocol

Mirrors `scripts/era_weighting_opener_read.py`'s weekly-refit archive/pairing
machinery exactly (**read** this session), restricted to the assigned window
via `nfl_ats.rotation.confirmation_split` -- the same split discipline
`scripts/mod07_weak_stack.py` used for its opener-graded confirmation look
(**read** this session):

- **Feature table**: `data/processed/game_features_weak_stack.parquet`,
  regular-season rows only.
- **Population**: the `docs/opener_evaluation.md` pairing archive
  (`build_pairing_table` / `close_reference_table`, `tue_open` +
  close-reference labels, `HISTORICAL_CAPTURE_KIND`), restricted to the
  assigned window's seasons; pushes excluded (games without a resolvable
  opener cover outcome do not score).
- **Walk-forward**: per scored week, training = every completed game strictly
  earlier than that week's earliest gameday (forward-chaining only, rule 3);
  both arms fit on IDENTICAL training rows via
  `era_weighting_lib.fit_weighted_ridge_margin`, differing only in the sample
  weight vector.
- **Self-check, run BEFORE `half_life_8` is interpreted**: the `baseline`
  arm's window-season predictions must reproduce the Section 8 artifact's
  baseline predictions
  (`artifacts/era_weighting_opener_read/20260820T002230Z/predictions.parquet`)
  game-for-game -- same machinery, same weekly training rows, so the
  probability-rule pick must agree on every game and the opener cover
  probability must match to floating-point noise. A miss is a bug in this
  script's adaptation, not a finding, and is fixed before the candidate
  arm's numbers are read.
- **Recording**: `rotation record` (verdict per Section 4's decision rule;
  `unresolved` unless a threshold mechanically fires) AND
  `nfl-ats weak-signals record` as
  `era_weighting_half_life_8_opener_confirmation`, league `nfl`,
  `effect_units=accuracy_points`, effect = week-blocked paired opener
  accuracy improvement in points, classification per Section 4. Registry read
  back after each write.

## Results

**Measured**, `scripts/era_weighting_promotion_look.py`, artifact
`artifacts/era_weighting_promotion_look/20260821T174753Z/`, run 2026-08-21.
Family `era_weighting_half_life_8` declared and window **[2020, 2021]**
assigned (deterministic earliest eligible opener block) before the run;
production recipe (`weak_stack`/`ridge`/`ridge_alpha=10.0`/
`market_residual`, `min_train_games=500`), seed 20260819, 20,000 bootstrap
samples, forward-chained weekly refits via
`nfl_ats.rotation.confirmation_split`.

**Self-check: passes exactly, before `half_life_8` was interpreted.** The
`baseline` arm's window-season predictions reproduce the Section 8
information-read artifact game-for-game on all **456** scored games: max
|probability difference| = **0.0**, probability-rule picks agree on every
game (**measured**, printed by the run). Both arms then scored the identical
932 rows (2 arms x 456 games + close-grade rows excluded here; `skip_counts`
all zero).

**Primary: opener grade, production probability rule, week-blocked, paired
accuracy improvement (`half_life_8` vs. `baseline`):**

| Metric | Estimate | 95% CI (week-blocked) | P+ | Paired games |
|---|---:|---:|---:|---:|
| Accuracy | **-0.2193 pts** | **[-3.3898, +2.6786] pts** | **0.4246** | 456 |

Absolute accuracy: baseline **48.03%**, candidate **48.03%** (35 week blocks).
Season-blocked context (degenerate -- only 2 blocks,
`BootstrapDegeneracyWarning`, estimate/P+ only, not a valid interval):
-0.2193 pts, P+ 0.2535.

**Secondary (direction only, no gate):** Brier improvement +0.000595
(P+ 0.7083); log-loss improvement +0.001076 (P+ 0.6841) -- both lean mildly
toward the candidate while the primary accuracy read leans mildly against it,
the same accuracy-vs-calibration divergence this family's other reads showed.

**Mechanical classification** (recorder-read from the artifact, never asserted
in prose first): the week-blocked interval does not sit entirely below zero,
so neither admissible closing ground applies and no positive control was run
-- **`unresolved_below_power`**, reported with `probability_positive` 0.4246.
The predeclared promotion-claim threshold (P+ >= 0.90) is not met, so these
docs claim nothing; per AGENTS.md that threshold governs claims only, never
which card is played.

**Recorded** (**measured**, both registries read back after the write):

- `rotation record`: family `era_weighting_half_life_8`, window [2020, 2021]
  marked **spent** 2026-08-21, verdict **unresolved**, effect -0.2193
  accuracy points, interval [-3.3898, +2.6786], P+ 0.4246, 35 blocks, no
  closing ground; `season_usage` for 2020/2021 rose to 3 each.
- `weak-signals record`: `era_weighting_half_life_8_opener_confirmation`,
  league nfl, `accuracy_points`, classification `unresolved_below_power`,
  `total_signals` incremented to 341, single call, no collision.

### Reading

This is the arm's FOURTH look (disclosures (a)-(d) above all apply and are
pasted into both registry entries). On the decision-grade instrument, on its
own assigned confirmation window, the selected arm's point estimate again
leans negative (-0.2193 pts, P+ 0.4246) -- consistent in direction with the
Section 8 information read (-0.3992 pts, P+ 0.2990) though flatter -- while
both continuous metrics lean mildly positive and nothing resolves in either
direction. Per the binding taxonomy this is category 3, unresolved: a
resolved wrong sign would require the whole interval below zero, which is not
the case, and no positive control bounds it. The line of work stays open; a
future play decision weighs all four looks together at the opener, on
expected value, with the selection-inflation and overlap discounts attached.
