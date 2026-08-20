# Smooth CDF mapping — MOD-08's Gaussian probability read, wired as a challenger

Written 2026-08-19. This item continues the MOD-08 lead already predeclared
and screened in `docs/ecdf_smoothing.md` (method selection: `gaussian` over
`gaussian_kde`/`skew_normal`, on the CFB benchmark plus an informational NFL
walk-forward) and already implemented in `src/nfl_ats/calibration.py`
(`fit_residual_smoother` / `smoothed_home_cover_probability`, an opt-in
reader that never touches `margin.py`). This document does three things that
had not yet been done: (1) a fresh, clearly-labeled historical measurement on
the CURRENT production recipe (`weak_stack`, not the `player` profile the
original screen used) restricted to seasons no rotation-registry family has
reserved, so it spends no window; (2) recording that measurement to
`registry/weak_signals.json`; (3) wiring the candidate as a side-ledger-only
2026 prospective challenger, mirroring the existing overlay challengers
exactly.

## Binding closing-grounds taxonomy (verbatim, per AGENTS.md/CLAUDE.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

## What the production mapping actually is (read, `src/nfl_ats/margin.py`)

`fit_margin_model` fits a Ridge (or HGB) mean model on the full training set,
then holds out the trailing 20% chronologically (`distribution_part`) and
computes `residuals = target - temporary_prediction` on that out-of-time
slice — genuinely out-of-sample residuals, not in-sample fit error.
`MarginModel.residuals` holds those draws (a few hundred: 102-883 across the
walk measured below). `MarginModel.predict` builds each game's predictive
sample as `center + residuals` and reads `home_cover_probability` off
`_smoothed_probability`:

```python
def _smoothed_probability(samples, threshold):
    successes = float(np.count_nonzero(samples > threshold))
    return (successes + 0.5) / (len(samples) + 1.0)
```

This is a Laplace/Krichevsky-Trofimov continuity-corrected empirical CDF —
i.e. it discretizes the probability into `n + 1` possible values (n ≈
100-900) and re-derives the distribution's location and shape from a raw
count every week. The pool's forced pick is
`home_cover_probability >= 0.5` (`nfl_ats.pool.build_ats_pool_card`,
`nfl_ats.backtest.py`), which is the empirical **median** of
`center + residuals` against the line — not `sign(predicted_market_residual)`
— so (per the MOD-06 retraction, `docs/pool_edge_plan.md`) a shift in how
that median/threshold-crossing is *read* from the same draws can flip a pick
even though a pure rescale of the point prediction cannot. MOD-05
(`docs/margin_variance.md`) already measured the ATS residual conditional on
the line as near-Gaussian (dip test p = 1.000, roughness chi2/df 1.6-1.9
against 21.2 for the raw margin), which is why a Gaussian CDF is an
admissible, theoretically motivated replacement for the raw count rather
than an arbitrary alternative.

## The mapping formula (parameter-free, derived)

`nfl_ats.calibration.smoothed_home_cover_probability(residuals, centers,
lines, method="gaussian")`, already implemented and unit-pinned
(`tests/test_calibration_ecdf_smoothing.py`):

```
mean = mean(residuals)
std  = sample_std(residuals, ddof=1)
threshold = line - center
home_cover_probability = 1 - Phi((threshold - mean) / std)   # scipy.stats.norm.sf
```

Every quantity is derived from the SAME trailing out-of-time residual sample
the production ECDF reads — no threshold, no bandwidth, no hand-picked
constant. This is the simplest defensible smooth CDF: a Gaussian fit by
method of moments, chosen (in `docs/ecdf_smoothing.md`'s screen) over a
kernel density or a skew-normal specifically because the residual is already
measured near-Gaussian (MOD-05) and a flexible nonparametric fit is more
likely to chase noise in a 100-900-draw sample than to improve on a
distributional match that is already this good. `gaussian_kde` and
`skew_normal` remain available in `calibration.py` as rejected-on-evidence
alternatives, not deleted.

## Implementation: opt-in, default path bit-identical

Nothing in `margin.py`, `pool.py`, or `backtest.py` changes.
`src/nfl_ats/calibration.py`'s residual-smoothing section (already present,
not added by this document) is a pure reader of `MarginModel.residuals`; its
`method="ecdf"` control arm reproduces `margin._smoothed_probability` to
floating-point precision, pinned by
`tests/test_calibration_ecdf_smoothing.py::test_ecdf_control_arm_reproduces_production_probabilities`.
`calibration.py` is not imported by `margin.py`, `pool.py`, `backtest.py`, or
`outcomes.py`, so the frozen active model is bit-identical whether or not it
is ever imported.

The new piece in this item is the challenger wiring,
`src/nfl_ats/smooth_cdf_mapping_overlay.py`
(`tests/test_smooth_cdf_mapping_overlay.py`, 14 tests). Unlike the flag-based
research module, a prospective challenger has to reproduce a *specific* past
week's fitted model without re-running `margin-predict`. It does so through
`nfl_ats.outcomes.fit_margin_models_for_week` — a public entry point that
exists precisely to expose the fitted `MarginModel` for one week rather than
a pre-summarized card — using the exact recipe (feature profile, regressor,
ridge alpha, `min_train_games`) recorded in the active model's own forecast
metadata, at the same leak-safe training cutoff `score_outcome_week` used.
Ridge and the Gaussian fit are both deterministic
(`random_state=42` throughout `margin.py`), so the refit reproduces the same
center and residual sample the card was built from — proven, not assumed,
by re-deriving the ECDF probability from the refit and requiring it to match
the card's own `home_cover_probability` to `atol=1e-9` before any Gaussian
probability is trusted (`apply_smooth_cdf_mapping_overlay`; a mismatch — e.g.
because the feature table was rebuilt between card generation and this
overlay running — raises `DataContractError` rather than silently comparing
against a moved target).

## Historical measurement (this document's own, `measured`)

Protocol: production recipe exactly (`target=market_residual`,
`regressor=ridge`, `ridge_alpha=10.0`, `feature_profile=weak_stack`,
`min_train_games=500` — `artifacts/active_ats_model.json` as of
2026-08-19), on `data/processed/game_features_weak_stack.parquet` (the
CLOSE grade: nflverse's own `spread_line`, not an archived Tuesday-opener
snapshot — matching the same grade the earlier MOD-08 figure and
`docs/ecdf_smoothing.md`'s screen both used). Every non-reserved NFL test
week 2009-2025 is scored twice from the identical fitted model and residual
sample — the `ecdf` control and the `gaussian` candidate — via
`scripts/smooth_cdf_mapping_measurement.py`. Non-reserved seasons are
computed from `nfl_ats.rotation.season_usage` (every season ANY family has
spent a window over, registry-wide): **{2013, 2014, 2015, 2016, 2017, 2020,
2021} are reserved and excluded**; the evaluation seasons are **{2009, 2010,
2011, 2012, 2018, 2019, 2022, 2023, 2024, 2025}** — 10 non-contiguous
seasons. Reserved-season weeks are still walked over so their games enter
the training pool for later cutoffs (exactly as in real production), but
they are never fit-and-scored as evaluation rows, so no registry-governed
season contributes evidence to this comparison and no window is spent or
implied (rule 8, `docs/rotation_registry.md`: "CFB and non-reserved seasons
stay free"). Bootstrap: seed 20260819, 20,000 samples, week-blocked primary
(within-week correlation is zero by mandate), season-blocked secondary.

Artifact:
`<scratchpad>/smooth_cdf_mapping/{predictions.parquet,paired_comparisons.csv,diagnostics.json}`
(measured this session; not committed — regenerate with
`scripts/smooth_cdf_mapping_measurement.py --output <dir>`).

**Week-blocked, paired (`gaussian` vs `ecdf`, 2,047 scored games, 140 week
blocks):**

| metric | estimate | 95% interval | probability_positive |
|---|---|---|---|
| accuracy (points) | **+0.684** | [-0.444, +1.841] | **0.8666** |
| Brier (raw, positive = better) | **+0.001861** | [+0.000871, +0.002868] | **0.9999** |
| log loss (raw, positive = better) | **+0.004185** | [+0.002068, +0.006358] | **1.0000** |

Season-blocked (8 blocks, flagged degenerate by
`experiments.paired_feature_comparisons`'s own guard — the point estimate
and `probability_positive` are still meaningful, the interval is not):
accuracy +0.684 points, `probability_positive` 0.9524; Brier +0.001861,
`probability_positive` 1.0000; log loss +0.004185, `probability_positive`
1.0000.

**Pick movement:** 159 of 2,111 predicted games flip sides (7.53%), close to
the 7.4-7.9% measured in `docs/ecdf_smoothing.md`'s CFB/NFL screen on the
`player` profile — the mechanism (denoising a noisy median near the 0.5
threshold) reproduces across both feature profiles. Flip rate by
`key_numbers.line_bucket`: `under_3` 9.43%, `three_five_to_six_five` 8.14%,
`three` 6.54%, `over_seven` 6.00%, `seven` 4.80% — again the same pattern as
the earlier screen (flips concentrate near pick'em lines, least often at the
`|line| = 3` key number).

**Read against the existing `ecdf_smoothing_accuracy` registry entry.** That
entry (CFB, `player`-style screen, `docs/ecdf_smoothing.md`) recorded
accuracy leaning NEGATIVE (`probability_positive` 0.11). This measurement —
NFL, `weak_stack` production profile, non-reserved seasons only — leans
POSITIVE (`probability_positive` 0.8666) on the same primary metric. Both
numbers are `measured`, from different leagues, feature profiles, and
seasons; this is not a re-run of the same look, and the disagreement is not
resolved by either alone. Per AGENTS.md, neither an interval crossing zero
nor a directional disagreement between two independently-measured contexts
is grounds to reject this candidate — both are recorded, separately, exactly
as measured. What is NOT in dispute across every version of this measurement
(this document, the earlier CFB screen, the earlier NFL informational run)
is the calibration gain: Brier and log loss improve with `probability_positive`
at or extremely close to 1.0 every time this has been measured, on both
leagues and both feature profiles.

## Frozen decision rule

Per this project's own bar (forced-pick accuracy vs. the coin flip, not a
calibration metric — AGENTS.md, "Edge means beating 50%"), accuracy is the
primary metric and Brier/log-loss are reported as secondary coherence
checks that do not override it in either direction. The interval crosses
zero, so per the binding taxonomy above this is **not** a closable result —
it is recorded `unresolved_below_power` with `probability_positive` 0.8666,
and it earns continued prospective evidence rather than a rotation-registry
confirmation window (none is available cheaply — see `docs/ecdf_smoothing.md`'s
own discussion of the two scarce remaining opener windows). No method,
feature, or split retuning follows this measurement; a different method
(`gaussian_kde`, `skew_normal`) would be a new predeclaration.

## Promotion / EV framing

The pool is forced picks: every one of the 285 cards must be played either
way, so the decision bar is expected value, not a promotion threshold
(AGENTS.md, "A promotion bar is not a decision bar"). At `probability_positive`
0.8666 for accuracy — favouring the Gaussian candidate roughly 6.5:1 — EV
already favours the candidate over the production ECDF on a per-pick basis,
even though the interval has not resolved. That is a real, actionable lean,
not a reason to wait for more data before it is even recorded. It is
**not**, by itself, a case for silently swapping the production mapping:
this document's constraints call for recording and side-ledger prospective
tracking only; the default probability path stays bit-identical, and
promotion is the owner's/orchestrator's decision, made with the accumulating
2026 prospective evidence in hand (see "Wiring" below) rather than on this
measurement alone.

## Wiring: `smooth_cdf_mapping` prospective challenger

Registered in `artifacts/prospective/challengers.json` as
`smooth_cdf_mapping`, `ACTIVE_PROSPECTIVE`, following the
`hc_year_one_fade_overlay` / `injury_value_lost_tilt_overlay` /
`spread_gap_zone_fade_overlay` pattern exactly: this is **not** a retrained
model with its own `margin-predict` artifact. Its weekly picks ARE the
active model's own picks, transformed post-prediction — here, by replacing
`home_cover_probability` with the Gaussian read of the same residual sample,
via `nfl_ats.smooth_cdf_mapping_overlay.apply_smooth_cdf_mapping_overlay` —
so `record_smooth_cdf_mapping_challenger_decisions` reads the active model's
own synchronized weekly forecast and refuses if the active model's live
fingerprint no longer matches the snapshot this challenger pinned at
registration (a promotion under the mapping's feet must not silently
convert into "prospective evidence" for a different base model). It is
wired into `_cmd_publish_predictions` (`src/nfl_ats/cli.py`) as an additive,
fail-open step exactly like every other overlay challenger: a failure there
is reported in the publish result but never un-publishes the card, and
nothing is recorded without the explicit `--record-decisions` flag
(`nfl_ats.clv.refuse_if_outside_recording_lock_window` is the second,
function-level guard against a rehearsal run reaching the real ledger).

`bet_side` is always `"PASS"` and `edge` is always NaN: this challenger
tracks the mapping's forced-pick (`decision_line`) accuracy only, never a
fabricated paper-bet edge for the post-mapping side. No rotation-registry
window is spent or implied by this registration
(`docs/rotation_registry.md`, "what this deliberately does not do" —
prospective scoring needs no registry window at all).

## Opener-grade decision measurement (2026-08-19, second look — multiplicity disclosed)

Per AGENTS.md ("A promotion bar is not a decision bar", "Grade the decision at
the OPENER") and the MOD-07 precedent (commit `68b4dc0`): a close-graded
comparison may inform but must never veto a forced-pick promotion decision,
because the pool's forced picks are graded at (or near) the Tuesday opener,
not the market's sharpest closing number, which systematically understates
pool-relevant edge. This section is a SECOND, independently-labeled look at
the same MOD-08 `gaussian`-vs-`ecdf` candidate already measured close-graded
above (accuracy `probability_positive` 0.8666, recorded
`mod08_smooth_cdf_mapping`, `unresolved_below_power`) — disclosed as a second
look, same convention as `docs/opener_evaluation.md`'s addendum.

### Protocol (frozen before running)

Reuses `docs/opener_evaluation.md`'s exact 1,537-paired-game archive and
weekly-refit protocol (`nfl_ats.clv.build_pairing_table`/
`close_reference_table`, `tue_open` + close, 2020-2025 regular season,
production recipe `weak_stack`/`ridge`/alpha 10.0/`market_residual`,
`min_train_games=500`). Every paired week's SAME fitted model and out-of-time
residual sample is read twice: once through the production `ecdf` control
(`margin._smoothed_probability`, reproduced via
`nfl_ats.calibration.smoothed_home_cover_probability(method="ecdf")`) and
once through the `gaussian` candidate (`method="gaussian"`), at BOTH the
opener and close lines. `scripts/smooth_cdf_mapping_opener_measurement.py`
runs this.

Per `docs/opener_evaluation.md`'s own admissibility argument (reused
verbatim in spirit): this reads inside the ledgered 2018-2025 era, but
nothing is being selected or tuned here — the two mappings (`ecdf`,
`gaussian`) were both already fully specified by the close-graded
measurement above, so this is a re-measurement of an already-fixed pair of
pick-stream generators at a different settlement line, not a new selection
dimension; no candidate gains or loses standing from this run. No
rotation-registry window is spent or implied (rule 8). The opener archive's
fixed 2020-2025 span cannot itself be restricted to non-reserved seasons the
way the close-grade script restricts to `{2009, 2010, 2011, 2012, 2018,
2019, 2022, 2023, 2024, 2025}` — it is "nothing selected," not "no reserved
season touched," that makes this admissible, exactly as it was for the
original `opener-evaluation` predeclaration.

Primary metric: paired forced-pick accuracy under **production's actual
pick rule** (`home_cover_probability >= 0.5`), settled at the opener,
week-blocked bootstrap, seed 20260819, 20,000 samples. Secondary/reported:
the same comparison at the close (context against the close-grade number
above), and the **sign rule** (`residual > 0`) at both lines — reported to
disclose that the sign rule is mapping-invariant BY CONSTRUCTION (it never
calls the probability-mapping function at all: `predicted_market_residual`
depends only on the fitted centre, not on `ecdf` vs `gaussian`), so it
necessarily shows zero flips between the `ecdf` and `gaussian` arms at a
fixed line. That is not a null result; it is a structural fact about which
lever the mapping pulls (`docs/ecdf_smoothing.md`'s "why smoothing can move
picks even though rescaling cannot").

### Frozen decision rule

- If the opener-grade, production-pick-rule paired accuracy delta's
  `probability_positive` > 0.5: **PROMOTE** the Gaussian mapping to the
  production default. Per AGENTS.md ("A promotion bar is not a decision
  bar"): the pool is forced picks, so declining a candidate that is
  more-likely-than-not better is taking the other side of that bet; this is
  an EV decision, not a threshold-clearing decision, and it fires however
  far above 0.5 the number lands (owner standing order: EV-positive plays
  execute; record both arms).
- If `probability_positive` <= 0.5: do not promote; the `smooth_cdf_mapping`
  challenger keeps accruing 2026 prospective evidence.
- Either way, record this opener measurement to the registry as
  `mod08_smooth_cdf_mapping_opener`, classification `unresolved_below_power`
  unless a genuinely admissible terminal ground (refuted mechanism or
  positive-control bound) fires, and report `probability_positive`, never
  "contains zero" (binding taxonomy, restated below). Multiplicity is
  disclosed: this is the second recorded look at this candidate's accuracy,
  after the close-graded `mod08_smooth_cdf_mapping` entry above.
- One run; no method, feature, or split retuning after seeing the result.

### Binding closing-grounds taxonomy (verbatim, per AGENTS.md/CLAUDE.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

### Result (measured, appended after running)

Run 2026-08-19, `scripts/smooth_cdf_mapping_opener_measurement.py --output
<scratchpad>/smooth_cdf_mapping_opener`, week-blocked, seed 20260819, 20,000
samples. `sign_rule_diagnostic` confirms the harness reproduces
`docs/opener_evaluation.md`'s tracked numbers exactly (opener accuracy
52.8277%, close accuracy 51.5594%) and that the sign rule shows **0 flips**
between the `ecdf` and `gaussian` arms at either line (1,537/1,537 games) —
the predicted structural invariance holds.

**Primary (production pick rule, settled at the opener, week-blocked, 107
blocks, not degenerate):**

| metric | estimate | 95% interval | probability_positive | paired games |
|---|---|---|---|---|
| accuracy (points) | **+0.1331** | [-1.397, +1.715] | **0.5536** | 1,503 |
| Brier improvement | +0.000723 | [-0.000208, +0.001667] | 0.9368 | 1,503 |
| log-loss improvement | +0.001581 | [-0.000321, +0.003521] | 0.9488 | 1,503 |

**Secondary (production pick rule, settled at the CLOSE, same 1,537-game
archive — context only, not the decision metric):**

| metric | estimate | 95% interval | probability_positive | paired games |
|---|---|---|---|---|
| accuracy (points) | +0.7963 | [-0.649, +2.262] | 0.8541 | 1,507 |

Season-blocked (6 blocks) is flagged degenerate by
`paired_feature_comparisons`'s own guard; only the point estimate/P+ are
meaningful: open accuracy P+ 0.5632, close accuracy P+ 0.8658 (both agree in
direction and magnitude with the week-blocked reads above).

**Reading, measured, plainly stated.** The opener-grade probability_positive
(0.5536) is real but markedly weaker than every close-graded reading of this
same candidate: 0.8666 on the full 10-season non-reserved-season archive
(`mod08_smooth_cdf_mapping`) and 0.8541 on this identical 1,537-game archive
graded at the close instead of the opener. The candidate's edge over the raw
ECDF concentrates more at the close than the opener on this archive — the
opposite pattern from MOD-07's own opener-vs-close story. That is disclosed,
not smoothed over. Brier/log-loss stay strongly positive at both grades
(P+ 0.94-0.95 at the opener), consistent with every other measurement of
this candidate.

**Decision.** `probability_positive` 0.5536 > 0.5, so the frozen rule above
fires: **PROMOTE**. Per AGENTS.md ("a promotion bar is not a decision bar";
"grade the decision at the opener; a close-graded number may never veto a
play"): the pool plays forced picks, and this is an EV rule, not a
threshold-clearing rule — it was written to fire "however far above 0.5 the
number lands," and this result, while close to a coin flip, lands on the
promote side of it. Recorded to the registry as
`mod08_smooth_cdf_mapping_opener`, `unresolved_below_power` (interval
crosses zero; not a terminal ground), `probability_positive` 0.5536,
multiplicity disclosed as the second look at this candidate. See "Promotion
executed" below for what changed in the codebase as a result.

## Promotion executed (2026-08-19)

The frozen rule above fired PROMOTE. What changed, concretely:

1. **Production default.** `nfl_ats.outcomes.score_outcome_week` -- the sole
   caller of which is the `margin-predict` CLI command, the single
   production weekly-forecast entry point -- now defaults
   `probability_method="gaussian"`. Every other probability-reading call
   site keeps the pre-promotion `"ecdf"` default: `MarginModel.predict`
   itself (dozens of research/backtest call sites), `walk_forward_outcomes`
   (backs `margin-backtest`, player ablations, experiment comparisons), and
   the `margin-backtest` CLI's own `--probability-method` flag. Only
   `home_cover_probability` is affected; `home_win_probability` and the
   push/three-way split stay on the raw ECDF unconditionally, unchanged from
   this document's declared scope.
2. **Cannot silently revert.** `nfl_ats.active_model`'s SYNCHRONIZED-matching
   identity (`_matching_evaluation`/`model_identity`) now carries
   `probability_method`, mirroring the existing `calibration_method` field
   exactly, defaulting to `"ecdf"` for legacy metadata that predates this
   field. A forecast built with one probability method can now only
   synchronize against an evaluation recorded with the same one -- the
   specific guard against a `--probability-method ecdf` run (e.g. a naively
   -built incumbent-tracking challenger) silently re-activating the
   pre-promotion evaluation and reverting the promotion. Because
   `margin-backtest`'s own default stayed `"ecdf"` (by design, for every
   other caller), `weekly.py`'s steps 4 (`margin-backtest`) and 5
   (`margin-predict`) now both pass `--probability-method gaussian`
   explicitly rather than relying on two independent CLI defaults staying in
   sync -- without this, real weekly-run would never synchronize
   (`assert-synchronized` would abort every week). Caught by running the
   full `tests/test_weekly.py` suite, not assumed.
3. **Regression coverage.** `tests/test_probability_method_promotion.py` (9
   tests) pins: `MarginModel.predict`'s default is unchanged and
   bit-for-bit; an explicit `"gaussian"` request changes only
   `home_cover_probability`; `score_outcome_week`'s own default is
   `"gaussian"` while `walk_forward_outcomes`' stays `"ecdf"`; the
   `margin-predict`/`margin-backtest` CLI defaults; and that
   `activate_matching_ats_model` only synchronizes a forecast against an
   evaluation recorded with the SAME `probability_method`. One existing test
   (`tests/test_outcomes.py::test_fit_margin_models_for_week_matches_score_outcome_week`)
   was updated to pass `probability_method="ecdf"` explicitly, since its
   point is refit equivalence at a fixed method, not a claim about which
   method is the production default. `tests/test_cli.py::test_cli_model_workflow`
   and the two `publish-predictions` recording tests were updated for the
   new default and the renamed challenger (below). No existing test's
   *expected value* changed, only which method it now must name explicitly
   to keep testing what it always tested.
4. **Challenger flip.** The published card's `home_cover_probability` IS now
   the Gaussian arm, so the `smooth_cdf_mapping` prospective challenger
   (which verified a card reproduces the ECDF control before mapping it to
   Gaussian) would fail by construction on every future week -- its
   `artifacts/prospective/challengers.json` entry is retired
   (`status: "SUPERSEDED_BY_PROMOTION"`, kept in place as an audit trail, not
   deleted). A new challenger, `ecdf_mapping_incumbent`
   (`src/nfl_ats/ecdf_mapping_incumbent_overlay.py`, 14 tests in
   `tests/test_ecdf_mapping_incumbent_overlay.py`), tracks the FORMER
   production ECDF read: it verifies a refit's **Gaussian** reproduction
   matches the (now Gaussian-native) card before mapping to **ECDF** --
   the exact mirror image of the retired challenger's mechanism. Wired into
   `_cmd_publish_predictions` (`src/nfl_ats/cli.py`) exactly where the old
   challenger was, same fail-open guarantee (a failure here is reported but
   never un-publishes the card), same `--record-decisions` gating. Both
   arms are now recorded prospectively: the ordinary paper ledger settles
   the (now Gaussian) published pick, and `ecdf_mapping_incumbent` settles
   the former ECDF pick, dual-tracked -- per AGENTS.md's "EV-positive plays
   execute; record both arms."
5. **Manifest schema.** `artifacts/active_ats_model.json` (generated, not
   committed) gains a `probability_method` field alongside the existing
   `calibration_method`, following that field's exact precedent (item 2
   above) -- not a new ad hoc key, the same slot MOD-07's promotion would
   have used had it needed one.
6. **Week 1 published card.** Checked read-only, using the already-tested
   `apply_smooth_cdf_mapping_overlay` against the live
   `artifacts/margin_predictions/2026-week-01-20260818T013139Z/recommendations.csv`
   (16 games) -- nothing was regenerated or republished. Exactly **one**
   pick would flip: `NE at SEA`, from `NE +3.5` (49.89% ECDF, the currently
   published pick) to `SEA -3.5` (51.69% Gaussian). The other 15 picks, and
   the Best Pick nomination (`MIA +3.5` in `MIA at LV` -- a separate
   alpha=2000 ranking model, out of this promotion's scope), are unchanged.
   `CURRENT_PREDICTIONS.md` and the live `artifacts/active_ats_model.json`
   were not touched by this session; republishing (running `margin-backtest`
   / `margin-predict` / `publish-predictions` for real against the live
   artifacts root) is the orchestrator's call, not made here.
7. **Not touched.** `margin.py`'s `line_sweep` (line-shopping diagnostic)
   and the Best Pick nomination's own alpha=2000 refit
   (`best_pick_nomination.fit_candidate_probabilities`) stay ECDF-only,
   matching this document's declared scope (item 2, "Only
   `home_cover_probability`... is mapped"); a future item wanting either
   would need its own predeclaration.

## Declared limitations

1. This document's own measurement (weak_stack, non-reserved seasons) is a
   fresh, independently-labeled look, not a re-run or extension of
   `docs/ecdf_smoothing.md`'s CFB/`player`-profile screen. The two disagree
   on the sign of the accuracy lean; both are recorded as measured, and
   neither overrides the other.
2. Only `home_cover_probability` (the two-way forced-pick threshold) is
   mapped. Push/three-way probabilities keep
   `margin._three_way_probabilities`'s existing integer-rounding treatment,
   unchanged — out of scope here, as in `docs/ecdf_smoothing.md`.
3. `gaussian_kde`/`skew_normal` are not wired as challengers. If `gaussian`
   is ever promoted or closed, either alternative would need its own fresh
   predeclaration, not a retry inside this one.
4. The season-blocked interval above is flagged degenerate (8 blocks); only
   its point estimate and `probability_positive` are reported as meaningful,
   per `experiments.paired_feature_comparisons`'s own guard.
5. This item does not change default production behavior. The default
   mapping (`margin._smoothed_probability`, the raw ECDF) stays bit-identical
   for every existing caller; only the new opt-in `calibration.py` reader and
   the side-ledger `smooth_cdf_mapping` challenger exist on top of it.
   **Superseded 2026-08-19 (see "Promotion executed" below): this limitation
   described the state before the opener-grade decision measurement fired
   the frozen promotion rule.** `margin._smoothed_probability` itself is
   still called unconditionally for `home_win_probability` and the
   push/three-way split, and every historical/research caller of
   `MarginModel.predict` still gets the ECDF by default -- but
   `nfl_ats.outcomes.score_outcome_week`, the sole production
   weekly-forecast entry point, now defaults to the Gaussian read, so
   "the default mapping stays bit-identical for every existing caller" is no
   longer true of THAT one caller. Kept here, corrected rather than deleted,
   per this project's convention for stale claims.
