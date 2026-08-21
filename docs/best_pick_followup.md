# Best Pick ranker follow-up (CFB screen, free) — predeclaration and result

Written 2026-08-21. Follow-up to `docs/best_pick_ranker.md` (POL-09) and
`docs/pool_format_levers.md`, executing the routing recommendation MOD-12 made
(`docs/ridge_alpha.md`: "route the Brier gain to the Best Pick ranker") and
MOD-08's promoted smooth-CDF mapping (`docs/ecdf_smoothing.md`). Spends **no
NFL rotation window**: everything here runs on the free CFB XLG-03 benchmark
(rotation rule 8) plus attribution on already-scored artifacts. Script:
`scripts/best_pick_ranker_followup.py`; artifact under
`artifacts/best_pick_followup/<ts>/`.

## Frozen predeclaration (written before any score was computed)

The four candidates below, the status-quo comparator, the population, and the
success rule were fixed and written down before the script's scoring pass ran.
No fifth candidate, no variant, no re-tuning.

### Population

The frozen CFB XLG-03 walk-forward population: seasons 2006–2025,
`market_residual` target, ridge alpha 10, `min_train_games = 500` — exactly the
280 scored weeks / 11,780 resolved picks already stored in
`artifacts/best_pick_tiebreak_cfb/20260818T212916Z/sweep_picks.parquet`
(reused read-only; not recomputed). Primary gate population: **all scored
weeks**. The `clean_core` cut (2012–2019, 2021–2025) is reported alongside as a
descriptive secondary, never gated on.

### Status-quo ranker (the comparator)

`sweep_robustness` descending, ties broken ascending `game_id` — the deployed
NFL Best Pick rule (frozen in `nfl_ats.best_pick`), computed for CFB by the
stage-0 harness (`scripts/best_pick_tiebreak_cfb_screen.py`) whose sweep loop
reproduced the weekly fit at `max |diff| = 0.0`. This is the rule a candidate
would actually replace.

### Predeclared candidates (exactly four)

Each candidate ranks one week's picks by its score descending (ties ascending
`game_id`; a missing score ranks last) and nominates the top-1. Sides are never
re-decided: every game's forced pick stays the alpha=10 model's sign.

1. **`best_pick_followup_smooth_cdf_distance`** — |pick-side cover probability
   − 0.5| under the PROMOTED smooth CDF mapping (analytic Gaussian smoother,
   `feature_set == "gaussian"`), read from the already-stored
   `artifacts/ecdf_smoothing/20260818T000600Z/cfb_predictions.parquet`. This is
   MOD-08's promoted mapping, not the quantised ECDF and not the rejected
   `gaussian_kde`/`skew_normal` alternatives.
2. **`best_pick_followup_alpha2000_distance`** — |pick-side cover probability
   − 0.5| from a fresh walk-forward refit at `ridge_alpha = 2000` (the
   walk-forward Brier optimum per `docs/ridge_alpha.md`), same weekly cutoffs,
   same feature contract, compared against the alpha=10 baseline ordering.
3. **`best_pick_followup_dispersion_gated_smooth_distance`** — composite.
   Week-level residual-sample dispersion is the standard deviation of the
   alpha=10 model's out-of-time residual draws (the SAME sample the ECDF reads;
   known pregame). A week is LOW-dispersion if its sd is strictly below the
   expanding median of all PRIOR scored weeks' sds (first scored week: no prior
   weeks → status quo). Low-dispersion weeks are ranked by candidate 1's score;
   high-dispersion weeks keep the status-quo ranking. The split is structural
   (expanding median), no tuned constant.
4. **`best_pick_followup_ensemble_distance`** — equal-weight mean of
   candidates 1 and 2's distances. Grounded in the reading: MOD-08 (smoothing)
   and MOD-12 (shrinkage) are two independently measured Brier-positive
   improvements to the SAME probability read; their average is the predeclared
   robustness ensemble of exactly those two, nothing else.

Excluded because already tried and closed or measured elsewhere:
`calibrated_probability` and `key_number_distance` (closed negative,
`docs/best_pick_ranker.md`); `sweep_robustness` itself (deployed incumbent);
within-sweep-tie probability tie-breaks (measured by stage 0,
`best_pick_tiebreak_cfb_stage0_ecdf_gaussian`).

### Metrics and success rule (fixed before scoring)

Per candidate: weekly top-1 correctness; the paired weekly difference vs the
status-quo nominee; week-blocked bootstrap (20,000 samples, seed 20260821) of
the mean paired difference → delta in accuracy points, 95% interval,
`probability_positive`. Descriptive only: top-1 accuracy levels, Kendall tau
between candidate score and pick correctness across all picks, weeks where the
nominee diverges.

**Gate**: a candidate passes the screen iff its full-population week-blocked
`probability_positive >= 0.75`. A pass does NOT activate anything: it makes the
signal eligible to be PREDECLARED for a future NFL look. **NFL activation would
need its own predeclared look** — no NFL window is spent or implied here.

Every cell is recorded to `registry/weak_signals.json` via
`nfl-ats weak-signals record` (league `cfb`, names prefixed
`best_pick_followup_`), whatever the numbers say. Classification per the
binding taxonomy: `unresolved_below_power` by default; `refuted_mechanism`
with `wrong_sign_resolved` ONLY if the whole interval sits below zero; no
positive control was run, so `bounded_by_control` is unavailable.

## Results

Measured 2026-08-21 (`scripts/best_pick_ranker_followup.py`, artifact
`artifacts/best_pick_followup/20260821T175357Z/`). Reproduction check: the
fresh alpha=10 walk-forward reproduced the stored stage-0 artifact's
`home_cover_probability` at `max |diff| = 0.0` over all 11,780 games, so every
signal below is scored off the same model the evaluation scored. No smooth-CDF
probability or residual sd was missing for any pick. The scoring pass is fully
deterministic (fixed bootstrap seed) and was re-run once after wiring in the
provenance helper; both runs produced identical numbers to the fourth decimal,
and the registry cells were recorded from the first of the two identical runs.

Population: **280 weeks / 11,780 resolved picks, seasons 2007–2025** (2006 is
warm-up: no week reaches the 500-game training floor until 2007). Status-quo
top-1 accuracy: **59.29%** (166/280).

| candidate | top-1 acc | delta vs status quo | week-blocked 95% | P+ | divergent weeks | clean-core delta (descriptive) |
|---|---|---|---|---|---|---|
| `smooth_cdf_distance` | 60.00% | **+0.71 pts** | [−4.29, +5.71] | **0.584** | 101/280 | +2.01 pts (P+ 0.730) |
| `alpha2000_distance` | 57.50% | −1.79 pts | [−7.50, +3.57] | 0.239 | 129/280 | +0.50 pts (P+ 0.534) |
| `dispersion_gated_smooth_distance` | 57.86% | −1.43 pts | [−5.36, +2.50] | 0.217 | 54/280 | −0.50 pts (P+ 0.362) |
| `ensemble_distance` | 59.64% | +0.36 pts | [−4.64, +5.36] | 0.527 | 109/280 | +3.52 pts (P+ 0.868) |

Kendall tau between each candidate score and pick correctness across all
11,780 picks: +0.009 / +0.010 / +0.006 / +0.011 (p = 0.22 / 0.20 / 0.40 /
0.16) — none of the orderings carries resolvable rank information, consistent
with the flat-confidence finding this family keeps reproducing.

### What this says

**No cell passes the predeclared 0.75 screen gate on the full population**, so
**no signal earned an NFL-window predeclaration** from this screen. The two
positive-leaning cells (`smooth_cdf_distance` at P+ 0.584,
`ensemble_distance` at P+ 0.527) are directionally interesting but sit close
to coin-flip territory on 280 weeks; per the binding taxonomy an interval
crossing zero is the EXPECTED shape of a real-but-small signal here, so all
four cells are recorded as `unresolved_below_power`, not closed:

- `best_pick_followup_smooth_cdf_distance`: +0.71 pts, P+ 0.584.
- `best_pick_followup_alpha2000_distance`: −1.79 pts, P+ 0.239. The interval's
  upper bound (+3.57) is positive, so `wrong_sign_resolved` is inadmissible
  despite the negative point estimate — MOD-12's routing hypothesis ("the
  Brier gain should help the ranker") is NOT refuted by this cell; it is
  unresolved, with the sign currently leaning against it.
- `best_pick_followup_dispersion_gated_smooth_distance`: −1.43 pts, P+ 0.217.
  Same situation: unresolved, sign leaning against, upper bound positive.
- `best_pick_followup_ensemble_distance`: +0.36 pts, P+ 0.527.

All four are in `registry/weak_signals.json` (league `cfb`, seasons 2007–2025,
recorded via `nfl-ats weak-signals record` reading the artifact — no
hand-typed numbers).

**The clean-core column must be read with discipline.** The ensemble's
clean-core P+ of 0.868 exceeds 0.75, but the PREDECLARED gate was the full
population precisely so that no post-hoc cut could manufacture a pass; the
clean-core cut is a descriptive secondary on the same 11,780 picks, not a new
look and not a pass. It is reported because signs accumulate even when gates
are not met, and it is the one number here that would justify RE-declaring
this exact four-cell design (unchanged) if a future session wants a second
CFB window — that would be a fresh predeclaration decision, not a default.

### NFL activation

**NFL activation would need its own predeclared look.** Nothing here touches
the NFL rotation registry, the live `sweep_robustness` rule, or the v2/v3
nomination pipeline; no NFL window is spent or reserved. If any of these four
signals is ever carried to NFL data, the carry must be predeclared (signal
definition, comparator, window assignment via the rotation rules, gate) before
any NFL number is computed.

### Multiplicity disclosure

This is the second look at the CFB sweep population for a Best-Pick question
(the first being stage 0's within-tie tie-break screen on the same 11,780
picks); these results carry that reuse discount. The four candidates were
frozen before scoring, but the family has a prior look on the data, and the
registry pool (`nfl-ats weak-signals pool --league cfb --effect-units
accuracy_points`) should be re-run rather than quoted from this document.

