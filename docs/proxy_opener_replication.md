# Proxy-opener replication: production model on 2009-2019 SBR-Open lines

Predeclared: 2026-08-19 (US), before any accuracy number in this document is
computed. Written before `scripts/proxy_opener_replication.py` produces any
output. Provenance tags used throughout: **measured** (run this session,
command/path given), **read** (file opened this session), **reported**
(another doc's claim, not reverified here), **inferred** (reasoning, not
evidence).

## Question

`docs/opener_evaluation.md` measured the frozen production model
(`market_residual`/`weak_stack`/ridge alpha 10.0) at the true Tuesday-opener
grade on 2020-2025 (1,537 paired games): **53.36%** under the production
probability rule, season-blocked 95% **[51.97%, 54.56%]**
(`artifacts/opener_evaluation/20260819T174244Z/`, **read** this session). That
archive only reaches back to 2020 (the purchased point-in-time market store's
floor). `docs/sbr_odds_archive.md` (**read** this session, all caveats below
taken from it) just ingested a second, independently-sourced historical odds
archive (sportsbookreviewsonline.com) whose "Open" column reaches back to
2009 -- eleven more seasons an opener-graded look has never touched. This
document predeclares scoring the same frozen model, same pick rules, same
walk-forward recipe, on those seasons, using SBR's Open as a **proxy** for
the Tuesday opener the true archive would have recorded.

## What this is and is not

- Single arm. Production `weak_stack`/ridge/alpha=10.0 only. No candidate,
  no tuning, no variant sweep -- a re-measurement of an already-fixed pick
  stream generator on a different, proxy settlement line, exactly the
  epistemic posture `docs/opener_evaluation.md` used to justify scoring
  inside the ledgered era.
- The line substitution mirrors `nfl_ats.clv.opener_pick_evaluation` exactly:
  the SBR Open home spread replaces `spread_line` in the frame handed to
  `MarginModel.predict` at scoring time (so it is both the grading line and
  the line the pick rules see), while training rows keep the feature table's
  own native `spread_line` (close-era) unmodified -- same "only the settling
  line changes" convention the true-opener evaluation uses for its
  opener-vs-close arms.
- **This is a proxy, not the true Tuesday opener.** `docs/sbr_odds_archive.md`
  section 3b measured SBR's Open against the repo's own `tue_open` archive on
  the only two overlapping seasons (2020-2021): Pearson correlation **0.949**,
  mean signed diff **+0.28**, mean absolute diff **1.36** pts (491 matched
  games, blending both overlap seasons), median absolute diff 1.0 pt, only
  ~45% of games agreeing within half a point. SBR's Open is correlated with,
  but not point-identical to, the true opener -- direction of any resulting
  attenuation is **not** assumed here; it is measured directly in the
  calibration arm below on the same two overlap seasons.

## Population (measured, before any accuracy is computed)

- Feature table: `data/processed/game_features_weak_stack.parquet` (production
  `feature_profile="weak_stack"`, matches `artifacts/active_ats_model.json`'s
  `feature_profile`/`regressor`/`ridge_alpha`, **read** this session).
- Regular season only (`game_type == "REG"`, via
  `nfl_ats.modeling.regular_season_rows`, the same filter
  `opener_pick_evaluation` applies).
- **Main population**: REG games, seasons 2009-2019 inclusive, joined to
  `data/processed/sbr_odds.parquet` on `game_id` (inner join, requiring a
  non-null `open_home_spread`). **Measured**: 2,816 of 2,816 REG games in
  this range have an SBR Open -- **100% match rate**, 256 games/season x 11
  seasons, zero missing opens, 2 of 2,816 (0.07%) flagged
  `open_ambiguous` by the ingestion script's spread/total disambiguation
  heuristic (`2017_14_IND_BUF`, `2019_14_DEN_HOU`, both near-pick'em games).
  Kept in the population -- no selection dimension is opened by dropping
  0.07% of rows on an outcome-blind flag, and dropping them is not necessary
  to keep this predeclaration outcome-blind, so they stay in.
- **Calibration population**: REG games, seasons 2020-2021 inclusive, that
  have BOTH an SBR Open (inner join to `sbr_odds.parquet`) AND a true
  Tuesday-opener quote from `opener_pick_evaluation`'s own paired archive
  (`data/market/raw`, `tue_open` decision label). Exact count is measured by
  the script at run time (join is data-dependent, not assumed in advance);
  `docs/sbr_odds_archive.md` section 3b's own (all-game-type) overlap check
  found 491 of 2020+2021's 554 SBR games matched to `tue_open` -- the REG-only
  paired count in this document's calibration arm will be at or below that.

## Warm-up floor (measured, before any accuracy is computed)

`opener_pick_evaluation`'s convention -- and this script's, which mirrors it
exactly -- requires >= `DEFAULT_MIN_TRAIN_GAMES` (500, **read**,
`src/nfl_ats/constants.py`) completed REG games strictly before a week's
first kickoff before that week is scored at all. The feature table's floor is
season 2009 (**read**, `data/processed/game_features.manifest.json`,
`first_season: 2009`), so early seasons have insufficient history in front of
them. **Measured** this session (cumulative completed-REG-game count before
each week's cutoff, full `game_features_weak_stack.parquet` REG history):

| Season | Weeks scorable | Games scorable |
|---|---|---|
| 2009 | 0 of 17 | 0 of 256 |
| 2010 | 0 of 17 | 0 of 256 |
| 2011 | 17 of 17 | 256 of 256 |
| 2012-2019 (8 seasons) | 17 of 17 each | 256 of 256 each |

2009 and 2010 fail the warm-up floor **entirely** (max cumulative training
games before 2010's last cutoff is 496, still under 500); week 1 of 2011
clears it at 512. This reproduces, on this specific population, the same
warm-up conclusion already documented for the rotation registry's identical
500-game floor (`src/nfl_ats/rotation.py`'s comment: "[2009, 2011] yields 17
scorable weeks, all in 2011" -- **reported** there, **independently
re-measured** here on the SBR-joined population with the same result). This
is a population fact, not an accuracy number, so declaring it here before any
pick is graded does not violate the predeclaration.

**Consequence, declared now**: the scored 2009-2019 population is therefore
**2,304 games (2011-2019, 9 seasons x 256 games)**, not the full 2,816-game
SBR-matched set. `n` for every headline number in this document is 2,304
unless stated otherwise. This is a feasibility floor identical in kind (same
constant, same mechanism) to the one already binding on the true-opener
evaluation and the rotation registry -- it is not a new discount invented for
this run.

## Grade

For every population game: one weekly-refit `market_residual`/`weak_stack`/
ridge(alpha=10.0) model, trained on completed REG games strictly before the
week's first kickoff (identical recipe to `opener_pick_evaluation`), scored
with `spread_line` overridden to SBR's `open_home_spread` for that game. Both
pick rules are reported, from the model's own predicted distribution
(`MarginModel.predict`):

- **Sign rule** (secondary, historical protocol comparability):
  `residual_at_open_proxy > 0`.
- **Probability rule** (**primary** -- what `pool.py`/`backtest.py` actually
  play, per `docs/opener_evaluation.md`'s 2026-08-19 addendum):
  `home_cover_probability_at_open_proxy >= 0.5`.

Settlement margin for both rules: `result - open_home_spread` (SBR Open
substituted as the line for grading cover, mirroring exactly how the true
opener evaluation settles `margin_vs_open = result - tue_open_home_spread`).
Pushes (`margin == 0`) excluded via `nfl_ats.clv.pick_correct`'s existing
push convention -- no new push logic is written.

## Reporting (no accept/reject gate -- this is a measurement)

- Per-season accuracy, both rules, 2011-2019.
- Pooled 2009-2019 (i.e., the 2,304-game scorable subset) accuracy under
  both rules, each with:
  - week-blocked bootstrap interval vs the 50% coin flip,
  - season-blocked bootstrap interval vs the 50% coin flip,
  - `probability_positive` (fraction of blocked resamples with accuracy
    above 0.5) for both blockings,
  - 20,000 samples, seed 20260817 (the project's standing opener-bootstrap
    seed, reused from `docs/opener_evaluation.md` /
    `scripts/surface_profile_opener_eval.py` / `scripts/ridge_alpha_promotion_eval.py`,
    not a fresh choice).
- **Calibration arm**: the same two pick rules, same walk-forward model (by
  construction the training data and per-week model are identical between
  the two grades on 2020-2021, since training never depends on which line is
  substituted at scoring time -- only the settlement/prediction line
  differs), scored on the calibration population at BOTH:
  - the SBR-Open grade (this script's own substitution path), and
  - the true Tuesday-opener grade (`nfl_ats.clv.opener_pick_evaluation`,
    unmodified, restricted post hoc to seasons 2020-2021 from its native
    2020-2025 output).

  Reported paired (same `game_id` set on both grades): accuracy under each
  grade, both rules, and the paired delta (SBR-Open minus true-opener) with a
  week-blocked bootstrap interval, 20,000 samples, same seed. This measures
  the proxy discount directly on overlap instead of assuming the 1.36-pt mean
  |diff| from `docs/sbr_odds_archive.md` translates linearly into an accuracy
  effect.
- Comparison, stated plainly once both arms are computed: how the
  2009-2019 (2011-2019 scorable) proxy-opener pooled accuracy compares to the
  2020-2025 true-opener 53.36% [51.97%, 54.56%], with the calibration arm's
  measured discount as the explicit bridge between the two eras/instruments.

## Frozen expectation, stated before any number is seen

The 2020-2025 true-opener figure is 53.36%, season-blocked 95%
[51.97%, 54.56%] (production probability rule, **read**,
`docs/opener_evaluation.md`). SBR's Open sits somewhere between the true
opener and the close in informational sharpness -- **direction of the
resulting accuracy effect is not assumed**: SBR's Open could read closer to
the sharper true opener (small attenuation toward it), or it could behave
more like an intermediate/blended quote (larger attenuation toward the close,
which the same frozen model scores at ~52.1% under the production rule per
`docs/opener_evaluation.md`), or -- because SBR aggregates across books with
an unpublished capture methodology (`docs/sbr_odds_archive.md` section 3b) --
it could occasionally be noisier than either endpoint on a per-game basis
even while correlating at r=0.949 in aggregate. **Per AGENTS.md, whatever the
resulting interval's relationship to zero, that is not itself grounds to
close this line of work.** The calibration arm exists precisely so this
document does not have to guess: it measures the same rules on the same 2020-
2021 games at both grades and reports the discount directly.

## Registry recording

At the end of this work, the pooled 2009-2019 (2011-2019 scorable)
production-probability-rule result is recorded to
`registry/weak_signals.json` via `nfl-ats weak-signals record`
(`league=nfl`, `effect_units=accuracy_points`, effect = accuracy points
**above 50** i.e. `(accuracy - 0.5) * 100`), name
`proxy_opener_production_rule_2009_2019`, classification
`unresolved_below_power` unless the whole week-blocked interval sits below
zero (which would make `wrong_sign_resolved` the admissible closing ground)
or a positive control bounds the instrument (neither expected, neither
assumed in advance). Every numeric CLI argument is read programmatically from
this run's artifact JSON -- no hand-typed numbers, matching
`scripts/surface_profile_opener_record.py`'s precedent. The registry is read
back after recording to verify the write; `--replace` is used only if the
name was already present from a prior partial run of this exact task.

## Files

- `scripts/proxy_opener_replication.py` -- implementation (single production
  arm, SBR-Open substitution, calibration arm against
  `nfl_ats.clv.opener_pick_evaluation`).
- `scripts/proxy_opener_replication_record.py` -- reads the artifact JSON and
  calls `nfl-ats weak-signals record`.
- `artifacts/proxy_opener_replication/<run-id>/` -- output artifact (summary
  JSON plus per-game parquet frames).

---

## Results

**Measured**, `scripts/proxy_opener_replication.py`, artifact
`artifacts/proxy_opener_replication/20260819T194330Z/summary.json`, run
2026-08-19. Config: `weak_stack`/ridge/alpha=10.0 (matches
`artifacts/active_ats_model.json` exactly), `min_train_games=500`, 20,000
bootstrap samples, seed 20260817.

### Main arm: 2009-2019 proxy-opener grade

Population confirms the predeclaration exactly: 2,816 of 2,816 REG games
2009-2019 have an SBR Open (100% match, 2 flagged `open_ambiguous`, kept
in). Warm-up reproduces the predeclared table exactly: **2009 and 2010 score
zero weeks** (max cumulative training games before 2010's last cutoff is
496, still under 500), **2011-2019 score all 17 weeks each**. Scored
population: **2,304 games, 153 weeks, 9 seasons (2011-2019)**.

| Season | Games | Sign-rule accuracy | Production-rule accuracy |
|---:|---:|---:|---:|
| 2009 | 0 | -- (warm-up, not scorable) | -- |
| 2010 | 0 | -- (warm-up, not scorable) | -- |
| 2011 | 256 | 46.80% | 45.20% |
| 2012 | 256 | 50.00% | 50.00% |
| 2013 | 256 | 52.05% | 52.46% |
| 2014 | 256 | 51.78% | 51.38% |
| 2015 | 256 | 49.60% | 50.00% |
| 2016 | 256 | 51.23% | 52.87% |
| 2017 | 256 | 54.03% | 52.02% |
| 2018 | 256 | 49.59% | 47.13% |
| 2019 | 256 | 49.19% | 52.42% |

Positive in 6 of 9 scorable seasons under the production rule, 6 of 9 under
the sign rule (both rules negative in 2011 and 2018; the rules disagree in
sign only on 2019, where sign is negative-ish/flat at 49.19% and production
is positive at 52.42%).

**Pooled 2011-2019 (n=2,304), vs the 50% coin flip:**

| Rule | Absolute accuracy | Week-blocked 95% (pts above 50) | Week-blocked P+ | Season-blocked 95% (pts above 50) | Season-blocked P+ |
|---|---:|---:|---:|---:|---:|
| **Production probability rule (primary)** | **50.38%** | [-1.60, +2.37] | **0.6468** | [-1.37, +1.87] | **0.6829** |
| Sign rule (secondary) | 50.47% | [-1.66, +2.60] | 0.6669 | [-0.81, +1.75] | 0.7582 |

Both intervals contain zero under both blockings. **Per AGENTS.md, that is
the EXPECTED shape for a real small signal at this evaluator's resolution and
is not grounds to reject** -- the correct read is `probability_positive`:
65-68% under the primary production rule, 67-76% under the sign rule, i.e.
leaning positive, unresolved either direction. This is markedly closer to a
coin flip than the 2020-2025 true-opener figure (53.36%, season-blocked
[51.97%, 54.56%], P+ effectively 100% by the season-blocked interval
excluding 50% entirely) -- see the calibration arm below for how much of that
gap the proxy-line discount can explain.

### Calibration arm: 2020-2021 dual grade (same instrument, same games)

466 REG games in 2020-2021 have both an SBR Open and a true `tue_open`
quote (of 528 SBR-matched, 466 also matched to the store's opener archive).
**Measured** mean absolute line difference on this exact paired set: **1.371
pts** -- reproduces `docs/sbr_odds_archive.md`'s independently-computed
1.36-pt figure (that document's number spanned all game types 2020-2021;
this one is the REG-only walk-forward-scored subset) almost exactly, a good
cross-check that the two ingestions agree.

| Grade | Sign-rule accuracy | Production-rule accuracy |
|---|---:|---:|
| SBR Open (proxy) | 54.85% | 52.86% |
| True Tuesday opener | 53.29% | **53.73%** |

(The true-opener production-rule figure here, 53.73% on just the 2020-2021
subset, closely reproduces what the six-season 2020-2025 per-season table in
`docs/opener_evaluation.md`'s addendum implies for these two seasons alone --
weighting 2020's 52.3% and 2021's 55.1% by 256/272 games gives 53.75% --
consistent, not a new independent fact.)

**Paired delta (SBR-Open minus true-opener), same games, same weekly-refit
models:**

| Rule | Estimate (pts) | Week-blocked 95% | Week-blocked P+ | Season-blocked 95% | Season-blocked P+ |
|---|---:|---:|---:|---:|---:|
| Sign rule | +1.556 | [-0.88, +3.95] | 0.8946 | [-0.80, +4.09] | 0.7494 |
| **Production rule (primary)** | **-0.865** | [-2.62, +0.91] | **0.1670** | [-2.09, +0.45] | 0.2487 |

**The two rules disagree on direction.** Under the sign rule, SBR's Open
reads *easier* than the true opener (+1.56 pts, leans positive, P+ 0.89).
Under the production probability rule -- the primary rule this whole
document is built around -- SBR's Open reads *harder*, i.e. the proxy
understates the true opener's accuracy by 0.865 pts (P+ 0.167, meaning 83.3%
of week-blocked resamples are negative). Both intervals contain zero at
n=466/35 weeks; **per AGENTS.md this is not grounds to reject either
reading** -- it is unresolved, reported with `probability_positive`, not
collapsed to "contains zero." What it rules out is treating the proxy
discount as a settled, single-direction correction: on this evidence it
leans toward SBR-Open modestly *understating* true-opener accuracy under the
rule that matters, but the interval does not exclude the proxy overstating
it either.

### Reading, stated plainly

- **Proxy-opener 2011-2019 pooled accuracy (50.38% production rule) sits
  much closer to a coin flip than the true-opener 2020-2025 figure (53.36%)
  and than the true-opener reading on the very same 2020-2021 seasons
  measured here (53.73%).** The gap is roughly 3 points.
- The calibration arm's measured discount under the production rule is only
  -0.865 pts (proxy reads lower) -- **nowhere near large enough by itself to
  explain a ~3-point gap**, and it is itself unresolved (P+ 0.167, interval
  crosses zero). Applying it naively (**inferred**, not measured, a back-of-
  envelope extrapolation only) would move the 2009-2019 headline to roughly
  51.2%, still well short of 53.36% and still not resolved either.
- **This does not refute the model or the true-opener finding.** Neither
  admissible closing ground applies: the whole interval is not on the wrong
  side of zero (both blockings' upper bounds are positive), and no positive
  control was run here. The correct classification is
  `unresolved_below_power`, and the honest headline is that **11 more
  seasons of a proxy instrument do not confirm the 2020-2025 opener edge at
  anywhere near the same size** -- whether that is because (a) the proxy
  line is a genuinely worse settlement instrument than the true opener in a
  way the 2-season calibration sample is too small to pin down, (b) the
  model's edge over the market is itself era-dependent and smaller in
  2011-2019 than 2020-2025, or (c) some combination, is not resolved by this
  measurement and is not guessed at further here.
- Consistent with `probability_positive` framing throughout: the primary
  2009-2019 pooled number leans positive (65-68% probability) but is not
  resolved: it must be recorded, not discarded, and not described as a
  negative result.

## Registry recording

Recorded via `scripts/proxy_opener_replication_record.py` (reads
`artifacts/proxy_opener_replication/20260819T194330Z/summary.json`
programmatically; no hand-typed numbers). Name
`proxy_opener_production_rule_2009_2019`, `classification
unresolved_below_power` (neither admissible closing ground applies, per the
reading above), effect = week-blocked pooled production-rule accuracy above
50 (+0.381 pts), full detail -- including the sign-rule read, the calibration
arm's rule-dependent discount, and the comparison to 53.36% -- in the
recorded `--notes`. Registry read back after write to confirm.
