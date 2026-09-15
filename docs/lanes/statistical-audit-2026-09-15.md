# Lane: statistical audit 2026-09-15

## Goal
Owner audit request: find where statistics/probability/ML principles have been
violated or read too liberally, with the suspicion that past sessions chased
in-sample backtest performance.

## State
Six audit areas. Five ran as subagents (model selection, leakage/chronology,
in-sample constants, inference machinery, probability/calibration). The sixth
(reported numbers and reader-facing framing) was done in-session; findings below
are measured unless labelled otherwise.

### Measured findings, most severe first

1. **The served policy is a greedy-forward-selection argmax scored on the games
   it is reported on.** `artifacts/overlay_subset_composition/20260914T161418942653Z/result.json`
   enumerates 127 subsets and records a greedy path; the played chain
   (coach fade + division revenge + player arrests) is step 3, the in-sample max
   at +1.331 accuracy points. Steps 4-7 decline. The artifact's own
   `selection_caveat` says the top figure "is an upper bound inflated by
   selecting the maximum over 127 correlated candidates scored on the same
   2020-2025 archive" and "is not a prospective expectation" - yet the board
   headlines it as "Played policy - archive score 56.8% - 1,503 opener-graded
   games". 127-subset spread: min -5.26, median -2.20, max +1.33, sd 1.83.

2. **The composition is a hard flip, which AGENTS.md declares inadmissible.**
   The artifact's `combination_rule` reads: "each member's flip sets
   home_cover_probability to exactly 1 - baseline probability ... a subset flips
   a game if ANY member fires". AGENTS.md: "A member that can only return p or
   1-p is a flip, not a signal, and is inadmissible on the served card."
   CURRENT_PREDICTIONS.md line 11 states the rules "flip it once when any one of
   them fires" and "This week they changed 4 picks" - 4 of 16.

3. **Decisive-game records are not reported.** `member_flip_counts`: arrests 24,
   coach 106, division 156. The promoted arrest policy's headline is "+0.399
   accuracy points on 1,503 games, probability_positive 0.8562"; its decisive
   record is **15-9 on 24 flipped games** (inferred from +0.399 x 1503 = 6.0 net
   games over 24 flips). AGENTS.md requires the decisive-game record before the
   headline effect.

4. **The displayed cover chance is a 12-cell in-sample lookup, floored at 50%.**
   `src/nfl_ats/displayed_confidence.py:12-14,76-81`: 4 spread buckets x 3
   probability bands, `PSEUDO_OBSERVATIONS = 20.0` (underived), cell win rates
   from `prior_rows: 1503` - the same games. Games sharing a cell get the same
   number: five of this week's sixteen picks display exactly 57.0%.
   Lines 80-81 mask any served side whose calibrated probability is below 0.5 to
   exactly 0.5; three picks this week show 50.0%, meaning the system's own table
   puts the served side below a coin flip. The guard at lines 396-400 that raises
   on `displayed < 0.5` is **dead code** - `calibrate()` already floored it.

5. **Model-only probabilities are anti-calibrated.**
   `artifacts/pick_probability/20260914T232538Z/metadata.json`
   `model_only_confidence_bands` on 1,503 games:
   [.50,.52) n=344 -> 57.85% | [.52,.55) n=405 -> 54.81% | [.55,.58) n=353 ->
   52.41% | [.58,.62) n=251 -> 54.58% | [.62,1] n=150 -> 50.67%.
   Weighted reliability slope **-0.401** (perfect = +1.00); approximate Brier
   0.25182 vs 0.25000 for a flat 50% - worse than saying nothing.

6. **Accuracy is confined to short spreads.** Same artifact's cell table, 1,503
   games: spread <= 7 -> 55.94% (n=1178, +/-2.86); spread >= 7.5 -> 49.23%
   (n=325, +/-5.44); 10.5+ -> 48.09% (n=131). Known and honestly documented in
   `docs/big_spread_diagnosis.md`, but the card still serves those games and
   `CLE at JAX JAX -8.5` displays 56.1% off an n=88 cell whose whole bucket is
   50.0%.

7. **6,577 recorded looks; the promoted component is not near the top.**
   `registry/weak_signals.json`: 6,577 signals, 6,534 `unresolved_below_power`.
   Of 6,131 with a non-placeholder probability_positive, **1,148 exceed the
   promoted arrest policy's 0.8562**. Both tails are inflated together
   (>0.90: 914 obs vs 613 expected-if-uniform; <0.10: 956 vs 613), and mean P+ =
   0.484 - the signature of correlated re-measurement and/or too-narrow
   intervals, not of net signal.

8. **Measured effect sizes track the noise curve with no floor.** 3,675
   accuracy-point measurements, n>=30, controls excluded: median |effect| is
   2.47 pts at median n=70, 0.96 at n=266, 0.60 at n=1503, 0.155 at n=4497 -
   a roughly constant ~0.5 ratio to the sampling-noise scale across two orders
   of magnitude of n. A real fixed-size effect would flatten; this does not.

9. **The production window is not fresh.** `registry/rotation_registry.json`
   `season_usage`: 2020 and 2021 have each been spent **72 times**, versus 2-10
   for every other season. The pick-probability fit population is seasons
   2020-2025, so every promoted component was selected partly inside the two
   most heavily mined seasons.

10. **A reader-facing percentage is a constant.**
    `src/nfl_ats/dashboard/findings_content.py:107` `PLAYED_CARD_EXPECTATION_PERCENT = 55`,
    rendered by `src/nfl_ats/publishing.py:285` as "The planning estimate remains
    ~55%". AGENTS.md: "every percentage a reader sees is read from an artifact
    keyed to the active model, never from a constant."

### What is built correctly (checked, not a problem)
- `src/nfl_ats/pick_probability_fit.py:307-340`: leave-one-season-out is real -
  per-fold standardisers fitted on train only, per-fold coefficients, held-out
  season predictions. Fold coefficients are stable
  (`composition_flag_sum` .238-.295, `market_move_toward_home` .159-.248).
  Out-of-season record 841-662 (55.96%) vs model-only 819-684 (54.49%).
- Strength band edges are tertile quantiles
  (`STRENGTH_BAND_QUANTILES = (1/3, 2/3)`), not a best-cell search.
- The overlay-composition artifact carries an accurate, explicit
  `selection_caveat` and `predeclaration_note`.
- `docs/big_spread_diagnosis.md` and `docs/spread_gap_zone_retired.md` are
  honest about mined archives and refuse to bolt on a threshold flip.

### Open
- The four-term probability collapses nine heterogeneous overlay members into a
  single `composition_flag_sum` count sharing one coefficient
  (`src/nfl_ats/pick_probability.py:49-55`). That is not "every situational
  signal as a fitted term"; it forces a weather flag and an arrest flag to carry
  identical weight, and treats two flags as evidence "2".
- `model_logit` fold coefficients range .090-.317 (3.5x). The ATS model is the
  least stable and least weighted term in its own served probability.
- No reliability table, log loss or Brier is computed for the SERVED probability
  against a market baseline anywhere found this session.

## Next
Owner decision on which findings to act on. The cheapest high-value fixes:
report the decisive-game record beside every overlay headline; replace the
displayed-confidence floor with a served-side that follows the one probability;
read the planning estimate from an artifact.

## Tried
Nothing changed in code. Read-only audit.

## Subagent report 1 of 6: model selection and search (verified in-session where noted)

**The decisive fact, already in the repo.** `docs/edge_audit_redteam.md:41-49`
(read, verified) LOSO-CV'd the subset-SELECTION PROCEDURE: argmax on five
seasons, score the frozen choice on the held-out sixth, all six folds. Fold
deltas +2.73 / +0.85 / +1.21 / -1.13 / -4.51 / +1.50. Pooled over 1,503
held-out games: **0.0000 accuracy points, week-blocked [-2.1462, +2.1348],
probability_positive 0.4930.** Rank-stability rho 0.7207 does not survive a
within-week flip-shuffle null (empirical p 0.2375). Recorded, never acted on.

**Headline decomposition** (reported, from `opener_evaluation/20260914T161406Z/per_game.parquet`):
sign rule 52.83% -> +1.00 gaussian_median (208 picks) -> +0.66 home-side offset
(46 picks) = 54.49% headline -> +1.33 flip card (272 flips) = 55.82%.
Binomial SE 1.29 pts. **+2.99 of the +5.82 over a coin flip comes from three
layers each maximized on those same 1,503 games.** The sign rule itself moved
only 52.50 -> 52.83 across 33 re-scorings of the frozen 1,537-game sample.

**Additional findings (agent, unverified unless marked):**
- `home_side_location.py:20` `HOME_SIDE_OFFSET_BUCKETS` is a post-hoc 3-of-5
  mask; `docs/home_side_offset_promotion.md:207` admits the restriction was
  chosen on the same mined archive. Also underived: `PRIOR_WEIGHT_GAMES=100.0`,
  `TRAILING_SEASONS=5`.
- `gaussian_median` location shift is noise-dominated: residual median +0.3387,
  std 12.6460, n=886 -> SE 0.547, |median|/SE 0.62. Worse, `margin.py:900-935`
  draws residuals from a temporary 80%-fit estimator while the deployed
  estimator is refit on 100% - it corrects the wrong model's bias.
- `docs/unserved_tilt_marginals.md:136-140` promoted six members by the rule
  "all six with probability_positive above 0.5 are served", from a 15-way
  in-sample screen. Two move 1 and 14 games. AGENTS.md forbids exactly this.
- **No outer test period exists.** `nested_walk_forward_evaluation`
  (`evaluation.py:125-215`) is correct and dormant since 2026-08-12;
  `DEFAULT_EVALUATION_CANDIDATES` does not contain ridge/market_residual/
  weak_stack. The only untouched evidence in the repo is 2026 Week 1 (9-7).
- **VERIFIED:** `cli_commands/clv.py:468` hard-codes
  `"hypothesis_frozen_before_scoring": True` on every run, hashed into
  configuration_sha256 -> model_id. Launders repeated looks as predeclaration.
- **VERIFIED:** headline denominator. per_game.parquet has 1,537 rows, 1,503
  non-null, 819 correct, acc 0.54491; metadata writes `"games": 1537` and
  HANDOFF.md:56 quotes "1,537 games". `active_ats_model.json` `"correct": 1085`
  is round(accuracy x games), not a count.
- `model_ledger.py:373,410,507,726` ranks arms by `max(probability_positive)`
  over linked evidence, rendered as "best evidence X% likely real". 17 of 68
  rows link more than one entry; one shows 99.5% from 7 entries.
- No promotion protocol: `activate_matching_ats_model` is called only from the
  end of `margin-predict` (`cli_commands/prediction.py:380`). Declared config
  space 72 profiles x 2 regressors x 2 targets x 15 mappings = 4,320. Nothing
  records how many were scored. No shrinkage/multiplicity field anywhere.
- `git show 68b4dc0`: weak_stack lost on the close grade (51.57 vs 52.05), won
  on the opener grade (+0.33), was promoted, and AGENTS.md gained the
  "close-graded number may never veto a play" rule in the same commit.

**Cleared by the agent (corrects earlier entries in this lane):**
- `ridge_alpha=10.0` is an inherited sklearn default derived on the CFB
  instrument, not tuned on the NFL evaluation set; the one NFL look kept the
  incumbent when accuracy fell.
- `weak_stack` is not an argmax: 57 of 72 profiles are weak_stack_* one-family
  extensions and the served profile is the bare base - all 56 failed to displace
  it.
- 1,537 / 1,503 / 2,075 are not unlike-population laundering; HANDOFF.md:61
  warns against conflating them. Caveat: opener coverage is time-structured
  (227/256 in 2020 rising to 272/272 in 2023-25), so era splits inside the
  archive compare unlike coverage.
- Bootstrap machinery (week/season-blocked, fixed seeds, probability_positive
  over binary verdicts) is correct on every artifact opened.
- `docs/joint_probability_model.md` and `docs/four_term_probability.md` already
  use proper LOSO with an explicit look_log. The pattern to generalize exists.

**Agent's overall read, which matches mine:** the failure is not concealment.
Every damaging result was recorded honestly, often in the artifact itself. No
disclosure ever changed what is served.

## CORRECTION to finding 2 above (verified this session)

**The served card does NOT hard-flip.** `card_view.py:342-347` `_served_probability_frame`
takes the overlaid frame but overwrites `home_cover_probability` with the RAW
model value from `predictions`, then calls `attach_pick_probability`, which
applies the four-term logistic with the members as signed +1/-1/0 terms in
`flag_sum`. The `p -> 1-p` chain survives only in the challenger ledgers, the
composition artifact and the card's disclosure prose. The served architecture is
compliant with the one-probability rule. Finding 2 as originally written is
withdrawn; the card's prose "flip it once when any one fires" describes the
attribution semantics, not the served path.

**Latent risk (read):** if `active_pick_probability.json` is missing,
`card_view.py:416-420` falls back to `production_overlay.overlaid_predictions` -
the hard-flip chain - as the served card. `publishing.py:1313-1331` fails closed,
so it is contained.

## Subagent report 2 of 6: in-sample constants

**Headline (verified in-session).** The most decision-moving quantity is not a
constant in `src/` but the fitted `flag_sum` coefficient.
From `artifacts/pick_probability/20260914T232538Z/coefficients.json`:
intercept -0.0662, model_logit 0.2168, flag_sum **0.2599**, move_toward_home
0.2103, move_available 0.0464. The model probability spans only 0.210-0.674, so
`model_logit x logit(p)` contributes at most +/-0.09 to z while one flag
contributes 0.26. Crossover model p to pick HOME: **0.5757** with no flags/move,
**0.2903** with one +1 flag. **91.0% of the 1,537 opener games fall inside that
band** (measured), i.e. one flag alone decides the side. Structural away tilt:
a no-flag no-move game is served AWAY 89.3% of the time vs the model's 56.7%.
`docs/four_term_probability.md:365-385` (agent read): the nine flags "were each
promoted, and then all nine were jointly fit as covariates, on this exact same
1,503-game 2020-2025 population"; flag_sum "is not an independent replication";
51 looks in that doc alone, on top of 33 and 19 in two prior lanes.

**Constants - clean:**
- `TOTAL_LOW_SIDE_SHADE_POINTS = -1.0` (`tiebreaker.py:49`) - trained 2009-2017,
  tested 2018-2025, OOS +0.0780 AE, P+ 0.843, source tagged in src/. The
  template for how this should be done.
- `WINDOW_DAYS = 14` (`player_arrests_back_side_overlay.py:30`) - selected from
  exposure counts without reading outcomes.
- tank-zone and precip cut points - predeclared cells.
- `STRENGTH_BAND_QUANTILES` - tertiles. `CONFIDENCE_BAND_EDGES` accuracies come
  out monotone (0.489/0.564/0.583/0.593/0.602), so no lopsided cell selected.
  (Separately: the lowest band [0.50,0.52) runs 48.9% on 319 games out-of-season.)

**Constants - in-sample fitted:**
- `TEMP_GAP_THRESHOLD_F = 25.0` - chosen as "the strongest tuesday_noon cell -
  the only one whose recorded interval already excludes zero" (disclosed).
- `SERVED_SPREAD_THRESHOLD = 7.0` (`best_pick_nomination.py:36`) - arm B2 of
  three (B1 +0.00 P+0.500, B2 +1.96 P+0.688, B3 -4.90 P+0.179); the doc admits
  "the decision to look at buckets at all was made after seeing the bucket
  table". No OOS number reported beside the 58.82%. Per the binding rule
  P+ 0.688 on 26 decisive weeks does NOT close this - it requires the OOS number.
- `NOMINATION_RIDGE_ALPHA = 2000.0` - its own ranker measures 50.96% (53/104).
- `MODEL_RESIDUAL_WEIGHT = 0.2` - "measured on 1,537 opener-graded games", one
  number, full sample; provenance lost when the no-comments rule deleted its
  docstring.
- `OVERLAY_WEEK_MAX = 8` (coach fade) - magnitude mined inside 2018-2025.

**Two more live defects:**
- Frozen accuracy literals reach readers: `public_board.py:1106-1109` prints
  "53.76% versus 53.36% ... (+0.399 points, P+ 0.86)" from constants at
  `player_arrests_back_side_overlay.py:32-36` introduced 2026-08-20 under a
  DIFFERENT active model. Same in `handoff.py:213-217`,
  `dashboard/findings_content.py:237`. Stale-number rule; publish-board should
  fail closed.
- `board_content.py:2382-2434` `_flip_line` never receives the
  PickProbabilityModel; it searches where the RAW model curve crosses 0.5, while
  the served card crosses at 0.5757 / 0.5227 / 0.2903. Off by up to 28
  probability points. `_cover_curve_offset_zero_note` already covers the chart
  but nothing covers the flip line.
- `NO_SPLIT_HALF_RELIABILITY_MAX = 0.10` (`weak_signals.py:36`) is looser than
  AGENTS.md's "zero split-half reliability" - makes closing easier than written.

**Hard-flip switches all verified OFF:** `pick_refresh.py:118-120`, gated at
:1270 / :1275 / :1306; with them off `new_side = model_only_side` (:1290) which
reads the calibrated probability. `OWNER_HELD_MEMBERS` flags hardcoded to 0 at
`pick_probability.py:474-475`.

**Cleared:** displayed_confidence reliability cells are strictly walk-forward via
`prior_rows_before` and the shrinkage can never flip a side; `_flip_line`
direction is adverse-only (compliant); no hard-coded accuracy percentages in
board body text (all f-string from artifacts); `schedule_flag_features.py`
constants are not on any served path; `crew_tilt_refresh_overlay.py` floats are
challenger-only.

## Subagent report 3 of 6: inference machinery

**Headline (VERIFIED in-session via `nfl-ats weak-signals pool --league nfl
--effect-units accuracy_points`).** The pooled read and the sign test on the
IDENTICAL pile contradict each other:
- pooled_effect +0.4685, standard_error 0.03202, probability_positive 1.000000,
  signals 3351, standard_errors_floored 55
- sign_test on the same pile: 1470 favouring candidate vs 1671 favouring
  baseline, share 0.4680, **p = 0.00036** - leaning NEGATIVE.

**The pooled SE is not a possible number (measured).** 95% width 3.92 x 0.032 =
0.126 accuracy points = **5.6 games** on the 4,431-game history. An effect of
+0.4685 pts requires >= 21 discordant games, flooring the paired SE at 0.103 -
so the pooled SE is **3.2x below the best case any measurement of this
population could achieve**, and ~14x below a realistic one (400 discordant
games -> SE 0.451).

Two causes:
- **Independence.** `weak_signals.py:1082` computes
  `standard_error = sqrt(1/total_weight)` with no correlation term.
  `overlap_pairwise_count` = 5,213,964 of 6,213,975 possible pairs = **84%**.
  The code's own note at :1116 and `family_overlap_warnings` at :1332 both say
  not to believe the interval; nothing in the arithmetic reads either.
  Family-level `tau/sqrt(495)` = 0.0567 is 2.6x wider; honest figure is nearer
  tau itself (~1.26), at which point the interval spans zero.
- **Tail dominance (measured decomposition, reproduces +0.468455 exactly).**
  effect > +10 pts: k=158, 1.82% of weight, contributes **+0.5597**.
  effect < -10: k=23, 0.04%, -0.0047. |effect| <= 10: k=3170, **98.14%** of
  weight, contributes **-0.0865**. Re-runs of the shipped function: trimmed 5%
  ends -> **-0.0138, P+ 0.144**; restricted to n>=1000 games -> **+0.0430**.
  Top rows are n=8-game cells with effect +100.00 and P+ 1.0.

**Recorder accepts mutually contradictory numbers.** `weak_signals.py:294-313`
checks only finiteness/ordering. Agent recorded and had ACCEPTED:
effect +0.30, SE 4.0, interval [-9,+9], probability_positive 0.9999 (implied
0.5299); 0.0001 equally accepted. Live registry: **52 rows disagree between SE
and interval by >2x, 48 by >4x** (worst 25.8x). Pool weight goes as 1/SE^2, so
those enter with weight wrong by up to ~600x. Also corrupts
`_plausibility_curve`'s variance floor (:941).
`record_signal` at :673-683 silently widens a floored SE and leaves the interval
alone - manufacturing this same incoherence (55 rows floored in the live pool).

**271 rows store probability_positive at exactly 0.0 or 1.0** (216 at 1.0).
`rows_needing_remeasurement` (:1181-1190) flags only the 55 pessimistic ones -
the 216 optimistic saturations are invisible to the ledger's own audit.

**A closing ground fires on a point estimate.** `set_reliability` (:782-832)
accepts reliability_low/high, validates them, then stores only the point;
`_SIGNAL_FIELDS` (:84-111) has no interval field. Agent recorded reliability
0.08 with interval [-0.40,+0.55] and `refuted_mechanism` was ACCEPTED. Live:
**230 rows at or below the 0.10 ceiling could be closed today with no interval
requirement**, 223 still unresolved_below_power; 268 rows have a reliability
interval existing only as free text. Test is `reliability <= 0.10` not
`|reliability| <= 0.10` (registry min -0.741). Spearman-Brown applied to
negative r at `pbp_coaching_traits.py:538-541` and `cfb_qb_dependence.py:316`;
the one live closure stores -0.6257 = SB of raw r ~ -0.24.

**`rotation._validate_closing_ground` (`rotation.py:336-362`) is the weaker
gate** and is the one AGENTS.md routes verdicts through. ACCEPTED by
`record_look`: wrong_sign_resolved with no interval; wrong_sign_resolved with
interval [-2.0,+5.0]; no_split_half_reliability with no reliability number;
verdict "confirmed" on P+ 0.51. The weak-signal path correctly rejects the first
two. Currently latent - the one live closed_negative window carries
[-2.86,-0.22]. Fix: delegate to `weak_signals.validate_closure`.

**Other:** `refresh_triggers.py:761-763` closes on raw upper<0 with no widening
margin while `estimation_variance.py:80-84` documents coverage 0.90-0.94 until
~50 blocks; `experiment_runner.py:2801-2845` does it correctly with a 1.099x
margin. Split-half is a single fixed `week % 2` split reused in every bootstrap
draw, so the CI carries zero split-choice noise. `purged_cv.permute_target`
permutes globally, not within season. `cfb_audit.py:191-198` "permutes" i.i.d.
normals - a no-op. **No permutation null exists anywhere for the small-lopsided-
split comparison the binding rule names.** `inflate_recorded_interval`
(`estimation_variance.py:698`) has an abs() that silently flips a side (latent,
no caller). Nothing counts looks numerically: grep for bonferroni/fdr/
multiplicity/n_looks returns one boolean and one text warning.

**VERIFIED CORRECT (agent checked each numerically):**
- probability_positive is a correct bootstrap tail fraction;
  `ZERO_ATOM_CREDIT = 0.5` is the right convention for a lattice-valued paired
  difference.
- Normal approximation is NOT the error: median |pp - Phi(effect/SE)| = 0.004
  across the registry; only 1.74% of 6,156 rows differ by >0.10.
- The comparison is genuinely paired (McNemar-equivalent). Synthetic, 60
  discordant of 1,600: repo SE 0.00478, exact McNemar 0.00484,
  independent-proportion 0.01768 (3.7x too WIDE). No proportion SE in src/.
- Signs are predeclared integer literals, never fitted; no abs() before
  probability_positive.
- `block_bootstrap_means` is the correct ratio-form block bootstrap; week and
  season blocking run side by side; season blocking comes out wider.
- `wrong_sign_resolved` in the weak-signal registry requires the whole interval
  on the wrong side (verified by rejection tests).
- `dependence.py:92` and `pbp_coaching_traits:459-474` permute within season.
- Dead-heat 0.5 rows drop out of the pool entirely; flagging them closes nothing.
- DerSimonian-Laird algebra is textbook-correct; the error is the independence
  assumption feeding it.
- `prospective.py` contains no inference machinery; `freeze_forecast` refuses
  rows at or after kickoff.

**Agent's bottom line, which matches mine:** the per-experiment machinery is
sound; the failure is concentrated at the aggregation and recording layer.
Nothing here justifies closing any individual signal - the fix is to stop
quoting the pooled figure and `excludes_zero` while the assumption behind them
is known false.

## Subagent reports 4 and 5 of 6, plus the flip question RESOLVED

### RESOLVED: the hard flip does NOT decide the served side (measured)

Two agents disagreed. Settled by reproducing the served sides arithmetically
from `artifacts/pick_probability/20260914T232538Z/coefficients.json`:

| game | raw p(home) | 4-term, 0 flags | 4-term, +1 home flag | hard 1-p | served |
|---|---|---|---|---|---|
| ARI@LAC | 0.362 | 0.4529 AWAY | **0.5178 HOME** | 0.638 AWAY | LAC (HOME) |
| ATL@PIT | 0.475 | 0.4779 AWAY | **0.5428 HOME** | 0.525 AWAY | PIT (HOME) |
| BAL@IND | 0.456 | 0.4739 AWAY | **0.5388 HOME** | 0.544 AWAY | IND (HOME) |
| WAS@PHI | 0.367 | 0.4541 AWAY | **0.5190 HOME** | 0.633 AWAY | PHI (HOME) |

All four served sides are reproduced exactly by the fitted logistic with one +1
flag; the hard `1 - p` write would serve the OPPOSITE side on all four. The
`1.0 - probabilities` write at `four_overlay_composition.py:557` exists and is
mandatory for FAIL_CLOSED_MEMBERS, but its output is discarded by
`card_view.py:342-347`. **The served architecture is compliant with the
one-probability rule.** Finding 2 of this lane stays withdrawn.

Also measured: `pool_card.csv` in the week-1 artifact is the RAW model card
(pick_probability is exactly max of p and 1-p of the raw probability) and differs
from the published card on at least 5 of 16 games. It is not the served card.

### Leakage and chronology (agent 4)

**CONFIRMED LEAK, verified in-session: observed game-time weather is a served
model feature.** `margin_feature_columns` for market_residual/weak_stack returns
90 columns including `temp` and `wind` (measured). Joined to
`data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet` over
n=3,010 games, the feature-table `temp` equals `actual_temp_f` **exactly on
100.0% of rows**; the honest pool-decision forecast differs by MAE 2.94 F
(wind corr 0.662, MAE 3.13 mph). Source: `features.py:683-684` ->
`constants.py:319-320` -> SCHEDULE_FEATURES -> FEATURE_SETS full.

**Measured effect on the headline: a dead heat.** Agent re-ran the real
`opener_pick_evaluation` (n=1,537) three ways: observed weather 0.52828 sign /
0.54491 prob; pool-decision forecast at identical coverage 0.52828 / 0.54691;
weather dropped 0.54291 / 0.54424. Swapping observed for forecast changes 10 of
1,537 picks, decisive record **5-5 vs 5-5**, probability_positive about 0.5. The
honest forecast is if anything slightly better. **Fix on invariant grounds, not
because the backtest is inflated.** Side effect: the
`weak_stack_oracle_weather` leak control (`constants.py:617-621`) cannot work,
because its incumbent already carries observed weather - any past result from
that control is uninterpretable.

**CONFIRMED, verified in-session: the 2025 injury table is a post-season scrape
whose point-in-time filter admits everything.** `observed_at_basis` by season in
`data/players/raw/20260913T131524Z/injuries.parquet`: 2020-2024 all
`date_modified`; **2025 = 6,068 of 6,068 `week_proxy`**; 2026 = 182
`first_seen_capture`. The proxy (`players.py:243`) equals the cutoff
(`players.py:1713`) and the comparison is `.le` (`players.py:1307`), so every
proxied row passes. **Positive-control bound (agent measured):** on the 12
seasons with real `date_modified` (n=64,851), only 0.42% of rows were modified
after kickoff-24h and 0.04% after the pool lock. Real, small, quantified.
Second defect: the manifest records injury_timestamp_fallback drop but
`players.py:404-413` takes the `already_has_basis` branch, which never applies
it.

**Third:** player/injury/QB features use kickoff minus 24h, not the repo's own
`pool_decision_cutoff` = min(kickoff, Sun 16:00 ET) (`nfl_week.py:17-25`).
Measured: **329 rows (6.7%), all Monday games**, sit a median 4.25h past the
lock. `sharp_book_movement_features.py:55-60` already does it right.

**The guard that should catch all this is a column-name check that never runs
on the served path (verified in-session).** `modeling.py:45-47` intersects
feature names with OUTCOME_COLUMNS - it compares names, never values, so it
cannot see a leaking column called `temp`. And `validate_model_frame` is called
only from `backtest.py`, `modeling.py`, `outcomes.py` - **`margin.py` never
calls it**, and the margin model is the served model. AGENTS.md requires a
runtime assertion in the feature builder; there is none in `_build_features_pass`.

**Also named (not a leak, wrong basis):** the feature table's `spread_line` is
the CLOSING line - matches the archived close 68.6% (MAE 0.193) vs the Tuesday
opener 25.1% (MAE 0.983). So `active_ats_model.json`'s 52.29% is graded at the
close WITH the close as a feature.

**Cleared by agent 4 (21 items):** Elo writes before updating; graph/PageRank/
HITS ratings use state through W-1; team-state EWMA joins via searchsorted minus
1; `league_mean_*` consumed only at completed season boundaries;
`gap_sandwich_spot`'s shift(-1) reads the May-published schedule; prior-season
rates lagged; PBP traits cumsum-then-shift; sharp-book movement has the
strongest point-in-time filter in the repo; historical odds are genuine
/v4/historical snapshots with pre-2020 openers ABSENT rather than back-filled;
retroactive pool lines raise; availability rates carry a real runtime chronology
assertion; all walk-forward loops refit weekly; nested selection/test are
disjoint; **no full-sample scaler anywhere in the model path** (StandardScaler
is inside the sklearn Pipeline in both `margin.py:610-612` and
`modeling.py:54-56`); `pick_probability_fit.py:321-334` uses fold-specific
standardisers with in-sample reported beside LOSO.
Low-severity, not on a served path: `officials_flag_features.py:202` full-sample
qcut; `tv_attention_fade_overlay.py:115-121` full-sample z-scores.

### Probability construction and calibration (agent 5)

**The discrete lattice is computed for every game and then discarded
(VERIFIED in-session).** `key_line_pick_read.json` for week 1 reads
status inapplicable, served true, challenger smooth_gaussian_median_every_line.
`KEY_LINE_ATOMS = (3.0, 7.0)` with an EXACT match (`key_line_pick_read.py:24,51`),
and the pool always quotes half points - **16 of 16 lines**. So the lattice fires
on **zero served games, structurally, every week**, and the smooth Gaussian
challenger serves on 100% of them with no comparison ever run. AGENTS.md: ANY
served cover probability is computed against the discrete margin distribution;
a smooth pooled-residual read may run only as a challenger and must beat the
discrete read on the opener grade before it is served.
Agent measured both reads in the same artifact: mean gap 3.5 pts, max 8.0 pts,
**5 of 16 games take a different side** (CHI@CAR 0.5121 vs 0.4317).
`serve_discrete_three_way` overwrites only the excluding_push, push_probability
and home_loss_probability fields, never `home_cover_probability`.
Consequence: `push_probability == 0.0` on all 16 games (correct, all half
points) yet `home_cover_probability` and `home_cover_probability_excluding_push`
differ by up to 8.0 points - an arithmetic contradiction shipped on every row.

**Proper scoring rules (VERIFIED in-session,
`artifacts/margins/20260914T160810Z/summary.csv`, same 2,075 games):**

| | Brier | log loss | ECE | accuracy |
|---|---|---|---|---|
| constant 0.5 | 0.250000 | 0.693147 | - | - |
| market (no-vig) | **0.249637** | **0.692420** | **0.0132** | 0.5094 |
| market_residual (served) | 0.252560 | 0.698648 | 0.0353 | **0.5229** |

The argmax is fine - 52.29% beats the market's 50.94%, and that is what the pool
pays for. The MAGNITUDE is not: the served probability is beaten by a constant
0.5 on both proper scoring rules and its ECE is 2.7x the market's. Agent's Platt
fit on the served logit gives **slope 0.2146** - distance from 0.5 is about 4.7x
too wide; top decile predicted 0.6029 vs realised 0.5385, bottom decile 0.3587
vs 0.4760. Note the card's own `model_logit` coefficient is 0.2168, almost
exactly that slope - **the card's probability is shrunk correctly even though
`predictions.csv` is not.** Brier/log loss/ECE are computed in the same run
(`backtest.py:155-157`) and surfaced nowhere beside the 52.29% headline.

**One global sigma.** Fitted residual sample n=886, median 0.3387, std 12.646,
one number for every line. Conditional sd is about 8-9% larger at the key
numbers (13.79 at line 3, 13.83 at 7). The smooth read understates the push at 3
by about 2.2x (3.6-4.0% vs empirical 8.87%, n=665) and at 7 (2.1-2.3% vs 6.12%,
n=294). That bad push is baked into `home_cover_probability`, which picks the
side. The flip line and spread-explorer curve are Gaussian everywhere with two
lattice points spliced in at exactly 3 and 7, jumping 0.8-4.5 points in
inconsistent directions (`key_line_pick_read.py:280-300`).

**Cleared by agent 5:** the combination IS one fitted logistic with per-fold
coefficients and the out-of-season record stated beside model-only; all four off
switches are module constants set False (no env var in `src/nfl_ats` touches a
served pick); nothing multiplies per-game probabilities on the served path
(`pool.py`'s simulator is dead code - but if revived, `pool.py:161` hard-codes
public_lean 0.65 and `:232-234` draws 285 entrants as INDEPENDENT binomials,
which flattens the field's upper tail); the lattice fit itself is sound
(mass-preserving tilt, at least 200 prior games, trailing 5 seasons, target week
excluded); `home_side_offset` changed 0 sides on week 1.

## Audit complete: the three things that would change a pick

1. Serve the discrete lattice at every line, not only exact 3 and 7. It is
   already fitted and already computed for every game; it currently decides
   nothing while a labelled challenger serves 100% of the card.
2. Replace the greedy-argmax overlay subset with the members entering only
   through `composition_flag_sum` (already a fitted term), and quote the LOSO
   number (0.00 pts, P+ 0.493) rather than the in-sample +1.33.
3. Freeze 2026 forward as an outer test nothing tunes on. Every game in
   2020-2025 has been touched by selection; 2020 and 2021 have been spent 72
   times each.

Cheap fixes that change no pick: source `temp`/`wind` from the forecast archive;
call `validate_model_frame` from `fit_margin_model`; read the planning estimate
and the arrest-policy percentages from artifacts; report Brier/log loss/ECE
beside the 52.29%; print the decisive-game record beside every overlay headline.

## Methodology, so this audit can itself be audited

### Reproduce every number

`.\.tools\uv.exe run --no-sync python scripts\statistical_audit_20260915.py`

The script re-reads the pinned artifacts and reprints the nine measurement
blocks this lane relies on. Artifact timestamps are constants at the top of the
file; change them to re-run against a newer active model. It writes nothing and
asserts nothing - it prints, so a reader can disagree with a number rather than
trust a pass/fail.

Artifacts pinned at the time of the audit:
- `artifacts/pick_probability/20260914T232538Z` (coefficients, fold coefficients, bands)
- `artifacts/opener_evaluation/20260914T161406Z` (per_game.parquet, 1,537 rows)
- `artifacts/overlay_subset_composition/20260914T161418942653Z` (127 subsets, greedy path)
- `artifacts/margins/20260914T160810Z` (summary.csv: Brier, log loss, ECE)
- `artifacts/margin_predictions/2026-week-01-20260914T160928Z` (served week-1 card)
- `data/players/raw/20260913T131524Z/injuries.parquet`
- `data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet`

Commands run outside the script:
- `nfl-ats weak-signals pool --league nfl --effect-units accuracy_points`
- `grep -rn "validate_model_frame" src/ --include=*.py`
- `git show 68b4dc0`

### How the audit was carried out

Six areas, five delegated to subagents in parallel, one (reported numbers and
reader-facing framing) done in-session. Each subagent prompt carried the
interval-crossing-zero rule and the one-calibrated-probability rule verbatim,
because hooks do not reach subagents. Each was told to audit read-only, to label
every claim measured / read / inferred, and to return a NOT-a-problem list
alongside its findings so that cleared machinery is on the record too.

The six areas: model selection and search; leakage and chronology; in-sample
constants; inference machinery; probability construction and calibration;
reported numbers and reader-facing framing.

### What is verified and what is not

Subagent reports are claims until a command confirms them. Verified in-session
by re-running the command or re-reading the artifact:
- the pooled effect, SE and sign-test contradiction
- the 6,577 look count and both tails of probability_positive
- the effect-size-versus-noise table
- the reliability bands, slope and Brier comparison
- accuracy by spread bucket
- the 127-subset spread and the greedy path
- observed weather as a served feature (100.0% exact match, n=3,010)
- the 2025 injury `week_proxy` provenance (6,068 of 6,068)
- `validate_model_frame` call sites (margin.py absent)
- `key_line_pick_read.json` status inapplicable
- the Brier / log loss / ECE table from summary.csv
- the four contested week-1 served sides, reproduced from the fitted coefficients
- `hypothesis_frozen_before_scoring` as a hard-coded literal
- the headline denominator (1,537 rows, 1,503 non-null, 819 correct)

NOT independently verified, carried as reported:
- the headline decomposition table in subagent report 1 (52.83 -> 53.83 -> 54.49
  -> 55.82) and the pick-change counts inside it
- the refit that recovered the residual sample (n=886, median 0.3387, std 12.646)
  and the Platt slope 0.2146
- the weather swap experiment (10 of 1,537 picks, 5-5 vs 5-5)
- the positive-control bound on injury timestamps (0.42% / 0.04%)
- the 329 Monday rows past the pool lock
- the lattice-versus-Gaussian per-game gap table
- the SE/interval disagreement counts (52 rows >2x, 48 rows >4x)
- the 230 rows closable at the reliability ceiling

### Errors made during this audit, and corrected

1. **Hard flips claimed to decide the served card.** Withdrawn. The `1 - p` write
   at `four_overlay_composition.py:557` exists but `card_view.py:342-347`
   discards it. Settled by reproducing four contested served sides from the
   fitted coefficients; the hard flip would serve the opposite side on all four.
   One subagent repeated the same error independently.
2. **An opener-graded number compared to a close-graded market number.** In
   conversation the card's 55.8% (opener, 1,503 games) was set against the
   market's 50.9% (close, 2,075 games). Invalid, and the same defect this audit
   flags elsewhere. Against the spread the baseline is a coin flip by
   construction, so the honest framing is 52.8% against 50%, with a +/-2.5 point
   margin of error on 1,537 games.

### Limits of this audit

- Nothing was executed against a held-out sample, because none exists. Every
  number is measured on games that selection has already touched, so each is an
  upper bound on what survives forward.
- The subset-selection LOSO result (0.0000 points, P+ 0.4930) is read from
  `docs/edge_audit_redteam.md`, not re-derived here.
- The decomposition attributing roughly 3 of the 5.8 points above a coin flip to
  selected layers is arithmetic on the reported layer table, not an independent
  refit.
- No code path was changed, so no claim here is confirmed by a behaviour change.
