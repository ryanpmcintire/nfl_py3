# Forward pooling declaration, 2026-11-03

Written before any 2026 forward outcome is read for these rows. Covers LEAD-65, MOD-07, MKT-08, LEAD-75, SKY-01. Governed by the Research invariants in AGENTS.md: pooling only across commensurable inputs, family declared before signs are seen, chronological evaluation, one fitted calibrated probability selects every side, zero crossing closes nothing.

Labels: **measured** (from the named lane, 2026-11-03), **read** (file), **inferred** (arithmetic from those), **declared** (a choice fixed here).

## Shared design (all five rows)

- **Family.** Each row is one family, `fwd_pool_<row>_v1`. Seasons 2026, 2027, 2028, ... append to it. Season is the block.
- **Unit.** Accuracy points of the whole card at the opener grade. The paired outcome per differing game is +1 or -1, so the season contrast is 100 * (wins_variant - wins_frozen) / N games. LEAD-75 and SKY-01 use the units stated in their sections. Contrasts are never pooled across rows.
- **Estimator (declared).** Pooled contrast = total paired difference over total eligible games across all seasons (fixed effect, no per-season weights). Interval: 2,000-draw bootstrap, seed 20261103, resampling weeks within season while fewer than 5 seasons exist, and resampling whole seasons (season-block bootstrap) once 5 or more exist. Report `probability_positive`, per-season contrasts, and the decisive record (games where the arms differ) before the headline. Each annual append is recorded with `nfl-ats weak-signals record` (the orchestrator runs it).
- **Power (declared).** Two-sided alpha 0.05, 80% power, so the effect must equal 2.80 standard errors. Paired SE per season = 100 * sqrt(d) / N, with d differing games per season and N = 272 scheduled regular-season games. Pooled SE = per-season SE / sqrt(k) for k seasons (inferred; assumes constant d and zero within-week correlation as AGENTS.md sets).
- **Decision rule at the power season, or at the end of 2030 if power is not reached (declared review date).**
  1. Serve: only by adding the frozen arm's quantity as a fitted term in the single calibrated probability, with leave-one-season-out coefficients, out-of-sample Brier and log loss against the market and model-only baselines, and per-fold coefficients. Requires a pooled point estimate positive in the declared direction and a positive out-of-sample log-loss gain. A probability just above 0.5 is not sufficient, and nothing flips a side alone.
  2. Keep tracking: every other outcome, including an interval across zero.
  3. Close: only on an admissible ground, meaning the whole pooled interval on the wrong side of zero (`wrong_sign_resolved`) or a positive control proven able to detect the declared effect. "Not powered" is `unresolved_below_power` and never closes.

## Power table (inferred from measured rates; no declared effect sizes)

Two effects per row. Effect (a) is the row's measured estimate with its interval. Effect (b) is the smallest effect that can change the card: the smallest nonzero season contrast is one net flipped pick, 100 / 272 = 0.368 accuracy points (**inferred** from N = 272), because a positive pooled estimate is the first serving requirement and every game moves it by that quantum. Seasons k = (2.8 x SE per season / effect)^2, rounded up; season reached = 2025 + k. SE per season = 100 x sqrt(d) / 272 with d the measured differing games per season. Power is undefined for (a) where the measured estimate is zero or wrong-signed.

| Row | d per season (source) | SE per season | (a) measured effect [interval], source | (a) seasons, reached | (b) effect | (b) seasons, reached | Detectable by end of 2030 (5 seasons) |
|---|---|---|---|---|---|---|---|
| LEAD-65 | 10 (58 of 1,503 over 2020-2025, registry cell) | 1.16 pts | +0.53 [-0.20, +1.27], P+ 0.874; `registry/weak_signals/pbp08_protection_mismatch_early_window_v1.json` cell `..._2020_2025` | 38, 2063 (upper bound 1.27: 7, 2032) | 0.368 pts | 79, 2104 | 1.46 pts |
| MOD-07 | 100 (45 differing in Weeks 1-8, `docs/lanes/done/mod07-prospective-status-audit.md` line 36) | 3.68 pts | historical +0.33 (52.83 v 52.50 on 1,537 games, same lane line 35; interval not stored in the lane or registry); forward Weeks 1-8 is active 24, stack 21, wrong-signed, so undefined for the forward arm | 974, 2999 on the historical point only | 0.368 pts | 783, 2808 | 4.6 pts |
| MKT-08 | 97 (43 changed-side games in Weeks 1-8, `docs/lanes/news-trigger-refresh.md` line 235) | 3.62 pts | forward interim: refreshed 25 wins v pre-refresh 21 on 43, net +4, scaled to 97 games a season is +3.3 pts (**inferred**; no interval stored, one interim look). The retrospective term (-1.22 [-3.18, +0.49], P+ 0.094) is a different arm and wrong-signed, so undefined there | 10, 2035 on the interim point | 0.368 pts | 760, 2785 | 4.5 pts |
| LEAD-75 | about 119 outdoor MEX pairs (7 settled in Week 7 x 17 weeks, `docs/lanes/lead75.md` line 27; upper bound, Weeks 4-6 had none) | slope SE 180 at 7 pairs (registry `lead75_paired_wind.json`), 180 x sqrt(7/n) | slope +46.3 [-497, 184], P+ 0.64, declared sign negative, so wrong-signed and undefined | undefined (magnitude 46.3 alone: 7 seasons, 2032) | not derivable yet: needs the measured SD of the term and the gap between the model margin and the line per game, neither stored | owed at the next look | 54.7 pts per unit |
| SKY-01 | about 8 decisive games (3% of 272, `docs/lanes/sky01-news-availability.md` line 10) | binomial, SD 0.5 | decisive 8-8 on 16 (win rate 0.50); Brier +0.000063 [-0.00149, +0.00142], P+ 0.55; registry family `sky01_news_availability`. Zero, so undefined | undefined | one net decisive win a season: win rate 0.5625 | about 502 decisive games, 62 seasons, 2087 | win rate 0.72 on 40 decisive games |

Reading: LEAD-65, MOD-07 and MKT-08 cannot reach power at the single-pick scale inside a usable horizon through pooling alone, and MKT-08 reaches it only if the large interim forward estimate holds (**inferred**). That is a statement about this evaluator's resolution, not a negative result. The 2030 review reports the smallest detectable effect, the pooled point estimate and `probability_positive`; every row stays `unresolved_below_power` unless an admissible closing ground appears. The LEAD-75 pair count and SE come from the interim look (SE 180 at 7 pairs); the earlier lane figure of about 160 pairs for SE 20 does not follow from that SE (**inferred**: SE 20 needs about 570 pairs) and is withdrawn. The LEAD-75 (b) effect and the SKY-01 interval-based SE are not yet measured and are remeasured at the first annual append.

## LEAD-65: early-window protection tilt

- **Frozen arm (read).** `src/nfl_ats/pbp08_protection_mismatch_early_window_overlay.py` at commit 93af6902, sha256 prefix 3f84dee1c444d37e. Flips are gated to week 4 and earlier via `PREDECLARED_SPLIT_BOUNDARY_WEEK`; it imports the parent flag build and flip logic and has no fitted constants. Registry entry `pbp08_protection_mismatch_early_window_v1`, ACTIVE_PROSPECTIVE (`docs/lanes/done/lead65-protection-window-split.md`).
- **Contrast.** Variant side minus served unconditional-tilt side, accuracy points of the card, opener grade, both sides read from the same `publish-predictions --record-decisions` run.
- **Eligible games.** Every regular-season game the overlay recorded with both sides present; Thursday post-kickoff skips are excluded and counted. Historical 2020-2025: 58 differing of 1,503 (about 10 a season), +0.53 [-0.20, +1.27], P+ 0.874 (**measured**, not pooled with forward).
- **Append rule.** No refit: the variant is a rule with no coefficients. Each season appends its ledger rows. The served arm is production and is not frozen; the contrast uses sides recorded in the same run, so a served refit does not break pairing. A change to the served flag sum or to the overlay file hash triggers the invalidation section.

## MOD-07: stacking challenger

- **Frozen arm (read).** Challenger fingerprint `bc77638d47e2748c`; ledger `artifacts/prospective/challenger_decisions.parquet`, SHA-256 `fb15413901b262b02dca1f4f6ffe48acda7f6e2da9dbb860b5386b8603d62d98` at 48 games (Weeks 1-3). Retuning is inadmissible (lane).
- **Contrast.** Stack pick minus active-model pick, accuracy points, opener grade, over games where they differ. Forward Weeks 1-8: 45 differing, active 24, stack 21 (**measured**).
- **Eligible games.** All regular-season games with that fingerprint on the ledger, one row per game.
- **Append rule.** No refit. Any retrain or new feature set takes a new fingerprint and a new family.

## MKT-08: news-triggered refresh timing

- **Frozen arm (read).** The refresh policy `registry/weak_signals/refresh_timing_policy.json` (sha256 prefix e99d25644e5f758e) and the dispatch module `src/nfl_ats/best_pick_refresh_prospective.py` with its activation watermark, at commit ef986916.
- **Contrast.** Refreshed side minus pre-refresh side, accuracy points of the card, opener grade, over changed-side games. Forward Weeks 1-8: 43 changed-side games, refreshed 25-18 versus pre-refresh 21-22 (**measured**).
- **Eligible games.** Games with a successful child refresh receipt and a changed side. Evidence rows, dry runs and failed refreshes are excluded and counted.
- **Append rule.** No refit. Trigger sources, watermark and grade are fixed. The retrospective Friday-resolution term (-1.22 pts, P+ 0.094) is a separate historical family and is not pooled with this forward arm.

## LEAD-75: paired wind-forecast innovation

- **Frozen arm (read).** Collector `scripts/capture_paired_wind_forecast.py` at commit 1e65cc23, sha256 prefix 3710fca23ded05c1, and the protocol in `docs/lanes/lead75.md` (Tuesday 12:05-12:20 ET, deadline final 90 minutes, identical product/station/valid-time key, MEX and GFS kept separate, no catch-up).
- **Contrast and unit.** Slope of home ATS margin, points per unit, on (MEX revision) x (home-minus-away pass-rate difference); declared sign negative. Interim look 1: slope +46.3, bootstrap 95% [-497, 184], P+ 0.64, 7 settled outdoor MEX pairs (**measured**).
- **Eligible games.** Outdoor games with a mapped station, a valid Tuesday receipt and a valid deadline receipt with an identical key. Weeks 4-6 had no usable Tuesday receipts and are not eligible; rows not attempted after a source error are excluded and counted.
- **Append rule.** The slope is the statistic itself, re-estimated on the pooled pairs with no coefficients carried forward. The pass-rate input uses only games before the prediction timestamp, with the definition at the commit above. Serving requires the term to enter the four-term probability under leave-one-season-out.

## SKY-01: earliest-source news availability

- **Frozen arm (read).** `scripts/sky01_news_availability.py` (commit 470093ba, sha256 prefix 29399d3b34efbbfb) and `scripts/sky01_drift_weighted.py` (commit 7164e885): the snap-share-weighted term with unmatched players at weight 0. The repaired weighting (`sky01_snap_repair.py`, OOS Brier -0.000097) adds no signal and is not the frozen arm.
- **Contrast.** Four-term opener probability plus the frozen term versus the four-term probability alone: out-of-sample Brier and log-loss difference, decisive record first. Historical: Brier +0.000063 [-0.00149, +0.00142], P+ 0.55, 8-8 on 16 (**measured**).
- **Eligible games.** 2026 games from Week 9 onward (four-term opener probabilities exist from then; first score after the Nov 10 games), then all regular-season games from 2027.
- **Refit protocol (declared).** This design refits each offseason. For season s, coefficients are fit on seasons 2024 through s-1 only (expanding window), applied unchanged to season s and never revised mid-season. Predictor definition, source list and snap-share weighting are frozen. Season 2026 uses the 2024-2025 fit.

## What invalidates pooling, and handling

Any one of these ends the family's continuation:

1. A rules change in the pool that alters what picks are graded on (scoring, lock time, opener definition, number of games).
2. A replacement of the active model or of the served four-term probability structure, because the paired arm is then a different comparison.
3. A change to an input definition: a different feed, station map, injury or coordinator source, or a change to a file hash or fingerprint listed above.
4. A schedule or ruleset change that alters the game count or key-number lattice materially (for example, an extra regular-season game).

Handling: the old family is closed to appends and its last pooled result stays recorded as is. The new arm starts a new family (`..._v2`) with its own frozen record, power table and declaration. Seasons are never spliced across the break, and old and new numbers are not pooled. Source outages and missed windows are not invalidation; they are counted exclusions inside the family.
