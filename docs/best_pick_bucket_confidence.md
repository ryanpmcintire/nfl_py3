# Best Pick bucket confidence (lane U)

**Question.** The pool scores one Best Pick per week separately. The served
nominator (`nfl_ats.best_pick_nomination.nominate_v2`) ranks by an alpha=2000
`market_residual` probability's distance from 0.5, inside that week's
below-median cross-book opener `spread_std` pool. That ranking score is
**bucket-blind**: the model's stated confidence is 55.3-56.6% in every spread
bucket (read: `docs/spread_hole_diagnosis.md:166-173`) while its realised
served accuracy is 56.16% on 0-6.5, 51.35% at exactly 7, 51.55% on 7.5-10 and
47.33% on 10.5+ (read: `docs/spread_hole_diagnosis.md:25-28`). A Best Pick
chosen by raw probability distance therefore over-nominates the games where
the model is worst. **Does a bucket-reliability-weighted nomination beat the
served nominator on Best-Pick accuracy?**

This lane changes a **RANKING**, never a side. Every game's forced pick stays
exactly as published; only which single game carries the ★ can move. No
threshold flip is proposed on any side.

## Predeclaration (written before any number in "Results" was computed)

### Binding closing-grounds taxonomy, verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (b) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it, report `probability_positive`, never
"contains zero". Within-week correlation is ZERO; week-blocked bootstrap.
Decide on expected value: forced picks, the nominator with the higher expected
Best-Pick accuracy is served; 0.90-style thresholds govern only what docs may
claim.

### Population

The frozen 2020-2025 opener archive, no new fitting:

- **Nomination inputs** — `artifacts/ridge_alpha_promotion/20260818T221459Z/`
  (`opener_paired.parquet`, `opener_baseline.parquet`): 1,537 games, 107 REG
  weeks, seasons 2020-2025. Supplies `candidate_dist` (the alpha=2000
  `market_residual` probability's distance from 0.5 — v2's own ranking score),
  the frozen Tuesday opener line `tue_open_home_spread`, and the archive arm's
  played side and settlement.
- **Tuesday cross-book dispersion** —
  `artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet`'s
  `spread_std`, fed through production's own
  `nfl_ats.best_pick_nomination.dispersion_pool_from_frame` per week, with the
  eval-script pool cross-checked row-for-row against the production pool
  (the check `scripts/best_pick_composed_rule_eval.py::build_archive_frame`
  already performs; the run aborts on any disagreement).
- **Served settlement** —
  `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`, the ACTIVE
  model `c657058903f3232b`'s own opener stream **including the S3 home-side
  offset**, same 1,537 games / 107 weeks.

### Two grades, both reported; PRIMARY named in advance

The archive that carries v2's ranking input predates the S3 home-side offset
promoted 2026-09-07, so the archive's played side and the card's served side
are not the same stream on big spreads. Both are graded, and each grade builds
its own reliability history from its own stream, so no arm ever mixes them.

- **PRIMARY — SERVED grade.** `correct_at_open` / `pick_home_at_open` from
  `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`. This is the
  side the card actually submits today and the stream the bucket table that
  motivates this lane was computed on.
- **SECONDARY — ARCHIVE grade.** `baseline_correct_open` /
  `baseline_pick_home` from `opener_paired.parquet`. This is the stream
  `scripts/best_pick_deadline_renomination_eval.py` and
  `scripts/best_pick_composed_rule_eval.py` score on, and it is the grade B0's
  replay is checked against.

### Spread buckets

`|tue_open_home_spread|` in **0-6.5**, **exactly 7**, **7.5-10**, **10.5+** —
the four buckets `docs/spread_hole_diagnosis.md` reports realised accuracy on,
used unchanged so no boundary is mined in this lane.

Pick side, for B3: **favourite** if the served/played pick is on the side the
line favours, **underdog** if not, **pickem** when the opener line is 0.

### Arms (one nomination per week, all graded at the frozen Tuesday opener)

- **B0 — v2 as served (incumbent).** `select_nominee` on the production
  below-median-`spread_std` pool, exactly as played. Replay check: its weekly
  nominees must equal `a0_tuesday_game_id` in
  `artifacts/best_pick_deadline_renomination/20260909T214500Z/weekly.csv` on
  every week that lane covered, which is what reproduces its 33/51.
- **B1 — bucket-reliability weighted.** Same eligibility pool. For each
  candidate, `bucket_reliability` = the realised accuracy of the graded side
  in that candidate's own spread bucket over **all prior completed weeks** in
  this archive (expanding, strictly earlier `(season, week)`), shrunk toward
  0.5 with 20 pseudo-observations — `(prior wins + 20 * 0.5) / (prior n + 20)`,
  the count 20 and the form taken from MOD-18's C3 rule
  (read: `docs/spread_regime_program.md:61-71`; C3 shrinks toward the stated
  probability, this lane shrinks toward 0.5 as specified for this lane).
  Score = `(bucket_reliability - 0.5) + (p - 0.5)`, where `(p - 0.5)` is
  `candidate_dist`, v2's own ranking score. Nominate the max within v2's own
  eligibility pool, ties broken by production's own rule (lower `spread_std`,
  then ascending `game_id`) by feeding the score to `select_nominee`. Weeks
  with no prior history give every candidate reliability 0.5, so B1 reduces to
  B0 there by construction.
- **B2 — small spreads only.** v2 restricted to pool games with
  `|tue_open_home_spread| <= 6.5`; falls back to B0's nominee when no eligible
  game exists that week.
- **B3 — pick-side reliability.** B1 with the reliability taken from the
  (bucket, pick side) cell — favourite / underdog / pickem within the bucket —
  instead of the bucket alone, same 20-pseudo-observation shrinkage toward 0.5.
- **Positive control — perfect-foresight nomination within the pool.** Among
  the same eligible pool, any game whose graded pick is correct
  (`select_nominee` order among the correct ones); else B0's nominee. This is
  the instrument's ceiling on this exact population.

### Metric

Best-Pick correctness per week (one pick per week, 0/1) at the frozen Tuesday
opener line, over 2020-2025 weeks with a v2 nominee, **paired against B0**,
week-blocked bootstrap (`nfl_ats.clv.week_blocked_bootstrap`, blocks
`(season, week)`), **20,000 samples, seed 20260821**, in accuracy points,
reported as `probability_positive` plus the interval. Raw win counts and the
number of weeks the nominee differs are reported alongside because n is ~107
weeks and one pick a week makes the interval necessarily wide. A push on the
nominee is NaN and drops that week from the paired cell.

### Declared limitations, stated before the run

1. **Look reuse.** This is the fifth reuse of the ~107-week opener population
   for the Best Pick family (ridge_alpha promotion look, odds-microstructure
   battery, the 2026-08-18 ranker screen that SELECTED v2, the 2026-08-19 v3
   audit, the 2026-09-09 composed-rule scoring), and the bucket accuracies
   that motivate B1/B3 were themselves measured on this same archive. Every
   number here carries that compounding discount; the reliability inputs are
   walk-forward, but the decision to look at buckets at all was made after
   seeing the bucket table.
2. **B1/B3 are nearly self-referential by construction.** The reliability is
   an expanding mean of the same graded stream the arm is scored on, one week
   ahead. It cannot leak an outcome (only strictly earlier weeks enter), but it
   is not independent evidence about buckets — it is a re-weighting of a
   pattern already established on this window.
3. **The archive's played side is not today's served side.** The SECONDARY
   grade is the alpha=10 arm without the S3 home-side offset; the PRIMARY grade
   is the served stream. Where the two disagree, prefer the PRIMARY.
4. **One pick a week is a ~107-observation instrument.** A 2-3 accuracy-point
   nomination edge is far below what it can resolve, which is exactly why the
   positive control is run: to state what this instrument CAN see, and
   therefore what it does not bound.

## Results

All measured 2026-09-09; commands and artifact path at the bottom of this
document. Every arm's nominee comes from production's own
`nfl_ats.best_pick_nomination` selector; nothing was refit.

Artifact: `artifacts/best_pick_bucket_confidence/20260909T233823Z/`
(`summary.json`, `weekly_served.csv`, `weekly_archive.csv`,
`record_commands.json`).

**Decision: B2 — never star a game with a spread bigger than 6.5 — has the
higher expected Best-Pick accuracy.** It scores **58.82% (60/102)** against the
served nominator's **56.86% (58/102)** on the same paired weeks: **+1.96
accuracy points, `probability_positive` 0.688**. On a forced weekly nomination
that is a 69/31 bet, and per `AGENTS.md` a promotion bar is not a decision bar,
so the expected-value call is B2. **Served the same evening** (see docs/best_pick_ranker.md);
and on the current Week 1 card it would change nothing (below). B1 is a dead
heat and B3 leans against the incumbent.

Population: 1,537 games, 107 REG weeks, 2020-2025, 34 pushes on each grade; 101
to 103 weeks pair depending on the arm. The dispersion pool fell back to the
full week in 10 of 107 weeks (8 missing data, 2 empty filter), the production
pool agreed with the eval pool on every game, and the served and archive picks
disagree on 60 of 1,537 games.

### B0 replay

B0's nominee equals `a0_tuesday_game_id` in **all 54 weeks**
`artifacts/best_pick_deadline_renomination/20260909T214500Z/weekly.csv` covers
(`nominee_reproduces_exactly: true`), where that lane scores A0 **33/51** on the
served late-week grade and 32/51 at the frozen Tuesday side. Independently, B0's
2020-2025 hit rate is **56.31% (58/103)**, matching `docs/best_pick_composed_rule.md`'s
own "v2 as played" line (56.31%, 58/103) exactly.

### Arms, paired against B0 (PRIMARY, served grade)

| Arm | Weeks won | Accuracy | Effect (acc. pts) | 95% week-blocked | `probability_positive` | Weeks nominee differs |
|---|---:|---:|---:|---|---:|---:|
| **B0 v2 as served (incumbent)** | 58/103 | 56.31% | — | — | — | — |
| B1 bucket-reliability weighted | 58/101 | 57.43% | **0.00** | [−5.94, +5.94] | **0.500** | 19 |
| **B2 spreads 6.5 or less only** | **60/102** | **58.82%** | **+1.96** | [−5.88, +9.80] | **0.688** | 26 |
| B3 pick-side reliability | 53/102 | 51.96% | **−4.90** | [−15.69, +4.90] | **0.179** | 52 |
| Positive control (perfect foresight) | 101/103 | 98.06% | **+41.75** | [+32.04, +51.46] | **1.000** | 43 |

SECONDARY (archive grade, the alpha=10 arm without the S3 home-side offset),
same weeks:

| Arm | Weeks won | Accuracy | Effect (acc. pts) | 95% week-blocked | `probability_positive` |
|---|---:|---:|---:|---|---:|
| **B0 v2 as served** | 58/103 | 56.31% | — | — | — |
| B1 bucket-reliability weighted | 57/101 | 56.44% | −0.99 | [−6.93, +4.95] | 0.383 |
| **B2 spreads 6.5 or less only** | 60/102 | 58.82% | **+1.96** | [−5.88, +9.80] | **0.688** |
| B3 pick-side reliability | 53/102 | 51.96% | −4.90 | [−15.69, +4.90] | 0.179 |
| Positive control | 101/103 | 98.06% | +41.75 | [+32.04, +51.46] | 1.000 |

B2, B3 and the control score identically on both grades because their nominees
sit almost entirely in the 0-6.5 bucket, where the S3 home-side offset (spreads
7 and up only) cannot change a side. Only B1's reliability history differs
between the two streams, and that is what moves it from 0.00 to −0.99.

### The defect on the Best Pick population itself

The lane's premise reproduces directly on the 107 nominated games, not just on
the 1,537-game card:

| B0's own nominee's bucket | weeks | Best Pick right |
|---|---:|---:|
| 0-6.5 | 76 | **59.21%** (45) |
| exactly 7 | 9 | 44.44% (4) |
| 7.5-10 | 12 | 58.33% (7) |
| 10.5+ | 6 | 33.33% (2) |
| **7 and up, pooled** | **27** | **48.15%** (13) |

The served nominator sends **27 of 107 Best Picks (25.2%)** into buckets where
the model is measurably weakest, and hits 48.15% on them against 59.21% on its
small-spread nominations. B2 removes exactly that quarter of the nominations and
never once has to fall back: every one of the 107 weeks had at least one
eligible game at 6.5 or less (mean pool 7.59 games, of which 5.88 small).

### Why B1 and B3 do not help

The walk-forward bucket reliabilities entering 2026 (served grade, expanding
over all 1,503 settled archive games) are 0.5436 on 0-6.5 (n=1,104), 0.5222 at
exactly 7 (n=70), **0.4679 on 7.5-10** (n=198) and 0.5033 on 10.5+ (n=131). The
whole span between the best and worst bucket is **0.076**, which is the same
order as the spread of `candidate_dist` within a week, so B1 does re-rank — 19
weeks of 101 — but it re-ranks softly and the changes cancel: 10 weeks change
outcome, 5 each way. B1's own nomination mix moves from 80/9/12/6 across the
four buckets to 96/7/3/1, so it is directionally doing what B2 does, just
without committing.

B3 splits the same history into (bucket, favourite/underdog/pickem) cells, which
are 12 to 562 games each. The small cells are noisy — the 12 pick'em games sit
at a raw 25.0%, shrunk to 0.4062 — and B3 re-ranks **52 of 102 weeks**, half the
season, on that noise. It loses 4.90 points, `probability_positive` 0.179. That
is a lean against, not a resolution.

### Season stability of the decision arm

| season | B0 | B2 |
|---|---:|---:|
| 2020 | 58.8% | 52.9% |
| 2021 | 50.0% | 66.7% |
| 2022 | 41.2% | 52.9% |
| 2023 | 47.1% | 47.1% |
| 2024 | 70.6% | 75.0% |
| 2025 | 70.6% | 58.8% |

B2 ahead in three seasons, level in one, behind in two. Seventeen weeks a season
is not a season-level instrument; this is reported as texture, not confirmation.

### Nothing here closes, and why

All four cells are recorded `unresolved_below_power`
(`registry/weak_signals.json`, family `best_pick_bucket_confidence`, names
`best_pick_bucket_confidence_<arm>_2020_2025`; argv in the artifact's
`record_commands.json`). Neither terminal ground is available:

- **`wrong_sign_resolved`** — no interval sits wholly on the wrong side of zero.
  B3 comes closest at [−15.69, +4.90] and still crosses.
- **`bounded_by_control`** — the positive control resolves cleanly at +41.75
  points, `probability_positive` 1.000, so the instrument demonstrably sees a
  ~42-point renomination effect on 103 weeks. It was **not** shown able to see a
  2-point one, so it bounds nothing at the scale B1 and B2 live at. One
  nomination a week is ~103 observations; at that n the interval is ±8 points
  wide and this instrument cannot resolve a small nomination edge either way.

### What Week 1 2026 would be

Measured on the live card (`artifacts/margin_predictions/2026-week-01-20260909T220047Z`,
active model `c657058903f3232b`, refit through production's own `nominate_v2`):
**every arm nominates the same game — `2026_01_MIA_LV`, MIA +3.5, the ★ already
on the card.** The dispersion pool holds 8 games this week and **all 8 are at
6.5 or less**, so B2's restriction excludes nothing, and with a single bucket in
play B1's and B3's reliability terms are constants that cannot re-rank anything.
Serving B2 would change **no pick and no star** in Week 1.

### What serving B2 took (served 2026-09-09)

1. **The rule already exists.**
   `nfl_ats.best_pick_big_spread_challenger.apply_big_spread_eligibility`
   excludes candidates whose absolute decision spread is **at least**
   `threshold` and falls back to the unmodified v2 pool when every eligible game
   is excluded (read: `src/nfl_ats/best_pick_big_spread_challenger.py:10-16`,
   signature at `:69-74`). Since lines are half-points, calling it with
   `threshold=7.0` is B2 exactly. Serving it means a new served constant and a
   `nominate_v2`-shaped wrapper composing the two, not a new algorithm.
2. **The served call site** is `nfl_ats.publishing._publication_context`, which passes
   `nominate_v2_fn=nominate_v2` into `nfl_ats.card_view.resolve_card_view` and
   stars `view.nomination.active_game_id` (read:
   `src/nfl_ats/publishing.py:153-162`). That is the one place the served
   nominator changes.
3. **The ledgers.** The played ★ is recorded as the active model's own
   `is_best_pick` flag on `artifacts/clv_ledger/decisions.parquet`
   (`nfl_ats.clv.record_paper_decisions`); the Tuesday nomination ledger rows
   are written by
   `nfl_ats.best_pick_nomination.record_nomination_challenger_decisions` under
   `best_pick_nomination_v2` and by
   `best_pick_big_spread_challenger.record_...` under
   `best_pick_big_spread_eligibility` (currently threshold 10.0). Serving B2
   means the ★ on the primary ledger moves, and the side-ledger challenger has
   to be re-registered if its threshold moves with it.
4. **The card's star text** comes from `_published_card` plus `_best_pick_note`
   in `publishing.py`; the disclosure sentence would need to name the new
   eligibility rule in reader words.

### Stated against the decision

- **Fifth reuse.** The bucket accuracies that motivated every arm here were
  measured on this same 107-week archive, and the archive has been looked at
  four times before for this family. B2's 6.5 boundary is inherited from
  `docs/spread_hole_diagnosis.md`, not mined in this lane, but that diagnosis
  was itself measured on these weeks.
- **The interval is wide and B2 changes 26 weeks.** +1.96 points is 2 extra
  correct weeks out of 102, and the outcome differs in 16 of the 26 weeks the
  nominee moves — 9 to B2, 7 to B0.
- **The existing 10+ challenger is the weaker version of the same idea.**
  `docs/best_pick_composed_rule.md` measured it at +0.97 points,
  `probability_positive` 0.592, changing 8 of 103 weeks. B2 is the same
  mechanism pushed to the boundary the diagnosis actually locates, and it lands
  further from zero on more weeks — consistent, not independent.

## Commands run

```
uv run --no-sync python scripts/best_pick_bucket_confidence_eval.py \
  --repo F:/Repos/nfl_py3 \
  --out-dir F:/Repos/nfl_py3/artifacts/best_pick_bucket_confidence/20260909T233823Z

uv run --no-sync nfl-ats weak-signals record ...   # 4 cells, family best_pick_bucket_confidence
                                                   # exact argv in the artifact's record_commands.json
```

