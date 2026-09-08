# Best Pick composed nomination rule, scored as played (POL-09)

Written 2026-09-07 (lane U). Family `pol09_best_pick_composed_v1`. Script
`scripts/best_pick_composed_rule_eval.py`; artifact under
`artifacts/best_pick_composed_rule/<ts>/`; tests
`tests/test_best_pick_composed_rule_eval.py`.

## Why this exists

The pool pays one Best Pick per week, so the top-1 nomination is a real
payout slot. The production nomination rule (v2, live since 2026-08-18)
composes chooser 6's below-median cross-book-dispersion filter with a
SEPARATE chooser's dispersion tie-break, applied inside the filtered pool.
`ROADMAP.md` row POL-09 states that this composed rule "was never itself
scored" — only its two halves were. The 2026-08-19 v3 audit
(`docs/best_pick_ranker.md` § "v3 audit") reproduced the composition on the
107-week archive, but scored it on the **candidate arm's** pick side
(`candidate_correct_open`, the alpha=2000 model's own pick), which is NOT
what the pool grades: sides never change, so the nominee's Best Pick settles
on the **active model's** pick (`baseline_correct_open`). The two arms
disagree on the side in 255 of 1,537 archive games [measured 2026-09-07,
`opener_paired.parquet`], so the two readings can differ. This document
scores the rule exactly as it is played.

## Frozen predeclaration (written before any score was computed)

### Population

The same 1,537-game / 107-week paired opener archive every `best_pick_*`
registry entry uses: `artifacts/ridge_alpha_promotion/20260818T221459Z/`
(`opener_paired.parquet`, `opener_baseline.parquet`; baseline arm =
`weak_stack` / ridge alpha 10 / `market_residual`, the active production
recipe; candidate arm = alpha 2000) and
`artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet`
(cross-book opener `spread_std`). Seasons 2020–2025, regular season, both a
Tuesday opener and a close on file. Pushes (34 games) carry no correctness;
a week whose nominee pushed is dropped from that pair only.

**Multiplicity, stated plainly: this is the FOURTH reuse of this ~107-week
opener population for the Best Pick family** (ridge_alpha promotion look,
odds-microstructure battery, the 2026-08-18 ranker screen, the 2026-08-19 v3
audit; this scoring is a fifth look at the same weeks). Every number below
carries that compounding look-reuse discount. **No rotation window is
assigned or spent**: this is a scoring of an already-played rule, like the
interim-HC screen, not a candidate look.

### Reproduction, not re-implementation

- The v2 nominee for every archive week is produced by calling
  `nfl_ats.best_pick_nomination.select_nominee` on that week's eligible pool.
  The pool is built by `nfl_ats.best_pick_nomination.dispersion_pool_from_frame`
  — the SAME function `week_dispersion_pool` (production) delegates to,
  factored out additively this session so the archive can be fed to it — and
  cross-checked row-for-row against the predeclared 2026-08-18 eval script's
  own `build_dispersion_pool`. Any disagreement aborts the run.
- U2's discount is `nfl_ats.best_pick_big_spread_challenger.apply_big_spread_eligibility`
  called on a `NominationV2Result` built from the same week table, with
  `spread_line` = the archived Tuesday opener.
- v1 is `nfl_ats.best_pick.select_best_pick` (frozen `sweep_robustness`)
  on a line sweep produced by refitting the active recipe walk-forward per
  week (`fit_margin_model`, `weak_stack`, alpha 10, training strictly before
  the week's first kickoff — the archive's own loop, mirrored) and sweeping
  at the archived opener with `probability_method="gaussian_median"` (the
  active manifest's mapping). A reproduction check reports the max absolute
  difference between the refit's opener residual and the archive's
  `residual_at_open`; the pick SIDE scored is always the archive's.
- The raw model's top-|residual| pick is the 2026-08-18 eval script's
  chooser 1 (`status_quo_residual`), unchanged.
- The alphabetical fallback for the tie-break audit is
  `nfl_ats.best_pick_nomination.select_nominee_v3` on the same pool.

### Cells (exactly these; no other variants, no tuning, no new chooser)

Scoring: opener grade, top-1 hit rate (nominee's active-model pick covered),
paired within week, week-blocked bootstrap of the mean paired difference,
20,000 draws, seed 20260817, within-week correlation ZERO (owner mandate).
Effects in accuracy points (percentage points), positive favours the
first-named rule.

- **U1 `v2_as_played`**: hit rate; paired delta vs v1 (`sweep_robustness`)
  and vs top-|residual|; number of weeks the composed nominee differs from
  each part alone (chooser 6 = filter with game_id tie-break, i.e. v3;
  chooser 8 = dispersion tie-break on the unfiltered week).
- **U2 `v2_plus_big_spread_discount`**: U1 with the 10+ opener-spread
  eligibility exclusion (fallback to the unmodified v2 pool when every
  eligible game is 10+); same two comparisons, plus its marginal vs U1.
- **Tie-break audit (descriptive)**: how many weeks tie at the top of the
  filtered pool; how many of those the dispersion tie-break actually
  decides; hit rate of v2's nominee in exactly those weeks vs the
  alphabetical (`select_nominee_v3`) fallback the 2026-08-18 audit found had
  been deciding under v1. The finding that alphabetical tie-breaks had been
  deciding is not re-litigated here.
- **Reconciliation row (descriptive, not a cell)**: U1 re-scored on the
  candidate arm's side, to tie out to the v3 audit's 54.37%.

Registry: every cell is recorded with `nfl-ats weak-signals record` under
the prefix `pol09_best_pick_composed_v1_`, family
`pol09_best_pick_composed_v1`, with a plain summary. Classification per the
binding taxonomy: `unresolved_below_power` by default; `refuted_mechanism`
with `wrong_sign_resolved` ONLY if the whole interval sits below zero; no
positive control was run, so `bounded_by_control` is unavailable. An
interval containing zero is never a rejection; `probability_positive` is
reported, and above 0.5 favours playing the first-named rule.

Decision rule, fixed now: **nothing served changes from this document.** A
change to the served nomination is proposed in the session report only if a
measured alternative among the declared cells has `probability_positive`
above 0.5 against v2 as played.

## Results (measured 2026-09-07/08, `artifacts/best_pick_composed_rule/20260908T003306Z/summary.json`)

Population as predeclared: 1,537 games, 107 weeks, 2020–2025, 34 pushes;
the production pool (`dispersion_pool_from_frame`) matched the eval script's
predeclared pool on every one of the 1,537 games; 10 weeks fell back to the
full slate (8 missing data, 2 empty strict filter). The bootstrap is
deterministic: a scratch run and the artifact run produced identical cells.

### Decision read — top-1 hit rate, opener grade, active-model pick side

| rule | hit rate (weeks scored) |
|---|---|
| **v2 as played** (production) | **56.31% (58/103)** |
| v2 + 10+ spread discount (side-ledger challenger) | 57.28% (59/103) |
| v1 `sweep_robustness` (what v2 replaced; tied at the top in 58 of 107 weeks, alphabetical decided those) | 55.66% (59/106) |
| raw top-\|residual\| | 52.83% (56/106) |
| chooser 6 alone (filter, game_id tie-break = v3) | 57.28% (59/103) |
| chooser 8 alone (dispersion tie-break, unfiltered) | 50.00% (52/104) |
| chooser 4 (unfiltered candidate distance) | 50.96% (53/104) |
| all-pick week average | 52.66% |
| *reconciliation: v2 on the candidate arm's side (the v3 audit's read)* | *54.37% (56/103) — ties out exactly* |

Paired within week, week-blocked bootstrap 20,000 draws seed 20260817,
accuracy points, positive favours the first-named rule:

| cell | delta | 95% interval | `probability_positive` | weeks paired / nominee differs / outcome differs |
|---|---|---|---|---|
| U1 v2 as played vs v1 | **+0.97** | [−10.68, +11.65] | **0.536** | 103 / 69 / 35 |
| U1 v2 as played vs top-\|residual\| | **+3.88** | [−6.80, +14.56] | **0.729** | 103 / 72 / 34 |
| U2 v2+discount vs v1 | +1.94 | [−8.74, +12.62] | 0.602 | 103 / 67 / 34 |
| U2 v2+discount vs top-\|residual\| | +4.85 | [−6.80, +16.50] | 0.774 | 103 / 73 / 37 |
| U2 v2+discount vs U1 v2 as played | +0.97 | [−2.91, +4.85] | 0.592 | 103 / 8 / 5 |

All five recorded `unresolved_below_power` (no resolved wrong sign, no
positive control), registry ids `pol09_best_pick_composed_v1_*`, family
`pol09_best_pick_composed_v1`. **What it implies for the decision, stated
before what is wrong with it:** the rule the pool is actually playing leans
ahead of both rules it replaced (P+ 0.54 vs v1, 0.73 vs top-|residual|), so
nothing measured here argues for reverting. The discount is the one declared
alternative with P+ above 0.5 against v2 as played (0.592) — but it changes
the nominee in only 8 of 103 weeks and the outcome in 5, and the +0.97-point
delta is a 3–2 split of those five weeks. It stays a side-ledger challenger;
this document proposes no change to the served nomination on five weeks.

### Where the composition mattered

- v2 differs from chooser 6 alone (its filter with the game_id tie-break) in
  **2 of 107 weeks** — the composition question is really only those two
  weeks (2020 wk 8, both lost; 2023 wk 15, where the dispersion tie-break's
  MIN_CIN lost and the alphabetical DAL_BUF won). On the as-played side the
  same two weeks decide it: chooser 6 alone reads 57.28% vs v2's 56.31%.
- v2 differs from chooser 8 alone (dispersion tie-break, unfiltered) in 43
  weeks, from the unfiltered chooser 4 in 45, from v1 in 72, from
  top-|residual| in 75. The filter is where the rule's whole difference
  from its predecessors lives; the tie-break layer is nearly inert.

### Tie-break audit (descriptive)

The filtered pool tied at the top in **5 of 107 weeks**. The dispersion
tie-break decided 3 of those; the other 2 fell through to game_id. In
exactly **1** week (2023 wk 15) the dispersion nominee differs from the
alphabetical one. In the five tie weeks the dispersion nominee hit 1 of 5
(20%), the alphabetical fallback 2 of 5 (40%): paired −20.0 points,
[−60.0, 0.0], P+ 0.00 on 5 blocks — recorded
`pol09_best_pick_composed_v1_tiebreak_dispersion_vs_alphabetical` as
`unresolved_below_power`, because the whole interval is not below zero
(its upper bound sits at 0.0) and the entire number is one week. A P+ of
0.00 built from one informative week is not a resolved wrong sign; it is
the shape of N=1.

### Caveats, stated plainly

- **The v1 comparator's sweep is a refit, not the archived fit.** The
  archive was built 2026-08-18 from a `weak_stack` table (sha `0a18e2d9…`)
  that no longer exists on disk; the refit used the ACTIVE forecast's table
  (`data/processed/game_features_weak_stack.parquet`, sha `457aafb7…`, the
  same table the live v2 nomination reads). Against the archive the refit's
  opener residuals correlate 0.983, mean |diff| 0.20 points (median 0.09,
  p95 0.84, max 3.47), pick-side agreement 96.1% (100% in 2020, 87.9% in
  2025 — the recent seasons are where the feature builders changed), median
  within-week rank agreement of |residual| 0.978. The v1 nominee's
  correctness was always scored on the ARCHIVE's played side. Both v1 cells
  carry this discount; the v2/U2/top-|residual| cells do not (they read the
  archive directly).
- v1 was swept with `probability_method="gaussian_median"` (the active
  manifest's mapping). Its 2020–2021 confirmation ran under ECDF; the
  0.50 threshold makes the width nearly mapping-independent, but the two
  are not byte-identical.
- Fourth/fifth reuse of the same ~107 weeks by this family; no rotation
  window spent. The intervals are the honest ~10-point width of a top-1
  rate on ~100 weeks; nothing here is resolved, and the decision read above
  is an EV read, not a claim.

### Week 1 2026 reproduction (read-only, measured 2026-09-08)

`scripts/best_pick_composed_rule_eval.py --week1-check`, reading the active
manifest's linked forecast (`margin_predictions/2026-week-01-20260907T232720Z`,
model `a4c757efd2525da6`) through `nominate_v2` and
`apply_big_spread_eligibility` with production's own inputs: nominee
**`2026_01_MIA_LV`, MIA +3.5**, untied (`tie_break="none"`), no pool fallback
(8 of the week's games eligible); the big-spread discount excludes nothing
and names the same game. Matches the ROADMAP's "Week 1 stays MIA +3.5 under
both rules". Nothing was written.
