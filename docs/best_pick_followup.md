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

## 2026-08-31: NFL dispersion-filter confirmation — predeclaration, then STOP

Written to close out `docs/pool_edge_plan.md`'s 2026-08-31 addendum item 5
("Best Pick ranker: dispersion-filtered chooser vs unfiltered, confirmed on a
genuinely fresh window"). This section predeclares the confirmation design
FIRST, then reports why the window-legality check that design requires came
back negative, before any outcome was scored. Nothing below is a scored
result; no `weak-signals record` or `rotation record` call was made, because
none produced an admissible measurement. Binding taxonomy, pasted verbatim
per AGENTS.md's subagent-portability requirement even though no subagent was
used here: *an interval or CI that contains zero is NEVER grounds to reject,
fail, or close an experiment. At this evaluator's ~2-point resolution,
"contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism — a RESOLVED wrong
sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is `unresolved_below_power`.*

### What the candidate IS (read, `src/nfl_ats/best_pick_nomination.py`)

The live production nomination (`NOMINATION_V2_ENABLED = True`,
`select_nominee`) and its side-ledger sibling v3 (`select_nominee_v3`) both
rank a week's games by an alpha=2000 `market_residual` calibrated
probability's distance from 0.5 (`candidate_dist`), restricted to that
week's below-median cross-book Tuesday-opener `spread_std` pool
(`week_dispersion_pool`, reading `nfl_ats.market_data.tuesday_opener_quotes`),
falling back to the full week on missing data or an empty strict filter. v2
additionally tie-breaks on lower dispersion before falling through to
ascending `game_id`; v3 (and the historically-scored "chooser 6",
`dispersion_filtered_candidate`) breaks ties on ascending `game_id` alone.
The disclosed method sentence is verbatim `NOMINATION_V2_METHOD_SENTENCE`,
"nominated by calibrated probability among low-disagreement games"
(`src/nfl_ats/best_pick_nomination.py:583`).

Two registry cells already measure this exact mechanism, both on the
identical 1,537-game/107-week paired opener archive, seasons 2020-2025
(read, `registry/weak_signals.json`):

- `best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered`:
  dispersion-filtered chooser (no dispersion tie-break — the v3/chooser-6
  form) vs the SAME candidate signal unfiltered (ranked over the whole
  week). +3.9216 accuracy points, week-blocked 95% `[-3.92, +11.76]`,
  `probability_positive` 0.8132, "**Third** reuse of the same 107 opener
  weeks" (ridge_alpha promotion look, odds-microstructure battery, this
  ranker screen). This is the cell `docs/pool_edge_plan.md`'s 2026-08-31
  addendum item 5 calls "the strongest of the four current ranker
  variants."
- `best_pick_opener_ranker_dispersion_filtered_candidate_vs_live_v2`: the
  same dispersion-filtered chooser vs the ACTUAL live v2 rule (filter plus
  dispersion tie-break). +0.9709 accuracy points, `[0.0, +2.91]`, P+
  0.6309, "**Fourth** reuse of the same 107 opener weeks," and the entire
  point estimate traces to one diverging week (2023 week 15) out of 103
  paired weeks.

Neither the `best_pick_ranker` nor `best_pick_ranker_opener` rotation
families cover this question: both test `sweep_robustness` top-1 (a
different ranker entirely) and hold unrelated spent windows (`[2013,2015]`
nflverse_spread grade; `[2020,2021]` opener grade — read,
`registry/rotation_registry.json`). No rotation family exists for the
dispersion-filter mechanism itself; both cells above were scored and
recorded outside the rotation registry (`registry/weak_signals.json`'s own
notes on both cells: "Not a rotation-registry window; no window is spent or
implied").

### Predeclared confirmation design (frozen before any window-legality check)

- **Candidate**: chooser-6/v3 form — alpha=2000 `candidate_dist`, ranked
  within that week's below-median cross-book Tuesday-opener `spread_std`
  pool (same fallback rule: full week on missing data or an empty strict
  filter), ties broken ascending `game_id`.
- **Comparator**: the SAME `candidate_dist` signal, unfiltered (ranked over
  the whole week), ties broken ascending `game_id` — matching the strongest
  registry lead above so the confirmation targets the cell already flagged
  as a next shot, not a new comparison invented after seeing this session's
  data.
- **Metric**: per-week Best-Pick lift (nominee's own-arm correctness minus
  that week's full-slate average correctness, pushes excluded), paired
  weekly delta (candidate minus comparator), in accuracy points — identical
  unit and construction to every cell already recorded for this family, so
  a pass would be directly commensurable and poolable with them.
- **Pairing / blocks**: one row per (season, week); week-blocked bootstrap,
  20,000 samples, seed 20260831, matching the family's existing seed
  convention (20260818/20260821) in kind, not value, so this look is never
  mistaken for a re-run of a prior one.
- **Tie-handling**: ascending `game_id` for BOTH arms symmetrically (never
  alphabetical resolution favoring one arm); no chooser in this design ever
  breaks a tie by an unweighted alphabetical rule the way the retired
  `sweep_robustness` did.
- **Positive control**: a perfect-foresight oracle chooser (nominate the
  week's actual highest-lift pick) vs the same unfiltered comparator, on
  the identical fresh population — mirroring `movement_expansion_battery`'s
  oracle design (+46.27 pts, both intervals fully positive, P+ 1.0), to
  prove the harness can detect a real effect of that size before trusting
  a null read on the real cells.
- **Window requirement**: seasons disjoint from `{2020, 2021, 2022, 2023,
  2024, 2025}` — the union already touched by the two cells above across
  their third and fourth reuses.

### Window-legality check — every avenue closed, in order

1. **Rotation-registry pool is hard-capped at 2020-2025 for this grade, in
   code, not just prose.** Read, `src/nfl_ats/rotation.py:37-43`:
   `GRADE_POOLS["opener"] = (2020, 2025)`, commented "Opener-graded
   confirmations need the paired Tuesday-opener archive, which only covers
   2020-2025." `close` and `nflverse_spread` grades report a wider nominal
   pool, `(2009, 2025)`, but that width describes ordinary single-book
   `market_residual` features (available since 2009); the cross-book
   `spread_std` dispersion feature this candidate is defined on comes from
   one specific store regardless of grade (see next point), so switching
   grade would not actually reach a different dispersion population — it
   would silently swap the candidate's own defining feature for something
   else.
2. **The underlying multi-book archive is a purchased, fixed-span product,
   confirmed independently of the registry code.** Read,
   `docs/mkt09_licensing_audit.md:47`: "The Odds API — paid historical
   snapshots (the 'purchased 2020–2025 archive')... Point-in-time
   multi-book odds boards... 2020–2025." Measured this session,
   `ls data/market/raw | sort | head`: the earliest local snapshot
   directory is `20200825T115500Z-futures`; nothing precedes it.
3. **Simulated (in-memory, no file written) what a brand-new rotation
   family for this exact question would actually be offered.** Measured
   this session by calling `nfl_ats.rotation.declare_family` +
   `eligible_blocks` directly against the loaded registry, never writing
   it back:

   ```
   sim = declare_family(registry, "<new family>", grade="opener",
                         acknowledges_mined_2018_2025=True)
   eligible_blocks(sim, "<new family>")
   → ((2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025))
   ```

   Every one of those five candidate blocks is a literal subset of the
   identical 1,537-game/107-week archive already scored for this exact
   dispersion-filter comparison three and four times over. The rotation
   registry's per-family bookkeeping (rule 4: windows retire per family,
   not globally — deliberately, so independent hypotheses can share a
   season range) would label any of them "unspent" for a family that has
   never drawn a window under that name before. That is a bookkeeping
   artifact of naming a new family, not evidence of an unseen population:
   assigning one would relabel already-scored games as fresh, exactly the
   look-reuse this mission's mandate rules out. No family was actually
   declared or assigned; `registry/rotation_registry.json` is untouched.
4. **A structurally different cross-book dispersion proxy exists for a
   disjoint era, but it fails on reliability, not availability.**
   `docs/vi_dispersion_screen.md` (read) scores Wayback-archived
   VegasInsider multi-book boards for REG 2009-2016 — seasons entirely
   outside the 2020-2025 union above. But that document's own measured
   split-half reliability of the spread-SD trait is approximately zero:
   mean split correlation −0.0206 (Spearman−Brown −0.0421) in the first
   measurement, −0.0421 recomputed in the second (618 ≥6-book games both
   times). A trait with no split-half reliability cannot be rescued by any
   sample size (AGENTS.md's own second closing ground), and substituting
   it here would not confirm or refute the production `spread_std`
   mechanism — it would score a different, already-disclosed-as-noisy
   instrument and misreport it as a confirmation of this one, violating
   the commensurability rule ("same units, same scale, same population")
   this project already treats as binding.
5. **The 2026 season has no graded outcomes yet.** Measured this session,
   `ls data/market/raw | sort | tail`: latest local capture is
   `20260830T201559Z` (pre-kickoff). `HANDOFF.md` (read): the linked 2026
   Week 1 forecast was created 2026-08-24 and is not yet graded. No week
   exists there to block-bootstrap.

### Determination: STOP

No legal, commensurable, genuinely non-reused window exists for a
confirmation look at this specific dispersion-filtered-ranker mechanism.
Per this mission's own predeclared contingency, no scoring was performed:
nothing was recorded to `registry/weak_signals.json`, no family was
declared or window assigned in `registry/rotation_registry.json`, and
neither registry changed as a result of this section (verified,
`registry/weak_signals.json` 615 signals — 548 NFL / 67 CFB, 612
`unresolved_below_power` / 3 `refuted_mechanism` — before and after; `rotation
status` families list unchanged at 14). Fabricating a look on an exhausted
population, or writing a registry entry for "no experiment ran," would
misuse both registries' schemas (`weak-signals record` requires an outcome;
`rotation record` requires a spent window) rather than honor them.

### EV implication (first, per the binding order)

**No change to the live nomination.** This session neither strengthens nor
weakens the case for `NOMINATION_V2_ENABLED`; the two already-recorded cells
stand exactly as before — dispersion-filtered-vs-unfiltered +3.92 pts (P+
0.8132) and dispersion-filtered-vs-live-v2 +0.97 pts (P+ 0.6309, one
diverging week carrying the whole estimate) — both `unresolved_below_power`,
both EV-positive leans on their own reuse-discounted terms, neither upgraded
nor downgraded here. No production code path (`best_pick_nomination.py`,
`publishing.py`) was touched.

### What would actually unlock a fresh window

Not another agent session re-running this same legality check. Either (a)
the purchased Odds API archive is ever extended to seasons outside
2020-2025 (a purchasing decision, out of this session's scope), or (b) the
2026 season's opener weeks are played and accrue enough weeks to
week-block-bootstrap on their own account — a matter of time, not of
drawing harder from the population that already exists. Recommend
`docs/pool_edge_plan.md`'s item 5 be re-flagged, next time it is edited by
whichever lane owns it, as "blocked on new seasons of the 2020-2025 archive
existing," not as an open next shot a fresh agent can just go draw.

