# MOD-18 spread-size regime

## Goal
ROADMAP MOD-18: "calibrate the model by spread instead of flipping it." The
2026-09-15 audit read the OLDER served model at 55.94% on |spread|<=7
(n=1,178) vs 49.23% on |spread|>=7.5 (n=325). Test whether that deficit
survives under the CURRENT served four-term fitted probability
(`nfl_ats.pick_probability_fit`, activated 2026-09-20), and whether adding a
fitted spread-magnitude term (never a flip) improves the calibrated
probability LOSO by season, on 2020-2025 and on the extended 2011-2025
population. Decision implication: would it change any Week 3 served pick.

## Predeclaration (before any fit ran)
- Candidate terms, fixed a priori, two continuous terms, no bucket dummy (a
  dummy at a threshold would be a bolted-on flip, banned by AGENTS.md):
  `abs_open_spread` (|opener home spread|) and
  `abs_open_spread_x_model_logit` (`abs_open_spread * model_logit`,
  raw product before standardization). Both enter the same standardized
  ridge-logit design as the served `FIT_FEATURES`.
- Measurement-only bucket split (not a fitted term, not a flip): |spread|<=7
  vs |spread|>=7.5, reusing the 2026-09-15 audit's own predeclared cut so the
  comparison is apples-to-apples, not a newly-mined threshold.
- Populations: (a) served 2020-2025 `build_fit_population` output (n=1,503),
  (b) extended 2011-2025 discrete-read population
  `artifacts/extended_fit_population/20260923T205910Z/population.parquet`
  (n=3,734), spread magnitude re-attached by game_id from
  `artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet`
  (`proxy_open_home_spread`, seasons<=2019) and from a fresh
  `build_fit_population` call (`tue_open_home_spread`, seasons>=2020).
- Method: LOSO by season for both the served 4-term base and the 6-term
  candidate; paired season-block bootstrap (`nfl_ats.clv.week_blocked_bootstrap`,
  `block="season"`, 4000 samples, seed 20260924) against the base on the same
  games, reusing the machinery `scripts/opener_error_transfer_unit2.py` used.
- Script: new file `scripts/mod18_spread_regime.py`, run once, artifacts under
  `artifacts/mod18_spread_regime/<ts>/`.
- Looks counted: bucket accuracy/reliability read (2020-2025), bucket
  accuracy/reliability read (2011-2025), candidate-vs-base LOSO fit +
  bootstrap on 2020-2025, same on 2011-2025 = 4 predeclared looks plus any
  calibration-table reads logged by the script's own `record_look`.

## State
Unit 1 DONE. `scripts/mod18_spread_regime.py` (new, ruff format+check clean),
run once for real. Artifact:
`artifacts/mod18_spread_regime/20260924T183858Z/summary.json` (+
`per_game_2020_2025.csv`, `per_game_2011_2025.csv`). 19 looks logged
(`look_log` in the summary).

**Served four-term probability, LOSO by season, bucket read (measured)**:
- 2020-2025 (n=1,503 fit population): small `|spread|<=7` n=1,177,
  **58.20%** (685-492); large `|spread|>=7.5` n=325, **53.54%** (174-151).
  Deficit 4.66 pts — smaller than the 2026-09-15 audit's older-model read
  (55.94% vs 49.23%, 6.71 pts) but the direction persists.
- 2011-2025 extended population (n=3,734): small n=2,936, **53.34%**
  (1566-1370); large n=797, **53.70%** (428-369) — the deficit **does not
  replicate**; large-spread bucket reads marginally better on the wider
  window.

**Candidate (base 4 terms + `abs_open_spread` + `abs_open_spread_x_model_logit`,
LOSO by season) vs base, paired season-block bootstrap (4000 samples, seed
20260924)**:
- 2020-2025: effect **-0.27 accuracy pts**, interval **[-1.31, +0.69]**,
  **P+ 0.336**. Decisive (disagreement) games n=140: candidate 68-72, base
  72-68 — candidate loses the head-to-head.
  Overall: base acc 57.15%/Brier 0.2454, candidate 56.89%/Brier 0.2452
  (accuracy down, Brier flat).
- 2011-2025: effect **-0.27 accuracy pts**, interval **[-1.05, +0.56]**,
  **P+ 0.259**. Decisive games n=212: candidate 101-111, base 111-101.
  Overall: base acc 53.40%/Brier 0.2486, candidate 53.13%/Brier 0.2489
  (both flat-to-worse).
- Per-fold LOSO coefficients are sign-stable in every one of 6 (served) and
  15 (extended) folds (`abs_open_spread` positive every fold,
  `abs_open_spread_x_model_logit` negative every fold) — not noise-flipping,
  but the sign-stable term still does not beat the base on paired accuracy.
  In-sample-minus-OOS gap is small both ways (served -0.0047 acc, extended
  +0.0032 acc) — no overfitting explains the miss.
- Neither interval sits wholly on one side of zero, so per AGENTS.md this is
  `unresolved_below_power`, not `refuted_mechanism`; P+ well under 0.5 in
  both reads is evidence against, not proof against.

**Week 3 decision check (measured, full-sample refit, not LOSO, sanity only)**:
16 season=2025/week=3 games in the fit population; 3 would flip side under a
full-sample candidate refit (`2025_03_ARI_SF`, `2025_03_DAL_CHI`,
`2025_03_MIA_BUF`, all spreads <=12.5, none flip because of a large-spread
adjustment specifically). **No served pick changes**: the candidate reads a
negative point estimate with P+ 0.26-0.34 in the predeclared LOSO test, so it
does not clear even "probability just above 0.5" — AGENTS.md's promotion bar
does not apply here since this isn't a marginal-positive case being held to
0.90; it is a marginal-negative case, and no rule flips a side on its own
regardless.

**Record commands drafted (not run — root records)**:
```
nfl-ats weak-signals record --name mod18_spread_regime_served_2020_2025 \
  --description "abs opener spread + its interaction with model_logit added as 2 fitted terms to the served four-term NFL pick probability, LOSO by season, vs the four-term base, on the served 2020-2025 fit population" \
  --source artifacts/mod18_spread_regime/20260924T183858Z/summary.json \
  --effect -0.26613439787092075 --effect-units accuracy_points \
  --classification unresolved_below_power --league nfl \
  --season-start 2020 --season-end 2025 \
  --interval-low -1.314965560425807 --interval-high 0.692041522491349 \
  --probability-positive 0.336125 --sample-games 1503 --sample-blocks 6 \
  --family mod18_spread_regime \
  --classification-evidence "Interval crosses zero (P+=0.336, negative point estimate); per-fold LOSO coefficients are sign-stable across all 6 seasons (abs_open_spread always positive, interaction always negative) so this is not a noise-flip artifact, it is a stable term that still loses the paired accuracy comparison (decisive-game record 68-72 vs base 72-68); in-sample-minus-OOS accuracy gap -0.0047 rules out overfitting as the explanation"

nfl-ats weak-signals record --name mod18_spread_regime_extended_2011_2025 \
  --description "same two spread-size terms fit on the extended 2011-2025 discrete-read population (artifacts/extended_fit_population/20260923T205910Z), LOSO by season, vs the four-term base; also used to check whether the served-population large-spread deficit replicates on more games" \
  --source artifacts/mod18_spread_regime/20260924T183858Z/summary.json \
  --effect -0.26780931976432276 --effect-units accuracy_points \
  --classification unresolved_below_power --league nfl \
  --season-start 2011 --season-end 2025 \
  --interval-low -1.0543310387294587 --interval-high 0.5595235402853109 \
  --probability-positive 0.259 --sample-games 3734 --sample-blocks 15 \
  --family mod18_spread_regime \
  --classification-evidence "Interval crosses zero (P+=0.259, negative point estimate); the large-spread deficit this unit was chartered to test does NOT replicate on the wider window (large-spread bucket 53.70% vs small-spread 53.34%, base four-term LOSO) so there is less mechanism to fit here than the served-population read suggested; per-fold coefficients sign-stable across all 15 seasons; in-sample-minus-OOS accuracy gap +0.0032"
```

## Tried
- Reused `week_blocked_bootstrap(block="season")` and the LOSO/ridge-logit
  machinery from `scripts/opener_error_transfer_unit2.py` unmodified in
  structure.
- Re-attached opener spread magnitude to the extended population (which
  dropped it in `KEEP_COLUMNS`) via two joins: `proxy_open_home_spread` from
  `artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet` for
  seasons<=2019, `tue_open_home_spread` from a fresh `build_fit_population`
  call for seasons>=2020. Zero missing after the join (verified).
- Considered a bucket-dummy fitted term instead of continuous magnitude;
  rejected per AGENTS.md ("a dip located only at a spread threshold is a
  diagnosis to publish and repair, never a flip to bolt on") — used the
  bucket split only for the read-only reliability/accuracy report, not as a
  fitted term.

## Next
Root runs the two `weak-signals record` commands above (Unit 1, both
`unresolved_below_power`). No further fitting is warranted from this unit's
evidence: the mechanism this row was chartered to test (large-spread deficit)
did not replicate on the larger 2011-2025 window, and the candidate term,
though sign-stable, loses the paired accuracy comparison on both windows. A
future unit could try the spread term as a `market_move`-style interaction
restricted to `MOVE_AVAILABLE_COLUMN=1` rows, or test a total-based analogue,
but neither was predeclared here.

## Open
None — measurement complete, no owner decision required. Root should record
the two cells above.
