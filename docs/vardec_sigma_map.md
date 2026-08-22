# Variance decomposition sigma map (heteroskedasticity screen)

Question: is the ~13.1-point ATS residual sd a constant, or does residual sigma
vary measurably by observable pregame conditions? If sigma is conditionally
predictable, that is directly relevant to a forced-pick pool via BEST-PICK
selection even with zero mean edge (POL-09 connection).

Status: measure-only mined battery, first run 2026-08-22. Nothing here is a
validated selection strategy and nothing was recorded without review.

## Population and method

- Population: REG 2009-2025, newest snapshot `data/raw/20260817T235649Z/schedules.parquet`,
  4,431 games scored after dropping pushes/missing lines (**measured**).
- Target: `ats_margin = result - spread_line` via `add_ats_outcomes`
  (`src/nfl_ats/features.py:42`). Overall residual sd **13.130 points** (**measured**).
- Statistic per cell: subset sd / complement sd of `ats_margin`, joint
  week-blocked bootstrap (season-blocked secondary), 20,000 resamples,
  seed 20260822, percentile 95% interval.
- Every condition is pregame-known. The two derived cells use prior-games-only
  information: late-contention standings are computed from prior-week records
  only; non-incumbent-QB uses each team's modal starter over PRIOR games of the
  same season (weeks >= 2). No game-time actuals enter any flag.
- Artifact: `artifacts/vardec_sigma/20260822T194037Z/results.json` (**measured**).

## MINED multiplicity disclosure

The 18 cells were chosen from the task brief after the data landscape was known.
At uncorrected 95%, roughly one spurious interval excluding 1.0 is expected by
chance alone. No correction was applied. Per AGENTS.md, an interval containing
1.0 is never grounds for closing a line; `probability_ratio_below_one` is
reported instead of any "contains" phrasing.

## Sigma map (week-blocked primary)

| # | Condition | Family | n | sigma | complement sigma | ratio [95% CI] | P(ratio<1) |
|---|-----------|--------|---|-------|------------------|----------------|------------|
| 1 | dome/closed roof | roof_type | 1215 | 13.21 | 13.10 | 1.008 [0.953, 1.066] | 0.392 |
| 2 | retractable open-air | roof_type | 84 | 11.85 | 13.15 | 0.901 [0.708, 1.087] | 0.870 |
| 3 | outdoor temp < 40F | weather_bands | 455 | 12.99 | 13.23 | 0.982 [0.909, 1.056] | 0.695 |
| 4 | outdoor temp >= 82F | weather_bands | 248 | 14.41 | 13.08 | 1.102 [0.995, 1.204] | 0.031 |
| 5 | wind >= 13 mph | weather_bands | 551 | 12.95 | 13.25 | 0.977 [0.906, 1.048] | 0.742 |
| 6 | division game | matchup | 1632 | 12.76 | 13.34 | 0.956 [0.913, 1.002] | 0.971 |
| 7 | late contention wk14+ | motivation | 706 | 13.20 | 13.12 | 1.007 [0.945, 1.071] | 0.429 |
| 8 | primetime kick >= 20:00 ET | slot | 849 | 12.93 | 13.18 | 0.981 [0.925, 1.037] | 0.751 |
| 9 | non-incumbent QB start | qb | 1252 | 13.23 | 13.09 | 1.011 [0.961, 1.061] | 0.342 |
| 10 | abs spread <= 3 | spread_buckets | 1622 | 13.34 | 13.01 | 1.025 [0.976, 1.077] | 0.165 |
| 11 | abs spread 3-7 | spread_buckets | 1769 | 12.96 | 13.24 | 0.979 [0.932, 1.028] | 0.804 |
| 12 | abs spread 7-10 | spread_buckets | 597 | 13.15 | 13.13 | 1.002 [0.934, 1.069] | 0.492 |
| 13 | abs spread > 10 | spread_buckets | 443 | 12.96 | 13.14 | 0.986 [0.913, 1.058] | 0.669 |
| 14 | season weeks 1-3 | season_week | 815 | 13.04 | 13.15 | 0.991 [0.931, 1.049] | 0.629 |
| 15 | season weeks 15+ | season_week | 895 | 13.29 | 13.09 | 1.015 [0.958, 1.075] | 0.310 |
| 16 | turf surface (vs grass) | surface | 1946 | 13.17 | 13.09 | 1.007 [0.962, 1.053] | 0.388 |
| 17 | altitude (DEN home, Azteca) | altitude | 143 | 13.16 | 13.13 | 1.002 [0.861, 1.149] | 0.529 |
| 18 | abs rest diff >= 4 days | rest | 502 | 12.12 | 13.25 | **0.915 [0.852, 0.977]** | 0.996 |

Season-blocked secondary for cell 18: 0.915 [0.849, 0.980] (**measured**) —
direction replicates under both blockings.

## Verdicts

- **Low-sigma bucket (only interval excluding 1.0): rest differential >= 4
  days** — sd 12.12 vs 13.25, ratio 0.915, 95% [0.852, 0.977],
  `probability_ratio_below_one` 0.996 (**measured**, artifact above). These are
  mostly bye-week-differential and post-bye/short-week mismatch games. This is
  a category-3 result: real-looking at both blockings but mined, so it must be
  treated as a lead, not a finding, until a predeclared confirmation look.
- **High-sigma bucket: none with an interval excluding 1.0.** Nearest is hot
  outdoor (>= 82F) at 1.102 [0.995, 1.204] — directionally interesting,
  unresolved (**measured**).
- Everything else: ratios sit in a narrow 0.91-1.10 band around 1. The headline
  answer is that the 13.1 residual sd is *nearly* constant across observable
  pregame conditions; the one measurable departure is rest-differential games.

## Best Pick implication (attribution only)

For the rest-differential low-sigma bucket (**measured**, attribution only):

- Historical favourite-cover accuracy inside the bucket: **49.20%** vs
  **47.64%** on the whole slate (+1.56 points). Both are below 50% because the
  flat favourite rule is itself sub-.500 on this population; the gap, not the
  level, is the signal.
- Analytic sensitivity: at a hypothetical +1-point mean edge per pick,
  cover probability is Phi(1/sigma) = **53.29%** inside the bucket versus
  **53.04%** at the slate sigma — about +0.25 points per pick purely from
  lower variance. That is the honest size of the POL-09 lever: variance
  selection amplifies whatever mean edge exists; it does not create one.
- Caveats: single mined look; bucket membership overlaps other cells;
  no predeclared confirmation window has been spent.

## Proposed record lines

None executed. The registry's effect units (`ats_points`, `accuracy_points`,
brier, log_loss, mae) cannot represent an sd-ratio without misreading it as a
mean shift, which would corrupt pool commensurability. If the owner wants this
in `registry/weak_signals.json`, the least-bad encoding is the point-scale sd
gap with explicit notes:

```
nfl-ats weak-signals record --name vardec_rest_diff_ge4_low_sigma --league nfl --season-start 2009 --season-end 2025 --source artifacts/vardec_sigma/20260822T194037Z/results.json --effect -1.129 --effect-units ats_points --interval-low -1.876 --interval-high -0.372 --probability-positive 0.996 --sample-games 502 --sample-blocks 294 --classification unresolved_below_power --classification-evidence "mined cell, direction replicates week- and season-blocked; mechanism open (bye/rest mismatch games show tighter ATS residuals); no predeclared confirmation look yet" --notes "SD DIFFERENCE not mean shift: subset sd 12.125 vs complement 13.254, ratio 0.915 [0.852,0.977] week-blocked, [0.849,0.980] season-blocked; units field cannot represent sd-ratio"
```

(`--effect` is sigma_subset - sigma_complement in points; the interval endpoints
are that difference scaled by the bootstrap ratio CI. Owner decision whether
that encoding is admissible.)
