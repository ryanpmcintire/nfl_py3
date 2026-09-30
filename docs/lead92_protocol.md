# LEAD-92 protocol saved before outcomes

## Protocol declaration (verbatim text, wrapped)

Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week,
observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly noon boundary. No closing inputs.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros.
LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs. Reused archives are retrospective; freeze
survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and
gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing;
default `unresolved_below_power`.

Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four contrasts × IS/OOS/gap ×
K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly reward/nominee Brier; 88 adds all-game/last-
game total MAE. No unlisted variants.

For 90–97 use C's chronology, five controls/arms, reporting and look formula with F=3; historical OPENER is the frozen Splash-line proxy, never pre-2026 pool captures. Shared **measured** inputs:
`artifacts/pick_probability/20260929T192747Z/per_game.parquet` (1,503 rows/107 weeks), `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet` (1,537 including pushes),
`data/processed/game_features_weak_stack.parquet` (4,902). Reuse the upstream cutoff/refit machinery documented in `docs/lanes/lead83.md:14–19`; a downstream LOSO label alone does not certify upstream
chronology. Fixed ridge .001, no tuning grid; selection and calibration years remain separate. Each row is an independent challenger, not an eight-way winner-selection exercise. All features/quotes
are pre-deadline; historical scores/plays are training labels only. One calibrated line-conditioned PMF decides sides, with the served four-term recipe as the paired base. K=4 normally; 90/91/93/95
add local RPS/worst-season loss/tail RPS/regulation log score respectively; 92 adds nominee reward/Brier; 94 adds all-game/last-game total MAE. Tiebreaker controls follow C/LEAD-88; tiebreaker changes
keep ATS sides fixed. All population filters are outcome-blind, all IS/OOS gaps and diagnostic looks reported. Zero crossing closes nothing; RPS or tiebreaker gains alone cannot promote ATS sides
(AGENTS.md, Margins/Promotion). Freeze survivors for subsequent normal 2026 capture collection.

| LEAD-92 | ⬜ | Learn within-week Best Pick ordering directly (rank 3/8) | **Inferred mechanism:** several stale pool lines can offer similar chances, while all-game loss underweights choosing the
week's bonus candidate. Fit the same four-term score with pairwise logistic loss on within-week discordant cover labels, both team-side orientations, excluding same-game pairs; normalize to one weight
per week. Separate-year calibration produces one home probability/PMF that selects sides and legal nominee. **Measured input:** `artifacts/pick_probability/20260929T192747Z/per_game.parquet`, 1,503
rows/107 weeks. D: nominee reward primary, B=6/K=6, **713 looks**. **Read/checked:** POL-09, LEAD-53/70/72/80/85/89, `docs/confidence_top_calibration.md:5–9`; ranking-loss training differs from
temperature, push arithmetic, uncertainty integration and timing. Not a fifth term. Rank reason: bonus-specific target, cheap pairs; pairs do not multiply independent games. Units 15–20/20–25 calls. |

## Implementation details fixed before outcomes
Use source-complete rows from the frozen opener population, requiring a dated move,
a pre-deadline Sunday market quote, and an archived Tuesday total. No minimum book-count filter.
Reconstruct moves using LEAD-73 and the existing Sunday-move function; reuse audited
composition flags including pushes. Refit margin, line-conditioned integer lattice,
home offsets, four-term comparator and Elo at each outer cutoff using LEAD-83 machinery.
Pair loss is the sum of within-week mean discordant-pair logistic losses plus
ridge .001/2 times squared coefficients, both orientations, different games only.
Keep the existing positive temperature selection on the tuning year's Brier score and
intercept calibration on the following year, identically for all five arms.
Nomination uses the existing Sunday eligibility, dispersion pool and deterministic
tie rule, expected win plus half a push; final unresolved ties average legal nominations.
Nominee reward is mean points per week; nominee Brier is conditional on a non-push.
Use 10,000 paired season/week-block resamples, seed 20260929, fixed fitted predictions;
IS means optimistic fitting-year predictions, gap is OOS minus IS. Positive contrasts
mean challenger improvements. All four baselines receive the same source-complete games.
The sixth specification counted in B is the upstream margin/discrete nuisance fit.
