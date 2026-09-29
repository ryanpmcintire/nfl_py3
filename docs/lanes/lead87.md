# LEAD-87
## Goal
Complete the declared first unit: postseason clock/label join for the shared news-response coefficient.
## State
**Measured:** join complete; 262 verified quote files, 52 paired POST games (8/9/9/8/9/9), one push, 51 conditional-fit rows. All 1,503 frozen REG rows retained; 1,272 have leader pairs (686 in outer 2023–2025). No fit or score.
**Measured:** companion cutoffs precede the game for 1,503/1,503 REG rows; 1,415 use earlier games within their own season. Strict outer-season exclusion and exact opener lineage remain unverified.
## Protocol declaration — copied before outcomes; unchanged
| LEAD-87 | ⬜ | Borrow postseason games for the news-response coefficient (batch C rank 6 of 8) | **Inferred mechanism:** a Tuesday price also becomes stale before playoff
kickoffs; excluding those independent NFL games discards evidence about how observed price corrections map to cover probability. Predeclare one joint likelihood: retain every REG
model term, add a postseason-only intercept, and share only the move coefficient with an auxiliary postseason market-only logit. Do not fabricate postseason model/flag columns: no
matching cached margin-model predictions were found. Keep every headline grade on the original REG frozen-opener population; postseason games are training evidence only and all
games from the held-out/later season are excluded. **Measured inputs:** `data/raw/20260908T162105Z/schedules.parquet` (4,902 rows);
`data/market/raw/20210105T135500Z/quotes.parquet` (576) is a playoff-source witness; the existing Tuesday/Sunday archive supplies 52 paired postseason games (8/9/9/8/9/9 by
2020–2025). `artifacts/pick_probability/20260929T192747Z/per_game.parquet` has 1,503 REG rows. Protocol C: opener Brier primary, F=3, B=6, K=4, **497 looks**; one auxiliary arm,
unchanged as more eligible seasons accrue. **Read/checked:** MOD-20/21, XLG-09, LEAD-73/76/77. This borrows same-league postseason news-response labels, not older rule-shape
priors, cross-league opener residuals or extra quote rows presented as games. Units: playoff clock/label join, 10–15 calls; joint likelihood replay, 20–25. Rank rationale: actual
extra independent games with no new source, but only 52 available games and an explicit transportability assumption. |

## Protocol C

Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting.
Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly
noon boundary. No closing inputs.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing
missing-archive zeros. LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs.
Reused archives are retrospective; freeze survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first;
optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and
`probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`.

Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four
contrasts × IS/OOS/gap × K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly
reward/nominee Brier; 88 adds all-game/last-game total MAE. No unlisted variants.
## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead87_unit1.py` completed, exit 0, after two guard diagnostics: cache arm disambiguation and separation of pregame from prior-season chronology. Resource caps: two threads; one process.
**Measured:** docs/lead87_unit1.md contains counts and folds; tests/scratch/codex/lead87_unit1/ contains joins, book evidence, source hashes, cutoff audit, summary, and run logs. UV cache redirected to tests/scratch/codex/lead87_uv_cache because the default cache was inaccessible.
## Record commands
None: unit 1 fits/scores zero looks and makes no signal verdict. Orchestrator alone records after the replay; commands must be bash-compatible and include --plain-summary in pool-player English.
## Next
Joint-likelihood replay under the unchanged declaration, after verifying opener lineage/fold eligibility; this first-unit packet did not fit the auxiliary arm.
## Open
Decisive record, IS/OOS and gap, per-fold coefficients, effect intervals, probability_positive and calibration remain unmeasured. Local postseason source is present; no closure or promotion decision. No Git mutation, registry write, publication, new tests, or served change.
