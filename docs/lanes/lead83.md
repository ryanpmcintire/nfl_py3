# LEAD-83 — shrink noisy move measurements

## Goal
Execute the declared variance-feature unit and its cached replay when source and upstream chronology permit, within the worker packet; no served changes or registry writes.

## State
Protocol frozen before any outcome computation. **Measured:** variance-feature unit complete: 1,311 declared games plus 33 retained pushes; 131 Tuesday/131 Sunday files (145,340/171,190 rows). **Read:** declaration below is ROADMAP.md:866 and ideation-2026-09-29c.md:9–16. Cached replay remains unscored.

## Protocol (predeclared)
**Inferred mechanism:** post-Tuesday news makes the pool opener stale, but a move supported by disagreeing books is a noisier estimate of that news than an equally sized unanimous move. Predeclare a game-level empirical-Bayes move: multiply the matched-book median move by tau²/(tau²+v), where v is its leave-book-out jackknife variance and tau² is nonnegative between-game signal variance estimated only on training seasons; no book ranking or outcome-selected bands. Add the shrunk move alongside the original move in the calibrated probability. Fixed population: the 1,311 target-week games with at least two common books before Sunday 12:30. Inputs: `artifacts/pick_probability/20260929T192747Z/per_game.parquet` and the 131 Tuesday/131 Sunday quote files, including `data/market/raw/20200908T125500Z/quotes.parquet` and `data/market/raw/20200913T162500Z/quotes.parquet`.

Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. No closing inputs. Use the historical opener as the frozen pool-line proxy and the served four-term probability as the base; no pre-2026 pool captures.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros. Verify upstream training cutoffs. Reused archives are retrospective; freeze survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`. Primary endpoint: opener Brier. F=3, B=7, K=4; L=(27K+B+4)(F+1)+25=501 looks. Book rows never multiply independent games. No unlisted variants.

## Tried
- Frozen implementation: tau²=max(0, sample variance of training game medians − mean jackknife variance); zero weight if tau²+v=0; train through Y−3. Preserve pushes; require the declared 1,311 non-push games.
- **Measured:** `.tools/uv.exe run --no-sync --no-cache python scripts/lead83_unit1.py` exited 0 once; scoped Ruff format/check pass. Cached verification corrected the variance fit to retain pushes (Protocol C), without outcomes or a repeat source scan. Final tau²=2.31476587/2.71310943/2.37679326 for outer 2023/24/25; variance-fit n=198/414/630, outer non-push n=233/232/232. Zero outcome looks; IS/OOS/gap, probability coefficients, intervals, decisive record, and probability_positive remain unestimated. `cache-verification.log` supersedes the original variance estimates in `run.log`.
- **Read:** `src/nfl_ats/clv.py:2177–2189` trains the named opener parent weekly on all completed prior games. **Measured:** parent rows have no training-cutoff columns. **Inferred:** this cache cannot substantiate Protocol C's fixed upstream Y−3 cutoff; no score or research verdict follows.

## Record commands
None: feature construction supplies no outcome estimate or research verdict to record. The orchestrator alone runs subsequent bash-compatible commands with `--plain-summary` in pool-player English.

## Next
Fresh-thread cached replay: reconstruct/verify fold-specific upstream discrete probabilities under Protocol C before fitting the five arms. Reuse `tests/scratch/codex/lead83_unit1/`; report is `docs/lead83_unit1.md`.

## Open
Named sources are present; a protocol-compliant upstream cache is not established. No full-history rebuild, registry write, served change, Git mutation, or new test. IS/OOS findings remain outstanding; no signal is closed.
