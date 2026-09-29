# Backlog sweep — 2026-09-29

## Goal
Classify open ROADMAP rows excluding LEAD-66–81; prepare at most six local worker packets.

## State
**Measured:** 42 eligible rows including eight concurrent additions (LEAD-82–89); final inventory/packet/look-arithmetic check PASS: 13 a, 6 b, 8 c, 15 d, no omissions/duplicates, zero experiments. **Inferred:** ranks estimate held-out gain/work, not measured gains. Classes concern remaining work, never signal closure: a = local; b = future observations; c = source/access prerequisite; d = specified unit completed/superseded/duplicated.

| Rows | Class/rank | Evidence / disposition |
|---|---|---|
| LEAD-82 | a / 1 | **Measured:** 3,820,296 cached quotes; **inferred:** omitted Tuesday information, cheap term. |
| LEAD-83 | a / 2 | **Measured:** paired quote witnesses exist; **inferred:** shrink noise around the existing move input. |
| LEAD-85 | a / 3 | **Measured:** 1,503 probability / 1,537 opener rows; no source join needed. |
| LEAD-86 | a / 4 | **Inferred:** compact training-loss change using verified quote inputs. |
| LEAD-84 | a / 5 | **Inferred:** precise distribution-aggregation mechanism, higher construction cost. |
| PER-09 | a / 6 | **Measured:** 7,413 prior-season ST ratings; **read:** ATS follow-up open (docs/st_player_ratings.md:175). |
| LEAD-28, LEAD-29, LEAD-43, LEAD-59, LEAD-87, LEAD-88, LEAD-89 | a / below six | **Measured:** 3,107 coordinator assignments, 12 adjudicated changes, 47 Denver base games, 23,962 penalty-type rows; postseason/total/three-stage quote witnesses exist. **Inferred:** sparse effects, indirect gain, or greater cost. |
| MKT-08, MOD-07, POL-10, LEAD-53, LEAD-64, LEAD-65 | b | **Read:** news/refresh, stack, nominee, designation-agreement and early-window prospective pairs pending. Leave the 2026 cohort alone. |
| PER-03, LEAD-55, ENV-02, MOD-10 | c | **Read:** injury credentials/revisions, roof issuance chronology/status, or reliable dated player-state prerequisites. |
| POL-05, POOL-01, LEAD-61, OPS-06 | c | **Read:** actual pool field/prizes/picks, same-book half-line coverage, or external-host access/deployment prerequisites. |
| XLG-09, PER-10, MOD-09, MOD-22 | d | **Read:** cached units scored; drive proposal redirected by docs/play_level_audit.md:162. No new family closure. |
| PER-07, MOD-17, MOD-18, MOD-19, POL-11, POL-12, UI-20 | d | **Read:** specified implementation units completed; coordinator follow-ups belong to LEAD-28/29. Paired-opener producer is done (docs/lanes/done/mod18-paired-opener-evaluation.md:6), superseding ROADMAP's note. |
| SIM-08, OPS-02, ENG-45, POL-08 | d | **Read:** declared simulator repair done (docs/lanes/sim08-simulator-rebuild.md:43); retention inventory done/deletion awaits approval; replay units done; field simulation duplicates POL-05/POOL-01. Simulator calibration remains unresolved. |

## Tried
**Measured:** scoped cs.ps1/cr.ps1 reads; bounded extraction recovered shortened rows. `.tools/uv.exe run --no-sync --no-cache python -` metadata checks passed; default uv cache was inaccessible. Inputs: **N** `artifacts/pick_probability/20260929T192747Z/per_game.parquet` (1,503); **O** `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet` (1,537); **Q** `artifacts/sharp_book_weighted_movement/spread_quotes.parquet` (3,820,296). Under `data/market/raw/`, verified quote witnesses: `20200908T125500Z/quotes.parquet` (1,862), `20200910T215500Z/quotes.parquet` (1,640), `20200913T162500Z/quotes.parquet` (2,088), `20250907T162538Z/quotes.parquet` (1,808), `20210105T135500Z/quotes.parquet` (576). New-row population counts remain **read**, not independently rejoined here.

## Next
Dispatch these packets; each owns its three named files plus `tests/scratch/codex/<script-stem>/`. No comments/docstrings, new tests, registry writes, serving/publication/Git actions; one job, n_jobs<=2. Freeze before outcomes; run once with `.tools/uv.exe run --no-sync python scripts/<script-stem>.py`. All inherit **Protocol C** (docs/lanes/ideation-2026-09-29c.md:9): historical opener proxy, four-term/model-only/market/Elo comparisons, separate earlier-season fit/selection/calibration, one calibrated discrete probability, decisive records, accuracy/Brier/log loss/RPS, reliability, IS/OOS gaps, coefficients, blocked 95% intervals and probability_positive. Zero crossing closes nothing. Exact bash-compatible record commands with --plain-summary, pool-player English and report --source go in each lane for the orchestrator.

**1 — Tuesday block.** Lane `docs/lanes/backlog82.md`; files `scripts/backlog82.py`, `docs/backlog82.md`. Use N/Q; restore matched-leader-book Tuesday-noon-to-Wednesday-midnight signed move beside the later move. Pin clocks/pre-kickoff rules; inherit LEAD-82's three books. Fit 2023, calibrate 2024, outer 2025; Brier primary, **261 looks** (C: F=1/B=6/K=4). One outer year establishes no season stability. Save record commands in lane.

**2 — Move uncertainty.** Lane `docs/lanes/backlog83.md`; files `scripts/backlog83.py`, `docs/backlog83.md`. Use N and Tuesday/Sunday quotes; verify LEAD-83's declared 1,311-game subset before outcomes. Add median move multiplied by training-only tau²/(tau²+leave-book-out variance), alongside original move; no book ranking/outcome bands. Brier primary, **501 looks** (C: F=3/B=7/K=4). Save record commands in lane.

**3 — Best Pick uncertainty.** Lane `docs/lanes/backlog85.md`; files `scripts/backlog85.py`, `docs/backlog85.md`. Use N/O and `data/processed/game_features_weak_stack.parquet` (**measured:** 4,902 rows). Integrate earlier-fit coefficient covariance with fixed 20-node quadrature; calibrate separately and mix discrete PMFs. Preserve contenders/push rules; no confidence haircut. Nominee Brier primary, weekly reward secondary; **713 looks** (C: F=3/B=6/K=6). Save record commands in lane.

**4 — Market regularization.** Lane `docs/lanes/backlog86.md`; files `scripts/backlog86.py`, `docs/backlog86.md`. Use N and Tuesday/Sunday quotes; verify LEAD-86's 1,309 complete-product games. Keep four predictor terms; penalize training probability toward the timestamp-matched market lattice with weights exactly 0/0.1/1, selected before separate calibration/outer scoring. Brier primary, **505 looks** (C: F=3/B=8/K=4). Save record commands in lane.

**5 — Book-lattice mixture.** Lane `docs/lanes/backlog84.md`; files `scripts/backlog84.py`, `docs/backlog84.md`. Use N/Sunday quotes; verify LEAD-84's 1,321 complete panels. Equal-weight per-book PMFs with an earlier-season prior; add mixture-minus-averaged-constraints logit as one fitted term. No weight search. RPS primary plus opener/calibration endpoints; **505 looks** (C: F=3/B=8/K=4). RPS alone authorizes no side change. Save record commands in lane.

**6 — Special teams.** Lane `docs/lanes/backlog-st-ats.md`; files `scripts/backlog_st_ats.py`, `docs/backlog_st_ats.md`. Use N, `artifacts/st_player_ratings/season_lagged_2019_2024/ratings.parquet`, `data/players/participation/raw/20260813T131635Z/` (**measured:** 478,989 rows). One home-minus-away prior-snap-weighted ST-rating term; reuse `scripts/latent_ratings_on_production.py`'s aggregation recipe read-only. Retain missing-season/base fallbacks and every chronological fold. Brier primary; **497 looks** (C: F=3/B=6/K=4); report coverage metadata separately. Save record commands in lane.

## Record commands
None here: planning only. Future commands require measured estimates; no placeholder records.

## Open
Only this lane was written. Rankings are **inferred**; reused discovery seasons are not independent confirmation. Clock, upstream-fit provenance and source-subset validation remain runtime gates; mismatches require pre-outcome amendment, never silent changes. Orchestrator owns dispatch/commits.
