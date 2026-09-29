# LEAD-82 unit 1: Tuesday boundary and clock extraction

**Measured:** boundary/clock unit only; no outcomes or closing inputs loaded, no fitting or scoring.
**Read:** ROADMAP.md:865 splits boundary/clock extraction from the later cached fitted replay.
The declaration was saved in `docs/lanes/lead82.md` before execution and remains unchanged.

## Frozen protocol

Median same-leader-book signed change from the last quote at/before Tuesday noon Eastern
to the last quote before Wednesday 00:00 Eastern. Add this term beside Wednesday-to-decision
movement in the existing four-term discrete conditional probability. Historical opener is
the frozen pool-line proxy. Fit 2023, calibrate 2024, outer 2025; existing ridge 0.001.
Primary Brier; accuracy/log loss/RPS, five arms, B=6, K=4, F=1, 261 declared looks.
**Measured:** 0 outcome looks and 0 fits executed. One outer season cannot establish season stability.

## Source and coverage inventory

**Measured:** quote cache 3,820,296 rows; frozen fit 1,503 games; opener source 1,537 games including pushes.
No outcome filter was applied to the opener source. Source gaps remain missing rather than becoming zero moves.
Tuesday boundaries use only `intraday_hourly`; scheduled `tue_open` captures are not noon-boundary evidence.
**Measured:** exact counts below; confidence intervals are not applicable to this finite local inventory.

| Season | Openers incl. pushes | Frozen fit | Opener early | Opener later | Fit early | Fit later | Fit both | Fit book pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 227 | 220 | 0 | 225 | 0 | 218 | 0 | 0 |
| 2021 | 239 | 236 | 0 | 237 | 0 | 234 | 0 | 0 |
| 2022 | 255 | 248 | 0 | 253 | 0 | 246 | 0 | 0 |
| 2023 | 272 | 266 | 272 | 272 | 266 | 266 | 266 | 783 |
| 2024 | 272 | 266 | 272 | 272 | 266 | 266 | 266 | 781 |
| 2025 | 272 | 267 | 272 | 272 | 267 | 267 | 267 | 785 |

**Measured:** 2,349 same-book boundary pairs for frozen-fit games in 2023-2025; 2,398 including the wider opener source.
**Measured:** reconstructed later moves compared on 799 frozen-fit games; 164 changed; maximum absolute difference from the saved feature = 2.5; availability mismatches = 0.

## Runtime timing checks

**Measured:** every saved pair passes target-week, source-season, observation, snapshot,
bookmaker-update and both source/schedule kickoff gates. Noon is inclusive; Wednesday and decision boundaries
are strict. Decisions are capped at the earlier of kickoff and Sunday 12:45 Eastern.
Same-book early plus later changes telescope to noon-to-decision movement; zero violations.
Both arms' later move is rebuilt with the existing `sunday_move` implementation after these gates.
**Measured:** 1,058,246 selected quote rows; 431,744 outside one or more gates; 313,148 eligible observation/source rows.
Gate rejection counts overlap and include snapshots outside the target-week decision window.
Archived kickoff timestamps may differ from the final schedule; quotes must precede both clocks, with no exact-equality requirement.

| Gate | Rejected quote rows |
| --- | --- |
| spread_market | 0 |
| finite_line | 0 |
| source_season | 0 |
| target_week | 423456 |
| book_before_observation | 0 |
| observation_before_snapshot | 0 |
| before_decision | 8288 |
| before_source_kickoff | 4362 |

## Replay handoff

**Measured:** in-sample/out-of-sample metrics, gaps, coefficients, decisive-game records,
intervals and `probability_positive` are not estimated in this outcome-free unit.
Before the separately declared fitted replay, verify upstream model/discrete-distribution
training cutoffs at row level and retain the opener pushes for distribution/RPS work.
Then use the unchanged fit/calibration/test split and paired baselines from the lane.
**Inferred:** this inventory establishes feature availability only; it makes no claim of a useful signal.
No research closure, registry entry, promotion or served-card change follows from coverage.

Saved outcome-free feature rows: `tests/scratch/codex/lead82_unit1/features.parquet`.
Saved per-book boundary evidence: `tests/scratch/codex/lead82_unit1/boundary_pairs.parquet`.
Source hashes, exact counts and protocol hash: `tests/scratch/codex/lead82_unit1/summary.json`.
Run: `.tools/uv.exe run --no-sync python scripts/lead82_unit1.py` with `UV_CACHE_DIR` set to a writable temporary directory.
