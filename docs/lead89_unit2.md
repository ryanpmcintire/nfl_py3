# LEAD-89 unit 2 - missing stage-timed upstream forecasts

**Measured:** the supplied season-held-out base clears the old training-overlap
blocker. A separate input-clock failure prevents the Best Pick replay:
**0 of 816 Tuesday candidates across the 54 declared 2023-2025 scoring weeks**
have both injury inputs available at the decision time. These are source census
counts; confidence intervals do not apply.

## Amendment and unchanged protocol

**Read:** the lane amendment was saved before the clock census or any scoring.
It pins `artifacts/loso_upstream/20260929T235904133782Z/`, replacing the rejected
companion margins. The supplied upstream/calibration LOSO is retrospective;
it does not establish the original chronological upstream training claim.
The exact declaration and amendment are preserved in
`tests/scratch/codex/lead89_unit2/protocol_before_run.md` with a digest.

The declared population remains 2020-2025 historical openers, retaining pushes
and using row-specific source-complete subsets. Outer scoring seasons remain
2023/2024/2025; downstream fit through Y-3, tune Y-2 and calibrate Y-1.
The three decisions remain Tuesday noon, Thursday's archived pre-kick capture
and Sunday 12:30. One legal nominee is retained; locked nominees cannot change.
Waiting compares an early nominee's expected reward with the expected best
still-playable reward from one earlier-season joint weekly transition law.
One fitted calibrated discrete probability chooses every side. No policy,
threshold, metric or look-budget change was made after inspecting outcomes.

Candidate, early-lock four-term recipe, model-only, timestamp-matched market
and Elo remain the five arms. Required metrics remain opener accuracy, Brier,
log loss, RPS, weekly Best Pick reward and nominee Brier, with IS/OOS/gaps,
fold coefficients, five reliability bands, season/week uncertainty and
probability_positive. The family remains F=3, B=7, K=6:
L=(27K+B+4)(F+1)+25 = **717 looks; zero outcome looks consumed**.

## Measured source checks

**Measured:** all 1,537 frozen-opener games across 107 weeks match one-to-one.
Prediction, feature-table and opener digests pass. All five saved season-overlap
counters are zero; the upstream and calibration training-season lists exclude
the prediction season. All 61,075 saved quote rows pass observation, bookmaker,
market-update and kickoff clock checks. No outcome columns were loaded.

**Read:** `scripts/build_loso_upstream.py:88` loads the frozen feature table;
`:102` joins its features to the opener games; `:293` predicts from those rows.
The artifact's margin-feature list includes injury terms.
`docs/loso_base_artifact.md:308` leaves early-stage feature availability
unverified. Unit 1 requires that check at `docs/lead89_unit1.md:91`.
AGENTS.md:41 requires information available before the prediction timestamp.

**Measured:** each row below compares both teams' injury observation timestamps
with the unit-1 decision timestamp. Late and missing can overlap; the union
counts each game once. Passing this injury clock does not certify other inputs.

| Population | Stage | Games | Late injury | Missing timestamp | Unavailable union | Available | Weeks with any available game |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-2025 | Tuesday | 1,537 | 1,534 | 11 | 1,536 | 1 | 1/107 |
| 2020-2025 | Thursday | 1,522 | 1,389 | 6 | 1,389 | 133 | 100/107 |
| 2020-2025 | Sunday | 1,354 | 27 | 6 | 33 | 1,321 | 107/107 |
| Outer 2023-2025 | Tuesday | 816 | 814 | 5 | 816 | 0 | 0/54 |
| Outer 2023-2025 | Thursday | 806 | 739 | 2 | 739 | 67 | 53/54 |
| Outer 2023-2025 | Sunday | 714 | 23 | 2 | 25 | 689 | 54/54 |

These are complete censuses, not estimated rates. Timestamp bases and proxy
flags accompany the saved game/stage audit. A proxy timestamp establishes a
provenance limitation, not when an injury actually happened.

**Inferred:** adjusting only the four-term market movement cannot remove later
injury information already embedded in the upstream forecast. A source-complete
subset does not rescue the declared evaluation: no outer scoring week has an
eligible Tuesday nominee. This is a missing-input finding, not a failed or
refuted waiting mechanism.

## Missing source and uncomputed results

Required source: season-held-out **stage-specific upstream forecasts** built
from inputs available by Tuesday noon, Thursday's pre-kick capture and Sunday
12:30, with input observation provenance and discrete opener cover/push masses.
Their four-term calibration must also exclude the prediction season. The
supplied per-game artifact has no replacement early-stage forecasts. Earlier
injury snapshots alone require matching forecasts; zeroing or backdating the
existing inputs would not reconstruct the declared model.

The weekly W-L-P records for both policies, differing-week records, exact paired
null, Wilson intervals, effect intervals, probability_positive, IS/OOS/gaps,
calibration and policy coefficients are **uncomputed**. No candidate-versus-served
result exists to register. No record command is supplied or run, and no signal
is closed. AGENTS.md:65-85 does not permit unsupported research closure.

## Verification and saved outputs

```bash
UV_CACHE_DIR=tests/scratch/codex/lead89_uv_cache .tools/uv.exe run --no-sync python scripts/lead89_unit2.py
UV_CACHE_DIR=tests/scratch/codex/lead89_uv_cache .tools/uv.exe run --no-sync ruff check scripts/lead89_unit2.py
UV_CACHE_DIR=tests/scratch/codex/lead89_uv_cache .tools/uv.exe run --no-sync ruff format --check scripts/lead89_unit2.py
```

**Measured:** one completed input audit, exit 2 (required source unavailable);
zero scored replays. An initial command attempt exited 1 on an incorrect
reporting assertion that no Tuesday game survived anywhere. That assertion was
removed; the census shows one historical survivor and zero outer-season
survivors. Neither attempt read outcome columns or fitted/scored a policy.
Both final Ruff commands exit 0. No tests were added.

`tests/scratch/codex/lead89_unit2/` contains `input_clock_audit.parquet`,
`stage_counts.csv`, `season_counts.csv`, `summary.json`, and the pre-run protocol.
The summary hashes the exact script and sources. The completed-command log is
`tests/scratch/codex/lead89_unit2_verified.log`; the initial failure is retained
in `tests/scratch/codex/lead89_unit2.log`.

The script is a reproducible blocker audit, not a completed policy evaluator.
The next unit must obtain the named source and then implement and execute the
unchanged transition/policy replay. No pipeline rebuild, registry write,
publication, serving change or Git mutation occurred.