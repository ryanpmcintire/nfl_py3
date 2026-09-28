# Seven backlog categories: September 28 completion report

The owner requested work on all seven remaining categories. This report separates
implemented and verified work from observations that cannot exist yet. No result
in this session selects a new served model or retunes the enrolled future study.

## Independent validation

**Measured:** the Monday study job missed its former 15-minute window behind the
noon forecast/injury work. Its window is now 60 minutes, without changing any
per-game deadline. All ten capture aliases completed through real scheduler argv;
the active Week 3 forecast correctly returned zero eligible study rows. The
Week 4-18 enrollment remains 224 games, source hashes unchanged, with interim
performance withheld. This operational rehearsal is not an actual Week 4 capture.
The first real capture must be checked after Tuesday's fresh-data lock.

## Simulator

**Measured:** historical overtime now ends after a failed second possession and
removes conversions after a winning touchdown. A separate defensive-return flag
handles transitions that omit possession change. Production-call probes cover
punts, downs, offensive/defensive touchdowns, safeties and answering field goals.
All three arms and 30,006 simulations are preserved; final engine log-loss
improvement versus the naive training histogram is -0.000656
[-0.013844, +0.014832], probability_positive 0.370370. Only 2/5 key-number bands
pass. All 147 diagnostics and four accepted unresolved registry records are in
[the complete simulator report](simulator_overtime_repair_20260928.md).
Correct rules alone have not made this simulator competitive; it remains unserved.

## Confidence and Best Pick

**Measured:** the readiness audit found 72 historical nominee weeks, only three
new paper nominee weeks, and incomplete paired records/full contender histories.
The declared growth trigger is eight new weeks. Weeks 4-18 are embargoed against
interim performance selection; the old Week 10 retuning suggestion is superseded.
The future recorder now saves the full contender distribution at the same
Tuesday/Sunday instant. A held Tuesday nominee remains fixed while the other
playable candidates are retained. Canonical hashes and uniform provenance reject
altered/ambiguous snapshots; an interrupted write recovers the original observation.
Root's isolated real-data probe retained all 16 Week 3 candidates and preserved
both original ledgers. The readiness audit verified hash rejection and reports
historical rows explicitly. No historical lock-time capture is claimed and no
confidence parameter was fitted. Evidence:
`artifacts/confidence_nominee_readiness/20260928_verified.json`.

## Evaluator power

**Measured:** the predeclared full curve completed on original inputs, 200 fixed
synthetic replicates per coefficient, and 400/2,000/20,000 nested draw prefixes.
It retains all 16 original coefficient cells and 96 diagnostic summaries, with
4,809,600 prediction rows and 9,600 replicate-prefix rows. Four completed
null/strong-control cells were reused by exact hashes. At the strongest control,
20,000-draw detection is DPI 34.5% [28.26%, 41.32%] and holding 75.0%
[68.57%, 80.49%]; both null rates are 0.5% [0.09%, 2.78%].
Every coefficient and uncertainty interval is in [the full curve report](positive_control_precision_curve_20260928.md).
The controls do not match the natural signals' direction and cannot close them.

**Measured:** an audit detected a runner write five seconds after its source hash
was frozen. The exact original bytes were recovered and matched that hash; the
sole difference was presentation sorting of two completed summary lists. Frozen
source, exact diff and provenance are archived without changing the manifest.
Root's independent verifier passed all 16 exact 400-draw reproductions, every
prediction/replicate arithmetic check, all 96 rate/interval summaries, reused
artifact hashes and unchanged input hashes. The lane has advanced; its original
predeclaration remains hash-verified in the artifact.

## Pool field inputs

**Measured:** the full fixed Week 1-2 population is 32 games. All remain unavailable
for fitting: 31 historical public captures record request start without proof of
response completion, and DET at BUF has no saved split before its deadline.
Tuesday-noon and Thursday-noon public captures have been added; both real job
commands succeeded with 16 games. Tuesday precedes the normal card publication. Historical inputs are not backfilled or silently dropped.
The stricter real fit command stops on ATL at PIT's baseline binding after
publication before any fit is run. Exact source proof requires one bookmaker's opposing lines and both recorded
prices at the saved observation time. A game-level timestamp cannot establish
this: schedule odds may remain when the pool supplies a different spread.
Nominal pool forecast prices remain lawful; unproven market baselines are unavailable.

## Input reliability

**Measured:** completed-snapshot freshness now accepts valid suffixed timestamps
and ignores incomplete/malformed captures; future-dated captures fail freshness. Private odds age skips malformed and
future manifests. Bovada validates nested provider data and reports totals-only
responses as missing spreads. UTC normalization fixes aware non-UTC injected
capture clocks. Production probes retained all 60 rows from an archived Bovada
response; a live capture returned 44 quotes for 11 games. The failed sandbox
public capture left an incomplete directory that freshness correctly skips.
The final daemon is PID 28232, with loaded code/schedule hashes verified. All new
jobs report success; the September 22 weekly-lock miss remains acknowledged.

**Measured:** public-betting team arrays are unordered. Explicit game team IDs
correct 8/16 home/away labels in an actual saved page while retaining every numeric
percentage. Capture now saves team IDs, each side's spread/price/live flag, and
uses response-received time. All 16 source pairs are retained, including two
nonopposing consensus lines. Malformed/empty pages fail before a completion
manifest is written. An independent review caught Windows newline translation
changing the saved raw-file bytes; the manifest now hashes the actual saved file.
The final `20260928T174637Z` capture has 16 verified rows and matching raw/index
hashes. Two earlier captures with incorrect raw-file hashes remain unchanged
and are excluded by proof validation. Evidence:
`artifacts/input_reliability/20260928_seven_backlog/`.

## Prospective coverage and visible history

**Measured:** future published records now bind the exact forecast, creation time,
original line and both immutable file hashes. Existing locked rows are preserved.
Raw-model and market eligibility are reported separately; unavailable inputs stay
visible. The real scorecard runs over all 48 saved picks: 28-19, one pending.
Exact-source audit supports 24 raw baselines but only three market baselines;
fifteen raw-eligible rows lack quote timestamps and six further rows lack a matching
source price pair. Old public indexes can be re-extracted in memory from their
immutable raw HTML with explicit team IDs; this repairs parsing without substituting
later information, but cannot prove response completion before publication.
All 32 pool rows remain visible and unavailable. Two incorrect declared raw hashes
fail validation; the final response-received capture passes.

**Measured:** `artifacts/prospective_scorecard/20260928T175751Z/` preserves all 48
predictions. Root independently recomputed 80 paired metric values, all four served
periods, and verified all 26 source hashes. The 47 decisive served picks have Brier
0.238437 and log loss 0.669702. On the 23 decisive raw-matched games, served versus
raw Brier is 0.251553 versus 0.240956, and log loss 0.696079 versus 0.674755.
On only three exact-market-matched games, served/raw/market Brier is
0.230852/0.227013/0.252214 and log loss 0.654252/0.646846/0.697575.
The neutral baseline is Brier 0.25 and log loss 0.693147 on each population.
These descriptive comparisons do not establish a stable edge or select a model;
all reliability tables and counted looks remain in the artifact. This supersedes
the earlier timestamp-only market comparison and the earlier raw cover-mass read.

**Measured:** current History content and all 48 rendered rows already agree with
frozen published sides, lines and probabilities. Its reader-facing heading now says
"Saved cover chance," with a tooltip and explanation that model updates retain
these original estimates. Publication completed and the rendered-page diff was
reviewed: History adds this explanation, Books now reflects the newest verified
capture, injury age advances, and Findings includes the four unresolved simulator
records. No frozen side, line, probability, or Best Pick changed.

## Verification and remaining observations

**Measured:** final `ruff format --check .` passed (1,165 files), `ruff check .`
passed, `mypy src` passed (252 source files), and
`pytest -q --basetemp .tmp/pytest-seven-final` passed: 1,645 tests, 91 warnings.
All commands used the locked Windows environment via `.tools/uv.exe run --no-sync`.
No test files or functions were added. Production commands exercised the scheduler,
capture, scorecard, readiness, simulator, and publication paths; isolated probes
checked interrupted contender writes, tampered snapshots, and provider failures.
Root verified unchanged locked picks and enrollment hashes. Independent code
reviews found no remaining defect or redundant contract test to propose removing.
Handoff refresh and publication commits complete the repository checkpoint.

Future games, missing historical predeadline records, and unresolved simulator
accuracy remain genuine limits. They are not grounds to invent data, choose the
best historical cell, or tune against the frozen future evaluation.
