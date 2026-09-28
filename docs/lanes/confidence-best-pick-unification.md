# Confidence and Best Pick unification

## Goal
Assess whether the one served probability ranks games and calibrates Best Pick,
using complete prospective contender distributions and chronological evaluation.

## State
**Measured:** the future Tuesday/Sunday recorder now preserves all candidates,
probabilities, eligibility, deadlines, UTC observation time and immutable hashes.
A held Tuesday nominee stays selected while Sunday's other playable candidates
are retained. Interrupted writes recover the original observation; changed rows,
ambiguous phases and inconsistent metadata fail both recovery and readiness.
The real-data isolated probe retained all 16 Week 3 candidates without changing
historical ledgers. No historical capture was backfilled or parameter fitted.

## Tried
**Measured:** coverage-only audit: 72 historical weeks, three new paper nominees,
two dedicated Tuesday nominees, one Sunday nominee, no historical full contender
ledger. Week 2 lacks the dedicated pair; Week 1's dedicated and paper nominees
differ. Readiness remains false; no outcomes were read. Evidence:
`artifacts/confidence_nominee_readiness/20260928_verified.json` and
[the seven-category report](../seven_backlog_completion_20260928.md).
Historical ranking/calibration cells remain unresolved in
`../confidence_best_pick_sunday_matched.md` and `../confidence_top_calibration.md`.

## Next
Verify the first genuine future contender capture. Run
`.tools/uv.exe run --no-sync python scripts/confidence_nominee_readiness.py --output .tmp/confidence_nominee_readiness.json`.
The +8-week growth trigger does not authorize interim tuning: the frozen Week 4-18
embargo supersedes the old Week 10 suggestion. Any later calibration needs a new
protocol, complete inputs and an untouched outer evaluation after season completion.

## Open
Past missing lock-time distributions cannot be reconstructed as historical facts.
Future collection remains pending; ranking/calibration are unresolved, not closed.
Zero crossing closes no signal; one fitted probability continues selecting the side.
