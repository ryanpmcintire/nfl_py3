# MOD-18 paired opener evaluation

## Goal
Compare FanDuel and consensus opener probabilities on identical eligible games,
with completed-prior-season fitting, calibration, and joint uncertainty estimates.

## State
The evaluation core and strict paired-price extractor are implemented. The prior-season input producer is in progress; authenticated end-to-end evaluation remains open.
**Measured:** the root verification passed with 16 synthetic paired rows, 12 jointly
decisive rows, two pushes per arm, and four calibration folds. Changing held-out
outcomes left that season's probabilities and coefficients unchanged; shuffling
input rows left outputs unchanged. Fourteen malformed input cases were rejected.
These are arithmetic and chronology checks, not research results.

## Tried
- `src/nfl_ats/paired_opener_evaluation.py` retains game predictions, fold
  coefficients and training seasons, calibration tables, market/model baselines,
  paired decisive metrics, and joint season-bootstrap draws.
- **Measured:** the root callable verification, scoped Ruff, and scoped mypy
  exited 0; evidence is in `.tmp/mod18-paired-core-root*.log`.
- **Read:** `.tmp/mod18-paired-producer-design.md` requires both validated opener
  manifests, identical feature/raw-source identities, exact line-bound price
  pairs, and separate strictly prior-season margin and lattice support.
- **Measured:** final synthetic price-pair verification rejected malformed timing, missing event identity, duplicate legs, and mismatched lines; two-book consensus arithmetic and all five row-order hashes matched. Evidence: `.tmp/mod18-paired-price-final-verification.md`.
- Price availability and the first eligible evaluation year are being checked
  before fitting. Missing prices must never become default -110 quotes.

## Next
Finish price/support preflight, implement the authenticated input producer, then
verify its chronology and probability components before any research evaluation.
Run repository checks at the next reviewed code checkpoint.

## Open
The core trusts caller-supplied probabilities and provenance; it is not a serving
path or an authenticated end-to-end evaluation command. Pre-2020 lattice support
is not established. No signal is closed and no card probability changes here.
