# LEAD-80: tiebreak guesses conditional on a weekly tie

## Goal
Execute the rules/data inventory unit; replay only with authentic local observations.

## State
**Measured:** inventory complete; 149,356 filenames, 0 registered observations,
5 unreadable directories. **Read:** 4 Splash snapshots and 2 field tables are
2026-only; historical entrant cards/guesses/rules remain unverified. Details:
../lead80_inventory.md and ../lead80_source_review.md. Replay stays source-gated.

### Predeclared protocol
**Read:** copied before outcomes from ROADMAP.md:863; full source in ../lead80_protocol.md.
Predeclare 2020-2025 weeks with authentic pool rules, archived guesses/cards for
evaluation, and pre-deadline spread/total captures. First verify closest-total
versus exact-score rules, lock timing, push/Best Pick points and field observations.
Fit the joint discrete score law and field distribution only on earlier seasons;
held-out opponent cards/guesses are evaluation outcomes, never pre-deadline inputs.
With sides and Best Pick selection fixed, choose the guess minimizing the verified
rule loss conditional on a weekly score tie, integrating uncertainty in the field.
Protocol B: chronological LOSO, paired ordinary prediction baselines plus the
same-law unconditional guess, IS/OOS gap and season-block intervals. B=3 (joint
score fit, field fit, conditional-choice arm), F=6, E=2 (joint-score log score and
actual tiebreak utility); 445 counted looks including the additional
unconditional-guess comparator (35 cells beyond Protocol B).

Inventory metadata first; require issuance/ingestion before the pool deadline;
grade authentic frozen openers. No retrospective source substitutes. Exclude
target/later seasons from fitting; separate earlier training, selection and
calibration; unavailable training means unavailable fold. Reuse certified pregame
predictions, choose coefficients/penalties earlier without new grids, freeze before
prospective evaluation. One calibrated discrete-margin probability selects sides.
Pair combined/model-only/market-only/Elo; report decisive record first, IS/OOS
accuracy/Brier/log loss/margin MAE and gap, fold coefficients, season stability,
five training-quantile reliability bands, season-block 95% intervals and
`probability_positive`. Zero crossing closes nothing: `unresolved_below_power`.
L=(B+36+8E)(F+1)+25+35=445; reporting cells count, nested identical refits add no
specification, and new splits/specifications need preregistration. No assumed-field
simulation may be reported as measured pool improvement.

## Tried
**Measured:** `UV_NO_CACHE=1 UV_OFFLINE=1 .tools/uv.exe run --no-sync python scripts/lead80_unit1.py`
ran once, exit 0. No fit/score: 0 decisive games, 0 outcome looks, 0/6 folds.
IS/OOS/gaps, coefficients, 95% intervals and `probability_positive`: not estimable.
**Measured:** Ruff passed; scoped whitespace check passed. No tests added.

## Record commands
None: inventory only, no effect estimate or research verdict to register.

## Next
Orchestrator obtains authentic 2020-2025 rules and entrant cards/Best Picks/guesses,
then verifies frozen spread/total captures and issuance/ingestion before deadlines.

## Open
Missing historical sources; exact loss and guess deadline; five scan access gaps.
Mechanism remains open. No registry write, served change, commit or push.
