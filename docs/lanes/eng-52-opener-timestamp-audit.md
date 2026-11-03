# ENG-52

## Goal
Opener timestamp audit: capture time versus first public posting, with graded results split by capture lag (ROADMAP ENG-52).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: An opener captured hours late is partly a moved line; the fair opener test assumes it is the first number. Report the lag distribution per game and regrade the fair test with late captures excluded. Capture lag enters as a grade filter, never a model term.

## Tried
Novelty check only. Nearest: ENG-51 (input provenance guard), LEAD-64 (capture integrity), MKT-08; grep 'timestamp drift' found none. Differs: Capture lag is a grade input, not a model input.

## Next
First unit: Compute lag per graded game from capture timestamps and any archived first-post times; regrade with and without late games with a paired bootstrap.

## Open
Data: Local capture logs (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
