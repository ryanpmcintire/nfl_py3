# LEAD-71: unit 4 complete; serial records pending

## Goal
Compare market-joint and served total guesses at the historical frozen opener.

## State
**Measured:** 1,472 paired games, 103 week blocks; 98 last-of-week games.
Last games: candidate/served MAE 10.163/10.235; gain +0.071429 points,
95% [-0.326531, +0.438776], probability_positive 0.6538; closer/worse/tied 30/40/28,
exact binomial p=0.281979. All games: MAE 10.490/10.689; gain +0.199049,
[+0.048877, +0.366854], probability_positive 0.9972; 512/500/460, p=0.729525.
**Read:** the unchanged predeclared two-look protocol and IS/OOS gaps are in
[the report](../lead71_unit4.md#predeclared-protocol); its hash and predictions are in
tests/scratch/codex/lead71_unit4/. No registry verdict has been written.

## Tried
**Measured:** .tools/uv.exe run --no-sync python scripts/lead71_unit4.py completed once;
12 seasonal ridge fits, 2,960 joint reconstructions, <=2 threads; exit 0.
Ruff check and format --check passed; read-only record-payload, protocol/hash and
no-comments/docstrings checks passed. Report refresh reused saved rows, with no refits.
Owned changes: scripts/lead71_unit4.py, docs/lead71_unit4.md, this lane.
No new tests, Git mutations, operational jobs, publication or served-card changes.

## Next
Orchestrator reviews reconstruction limits and runs the two commands below serially.
Prior Unit 2/3 commands are preserved unchanged in the Unit 4 report appendix.

## Open
**Inferred:** unresolved_below_power for last-game usefulness; no admissible closure.
All-game gain is positive in four seasons, with the largest gain in 2020; last-game
gain is positive in two, negative in three, tied in one. No independent outer test.
Eight games lack admissible scores in both arms/schemes. Four-term LOSO is retrospective;
total/lattice OOS training uses five prior seasons. This reconstructs season-frozen served
rules, not archived weekly guesses. Week intervals condition on fitted predictions.

## Record commands
Proposed only, Bash-compatible; candidate versus served. Not executed.

```bash
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit4-last --description 'Market joint score tiebreaker versus served; last game each week; two-look family' --source docs/lead71_unit4.md --league nfl --category market --effect=0.0714285714286 --effect-units mae_improvement --standard-error=0.195866397605 --interval-low=-0.326530612245 --interval-high=0.438775510204 --probability-positive=0.6538 --sample-games 98 --sample-blocks 98 --season-start 2020 --season-end 2025 --family LEAD-71-unit4 --classification unresolved_below_power --classification-evidence 'No admissible closure established; descriptive exact null only' --plain-summary 'Using the market score table made the total guess 0.071 points closer on average for last game each week. The result is still uncertain and the pool guesses stay unchanged.' --notes 'Proposed serial record; two primary MAE contrasts; week-block intervals condition on predictions; retrospective four-term base'
.tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit4-all --description 'Market joint score tiebreaker versus served; all eligible games; two-look family' --source docs/lead71_unit4.md --league nfl --category market --effect=0.199048913043 --effect-units mae_improvement --standard-error=0.0811389775323 --interval-low=0.0488773533357 --interval-high=0.366853830509 --probability-positive=0.9972 --sample-games 1472 --sample-blocks 103 --season-start 2020 --season-end 2025 --family LEAD-71-unit4 --classification unresolved_below_power --classification-evidence 'No admissible closure established; descriptive exact null only' --plain-summary 'Using the market score table made the total guess 0.199 points closer on average for all eligible games. The result is still uncertain and the pool guesses stay unchanged.' --notes 'Proposed serial record; two primary MAE contrasts; week-block intervals condition on predictions; retrospective four-term base'
```
