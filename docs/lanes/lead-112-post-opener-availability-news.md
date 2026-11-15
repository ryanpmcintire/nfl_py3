# LEAD-112

## Goal
Post-opener availability news: change in read availability between the Wednesday report and the Friday and game-day reports, against the line move (ROADMAP LEAD-112). Research only; one fitted probability selects every side; nothing here changes a pick.

## Declaration (2026-11-15, written before any sign was computed)
Family `lead112_post_opener_availability`. 2026 only, so coefficients are leave-one-week-out. Population: graded games of weeks 2-9 in artifacts/forward_ledger_core/ledger.parquet (week 1 has no prior ladder or snap weights; counted, excluded). Language reads: 0 games have a read at two times (Thursday view covers only the Thursday game, Sunday view the rest), so the term is ladder-based. Late term per team = sum over players of (P(plays | state at first team report) - P(plays | state at last snapshot before the deadline)) x prior-week mean snap share; game term = home minus away. P(plays) from expanding prior-week ladders (add-one), LEAD-106 outcomes; unlisted = `none`. Base: served card probability at the deadline (per_game holds no 2026 rows). Move = deadline market minus Tuesday market (lead99 quote machinery; Tuesday captures run 12:00-18:00 ET, so the Tuesday market is the day's latest quotes; weeks without a Tuesday capture excluded). Arms 3 (move only; designation late + move; designation x practice late + move) x metrics 2 (log loss, Brier vs served base) x subsets 2 (all; nonzero term) = 12 looks, plus 4 reference (raw model, intercept-only) = 16.

## State (2026-11-15, all measured by scripts/lead112_post_opener_availability.py, output tests/scratch/lead112/result.json)
- 136 ledger games; excluded 16 (week 1), 15 (week 5, no Tuesday quote capture), 0 for missing revision history; 105 eligible, 7 week blocks. 2276 listed player-weeks; 972 designation changes after the first report, every one from `none` (481 to Q, 458 to Out, 33 to D): the Wednesday report carries no game designations, so the late term is nearly the whole designation.
- Improvement over the served card (week-block bootstrap, P+), all games: move only LL -.0095 [-.0201, -.0022] P+ 0, Brier -.0045; designation LL -.0200 [-.0344, -.0070] P+ 0, Brier -.0097 [-.0168, -.0034]; desig x practice LL -.0217 [-.0379, -.0094] P+ 0, Brier -.0105. Nonzero-term subsets the same within .002 (move nonzero 59 games -.0169).
- Within-week permutation null means -.0062 / -.0168 / -.0166, p .62 / .68 / .75: shuffled terms lose the same, so the loss is the cost of fitting on seven folds, not a wrong sign.
- In-sample late coefficient (logit per snap-share unit): designation -.012 (SE .203), desig x practice -.033 (SE .221), expected sign; move -.084 (SE .183). Folds: designation -.11 to +.13, sign unstable. In-sample LL gain .0010-.0011; out-of-sample gap .021-.023.
- Decisive records (side differs from card): move 5-3, designation 9-12, desig x practice 10-11.
- Reference: served base LL .6987, Brier .2525, 53-52; raw model better than served by LL +.0121 [-.0047, +.0306] P+ .90, 57-48 (decisive 19-15); intercept-only -.0085 P+ .35.
- Registry: 12 cells `lead112_*` unresolved_below_power in registry/weak_signals/lead112_post_opener_availability.json (record exit 0); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.

## Tried
- Unit 1 (2026-11-15): `scripts/lead112_post_opener_availability.py` (imports lead99_forward_decision_read and lead106_injury_language unchanged). Revision history rebuilt through Week 10 to `tests/scratch/lead112/injury_revisions_2026.parquet` via `scripts/per03_injury_revisions.py --out` (167 snapshots, 5742 rows; production parquet untouched). Designations first seen before Friday 16:00 ET: 0-3.6% per week. Batch file tests/scratch/lead112/batch.json.
- Script lint: ruff format applied; one RUF046 remains at the `"games": int(len(delta))` line in block_boot (cosmetic; the edit was blocked by the tool cap).

## Next
1. Fix the RUF046 line (`len(delta)` without int) and rerun ruff check.
2. Week 10 and later: rerun the script once Week 10 grades (extend LAST, snap file before_week_11) to add blocks; same 12 cells with --replace.
3. Give the term a level-free form: the first report has no designations, so test practice-status change Wed to Fri separately from the Saturday designation level, declared as a new look family before signs.

## Open
- Every out-of-sample cell has its whole interval below zero; recorded unresolved_below_power because the permutation null loses the same amount (fitting cost) and the coefficient sign is not resolved. Orchestrator to adjudicate.
- Language-read change cannot be measured until the forward capture reads each Sunday game at two times (e.g. Wednesday and Saturday views).
