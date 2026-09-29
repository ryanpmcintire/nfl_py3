# LEAD-79 - earlier fixture lookahead anchor

## Goal
Evaluate an earlier same-fixture price as a fitted addition to the four-term base.

## State
**Measured:** unit 2 ran once, exit 0; 1,059 eligible games, 970 OOS, 76 week blocks.
Changed picks: 8-14; accuracy 36.36% [15.99%, 60.00%]. Overall: 553-417,
57.01% [54.03%, 59.94%]; base 559-411. Paired accuracy: -0.61856 points
[-1.61780, 0.41537], probability_positive=0.1182. LL/Brier gains are adverse;
full intervals, IS/OOS gaps, reliability and fold coefficients: `docs/lead79_unit2.md`.

### Unit-2 amendment (saved before outcomes; unchanged)
2020-2025 REG, historical OPENER proxy, served four-term inputs; availability
assumes provider observed_at_utc under the root provenance Decision. Latest complete
same-fixture snapshot before BOTH previous same-season games and Tuesday 09:00 ET;
equal-book median. Term: lookahead-implied margin minus opener-implied margin.
Earlier seasons train each held-out season; fixed ridge 0.001, no tuning; one fitted
probability selects the side. Opener grading retains the served pregame movement horizon.
Accuracy/Brier/log loss, available cached MAE, paired baselines, IS/OOS gaps,
coefficients, reliability, decisive record; 10,000 week-block draws, seed 79.
B=2/F=6/E=0, 291-look ceiling, 209 numeric reporting looks. Original full declaration
is copied verbatim in the report appendix and saved/hashed in scratch. Zero crossing
closes nothing; no power control or reliability closure is claimed.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead79_unit2.py` with
UV_NO_CACHE=1; lint/format, saved-row arithmetic, chronology and syntax checks passed.
No new tests. Run log: `%TEMP%/lead79-unit2-run.log`; rows/summary in
`tests/scratch/codex/lead79_unit2/`. No registry command was run.

## Record commands
Orchestrator only; exact bash loop records eight paired OOS endpoints serially.
They overlap and are not independent evidence; primary accuracy remains unresolved.
```bash
while read -r comparator units effect low high positive se; do
  UV_NO_CACHE=1 .tools/uv.exe run --no-sync nfl-ats weak-signals record \
    --name "lead79_unit2_${comparator}_${units}" --source docs/lead79_unit2.md \
    --description "Earlier matchup price versus ${comparator}, ${units}" \
    --effect="$effect" --effect-units="$units" --interval-low="$low" --interval-high="$high" \
    --probability-positive="$positive" --standard-error="$se" --sample-games 970 --sample-blocks 76 \
    --classification unresolved_below_power --league nfl --season-start 2021 --season-end 2025 \
    --family lead79_same_fixture_anchor --category market \
    --classification-evidence 'No closure requested; retain adverse losses and uncertain accuracy.' \
    --notes 'Provider clock assumed available; five forward folds; overlapping comparisons; no reliability estimate.' \
    --plain-summary 'Using an earlier matchup price lost 14 of 22 changed picks against the current method. This study does not support changing the pool picks.'
done <<'RESULTS'
four_term accuracy_points -0.6185567010309279 -1.6177957532861476 0.4153686396677051 0.1182 0.5134330489278933
four_term log_loss_improvement -0.002114363572317551 -0.004281188727584316 -5.598091357649272e-05 0.0218 0.0010793929114343618
four_term brier_improvement -0.0010119157608972815 -0.002038667943899945 -3.9532948170922026e-05 0.0193 0.0005101045794968037
model accuracy_points 3.2989690721649483 0.0 6.673618352450469 0.97575 1.7023907981246138
model log_loss_improvement 0.007425979287467152 -0.005003368655367486 0.02007039451150739 0.8822 0.006352988058260747
model brier_improvement 0.0040465224089098065 -0.001755005774931721 0.009963487468937786 0.9157 0.002974942104480966
market log_loss_improvement 0.006138367955529108 -0.00749228379604952 0.019709829233888497 0.8109 0.006980514241032028
market brier_improvement 0.003497407042699485 -0.00290831643311551 0.00986872045755164 0.8587 0.0032708506965016737
RESULTS
```

## Next
Orchestrator: review loss evidence, execute the commands serially, and choose the next unit.

## Open
**Inferred:** no serving change; accuracy remains unresolved pending registry review.
Elo lacks a matching cached prediction; candidate margin MAE is undefined for a probability fit.
No registry/served edits, publication, commits, or other lane changes.

**Recorded 2026-09-29 (root):** the three four_term comparison cells only, unresolved_below_power.
