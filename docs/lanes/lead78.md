# LEAD-78: within-week common-opponent news

## Goal
Measure one fitted fifth term without changing the served card.

## State
**Measured:** unit two complete: 1,503 games, decisive 4-9-0; accuracy -0.332668 points [95% -0.805369, +0.133511], probability_positive 0.08435. Log-loss gain -0.000566351 [-0.001148588, -0.000027260], probability_positive 0.0197; Brier gain -0.000283074 [-0.000568985, -0.000017961], probability_positive 0.0181. Provisional unresolved_below_power; registry pending. Full tables, original declaration and limitations: `docs/lead78_unit2.md`.

### Unit-two amendment (declared before outcomes)
Use the historical OPENER as the frozen-line proxy on the same 1,503 non-push 2020-2025 REG games, with the served four-term probability as base (`artifacts/pick_probability/20260929T192747Z/per_game.parquet`). Replace nonexistent published-final timestamps with scheduled kickoff + 4 hours from `data/raw/20260908T162105Z/schedules.parquet`; assert each included Thursday/Saturday result completes after Tuesday noon and strictly before the target deadline, the earlier of kickoff and Sunday 12:45 Eastern. These are explicit proxies, not reconstructed pool captures or verified final-publication times.
Retain the original score-state delta as ONE fifth term, market-move control, six season holdouts, paired metrics and B=3/F=6/E=0: the look budget is unchanged at 298. Per the unit-two packet, use the standard retrospective LOSO (other five seasons train each held-out season) and season-stratified week-block bootstrap (10,000 draws; seed 78); disclose that this is not forward-only evaluation. No searches, extra endpoints, or selected subsets.
Freeze one rating implementation: ridge team-margin graph with unit prior precision, prior ratings and home advantage estimated only from the preceding season. At Tuesday noon fit current-season completed scores around that prior; append all eligible Thursday/Saturday scores, then subtract Tuesday home-minus-away rating from the updated rating difference. The fixed graph propagates information through already played opponents; unavailable early news gives zero. Refit all four existing terms plus this delta with the existing ridge 0.001 and training-only standardization in every fold. Fixed baselines: served four-term, raw discrete model, even market, and score-only Elo (K=20, scale=400, offseason carry=0.75; training-fold logistic opener calibration). No data-chosen cutoffs or grids.
Report decisive W-L-P before overall accuracy; paired accuracy-point, log-loss and Brier gains with 95% intervals and `probability_positive`; train/test/gap pooled and by season; every fold's natural coefficients; five training-quantile reliability bands. Margin MAE uses fold-trained linear margin calibrations of the same score (reporting only). Preserve all prediction rows and timing audit in `tests/scratch/codex/lead78_unit2/`. An interval crossing zero closes nothing; one fitted calibrated probability selects the side.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead78_unit2.py`: one pre-fit opener-sign validation failure, then one completed scoring run, exit 0 (8.9 seconds). Two numerical threads; 1,886 timing pairs passed; base reproduction error 2.22e-16. Ruff format/check, scoped diff check and saved-artifact review pass (rows, timing, decisive record, four record cells). Predictions, audit, summary, logs and record batch: `tests/scratch/codex/lead78_unit2/`. No tests added or registry writes.

## Record commands
Prepared only; batch holds the four primary metric cells with intervals and plain-English summaries (diagnostic baseline comparisons are not separate signals).
```bash
UV_CACHE_DIR=tests/scratch/codex/lead78_unit2/uv-cache UV_OFFLINE=1 .tools/uv.exe run --no-sync nfl-ats weak-signals record \
  --batch tests/scratch/codex/lead78_unit2/registry_batch.json \
  --source docs/lead78_unit2.md \
  --plain-summary 'Using early results to revisit common opponents lost five extra picks. Keep the current picks; the idea remains unresolved.'
```

## Next
Orchestrator: review and run the prepared record command serially; retain research-only status. No commit, push, publication or served-card change by this worker.

## Open
**Inferred:** opener/completion proxies and retrospective LOSO limit prospective interpretation. Proper-score deterioration does not alone refute the mechanism (news coefficient positive 4/6); no power control was established. Registry entries are not yet settled.
