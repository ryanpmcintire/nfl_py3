# LEAD-66

## Goal
Execute the three-arm whole-margin PMF protocol in ROADMAP.md LEAD-66.

## State
**Measured:** one run passed; 1,537 games (1,503 non-push), 2020-2025.
RPS gains (95% season-block interval; probability_positive):
- Served vs market: IS -0.024680; OOS -0.026309 [-0.070629, 0.004613]; gap -0.001629; 0.0765.
- Model-centred vs market: IS -0.037600; OOS -0.037718 [-0.081865, -0.010954]; gap -0.000119; 0.
- Model-centred vs served: IS -0.012920; OOS -0.011409 [-0.019049, -0.003794]; gap +0.001510; 0.0029.
First contrast unresolved; other two have proposed wrong_sign_resolved grounds, pending records.
Report/fold coefficients: `docs/lead66_unit1.md`; per-game scores: `docs/lead66_prediction_scores.md`.
Keep the generated per-game score appendix local; AGENTS.md forbids committing processed data.

## Tried
Copied the roadmap row and full implementation declaration here before outcomes; unchanged copy
retained in `docs/lead66_protocol.md`. Three primary contrasts + 15 calibration cells = 18 looks.
Ran `.tools/uv.exe run --no-sync python scripts/lead66_unit1.py` once (exit 0), one thread;
ruff check/format and syntax/comment checks passed. All pooled exact log scores are infinite:
market/served/model-centred zero-mass games = 26/69/81. No registry, serving or Git writes.

## Next
Orchestrator: review source/protocol limitations, resolve RPS unit support, then run records serially.

## Open
**Measured:** same-configuration secondary predictions lack 2009-2017; no secondary score or rebuild.
**Read:** `weak_signals.py:56` lacks `rps_improvement`; the required commands below are parser-blocked.
Chronological LOSO applies to new PMF mass fits; frozen base forecasts remain walk-forward.
Cover disagreements are diagnostics only. Full log-score comparison needs a separately declared
tail-support repair; no smoothing or alternate arm was introduced after seeing outcomes.

## Record commands
Prepared only; not executed. Resolve the unit schema first; retain the declared RPS units.
```powershell
$lead66Common = @('--source', 'docs/lead66_unit1.md', '--effect-units', 'rps_improvement',
  '--league', 'nfl', '--season-start', '2020', '--season-end', '2025', '--sample-games', '1537',
  '--sample-blocks', '6', '--family', 'lead66_margin_pmf', '--category', 'modeling',
  '--notes', '3 primary contrasts and 15 calibration cells; chronological LOSO PMF fits; frozen walk-forward base; no serving change.')
.tools/uv.exe run --no-sync nfl-ats weak-signals record @lead66Common --name lead66_served_lattice_vs_market_residual `
  --description 'Served lattice vs market-centred residual PMF' --classification unresolved_below_power `
  --effect -0.0263092311 --interval-low -0.0706287621 --interval-high 0.0046126176 --probability-positive 0.0765 `
  --classification-evidence 'No admissible closing ground; retain the unresolved PMF contrast.'
.tools/uv.exe run --no-sync nfl-ats weak-signals record @lead66Common --name lead66_model_centred_lattice_vs_market_residual `
  --description 'Model-centred lattice vs market-centred residual PMF' --classification refuted_mechanism --closing-ground wrong_sign_resolved `
  --effect -0.0377183695 --interval-low -0.0818652969 --interval-high -0.0109540449 --probability-positive 0 `
  --classification-evidence 'The whole six-season bootstrap 95 percent RPS-improvement interval is on the wrong side of zero.'
.tools/uv.exe run --no-sync nfl-ats weak-signals record @lead66Common --name lead66_model_centred_lattice_vs_served_lattice `
  --description 'Model-centred conditioning vs served line conditioning' --classification refuted_mechanism --closing-ground wrong_sign_resolved `
  --effect -0.0114091385 --interval-low -0.0190492490 --interval-high -0.0037943140 --probability-positive 0.0029 `
  --classification-evidence 'The whole six-season bootstrap 95 percent RPS-improvement interval is on the wrong side of zero.'
```
