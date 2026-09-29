# MOD-22 unit 5: QB quality remeasurement

## Goal
Repair unit 5 with unit 4's evidenced report times and suffix matching; rerun once.

## Protocol (locked before corrected outcomes, 2026-09-29; unchanged)
2020–2025 REG nonpush `build_fit_population` games with Tuesday opener; target
`home_covered`; decision kickoff minus 24h, freeze Tuesday noon Eastern.
Quality: previous REG season EPA/QB dropback, valid EPA/named passer/no no-play,
2019–2024 PBP snapshot `20260812T142851Z`; missing quality fixed at -0.15,
no shrinkage/minimum cutoff. Starter: unit 4's previous-game most-used QB.
Backup: available other QB on latest strictly prior roster, most cumulative prior
team offense snaps, ties GSIS ID; no backup uses -0.15. Latest evidenced report
must be Out/Doubtful; first visible Out/Doubtful after freeze. Otherwise loss=0.
Signed loss=starter minus backup quality; term=away minus home loss.
Import repaired `norm`; mirror non-proxy `date_modified` evidence and cutoff;
keep prior-roster/quality/snap guards. One calibrated probability selects sides.
Same four terms plus fifth term, LOSO 2020–25 scaling/coefficients, inherited ridge;
IS diagnostic; accuracy/Brier/log loss, gaps, baselines, decisive/null, all folds,
season/reliability diagnostics. Paired 2,000 season/week bootstrap draws,
seed 20260923, 95%; inherited bins [0,.4,.45,.5,.55,.6,1]. Family
`qb_quality_loss_v1`, look **1**, remeasurement; no tuning or additional selection.
Report probability_positive; zero crossing never closes a signal.

## State
**Measured:** exit 0; 1,503 games/107 blocks, 55 late-out flags, 53 nonzero terms;
18 proxy reports excluded, one evidenced report late; matches 3,230/3,230 source,
3,006/3,006 scored. Decisive **4–12**, exact-null p=0.0768127. Brier improvement
-0.001059849, 95% [-0.003162455,-0.000080059], probability_positive=0.0045;
accuracy -0.532269 pp, 95% [-1.504276,0.194430], probability_positive=0.09625.
IS/OOS Brier 0.244387627/0.245898235, gap -0.001510609; coefficient positive 3/6
folds, range [-1.268117,.403065]. Full scores, log loss, gaps and reliability:
`docs/mod22_unit5.md`. **Inferred:** proposed closure only for this construction
under AGENTS.md:70–78 (wholly adverse Brier interval); accuracy remains unresolved.

## Tried
Ran `.tools/uv.exe run --no-sync python scripts/mod22_unit5.py` once (2 threads).
Scratch `tests/scratch/codex/mod22_unit5/20260929T214747Z/`; log
`tests/scratch/codex/mod22-unit5-remeasurement.log`. **Measured:** evidence-filter
AST matches unit 4; shared normalizer, unchanged population/targets/opener/base
probabilities/starter IDs; all 55 saved injury gates and Brier arithmetic pass.
Prepared Bash command parses and passes numeric/closure validation without its
handler; script syntax and no-comment/docstring checks pass. No second fit.

## Next
Orchestrator verifies/runs replacement below serially; worker did not record it.

## Open
Prior roster/depth changes, missing quality, inherited later market inputs;
no split-half reliability or power-matched control; broader QB mechanisms open.

## Record commands
Prepared, not executed; replaces old timestamp-proxy cell, look count unchanged.
```bash
nfl-ats weak-signals record --replace \
  --name qb_quality_loss_fifth_term_brier --family qb_quality_loss_v1 \
  --category health --league nfl --season-start 2020 --season-end 2025 \
  --description "MOD-22 unit 5: evidenced QB quality-loss fifth term; single-look remeasurement" \
  --source "tests/scratch/codex/mod22_unit5/20260929T214747Z/summary.json" \
  --effect=-0.00105984891368009 --effect-units brier_improvement \
  --interval-low=-0.0031624546186375373 --interval-high=-8.005893897342283e-05 \
  --probability-positive 0.0045 --sample-games 1503 --sample-blocks 107 \
  --classification refuted_mechanism --closing-ground wrong_sign_resolved \
  --classification-evidence "Paired 95% Brier improvement is wholly negative; closure covers only this quality proxy, timing gate and fitted fifth term" \
  --notes "Look count remains 1; 18 proxy reports excluded; no split-half reliability measured; broader QB-quality mechanisms remain open" \
  --plain-summary "When the expected quarterback is out or doubtful after Tuesday, using last year's gap to his backup made predictions worse. Keep this adjustment out of the picks."
```
