# LEAD-73 unit 2: extended dated moves

## Goal

Refit the same four-term probability with verified 2020–2022 moves; research only.

## State

**Measured:** complete. Protocol saved here before outcomes; full declaration and
results now in `docs/lead73_unit2.md`. Frozen 1,503-game 2020–2025 population;
identical model/composition/opener rows and 2023–2025 moves in both arms. Six LOSO
folds, four terms plus intercept, ridge .001, no tuning; full-data IS comparison.
10,000 paired season-stratified week-block resamples, seed 20260929; one primary
comparison and 144 reporting looks. Raw model and neutral market baselines;
fixed calibration bins. Zero crossing never closes a signal; one fitted
probability chooses the side. No protocol changes after outcomes.

**Measured:** 757 source games, 698 aligned additions; available moves 799→1,497.
Decisive record: extended 69–94. OOS accuracy 55.755% versus 57.418%; gain
−1.663 points [−3.275, −0.067], probability_positive 0.0202. Extended log loss
0.6830, Brier 0.2450; gains versus current have probability_positive .4707/.4496.
Move coefficients positive 6/6 folds: .119636–.165644 (current .188739–.246066).
Extended IS/OOS accuracy gap .732 points [.133, 1.390].

## Tried

**Measured:** `.tools/uv.exe run --no-sync --no-cache python scripts/lead73_unit2.py`
exited 0 once; 455 quote manifests hash-verified; sign/time/team gates passed;
current IS/LOSO predictions reproduced exactly. Scoped Ruff format/check passed.
Report refreshed from saved predictions after review, without refitting or
resampling. Independent pooled arithmetic and nine registry closure/coherence
checks passed in memory. No registry, served-fit, src/, publication or git writes.

## Next

Owner reviews report, runs the batch below serially, and decides subsequent work.
**Inferred:** primary accuracy comparison supports `wrong_sign_resolved` only for
this fixed extension. Eight diagnostic contrasts remain `unresolved_below_power`;
this does not close book moves or older-source research. No promotion proposed.

## Open

Upstream model/composition construction is inherited. LOSO is not chronological
deployment; bootstrap holds fits fixed. Capture density may differ by archive.
No split-half reliability or positive-control experiment. Batch not yet recorded.
Artifacts/rows: `tests/scratch/codex/lead73_unit2/`; preserve them for owner review.

## Record commands

Bash-compatible; prepared, not executed. Batch contains all nine contrasts,
plain-English summaries, intervals and explicit primary closing ground.

```bash
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record \
  --batch tests/scratch/codex/lead73_unit2/registry_batch.json \
  --plain-summary 'Adding older sportsbook line moves did not improve the current picks. Keep the current version while the older moves remain available for research.'
```

**Recorded 2026-09-29 (root):** only the three extended-vs-current cells, all unresolved_below_power (accuracy -1.66 [-3.28,-0.07] with flat proper scores, 144 looks, move coefficient positive 6/6); baseline contrasts not recorded as signals. Served fit unchanged.
