# SIMULATOR HANDOFF - START HERE (2026-10-24)

Active task: finish the SIM-10 drive-level simulator. The owner wants it finished, not explored.

Full state: `docs/lanes/sim10-drive-resampler.md`. Code: `scripts/sim10_drive.py`.
Data and fitted artifacts: `artifacts/sim10/` (all built; do not rebuild).

## Where it stands
- Passes 10 of 13 gates against real 2009-17 games (3-seed average).
- Fails strength variance (48.8 v [51.4, 64.9]), late R2 (.126 v [.131, .161]), cross-quarter
  catch-up (-19.3 v [-17.7, 6.6]). One cause: team strength transmits ~5% weak because the last
  2 minutes of each half (~8% of plays) are not strength-matched.
- Test `SIM10_MU_ONLY=1,4` (strength-only tilt in those minutes), seed 12: strength 52.5 and catch-up
  pass, but mass at 3 drops to .132 (bound .135). Seed 13 result: `artifacts/sim10/muonly_s13.log`.

## Do exactly this, then stop and report
1. Read `artifacts/sim10/muonly_s13.log`.
2. Implement: in regimes 1 and 4 tilt to strength AND td/fg/ds targets, not elapsed. Run seeds 12
   and 13 (one sim at a time, background, `--workers 11`, ~10 min each):
   `python scripts/sim10_drive.py sim --label X --seed S --workers 11 --models 1 --pat 1 --mufree 1,4 --dr 1 --h artifacts/sim10/bandwidths_week.json`
3. Pick the config passing the most gates over 3 seeds (fallback: the command above, no env vars),
   make it the CLI default, commit, push, give the owner the pass/fail table. No more experiments.
