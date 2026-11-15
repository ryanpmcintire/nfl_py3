# Lane: sim-finish-amendment5 (done 2026-11-11 10:55)
## Goal
Repair the conditioning step (exponential tilt of draws to the line), declare it, grade it on saved draws, add the driver option, leave the final-run command.
## State
- Amendment 5 appended to docs/sim08_market_anchored_declaration.md (declared before the tilt was applied; 33 cells counted; second look at the diagnosis shift).
- scripts/sim08_amendment5_grade.py -> artifacts/sim08_amendment5/grade.json, tilt_weights.npz, record_batch.json. 33 cells recorded (family sim08_market_anchored, names sim08_a5_*, unresolved_below_power); shard registry/weak_signals/sim08_market_anchored.json 957 -> 1881 lines.
- scripts/sim08_market_anchored.py: `--amend5 1` (env SIM08_AMEND5, default off) on fold/full; saves `weights` and `ess` beside unchanged margins/totals/ot. Off path: anchored_draws, game_task, npz keys, smoke records unchanged (checked with a stub play: first three columns equal, 3-column shape off, weighted mean -3.5 exact on). Also hand-fixed four pre-existing ruff findings (no behaviour change).
- Result (2026-11-03 engine, no A4 terminal transitions): tilt is exact (max |mean-line| 9e-15) but does NOT reproduce the shift diagnostic. Solo cover-logit coef tilt -0.498 (CI -1.33 to 0.34, P+ .13), all 6 folds negative; the shift gave +0.448. Same draws: shift on half-lines +0.16, tilt -0.99. Log loss tilt .7661 v served .7638, market hist .7613, raw .7773. See grade.json.
## Tried
Margin-only tilt via sim04_loso.tilt_to_mean; total left at the anchor. n_eff mean 363.5, min 259; 923 of 1,537 below the 369.8 floor (floor sits at N by construction).
## Next
Final run (after SIM10_TERMINAL identity and Tier B pass, one heavy job at a time):
1. Smoke, one core, 16 draws (a 2-draw tilt cannot bracket the line; 16 brackets it with probability above 0.9999 per game; always finite ESS): `New-Item -ItemType Directory -Force F:\Repos\nfl_py3\artifacts\sim08_market_anchored\smoke_a5; Copy-Item F:\Repos\nfl_py3\artifacts\sim08_market_anchored\main\fit_2025.json F:\Repos\nfl_py3\artifacts\sim08_market_anchored\smoke_a5\; $env:SIM08_HELD='2025'; $env:SIM10_TERMINAL='1'; F:\Repos\nfl_py3\.tools\uv.exe run --project F:\Repos\nfl_py3 --no-sync python F:\Repos\nfl_py3\scripts\sim08_market_anchored.py fold --held-out 2025 --arm primary --run smoke_a5 --workers 1 --draws 16 --amend4 1 --amend5 1` (`fold` has no game limit: 272 games, up to 6 passes x 16 draws x 0.13 s, about 25 to 60 min; background call, wait for notification).
2. Smoke must show: exit 0; primary_2025.npz holds weights (272,16) and ess with no NaN; every game's weighted mean margin within 1e-6 of its line; ess between 1 and 16; total_target_2025.json written; then delete smoke_a5.
3. Full: `$env:SIM10_TERMINAL='1'; F:\Repos\nfl_py3\.tools\uv.exe run --project F:\Repos\nfl_py3 --no-sync python F:\Repos\nfl_py3\scripts\sim08_market_anchored.py full --run amend5 --arms primary --workers 4 --draws 372 --amend4 1 --amend5 1` (parent lane estimate 12 to 30 h; workers 204 MB each, asserts 16 GB free, 36 GB measured free 2026-11-11).
4. Grade: `sim08_evaluate.py --run amend5 --arms primary` (raw) and `sim08_amendment5_grade.py --run amend5` (tilted; names sim08_a5_amend5_*; recount looks under A4 before recording).
Before the 12 to 30 h run, decide whether more draws or a different conditioning is worth it: the exact-mean tilt did not repair the inversion on these draws.
## Open
- Shift (+0.448) and tilt (-0.498) differ by about one coefficient SE (0.415 for shift) on the same draws; which conditioning is the definition is a declared-before-the-run choice and the data do not resolve it. Not identified: why half-line cover logit stays anti-informative after exact mean conditioning.
