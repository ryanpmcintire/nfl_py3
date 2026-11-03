# SIM-08 simulator rebuild

## Goal
Repair transition and terminal-state behavior, then diagnose late-game bias
without fitting weights or changing served probabilities from these diagnostics.

## State
**Measured:** September 28 repaired quarter boundaries, zero elapsed transitions,
timeouts, and historical overtime termination/conversion rules. The final fixed
replay has 10,002 games, zero caps, but only 2/5 key-number bands pass.
Log-loss improvement versus the training histogram is -0.000656
[-0.013844, +0.014832], probability_positive 0.370370. Versus the preceding
engine, improvement is -0.001595 [-0.002528, -0.000795], probability_positive 0
in 27 conditional season draws; Monte Carlo and prior selection are omitted.
All four effects remain registered unresolved; no serving change.
[Complete 147-look result, including superseded arm](../simulator_overtime_repair_20260928.md).

## Tried
- **Measured:** conversion repair reconciles 768/768 final margins. Full repair
  history: `done/sim08-transition-repair-history.md`.
- **Measured:** 12 neighbor looks and four chronological comparisons diagnose
  conditional scoring and clock behavior. Chronological one-score elapsed delta
  is -2.11 seconds [-2.88, -1.43], probability_positive 0 in 2,000 draws;
  scoring frequency is +0.17 points [-0.43, +0.77], probability_positive 0.706.
- **Measured:** three validation source-age slopes and six training rolling-era
  slopes bring that diagnostic family to 25 looks. Excluding 2009, training slopes
  are +0.830 [0.282, 1.391], +0.741 [0.118, 1.315], and +0.802 [0.334, 1.286]
  seconds/year, probability_positive 0.999, 0.989, and 0.9995. Cross-role game
  identity limits those bootstrap intervals; these are not causal rollout effects.
- **Measured:** 144 instrumented rollouts preserve all same-seed outputs, with
  zero caps. Exact matching yields 2 selected scoring transitions; the separately
  declared timeout-omission sensitivity yields 9 in 9 rollout clusters from
  6 scheduled games. Both fail declared 30-transition/10-cluster support gates.
  Neither computes an effect. Two rollout endpoint looks are recorded.
- **Measured:** the expanded 96-game/eight-seed sample yields 67 matched scoring
  transitions in 66 rollouts from 40 scheduled games. Its fixed-selection paired
  bootstrap estimates selected-minus-actual elapsed discrepancy -0.751 seconds
  [-2.728, +1.173], probability_positive 0.2175 (2,000 draws; 55 actual games).
  This fourth rollout endpoint look is recorded as `unresolved_below_power`.
- Full methods, artifacts, uncertainty and limitations are retained in
  `done/sim08-late-game-diagnostic-history.md`.

## Next
Historical rule defects are repaired; broader predictive accuracy is unresolved.
Preserve `artifacts/sim08_overtime_repair/20260928T171712Z/` with all three arms,
source/input hashes, 30,006 simulations and actual-game losses. Any new mechanism
or 2025+ rules extension needs a declaration before another endpoint is scored.
Do not patch key-number masses, fit on these diagnostics, or promote this engine.

## Declaration (2026-11-02)
Unit 3 market-anchored sim is predeclared in `docs/sim08_market_anchored_declaration.md`
(34 looks, primary L4 blend v served-only recalibration, 2020-2025 1,537 games, LOSO 6
folds, planted positive control). Next: commit it, then build the anchored arm (no code
comments), 20-game smoke, planted control, full run one job, <=4 workers.

## Open
Finite-draw probability_positive of 0 or 1 is not impossibility or certainty.
The deficient margin-3 mass and rollout calibration remain unresolved. Support
failure closes no signal; these diagnostics authorize no weight or serving change.

## Unit 3 build (2026-11-02, subagent)
Built `scripts/sim08_market_anchored.py` (imports sim10_drive engine unmodified; `engine()` rebuilds locked
defaults, `play()` calls `Engine.game` with ratings set from latents x_diff, x_env). Anchoring solve:
`transmission_fit` (OLS of simulated margin/total on design latents, 2x2 inverse in `solve_latents`).
Smoke: `python scripts/sim08_market_anchored.py smoke --design-n 1500 --draws 150 --x-diff-range 0.8 --x-env-range 0.5`
-> tests/scratch/sim08_anchor/smoke.json. Positive control (as declared, planted on the LOSO
recalibrated served logit, 500 reps): `... control --deltas 0.0005 0.0010 0.0020 0.0040 0.0080` ->
control.json; detects all (pp>=0.90 in 100% of reps) because the planted gain is near-deterministic per game
(caveat; a noisy plant would need an amendment). Engine 0.44 s/game-draw at 1 worker.
Open before full run: per-fold kernel pool needs LOSO upstream (blocked per 9a0a951d); opener totals
missing for 11 of 1537 games (spread-only fallback in `anchored_draws`).
Smoke result (measured, fold 2025, 20 games, 150 draws, design 1500, x_diff +-0.8, x_env +-0.5): ANCHOR FAILED.
Margin slope 17.2 (SE 0.9) fine but total response to x_env is non-monotone (linear R2 .074, quadratic .181;
probe: total 70/47/59 at x_env -1/0/+1), so the 2x2 linear inverse put x_env at -0.3..-1.6 and sim totals 65-79
v opener 36-55 (signed err +23.8). Margin mean err -2.7 (z rms 4.3). Cause: design range far outside engine
support (s=0.969, real xq ~ +-0.2). Next: re-measure with design ranges from the sim-world xq distribution, fit
total on x_env with |x_diff| term or monotone local lookup (a mechanism measured, not a knob), re-smoke.
Cost: 0.26 s/game-draw design, 0.58 s/draw anchored (contended) -> 1000 draws x 1537 games = 100-250 h single
worker; infeasible at declared 1000 draws, needs a declared amendment (draw count).

## Unit 3 re-anchor (2026-11-02, subagent 2)
Support from engine latents (support.json) x_diff +-0.5, x_env +-0.21; script now solves bounded joint quadratic
(response/solve_latents), smoke takes --design-csv. Smoke 9,600 design/300 draws: margin err -0.41 z -0.97 pass;
total +0.95 z +3.10 FAIL (persists across design size). Amendment 1 appended to the declaration (N=372, noisy plant).
Next: smoke with design pool drawn from held-out fold contexts (code change to design_run pool), then the noisy
control code, then the full run (one job, 4 workers). Per-fold LOSO kernel pool still open.

## Unit 3 re-anchor 2 (2026-11-02, subagent 3)
Code: engine() now sets SIM10_JOINT=1 (new default; was unset in old smokes = pre-SIM-11 engine), `support` command,
`smoke --heldout-pool 1 --save-design`, per-game sim_home_cover/loss stored, `control --noisy 1` (Amendment 1 item 5:
kappa bisection on LOSO blend, sd = sqrt(train_sd^2 + (2/sqrt(draws))^2); --sd-from-smoke or --train-sd).
Support on joint engine (tests/scratch/sim08_anchor/support_joint.json, 92,160 games): x_diff 0.1/1/5/25/50/75/95/99/99.9% =
-0.492/-0.380/-0.267/-0.109/0.001/0.109/0.267/0.376/0.502; x_env -0.205/-0.153/-0.107/-0.041/0.003/0.047/0.111/0.155/0.205;
sd 0.162/0.066, corr 0.007. Unchanged vs old (latents are engine-independent); box stays +-0.5/+-0.21.
Smoke5 (joint, held-out-fold pool, 9600 design, 300 draws, 20 games) launched PID 33748 (python child; parent 33164),
out smoke5.json/log, design saved design_joint_heldout.csv. Next: read z, then noisy control on smoke population.
Result (measured): smoke5 and identical-seed rerun smoke6: margin err -0.645 (z -1.56), total err +0.145 (z +0.38), pass.
Total bias cleared by held-out-fold context pool (confounded with joint-engine switch, inferred). 0.116 s/draw, 36 GB free.
Transmission: margin 1.93+37.5 xd+3.9 xe (R2 .403); total 45.7+2.5 xd+47.1 xe (R2 .162, quad .172). Amendment 2 appended.
Noisy control coded and executed (smoke6 population, 20 reps, not a look): deltas .001/.008 kappa .017/.045, realized .00044/.0071.
Next: per-fold LOSO kernel pool decision, then full run (one job, 4 workers, N=372, heldout-pool per fold); control verdict
only after full-run train sd. No processes left running.

## Unit 3 full run (2026-11-03 subagent 4) - IN FLIGHT
Amendment 3 appended to the declaration (per-fold state `artifacts/sim10_wf/T<s>` = seasons 2009..s-1 for library, models, latent, strength,
q4dec, bandwidths; runtime asserts `assert_fold_state`; Jeffreys smoothing; baselines; operational look definitions; A1 only if time allows).
Code: `scripts/sim08_market_anchored.py` (`fold`, `full` commands), `scripts/sim08_evaluate.py` (all declared cells; tested on fake data only, removed).
Launched 23:46 Mon: `python scripts/sim08_market_anchored.py full --run main --arms primary,spread --workers 4 --draws 372`
(background task b58jmidj2; driver PID 31404, fold PID 8392, workers 17520 19684 21104 26696, stubs 21616 22224; PIDs 13088 15776 are NOT mine).
Output `artifacts/sim08_market_anchored/main/{primary,spread}_<season>.npz` per fold (checkpoint; driver skips finished), `run.log`.
Free RAM 36 GB, workers 204 MB each, sampled page faults ~0/s. Expected ~1 h per fold: primary done ~06:00, spread ~11:30.
Deadline: at 11:30 ET Tue kill by PID, keep finished folds; spread incomplete -> A1 not run, 5 cells unrecorded.
Next: after primary folds (6 npz): `python scripts/sim08_evaluate.py --run main --arms primary` (then `primary,spread` if complete) ->
evaluation.json; noisy control `python scripts/sim08_market_anchored.py control --noisy 1 --train-sd <mean of sim_logit_minus_served_sd_by_train_fold>
--draws 372 --deltas 0.001 0.008 --replicates 500 --out artifacts/sim08_market_anchored/main/control.json`; then record cells (families
sim08_market_anchored 29 incl A1, sim08_market_anchored_shape K x5), `nfl-ats research status --family sim08_market_anchored`. Nothing recorded yet.

## Unit 3 RESULT (2026-11-03, measured; run finished, no processes left)
Artifacts: artifacts/sim08_market_anchored/main/{evaluation.json,control.json,*.npz}. Both arms complete (N=372, 6 folds, per-fold sim10_wf state).
Primary L4 (blend v served-only recal) +0.00153 [-0.00114,+0.00412] pp 0.864; spread-only A1 L4 +0.00039 pp 0.61. L1 raw sim v served -0.0136 pp 0 (raw sim worse); L5 +0.0062 pp 0.98.
Blend sim coef NEGATIVE in all 6 folds (-0.48..-0.39) with served 0.28..0.42: odd sign, inferred attenuation/overlap, unexplained. Decisive games 390: blend 196 v recal 194.
Shape: mass3 sim .130 v real .143 (K3 -0.0205 pp .01), K7 +.0096, K17 -.0094; total slope sim 1.111 v real (T +0.223 pp 1.0, overshoot); OT diff -0.0025.
Noisy control: delta .001 detected 31.8% (not detected), delta .008 84.4% (detected) -> power statement: detects ~0.008 not 0.001; L4 stays unresolved_below_power.
Next: weak-signals record --batch (family sim08_market_anchored 29 cells, sim08_market_anchored_shape K x5; T,O in main family), then research status. RECORDED 2026-11-03: 34 cells via weak-signals record --batch (29 main incl A1, 5 shape), all unresolved_below_power; status 29/34 main (5 are the shape family), no closure. Lane correction: real mass at 3 is .151 (evaluation.json), not .143. Decision rule (decl. s10): L4 pp 0.864 is between 0.5 and 0.9, so unresolved, work stays open; nothing served. SIM-08 bar (catch the served read AND beat it on push pricing) NOT met: raw sim worse than served (L1 -0.0136) and on push lines (P -0.1375 pp .006). Negative blend sim coefficient in all 6 folds: measured, unexplained.

## Unit 3 shape decomposition (2026-11-03, subagent 5; measured, no simulation run, no engine edit)
Script `scripts/sim08_shape_decompose.py` -> `artifacts/sim08_market_anchored/main/shape_decompose.json` (ruff clean; reads npz + 2020-25 pbp).
The npz store only margins/totals/ot per draw, so the sim cannot be split by scoring path or drives per game from stored outputs.
Mass at 3 (1537 games x 372): sim .1304 v real .1509. Regulation: sim .0978 v real .1152 (-1.73 pp, 85% of gap); OT: .0326 v .0358 (-0.32 pp).
OT rate sim .0580 v real .0605; margin 0 (tied after OT) sim .0089 v real .0033 (2.7x; sim OT tie share 15% v real 5%, cause unmeasured).
By |opener line|: <=3 sim .146 v .152; (3,4] .143 v .197 (n 229, -5.4 pp); (4,6.5] .130 v .152; >6.5 .101 v .123. Sim mass at 3 is nearly flat in the line
(.146->.143->.130->.101) while real peaks at 3-3.5 (.176 at 3, .209 at 3.5): the shortage is where the expected margin sits on 3, not a uniform deficit.
By winner score: sim short at 20-17, 23-20, 27-24 (-.0064/-.0053/-.0039 of games), long at 24-21, 17-14, 13-10 (+.0045/+.003/+.002).
Real margin-3 endings (232 games, pbp last scoring event): FG from tie in regulation 94, OT FG 51, loser TD+PAT trimming a 10 lead 35, winner TD+PAT from -4 23+3, loser FG from -6 5.
Per-path sim shares are not identifiable from stored outputs; named candidates (not isolated): late-game FG v TD v kneel/clock choice by margin cell near 3 (q4dec/fgdec offsets
cover only regime x margin-bin logit shifts), and OT possession-end rule. Read defect candidate: Engine.game skips run_half after an OT opening score, so a first-possession FG ends the
game under the modified rule (2012-2024 seasons) with no reply drive; it inflates OT margin 3 and cannot explain the regulation gap.
Total slope (sim mean total on opener total 1.111 v real 0.888): sim quintile means 38.4/42.2/44.8/47.8/52.4 v opener 38.8/42.6/44.9/47.5/51.5 v real 39.8/43.3/46.0/47.8/51.2.
Two halves: anchor target (market total spread exceeds realized: real slope 0.888, so 0.112) and anchor inversion overshoot at the tails (sim 1.111 v 1.0 target, 0.111).
Real pace is flat in the total (drives/game slope -0.087; 22.6 -> 21.3 across quintiles) and points/drive carries it (1.76 -> 2.41, slope +0.048 per total point), so the sim's excess is not a pace effect in the real data;
sim drives per game and points/drive by strength bin are NOT measured here (needs engine instrumentation, held until sim10_drive is free).
By |line| quartile real drives 21.7/21.9/21.9/21.6, points/drive 2.08/2.04/2.14/2.15; sim abs margin 9.76/10.07/11.00/13.66 v real 10.14/9.86/10.50/13.82 (strength gap transmission roughly right; deviations of +-0.5 not resolved).
Not recorded with weak-signals record: these are descriptive splits without intervals (about 60 cells counted as looks, family sim08_market_anchored_shape); intervals needed before recording.
Repair specs:
A (mass 3) structure change, fitted LOSO per fold from 2009..s-1 pbp: terminal-state transition table P(final margin | margin at 5:00 and 2:00 left, possession, timeouts, line cell) with kneel and run-out-the-clock
and FG v TD v go decisions as separate fitted outcomes, replacing the single regime x margin-bin logit offsets; first diagnose by instrumenting the sim to log margin at 300 s and 120 s and the last scoring event
type, then compare conditional P(final = 3 | cell) sim v real to locate the cell. Also fix the OT opening-score branch to use the declared rule (reply possession) and measure OT drive elapsed time/tie share from real OT.
Gate: K3 effect interval includes 0 with sim mass >= real - one SE, K7 and K17 not worse than now (+.0096, -.0094), O not worse, L1 not worse than -0.0136, SIM-11 14 gates pass, planted control re-run.
B (total slope) fitted input: replace the target total with the LOSO regression E[real total | opener total] fitted on 2009..s-1 (measured slope 0.888 in 2020-25 to be re-measured out of season), and replace the one-shot quadratic inverse with an iterated
solve at the tails (common random numbers) so sim slope on target is 1.0. Gate: T slope difference interval includes 0 with |effect| < .05, anchor_check total err |.1|, K cells and L4 not worse, SIM-11 14 gates pass.
Heavy command for the next declared unit (after sim10_drive is released and a declaration amendment with the instrumentation): F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim08_market_anchored.py full --run terminal_diag --arms primary --workers 4 --draws 372

## 2026-11-03 OT rule per season (measured, play-by-play data/pbp/raw/20261027T220410Z, REG, qtr 5)
Script tests/scratch/ot_rule.py. Season: OT games / ties / OT length / opening-drive TDs (ended game, followed by opponent scrimmage play) / opening FG.
2012 22/2/900/3(3,0)/4; 2013 16/1/900/2(2,0)/2; 2014 11/1/900/2(2,0)/2; 2015 21/0/900/4(4,0)/6; 2016 13/2/900/3(3,0)/3;
2017 14/0/600/2(2,0)/1; 2018 15/2/600/3(3,0)/4; 2019 9/1/600/4(4,0)/1; 2020 10/1/600/1(1,0)/2; 2021 21/1/600/4(4,0)/5;
2022 20/2/600/1(1,0)/3; 2023 13/0/600/2(2,0)/2; 2024 16/0/600/6(6,0)/2; 2025 14/1/600/4(0,4)/4.
Rule measured: sudden-death-style (opening TD ends game) 2012-2024; both-possess after opening TD 2025 (4 of 4). Period 900 s through 2016, 600 s from 2017 (matches engine).
Real tie rate 2012-2025: 13/213 = 6.1% of OT games (2012 9%, 2013 6%, 2014 9%, 2016 15%, 2018 13%, 2019 11%, 2020 10%, 2021 5%, 2022 10%, 2025 7%; 0 in 2015, 2017, 2023, 2024). Ties computed from last scoring play margin.
Engine (sim10_drive.py:1332-1336,1376) "mod" 2012-2024: both-possess-unless-opening-TD-or-defensive-score; opening FG leads to a reply in both rules, so 2012-2024 matches. 2009-2011 sudden death: not re-measured here (outside 2012-2025 ask).
Change for root after sim10_drive.py frees: add a "both_possess" (playoff) rule for season set {2025} (and later seasons): opening-drive TD does not end OT; the other team gets a possession; then mod/sudden-death continues. Not "mod".
Tie comparison: real 6.1% (not 5%) in 12-... OT games, simulator 15% is 2.5x too high; 2025 rule raises tie chance only slightly.

## Spec B coded (2026-11-03, subagent; no simulation run)
Amendment 4 in docs/sim08_market_anchored_declaration.md. Script flag `--amend4 1` (env SIM08_AMEND4=1; default = Amendment 3), `target` command.
LOSO target (train = other 2020-25 seasons; measured, no sim) slope on opener per held-out 2020..2025: .833 .943 .886 1.000 1.000 1.000
(forms linear x3, identity x3 by nested LOO SSE); real slopes .900 .602 .825 .720 .939 1.108. Production folds train on seasons < s from 2009 (not yet measured).
Iterated Newton anchor (CRN, stop within 1 SE or no improvement, cap 4 steps = budget) is untested by execution; cost up to ~5x pilot draws per game.
Heavy command (only after spec A + OT-rule edits to sim10_drive.py land and heavy slot frees):
F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python F:\Repos\nfl_py3\scripts\sim08_market_anchored.py full --run amend4 --arms primary --workers 4 --draws 372 --amend4 1
Then evaluate: sim08_evaluate.py --run amend4 --arms primary.
