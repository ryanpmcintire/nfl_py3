# Variance decomposition noise floor (SIM-02 lite)

Question: what is the MINIMUM achievable outcome-margin variance given perfect
team-strength knowledge? Equivalently: how much of NFL game-to-game margin
variance is irreducible play-level execution noise versus scheduling/matchup
structure that a model could in principle capture?

Status: measure-only simulator milestone, first run 2026-08-22. Nothing here is
a selection strategy and nothing was recorded to the registry.

## Method

Empirical resampling simulator (`scripts/vardec_noisefloor.py`):

- Population: REG 2021-2025, 1,359 scored games; per (season, team) offensive
  play pools from `data/pbp/raw/20260817T184927Z` (pass/run plays only,
  kneels/spikes/aborted/no-play excluded), 160 pools, 167,772 plays, smallest
  pool 939 (**measured**).
- Games are simulated by sampling plays WITHOUT replacement from each
  possessing team-season's shuffled deck. Drive structure approximated:
  alternating possessions, chain/down bookkeeping, league field-goal-rate model
  by distance measured from pbp, punt net 38 yards with pinning, turnover spots
  mirrored, garbage-time kneel heuristic when leading by 9+ with at most 2
  drives left, resampled real (home, away) drive-count pairs, Poisson
  non-offensive-TD events at the measured 0.261/games rate.
- Defense enters through a half-weight additive per-play yardage adjustment
  from yards allowed per play (**declared limitation**: no coverage/pass-rush
  split).
- Offensive penalties replayed as distance-only events at the measured
  0.0868/snap rate with the empirical penalty-yardage distribution
  (**measured**).
- Two calibration stages against measured targets:
  1. series drag (extra required yards per first-down series) bisected so chain
     series conversion matches the real 0.6616 rate (**measured**); selected
     drag = **2.001**;
  2. execution-dispersion dial `disp`, interpolating every sampled play between
     its pool mean (disp=0) and its observed value (disp=1), bisected so the
     simulated margin sd matches the real margin sd within 0.5 points.
- ABLATION: disp=0 is exactly the brief's "replace each team's pool by its own
  mean-EPA-equivalent play" world: play mix retained at the pool-mean level,
  play-level execution noise removed. The floor is the between-matchup sd of
  per-game mean margins over 192 ablated reps, corrected for finite-rep Monte
  Carlo noise.

## Results

All numbers from `artifacts/vardec_floor/20260822T213001Z/results.json`
(**measured**):

| Quantity | Value |
|---|---|
| Real margin sd (target) | 14.188 points |
| Calibrated sim margin sd | 14.244 points (gap +0.056, gate <=0.5 **pass**) |
| Selected dispersion | disp = 0.3125 |
| Ablated floor sd | 6.441 points (raw between 6.482, MC noise 0.727) |
| Execution-noise sd | sqrt(14.244^2 - 6.441^2) = **12.71 points** |
| Execution-noise share | 1 - (6.441/14.244)^2 = **0.795** |

Reading of disp = 0.3125: full-strength empirical resampling OVER-disperses
margins (sd 17.69 at disp=1) because resampled sequences omit correlated
real-world dampers (clock/game-script conservatism beyond the kneel rule,
weather shared within a game, within-season churn). The calibrated world
retains about 31% of per-play deviation amplitude; this is a declared
structural approximation, not a finding.

## Validation vs real margins

(**measured**, calibrated run vs real games)

| Statistic | Real | Simulated |
|---|---|---|
| mean margin | +2.06 | +0.96 |
| sd margin | 14.19 | 14.24 |
| abs-margin quantiles 10/25/50/75/90 | 2 / 3 / 8 / 17 / 24 | 2 / 5 / 9 / 15 / 23 |
| share abs(margin) = 3 | 0.146 | 0.024 |
| share abs(margin) = 7 | 0.077 | 0.016 |
| share abs(margin) <= 3 | 0.252 | 0.172 |
| share margin in 2/3/6/7 either side | 0.345 | 0.203 |

Shape verdict: quantiles reproduce reasonably; KEY-NUMBER CLUSTERING IS
ATTENUATED in the calibrated sim (real margins cluster on 3/7 far more). The
dispersion dial compresses discrete scoring lumps and the FG model is
league-level rather than team-kicker-specific. Treat simulated key-number mass
as a lower bound; the floor estimate is computed in the expectation-based
ablated world and is not affected by this.

Mean margin gap (+0.96 vs +2.06): HFA injected as a constant 2.0 points; the
residual -1.0 is engine asymmetry (misc-event side split, kneel timing), noted
for honesty.

## Interpretation for the capturable ceiling

(**inferred** from the measured decomposition, not directly measured)

- Outcome variance decomposes into matchup/schedule variance (6.44^2 = 41.5)
  and execution noise (12.71^2 = 161.6): execution dominates at ~80%.
- Even perfect team-strength knowledge leaves at least ~6.4-point sd across the
  real schedule; perfect knowledge can remove at most ~6.8 points of sd versus
  a strength-blind baseline, and only if that matchup signal is not already
  priced.
- Consistency check: the project ATS residual sd is ~13.1 points
  (`docs/vardec_sigma_map.md`). If the market already prices most of the
  6.4-point matchup component, the ATS residual should be approximately the
  execution-noise sd (~12.7) plus unmatched matchup residue, which is what is
  observed. This supports treating the ~13.1 ATS residual sd as mostly
  irreducible and the capturable ceiling from better strength modeling as
  SMALL. Per AGENTS.md this is a variance statement, never a claim about mean
  edge.

## Caveats

- Season-level pools ignore within-season drift by design ("perfect" knowledge
  idealized at season resolution).
- No overtime, no two-point attempts, possession starts fixed at the 25 except
  where punt/turnover spots apply, FG model is league-level by distance.
- The ablated floor keeps turnover skill as an expectation inside the drive
  value; it contains zero Bernoulli execution noise by construction.
- Single-season-window population (2021-2025). Era stability untested; treat
  the floor as current-era only until replicated on earlier windows.
