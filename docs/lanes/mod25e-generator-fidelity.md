# mod25e-generator-fidelity

## Goal
Keep improving the MOD-25 generator (owner 2026-10-02). Gate failures to
attack: margin SD (band 13.97-14.87), non-strength variance (real 178),
late-season R2 (band .132-.166), held-out shape (mass 3 real .144). Parent
docs/lanes/mod25-synthetic-data.md; prior docs/lanes/mod25d-variance-source.md.
Rules: nothing targets mass at 3/7/10/14/17; aggregates (pts/g, margin SD,
variance) are checks only. Owner 2026-10-02: NO compensating constants; every
parameter is a named mechanism fitted to its own real behaviour.

## State (measured 2026-10-02, gate @ .9 held out 2018-25)
| variant | SD | nonstr var | r2 w10-18 | mass3 | pts/g |
|---|---|---|---|---|---|
| crj (k4; has yard_bias .75 hack) | 15.62 | 220 | .107 | .113 | 46.2 |
| crk06 = no bias + 1-yd dist matching (E1) | 15.41 | 213 | .122 | .114 | 41.9 |
| crf4 = crk06 + 4th-down bug fix (E4) | 15.16 | 205 | .111 | .118 | 42.5 |
| crf4m = crf4 + gain 6.5 + scale 1.0 (E7, 3 seeds @1.0) | 16.16 | 216 | .197 | .108 | 42.5 |
| css2 = crj + whole-game play-class policy (E3) | 15.71 | 226 | n/a | .123 | n/a |
| crf4 3 seeds (E5) | 15.16 +-.12 | 207 +-2.4 | .112 +-.005 | n/a | 42.0 |
| cf4s = crf4 + css2 policy, 3 seeds (E5) | 15.12 +-.20 | 210 +-5.2 | .091 +-.003 | n/a | 42.8 |
Code: scripts/mod25d_variance.py (DV variants, gate, downdiag), mod25e_budget.py
(variance budget), mod25e_scorestate.py (css/css2), mod25e_deficit.py.
Artifacts: artifacts/mod25d/gate_e1*, gate_e4a.json, downdiag_e1*/e4a;
artifacts/mod25e/budget.txt, deficit_e4*; artifacts/mod25e3/.

## Tried (measured)
- E2 budget (crj vs real): margin var 250 vs 205; strength 56 vs 52 (not the
  excess); noise 194 vs 152. Excess = cross-quarter covariance +16 vs -12
  (real leads mean-revert; Q4 slope -.073 real vs -.030) and drives/g 22.6 vs
  21.3 (~13). Per-drive scoring variance matches.
- E1: yard_bias .75 (MOD-25a/b hack) removed; dist cache rounding 2 yd ->
  1 yd plus distance kernel: 3rd&1 run share .66 (real .72), conv .61 (.66).
  Residual: 3rd 10+ share .232 vs .19; 3rd<=1 share .073 vs .11.
- E3: play choice, tempo, 4th-go by lead already match. Gap is drive outcomes:
  leader TD .269 vs .230 (Q3), punts .352 vs .405, trailer 2-min drives 3.0 vs
  4.1 plays. css2 halves cross-quarter cov (+6.8) but gate not improved; one
  sim stream only.
- E4: engine bug turned drawn 4th-down successes into turnovers on downs when
  sampled gain < sim distance (0.93 spurious/g; 4th conv .416 vs .487). Fixed
  in crf4. Field position, ypp, FG, turnovers match. Remaining pts gap 42.5 vs
  ~44-46: first-down move-on 0.7-1.3 pts low, 20+ yd share 5.25 vs 5.62,
  drives/g +0.8.

## E5 (merge)
cf4s (mod25e_scorestate.py: e5, e5real, e5report; seeds 11-13, 8x8 @.9, artifacts/mod25e3/e5_*/e5.json, e5_real.json).
Real vs crf4 vs cf4s (mean +-sd): margin var 205 / 229.8+-3.7 / 228.8+-6.1; strength RE 52.3 / 49.8+-2.8 / 43.7+-1.6
(merged LOSES strength variance); noise RE 152 / 180.0+-2.5 / 185.1+-4.5; xq cov -12.0 / +7.5+-3.0 / +4.1+-4.2;
Q4 slope -.073 / -.041+-.004 / -.043+-.007; drives/g 21.3 / 23.83+-.02 / 23.89+-.04; r2 w10-18 .150 / .112 / .091.
Merged is not distinguishable on margin var, noise, Q4 slope, drives; only xq cov (p~.15 n=3) improves, strength RE and r2 worsen (>2 sd).
Policy alone does not fix drives/g or Q4 slope; next target drives/g (+2.5 vs real) and strength/r2 loss.

## Next
E7 sets gain 6.5 and scale 1.0 (below). Then E8 residual: 3rd&1 conversion
(.60 vs real .66-.69), sim first-down rates still 1-2 pp low, strength RE and
R2 down vs crf4; leader drive-end outcomes, explosive tail.

## Open
v2 distilled challenger (forward log) was trained on the biased crj world;
rebuild on the cleaned generator before any serving question. Every MOD-25
result built on yard_bias .75 (pipeline v1/v2, mod25_distillation and
mod25_distillation_linemove registry cells) is contaminated: re-run on the
cleaned generator once E5/E6 land. fg_boost (never used) deleted 2026-10-02.

## E6 (knob audit, measured; scripts/mod25e_knobs.py, artifacts/mod25e/knobs.json)
- yard_gain 2.0 (mod25d_variance.py:991, pipeline.py:18): real play yards per unit team EPA/play, cross-fit BLUP (half-games), state-controlled: off 6.35 (5.7-7.0), def 7.24 (6.2-8.3), sum 6.5 (5.9-7.0); raw 5.7; in-sample 6.9-8.1; probability_positive 1.0. Current 2.0 is 3x low; consistency: 6.5 yd x .12 EPA/yd = .78 of the EPA unit.
- def_sign +1: real def/off slope ratio 1.1, matches.
- epa_per_yard .07 (mod25_generator.py:329): real .120 (.119-.120) state-controlled; DEAD in crf4 (chain2 EPA replaces line 444/1047 path).
- pace .04: real net-of-state between-game tempo sigma .035-.037 (CI .033-.039, el capped 45-60s; uncapped is outlier-contaminated) matches.
- strength scale .9 (pipeline.py:17): no real target; fit to R2; with measured gain should be 1.0 (var_mu is measured).
- Structural, no single target: inner/outer/stime .5/2/60 (mechanisms.py:563), K 40, h-scale .5, SCALE_* (engine:31-48), dkern .6, ipw shrink 50 clip .25-4 (d25:394), cal4 shrink 10 (d25:592/654), ep table shrink 20 (d25:805). tilt_coef.json fitted from real. fg_boost removed.

## E7 (gain 6.5/scale 1.0 double counts; scripts/mod25e_ygain.py, artifacts/mod25e/ygain_{m,c}, m25e7/)
- Same cross-fit estimator on sim plays: crf4m slope 6.98 (off 6.67, def 7.96), spread SD .083/.047 vs real 6.5, .057/.028; control crf4 (2.0/.9) 7.01, .059/.024. Gate crf4m margin SD 16.2, r2 .197, budget strength var 80 vs 52: overshoot. Slope is carried by the kernel.

## E7b (derived gain/scale; artifacts/mod25e/e7b/, final/)
- Declared grid (mechanism targets only: cross-fit off/def slope 6.35/7.24, EPA-effect SD .057/.028, sum of squared relative errors): gain {0,1} x scale {.8,.9,1.0} one seed (gain 2/3 not run, E7 covers 2.0@.9 and 6.5@1.0). Kernel-only (gain 0): scale .8/.9/1.0 loss .218/.053/.024 (slope off 6.17, def 7.20, SD .065/.026 at 1.0). Gain 1: .076/.011/.045. Best cell gain 1@.9 (.011) is within single-seed noise of gain 0@1.0; gain 0 needs no yard shift, scale 1.0 is the measured var_mu.
- Derived: yard_gain 0 (kernel carries strength), scale 1.0. Off/def need no separate gain (def slope 7.2 vs 7.24).
- 3 seeds crf4@gain0/1.0 (gate 9/10/11): margin SD 15.29/15.36/15.20, r2 w10-18 .116/.131/.094, mass3 .116/.119/.118, nonstr var ~211; budget margin var 236/229/234 (real 205), strength RE 52.0/50.7/48.9 (real 52.3), noise 185/178/185, nposs 23.3-23.4. Equals crf4 (15.16, r2 .112, 230/49.8/180). Shift removal costs nothing; gate failures unchanged. Next: drives/g, xq cov, Q4 slope.
